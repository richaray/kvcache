#pragma once

#include <chrono>
#include <cstddef>
#include <limits>
#include <list>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace kvcache {

// An in-memory key-value cache with fixed capacity, LFU (Least Frequently Used)
// eviction, optional per-entry TTL expiry, and full thread safety via a single mutex.
//
// Design — O(1) LFU (Shah et al., 2010):
//   - freq_map_: unordered_map<int, list<Node>>
//       Each entry maps a frequency f to the list of all nodes currently at f.
//       Within each bucket, the list is in LRU order (front = MRU, back = LRU),
//       giving LRU tiebreaking among entries with equal frequency.
//   - index_: unordered_map<Key, ListIter>
//       Direct iterator to each node — O(1) lookup without searching.
//   - min_freq_: current minimum frequency across all entries.
//       The eviction victim is always back of freq_map_[min_freq_].
//
// LFU vs LRU — when each wins:
//   LRU: good when access patterns shift over time (recency matters).
//        A key accessed 1000× yesterday but never today will be evicted quickly.
//   LFU: good when some keys are structurally "hot" and stay popular long-term.
//        A key accessed 1000× is protected even if not accessed in the last minute.
//   LFU weakness ("cache pollution"): a key accessed many times historically
//        but now irrelevant keeps its high frequency and resists eviction.
//
// Same public interface as LRUCache — drop-in replacement for benchmarking.
template <typename Key, typename Value>
class LFUCache {
public:
    using Clock     = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    using Duration  = Clock::duration;

    explicit LFUCache(std::size_t capacity) : capacity_(capacity) {}

    LFUCache(const LFUCache&)            = delete;
    LFUCache& operator=(const LFUCache&) = delete;
    LFUCache(LFUCache&&)                 = delete;
    LFUCache& operator=(LFUCache&&)      = delete;

    // Hit: bump frequency and return value.
    // Miss or expired: remove (if expired) and return nullopt.
    std::optional<Value> get(const Key& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = index_.find(key);
        if (it == index_.end()) return std::nullopt;

        if (isExpired(it->second)) {
            removeNode(it->second);
            index_.erase(it);
            return std::nullopt;
        }

        incrementFreq(it->second);
        return it->second->value;
    }

    void put(const Key& key, const Value& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        putImpl(key, value, TimePoint::max());
    }

    void put(const Key& key, const Value& value, Duration ttl) {
        std::lock_guard<std::mutex> lock(mutex_);
        putImpl(key, value, Clock::now() + ttl);
    }

    bool erase(const Key& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = index_.find(key);
        if (it == index_.end()) return false;
        removeNode(it->second);
        index_.erase(it);
        return true;
    }

    std::size_t purge_expired() {
        std::lock_guard<std::mutex> lock(mutex_);
        return purgeExpiredUnlocked();
    }

    bool contains(const Key& key) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return index_.count(key) > 0;
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return index_.size();
    }

    std::size_t capacity() const { return capacity_; }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return index_.empty();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        index_.clear();
        freq_map_.clear();
        min_freq_ = 1;
    }

private:
    struct Node {
        Key       key;
        Value     value;
        int       freq;
        TimePoint expires_at;
    };
    using NodeList = std::list<Node>;
    using ListIter = typename NodeList::iterator;

    static bool isExpired(ListIter it) {
        return it->expires_at != TimePoint::max() &&
               Clock::now() >= it->expires_at;
    }

    // Move node from freq f to f+1. Splice preserves iterator validity so
    // index_ entries stay correct without any update.
    void incrementFreq(ListIter node_it) {
        const int old_f = node_it->freq;
        const int new_f = old_f + 1;

        // Splice to the front of the new bucket (MRU position at freq new_f).
        freq_map_[new_f].splice(freq_map_[new_f].begin(),
                                freq_map_[old_f],
                                node_it);
        node_it->freq = new_f;

        if (freq_map_[old_f].empty()) {
            freq_map_.erase(old_f);
            if (min_freq_ == old_f) min_freq_ = new_f;
        }
    }

    // Remove a node from its frequency bucket. Cleans up the bucket if empty.
    void removeNode(ListIter node_it) {
        const int f = node_it->freq;
        freq_map_[f].erase(node_it);
        if (freq_map_[f].empty()) {
            freq_map_.erase(f);
            if (min_freq_ == f) recomputeMinFreq();
        }
    }

    // Evict the LFU entry. Among entries with the same minimum frequency,
    // the one at the back of the bucket (LRU) is chosen.
    void evictLFU() {
        auto& bucket = freq_map_[min_freq_];
        Key victim_key = bucket.back().key;   // copy before pop
        bucket.pop_back();
        index_.erase(victim_key);
        if (bucket.empty()) freq_map_.erase(min_freq_);
        // min_freq_ is always reset to 1 on the next new insert,
        // so no update needed here.
    }

    // Scan all buckets for the new minimum. O(F) where F = distinct frequencies.
    // Called only after erase/purge — not on the hot path.
    void recomputeMinFreq() {
        min_freq_ = std::numeric_limits<int>::max();
        for (const auto& [f, lst] : freq_map_) {
            if (!lst.empty()) min_freq_ = std::min(min_freq_, f);
        }
        if (min_freq_ == std::numeric_limits<int>::max()) min_freq_ = 1;
    }

    // Must be called with mutex_ already held.
    std::size_t purgeExpiredUnlocked() {
        std::size_t count = 0;
        for (auto freq_it = freq_map_.begin(); freq_it != freq_map_.end(); ) {
            auto& lst = freq_it->second;
            for (auto node_it = lst.begin(); node_it != lst.end(); ) {
                if (isExpired(node_it)) {
                    index_.erase(node_it->key);
                    node_it = lst.erase(node_it);
                    ++count;
                } else {
                    ++node_it;
                }
            }
            freq_it = lst.empty() ? freq_map_.erase(freq_it) : ++freq_it;
        }
        if (count > 0) recomputeMinFreq();
        return count;
    }

    // Must be called with mutex_ already held.
    void putImpl(const Key& key, const Value& value, TimePoint expires_at) {
        auto it = index_.find(key);
        if (it != index_.end()) {
            // Update: refresh value + bump frequency (same as a get hit).
            it->second->value      = value;
            it->second->expires_at = expires_at;
            incrementFreq(it->second);
            return;
        }
        if (capacity_ == 0) return;
        if (index_.size() >= capacity_) {
            if (purgeExpiredUnlocked() == 0) evictLFU();
        }
        // New entries always start at frequency 1, MRU position in that bucket.
        freq_map_[1].push_front(Node{key, value, 1, expires_at});
        index_[key] = freq_map_[1].begin();
        min_freq_ = 1;
    }

    const std::size_t                       capacity_;
    int                                     min_freq_ = 1;
    std::unordered_map<Key, ListIter>       index_;
    std::unordered_map<int, NodeList>       freq_map_;
    mutable std::mutex                      mutex_;
};

}  // namespace kvcache
