#pragma once

#include <chrono>
#include <cstddef>
#include <list>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace kvcache {

// An in-memory key-value cache with fixed capacity, LRU eviction, optional
// per-entry TTL expiry, and full thread safety via a single mutex.
//
// Design:
//   - A std::list<Node> holds entries in usage order (front = MRU, back = LRU).
//   - An unordered_map<Key, list-iterator> enables O(1) lookup and splice.
//   - Each Node carries an expires_at timestamp. TimePoint::max() means "never".
//   - A single std::mutex serialises every public operation (coarse-grained lock).
//
// Thread-safety contract: all public methods are safe to call concurrently.
// Each method acquires the mutex on entry and releases it on return.
//
// Locking rule for private helpers:
//   Public methods  → acquire mutex_, then delegate to *Unlocked helpers.
//   *Unlocked helpers → must only be called while mutex_ is already held.
//   This avoids recursive locking (std::mutex is not re-entrant).
//
// NOTE: capacity() is the only method with no lock — capacity_ is set in the
// constructor and never modified, so reading it is always safe.
template <typename Key, typename Value>
class LRUCache {
public:
    using Clock     = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;
    using Duration  = Clock::duration;

    explicit LRUCache(std::size_t capacity) : capacity_(capacity) {}

    // Mutex is not copyable or movable, so neither is the cache.
    LRUCache(const LRUCache&)            = delete;
    LRUCache& operator=(const LRUCache&) = delete;
    LRUCache(LRUCache&&)                 = delete;
    LRUCache& operator=(LRUCache&&)      = delete;

    std::optional<Value> get(const Key& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = index_.find(key);
        if (it == index_.end()) return std::nullopt;

        if (isExpired(it->second)) {
            order_.erase(it->second);
            index_.erase(it);
            return std::nullopt;
        }

        order_.splice(order_.begin(), order_, it->second);
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
        order_.erase(it->second);
        index_.erase(it);
        return true;
    }

    // Active expiry: scan and remove all expired entries.
    // Safe to call from any thread at any time.
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

    // No lock needed — capacity_ is immutable after construction.
    std::size_t capacity() const { return capacity_; }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return index_.empty();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        order_.clear();
        index_.clear();
    }

private:
    struct Node {
        Key       key;
        Value     value;
        TimePoint expires_at;
    };
    using ListIter = typename std::list<Node>::iterator;

    static bool isExpired(ListIter it) {
        return it->expires_at != TimePoint::max() && Clock::now() >= it->expires_at;
    }

    void evictLRU() {
        index_.erase(order_.back().key);
        order_.pop_back();
    }

    // Must be called with mutex_ already held.
    std::size_t purgeExpiredUnlocked() {
        std::size_t count = 0;
        for (auto it = order_.begin(); it != order_.end(); ) {
            if (isExpired(it)) {
                index_.erase(it->key);
                it = order_.erase(it);
                ++count;
            } else {
                ++it;
            }
        }
        return count;
    }

    // Must be called with mutex_ already held.
    void putImpl(const Key& key, const Value& value, TimePoint expires_at) {
        auto it = index_.find(key);
        if (it != index_.end()) {
            it->second->value      = value;
            it->second->expires_at = expires_at;
            order_.splice(order_.begin(), order_, it->second);
            return;
        }
        if (capacity_ == 0) return;
        if (order_.size() >= capacity_) {
            if (purgeExpiredUnlocked() == 0) evictLRU();
        }
        order_.push_front(Node{key, value, expires_at});
        index_[key] = order_.begin();
    }

    const std::size_t                 capacity_;
    std::list<Node>                   order_;
    std::unordered_map<Key, ListIter> index_;
    mutable std::mutex                mutex_;
};

}  // namespace kvcache
