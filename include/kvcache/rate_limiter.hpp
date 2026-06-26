#pragma once

#include <chrono>
#include <deque>
#include <mutex>
#include <unordered_map>

namespace kvcache {

// Token-bucket rate limiter.
//
// Each key gets an independent bucket. The bucket holds up to `capacity`
// tokens and refills at `rate` tokens per second. Every allowed request
// costs 1 token (or a custom cost via the two-argument overload).
//
// Key property — burst allowance:
//   An idle key accumulates tokens up to `capacity` and can spend them all
//   at once. Useful for APIs where short spikes are fine as long as the
//   sustained rate stays within budget.
//
// Algorithm (O(1) per operation):
//   On allow(key):
//     1. Compute elapsed = now - last_refill
//     2. tokens = min(capacity, tokens + elapsed * rate)
//     3. If tokens >= cost: tokens -= cost, return true; else return false.
template <typename Key>
class TokenBucketLimiter {
public:
    using Clock = std::chrono::steady_clock;

    // capacity:        max burst size in tokens
    // rate_per_second: tokens added per second (0 = no refill)
    TokenBucketLimiter(double capacity, double rate_per_second)
        : capacity_(capacity), rate_(rate_per_second) {}

    TokenBucketLimiter(const TokenBucketLimiter&)            = delete;
    TokenBucketLimiter& operator=(const TokenBucketLimiter&) = delete;

    // Returns true and deducts 1 token if the bucket is non-empty.
    bool allow(const Key& key) { return allow(key, 1.0); }

    // Returns true and deducts `cost` tokens if enough are available.
    bool allow(const Key& key, double cost) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto [it, inserted] = buckets_.try_emplace(key, capacity_, Clock::now());
        if (!inserted) refill(it->second);
        if (it->second.tokens >= cost) {
            it->second.tokens -= cost;
            return true;
        }
        return false;
    }

    // Current token level for key (triggers a refill calculation).
    double tokens_remaining(const Key& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto [it, inserted] = buckets_.try_emplace(key, capacity_, Clock::now());
        if (!inserted) refill(it->second);
        return it->second.tokens;
    }

    void reset(const Key& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        buckets_.erase(key);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        buckets_.clear();
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buckets_.size();
    }

private:
    struct Bucket {
        double            tokens;
        Clock::time_point last_refill;
        Bucket(double t, Clock::time_point tp) : tokens(t), last_refill(std::move(tp)) {}
    };

    void refill(Bucket& b) {
        auto   now     = Clock::now();
        double elapsed = std::chrono::duration<double>(now - b.last_refill).count();
        b.tokens       = std::min(capacity_, b.tokens + elapsed * rate_);
        b.last_refill  = now;
    }

    const double                    capacity_;
    const double                    rate_;
    std::unordered_map<Key, Bucket> buckets_;
    mutable std::mutex              mutex_;
};

// Sliding-window rate limiter.
//
// Counts requests in a rolling time window per key. No burst allowance —
// once a key hits `limit` requests within the window, every subsequent
// request in that window is denied, even if the requests were spread evenly.
//
// Algorithm (O(count_in_window) per operation):
//   On allow(key):
//     1. Drop timestamps older than now - window from the front of the log.
//     2. If log.size() < limit: record now, return true; else return false.
//
// Memory: O(limit) timestamps per active key (old entries are dropped lazily).
template <typename Key>
class SlidingWindowLimiter {
public:
    using Clock    = std::chrono::steady_clock;
    using Duration = Clock::duration;

    // limit:  max requests allowed within one window
    // window: rolling window duration
    SlidingWindowLimiter(int limit, Duration window)
        : limit_(limit), window_(window) {}

    SlidingWindowLimiter(const SlidingWindowLimiter&)            = delete;
    SlidingWindowLimiter& operator=(const SlidingWindowLimiter&) = delete;

    // Returns true if the key has fewer than `limit` requests in the last window.
    bool allow(const Key& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& log = logs_[key];
        evict(log);
        if (static_cast<int>(log.size()) < limit_) {
            log.push_back(Clock::now());
            return true;
        }
        return false;
    }

    // How many requests this key has made in the current window.
    int requests_in_window(const Key& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = logs_.find(key);
        if (it == logs_.end()) return 0;
        evict(it->second);
        return static_cast<int>(it->second.size());
    }

    void reset(const Key& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        logs_.erase(key);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        logs_.clear();
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return logs_.size();
    }

private:
    using Log = std::deque<Clock::time_point>;

    void evict(Log& log) {
        auto cutoff = Clock::now() - window_;
        while (!log.empty() && log.front() <= cutoff) log.pop_front();
    }

    const int                        limit_;
    const Duration                   window_;
    std::unordered_map<Key, Log>     logs_;
    mutable std::mutex               mutex_;
};

}  // namespace kvcache
