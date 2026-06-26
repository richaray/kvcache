#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <memory>

#include "kvcache/lru_cache.hpp"

namespace kvcache {

// A sharded cache that reduces lock contention by routing each key to one of
// NumShards independent LRUCache instances, each with its own mutex.
//
// Layer 3 concept — lock granularity vs throughput:
//   Single mutex (Layer 2): every operation acquires one global lock, so all
//     threads serialise even when touching completely different keys.
//   Sharded (Layer 3): key K always goes to shard hash(K) % NumShards. Two
//     threads that happen to hit different shards hold different locks and run
//     truly in parallel. Under N threads with N shards, throughput can approach
//     N× single-threaded speed when keys are spread uniformly.
//
// Trade-offs vs a single mutex:
//   + Throughput scales with the number of shards (up to CPU core count).
//   - Capacity is split: total_capacity / NumShards per shard. Hot shards may
//     evict faster than cold ones if keys are not uniformly distributed.
//   - size() is not a consistent snapshot — each shard is locked separately,
//     so a concurrent write can land between two shard reads.
//   - Operations that span multiple keys (e.g., atomic swap of two keys) are
//     not possible without additional coordination.
//
// NumShards defaults to 8. A power-of-two that roughly matches core count
// works well in practice.
template <typename Key, typename Value, std::size_t NumShards = 8>
class ShardedCache {
    static_assert(NumShards >= 1, "NumShards must be at least 1");

public:
    explicit ShardedCache(std::size_t total_capacity) {
        const std::size_t per_shard = std::max(std::size_t{1}, total_capacity / NumShards);
        for (auto& s : shards_)
            s = std::make_unique<LRUCache<Key, Value>>(per_shard);
    }

    ShardedCache(const ShardedCache&)            = delete;
    ShardedCache& operator=(const ShardedCache&) = delete;
    ShardedCache(ShardedCache&&)                 = delete;
    ShardedCache& operator=(ShardedCache&&)      = delete;

    std::optional<Value> get(const Key& key) {
        return shardFor(key).get(key);
    }

    void put(const Key& key, const Value& value) {
        shardFor(key).put(key, value);
    }

    void put(const Key& key, const Value& value,
             typename LRUCache<Key, Value>::Duration ttl) {
        shardFor(key).put(key, value, ttl);
    }

    bool erase(const Key& key) {
        return shardFor(key).erase(key);
    }

    bool contains(const Key& key) const {
        return shardFor(key).contains(key);
    }

    // Sweeps all shards in sequence. Each shard is locked independently —
    // other threads can still access unlocked shards while the sweep runs.
    std::size_t purge_expired() {
        std::size_t total = 0;
        for (auto& s : shards_) total += s->purge_expired();
        return total;
    }

    // Sum of all shard sizes. Not a consistent snapshot under concurrent writes.
    std::size_t size() const {
        std::size_t total = 0;
        for (const auto& s : shards_) total += s->size();
        return total;
    }

    // Total capacity across all shards.
    std::size_t capacity() const {
        std::size_t total = 0;
        for (const auto& s : shards_) total += s->capacity();
        return total;
    }

    static constexpr std::size_t num_shards() { return NumShards; }

    void clear() {
        for (auto& s : shards_) s->clear();
    }

private:
    std::array<std::unique_ptr<LRUCache<Key, Value>>, NumShards> shards_;

    LRUCache<Key, Value>& shardFor(const Key& key) {
        return *shards_[std::hash<Key>{}(key) % NumShards];
    }

    const LRUCache<Key, Value>& shardFor(const Key& key) const {
        return *shards_[std::hash<Key>{}(key) % NumShards];
    }
};

}  // namespace kvcache
