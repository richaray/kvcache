#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "kvcache/lfu_cache.hpp"
#include "kvcache/lru_cache.hpp"
#include "kvcache/sharded_cache.hpp"

using kvcache::LFUCache;
using kvcache::LRUCache;
using kvcache::ShardedCache;
using namespace std::chrono_literals;

// ── Layer 0: LRU eviction ────────────────────────────────────────────────────

TEST(LRUCache, MissOnEmpty) {
    LRUCache<int, int> c(2);
    EXPECT_FALSE(c.get(1).has_value());
    EXPECT_EQ(c.size(), 0u);
}

TEST(LRUCache, PutThenGet) {
    LRUCache<int, std::string> c(2);
    c.put(1, "one");
    auto v = c.get(1);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, "one");
}

TEST(LRUCache, UpdateExistingKeyDoesNotGrow) {
    LRUCache<int, int> c(2);
    c.put(1, 10);
    c.put(1, 20);
    EXPECT_EQ(c.size(), 1u);
    EXPECT_EQ(*c.get(1), 20);
}

TEST(LRUCache, EvictsLeastRecentlyUsed) {
    LRUCache<int, int> c(2);
    c.put(1, 1);
    c.put(2, 2);
    c.put(3, 3);
    EXPECT_FALSE(c.get(1).has_value());
    EXPECT_TRUE(c.get(2).has_value());
    EXPECT_TRUE(c.get(3).has_value());
    EXPECT_EQ(c.size(), 2u);
}

TEST(LRUCache, GetPromotesToMostRecentlyUsed) {
    LRUCache<int, int> c(2);
    c.put(1, 1);
    c.put(2, 2);
    EXPECT_TRUE(c.get(1).has_value());
    c.put(3, 3);
    EXPECT_TRUE(c.get(1).has_value());
    EXPECT_FALSE(c.get(2).has_value());
    EXPECT_TRUE(c.get(3).has_value());
}

TEST(LRUCache, Erase) {
    LRUCache<int, int> c(2);
    c.put(1, 1);
    EXPECT_TRUE(c.erase(1));
    EXPECT_FALSE(c.erase(1));
    EXPECT_FALSE(c.contains(1));
    EXPECT_EQ(c.size(), 0u);
}

TEST(LRUCache, ZeroCapacityStoresNothing) {
    LRUCache<int, int> c(0);
    c.put(1, 1);
    EXPECT_FALSE(c.get(1).has_value());
    EXPECT_EQ(c.size(), 0u);
}

// ── Layer 1: TTL expiry ──────────────────────────────────────────────────────

TEST(LRUCache, TTLEntryVisibleBeforeExpiry) {
    LRUCache<int, int> c(4);
    c.put(1, 42, 500ms);
    ASSERT_TRUE(c.get(1).has_value());
    EXPECT_EQ(*c.get(1), 42);
}

TEST(LRUCache, TTLLazyExpiryOnGet) {
    LRUCache<int, int> c(4);
    c.put(1, 99, 50ms);
    std::this_thread::sleep_for(100ms);
    EXPECT_FALSE(c.get(1).has_value());
    EXPECT_EQ(c.size(), 0u);
}

TEST(LRUCache, NoTTLEntryNeverExpires) {
    LRUCache<int, int> c(4);
    c.put(1, 7);
    std::this_thread::sleep_for(100ms);
    ASSERT_TRUE(c.get(1).has_value());
    EXPECT_EQ(*c.get(1), 7);
}

TEST(LRUCache, TTLUpdateRefreshesExpiry) {
    LRUCache<int, int> c(4);
    c.put(1, 10, 50ms);
    std::this_thread::sleep_for(30ms);
    c.put(1, 20, 300ms);
    std::this_thread::sleep_for(80ms);
    ASSERT_TRUE(c.get(1).has_value());
    EXPECT_EQ(*c.get(1), 20);
}

TEST(LRUCache, PurgeExpiredRemovesStaleEntries) {
    LRUCache<int, int> c(4);
    c.put(1, 1, 50ms);
    c.put(2, 2, 50ms);
    c.put(3, 3);
    std::this_thread::sleep_for(100ms);
    EXPECT_EQ(c.purge_expired(), 2u);
    EXPECT_EQ(c.size(), 1u);
    EXPECT_TRUE(c.get(3).has_value());
}

TEST(LRUCache, PurgeExpiredOnFreshCacheReturnsZero) {
    LRUCache<int, int> c(4);
    c.put(1, 1);
    c.put(2, 2, 500ms);
    EXPECT_EQ(c.purge_expired(), 0u);
    EXPECT_EQ(c.size(), 2u);
}

TEST(LRUCache, ExpiredEntryEvictedBeforeValidLRU) {
    LRUCache<int, int> c(2);
    c.put(1, 1, 50ms);
    c.put(2, 2);
    std::this_thread::sleep_for(100ms);
    c.put(3, 3);
    EXPECT_FALSE(c.get(1).has_value());
    EXPECT_TRUE(c.get(2).has_value());
    EXPECT_TRUE(c.get(3).has_value());
}

// ── Layer 2: thread safety ───────────────────────────────────────────────────

TEST(LRUCache, NonCopyable) {
    static_assert(!std::is_copy_constructible<LRUCache<int, int>>::value);
    static_assert(!std::is_copy_assignable<LRUCache<int, int>>::value);
    static_assert(!std::is_move_constructible<LRUCache<int, int>>::value);
}

TEST(LRUCache, ConcurrentWritesNoCorruption) {
    LRUCache<int, int> c(200);
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&c, t]() {
            for (int i = 0; i < 50; ++i) c.put(t * 50 + i, i);
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(c.size(), 200u);
}

TEST(LRUCache, ConcurrentReadsNoCorruption) {
    LRUCache<int, int> c(100);
    for (int i = 0; i < 100; ++i) c.put(i, i);
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&c]() {
            for (int i = 0; i < 100; ++i) c.get(i);
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(c.size(), 100u);
}

TEST(LRUCache, ConcurrentReadWriteNoCorruption) {
    LRUCache<int, int> c(50);
    for (int i = 0; i < 50; ++i) c.put(i, i);
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&c]() {
            for (int i = 0; i < 50; ++i) c.get(i);
        });
    }
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&c, t]() {
            for (int i = 0; i < 50; ++i) c.put(t * 50 + i, i);
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_LE(c.size(), 50u);
}

TEST(LRUCache, ConcurrentSweepAndWrite) {
    LRUCache<int, int> c(100);
    std::atomic<bool> done{false};
    std::thread sweeper([&c, &done]() {
        while (!done.load()) {
            c.purge_expired();
            std::this_thread::sleep_for(5ms);
        }
    });
    std::vector<std::thread> writers;
    for (int t = 0; t < 4; ++t) {
        writers.emplace_back([&c, t]() {
            for (int i = 0; i < 50; ++i) c.put(t * 50 + i, i, 20ms);
        });
    }
    for (auto& w : writers) w.join();
    done.store(true);
    sweeper.join();
    EXPECT_LE(c.size(), 100u);
}

// ── Layer 3: sharded cache ───────────────────────────────────────────────────

TEST(ShardedCache, MissOnEmpty) {
    ShardedCache<int, int, 4> c(400);
    EXPECT_FALSE(c.get(42).has_value());
}

TEST(ShardedCache, BasicPutGet) {
    ShardedCache<int, int, 4> c(400);
    c.put(1, 100);
    auto v = c.get(1);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, 100);
}

TEST(ShardedCache, UpdateExistingKey) {
    ShardedCache<int, int, 4> c(400);
    c.put(1, 10);
    c.put(1, 20);
    EXPECT_EQ(*c.get(1), 20);
}

TEST(ShardedCache, EraseAndContains) {
    ShardedCache<int, int, 4> c(400);
    c.put(7, 77);
    EXPECT_TRUE(c.contains(7));
    EXPECT_TRUE(c.erase(7));
    EXPECT_FALSE(c.erase(7));   // already gone
    EXPECT_FALSE(c.contains(7));
}

// Large capacity ensures no evictions so size() == number of inserts.
TEST(ShardedCache, SizeIsSumAcrossShards) {
    ShardedCache<int, int, 4> c(4000);  // 1000 per shard
    for (int i = 0; i < 40; ++i) c.put(i, i);
    EXPECT_EQ(c.size(), 40u);
}

TEST(ShardedCache, CapacityIsSumAcrossShards) {
    ShardedCache<int, int, 4> c(400);
    EXPECT_EQ(c.capacity(), 400u);  // 4 × 100
    EXPECT_EQ(c.num_shards(), 4u);
}

TEST(ShardedCache, TTLExpiry) {
    ShardedCache<int, int, 4> c(400);
    c.put(1, 99, 50ms);
    EXPECT_TRUE(c.get(1).has_value());
    std::this_thread::sleep_for(100ms);
    EXPECT_FALSE(c.get(1).has_value());
}

// Large capacity + enough shards so all expired entries land somewhere.
TEST(ShardedCache, PurgeExpiredAcrossAllShards) {
    ShardedCache<int, int, 4> c(4000);
    for (int i = 0; i < 40; ++i) c.put(i, i, 50ms);
    c.put(9999, 1);  // no TTL — survives
    std::this_thread::sleep_for(100ms);
    EXPECT_EQ(c.purge_expired(), 40u);
    EXPECT_EQ(c.size(), 1u);
    EXPECT_TRUE(c.get(9999).has_value());
}

// 8 threads × 100 unique keys = 800. Capacity 8000 (1000/shard) ensures no
// evictions regardless of hash distribution.
TEST(ShardedCache, ConcurrentWritesNoCorruption) {
    ShardedCache<int, int, 8> c(8000);
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&c, t]() {
            for (int i = 0; i < 100; ++i) c.put(t * 100 + i, i);
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(c.size(), 800u);
}

// Mixed concurrent load — only guarantee is size ≤ capacity.
TEST(ShardedCache, ConcurrentReadWriteNoCorruption) {
    ShardedCache<int, int, 8> c(100);
    for (int i = 0; i < 100; ++i) c.put(i, i);
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&c]() {
            for (int i = 0; i < 100; ++i) c.get(i);
        });
    }
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&c, t]() {
            for (int i = 0; i < 100; ++i) c.put(t * 100 + i, i);
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_LE(c.size(), c.capacity());
}

// Deterministic routing: the same key always lands in the same shard.
TEST(ShardedCache, KeyRoutingIsDeterministic) {
    ShardedCache<int, int, 4> c(400);
    c.put(42, 1);
    // Write 42 many times — if routing were random, shard sizes would diverge.
    for (int i = 0; i < 100; ++i) c.put(42, i);
    // Key 42 should be present exactly once, with the last written value.
    ASSERT_TRUE(c.get(42).has_value());
    EXPECT_EQ(*c.get(42), 99);
    EXPECT_EQ(c.size(), 1u);
}

// ── Layer 5: LFU eviction ────────────────────────────────────────────────────

TEST(LFUCache, MissOnEmpty) {
    LFUCache<int, int> c(4);
    EXPECT_FALSE(c.get(1).has_value());
    EXPECT_EQ(c.size(), 0u);
}

TEST(LFUCache, PutThenGet) {
    LFUCache<int, int> c(4);
    c.put(1, 42);
    auto v = c.get(1);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(*v, 42);
}

TEST(LFUCache, ZeroCapacityStoresNothing) {
    LFUCache<int, int> c(0);
    c.put(1, 1);
    EXPECT_FALSE(c.get(1).has_value());
    EXPECT_EQ(c.size(), 0u);
}

TEST(LFUCache, UpdateExistingKeyDoesNotGrow) {
    LFUCache<int, int> c(4);
    c.put(1, 10);
    c.put(1, 20);
    EXPECT_EQ(c.size(), 1u);
    EXPECT_EQ(*c.get(1), 20);
}

// Core LFU property: the entry with the lowest access count is evicted.
// Key 1 is accessed twice (put + get), key 2 once (put only).
// When key 3 arrives, key 2 (freq=1) is evicted, not key 1 (freq=2).
TEST(LFUCache, EvictsLeastFrequentlyUsed) {
    LFUCache<int, int> c(2);
    c.put(1, 1);   // freq[1] = 1
    c.get(1);      // freq[1] = 2
    c.put(2, 2);   // freq[2] = 1
    c.put(3, 3);   // full → evicts key 2 (freq=1, LFU)
    EXPECT_TRUE(c.get(1).has_value());
    EXPECT_FALSE(c.get(2).has_value());
    EXPECT_TRUE(c.get(3).has_value());
}

// Among entries with equal frequency, the LRU one is evicted (tiebreaker).
// All three keys start at freq=1. Key 1 is LRU (inserted first, never re-accessed).
TEST(LFUCache, LRUTiebreakerAmongSameFrequency) {
    LFUCache<int, int> c(3);
    c.put(1, 1);   // freq=1, LRU order: back=1
    c.put(2, 2);   // freq=1, LRU order: back=1
    c.put(3, 3);   // freq=1, LRU order: [front=3, 2, back=1]
    c.put(4, 4);   // full → evicts key 1 (freq=1, LRU among freq=1)
    EXPECT_FALSE(c.get(1).has_value());
    EXPECT_TRUE(c.get(2).has_value());
    EXPECT_TRUE(c.get(3).has_value());
    EXPECT_TRUE(c.get(4).has_value());
}

// get() increments frequency, which protects an entry from eviction.
TEST(LFUCache, GetIncrementsFrequency) {
    LFUCache<int, int> c(2);
    c.put(1, 1);
    c.put(2, 2);
    c.get(1); c.get(1);   // freq[1]=3, freq[2]=1
    c.put(3, 3);           // evicts key 2 (freq=1), not key 1 (freq=3)
    EXPECT_TRUE(c.get(1).has_value());
    EXPECT_FALSE(c.get(2).has_value());
    EXPECT_TRUE(c.get(3).has_value());
}

// put() on an existing key updates its value AND increments its frequency.
TEST(LFUCache, PutExistingKeyIncrementsFrequency) {
    LFUCache<int, int> c(2);
    c.put(1, 10);           // freq[1]=1
    c.put(2, 20);           // freq[2]=1
    c.put(1, 11);           // update key 1 → freq[1]=2
    c.put(3, 30);           // full → evicts key 2 (freq=1)
    EXPECT_EQ(*c.get(1), 11);
    EXPECT_FALSE(c.get(2).has_value());
    EXPECT_TRUE(c.get(3).has_value());
}

TEST(LFUCache, EraseAndContains) {
    LFUCache<int, int> c(4);
    c.put(1, 1);
    EXPECT_TRUE(c.erase(1));
    EXPECT_FALSE(c.erase(1));
    EXPECT_FALSE(c.contains(1));
    EXPECT_EQ(c.size(), 0u);
}

TEST(LFUCache, TTLLazyExpiryOnGet) {
    LFUCache<int, int> c(4);
    c.put(1, 99, 50ms);
    std::this_thread::sleep_for(100ms);
    EXPECT_FALSE(c.get(1).has_value());
    EXPECT_EQ(c.size(), 0u);
}

TEST(LFUCache, PurgeExpired) {
    LFUCache<int, int> c(4);
    c.put(1, 1, 50ms);
    c.put(2, 2, 50ms);
    c.put(3, 3);
    std::this_thread::sleep_for(100ms);
    EXPECT_EQ(c.purge_expired(), 2u);
    EXPECT_EQ(c.size(), 1u);
    EXPECT_TRUE(c.get(3).has_value());
}

// An expired entry is evicted before a valid LFU entry.
TEST(LFUCache, ExpiredEntryEvictedBeforeValidLFU) {
    LFUCache<int, int> c(2);
    c.put(1, 1, 50ms);    // will expire
    c.put(2, 2);
    c.get(2); c.get(2);   // make key 2 high-frequency (freq=3)
    std::this_thread::sleep_for(100ms);
    c.put(3, 3);          // evicts expired key 1, not high-freq key 2
    EXPECT_FALSE(c.get(1).has_value());
    EXPECT_TRUE(c.get(2).has_value());
    EXPECT_TRUE(c.get(3).has_value());
}

TEST(LFUCache, NonCopyable) {
    static_assert(!std::is_copy_constructible<LFUCache<int, int>>::value);
    static_assert(!std::is_copy_assignable<LFUCache<int, int>>::value);
    static_assert(!std::is_move_constructible<LFUCache<int, int>>::value);
}

TEST(LFUCache, ConcurrentWritesNoCorruption) {
    LFUCache<int, int> c(200);
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&c, t]() {
            for (int i = 0; i < 50; ++i) c.put(t * 50 + i, i);
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(c.size(), 200u);
}

TEST(LFUCache, ConcurrentReadWriteNoCorruption) {
    LFUCache<int, int> c(50);
    for (int i = 0; i < 50; ++i) c.put(i, i);
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&c]() {
            for (int i = 0; i < 50; ++i) c.get(i);
        });
    }
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&c, t]() {
            for (int i = 0; i < 50; ++i) c.put(t * 50 + i, i);
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_LE(c.size(), 50u);
}
