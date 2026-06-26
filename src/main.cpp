#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "kvcache/lfu_cache.hpp"
#include "kvcache/lru_cache.hpp"
#include "kvcache/rate_limiter.hpp"
#include "kvcache/sharded_cache.hpp"

using namespace std::chrono_literals;

int main() {
    // ── Layers 0–2 recap ─────────────────────────────────────────────────────
    std::cout << "=== Layers 0-2 recap ===\n";
    {
        kvcache::LRUCache<std::string, std::string> cache(2);
        cache.put("a", "apple");
        cache.put("b", "banana");
        cache.get("a");
        cache.put("c", "cherry");
        std::cout << "get b      -> " << cache.get("b").value_or("MISS") << "  (LRU evicted)\n";

        kvcache::LRUCache<std::string, std::string> c2(4);
        c2.put("tok", "xyz", 80ms);
        std::this_thread::sleep_for(120ms);
        std::cout << "get tok    -> " << c2.get("tok").value_or("MISS") << "  (TTL expired)\n";
    }

    // ── Layer 3: basic ShardedCache usage ─────────────────────────────────────
    std::cout << "\n=== ShardedCache basics (Layer 3) ===\n";
    {
        kvcache::ShardedCache<std::string, std::string, 4> cache(400);
        cache.put("hello", "world");
        cache.put("foo",   "bar", 500ms);

        std::cout << "num_shards = " << cache.num_shards() << "\n";
        std::cout << "capacity   = " << cache.capacity()   << "  (4 shards × 100)\n";
        std::cout << "get hello  -> " << cache.get("hello").value_or("MISS") << "\n";
        std::cout << "get foo    -> " << cache.get("foo").value_or("MISS")   << "\n";
        std::cout << "erase foo  -> " << std::boolalpha << cache.erase("foo") << "\n";
        std::cout << "get foo    -> " << cache.get("foo").value_or("MISS")   << "  (erased)\n";
    }

    // ── Layer 3: throughput comparison ───────────────────────────────────────
    // Same workload on a single-mutex cache vs an 8-shard cache.
    // With 8 threads and 8 shards, threads that land on different shards run
    // in parallel — no waiting. The single-mutex version serialises all of them.
    std::cout << "\n=== Throughput: single mutex vs sharded ===\n";
    {
        constexpr int THREADS = 8;
        constexpr int OPS     = 100'000;

        // Each thread owns a disjoint key range of size KEY_RANGE.
        // Thread t touches keys [t*KEY_RANGE, t*KEY_RANGE + KEY_RANGE).
        // With 8 shards those ranges land on different shards, so threads
        // rarely contend — that is when sharding actually helps.
        constexpr int KEY_RANGE = 1000;   // keys per thread
        constexpr int CAPACITY  = THREADS * KEY_RANGE;

        auto bench = [](const char* label, auto& cache, int threads,
                        int ops, int key_range) {
            auto t0 = std::chrono::high_resolution_clock::now();

            std::vector<std::thread> ts;
            for (int t = 0; t < threads; ++t) {
                ts.emplace_back([&cache, t, ops, key_range]() {
                    const int base = t * key_range;
                    for (int i = 0; i < ops; ++i) {
                        int key = base + (i % key_range);
                        if (i % 4 == 0) cache.put(key, key);
                        else            cache.get(key);
                    }
                });
            }
            for (auto& th : ts) th.join();

            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::high_resolution_clock::now() - t0).count();
            std::cout << label << ": " << ms << " ms\n";
        };

        kvcache::LRUCache<int, int>        single(CAPACITY);
        kvcache::ShardedCache<int, int, 8> sharded(CAPACITY);

        bench("LRUCache     (1 mutex,  8 threads)", single,  THREADS, OPS, KEY_RANGE);
        bench("ShardedCache (8 shards, 8 threads)", sharded, THREADS, OPS, KEY_RANGE);

        std::cout << "(each thread owns a disjoint key range — threads land on\n"
                  << " different shards and run in parallel)\n";
    }

    // ── Layer 5: LFU eviction ─────────────────────────────────────────────────
    // The key question: which entry gets evicted when the cache is full?
    // LRU: the entry not accessed for the longest time.
    // LFU: the entry accessed the fewest total times.
    std::cout << "\n=== LFU vs LRU eviction (Layer 5) ===\n";
    {
        // Setup: cache size 3.
        // A is accessed 3 times (very "hot"), B twice, C once (just inserted).
        // Then D arrives and forces an eviction.
        //
        // LRU evicts A — it was the least recently used because B and C came after.
        // LFU evicts C — it has the lowest frequency (1), protecting the hot entry A.
        auto setup = [](auto& cache) {
            cache.put("A", 1);
            cache.get("A");
            cache.get("A");   // A.freq = 3 (put counts as freq=1, two gets → 3)
            cache.put("B", 2);
            cache.get("B");   // B.freq = 2
            cache.put("C", 3);// C.freq = 1  — LRU order after all this: [C, B, A]
        };

        kvcache::LRUCache<std::string, int> lru(3);
        kvcache::LFUCache<std::string, int> lfu(3);

        setup(lru);
        setup(lfu);

        lru.put("D", 4);  // triggers eviction
        lfu.put("D", 4);  // triggers eviction

        std::cout << "After put(A)×1+get(A)×2, put(B)+get(B), put(C), put(D):\n";
        std::cout << "  LRU evicts: " << (!lru.contains("A") ? "A" :
                                          !lru.contains("B") ? "B" :
                                          !lru.contains("C") ? "C" : "?")
                  << "  (least recently used — but A was the hottest key!)\n";
        std::cout << "  LFU evicts: " << (!lfu.contains("A") ? "A" :
                                          !lfu.contains("B") ? "B" :
                                          !lfu.contains("C") ? "C" : "?")
                  << "  (least frequently used — protects hot key A)\n";
    }

    // LFU weakness: cache pollution from historically popular but now stale keys.
    std::cout << "\n=== LFU cache pollution (the trade-off) ===\n";
    {
        kvcache::LFUCache<std::string, int> lfu(2);
        lfu.put("old", 1);
        for (int i = 0; i < 10; ++i) lfu.get("old");  // old.freq = 11

        lfu.put("new1", 2);  // new1.freq = 1 — immediately at risk
        lfu.put("new2", 3);  // triggers eviction: evicts new1 (freq=1), not old (freq=11)

        std::cout << "old (freq=11, never accessed again): "
                  << (lfu.contains("old")  ? "still in cache" : "evicted") << "\n";
        std::cout << "new1 (freq=1, just inserted):        "
                  << (lfu.contains("new1") ? "still in cache" : "evicted") << "\n";
        std::cout << "(LFU keeps stale 'old' because of its historical frequency)\n";
    }

    // ── Layer 7: Rate Limiter ─────────────────────────────────────────────────
    std::cout << "\n=== Rate Limiter (Layer 7) ===\n";
    {
        // Token bucket: capacity=5, rate=10/s.
        // The first 5 requests (burst) are always allowed regardless of timing.
        // After the burst, requests are allowed only as tokens refill.
        std::cout << "\n-- Token bucket (capacity=5, rate=10/sec) --\n";
        {
            kvcache::TokenBucketLimiter<std::string> tb(5.0, 10.0);
            std::cout << "Burst of 5:    ";
            for (int i = 0; i < 5; ++i)
                std::cout << (tb.allow("user1") ? "ok " : "DENY ");
            std::cout << "\n";
            std::cout << "Request 6:     " << (tb.allow("user1") ? "ok" : "DENY")
                      << "  (bucket empty)\n";
            std::this_thread::sleep_for(300ms);
            std::cout << "After 300ms:   " << (tb.allow("user1") ? "ok" : "DENY")
                      << "  (~3 tokens refilled, 1 consumed)\n";
        }

        // Sliding window: limit=5 per 500ms.
        // Strict count — no burst credit. Once the window fills, all further
        // requests are denied until the window slides past the earliest entry.
        std::cout << "\n-- Sliding window (limit=5 per 500ms) --\n";
        {
            kvcache::SlidingWindowLimiter<std::string> sw(5, 500ms);
            std::cout << "First 5:       ";
            for (int i = 0; i < 5; ++i)
                std::cout << (sw.allow("user1") ? "ok " : "DENY ");
            std::cout << "\n";
            std::cout << "Request 6:     " << (sw.allow("user1") ? "ok" : "DENY")
                      << "  (window full)\n";
            std::this_thread::sleep_for(600ms);
            std::cout << "After 600ms:   " << (sw.allow("user1") ? "ok" : "DENY")
                      << "  (window slid past all old entries)\n";
        }

        std::cout << "\n-- Trade-off --\n";
        std::cout << "Token bucket:   burst up to capacity allowed; "
                     "sustained rate is bounded by refill speed\n";
        std::cout << "Sliding window: no burst; exact count per rolling window\n";
        std::cout << "Use token bucket for APIs where short spikes are acceptable.\n";
        std::cout << "Use sliding window for abuse prevention where strict counts matter.\n";
    }

    return 0;
}
