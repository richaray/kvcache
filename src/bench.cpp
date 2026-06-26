#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "kvcache/lfu_cache.hpp"
#include "kvcache/lru_cache.hpp"
#include "kvcache/sharded_cache.hpp"

// ── Config & Result ──────────────────────────────────────────────────────────

struct Config {
    int  threads        = 4;
    int  ops_per_thread = 200'000;
    int  key_space      = 10'000;
    int  read_pct       = 80;    // % of ops that are gets (rest are puts)
    bool hotspot        = false; // true → 80% of ops target the hot 10% of keys
};

struct Result {
    long long total_ops  = 0;
    double    throughput = 0;  // ops/sec
    long long p50_ns     = 0;
    long long p95_ns     = 0;
    long long p99_ns     = 0;
    long long max_ns     = 0;
};

// ── Benchmark runner (generic over any cache type) ───────────────────────────
//
// Measures:
//   Throughput — total operations / wall-clock time (most reliable)
//   Latency    — per-operation time sampled by high_resolution_clock.
//                Includes ~10-30 ns chrono overhead; treat relative
//                comparisons as meaningful, not absolute values.

template <typename Cache>
Result run(const Config& cfg, Cache& cache) {
    // Warm up: fill the cache so subsequent gets are real hits, not misses.
    for (int i = 0; i < cfg.key_space; ++i)
        cache.put(i, i);

    // One latency vector per thread — no cross-thread writes, no data race.
    std::vector<std::vector<long long>> lat(cfg.threads);
    for (auto& v : lat) v.reserve(cfg.ops_per_thread);

    auto wall_start = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    for (int t = 0; t < cfg.threads; ++t) {
        threads.emplace_back([&lat, &cache, &cfg, t]() {
            // Each thread gets its own seeded RNG — no shared state.
            std::mt19937 rng(static_cast<uint32_t>(t) * 2654435761u + 1);
            std::uniform_int_distribution<int> full_dist(0, cfg.key_space - 1);
            std::uniform_int_distribution<int> hot_dist(0, cfg.key_space / 10 - 1);
            std::uniform_int_distribution<int> pct(0, 99);

            for (int i = 0; i < cfg.ops_per_thread; ++i) {
                // Key selection: hotspot sends 80% of traffic to 10% of keys.
                int key = (cfg.hotspot && pct(rng) < 80) ? hot_dist(rng)
                                                          : full_dist(rng);

                auto t0 = std::chrono::high_resolution_clock::now();
                if (pct(rng) < cfg.read_pct) cache.get(key);
                else                          cache.put(key, key);
                auto t1 = std::chrono::high_resolution_clock::now();

                lat[t].push_back(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        t1 - t0).count());
            }
        });
    }
    for (auto& th : threads) th.join();

    long long wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - wall_start).count();

    // Flatten all per-thread latency samples and sort for percentiles.
    std::vector<long long> all;
    all.reserve(static_cast<std::size_t>(cfg.threads) * cfg.ops_per_thread);
    for (auto& v : lat) all.insert(all.end(), v.begin(), v.end());
    std::sort(all.begin(), all.end());

    const std::size_t n = all.size();
    Result r;
    r.total_ops  = static_cast<long long>(n);
    r.throughput = static_cast<double>(n) / (wall_ns / 1e9);
    r.p50_ns     = all[n * 50 / 100];
    r.p95_ns     = all[n * 95 / 100];
    r.p99_ns     = all[n * 99 / 100];
    r.max_ns     = all.back();
    return r;
}

// ── Output helpers ───────────────────────────────────────────────────────────

void print(const char* label, const Result& r) {
    std::printf("  %-46s tput: %6.1f Mops/s   p50: %5lld ns   p95: %5lld ns   p99: %5lld ns\n",
                label,
                r.throughput / 1e6,
                r.p50_ns, r.p95_ns, r.p99_ns);
}

void section(const char* title) {
    std::printf("\n── %s ──\n", title);
}

// ── Scenarios ────────────────────────────────────────────────────────────────

int main() {
    std::printf("kvcache benchmark\n");
    std::printf("Build in Release mode for meaningful numbers.\n");
    std::printf("Latency includes ~10-30 ns chrono overhead — use throughput for comparisons.\n");

    // ── 1. Op mix: read-heavy vs write-heavy ─────────────────────────────────
    // Real caches are typically 90-95% reads. Write-heavy is the stress case.
    section("Op mix  |  8 threads  |  uniform keys  |  10K key space");
    {
        Config cfg;
        cfg.threads = 8; cfg.key_space = 10'000;

        for (int rpct : {95, 50}) {
            cfg.read_pct = rpct;
            char label[64];

            std::snprintf(label, sizeof(label),
                          "LRUCache     %2d%% read / %2d%% write", rpct, 100 - rpct);
            { kvcache::LRUCache<int,int> c(cfg.key_space); print(label, run(cfg, c)); }

            std::snprintf(label, sizeof(label),
                          "LFUCache     %2d%% read / %2d%% write", rpct, 100 - rpct);
            { kvcache::LFUCache<int,int> c(cfg.key_space); print(label, run(cfg, c)); }

            std::snprintf(label, sizeof(label),
                          "ShardedCache %2d%% read / %2d%% write", rpct, 100 - rpct);
            { kvcache::ShardedCache<int,int,8> c(cfg.key_space); print(label, run(cfg, c)); }
        }
    }

    // ── 2. Key distribution: uniform vs hotspot ───────────────────────────────
    // Hotspot (Zipfian-like): 80% of traffic hits 10% of keys.
    // This is the common real-world pattern — think "trending" items.
    // Under hotspot, all threads contend on the same hot shard → sharding hurts.
    section("Key dist  |  8 threads  |  80% read  |  10K key space");
    {
        Config cfg;
        cfg.threads = 8; cfg.key_space = 10'000; cfg.read_pct = 80;

        cfg.hotspot = false;
        { kvcache::LRUCache<int,int>       c(cfg.key_space); print("LRUCache     uniform",  run(cfg, c)); }
        { kvcache::LFUCache<int,int>       c(cfg.key_space); print("LFUCache     uniform",  run(cfg, c)); }
        { kvcache::ShardedCache<int,int,8> c(cfg.key_space); print("ShardedCache uniform",  run(cfg, c)); }

        cfg.hotspot = true;
        { kvcache::LRUCache<int,int>       c(cfg.key_space); print("LRUCache     hotspot",  run(cfg, c)); }
        { kvcache::LFUCache<int,int>       c(cfg.key_space); print("LFUCache     hotspot",  run(cfg, c)); }
        { kvcache::ShardedCache<int,int,8> c(cfg.key_space); print("ShardedCache hotspot",  run(cfg, c)); }
    }

    // ── 3. Thread scaling ─────────────────────────────────────────────────────
    // Shows how each cache type scales as thread count increases.
    // Single mutex should plateau (or degrade) past 2 threads.
    // Sharded should scale better at higher thread counts.
    section("Thread scaling  |  80% read  |  uniform  |  10K key space");
    {
        Config cfg;
        cfg.key_space = 10'000; cfg.read_pct = 80;

        for (int t : {1, 2, 4, 8}) {
            cfg.threads = t;
            char label[64];

            std::snprintf(label, sizeof(label),
                          "LRUCache     %d thread%s", t, t > 1 ? "s" : " ");
            { kvcache::LRUCache<int,int>       c(cfg.key_space); print(label, run(cfg, c)); }

            std::snprintf(label, sizeof(label),
                          "LFUCache     %d thread%s", t, t > 1 ? "s" : " ");
            { kvcache::LFUCache<int,int>       c(cfg.key_space); print(label, run(cfg, c)); }

            std::snprintf(label, sizeof(label),
                          "ShardedCache %d thread%s", t, t > 1 ? "s" : " ");
            { kvcache::ShardedCache<int,int,8> c(cfg.key_space); print(label, run(cfg, c)); }
        }
    }

    return 0;
}
