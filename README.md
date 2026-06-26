# kvcache — in-memory key-value cache in C++

A from-scratch, thread-safe, in-memory key-value cache in modern C++17, built in independent layers. Each layer adds exactly one concept and is fully tested and benchmarked.

## Benchmark results

Measured on 8 threads, 10K key space, 80% reads, Release build:

| Implementation | Throughput | p99 latency |
|---|---|---|
| LRUCache (single mutex) | 1.0 Mops/s | 166 µs |
| LFUCache (single mutex) | 0.7 Mops/s | 199 µs |
| ShardedCache (8 shards) | **3.0 Mops/s** | **19 µs** |

Sharded locking delivers **3× throughput** and **9× lower p99** vs a single mutex under 8-thread load. At 1 thread all three are equivalent (~3 Mops/s) — the gap only opens under contention.

## Layers

| Layer | Adds | Concept |
|---|---|---|
| 0 | LRU cache, O(1) get/put | doubly-linked list + hash map eviction |
| 1 | TTL expiry | lazy expiry on get + active `purge_expired()` sweep |
| 2 | Thread safety | `std::mutex`, RAII lock guards, `*Unlocked` helper pattern |
| 3 | Sharded locks | N independent shards, lock granularity vs throughput |
| 4 | Benchmark harness | throughput (Mops/s), p50/p95/p99 latency percentiles |
| 5 | LFU eviction | O(1) LFU via frequency bucket list, LRU vs LFU trade-offs |
| 6 | Rate limiter | token bucket (burst-friendly) + sliding window (strict) |

## How the key data structures work

**LRU cache** — `std::list` keeps entries in usage order (front = MRU, back = eviction victim). An `unordered_map<Key, list-iterator>` gives O(1) lookup and splice. On every `get`, the entry is spliced to the front in O(1).

**LFU cache** — two hash maps + a per-frequency list (Shah et al. 2010). `index_` maps each key to its node. `freq_map_[f]` holds all nodes at frequency `f` in LRU order. `min_freq_` tracks the eviction target. `list::splice` moves a node between frequency buckets in O(1) while preserving iterator validity.

**Token bucket rate limiter** — each key has a token count and a last-refill timestamp. On every `allow()`, elapsed time is computed and tokens are added (`min(capacity, tokens + elapsed * rate)`). Allows bursting up to capacity.

**Sliding window rate limiter** — each key has a deque of request timestamps. Old entries (outside the window) are dropped on each call. Strict per-window counting with no burst allowance.

## Build & run

Requirements: C++17 compiler, CMake ≥ 3.14. GoogleTest is fetched automatically on first configure.

```bash
# Configure (only needed once)
cmake -S . -B build_rel

# Build (Windows/MSVC)
cmake --build build_rel -j --config Release

# Run
./build_rel/Release/demo.exe        # behavioral walk-through of every layer
./build_rel/Release/bench.exe       # throughput + latency benchmark
./build_rel/Release/lru_tests.exe   # 45 cache tests
./build_rel/Release/rate_tests.exe  # 21 rate limiter tests
```

On Linux/macOS replace `--config Release` with `-DCMAKE_BUILD_TYPE=Release` at configure time, and omit `/Release` from the binary paths.

## Project layout

```
kvcache/
├── CMakeLists.txt
├── include/kvcache/
│   ├── lru_cache.hpp       # LRU cache (Layers 0-2)
│   ├── sharded_cache.hpp   # sharded lock wrapper (Layer 3)
│   ├── lfu_cache.hpp       # LFU cache (Layer 5)
│   └── rate_limiter.hpp    # token bucket + sliding window (Layer 6)
├── src/
│   ├── main.cpp            # demo: walks through every layer
│   └── bench.cpp           # benchmark harness
└── tests/
    ├── test_lru_cache.cpp  # 45 tests: LRU, TTL, threads, sharding, LFU
    └── test_rate_limiter.cpp # 21 tests: token bucket + sliding window
```
