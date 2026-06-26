#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "kvcache/rate_limiter.hpp"

using namespace std::chrono_literals;
using kvcache::SlidingWindowLimiter;
using kvcache::TokenBucketLimiter;

// ── TokenBucketLimiter ───────────────────────────────────────────────────────

TEST(TokenBucket, AllowsBurst) {
    TokenBucketLimiter<int> limiter(5.0, 0.0);  // 5 tokens, no refill
    for (int i = 0; i < 5; ++i) EXPECT_TRUE(limiter.allow(1));
    EXPECT_FALSE(limiter.allow(1));
}

TEST(TokenBucket, DeniesWhenBucketEmpty) {
    TokenBucketLimiter<int> limiter(3.0, 0.0);
    limiter.allow(1); limiter.allow(1); limiter.allow(1);
    EXPECT_FALSE(limiter.allow(1));
    EXPECT_FALSE(limiter.allow(1));
}

TEST(TokenBucket, RefillsOverTime) {
    // rate=20/s: after 400ms we recover at least 7 tokens (capped at 5).
    // Exhaust all 5, sleep, then 5 more should all pass.
    TokenBucketLimiter<int> limiter(5.0, 20.0);
    for (int i = 0; i < 5; ++i) limiter.allow(1);  // drain
    std::this_thread::sleep_for(400ms);
    for (int i = 0; i < 5; ++i) EXPECT_TRUE(limiter.allow(1));
}

TEST(TokenBucket, TokensAreCapedAtCapacity) {
    TokenBucketLimiter<int> limiter(3.0, 100.0);
    std::this_thread::sleep_for(200ms);  // would add 20 tokens but cap is 3
    EXPECT_LE(limiter.tokens_remaining(1), 3.0);
    for (int i = 0; i < 3; ++i) EXPECT_TRUE(limiter.allow(1));
    EXPECT_FALSE(limiter.allow(1));
}

TEST(TokenBucket, MultipleKeysAreIndependent) {
    TokenBucketLimiter<int> limiter(2.0, 0.0);
    limiter.allow(1); limiter.allow(1);   // drain key 1
    EXPECT_FALSE(limiter.allow(1));        // key 1 empty
    EXPECT_TRUE(limiter.allow(2));         // key 2 untouched — still full
}

TEST(TokenBucket, CostParameter) {
    TokenBucketLimiter<int> limiter(5.0, 0.0);
    EXPECT_TRUE(limiter.allow(1, 3.0));   // costs 3 tokens (2 remaining)
    EXPECT_FALSE(limiter.allow(1, 3.0));  // needs 3, only 2 left
    EXPECT_TRUE(limiter.allow(1, 2.0));   // costs 2 (0 remaining)
}

TEST(TokenBucket, Reset) {
    TokenBucketLimiter<int> limiter(2.0, 0.0);
    limiter.allow(1); limiter.allow(1);
    EXPECT_FALSE(limiter.allow(1));
    limiter.reset(1);
    EXPECT_TRUE(limiter.allow(1));   // fresh bucket after reset
}

TEST(TokenBucket, TokensRemaining) {
    TokenBucketLimiter<int> limiter(5.0, 0.0);
    EXPECT_DOUBLE_EQ(limiter.tokens_remaining(1), 5.0);
    limiter.allow(1); limiter.allow(1);
    EXPECT_DOUBLE_EQ(limiter.tokens_remaining(1), 3.0);
}

TEST(TokenBucket, SizeTracksDistinctKeys) {
    TokenBucketLimiter<int> limiter(5.0, 0.0);
    limiter.allow(10); limiter.allow(20); limiter.allow(30);
    EXPECT_EQ(limiter.size(), 3u);
    limiter.reset(20);
    EXPECT_EQ(limiter.size(), 2u);
}

// Critical correctness property: concurrent allow() on the same key must never
// approve more requests than the bucket capacity allows.
// Rate=0 means no refill, so initial 10 tokens are the only ones available.
// 10 threads × 5 attempts = 50 total tries; exactly 10 should succeed.
TEST(TokenBucket, ConcurrentAllowNeverExceedsCapacity) {
    TokenBucketLimiter<int> limiter(10.0, 0.0);
    std::atomic<int> approved{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 10; ++t) {
        threads.emplace_back([&limiter, &approved]() {
            for (int i = 0; i < 5; ++i)
                if (limiter.allow(42)) ++approved;
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(approved.load(), 10);
}

TEST(TokenBucket, ConcurrentDistinctKeysNoCorruption) {
    TokenBucketLimiter<int> limiter(100.0, 0.0);
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&limiter, t]() {
            for (int i = 0; i < 50; ++i) limiter.allow(t);
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(limiter.size(), 8u);
}

// ── SlidingWindowLimiter ─────────────────────────────────────────────────────

TEST(SlidingWindow, AllowsUpToLimit) {
    SlidingWindowLimiter<int> limiter(5, 1s);
    for (int i = 0; i < 5; ++i) EXPECT_TRUE(limiter.allow(1));
}

TEST(SlidingWindow, DeniesWhenLimitReached) {
    SlidingWindowLimiter<int> limiter(3, 1s);
    limiter.allow(1); limiter.allow(1); limiter.allow(1);
    EXPECT_FALSE(limiter.allow(1));
    EXPECT_FALSE(limiter.allow(1));
}

TEST(SlidingWindow, AllowsAfterWindowSlides) {
    SlidingWindowLimiter<int> limiter(3, 100ms);
    limiter.allow(1); limiter.allow(1); limiter.allow(1);
    EXPECT_FALSE(limiter.allow(1));
    std::this_thread::sleep_for(200ms);   // window fully expired
    for (int i = 0; i < 3; ++i) EXPECT_TRUE(limiter.allow(1));
}

TEST(SlidingWindow, RequestsInWindow) {
    SlidingWindowLimiter<int> limiter(10, 1s);
    EXPECT_EQ(limiter.requests_in_window(1), 0);
    limiter.allow(1); limiter.allow(1); limiter.allow(1);
    EXPECT_EQ(limiter.requests_in_window(1), 3);
}

TEST(SlidingWindow, RequestsInWindowDropsAfterExpiry) {
    SlidingWindowLimiter<int> limiter(10, 100ms);
    limiter.allow(1); limiter.allow(1);
    std::this_thread::sleep_for(200ms);
    EXPECT_EQ(limiter.requests_in_window(1), 0);
}

TEST(SlidingWindow, MultipleKeysAreIndependent) {
    SlidingWindowLimiter<int> limiter(2, 1s);
    limiter.allow(1); limiter.allow(1);
    EXPECT_FALSE(limiter.allow(1));
    EXPECT_TRUE(limiter.allow(2));  // key 2 has its own empty window
}

TEST(SlidingWindow, Reset) {
    SlidingWindowLimiter<int> limiter(2, 1s);
    limiter.allow(1); limiter.allow(1);
    EXPECT_FALSE(limiter.allow(1));
    limiter.reset(1);
    EXPECT_TRUE(limiter.allow(1));
}

TEST(SlidingWindow, SizeTracksDistinctKeys) {
    SlidingWindowLimiter<int> limiter(10, 1s);
    limiter.allow(10); limiter.allow(20);
    EXPECT_EQ(limiter.size(), 2u);
}

// Concurrent allow on the same key: total approved must not exceed the limit.
// (Window is 10 seconds so it won't slide during the test.)
TEST(SlidingWindow, ConcurrentAllowNeverExceedsLimit) {
    SlidingWindowLimiter<int> limiter(10, 10s);
    std::atomic<int> approved{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 10; ++t) {
        threads.emplace_back([&limiter, &approved]() {
            for (int i = 0; i < 5; ++i)
                if (limiter.allow(42)) ++approved;
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(approved.load(), 10);
}

TEST(SlidingWindow, ConcurrentDistinctKeysNoCorruption) {
    SlidingWindowLimiter<int> limiter(1000, 10s);
    std::vector<std::thread> threads;
    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&limiter, t]() {
            for (int i = 0; i < 50; ++i) limiter.allow(t);
        });
    }
    for (auto& th : threads) th.join();
    EXPECT_EQ(limiter.size(), 8u);
}
