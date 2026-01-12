// Stress tests for unified_cache concurrent access.
//
// Usage:
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-unified-cache-concurrent

#include "ggml-sycl.h"
#include "ggml-sycl/unified-cache.hpp"
#include "ggml.h"

#include <atomic>
#include <cstdio>
#include <random>
#include <sycl/sycl.hpp>
#include <thread>
#include <vector>

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

static void spin_wait(std::atomic<bool> & start) {
    while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

static bool test_concurrent_ensure_cached(sycl::queue & q) {
    printf("\n=== Test: concurrent ensure_cached ===\n");

    ggml_sycl::unified_cache          cache(q, 2 * 1024 * 1024);
    std::vector<std::vector<uint8_t>> payloads(64, std::vector<uint8_t>(256, 0x11));

    const int                threads = 8;
    const int                iters   = 200;
    std::atomic<bool>        start{ false };
    std::atomic<bool>        ok{ true };
    std::vector<std::thread> workers;
    workers.reserve(threads);

    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&cache, &payloads, &start, &ok, t]() {
            std::mt19937                          rng(0x1234u + static_cast<unsigned int>(t));
            std::uniform_int_distribution<size_t> dist(0, payloads.size() - 1);
            spin_wait(start);
            for (int i = 0; i < iters; ++i) {
                size_t idx = dist(rng);
                void * ptr =
                    cache.ensure_cached(payloads[idx].data(), payloads[idx].data(), payloads[idx].size(),
                                        ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false);
                if (!ptr) {
                    ok.store(false, std::memory_order_release);
                    return;
                }
            }
        });
    }

    start.store(true, std::memory_order_release);
    for (auto & worker : workers) {
        worker.join();
    }

    if (!ok.load(std::memory_order_acquire)) {
        fprintf(stderr, "Concurrent ensure_cached returned nullptr\n");
        return false;
    }

    if (cache.used() > cache.budget()) {
        fprintf(stderr, "Cache used() exceeded budget after concurrent ensure_cached\n");
        return false;
    }

    return true;
}

static bool test_concurrent_evict(sycl::queue & q) {
    printf("\n=== Test: concurrent evict ===\n");

    ggml_sycl::unified_cache          cache(q, 4096);
    std::vector<std::vector<uint8_t>> payloads(4, std::vector<uint8_t>(1024, 0x22));

    bool needs_fill = false;
    for (auto & payload : payloads) {
        void * ptr = cache.ensure_cached_alloc(payload.data(), payload.data(), payload.size(), payload.size(),
                                               ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS,
                                               false, &needs_fill);
        if (!ptr) {
            fprintf(stderr, "Failed to allocate entry for concurrent evict test\n");
            return false;
        }
        cache.pin(payload.data(), GGML_LAYOUT_AOS);
    }

    const size_t used_before = cache.used();

    const int                threads = 4;
    const int                iters   = 100;
    std::atomic<bool>        start{ false };
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&cache, &start]() {
            spin_wait(start);
            for (int i = 0; i < iters; ++i) {
                cache.evict(1024);
            }
        });
    }

    start.store(true, std::memory_order_release);
    for (auto & worker : workers) {
        worker.join();
    }

    if (cache.used() != used_before) {
        fprintf(stderr, "Pinned entries evicted unexpectedly (used_before=%zu used_after=%zu)\n", used_before,
                cache.used());
        return false;
    }

    for (auto & payload : payloads) {
        if (!cache.is_cached(payload.data(), GGML_LAYOUT_AOS)) {
            fprintf(stderr, "Pinned entry missing after concurrent evict\n");
            return false;
        }
    }

    return true;
}

static bool test_mixed_ops(sycl::queue & q) {
    printf("\n=== Test: mixed concurrent ops ===\n");

    ggml_sycl::unified_cache          cache(q, 2 * 1024 * 1024);
    std::vector<std::vector<uint8_t>> payloads(64, std::vector<uint8_t>(256, 0x33));

    const int                threads = 8;
    const int                iters   = 200;
    std::atomic<bool>        start{ false };
    std::atomic<bool>        ok{ true };
    std::vector<std::thread> workers;
    workers.reserve(threads);

    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&cache, &payloads, &start, &ok, t]() {
            std::mt19937                          rng(0x9e37u + static_cast<unsigned int>(t));
            std::uniform_int_distribution<size_t> key_dist(0, payloads.size() - 1);
            std::uniform_int_distribution<int>    op_dist(0, 99);
            spin_wait(start);
            for (int i = 0; i < iters; ++i) {
                size_t idx = key_dist(rng);
                int    op  = op_dist(rng);
                if (op < 60) {
                    void * ptr =
                        cache.ensure_cached(payloads[idx].data(), payloads[idx].data(), payloads[idx].size(),
                                            ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1, GGML_LAYOUT_AOS, false);
                    if (!ptr) {
                        ok.store(false, std::memory_order_release);
                        return;
                    }
                } else if (op < 75) {
                    cache.pin(payloads[idx].data(), GGML_LAYOUT_AOS);
                } else if (op < 90) {
                    cache.unpin(payloads[idx].data(), GGML_LAYOUT_AOS);
                } else if (op < 95) {
                    cache.evict(256);
                } else {
                    cache.remove(payloads[idx].data(), ggml_sycl::cache_entry_type::DENSE_WEIGHT, -1, -1,
                                 GGML_LAYOUT_AOS);
                }
            }
        });
    }

    start.store(true, std::memory_order_release);
    for (auto & worker : workers) {
        worker.join();
    }

    cache.evict(0);

    if (!ok.load(std::memory_order_acquire)) {
        fprintf(stderr, "Mixed ops returned nullptr from ensure_cached\n");
        return false;
    }

    if (cache.used() > cache.budget()) {
        fprintf(stderr, "Cache used() exceeded budget after mixed ops\n");
        return false;
    }

    return true;
}

int main() {
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);
    }

    sycl::queue q;
    try {
        printf("Using device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    } catch (const sycl::exception & e) {
        fprintf(stderr, "SYCL error: %s\n", e.what());
        return 1;
    }

    bool ok = true;
    ok &= test_concurrent_ensure_cached(q);
    ok &= test_concurrent_evict(q);
    ok &= test_mixed_ops(q);

    printf("\nUnified cache concurrency tests: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

#endif
