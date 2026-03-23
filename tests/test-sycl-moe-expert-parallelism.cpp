// SYCL MoE Expert Parallelism Unit Tests
//
// Pure C++ unit tests for MoE expert dispatch data structures:
// 1. Expert popularity ranking: set/get roundtrip, thread safety
// 2. Key uniqueness: different (layer_id, expert_id) pairs produce independent ranks
// 3. Initialization state tracking
//
// No GPU dependency — tests only the CPU-side data structures.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <thread>
#include <vector>
#include <unordered_set>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl/unified-cache.hpp"

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

// Test counters
static int g_tests_run    = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, msg)                                                 \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "  FAIL: %s\n", msg);                              \
            return false;                                                      \
        }                                                                      \
    } while (0)

#define RUN_TEST(fn)                                                           \
    do {                                                                       \
        g_tests_run++;                                                         \
        if (fn()) {                                                            \
            g_tests_passed++;                                                  \
            printf("  PASS: %s\n", #fn);                                       \
        } else {                                                               \
            g_tests_failed++;                                                  \
            fprintf(stderr, "  FAIL: %s\n", #fn);                              \
        }                                                                      \
    } while (0)

// ========================================================================
// Test 1: Popularity ranking basic init state
// ========================================================================
static bool test_popularity_init_state() {
    // Before any ranks are set, is_initialized should return false
    // and get should return -1 (unranked).
    // Note: if prior tests ran in the same process, this may already be true.
    // We just verify that get returns -1 for unknown entries.
    int rank = ggml_sycl::get_expert_popularity_rank(99999, 99999);
    TEST_ASSERT(rank == -1, "unknown entry should return rank -1 (unranked)");
    return true;
}

// ========================================================================
// Test 2: Set/get roundtrip for popularity ranking
// ========================================================================
static bool test_popularity_set_get_roundtrip() {
    const int layer_id  = 42;
    const int expert_id = 7;

    ggml_sycl::set_expert_popularity_rank(layer_id, expert_id, 3);
    int rank = ggml_sycl::get_expert_popularity_rank(layer_id, expert_id);
    TEST_ASSERT(rank == 3, "popularity_rank roundtrip should be 3");

    // Update to a different rank
    ggml_sycl::set_expert_popularity_rank(layer_id, expert_id, 0);
    rank = ggml_sycl::get_expert_popularity_rank(layer_id, expert_id);
    TEST_ASSERT(rank == 0, "popularity_rank should be updated to 0");

    // Getting a non-existent entry returns -1
    int missing_rank = ggml_sycl::get_expert_popularity_rank(88888, 77777);
    TEST_ASSERT(missing_rank == -1, "missing entry should return -1");

    return true;
}

// ========================================================================
// Test 3: is_expert_popularity_initialized tracks state
// ========================================================================
static bool test_popularity_initialized() {
    // After setting at least one rank (from test 2), should be initialized
    TEST_ASSERT(ggml_sycl::is_expert_popularity_initialized(),
                "should be initialized after set_expert_popularity_rank");
    return true;
}

// ========================================================================
// Test 4: Key uniqueness — different (layer_id, expert_id) pairs must
//         produce independent popularity entries
// ========================================================================
static bool test_popularity_key_uniqueness() {
    const int n_layers  = 32;
    const int n_experts = 8;

    // Set ranks with unique values per (layer, expert)
    for (int l = 0; l < n_layers; l++) {
        for (int e = 0; e < n_experts; e++) {
            int rank = l * 100 + e;
            ggml_sycl::set_expert_popularity_rank(l + 10000, e, rank);
        }
    }

    // Verify each entry is independent
    for (int l = 0; l < n_layers; l++) {
        for (int e = 0; e < n_experts; e++) {
            int expected = l * 100 + e;
            int got = ggml_sycl::get_expert_popularity_rank(l + 10000, e);
            TEST_ASSERT(got == expected, "rank should be unique per entry");
        }
    }

    // Verify that swapped (layer_id, expert_id) are different keys
    ggml_sycl::set_expert_popularity_rank(20001, 20002, 111);
    ggml_sycl::set_expert_popularity_rank(20002, 20001, 222);

    int r1 = ggml_sycl::get_expert_popularity_rank(20001, 20002);
    int r2 = ggml_sycl::get_expert_popularity_rank(20002, 20001);
    TEST_ASSERT(r1 == 111, "swapped key 1 rank");
    TEST_ASSERT(r2 == 222, "swapped key 2 rank");

    return true;
}

// ========================================================================
// Test 5: FNV hash-based layer IDs (large values typical of moe_cache_layer_id)
// ========================================================================
static bool test_popularity_hash_based_ids() {
    // Simulate FNV-1a 32-bit hash values
    const int layer_ids[] = {
        static_cast<int>(2166136261u),  // FNV offset basis
        static_cast<int>(0xCAFEBABEu),
        static_cast<int>(0xDEADBEEFu),
        static_cast<int>(0x00000001u),
        static_cast<int>(0xFFFFFFFFu),  // -1 as signed int
        0,
    };

    for (int i = 0; i < 6; i++) {
        ggml_sycl::set_expert_popularity_rank(layer_ids[i], 0, i * 10);
    }

    for (int i = 0; i < 6; i++) {
        int got = ggml_sycl::get_expert_popularity_rank(layer_ids[i], 0);
        TEST_ASSERT(got == i * 10, "hash-based layer_id roundtrip");
    }

    return true;
}

// ========================================================================
// Test 6: Thread safety — concurrent readers and writers
// ========================================================================
static bool test_popularity_thread_safety() {
    // Pre-populate entries
    for (int l = 0; l < 64; l++) {
        for (int e = 0; e < 32; e++) {
            ggml_sycl::set_expert_popularity_rank(l + 30000, e, 0);
        }
    }

    std::atomic<int> errors{0};
    std::atomic<bool> go{false};

    // Writer threads: update popularity ranks
    auto writer_fn = [&](int thread_id) {
        while (!go.load(std::memory_order_acquire)) {}
        for (int iter = 0; iter < 1000; iter++) {
            int l = (thread_id * 7 + iter) % 64;
            int e = (thread_id * 13 + iter) % 32;
            ggml_sycl::set_expert_popularity_rank(l + 30000, e, iter % 100);
        }
    };

    // Reader threads: get popularity and verify range
    auto reader_fn = [&](int /* thread_id */) {
        while (!go.load(std::memory_order_acquire)) {}
        for (int iter = 0; iter < 2000; iter++) {
            int l = iter % 64;
            int e = iter % 32;
            int rank = ggml_sycl::get_expert_popularity_rank(l + 30000, e);
            // rank should be in valid range (0-99 from writers, or 0 from init)
            if (rank < -1 || rank > 99) {
                errors.fetch_add(1, std::memory_order_relaxed);
            }
        }
    };

    const int n_writers = 4;
    const int n_readers = 8;
    std::vector<std::thread> threads;
    threads.reserve(n_writers + n_readers);

    for (int i = 0; i < n_writers; i++) {
        threads.emplace_back(writer_fn, i);
    }
    for (int i = 0; i < n_readers; i++) {
        threads.emplace_back(reader_fn, i);
    }

    go.store(true, std::memory_order_release);

    for (auto & t : threads) {
        t.join();
    }

    TEST_ASSERT(errors.load() == 0, "no data races detected in concurrent access");
    return true;
}

// ========================================================================
// Test 7: Popularity rank sorting — verify that rank values enable
//         correct priority ordering for prestage/eviction
// ========================================================================
static bool test_popularity_sorting() {
    const int layer_id = 50000;
    // Set experts with varying popularity (out of order)
    int popularities[] = { 5, 2, 7, 1, 3, 6, 0, 4 };
    for (int e = 0; e < 8; e++) {
        ggml_sycl::set_expert_popularity_rank(layer_id, e, popularities[e]);
    }

    // Build sorted list by popularity rank (ascending = most popular first)
    std::vector<std::pair<int, int>> sorted_experts;  // (rank, expert_id)
    for (int e = 0; e < 8; e++) {
        int rank = ggml_sycl::get_expert_popularity_rank(layer_id, e);
        sorted_experts.push_back({rank, e});
    }
    std::sort(sorted_experts.begin(), sorted_experts.end());

    // Verify sorted order
    for (size_t i = 1; i < sorted_experts.size(); i++) {
        TEST_ASSERT(sorted_experts[i].first >= sorted_experts[i - 1].first,
                    "experts should be sortable by popularity_rank ascending");
    }

    // Most popular (rank 0) should be expert 6
    TEST_ASSERT(sorted_experts[0].second == 6, "most popular expert should be expert 6 (rank 0)");

    return true;
}

int main() {
    printf("=== MoE Expert Popularity Ranking Unit Tests ===\n\n");

    RUN_TEST(test_popularity_init_state);
    RUN_TEST(test_popularity_set_get_roundtrip);
    RUN_TEST(test_popularity_initialized);
    RUN_TEST(test_popularity_key_uniqueness);
    RUN_TEST(test_popularity_hash_based_ids);
    RUN_TEST(test_popularity_thread_safety);
    RUN_TEST(test_popularity_sorting);

    printf("\n=== Results: %d/%d passed, %d failed ===\n",
           g_tests_passed, g_tests_run, g_tests_failed);

    return g_tests_failed > 0 ? 1 : 0;
}

#endif  // GGML_USE_SYCL
