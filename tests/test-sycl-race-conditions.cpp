// SYCL Race Condition Tests for Lazy-Initialized Config Variables
//
// TDD: These tests verify that lazy-initialized configuration variables
// are thread-safe. Multiple threads calling config functions simultaneously
// should not cause data races or undefined behavior.
//
// Build with ThreadSanitizer to detect races:
//   cmake -B build-tsan -G Ninja -DGGML_SYCL=ON \
//         -DCMAKE_CXX_FLAGS="-fsanitize=thread" \
//         -DCMAKE_C_FLAGS="-fsanitize=thread" \
//         -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread" ...
//
// The test spawns many threads that all call config functions concurrently.
// Without proper atomic operations, ThreadSanitizer will report data races.

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

// Forward declarations for the functions we're testing
// These are defined in ggml-sycl.cpp and common.cpp
namespace ggml_sycl_test {

// Functions from ggml-sycl.cpp (static, so we need test hooks)
// We test via the public API or by exposing test hooks

}  // namespace ggml_sycl_test

// Test counters
static int g_tests_run    = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, msg)                                                  \
    do {                                                                        \
        g_tests_run++;                                                          \
        if (!(cond)) {                                                          \
            fprintf(stderr, "FAIL: %s (line %d): %s\n", __func__, __LINE__, msg); \
            g_tests_failed++;                                                   \
        } else {                                                                \
            g_tests_passed++;                                                   \
        }                                                                       \
    } while (0)

// Include SYCL headers for testing
#include "ggml-sycl.h"
#include "ggml-sycl/common.hpp"

// Number of threads and iterations for stress testing
constexpr int NUM_THREADS    = 16;
constexpr int NUM_ITERATIONS = 1000;

// =============================================================================
// Test 1: ggml_sycl_quant_allreduce_enabled() thread safety
// =============================================================================
static void test_quant_allreduce_enabled_thread_safety() {
    printf("Testing ggml_sycl_quant_allreduce_enabled() thread safety...\n");

    std::atomic<int> ready{0};
    std::atomic<int> done{0};
    std::vector<bool> results(NUM_THREADS * NUM_ITERATIONS);
    std::vector<std::thread> threads;

    // Spawn threads that all wait at a barrier, then call concurrently
    for (int t = 0; t < NUM_THREADS; t++) {
        threads.emplace_back([&, t]() {
            ready.fetch_add(1, std::memory_order_release);
            // Spin until all threads are ready
            while (ready.load(std::memory_order_acquire) < NUM_THREADS) {
                // spin
            }
            // Now all threads call the function concurrently
            for (int i = 0; i < NUM_ITERATIONS; i++) {
                results[t * NUM_ITERATIONS + i] = ggml_sycl_quant_allreduce_enabled();
            }
            done.fetch_add(1, std::memory_order_release);
        });
    }

    for (auto & th : threads) {
        th.join();
    }

    // All results should be consistent (all true or all false)
    bool first = results[0];
    bool all_consistent = true;
    for (size_t i = 1; i < results.size(); i++) {
        if (results[i] != first) {
            all_consistent = false;
            break;
        }
    }

    TEST_ASSERT(all_consistent, "ggml_sycl_quant_allreduce_enabled() returned inconsistent results");
    printf("  All %d calls returned consistent value: %s\n",
           NUM_THREADS * NUM_ITERATIONS, first ? "true" : "false");
}

// =============================================================================
// Test 2: ggml_sycl_should_use_quant_allreduce() thread safety
// =============================================================================
static void test_should_use_quant_allreduce_thread_safety() {
    printf("Testing ggml_sycl_should_use_quant_allreduce() thread safety...\n");

    std::atomic<int> ready{0};
    std::vector<bool> results_small(NUM_THREADS * NUM_ITERATIONS);
    std::vector<bool> results_large(NUM_THREADS * NUM_ITERATIONS);
    std::vector<std::thread> threads;

    // Test with both small and large element counts
    constexpr size_t small_size = 1000;    // Below default threshold
    constexpr size_t large_size = 100000;  // Above default threshold

    for (int t = 0; t < NUM_THREADS; t++) {
        threads.emplace_back([&, t]() {
            ready.fetch_add(1, std::memory_order_release);
            while (ready.load(std::memory_order_acquire) < NUM_THREADS) {
                // spin
            }
            for (int i = 0; i < NUM_ITERATIONS; i++) {
                results_small[t * NUM_ITERATIONS + i] = ggml_sycl_should_use_quant_allreduce(small_size);
                results_large[t * NUM_ITERATIONS + i] = ggml_sycl_should_use_quant_allreduce(large_size);
            }
        });
    }

    for (auto & th : threads) {
        th.join();
    }

    // Check consistency for small sizes
    bool first_small = results_small[0];
    bool small_consistent = true;
    for (size_t i = 1; i < results_small.size(); i++) {
        if (results_small[i] != first_small) {
            small_consistent = false;
            break;
        }
    }

    // Check consistency for large sizes
    bool first_large = results_large[0];
    bool large_consistent = true;
    for (size_t i = 1; i < results_large.size(); i++) {
        if (results_large[i] != first_large) {
            large_consistent = false;
            break;
        }
    }

    TEST_ASSERT(small_consistent, "ggml_sycl_should_use_quant_allreduce(small) returned inconsistent results");
    TEST_ASSERT(large_consistent, "ggml_sycl_should_use_quant_allreduce(large) returned inconsistent results");
    printf("  Small size results consistent: %s, Large size results consistent: %s\n",
           first_small ? "true" : "false", first_large ? "true" : "false");
}

// =============================================================================
// Test 3: Stress test with rapid repeated calls
// =============================================================================
static void test_rapid_concurrent_calls() {
    printf("Testing rapid concurrent config calls...\n");

    std::atomic<int> ready{0};
    std::atomic<int> errors{0};
    std::vector<std::thread> threads;

    // More aggressive test - many rapid calls
    constexpr int RAPID_ITERATIONS = 10000;

    for (int t = 0; t < NUM_THREADS; t++) {
        threads.emplace_back([&]() {
            ready.fetch_add(1, std::memory_order_release);
            while (ready.load(std::memory_order_acquire) < NUM_THREADS) {
                // spin
            }

            bool prev_quant = ggml_sycl_quant_allreduce_enabled();
            for (int i = 0; i < RAPID_ITERATIONS; i++) {
                bool curr_quant = ggml_sycl_quant_allreduce_enabled();
                // Value should never change after initialization
                if (curr_quant != prev_quant) {
                    errors.fetch_add(1, std::memory_order_relaxed);
                }
                prev_quant = curr_quant;
            }
        });
    }

    for (auto & th : threads) {
        th.join();
    }

    TEST_ASSERT(errors.load() == 0, "Config values changed during concurrent access");
    printf("  No value changes detected across %d total calls\n",
           NUM_THREADS * RAPID_ITERATIONS);
}

// =============================================================================
// Main
// =============================================================================
int main() {
    printf("=== SYCL Race Condition Tests ===\n");
    printf("Threads: %d, Iterations per thread: %d\n\n", NUM_THREADS, NUM_ITERATIONS);

    // Note: For full race detection, build with ThreadSanitizer (-fsanitize=thread)
    // Even without TSan, these tests verify functional correctness under concurrency.

    test_quant_allreduce_enabled_thread_safety();
    test_should_use_quant_allreduce_thread_safety();
    test_rapid_concurrent_calls();

    printf("\n=== Summary ===\n");
    printf("Tests run: %d\n", g_tests_run);
    printf("Passed: %d\n", g_tests_passed);
    printf("Failed: %d\n", g_tests_failed);

    if (g_tests_failed > 0) {
        printf("\nRace condition tests FAILED!\n");
        printf("If running with ThreadSanitizer, check for data race reports.\n");
        return 1;
    }

    printf("\nAll race condition tests passed.\n");
    printf("For complete verification, rebuild with -fsanitize=thread\n");
    return 0;
}
