// Unit tests for parallel tensor repacking correctness
//
// This test validates that parallel repacking (with OpenMP) produces
// byte-identical results to sequential repacking. Since repacking is
// deterministic (no floating-point approximation), any thread count
// must produce exactly the same output given the same input.
//
// The test catches:
// - Race conditions in parallel execution
// - Incorrect index calculations in the parallelized loops
// - Thread-local buffer issues
//
// Usage: test-repack-parallel [-v]
//   -v : verbose output (show all test results, not just failures)

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "repack.h"  // For ggml_backend_cpu_repack_buffer_type

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

// Test configuration
static constexpr int64_t TEST_COLS = 4096;   // ne[0]: elements per row (must be divisible by block size)
static constexpr int64_t TEST_ROWS = 128;    // ne[1]: number of rows (must be divisible by nrows_interleaved)

static bool g_verbose = false;

// Generate reproducible test data using simple LCG
static void generate_test_data(uint8_t* data, size_t size, uint32_t seed) {
    uint32_t state = seed;
    for (size_t i = 0; i < size; i++) {
        state = state * 1103515245u + 12345u;
        data[i] = static_cast<uint8_t>((state >> 16) & 0xFF);
    }
}

// Compare two buffers byte-by-byte, return first difference index or -1 if equal
static int64_t compare_buffers(const uint8_t* a, const uint8_t* b, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (a[i] != b[i]) {
            return static_cast<int64_t>(i);
        }
    }
    return -1;
}

// Test structure for a specific quantization type
struct repack_test_case {
    const char* name;
    ggml_type   src_type;      // Original quantization type
    int64_t     ne0;           // Elements per row
    int64_t     ne1;           // Number of rows
};

// Run a single repack test with specified thread counts
// Returns true if serial and parallel produce identical results
static bool test_repack_determinism(const repack_test_case& tc, int serial_threads, int parallel_threads) {
    // Get repack buffer type
    ggml_backend_buffer_type_t repack_buft = ggml_backend_cpu_repack_buffer_type();
    if (!repack_buft) {
        printf("  %s: SKIPPED (repack buffer type not available)\n", tc.name);
        return true;  // Not a failure, just not supported
    }

    // Calculate source data size
    size_t type_size = ggml_type_size(tc.src_type);
    size_t block_size = ggml_blck_size(tc.src_type);
    if (block_size == 0) {
        printf("  %s: SKIPPED (invalid block size)\n", tc.name);
        return true;
    }

    int64_t n_blocks_per_row = tc.ne0 / block_size;
    size_t src_data_size = static_cast<size_t>(tc.ne1 * n_blocks_per_row) * type_size;

    // Generate source data
    std::vector<uint8_t> src_data(src_data_size);
    generate_test_data(src_data.data(), src_data_size, 0xDEADBEEF);

    // Create contexts for tensor allocation (with no_alloc=true)
    struct ggml_init_params params = {
        /*.mem_size   =*/ 2 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context* ctx_serial = ggml_init(params);
    struct ggml_context* ctx_parallel = ggml_init(params);

    if (!ctx_serial || !ctx_parallel) {
        printf("  %s: FAILED (context allocation)\n", tc.name);
        if (ctx_serial) ggml_free(ctx_serial);
        if (ctx_parallel) ggml_free(ctx_parallel);
        return false;
    }

    // Create tensors
    struct ggml_tensor* tensor_serial = ggml_new_tensor_2d(ctx_serial, tc.src_type, tc.ne0, tc.ne1);
    struct ggml_tensor* tensor_parallel = ggml_new_tensor_2d(ctx_parallel, tc.src_type, tc.ne0, tc.ne1);

    if (!tensor_serial || !tensor_parallel) {
        printf("  %s: FAILED (tensor creation)\n", tc.name);
        ggml_free(ctx_serial);
        ggml_free(ctx_parallel);
        return false;
    }

    ggml_set_name(tensor_serial, "serial");
    ggml_set_name(tensor_parallel, "parallel");

    // Allocate buffers with repack buffer type using the proper API
    ggml_backend_buffer_t buf_serial = ggml_backend_alloc_ctx_tensors_from_buft(ctx_serial, repack_buft);
    ggml_backend_buffer_t buf_parallel = ggml_backend_alloc_ctx_tensors_from_buft(ctx_parallel, repack_buft);

    if (!buf_serial || !buf_parallel) {
        printf("  %s: SKIPPED (buffer allocation failed - type may not be supported)\n", tc.name);
        if (buf_serial) ggml_backend_buffer_free(buf_serial);
        if (buf_parallel) ggml_backend_buffer_free(buf_parallel);
        ggml_free(ctx_serial);
        ggml_free(ctx_parallel);
        return true;  // Not necessarily a failure - might just be unsupported config
    }

#if defined(_OPENMP)
    // Run serial repack
    omp_set_num_threads(serial_threads);
#else
    (void)serial_threads;
#endif

    ggml_backend_tensor_set(tensor_serial, src_data.data(), 0, src_data_size);

#if defined(_OPENMP)
    // Run parallel repack
    omp_set_num_threads(parallel_threads);
#else
    (void)parallel_threads;
#endif

    ggml_backend_tensor_set(tensor_parallel, src_data.data(), 0, src_data_size);

    // Compare results
    // After repacking, tensor->data points to the repacked data
    size_t repacked_size = ggml_nbytes(tensor_serial);

    const uint8_t* data_serial = static_cast<const uint8_t*>(tensor_serial->data);
    const uint8_t* data_parallel = static_cast<const uint8_t*>(tensor_parallel->data);

    int64_t diff_idx = compare_buffers(data_serial, data_parallel, repacked_size);

    bool passed = (diff_idx < 0);

    if (!passed) {
        printf("  %s [%d vs %d threads]: FAILED\n", tc.name, serial_threads, parallel_threads);
        printf("    First difference at byte %lld: serial=0x%02X parallel=0x%02X\n",
               (long long)diff_idx, data_serial[diff_idx], data_parallel[diff_idx]);

        // Show a few more bytes for context
        printf("    Context around difference:\n");
        int64_t start = (diff_idx > 8) ? diff_idx - 8 : 0;
        int64_t end = (diff_idx + 8 < (int64_t)repacked_size) ? diff_idx + 8 : (int64_t)repacked_size;
        printf("    Serial:   ");
        for (int64_t i = start; i < end; i++) {
            printf("%02X ", data_serial[i]);
        }
        printf("\n    Parallel: ");
        for (int64_t i = start; i < end; i++) {
            printf("%02X ", data_parallel[i]);
        }
        printf("\n");
    } else if (g_verbose) {
        printf("  %s [%d vs %d threads]: ok (%zu bytes verified)\n",
               tc.name, serial_threads, parallel_threads, repacked_size);
    }

    // Cleanup
    ggml_backend_buffer_free(buf_serial);
    ggml_backend_buffer_free(buf_parallel);
    ggml_free(ctx_serial);
    ggml_free(ctx_parallel);

    return passed;
}

int main(int argc, char* argv[]) {
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0) {
            g_verbose = true;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s [-v]\n", argv[0]);
            return 1;
        }
    }

    printf("Testing parallel repack determinism\n");
    printf("====================================\n\n");

#if defined(_OPENMP)
    int max_threads = omp_get_max_threads();
    printf("OpenMP enabled, max threads: %d\n", max_threads);
#else
    printf("OpenMP not enabled - tests will verify single-threaded consistency\n");
    int max_threads = 1;
#endif

    // Initialize CPU backend
    ggml_cpu_init();

    // Define test cases for each quantization type that has repacking
    // These correspond to the parallelized repack functions
    std::vector<repack_test_case> test_cases = {
        // Q4_0 with 4-way interleave (repack_q4_0_to_q4_0_4_bl)
        {"Q4_0 x4 interleave", GGML_TYPE_Q4_0, TEST_COLS, TEST_ROWS},

        // Q4_K with 8-way interleave (repack_q4_K_to_q4_K_8_bl)
        {"Q4_K x8 interleave", GGML_TYPE_Q4_K, TEST_COLS, TEST_ROWS},

        // Q2_K with 8-way interleave (repack_q2_K_to_q2_K_8_bl)
        {"Q2_K x8 interleave", GGML_TYPE_Q2_K, TEST_COLS, TEST_ROWS},

        // IQ4_NL (repack_iq4_nl_to_iq4_nl_4_bl and repack_iq4_nl_to_iq4_nl_8_bl)
        {"IQ4_NL interleave", GGML_TYPE_IQ4_NL, TEST_COLS, TEST_ROWS},
    };

    int num_failed = 0;

    // Thread count configurations to test
    // We test 1 vs N threads to catch parallelization bugs
    std::vector<std::pair<int, int>> thread_configs;
    thread_configs.push_back({1, 1});  // Baseline: both single-threaded

    if (max_threads >= 2) {
        thread_configs.push_back({1, 2});   // 1 vs 2 threads
    }
    if (max_threads >= 4) {
        thread_configs.push_back({1, 4});   // 1 vs 4 threads
    }
    if (max_threads >= 8) {
        thread_configs.push_back({1, 8});   // 1 vs 8 threads
    }
    if (max_threads >= 16) {
        thread_configs.push_back({1, 16});  // 1 vs 16 threads
    }
    if (max_threads > 16) {
        thread_configs.push_back({1, max_threads});  // 1 vs max threads
    }

    // Also test different parallel counts against each other
    if (max_threads >= 4) {
        thread_configs.push_back({2, 4});   // 2 vs 4 threads
    }
    if (max_threads >= 8) {
        thread_configs.push_back({4, 8});   // 4 vs 8 threads
    }

    for (const auto& tc : test_cases) {
        printf("Testing %s (ne0=%lld, ne1=%lld):\n", tc.name, (long long)tc.ne0, (long long)tc.ne1);

        for (const auto& threads : thread_configs) {
            bool passed = test_repack_determinism(tc, threads.first, threads.second);
            if (!passed) {
                num_failed++;
            }
        }
        printf("\n");
    }

    printf("====================================\n");
    if (num_failed == 0) {
        printf("All tests passed!\n");
    } else {
        printf("%d test(s) FAILED\n", num_failed);
    }

    return num_failed > 0 ? 1 : 0;
}
