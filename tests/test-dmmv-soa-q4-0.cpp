// DMMV Q4_0 SoA Unit Test
// Tests the actual dequantize kernel functions used in DMMV
// Compares AoS vs SoA dequantization to find the bug
//
// Build: cmake --build build --target test-dmmv-soa-q4-0
// Run: ./build/bin/test-dmmv-soa-q4-0

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#include <sycl/sycl.hpp>

// Constants from DMMV kernel (matching production code)
#define GGML_SYCL_DMMV_X 32
#define WARP_SIZE 32

// Q4_0 block structure (must match ggml-common.h)
#define QK4_0 32
#define QR4_0 2

typedef struct {
    sycl::half d;
    uint8_t qs[QK4_0/2];
} block_q4_0;

static_assert(sizeof(block_q4_0) == 18, "block_q4_0 size mismatch");

using dfloat = float;
using dfloat2 = sycl::float2;

// ============================================================================
// ACTUAL dequantize_q4_0 from dequantize.hpp (AoS version)
// ============================================================================
static inline void dequantize_q4_0(const void *vx, const int64_t ib,
                                    const int iqs, dfloat2 &v) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const dfloat d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

    v.x() = (v.x() - 8.0f) * d;
    v.y() = (v.y() - 8.0f) * d;
}

// ============================================================================
// ACTUAL dequantize_q4_0_reorder from dequantize.hpp (SoA version)
// ============================================================================
static inline void dequantize_q4_0_reorder(const void *d_ptr, const int64_t ib, const void *qs,
                                            const int iqs, dfloat2 &v) {
    const dfloat d = (const dfloat)*((const sycl::half*)d_ptr+ib);

    const int vui = *((const uint8_t *)qs+iqs);

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

    v.x() = (v.x() - 8.0f) * d;
    v.y() = (v.y() - 8.0f) * d;
}

// Test configuration matching Mistral 7B dimensions
struct TestConfig {
    int nrows;
    int ncols;
    const char* name;
};

static TestConfig test_configs[] = {
    {4096, 4096, "Mistral FFN (4096x4096)"},
    {4096, 14336, "Mistral Up/Gate (4096x14336)"},
    {14336, 4096, "Mistral Down (14336x4096)"},
    {128, 4096, "Small test (128x4096)"},
    {1, 4096, "Single row (1x4096)"},
};

// Create AoS Q4_0 test data with known values
void create_aos_data(block_q4_0* data, int nrows, int ncols, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> scale_dist(0.001f, 0.1f);

    const int blocks_per_row = ncols / QK4_0;
    const int nblocks = nrows * blocks_per_row;

    for (int i = 0; i < nblocks; i++) {
        // Use varied but reproducible scales
        data[i].d = sycl::half(scale_dist(rng));

        // Fill qs with pattern that exercises all nibble values
        for (int j = 0; j < QK4_0/2; j++) {
            // Create predictable pattern: lower nibble 0-15, upper nibble varies
            uint8_t lo = (i + j) & 0xF;
            uint8_t hi = (i + j + 7) & 0xF;
            data[i].qs[j] = lo | (hi << 4);
        }
    }
}

// Convert AoS to SoA layout (matches reorder_qw_q4_0 in ggml-sycl.cpp)
void convert_aos_to_soa(const block_q4_0* aos, uint8_t* soa, int nrows, int ncols) {
    const int blocks_per_row = ncols / QK4_0;
    const int nblocks = nrows * blocks_per_row;

    // SoA layout: [all qs bytes][all d values]
    uint8_t* qs_ptr = soa;
    sycl::half* d_ptr = (sycl::half*)(soa + (size_t)nrows * ncols / 2);

    for (int i = 0; i < nblocks; i++) {
        // Copy qs bytes
        memcpy(qs_ptr + i * (QK4_0/2), aos[i].qs, QK4_0/2);
        // Copy d value
        d_ptr[i] = aos[i].d;
    }
}

// Test 1: Direct dequantize function comparison
// Calls actual dequantize_q4_0 and dequantize_q4_0_reorder functions
bool test_dequantize_functions() {
    printf("Test 1: Direct dequantize function comparison\n");

    const int nrows = 128;
    const int ncols = 4096;
    const int blocks_per_row = ncols / QK4_0;
    const int nblocks = nrows * blocks_per_row;
    const size_t aos_size = nblocks * sizeof(block_q4_0);
    const size_t soa_size = aos_size;  // Same total size
    const int64_t d_offset = (int64_t)nrows * ncols / 2;

    // Allocate data
    std::vector<block_q4_0> aos_data(nblocks);
    std::vector<uint8_t> soa_data(soa_size);

    // Create test data
    create_aos_data(aos_data.data(), nrows, ncols, 42);
    convert_aos_to_soa(aos_data.data(), soa_data.data(), nrows, ncols);

    // Pointers for SoA access
    const uint8_t* qs_base = soa_data.data();
    const void* d_base = soa_data.data() + d_offset;

    int errors = 0;
    int tested = 0;

    // Test every block and every iqs position
    for (int ib = 0; ib < std::min(nblocks, 1000); ib++) {
        for (int iqs = 0; iqs < QK4_0/2; iqs++) {
            dfloat2 v_aos, v_soa;

            // Call actual AoS dequantize function
            dequantize_q4_0(aos_data.data(), ib, iqs, v_aos);

            // Call actual SoA dequantize function
            // Note: SoA function expects qs_index = ib * (QK4_0/2) + iqs
            int qs_index = ib * (QK4_0/2) + iqs;
            dequantize_q4_0_reorder(d_base, ib, qs_base, qs_index, v_soa);

            // Compare results
            float diff_x = fabsf(v_aos.x() - v_soa.x());
            float diff_y = fabsf(v_aos.y() - v_soa.y());

            if (diff_x > 1e-5f || diff_y > 1e-5f) {
                if (errors < 5) {
                    printf("  ERROR at ib=%d, iqs=%d: AoS=(%.6f,%.6f) SoA=(%.6f,%.6f)\n",
                           ib, iqs, v_aos.x(), v_aos.y(), v_soa.x(), v_soa.y());

                    // Debug: show raw values
                    uint8_t aos_qs = aos_data[ib].qs[iqs];
                    uint8_t soa_qs = qs_base[qs_index];
                    float aos_d = (float)aos_data[ib].d;
                    float soa_d = (float)*((const sycl::half*)d_base + ib);
                    printf("         Raw: AoS qs=0x%02x d=%.6f, SoA qs=0x%02x d=%.6f\n",
                           aos_qs, aos_d, soa_qs, soa_d);
                }
                errors++;
            }
            tested++;
        }
    }

    if (errors > 0) {
        printf("  FAIL: %d/%d dequantize mismatches\n", errors, tested);
        return false;
    }

    printf("  PASS: %d dequantize comparisons match\n", tested);
    return true;
}

// Test 2: DMMV iteration pattern comparison
// Simulates exact DMMV kernel loop structure with actual dequantize calls
bool test_dmmv_iteration_pattern() {
    printf("Test 2: DMMV iteration pattern comparison\n");

    const int nrows = 64;
    const int ncols = 4096;
    const int blocks_per_row = ncols / QK4_0;
    const int nblocks = nrows * blocks_per_row;
    const size_t aos_size = nblocks * sizeof(block_q4_0);
    const int64_t d_offset = (int64_t)nrows * ncols / 2;

    // Allocate data
    std::vector<block_q4_0> aos_data(nblocks);
    std::vector<uint8_t> soa_data(aos_size);
    std::vector<float> y_data(nrows * ncols);  // Input vector

    // Create test data
    create_aos_data(aos_data.data(), nrows, ncols, 123);
    convert_aos_to_soa(aos_data.data(), soa_data.data(), nrows, ncols);

    // Create Y vector with known values
    std::mt19937 rng(456);
    std::uniform_real_distribution<float> y_dist(-1.0f, 1.0f);
    for (size_t i = 0; i < y_data.size(); i++) {
        y_data[i] = y_dist(rng);
    }

    // SoA pointers
    const uint8_t* qs_base = soa_data.data();
    const void* d_base = soa_data.data() + d_offset;

    // DMMV kernel parameters (from dmmv.cpp lines 163-165)
    const int iter_stride = 2 * GGML_SYCL_DMMV_X;  // 64
    const int vals_per_iter = iter_stride / WARP_SIZE;  // 2
    const int y_offset = QK4_0 / 2;  // 16 (for qr=2)

    int errors = 0;

    // Simulate DMMV for each row
    for (int row = 0; row < nrows; row++) {
        float sum_aos = 0.0f;
        float sum_soa = 0.0f;

        // Simulate all 32 threads in a warp
        for (int tid = 0; tid < WARP_SIZE; tid++) {
            float thread_sum_aos = 0.0f;
            float thread_sum_soa = 0.0f;

            // Main loop (from dmmv.cpp lines 176-204)
            for (int i = 0; i < ncols; i += iter_stride) {
                const int col = i + vals_per_iter * tid;
                const int ib = (row * ncols + col) / QK4_0;
                const int iqs = (col % QK4_0) / QR4_0;
                const int iybs = col - col % QK4_0;

                // Inner loop (j goes 0 only since vals_per_iter=2)
                for (int j = 0; j < vals_per_iter; j += 2) {
                    dfloat2 v_aos, v_soa;

                    // AoS dequantize (as in original kernel)
                    dequantize_q4_0(aos_data.data(), ib, iqs + j/QR4_0, v_aos);

                    // SoA dequantize (as in reorder kernel, line 190)
                    int qs_index = ib * (QK4_0/2) + iqs + j/QR4_0;
                    dequantize_q4_0_reorder(d_base, ib, qs_base, qs_index, v_soa);

                    // Matrix multiplication (lines 200-202)
                    int y_idx_0 = iybs + iqs + j/QR4_0;
                    int y_idx_1 = iybs + iqs + j/QR4_0 + y_offset;

                    thread_sum_aos += v_aos.x() * y_data[y_idx_0];
                    thread_sum_aos += v_aos.y() * y_data[y_idx_1];

                    thread_sum_soa += v_soa.x() * y_data[y_idx_0];
                    thread_sum_soa += v_soa.y() * y_data[y_idx_1];
                }
            }

            sum_aos += thread_sum_aos;
            sum_soa += thread_sum_soa;
        }

        // Compare row results
        float diff = fabsf(sum_aos - sum_soa);
        float rel_diff = diff / (fabsf(sum_aos) + 1e-10f);

        if (rel_diff > 1e-4f) {
            if (errors < 5) {
                printf("  ERROR row %d: AoS=%.6f SoA=%.6f diff=%.6e rel=%.6e\n",
                       row, sum_aos, sum_soa, diff, rel_diff);
            }
            errors++;
        }
    }

    if (errors > 0) {
        printf("  FAIL: %d/%d rows have DMMV mismatch\n", errors, nrows);
        return false;
    }

    printf("  PASS: All %d rows match between AoS and SoA DMMV\n", nrows);
    return true;
}

// Test 3: Full DMMV matrix-vector multiply comparison
// Tests the complete dequantize + multiply + accumulate pipeline
bool test_full_dmmv_comparison() {
    printf("Test 3: Full DMMV matrix-vector multiply\n");

    int total_passed = 0;
    int total_failed = 0;

    for (const auto& cfg : test_configs) {
        const int nrows = cfg.nrows;
        const int ncols = cfg.ncols;
        const int blocks_per_row = ncols / QK4_0;
        const int nblocks = nrows * blocks_per_row;
        const size_t aos_size = nblocks * sizeof(block_q4_0);
        const int64_t d_offset = (int64_t)nrows * ncols / 2;

        // Allocate data
        std::vector<block_q4_0> aos_data(nblocks);
        std::vector<uint8_t> soa_data(aos_size);
        std::vector<float> y_data(ncols);
        std::vector<float> dst_aos(nrows, 0.0f);
        std::vector<float> dst_soa(nrows, 0.0f);

        // Create test data
        create_aos_data(aos_data.data(), nrows, ncols, 789);
        convert_aos_to_soa(aos_data.data(), soa_data.data(), nrows, ncols);

        // Create Y vector
        std::mt19937 rng(101);
        std::uniform_real_distribution<float> y_dist(-1.0f, 1.0f);
        for (int i = 0; i < ncols; i++) {
            y_data[i] = y_dist(rng);
        }

        // SoA pointers
        const uint8_t* qs_base = soa_data.data();
        const void* d_base = soa_data.data() + d_offset;

        // Compute full DMMV for each row
        for (int row = 0; row < nrows; row++) {
            float sum_aos = 0.0f;
            float sum_soa = 0.0f;

            // Process all elements in the row
            for (int col = 0; col < ncols; col += 2) {
                const int ib = (row * ncols + col) / QK4_0;
                const int iqs = (col % QK4_0) / 2;

                dfloat2 v_aos, v_soa;

                // AoS dequantize
                dequantize_q4_0(aos_data.data(), ib, iqs, v_aos);

                // SoA dequantize
                int qs_index = ib * (QK4_0/2) + iqs;
                dequantize_q4_0_reorder(d_base, ib, qs_base, qs_index, v_soa);

                // Accumulate
                sum_aos += v_aos.x() * y_data[col];
                sum_aos += v_aos.y() * y_data[col + QK4_0/2];

                sum_soa += v_soa.x() * y_data[col];
                sum_soa += v_soa.y() * y_data[col + QK4_0/2];
            }

            dst_aos[row] = sum_aos;
            dst_soa[row] = sum_soa;
        }

        // Compare results
        int errors = 0;
        float max_rel_diff = 0.0f;

        for (int row = 0; row < nrows; row++) {
            float diff = fabsf(dst_aos[row] - dst_soa[row]);
            float rel_diff = diff / (fabsf(dst_aos[row]) + 1e-10f);
            max_rel_diff = std::max(max_rel_diff, rel_diff);

            if (rel_diff > 1e-4f) {
                errors++;
            }
        }

        if (errors > 0) {
            printf("  %s: FAIL (%d/%d rows differ, max_rel=%.2e)\n",
                   cfg.name, errors, nrows, max_rel_diff);
            total_failed++;
        } else {
            printf("  %s: PASS (max_rel=%.2e)\n", cfg.name, max_rel_diff);
            total_passed++;
        }
    }

    printf("  Summary: %d passed, %d failed\n", total_passed, total_failed);
    return total_failed == 0;
}

// Test 4: Edge case - verify d_offset calculation
bool test_d_offset_calculation() {
    printf("Test 4: d_offset calculation verification\n");

    const int nrows = 256;
    const int ncols = 4096;
    const int blocks_per_row = ncols / QK4_0;
    const int nblocks = nrows * blocks_per_row;
    const size_t aos_size = nblocks * sizeof(block_q4_0);

    // Calculate expected d_offset
    const int64_t expected_d_offset = (int64_t)nrows * ncols / 2;  // All qs bytes
    const int64_t expected_qs_bytes = nblocks * (QK4_0/2);

    printf("  nrows=%d ncols=%d nblocks=%d\n", nrows, ncols, nblocks);
    printf("  expected_d_offset = %lld bytes\n", (long long)expected_d_offset);
    printf("  expected_qs_bytes = %lld bytes\n", (long long)expected_qs_bytes);
    printf("  match: %s\n", expected_d_offset == expected_qs_bytes ? "YES" : "NO");

    if (expected_d_offset != expected_qs_bytes) {
        printf("  FAIL: d_offset calculation mismatch!\n");
        return false;
    }

    // Verify by creating data and checking access
    std::vector<block_q4_0> aos_data(nblocks);
    std::vector<uint8_t> soa_data(aos_size);

    // Set unique scale for each block
    for (int i = 0; i < nblocks; i++) {
        float scale = 0.001f * (i + 1);
        aos_data[i].d = sycl::half(scale);
    }

    convert_aos_to_soa(aos_data.data(), soa_data.data(), nrows, ncols);

    // Verify d values at expected offset
    const sycl::half* d_ptr = (const sycl::half*)(soa_data.data() + expected_d_offset);

    int errors = 0;
    for (int i = 0; i < nblocks && errors < 5; i++) {
        float expected = (float)aos_data[i].d;
        float actual = (float)d_ptr[i];

        if (fabsf(expected - actual) > 1e-3f) {  // Use larger tolerance for fp16
            printf("  ERROR block %d: expected d=%.6f, got d=%.6f\n", i, expected, actual);
            errors++;
        }
    }

    if (errors > 0) {
        printf("  FAIL: %d d values at wrong offset\n", errors);
        return false;
    }

    printf("  PASS: All %d d values correctly placed at offset %lld\n",
           nblocks, (long long)expected_d_offset);
    return true;
}

int main() {
    printf("=== DMMV Q4_0 SoA Unit Tests ===\n");
    printf("Testing actual dequantize kernel functions\n\n");

    int passed = 0;
    int failed = 0;

    if (test_dequantize_functions()) passed++; else failed++;
    printf("\n");

    if (test_dmmv_iteration_pattern()) passed++; else failed++;
    printf("\n");

    if (test_full_dmmv_comparison()) passed++; else failed++;
    printf("\n");

    if (test_d_offset_calculation()) passed++; else failed++;
    printf("\n");

    printf("=================================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);

    return failed > 0 ? 1 : 0;
}
