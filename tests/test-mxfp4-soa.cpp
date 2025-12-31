// Comprehensive MXFP4 SoA vs AoS unit test
// Tests the CPU-side reorder and GPU kernel computation for correctness
//
// Usage:
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-mxfp4-soa
//
// Expected: All tests pass, AoS and SoA produce identical results

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <sycl/sycl.hpp>

// MXFP4 constants
#define QK_MXFP4 32
#define BYTES_PER_BLOCK 17  // 1 byte e + 16 bytes qs

// MXFP4 block structure (AoS layout) - matches ggml-common.h
typedef struct {
    uint8_t e;              // E8M0 scale
    uint8_t qs[QK_MXFP4/2]; // 16 packed bytes (32 4-bit values)
} block_mxfp4;

static_assert(sizeof(block_mxfp4) == BYTES_PER_BLOCK, "block_mxfp4 size mismatch");

// MXFP4 dequantization lookup table
// Maps 4-bit values to floats: 0->0, 1->0.5, 2->1, 3->1.5, ..., 8-15 are negative
static const float kvalues_mxfp4[16] = {
    0, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
    -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
};

// E8M0 to FP32 conversion (exponent-only format, no mantissa)
static inline float e8m0_to_fp32(uint8_t x) {
    if (x == 0) {
        // Special case: 2^(-127) represented as subnormal-ish value
        union { uint32_t u; float f; } bits = { 0x00400000 };
        return bits.f;
    }
    union { uint32_t u; float f; } bits = { ((uint32_t)x) << 23 };
    return bits.f;
}

// =============================================================================
// CPU Reference Implementation (AoS)
// =============================================================================

// Dot product of one MXFP4 block (AoS) with float input
float cpu_dot_mxfp4_aos(const block_mxfp4* block, const float* input) {
    const float scale = e8m0_to_fp32(block->e);
    float sum = 0.0f;

    for (int i = 0; i < QK_MXFP4 / 2; i++) {
        uint8_t packed = block->qs[i];
        float w_lo = scale * kvalues_mxfp4[packed & 0xF];
        float w_hi = scale * kvalues_mxfp4[packed >> 4];
        sum += w_lo * input[i] + w_hi * input[i + 16];
    }
    return sum;
}

// Dot product of a row with nblocks blocks (AoS)
float cpu_row_dot_aos(const block_mxfp4* row, const float* input, int nblocks) {
    float sum = 0.0f;
    for (int b = 0; b < nblocks; b++) {
        sum += cpu_dot_mxfp4_aos(&row[b], input + b * QK_MXFP4);
    }
    return sum;
}

// =============================================================================
// CPU Reference Implementation (SoA)
// =============================================================================

// SoA layout: [all qs (nblocks * 16 bytes)][all e (nblocks * 1 byte)]
struct soa_mxfp4_ptrs {
    const uint8_t* qs;  // Pointer to qs region
    const uint8_t* e;   // Pointer to e (scale) region
};

// Get SoA pointers for a given base and total blocks
soa_mxfp4_ptrs get_soa_ptrs(const void* soa_data, size_t nblocks) {
    const uint8_t* base = (const uint8_t*)soa_data;
    return {
        base,                                // qs starts at offset 0
        base + nblocks * (QK_MXFP4 / 2)      // e starts after all qs
    };
}

// Dot product of one MXFP4 block (SoA) with float input
// block_idx is the absolute index into the SoA array
float cpu_dot_mxfp4_soa(const soa_mxfp4_ptrs& ptrs, int block_idx, const float* input) {
    const uint8_t* qs_block = ptrs.qs + block_idx * (QK_MXFP4 / 2);
    const float scale = e8m0_to_fp32(ptrs.e[block_idx]);
    float sum = 0.0f;

    for (int i = 0; i < QK_MXFP4 / 2; i++) {
        uint8_t packed = qs_block[i];
        float w_lo = scale * kvalues_mxfp4[packed & 0xF];
        float w_hi = scale * kvalues_mxfp4[packed >> 4];
        sum += w_lo * input[i] + w_hi * input[i + 16];
    }
    return sum;
}

// Dot product of a row with nblocks_per_row blocks (SoA)
// row_idx: which row (0 to nrows-1)
// nblocks_per_row: blocks per row
float cpu_row_dot_soa(const void* soa_data, int row_idx, int nblocks_per_row,
                      int total_nblocks, const float* input) {
    soa_mxfp4_ptrs ptrs = get_soa_ptrs(soa_data, total_nblocks);
    float sum = 0.0f;

    int start_block = row_idx * nblocks_per_row;
    for (int b = 0; b < nblocks_per_row; b++) {
        sum += cpu_dot_mxfp4_soa(ptrs, start_block + b, input + b * QK_MXFP4);
    }
    return sum;
}

// =============================================================================
// CPU Reorder: AoS → SoA (matches production code in ggml-sycl.cpp)
// =============================================================================

void reorder_mxfp4_aos_to_soa(void* dst_soa, const void* src_aos, int ncols, int nrows) {
    const size_t blocks_per_row = ncols / QK_MXFP4;
    const size_t nblocks = blocks_per_row * nrows;

    const uint8_t* aos = (const uint8_t*)src_aos;
    uint8_t* soa_qs = (uint8_t*)dst_soa;
    uint8_t* soa_e = soa_qs + nblocks * (QK_MXFP4 / 2);  // e values start after all qs

    for (size_t ib = 0; ib < nblocks; ib++) {
        const block_mxfp4* block = (const block_mxfp4*)(aos + ib * sizeof(block_mxfp4));

        // Copy qs array (16 bytes)
        memcpy(soa_qs + ib * (QK_MXFP4 / 2), block->qs, QK_MXFP4 / 2);

        // Copy e (scale) value (1 byte)
        soa_e[ib] = block->e;
    }
}

// =============================================================================
// GPU Kernel: MXFP4 Row Dot Product (AoS)
// =============================================================================

void gpu_row_dot_aos_kernel(
    const uint8_t* __restrict__ weights_aos,  // AoS layout
    const float* __restrict__ input,
    float* __restrict__ output,
    int nblocks_per_row,
    int nrows,
    const sycl::nd_item<1>& item
) {
    const int row = item.get_global_id(0);
    if (row >= nrows) return;

    const block_mxfp4* row_blocks = (const block_mxfp4*)(weights_aos + row * nblocks_per_row * sizeof(block_mxfp4));

    float sum = 0.0f;
    for (int b = 0; b < nblocks_per_row; b++) {
        const float scale = e8m0_to_fp32(row_blocks[b].e);

        for (int i = 0; i < QK_MXFP4 / 2; i++) {
            uint8_t packed = row_blocks[b].qs[i];
            float w_lo = scale * kvalues_mxfp4[packed & 0xF];
            float w_hi = scale * kvalues_mxfp4[packed >> 4];
            sum += w_lo * input[b * QK_MXFP4 + i] + w_hi * input[b * QK_MXFP4 + i + 16];
        }
    }
    output[row] = sum;
}

// =============================================================================
// GPU Kernel: MXFP4 Row Dot Product (SoA) - matches fused_moe_mxfp4_soa_kernel
// =============================================================================

void gpu_row_dot_soa_kernel(
    const uint8_t* __restrict__ weights_qs,     // SoA: qs region
    const uint8_t* __restrict__ weights_scales, // SoA: scales region
    const float* __restrict__ input,
    float* __restrict__ output,
    int nblocks_per_row,
    int nrows,
    const sycl::nd_item<1>& item
) {
    const int row = item.get_global_id(0);
    if (row >= nrows) return;

    // SoA layout: compute absolute block offset for this row
    const int64_t row_block_offset = row * nblocks_per_row;

    // qs region: each block contributes 16 bytes
    const uint8_t* qs_row = weights_qs + row_block_offset * (QK_MXFP4 / 2);
    // scales region: each block contributes 1 byte
    const uint8_t* scale_row = weights_scales + row_block_offset;

    float sum = 0.0f;
    for (int b = 0; b < nblocks_per_row; b++) {
        const float scale = e8m0_to_fp32(scale_row[b]);

        for (int i = 0; i < QK_MXFP4 / 2; i++) {
            uint8_t packed = qs_row[b * (QK_MXFP4 / 2) + i];
            float w_lo = scale * kvalues_mxfp4[packed & 0xF];
            float w_hi = scale * kvalues_mxfp4[packed >> 4];
            sum += w_lo * input[b * QK_MXFP4 + i] + w_hi * input[b * QK_MXFP4 + i + 16];
        }
    }
    output[row] = sum;
}

// =============================================================================
// Test Utilities
// =============================================================================

void generate_random_mxfp4_blocks(block_mxfp4* blocks, int nblocks, std::mt19937& rng) {
    std::uniform_int_distribution<int> scale_dist(100, 140);  // Reasonable E8M0 range
    std::uniform_int_distribution<int> qs_dist(0, 255);

    for (int i = 0; i < nblocks; i++) {
        blocks[i].e = scale_dist(rng);
        for (int j = 0; j < QK_MXFP4 / 2; j++) {
            blocks[i].qs[j] = qs_dist(rng);
        }
    }
}

void generate_random_input(float* input, int n, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; i++) {
        input[i] = dist(rng);
    }
}

bool compare_results(const float* a, const float* b, int n, float rtol = 1e-4f, float atol = 1e-6f) {
    bool all_match = true;
    for (int i = 0; i < n; i++) {
        float diff = std::abs(a[i] - b[i]);
        float threshold = atol + rtol * std::abs(b[i]);
        if (diff > threshold) {
            fprintf(stderr, "  Mismatch at [%d]: got %.6f, expected %.6f (diff=%.6e)\n",
                    i, a[i], b[i], diff);
            all_match = false;
            if (i > 5) {
                fprintf(stderr, "  ... (more mismatches)\n");
                break;
            }
        }
    }
    return all_match;
}

// =============================================================================
// Test Cases
// =============================================================================

bool test_cpu_reorder_correctness() {
    printf("Test: CPU AoS→SoA reorder correctness...\n");

    const int ncols = 2880;  // GPT-OSS hidden dim (divisible by 32)
    const int nrows = 8;     // Test with 8 rows
    const int blocks_per_row = ncols / QK_MXFP4;  // 90
    const int nblocks = blocks_per_row * nrows;   // 720

    std::mt19937 rng(42);

    // Allocate and fill AoS data
    std::vector<block_mxfp4> aos_data(nblocks);
    generate_random_mxfp4_blocks(aos_data.data(), nblocks, rng);

    // Allocate SoA buffer (same total size)
    std::vector<uint8_t> soa_data(nblocks * BYTES_PER_BLOCK);

    // Perform reorder
    reorder_mxfp4_aos_to_soa(soa_data.data(), aos_data.data(), ncols, nrows);

    // Verify: qs values should be contiguous first
    const uint8_t* soa_qs = soa_data.data();
    const uint8_t* soa_e = soa_qs + nblocks * (QK_MXFP4 / 2);

    bool pass = true;
    for (int ib = 0; ib < nblocks; ib++) {
        // Check qs bytes
        for (int j = 0; j < QK_MXFP4 / 2; j++) {
            uint8_t expected = aos_data[ib].qs[j];
            uint8_t actual = soa_qs[ib * (QK_MXFP4 / 2) + j];
            if (expected != actual) {
                fprintf(stderr, "  qs mismatch at block %d byte %d: got %d, expected %d\n",
                        ib, j, actual, expected);
                pass = false;
            }
        }
        // Check e (scale) value
        if (aos_data[ib].e != soa_e[ib]) {
            fprintf(stderr, "  e mismatch at block %d: got %d, expected %d\n",
                    ib, soa_e[ib], aos_data[ib].e);
            pass = false;
        }
    }

    printf("  %s\n", pass ? "PASSED" : "FAILED");
    return pass;
}

bool test_cpu_aos_vs_soa_dot() {
    printf("Test: CPU AoS vs SoA dot product equivalence...\n");

    const int ncols = 2880;
    const int nrows = 8;
    const int blocks_per_row = ncols / QK_MXFP4;
    const int nblocks = blocks_per_row * nrows;

    std::mt19937 rng(123);

    // Generate test data
    std::vector<block_mxfp4> aos_data(nblocks);
    generate_random_mxfp4_blocks(aos_data.data(), nblocks, rng);

    std::vector<float> input(ncols);
    generate_random_input(input.data(), ncols, rng);

    // Reorder to SoA
    std::vector<uint8_t> soa_data(nblocks * BYTES_PER_BLOCK);
    reorder_mxfp4_aos_to_soa(soa_data.data(), aos_data.data(), ncols, nrows);

    // Compute with both layouts
    std::vector<float> aos_results(nrows);
    std::vector<float> soa_results(nrows);

    for (int row = 0; row < nrows; row++) {
        aos_results[row] = cpu_row_dot_aos(&aos_data[row * blocks_per_row], input.data(), blocks_per_row);
        soa_results[row] = cpu_row_dot_soa(soa_data.data(), row, blocks_per_row, nblocks, input.data());
    }

    bool pass = compare_results(soa_results.data(), aos_results.data(), nrows);
    printf("  %s\n", pass ? "PASSED" : "FAILED");
    return pass;
}

bool test_gpu_aos_kernel() {
    printf("Test: GPU AoS kernel correctness...\n");

    const int ncols = 2880;
    const int nrows = 32;  // More rows for GPU
    const int blocks_per_row = ncols / QK_MXFP4;
    const int nblocks = blocks_per_row * nrows;

    std::mt19937 rng(456);

    // Generate test data
    std::vector<block_mxfp4> aos_data(nblocks);
    generate_random_mxfp4_blocks(aos_data.data(), nblocks, rng);

    std::vector<float> input(ncols);
    generate_random_input(input.data(), ncols, rng);

    // CPU reference results
    std::vector<float> cpu_results(nrows);
    for (int row = 0; row < nrows; row++) {
        cpu_results[row] = cpu_row_dot_aos(&aos_data[row * blocks_per_row], input.data(), blocks_per_row);
    }

    // GPU computation
    sycl::queue queue(sycl::default_selector_v);
    printf("  Device: %s\n", queue.get_device().get_info<sycl::info::device::name>().c_str());

    uint8_t* d_weights = sycl::malloc_device<uint8_t>(nblocks * BYTES_PER_BLOCK, queue);
    float* d_input = sycl::malloc_device<float>(ncols, queue);
    float* d_output = sycl::malloc_device<float>(nrows, queue);

    queue.memcpy(d_weights, aos_data.data(), nblocks * BYTES_PER_BLOCK).wait();
    queue.memcpy(d_input, input.data(), ncols * sizeof(float)).wait();

    queue.parallel_for(sycl::range<1>(nrows), [=](sycl::id<1> id) {
        sycl::nd_item<1> item = sycl::ext::oneapi::experimental::this_nd_item<1>();
        // Inline kernel since we can't call a device function directly
        const int row = id[0];
        const block_mxfp4* row_blocks = (const block_mxfp4*)(d_weights + row * blocks_per_row * sizeof(block_mxfp4));

        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            float scale_val;
            uint8_t e = row_blocks[b].e;
            if (e == 0) {
                union { uint32_t u; float f; } bits = { 0x00400000 };
                scale_val = bits.f;
            } else {
                union { uint32_t u; float f; } bits = { ((uint32_t)e) << 23 };
                scale_val = bits.f;
            }

            for (int i = 0; i < 16; i++) {
                uint8_t packed = row_blocks[b].qs[i];
                // Inline kvalues
                const float kv[16] = {0, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                      -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
                float w_lo = scale_val * kv[packed & 0xF];
                float w_hi = scale_val * kv[packed >> 4];
                sum += w_lo * d_input[b * 32 + i] + w_hi * d_input[b * 32 + i + 16];
            }
        }
        d_output[row] = sum;
    }).wait();

    std::vector<float> gpu_results(nrows);
    queue.memcpy(gpu_results.data(), d_output, nrows * sizeof(float)).wait();

    sycl::free(d_weights, queue);
    sycl::free(d_input, queue);
    sycl::free(d_output, queue);

    bool pass = compare_results(gpu_results.data(), cpu_results.data(), nrows);
    printf("  %s\n", pass ? "PASSED" : "FAILED");
    return pass;
}

bool test_gpu_soa_kernel() {
    printf("Test: GPU SoA kernel correctness...\n");

    const int ncols = 2880;
    const int nrows = 32;
    const int blocks_per_row = ncols / QK_MXFP4;
    const int nblocks = blocks_per_row * nrows;
    const int qs_region_size = nblocks * (QK_MXFP4 / 2);  // Total qs bytes

    std::mt19937 rng(789);

    // Generate AoS test data
    std::vector<block_mxfp4> aos_data(nblocks);
    generate_random_mxfp4_blocks(aos_data.data(), nblocks, rng);

    std::vector<float> input(ncols);
    generate_random_input(input.data(), ncols, rng);

    // Reorder to SoA
    std::vector<uint8_t> soa_data(nblocks * BYTES_PER_BLOCK);
    reorder_mxfp4_aos_to_soa(soa_data.data(), aos_data.data(), ncols, nrows);

    // CPU reference results (using AoS for ground truth)
    std::vector<float> cpu_results(nrows);
    for (int row = 0; row < nrows; row++) {
        cpu_results[row] = cpu_row_dot_aos(&aos_data[row * blocks_per_row], input.data(), blocks_per_row);
    }

    // GPU computation
    sycl::queue queue(sycl::default_selector_v);
    printf("  Device: %s\n", queue.get_device().get_info<sycl::info::device::name>().c_str());

    uint8_t* d_soa = sycl::malloc_device<uint8_t>(nblocks * BYTES_PER_BLOCK, queue);
    float* d_input = sycl::malloc_device<float>(ncols, queue);
    float* d_output = sycl::malloc_device<float>(nrows, queue);

    queue.memcpy(d_soa, soa_data.data(), nblocks * BYTES_PER_BLOCK).wait();
    queue.memcpy(d_input, input.data(), ncols * sizeof(float)).wait();

    // Compute SoA pointers (matching kernel expectations)
    const uint8_t* weights_qs = d_soa;
    const uint8_t* weights_scales = d_soa + qs_region_size;

    queue.parallel_for(sycl::range<1>(nrows), [=](sycl::id<1> id) {
        const int row = id[0];

        // SoA layout: compute absolute block offset for this row
        const int64_t row_block_offset = row * blocks_per_row;

        // qs region: each block contributes 16 bytes
        const uint8_t* qs_row = weights_qs + row_block_offset * 16;
        // scales region: each block contributes 1 byte
        const uint8_t* scale_row = weights_scales + row_block_offset;

        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            // E8M0 to float
            float scale_val;
            uint8_t e = scale_row[b];
            if (e == 0) {
                union { uint32_t u; float f; } bits = { 0x00400000 };
                scale_val = bits.f;
            } else {
                union { uint32_t u; float f; } bits = { ((uint32_t)e) << 23 };
                scale_val = bits.f;
            }

            for (int i = 0; i < 16; i++) {
                uint8_t packed = qs_row[b * 16 + i];
                const float kv[16] = {0, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                      -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
                float w_lo = scale_val * kv[packed & 0xF];
                float w_hi = scale_val * kv[packed >> 4];
                sum += w_lo * d_input[b * 32 + i] + w_hi * d_input[b * 32 + i + 16];
            }
        }
        d_output[row] = sum;
    }).wait();

    std::vector<float> gpu_results(nrows);
    queue.memcpy(gpu_results.data(), d_output, nrows * sizeof(float)).wait();

    sycl::free(d_soa, queue);
    sycl::free(d_input, queue);
    sycl::free(d_output, queue);

    bool pass = compare_results(gpu_results.data(), cpu_results.data(), nrows);
    printf("  %s\n", pass ? "PASSED" : "FAILED");
    return pass;
}

bool test_gpu_aos_vs_soa() {
    printf("Test: GPU AoS vs SoA equivalence...\n");

    const int ncols = 2880;
    const int nrows = 32;
    const int blocks_per_row = ncols / QK_MXFP4;
    const int nblocks = blocks_per_row * nrows;
    const int qs_region_size = nblocks * (QK_MXFP4 / 2);

    std::mt19937 rng(101112);

    // Generate test data
    std::vector<block_mxfp4> aos_data(nblocks);
    generate_random_mxfp4_blocks(aos_data.data(), nblocks, rng);

    std::vector<float> input(ncols);
    generate_random_input(input.data(), ncols, rng);

    // Reorder to SoA
    std::vector<uint8_t> soa_data(nblocks * BYTES_PER_BLOCK);
    reorder_mxfp4_aos_to_soa(soa_data.data(), aos_data.data(), ncols, nrows);

    sycl::queue queue(sycl::default_selector_v);
    printf("  Device: %s\n", queue.get_device().get_info<sycl::info::device::name>().c_str());

    // Allocate device memory for both layouts
    uint8_t* d_aos = sycl::malloc_device<uint8_t>(nblocks * BYTES_PER_BLOCK, queue);
    uint8_t* d_soa = sycl::malloc_device<uint8_t>(nblocks * BYTES_PER_BLOCK, queue);
    float* d_input = sycl::malloc_device<float>(ncols, queue);
    float* d_aos_output = sycl::malloc_device<float>(nrows, queue);
    float* d_soa_output = sycl::malloc_device<float>(nrows, queue);

    queue.memcpy(d_aos, aos_data.data(), nblocks * BYTES_PER_BLOCK).wait();
    queue.memcpy(d_soa, soa_data.data(), nblocks * BYTES_PER_BLOCK).wait();
    queue.memcpy(d_input, input.data(), ncols * sizeof(float)).wait();

    // Run AoS kernel
    queue.parallel_for(sycl::range<1>(nrows), [=](sycl::id<1> id) {
        const int row = id[0];
        const block_mxfp4* row_blocks = (const block_mxfp4*)(d_aos + row * blocks_per_row * sizeof(block_mxfp4));

        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            float scale_val;
            uint8_t e = row_blocks[b].e;
            if (e == 0) {
                union { uint32_t u; float f; } bits = { 0x00400000 };
                scale_val = bits.f;
            } else {
                union { uint32_t u; float f; } bits = { ((uint32_t)e) << 23 };
                scale_val = bits.f;
            }

            for (int i = 0; i < 16; i++) {
                uint8_t packed = row_blocks[b].qs[i];
                const float kv[16] = {0, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                      -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
                float w_lo = scale_val * kv[packed & 0xF];
                float w_hi = scale_val * kv[packed >> 4];
                sum += w_lo * d_input[b * 32 + i] + w_hi * d_input[b * 32 + i + 16];
            }
        }
        d_aos_output[row] = sum;
    }).wait();

    // Run SoA kernel
    const uint8_t* weights_qs = d_soa;
    const uint8_t* weights_scales = d_soa + qs_region_size;

    queue.parallel_for(sycl::range<1>(nrows), [=](sycl::id<1> id) {
        const int row = id[0];

        const int64_t row_block_offset = row * blocks_per_row;
        const uint8_t* qs_row = weights_qs + row_block_offset * 16;
        const uint8_t* scale_row = weights_scales + row_block_offset;

        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            float scale_val;
            uint8_t e = scale_row[b];
            if (e == 0) {
                union { uint32_t u; float f; } bits = { 0x00400000 };
                scale_val = bits.f;
            } else {
                union { uint32_t u; float f; } bits = { ((uint32_t)e) << 23 };
                scale_val = bits.f;
            }

            for (int i = 0; i < 16; i++) {
                uint8_t packed = qs_row[b * 16 + i];
                const float kv[16] = {0, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                      -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
                float w_lo = scale_val * kv[packed & 0xF];
                float w_hi = scale_val * kv[packed >> 4];
                sum += w_lo * d_input[b * 32 + i] + w_hi * d_input[b * 32 + i + 16];
            }
        }
        d_soa_output[row] = sum;
    }).wait();

    std::vector<float> aos_results(nrows);
    std::vector<float> soa_results(nrows);
    queue.memcpy(aos_results.data(), d_aos_output, nrows * sizeof(float)).wait();
    queue.memcpy(soa_results.data(), d_soa_output, nrows * sizeof(float)).wait();

    sycl::free(d_aos, queue);
    sycl::free(d_soa, queue);
    sycl::free(d_input, queue);
    sycl::free(d_aos_output, queue);
    sycl::free(d_soa_output, queue);

    bool pass = compare_results(soa_results.data(), aos_results.data(), nrows);
    printf("  %s\n", pass ? "PASSED" : "FAILED");
    return pass;
}

bool test_moe_expert_layout() {
    printf("Test: MoE expert layout (multi-expert SoA)...\n");

    const int ncols = 2880;           // Hidden dim per expert
    const int nrows_per_expert = 32;  // Rows per expert
    const int num_experts = 8;        // Number of experts
    const int blocks_per_row = ncols / QK_MXFP4;
    const int blocks_per_expert = blocks_per_row * nrows_per_expert;
    const int total_blocks = blocks_per_expert * num_experts;
    const int qs_region_size = total_blocks * (QK_MXFP4 / 2);

    std::mt19937 rng(131415);

    // Generate AoS data for all experts (contiguous)
    std::vector<block_mxfp4> aos_data(total_blocks);
    generate_random_mxfp4_blocks(aos_data.data(), total_blocks, rng);

    std::vector<float> input(ncols);
    generate_random_input(input.data(), ncols, rng);

    // Reorder entire tensor to SoA (as production code does)
    // nrows for reorder = nrows_per_expert * num_experts (total rows)
    std::vector<uint8_t> soa_data(total_blocks * BYTES_PER_BLOCK);
    reorder_mxfp4_aos_to_soa(soa_data.data(), aos_data.data(), ncols, nrows_per_expert * num_experts);

    sycl::queue queue(sycl::default_selector_v);
    printf("  Device: %s\n", queue.get_device().get_info<sycl::info::device::name>().c_str());
    printf("  Experts: %d, Rows/expert: %d, Blocks/row: %d\n",
           num_experts, nrows_per_expert, blocks_per_row);

    // Test each expert's first row
    uint8_t* d_soa = sycl::malloc_device<uint8_t>(total_blocks * BYTES_PER_BLOCK, queue);
    float* d_input = sycl::malloc_device<float>(ncols, queue);
    float* d_output = sycl::malloc_device<float>(num_experts, queue);

    queue.memcpy(d_soa, soa_data.data(), total_blocks * BYTES_PER_BLOCK).wait();
    queue.memcpy(d_input, input.data(), ncols * sizeof(float)).wait();

    const uint8_t* weights_qs = d_soa;
    const uint8_t* weights_scales = d_soa + qs_region_size;

    // Each work-item computes one expert's first row
    queue.parallel_for(sycl::range<1>(num_experts), [=](sycl::id<1> id) {
        const int expert_id = id[0];
        const int row = 0;  // First row of each expert

        // Same calculation as fused_moe_mxfp4_soa_kernel
        const int64_t row_block_offset = (expert_id * nrows_per_expert + row) * blocks_per_row;
        const uint8_t* qs_row = weights_qs + row_block_offset * 16;
        const uint8_t* scale_row = weights_scales + row_block_offset;

        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            float scale_val;
            uint8_t e = scale_row[b];
            if (e == 0) {
                union { uint32_t u; float f; } bits = { 0x00400000 };
                scale_val = bits.f;
            } else {
                union { uint32_t u; float f; } bits = { ((uint32_t)e) << 23 };
                scale_val = bits.f;
            }

            for (int i = 0; i < 16; i++) {
                uint8_t packed = qs_row[b * 16 + i];
                const float kv[16] = {0, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                      -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
                float w_lo = scale_val * kv[packed & 0xF];
                float w_hi = scale_val * kv[packed >> 4];
                sum += w_lo * d_input[b * 32 + i] + w_hi * d_input[b * 32 + i + 16];
            }
        }
        d_output[expert_id] = sum;
    }).wait();

    std::vector<float> gpu_results(num_experts);
    queue.memcpy(gpu_results.data(), d_output, num_experts * sizeof(float)).wait();

    // CPU reference using AoS
    std::vector<float> cpu_results(num_experts);
    for (int expert_id = 0; expert_id < num_experts; expert_id++) {
        int expert_start_block = expert_id * blocks_per_expert;
        cpu_results[expert_id] = cpu_row_dot_aos(&aos_data[expert_start_block], input.data(), blocks_per_row);
    }

    sycl::free(d_soa, queue);
    sycl::free(d_input, queue);
    sycl::free(d_output, queue);

    bool pass = compare_results(gpu_results.data(), cpu_results.data(), num_experts);
    printf("  %s\n", pass ? "PASSED" : "FAILED");
    return pass;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("MXFP4 SoA vs AoS Unit Tests\n");
    printf("===========================\n\n");

    int passed = 0;
    int total = 0;

    auto run_test = [&](bool (*test_fn)()) {
        total++;
        if (test_fn()) passed++;
        printf("\n");
    };

    // CPU tests
    run_test(test_cpu_reorder_correctness);
    run_test(test_cpu_aos_vs_soa_dot);

    // GPU tests
    run_test(test_gpu_aos_kernel);
    run_test(test_gpu_soa_kernel);
    run_test(test_gpu_aos_vs_soa);
    run_test(test_moe_expert_layout);

    printf("===========================\n");
    printf("Results: %d/%d tests passed\n", passed, total);

    return (passed == total) ? 0 : 1;
}
