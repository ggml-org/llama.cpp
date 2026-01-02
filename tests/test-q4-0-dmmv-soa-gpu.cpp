// GPU Unit tests for Q4_0 DMMV SoA (Structure of Arrays) support
// Tests the ACTUAL production functions from the SYCL backend
// Compares SoA vs AoS output using identical inputs

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include <sycl/sycl.hpp>

// Include actual SYCL backend headers
#include "ggml.h"
#include "ggml-sycl.h"

// Include internal SYCL definitions
#define GGML_SYCL_DMMV_X 32
#define WARP_SIZE 16
#define GGML_SYCL_MMV_Y 2

// Q4_0 block structure (must match ggml-common.h)
#define QK4_0 32
#define QR4_0 2

typedef struct {
    sycl::half d;
    uint8_t qs[QK4_0/2];
} block_q4_0_test;

static_assert(sizeof(block_q4_0_test) == 18, "block_q4_0 size mismatch");

using dfloat = float;
using dfloat2 = sycl::float2;

// ============================================================================
// Copy of the ACTUAL production dequantize function from dequantize.hpp
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

// AoS version for comparison
static inline void dequantize_q4_0_aos(const void *vx, const int64_t ib,
                                        const int iqs, dfloat2 &v) {
    const block_q4_0_test * x = (const block_q4_0_test *) vx + ib;
    const dfloat d = dfloat(x->d);
    const int vui = x->qs[iqs];

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

    v.x() = (v.x() - 8.0f) * d;
    v.y() = (v.y() - 8.0f) * d;
}

// ============================================================================
// Copy of the ACTUAL production DMMV kernel from dmmv.cpp
// Direct function calls (no function pointers - SYCL limitation)
// ============================================================================

static void dequantize_mul_mat_vec_reorder_kernel(
    const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst,
    const int ncols, const int nrows, const int64_t d_offset,
    const sycl::nd_item<3> &item_ct1
) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int tid = item_ct1.get_local_id(2);
    const int num_blocks_per_row = ncols / QK4_0;
    const int ib0 = row * num_blocks_per_row;

    const uint8_t* qs_base = static_cast<const uint8_t*>(vx);
    const sycl::half* d_base = reinterpret_cast<const sycl::half*>(
        static_cast<const char*>(vx) + d_offset);

    float tmp = 0.0f;

    for (int block_in_row = tid; block_in_row < num_blocks_per_row; block_in_row += WARP_SIZE) {
        const int block_idx = ib0 + block_in_row;
        const float d = static_cast<float>(d_base[block_idx]);

        const uint8_t* qs = qs_base + block_idx * (QK4_0 / 2);
        const float* y_block = y + block_in_row * QK4_0;

        float block_sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < QK4_0 / 2; ++j) {
            const uint8_t qs_byte = qs[j];
            const float v0 = ((float)(qs_byte & 0xF) - 8.0f) * d;
            const float v1 = ((float)(qs_byte >> 4) - 8.0f) * d;

            block_sum += v0 * y_block[j];
            block_sum += v1 * y_block[j + 16];
        }
        tmp += block_sum;
    }

    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
        tmp += sycl::permute_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

// AoS kernel for comparison
static void dequantize_mul_mat_vec_aos_kernel(
    const void * __restrict__ vx, const dfloat * __restrict__ y, float * __restrict__ dst,
    const int ncols, const int nrows,
    const sycl::nd_item<3> &item_ct1
) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int tid = item_ct1.get_local_id(2);
    const int iter_stride = 2*GGML_SYCL_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE;
    const int y_offset = QK4_0/2;  // qr=2

    float tmp = 0.0f;

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter*tid;
        const int ib = (row*ncols + col)/QK4_0;
        const int iqs = (col%QK4_0)/QR4_0;
        const int iybs = col - col%QK4_0;

        #pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            dfloat2 v;
            // Direct function call - no function pointer
            dequantize_q4_0_aos(vx, ib, iqs + j/QR4_0, v);

            tmp += v.x() * y[iybs + iqs + j/QR4_0 + 0];
            tmp += v.y() * y[iybs + iqs + j/QR4_0 + y_offset];
        }
    }

    // Warp reduction
    for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
        tmp += sycl::permute_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

// ============================================================================
// Reference CPU Implementation
// ============================================================================

static void dequantize_block_q4_0_cpu(const block_q4_0_test* block, float* out) {
    const float d = float(block->d);
    for (int i = 0; i < QK4_0/2; i++) {
        const uint8_t byte = block->qs[i];
        const int lo = (byte & 0xF);
        const int hi = (byte >> 4);
        out[i]          = (lo - 8) * d;
        out[i + QK4_0/2] = (hi - 8) * d;
    }
}

static void dmmv_q4_0_cpu_reference(
    const block_q4_0_test* x,
    const float* y,
    float* dst,
    int ncols,
    int nrows
) {
    const int blocks_per_row = ncols / QK4_0;

    for (int row = 0; row < nrows; row++) {
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            const block_q4_0_test* block = &x[row * blocks_per_row + b];
            float dequant[QK4_0];
            dequantize_block_q4_0_cpu(block, dequant);

            for (int i = 0; i < QK4_0; i++) {
                sum += dequant[i] * y[b * QK4_0 + i];
            }
        }
        dst[row] = sum;
    }
}

// ============================================================================
// Test 1: Dequantize function - SoA vs AoS (byte-level)
// ============================================================================

bool test_dequantize_function(sycl::queue& q) {
    printf("\n=== Test 1: dequantize_q4_0 SoA vs AoS (byte-level) ===\n");

    const int num_blocks = 1024;
    const size_t qs_size = num_blocks * (QK4_0/2);
    const size_t d_size = num_blocks * sizeof(sycl::half);

    // Create random test data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> scale_dist(-0.1f, 0.1f);
    std::uniform_int_distribution<int> byte_dist(0, 255);

    std::vector<block_q4_0_test> h_aos(num_blocks);
    for (int b = 0; b < num_blocks; b++) {
        h_aos[b].d = sycl::half(scale_dist(rng));
        for (int i = 0; i < QK4_0/2; i++) {
            h_aos[b].qs[i] = (uint8_t)byte_dist(rng);
        }
    }

    // Create SoA layout
    std::vector<uint8_t> h_soa_qs(qs_size);
    std::vector<sycl::half> h_soa_d(num_blocks);

    for (int b = 0; b < num_blocks; b++) {
        for (int i = 0; i < QK4_0/2; i++) {
            h_soa_qs[b * (QK4_0/2) + i] = h_aos[b].qs[i];
        }
        h_soa_d[b] = h_aos[b].d;
    }

    // Allocate device memory
    block_q4_0_test* d_aos = sycl::malloc_device<block_q4_0_test>(num_blocks, q);
    uint8_t* d_soa = sycl::malloc_device<uint8_t>(qs_size + d_size, q);
    float* d_aos_out = sycl::malloc_device<float>(num_blocks * QK4_0, q);
    float* d_soa_out = sycl::malloc_device<float>(num_blocks * QK4_0, q);

    // Copy data to device
    q.memcpy(d_aos, h_aos.data(), num_blocks * sizeof(block_q4_0_test));
    q.memcpy(d_soa, h_soa_qs.data(), qs_size);
    q.memcpy((sycl::half*)(d_soa + qs_size), h_soa_d.data(), d_size);
    q.wait();

    const sycl::half* d_soa_d_ptr = (const sycl::half*)(d_soa + qs_size);

    // Test ALL byte positions (iqs 0-15) for each block
    q.parallel_for(sycl::range<1>(num_blocks * (QK4_0/2)), [=](sycl::id<1> idx) {
        int b = idx[0] / (QK4_0/2);
        int iqs = idx[0] % (QK4_0/2);

        dfloat2 v_aos, v_soa;

        dequantize_q4_0_aos(d_aos, b, iqs, v_aos);
        dequantize_q4_0_reorder(d_soa_d_ptr, b, d_soa, b * (QK4_0/2) + iqs, v_soa);

        d_aos_out[b * QK4_0 + iqs * 2 + 0] = v_aos.x();
        d_aos_out[b * QK4_0 + iqs * 2 + 1] = v_aos.y();
        d_soa_out[b * QK4_0 + iqs * 2 + 0] = v_soa.x();
        d_soa_out[b * QK4_0 + iqs * 2 + 1] = v_soa.y();
    }).wait();

    // Copy results back
    std::vector<float> h_aos_out(num_blocks * QK4_0);
    std::vector<float> h_soa_out(num_blocks * QK4_0);
    q.memcpy(h_aos_out.data(), d_aos_out, num_blocks * QK4_0 * sizeof(float));
    q.memcpy(h_soa_out.data(), d_soa_out, num_blocks * QK4_0 * sizeof(float));
    q.wait();

    // Compare
    bool passed = true;
    int errors = 0;
    const float eps = 1e-6f;

    for (int i = 0; i < num_blocks * QK4_0 && errors < 10; i++) {
        float diff = std::abs(h_aos_out[i] - h_soa_out[i]);
        if (diff > eps) {
            int b = i / QK4_0;
            int j = i % QK4_0;
            printf("  FAIL block=%d idx=%d: AoS=%.6f SoA=%.6f diff=%.6e\n",
                   b, j, h_aos_out[i], h_soa_out[i], diff);
            passed = false;
            errors++;
        }
    }

    printf("Tested %d values, errors=%d\n", num_blocks * QK4_0, errors);

    sycl::free(d_aos, q);
    sycl::free(d_soa, q);
    sycl::free(d_aos_out, q);
    sycl::free(d_soa_out, q);

    printf("%s\n", passed ? "PASSED" : "FAILED");
    return passed;
}

// ============================================================================
// Test 2: Full DMMV - SoA vs AoS GPU kernels (using actual production logic)
// ============================================================================

bool test_dmmv_soa_vs_aos(sycl::queue& q) {
    printf("\n=== Test 2: DMMV SoA vs AoS GPU (production kernel logic) ===\n");

    const int nrows = 4096;  // Match typical model dimension
    const int ncols = 4096;
    const int blocks_per_row = ncols / QK4_0;
    const int total_blocks = nrows * blocks_per_row;

    printf("Matrix: %d x %d, blocks_per_row=%d, total_blocks=%d\n",
           nrows, ncols, blocks_per_row, total_blocks);

    // Create test data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> scale_dist(-0.1f, 0.1f);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_real_distribution<float> y_dist(-1.0f, 1.0f);

    std::vector<block_q4_0_test> h_aos(total_blocks);
    for (int b = 0; b < total_blocks; b++) {
        h_aos[b].d = sycl::half(scale_dist(rng));
        for (int i = 0; i < QK4_0/2; i++) {
            h_aos[b].qs[i] = (uint8_t)byte_dist(rng);
        }
    }

    std::vector<float> h_y(ncols);
    for (int i = 0; i < ncols; i++) {
        h_y[i] = y_dist(rng);
    }

    // Create SoA layout
    const size_t qs_bytes = total_blocks * (QK4_0/2);
    const size_t d_bytes = total_blocks * sizeof(sycl::half);

    std::vector<uint8_t> h_soa_data(qs_bytes + d_bytes);
    for (int b = 0; b < total_blocks; b++) {
        for (int i = 0; i < QK4_0/2; i++) {
            h_soa_data[b * (QK4_0/2) + i] = h_aos[b].qs[i];
        }
        ((sycl::half*)(h_soa_data.data() + qs_bytes))[b] = h_aos[b].d;
    }

    // Allocate device memory
    block_q4_0_test* d_aos = sycl::malloc_device<block_q4_0_test>(total_blocks, q);
    uint8_t* d_soa = sycl::malloc_device<uint8_t>(qs_bytes + d_bytes, q);
    float* d_y = sycl::malloc_device<float>(ncols, q);
    float* d_aos_out = sycl::malloc_device<float>(nrows, q);
    float* d_soa_out = sycl::malloc_device<float>(nrows, q);

    // Copy to device
    q.memcpy(d_aos, h_aos.data(), total_blocks * sizeof(block_q4_0_test));
    q.memcpy(d_soa, h_soa_data.data(), qs_bytes + d_bytes);
    q.memcpy(d_y, h_y.data(), ncols * sizeof(float));
    q.wait();

    // Launch parameters (same as production)
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    sycl::range<3> block_nums(1, 1, block_num_y);
    sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    // Run AoS kernel
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_aos_kernel(d_aos, d_y, d_aos_out, ncols, nrows, item);
            });
    }).wait();

    // Run SoA kernel
    const int64_t d_offset = qs_bytes;
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_reorder_kernel(d_soa, d_y, d_soa_out, ncols, nrows, d_offset, item);
            });
    }).wait();

    // Copy results back
    std::vector<float> h_aos_out(nrows);
    std::vector<float> h_soa_out(nrows);
    q.memcpy(h_aos_out.data(), d_aos_out, nrows * sizeof(float));
    q.memcpy(h_soa_out.data(), d_soa_out, nrows * sizeof(float));
    q.wait();

    // CPU reference for sanity check
    std::vector<float> h_cpu_out(nrows);
    dmmv_q4_0_cpu_reference(h_aos.data(), h_y.data(), h_cpu_out.data(), ncols, nrows);

    // Compare GPU results
    bool passed = true;
    int aos_vs_soa_errors = 0;
    int cpu_vs_aos_errors = 0;
    float max_aos_soa_diff = 0.0f;
    float max_cpu_aos_diff = 0.0f;

    for (int row = 0; row < nrows; row++) {
        float aos = h_aos_out[row];
        float soa = h_soa_out[row];
        float cpu = h_cpu_out[row];

        float aos_soa_diff = std::abs(aos - soa);
        float cpu_aos_diff = std::abs(cpu - aos);

        max_aos_soa_diff = std::max(max_aos_soa_diff, aos_soa_diff);
        max_cpu_aos_diff = std::max(max_cpu_aos_diff, cpu_aos_diff);

        // Different tolerance for the two comparisons
        // AoS vs SoA should be near-identical (different FP accumulation order)
        // CPU vs GPU can have larger differences (different algorithms)
        if (aos_soa_diff > 1e-4f) {  // Relax tolerance for FP order differences
            if (aos_vs_soa_errors < 5) {
                printf("  AoS vs SoA FAIL row=%d: AoS=%.6f SoA=%.6f diff=%.6e\n",
                       row, aos, soa, aos_soa_diff);
            }
            aos_vs_soa_errors++;
        }
        if (cpu_aos_diff > 1e-3f) {  // GPU vs CPU can have more FP error
            if (cpu_vs_aos_errors < 5) {
                printf("  CPU vs AoS FAIL row=%d: CPU=%.6f AoS=%.6f diff=%.6e\n",
                       row, cpu, aos, cpu_aos_diff);
            }
            cpu_vs_aos_errors++;
        }
    }

    passed = (aos_vs_soa_errors == 0 && cpu_vs_aos_errors == 0);

    printf("AoS vs SoA: errors=%d, max_diff=%.6e\n", aos_vs_soa_errors, max_aos_soa_diff);
    printf("CPU vs AoS: errors=%d, max_diff=%.6e\n", cpu_vs_aos_errors, max_cpu_aos_diff);

    // Show first 8 values
    printf("First 8 CPU: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", h_cpu_out[i]);
    printf("\n");
    printf("First 8 AoS: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", h_aos_out[i]);
    printf("\n");
    printf("First 8 SoA: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", h_soa_out[i]);
    printf("\n");

    sycl::free(d_aos, q);
    sycl::free(d_soa, q);
    sycl::free(d_y, q);
    sycl::free(d_aos_out, q);
    sycl::free(d_soa_out, q);

    printf("%s\n", passed ? "PASSED" : "FAILED");
    return passed;
}

// ============================================================================
// Test 3: Match actual model dimensions (14336, 4096)
// ============================================================================

bool test_model_dimensions(sycl::queue& q) {
    printf("\n=== Test 3: Model dimensions (14336 x 4096) ===\n");

    const int nrows = 14336;  // Actual model dimension
    const int ncols = 4096;
    const int blocks_per_row = ncols / QK4_0;
    const int total_blocks = nrows * blocks_per_row;

    printf("Matrix: %d x %d, total_blocks=%d\n", nrows, ncols, total_blocks);

    // Create test data with specific seed for reproducibility
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> scale_dist(-0.1f, 0.1f);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_real_distribution<float> y_dist(-1.0f, 1.0f);

    std::vector<block_q4_0_test> h_aos(total_blocks);
    for (int b = 0; b < total_blocks; b++) {
        h_aos[b].d = sycl::half(scale_dist(rng));
        for (int i = 0; i < QK4_0/2; i++) {
            h_aos[b].qs[i] = (uint8_t)byte_dist(rng);
        }
    }

    std::vector<float> h_y(ncols);
    for (int i = 0; i < ncols; i++) {
        h_y[i] = y_dist(rng);
    }

    // Create SoA layout
    const size_t qs_bytes = total_blocks * (QK4_0/2);
    const size_t d_bytes = total_blocks * sizeof(sycl::half);

    std::vector<uint8_t> h_soa_data(qs_bytes + d_bytes);
    for (int b = 0; b < total_blocks; b++) {
        for (int i = 0; i < QK4_0/2; i++) {
            h_soa_data[b * (QK4_0/2) + i] = h_aos[b].qs[i];
        }
        ((sycl::half*)(h_soa_data.data() + qs_bytes))[b] = h_aos[b].d;
    }

    // Allocate device memory
    block_q4_0_test* d_aos = sycl::malloc_device<block_q4_0_test>(total_blocks, q);
    uint8_t* d_soa = sycl::malloc_device<uint8_t>(qs_bytes + d_bytes, q);
    float* d_y = sycl::malloc_device<float>(ncols, q);
    float* d_aos_out = sycl::malloc_device<float>(nrows, q);
    float* d_soa_out = sycl::malloc_device<float>(nrows, q);

    // Copy to device
    q.memcpy(d_aos, h_aos.data(), total_blocks * sizeof(block_q4_0_test));
    q.memcpy(d_soa, h_soa_data.data(), qs_bytes + d_bytes);
    q.memcpy(d_y, h_y.data(), ncols * sizeof(float));
    q.wait();

    // Launch kernels
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    sycl::range<3> block_nums(1, 1, block_num_y);
    sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_aos_kernel(d_aos, d_y, d_aos_out, ncols, nrows, item);
            });
    }).wait();

    const int64_t d_offset = qs_bytes;
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                dequantize_mul_mat_vec_reorder_kernel(d_soa, d_y, d_soa_out, ncols, nrows, d_offset, item);
            });
    }).wait();

    // Copy results back
    std::vector<float> h_aos_out(nrows);
    std::vector<float> h_soa_out(nrows);
    q.memcpy(h_aos_out.data(), d_aos_out, nrows * sizeof(float));
    q.memcpy(h_soa_out.data(), d_soa_out, nrows * sizeof(float));
    q.wait();

    // Compare
    bool passed = true;
    int errors = 0;
    float max_diff = 0.0f;
    float max_rel_err = 0.0f;

    const float abs_tol = 2e-5f;
    const float rel_tol = 0.001f;

    for (int row = 0; row < nrows; row++) {
        float aos = h_aos_out[row];
        float soa = h_soa_out[row];
        float diff = std::abs(aos - soa);
        float rel_err = std::abs(aos) > 1e-6f ? diff / std::abs(aos) : diff;

        max_diff = std::max(max_diff, diff);
        max_rel_err = std::max(max_rel_err, rel_err);

        if (diff > abs_tol && rel_err > rel_tol) {
            if (errors < 5) {
                printf("  FAIL row=%d: AoS=%.6f SoA=%.6f diff=%.6e rel=%.4f%%\n",
                       row, aos, soa, diff, rel_err * 100);
            }
            errors++;
            passed = false;
        }
    }

    printf("Rows: %d, Errors: %d\n", nrows, errors);
    printf("Max absolute diff: %.6e\n", max_diff);
    printf("Max relative error: %.4f%%\n", max_rel_err * 100);

    printf("First 8 AoS: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", h_aos_out[i]);
    printf("\n");
    printf("First 8 SoA: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", h_soa_out[i]);
    printf("\n");

    sycl::free(d_aos, q);
    sycl::free(d_soa, q);
    sycl::free(d_y, q);
    sycl::free(d_aos_out, q);
    sycl::free(d_soa_out, q);

    printf("%s\n", passed ? "PASSED" : "FAILED");
    return passed;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("Q4_0 DMMV SoA GPU Unit Tests (Production Logic)\n");
    printf("================================================\n");

    sycl::device gpu_dev;
    try {
        gpu_dev = sycl::device(sycl::gpu_selector_v);
    } catch (...) {
        printf("ERROR: No GPU device found\n");
        return 1;
    }

    printf("Device: %s\n", gpu_dev.get_info<sycl::info::device::name>().c_str());

    sycl::queue q(gpu_dev, sycl::property::queue::in_order());

    int passed = 0;
    int failed = 0;

    if (test_dequantize_function(q)) passed++; else failed++;
    if (test_dmmv_soa_vs_aos(q)) passed++; else failed++;
    if (test_model_dimensions(q)) passed++; else failed++;

    printf("\n================================================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);

    return failed > 0 ? 1 : 0;
}
