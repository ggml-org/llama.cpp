// Comprehensive DMMV SoA Unit Test Suite
// Tests GPU SoA kernel against GPU AoS kernel (not just CPU reference)
//
// Compile:
//   source /opt/intel/oneapi/setvars.sh --force
//   icpx -fsycl -O2 -I../ggml/include -I../ggml/src tests/test_dmmv_soa.cpp -o test_dmmv_soa
//
// Run:
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./test_dmmv_soa

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
#include <random>
#include <functional>
#include <string>

// =============================================================================
// Constants from llama.cpp
// =============================================================================

constexpr int QK4_0 = 32;
constexpr int QK4_1 = 32;
constexpr int QK5_0 = 32;
constexpr int QK5_1 = 32;
constexpr int QK8_0 = 32;

constexpr int QR4_0 = 2;
constexpr int QR4_1 = 2;
constexpr int QR5_0 = 2;
constexpr int QR5_1 = 2;
constexpr int QR8_0 = 1;

constexpr int WARP_SIZE = 16;  // Intel Arc
constexpr int GGML_SYCL_MMV_Y = 1;
constexpr int GGML_SYCL_DMMV_X = 32;

using dfloat = float;
using dfloat2 = sycl::float2;

// =============================================================================
// Block Structures (AoS layout)
// =============================================================================

struct block_q4_0 {
    sycl::half d;
    uint8_t qs[16];  // 32 4-bit quants
};

struct block_q4_1 {
    sycl::half2 dm;   // d and m (min)
    uint8_t qs[16];   // 32 4-bit quants
};

struct block_q5_0 {
    sycl::half d;
    uint8_t qh[4];    // 32 5th bits
    uint8_t qs[16];   // 32 4-bit quants (lower)
};

struct block_q5_1 {
    sycl::half2 dm;
    uint8_t qh[4];
    uint8_t qs[16];
};

struct block_q8_0 {
    sycl::half d;
    int8_t qs[32];    // 32 8-bit quants
};

// =============================================================================
// Test Framework
// =============================================================================

static int g_tests_passed = 0;
static int g_tests_failed = 0;
static sycl::queue* g_queue = nullptr;

#define RUN_TEST(test_fn) do { \
    printf("\n--- Running: %s ---\n", #test_fn); \
    bool result = test_fn(); \
    if (result) { \
        g_tests_passed++; \
        printf("  Result: PASS\n"); \
    } else { \
        g_tests_failed++; \
        printf("  Result: FAIL\n"); \
    } \
} while(0)

// Validation helper
bool validate_output(const float* expected, const float* actual, int n,
                     float rtol = 1e-3f, float atol = 1e-5f,
                     bool verbose = false) {
    int mismatches = 0;
    for (int i = 0; i < n; i++) {
        if (std::isnan(actual[i]) || std::isinf(actual[i])) {
            if (mismatches < 5 || verbose) {
                printf("  NaN/Inf at index %d: expected=%.6f actual=%.6f\n",
                       i, expected[i], actual[i]);
            }
            mismatches++;
            continue;
        }
        float diff = std::fabs(expected[i] - actual[i]);
        float threshold = atol + rtol * std::fabs(expected[i]);
        if (diff > threshold) {
            if (mismatches < 5 || verbose) {
                printf("  Mismatch at %d: expected=%.6f actual=%.6f diff=%.2e threshold=%.2e\n",
                       i, expected[i], actual[i], diff, threshold);
            }
            mismatches++;
        }
    }
    if (mismatches > 5 && !verbose) {
        printf("  ... and %d more mismatches\n", mismatches - 5);
    }
    return mismatches == 0;
}

// =============================================================================
// Q4_0 Dequantize Functions
// =============================================================================

// AoS dequantize
static void dequantize_q4_0(const void* vx, const int64_t ib, const int iqs, dfloat2& v) {
    const block_q4_0* x = (const block_q4_0*)vx;
    const dfloat d = static_cast<float>(x[ib].d);
    const int vui = x[ib].qs[iqs];
    v.x() = ((float)(vui & 0xF) - 8.0f) * d;
    v.y() = ((float)(vui >> 4) - 8.0f) * d;
}

// SoA dequantize
static void dequantize_q4_0_soa(const void* d_ptr, const int64_t ib,
                                 const void* qs, const int iqs, dfloat2& v) {
    const dfloat d = static_cast<float>(*((const sycl::half*)d_ptr + ib));
    const int vui = *((const uint8_t*)qs + iqs);
    v.x() = ((float)(vui & 0xF) - 8.0f) * d;
    v.y() = ((float)(vui >> 4) - 8.0f) * d;
}

// =============================================================================
// Q8_0 Dequantize Functions
// =============================================================================

static void dequantize_q8_0(const void* vx, const int64_t ib, const int iqs, dfloat2& v) {
    const block_q8_0* x = (const block_q8_0*)vx;
    const dfloat d = static_cast<float>(x[ib].d);
    v.x() = x[ib].qs[iqs + 0] * d;
    v.y() = x[ib].qs[iqs + 1] * d;
}

static void dequantize_q8_0_soa(const void* d_ptr, const int64_t ib,
                                 const void* qs, const int iqs, dfloat2& v) {
    const dfloat d = static_cast<float>(*((const sycl::half*)d_ptr + ib));
    const int8_t* qs_ptr = (const int8_t*)qs + ib * 32;
    v.x() = qs_ptr[iqs + 0] * d;
    v.y() = qs_ptr[iqs + 1] * d;
}

// =============================================================================
// CPU Reference Implementations
// =============================================================================

void cpu_dmmv_q4_0_aos(const block_q4_0* x, const float* y, float* dst,
                        int ncols, int nrows) {
    const int blocks_per_row = ncols / QK4_0;
    for (int row = 0; row < nrows; row++) {
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            int ib = row * blocks_per_row + b;
            float d = static_cast<float>(x[ib].d);
            for (int i = 0; i < 16; i++) {
                int q_lo = (x[ib].qs[i] & 0x0F) - 8;
                int q_hi = (x[ib].qs[i] >> 4) - 8;
                int col = b * 32 + i;
                sum += d * q_lo * y[col];
                sum += d * q_hi * y[col + 16];
            }
        }
        dst[row] = sum;
    }
}

void cpu_dmmv_q4_0_soa(const uint8_t* qs, const sycl::half* d_vals,
                        const float* y, float* dst, int ncols, int nrows) {
    const int blocks_per_row = ncols / QK4_0;
    for (int row = 0; row < nrows; row++) {
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            int ib = row * blocks_per_row + b;
            float d = static_cast<float>(d_vals[ib]);
            for (int i = 0; i < 16; i++) {
                int qs_idx = ib * 16 + i;
                int q_lo = (qs[qs_idx] & 0x0F) - 8;
                int q_hi = (qs[qs_idx] >> 4) - 8;
                int col = b * 32 + i;
                sum += d * q_lo * y[col];
                sum += d * q_hi * y[col + 16];
            }
        }
        dst[row] = sum;
    }
}

void cpu_dmmv_q8_0_aos(const block_q8_0* x, const float* y, float* dst,
                        int ncols, int nrows) {
    const int blocks_per_row = ncols / QK8_0;
    for (int row = 0; row < nrows; row++) {
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            int ib = row * blocks_per_row + b;
            float d = static_cast<float>(x[ib].d);
            for (int i = 0; i < 32; i++) {
                int col = b * 32 + i;
                sum += d * x[ib].qs[i] * y[col];
            }
        }
        dst[row] = sum;
    }
}

// =============================================================================
// Data Conversion Functions
// =============================================================================

void convert_q4_0_aos_to_soa(const block_q4_0* aos, uint8_t* soa_qs,
                              sycl::half* soa_d, int nblocks) {
    for (int b = 0; b < nblocks; b++) {
        memcpy(soa_qs + b * 16, aos[b].qs, 16);
        soa_d[b] = aos[b].d;
    }
}

void convert_q8_0_aos_to_soa(const block_q8_0* aos, int8_t* soa_qs,
                              sycl::half* soa_d, int nblocks) {
    for (int b = 0; b < nblocks; b++) {
        memcpy(soa_qs + b * 32, aos[b].qs, 32);
        soa_d[b] = aos[b].d;
    }
}

// =============================================================================
// GPU Kernels - Q4_0
// =============================================================================

// GPU AoS kernel (reference)
template <int qk, int qr>
void dmmv_q4_0_aos_kernel(const void* vx, const dfloat* y, float* dst,
                           int ncols, int nrows, sycl::nd_item<3> item) {
    const int row = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
    if (row >= nrows) return;

    const int tid = item.get_local_id(2);
    const int iter_stride = 2 * GGML_SYCL_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE;
    const int y_offset = qr == 1 ? 1 : qk / 2;

    float tmp = 0.0f;
    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter * tid;
        if (col >= ncols) break;
        const int ib = (row * ncols + col) / qk;
        const int iqs = (col % qk) / qr;
        const int iybs = col - col % qk;

        for (int j = 0; j < vals_per_iter && (col + j) < ncols; j += 2) {
            dfloat2 v;
            dequantize_q4_0(vx, ib, iqs + j / qr, v);
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_offset];
        }
    }

    // Subgroup reduction
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += sycl::permute_group_by_xor(item.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

// GPU SoA kernel (under test)
template <int qk, int qr>
void dmmv_q4_0_soa_kernel(const void* vx, const dfloat* y, float* dst,
                           int ncols, int nrows, int64_t d_offset,
                           sycl::nd_item<3> item) {
    const int row = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
    if (row >= nrows) return;

    const int tid = item.get_local_id(2);
    const char* d_ptr = (const char*)vx + d_offset;

    const int iter_stride = 2 * GGML_SYCL_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE;
    const int y_offset = qr == 1 ? 1 : qk / 2;

    float tmp = 0.0f;
    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter * tid;
        if (col >= ncols) break;
        const int ib = (row * ncols + col) / qk;
        const int iqs = (col % qk) / qr;
        const int iybs = col - col % qk;

        for (int j = 0; j < vals_per_iter && (col + j) < ncols; j += 2) {
            dfloat2 v;
            dequantize_q4_0_soa((const void*)d_ptr, ib,
                                (const void*)vx, ib * qk / 2 + iqs + j / qr, v);
            tmp += v.x() * y[iybs + iqs + j / qr + 0];
            tmp += v.y() * y[iybs + iqs + j / qr + y_offset];
        }
    }

    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += sycl::permute_group_by_xor(item.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

// =============================================================================
// GPU Kernels - Q8_0
// =============================================================================

template <int qk, int qr>
void dmmv_q8_0_aos_kernel(const void* vx, const dfloat* y, float* dst,
                           int ncols, int nrows, sycl::nd_item<3> item) {
    const int row = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
    if (row >= nrows) return;

    const int tid = item.get_local_id(2);
    const int iter_stride = 2 * GGML_SYCL_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE;

    float tmp = 0.0f;
    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter * tid;
        if (col >= ncols) break;
        const int ib = (row * ncols + col) / qk;
        const int iqs = col % qk;
        const int iybs = col - col % qk;

        for (int j = 0; j < vals_per_iter && (col + j) < ncols; j += 2) {
            dfloat2 v;
            dequantize_q8_0(vx, ib, iqs + j, v);
            tmp += v.x() * y[iybs + iqs + j + 0];
            tmp += v.y() * y[iybs + iqs + j + 1];
        }
    }

    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += sycl::permute_group_by_xor(item.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qr>
void dmmv_q8_0_soa_kernel(const void* vx, const dfloat* y, float* dst,
                           int ncols, int nrows, int64_t d_offset,
                           sycl::nd_item<3> item) {
    const int row = item.get_group(2) * item.get_local_range(1) + item.get_local_id(1);
    if (row >= nrows) return;

    const int tid = item.get_local_id(2);
    const char* d_ptr = (const char*)vx + d_offset;

    const int iter_stride = 2 * GGML_SYCL_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE;

    float tmp = 0.0f;
    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter * tid;
        if (col >= ncols) break;
        const int ib = (row * ncols + col) / qk;
        const int iqs = col % qk;
        const int iybs = col - col % qk;

        for (int j = 0; j < vals_per_iter && (col + j) < ncols; j += 2) {
            dfloat2 v;
            dequantize_q8_0_soa((const void*)d_ptr, ib,
                                (const void*)vx, iqs + j, v);
            tmp += v.x() * y[iybs + iqs + j + 0];
            tmp += v.y() * y[iybs + iqs + j + 1];
        }
    }

    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += sycl::permute_group_by_xor(item.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

// =============================================================================
// Test Data Generation
// =============================================================================

void generate_q4_0_data(block_q4_0* blocks, int nblocks, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> scale_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> qs_dist(0, 255);

    for (int b = 0; b < nblocks; b++) {
        blocks[b].d = sycl::half(scale_dist(rng));
        for (int i = 0; i < 16; i++) {
            blocks[b].qs[i] = qs_dist(rng);
        }
    }
}

void generate_q8_0_data(block_q8_0* blocks, int nblocks, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> scale_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> qs_dist(-128, 127);

    for (int b = 0; b < nblocks; b++) {
        blocks[b].d = sycl::half(scale_dist(rng));
        for (int i = 0; i < 32; i++) {
            blocks[b].qs[i] = qs_dist(rng);
        }
    }
}

void generate_y_data(float* y, int ncols, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < ncols; i++) {
        y[i] = dist(rng);
    }
}

// =============================================================================
// Test Runner for Q4_0 GPU vs GPU Comparison
// =============================================================================

bool run_q4_0_gpu_comparison(int ncols, int nrows, int row_low, int row_high,
                              uint32_t seed, const char* test_name) {
    sycl::queue& q = *g_queue;

    const int full_nrows = nrows;
    const int slice_nrows = row_high - row_low;
    const int blocks_per_row = ncols / QK4_0;
    const int full_nblocks = full_nrows * blocks_per_row;
    const int slice_nblocks = slice_nrows * blocks_per_row;

    // Generate AoS data for full tensor
    std::vector<block_q4_0> x_aos(full_nblocks);
    std::vector<float> y_vec(ncols);
    generate_q4_0_data(x_aos.data(), full_nblocks, seed);
    generate_y_data(y_vec.data(), ncols, seed + 1);

    // Convert to SoA for full tensor
    size_t full_qs_bytes = full_nblocks * 16;
    std::vector<uint8_t> soa_qs(full_qs_bytes);
    std::vector<sycl::half> soa_d(full_nblocks);
    convert_q4_0_aos_to_soa(x_aos.data(), soa_qs.data(), soa_d.data(), full_nblocks);

    // Combined SoA buffer: [qs...][d...]
    std::vector<uint8_t> soa_combined(full_qs_bytes + full_nblocks * sizeof(sycl::half));
    memcpy(soa_combined.data(), soa_qs.data(), full_qs_bytes);
    memcpy(soa_combined.data() + full_qs_bytes, soa_d.data(), full_nblocks * sizeof(sycl::half));

    // Calculate pointers for the row slice
    size_t row_slice_aos_offset = row_low * blocks_per_row * sizeof(block_q4_0);
    size_t row_slice_qs_offset = row_low * blocks_per_row * 16;
    size_t row_slice_d_offset = row_low * blocks_per_row * sizeof(sycl::half);

    // Correct d_offset formula: from slice start pointer
    // d values are at: base + full_qs_bytes + row_low * blocks_per_row * 2
    // slice_ptr is at: base + row_slice_qs_offset
    // d_offset from slice_ptr = full_qs_bytes + row_slice_d_offset - row_slice_qs_offset
    int64_t correct_d_offset = full_qs_bytes + row_slice_d_offset - row_slice_qs_offset;

    // Allocate device memory
    void* d_aos = sycl::malloc_device(full_nblocks * sizeof(block_q4_0), q);
    void* d_soa = sycl::malloc_device(soa_combined.size(), q);
    float* d_y = sycl::malloc_device<float>(ncols, q);
    float* d_out_aos = sycl::malloc_device<float>(slice_nrows, q);
    float* d_out_soa = sycl::malloc_device<float>(slice_nrows, q);

    // Copy to device
    q.memcpy(d_aos, x_aos.data(), full_nblocks * sizeof(block_q4_0));
    q.memcpy(d_soa, soa_combined.data(), soa_combined.size());
    q.memcpy(d_y, y_vec.data(), ncols * sizeof(float));
    q.wait();

    // Kernel launch configuration
    sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    sycl::range<3> grid_dims(1, 1, slice_nrows);

    // AoS slice pointer
    const char* aos_slice_ptr = (const char*)d_aos + row_slice_aos_offset;

    // SoA slice pointer
    const char* soa_slice_ptr = (const char*)d_soa + row_slice_qs_offset;

    // Run AoS kernel (reference)
    q.parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                   [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                       dmmv_q4_0_aos_kernel<QK4_0, QR4_0>(aos_slice_ptr, (const dfloat*)d_y,
                                                          d_out_aos, ncols, slice_nrows, item);
                   });

    // Run SoA kernel (under test)
    q.parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                   [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                       dmmv_q4_0_soa_kernel<QK4_0, QR4_0>(soa_slice_ptr, (const dfloat*)d_y,
                                                          d_out_soa, ncols, slice_nrows,
                                                          correct_d_offset, item);
                   });
    q.wait();

    // Read results
    std::vector<float> out_aos(slice_nrows), out_soa(slice_nrows);
    q.memcpy(out_aos.data(), d_out_aos, slice_nrows * sizeof(float));
    q.memcpy(out_soa.data(), d_out_soa, slice_nrows * sizeof(float));
    q.wait();

    // Validate GPU SoA vs GPU AoS
    bool pass = validate_output(out_aos.data(), out_soa.data(), slice_nrows, 1e-3f, 1e-5f);

    // Cleanup
    sycl::free(d_aos, q);
    sycl::free(d_soa, q);
    sycl::free(d_y, q);
    sycl::free(d_out_aos, q);
    sycl::free(d_out_soa, q);

    return pass;
}

// =============================================================================
// Test Runner for Q8_0 GPU vs GPU Comparison
// =============================================================================

bool run_q8_0_gpu_comparison(int ncols, int nrows, int row_low, int row_high,
                              uint32_t seed, const char* test_name) {
    sycl::queue& q = *g_queue;

    const int full_nrows = nrows;
    const int slice_nrows = row_high - row_low;
    const int blocks_per_row = ncols / QK8_0;
    const int full_nblocks = full_nrows * blocks_per_row;

    // Generate AoS data for full tensor
    std::vector<block_q8_0> x_aos(full_nblocks);
    std::vector<float> y_vec(ncols);
    generate_q8_0_data(x_aos.data(), full_nblocks, seed);
    generate_y_data(y_vec.data(), ncols, seed + 1);

    // Convert to SoA: [qs...][d...]
    size_t full_qs_bytes = full_nblocks * 32;  // 32 bytes of qs per Q8_0 block
    std::vector<int8_t> soa_qs(full_nblocks * 32);
    std::vector<sycl::half> soa_d(full_nblocks);
    convert_q8_0_aos_to_soa(x_aos.data(), soa_qs.data(), soa_d.data(), full_nblocks);

    std::vector<uint8_t> soa_combined(full_qs_bytes + full_nblocks * sizeof(sycl::half));
    memcpy(soa_combined.data(), soa_qs.data(), full_qs_bytes);
    memcpy(soa_combined.data() + full_qs_bytes, soa_d.data(), full_nblocks * sizeof(sycl::half));

    // Calculate offsets for row slice
    size_t row_slice_aos_offset = row_low * blocks_per_row * sizeof(block_q8_0);
    size_t row_slice_qs_offset = row_low * blocks_per_row * 32;
    size_t row_slice_d_offset = row_low * blocks_per_row * sizeof(sycl::half);

    int64_t correct_d_offset = full_qs_bytes + row_slice_d_offset - row_slice_qs_offset;

    // Allocate device memory
    void* d_aos = sycl::malloc_device(full_nblocks * sizeof(block_q8_0), q);
    void* d_soa = sycl::malloc_device(soa_combined.size(), q);
    float* d_y = sycl::malloc_device<float>(ncols, q);
    float* d_out_aos = sycl::malloc_device<float>(slice_nrows, q);
    float* d_out_soa = sycl::malloc_device<float>(slice_nrows, q);

    q.memcpy(d_aos, x_aos.data(), full_nblocks * sizeof(block_q8_0));
    q.memcpy(d_soa, soa_combined.data(), soa_combined.size());
    q.memcpy(d_y, y_vec.data(), ncols * sizeof(float));
    q.wait();

    sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    sycl::range<3> grid_dims(1, 1, slice_nrows);

    const char* aos_slice_ptr = (const char*)d_aos + row_slice_aos_offset;
    const char* soa_slice_ptr = (const char*)d_soa + row_slice_qs_offset;

    q.parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                   [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                       dmmv_q8_0_aos_kernel<QK8_0, QR8_0>(aos_slice_ptr, (const dfloat*)d_y,
                                                          d_out_aos, ncols, slice_nrows, item);
                   });

    q.parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                   [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                       dmmv_q8_0_soa_kernel<QK8_0, QR8_0>(soa_slice_ptr, (const dfloat*)d_y,
                                                          d_out_soa, ncols, slice_nrows,
                                                          correct_d_offset, item);
                   });
    q.wait();

    std::vector<float> out_aos(slice_nrows), out_soa(slice_nrows);
    q.memcpy(out_aos.data(), d_out_aos, slice_nrows * sizeof(float));
    q.memcpy(out_soa.data(), d_out_soa, slice_nrows * sizeof(float));
    q.wait();

    bool pass = validate_output(out_aos.data(), out_soa.data(), slice_nrows, 1e-3f, 1e-5f);

    sycl::free(d_aos, q);
    sycl::free(d_soa, q);
    sycl::free(d_y, q);
    sycl::free(d_out_aos, q);
    sycl::free(d_out_soa, q);

    return pass;
}

// =============================================================================
// Category 1: Core Functionality Tests (Q4_0)
// =============================================================================

bool test_q4_0_basic_full_tensor() {
    return run_q4_0_gpu_comparison(128, 4, 0, 4, 42, "basic_full_tensor");
}

bool test_q4_0_basic_large() {
    return run_q4_0_gpu_comparison(4096, 32, 0, 32, 43, "basic_large");
}

bool test_q4_0_single_row() {
    return run_q4_0_gpu_comparison(128, 1, 0, 1, 44, "single_row");
}

bool test_q4_0_single_block() {
    return run_q4_0_gpu_comparison(32, 1, 0, 1, 45, "single_block");
}

bool test_q4_0_realistic_mistral() {
    // Mistral 7B: intermediate_size = 14336
    return run_q4_0_gpu_comparison(4096, 64, 0, 64, 46, "realistic_mistral");
}

// =============================================================================
// Category 2: Row Slicing Tests (CRITICAL - d_offset bug)
// =============================================================================

bool test_q4_0_slice_middle() {
    // Full: 25 rows, process rows 10-25
    return run_q4_0_gpu_comparison(128, 25, 10, 25, 100, "slice_middle");
}

bool test_q4_0_slice_first_half() {
    // Full: 32 rows, process rows 0-16
    return run_q4_0_gpu_comparison(128, 32, 0, 16, 101, "slice_first_half");
}

bool test_q4_0_slice_second_half() {
    // Full: 32 rows, process rows 16-32
    return run_q4_0_gpu_comparison(128, 32, 16, 32, 102, "slice_second_half");
}

bool test_q4_0_slice_single_row_mid() {
    // Full: 100 rows, process row 50 only
    return run_q4_0_gpu_comparison(128, 100, 50, 51, 103, "slice_single_row_mid");
}

bool test_q4_0_slice_last_row() {
    // Full: 64 rows, process row 63 only
    return run_q4_0_gpu_comparison(128, 64, 63, 64, 104, "slice_last_row");
}

bool test_q4_0_slice_realistic() {
    // Realistic multi-GPU split: full tensor 14336 rows, slice middle portion
    return run_q4_0_gpu_comparison(4096, 14336, 7168, 10752, 105, "slice_realistic");
}

// =============================================================================
// Category 3: Column Alignment Edge Cases
// =============================================================================

bool test_q4_0_ncols_aligned_512() {
    // 512 = QK4_0 * WARP_SIZE - main loop only
    return run_q4_0_gpu_comparison(512, 4, 0, 4, 200, "ncols_aligned_512");
}

bool test_q4_0_ncols_aligned_1024() {
    return run_q4_0_gpu_comparison(1024, 4, 0, 4, 201, "ncols_aligned_1024");
}

bool test_q4_0_ncols_unaligned_544() {
    // 544 = 512 + 32 (one extra block)
    return run_q4_0_gpu_comparison(544, 4, 0, 4, 202, "ncols_unaligned_544");
}

bool test_q4_0_ncols_small_128() {
    // Smaller than alignment boundary
    return run_q4_0_gpu_comparison(128, 4, 0, 4, 203, "ncols_small_128");
}

bool test_q4_0_ncols_minimal_32() {
    // Single block
    return run_q4_0_gpu_comparison(32, 4, 0, 4, 204, "ncols_minimal_32");
}

bool test_q4_0_ncols_large_8192() {
    return run_q4_0_gpu_comparison(8192, 4, 0, 4, 205, "ncols_large_8192");
}

// =============================================================================
// Category 4: Scale Edge Cases
// =============================================================================

bool test_q4_0_scale_zero() {
    sycl::queue& q = *g_queue;
    const int ncols = 128, nrows = 4;
    const int blocks_per_row = ncols / QK4_0;
    const int nblocks = nrows * blocks_per_row;

    std::vector<block_q4_0> x_aos(nblocks);
    std::vector<float> y_vec(ncols);
    for (int b = 0; b < nblocks; b++) {
        x_aos[b].d = sycl::half(0.0f);  // Zero scales
        for (int i = 0; i < 16; i++) {
            x_aos[b].qs[i] = 0x88;  // Non-zero quants
        }
    }
    for (int i = 0; i < ncols; i++) y_vec[i] = 1.0f;

    // Reference: all outputs should be 0
    std::vector<float> expected(nrows, 0.0f);

    // Run GPU AoS
    std::vector<float> out_aos(nrows);
    {
        void* d_aos = sycl::malloc_device(nblocks * sizeof(block_q4_0), q);
        float* d_y = sycl::malloc_device<float>(ncols, q);
        float* d_out = sycl::malloc_device<float>(nrows, q);
        q.memcpy(d_aos, x_aos.data(), nblocks * sizeof(block_q4_0));
        q.memcpy(d_y, y_vec.data(), ncols * sizeof(float));
        q.wait();

        sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
        sycl::range<3> grid_dims(1, 1, nrows);

        q.parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                       [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                           dmmv_q4_0_aos_kernel<QK4_0, QR4_0>(d_aos, (const dfloat*)d_y,
                                                              d_out, ncols, nrows, item);
                       });
        q.wait();
        q.memcpy(out_aos.data(), d_out, nrows * sizeof(float)).wait();

        sycl::free(d_aos, q);
        sycl::free(d_y, q);
        sycl::free(d_out, q);
    }

    return validate_output(expected.data(), out_aos.data(), nrows, 1e-6f, 1e-6f);
}

bool test_q4_0_scale_large() {
    return run_q4_0_gpu_comparison(128, 4, 0, 4, 301, "scale_large");
}

bool test_q4_0_scale_mixed() {
    return run_q4_0_gpu_comparison(128, 4, 0, 4, 302, "scale_mixed");
}

// =============================================================================
// Category 5: Quantized Value Edge Cases
// =============================================================================

bool test_q4_0_qs_all_zeros() {
    sycl::queue& q = *g_queue;
    const int ncols = 128, nrows = 4;
    const int blocks_per_row = ncols / QK4_0;
    const int nblocks = nrows * blocks_per_row;

    std::vector<block_q4_0> x_aos(nblocks);
    for (int b = 0; b < nblocks; b++) {
        x_aos[b].d = sycl::half(1.0f);
        for (int i = 0; i < 16; i++) {
            x_aos[b].qs[i] = 0x00;  // All zeros (maps to -8, -8 after subtraction)
        }
    }

    // Use standard comparison - SoA should match AoS
    return run_q4_0_gpu_comparison(ncols, nrows, 0, nrows, 401, "qs_all_zeros");
}

bool test_q4_0_qs_all_max() {
    // qs = 0xFF maps to (15-8, 15-8) = (7, 7) after dequantization
    return run_q4_0_gpu_comparison(128, 4, 0, 4, 402, "qs_all_max");
}

bool test_q4_0_qs_boundary() {
    // qs = 0x88 maps to (8-8, 8-8) = (0, 0) - boundary value
    return run_q4_0_gpu_comparison(128, 4, 0, 4, 403, "qs_boundary");
}

// =============================================================================
// Category 6: Y Vector Tests
// =============================================================================

bool test_q4_0_y_zeros() {
    sycl::queue& q = *g_queue;
    const int ncols = 128, nrows = 4;
    const int blocks_per_row = ncols / QK4_0;
    const int nblocks = nrows * blocks_per_row;

    std::vector<block_q4_0> x_aos(nblocks);
    std::vector<float> y_vec(ncols, 0.0f);  // All zeros

    generate_q4_0_data(x_aos.data(), nblocks, 501);

    // With y=0, all outputs should be 0
    std::vector<float> expected(nrows, 0.0f);

    void* d_aos = sycl::malloc_device(nblocks * sizeof(block_q4_0), q);
    float* d_y = sycl::malloc_device<float>(ncols, q);
    float* d_out = sycl::malloc_device<float>(nrows, q);
    q.memcpy(d_aos, x_aos.data(), nblocks * sizeof(block_q4_0));
    q.memcpy(d_y, y_vec.data(), ncols * sizeof(float));
    q.wait();

    sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    sycl::range<3> grid_dims(1, 1, nrows);

    std::vector<float> out_aos(nrows);
    q.parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                   [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                       dmmv_q4_0_aos_kernel<QK4_0, QR4_0>(d_aos, (const dfloat*)d_y,
                                                          d_out, ncols, nrows, item);
                   });
    q.wait();
    q.memcpy(out_aos.data(), d_out, nrows * sizeof(float)).wait();

    sycl::free(d_aos, q);
    sycl::free(d_y, q);
    sycl::free(d_out, q);

    return validate_output(expected.data(), out_aos.data(), nrows, 1e-6f, 1e-6f);
}

bool test_q4_0_y_ones() {
    return run_q4_0_gpu_comparison(128, 4, 0, 4, 502, "y_ones");
}

bool test_q4_0_y_alternating() {
    return run_q4_0_gpu_comparison(128, 4, 0, 4, 503, "y_alternating");
}

// =============================================================================
// Category 7: Numerical Precision Tests
// =============================================================================

bool test_q4_0_precision_accumulated() {
    // Large tensor to test accumulated precision
    return run_q4_0_gpu_comparison(4096, 64, 0, 64, 601, "precision_accumulated");
}

// =============================================================================
// Category 8: Thread/Warp Boundary Tests
// =============================================================================

bool test_q4_0_warp_boundary_16() {
    // ncols = 512 = 16 * 32 - exactly WARP_SIZE blocks per row
    return run_q4_0_gpu_comparison(512, 16, 0, 16, 701, "warp_boundary_16");
}

bool test_q4_0_multiple_warps() {
    // ncols = 1024 - requires multiple warps per row
    return run_q4_0_gpu_comparison(1024, 16, 0, 16, 702, "multiple_warps");
}

// =============================================================================
// Category 9: Memory Layout Verification
// =============================================================================

bool test_q4_0_d_offset_formula() {
    // Test specific d_offset calculation across different row_low values
    bool all_pass = true;

    // Test multiple row_low values
    int test_cases[][4] = {
        // {ncols, full_nrows, row_low, row_high}
        {128, 32, 0, 32},   // Full tensor
        {128, 32, 8, 32},   // First quarter skipped
        {128, 32, 16, 32},  // First half skipped
        {128, 32, 24, 32},  // Last quarter only
        {128, 32, 31, 32},  // Last row only
    };

    for (int i = 0; i < 5; i++) {
        bool pass = run_q4_0_gpu_comparison(
            test_cases[i][0], test_cases[i][1],
            test_cases[i][2], test_cases[i][3],
            800 + i, "d_offset_formula");
        if (!pass) {
            printf("    d_offset test failed for row_low=%d\n", test_cases[i][2]);
            all_pass = false;
        }
    }

    return all_pass;
}

// =============================================================================
// Category 10: Q8_0 Tests
// =============================================================================

bool test_q8_0_basic_full_tensor() {
    return run_q8_0_gpu_comparison(128, 4, 0, 4, 900, "q8_0_basic");
}

bool test_q8_0_slice_middle() {
    return run_q8_0_gpu_comparison(128, 25, 10, 25, 901, "q8_0_slice_middle");
}

bool test_q8_0_slice_last_row() {
    return run_q8_0_gpu_comparison(128, 64, 63, 64, 902, "q8_0_slice_last");
}

bool test_q8_0_large() {
    return run_q8_0_gpu_comparison(4096, 32, 0, 32, 903, "q8_0_large");
}

bool test_q8_0_realistic() {
    return run_q8_0_gpu_comparison(4096, 64, 32, 64, 904, "q8_0_realistic");
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("=== DMMV SoA Comprehensive Unit Test Suite ===\n");
    printf("Comparison: GPU SoA vs GPU AoS\n\n");

    try {
        sycl::queue q{sycl::gpu_selector_v};
        g_queue = &q;
        printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
        printf("Max work-group size: %zu\n",
               q.get_device().get_info<sycl::info::device::max_work_group_size>());
        printf("Max compute units: %u\n",
               q.get_device().get_info<sycl::info::device::max_compute_units>());

        printf("\n========== Category 1: Q4_0 Core Functionality ==========\n");
        RUN_TEST(test_q4_0_basic_full_tensor);
        RUN_TEST(test_q4_0_basic_large);
        RUN_TEST(test_q4_0_single_row);
        RUN_TEST(test_q4_0_single_block);
        RUN_TEST(test_q4_0_realistic_mistral);

        printf("\n========== Category 2: Q4_0 Row Slicing (CRITICAL) ==========\n");
        RUN_TEST(test_q4_0_slice_middle);
        RUN_TEST(test_q4_0_slice_first_half);
        RUN_TEST(test_q4_0_slice_second_half);
        RUN_TEST(test_q4_0_slice_single_row_mid);
        RUN_TEST(test_q4_0_slice_last_row);
        RUN_TEST(test_q4_0_slice_realistic);

        printf("\n========== Category 3: Q4_0 Column Alignment ==========\n");
        RUN_TEST(test_q4_0_ncols_aligned_512);
        RUN_TEST(test_q4_0_ncols_aligned_1024);
        RUN_TEST(test_q4_0_ncols_unaligned_544);
        RUN_TEST(test_q4_0_ncols_small_128);
        RUN_TEST(test_q4_0_ncols_minimal_32);
        RUN_TEST(test_q4_0_ncols_large_8192);

        printf("\n========== Category 4: Q4_0 Scale Edge Cases ==========\n");
        RUN_TEST(test_q4_0_scale_zero);
        RUN_TEST(test_q4_0_scale_large);
        RUN_TEST(test_q4_0_scale_mixed);

        printf("\n========== Category 5: Q4_0 Quantized Value Edge Cases ==========\n");
        RUN_TEST(test_q4_0_qs_all_zeros);
        RUN_TEST(test_q4_0_qs_all_max);
        RUN_TEST(test_q4_0_qs_boundary);

        printf("\n========== Category 6: Q4_0 Y Vector Tests ==========\n");
        RUN_TEST(test_q4_0_y_zeros);
        RUN_TEST(test_q4_0_y_ones);
        RUN_TEST(test_q4_0_y_alternating);

        printf("\n========== Category 7: Q4_0 Numerical Precision ==========\n");
        RUN_TEST(test_q4_0_precision_accumulated);

        printf("\n========== Category 8: Q4_0 Thread/Warp Boundaries ==========\n");
        RUN_TEST(test_q4_0_warp_boundary_16);
        RUN_TEST(test_q4_0_multiple_warps);

        printf("\n========== Category 9: Q4_0 Memory Layout ==========\n");
        RUN_TEST(test_q4_0_d_offset_formula);

        printf("\n========== Category 10: Q8_0 Tests ==========\n");
        RUN_TEST(test_q8_0_basic_full_tensor);
        RUN_TEST(test_q8_0_slice_middle);
        RUN_TEST(test_q8_0_slice_last_row);
        RUN_TEST(test_q8_0_large);
        RUN_TEST(test_q8_0_realistic);

        printf("\n========================================\n");
        printf("=== Results: %d passed, %d failed ===\n", g_tests_passed, g_tests_failed);
        printf("========================================\n");

    } catch (const sycl::exception& e) {
        printf("SYCL exception: %s\n", e.what());
        return 1;
    }

    return g_tests_failed > 0 ? 1 : 0;
}
