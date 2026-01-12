// DMMV Q8_0 SoA Unit Test
// Tests dequantize_q8_0 vs dequantize_q8_0_reorder and DMMV iteration parity

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include <sycl/sycl.hpp>

#define GGML_SYCL_DMMV_X 32
#define WARP_SIZE 32
#define QK8_0 32
#define QR8_0 1

typedef struct {
    sycl::half d;
    int8_t qs[QK8_0];
} block_q8_0_test;

static_assert(sizeof(block_q8_0_test) == 34, "block_q8_0 size mismatch");

using dfloat = float;
using dfloat2 = sycl::float2;

// Copy of production dequantize_q8_0
static inline void dequantize_q8_0_aos(const void *vx, const int64_t ib,
                                       const int iqs, dfloat2 &v) {
    const block_q8_0_test * x = (const block_q8_0_test *)vx;
    const dfloat d = dfloat(x[ib].d);
    v.x() = x[ib].qs[iqs + 0];
    v.y() = x[ib].qs[iqs + 1];
    v.x() *= d;
    v.y() *= d;
}

// Copy of production dequantize_q8_0_reorder
static inline void dequantize_q8_0_reorder(const void *d_ptr, const int64_t ib, const void *qs,
                                           const int iqs, dfloat2 &v) {
    const dfloat d = (const dfloat)*((const sycl::half*)d_ptr + ib);
    const int8_t *qs_ptr = (const int8_t *)qs;
    v.x() = qs_ptr[iqs + 0];
    v.y() = qs_ptr[iqs + 1];
    v.x() *= d;
    v.y() *= d;
}

static void create_aos_data(block_q8_0_test *data, int nrows, int ncols, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> scale_dist(0.001f, 0.1f);
    std::uniform_int_distribution<int> quant_dist(-127, 127);

    const int blocks_per_row = ncols / QK8_0;
    const int nblocks = nrows * blocks_per_row;

    for (int i = 0; i < nblocks; i++) {
        data[i].d = sycl::half(scale_dist(rng));
        for (int j = 0; j < QK8_0; j++) {
            data[i].qs[j] = (int8_t)quant_dist(rng);
        }
    }
}

static void convert_aos_to_soa(const block_q8_0_test *aos, uint8_t *soa, int nrows, int ncols) {
    const int blocks_per_row = ncols / QK8_0;
    const int nblocks = nrows * blocks_per_row;
    const size_t d_offset = (size_t)nrows * ncols;
    int8_t *qs_ptr = (int8_t *)soa;
    sycl::half *d_ptr = (sycl::half *)(soa + d_offset);

    for (int i = 0; i < nblocks; i++) {
        memcpy(qs_ptr + i * QK8_0, aos[i].qs, QK8_0);
        d_ptr[i] = aos[i].d;
    }
}

static bool test_dequantize_functions() {
    printf("Test 1: dequantize_q8_0 SoA vs AoS\n");

    const int nrows = 128;
    const int ncols = 4096;
    const int blocks_per_row = ncols / QK8_0;
    const int nblocks = nrows * blocks_per_row;
    const size_t d_offset = (size_t)nrows * ncols;
    const size_t soa_size = d_offset + (size_t)nblocks * sizeof(sycl::half);

    std::vector<block_q8_0_test> aos_data(nblocks);
    std::vector<uint8_t> soa_data(soa_size);

    create_aos_data(aos_data.data(), nrows, ncols, 42);
    convert_aos_to_soa(aos_data.data(), soa_data.data(), nrows, ncols);

    const void *d_ptr = soa_data.data() + d_offset;
    const void *qs_base = soa_data.data();

    int errors = 0;
    int tested = 0;

    const int max_blocks = std::min(nblocks, 1000);
    for (int ib = 0; ib < max_blocks; ib++) {
        for (int iqs = 0; iqs < QK8_0 - 1; iqs++) {
            dfloat2 v_aos;
            dfloat2 v_soa;

            dequantize_q8_0_aos(aos_data.data(), ib, iqs, v_aos);
            dequantize_q8_0_reorder(d_ptr, ib, qs_base, ib * QK8_0 + iqs, v_soa);

            float diff_x = fabsf(v_aos.x() - v_soa.x());
            float diff_y = fabsf(v_aos.y() - v_soa.y());

            if (diff_x > 1e-6f || diff_y > 1e-6f) {
                if (errors < 5) {
                    printf("  ERROR at ib=%d iqs=%d: AoS=(%.6f,%.6f) SoA=(%.6f,%.6f)\n",
                           ib, iqs, v_aos.x(), v_aos.y(), v_soa.x(), v_soa.y());
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

static float dmmv_reference_row(const block_q8_0_test *aos, const float *y,
                                int ncols, int row) {
    const int blocks_per_row = ncols / QK8_0;
    float sum = 0.0f;

    for (int b = 0; b < blocks_per_row; b++) {
        const block_q8_0_test *block = aos + row * blocks_per_row + b;
        const float d = float(block->d);
        const float *y_block = y + b * QK8_0;
        for (int j = 0; j < QK8_0; j++) {
            sum += d * (float)block->qs[j] * y_block[j];
        }
    }

    return sum;
}

static float dmmv_iter_row_aos(const block_q8_0_test *aos, const float *y,
                               int ncols, int row) {
    const int iter_stride = 2 * GGML_SYCL_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE;
    const int y_offset = 1;
    float warp_sum = 0.0f;

    for (int tid = 0; tid < WARP_SIZE; tid++) {
        float tmp = 0.0f;
        for (int i = 0; i < ncols; i += iter_stride) {
            const int col = i + vals_per_iter * tid;
            if (col >= ncols) {
                continue;
            }

            const int ib = (row * ncols + col) / QK8_0;
            const int iqs = (col % QK8_0) / QR8_0;
            const int iybs = col - col % QK8_0;

            for (int j = 0; j < vals_per_iter; j += 2) {
                dfloat2 v;
                dequantize_q8_0_aos(aos, ib, iqs + j / QR8_0, v);
                tmp += v.x() * y[iybs + iqs + j / QR8_0 + 0];
                tmp += v.y() * y[iybs + iqs + j / QR8_0 + y_offset];
            }
        }
        warp_sum += tmp;
    }

    return warp_sum;
}

static float dmmv_iter_row_soa(const uint8_t *soa, const float *y,
                               int ncols, int row, int nrows) {
    const int iter_stride = 2 * GGML_SYCL_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE;
    const int y_offset = 1;
    const size_t d_offset = (size_t)nrows * ncols;
    const void *d_ptr = soa + d_offset;
    const void *qs_base = soa;
    float warp_sum = 0.0f;

    for (int tid = 0; tid < WARP_SIZE; tid++) {
        float tmp = 0.0f;
        for (int i = 0; i < ncols; i += iter_stride) {
            const int col = i + vals_per_iter * tid;
            if (col >= ncols) {
                continue;
            }

            const int ib = (row * ncols + col) / QK8_0;
            const int iqs = (col % QK8_0) / QR8_0;
            const int iybs = col - col % QK8_0;

            for (int j = 0; j < vals_per_iter; j += 2) {
                dfloat2 v;
                dequantize_q8_0_reorder(d_ptr, ib, qs_base,
                                        ib * QK8_0 + iqs + j / QR8_0, v);
                tmp += v.x() * y[iybs + iqs + j / QR8_0 + 0];
                tmp += v.y() * y[iybs + iqs + j / QR8_0 + y_offset];
            }
        }
        warp_sum += tmp;
    }

    return warp_sum;
}

static bool test_dmmv_iteration_pattern() {
    printf("Test 2: DMMV iteration pattern (AoS vs SoA)\n");

    const int nrows = 64;
    const int ncols = 4096;
    const int blocks_per_row = ncols / QK8_0;
    const int nblocks = nrows * blocks_per_row;
    const size_t d_offset = (size_t)nrows * ncols;
    const size_t soa_size = d_offset + (size_t)nblocks * sizeof(sycl::half);

    std::vector<block_q8_0_test> aos_data(nblocks);
    std::vector<uint8_t> soa_data(soa_size);
    std::vector<float> y_data(ncols);

    create_aos_data(aos_data.data(), nrows, ncols, 123);
    convert_aos_to_soa(aos_data.data(), soa_data.data(), nrows, ncols);

    std::mt19937 rng(456);
    std::uniform_real_distribution<float> y_dist(-1.0f, 1.0f);
    for (int i = 0; i < ncols; i++) {
        y_data[i] = y_dist(rng);
    }

    int errors = 0;
    float max_diff = 0.0f;
    float max_rel = 0.0f;

    for (int row = 0; row < nrows; row++) {
        float ref = dmmv_reference_row(aos_data.data(), y_data.data(), ncols, row);
        float aos_iter = dmmv_iter_row_aos(aos_data.data(), y_data.data(), ncols, row);
        float soa_iter = dmmv_iter_row_soa(soa_data.data(), y_data.data(), ncols, row, nrows);

        float diff = fabsf(ref - soa_iter);
        float rel = fabsf(ref) > 1e-6f ? diff / fabsf(ref) : diff;
        max_diff = std::max(max_diff, diff);
        max_rel = std::max(max_rel, rel);

        if (fabsf(ref - aos_iter) > 1e-4f) {
            printf("  ERROR row %d: AoS iter mismatch ref=%.6f aos=%.6f\n", row, ref, aos_iter);
            errors++;
        }
        if (rel > 1e-4f) {
            if (errors < 5) {
                printf("  ERROR row %d: SoA iter mismatch ref=%.6f soa=%.6f rel=%.6e\n",
                       row, ref, soa_iter, rel);
            }
            errors++;
        }
    }

    printf("  Max diff: %.6e, max rel: %.6e\n", max_diff, max_rel);
    if (errors > 0) {
        printf("  FAIL: %d mismatches\n", errors);
        return false;
    }

    printf("  PASS: DMMV iteration matches reference\n");
    return true;
}

int main() {
    printf("=== DMMV SoA Q8_0 Unit Test ===\n\n");

    bool ok = true;
    ok &= test_dequantize_functions();
    ok &= test_dmmv_iteration_pattern();

    printf("\nOverall: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
