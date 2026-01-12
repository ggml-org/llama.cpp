// GPU Unit tests for Q8_0 DMMV SoA support
// Compares SoA vs AoS outputs using production-like DMMV kernels

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include <sycl/sycl.hpp>

#ifndef GGML_SYCL_WARP_SIZE
#define GGML_SYCL_WARP_SIZE 32
#endif

#define WARP_SIZE GGML_SYCL_WARP_SIZE
#define GGML_SYCL_DMMV_X 32
#define GGML_SYCL_MMV_Y 1

#define QK8_0 32
#define QR8_0 1

typedef struct {
    sycl::half d;
    int8_t qs[QK8_0];
} block_q8_0_test;

static_assert(sizeof(block_q8_0_test) == 34, "block_q8_0 size mismatch");

using dfloat = float;
using dfloat2 = sycl::float2;

static inline void dequantize_q8_0_aos(const void *vx, const int64_t ib,
                                       const int iqs, dfloat2 &v) {
    const block_q8_0_test *x = (const block_q8_0_test *)vx;
    const dfloat d = dfloat(x[ib].d);
    v.x() = x[ib].qs[iqs + 0];
    v.y() = x[ib].qs[iqs + 1];
    v.x() *= d;
    v.y() *= d;
}

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

static void dmmv_q8_0_cpu_reference(const block_q8_0_test *aos, const float *y,
                                   float *dst, int ncols, int nrows) {
    const int blocks_per_row = ncols / QK8_0;
    for (int row = 0; row < nrows; row++) {
        float sum = 0.0f;
        for (int b = 0; b < blocks_per_row; b++) {
            const block_q8_0_test *block = aos + row * blocks_per_row + b;
            const float d = float(block->d);
            const float *y_block = y + b * QK8_0;
            for (int j = 0; j < QK8_0; j++) {
                sum += d * (float)block->qs[j] * y_block[j];
            }
        }
        dst[row] = sum;
    }
}

static void dmmv_q8_0_aos_kernel(const void * __restrict__ vx, const dfloat * __restrict__ y,
                                float * __restrict__ dst, const int ncols, const int nrows,
                                const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int tid = item_ct1.get_local_id(2);
    const int iter_stride = 2 * GGML_SYCL_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE;
    const int y_offset = 1;

    float tmp = 0.0f;

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter * tid;
        if (col >= ncols) {
            continue;
        }

        const int ib = (row * ncols + col) / QK8_0;
        const int iqs = (col % QK8_0) / QR8_0;
        const int iybs = col - col % QK8_0;

#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            dfloat2 v;
            dequantize_q8_0_aos(vx, ib, iqs + j / QR8_0, v);
            tmp += v.x() * y[iybs + iqs + j / QR8_0 + 0];
            tmp += v.y() * y[iybs + iqs + j / QR8_0 + y_offset];
        }
    }

    const int mask_start = ncols > GGML_SYCL_DMMV_X ? WARP_SIZE >> 1 : WARP_SIZE >> 2;
    for (int mask = mask_start; mask > 0; mask >>= 1) {
        tmp += sycl::permute_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

static void dmmv_q8_0_soa_kernel(const void * __restrict__ vx, const dfloat * __restrict__ y,
                                float * __restrict__ dst, const int ncols, const int nrows,
                                const int64_t d_offset, const int row_low,
                                const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int tid = item_ct1.get_local_id(2);
    const int iter_stride = 2 * GGML_SYCL_DMMV_X;
    const int vals_per_iter = iter_stride / WARP_SIZE;
    const int y_offset = 1;

    float tmp = 0.0f;
    const char *d_ptr = (const char *)vx + d_offset;
    const int global_row = row_low + row;

    for (int i = 0; i < ncols; i += iter_stride) {
        const int col = i + vals_per_iter * tid;
        if (col >= ncols) {
            continue;
        }

        const int ib = (global_row * ncols + col) / QK8_0;
        const int iqs = (col % QK8_0) / QR8_0;
        const int iybs = col - col % QK8_0;

#pragma unroll
        for (int j = 0; j < vals_per_iter; j += 2) {
            dfloat2 v;
            dequantize_q8_0_reorder(d_ptr, ib, vx, ib * QK8_0 + iqs + j / QR8_0, v);
            tmp += v.x() * y[iybs + iqs + j / QR8_0 + 0];
            tmp += v.y() * y[iybs + iqs + j / QR8_0 + y_offset];
        }
    }

    const int mask_start = ncols > GGML_SYCL_DMMV_X ? WARP_SIZE >> 1 : WARP_SIZE >> 2;
    for (int mask = mask_start; mask > 0; mask >>= 1) {
        tmp += sycl::permute_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (tid == 0) {
        dst[row] = tmp;
    }
}

struct TestCase {
    int nrows_full;
    int nrows_slice;
    int row_low;
    int ncols;
    const char *name;
};

static bool run_case(sycl::queue &q, const TestCase &tc) {
    const int blocks_per_row = tc.ncols / QK8_0;
    const int nblocks_full = tc.nrows_full * blocks_per_row;
    const size_t d_offset = (size_t)tc.nrows_full * tc.ncols;
    const size_t soa_bytes = d_offset + (size_t)nblocks_full * sizeof(sycl::half);

    std::vector<block_q8_0_test> h_aos(nblocks_full);
    std::vector<uint8_t> h_soa(soa_bytes);
    std::vector<float> h_y(tc.ncols);

    create_aos_data(h_aos.data(), tc.nrows_full, tc.ncols, 42 + tc.row_low);
    convert_aos_to_soa(h_aos.data(), h_soa.data(), tc.nrows_full, tc.ncols);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> y_dist(-1.0f, 1.0f);
    for (int i = 0; i < tc.ncols; i++) {
        h_y[i] = y_dist(rng);
    }

    std::vector<float> h_ref(tc.nrows_slice);
    dmmv_q8_0_cpu_reference(h_aos.data() + tc.row_low * blocks_per_row,
                           h_y.data(), h_ref.data(), tc.ncols, tc.nrows_slice);

    block_q8_0_test *d_aos = sycl::malloc_device<block_q8_0_test>(nblocks_full, q);
    uint8_t *d_soa = sycl::malloc_device<uint8_t>(soa_bytes, q);
    float *d_y = sycl::malloc_device<float>(tc.ncols, q);
    float *d_out_aos = sycl::malloc_device<float>(tc.nrows_slice, q);
    float *d_out_soa = sycl::malloc_device<float>(tc.nrows_slice, q);

    q.memcpy(d_aos, h_aos.data(), nblocks_full * sizeof(block_q8_0_test));
    q.memcpy(d_soa, h_soa.data(), soa_bytes);
    q.memcpy(d_y, h_y.data(), tc.ncols * sizeof(float));
    q.wait();

    const int block_num_y = (tc.nrows_slice + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    sycl::range<3> grid(1, 1, block_num_y);

    const block_q8_0_test *d_aos_slice = d_aos + tc.row_low * blocks_per_row;

    q.parallel_for(
        sycl::nd_range<3>(grid * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dmmv_q8_0_aos_kernel(d_aos_slice, d_y, d_out_aos, tc.ncols, tc.nrows_slice, item_ct1);
        });

    q.parallel_for(
        sycl::nd_range<3>(grid * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            dmmv_q8_0_soa_kernel(d_soa, d_y, d_out_soa, tc.ncols, tc.nrows_slice,
                                (int64_t)d_offset, tc.row_low, item_ct1);
        });

    q.wait();

    std::vector<float> h_out_aos(tc.nrows_slice);
    std::vector<float> h_out_soa(tc.nrows_slice);
    q.memcpy(h_out_aos.data(), d_out_aos, tc.nrows_slice * sizeof(float)).wait();
    q.memcpy(h_out_soa.data(), d_out_soa, tc.nrows_slice * sizeof(float)).wait();

    float max_diff_aos = 0.0f;
    float max_diff_soa = 0.0f;
    float max_rel_soa = 0.0f;
    int errors = 0;

    for (int i = 0; i < tc.nrows_slice; i++) {
        float ref = h_ref[i];
        float diff_aos = fabsf(h_out_aos[i] - ref);
        float diff_soa = fabsf(h_out_soa[i] - ref);
        float rel_soa = fabsf(ref) > 1e-6f ? diff_soa / fabsf(ref) : diff_soa;

        max_diff_aos = std::max(max_diff_aos, diff_aos);
        max_diff_soa = std::max(max_diff_soa, diff_soa);
        max_rel_soa = std::max(max_rel_soa, rel_soa);

        if (rel_soa > 1e-3f) {
            if (errors < 5) {
                printf("  FAIL row %d: ref=%.6f aos=%.6f soa=%.6f rel=%.6e\n",
                       i, ref, h_out_aos[i], h_out_soa[i], rel_soa);
            }
            errors++;
        }
    }

    printf("  Case %s: max diff AoS %.6e, max diff SoA %.6e, max rel SoA %.6e\n",
           tc.name, max_diff_aos, max_diff_soa, max_rel_soa);

    sycl::free(d_aos, q);
    sycl::free(d_soa, q);
    sycl::free(d_y, q);
    sycl::free(d_out_aos, q);
    sycl::free(d_out_soa, q);

    return errors == 0;
}

int main() {
    printf("Q8_0 DMMV SoA GPU Unit Tests (Production Logic)\n");
    printf("================================================\n");

    sycl::queue q{sycl::property::queue::in_order()};

    printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    const TestCase cases[] = {
        {128, 128, 0, 4096, "full-128x4096"},
        {256, 128, 64, 4096, "slice-128x4096@64"},
    };

    int passed = 0;
    int failed = 0;

    for (const auto &tc : cases) {
        printf("\n=== Case: %s ===\n", tc.name);
        if (run_case(q, tc)) {
            passed++;
        } else {
            failed++;
        }
    }

    printf("\n================================================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
