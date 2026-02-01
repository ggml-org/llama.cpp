#pragma once

#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "ggml.h"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/ggml-sycl-bench.hpp"
#include "ggml-sycl/vecdotq.hpp"

namespace sycl_bench {
namespace mmvq_tier1 {

enum class AosVariant {
    Baseline = 0,
    Prefetch = 1,
    WideLoad = 2,
};

constexpr int kVariantBaseline = 0;
constexpr int kVariantPrefetch = 1;
constexpr int kVariantWideLoad = 2;
constexpr int kVariantSlm      = 3;

#if defined(__clang__) || defined(__GNUC__)
#define MMVQ_TIER1_PREFETCH(ptr) __builtin_prefetch((ptr), 0, 3)
#else
#define MMVQ_TIER1_PREFETCH(ptr) ((void) 0)
#endif

template <int qtype, int variant>
class mmvq_tier1_kernel_name;

template <int qtype, int variant>
class mmvq_tier1_multirow_kernel_name;

static inline block_q8_1 load_block_q8_1_wide(const block_q8_1 * src) {
    block_q8_1 out;
    out.ds = src->ds;

    const int * qs_src = reinterpret_cast<const int *>(src->qs);
    int * qs_dst = reinterpret_cast<int *>(out.qs);

    sycl::vec<int, 4> v0;
    sycl::vec<int, 4> v1;
    v0.load(0, qs_src);
    v1.load(0, qs_src + 4);
    v0.store(0, qs_dst);
    v1.store(0, qs_dst + 4);

    return out;
}

template <vec_dot_q_sycl_t vec_dot_q_sycl>
static __dpct_inline__ float vec_dot_q_slm_generic(const void * __restrict__ vbq,
                                                   const int * __restrict__ slm_y_qs,
                                                   const sycl::half2 * __restrict__ slm_y_ds,
                                                   int iby,
                                                   int stride,
                                                   const int & iqs) {
    block_q8_1 local{};
    local.ds = slm_y_ds[iby];

    const int * src_qs = slm_y_qs + iby * stride;
    int * dst_qs = reinterpret_cast<int *>(local.qs);

#pragma unroll
    for (int j = 0; j < QI8_1; ++j) {
        dst_qs[j] = src_qs[j];
    }

    return vec_dot_q_sycl(vbq, &local, iqs);
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void mul_mat_vec_q_baseline(const void * __restrict__ vx,
                                   const void * __restrict__ vy,
                                   float * __restrict__ dst,
                                   const int                ncols,
                                   const int                nrows,
                                   const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row  = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    constexpr int qi_div_vdr = qi / vdr;
    const int     lane_id    = item_ct1.get_local_id(2);
    const int     base_iqs   = vdr * (lane_id % qi_div_vdr);
    const int     row_offset = row * blocks_per_row;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    int       i       = lane_id / qi_div_vdr;
    const int stride  = blocks_per_warp;
    const int stride4 = 4 * stride;

    for (; i + 3 * stride < blocks_per_row; i += stride4) {
        const int ibx0 = row_offset + i;
        const int ibx1 = row_offset + i + stride;
        const int ibx2 = row_offset + i + 2 * stride;
        const int ibx3 = row_offset + i + 3 * stride;

        const int iby0 = i * (qk / QK8_1);
        const int iby1 = (i + stride) * (qk / QK8_1);
        const int iby2 = (i + 2 * stride) * (qk / QK8_1);
        const int iby3 = (i + 3 * stride) * (qk / QK8_1);

#pragma unroll
        for (size_t elem = 0; elem < qi_div_vdr; elem += WARP_SIZE) {
            const int iqs = elem + base_iqs;
            acc0 += vec_dot_q_sycl(&x[ibx0], &y[iby0], iqs);
            acc1 += vec_dot_q_sycl(&x[ibx1], &y[iby1], iqs);
            acc2 += vec_dot_q_sycl(&x[ibx2], &y[iby2], iqs);
            acc3 += vec_dot_q_sycl(&x[ibx3], &y[iby3], iqs);
        }
    }

    for (; i < blocks_per_row; i += stride) {
        const int ibx = row_offset + i;
        const int iby = i * (qk / QK8_1);

#pragma unroll
        for (size_t elem = 0; elem < qi_div_vdr; elem += WARP_SIZE) {
            const int iqs = elem + base_iqs;
            acc0 += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

    float tmp = (acc0 + acc1) + (acc2 + acc3);
    tmp = sycl::reduce_over_group(item_ct1.get_sub_group(), tmp, sycl::plus<float>());

    if (lane_id == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void mul_mat_vec_q_prefetch(const void * __restrict__ vx,
                                   const void * __restrict__ vy,
                                   float * __restrict__ dst,
                                   const int                ncols,
                                   const int                nrows,
                                   const sycl::nd_item<3> & item_ct1) {
    if constexpr (qk > QK8_1) {
        mul_mat_vec_q_baseline<qk, qi, block_q_t, vdr, vec_dot_q_sycl>(vx, vy, dst, ncols, nrows, item_ct1);
        return;
    }

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row  = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    constexpr int qi_div_vdr = qi / vdr;
    const int     lane_id    = item_ct1.get_local_id(2);
    const int     base_iqs   = vdr * (lane_id % qi_div_vdr);
    const int     row_offset = row * blocks_per_row;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    int       i       = lane_id / qi_div_vdr;
    const int stride  = blocks_per_warp;
    const int stride4 = 4 * stride;

    block_q8_1 y0{};
    block_q8_1 y1{};
    block_q8_1 y2{};
    block_q8_1 y3{};

    if (i + 3 * stride < blocks_per_row) {
        y0 = load_block_q8_1_wide(&y[i * (qk / QK8_1)]);
        y1 = load_block_q8_1_wide(&y[(i + stride) * (qk / QK8_1)]);
        y2 = load_block_q8_1_wide(&y[(i + 2 * stride) * (qk / QK8_1)]);
        y3 = load_block_q8_1_wide(&y[(i + 3 * stride) * (qk / QK8_1)]);
    }

    for (; i + 3 * stride < blocks_per_row; i += stride4) {
        const int ibx0 = row_offset + i;
        const int ibx1 = row_offset + i + stride;
        const int ibx2 = row_offset + i + 2 * stride;
        const int ibx3 = row_offset + i + 3 * stride;

#pragma unroll
        for (size_t elem = 0; elem < qi_div_vdr; elem += WARP_SIZE) {
            const int iqs = elem + base_iqs;
            acc0 += vec_dot_q_sycl(&x[ibx0], &y0, iqs);
            acc1 += vec_dot_q_sycl(&x[ibx1], &y1, iqs);
            acc2 += vec_dot_q_sycl(&x[ibx2], &y2, iqs);
            acc3 += vec_dot_q_sycl(&x[ibx3], &y3, iqs);
        }

        const int next_i = i + stride4;
        if (next_i + 3 * stride < blocks_per_row) {
            y0 = load_block_q8_1_wide(&y[next_i * (qk / QK8_1)]);
            y1 = load_block_q8_1_wide(&y[(next_i + stride) * (qk / QK8_1)]);
            y2 = load_block_q8_1_wide(&y[(next_i + 2 * stride) * (qk / QK8_1)]);
            y3 = load_block_q8_1_wide(&y[(next_i + 3 * stride) * (qk / QK8_1)]);
        }
    }

    for (; i < blocks_per_row; i += stride) {
        const int ibx = row_offset + i;
        const int iby = i * (qk / QK8_1);
        const block_q8_1 y_local = load_block_q8_1_wide(&y[iby]);

#pragma unroll
        for (size_t elem = 0; elem < qi_div_vdr; elem += WARP_SIZE) {
            const int iqs = elem + base_iqs;
            acc0 += vec_dot_q_sycl(&x[ibx], &y_local, iqs);
        }
    }

    float tmp = (acc0 + acc1) + (acc2 + acc3);
    tmp = sycl::reduce_over_group(item_ct1.get_sub_group(), tmp, sycl::plus<float>());

    if (lane_id == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void mul_mat_vec_q_wide(const void * __restrict__ vx,
                               const void * __restrict__ vy,
                               float * __restrict__ dst,
                               const int                ncols,
                               const int                nrows,
                               const sycl::nd_item<3> & item_ct1) {
    if constexpr (qk > QK8_1) {
        mul_mat_vec_q_baseline<qk, qi, block_q_t, vdr, vec_dot_q_sycl>(vx, vy, dst, ncols, nrows, item_ct1);
        return;
    }

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row  = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    constexpr int qi_div_vdr = qi / vdr;
    const int     lane_id    = item_ct1.get_local_id(2);
    const int     base_iqs   = vdr * (lane_id % qi_div_vdr);
    const int     row_offset = row * blocks_per_row;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    int       i       = lane_id / qi_div_vdr;
    const int stride  = blocks_per_warp;
    const int stride4 = 4 * stride;

    for (; i + 3 * stride < blocks_per_row; i += stride4) {
        const int ibx0 = row_offset + i;
        const int ibx1 = row_offset + i + stride;
        const int ibx2 = row_offset + i + 2 * stride;
        const int ibx3 = row_offset + i + 3 * stride;

        const int iby0 = i * (qk / QK8_1);
        const int iby1 = (i + stride) * (qk / QK8_1);
        const int iby2 = (i + 2 * stride) * (qk / QK8_1);
        const int iby3 = (i + 3 * stride) * (qk / QK8_1);

        const block_q8_1 y0 = load_block_q8_1_wide(&y[iby0]);
        const block_q8_1 y1 = load_block_q8_1_wide(&y[iby1]);
        const block_q8_1 y2 = load_block_q8_1_wide(&y[iby2]);
        const block_q8_1 y3 = load_block_q8_1_wide(&y[iby3]);

#pragma unroll
        for (size_t elem = 0; elem < qi_div_vdr; elem += WARP_SIZE) {
            const int iqs = elem + base_iqs;
            acc0 += vec_dot_q_sycl(&x[ibx0], &y0, iqs);
            acc1 += vec_dot_q_sycl(&x[ibx1], &y1, iqs);
            acc2 += vec_dot_q_sycl(&x[ibx2], &y2, iqs);
            acc3 += vec_dot_q_sycl(&x[ibx3], &y3, iqs);
        }
    }

    for (; i < blocks_per_row; i += stride) {
        const int ibx = row_offset + i;
        const int iby = i * (qk / QK8_1);
        const block_q8_1 y_local = load_block_q8_1_wide(&y[iby]);

#pragma unroll
        for (size_t elem = 0; elem < qi_div_vdr; elem += WARP_SIZE) {
            const int iqs = elem + base_iqs;
            acc0 += vec_dot_q_sycl(&x[ibx], &y_local, iqs);
        }
    }

    float tmp = (acc0 + acc1) + (acc2 + acc3);
    tmp = sycl::reduce_over_group(item_ct1.get_sub_group(), tmp, sycl::plus<float>());

    if (lane_id == 0) {
        dst[row] = tmp;
    }
}

template <int qk,
          int qi,
          typename block_q_t,
          int vdr,
          float (*vec_dot_q_slm)(const void *, const int *, const sycl::half2 *, int, int, const int &),
          int nrows_per_wg>
static void mul_mat_vec_q_multirow(const void * __restrict__ vx,
                                   const void * __restrict__ vy,
                                   float * __restrict__ dst,
                                   const int                ncols,
                                   const int                nrows,
                                   const sycl::nd_item<3> & item_ct1,
                                   int * __restrict__ slm_y_qs,
                                   sycl::half2 * __restrict__ slm_y_ds) {
    const int local_row = item_ct1.get_local_id(1);
    const int lane_id   = item_ct1.get_local_id(2);
    const int wg_idx    = item_ct1.get_group(2);
    const int row       = wg_idx * nrows_per_wg + local_row;

    const int blocks_per_row = ncols / qk;
    constexpr int blocks_per_y = qk / QK8_1;
    const int total_y_blocks = blocks_per_row * blocks_per_y;

    if (local_row == 0) {
        const block_q8_1 * y = (const block_q8_1 *) vy;
        for (int blk = lane_id; blk < total_y_blocks; blk += WARP_SIZE) {
            const int slm_offset = blk * MMVQ_SLM_Y_QS_STRIDE;
#pragma unroll
            for (int j = 0; j < QI8_1; ++j) {
                slm_y_qs[slm_offset + j] = get_int_from_int8_aligned(y[blk].qs, j);
            }
            slm_y_ds[blk] = *((const sycl::half2 *) &y[blk].ds);
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    if (row >= nrows) {
        return;
    }

    const block_q_t * x = (const block_q_t *) vx;

    constexpr int qi_div_vdr      = qi / vdr;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;
    const int     base_iqs        = vdr * (lane_id % qi_div_vdr);
    const int     row_offset      = row * blocks_per_row;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    int       i       = lane_id / qi_div_vdr;
    const int stride  = blocks_per_warp;
    const int stride4 = 4 * stride;

    for (; i + 3 * stride < blocks_per_row; i += stride4) {
        const int ibx0 = row_offset + i;
        const int ibx1 = row_offset + i + stride;
        const int ibx2 = row_offset + i + 2 * stride;
        const int ibx3 = row_offset + i + 3 * stride;

        const int iby0 = i * (qk / QK8_1);
        const int iby1 = (i + stride) * (qk / QK8_1);
        const int iby2 = (i + 2 * stride) * (qk / QK8_1);
        const int iby3 = (i + 3 * stride) * (qk / QK8_1);

#pragma unroll
        for (size_t elem = 0; elem < qi_div_vdr; elem += WARP_SIZE) {
            const int iqs = elem + base_iqs;
            acc0 += vec_dot_q_slm(&x[ibx0], slm_y_qs, slm_y_ds, iby0, MMVQ_SLM_Y_QS_STRIDE, iqs);
            acc1 += vec_dot_q_slm(&x[ibx1], slm_y_qs, slm_y_ds, iby1, MMVQ_SLM_Y_QS_STRIDE, iqs);
            acc2 += vec_dot_q_slm(&x[ibx2], slm_y_qs, slm_y_ds, iby2, MMVQ_SLM_Y_QS_STRIDE, iqs);
            acc3 += vec_dot_q_slm(&x[ibx3], slm_y_qs, slm_y_ds, iby3, MMVQ_SLM_Y_QS_STRIDE, iqs);
        }
    }

    for (; i < blocks_per_row; i += stride) {
        const int ibx = row_offset + i;
        const int iby = i * (qk / QK8_1);

#pragma unroll
        for (size_t elem = 0; elem < qi_div_vdr; elem += WARP_SIZE) {
            const int iqs = elem + base_iqs;
            acc0 += vec_dot_q_slm(&x[ibx], slm_y_qs, slm_y_ds, iby, MMVQ_SLM_Y_QS_STRIDE, iqs);
        }
    }

    float tmp = (acc0 + acc1) + (acc2 + acc3);
    tmp = sycl::reduce_over_group(item_ct1.get_sub_group(), tmp, sycl::plus<float>());

    if (lane_id == 0) {
        dst[row] = tmp;
    }
}

template <int qtype,
          int qk,
          int qi,
          typename block_q_t,
          int vdr,
          vec_dot_q_sycl_t vec_dot_q_sycl,
          int variant>
static sycl::event launch_aos_kernel(const void *    vx,
                                     const void *    vy,
                                     float *         dst,
                                     const int       ncols,
                                     const int       nrows,
                                     sycl::queue &   queue) {
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    return queue.submit([&](sycl::handler & cgh) {
        cgh.parallel_for<mmvq_tier1_kernel_name<qtype, variant>>(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                if constexpr (variant == kVariantBaseline) {
                    mul_mat_vec_q_baseline<qk, qi, block_q_t, vdr, vec_dot_q_sycl>(
                        vx, vy, dst, ncols, nrows, item_ct1);
                } else if constexpr (variant == kVariantPrefetch) {
                    mul_mat_vec_q_prefetch<qk, qi, block_q_t, vdr, vec_dot_q_sycl>(
                        vx, vy, dst, ncols, nrows, item_ct1);
                } else {
                    mul_mat_vec_q_wide<qk, qi, block_q_t, vdr, vec_dot_q_sycl>(
                        vx, vy, dst, ncols, nrows, item_ct1);
                }
            });
    });
}

template <int qtype,
          int qk,
          int qi,
          typename block_q_t,
          int vdr,
          float (*vec_dot_q_slm)(const void *, const int *, const sycl::half2 *, int, int, const int &),
          int nrows_per_wg>
static sycl::event launch_slm_kernel(const void *    vx,
                                     const void *    vy,
                                     float *         dst,
                                     const int       ncols,
                                     const int       nrows,
                                     sycl::queue &   queue) {
    const int            block_num_z = (nrows + nrows_per_wg - 1) / nrows_per_wg;
    const sycl::range<3> block_nums(1, 1, block_num_z);
    const sycl::range<3> block_dims(1, nrows_per_wg, WARP_SIZE);

    const int blocks_per_row = ncols / qk;
    constexpr int blocks_per_y = qk / QK8_1;
    const int total_y_blocks = blocks_per_row * blocks_per_y;
    const int slm_y_qs_size  = total_y_blocks * MMVQ_SLM_Y_QS_STRIDE;
    const int slm_y_ds_size  = total_y_blocks + 1;

    return queue.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<int, 1>         slm_y_qs(slm_y_qs_size, cgh);
        sycl::local_accessor<sycl::half2, 1> slm_y_ds(slm_y_ds_size, cgh);

        cgh.parallel_for<mmvq_tier1_multirow_kernel_name<qtype, kVariantSlm>>(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                auto slm_y_qs_ptr = slm_y_qs.template get_multi_ptr<sycl::access::decorated::no>();
                auto slm_y_ds_ptr = slm_y_ds.template get_multi_ptr<sycl::access::decorated::no>();
                mul_mat_vec_q_multirow<qk, qi, block_q_t, vdr, vec_dot_q_slm, nrows_per_wg>(
                    vx, vy, dst, ncols, nrows, item_ct1, slm_y_qs_ptr.get(), slm_y_ds_ptr.get());
            });
    });
}

static inline bool validate_aos_args(const ggml_sycl::mmvq_bench_args & args, std::string & error) {
    if (!args.stream || !args.weights || !args.activations || !args.output) {
        error = "Invalid null pointers for MMVQ args.";
        return false;
    }
    if (args.ncols <= 0 || args.nrows <= 0 || args.batch <= 0) {
        error = "Invalid MMVQ dimensions.";
        return false;
    }
    if (args.ncols % QK8_1 != 0) {
        error = "MMVQ requires ncols multiple of QK8_1.";
        return false;
    }
    if (args.row_low < 0 || args.row_high <= args.row_low || args.row_high > args.nrows) {
        error = "Invalid MMVQ row range.";
        return false;
    }
    if (args.src1_padded_col_size <= 0 || args.dst_row_stride <= 0) {
        error = "Invalid MMVQ strides.";
        return false;
    }
    if (args.layout != GGML_LAYOUT_AOS) {
        error = "Tier1 AOS variants require AOS layout.";
        return false;
    }
    return true;
}

template <AosVariant variant>
static bool launch_aos_variant(const ggml_sycl::mmvq_bench_args & args,
                               std::vector<sycl::event> * events,
                               std::string & error) {
    if (!validate_aos_args(args, error)) {
        return false;
    }

    sycl::queue & queue = *args.stream;

    const int64_t row_low  = args.row_low;
    const int64_t row_high = args.row_high;
    const int64_t nrows    = row_high - row_low;

    const size_t row_bytes = ggml_row_size(args.weight_type, args.ncols);
    const char * weight_base = static_cast<const char *>(args.weights);
    const char * weight_ptr  = weight_base + row_low * row_bytes;

    float * output_base = args.output + row_low;

    const size_t src1_row_bytes =
        static_cast<size_t>(args.src1_padded_col_size) * sizeof(block_q8_1) / QK8_1;

    auto record_event = [&](sycl::event evt) {
        if (events) {
            events->push_back(std::move(evt));
        }
    };

    constexpr int variant_id = (variant == AosVariant::Baseline)
        ? kVariantBaseline
        : (variant == AosVariant::Prefetch ? kVariantPrefetch : kVariantWideLoad);

    for (int64_t i = 0; i < args.batch; ++i) {
        const char * y_ptr = static_cast<const char *>(args.activations) + i * src1_row_bytes;
        float * dst_ptr = output_base + i * args.dst_row_stride;

        switch (args.weight_type) {
            case GGML_TYPE_Q4_0:
                record_event(launch_aos_kernel<GGML_TYPE_Q4_0, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ,
                                              vec_dot_q4_0_q8_1, variant_id>(
                    weight_ptr, y_ptr, dst_ptr, args.ncols, static_cast<int>(nrows), queue));
                break;
            case GGML_TYPE_Q4_1:
                {
                    record_event(launch_aos_kernel<GGML_TYPE_Q4_1, QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ,
                                                  vec_dot_q4_1_q8_1, variant_id>(
                        weight_ptr, y_ptr, dst_ptr, args.ncols, static_cast<int>(nrows), queue));
                }
                break;
            case GGML_TYPE_Q5_0:
                {
                    record_event(launch_aos_kernel<GGML_TYPE_Q5_0, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ,
                                                  vec_dot_q5_0_q8_1, variant_id>(
                        weight_ptr, y_ptr, dst_ptr, args.ncols, static_cast<int>(nrows), queue));
                }
                break;
            case GGML_TYPE_Q5_1:
                {
                    record_event(launch_aos_kernel<GGML_TYPE_Q5_1, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ,
                                                  vec_dot_q5_1_q8_1, variant_id>(
                        weight_ptr, y_ptr, dst_ptr, args.ncols, static_cast<int>(nrows), queue));
                }
                break;
            case GGML_TYPE_Q8_0:
                {
                    record_event(launch_aos_kernel<GGML_TYPE_Q8_0, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
                                                  vec_dot_q8_0_q8_1, variant_id>(
                        weight_ptr, y_ptr, dst_ptr, args.ncols, static_cast<int>(nrows), queue));
                }
                break;
            case GGML_TYPE_MXFP4:
                {
                    record_event(launch_aos_kernel<GGML_TYPE_MXFP4, QK_MXFP4, QI_MXFP4, block_mxfp4,
                                                  VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1, variant_id>(
                        weight_ptr, y_ptr, dst_ptr, args.ncols, static_cast<int>(nrows), queue));
                }
                break;
            case GGML_TYPE_Q2_K:
                {
                    record_event(launch_aos_kernel<GGML_TYPE_Q2_K, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ,
                                                  vec_dot_q2_K_q8_1, variant_id>(
                        weight_ptr, y_ptr, dst_ptr, args.ncols, static_cast<int>(nrows), queue));
                }
                break;
            case GGML_TYPE_Q3_K:
                {
                    record_event(launch_aos_kernel<GGML_TYPE_Q3_K, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ,
                                                  vec_dot_q3_K_q8_1, variant_id>(
                        weight_ptr, y_ptr, dst_ptr, args.ncols, static_cast<int>(nrows), queue));
                }
                break;
            case GGML_TYPE_Q4_K:
                {
                    record_event(launch_aos_kernel<GGML_TYPE_Q4_K, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ,
                                                  vec_dot_q4_K_q8_1, variant_id>(
                        weight_ptr, y_ptr, dst_ptr, args.ncols, static_cast<int>(nrows), queue));
                }
                break;
            case GGML_TYPE_Q5_K:
                {
                    record_event(launch_aos_kernel<GGML_TYPE_Q5_K, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ,
                                                  vec_dot_q5_K_q8_1, variant_id>(
                        weight_ptr, y_ptr, dst_ptr, args.ncols, static_cast<int>(nrows), queue));
                }
                break;
            case GGML_TYPE_Q6_K:
                {
                    record_event(launch_aos_kernel<GGML_TYPE_Q6_K, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ,
                                                  vec_dot_q6_K_q8_1, variant_id>(
                        weight_ptr, y_ptr, dst_ptr, args.ncols, static_cast<int>(nrows), queue));
                }
                break;
            default:
                error = "Unsupported quant type for Tier1 AOS variant.";
                return false;
        }
    }

    return true;
}

static bool launch_slm_cached(const ggml_sycl::mmvq_bench_args & args,
                              std::vector<sycl::event> * events,
                              std::string & error) {
    if (!validate_aos_args(args, error)) {
        return false;
    }

    sycl::queue & queue = *args.stream;

    const int64_t row_low  = args.row_low;
    const int64_t row_high = args.row_high;
    const int64_t nrows    = row_high - row_low;

    const size_t row_bytes = ggml_row_size(args.weight_type, args.ncols);
    const char * weight_base = static_cast<const char *>(args.weights);
    const char * weight_ptr  = weight_base + row_low * row_bytes;

    float * output_base = args.output + row_low;

    const size_t src1_row_bytes =
        static_cast<size_t>(args.src1_padded_col_size) * sizeof(block_q8_1) / QK8_1;

    auto record_event = [&](sycl::event evt) {
        if (events) {
            events->push_back(std::move(evt));
        }
    };

    for (int64_t i = 0; i < args.batch; ++i) {
        const char * y_ptr = static_cast<const char *>(args.activations) + i * src1_row_bytes;
        float * dst_ptr = output_base + i * args.dst_row_stride;

        switch (args.weight_type) {
            case GGML_TYPE_Q4_0:
                record_event(launch_slm_kernel<GGML_TYPE_Q4_0, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ,
                                               vec_dot_q4_0_q8_1_slm, MMVQ_NROWS_PER_WG>(
                    weight_ptr, y_ptr, dst_ptr, args.ncols, static_cast<int>(nrows), queue));
                break;
            case GGML_TYPE_Q8_0:
                record_event(launch_slm_kernel<GGML_TYPE_Q8_0, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
                                               vec_dot_q8_0_q8_1_slm, MMVQ_NROWS_PER_WG>(
                    weight_ptr, y_ptr, dst_ptr, args.ncols, static_cast<int>(nrows), queue));
                break;
            case GGML_TYPE_Q4_1:
                record_event(launch_slm_kernel<GGML_TYPE_Q4_1, QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ,
                                               vec_dot_q_slm_generic<vec_dot_q4_1_q8_1>, MMVQ_NROWS_PER_WG>(
                    weight_ptr, y_ptr, dst_ptr, args.ncols, static_cast<int>(nrows), queue));
                break;
            case GGML_TYPE_Q5_0:
                record_event(launch_slm_kernel<GGML_TYPE_Q5_0, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ,
                                               vec_dot_q_slm_generic<vec_dot_q5_0_q8_1>, MMVQ_NROWS_PER_WG>(
                    weight_ptr, y_ptr, dst_ptr, args.ncols, static_cast<int>(nrows), queue));
                break;
            case GGML_TYPE_Q5_1:
                record_event(launch_slm_kernel<GGML_TYPE_Q5_1, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ,
                                               vec_dot_q_slm_generic<vec_dot_q5_1_q8_1>, MMVQ_NROWS_PER_WG>(
                    weight_ptr, y_ptr, dst_ptr, args.ncols, static_cast<int>(nrows), queue));
                break;
            case GGML_TYPE_MXFP4:
                record_event(launch_slm_kernel<GGML_TYPE_MXFP4, QK_MXFP4, QI_MXFP4, block_mxfp4,
                                               VDR_MXFP4_Q8_1_MMVQ, vec_dot_q_slm_generic<vec_dot_mxfp4_q8_1>,
                                               MMVQ_NROWS_PER_WG>(
                    weight_ptr, y_ptr, dst_ptr, args.ncols, static_cast<int>(nrows), queue));
                break;
            case GGML_TYPE_Q2_K:
                record_event(launch_aos_kernel<GGML_TYPE_Q2_K, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ,
                                              vec_dot_q2_K_q8_1, kVariantBaseline>(
                    weight_ptr, y_ptr, dst_ptr, args.ncols, static_cast<int>(nrows), queue));
                break;
            case GGML_TYPE_Q3_K:
                record_event(launch_aos_kernel<GGML_TYPE_Q3_K, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ,
                                              vec_dot_q3_K_q8_1, kVariantBaseline>(
                    weight_ptr, y_ptr, dst_ptr, args.ncols, static_cast<int>(nrows), queue));
                break;
            case GGML_TYPE_Q4_K:
                record_event(launch_aos_kernel<GGML_TYPE_Q4_K, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ,
                                              vec_dot_q4_K_q8_1, kVariantBaseline>(
                    weight_ptr, y_ptr, dst_ptr, args.ncols, static_cast<int>(nrows), queue));
                break;
            case GGML_TYPE_Q5_K:
                record_event(launch_aos_kernel<GGML_TYPE_Q5_K, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ,
                                              vec_dot_q5_K_q8_1, kVariantBaseline>(
                    weight_ptr, y_ptr, dst_ptr, args.ncols, static_cast<int>(nrows), queue));
                break;
            case GGML_TYPE_Q6_K:
                record_event(launch_aos_kernel<GGML_TYPE_Q6_K, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ,
                                              vec_dot_q6_K_q8_1, kVariantBaseline>(
                    weight_ptr, y_ptr, dst_ptr, args.ncols, static_cast<int>(nrows), queue));
                break;
            default:
                error = "Unsupported quant type for Tier1 SLM variant.";
                return false;
        }
    }

    return true;
}

}  // namespace mmvq_tier1
}  // namespace sycl_bench
