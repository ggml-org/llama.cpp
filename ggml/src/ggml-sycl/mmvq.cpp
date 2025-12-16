#include "mmvq.hpp"

#include "ggml.h"
#include "common.hpp"
#include "quants.hpp"
#include "quantize.hpp"
#include "vecdotq.hpp"

template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_reorder(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                                  const int ncols, const int nrows, const sycl::nd_item<3> & nd_item) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const auto sg           = nd_item.get_sub_group();
    const int  sg_range     = sg.get_group_linear_range();
    const int  workgroup_id = nd_item.get_group_linear_id();
    const int  sg_id        = sg.get_group_linear_id();
    const int  row          = workgroup_id * sg_range + sg_id;

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row              = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    const int     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    float partial_sum = 0.0f;
    for (int i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const auto         bx_offset      = block_type::get_block_offset(ibx, nblocks);
        const auto         d_offset       = block_type::get_d_offset(nrows, ncols, ibx);
        // Y block index that aligns with ibx
        const int iby = i * block_type::block_to_q8_1_ratio();
        const int8_t* q8_1_quant_ptr = (const int8_t*)vy + iby * QK8_1;
        const sycl::half2* q8_1_ds_ptr = (const sycl::half2*)((const char*)vy + ncols + iby * sizeof(sycl::half2));

#pragma unroll
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            // x block quant index when casting the quants to int
            const int iqs = elem + block_traits::vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);

            partial_sum += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
        }
    }

    auto sum = sycl::reduce_over_group(nd_item.get_sub_group(), partial_sum, std::plus<>());

    if (sg.leader()) {
        dst[row] = sum;
    }
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void mul_mat_vec_q(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                          const int ncols, const int nrows, const sycl::nd_item<3> & item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int     blocks_per_row  = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;  // Ensuring blocks_per_warp > 0

    assert(blocks_per_warp > 0);

    // partial sum for each thread
    float tmp = 0.0f;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;  // x block index

        const int iby = i * (qk / QK8_1);          // y block index that aligns with ibx

        for (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
            const int iqs = elem + vdr * (item_ct1.get_local_id(2) %
                                          (qi / vdr));  // x block quant index when casting the quants to int

            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

// MoE-aware kernel: routes to different experts based on ids tensor
// Handles 2D iteration: (iid1, id) over tokens and expert selections
// For MUL_MAT_ID: reads ids[iid1][id] to determine which expert weights to use
// This allows GPU-side expert routing without host sync
template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void mul_mat_vec_q_id(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                              const int32_t * __restrict__ ids, const int ncols, const int nrows_per_expert,
                              const int n_ids, const int ne11,
                              const int64_t stride_expert_x,
                              const int64_t ids_nb0, const int64_t ids_nb1,
                              const int64_t nb11, const int64_t nb12,
                              const int64_t nb1, const int64_t nb2,
                              const sycl::nd_item<3> & item_ct1) {
    // batch_idx from block.y dimension (linearized over iid1 * n_ids + id)
    const int batch_idx = item_ct1.get_group(1);
    // row within expert from block.z dimension
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows_per_expert) {
        return;
    }

    // Decompose batch_idx into (iid1, id) - 2D iteration structure
    const int iid1 = batch_idx / n_ids;  // Token position
    const int id = batch_idx % n_ids;    // Expert selection index

    // Read expert ID from ids tensor using proper 2D indexing
    const int32_t expert_id = *(const int32_t *)((const char*)ids + iid1 * ids_nb1 + id * ids_nb0);

    // Compute src1 and dst offsets matching host-side logic
    const int64_t i11 = id % ne11;
    const int64_t i12 = iid1;
    const int64_t i1 = id;
    const int64_t i2 = iid1;

    const int     blocks_per_row  = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;

    assert(blocks_per_warp > 0);

    float tmp = 0.0f;

    // Expert weights: offset by expert_id * stride_expert_x
    const block_q_t *  x = (const block_q_t *) ((const char*)vx + expert_id * stride_expert_x);
    // Input: offset using proper 2D indexing
    const block_q8_1 * y = (const block_q8_1 *) ((const char*)vy + i11 * nb11 + i12 * nb12);

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;
        const int iby = i * (qk / QK8_1);

        for (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
            const int iqs = elem + vdr * (item_ct1.get_local_id(2) % (qi / vdr));
            tmp += vec_dot_q_sycl(&x[ibx], &y[iby], iqs);
        }
    }

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        // Output: offset using proper 2D indexing
        float * dst_out = (float*)((char*)dst + i1 * nb1 + i2 * nb2);
        dst_out[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_xxs_q8_1(const void *__restrict__ vx,
                                       const void *__restrict__ vy,
                                       float *__restrict__ dst, const int ncols,
                                       const int nrows,
                                       const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);

// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xxs_q8_1(&x[ibx], &y[iby], iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_xs_q8_1(const void *__restrict__ vx,
                                      const void *__restrict__ vy,
                                      float *__restrict__ dst, const int ncols,
                                      const int nrows,
                                      const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_xs_q8_1(&x[ibx], &y[iby], iqs, iq2xs_grid, ksigns64);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq2_s_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq2_s_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq3_xxs_q8_1(const void *__restrict__ vx,
                                       const void *__restrict__ vy,
                                       float *__restrict__ dst, const int ncols,
                                       const int nrows,
                                       const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_xxs_q8_1(&x[ibx], &y[iby], iqs, iq3xxs_grid, ksigns64);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq3_s_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq3_s_q8_1(&x[ibx], &y[iby], iqs, iq3s_grid);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq1_s_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_s_q8_1(&x[ibx], &y[iby], iqs, iq1s_grid_gpu);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq1_m_q8_1(const void *__restrict__ vx,
                                     const void *__restrict__ vy,
                                     float *__restrict__ dst, const int ncols,
                                     const int nrows,
                                     const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq1_m_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq4_nl_q8_1(const void *__restrict__ vx,
                                      const void *__restrict__ vy,
                                      float *__restrict__ dst, const int ncols,
                                      const int nrows,
                                      const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_nl_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}


template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_iq4_xs_q8_1(const void *__restrict__ vx,
                                      const void *__restrict__ vy,
                                      float *__restrict__ dst, const int ncols,
                                      const int nrows,
                                      const sycl::nd_item<3> &item_ct1) {
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp>0);
// partial sum for each thread
    float tmp = 0.0f;

    const block_q_t  * x = (const block_q_t  *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row;
         i += blocks_per_warp) {
        const int ibx = row*blocks_per_row + i; // x block index

        const int iby = i * (qk/QK8_1); // y block index that aligns with ibx

        const int iqs =
            vdr *
            (item_ct1.get_local_id(2) %
             (qi / vdr)); // x block quant index when casting the quants to int

        tmp += vec_dot_iq4_xs_q8_1(&x[ibx], &y[iby], iqs);
    }

    // sum up partial sums and write back result
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        dst[row] = tmp;
    }
}

static void reorder_mul_mat_vec_q4_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                                    const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, (block_num_y * WARP_SIZE));
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

static void mul_mat_vec_q4_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols, const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                                     vx, vy, dst, ncols, nrows, item_ct1);
                             });
        });
    }
}

// MoE dispatch: Q4_0 with expert routing via ids tensor (GPU-side, no host sync)
static void mul_mat_vec_q4_0_q8_1_id_sycl(const void * vx, const void * vy, float * dst, const int32_t * ids,
                                          const int ncols, const int nrows_per_expert,
                                          const int total_batches, const int n_ids, const int ne11,
                                          const int64_t stride_expert_x,
                                          const int64_t ids_nb0, const int64_t ids_nb1,
                                          const int64_t nb11, const int64_t nb12,
                                          const int64_t nb1, const int64_t nb2,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    const int block_num_z = (nrows_per_expert + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    // 3D dispatch: (1, total_batches, block_num_z) - batch in y dimension, rows in z
    const sycl::range<3> block_nums(1, total_batches, block_num_z);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_id<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                                 vx, vy, dst, ids, ncols, nrows_per_expert,
                                 n_ids, ne11, stride_expert_x,
                                 ids_nb0, ids_nb1, nb11, nb12, nb1, nb2, item_ct1);
                         });
    });
}

static void mul_mat_vec_q4_1_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK4_0, QI4_1, block_q4_1,
                                      VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_mxfp4_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols, const int nrows,
                                        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q<QK_MXFP4, QI_MXFP4, block_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                                     vx, vy, dst, ncols, nrows, item_ct1);
                             });
        });
    }
}


static void mul_mat_vec_q5_0_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK5_0, QI5_0, block_q5_0,
                                      VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_q5_1_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK5_1 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK5_1, QI5_1, block_q5_1,
                                      VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_q8_0_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK8_0, QI8_0, block_q8_0,
                                      VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

// MoE dispatch: Q8_0 with expert routing via ids tensor (GPU-side, no host sync)
static void mul_mat_vec_q8_0_q8_1_id_sycl(const void * vx, const void * vy, float * dst, const int32_t * ids,
                                          const int ncols, const int nrows_per_expert,
                                          const int total_batches, const int n_ids, const int ne11,
                                          const int64_t stride_expert_x,
                                          const int64_t ids_nb0, const int64_t ids_nb1,
                                          const int64_t nb11, const int64_t nb12,
                                          const int64_t nb1, const int64_t nb2,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    const int block_num_z = (nrows_per_expert + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, total_batches, block_num_z);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_id<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                                 vx, vy, dst, ids, ncols, nrows_per_expert,
                                 n_ids, ne11, stride_expert_x,
                                 ids_nb0, ids_nb1, nb11, nb12, nb1, nb2, item_ct1);
                         });
    });
}

static void mul_mat_vec_q2_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI2_K, block_q2_K,
                                      VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_q3_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI3_K, block_q3_K,
                                      VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_q4_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI4_K, block_q4_K,
                                      VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void reorder_mul_mat_vec_q4_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
    const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_item);
                            });
    });
}


static void mul_mat_vec_q5_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI5_K, block_q5_K,
                                      VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void reorder_mul_mat_vec_q6_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                               const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int        block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y);
    constexpr size_t num_subgroups = 16;
    GGML_ASSERT(block_num_y % num_subgroups == 0);

    const sycl::range<3> global_size(1, GGML_SYCL_MMV_Y, block_num_y * WARP_SIZE);
    const sycl::range<3> workgroup_size(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(global_size, workgroup_size),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}
static void mul_mat_vec_q6_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q<QK_K, QI6_K, block_q6_K,
                                      VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}


static void mul_mat_vec_iq2_xxs_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq2_xxs_q8_1<QK_K, QI2_XXS/2, block_iq2_xxs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq2_xs_q8_1_sycl(const void *vx, const void *vy,
                                         float *dst, const int ncols,
                                         const int nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq2_xs_q8_1<QK_K, QI2_XS/2, block_iq2_xs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq2_s_q8_1_sycl(const void *vx, const void *vy,
                                         float *dst, const int ncols,
                                         const int nrows,
                                         dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq2_s_q8_1<QK_K, QI2_S/2, block_iq2_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq3_xxs_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq3_xxs_q8_1<QK_K, QI3_XXS/2, block_iq3_xxs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq3_s_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq3_s_q8_1<QK_K, QI3_S/2, block_iq3_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq1_s_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq1_s_q8_1<QK_K, QI1_S, block_iq1_s, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq1_m_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {
        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq1_m_q8_1<QK_K, QI1_S, block_iq1_m, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq4_nl_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_NL == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq4_nl_q8_1<QK4_NL, QI4_NL, block_iq4_nl, 2>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

static void mul_mat_vec_iq4_xs_q8_1_sycl(const void *vx, const void *vy,
                                          float *dst, const int ncols,
                                          const int nrows,
                                          dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    {

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        mul_mat_vec_q_iq4_xs_q8_1<QK_K, QI4_XS/4, block_iq4_xs, 1>(
                            vx, vy, dst, ncols, nrows, item_ct1);
                    });
        });
    }
}

// MoE-aware MXFP4 kernel: routes to different experts based on ids tensor
// Handles 2D iteration: (iid1, id) over tokens and expert selections
template <int qk, int qi, typename block_q_t, int vdr>
static void mul_mat_vec_q_mxfp4_q8_1_id(const void * __restrict__ vx, const void * __restrict__ vy, float * __restrict__ dst,
                                         const int32_t * __restrict__ ids, const int ncols, const int nrows_per_expert,
                                         const int n_ids, const int ne11,
                                         const int64_t stride_expert_x,
                                         const int64_t ids_nb0, const int64_t ids_nb1,
                                         const int64_t nb11, const int64_t nb12,
                                         const int64_t nb1, const int64_t nb2,
                                         const sycl::nd_item<3> & item_ct1) {
    const int batch_idx = item_ct1.get_group(1);
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows_per_expert) {
        return;
    }

    // Decompose batch_idx into (iid1, id) - 2D iteration structure
    const int iid1 = batch_idx / n_ids;  // Token position
    const int id = batch_idx % n_ids;    // Expert selection index

    // Read expert ID from ids tensor using proper 2D indexing
    const int32_t expert_id = *(const int32_t *)((const char*)ids + iid1 * ids_nb1 + id * ids_nb0);

    // Compute src1 and dst offsets matching host-side logic
    const int64_t i11 = id % ne11;
    const int64_t i12 = iid1;
    const int64_t i1 = id;
    const int64_t i2 = iid1;

    const int blocks_per_row = ncols / qk;
    const int blocks_per_warp = vdr * WARP_SIZE / qi;
    assert(blocks_per_warp > 0);

    float tmp = 0.0f;

    // Expert weights: offset by expert_id * stride_expert_x
    const block_q_t *  x = (const block_q_t *) ((const char*)vx + expert_id * stride_expert_x);
    // Input: offset using proper 2D indexing
    const block_q8_1 * y = (const block_q8_1 *) ((const char*)vy + i11 * nb11 + i12 * nb12);

    for (int i = item_ct1.get_local_id(2) / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
        const int ibx = row * blocks_per_row + i;
        const int iby = i * (qk / QK8_1);
        const int iqs = vdr * (item_ct1.get_local_id(2) % (qi / vdr));

        tmp += vec_dot_mxfp4_q8_1(&x[ibx], &y[iby], iqs);
    }

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    if (item_ct1.get_local_id(2) == 0) {
        // Output: offset using proper 2D indexing
        float * dst_out = (float*)((char*)dst + i1 * nb1 + i2 * nb2);
        dst_out[row] = tmp;
    }
}

// MoE dispatch: MXFP4 with expert routing via ids tensor (GPU-side, no host sync)
static void mul_mat_vec_mxfp4_q8_1_id_sycl(const void * vx, const void * vy, float * dst, const int32_t * ids,
                                           const int ncols, const int nrows_per_expert,
                                           const int total_batches, const int n_ids, const int ne11,
                                           const int64_t stride_expert_x,
                                           const int64_t ids_nb0, const int64_t ids_nb1,
                                           const int64_t nb11, const int64_t nb12,
                                           const int64_t nb1, const int64_t nb2,
                                           dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    const int block_num_z = (nrows_per_expert + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, total_batches, block_num_z);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    // Use generic template with vec_dot_mxfp4_q8_1 function pointer
    // This matches how Q4_0 and Q8_0 work
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_id<QK_MXFP4, QI_MXFP4, block_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                                 vx, vy, dst, ids, ncols, nrows_per_expert,
                                 n_ids, ne11, stride_expert_x,
                                 ids_nb0, ids_nb1, nb11, nb12, nb1, nb2, item_ct1);
                         });
    });
}

// MoE-aware MUL_MAT_ID dispatch: GPU-side expert routing without host sync
// This allows SYCL graph recording to work with MoE models
// Returns true if handled, false to fall back to host-side routing
bool ggml_sycl_mul_mat_id_vec_q(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, const ggml_tensor *ids, ggml_tensor *dst) {

    GGML_TENSOR_BINARY_OP_LOCALS;

    // Supports both ne12 == 1 (decode) and ne12 > 1 (prompt)
    // The kernel dispatches over (iid1, id) pairs in parallel

    // Only handle quantized types that have _id kernels
    if (!ggml_is_quantized(src0->type)) {
        return false;
    }

    // Check for supported types
    switch (src0->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
            break;
        default:
            return false;  // Fall back to host routing for other types
    }

    const queue_ptr stream = ctx.stream();

    // Quantize src1 to Q8_1 format - need to handle all rows (ne11 * ne12)
    const int64_t ne10_padded = GGML_PAD(ne10, QK8_1);
    const int64_t q8_1_row_size = ne10_padded * sizeof(block_q8_1) / QK8_1;

    // Total rows = ne11 * ne12 (e.g., 4 expert outputs * 1 token = 4 rows)
    const int64_t total_src1_rows = ne11 * ne12;

    // Allocate space for all rows
    ggml_sycl_pool_alloc<int8_t> src1_q8_1(ctx.pool(), total_src1_rows * q8_1_row_size);

    // Get pointers
    const void * src0_d = src0->data;
    const float * src1_d = (const float *)src1->data;
    const int32_t * ids_d = (const int32_t *)ids->data;
    float * dst_d = (float *)dst->data;

    // Quantize all rows to Q8_1
    // The quantize function handles multiple rows when nrows > 1
    quantize_row_q8_1_sycl<quantize_q8_1>(src1_d, src1_q8_1.get(), ne10, total_src1_rows, ne10_padded, stream);

    // Calculate strides from tensors (matching host-side logic)
    const int64_t n_ids = ids->ne[0];  // Number of expert selections per token
    const int64_t num_tokens = ids->ne[1];  // Number of tokens (should equal ne12)

    // Total batches = tokens * expert_selections_per_token
    const int64_t total_batches = num_tokens * n_ids;

    // stride_expert_x: offset between expert weight matrices
    const int64_t stride_expert_x = nb02;

    // Quantized src1 strides (map to Q8_1 layout)
    // For Q8_1: each row is q8_1_row_size bytes
    // q8_nb11: stride between consecutive rows (dimension 1)
    // q8_nb12: stride between tokens (dimension 2) = ne11 rows per token
    const int64_t q8_nb11 = q8_1_row_size;
    const int64_t q8_nb12 = ne11 * q8_1_row_size;

    // Dispatch based on type
    switch (src0->type) {
        case GGML_TYPE_Q4_0:
            mul_mat_vec_q4_0_q8_1_id_sycl(
                src0_d, src1_q8_1.get(), dst_d, ids_d,
                ne00,  // ncols
                ne01,  // nrows_per_expert
                total_batches,
                n_ids,
                ne11,
                stride_expert_x,
                ids->nb[0], ids->nb[1],  // ids strides
                q8_nb11, q8_nb12,        // Q8_1 strides
                nb1, nb2,                // dst strides
                stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_vec_q8_0_q8_1_id_sycl(
                src0_d, src1_q8_1.get(), dst_d, ids_d,
                ne00,  // ncols
                ne01,  // nrows_per_expert
                total_batches,
                n_ids,
                ne11,
                stride_expert_x,
                ids->nb[0], ids->nb[1],  // ids strides
                q8_nb11, q8_nb12,        // Q8_1 strides
                nb1, nb2,                // dst strides
                stream);
            break;
        case GGML_TYPE_MXFP4:
            {
                // Debug: dump first MXFP4 block and Q8_1 block for debugging
                static int mxfp4_debug_call = 0;
                if (mxfp4_debug_call++ < 1 && std::getenv("GGML_SYCL_MXFP4_DEBUG")) {
                    // Copy first expert's first block to host
                    block_mxfp4 host_mxfp4;
                    block_q8_1 host_q8_1;
                    int32_t host_expert_id;

                    stream->memcpy(&host_expert_id, ids_d, sizeof(int32_t)).wait();
                    stream->memcpy(&host_mxfp4, (const char*)src0_d + host_expert_id * stride_expert_x, sizeof(block_mxfp4)).wait();
                    stream->memcpy(&host_q8_1, src1_q8_1.get(), sizeof(block_q8_1)).wait();

                    fprintf(stderr, "\n[MXFP4 DEBUG] expert_id=%d, ne00=%lld, ne01=%lld, stride_expert_x=%lld\n",
                            host_expert_id, (long long)ne00, (long long)ne01, (long long)stride_expert_x);
                    fprintf(stderr, "[MXFP4 DEBUG] E8M0 exponent: %d\n", host_mxfp4.e);
                    fprintf(stderr, "[MXFP4 DEBUG] qs bytes: ");
                    for (int i = 0; i < 16; i++) {
                        fprintf(stderr, "%02x ", host_mxfp4.qs[i]);
                    }
                    fprintf(stderr, "\n[MXFP4 DEBUG] Q8_1 d=%f, qs: ", host_q8_1.ds[0]);
                    for (int i = 0; i < 32; i++) {
                        fprintf(stderr, "%d ", host_q8_1.qs[i]);
                    }
                    fprintf(stderr, "\n");

                    // Compute CPU reference for first block
                    static const int8_t kvalues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
                    int cpu_sum = 0;
                    for (int i = 0; i < 16; i++) {
                        int8_t w_lo = kvalues[host_mxfp4.qs[i] & 0xF];
                        int8_t w_hi = kvalues[host_mxfp4.qs[i] >> 4];
                        cpu_sum += w_lo * host_q8_1.qs[i];
                        cpu_sum += w_hi * host_q8_1.qs[i + 16];
                    }
                    // sycl_e8m0_to_fp32_half includes 0.5f factor
                    float scale = sycl_e8m0_to_fp32_half(host_mxfp4.e) * host_q8_1.ds[0];
                    fprintf(stderr, "[MXFP4 DEBUG] CPU: raw_sum=%d, scale=%f, result=%f\n",
                            cpu_sum, scale, scale * cpu_sum);
                }
            }
            mul_mat_vec_mxfp4_q8_1_id_sycl(
                src0_d, src1_q8_1.get(), dst_d, ids_d,
                ne00,  // ncols
                ne01,  // nrows_per_expert
                total_batches,
                n_ids,
                ne11,
                stride_expert_x,
                ids->nb[0], ids->nb[1],  // ids strides
                q8_nb11, q8_nb12,        // Q8_1 strides
                nb1, nb2,                // dst strides
                stream);
            break;
        default:
            GGML_ABORT("Unsupported type for MoE GPU dispatch");
    }

    return true;
}

void ggml_sycl_op_mul_mat_vec_q(ggml_backend_sycl_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                                ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
                                const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low,
                                const int64_t row_high, const int64_t src1_ncols, const int64_t src1_padded_col_size,
                                const dpct::queue_ptr & stream) {
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 % QK8_1 == 0);

    const int64_t ne00     = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    // DEBUG: Check dimensions and src1 values for layer 0 projections
    static int mmvq_dbg = 0;
    static int mmvq_b1_dbg = 0;
    static int mmvq_wq_dbg = 0;
    bool is_attn_out_l0 = src0->name && strstr(src0->name, "blk.0.attn_output");
    bool is_wq_l0 = src0->name && strstr(src0->name, "blk.0.attn_q");
    bool do_mmvq_dbg = (mmvq_dbg++ < 5 && is_attn_out_l0) || (src1_ncols == 1 && is_attn_out_l0 && mmvq_b1_dbg++ < 5);

    // DEBUG wq projection for batch=1 NaN issue
    // Controlled by GGML_SYCL_TP_DEBUG environment variable
    if (g_ggml_sycl_tp_debug && is_wq_l0 && src1_ncols == 1 && mmvq_wq_dbg++ < 5) {
        fprintf(stderr, "TP DEBUG MMVQ wq layer0 batch=1: ne00=%lld, row_diff=%lld, src1_ddf_i=%p, src1_ddq_i=%p\n",
                (long long)ne00, (long long)row_diff, (void*)src1_ddf_i, (void*)src1_ddq_i);

        // Check input values (RMS norm output, quantized as Q8_1)
        struct block_q8_1_debug {
            sycl::half d;
            sycl::half s;
            int8_t qs[32];
        };
        block_q8_1_debug blk;
        stream->memcpy(&blk, src1_ddq_i, sizeof(blk)).wait();
        float d_f = static_cast<float>(blk.d);
        fprintf(stderr, "TP DEBUG MMVQ wq input (norm output): d=%f, qs[0..3]=[%d,%d,%d,%d]\n",
                d_f, blk.qs[0], blk.qs[1], blk.qs[2], blk.qs[3]);

        // Check for NaN/inf in d
        if (d_f != d_f || d_f == std::numeric_limits<float>::infinity() || d_f == -std::numeric_limits<float>::infinity()) {
            fprintf(stderr, "TP DEBUG MMVQ wq input: INVALID d value detected!\n");
        }
    }

    // DEBUG: Layer 30 and 31 FFN gate Q8_1 quantized input check
    static int mmvq_l30_gate_dbg = 0;
    static int mmvq_l31_gate_dbg = 0;
    bool is_ffn_gate_l30 = src0->name && strstr(src0->name, "blk.30.ffn_gate");
    bool is_ffn_gate_l31 = src0->name && strstr(src0->name, "blk.31.ffn_gate");

    // Check layer 30 first to compare
    if (g_ggml_sycl_tp_debug && is_ffn_gate_l30 && src1_ncols == 1 && mmvq_l30_gate_dbg++ < 3) {
        struct block_q4_0_l30 {
            sycl::half d;
            uint8_t qs[16];
        };
        block_q4_0_l30 wblk;
        stream->memcpy(&wblk, src0_dd_i, sizeof(wblk)).wait();
        float wd = static_cast<float>(wblk.d);
        int w0 = (wblk.qs[0] & 0xF) - 8;
        int w1 = (wblk.qs[0] >> 4) - 8;
        fprintf(stderr, "TP DEBUG MMVQ L30 Q4_0 weight: ptr=%p d=%f raw[0..1]=[%d,%d] -> deq=[%f,%f]\n",
                (void*)src0_dd_i, wd, w0, w1, w0 * wd, w1 * wd);
    }
    if (g_ggml_sycl_tp_debug && is_ffn_gate_l31 && src1_ncols == 1 && mmvq_l31_gate_dbg++ < 5) {
        fprintf(stderr, "TP DEBUG MMVQ L31 FFN_GATE batch=1: ne00=%lld (K), row_diff=%lld (N), src1_ddq_i=%p\n",
                (long long)ne00, (long long)row_diff, (void*)src1_ddq_i);

        // Read first Q8_1 block (ffn_norm output quantized)
        struct block_q8_1_l31 {
            sycl::half d;
            sycl::half s;
            int8_t qs[32];
        };
        block_q8_1_l31 blk;
        stream->memcpy(&blk, src1_ddq_i, sizeof(blk)).wait();
        float d_f = static_cast<float>(blk.d);
        float s_f = static_cast<float>(blk.s);
        float v0 = blk.qs[0] * d_f;
        float v1 = blk.qs[1] * d_f;
        float v2 = blk.qs[2] * d_f;
        float v3 = blk.qs[3] * d_f;

        bool d_invalid = std::isnan(d_f) || std::isinf(d_f);
        fprintf(stderr, "TP DEBUG MMVQ L31 Q8_1 input: d=%f (%s), s=%f, qs[0..3]=[%d,%d,%d,%d] -> deq=[%f,%f,%f,%f]\n",
                d_f, d_invalid ? "INVALID" : "ok", s_f,
                blk.qs[0], blk.qs[1], blk.qs[2], blk.qs[3], v0, v1, v2, v3);

        // Also check weight Q4_0 block at layer 31
        struct block_q4_0_l31 {
            sycl::half d;
            uint8_t qs[16];  // 32 x 4-bit
        };
        block_q4_0_l31 wblk;
        stream->memcpy(&wblk, src0_dd_i, sizeof(wblk)).wait();
        float wd = static_cast<float>(wblk.d);
        int w0 = (wblk.qs[0] & 0xF) - 8;
        int w1 = (wblk.qs[0] >> 4) - 8;
        bool wd_invalid = std::isnan(wd) || std::isinf(wd);
        fprintf(stderr, "TP DEBUG MMVQ L31 Q4_0 weight: d=%f (%s), raw[0..1]=[%d,%d] -> deq=[%f,%f]\n",
                wd, wd_invalid ? "INVALID" : "ok", w0, w1, w0 * wd, w1 * wd);

        // DEBUG: Check if dst pointer overlaps with weight
        uintptr_t weight_start = (uintptr_t)src0_dd_i;
        uintptr_t weight_end = weight_start + (ne00 / 32) * 18 * row_diff;  // Q4_0 block size
        uintptr_t dst_start = (uintptr_t)dst_dd_i;
        uintptr_t dst_end = dst_start + row_diff * src1_ncols * sizeof(float);
        bool overlap = (dst_start < weight_end && dst_end > weight_start);
        fprintf(stderr, "TP DEBUG MMVQ L31 ptrs: weight=[%p,%p), dst=[%p,%p), overlap=%d\n",
                (void*)weight_start, (void*)weight_end, (void*)dst_start, (void*)dst_end, overlap);
    }
    if (g_ggml_sycl_tp_debug && do_mmvq_dbg) {
        fprintf(stderr, "TP DEBUG MMVQ %s: ne00=%lld (K), ne10=%lld (src1_K), row_diff=%lld (N), src1_ncols=%lld, padded=%lld, src1_ddq_i=%p\n",
                src0->name, (long long)ne00, (long long)ne10, (long long)row_diff,
                (long long)src1_ncols, (long long)src1_padded_col_size, (void*)src1_ddq_i);

        // Read and dequantize first Q8_1 block to check input values
        // Q8_1 block: ggml_half d (2 bytes), ggml_half s (2 bytes), int8_t qs[32]
        struct block_q8_1_debug {
            sycl::half d;
            sycl::half s;
            int8_t qs[32];
        };
        block_q8_1_debug blk;
        stream->memcpy(&blk, src1_ddq_i, sizeof(blk)).wait();
        // Dequantize first 4 values: val[i] = qs[i] * d
        float d_f = static_cast<float>(blk.d);
        float s_f = static_cast<float>(blk.s);
        float v0 = blk.qs[0] * d_f;
        float v1 = blk.qs[1] * d_f;
        float v2 = blk.qs[2] * d_f;
        float v3 = blk.qs[3] * d_f;
        fprintf(stderr, "TP DEBUG MMVQ src1 (attn_out quantized): d=%f, s=%f, qs[0..3]=[%d,%d,%d,%d] -> deq=[%f,%f,%f,%f]\n",
                d_f, s_f, blk.qs[0], blk.qs[1], blk.qs[2], blk.qs[3], v0, v1, v2, v3);

        // Also read some middle values from quantized input (block 32 = position 1024-1055)
        block_q8_1_debug blk_mid;
        stream->memcpy(&blk_mid, src1_ddq_i + 32*sizeof(blk_mid), sizeof(blk_mid)).wait();
        float d_mid = static_cast<float>(blk_mid.d);
        fprintf(stderr, "TP DEBUG MMVQ src1 middle (pos 1024): d=%f, qs[0..3]=[%d,%d,%d,%d] -> deq=[%f,%f,%f,%f]\n",
                d_mid, blk_mid.qs[0], blk_mid.qs[1], blk_mid.qs[2], blk_mid.qs[3],
                blk_mid.qs[0]*d_mid, blk_mid.qs[1]*d_mid, blk_mid.qs[2]*d_mid, blk_mid.qs[3]*d_mid);
    }

    int id;
    SYCL_CHECK(CHECK_TRY_ERROR(id = get_current_device_id()));

    // DEBUG: Also print device ID and weight data for layer 0 attn_output
    static int mmvq_dev_dbg = 0;
    if (g_ggml_sycl_tp_debug && mmvq_dev_dbg++ < 10 && src0->name && strstr(src0->name, "blk.0.attn_output")) {
        fprintf(stderr, "TP DEBUG MMVQ device=%d for %s, src0_dd_i=%p, ne00=%lld, row_diff=%lld\n",
                id, src0->name, (void*)src0_dd_i, (long long)ne00, (long long)row_diff);

        // Read first Q4_0 block from weight (output row 0, first 32 input elements)
        struct block_q4_0_debug {
            sycl::half d;
            uint8_t qs[16];  // 32 x 4-bit values packed
        };
        block_q4_0_debug w_blk0, w_blk64;
        stream->memcpy(&w_blk0, src0_dd_i, sizeof(w_blk0)).wait();
        // Also read block 64 (second row of output, position [1,0]) for TP mode
        // Block index 64 = ne00/32 blocks per row = 64 for TP, 128 for single
        size_t blocks_per_row = ne00 / 32;
        stream->memcpy(&w_blk64, src0_dd_i + blocks_per_row * sizeof(w_blk0), sizeof(w_blk64)).wait();

        float d0 = static_cast<float>(w_blk0.d);
        float d64 = static_cast<float>(w_blk64.d);
        // Dequantize first 4 values: val[i] = (qs[i/2] >> (4*(i%2)) & 0xF) - 8) * d
        int v0 = (w_blk0.qs[0] & 0xF) - 8;
        int v1 = (w_blk0.qs[0] >> 4) - 8;
        int v2 = (w_blk0.qs[1] & 0xF) - 8;
        int v3 = (w_blk0.qs[1] >> 4) - 8;
        fprintf(stderr, "TP DEBUG MMVQ weight row0 blk0: d=%f, qs[0-1]=0x%02x%02x, vals=[%d,%d,%d,%d] -> deq=[%f,%f,%f,%f]\n",
                d0, w_blk0.qs[0], w_blk0.qs[1], v0, v1, v2, v3, v0*d0, v1*d0, v2*d0, v3*d0);

        int v0_64 = (w_blk64.qs[0] & 0xF) - 8;
        int v1_64 = (w_blk64.qs[0] >> 4) - 8;
        fprintf(stderr, "TP DEBUG MMVQ weight row1 blk0: d=%f, qs[0]=0x%02x, vals=[%d,%d] -> deq=[%f,%f]\n",
                d64, w_blk64.qs[0], v0_64, v1_64, v0_64*d64, v1_64*d64);
    }

    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;
    // the main device has a larger memory buffer to hold the results from all GPUs
    // nrows_dst == nrows of the matrix that the kernel writes into

    for (int i = 0; i < src1_ncols; i++) {
        const size_t src1_ddq_i_offset = i * src1_padded_col_size * q8_1_ts / q8_1_bs;
        const char * src1_ddq_i_bs     = src1_ddq_i + src1_ddq_i_offset;
        float *      dst_dd_i_bs       = dst_dd_i + i * dst->ne[0];
        switch (src0->type) {
            case GGML_TYPE_Q4_0:
                {
                    // Use src0->extra for reorder check since src0 is the actual weight tensor being used
                    // This is important for TP where we may pass different tensors than dst->src[0]
                    auto * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;
                    bool use_reorder = src0_extra && src0_extra->optimized_feature.reorder;

                    // For TP: DISABLE reorder completely to rule out issues
                    if (src0_extra && src0_extra->tp_sharded) {
                        use_reorder = false;
                    }

                    if (use_reorder) {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    } else {
                        GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl\n");
                        mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                }
                break;
            case GGML_TYPE_Q4_1:
                mul_mat_vec_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q5_0:
                mul_mat_vec_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q5_1:
                mul_mat_vec_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q8_0:
                mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_MXFP4:
                mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q2_K:
                mul_mat_vec_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q3_K:
                mul_mat_vec_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q4_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl\n");
                    reorder_mul_mat_vec_q4_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                } else {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl\n");
                    mul_mat_vec_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q5_K:
                mul_mat_vec_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_Q6_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl\n");
                    reorder_mul_mat_vec_q6_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                } else {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_k_q8_1_sycl\n");
                    mul_mat_vec_q6_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_IQ1_S:
                mul_mat_vec_iq1_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ1_M:
                mul_mat_vec_iq1_m_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_XXS:
                mul_mat_vec_iq2_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_XS:
                mul_mat_vec_iq2_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ2_S:
                mul_mat_vec_iq2_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ3_XXS:
                mul_mat_vec_iq3_xxs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ3_S:
                mul_mat_vec_iq3_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ4_NL:
                mul_mat_vec_iq4_nl_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ4_XS:
                mul_mat_vec_iq4_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            default:
                GGML_ABORT("fatal error");
        }

        // DEBUG: Check output values for attn_output after kernel (layer 0 only)
        // Controlled by GGML_SYCL_TP_DEBUG environment variable
        static int mmvq_out_dbg = 0;
        if (g_ggml_sycl_tp_debug && mmvq_out_dbg++ < 10 && src0->name && strstr(src0->name, "blk.0.attn_output")) {
            stream->wait();  // Wait for kernel to complete
            float out_vals[8];
            stream->memcpy(out_vals, dst_dd_i_bs, 8*sizeof(float)).wait();
            fprintf(stderr, "TP DEBUG MMVQ output device=%d %s col=%d dst[0..7]=[%f, %f, %f, %f, %f, %f, %f, %f]\n",
                    id, src0->name, i, out_vals[0], out_vals[1], out_vals[2], out_vals[3],
                    out_vals[4], out_vals[5], out_vals[6], out_vals[7]);
        }

        // DEBUG: Check FFN down output for layer 0 (works in both TP and single GPU)
        // Controlled by GGML_SYCL_TP_DEBUG environment variable
        static int ffn_down_dbg = 0;
        if (g_ggml_sycl_tp_debug && ffn_down_dbg++ < 5 && src0->name && strstr(src0->name, "blk.0.ffn_down")) {
            stream->wait();
            float out_vals[8];
            stream->memcpy(out_vals, dst_dd_i_bs, 8*sizeof(float)).wait();
            fprintf(stderr, "DEBUG FFN_DOWN layer0 device=%d col=%d ne00=%lld row_diff=%lld dst[0..7]=[%f, %f, %f, %f, %f, %f, %f, %f]\n",
                    id, (int)i, (long long)ne00, (long long)row_diff, out_vals[0], out_vals[1], out_vals[2], out_vals[3],
                    out_vals[4], out_vals[5], out_vals[6], out_vals[7]);
        }
    }
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(ctx);
}
