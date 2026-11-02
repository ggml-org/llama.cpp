#include "mmvq.hpp"

#include "ggml.h"
#include "common.hpp"
#include "element_wise.hpp"
#include "quants.hpp"
#include "vecdotq.hpp"

#ifdef GGML_SYCL_MMVQ_HAS_XMX
#    include "esimd.hpp"
#endif // GGML_SYCL_MMVQ_HAS_XMX

// vec_dot_q_sycl_t adapters for the IQ vec_dots that take their codebook tables as extra
// arguments: bind the constant tables here (as vec_dot_iq2_s_q8_1 / vec_dot_iq1_m_q8_1 already do
// internally) so they can be used as template arguments of mul_mat_vec_q_moe.
static __dpct_inline__ float vec_dot_iq2_xxs_q8_1_moe(const void * __restrict__ vbq,
                                                      const block_q8_1 * __restrict__ bq8_1,
                                                      const int & iqs) {
    return vec_dot_iq2_xxs_q8_1(vbq, bq8_1, iqs, iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs);
}

static __dpct_inline__ float vec_dot_iq2_xs_q8_1_moe(const void * __restrict__ vbq,
                                                     const block_q8_1 * __restrict__ bq8_1,
                                                     const int & iqs) {
    return vec_dot_iq2_xs_q8_1(vbq, bq8_1, iqs, iq2xs_grid);
}

static __dpct_inline__ float vec_dot_iq3_xxs_q8_1_moe(const void * __restrict__ vbq,
                                                      const block_q8_1 * __restrict__ bq8_1,
                                                      const int & iqs) {
    return vec_dot_iq3_xxs_q8_1(vbq, bq8_1, iqs, iq3xxs_grid);
}

static __dpct_inline__ float vec_dot_iq3_s_q8_1_moe(const void * __restrict__ vbq,
                                                    const block_q8_1 * __restrict__ bq8_1,
                                                    const int & iqs) {
    return vec_dot_iq3_s_q8_1(vbq, bq8_1, iqs, iq3s_grid);
}

static __dpct_inline__ float vec_dot_iq1_s_q8_1_moe(const void * __restrict__ vbq,
                                                    const block_q8_1 * __restrict__ bq8_1,
                                                    const int & iqs) {
    return vec_dot_iq1_s_q8_1(vbq, bq8_1, iqs, iq1s_grid_gpu);
}

// Minimum weight-row count at which the Q4_K multi-column MMVQ kernel handles two output rows per
// subgroup (rows_per_sg == 2) instead of one, when ncols_dst == 2.
//
// Pairing rows lets a subgroup load each activation block once and apply it to two rows, at the cost
// of halving the number of subgroups in the launch. With only two destination columns there is too
// little work per row to hide that loss of parallelism, so pairing only pays off once there are
// enough rows to keep the device occupied. This is a measured performance crossover, not a
// correctness or hardware limit - both variants compute the same result for any nrows.
//
// Derived on Intel Arc Pro B70 with `test-backend-ops perf -o MUL_MAT` (Q4_K, ncols_dst == 2),
// sweeping nrows over 5120..6912 at ncols 17408 and 19968: one row per subgroup was up to 9% faster
// below the crossover, two rows per subgroup 8-15% faster above it, and the crossover fell inside
// (6144, 6272] for both ncols with no measurable ncols dependence. A later 32-row granularity sweep
// narrowed it to (6144, 6176], so 6272 is a conservative gate rather than the exact crossover.
// ncols_dst >= 3 amortizes the activation loads over more columns and is faster with two rows at
// every row count, so it does not consult this threshold.
static constexpr int Q4_K_MMVQ_ROW_PAIR_MIN_NROWS = 6272;

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

// With has_fusion, `vgate` is a second weight matrix sharing vx's shape, stride and reorder
// layout: one pass computes both row dot products and the epilogue writes glu(gate, up).
template <typename reorder_vec_dot_q_sycl, int ncols_dst, bool has_fusion = false, int rows_per_sg = 1,
          bool shared_weights = reorder_vec_dot_shared_weights<reorder_vec_dot_q_sycl::gtype>::value>
static void mul_mat_vec_q_reorder_ncols(const void * __restrict__ vx, const void * __restrict__ vgate,
                                        const void * __restrict__ vy, float * __restrict__ dst, const int ncols,
                                        const int nrows, const int stride_col_y_bytes, const int stride_col_dst,
                                        const ggml_glu_op glu_op, const sycl::nd_item<3> & nd_item) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const auto sg           = nd_item.get_sub_group();
    const int  sg_range     = sg.get_group_linear_range();
    const int  workgroup_id = nd_item.get_group_linear_id();
    const int  sg_id        = sg.get_group_linear_id();
    const int  row0         = (workgroup_id * sg_range + sg_id) * rows_per_sg;

    // row is sub-group uniform, so this retires whole sub-groups and the collectives below
    // stay convergent
    if (row0 >= nrows) {
        return;
    }

    static_assert(rows_per_sg == 1 ||
                  reorder_vec_dot_shared_activations<reorder_vec_dot_q_sycl::gtype>::value);

    const int     blocks_per_row              = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup         = ceil_div(block_traits::vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / block_traits::vdr_mmvq;
    const int     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    float partial_sum[ncols_dst][rows_per_sg] = {};
    // sized 1 rather than 0 when unused: zero-length arrays are not standard C++, and the
    // array is dead and eliminated in that case
    [[maybe_unused]] float partial_gate[has_fusion ? ncols_dst : 1][has_fusion ? rows_per_sg : 1] = {};
    for (int i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        const int  iby       = i * block_type::block_to_q8_1_ratio();

#pragma unroll
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            const int iqs = elem + block_traits::vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);

            if constexpr (rows_per_sg > 1) {
                typename reorder_vec_dot_q_sycl::weights wx[rows_per_sg];
                [[maybe_unused]] typename reorder_vec_dot_q_sycl::weights wg[rows_per_sg];
#pragma unroll
                for (int r = 0; r < rows_per_sg; ++r) {
                    const int row = sycl::min(row0 + r, nrows - 1);
                    const int ibx = row * blocks_per_row + i;
                    const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
                    const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
                    wx[r] = reorder_vec_dot_q_sycl::load(vx, bx_offset, d_offset, iqs);
                    if constexpr (has_fusion) {
                        wg[r] = reorder_vec_dot_q_sycl::load(vgate, bx_offset, d_offset, iqs);
                    }
                }
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const char        * vy_j           = (const char *) vy + j * stride_col_y_bytes;
                    const int8_t      * q8_1_quant_ptr = (const int8_t *) vy_j + iby * QK8_1;
                    const sycl::half2 * q8_1_ds_ptr =
                        (const sycl::half2 *) (vy_j + ncols + iby * sizeof(sycl::half2));
                    const auto a = reorder_vec_dot_q_sycl::load_activations(q8_1_quant_ptr, q8_1_ds_ptr, iqs);
#pragma unroll
                    for (int r = 0; r < rows_per_sg; ++r) {
                        partial_sum[j][r] += reorder_vec_dot_q_sycl::apply(wx[r], a);
                        if constexpr (has_fusion) {
                            partial_gate[j][r] += reorder_vec_dot_q_sycl::apply(wg[r], a);
                        }
                    }
                }
            } else if constexpr (shared_weights) {
                const int ibx = row0 * blocks_per_row + i;
                const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
                const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
                const auto wx = reorder_vec_dot_q_sycl::load(vx, bx_offset, d_offset, iqs);
                if constexpr (has_fusion) {
                    const auto wg = reorder_vec_dot_q_sycl::load(vgate, bx_offset, d_offset, iqs);

#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        const char        * vy_j           = (const char *) vy + j * stride_col_y_bytes;
                        const int8_t      * q8_1_quant_ptr = (const int8_t *) vy_j + iby * QK8_1;
                        const sycl::half2 * q8_1_ds_ptr =
                            (const sycl::half2 *) (vy_j + ncols + iby * sizeof(sycl::half2));

                        // up and gate share the activation, so load it once and apply it twice
                        const auto a = reorder_vec_dot_q_sycl::load_activations(q8_1_quant_ptr, q8_1_ds_ptr, iqs);

                        partial_sum[j][0] += reorder_vec_dot_q_sycl::apply(wx, a);
                        partial_gate[j][0] += reorder_vec_dot_q_sycl::apply(wg, a);
                    }
                } else {
#pragma unroll
                    for (int j = 0; j < ncols_dst; ++j) {
                        const char        * vy_j           = (const char *) vy + j * stride_col_y_bytes;
                        const int8_t      * q8_1_quant_ptr = (const int8_t *) vy_j + iby * QK8_1;
                        const sycl::half2 * q8_1_ds_ptr =
                            (const sycl::half2 *) (vy_j + ncols + iby * sizeof(sycl::half2));

                        partial_sum[j][0] += reorder_vec_dot_q_sycl::dot(wx, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                    }
                }
            } else {
                const int ibx = row0 * blocks_per_row + i;
                const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
                const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
#pragma unroll
                for (int j = 0; j < ncols_dst; ++j) {
                    const char        * vy_j           = (const char *) vy + j * stride_col_y_bytes;
                    const int8_t      * q8_1_quant_ptr = (const int8_t *) vy_j + iby * QK8_1;
                    const sycl::half2 * q8_1_ds_ptr =
                        (const sycl::half2 *) (vy_j + ncols + iby * sizeof(sycl::half2));

                    partial_sum[j][0] +=
                        reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);

                    if constexpr (has_fusion) {
                        partial_gate[j][0] +=
                            reorder_vec_dot_q_sycl()(vgate, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int r = 0; r < rows_per_sg; ++r) {
            float sum = sycl::reduce_over_group(nd_item.get_sub_group(), partial_sum[j][r], std::plus<>());

            if constexpr (has_fusion) {
                const float gate = sycl::reduce_over_group(nd_item.get_sub_group(), partial_gate[j][r], std::plus<>());

                // uniform across the launch; the launcher only instantiates SWIGLU and GEGLU
                sum *= glu_op == GGML_GLU_OP_SWIGLU ? op_silu(gate) : op_gelu(gate);
            }

            if (sg.leader() && row0 + r < nrows) {
                dst[j * stride_col_dst + row0 + r] = sum;
            }
        }
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

template <int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_sycl_t vec_dot_q_sycl, int ncols_dst>
static void mul_mat_vec_q_ncols(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float * __restrict__ dst,
        const int ncols,
        const int nrows,
        const int stride_col_y,
        const int stride_col_dst,
        const sycl::nd_item<3> & item_ct1) {

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1)
                  + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    const int blocks_per_row = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;

    // partial sums: one per output column
    float tmp[ncols_dst] = {0.0f};

    const block_q_t  * x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

    for (int i = item_ct1.get_local_id(2) / (qi / vdr);
         i < blocks_per_row;
         i += blocks_per_warp) {

        const int ibx = row * blocks_per_row + i;
        const int iby = i * (qk / QK8_1);

        // read weight block once, dot against all columns
        for (size_t elem = 0; elem < qi / vdr; elem += WARP_SIZE) {
            const int iqs = elem + vdr * (item_ct1.get_local_id(2) % (qi / vdr));

#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                tmp[j] += vec_dot_q_sycl(&x[ibx], &y[j * stride_col_y + iby], iqs);
            }
        }
    }

    // reduce within subgroup
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            tmp[j] += dpct::permute_sub_group_by_xor(
                item_ct1.get_sub_group(), tmp[j], mask);
        }
    }

    if (item_ct1.get_local_id(2) == 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            dst[j * stride_col_dst + row] = tmp[j];
        }
    }
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl, int assert_qk>
static void launch_mul_mat_vec_q(const void * vx, const void * vy, float * dst,
                                 const int ncols, const int nrows, const int ncols_dst,
                                 const int stride_col_y, const int stride_col_dst,
                                 dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % assert_qk == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    const auto launch = [&](auto ncols_tag) {
        constexpr int N = decltype(ncols_tag)::value;
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 if constexpr (N == 1) {
                                     mul_mat_vec_q<qk, qi, block_q_t, vdr, vec_dot_q_sycl>(
                                         vx, vy, dst, ncols, nrows, item_ct1);
                                 } else {
                                     mul_mat_vec_q_ncols<qk, qi, block_q_t, vdr, vec_dot_q_sycl, N>(
                                         vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
                                 }
                             });
        });
    };

    switch (ncols_dst) {
        case 1: launch(std::integral_constant<int, 1>{}); break;
        case 2: launch(std::integral_constant<int, 2>{}); break;
        case 3: launch(std::integral_constant<int, 3>{}); break;
        case 4: launch(std::integral_constant<int, 4>{}); break;
        case 5: launch(std::integral_constant<int, 5>{}); break;
        case 6: launch(std::integral_constant<int, 6>{}); break;
        case 7: launch(std::integral_constant<int, 7>{}); break;
        case 8: launch(std::integral_constant<int, 8>{}); break;
        default: GGML_ABORT("unsupported ncols_dst=%d", ncols_dst);
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

        tmp += vec_dot_iq2_xs_q8_1(&x[ibx], &y[iby], iqs, iq2xs_grid);
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

        tmp += vec_dot_iq3_xxs_q8_1(&x[ibx], &y[iby], iqs, iq3xxs_grid);
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

static __dpct_inline__ float vec_dot_iq3_s_q8_1_adapter(
        const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & iqs) {
    return vec_dot_iq3_s_q8_1(vbq, bq8_1, iqs, iq3s_grid);
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
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK4_0 == 0);
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q4_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q4_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q4_0 reorder multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q4_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols, const int nrows,
                                       dpct::queue_ptr stream) {
    launch_mul_mat_vec_q<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1, QK4_0>(
        vx, vy, dst, ncols, nrows, 1, 0, 0, stream);
}


static void mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    if (ncols_dst < 1 || ncols_dst > 8) {
        GGML_ABORT("unsupported ncols_dst=%d for Q4_0 multi-col MMVQ", ncols_dst);
    }
    launch_mul_mat_vec_q<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1, QK4_0>(
        vx, vy, dst, ncols, nrows, ncols_dst, stride_col_y, stride_col_dst, stream);
}


static void mul_mat_vec_q4_1_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    launch_mul_mat_vec_q<QK4_0, QI4_1, block_q4_1,
                                      VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1, QK4_1>(
        vx, vy, dst, ncols, nrows, 1, 0, 0, stream);
}


static void mul_mat_vec_q4_1_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    if (ncols_dst < 1 || ncols_dst > 8) {
        GGML_ABORT("unsupported ncols_dst=%d for Q4_1 multi-col MMVQ", ncols_dst);
    }
    launch_mul_mat_vec_q<QK4_0, QI4_1, block_q4_1,
                                      VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1, QK4_1>(
        vx, vy, dst, ncols, nrows, ncols_dst, stride_col_y, stride_col_dst, stream);
}


static void mul_mat_vec_mxfp4_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols, const int nrows,
                                        dpct::queue_ptr stream) {
    launch_mul_mat_vec_q<QK_MXFP4, QI_MXFP4, block_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1, QK_MXFP4>(
        vx, vy, dst, ncols, nrows, 1, 0, 0, stream);
}


static void mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    if (ncols_dst < 1 || ncols_dst > 8) {
        GGML_ABORT("unsupported ncols_dst=%d for MXFP4 multi-col MMVQ", ncols_dst);
    }
    launch_mul_mat_vec_q<QK_MXFP4, QI_MXFP4, block_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1, QK_MXFP4>(
        vx, vy, dst, ncols, nrows, ncols_dst, stride_col_y, stride_col_dst, stream);
}


static void mul_mat_vec_nvfp4_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols, const int nrows,
                                        dpct::queue_ptr stream) {
    launch_mul_mat_vec_q<QK_NVFP4, QI_NVFP4, block_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1, QK_NVFP4>(
        vx, vy, dst, ncols, nrows, 1, 0, 0, stream);
}


static void mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    if (ncols_dst < 1 || ncols_dst > 8) {
        GGML_ABORT("unsupported ncols_dst=%d for NVFP4 multi-col MMVQ", ncols_dst);
    }
    launch_mul_mat_vec_q<QK_NVFP4, QI_NVFP4, block_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1, QK_NVFP4>(
        vx, vy, dst, ncols, nrows, ncols_dst, stride_col_y, stride_col_dst, stream);
}


static void mul_mat_vec_q5_0_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    launch_mul_mat_vec_q<QK5_0, QI5_0, block_q5_0,
                                      VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1, QK5_0>(
        vx, vy, dst, ncols, nrows, 1, 0, 0, stream);
}


static void mul_mat_vec_q5_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    if (ncols_dst < 1 || ncols_dst > 8) {
        GGML_ABORT("unsupported ncols_dst=%d for Q5_0 multi-col MMVQ", ncols_dst);
    }
    launch_mul_mat_vec_q<QK5_0, QI5_0, block_q5_0,
                                      VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1, QK5_0>(
        vx, vy, dst, ncols, nrows, ncols_dst, stride_col_y, stride_col_dst, stream);
}


static void mul_mat_vec_q5_1_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    launch_mul_mat_vec_q<QK5_1, QI5_1, block_q5_1,
                                      VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1, QK5_1>(
        vx, vy, dst, ncols, nrows, 1, 0, 0, stream);
}


static void mul_mat_vec_q5_1_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    if (ncols_dst < 1 || ncols_dst > 8) {
        GGML_ABORT("unsupported ncols_dst=%d for Q5_1 multi-col MMVQ", ncols_dst);
    }
    launch_mul_mat_vec_q<QK5_1, QI5_1, block_q5_1,
                                      VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1, QK5_1>(
        vx, vy, dst, ncols, nrows, ncols_dst, stride_col_y, stride_col_dst, stream);
}


static void reorder_mul_mat_vec_q8_0_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                                    const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    if (g_ggml_sycl_mmvq_wide) {
        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                             [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 mul_mat_vec_q_reorder<reorder_vec_dot_q8_0_wide>(vx, vy, dst, ncols, nrows, nd_item);
                             });
        });
        return;
    }

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK8_0 == 0);
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q8_0_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q8_0_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q8_0 reorder multi-col MMVQ", ncols_dst);
    }
}

// MXFP4 reorder GEMV. Only MoE expert slices are reordered (opt_for_reorder_id); these dense entry
// points serve per-expert ggml_sycl_mul_mat calls from multi-token MUL_MAT_ID after that reorder.
static void reorder_mul_mat_vec_mxfp4_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                                const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    constexpr size_t     num_subgroups = WARP_SIZE;
    const int            block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>>(vx, vy, dst, ncols, nrows,
                                                                                            nd_item);
                         });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_MXFP4 == 0);
    constexpr size_t     num_subgroups = WARP_SIZE;
    const int            block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_mxfp4_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_mxfp4_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for MXFP4 reorder multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q8_0_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    launch_mul_mat_vec_q<QK8_0, QI8_0, block_q8_0,
                                      VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1, QK8_0>(
        vx, vy, dst, ncols, nrows, 1, 0, 0, stream);
}


static void mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    if (ncols_dst < 1 || ncols_dst > 8) {
        GGML_ABORT("unsupported ncols_dst=%d for Q8_0 multi-col MMVQ", ncols_dst);
    }
    launch_mul_mat_vec_q<QK8_0, QI8_0, block_q8_0,
                                      VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1, QK8_0>(
        vx, vy, dst, ncols, nrows, ncols_dst, stride_col_y, stride_col_dst, stream);
}


static void mul_mat_vec_q1_0_q8_1_sycl(const void * vx, const void * vy,
                                       float * dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    launch_mul_mat_vec_q<QK1_0, QI1_0, block_q1_0,
                              VDR_Q1_0_Q8_1_MMVQ, vec_dot_q1_0_q8_1, QK1_0>(
        vx, vy, dst, ncols, nrows, 1, 0, 0, stream);
}


static void mul_mat_vec_q1_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    if (ncols_dst < 1 || ncols_dst > 8) {
        GGML_ABORT("unsupported ncols_dst=%d for Q1_0 multi-col MMVQ", ncols_dst);
    }
    launch_mul_mat_vec_q<QK1_0, QI1_0, block_q1_0,
                              VDR_Q1_0_Q8_1_MMVQ, vec_dot_q1_0_q8_1, QK1_0>(
        vx, vy, dst, ncols, nrows, ncols_dst, stride_col_y, stride_col_dst, stream);
}


static void mul_mat_vec_q2_0_q8_1_sycl(const void * vx, const void * vy,
                                       float * dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    launch_mul_mat_vec_q<QK2_0, QI2_0, block_q2_0,
                              VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1, QK2_0>(
        vx, vy, dst, ncols, nrows, 1, 0, 0, stream);
}


static void mul_mat_vec_q2_0_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    if (ncols_dst < 1 || ncols_dst > 8) {
        GGML_ABORT("unsupported ncols_dst=%d for Q2_0 multi-col MMVQ", ncols_dst);
    }
    launch_mul_mat_vec_q<QK2_0, QI2_0, block_q2_0,
                              VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1, QK2_0>(
        vx, vy, dst, ncols, nrows, ncols_dst, stride_col_y, stride_col_dst, stream);
}


static void mul_mat_vec_q2_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    launch_mul_mat_vec_q<QK_K, QI2_K, block_q2_K,
                                      VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1, QK_K>(
        vx, vy, dst, ncols, nrows, 1, 0, 0, stream);
}


static void mul_mat_vec_q2_K_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    if (ncols_dst < 1 || ncols_dst > 8) {
        GGML_ABORT("unsupported ncols_dst=%d for Q2_K multi-col MMVQ", ncols_dst);
    }
    launch_mul_mat_vec_q<QK_K, QI2_K, block_q2_K,
                                      VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1, QK_K>(
        vx, vy, dst, ncols, nrows, ncols_dst, stride_col_y, stride_col_dst, stream);
}


static void reorder_mul_mat_vec_q2_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                               const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q2_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q2_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q2_K reorder multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q3_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    launch_mul_mat_vec_q<QK_K, QI3_K, block_q3_K,
                                      VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1, QK_K>(
        vx, vy, dst, ncols, nrows, 1, 0, 0, stream);
}


static void reorder_mul_mat_vec_q3_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                               const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K>>(vx, vy, dst, ncols, nrows,
                                                                                           nd_item);
                         });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q3_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q3_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q3_K reorder multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q3_K_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    if (ncols_dst < 1 || ncols_dst > 8) {
        GGML_ABORT("unsupported ncols_dst=%d for Q3_K multi-col MMVQ", ncols_dst);
    }
    launch_mul_mat_vec_q<QK_K, QI3_K, block_q3_K,
                                      VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1, QK_K>(
        vx, vy, dst, ncols, nrows, ncols_dst, stride_col_y, stride_col_dst, stream);
}



static void mul_mat_vec_q4_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    launch_mul_mat_vec_q<QK_K, QI4_K, block_q4_K,
                                      VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1, QK_K>(
        vx, vy, dst, ncols, nrows, 1, 0, 0, stream);
}


static void mul_mat_vec_q4_K_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    if (ncols_dst < 1 || ncols_dst > 8) {
        GGML_ABORT("unsupported ncols_dst=%d for Q4_K multi-col MMVQ", ncols_dst);
    }
    launch_mul_mat_vec_q<QK_K, QI4_K, block_q4_K,
                                      VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1, QK_K>(
        vx, vy, dst, ncols, nrows, ncols_dst, stride_col_y, stride_col_dst, stream);
}

// One lane's Q4_K partial for an already loaded weight quad, so multi-column kernels load each weight once.
static inline float mul_mat_vec_q4_K_reorder_wide_dot(const sycl::int4 & vq, const uint8_t * sc, const uint8_t * m,
                                 const sycl::float2 & dm4f, int half, const sycl::int4 & u0, const sycl::int4 & u1,
                                 float d8_0, float d8_1, float s8_0, float s8_1) {
    float sumf_d = 0.0f;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int vp  = vq[p];
        const int vi0 = (vp >> 0) & 0x0F0F0F0F;
        const int vi1 = (vp >> 4) & 0x0F0F0F0F;
        sumf_d += d8_0 * (dpct::dp4a(vi0, u0[p], 0) * sc[0]);
        sumf_d += d8_1 * (dpct::dp4a(vi1, u1[p], 0) * sc[1]);
    }
    const float sumf_m = (half == 0) ? (s8_0 * m[0] + s8_1 * m[1]) : 0.0f;
    return dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}

// Per-lane Q4_K partial for the wide reorder kernels below. Lane c (0..7) loads int4 c of the 128-byte block.
// Only even lanes add the min term, so it counts once per scale group.
static inline float mul_mat_vec_q4_K_reorder_wide_lane_partial(const uint8_t * qs, const uint16_t * scales, const sycl::half2 * dms,
                                          int c, const sycl::int4 & u0, const sycl::int4 & u1, float d8_0,
                                          float d8_1, float s8_0, float s8_1) {
    const int g    = c >> 1;
    const int half = c & 1;

    const sycl::int4 vq = *(reinterpret_cast<const sycl::int4 *>(qs) + c);

    // same scale and min extraction as the scalar path
    uint16_t  aux[2];
    const int j = g;
    if (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } else {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }
    const uint8_t * sc = reinterpret_cast<const uint8_t *>(aux);
    const uint8_t * m  = sc + 2;

    const sycl::float2 dm4f = (*dms).convert<float, sycl::rounding_mode::automatic>();

    return mul_mat_vec_q4_K_reorder_wide_dot(vq, sc, m, dm4f, half, u0, u1, d8_0, d8_1, s8_0, s8_1);
}

// Q4_K reorder mat-vec for one column with 128-bit weight loads.
// 8 lanes cover one 128-byte block and even and odd lanes take the two nibble halves, so no shuffle is needed.
static void mul_mat_vec_q4_K_reorder_wide(const void * __restrict__ vx, const void * __restrict__ vy,
                                          float * __restrict__ dst, const int ncols, const int nrows,
                                          const sycl::nd_item<3> & nd_item) {
    using q4_k_block = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_K>;

    const auto sg       = nd_item.get_sub_group();
    const int  sg_range = sg.get_group_linear_range();
    const int  row      = nd_item.get_group_linear_id() * sg_range + sg.get_group_linear_id();
    if (row >= nrows) {
        return;
    }

    const int blocks_per_row  = ncols / QK_K;
    const int nblocks         = nrows * blocks_per_row;
    const int lane            = sg.get_local_linear_id();
    const int blocks_per_wave = WARP_SIZE / 8;   // 8 lanes cover one 128-byte block
    const int lb              = lane / 8;         // block within the wave
    const int c               = lane % 8;         // int4 index within the block (0..7)
    const int g               = c >> 1;           // scale group (0..3)
    const int half            = c & 1;            // 0 -> v[0] quad, 1 -> v[1] quad
    const int uoff            = half ? 4 : 0;     // int offset into each q8_1 sub-block

    const uint8_t *     base = static_cast<const uint8_t *>(vx);
    const sycl::half2 * vyds = reinterpret_cast<const sycl::half2 *>(static_cast<const char *>(vy) + ncols);
    const int8_t *      vy8  = static_cast<const int8_t *>(vy);

    float partial = 0.0f;

    for (int blk = lb; blk < blocks_per_row; blk += blocks_per_wave) {
        const int  ibx     = row * blocks_per_row + blk;
        const auto ibx_off = q4_k_block::get_block_offset(ibx, nblocks);
        const auto d_off   = q4_k_block::get_d_offset(nrows, ncols, ibx);

        const int         gq8 = blk * (QK_K / QK8_1) + 2 * g;  // global q8_1 sub-block (i=0); i=1 -> +1
        const sycl::half2 ds0 = vyds[gq8 + 0];
        const sycl::half2 ds1 = vyds[gq8 + 1];
        // one activation int4 per q8_1 sub-block
        const sycl::int4  u0  = *reinterpret_cast<const sycl::int4 *>(reinterpret_cast<const int *>(vy8 + (gq8 + 0) * QK8_1) + uoff);
        const sycl::int4  u1  = *reinterpret_cast<const sycl::int4 *>(reinterpret_cast<const int *>(vy8 + (gq8 + 1) * QK8_1) + uoff);

        partial += mul_mat_vec_q4_K_reorder_wide_lane_partial(base + ibx_off.first, reinterpret_cast<const uint16_t *>(base + d_off.first),
                                         reinterpret_cast<const sycl::half2 *>(base + d_off.second), c, u0, u1,
                                         ds0[0], ds1[0], ds0[1], ds1[1]);
    }

    const float row_sum = sycl::reduce_over_group(sg, partial, std::plus<>());
    if (sg.leader()) {
        dst[row] = row_sum;
    }
}

static void reorder_mul_mat_vec_q4_k_q8_1_wide_sycl(const void * vx, const void * vy, float * dst,
                                                    const int ncols, const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t     num_subgroups = WARP_SIZE;
    const int            block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q4_K_reorder_wide(vx, vy, dst, ncols, nrows, nd_item);
                         });
    });
}

// SwiGLU-fused version of mul_mat_vec_q4_K_reorder_wide. It loads the up and gate weights and writes silu(gate) * up.
static void mul_mat_vec_q4_K_reorder_wide_glu(const void * __restrict__ vx, const void * __restrict__ vgate,
                                              const void * __restrict__ vy, float * __restrict__ dst,
                                              const int ncols, const int nrows, const sycl::nd_item<3> & nd_item) {
    using q4_k_block = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_K>;

    const auto sg       = nd_item.get_sub_group();
    const int  sg_range = sg.get_group_linear_range();
    const int  row      = nd_item.get_group_linear_id() * sg_range + sg.get_group_linear_id();
    if (row >= nrows) {
        return;
    }

    const int blocks_per_row  = ncols / QK_K;
    const int nblocks         = nrows * blocks_per_row;
    const int lane            = sg.get_local_linear_id();
    const int blocks_per_wave = WARP_SIZE / 8;
    const int lb              = lane / 8;
    const int c               = lane % 8;
    const int g               = c >> 1;
    const int uoff            = (c & 1) ? 4 : 0;

    const uint8_t *     base_up   = static_cast<const uint8_t *>(vx);
    const uint8_t *     base_gate = static_cast<const uint8_t *>(vgate);
    const sycl::half2 * vyds      = reinterpret_cast<const sycl::half2 *>(static_cast<const char *>(vy) + ncols);
    const int8_t *      vy8       = static_cast<const int8_t *>(vy);

    float partial_up   = 0.0f;
    float partial_gate = 0.0f;

    for (int blk = lb; blk < blocks_per_row; blk += blocks_per_wave) {
        const int  ibx     = row * blocks_per_row + blk;
        const auto ibx_off = q4_k_block::get_block_offset(ibx, nblocks);
        const auto d_off   = q4_k_block::get_d_offset(nrows, ncols, ibx);

        const int         gq8 = blk * (QK_K / QK8_1) + 2 * g;
        const sycl::half2 ds0 = vyds[gq8 + 0];
        const sycl::half2 ds1 = vyds[gq8 + 1];
        const sycl::int4  u0  = *reinterpret_cast<const sycl::int4 *>(reinterpret_cast<const int *>(vy8 + (gq8 + 0) * QK8_1) + uoff);
        const sycl::int4  u1  = *reinterpret_cast<const sycl::int4 *>(reinterpret_cast<const int *>(vy8 + (gq8 + 1) * QK8_1) + uoff);

        partial_up += mul_mat_vec_q4_K_reorder_wide_lane_partial(base_up + ibx_off.first,
                                            reinterpret_cast<const uint16_t *>(base_up + d_off.first),
                                            reinterpret_cast<const sycl::half2 *>(base_up + d_off.second), c, u0, u1,
                                            ds0[0], ds1[0], ds0[1], ds1[1]);
        partial_gate += mul_mat_vec_q4_K_reorder_wide_lane_partial(base_gate + ibx_off.first,
                                              reinterpret_cast<const uint16_t *>(base_gate + d_off.first),
                                              reinterpret_cast<const sycl::half2 *>(base_gate + d_off.second), c, u0, u1,
                                              ds0[0], ds1[0], ds0[1], ds1[1]);
    }

    const float up_sum   = sycl::reduce_over_group(sg, partial_up, std::plus<>());
    const float gate_sum = sycl::reduce_over_group(sg, partial_gate, std::plus<>());
    if (sg.leader()) {
        const float silu = gate_sum / (1.0f + sycl::exp(-gate_sum));
        dst[row]         = silu * up_sum;
    }
}

static void reorder_mul_mat_vec_q4_k_q8_1_glu_fused_wide_sycl(const void * vx, const void * vgate, const void * vy,
                                                              float * dst, const int ncols, const int nrows,
                                                              dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t     num_subgroups = WARP_SIZE;
    const int            block_num_y   = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q4_K_reorder_wide_glu(vx, vgate, vy, dst, ncols, nrows, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q4_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
    const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(vx, vy, dst, ncols,
                                                                                            nrows, nd_item);
                            });
    });
}

// Multi-column (2..8) version of the wide Q4_K kernel. Each block's weights and scales load once for all columns.
template <int ncols_dst>
static void mul_mat_vec_q4_K_reorder_wide_ncols(const void * __restrict__ vx, const void * __restrict__ vy,
                                                float * __restrict__ dst, const int ncols, const int nrows,
                                                const int stride_col_y_bytes, const int stride_col_dst,
                                                const sycl::nd_item<3> & nd_item) {
    using q4_k_block = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_K>;

    const auto sg       = nd_item.get_sub_group();
    const int  sg_range = sg.get_group_linear_range();
    const int  row      = nd_item.get_group_linear_id() * sg_range + sg.get_group_linear_id();
    if (row >= nrows) {
        return;
    }

    const int blocks_per_row  = ncols / QK_K;
    const int nblocks         = nrows * blocks_per_row;
    const int lane            = sg.get_local_linear_id();
    const int blocks_per_wave = WARP_SIZE / 8;
    const int lb              = lane / 8;
    const int c               = lane % 8;
    const int g               = c >> 1;
    const int half            = c & 1;
    const int uoff            = half ? 4 : 0;

    const uint8_t * base = static_cast<const uint8_t *>(vx);

    float partial[ncols_dst] = { 0.0f };

    for (int blk = lb; blk < blocks_per_row; blk += blocks_per_wave) {
        const int  ibx     = row * blocks_per_row + blk;
        const auto ibx_off = q4_k_block::get_block_offset(ibx, nblocks);
        const auto d_off   = q4_k_block::get_d_offset(nrows, ncols, ibx);

        const uint8_t *     qs     = base + ibx_off.first;
        const uint16_t *    scales = reinterpret_cast<const uint16_t *>(base + d_off.first);
        const sycl::half2 * dms    = reinterpret_cast<const sycl::half2 *>(base + d_off.second);

        const sycl::int4 vq = *(reinterpret_cast<const sycl::int4 *>(qs) + c);
        uint16_t  aux[2];
        const int j = g;
        if (j < 2) {
            aux[0] = scales[j + 0] & 0x3f3f;
            aux[1] = scales[j + 2] & 0x3f3f;
        } else {
            aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }
        const uint8_t *    sc   = reinterpret_cast<const uint8_t *>(aux);
        const uint8_t *    m    = sc + 2;
        const sycl::float2 dm4f = (*dms).convert<float, sycl::rounding_mode::automatic>();

        const int gq8 = blk * (QK_K / QK8_1) + 2 * g;

#pragma unroll
        for (int col = 0; col < ncols_dst; ++col) {
            const char *        vy_j = static_cast<const char *>(vy) + col * stride_col_y_bytes;
            const int8_t *      vy8  = reinterpret_cast<const int8_t *>(vy_j);
            const sycl::half2 * vyds = reinterpret_cast<const sycl::half2 *>(vy_j + ncols);
            const sycl::half2   ds0  = vyds[gq8 + 0];
            const sycl::half2   ds1  = vyds[gq8 + 1];
            const sycl::int4    u0   = *reinterpret_cast<const sycl::int4 *>(reinterpret_cast<const int *>(vy8 + (gq8 + 0) * QK8_1) + uoff);
            const sycl::int4    u1   = *reinterpret_cast<const sycl::int4 *>(reinterpret_cast<const int *>(vy8 + (gq8 + 1) * QK8_1) + uoff);

            partial[col] += mul_mat_vec_q4_K_reorder_wide_dot(vq, sc, m, dm4f, half, u0, u1, ds0[0], ds1[0], ds0[1], ds1[1]);
        }
    }

#pragma unroll
    for (int col = 0; col < ncols_dst; ++col) {
        const float sum = sycl::reduce_over_group(sg, partial[col], std::plus<>());
        if (sg.leader()) {
            dst[col * stride_col_dst + row] = sum;
        }
    }
}


// Split-K version for narrow outputs. SPLIT subgroups share one row and reduce through SLM.
// The launcher uses it only when the row count is too small to fill the device.
template <int ncols_dst, int SPLIT>
static void mul_mat_vec_q4_K_reorder_wide_ncols_split(const void * __restrict__ vx, const void * __restrict__ vy,
                                                      float * __restrict__ dst, const int ncols, const int nrows,
                                                      const int stride_col_y_bytes, const int stride_col_dst,
                                                      float * __restrict__ slm,
                                                      const sycl::nd_item<3> & nd_item) {
    using q4_k_block = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_K>;

    const auto sg       = nd_item.get_sub_group();
    const int  sg_range = sg.get_group_linear_range();
    const int  sg_id    = sg.get_group_linear_id();

    const int rows_per_group = sg_range / SPLIT;
    const int row_local      = sg_id / SPLIT;
    const int part           = sg_id % SPLIT;
    const int row            = nd_item.get_group_linear_id() * rows_per_group + row_local;

    // No early return here, because every subgroup must reach the barrier.
    const bool active = row < nrows && row_local < rows_per_group;

    const int blocks_per_row  = ncols / QK_K;
    const int nblocks         = nrows * blocks_per_row;
    const int lane            = sg.get_local_linear_id();
    const int blocks_per_wave = WARP_SIZE / 8;
    const int lb              = lane / 8;
    const int c               = lane % 8;
    const int g               = c >> 1;
    const int half            = c & 1;
    const int uoff            = half ? 4 : 0;

    const uint8_t * base = static_cast<const uint8_t *>(vx);

    float partial[ncols_dst] = { 0.0f };

    if (active) {
        // interleave whole block-waves so each subgroup keeps the 128-byte coalesced loads
        for (int blk = lb + part * blocks_per_wave; blk < blocks_per_row; blk += SPLIT * blocks_per_wave) {
        const int  ibx     = row * blocks_per_row + blk;
        const auto ibx_off = q4_k_block::get_block_offset(ibx, nblocks);
        const auto d_off   = q4_k_block::get_d_offset(nrows, ncols, ibx);

        const uint8_t *     qs     = base + ibx_off.first;
        const uint16_t *    scales = reinterpret_cast<const uint16_t *>(base + d_off.first);
        const sycl::half2 * dms    = reinterpret_cast<const sycl::half2 *>(base + d_off.second);

        const sycl::int4 vq = *(reinterpret_cast<const sycl::int4 *>(qs) + c);
        uint16_t  aux[2];
        const int j = g;
        if (j < 2) {
            aux[0] = scales[j + 0] & 0x3f3f;
            aux[1] = scales[j + 2] & 0x3f3f;
        } else {
            aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }
        const uint8_t *    sc   = reinterpret_cast<const uint8_t *>(aux);
        const uint8_t *    m    = sc + 2;
        const sycl::float2 dm4f = (*dms).convert<float, sycl::rounding_mode::automatic>();

        const int gq8 = blk * (QK_K / QK8_1) + 2 * g;

#pragma unroll
        for (int col = 0; col < ncols_dst; ++col) {
            const char *        vy_j = static_cast<const char *>(vy) + col * stride_col_y_bytes;
            const int8_t *      vy8  = reinterpret_cast<const int8_t *>(vy_j);
            const sycl::half2 * vyds = reinterpret_cast<const sycl::half2 *>(vy_j + ncols);
            const sycl::half2   ds0  = vyds[gq8 + 0];
            const sycl::half2   ds1  = vyds[gq8 + 1];
            const sycl::int4    u0   = *reinterpret_cast<const sycl::int4 *>(reinterpret_cast<const int *>(vy8 + (gq8 + 0) * QK8_1) + uoff);
            const sycl::int4    u1   = *reinterpret_cast<const sycl::int4 *>(reinterpret_cast<const int *>(vy8 + (gq8 + 1) * QK8_1) + uoff);

            partial[col] += mul_mat_vec_q4_K_reorder_wide_dot(vq, sc, m, dm4f, half, u0, u1, ds0[0], ds1[0], ds0[1], ds1[1]);
        }
        }
    }

#pragma unroll
    for (int col = 0; col < ncols_dst; ++col) {
        const float sum = sycl::reduce_over_group(sg, partial[col], std::plus<>());
        if (sg.leader()) {
            slm[sg_id * ncols_dst + col] = sum;
        }
    }

    nd_item.barrier(sycl::access::fence_space::local_space);

    if (part == 0 && active) {
#pragma unroll
        for (int col = 0; col < ncols_dst; ++col) {
            float total = 0.0f;
#pragma unroll
            for (int pt = 0; pt < SPLIT; ++pt) {
                total += slm[(row_local * SPLIT + pt) * ncols_dst + col];
            }
            if (sg.leader()) {
                dst[col * stride_col_dst + row] = total;
            }
        }
    }
}

// Split factor for a shape. It splits only when the rows cannot fill the device, and never past the row's block-waves.
// The default target is 16384 subgroups for one column and 8192 for more.
// GGML_SYCL_MMVQ_SPLIT_TARGET overrides the target (0 disables), GGML_SYCL_MMVQ_SPLIT forces a split.
static inline int ggml_sycl_mmvq_split(const int ncols, const int nrows, const int ncols_dst) {
    static const int forced     = ggml_sycl_get_env("GGML_SYCL_MMVQ_SPLIT", 0);
    static const int target_env = ggml_sycl_get_env("GGML_SYCL_MMVQ_SPLIT_TARGET", -1);

    const int target = target_env >= 0 ? target_env : (ncols_dst == 1 ? 16384 : 8192);

    const int waves_per_row = ceil_div(ncols / QK_K, WARP_SIZE / 8);

    int split = forced;
    if (split <= 0) {
        split = 1;
        if (target > 0 && nrows > 0 && nrows < target) {
            const int want = ceil_div(target, nrows);
            split = want >= 8 ? 8 : (want >= 4 ? 4 : (want >= 2 ? 2 : 1));
        }
    }
    while (split > 1 && split > waves_per_row) {
        split /= 2;
    }
    return split;
}

template <int ncols_dst, int SPLIT>
static void reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_split(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    constexpr size_t num_subgroups  = WARP_SIZE;
    constexpr int    rows_per_group = (int) num_subgroups / SPLIT;

    const int block_num_y = ceil_div(nrows, rows_per_group);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>(num_subgroups * ncols_dst), cgh);
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q4_K_reorder_wide_ncols_split<ncols_dst, SPLIT>(
                                 vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 get_pointer(slm), nd_item);
                         });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t num_subgroups = WARP_SIZE;

    switch (ggml_sycl_mmvq_split(ncols, nrows, ncols_dst)) {
        case 2:
            reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_split<ncols_dst, 2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
            return;
        case 4:
            reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_split<ncols_dst, 4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
            return;
        case 8:
            reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols_split<ncols_dst, 8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
            return;
        default:
            break;
    }

    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q4_K_reorder_wide_ncols<ncols_dst>(
                                 vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q4_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q4_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q4_K reorder multi-col MMVQ", ncols_dst);
    }
}

// Multi-column version of the SwiGLU-fused wide Q4_K kernel.
template <int ncols_dst>
static void mul_mat_vec_q4_K_reorder_wide_ncols_glu(const void * __restrict__ vx, const void * __restrict__ vgate,
                                                    const void * __restrict__ vy, float * __restrict__ dst,
                                                    const int ncols, const int nrows, const int stride_col_y_bytes,
                                                    const int stride_col_dst, const sycl::nd_item<3> & nd_item) {
    using q4_k_block = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_K>;

    const auto sg       = nd_item.get_sub_group();
    const int  sg_range = sg.get_group_linear_range();
    const int  row      = nd_item.get_group_linear_id() * sg_range + sg.get_group_linear_id();
    if (row >= nrows) {
        return;
    }

    const int blocks_per_row  = ncols / QK_K;
    const int nblocks         = nrows * blocks_per_row;
    const int lane            = sg.get_local_linear_id();
    const int blocks_per_wave = WARP_SIZE / 8;
    const int lb              = lane / 8;
    const int c               = lane % 8;
    const int g               = c >> 1;
    const int half            = c & 1;
    const int uoff            = half ? 4 : 0;

    const uint8_t * base_up   = static_cast<const uint8_t *>(vx);
    const uint8_t * base_gate = static_cast<const uint8_t *>(vgate);

    float partial_up[ncols_dst]   = { 0.0f };
    float partial_gate[ncols_dst] = { 0.0f };

    for (int blk = lb; blk < blocks_per_row; blk += blocks_per_wave) {
        const int  ibx     = row * blocks_per_row + blk;
        const auto ibx_off = q4_k_block::get_block_offset(ibx, nblocks);
        const auto d_off   = q4_k_block::get_d_offset(nrows, ncols, ibx);
        const int  gq8     = blk * (QK_K / QK8_1) + 2 * g;

        const sycl::int4 vq_u = *(reinterpret_cast<const sycl::int4 *>(base_up + ibx_off.first) + c);
        const sycl::int4 vq_g = *(reinterpret_cast<const sycl::int4 *>(base_gate + ibx_off.first) + c);
        const uint16_t * scu = reinterpret_cast<const uint16_t *>(base_up + d_off.first);
        const uint16_t * scg = reinterpret_cast<const uint16_t *>(base_gate + d_off.first);
        uint16_t  au[2], ag[2];
        const int j = g;
        if (j < 2) {
            au[0] = scu[j + 0] & 0x3f3f;  au[1] = scu[j + 2] & 0x3f3f;
            ag[0] = scg[j + 0] & 0x3f3f;  ag[1] = scg[j + 2] & 0x3f3f;
        } else {
            au[0] = ((scu[j + 2] >> 0) & 0x0f0f) | ((scu[j - 2] & 0xc0c0) >> 2);
            au[1] = ((scu[j + 2] >> 4) & 0x0f0f) | ((scu[j - 0] & 0xc0c0) >> 2);
            ag[0] = ((scg[j + 2] >> 0) & 0x0f0f) | ((scg[j - 2] & 0xc0c0) >> 2);
            ag[1] = ((scg[j + 2] >> 4) & 0x0f0f) | ((scg[j - 0] & 0xc0c0) >> 2);
        }
        const uint8_t *    scu8 = reinterpret_cast<const uint8_t *>(au);
        const uint8_t *    scg8 = reinterpret_cast<const uint8_t *>(ag);
        const sycl::float2 dmu  = (*reinterpret_cast<const sycl::half2 *>(base_up + d_off.second)).convert<float, sycl::rounding_mode::automatic>();
        const sycl::float2 dmg  = (*reinterpret_cast<const sycl::half2 *>(base_gate + d_off.second)).convert<float, sycl::rounding_mode::automatic>();

#pragma unroll
        for (int col = 0; col < ncols_dst; ++col) {
            const char *        vy_j = static_cast<const char *>(vy) + col * stride_col_y_bytes;
            const int8_t *      vy8  = reinterpret_cast<const int8_t *>(vy_j);
            const sycl::half2 * vyds = reinterpret_cast<const sycl::half2 *>(vy_j + ncols);
            const sycl::half2   ds0  = vyds[gq8 + 0];
            const sycl::half2   ds1  = vyds[gq8 + 1];
            const sycl::int4    u0   = *reinterpret_cast<const sycl::int4 *>(reinterpret_cast<const int *>(vy8 + (gq8 + 0) * QK8_1) + uoff);
            const sycl::int4    u1   = *reinterpret_cast<const sycl::int4 *>(reinterpret_cast<const int *>(vy8 + (gq8 + 1) * QK8_1) + uoff);

            partial_up[col]   += mul_mat_vec_q4_K_reorder_wide_dot(vq_u, scu8, scu8 + 2, dmu, half, u0, u1, ds0[0], ds1[0], ds0[1], ds1[1]);
            partial_gate[col] += mul_mat_vec_q4_K_reorder_wide_dot(vq_g, scg8, scg8 + 2, dmg, half, u0, u1, ds0[0], ds1[0], ds0[1], ds1[1]);
        }
    }

#pragma unroll
    for (int col = 0; col < ncols_dst; ++col) {
        const float up_sum   = sycl::reduce_over_group(sg, partial_up[col], std::plus<>());
        const float gate_sum = sycl::reduce_over_group(sg, partial_gate[col], std::plus<>());
        if (sg.leader()) {
            const float silu              = gate_sum / (1.0f + sycl::exp(-gate_sum));
            dst[col * stride_col_dst + row] = silu * up_sum;
        }
    }
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q4_k_q8_1_glu_fused_ncols_sycl(const void * vx, const void * vgate, const void * vy,
        float * dst, const int ncols, const int nrows, const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q4_K_reorder_wide_ncols_glu<ncols_dst>(
                                 vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q4_k_q8_1_glu_fused_switch_ncols(const void * vx, const void * vgate, const void * vy,
        float * dst, const int ncols, const int nrows, const int ncols_dst, const int stride_col_y_bytes,
        const int stride_col_dst, dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q4_k_q8_1_glu_fused_wide_sycl(vx, vgate, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q4_k_q8_1_glu_fused_ncols_sycl<2>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q4_k_q8_1_glu_fused_ncols_sycl<3>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q4_k_q8_1_glu_fused_ncols_sycl<4>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q4_k_q8_1_glu_fused_ncols_sycl<5>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q4_k_q8_1_glu_fused_ncols_sycl<6>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q4_k_q8_1_glu_fused_ncols_sycl<7>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q4_k_q8_1_glu_fused_ncols_sycl<8>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q4_K reorder glu-fused MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q5_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    launch_mul_mat_vec_q<QK_K, QI5_K, block_q5_K,
                                      VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1, QK_K>(
        vx, vy, dst, ncols, nrows, 1, 0, 0, stream);
}


static void mul_mat_vec_q5_K_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    if (ncols_dst < 1 || ncols_dst > 8) {
        GGML_ABORT("unsupported ncols_dst=%d for Q5_K multi-col MMVQ", ncols_dst);
    }
    launch_mul_mat_vec_q<QK_K, QI5_K, block_q5_K,
                                      VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1, QK_K>(
        vx, vy, dst, ncols, nrows, ncols_dst, stride_col_y, stride_col_dst, stream);
}

template <int ncols_dst>
static void mul_mat_vec_q5_K_reorder_wide_ncols(const void * __restrict__ vx, const void * __restrict__ vy,
                                                float * __restrict__ dst, const int ncols, const int nrows,
                                                const int stride_col_y_bytes, const int stride_col_dst,
                                                const sycl::nd_item<3> & nd_item);

template <int ncols_dst, int SPLIT>
static void reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_split(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream);

static void reorder_mul_mat_vec_q5_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                               const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    switch (ggml_sycl_mmvq_split(ncols, nrows, 1)) {
        case 2: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_split<1, 2>(vx, vy, dst, ncols, nrows, 0, 0, stream); return;
        case 4: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_split<1, 4>(vx, vy, dst, ncols, nrows, 0, 0, stream); return;
        case 8: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_split<1, 8>(vx, vy, dst, ncols, nrows, 0, 0, stream); return;
        default: break;
    }

    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    // One column also uses the wide kernel. Both strides are unused at ncols_dst == 1, so 0 is safe.
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                            [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                mul_mat_vec_q5_K_reorder_wide_ncols<1>(vx, vy, dst, ncols, nrows, 0, 0, nd_item);
                            });
    });
}

// Multi-column Q5_K reorder mat-vec. The weight load and 5-bit decode run once per block, outside the column loop.
template <int ncols_dst>
static void mul_mat_vec_q5_K_reorder_wide_ncols(const void * __restrict__ vx, const void * __restrict__ vy,
                                                float * __restrict__ dst, const int ncols, const int nrows,
                                                const int stride_col_y_bytes, const int stride_col_dst,
                                                const sycl::nd_item<3> & nd_item) {
    using block_type   = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q5_K>;
    using block_traits = typename block_type::traits;

    const auto sg       = nd_item.get_sub_group();
    const int  sg_range = sg.get_group_linear_range();
    const int  row      = nd_item.get_group_linear_id() * sg_range + sg.get_group_linear_id();
    if (row >= nrows) {
        return;
    }

    // Lane layout matches the wide Q4_K kernel. Lane c also loads one int4 of the qh plane.
    const int lane            = sg.get_local_linear_id();
    const int blocks_per_wave = WARP_SIZE / 8;
    const int lb              = lane / 8;
    const int c               = lane % 8;
    const int g               = c >> 1;
    const int half            = c & 1;
    const int uoff            = half ? 4 : 0;

    const int blocks_per_row = ncols / block_traits::qk;
    const int nblocks        = nrows * blocks_per_row;

    const uint8_t * vbq = static_cast<const uint8_t *>(vx);

    float partial[ncols_dst] = { 0.0f };
    for (int blk = lb; blk < blocks_per_row; blk += blocks_per_wave) {
        const int  ibx       = row * blocks_per_row + blk;
        const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
        const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
        const int  iby       = blk * block_type::block_to_q8_1_ratio();

        const uint8_t *     qs      = vbq + bx_offset.first;
        const uint8_t *     qh_base = vbq + bx_offset.second;
        const uint16_t *    scales  = reinterpret_cast<const uint16_t *>(vbq + d_offset.first);
        const sycl::half2 * dms     = reinterpret_cast<const sycl::half2 *>(vbq + d_offset.second);

        const sycl::int4 vq  = *(reinterpret_cast<const sycl::int4 *>(qs) + c);
        const sycl::int4 qh4 = *(reinterpret_cast<const sycl::int4 *>(qh_base) + half);

        uint16_t  aux[2];
        const int jsc = g;
        if (jsc < 2) {
            aux[0] = scales[jsc + 0] & 0x3f3f;
            aux[1] = scales[jsc + 2] & 0x3f3f;
        } else {
            aux[0] = ((scales[jsc + 2] >> 0) & 0x0f0f) | ((scales[jsc - 2] & 0xc0c0) >> 2);
            aux[1] = ((scales[jsc + 2] >> 4) & 0x0f0f) | ((scales[jsc - 0] & 0xc0c0) >> 2);
        }
        const uint8_t *    sc   = reinterpret_cast<const uint8_t *>(aux);
        const uint8_t *    m    = sc + 2;
        const sycl::float2 dm5f = (*dms).convert<float, sycl::rounding_mode::automatic>();

        // low nibbles are sub-block 2g, high nibbles 2g+1, and qh bit (2g+i) is the fifth bit
        // vq[p] and qh4[p] cover the same 16 positions
        int v_lo[4], v_hi[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            v_lo[p] = ((vq[p] >> 0) & 0x0F0F0F0F) | (((qh4[p] >> (2 * g + 0)) << 4) & 0x10101010);
            v_hi[p] = ((vq[p] >> 4) & 0x0F0F0F0F) | (((qh4[p] >> (2 * g + 1)) << 4) & 0x10101010);
        }

        const int gq8 = iby + 2 * g;

#pragma unroll
        for (int col = 0; col < ncols_dst; ++col) {
            const char *        vy_j = static_cast<const char *>(vy) + col * stride_col_y_bytes;
            const int8_t *      vy8  = reinterpret_cast<const int8_t *>(vy_j);
            const sycl::half2 * vyds = reinterpret_cast<const sycl::half2 *>(vy_j + ncols);
            const sycl::float2  ds0  = vyds[gq8 + 0].convert<float, sycl::rounding_mode::automatic>();
            const sycl::float2  ds1  = vyds[gq8 + 1].convert<float, sycl::rounding_mode::automatic>();
            const sycl::int4    u0   = *reinterpret_cast<const sycl::int4 *>(reinterpret_cast<const int *>(vy8 + (gq8 + 0) * QK8_1) + uoff);
            const sycl::int4    u1   = *reinterpret_cast<const sycl::int4 *>(reinterpret_cast<const int *>(vy8 + (gq8 + 1) * QK8_1) + uoff);

            int dot0 = 0;
            int dot1 = 0;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                dot0 = dpct::dp4a(v_lo[p], u0[p], dot0);
                dot1 = dpct::dp4a(v_hi[p], u1[p], dot1);
            }
            const float sumf_d = ds0.x() * (float) (dot0 * sc[0]) + ds1.x() * (float) (dot1 * sc[1]);
            const float sumf_m = (half == 0) ? (ds0.y() * m[0] + ds1.y() * m[1]) : 0.0f;
            partial[col] += dm5f.x() * sumf_d - dm5f.y() * sumf_m;
        }
    }

#pragma unroll
    for (int col = 0; col < ncols_dst; ++col) {
        const float sum = sycl::reduce_over_group(sg, partial[col], std::plus<>());
        if (sg.leader()) {
            dst[col * stride_col_dst + row] = sum;
        }
    }
}

// Split-K version of the wide Q5_K kernel for medium row counts, same scheme as the Q4_K split kernel.
template <int ncols_dst, int SPLIT>
static void mul_mat_vec_q5_K_reorder_wide_ncols_split(const void * __restrict__ vx, const void * __restrict__ vy,
                                                      float * __restrict__ dst, const int ncols, const int nrows,
                                                      const int stride_col_y_bytes, const int stride_col_dst,
                                                      float * __restrict__ slm,
                                                      const sycl::nd_item<3> & nd_item) {
    using block_type   = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q5_K>;
    using block_traits = typename block_type::traits;

    const auto sg       = nd_item.get_sub_group();
    const int  sg_range = sg.get_group_linear_range();
    const int  sg_id    = sg.get_group_linear_id();

    const int rows_per_group = sg_range / SPLIT;
    const int row_local      = sg_id / SPLIT;
    const int part           = sg_id % SPLIT;
    const int row            = nd_item.get_group_linear_id() * rows_per_group + row_local;

    // No early return here, because every subgroup must reach the barrier.
    const bool active = row < nrows;

    const int lane            = sg.get_local_linear_id();
    const int blocks_per_wave = WARP_SIZE / 8;
    const int lb              = lane / 8;
    const int c               = lane % 8;
    const int g               = c >> 1;
    const int half            = c & 1;
    const int uoff            = half ? 4 : 0;

    const int blocks_per_row = ncols / block_traits::qk;
    const int nblocks        = nrows * blocks_per_row;

    const uint8_t * vbq = static_cast<const uint8_t *>(vx);

    float partial[ncols_dst] = { 0.0f };
    if (active) {
        for (int blk = lb + part * blocks_per_wave; blk < blocks_per_row; blk += SPLIT * blocks_per_wave) {
            const int  ibx       = row * blocks_per_row + blk;
            const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
            const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
            const int  iby       = blk * block_type::block_to_q8_1_ratio();

            const uint8_t *     qs      = vbq + bx_offset.first;
            const uint8_t *     qh_base = vbq + bx_offset.second;
            const uint16_t *    scales  = reinterpret_cast<const uint16_t *>(vbq + d_offset.first);
            const sycl::half2 * dms     = reinterpret_cast<const sycl::half2 *>(vbq + d_offset.second);

            const sycl::int4 vq  = *(reinterpret_cast<const sycl::int4 *>(qs) + c);
            const sycl::int4 qh4 = *(reinterpret_cast<const sycl::int4 *>(qh_base) + half);

            uint16_t  aux[2];
            const int jsc = g;
            if (jsc < 2) {
                aux[0] = scales[jsc + 0] & 0x3f3f;
                aux[1] = scales[jsc + 2] & 0x3f3f;
            } else {
                aux[0] = ((scales[jsc + 2] >> 0) & 0x0f0f) | ((scales[jsc - 2] & 0xc0c0) >> 2);
                aux[1] = ((scales[jsc + 2] >> 4) & 0x0f0f) | ((scales[jsc - 0] & 0xc0c0) >> 2);
            }
            const uint8_t *    sc   = reinterpret_cast<const uint8_t *>(aux);
            const uint8_t *    m    = sc + 2;
            const sycl::float2 dm5f = (*dms).convert<float, sycl::rounding_mode::automatic>();

            int v_lo[4], v_hi[4];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                v_lo[p] = ((vq[p] >> 0) & 0x0F0F0F0F) | (((qh4[p] >> (2 * g + 0)) << 4) & 0x10101010);
                v_hi[p] = ((vq[p] >> 4) & 0x0F0F0F0F) | (((qh4[p] >> (2 * g + 1)) << 4) & 0x10101010);
            }

            const int gq8 = iby + 2 * g;

#pragma unroll
            for (int col = 0; col < ncols_dst; ++col) {
                const char *        vy_j = static_cast<const char *>(vy) + col * stride_col_y_bytes;
                const int8_t *      vy8  = reinterpret_cast<const int8_t *>(vy_j);
                const sycl::half2 * vyds = reinterpret_cast<const sycl::half2 *>(vy_j + ncols);
                const sycl::float2  ds0  = vyds[gq8 + 0].convert<float, sycl::rounding_mode::automatic>();
                const sycl::float2  ds1  = vyds[gq8 + 1].convert<float, sycl::rounding_mode::automatic>();
                const sycl::int4    u0   = *reinterpret_cast<const sycl::int4 *>(reinterpret_cast<const int *>(vy8 + (gq8 + 0) * QK8_1) + uoff);
                const sycl::int4    u1   = *reinterpret_cast<const sycl::int4 *>(reinterpret_cast<const int *>(vy8 + (gq8 + 1) * QK8_1) + uoff);

                int dot0 = 0;
                int dot1 = 0;
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    dot0 = dpct::dp4a(v_lo[p], u0[p], dot0);
                    dot1 = dpct::dp4a(v_hi[p], u1[p], dot1);
                }
                const float sumf_d = ds0.x() * (float) (dot0 * sc[0]) + ds1.x() * (float) (dot1 * sc[1]);
                const float sumf_m = (half == 0) ? (ds0.y() * m[0] + ds1.y() * m[1]) : 0.0f;
                partial[col] += dm5f.x() * sumf_d - dm5f.y() * sumf_m;
            }
        }
    }

#pragma unroll
    for (int col = 0; col < ncols_dst; ++col) {
        const float sum = sycl::reduce_over_group(sg, partial[col], std::plus<>());
        if (sg.leader()) {
            slm[sg_id * ncols_dst + col] = sum;
        }
    }

    nd_item.barrier(sycl::access::fence_space::local_space);

    if (part == 0 && active) {
#pragma unroll
        for (int col = 0; col < ncols_dst; ++col) {
            float total = 0.0f;
#pragma unroll
            for (int pt = 0; pt < SPLIT; ++pt) {
                total += slm[(row_local * SPLIT + pt) * ncols_dst + col];
            }
            if (sg.leader()) {
                dst[col * stride_col_dst + row] = total;
            }
        }
    }
}

template <int ncols_dst, int SPLIT>
static void reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_split(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    constexpr size_t num_subgroups  = WARP_SIZE;
    constexpr int    rows_per_group = (int) num_subgroups / SPLIT;

    const int block_num_y = ceil_div(nrows, rows_per_group);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> slm(sycl::range<1>(num_subgroups * ncols_dst), cgh);
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q5_K_reorder_wide_ncols_split<ncols_dst, SPLIT>(
                                 vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 get_pointer(slm), nd_item);
                         });
    });
}

// Multi-column Q6_K reorder mat-vec. The weight load and 6-bit decode run once per block, outside the column loop.
template <int ncols_dst>
static void mul_mat_vec_q6_K_reorder_wide_ncols(const void * __restrict__ vx, const void * __restrict__ vy,
                                                float * __restrict__ dst, const int ncols, const int nrows,
                                                const int stride_col_y_bytes, const int stride_col_dst,
                                                const sycl::nd_item<3> & nd_item) {
    using block_type   = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q6_K>;

    const auto sg       = nd_item.get_sub_group();
    const int  sg_range = sg.get_group_linear_range();
    const int  row      = nd_item.get_group_linear_id() * sg_range + sg.get_group_linear_id();
    if (row >= nrows) {
        return;
    }

    // 8 lanes per Q6_K block, each loading one int4 of ql, as in the wide Q4_K kernel.
    const int blocks_per_row  = ncols / QK_K;
    const int nblocks         = nrows * blocks_per_row;
    const int lane            = sg.get_local_linear_id();
    const int blocks_per_wave = WARP_SIZE / 8;
    const int lb              = lane / 8;
    const int c               = lane % 8;

    // per-lane constants for quant groups 4c..4c+3
    const int qh_i4_idx = 2 * (c / 4) + (c % 2);
    const int vh_shift  = 2 * (((4 * c) % 16) / 8);
    const int scale_off = 8 * (c / 4) + ((4 * c) % 16) / 4;
    const int bq8       = 4 * (c / 4) + ((4 * c) % 16) / 8;
    const int uoff_i4   = ((4 * c) % 8) / 4;

    const uint8_t * base = static_cast<const uint8_t *>(vx);

    float partial[ncols_dst] = { 0.0f };
    for (int blk = lb; blk < blocks_per_row; blk += blocks_per_wave) {
        const int  ibx       = row * blocks_per_row + blk;
        const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
        const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);
        const int  iby       = blk * block_type::block_to_q8_1_ratio();

        const sycl::int4 ql4 = *(reinterpret_cast<const sycl::int4 *>(base + bx_offset.first) + c);
        const sycl::int4 qh4 = *(reinterpret_cast<const sycl::int4 *>(base + bx_offset.second) + qh_i4_idx);
        const int8_t *   scs = reinterpret_cast<const int8_t *>(base + d_offset.first) + scale_off;
        const int8_t     sc0 = scs[0], sc1 = scs[4];
        const float      d   = *reinterpret_cast<const ggml_half *>(base + d_offset.second);

        int vi0[4], vi1[4];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int vl   = ql4[p];
            const int vh   = qh4[p] >> vh_shift;
            const int vih0 = ((vh >> 0) << 4) & 0x30303030;
            const int vih1 = ((vh >> 4) << 4) & 0x30303030;
            vi0[p] = byte_sub_4(((vl >> 0) & 0x0F0F0F0F) | vih0, 0x20202020);
            vi1[p] = byte_sub_4(((vl >> 4) & 0x0F0F0F0F) | vih1, 0x20202020);
        }

#pragma unroll
        for (int col = 0; col < ncols_dst; ++col) {
            const char *        vy_j = static_cast<const char *>(vy) + col * stride_col_y_bytes;
            const int8_t *      q8p  = reinterpret_cast<const int8_t *>(vy_j) + iby * QK8_1;
            const sycl::half2 * q8ds = reinterpret_cast<const sycl::half2 *>(vy_j + ncols + iby * sizeof(sycl::half2));
            const sycl::int4    u0   = *(reinterpret_cast<const sycl::int4 *>(q8p + bq8 * QK8_1) + uoff_i4);
            const sycl::int4    u1   = *(reinterpret_cast<const sycl::int4 *>(q8p + (bq8 + 2) * QK8_1) + uoff_i4);
            const float         d80  = (*(q8ds + bq8 + 0))[0];
            const float         d81  = (*(q8ds + bq8 + 2))[0];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                partial[col] += d * (d80 * (dpct::dp4a(vi0[p], u0[p], 0) * sc0) + d81 * (dpct::dp4a(vi1[p], u1[p], 0) * sc1));
            }
        }
    }

#pragma unroll
    for (int col = 0; col < ncols_dst; ++col) {
        const float sum = sycl::reduce_over_group(sg, partial[col], std::plus<>());
        if (sg.leader()) {
            dst[col * stride_col_dst + row] = sum;
        }
    }
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    switch (ggml_sycl_mmvq_split(ncols, nrows, ncols_dst)) {
        case 2:
            reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_split<ncols_dst, 2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
            return;
        case 4:
            reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_split<ncols_dst, 4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
            return;
        case 8:
            reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols_split<ncols_dst, 8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream);
            return;
        default:
            break;
    }

    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q5_K_reorder_wide_ncols<ncols_dst>(
                                 vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q5_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q5_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q5_K reorder multi-col MMVQ", ncols_dst);
    }
}

static void reorder_mul_mat_vec_q6_k_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                               const int nrows, dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    // Round up to a whole number of subgroup-sized workgroups; out-of-range rows are skipped inside the kernel.
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    // One column also uses the wide kernel, which is faster than the generic kernel at n=1.
    // Both strides are unused at ncols_dst == 1, so 0 is safe.
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q6_K_reorder_wide_ncols<1>(vx, vy, dst, ncols, nrows, 0, 0, nd_item);
                         });
    });
}

template <int ncols_dst>
static void reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q6_K_reorder_wide_ncols<ncols_dst>(
                                 vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, nd_item);
                         });
    });
}

static void reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_q6_k_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_q6_k_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for Q6_K reorder multi-col MMVQ", ncols_dst);
    }
}

static void mul_mat_vec_q6_K_q8_1_sycl(const void *vx, const void *vy,
                                       float *dst, const int ncols,
                                       const int nrows,
                                       dpct::queue_ptr stream) {
    launch_mul_mat_vec_q<QK_K, QI6_K, block_q6_K,
                                      VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1, QK_K>(
        vx, vy, dst, ncols, nrows, 1, 0, 0, stream);
}


static void mul_mat_vec_q6_K_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    if (ncols_dst < 1 || ncols_dst > 8) {
        GGML_ABORT("unsupported ncols_dst=%d for Q6_K multi-col MMVQ", ncols_dst);
    }
    launch_mul_mat_vec_q<QK_K, QI6_K, block_q6_K,
                                      VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1, QK_K>(
        vx, vy, dst, ncols, nrows, ncols_dst, stride_col_y, stride_col_dst, stream);
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

template <ggml_type type>
static void reorder_mul_mat_vec_iq3_q8_1_sycl(const void * vx, const void * vy, float * dst, const int ncols,
                                              const int nrows, dpct::queue_ptr stream) {
    static_assert(type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S);
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder<reorder_vec_dot_q_sycl<type>>(
                                 vx, vy, dst, ncols, nrows, nd_item);
                         });
    });
}

template <ggml_type type, int ncols_dst>
static void reorder_mul_mat_vec_iq3_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    static_assert(type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S);
    GGML_ASSERT(ncols % QK_K == 0);
    constexpr size_t num_subgroups = WARP_SIZE;
    const int block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl<type>, ncols_dst>(
                                 vx, /*vgate=*/ nullptr, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst,
                                 /*glu_op=*/ GGML_GLU_OP_SWIGLU, nd_item);
                         });
    });
}

template <ggml_type type>
static void reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows, const int ncols_dst,
        const int stride_col_y_bytes, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: reorder_mul_mat_vec_iq3_q8_1_sycl<type>(vx, vy, dst, ncols, nrows, stream); break;
        case 2: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 2>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 3: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 3>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 4: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 4>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 5: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 5>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 6: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 6>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 7: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 7>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        case 8: reorder_mul_mat_vec_iq3_q8_1_sycl_ncols<type, 8>(vx, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for reordered IQ3 multi-col MMVQ", ncols_dst);
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

template <int ncols_dst>
static void mul_mat_vec_iq3_s_q8_1_sycl_ncols(const void *    vx,
                                              const void *    vy,
                                              float *         dst,
                                              const int       ncols,
                                              const int       nrows,
                                              const int       stride_col_y,
                                              const int       stride_col_dst,
                                              dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_ncols<QK_K, QI3_S / 2, block_iq3_s, 1, vec_dot_iq3_s_q8_1_moe, ncols_dst>(
                                 vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, item_ct1);
                         });
    });
}

static void mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols(const void *    vx,
                                                     const void *    vy,
                                                     float *         dst,
                                                     const int       ncols,
                                                     const int       nrows,
                                                     const int       ncols_dst,
                                                     const int       stride_col_y,
                                                     const int       stride_col_dst,
                                                     dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1:
            mul_mat_vec_iq3_s_q8_1_sycl(vx, vy, dst, ncols, nrows, stream);
            break;
        case 2:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            break;
        case 3:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            break;
        case 4:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            break;
        case 5:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            break;
        case 6:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            break;
        case 7:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            break;
        case 8:
            mul_mat_vec_iq3_s_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream);
            break;
        default:
            GGML_ABORT("unsupported ncols_dst=%d for IQ3_S multi-col MMVQ", ncols_dst);
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

template <int ncols_dst>
static void mul_mat_vec_iq4_xs_q8_1_sycl_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    mul_mat_vec_q_ncols<QK_K, QI4_XS/4, block_iq4_xs,
                                        1,
                                        vec_dot_iq4_xs_q8_1,
                                        ncols_dst>(
                        vx, vy, dst, ncols, nrows,
                        stride_col_y, stride_col_dst, item_ct1);
                });
    });
}

static void mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols(
        const void * vx, const void * vy, float * dst,
        const int ncols, const int nrows,
        const int ncols_dst,
        const int stride_col_y, const int stride_col_dst,
        dpct::queue_ptr stream) {
    switch (ncols_dst) {
        case 1: mul_mat_vec_iq4_xs_q8_1_sycl(vx, vy, dst, ncols, nrows, stream); break;
        case 2: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<2>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 3: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<3>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 4: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<4>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 5: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<5>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 6: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<6>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 7: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<7>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        case 8: mul_mat_vec_iq4_xs_q8_1_sycl_ncols<8>(vx, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, stream); break;
        default: GGML_ABORT("unsupported ncols_dst=%d for IQ4_XS multi-col MMVQ", ncols_dst);
    }
}

#ifdef GGML_SYCL_MMVQ_HAS_XMX
// vgate is the gate weight of the fused gate and up kernel, nullptr for a plain mat-vec
template <template <bool> class Traits>
static void ggml_sycl_mul_mat_vec_q_xmx_run(bool            short_rows,
                                            const void *    vx,
                                            const void *    vgate,
                                            const char *    vy,
                                            float *         dst,
                                            int             ne00,
                                            int64_t         nrows,
                                            int64_t         ncols_dst,
                                            int             stride_y_bytes,
                                            int             ldd,
                                            int             glu_op,
                                            dpct::queue_ptr stream) {
    namespace esimd = ggml_sycl_esimd;
    if (vgate) {
        if (short_rows) {
            esimd::xmx_mul_mat_ncols<Traits<true>, true>(vx, vgate, vy, dst, ne00, nrows, ncols_dst, stride_y_bytes,
                                                         ldd, glu_op, stream);
        } else {
            esimd::xmx_mul_mat_ncols<Traits<false>, true>(vx, vgate, vy, dst, ne00, nrows, ncols_dst, stride_y_bytes,
                                                          ldd, glu_op, stream);
        }
    } else if (short_rows) {
        esimd::xmx_mul_mat_ncols<Traits<true>>(vx, nullptr, vy, dst, ne00, nrows, ncols_dst, stride_y_bytes, ldd, 0,
                                               stream);
    } else {
        esimd::xmx_mul_mat_ncols<Traits<false>>(vx, nullptr, vy, dst, ne00, nrows, ncols_dst, stride_y_bytes, ldd, 0,
                                                stream);
    }
}

static bool ggml_sycl_mul_mat_vec_q_xmx_type(ggml_type       type,
                                             bool            short_rows,
                                             const void *    vx,
                                             const void *    vgate,
                                             const char *    vy,
                                             float *         dst,
                                             int             ne00,
                                             int64_t         nrows,
                                             int64_t         ncols_dst,
                                             int             stride_y_bytes,
                                             int             ldd,
                                             int             glu_op,
                                             dpct::queue_ptr stream) {
#    define XMX_RUN(TRAITS)                                                                                       \
        ggml_sycl_mul_mat_vec_q_xmx_run<ggml_sycl_esimd::TRAITS>(short_rows, vx, vgate, vy, dst, ne00, nrows,     \
                                                                 ncols_dst, stride_y_bytes, ldd, glu_op, stream); \
        return true
    switch (type) {
        case GGML_TYPE_Q2_K:
            XMX_RUN(xmx_traits_q2_k);
        case GGML_TYPE_Q3_K:
            XMX_RUN(xmx_traits_q3_k);
        case GGML_TYPE_Q4_K:
            XMX_RUN(xmx_traits_q4_k);
        case GGML_TYPE_Q5_K:
            XMX_RUN(xmx_traits_q5_k);
        case GGML_TYPE_Q6_K:
            XMX_RUN(xmx_traits_q6_k);
        case GGML_TYPE_Q8_0:
            XMX_RUN(xmx_traits_q8_0);
        default:
            return false;
    }
#    undef XMX_RUN
}

// multi-column XMX path for reordered weights, returns false when it does not apply
static bool ggml_sycl_mul_mat_vec_q_xmx(int                 device,
                                        const ggml_tensor * src0,
                                        const ggml_tensor * dst,
                                        const char *        src0_dd_i,
                                        const char *        src1_ddq_i,
                                        float *             dst_dd_i,
                                        int64_t             row_diff,
                                        int64_t             src1_ncols,
                                        int64_t             src1_padded_col_size,
                                        dpct::queue_ptr     stream) {
    const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    if (!ggml_sycl_xmx_enabled(device) || !ggml_sycl_xmx_supports_type(src0->type) || !extra ||
        !extra->optimized_feature.reorder) {
        return false;
    }
    if (src1_ncols < ggml_sycl_xmx_min_cols(src0->type) || src1_ncols > g_ggml_sycl_xmx_max_cols) {
        return false;
    }

    const int ne00 = src0->ne[0];
    const int ldd  = dst->ne[0];
    if (!ggml_sycl_esimd::xmx_supported(src0_dd_i, ne00)) {
        return false;
    }

    // rows under 64 bytes use the variant without 2D block loads, it is not built for the widest tile
    const bool short_rows = ne00 < 4 * QK_K;
    if (short_rows && src1_ncols > 8) {
        return false;
    }

    const int stride_y_bytes = src1_padded_col_size * sizeof(block_q8_1) / QK8_1;
    // the 2D block loads of the activations need a 64 byte aligned base and a pitch of 16 bytes
    if ((uintptr_t) src1_ddq_i % 64 != 0 || stride_y_bytes % 16 != 0) {
        return false;
    }
    return ggml_sycl_mul_mat_vec_q_xmx_type(src0->type, short_rows, src0_dd_i, nullptr, src1_ddq_i, dst_dd_i, ne00,
                                            row_diff, src1_ncols, stride_y_bytes, ldd, 0, stream);
}
#endif // GGML_SYCL_MMVQ_HAS_XMX

bool ggml_sycl_mul_mat_vec_q_glu_xmx(int              device,
                                     enum ggml_type   src0_type,
                                     enum ggml_glu_op glu_op,
                                     const void *     vx,
                                     const void *     vgate,
                                     const void *     vy,
                                     float *          dst,
                                     int              ncols,
                                     int              nrows,
                                     int              ncols_dst,
                                     int              stride_col_y_bytes,
                                     int              stride_col_dst,
                                     dpct::queue_ptr  stream) {
#ifdef GGML_SYCL_MMVQ_HAS_XMX
    if (!ggml_sycl_xmx_enabled(device) || !ggml_sycl_xmx_supports_type(src0_type)) {
        return false;
    }
    if (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        return false;
    }
    if (ncols_dst < ggml_sycl_xmx_min_cols(src0_type) || ncols_dst > g_ggml_sycl_xmx_glu_max_cols) {
        return false;
    }
    if (!ggml_sycl_esimd::xmx_supported(vx, ncols) || !ggml_sycl_esimd::xmx_supported(vgate, ncols) ||
        (uintptr_t) vy % 64 != 0 || stride_col_y_bytes % 16 != 0) {
        return false;
    }
    // rows under 64 bytes use the variant without 2D block loads, it is not built for the wide tiles
    const bool short_rows = ncols < 4 * QK_K;
    if (short_rows && ncols_dst > 8) {
        return false;
    }
    return ggml_sycl_mul_mat_vec_q_xmx_type(src0_type, short_rows, vx, vgate, (const char *) vy, dst, ncols, nrows,
                                            ncols_dst, stride_col_y_bytes, stride_col_dst, (int) glu_op, stream);
#else
    GGML_UNUSED_VARS(device, src0_type, glu_op, vx, vgate, vy, dst, ncols, nrows, ncols_dst, stride_col_y_bytes,
                     stride_col_dst, stream);
    return false;
#endif // GGML_SYCL_MMVQ_HAS_XMX
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

#ifdef GGML_SYCL_MMVQ_HAS_XMX
    if (ggml_sycl_mul_mat_vec_q_xmx(ctx.device, src0, dst, src0_dd_i, src1_ddq_i, dst_dd_i, row_diff, src1_ncols,
                                    src1_padded_col_size, stream)) {
        return;
    }
#endif // GGML_SYCL_MMVQ_HAS_XMX

    int id;
    SYCL_CHECK(CHECK_TRY_ERROR(id = get_current_device_id()));
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
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q4_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_0_q8_1_sycl\n");
                    mul_mat_vec_q4_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q4_1:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_1_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q4_1_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q4_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q5_0:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q5_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q5_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q5_1:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_1_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q5_1_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_q5_1_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q8_0:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q8_0_q8_1_sycl\n");
                        reorder_mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q8_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q8_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q8_0_q8_1_sycl\n");
                    mul_mat_vec_q8_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q1_0:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q1_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q1_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q1_0_q8_1_sycl\n");
                    mul_mat_vec_q1_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q2_0:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_0_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q2_0_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_0_q8_1_sycl\n");
                    mul_mat_vec_q2_0_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q2_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q2_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q2_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q2_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_K_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q2_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q2_K_q8_1_sycl\n");
                    mul_mat_vec_q2_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q3_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q3_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q3_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q3_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q3_K_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q3_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q3_K_q8_1_sycl\n");
                    mul_mat_vec_q3_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q4_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q4_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q4_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q4_k_q8_1_wide_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q4_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q4_K_q8_1_sycl\n");
                    mul_mat_vec_q4_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q5_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q5_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q5_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q5_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_K_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q5_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q5_K_q8_1_sycl\n");
                    mul_mat_vec_q5_K_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_Q6_K:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_q6_k_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_q6_k_q8_1_sycl\n");
                        reorder_mul_mat_vec_q6_k_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_q6_K_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_q6_K_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
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
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reordered IQ3_XXS multi-col MMVQ ncols=%d\n", (int) src1_ncols);
                        reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols<GGML_TYPE_IQ3_XXS>(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reordered IQ3_XXS MMVQ\n");
                        reorder_mul_mat_vec_iq3_q8_1_sycl<GGML_TYPE_IQ3_XXS>(
                            src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else {
                    mul_mat_vec_iq3_xxs_q8_1_sycl(
                        src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_IQ3_S:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reordered IQ3_S multi-col MMVQ ncols=%d\n", (int) src1_ncols);
                        reorder_mul_mat_vec_iq3_q8_1_sycl_switch_ncols<GGML_TYPE_IQ3_S>(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reordered IQ3_S MMVQ\n");
                        reorder_mul_mat_vec_iq3_q8_1_sycl<GGML_TYPE_IQ3_S>(
                            src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols ncols=%d\n", (int) src1_ncols);
                    mul_mat_vec_iq3_s_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_iq3_s_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_IQ4_NL:
                mul_mat_vec_iq4_nl_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                break;
            case GGML_TYPE_IQ4_XS:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_iq4_xs_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_iq4_xs_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_MXFP4:
                if ((ggml_tensor_extra_gpu *) dst->src[0]->extra &&
                    ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                    if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                        const int stride_col_y_bytes = src1_padded_col_size * q8_1_ts / q8_1_bs;
                        const int stride_col_dst     = dst->ne[0];
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                        reorder_mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
                            src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                            src1_ncols, stride_col_y_bytes, stride_col_dst, stream);
                        return;
                    } else {
                        GGML_SYCL_DEBUG("Calling reorder_mul_mat_vec_mxfp4_q8_1_sycl\n");
                        reorder_mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                    }
                } else if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_mxfp4_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_mxfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            case GGML_TYPE_NVFP4:
                if (i == 0 && src1_ncols > 1 && src1_ncols <= 8) {
                    const int stride_col_y   = src1_padded_col_size / QK8_1;
                    const int stride_col_dst = dst->ne[0];
                    GGML_SYCL_DEBUG("Calling mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols ncols=%d\n", (int)src1_ncols);
                    mul_mat_vec_nvfp4_q8_1_sycl_switch_ncols(
                        src0_dd_i, src1_ddq_i, dst_dd_i, ne00, row_diff,
                        src1_ncols, stride_col_y, stride_col_dst, stream);
                    return;
                } else if (i == 0 || src1_ncols == 1) {
                    mul_mat_vec_nvfp4_q8_1_sycl(src0_dd_i, src1_ddq_i_bs, dst_dd_i_bs, ne00, row_diff, stream);
                }
                break;
            default:
                GGML_ABORT("fatal error: unsupport src0 data type %s\n", ggml_type_name(src0->type));
        }
    }
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(ctx);
}

// src1_row_stride: 0 for shared src1 (gate/up proj), else per-expert stride (down proj).
template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void mul_mat_vec_q_moe(
    const void * __restrict__ vx_base, const void * __restrict__ vy_base,
    float * __restrict__ dst_base, const int32_t * __restrict__ ids_dev,
    const int ncols, const int nrows,
    const size_t expert_weight_stride, const size_t dst_row_stride,
    const size_t src1_row_stride,
    const sycl::nd_item<3> & item_ct1) {

    const int expert_idx = item_ct1.get_group(1);
    const int i02        = ids_dev[expert_idx];

    const char * vy = (const char *) vy_base + (size_t) expert_idx * src1_row_stride;
    float *      dst = (float *) ((char *) dst_base + (size_t) expert_idx * dst_row_stride);

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    if (row >= nrows) {
        return;
    }

    if (i02 == -1) { // skipped slot
        if (item_ct1.get_local_id(2) == 0) {
            dst[row] = 0.0f;
        }
        return;
    }

    const char * vx = (const char *) vx_base + (size_t) i02 * expert_weight_stride;

    const int     blocks_per_row  = ncols / qk;
    constexpr int blocks_per_warp = (vdr * WARP_SIZE + qi - 1) / qi;

    float tmp = 0.0f;

    const block_q_t *  x = (const block_q_t *) vx;
    const block_q8_1 * y = (const block_q8_1 *) vy;

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
        dst[row] = tmp;
    }
}

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_sycl_t vec_dot_q_sycl>
static void launch_mul_mat_vec_q_moe(
    const void * vx_base, const void * vy, const int32_t * ids_dev,
    float * dst_base, const int ncols, const int nrows, const int n_experts_used,
    const size_t expert_weight_stride, const size_t dst_row_stride,
    const size_t src1_row_stride,
    dpct::queue_ptr stream) {
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, (unsigned) n_experts_used, (unsigned) block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_moe<qk, qi, block_q_t, vdr, vec_dot_q_sycl>(
                    vx_base, vy, dst_base, ids_dev, ncols, nrows,
                    expert_weight_stride, dst_row_stride, src1_row_stride, item);
            });
    });
}

bool ggml_sycl_mul_mat_vec_q_id(
    enum ggml_type     src0_type,
    const void *       vx_base,
    const void *       vy,
    const int32_t *    ids_dev,
    float *            dst_base,
    int                ncols,
    int                nrows,
    int                n_experts_used,
    size_t             expert_weight_stride,
    size_t             dst_row_stride,
    size_t             src1_row_stride,
    dpct::queue_ptr    stream) {
    // if adding a new type, ensure it is also added to ggml_sycl_supports_mmvq_id
    switch (src0_type) {
        case GGML_TYPE_Q4_0:
            launch_mul_mat_vec_q_moe<QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q4_1:
            launch_mul_mat_vec_q_moe<QK4_1, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q5_0:
            launch_mul_mat_vec_q_moe<QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q5_1:
            launch_mul_mat_vec_q_moe<QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q8_0:
            launch_mul_mat_vec_q_moe<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q2_0:
            launch_mul_mat_vec_q_moe<QK2_0, QI2_0, block_q2_0, VDR_Q2_0_Q8_1_MMVQ, vec_dot_q2_0_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q2_K:
            launch_mul_mat_vec_q_moe<QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q3_K:
            launch_mul_mat_vec_q_moe<QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe<QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_MXFP4:
            launch_mul_mat_vec_q_moe<QK_MXFP4, QI_MXFP4, block_mxfp4, VDR_MXFP4_Q8_1_MMVQ, vec_dot_mxfp4_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_NVFP4:
            launch_mul_mat_vec_q_moe<QK_NVFP4, QI_NVFP4, block_nvfp4, VDR_NVFP4_Q8_1_MMVQ, vec_dot_nvfp4_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_IQ2_XXS:
            launch_mul_mat_vec_q_moe<QK_K, QI2_XXS/2, block_iq2_xxs, VDR_IQ2_XXS_Q8_1_MMVQ, vec_dot_iq2_xxs_q8_1_moe>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_IQ2_XS:
            launch_mul_mat_vec_q_moe<QK_K, QI2_XS/2, block_iq2_xs, VDR_IQ2_XS_Q8_1_MMVQ, vec_dot_iq2_xs_q8_1_moe>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_IQ2_S:
            launch_mul_mat_vec_q_moe<QK_K, QI2_S/2, block_iq2_s, VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_IQ3_XXS:
            launch_mul_mat_vec_q_moe<QK_K, QI3_XXS/2, block_iq3_xxs, VDR_IQ3_XXS_Q8_1_MMVQ, vec_dot_iq3_xxs_q8_1_moe>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_IQ3_S:
            launch_mul_mat_vec_q_moe<QK_K, QI3_S/2, block_iq3_s, VDR_IQ3_S_Q8_1_MMVQ, vec_dot_iq3_s_q8_1_adapter>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_IQ1_S:
            launch_mul_mat_vec_q_moe<QK_K, QI1_S, block_iq1_s, VDR_IQ1_S_Q8_1_MMVQ, vec_dot_iq1_s_q8_1_moe>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_IQ1_M:
            launch_mul_mat_vec_q_moe<QK_K, QI1_S, block_iq1_m, VDR_IQ1_M_Q8_1_MMVQ, vec_dot_iq1_m_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_IQ4_NL:
            launch_mul_mat_vec_q_moe<QK4_NL, QI4_NL, block_iq4_nl, VDR_IQ4_NL_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_IQ4_XS:
            launch_mul_mat_vec_q_moe<QK_K, QI4_XS/4, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        default:
            return false;
    }
}

// Reorder (SoA) MoE expert GEMV: MoE expert/row/lane indexing (from mul_mat_vec_q_moe) with the
// dense-reorder per-block reads (from mul_mat_vec_q_reorder). Each expert slice in vx_base is a
// self-contained SoA, so nblocks = nrows*(ncols/qk) per expert and the constant expert stride holds.
template <typename reorder_vec_dot_q_sycl>
static void mul_mat_vec_q_moe_reorder(
    const void * __restrict__ vx_base, const void * __restrict__ vy_base,
    float * __restrict__ dst_base, const int32_t * __restrict__ ids_dev,
    const int ncols, const int nrows,
    const size_t expert_weight_stride, const size_t dst_row_stride,
    const size_t src1_row_stride,
    const sycl::nd_item<3> & item_ct1) {
    using block_type   = ggml_sycl_reordered::block_q_t<reorder_vec_dot_q_sycl::gtype>;
    using block_traits = typename block_type::traits;

    const int expert_idx = item_ct1.get_group(1);
    const int i02        = ids_dev[expert_idx];

    const char * vy  = (const char *) vy_base + (size_t) expert_idx * src1_row_stride;
    float *      dst = (float *) ((char *) dst_base + (size_t) expert_idx * dst_row_stride);

    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    if (row >= nrows) {
        return;
    }

    const auto sg = item_ct1.get_sub_group();

    if (i02 == -1) { // skipped slot
        if (sg.leader()) {
            dst[row] = 0.0f;
        }
        return;
    }

    const char * vx = (const char *) vx_base + (size_t) i02 * expert_weight_stride;

    constexpr int vdr_mmvq = reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0 ?
                            VDR_Q8_0_Q8_1_MMVQ : block_traits::vdr_mmvq;
    const int     blocks_per_row              = ncols / block_traits::qk;
    constexpr int blocks_per_subgroup         = ceil_div(vdr_mmvq * WARP_SIZE, block_traits::qi);
    constexpr int block_elements_per_subgroup = block_traits::qi / vdr_mmvq;
    const int     nblocks                     = nrows * (ncols / block_traits::qk);

    static_assert(blocks_per_subgroup > 0);
    static_assert(block_elements_per_subgroup > 0);

    float partial_sum = 0.0f;
    for (int i = sg.get_local_linear_id() / block_elements_per_subgroup; i < blocks_per_row; i += blocks_per_subgroup) {
        const int ibx = row * blocks_per_row + i;

        const auto bx_offset = block_type::get_block_offset(ibx, nblocks);
        const auto d_offset  = block_type::get_d_offset(nrows, ncols, ibx);

        const int           iby            = i * block_type::block_to_q8_1_ratio();
        const int8_t *      q8_1_quant_ptr = (const int8_t *) vy + iby * QK8_1;
        const sycl::half2 * q8_1_ds_ptr    = (const sycl::half2 *) ((const char *) vy + ncols + iby * sizeof(sycl::half2));

#pragma unroll
        for (int elem = 0; elem < block_elements_per_subgroup; elem += WARP_SIZE) {
            const int iqs = elem + vdr_mmvq * (sg.get_local_linear_id() % block_elements_per_subgroup);
            if constexpr (reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0) {
                partial_sum += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs, vdr_mmvq);
            } else {
                partial_sum += reorder_vec_dot_q_sycl()(vx, bx_offset, d_offset, q8_1_quant_ptr, q8_1_ds_ptr, iqs);
            }
        }
    }

    float sum = partial_sum;
    if constexpr (reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q8_0) {
        // Keep the accumulation order of the non-reordered MoE kernel.
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            sum += dpct::permute_sub_group_by_xor(sg, sum, mask);
        }
    } else {
        sum = sycl::reduce_over_group(sg, partial_sum, std::plus<>());
    }
    if (sg.leader()) {
        dst[row] = sum;
    }
}

template <typename reorder_vec_dot_q_sycl>
static void launch_mul_mat_vec_q_moe_reorder(
    const void * vx_base, const void * vy, const int32_t * ids_dev,
    float * dst_base, const int ncols, const int nrows, const int n_experts_used,
    const size_t expert_weight_stride, const size_t dst_row_stride,
    const size_t src1_row_stride,
    dpct::queue_ptr stream) {
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, (unsigned) n_experts_used, (unsigned) block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl>(
                    vx_base, vy, dst_base, ids_dev, ncols, nrows,
                    expert_weight_stride, dst_row_stride, src1_row_stride, item);
            });
    });
}

bool ggml_sycl_mul_mat_vec_q_id_reorder(
    enum ggml_type     src0_type,
    const void *       vx_base,
    const void *       vy,
    const int32_t *    ids_dev,
    float *            dst_base,
    int                ncols,
    int                nrows,
    int                n_experts_used,
    size_t             expert_weight_stride,
    size_t             dst_row_stride,
    size_t             src1_row_stride,
    dpct::queue_ptr    stream) {
    switch (src0_type) {
        case GGML_TYPE_Q8_0:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0>>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q4_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q5_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K>>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_Q6_K:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K>>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        case GGML_TYPE_MXFP4:
            launch_mul_mat_vec_q_moe_reorder<reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4>>(
                vx_base, vy, ids_dev, dst_base, ncols, nrows, n_experts_used,
                expert_weight_stride, dst_row_stride, src1_row_stride, stream);
            return true;
        default:
            return false;
    }
}

template <typename reorder_vec_dot_q_sycl, int ncols_dst, int rows_per_sg>
static void launch_mul_mat_vec_q_reorder_glu_impl(const void * vx, const void * vgate, const void * vy, float * dst,
                                             const int ncols, const int nrows, const int stride_col_y_bytes,
                                             const int stride_col_dst, const ggml_glu_op glu_op,
                                             dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);

    constexpr size_t num_subgroups = WARP_SIZE;

    const int            block_num_y = ceil_div(nrows, GGML_SYCL_MMV_Y * (int) num_subgroups * rows_per_sg);
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, num_subgroups * WARP_SIZE);

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_reorder_ncols<reorder_vec_dot_q_sycl, ncols_dst, /*has_fusion=*/ true,
                                                        rows_per_sg>(
                                 vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, glu_op,
                                 nd_item);
                         });
    });
}

template <typename reorder_vec_dot_q_sycl, int ncols_dst>
static void launch_mul_mat_vec_q_reorder_glu(const void * vx, const void * vgate, const void * vy, float * dst,
                                             const int ncols, const int nrows, const int stride_col_y_bytes,
                                             const int stride_col_dst, const ggml_glu_op glu_op,
                                             dpct::queue_ptr stream) {
    // q4_K pairs rows for 3..4 columns, q5_K for 3..5
    constexpr int row_pair_max = reorder_vec_dot_q_sycl::gtype == GGML_TYPE_Q5_K ? 5 : 4;
    constexpr int rows_per_sg =
        reorder_vec_dot_shared_activations<reorder_vec_dot_q_sycl::gtype>::value && ncols_dst >= 3 &&
                ncols_dst <= row_pair_max
            ? 2
            : 1;
    launch_mul_mat_vec_q_reorder_glu_impl<reorder_vec_dot_q_sycl, ncols_dst, rows_per_sg>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, glu_op, stream);
}

// ---------------------------------------------------------------------------
// Fused dense-FFN GEMV + GLU over the standard (non-reorder) weight layout.
//
// Unlike the reorder variant below, the two weights may carry different block
// types (e.g. an unsloth UD mix with an iq4_xs gate and a q5_K up), as long as
// both quantize in QK_K-sized super-blocks so that one q8_1 activation
// quantization serves both dots. Per-operand accumulation order matches
// mul_mat_vec_q exactly, so results are bit-identical to running the three
// nodes separately.
// ---------------------------------------------------------------------------
template <int qi_g, typename block_g_t, int vdr_g, vec_dot_q_sycl_t vec_dot_g,
          int qi_u, typename block_u_t, int vdr_u, vec_dot_q_sycl_t vec_dot_u, int ncols_dst>
static void mul_mat_vec_q_glu(const void * __restrict__ vxg, const void * __restrict__ vxu,
                              const void * __restrict__ vy, float * __restrict__ dst, const int ncols,
                              const int nrows, const int stride_col_y, const int stride_col_dst,
                              const ggml_glu_op glu_op, const sycl::nd_item<3> & item_ct1) {
    static_assert(QK_K % QK8_1 == 0);
    const int row = item_ct1.get_group(2) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);
    if (row >= nrows) {
        return;
    }
    const int     blocks_per_row  = ncols / QK_K;
    constexpr int blocks_per_warp_g = (vdr_g * WARP_SIZE + qi_g - 1) / qi_g;
    constexpr int blocks_per_warp_u = (vdr_u * WARP_SIZE + qi_u - 1) / qi_u;
    // one partial sum per output column, per operand
    float tmpg[ncols_dst] = {0.0f};
    float tmpu[ncols_dst] = {0.0f};
    const block_g_t  * xg = (const block_g_t *) vxg;
    const block_u_t  * xu = (const block_u_t *) vxu;
    const block_q8_1 * y  = (const block_q8_1 *) vy;
    for (int i = item_ct1.get_local_id(2) / (qi_g / vdr_g); i < blocks_per_row; i += blocks_per_warp_g) {
        const int ibx = row * blocks_per_row + i;
        const int iby = i * (QK_K / QK8_1);
        for (size_t elem = 0; elem < qi_g / vdr_g; elem += WARP_SIZE) {
            const int iqs = elem + vdr_g * (item_ct1.get_local_id(2) % (qi_g / vdr_g));
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                tmpg[j] += vec_dot_g(&xg[ibx], &y[j * stride_col_y + iby], iqs);
            }
        }
    }
    for (int i = item_ct1.get_local_id(2) / (qi_u / vdr_u); i < blocks_per_row; i += blocks_per_warp_u) {
        const int ibx = row * blocks_per_row + i;
        const int iby = i * (QK_K / QK8_1);
        for (size_t elem = 0; elem < qi_u / vdr_u; elem += WARP_SIZE) {
            const int iqs = elem + vdr_u * (item_ct1.get_local_id(2) % (qi_u / vdr_u));
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) {
                tmpu[j] += vec_dot_u(&xu[ibx], &y[j * stride_col_y + iby], iqs);
            }
        }
    }
    // sum up partial sums and write back the activated product
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
            tmpg[j] += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmpg[j], mask);
            tmpu[j] += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmpu[j], mask);
        }
    }
    if (item_ct1.get_local_id(2) == 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
            // uniform across the launch; the dispatcher only accepts SWIGLU and GEGLU
            const float gate = glu_op == GGML_GLU_OP_SWIGLU ? op_silu(tmpg[j]) : op_gelu(tmpg[j]);
            dst[j * stride_col_dst + row] = gate * tmpu[j];
        }
    }
}

template <int qi_g, typename block_g_t, int vdr_g, vec_dot_q_sycl_t vec_dot_g,
          int qi_u, typename block_u_t, int vdr_u, vec_dot_q_sycl_t vec_dot_u, int ncols_dst>
static void launch_mul_mat_vec_q_glu(const void * vxg, const void * vxu, const void * vy, float * dst,
                                     const int ncols, const int nrows, const int stride_col_y,
                                     const int stride_col_dst, const ggml_glu_op glu_op,
                                     dpct::queue_ptr stream) {
    GGML_ASSERT(ncols % QK_K == 0);
    const int            block_num_y = (nrows + GGML_SYCL_MMV_Y - 1) / GGML_SYCL_MMV_Y;
    const sycl::range<3> block_nums(1, 1, block_num_y);
    const sycl::range<3> block_dims(1, GGML_SYCL_MMV_Y, WARP_SIZE);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> nd_item) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                               qi_u, block_u_t, vdr_u, vec_dot_u, ncols_dst>(
                                 vxg, vxu, vy, dst, ncols, nrows, stride_col_y, stride_col_dst,
                                 glu_op, nd_item);
                         });
    });
}

// Dispatch the plain-layout fused GLU GEMV over the activation batch: ncols_dst
// selects the kernel's per-column template parameter. Returns false when the
// batch exceeds the instantiated range; the caller falls back to unfused nodes.
template <int qi_g, typename block_g_t, int vdr_g, vec_dot_q_sycl_t vec_dot_g,
          int qi_u, typename block_u_t, int vdr_u, vec_dot_q_sycl_t vec_dot_u>
static bool dispatch_mul_mat_vec_q_glu_plain(const void * vgate, const void * vup, const void * vy,
                                             float * dst, const int ncols, const int nrows,
                                             const int stride_col_y, const int stride_col_dst,
                                             const ggml_glu_op glu_op, dpct::queue_ptr stream,
                                             const int ncols_dst) {
    switch (ncols_dst) {
        case 1:
            launch_mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                     qi_u, block_u_t, vdr_u, vec_dot_u, 1>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            return true;
        case 2:
            launch_mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                     qi_u, block_u_t, vdr_u, vec_dot_u, 2>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            return true;
        case 3:
            launch_mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                     qi_u, block_u_t, vdr_u, vec_dot_u, 3>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            return true;
        case 4:
            launch_mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                     qi_u, block_u_t, vdr_u, vec_dot_u, 4>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            return true;
        case 5:
            launch_mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                     qi_u, block_u_t, vdr_u, vec_dot_u, 5>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            return true;
        case 6:
            launch_mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                     qi_u, block_u_t, vdr_u, vec_dot_u, 6>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            return true;
        case 7:
            launch_mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                     qi_u, block_u_t, vdr_u, vec_dot_u, 7>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            return true;
        case 8:
            launch_mul_mat_vec_q_glu<qi_g, block_g_t, vdr_g, vec_dot_g,
                                     qi_u, block_u_t, vdr_u, vec_dot_u, 8>(
                vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream);
            return true;
        default:
            return false;
    }
}

// Fused dense-FFN GEMV + GLU for weight pairs the reorder kernel does not cover.
// vgate/vup must be in the standard block layout; vy must be quantized with plain
// quantize_q8_1 (padded rows). stride_col_y is in block_q8_1 units.
// Returns false if the type pair or batch is unhandled; caller should fall back.
bool ggml_sycl_mul_mat_vec_q_glu_plain(enum ggml_type gate_type, enum ggml_type up_type,
                                       enum ggml_glu_op glu_op, const void * vgate, const void * vup,
                                       const void * vy, float * dst, int ncols, int nrows, int ncols_dst,
                                       int stride_col_y, int stride_col_dst, dpct::queue_ptr stream) {
    if (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        return false;
    }
    if (ncols % QK_K != 0) {
        return false;
    }
    if (gate_type == GGML_TYPE_Q5_K && up_type == GGML_TYPE_Q5_K) {
        return dispatch_mul_mat_vec_q_glu_plain<QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1,
                                                QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream, ncols_dst);
    }
    if (gate_type == GGML_TYPE_IQ4_XS && up_type == GGML_TYPE_IQ4_XS) {
        return dispatch_mul_mat_vec_q_glu_plain<QI4_XS / 4, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1,
                                                QI4_XS / 4, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream, ncols_dst);
    }
    if (gate_type == GGML_TYPE_IQ4_XS && up_type == GGML_TYPE_Q5_K) {
        return dispatch_mul_mat_vec_q_glu_plain<QI4_XS / 4, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1,
                                                QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream, ncols_dst);
    }
    if (gate_type == GGML_TYPE_Q5_K && up_type == GGML_TYPE_IQ4_XS) {
        return dispatch_mul_mat_vec_q_glu_plain<QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1,
                                                QI4_XS / 4, block_iq4_xs, VDR_IQ4_XS_Q8_1_MMVQ, vec_dot_iq4_xs_q8_1>(
            vgate, vup, vy, dst, ncols, nrows, stride_col_y, stride_col_dst, glu_op, stream, ncols_dst);
    }
    return false;
}

template <ggml_type type, int... Ns>
static bool mul_mat_vec_q_glu_reorder_ncols(enum ggml_glu_op glu_op, const void * vx, const void * vgate,
                                            const void * vy, float * dst, int ncols, int nrows, int ncols_dst,
                                            int stride_col_y_bytes, int stride_col_dst, dpct::queue_ptr stream) {
    using vec_dot = reorder_vec_dot_q_sycl<type>;

    auto launch = [&](auto I) -> bool {
        constexpr int n = decltype(I)::value;
        if (ncols_dst != n) {
            return false;
        }
        if constexpr (type == GGML_TYPE_Q4_K && n == 2) {
            if (nrows >= Q4_K_MMVQ_ROW_PAIR_MIN_NROWS) {
                launch_mul_mat_vec_q_reorder_glu_impl<vec_dot, 2, 2>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes, stride_col_dst, glu_op, stream);
                return true;
            }
        }
        launch_mul_mat_vec_q_reorder_glu<vec_dot, n>(vx, vgate, vy, dst, ncols, nrows, stride_col_y_bytes,
                                                     stride_col_dst, glu_op, stream);
        return true;
    };

    // unary fold over launch
    return (launch(std::integral_constant<int, Ns>{}) || ...);
}

bool ggml_sycl_mul_mat_vec_q_glu_reorder(enum ggml_type src0_type, enum ggml_glu_op glu_op, const void * vx,
                                         const void * vgate, const void * vy, float * dst, int ncols, int nrows,
                                         int ncols_dst, int stride_col_y_bytes, int stride_col_dst,
                                         dpct::queue_ptr stream) {
    if (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        return false;
    }

    // SwiGLU uses the fused wide kernels, which load each weight quad once for both dot products.
    if (src0_type == GGML_TYPE_Q4_K && glu_op == GGML_GLU_OP_SWIGLU) {
        if (ncols_dst == 1) {
            reorder_mul_mat_vec_q4_k_q8_1_glu_fused_wide_sycl(vx, vgate, vy, dst, ncols, nrows, stream);
            return true;
        }
        if (ncols_dst >= 2 && ncols_dst <= 8) {
            reorder_mul_mat_vec_q4_k_q8_1_glu_fused_switch_ncols(vx, vgate, vy, dst, ncols, nrows, ncols_dst,
                                                                 stride_col_y_bytes, stride_col_dst, stream);
            return true;
        }
    }

    switch (src0_type) {
        case GGML_TYPE_Q4_K:
            return mul_mat_vec_q_glu_reorder_ncols<GGML_TYPE_Q4_K, 1, 2, 3, 4, 5, 6, 7, 8>(
                glu_op, vx, vgate, vy, dst, ncols, nrows, ncols_dst, stride_col_y_bytes, stride_col_dst, stream);
        case GGML_TYPE_Q5_K:
            // fusion declines q5_K past 5 columns
            return mul_mat_vec_q_glu_reorder_ncols<GGML_TYPE_Q5_K, 1, 2, 3, 4, 5>(
                glu_op, vx, vgate, vy, dst, ncols, nrows, ncols_dst, stride_col_y_bytes, stride_col_dst, stream);
        default:
            return false;
    }
}
