#pragma once

#include <algorithm>

#include "../tier2/mmvq_tier2_spec_common.hpp"

namespace sycl_bench {
namespace mmvq_tier3 {

#if MMVQ_TIER2_XMX_AVAILABLE

using mmvq_tier2::AParams;
using mmvq_tier2::block_traits;
using mmvq_tier2::decode_block_aos;
using mmvq_tier2::validate_xmx_args_common;
namespace sycl_xmx = mmvq_tier2::sycl_xmx;
#if MMVQ_TIER2_ESIMD_AVAILABLE
namespace esimd = mmvq_tier2::esimd;
namespace esimd_xmx = mmvq_tier2::esimd_xmx;
#endif

constexpr int XMX_M       = mmvq_tier2::XMX_M;
constexpr int XMX_N       = mmvq_tier2::XMX_N;
constexpr int XMX_K       = mmvq_tier2::XMX_K;
constexpr int XMX_SG_SIZE = mmvq_tier2::XMX_SG_SIZE;

constexpr int SLM_A_SIZE        = mmvq_tier2::SLM_A_SIZE;
constexpr int SLM_B_SIZE        = mmvq_tier2::SLM_B_SIZE;
constexpr int SLM_C_SIZE        = mmvq_tier2::SLM_C_SIZE;
constexpr int SLM_SCALES_A_SIZE = mmvq_tier2::SLM_SCALES_A_SIZE;
constexpr int SLM_OFFSETS_A_SIZE = mmvq_tier2::SLM_OFFSETS_A_SIZE;
constexpr int SLM_SCALES_B_SIZE = mmvq_tier2::SLM_SCALES_B_SIZE;
constexpr int SLM_SUMS_B_SIZE   = mmvq_tier2::SLM_SUMS_B_SIZE;

static inline bool validate_tier3_args(const ggml_sycl::mmvq_bench_args & args,
                                       std::string & error) {
    return validate_xmx_args_common(args, GGML_LAYOUT_AOS, error);
}

template <ggml_type T, int RowSubgroups, int ColTiles, bool Persistent>
static inline void mmvq_tile_kernel_colfold(const void * __restrict__ weights,
                                            const void * __restrict__ activations,
                                            float * __restrict__ dst,
                                            const int ncols_x,
                                            const int ncols_y,
                                            const int nrows_dst,
                                            const int row_low,
                                            const int row_high,
                                            const int src1_stride_blocks,
                                            const int blocks_per_row,
                                            int8_t * __restrict__ slm_A,
                                            int8_t * __restrict__ slm_B,
                                            int32_t * __restrict__ slm_C,
                                            float * __restrict__ slm_scales_A,
                                            float * __restrict__ slm_offsets_A,
                                            float * __restrict__ slm_scales_B,
                                            float * __restrict__ slm_sums_B,
                                            sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];
    const int  row_sg  = item.get_local_id(0);

    const int tile_rows = RowSubgroups * XMX_M;
    const int row_base_group = item.get_group(0) * tile_rows + row_low;
    if (row_base_group >= row_high) {
        return;
    }

    const int col_tiles_span = ColTiles * XMX_N;
    const int col_groups = (ncols_y + col_tiles_span - 1) / col_tiles_span;
    const int col_group_start = item.get_group(1);
    const int col_group_stride = Persistent ? item.get_group_range(1) : col_groups;

    const int row_base = row_base_group + row_sg * XMX_M;
    const int num_k_blocks = ncols_x / XMX_K;

    const char * src1 = static_cast<const char *>(activations);
    constexpr int Q8_1_BLOCK_SIZE = sizeof(block_q8_1);
    constexpr int SUBBLOCKS = block_traits<T>::block_size / XMX_K;
    constexpr int PASSES = block_traits<T>::per_16 ? 2 : 1;

    for (int col_group = col_group_start; col_group < col_groups; col_group += col_group_stride) {
        const int col_base_group = col_group * col_tiles_span;
        if (col_base_group >= ncols_y) {
            continue;
        }

        float acc[ColTiles][8];
        for (int tile = 0; tile < ColTiles; ++tile) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[tile][i] = 0.0f;
            }
        }

        for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
            int8_t q_local[XMX_K];
            AParams params{};

            if (lane_id < XMX_M) {
                const int row = row_base + lane_id;
                if (row < row_high) {
                    const int block_idx = k_block / SUBBLOCKS;
                    const int sub_idx = k_block - block_idx * SUBBLOCKS;
                    decode_block_aos<T>(weights, row, blocks_per_row, block_idx, sub_idx, q_local, params);
                } else {
#pragma unroll
                    for (int j = 0; j < XMX_K; j++) {
                        q_local[j] = 0;
                    }
                    params.scale0  = 0.0f;
                    params.offset0 = 0.0f;
                    params.scale1  = 0.0f;
                    params.offset1 = 0.0f;
                }
            }

            for (int tile = 0; tile < ColTiles; ++tile) {
                const int col_base = col_base_group + tile * XMX_N;
                const bool active_tile = (col_base < ncols_y);

                if (lane_id < XMX_N) {
                    const int col = col_base + lane_id;
                    const bool active_col = active_tile && (col < ncols_y);
                    if (active_col) {
                        const char * block_ptr =
                            src1 + (static_cast<int64_t>(col) * src1_stride_blocks + k_block) * Q8_1_BLOCK_SIZE;
                        const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
                        const sycl::half s = *reinterpret_cast<const sycl::half *>(block_ptr + 2);

                        slm_scales_B[lane_id] = float(d);

                        float sum0 = 0.0f;
                        float sum1 = 0.0f;
                        const int8_t * qs = reinterpret_cast<const int8_t *>(block_ptr + 4);
#pragma unroll
                        for (int k = 0; k < XMX_K; ++k) {
                            const int8_t v = qs[k];
                            slm_B[lane_id * XMX_K + k] = v;
                            if constexpr (T == GGML_TYPE_Q2_K) {
                                if (k < 16) {
                                    sum0 += v;
                                } else {
                                    sum1 += v;
                                }
                            }
                        }
                        if constexpr (T == GGML_TYPE_Q2_K) {
                            const float d_f = float(d);
                            slm_sums_B[0 * XMX_N + lane_id] = d_f * sum0;
                            slm_sums_B[1 * XMX_N + lane_id] = d_f * sum1;
                        } else {
                            slm_sums_B[lane_id] = float(s);
                        }
                    } else {
                        slm_scales_B[lane_id] = 0.0f;
                        if constexpr (T == GGML_TYPE_Q2_K) {
                            slm_sums_B[0 * XMX_N + lane_id] = 0.0f;
                            slm_sums_B[1 * XMX_N + lane_id] = 0.0f;
                        } else {
                            slm_sums_B[lane_id] = 0.0f;
                        }
#pragma unroll
                        for (int k = 0; k < XMX_K; ++k) {
                            slm_B[lane_id * XMX_K + k] = 0;
                        }
                    }
                }

                item.barrier(sycl::access::fence_space::local_space);

                for (int pass = 0; pass < PASSES; ++pass) {
                    if (lane_id < XMX_M) {
                        const float scale = (pass == 0) ? params.scale0 : params.scale1;
                        const float offset = (pass == 0) ? params.offset0 : params.offset1;
                        slm_scales_A[pass * XMX_M + lane_id] = scale;
                        slm_offsets_A[pass * XMX_M + lane_id] = offset;
                        const int base_idx = lane_id * XMX_K;
#pragma unroll
                        for (int j = 0; j < XMX_K; ++j) {
                            if constexpr (PASSES == 1) {
                                slm_A[base_idx + j] = q_local[j];
                            } else {
                                const bool keep = (pass == 0) ? (j < 16) : (j >= 16);
                                slm_A[base_idx + j] = keep ? q_local[j] : 0;
                            }
                        }
                    }

                    item.barrier(sycl::access::fence_space::local_space);

                    sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K,
                                           sycl_xmx::layout::row_major>
                        matA;
                    sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N,
                                           sycl_xmx::layout::col_major>
                        matB;
                    sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

                    auto A_ptr =
                        sycl::address_space_cast<sycl::access::address_space::local_space,
                                                 sycl::access::decorated::no>(slm_A);
                    auto B_ptr =
                        sycl::address_space_cast<sycl::access::address_space::local_space,
                                                 sycl::access::decorated::no>(slm_B);
                    auto C_ptr =
                        sycl::address_space_cast<sycl::access::address_space::local_space,
                                                 sycl::access::decorated::no>(slm_C);

                    sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
                    sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
                    sycl_xmx::joint_matrix_fill(sg, matC, 0);
                    sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
                    sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

#pragma unroll
                    for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
                        const int   i    = idx / XMX_N;
                        const int   j    = idx % XMX_N;
                        const float d_A  = slm_scales_A[pass * XMX_M + i];
                        const float o_A  = slm_offsets_A[pass * XMX_M + i];
                        const float d_B  = slm_scales_B[j];
                        float s_B = 0.0f;
                        if constexpr (T == GGML_TYPE_Q2_K) {
                            s_B = slm_sums_B[pass * XMX_N + j];
                        } else {
                            s_B = slm_sums_B[j];
                        }
                        const float C_ij = static_cast<float>(slm_C[idx]);
                        acc[tile][acc_idx] += d_A * (C_ij * d_B) + o_A * s_B;
                    }

                    item.barrier(sycl::access::fence_space::local_space);
                }

                item.barrier(sycl::access::fence_space::local_space);
            }
        }

        for (int tile = 0; tile < ColTiles; ++tile) {
            const int col_base = col_base_group + tile * XMX_N;
            if (col_base >= ncols_y) {
                continue;
            }
#pragma unroll
            for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
                const int i   = idx / XMX_N;
                const int j   = idx % XMX_N;
                const int row = row_base + i;
                const int col = col_base + j;
                if (row < row_high && col < ncols_y) {
                    dst[col * nrows_dst + row] = acc[tile][acc_idx];
                }
            }
        }
    }
}

template <ggml_type T, int RowSubgroups, int ColTiles, bool Persistent>
static inline bool launch_xmx_colfold_typed(const ggml_sycl::mmvq_bench_args & args,
                                            std::vector<sycl::event> * events,
                                            std::string & error) {
    (void) error;
    const int row_low  = static_cast<int>(args.row_low);
    const int row_high = static_cast<int>(args.row_high);
    const int rows_span = row_high - row_low;
    const int tile_rows = RowSubgroups * XMX_M;
    const int col_tiles_span = ColTiles * XMX_N;
    const int num_row_tiles = (rows_span + tile_rows - 1) / tile_rows;
    const int num_col_tiles = (static_cast<int>(args.batch) + col_tiles_span - 1) / col_tiles_span;

    const int ncols_x = static_cast<int>(args.ncols);
    const int ncols_y = static_cast<int>(args.batch);
    const int nrows_dst = static_cast<int>(args.dst_row_stride);
    const int src1_stride_blocks = static_cast<int>(args.src1_padded_col_size / XMX_K);

    const int block_size = block_traits<T>::block_size;
    const int blocks_per_row = ncols_x / block_size;

    const void * weights = args.weights;
    const void * activations = args.activations;
    float * output = args.output;

    sycl::queue & queue = *args.stream;
    int grid_cols = num_col_tiles;
    if constexpr (Persistent) {
        grid_cols = std::min(grid_cols, 4);
        if (grid_cols < 1) {
            grid_cols = 1;
        }
    }
    const sycl::range<2> grid(num_row_tiles, grid_cols);
    const sycl::range<2> block(RowSubgroups, XMX_SG_SIZE);

    constexpr int TOTAL_SLM_SIZE =
        SLM_A_SIZE + SLM_B_SIZE + SLM_C_SIZE * 4 +
        SLM_SCALES_A_SIZE * 4 + SLM_OFFSETS_A_SIZE * 4 +
        SLM_SCALES_B_SIZE * 4 + SLM_SUMS_B_SIZE * 4;

    sycl::event ev = queue.submit([&](sycl::handler & h) {
        sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE * RowSubgroups), h);

        h.parallel_for(sycl::nd_range<2>(grid * block, block),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                           char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                           const int row_sg = item.get_local_id(0);
                           int offset = row_sg * TOTAL_SLM_SIZE;
                           int8_t * slm_A = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_A_SIZE;
                           int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_B_SIZE;
                           int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                           offset += SLM_C_SIZE * 4;
                           float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_A_SIZE * 4;
                           float * slm_offsets_A = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_OFFSETS_A_SIZE * 4;
                           float * slm_scales_B = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_B_SIZE * 4;
                           float * slm_sums_B = reinterpret_cast<float *>(shared + offset);

                           mmvq_tile_kernel_colfold<T, RowSubgroups, ColTiles, Persistent>(
                               weights, activations, output,
                               ncols_x, ncols_y, nrows_dst,
                               row_low, row_high, src1_stride_blocks,
                               blocks_per_row,
                               slm_A, slm_B, slm_C,
                               slm_scales_A, slm_offsets_A, slm_scales_B, slm_sums_B, item);
                       });
    });

    if (events != nullptr) {
        events->push_back(ev);
    }
    return true;
}

template <int RowSubgroups, int ColTiles, bool Persistent>
static inline bool launch_xmx_colfold_dispatch(const ggml_sycl::mmvq_bench_args & args,
                                               std::vector<sycl::event> * events,
                                               std::string & error) {
    switch (args.weight_type) {
        case GGML_TYPE_Q4_0:
            return launch_xmx_colfold_typed<GGML_TYPE_Q4_0, RowSubgroups, ColTiles, Persistent>(args, events, error);
        case GGML_TYPE_Q8_0:
            return launch_xmx_colfold_typed<GGML_TYPE_Q8_0, RowSubgroups, ColTiles, Persistent>(args, events, error);
        case GGML_TYPE_Q6_K:
            return launch_xmx_colfold_typed<GGML_TYPE_Q6_K, RowSubgroups, ColTiles, Persistent>(args, events, error);
        case GGML_TYPE_MXFP4:
            return launch_xmx_colfold_typed<GGML_TYPE_MXFP4, RowSubgroups, ColTiles, Persistent>(args, events, error);
        case GGML_TYPE_Q4_K:
            return launch_xmx_colfold_typed<GGML_TYPE_Q4_K, RowSubgroups, ColTiles, Persistent>(args, events, error);
        case GGML_TYPE_Q2_K:
            return launch_xmx_colfold_typed<GGML_TYPE_Q2_K, RowSubgroups, ColTiles, Persistent>(args, events, error);
        case GGML_TYPE_Q3_K:
            return launch_xmx_colfold_typed<GGML_TYPE_Q3_K, RowSubgroups, ColTiles, Persistent>(args, events, error);
        case GGML_TYPE_Q5_K:
            return launch_xmx_colfold_typed<GGML_TYPE_Q5_K, RowSubgroups, ColTiles, Persistent>(args, events, error);
        default:
            error = "Tier3 kernel: unsupported quant type.";
            return false;
    }
}

template <ggml_type T, int RowSubgroups, int ColSubgroups>
static inline void mmvq_tile_kernel_coop(const void * __restrict__ weights,
                                         const void * __restrict__ activations,
                                         float * __restrict__ dst,
                                         const int ncols_x,
                                         const int ncols_y,
                                         const int nrows_dst,
                                         const int row_low,
                                         const int row_high,
                                         const int src1_stride_blocks,
                                         const int blocks_per_row,
                                         int8_t * __restrict__ slm_A,
                                         int8_t * __restrict__ slm_B,
                                         int32_t * __restrict__ slm_C,
                                         float * __restrict__ slm_scales_A,
                                         float * __restrict__ slm_offsets_A,
                                         float * __restrict__ slm_scales_B,
                                         float * __restrict__ slm_sums_B,
                                         sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];
    const int  sg_linear = item.get_local_id(0);
    const int  row_sg = sg_linear / ColSubgroups;
    const int  col_sg = sg_linear - row_sg * ColSubgroups;

    const int tile_rows = RowSubgroups * XMX_M;
    const int tile_cols = ColSubgroups * XMX_N;
    const int row_base_group = item.get_group(0) * tile_rows + row_low;
    const int col_base_group = item.get_group(1) * tile_cols;
    if (row_base_group >= row_high || col_base_group >= ncols_y) {
        return;
    }

    const int row_base = row_base_group + row_sg * XMX_M;
    const int col_base = col_base_group + col_sg * XMX_N;
    if (row_base >= row_high) {
        return;
    }

    const int num_k_blocks = ncols_x / XMX_K;

    const char * src1 = static_cast<const char *>(activations);
    constexpr int Q8_1_BLOCK_SIZE = sizeof(block_q8_1);
    constexpr int SUBBLOCKS = block_traits<T>::block_size / XMX_K;
    constexpr int PASSES = block_traits<T>::per_16 ? 2 : 1;

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for (int k_block = 0; k_block < num_k_blocks; ++k_block) {
        int8_t q_local[XMX_K];
        AParams params{};

        if (lane_id < XMX_M) {
            const int row = row_base + lane_id;
            if (row < row_high) {
                const int block_idx = k_block / SUBBLOCKS;
                const int sub_idx = k_block - block_idx * SUBBLOCKS;
                decode_block_aos<T>(weights, row, blocks_per_row, block_idx, sub_idx, q_local, params);
            } else {
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    q_local[j] = 0;
                }
                params.scale0  = 0.0f;
                params.offset0 = 0.0f;
                params.scale1  = 0.0f;
                params.offset1 = 0.0f;
            }
        }

        if (lane_id < XMX_N) {
            const int col = col_base + lane_id;
            const bool active_col = (col < ncols_y);
            if (active_col) {
                const char * block_ptr =
                    src1 + (static_cast<int64_t>(col) * src1_stride_blocks + k_block) * Q8_1_BLOCK_SIZE;
                const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
                const sycl::half s = *reinterpret_cast<const sycl::half *>(block_ptr + 2);

                slm_scales_B[lane_id] = float(d);

                float sum0 = 0.0f;
                float sum1 = 0.0f;
                const int8_t * qs = reinterpret_cast<const int8_t *>(block_ptr + 4);
#pragma unroll
                for (int k = 0; k < XMX_K; ++k) {
                    const int8_t v = qs[k];
                    slm_B[lane_id * XMX_K + k] = v;
                    if constexpr (T == GGML_TYPE_Q2_K) {
                        if (k < 16) {
                            sum0 += v;
                        } else {
                            sum1 += v;
                        }
                    }
                }
                if constexpr (T == GGML_TYPE_Q2_K) {
                    const float d_f = float(d);
                    slm_sums_B[0 * XMX_N + lane_id] = d_f * sum0;
                    slm_sums_B[1 * XMX_N + lane_id] = d_f * sum1;
                } else {
                    slm_sums_B[lane_id] = float(s);
                }
            } else {
                slm_scales_B[lane_id] = 0.0f;
                if constexpr (T == GGML_TYPE_Q2_K) {
                    slm_sums_B[0 * XMX_N + lane_id] = 0.0f;
                    slm_sums_B[1 * XMX_N + lane_id] = 0.0f;
                } else {
                    slm_sums_B[lane_id] = 0.0f;
                }
#pragma unroll
                for (int k = 0; k < XMX_K; ++k) {
                    slm_B[lane_id * XMX_K + k] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        for (int pass = 0; pass < PASSES; ++pass) {
            if (lane_id < XMX_M) {
                const float scale = (pass == 0) ? params.scale0 : params.scale1;
                const float offset = (pass == 0) ? params.offset0 : params.offset1;
                slm_scales_A[pass * XMX_M + lane_id] = scale;
                slm_offsets_A[pass * XMX_M + lane_id] = offset;
                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; ++j) {
                    if constexpr (PASSES == 1) {
                        slm_A[base_idx + j] = q_local[j];
                    } else {
                        const bool keep = (pass == 0) ? (j < 16) : (j >= 16);
                        slm_A[base_idx + j] = keep ? q_local[j] : 0;
                    }
                }
            }

            item.barrier(sycl::access::fence_space::local_space);

            sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K,
                                   sycl_xmx::layout::row_major>
                matA;
            sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N,
                                   sycl_xmx::layout::col_major>
                matB;
            sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

            auto A_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space,
                                         sycl::access::decorated::no>(slm_A);
            auto B_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space,
                                         sycl::access::decorated::no>(slm_B);
            auto C_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space,
                                         sycl::access::decorated::no>(slm_C);

            sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
            sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
            sycl_xmx::joint_matrix_fill(sg, matC, 0);
            sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
            sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

#pragma unroll
            for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
                const int   i    = idx / XMX_N;
                const int   j    = idx % XMX_N;
                const float d_A  = slm_scales_A[pass * XMX_M + i];
                const float o_A  = slm_offsets_A[pass * XMX_M + i];
                const float d_B  = slm_scales_B[j];
                float s_B = 0.0f;
                if constexpr (T == GGML_TYPE_Q2_K) {
                    s_B = slm_sums_B[pass * XMX_N + j];
                } else {
                    s_B = slm_sums_B[j];
                }
                const float C_ij = static_cast<float>(slm_C[idx]);
                acc[acc_idx] += d_A * (C_ij * d_B) + o_A * s_B;
            }

            item.barrier(sycl::access::fence_space::local_space);
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

#pragma unroll
    for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
        const int i   = idx / XMX_N;
        const int j   = idx % XMX_N;
        const int row = row_base + i;
        const int col = col_base + j;
        if (row < row_high && col < ncols_y) {
            dst[col * nrows_dst + row] = acc[acc_idx];
        }
    }
}

template <ggml_type T, int RowSubgroups, int ColSubgroups>
static inline bool launch_xmx_coop_typed(const ggml_sycl::mmvq_bench_args & args,
                                         std::vector<sycl::event> * events,
                                         std::string & error) {
    (void) error;
    const int row_low  = static_cast<int>(args.row_low);
    const int row_high = static_cast<int>(args.row_high);
    const int rows_span = row_high - row_low;
    const int tile_rows = RowSubgroups * XMX_M;
    const int tile_cols = ColSubgroups * XMX_N;
    const int num_row_tiles = (rows_span + tile_rows - 1) / tile_rows;
    const int num_col_tiles = (static_cast<int>(args.batch) + tile_cols - 1) / tile_cols;

    const int ncols_x = static_cast<int>(args.ncols);
    const int ncols_y = static_cast<int>(args.batch);
    const int nrows_dst = static_cast<int>(args.dst_row_stride);
    const int src1_stride_blocks = static_cast<int>(args.src1_padded_col_size / XMX_K);

    const int block_size = block_traits<T>::block_size;
    const int blocks_per_row = ncols_x / block_size;

    const void * weights = args.weights;
    const void * activations = args.activations;
    float * output = args.output;

    sycl::queue & queue = *args.stream;
    const sycl::range<2> grid(num_row_tiles, num_col_tiles);
    const sycl::range<2> block(RowSubgroups * ColSubgroups, XMX_SG_SIZE);

    constexpr int TOTAL_SLM_SIZE =
        SLM_A_SIZE + SLM_B_SIZE + SLM_C_SIZE * 4 +
        SLM_SCALES_A_SIZE * 4 + SLM_OFFSETS_A_SIZE * 4 +
        SLM_SCALES_B_SIZE * 4 + SLM_SUMS_B_SIZE * 4;

    sycl::event ev = queue.submit([&](sycl::handler & h) {
        sycl::local_accessor<char, 1> shared_acc(
            sycl::range<1>(TOTAL_SLM_SIZE * RowSubgroups * ColSubgroups), h);

        h.parallel_for(sycl::nd_range<2>(grid * block, block),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                           char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                           const int sg_linear = item.get_local_id(0);
                           int offset = sg_linear * TOTAL_SLM_SIZE;
                           int8_t * slm_A = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_A_SIZE;
                           int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_B_SIZE;
                           int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                           offset += SLM_C_SIZE * 4;
                           float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_A_SIZE * 4;
                           float * slm_offsets_A = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_OFFSETS_A_SIZE * 4;
                           float * slm_scales_B = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_B_SIZE * 4;
                           float * slm_sums_B = reinterpret_cast<float *>(shared + offset);

                           mmvq_tile_kernel_coop<T, RowSubgroups, ColSubgroups>(
                               weights, activations, output,
                               ncols_x, ncols_y, nrows_dst,
                               row_low, row_high, src1_stride_blocks,
                               blocks_per_row,
                               slm_A, slm_B, slm_C,
                               slm_scales_A, slm_offsets_A, slm_scales_B, slm_sums_B, item);
                       });
    });

    if (events != nullptr) {
        events->push_back(ev);
    }
    return true;
}

template <int RowSubgroups, int ColSubgroups>
static inline bool launch_xmx_coop_dispatch(const ggml_sycl::mmvq_bench_args & args,
                                            std::vector<sycl::event> * events,
                                            std::string & error) {
    switch (args.weight_type) {
        case GGML_TYPE_Q4_0:
            return launch_xmx_coop_typed<GGML_TYPE_Q4_0, RowSubgroups, ColSubgroups>(args, events, error);
        case GGML_TYPE_Q8_0:
            return launch_xmx_coop_typed<GGML_TYPE_Q8_0, RowSubgroups, ColSubgroups>(args, events, error);
        case GGML_TYPE_Q6_K:
            return launch_xmx_coop_typed<GGML_TYPE_Q6_K, RowSubgroups, ColSubgroups>(args, events, error);
        case GGML_TYPE_MXFP4:
            return launch_xmx_coop_typed<GGML_TYPE_MXFP4, RowSubgroups, ColSubgroups>(args, events, error);
        case GGML_TYPE_Q4_K:
            return launch_xmx_coop_typed<GGML_TYPE_Q4_K, RowSubgroups, ColSubgroups>(args, events, error);
        case GGML_TYPE_Q2_K:
            return launch_xmx_coop_typed<GGML_TYPE_Q2_K, RowSubgroups, ColSubgroups>(args, events, error);
        case GGML_TYPE_Q3_K:
            return launch_xmx_coop_typed<GGML_TYPE_Q3_K, RowSubgroups, ColSubgroups>(args, events, error);
        case GGML_TYPE_Q5_K:
            return launch_xmx_coop_typed<GGML_TYPE_Q5_K, RowSubgroups, ColSubgroups>(args, events, error);
        default:
            error = "Tier3 kernel: unsupported quant type.";
            return false;
    }
}

template <ggml_type T, int RowSubgroups, int ColSubgroups, int KSplits>
static inline void mmvq_tile_kernel_multi_wg(const void * __restrict__ weights,
                                             const void * __restrict__ activations,
                                             float * __restrict__ dst,
                                             const int ncols_x,
                                             const int ncols_y,
                                             const int nrows_dst,
                                             const int row_low,
                                             const int row_high,
                                             const int src1_stride_blocks,
                                             const int blocks_per_row,
                                             const int num_col_tiles,
                                             int8_t * __restrict__ slm_A,
                                             int8_t * __restrict__ slm_B,
                                             int32_t * __restrict__ slm_C,
                                             float * __restrict__ slm_scales_A,
                                             float * __restrict__ slm_offsets_A,
                                             float * __restrict__ slm_scales_B,
                                             float * __restrict__ slm_sums_B,
                                             sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];
    const int  sg_linear = item.get_local_id(0);
    const int  row_sg = sg_linear / ColSubgroups;
    const int  col_sg = sg_linear - row_sg * ColSubgroups;

    const int tile_rows = RowSubgroups * XMX_M;
    const int tile_cols = ColSubgroups * XMX_N;
    const int row_base_group = item.get_group(0) * tile_rows + row_low;
    if (row_base_group >= row_high) {
        return;
    }

    const int group_linear = item.get_group(1);
    const int tile_col = num_col_tiles > 0 ? (group_linear % num_col_tiles) : 0;
    const int k_slice = num_col_tiles > 0 ? (group_linear / num_col_tiles) : 0;
    if (k_slice >= KSplits) {
        return;
    }

    const int col_base_group = tile_col * tile_cols;
    if (col_base_group >= ncols_y) {
        return;
    }

    const int row_base = row_base_group + row_sg * XMX_M;
    if (row_base >= row_high) {
        return;
    }
    const int col_base = col_base_group + col_sg * XMX_N;
    if (col_base >= ncols_y) {
        return;
    }

    const int num_k_blocks = ncols_x / XMX_K;
    const int blocks_per_slice = (num_k_blocks + KSplits - 1) / KSplits;
    const int k_start = k_slice * blocks_per_slice;
    const int k_end = std::min(num_k_blocks, k_start + blocks_per_slice);
    if (k_start >= num_k_blocks) {
        return;
    }

    const char * src1 = static_cast<const char *>(activations);
    constexpr int Q8_1_BLOCK_SIZE = sizeof(block_q8_1);
    constexpr int SUBBLOCKS = block_traits<T>::block_size / XMX_K;
    constexpr int PASSES = block_traits<T>::per_16 ? 2 : 1;

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for (int k_block = k_start; k_block < k_end; ++k_block) {
        int8_t q_local[XMX_K];
        AParams params{};

        if (lane_id < XMX_M) {
            const int row = row_base + lane_id;
            if (row < row_high) {
                const int block_idx = k_block / SUBBLOCKS;
                const int sub_idx = k_block - block_idx * SUBBLOCKS;
                decode_block_aos<T>(weights, row, blocks_per_row, block_idx, sub_idx, q_local, params);
            } else {
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    q_local[j] = 0;
                }
                params.scale0  = 0.0f;
                params.offset0 = 0.0f;
                params.scale1  = 0.0f;
                params.offset1 = 0.0f;
            }
        }

        if (lane_id < XMX_N) {
            const int col = col_base + lane_id;
            const bool active_col = (col < ncols_y);
            if (active_col) {
                const char * block_ptr =
                    src1 + (static_cast<int64_t>(col) * src1_stride_blocks + k_block) * Q8_1_BLOCK_SIZE;
                const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
                const sycl::half s = *reinterpret_cast<const sycl::half *>(block_ptr + 2);

                slm_scales_B[lane_id] = float(d);

                float sum0 = 0.0f;
                float sum1 = 0.0f;
                const int8_t * qs = reinterpret_cast<const int8_t *>(block_ptr + 4);
#pragma unroll
                for (int k = 0; k < XMX_K; ++k) {
                    const int8_t v = qs[k];
                    slm_B[lane_id * XMX_K + k] = v;
                    if constexpr (T == GGML_TYPE_Q2_K) {
                        if (k < 16) {
                            sum0 += v;
                        } else {
                            sum1 += v;
                        }
                    }
                }
                if constexpr (T == GGML_TYPE_Q2_K) {
                    const float d_f = float(d);
                    slm_sums_B[0 * XMX_N + lane_id] = d_f * sum0;
                    slm_sums_B[1 * XMX_N + lane_id] = d_f * sum1;
                } else {
                    slm_sums_B[lane_id] = float(s);
                }
            } else {
                slm_scales_B[lane_id] = 0.0f;
                if constexpr (T == GGML_TYPE_Q2_K) {
                    slm_sums_B[0 * XMX_N + lane_id] = 0.0f;
                    slm_sums_B[1 * XMX_N + lane_id] = 0.0f;
                } else {
                    slm_sums_B[lane_id] = 0.0f;
                }
#pragma unroll
                for (int k = 0; k < XMX_K; ++k) {
                    slm_B[lane_id * XMX_K + k] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        for (int pass = 0; pass < PASSES; ++pass) {
            if (lane_id < XMX_M) {
                const float scale = (pass == 0) ? params.scale0 : params.scale1;
                const float offset = (pass == 0) ? params.offset0 : params.offset1;
                slm_scales_A[pass * XMX_M + lane_id] = scale;
                slm_offsets_A[pass * XMX_M + lane_id] = offset;
                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; ++j) {
                    if constexpr (PASSES == 1) {
                        slm_A[base_idx + j] = q_local[j];
                    } else {
                        const bool keep = (pass == 0) ? (j < 16) : (j >= 16);
                        slm_A[base_idx + j] = keep ? q_local[j] : 0;
                    }
                }
            }

            item.barrier(sycl::access::fence_space::local_space);

            sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K,
                                   sycl_xmx::layout::row_major>
                matA;
            sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N,
                                   sycl_xmx::layout::col_major>
                matB;
            sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

            auto A_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space,
                                         sycl::access::decorated::no>(slm_A);
            auto B_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space,
                                         sycl::access::decorated::no>(slm_B);
            auto C_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space,
                                         sycl::access::decorated::no>(slm_C);

            sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
            sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
            sycl_xmx::joint_matrix_fill(sg, matC, 0);
            sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
            sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

#pragma unroll
            for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
                const int   i    = idx / XMX_N;
                const int   j    = idx % XMX_N;
                const float d_A  = slm_scales_A[pass * XMX_M + i];
                const float o_A  = slm_offsets_A[pass * XMX_M + i];
                const float d_B  = slm_scales_B[j];
                float s_B = 0.0f;
                if constexpr (T == GGML_TYPE_Q2_K) {
                    s_B = slm_sums_B[pass * XMX_N + j];
                } else {
                    s_B = slm_sums_B[j];
                }
                const float C_ij = static_cast<float>(slm_C[idx]);
                acc[acc_idx] += d_A * (C_ij * d_B) + o_A * s_B;
            }

            item.barrier(sycl::access::fence_space::local_space);
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

#pragma unroll
    for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
        const int i   = idx / XMX_N;
        const int j   = idx % XMX_N;
        const int row = row_base + i;
        const int col = col_base + j;
        if (row < row_high && col < ncols_y) {
            float * out_ptr = dst + col * nrows_dst + row;
            sycl::atomic_ref<float,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space> atom(*out_ptr);
            atom.fetch_add(acc[acc_idx]);
        }
    }
}

template <ggml_type T, int RowSubgroups, int ColSubgroups, int KSplits>
static inline bool launch_xmx_multi_wg_typed(const ggml_sycl::mmvq_bench_args & args,
                                             std::vector<sycl::event> * events,
                                             std::string & error) {
    (void) error;
    const int row_low  = static_cast<int>(args.row_low);
    const int row_high = static_cast<int>(args.row_high);
    const int rows_span = row_high - row_low;
    const int tile_rows = RowSubgroups * XMX_M;
    const int tile_cols = ColSubgroups * XMX_N;
    const int num_row_tiles = (rows_span + tile_rows - 1) / tile_rows;
    const int num_col_tiles = (static_cast<int>(args.batch) + tile_cols - 1) / tile_cols;

    const int ncols_x = static_cast<int>(args.ncols);
    const int ncols_y = static_cast<int>(args.batch);
    const int nrows_dst = static_cast<int>(args.dst_row_stride);
    const int src1_stride_blocks = static_cast<int>(args.src1_padded_col_size / XMX_K);

    const int block_size = block_traits<T>::block_size;
    const int blocks_per_row = ncols_x / block_size;

    const void * weights = args.weights;
    const void * activations = args.activations;
    float * output = args.output;

    sycl::queue & queue = *args.stream;
    const sycl::range<2> grid(num_row_tiles, num_col_tiles * KSplits);
    const sycl::range<2> block(RowSubgroups * ColSubgroups, XMX_SG_SIZE);

    constexpr int TOTAL_SLM_SIZE =
        SLM_A_SIZE + SLM_B_SIZE + SLM_C_SIZE * 4 +
        SLM_SCALES_A_SIZE * 4 + SLM_OFFSETS_A_SIZE * 4 +
        SLM_SCALES_B_SIZE * 4 + SLM_SUMS_B_SIZE * 4;

    sycl::event ev = queue.submit([&](sycl::handler & h) {
        sycl::local_accessor<char, 1> shared_acc(
            sycl::range<1>(TOTAL_SLM_SIZE * RowSubgroups * ColSubgroups), h);

        h.parallel_for(sycl::nd_range<2>(grid * block, block),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                           char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                           const int sg_linear = item.get_local_id(0);
                           int offset = sg_linear * TOTAL_SLM_SIZE;
                           int8_t * slm_A = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_A_SIZE;
                           int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_B_SIZE;
                           int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                           offset += SLM_C_SIZE * 4;
                           float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_A_SIZE * 4;
                           float * slm_offsets_A = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_OFFSETS_A_SIZE * 4;
                           float * slm_scales_B = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_B_SIZE * 4;
                           float * slm_sums_B = reinterpret_cast<float *>(shared + offset);

                           mmvq_tile_kernel_multi_wg<T, RowSubgroups, ColSubgroups, KSplits>(
                               weights, activations, output,
                               ncols_x, ncols_y, nrows_dst,
                               row_low, row_high, src1_stride_blocks,
                               blocks_per_row, num_col_tiles,
                               slm_A, slm_B, slm_C,
                               slm_scales_A, slm_offsets_A, slm_scales_B, slm_sums_B, item);
                       });
    });

    if (events != nullptr) {
        events->push_back(ev);
    }
    return true;
}

template <int RowSubgroups, int ColSubgroups, int KSplits>
static inline bool launch_xmx_multi_wg_dispatch(const ggml_sycl::mmvq_bench_args & args,
                                                std::vector<sycl::event> * events,
                                                std::string & error) {
    switch (args.weight_type) {
        case GGML_TYPE_Q4_0:
            return launch_xmx_multi_wg_typed<GGML_TYPE_Q4_0, RowSubgroups, ColSubgroups, KSplits>(args, events, error);
        case GGML_TYPE_Q8_0:
            return launch_xmx_multi_wg_typed<GGML_TYPE_Q8_0, RowSubgroups, ColSubgroups, KSplits>(args, events, error);
        case GGML_TYPE_Q6_K:
            return launch_xmx_multi_wg_typed<GGML_TYPE_Q6_K, RowSubgroups, ColSubgroups, KSplits>(args, events, error);
        case GGML_TYPE_MXFP4:
            return launch_xmx_multi_wg_typed<GGML_TYPE_MXFP4, RowSubgroups, ColSubgroups, KSplits>(args, events, error);
        case GGML_TYPE_Q4_K:
            return launch_xmx_multi_wg_typed<GGML_TYPE_Q4_K, RowSubgroups, ColSubgroups, KSplits>(args, events, error);
        case GGML_TYPE_Q2_K:
            return launch_xmx_multi_wg_typed<GGML_TYPE_Q2_K, RowSubgroups, ColSubgroups, KSplits>(args, events, error);
        case GGML_TYPE_Q3_K:
            return launch_xmx_multi_wg_typed<GGML_TYPE_Q3_K, RowSubgroups, ColSubgroups, KSplits>(args, events, error);
        case GGML_TYPE_Q5_K:
            return launch_xmx_multi_wg_typed<GGML_TYPE_Q5_K, RowSubgroups, ColSubgroups, KSplits>(args, events, error);
        default:
            error = "Tier3 kernel: unsupported quant type.";
            return false;
    }
}

inline bool launch_xmx_tile_64x64(const ggml_sycl::mmvq_bench_args & args,
                                  std::vector<sycl::event> * events,
                                  std::string & error) {
    if (!validate_tier3_args(args, error)) {
        return false;
    }
    return launch_xmx_coop_dispatch<8, 4>(args, events, error);
}

inline bool launch_xmx_register_accum(const ggml_sycl::mmvq_bench_args & args,
                                      std::vector<sycl::event> * events,
                                      std::string & error) {
    if (!validate_tier3_args(args, error)) {
        return false;
    }
    return launch_xmx_colfold_dispatch<8, 4, false>(args, events, error);
}

inline bool launch_xmx_multi_wg(const ggml_sycl::mmvq_bench_args & args,
                                std::vector<sycl::event> * events,
                                std::string & error) {
    if (!validate_tier3_args(args, error)) {
        return false;
    }
    return launch_xmx_multi_wg_dispatch<8, 4, 2>(args, events, error);
}

inline bool launch_xmx_persistent(const ggml_sycl::mmvq_bench_args & args,
                                  std::vector<sycl::event> * events,
                                  std::string & error) {
    if (!validate_tier3_args(args, error)) {
        return false;
    }
    return launch_xmx_colfold_dispatch<8, 4, true>(args, events, error);
}

#if MMVQ_TIER2_ESIMD_AVAILABLE

template <int TypeId, int RowsPerTile, int TileCols, bool Persistent, bool Prefetch>
class mmvq_esimd_tier3_kernel;

template <ggml_type T, int RowsPerTile, int TileCols, bool Persistent, bool Prefetch>
static inline bool launch_esimd_tier3_typed(const ggml_sycl::mmvq_bench_args & args,
                                            std::vector<sycl::event> * events,
                                            std::string & error) {
    static_assert(TileCols == 16, "ESIMD tier3 kernels assume 16 output columns.");
    static_assert(RowsPerTile >= 1 && RowsPerTile <= 8, "ESIMD tier3 RepeatCount must be in [1, 8].");

    if (!validate_tier3_args(args, error)) {
        return false;
    }

    const int row_low  = static_cast<int>(args.row_low);
    const int row_high = static_cast<int>(args.row_high);
    const int rows_span = row_high - row_low;
    const int num_row_tiles = (rows_span + RowsPerTile - 1) / RowsPerTile;
    const int num_col_tiles = (static_cast<int>(args.batch) + TileCols - 1) / TileCols;

    const int ncols_x = static_cast<int>(args.ncols);
    const int ncols_y = static_cast<int>(args.batch);
    const int nrows_dst = static_cast<int>(args.dst_row_stride);
    const int src1_stride_blocks = static_cast<int>(args.src1_padded_col_size / XMX_K);

    const int block_size = block_traits<T>::block_size;
    const int blocks_per_row = ncols_x / block_size;

    const void * weights = args.weights;
    const void * activations = args.activations;
    float * output = args.output;

    sycl::queue & queue = *args.stream;
    int grid_cols = num_col_tiles;
    if constexpr (Persistent) {
        grid_cols = std::min(grid_cols, 4);
        if (grid_cols < 1) {
            grid_cols = 1;
        }
    }
    const sycl::range<2> grid(num_row_tiles, grid_cols);
    const sycl::range<2> block(1, 1);

    sycl::event ev = queue.submit([&](sycl::handler & h) {
        h.parallel_for<mmvq_esimd_tier3_kernel<static_cast<int>(T), RowsPerTile, TileCols, Persistent, Prefetch>>(
            sycl::nd_range<2>(grid * block, block),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                constexpr int HALF_COLS = XMX_N;
                constexpr int NUM_HALVES = TileCols / HALF_COLS;
                constexpr int SUBBLOCKS = block_traits<T>::block_size / XMX_K;
                constexpr int PASSES = block_traits<T>::per_16 ? 2 : 1;
                constexpr int AN = RowsPerTile * XMX_K;
                constexpr int BN = XMX_K * HALF_COLS;

                const int tile_row = static_cast<int>(item.get_global_id(0));
                const int tile_col_start = static_cast<int>(item.get_global_id(1));
                const int row_base = row_low + tile_row * RowsPerTile;

                if (row_base >= row_high) {
                    return;
                }

                const int num_k_blocks = ncols_x / XMX_K;
                const char * src1 = static_cast<const char *>(activations);
                constexpr int Q8_1_BLOCK_SIZE = sizeof(block_q8_1);
                for (int tile_col = tile_col_start; tile_col < num_col_tiles; tile_col += grid_cols) {
                    const int col_base = tile_col * TileCols;
                    if (col_base >= ncols_y) {
                        continue;
                    }

                    float acc[RowsPerTile][TileCols];
                    for (int r = 0; r < RowsPerTile; ++r) {
                        for (int c = 0; c < TileCols; ++c) {
                            acc[r][c] = 0.0f;
                        }
                    }

                    for (int kb = 0; kb < num_k_blocks; ++kb) {
                        if constexpr (Prefetch) {
                            const auto prefetch_props =
                                esimd::properties{esimd::cache_hint_L1<esimd::cache_hint::cached>,
                                                  esimd::cache_hint_L2<esimd::cache_hint::cached>,
                                                  esimd::alignment<4>};
                            const int prefetch_kb = kb + 1;
                            if (prefetch_kb < num_k_blocks) {
                                for (int c = 0; c < TileCols; ++c) {
                                    const int col = col_base + c;
                                    if (col < ncols_y) {
                                        const char * pf_ptr =
                                            src1 + (static_cast<int64_t>(col) * src1_stride_blocks + prefetch_kb) * Q8_1_BLOCK_SIZE;
                                        const uintptr_t addr = reinterpret_cast<uintptr_t>(pf_ptr);
                                        if ((addr & 0x3u) == 0u) {
                                            auto pf_ptr_u32 = reinterpret_cast<const uint32_t *>(pf_ptr);
                                            esimd::prefetch(pf_ptr_u32, prefetch_props);
                                        }
                                    }
                                }
                            }
                        }

                        int8_t a_vals[RowsPerTile][XMX_K];
                        AParams params[RowsPerTile];
                        for (int r = 0; r < RowsPerTile; ++r) {
                            const int row = row_base + r;
                            if (row < row_high) {
                                const int block_idx = kb / SUBBLOCKS;
                                const int sub_idx = kb - block_idx * SUBBLOCKS;
                                decode_block_aos<T>(weights, row, blocks_per_row, block_idx, sub_idx, a_vals[r], params[r]);
                            } else {
#pragma unroll
                                for (int k = 0; k < XMX_K; ++k) {
                                    a_vals[r][k] = 0;
                                }
                                params[r].scale0  = 0.0f;
                                params[r].offset0 = 0.0f;
                                params[r].scale1  = 0.0f;
                                params[r].offset1 = 0.0f;
                            }
                        }

                        int8_t b_vals[TileCols][XMX_K];
                        float d_B[TileCols];
                        float s_B[2][TileCols];
                        for (int c = 0; c < TileCols; ++c) {
                            const int col = col_base + c;
                            if (col < ncols_y) {
                                const char * block_ptr = src1 +
                                    (static_cast<int64_t>(col) * src1_stride_blocks + kb) * Q8_1_BLOCK_SIZE;
                                const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
                                const sycl::half s = *reinterpret_cast<const sycl::half *>(block_ptr + 2);
                                d_B[c] = float(d);
                                const int8_t * qs = reinterpret_cast<const int8_t *>(block_ptr + 4);
                                if constexpr (T == GGML_TYPE_Q2_K) {
                                    float sum0 = 0.0f;
                                    float sum1 = 0.0f;
                                    for (int k = 0; k < XMX_K; ++k) {
                                        const int8_t v = qs[k];
                                        if (k < 16) {
                                            sum0 += v;
                                        } else {
                                            sum1 += v;
                                        }
                                    }
                                    s_B[0][c] = d_B[c] * sum0;
                                    s_B[1][c] = d_B[c] * sum1;
                                } else {
                                    s_B[0][c] = float(s);
                                    s_B[1][c] = 0.0f;
                                }
#pragma unroll
                                for (int k = 0; k < XMX_K; ++k) {
                                    b_vals[c][k] = qs[k];
                                }
                            } else {
                                d_B[c] = 0.0f;
                                s_B[0][c] = 0.0f;
                                s_B[1][c] = 0.0f;
#pragma unroll
                                for (int k = 0; k < XMX_K; ++k) {
                                    b_vals[c][k] = 0;
                                }
                            }
                        }

                        for (int pass = 0; pass < PASSES; ++pass) {
                            esimd::simd<int8_t, AN> a_vec;
                            int a_idx = 0;
                            for (int r = 0; r < RowsPerTile; ++r) {
                                for (int k = 0; k < XMX_K; ++k) {
                                    int8_t v = a_vals[r][k];
                                    if constexpr (PASSES == 2) {
                                        const bool keep = (pass == 0) ? (k < 16) : (k >= 16);
                                        v = keep ? v : 0;
                                    }
                                    a_vec[a_idx++] = v;
                                }
                            }

                            for (int half = 0; half < NUM_HALVES; ++half) {
                                esimd::simd<int8_t, BN> b_vec;
                                int b_idx = 0;
                                const int col_offset = half * HALF_COLS;
                                for (int k4 = 0; k4 < (XMX_K / 4); ++k4) {
                                    for (int j = 0; j < HALF_COLS; ++j) {
                                        const int col = col_offset + j;
                                        const int base = k4 * 4;
                                        b_vec[b_idx++] = b_vals[col][base + 0];
                                        b_vec[b_idx++] = b_vals[col][base + 1];
                                        b_vec[b_idx++] = b_vals[col][base + 2];
                                        b_vec[b_idx++] = b_vals[col][base + 3];
                                    }
                                }

                                auto c_vec = esimd_xmx::dpas<8, RowsPerTile, int, int8_t, int8_t>(b_vec, a_vec);
                                for (int r = 0; r < RowsPerTile; ++r) {
                                    const float d_A = (pass == 0) ? params[r].scale0 : params[r].scale1;
                                    const float o_A = (pass == 0) ? params[r].offset0 : params[r].offset1;
                                    for (int j = 0; j < HALF_COLS; ++j) {
                                        const int col = col_offset + j;
                                        const int idx = r * HALF_COLS + j;
                                        const float C_ij = static_cast<float>(c_vec[idx]);
                                        float s_val = 0.0f;
                                        if constexpr (T == GGML_TYPE_Q2_K) {
                                            s_val = s_B[pass][col];
                                        } else {
                                            s_val = s_B[0][col];
                                        }
                                        acc[r][col] += d_A * (C_ij * d_B[col]) + o_A * s_val;
                                    }
                                }
                            }
                        }
                    }

                    for (int r = 0; r < RowsPerTile; ++r) {
                        const int row = row_base + r;
                        if (row >= row_high) {
                            continue;
                        }
                        for (int c = 0; c < TileCols; ++c) {
                            const int col = col_base + c;
                            if (col < ncols_y) {
                                output[col * nrows_dst + row] = acc[r][c];
                            }
                        }
                    }
                }
            });
    });

    if (events != nullptr) {
        events->push_back(ev);
    }
    return true;
}

template <int RowsPerTile, int TileCols, bool Persistent, bool Prefetch>
static inline bool launch_esimd_tier3_dispatch(const ggml_sycl::mmvq_bench_args & args,
                                               std::vector<sycl::event> * events,
                                               std::string & error) {
    switch (args.weight_type) {
        case GGML_TYPE_Q4_0:
            return launch_esimd_tier3_typed<GGML_TYPE_Q4_0, RowsPerTile, TileCols, Persistent, Prefetch>(args, events, error);
        case GGML_TYPE_Q8_0:
            return launch_esimd_tier3_typed<GGML_TYPE_Q8_0, RowsPerTile, TileCols, Persistent, Prefetch>(args, events, error);
        case GGML_TYPE_Q6_K:
            return launch_esimd_tier3_typed<GGML_TYPE_Q6_K, RowsPerTile, TileCols, Persistent, Prefetch>(args, events, error);
        case GGML_TYPE_MXFP4:
            return launch_esimd_tier3_typed<GGML_TYPE_MXFP4, RowsPerTile, TileCols, Persistent, Prefetch>(args, events, error);
        case GGML_TYPE_Q4_K:
            return launch_esimd_tier3_typed<GGML_TYPE_Q4_K, RowsPerTile, TileCols, Persistent, Prefetch>(args, events, error);
        case GGML_TYPE_Q2_K:
            return launch_esimd_tier3_typed<GGML_TYPE_Q2_K, RowsPerTile, TileCols, Persistent, Prefetch>(args, events, error);
        case GGML_TYPE_Q3_K:
            return launch_esimd_tier3_typed<GGML_TYPE_Q3_K, RowsPerTile, TileCols, Persistent, Prefetch>(args, events, error);
        case GGML_TYPE_Q5_K:
            return launch_esimd_tier3_typed<GGML_TYPE_Q5_K, RowsPerTile, TileCols, Persistent, Prefetch>(args, events, error);
        default:
            error = "Tier3 ESIMD kernel: unsupported quant type.";
            return false;
    }
}

inline bool launch_esimd_large_tile(const ggml_sycl::mmvq_bench_args & args,
                                    std::vector<sycl::event> * events,
                                    std::string & error) {
    return launch_esimd_tier3_dispatch<8, 16, false, false>(args, events, error);
}

inline bool launch_esimd_persistent(const ggml_sycl::mmvq_bench_args & args,
                                    std::vector<sycl::event> * events,
                                    std::string & error) {
    return launch_esimd_tier3_dispatch<8, 16, true, false>(args, events, error);
}

inline bool launch_esimd_lsc_prefetch(const ggml_sycl::mmvq_bench_args & args,
                                      std::vector<sycl::event> * events,
                                      std::string & error) {
    return launch_esimd_tier3_dispatch<4, 16, false, true>(args, events, error);
}

#else

inline bool launch_esimd_large_tile(const ggml_sycl::mmvq_bench_args &,
                                    std::vector<sycl::event> *,
                                    std::string & error) {
    error = "SYCL ESIMD unavailable; tier3 ESIMD kernels disabled.";
    return false;
}

inline bool launch_esimd_persistent(const ggml_sycl::mmvq_bench_args &,
                                    std::vector<sycl::event> *,
                                    std::string & error) {
    error = "SYCL ESIMD unavailable; tier3 ESIMD kernels disabled.";
    return false;
}

inline bool launch_esimd_lsc_prefetch(const ggml_sycl::mmvq_bench_args &,
                                      std::vector<sycl::event> *,
                                      std::string & error) {
    error = "SYCL ESIMD unavailable; tier3 ESIMD kernels disabled.";
    return false;
}

#endif

#else

inline bool launch_xmx_tile_64x64(const ggml_sycl::mmvq_bench_args &,
                                  std::vector<sycl::event> *,
                                  std::string & error) {
    error = "SYCL joint_matrix unavailable; tier3 kernels disabled.";
    return false;
}

inline bool launch_xmx_register_accum(const ggml_sycl::mmvq_bench_args &,
                                      std::vector<sycl::event> *,
                                      std::string & error) {
    error = "SYCL joint_matrix unavailable; tier3 kernels disabled.";
    return false;
}

inline bool launch_xmx_multi_wg(const ggml_sycl::mmvq_bench_args &,
                                std::vector<sycl::event> *,
                                std::string & error) {
    error = "SYCL joint_matrix unavailable; tier3 kernels disabled.";
    return false;
}

inline bool launch_xmx_persistent(const ggml_sycl::mmvq_bench_args &,
                                  std::vector<sycl::event> *,
                                  std::string & error) {
    error = "SYCL joint_matrix unavailable; tier3 kernels disabled.";
    return false;
}

inline bool launch_esimd_large_tile(const ggml_sycl::mmvq_bench_args &,
                                    std::vector<sycl::event> *,
                                    std::string & error) {
    error = "SYCL ESIMD unavailable; tier3 ESIMD kernels disabled.";
    return false;
}

inline bool launch_esimd_persistent(const ggml_sycl::mmvq_bench_args &,
                                    std::vector<sycl::event> *,
                                    std::string & error) {
    error = "SYCL ESIMD unavailable; tier3 ESIMD kernels disabled.";
    return false;
}

inline bool launch_esimd_lsc_prefetch(const ggml_sycl::mmvq_bench_args &,
                                      std::vector<sycl::event> *,
                                      std::string & error) {
    error = "SYCL ESIMD unavailable; tier3 ESIMD kernels disabled.";
    return false;
}

#endif

}  // namespace mmvq_tier3
}  // namespace sycl_bench
