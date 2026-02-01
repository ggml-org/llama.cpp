#pragma once

#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "ggml.h"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/ggml-sycl-bench.hpp"

#if __has_include(<sycl/ext/oneapi/matrix/matrix.hpp>)
#    include <sycl/ext/oneapi/matrix/matrix.hpp>
#    define MMVQ_TIER2_XMX_AVAILABLE 1
#else
#    define MMVQ_TIER2_XMX_AVAILABLE 0
#endif

namespace sycl_bench {
namespace mmvq_tier2 {

#if MMVQ_TIER2_XMX_AVAILABLE

namespace sycl_xmx = sycl::ext::oneapi::experimental::matrix;

constexpr int XMX_M       = 8;
constexpr int XMX_N       = 16;
constexpr int XMX_K       = 32;
constexpr int XMX_SG_SIZE = 16;

constexpr int SLM_A_SIZE        = XMX_M * XMX_K;
constexpr int SLM_B_SIZE        = XMX_K * XMX_N;
constexpr int SLM_C_SIZE        = XMX_M * XMX_N;
constexpr int SLM_SCALES_A_SIZE = XMX_M;
constexpr int SLM_SCALES_B_SIZE = XMX_N;
constexpr int SLM_SUMS_B_SIZE   = XMX_N;

constexpr int SLM_A_SIZE_DB        = SLM_A_SIZE * 2;
constexpr int SLM_B_SIZE_DB        = SLM_B_SIZE * 2;
constexpr int SLM_SCALES_A_SIZE_DB = SLM_SCALES_A_SIZE * 2;
constexpr int SLM_SCALES_B_SIZE_DB = SLM_SCALES_B_SIZE * 2;
constexpr int SLM_SUMS_B_SIZE_DB   = SLM_SUMS_B_SIZE * 2;

constexpr int CF2_COL_TILES = 2;
constexpr int CF4_COL_TILES = 4;

static inline bool validate_xmx_args(const ggml_sycl::mmvq_bench_args & args,
                                     std::string & error) {
    if (args.stream == nullptr) {
        error = "SYCL stream is null.";
        return false;
    }
    if (args.layout != GGML_LAYOUT_AOS) {
        error = "XMX tier2 kernels require AOS weight layout.";
        return false;
    }
    if (args.weight_type != GGML_TYPE_Q4_0 && args.weight_type != GGML_TYPE_Q8_0) {
        error = "XMX tier2 kernels support Q4_0 and Q8_0 only.";
        return false;
    }
    if (args.ncols <= 0 || args.nrows <= 0 || args.batch <= 0) {
        error = "Invalid dimensions for XMX tier2 kernel.";
        return false;
    }
    if ((args.ncols % XMX_K) != 0) {
        error = "K dimension must be multiple of 32 for XMX tier2 kernel.";
        return false;
    }
    if ((args.src1_padded_col_size % XMX_K) != 0) {
        error = "Padded K dimension must be multiple of 32 for XMX tier2 kernel.";
        return false;
    }
    if (args.src1_padded_col_size < args.ncols) {
        error = "Padded K dimension is smaller than K for XMX tier2 kernel.";
        return false;
    }
    return true;
}

static inline bool validate_xmx_soa_args(const ggml_sycl::mmvq_bench_args & args,
                                         std::string & error) {
    if (args.stream == nullptr) {
        error = "SYCL stream is null.";
        return false;
    }
    if (args.layout != GGML_LAYOUT_SOA) {
        error = "XMX tier2 SoA kernels require SOA weight layout.";
        return false;
    }
    if (args.layout_base == nullptr) {
        error = "XMX tier2 SoA kernels require layout_base pointer.";
        return false;
    }
    if (args.weight_type != GGML_TYPE_Q4_0 && args.weight_type != GGML_TYPE_Q8_0) {
        error = "XMX tier2 SoA kernels support Q4_0 and Q8_0 only.";
        return false;
    }
    if (args.ncols <= 0 || args.nrows <= 0 || args.batch <= 0) {
        error = "Invalid dimensions for XMX tier2 SoA kernel.";
        return false;
    }
    if ((args.ncols % XMX_K) != 0) {
        error = "K dimension must be multiple of 32 for XMX tier2 SoA kernel.";
        return false;
    }
    if ((args.src1_padded_col_size % XMX_K) != 0) {
        error = "Padded K dimension must be multiple of 32 for XMX tier2 SoA kernel.";
        return false;
    }
    if (args.src1_padded_col_size < args.ncols) {
        error = "Padded K dimension is smaller than K for XMX tier2 SoA kernel.";
        return false;
    }
    return true;
}

static inline void get_soa_weight_ptrs(const ggml_sycl::mmvq_bench_args & args,
                                       const uint8_t *& weight_qs,
                                       const sycl::half *& weight_d) {
    const uint8_t * base = static_cast<const uint8_t *>(args.layout_base);
    if (args.weight_type == GGML_TYPE_Q4_0) {
        const size_t blocks_per_row = static_cast<size_t>(args.ncols / QK4_0);
        const size_t nblocks = blocks_per_row * static_cast<size_t>(args.nrows);
        weight_qs = base;
        weight_d = reinterpret_cast<const sycl::half *>(base + nblocks * (QK4_0 / 2));
        return;
    }
    const size_t blocks_per_row = static_cast<size_t>(args.ncols / QK8_0);
    const size_t nblocks = blocks_per_row * static_cast<size_t>(args.nrows);
    weight_qs = base;
    weight_d = reinterpret_cast<const sycl::half *>(base + nblocks * QK8_0);
}

static inline void q4_0_tile_kernel(const void * __restrict__ vx,
                                    const void * __restrict__ vy,
                                    float * __restrict__ dst,
                                    const int ncols_x,
                                    const int ncols_y,
                                    const int nrows_dst,
                                    const int src1_stride_blocks,
                                    const int row_low,
                                    const int row_high,
                                    int8_t * __restrict__ slm_A,
                                    int8_t * __restrict__ slm_B,
                                    int32_t * __restrict__ slm_C,
                                    float * __restrict__ slm_scales_A,
                                    float * __restrict__ slm_scales_B,
                                    float * __restrict__ slm_sums_B,
                                    sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];

    const int row_base = item.get_group(0) * XMX_M + row_low;
    const int col_base = item.get_group(1) * XMX_N;

    if (row_base >= row_high || col_base >= ncols_y) {
        return;
    }

    const int num_k_blocks = ncols_x / XMX_K;

    const char * src0 = static_cast<const char *>(vx);
    const char * src1 = static_cast<const char *>(vy);
    constexpr int Q4_0_BLOCK_SIZE = sizeof(block_q4_0);
    constexpr int Q8_1_BLOCK_SIZE = sizeof(block_q8_1);

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        if (lane_id < XMX_M) {
            const int row = row_base + lane_id;
            if (row < row_high) {
                const int64_t block_idx = static_cast<int64_t>(row) * num_k_blocks + k_block;
                const char * block_ptr = src0 + block_idx * Q4_0_BLOCK_SIZE;

                const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
                slm_scales_A[lane_id] = float(d);

                const uint8_t * nibbles  = reinterpret_cast<const uint8_t *>(block_ptr + 2);
                const int       base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < 16; j++) {
                    const uint8_t packed = nibbles[j];
                    slm_A[base_idx + j]      = static_cast<int8_t>(packed & 0x0F);
                    slm_A[base_idx + j + 16] = static_cast<int8_t>(packed >> 4);
                }
            } else {
                slm_scales_A[lane_id] = 0.0f;
                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        {
            const int col = col_base + lane_id;
            if (col < ncols_y) {
                const char * block_ptr = src1 +
                    (static_cast<int64_t>(col) * src1_stride_blocks + k_block) * Q8_1_BLOCK_SIZE;
                const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
                const sycl::half s = *reinterpret_cast<const sycl::half *>(block_ptr + 2);

                slm_scales_B[lane_id] = float(d);
                slm_sums_B[lane_id]   = float(s);

                const int8_t * qs = reinterpret_cast<const int8_t *>(block_ptr + 4);
                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[base_idx + k] = qs[k];
                }
            } else {
                slm_scales_B[lane_id] = 0.0f;
                slm_sums_B[lane_id]   = 0.0f;
                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[base_idx + k] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
            matA;
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
            matB;
        sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

        auto A_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                              sycl::access::decorated::no>(slm_A);
        auto B_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                              sycl::access::decorated::no>(slm_B);
        auto C_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
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
            const float d_A  = slm_scales_A[i];
            const float d_B  = slm_scales_B[j];
            const float s_B  = slm_sums_B[j];
            const float C_ij = static_cast<float>(slm_C[idx]);
            acc[acc_idx] += d_A * (C_ij * d_B - 8.0f * s_B);
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

static inline void q8_0_tile_kernel(const void * __restrict__ vx,
                                    const void * __restrict__ vy,
                                    float * __restrict__ dst,
                                    const int ncols_x,
                                    const int ncols_y,
                                    const int nrows_dst,
                                    const int src1_stride_blocks,
                                    const int row_low,
                                    const int row_high,
                                    int8_t * __restrict__ slm_A,
                                    int8_t * __restrict__ slm_B,
                                    int32_t * __restrict__ slm_C,
                                    float * __restrict__ slm_scales_A,
                                    float * __restrict__ slm_scales_B,
                                    sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];

    const int row_base = item.get_group(0) * XMX_M + row_low;
    const int col_base = item.get_group(1) * XMX_N;

    if (row_base >= row_high || col_base >= ncols_y) {
        return;
    }

    const int num_k_blocks = ncols_x / XMX_K;

    const char * src0 = static_cast<const char *>(vx);
    const char * src1 = static_cast<const char *>(vy);
    constexpr int Q8_0_BLOCK_SIZE = sizeof(block_q8_0);
    constexpr int Q8_1_BLOCK_SIZE = sizeof(block_q8_1);

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        if (lane_id < XMX_M) {
            const int row = row_base + lane_id;
            if (row < row_high) {
                const int64_t block_idx = static_cast<int64_t>(row) * num_k_blocks + k_block;
                const char * block_ptr = src0 + block_idx * Q8_0_BLOCK_SIZE;

                const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
                slm_scales_A[lane_id] = float(d);

                const int8_t * qs = reinterpret_cast<const int8_t *>(block_ptr + 2);
                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = qs[j];
                }
            } else {
                slm_scales_A[lane_id] = 0.0f;
                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        {
            const int col = col_base + lane_id;
            if (col < ncols_y) {
                const char * block_ptr = src1 +
                    (static_cast<int64_t>(col) * src1_stride_blocks + k_block) * Q8_1_BLOCK_SIZE;
                const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);

                slm_scales_B[lane_id] = float(d);

                const int8_t * qs = reinterpret_cast<const int8_t *>(block_ptr + 4);
                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[base_idx + k] = qs[k];
                }
            } else {
                slm_scales_B[lane_id] = 0.0f;
                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[base_idx + k] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
            matA;
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
            matB;
        sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

        auto A_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                              sycl::access::decorated::no>(slm_A);
        auto B_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                              sycl::access::decorated::no>(slm_B);
        auto C_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
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
            const float d_A  = slm_scales_A[i];
            const float d_B  = slm_scales_B[j];
            const float C_ij = static_cast<float>(slm_C[idx]);
            acc[acc_idx] += d_A * d_B * C_ij;
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

static inline void q4_0_cf2_kernel(const void * __restrict__ vx,
                                   const void * __restrict__ vy,
                                   float * __restrict__ dst,
                                   const int ncols_x,
                                   const int ncols_y,
                                   const int nrows_dst,
                                   const int src1_stride_blocks,
                                   const int row_low,
                                   const int row_high,
                                   int8_t * __restrict__ slm_A,
                                   int8_t * __restrict__ slm_B,
                                   int32_t * __restrict__ slm_C,
                                   float * __restrict__ slm_scales_A,
                                   float * __restrict__ slm_scales_B,
                                   float * __restrict__ slm_sums_B,
                                   sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  sg_id   = item.get_local_id(0);
    const int  lane_id = sg.get_local_id()[0];

    const int row_base  = item.get_group(0) * XMX_M + row_low;
    const int col_group = item.get_group(1);
    const int col_base  = col_group * (CF2_COL_TILES * XMX_N) + sg_id * XMX_N;

    if (row_base >= row_high) {
        return;
    }

    const int  num_k_blocks = ncols_x / XMX_K;
    const bool active_sg    = (col_base < ncols_y);

    const char * src0 = static_cast<const char *>(vx);
    const char * src1 = static_cast<const char *>(vy);
    constexpr int Q4_0_BLOCK_SIZE = sizeof(block_q4_0);
    constexpr int Q8_1_BLOCK_SIZE = sizeof(block_q8_1);

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    int8_t *  my_B_tile   = slm_B + sg_id * (XMX_K * XMX_N);
    int32_t * my_C_tile   = slm_C + sg_id * (XMX_M * XMX_N);
    float *   my_scales_B = slm_scales_B + sg_id * XMX_N;
    float *   my_sums_B   = slm_sums_B + sg_id * XMX_N;

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        if (sg_id == 0 && lane_id < XMX_M) {
            const int row = row_base + lane_id;
            if (row < row_high) {
                const int64_t block_idx = static_cast<int64_t>(row) * num_k_blocks + k_block;
                const char * block_ptr = src0 + block_idx * Q4_0_BLOCK_SIZE;

                const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
                slm_scales_A[lane_id] = float(d);

                const uint8_t * nibbles  = reinterpret_cast<const uint8_t *>(block_ptr + 2);
                const int       base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < 16; j++) {
                    const uint8_t packed = nibbles[j];
                    slm_A[base_idx + j]      = static_cast<int8_t>(packed & 0x0F);
                    slm_A[base_idx + j + 16] = static_cast<int8_t>(packed >> 4);
                }
            } else {
                slm_scales_A[lane_id] = 0.0f;
                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        if (active_sg) {
            const int col = col_base + lane_id;
            if (col < ncols_y) {
                const char * block_ptr =
                    src1 + (static_cast<int64_t>(col) * src1_stride_blocks + k_block) * Q8_1_BLOCK_SIZE;
                const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
                const sycl::half s = *reinterpret_cast<const sycl::half *>(block_ptr + 2);

                my_scales_B[lane_id] = float(d);
                my_sums_B[lane_id]   = float(s);

                const int8_t * qs = reinterpret_cast<const int8_t *>(block_ptr + 4);
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    my_B_tile[k + lane_id * XMX_K] = qs[k];
                }
            } else {
                my_scales_B[lane_id] = 0.0f;
                my_sums_B[lane_id]   = 0.0f;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    my_B_tile[k + lane_id * XMX_K] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        if (active_sg) {
            auto A_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                    slm_A);
            auto B_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                    my_B_tile);
            auto C_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                    my_C_tile);

            sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K,
                                   sycl_xmx::layout::row_major>
                matA;
            sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N,
                                   sycl_xmx::layout::col_major>
                matB;
            sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

            sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
            sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
            sycl_xmx::joint_matrix_fill(sg, matC, 0);
            sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
            sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

#pragma unroll
            for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
                const int   i    = idx / XMX_N;
                const int   j    = idx % XMX_N;
                const float d_A  = slm_scales_A[i];
                const float d_B  = my_scales_B[j];
                const float s_B  = my_sums_B[j];
                const float C_ij = static_cast<float>(my_C_tile[idx]);
                acc[acc_idx] += d_A * (C_ij * d_B - 8.0f * s_B);
            }
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

    if (active_sg) {
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
}

static inline void q8_0_cf2_kernel(const void * __restrict__ vx,
                                   const void * __restrict__ vy,
                                   float * __restrict__ dst,
                                   const int ncols_x,
                                   const int ncols_y,
                                   const int nrows_dst,
                                   const int src1_stride_blocks,
                                   const int row_low,
                                   const int row_high,
                                   int8_t * __restrict__ slm_A,
                                   int8_t * __restrict__ slm_B,
                                   int32_t * __restrict__ slm_C,
                                   float * __restrict__ slm_scales_A,
                                   float * __restrict__ slm_scales_B,
                                   sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  sg_id   = item.get_local_id(0);
    const int  lane_id = sg.get_local_id()[0];

    const int row_base  = item.get_group(0) * XMX_M + row_low;
    const int col_group = item.get_group(1);
    const int col_base  = col_group * (CF2_COL_TILES * XMX_N) + sg_id * XMX_N;

    if (row_base >= row_high) {
        return;
    }

    const int  num_k_blocks = ncols_x / XMX_K;
    const bool active_sg    = (col_base < ncols_y);

    const char * src0 = static_cast<const char *>(vx);
    const char * src1 = static_cast<const char *>(vy);
    constexpr int Q8_0_BLOCK_SIZE = sizeof(block_q8_0);
    constexpr int Q8_1_BLOCK_SIZE = sizeof(block_q8_1);

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    int8_t *  my_B_tile   = slm_B + sg_id * (XMX_K * XMX_N);
    int32_t * my_C_tile   = slm_C + sg_id * (XMX_M * XMX_N);
    float *   my_scales_B = slm_scales_B + sg_id * XMX_N;

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        if (sg_id == 0 && lane_id < XMX_M) {
            const int row = row_base + lane_id;
            if (row < row_high) {
                const int64_t block_idx = static_cast<int64_t>(row) * num_k_blocks + k_block;
                const char * block_ptr = src0 + block_idx * Q8_0_BLOCK_SIZE;

                const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
                slm_scales_A[lane_id] = float(d);

                const int8_t * qs = reinterpret_cast<const int8_t *>(block_ptr + 2);
                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = qs[j];
                }
            } else {
                slm_scales_A[lane_id] = 0.0f;
                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        if (active_sg) {
            const int col = col_base + lane_id;
            if (col < ncols_y) {
                const char * block_ptr =
                    src1 + (static_cast<int64_t>(col) * src1_stride_blocks + k_block) * Q8_1_BLOCK_SIZE;
                const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);

                my_scales_B[lane_id] = float(d);

                const int8_t * qs = reinterpret_cast<const int8_t *>(block_ptr + 4);
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    my_B_tile[k + lane_id * XMX_K] = qs[k];
                }
            } else {
                my_scales_B[lane_id] = 0.0f;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    my_B_tile[k + lane_id * XMX_K] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        if (active_sg) {
            auto A_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                    slm_A);
            auto B_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                    my_B_tile);
            auto C_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                    my_C_tile);

            sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K,
                                   sycl_xmx::layout::row_major>
                matA;
            sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N,
                                   sycl_xmx::layout::col_major>
                matB;
            sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

            sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
            sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
            sycl_xmx::joint_matrix_fill(sg, matC, 0);
            sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
            sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

#pragma unroll
            for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
                const int   i    = idx / XMX_N;
                const int   j    = idx % XMX_N;
                const float d_A  = slm_scales_A[i];
                const float d_B  = my_scales_B[j];
                const float C_ij = static_cast<float>(my_C_tile[idx]);
                acc[acc_idx] += d_A * d_B * C_ij;
            }
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

    if (active_sg) {
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
}

static inline void q4_0_cf4_kernel(const void * __restrict__ vx,
                                   const void * __restrict__ vy,
                                   float * __restrict__ dst,
                                   const int ncols_x,
                                   const int ncols_y,
                                   const int nrows_dst,
                                   const int src1_stride_blocks,
                                   const int row_low,
                                   const int row_high,
                                   int8_t * __restrict__ slm_A,
                                   int8_t * __restrict__ slm_B,
                                   int32_t * __restrict__ slm_C,
                                   float * __restrict__ slm_scales_A,
                                   float * __restrict__ slm_scales_B,
                                   float * __restrict__ slm_sums_B,
                                   sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];

    const int row_base  = item.get_group(0) * XMX_M + row_low;
    const int col_group = item.get_group(1);
    const int col_base  = col_group * (CF4_COL_TILES * XMX_N);

    if (row_base >= row_high) {
        return;
    }

    const int num_k_blocks = ncols_x / XMX_K;

    const char * src0 = static_cast<const char *>(vx);
    const char * src1 = static_cast<const char *>(vy);
    constexpr int Q4_0_BLOCK_SIZE = sizeof(block_q4_0);
    constexpr int Q8_1_BLOCK_SIZE = sizeof(block_q8_1);

    float acc[CF4_COL_TILES][8] = {};

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        if (lane_id < XMX_M) {
            const int row = row_base + lane_id;
            if (row < row_high) {
                const int64_t block_idx = static_cast<int64_t>(row) * num_k_blocks + k_block;
                const char * block_ptr = src0 + block_idx * Q4_0_BLOCK_SIZE;

                const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
                slm_scales_A[lane_id] = float(d);

                const uint8_t * nibbles  = reinterpret_cast<const uint8_t *>(block_ptr + 2);
                const int       base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < 16; j++) {
                    const uint8_t packed = nibbles[j];
                    slm_A[base_idx + j]      = static_cast<int8_t>(packed & 0x0F);
                    slm_A[base_idx + j + 16] = static_cast<int8_t>(packed >> 4);
                }
            } else {
                slm_scales_A[lane_id] = 0.0f;
                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        for (int tile = 0; tile < CF4_COL_TILES; tile++) {
            const int col_tile = col_base + tile * XMX_N;
            const bool active_tile = (col_tile < ncols_y);
            const int col = col_tile + lane_id;
            if (col < ncols_y) {
                const char * block_ptr =
                    src1 + (static_cast<int64_t>(col) * src1_stride_blocks + k_block) * Q8_1_BLOCK_SIZE;
                const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
                const sycl::half s = *reinterpret_cast<const sycl::half *>(block_ptr + 2);

                slm_scales_B[lane_id] = float(d);
                slm_sums_B[lane_id]   = float(s);

                const int8_t * qs = reinterpret_cast<const int8_t *>(block_ptr + 4);
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = qs[k];
                }
            } else {
                slm_scales_B[lane_id] = 0.0f;
                slm_sums_B[lane_id]   = 0.0f;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = 0;
                }
            }

            item.barrier(sycl::access::fence_space::local_space);

            if (active_tile) {
                auto A_ptr =
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        slm_A);
                auto B_ptr =
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        slm_B);
                auto C_ptr =
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        slm_C);

                sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K,
                                       sycl_xmx::layout::row_major>
                    matA;
                sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N,
                                       sycl_xmx::layout::col_major>
                    matB;
                sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

                sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
                sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
                sycl_xmx::joint_matrix_fill(sg, matC, 0);
                sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
                sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

#pragma unroll
                for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
                    const int   i    = idx / XMX_N;
                    const int   j    = idx % XMX_N;
                    const float d_A  = slm_scales_A[i];
                    const float d_B  = slm_scales_B[j];
                    const float s_B  = slm_sums_B[j];
                    const float C_ij = static_cast<float>(slm_C[idx]);
                    acc[tile][acc_idx] += d_A * (C_ij * d_B - 8.0f * s_B);
                }
            }

            item.barrier(sycl::access::fence_space::local_space);
        }
    }

    for (int tile = 0; tile < CF4_COL_TILES; tile++) {
        const int col_tile = col_base + tile * XMX_N;
        if (col_tile >= ncols_y) {
            continue;
        }
#pragma unroll
        for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
            const int i   = idx / XMX_N;
            const int j   = idx % XMX_N;
            const int row = row_base + i;
            const int col = col_tile + j;
            if (row < row_high && col < ncols_y) {
                dst[col * nrows_dst + row] = acc[tile][acc_idx];
            }
        }
    }
}

static inline void q8_0_cf4_kernel(const void * __restrict__ vx,
                                   const void * __restrict__ vy,
                                   float * __restrict__ dst,
                                   const int ncols_x,
                                   const int ncols_y,
                                   const int nrows_dst,
                                   const int src1_stride_blocks,
                                   const int row_low,
                                   const int row_high,
                                   int8_t * __restrict__ slm_A,
                                   int8_t * __restrict__ slm_B,
                                   int32_t * __restrict__ slm_C,
                                   float * __restrict__ slm_scales_A,
                                   float * __restrict__ slm_scales_B,
                                   sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];

    const int row_base  = item.get_group(0) * XMX_M + row_low;
    const int col_group = item.get_group(1);
    const int col_base  = col_group * (CF4_COL_TILES * XMX_N);

    if (row_base >= row_high) {
        return;
    }

    const int num_k_blocks = ncols_x / XMX_K;

    const char * src0 = static_cast<const char *>(vx);
    const char * src1 = static_cast<const char *>(vy);
    constexpr int Q8_0_BLOCK_SIZE = sizeof(block_q8_0);
    constexpr int Q8_1_BLOCK_SIZE = sizeof(block_q8_1);

    float acc[CF4_COL_TILES][8] = {};

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        if (lane_id < XMX_M) {
            const int row = row_base + lane_id;
            if (row < row_high) {
                const int64_t block_idx = static_cast<int64_t>(row) * num_k_blocks + k_block;
                const char * block_ptr = src0 + block_idx * Q8_0_BLOCK_SIZE;

                const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
                slm_scales_A[lane_id] = float(d);

                const int8_t * qs = reinterpret_cast<const int8_t *>(block_ptr + 2);
                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = qs[j];
                }
            } else {
                slm_scales_A[lane_id] = 0.0f;
                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        for (int tile = 0; tile < CF4_COL_TILES; tile++) {
            const int col_tile = col_base + tile * XMX_N;
            const bool active_tile = (col_tile < ncols_y);
            const int col = col_tile + lane_id;
            if (col < ncols_y) {
                const char * block_ptr =
                    src1 + (static_cast<int64_t>(col) * src1_stride_blocks + k_block) * Q8_1_BLOCK_SIZE;
                const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);

                slm_scales_B[lane_id] = float(d);

                const int8_t * qs = reinterpret_cast<const int8_t *>(block_ptr + 4);
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = qs[k];
                }
            } else {
                slm_scales_B[lane_id] = 0.0f;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = 0;
                }
            }

            item.barrier(sycl::access::fence_space::local_space);

            if (active_tile) {
                auto A_ptr =
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        slm_A);
                auto B_ptr =
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        slm_B);
                auto C_ptr =
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        slm_C);

                sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K,
                                       sycl_xmx::layout::row_major>
                    matA;
                sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N,
                                       sycl_xmx::layout::col_major>
                    matB;
                sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

                sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
                sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
                sycl_xmx::joint_matrix_fill(sg, matC, 0);
                sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
                sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

#pragma unroll
                for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
                    const int   i    = idx / XMX_N;
                    const int   j    = idx % XMX_N;
                    const float d_A  = slm_scales_A[i];
                    const float d_B  = slm_scales_B[j];
                    const float C_ij = static_cast<float>(slm_C[idx]);
                    acc[tile][acc_idx] += d_A * d_B * C_ij;
                }
            }

            item.barrier(sycl::access::fence_space::local_space);
        }
    }

    for (int tile = 0; tile < CF4_COL_TILES; tile++) {
        const int col_tile = col_base + tile * XMX_N;
        if (col_tile >= ncols_y) {
            continue;
        }
#pragma unroll
        for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
            const int i   = idx / XMX_N;
            const int j   = idx % XMX_N;
            const int row = row_base + i;
            const int col = col_tile + j;
            if (row < row_high && col < ncols_y) {
                dst[col * nrows_dst + row] = acc[tile][acc_idx];
            }
        }
    }
}

static inline void q4_0_tile_db_kernel(const void * __restrict__ vx,
                                       const void * __restrict__ vy,
                                       float * __restrict__ dst,
                                       const int ncols_x,
                                       const int ncols_y,
                                       const int nrows_dst,
                                       const int src1_stride_blocks,
                                       const int row_low,
                                       const int row_high,
                                       int8_t * __restrict__ slm_A,
                                       int8_t * __restrict__ slm_B,
                                       int32_t * __restrict__ slm_C,
                                       float * __restrict__ slm_scales_A,
                                       float * __restrict__ slm_scales_B,
                                       float * __restrict__ slm_sums_B,
                                       sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];

    const int row_base = item.get_group(0) * XMX_M + row_low;
    const int col_base = item.get_group(1) * XMX_N;

    if (row_base >= row_high || col_base >= ncols_y) {
        return;
    }

    const int num_k_blocks = ncols_x / XMX_K;

    const char * src0 = static_cast<const char *>(vx);
    const char * src1 = static_cast<const char *>(vy);
    constexpr int Q4_0_BLOCK_SIZE = sizeof(block_q4_0);
    constexpr int Q8_1_BLOCK_SIZE = sizeof(block_q8_1);

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    int buf_load = 0;
    int buf_comp = 0;

    if (num_k_blocks > 0) {
        if (lane_id < XMX_M) {
            const int row = row_base + lane_id;
            if (row < row_high) {
                const int64_t block_idx = static_cast<int64_t>(row) * num_k_blocks;
                const char * block_ptr = src0 + block_idx * Q4_0_BLOCK_SIZE;
                const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
                slm_scales_A[buf_load * SLM_SCALES_A_SIZE + lane_id] = float(d);

                const uint8_t * nibbles  = reinterpret_cast<const uint8_t *>(block_ptr + 2);
                const int       base_idx = buf_load * SLM_A_SIZE + lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < 16; j++) {
                    const uint8_t packed = nibbles[j];
                    slm_A[base_idx + j]      = static_cast<int8_t>(packed & 0x0F);
                    slm_A[base_idx + j + 16] = static_cast<int8_t>(packed >> 4);
                }
            } else {
                slm_scales_A[buf_load * SLM_SCALES_A_SIZE + lane_id] = 0.0f;
                const int base_idx = buf_load * SLM_A_SIZE + lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        {
            const int col = col_base + lane_id;
            if (col < ncols_y) {
                const char * block_ptr = src1 +
                    (static_cast<int64_t>(col) * src1_stride_blocks) * Q8_1_BLOCK_SIZE;
                const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
                const sycl::half s = *reinterpret_cast<const sycl::half *>(block_ptr + 2);

                slm_scales_B[buf_load * SLM_SCALES_B_SIZE + lane_id] = float(d);
                slm_sums_B[buf_load * SLM_SUMS_B_SIZE + lane_id]     = float(s);

                const int8_t * qs = reinterpret_cast<const int8_t *>(block_ptr + 4);
                const int base_idx = buf_load * SLM_B_SIZE + lane_id * XMX_K;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[base_idx + k] = qs[k];
                }
            } else {
                slm_scales_B[buf_load * SLM_SCALES_B_SIZE + lane_id] = 0.0f;
                slm_sums_B[buf_load * SLM_SUMS_B_SIZE + lane_id]     = 0.0f;
                const int base_idx = buf_load * SLM_B_SIZE + lane_id * XMX_K;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[base_idx + k] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        buf_comp = k_block & 1;
        buf_load = (k_block + 1) & 1;

        if (k_block + 1 < num_k_blocks) {
            const int next_k = k_block + 1;

            if (lane_id < XMX_M) {
                const int row = row_base + lane_id;
                if (row < row_high) {
                    const int64_t block_idx = static_cast<int64_t>(row) * num_k_blocks + next_k;
                    const char * block_ptr = src0 + block_idx * Q4_0_BLOCK_SIZE;
                    const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
                    slm_scales_A[buf_load * SLM_SCALES_A_SIZE + lane_id] = float(d);

                    const uint8_t * nibbles  = reinterpret_cast<const uint8_t *>(block_ptr + 2);
                    const int       base_idx = buf_load * SLM_A_SIZE + lane_id * XMX_K;
#pragma unroll
                    for (int j = 0; j < 16; j++) {
                        const uint8_t packed = nibbles[j];
                        slm_A[base_idx + j]      = static_cast<int8_t>(packed & 0x0F);
                        slm_A[base_idx + j + 16] = static_cast<int8_t>(packed >> 4);
                    }
                } else {
                    slm_scales_A[buf_load * SLM_SCALES_A_SIZE + lane_id] = 0.0f;
                    const int base_idx = buf_load * SLM_A_SIZE + lane_id * XMX_K;
#pragma unroll
                    for (int j = 0; j < XMX_K; j++) {
                        slm_A[base_idx + j] = 0;
                    }
                }
            }

            {
                const int col = col_base + lane_id;
                if (col < ncols_y) {
                    const char * block_ptr = src1 +
                        (static_cast<int64_t>(col) * src1_stride_blocks + next_k) * Q8_1_BLOCK_SIZE;
                    const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
                    const sycl::half s = *reinterpret_cast<const sycl::half *>(block_ptr + 2);

                    slm_scales_B[buf_load * SLM_SCALES_B_SIZE + lane_id] = float(d);
                    slm_sums_B[buf_load * SLM_SUMS_B_SIZE + lane_id]     = float(s);

                    const int8_t * qs = reinterpret_cast<const int8_t *>(block_ptr + 4);
                    const int base_idx = buf_load * SLM_B_SIZE + lane_id * XMX_K;
#pragma unroll
                    for (int k = 0; k < XMX_K; k++) {
                        slm_B[base_idx + k] = qs[k];
                    }
                } else {
                    slm_scales_B[buf_load * SLM_SCALES_B_SIZE + lane_id] = 0.0f;
                    slm_sums_B[buf_load * SLM_SUMS_B_SIZE + lane_id]     = 0.0f;
                    const int base_idx = buf_load * SLM_B_SIZE + lane_id * XMX_K;
#pragma unroll
                    for (int k = 0; k < XMX_K; k++) {
                        slm_B[base_idx + k] = 0;
                    }
                }
            }
        }

        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
            matA;
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
            matB;
        sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

        auto A_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                              sycl::access::decorated::no>(slm_A + buf_comp * SLM_A_SIZE);
        auto B_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                              sycl::access::decorated::no>(slm_B + buf_comp * SLM_B_SIZE);
        auto C_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
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
            const float d_A  = slm_scales_A[buf_comp * SLM_SCALES_A_SIZE + i];
            const float d_B  = slm_scales_B[buf_comp * SLM_SCALES_B_SIZE + j];
            const float s_B  = slm_sums_B[buf_comp * SLM_SUMS_B_SIZE + j];
            const float C_ij = static_cast<float>(slm_C[idx]);
            acc[acc_idx] += d_A * (C_ij * d_B - 8.0f * s_B);
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

static inline void q8_0_tile_db_kernel(const void * __restrict__ vx,
                                       const void * __restrict__ vy,
                                       float * __restrict__ dst,
                                       const int ncols_x,
                                       const int ncols_y,
                                       const int nrows_dst,
                                       const int src1_stride_blocks,
                                       const int row_low,
                                       const int row_high,
                                       int8_t * __restrict__ slm_A,
                                       int8_t * __restrict__ slm_B,
                                       int32_t * __restrict__ slm_C,
                                       float * __restrict__ slm_scales_A,
                                       float * __restrict__ slm_scales_B,
                                       sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];

    const int row_base = item.get_group(0) * XMX_M + row_low;
    const int col_base = item.get_group(1) * XMX_N;

    if (row_base >= row_high || col_base >= ncols_y) {
        return;
    }

    const int num_k_blocks = ncols_x / XMX_K;

    const char * src0 = static_cast<const char *>(vx);
    const char * src1 = static_cast<const char *>(vy);
    constexpr int Q8_0_BLOCK_SIZE = sizeof(block_q8_0);
    constexpr int Q8_1_BLOCK_SIZE = sizeof(block_q8_1);

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    int buf_load = 0;
    int buf_comp = 0;

    if (num_k_blocks > 0) {
        if (lane_id < XMX_M) {
            const int row = row_base + lane_id;
            if (row < row_high) {
                const int64_t block_idx = static_cast<int64_t>(row) * num_k_blocks;
                const char * block_ptr = src0 + block_idx * Q8_0_BLOCK_SIZE;
                const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
                slm_scales_A[buf_load * SLM_SCALES_A_SIZE + lane_id] = float(d);

                const int8_t * qs = reinterpret_cast<const int8_t *>(block_ptr + 2);
                const int base_idx = buf_load * SLM_A_SIZE + lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = qs[j];
                }
            } else {
                slm_scales_A[buf_load * SLM_SCALES_A_SIZE + lane_id] = 0.0f;
                const int base_idx = buf_load * SLM_A_SIZE + lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        {
            const int col = col_base + lane_id;
            if (col < ncols_y) {
                const char * block_ptr = src1 +
                    (static_cast<int64_t>(col) * src1_stride_blocks) * Q8_1_BLOCK_SIZE;
                const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);

                slm_scales_B[buf_load * SLM_SCALES_B_SIZE + lane_id] = float(d);

                const int8_t * qs = reinterpret_cast<const int8_t *>(block_ptr + 4);
                const int base_idx = buf_load * SLM_B_SIZE + lane_id * XMX_K;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[base_idx + k] = qs[k];
                }
            } else {
                slm_scales_B[buf_load * SLM_SCALES_B_SIZE + lane_id] = 0.0f;
                const int base_idx = buf_load * SLM_B_SIZE + lane_id * XMX_K;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[base_idx + k] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        buf_comp = k_block & 1;
        buf_load = (k_block + 1) & 1;

        if (k_block + 1 < num_k_blocks) {
            const int next_k = k_block + 1;

            if (lane_id < XMX_M) {
                const int row = row_base + lane_id;
                if (row < row_high) {
                    const int64_t block_idx = static_cast<int64_t>(row) * num_k_blocks + next_k;
                    const char * block_ptr = src0 + block_idx * Q8_0_BLOCK_SIZE;
                    const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
                    slm_scales_A[buf_load * SLM_SCALES_A_SIZE + lane_id] = float(d);

                    const int8_t * qs = reinterpret_cast<const int8_t *>(block_ptr + 2);
                    const int base_idx = buf_load * SLM_A_SIZE + lane_id * XMX_K;
#pragma unroll
                    for (int j = 0; j < XMX_K; j++) {
                        slm_A[base_idx + j] = qs[j];
                    }
                } else {
                    slm_scales_A[buf_load * SLM_SCALES_A_SIZE + lane_id] = 0.0f;
                    const int base_idx = buf_load * SLM_A_SIZE + lane_id * XMX_K;
#pragma unroll
                    for (int j = 0; j < XMX_K; j++) {
                        slm_A[base_idx + j] = 0;
                    }
                }
            }

            {
                const int col = col_base + lane_id;
                if (col < ncols_y) {
                    const char * block_ptr = src1 +
                        (static_cast<int64_t>(col) * src1_stride_blocks + next_k) * Q8_1_BLOCK_SIZE;
                    const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);

                    slm_scales_B[buf_load * SLM_SCALES_B_SIZE + lane_id] = float(d);

                    const int8_t * qs = reinterpret_cast<const int8_t *>(block_ptr + 4);
                    const int base_idx = buf_load * SLM_B_SIZE + lane_id * XMX_K;
#pragma unroll
                    for (int k = 0; k < XMX_K; k++) {
                        slm_B[base_idx + k] = qs[k];
                    }
                } else {
                    slm_scales_B[buf_load * SLM_SCALES_B_SIZE + lane_id] = 0.0f;
                    const int base_idx = buf_load * SLM_B_SIZE + lane_id * XMX_K;
#pragma unroll
                    for (int k = 0; k < XMX_K; k++) {
                        slm_B[base_idx + k] = 0;
                    }
                }
            }
        }

        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
            matA;
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
            matB;
        sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

        auto A_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                              sycl::access::decorated::no>(slm_A + buf_comp * SLM_A_SIZE);
        auto B_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                              sycl::access::decorated::no>(slm_B + buf_comp * SLM_B_SIZE);
        auto C_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
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
            const float d_A  = slm_scales_A[buf_comp * SLM_SCALES_A_SIZE + i];
            const float d_B  = slm_scales_B[buf_comp * SLM_SCALES_B_SIZE + j];
            const float C_ij = static_cast<float>(slm_C[idx]);
            acc[acc_idx] += d_A * d_B * C_ij;
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

static inline void q4_0_tile_kernel_soa(const uint8_t * __restrict__ weight_qs,
                                        const sycl::half * __restrict__ weight_d,
                                        const uint8_t * __restrict__ act_base,
                                        float * __restrict__ dst,
                                        const int ncols_x,
                                        const int ncols_y,
                                        const int nrows_dst,
                                        const int src1_stride_blocks,
                                        const int row_low,
                                        const int row_high,
                                        int8_t * __restrict__ slm_A,
                                        int8_t * __restrict__ slm_B,
                                        int32_t * __restrict__ slm_C,
                                        float * __restrict__ slm_scales_A,
                                        float * __restrict__ slm_scales_B,
                                        float * __restrict__ slm_sums_B,
                                        sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];

    const int row_base = item.get_group(0) * XMX_M + row_low;
    const int col_base = item.get_group(1) * XMX_N;

    if (row_base >= row_high || col_base >= ncols_y) {
        return;
    }

    const int num_k_blocks = ncols_x / XMX_K;
    constexpr int Q4_0_QS_BYTES = QK4_0 / 2;
    constexpr int Q8_1_BLOCK_SIZE = sizeof(block_q8_1);
    const size_t act_row_stride = static_cast<size_t>(src1_stride_blocks) * Q8_1_BLOCK_SIZE;

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        if (lane_id < XMX_M) {
            const int row = row_base + lane_id;
            if (row < row_high) {
                const int64_t block_idx = static_cast<int64_t>(row) * num_k_blocks + k_block;
                const uint8_t * nibbles = weight_qs + block_idx * Q4_0_QS_BYTES;
                slm_scales_A[lane_id] = static_cast<float>(weight_d[block_idx]);

                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < 16; j++) {
                    const uint8_t packed = nibbles[j];
                    slm_A[base_idx + j]      = static_cast<int8_t>(packed & 0x0F);
                    slm_A[base_idx + j + 16] = static_cast<int8_t>(packed >> 4);
                }
            } else {
                slm_scales_A[lane_id] = 0.0f;
                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        {
            const int col = col_base + lane_id;
            if (col < ncols_y) {
                const uint8_t * act_row = act_base + static_cast<size_t>(col) * act_row_stride;
                const int8_t * qs = reinterpret_cast<const int8_t *>(act_row);
                const sycl::half2 * ds =
                    reinterpret_cast<const sycl::half2 *>(act_row + ncols_x);
                const sycl::half2 ds_vals = ds[k_block];

                slm_scales_B[lane_id] = static_cast<float>(ds_vals.x());
                slm_sums_B[lane_id]   = static_cast<float>(ds_vals.y());

                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[base_idx + k] = qs[k_block * XMX_K + k];
                }
            } else {
                slm_scales_B[lane_id] = 0.0f;
                slm_sums_B[lane_id]   = 0.0f;
                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[base_idx + k] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
            matA;
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
            matB;
        sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

        auto A_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                              sycl::access::decorated::no>(slm_A);
        auto B_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                              sycl::access::decorated::no>(slm_B);
        auto C_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
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
            const float d_A  = slm_scales_A[i];
            const float d_B  = slm_scales_B[j];
            const float s_B  = slm_sums_B[j];
            const float C_ij = static_cast<float>(slm_C[idx]);
            acc[acc_idx] += d_A * (C_ij * d_B - 8.0f * s_B);
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

static inline void q8_0_tile_kernel_soa(const uint8_t * __restrict__ weight_qs,
                                        const sycl::half * __restrict__ weight_d,
                                        const uint8_t * __restrict__ act_base,
                                        float * __restrict__ dst,
                                        const int ncols_x,
                                        const int ncols_y,
                                        const int nrows_dst,
                                        const int src1_stride_blocks,
                                        const int row_low,
                                        const int row_high,
                                        int8_t * __restrict__ slm_A,
                                        int8_t * __restrict__ slm_B,
                                        int32_t * __restrict__ slm_C,
                                        float * __restrict__ slm_scales_A,
                                        float * __restrict__ slm_scales_B,
                                        sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];

    const int row_base = item.get_group(0) * XMX_M + row_low;
    const int col_base = item.get_group(1) * XMX_N;

    if (row_base >= row_high || col_base >= ncols_y) {
        return;
    }

    const int num_k_blocks = ncols_x / XMX_K;
    constexpr int Q8_0_QS_BYTES = QK8_0;
    constexpr int Q8_1_BLOCK_SIZE = sizeof(block_q8_1);
    const size_t act_row_stride = static_cast<size_t>(src1_stride_blocks) * Q8_1_BLOCK_SIZE;

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        if (lane_id < XMX_M) {
            const int row = row_base + lane_id;
            if (row < row_high) {
                const int64_t block_idx = static_cast<int64_t>(row) * num_k_blocks + k_block;
                const int8_t * qs = reinterpret_cast<const int8_t *>(weight_qs + block_idx * Q8_0_QS_BYTES);
                slm_scales_A[lane_id] = static_cast<float>(weight_d[block_idx]);

                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = qs[j];
                }
            } else {
                slm_scales_A[lane_id] = 0.0f;
                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        {
            const int col = col_base + lane_id;
            if (col < ncols_y) {
                const uint8_t * act_row = act_base + static_cast<size_t>(col) * act_row_stride;
                const int8_t * qs = reinterpret_cast<const int8_t *>(act_row);
                const sycl::half2 * ds =
                    reinterpret_cast<const sycl::half2 *>(act_row + ncols_x);
                const sycl::half2 ds_vals = ds[k_block];

                slm_scales_B[lane_id] = static_cast<float>(ds_vals.x());

                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[base_idx + k] = qs[k_block * XMX_K + k];
                }
            } else {
                slm_scales_B[lane_id] = 0.0f;
                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[base_idx + k] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
            matA;
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
            matB;
        sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

        auto A_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                              sycl::access::decorated::no>(slm_A);
        auto B_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                              sycl::access::decorated::no>(slm_B);
        auto C_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
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
            const float d_A  = slm_scales_A[i];
            const float d_B  = slm_scales_B[j];
            const float C_ij = static_cast<float>(slm_C[idx]);
            acc[acc_idx] += d_A * d_B * C_ij;
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

static inline void q4_0_tile_db_kernel_soa(const uint8_t * __restrict__ weight_qs,
                                           const sycl::half * __restrict__ weight_d,
                                           const uint8_t * __restrict__ act_base,
                                           float * __restrict__ dst,
                                           const int ncols_x,
                                           const int ncols_y,
                                           const int nrows_dst,
                                           const int src1_stride_blocks,
                                           const int row_low,
                                           const int row_high,
                                           int8_t * __restrict__ slm_A,
                                           int8_t * __restrict__ slm_B,
                                           int32_t * __restrict__ slm_C,
                                           float * __restrict__ slm_scales_A,
                                           float * __restrict__ slm_scales_B,
                                           float * __restrict__ slm_sums_B,
                                           sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];

    const int row_base = item.get_group(0) * XMX_M + row_low;
    const int col_base = item.get_group(1) * XMX_N;

    if (row_base >= row_high || col_base >= ncols_y) {
        return;
    }

    const int num_k_blocks = ncols_x / XMX_K;
    constexpr int Q4_0_QS_BYTES = QK4_0 / 2;
    constexpr int Q8_1_BLOCK_SIZE = sizeof(block_q8_1);
    const size_t act_row_stride = static_cast<size_t>(src1_stride_blocks) * Q8_1_BLOCK_SIZE;

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    int buf_load = 0;
    int buf_comp = 0;

    if (num_k_blocks > 0) {
        if (lane_id < XMX_M) {
            const int row = row_base + lane_id;
            if (row < row_high) {
                const int64_t block_idx = static_cast<int64_t>(row) * num_k_blocks;
                const uint8_t * nibbles = weight_qs + block_idx * Q4_0_QS_BYTES;
                slm_scales_A[buf_load * SLM_SCALES_A_SIZE + lane_id] =
                    static_cast<float>(weight_d[block_idx]);

                const int base_idx = buf_load * SLM_A_SIZE + lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < 16; j++) {
                    const uint8_t packed = nibbles[j];
                    slm_A[base_idx + j]      = static_cast<int8_t>(packed & 0x0F);
                    slm_A[base_idx + j + 16] = static_cast<int8_t>(packed >> 4);
                }
            } else {
                slm_scales_A[buf_load * SLM_SCALES_A_SIZE + lane_id] = 0.0f;
                const int base_idx = buf_load * SLM_A_SIZE + lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        {
            const int col = col_base + lane_id;
            if (col < ncols_y) {
                const uint8_t * act_row = act_base + static_cast<size_t>(col) * act_row_stride;
                const int8_t * qs = reinterpret_cast<const int8_t *>(act_row);
                const sycl::half2 * ds =
                    reinterpret_cast<const sycl::half2 *>(act_row + ncols_x);
                const sycl::half2 ds_vals = ds[0];

                slm_scales_B[buf_load * SLM_SCALES_B_SIZE + lane_id] = static_cast<float>(ds_vals.x());
                slm_sums_B[buf_load * SLM_SUMS_B_SIZE + lane_id]     = static_cast<float>(ds_vals.y());

                const int base_idx = buf_load * SLM_B_SIZE + lane_id * XMX_K;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[base_idx + k] = qs[k];
                }
            } else {
                slm_scales_B[buf_load * SLM_SCALES_B_SIZE + lane_id] = 0.0f;
                slm_sums_B[buf_load * SLM_SUMS_B_SIZE + lane_id]     = 0.0f;
                const int base_idx = buf_load * SLM_B_SIZE + lane_id * XMX_K;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[base_idx + k] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        buf_comp = k_block & 1;
        buf_load = (k_block + 1) & 1;

        if (k_block + 1 < num_k_blocks) {
            const int next_k = k_block + 1;

            if (lane_id < XMX_M) {
                const int row = row_base + lane_id;
                if (row < row_high) {
                    const int64_t block_idx = static_cast<int64_t>(row) * num_k_blocks + next_k;
                    const uint8_t * nibbles = weight_qs + block_idx * Q4_0_QS_BYTES;
                    slm_scales_A[buf_load * SLM_SCALES_A_SIZE + lane_id] =
                        static_cast<float>(weight_d[block_idx]);

                    const int base_idx = buf_load * SLM_A_SIZE + lane_id * XMX_K;
#pragma unroll
                    for (int j = 0; j < 16; j++) {
                        const uint8_t packed = nibbles[j];
                        slm_A[base_idx + j]      = static_cast<int8_t>(packed & 0x0F);
                        slm_A[base_idx + j + 16] = static_cast<int8_t>(packed >> 4);
                    }
                } else {
                    slm_scales_A[buf_load * SLM_SCALES_A_SIZE + lane_id] = 0.0f;
                    const int base_idx = buf_load * SLM_A_SIZE + lane_id * XMX_K;
#pragma unroll
                    for (int j = 0; j < XMX_K; j++) {
                        slm_A[base_idx + j] = 0;
                    }
                }
            }

            {
                const int col = col_base + lane_id;
                if (col < ncols_y) {
                    const uint8_t * act_row = act_base + static_cast<size_t>(col) * act_row_stride;
                    const int8_t * qs = reinterpret_cast<const int8_t *>(act_row);
                    const sycl::half2 * ds =
                        reinterpret_cast<const sycl::half2 *>(act_row + ncols_x);
                    const sycl::half2 ds_vals = ds[next_k];

                    slm_scales_B[buf_load * SLM_SCALES_B_SIZE + lane_id] = static_cast<float>(ds_vals.x());
                    slm_sums_B[buf_load * SLM_SUMS_B_SIZE + lane_id]     = static_cast<float>(ds_vals.y());

                    const int base_idx = buf_load * SLM_B_SIZE + lane_id * XMX_K;
#pragma unroll
                    for (int k = 0; k < XMX_K; k++) {
                        slm_B[base_idx + k] = qs[next_k * XMX_K + k];
                    }
                } else {
                    slm_scales_B[buf_load * SLM_SCALES_B_SIZE + lane_id] = 0.0f;
                    slm_sums_B[buf_load * SLM_SUMS_B_SIZE + lane_id]     = 0.0f;
                    const int base_idx = buf_load * SLM_B_SIZE + lane_id * XMX_K;
#pragma unroll
                    for (int k = 0; k < XMX_K; k++) {
                        slm_B[base_idx + k] = 0;
                    }
                }
            }
        }

        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
            matA;
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
            matB;
        sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

        auto A_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                              sycl::access::decorated::no>(slm_A + buf_comp * SLM_A_SIZE);
        auto B_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                              sycl::access::decorated::no>(slm_B + buf_comp * SLM_B_SIZE);
        auto C_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
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
            const float d_A  = slm_scales_A[buf_comp * SLM_SCALES_A_SIZE + i];
            const float d_B  = slm_scales_B[buf_comp * SLM_SCALES_B_SIZE + j];
            const float s_B  = slm_sums_B[buf_comp * SLM_SUMS_B_SIZE + j];
            const float C_ij = static_cast<float>(slm_C[idx]);
            acc[acc_idx] += d_A * (C_ij * d_B - 8.0f * s_B);
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

static inline void q8_0_tile_db_kernel_soa(const uint8_t * __restrict__ weight_qs,
                                           const sycl::half * __restrict__ weight_d,
                                           const uint8_t * __restrict__ act_base,
                                           float * __restrict__ dst,
                                           const int ncols_x,
                                           const int ncols_y,
                                           const int nrows_dst,
                                           const int src1_stride_blocks,
                                           const int row_low,
                                           const int row_high,
                                           int8_t * __restrict__ slm_A,
                                           int8_t * __restrict__ slm_B,
                                           int32_t * __restrict__ slm_C,
                                           float * __restrict__ slm_scales_A,
                                           float * __restrict__ slm_scales_B,
                                           sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];

    const int row_base = item.get_group(0) * XMX_M + row_low;
    const int col_base = item.get_group(1) * XMX_N;

    if (row_base >= row_high || col_base >= ncols_y) {
        return;
    }

    const int num_k_blocks = ncols_x / XMX_K;
    constexpr int Q8_0_QS_BYTES = QK8_0;
    constexpr int Q8_1_BLOCK_SIZE = sizeof(block_q8_1);
    const size_t act_row_stride = static_cast<size_t>(src1_stride_blocks) * Q8_1_BLOCK_SIZE;

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    int buf_load = 0;
    int buf_comp = 0;

    if (num_k_blocks > 0) {
        if (lane_id < XMX_M) {
            const int row = row_base + lane_id;
            if (row < row_high) {
                const int64_t block_idx = static_cast<int64_t>(row) * num_k_blocks;
                const int8_t * qs = reinterpret_cast<const int8_t *>(weight_qs + block_idx * Q8_0_QS_BYTES);
                slm_scales_A[buf_load * SLM_SCALES_A_SIZE + lane_id] =
                    static_cast<float>(weight_d[block_idx]);

                const int base_idx = buf_load * SLM_A_SIZE + lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = qs[j];
                }
            } else {
                slm_scales_A[buf_load * SLM_SCALES_A_SIZE + lane_id] = 0.0f;
                const int base_idx = buf_load * SLM_A_SIZE + lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        {
            const int col = col_base + lane_id;
            if (col < ncols_y) {
                const uint8_t * act_row = act_base + static_cast<size_t>(col) * act_row_stride;
                const int8_t * qs = reinterpret_cast<const int8_t *>(act_row);
                const sycl::half2 * ds =
                    reinterpret_cast<const sycl::half2 *>(act_row + ncols_x);
                const sycl::half2 ds_vals = ds[0];

                slm_scales_B[buf_load * SLM_SCALES_B_SIZE + lane_id] = static_cast<float>(ds_vals.x());

                const int base_idx = buf_load * SLM_B_SIZE + lane_id * XMX_K;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[base_idx + k] = qs[k];
                }
            } else {
                slm_scales_B[buf_load * SLM_SCALES_B_SIZE + lane_id] = 0.0f;
                const int base_idx = buf_load * SLM_B_SIZE + lane_id * XMX_K;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[base_idx + k] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        buf_comp = k_block & 1;
        buf_load = (k_block + 1) & 1;

        if (k_block + 1 < num_k_blocks) {
            const int next_k = k_block + 1;

            if (lane_id < XMX_M) {
                const int row = row_base + lane_id;
                if (row < row_high) {
                    const int64_t block_idx = static_cast<int64_t>(row) * num_k_blocks + next_k;
                    const int8_t * qs = reinterpret_cast<const int8_t *>(weight_qs + block_idx * Q8_0_QS_BYTES);
                    slm_scales_A[buf_load * SLM_SCALES_A_SIZE + lane_id] =
                        static_cast<float>(weight_d[block_idx]);

                    const int base_idx = buf_load * SLM_A_SIZE + lane_id * XMX_K;
#pragma unroll
                    for (int j = 0; j < XMX_K; j++) {
                        slm_A[base_idx + j] = qs[j];
                    }
                } else {
                    slm_scales_A[buf_load * SLM_SCALES_A_SIZE + lane_id] = 0.0f;
                    const int base_idx = buf_load * SLM_A_SIZE + lane_id * XMX_K;
#pragma unroll
                    for (int j = 0; j < XMX_K; j++) {
                        slm_A[base_idx + j] = 0;
                    }
                }
            }

            {
                const int col = col_base + lane_id;
                if (col < ncols_y) {
                    const uint8_t * act_row = act_base + static_cast<size_t>(col) * act_row_stride;
                    const int8_t * qs = reinterpret_cast<const int8_t *>(act_row);
                    const sycl::half2 * ds =
                        reinterpret_cast<const sycl::half2 *>(act_row + ncols_x);
                    const sycl::half2 ds_vals = ds[next_k];

                    slm_scales_B[buf_load * SLM_SCALES_B_SIZE + lane_id] =
                        static_cast<float>(ds_vals.x());

                    const int base_idx = buf_load * SLM_B_SIZE + lane_id * XMX_K;
#pragma unroll
                    for (int k = 0; k < XMX_K; k++) {
                        slm_B[base_idx + k] = qs[next_k * XMX_K + k];
                    }
                } else {
                    slm_scales_B[buf_load * SLM_SCALES_B_SIZE + lane_id] = 0.0f;
                    const int base_idx = buf_load * SLM_B_SIZE + lane_id * XMX_K;
#pragma unroll
                    for (int k = 0; k < XMX_K; k++) {
                        slm_B[base_idx + k] = 0;
                    }
                }
            }
        }

        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
            matA;
        sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
            matB;
        sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

        auto A_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                              sycl::access::decorated::no>(slm_A + buf_comp * SLM_A_SIZE);
        auto B_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                              sycl::access::decorated::no>(slm_B + buf_comp * SLM_B_SIZE);
        auto C_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
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
            const float d_A  = slm_scales_A[buf_comp * SLM_SCALES_A_SIZE + i];
            const float d_B  = slm_scales_B[buf_comp * SLM_SCALES_B_SIZE + j];
            const float C_ij = static_cast<float>(slm_C[idx]);
            acc[acc_idx] += d_A * d_B * C_ij;
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

static inline void q4_0_cf2_kernel_soa(const uint8_t * __restrict__ weight_qs,
                                       const sycl::half * __restrict__ weight_d,
                                       const uint8_t * __restrict__ act_base,
                                       float * __restrict__ dst,
                                       const int ncols_x,
                                       const int ncols_y,
                                       const int nrows_dst,
                                       const int src1_stride_blocks,
                                       const int row_low,
                                       const int row_high,
                                       int8_t * __restrict__ slm_A,
                                       int8_t * __restrict__ slm_B,
                                       int32_t * __restrict__ slm_C,
                                       float * __restrict__ slm_scales_A,
                                       float * __restrict__ slm_scales_B,
                                       float * __restrict__ slm_sums_B,
                                       sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  sg_id   = item.get_local_id(0);
    const int  lane_id = sg.get_local_id()[0];

    const int row_base  = item.get_group(0) * XMX_M + row_low;
    const int col_group = item.get_group(1);
    const int col_base  = col_group * (CF2_COL_TILES * XMX_N) + sg_id * XMX_N;

    if (row_base >= row_high) {
        return;
    }

    const int  num_k_blocks = ncols_x / XMX_K;
    const bool active_sg    = (col_base < ncols_y);
    constexpr int Q4_0_QS_BYTES = QK4_0 / 2;
    constexpr int Q8_1_BLOCK_SIZE = sizeof(block_q8_1);
    const size_t act_row_stride = static_cast<size_t>(src1_stride_blocks) * Q8_1_BLOCK_SIZE;

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    int8_t *  my_B_tile   = slm_B + sg_id * (XMX_K * XMX_N);
    int32_t * my_C_tile   = slm_C + sg_id * (XMX_M * XMX_N);
    float *   my_scales_B = slm_scales_B + sg_id * XMX_N;
    float *   my_sums_B   = slm_sums_B + sg_id * XMX_N;

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        if (sg_id == 0 && lane_id < XMX_M) {
            const int row = row_base + lane_id;
            if (row < row_high) {
                const int64_t block_idx = static_cast<int64_t>(row) * num_k_blocks + k_block;
                const uint8_t * nibbles = weight_qs + block_idx * Q4_0_QS_BYTES;
                slm_scales_A[lane_id] = static_cast<float>(weight_d[block_idx]);

                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < 16; j++) {
                    const uint8_t packed = nibbles[j];
                    slm_A[base_idx + j]      = static_cast<int8_t>(packed & 0x0F);
                    slm_A[base_idx + j + 16] = static_cast<int8_t>(packed >> 4);
                }
            } else {
                slm_scales_A[lane_id] = 0.0f;
                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        if (active_sg) {
            const int col = col_base + lane_id;
            if (col < ncols_y) {
                const uint8_t * act_row = act_base + static_cast<size_t>(col) * act_row_stride;
                const int8_t * qs = reinterpret_cast<const int8_t *>(act_row);
                const sycl::half2 * ds =
                    reinterpret_cast<const sycl::half2 *>(act_row + ncols_x);
                const sycl::half2 ds_vals = ds[k_block];

                my_scales_B[lane_id] = static_cast<float>(ds_vals.x());
                my_sums_B[lane_id]   = static_cast<float>(ds_vals.y());

#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    my_B_tile[k + lane_id * XMX_K] = qs[k_block * XMX_K + k];
                }
            } else {
                my_scales_B[lane_id] = 0.0f;
                my_sums_B[lane_id]   = 0.0f;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    my_B_tile[k + lane_id * XMX_K] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        if (active_sg) {
            auto A_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                    slm_A);
            auto B_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                    my_B_tile);
            auto C_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                    my_C_tile);

            sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K,
                                   sycl_xmx::layout::row_major>
                matA;
            sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N,
                                   sycl_xmx::layout::col_major>
                matB;
            sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

            sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
            sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
            sycl_xmx::joint_matrix_fill(sg, matC, 0);
            sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
            sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

#pragma unroll
            for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
                const int   i    = idx / XMX_N;
                const int   j    = idx % XMX_N;
                const float d_A  = slm_scales_A[i];
                const float d_B  = my_scales_B[j];
                const float s_B  = my_sums_B[j];
                const float C_ij = static_cast<float>(my_C_tile[idx]);
                acc[acc_idx] += d_A * (C_ij * d_B - 8.0f * s_B);
            }
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

    if (active_sg) {
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
}

static inline void q8_0_cf2_kernel_soa(const uint8_t * __restrict__ weight_qs,
                                       const sycl::half * __restrict__ weight_d,
                                       const uint8_t * __restrict__ act_base,
                                       float * __restrict__ dst,
                                       const int ncols_x,
                                       const int ncols_y,
                                       const int nrows_dst,
                                       const int src1_stride_blocks,
                                       const int row_low,
                                       const int row_high,
                                       int8_t * __restrict__ slm_A,
                                       int8_t * __restrict__ slm_B,
                                       int32_t * __restrict__ slm_C,
                                       float * __restrict__ slm_scales_A,
                                       float * __restrict__ slm_scales_B,
                                       sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  sg_id   = item.get_local_id(0);
    const int  lane_id = sg.get_local_id()[0];

    const int row_base  = item.get_group(0) * XMX_M + row_low;
    const int col_group = item.get_group(1);
    const int col_base  = col_group * (CF2_COL_TILES * XMX_N) + sg_id * XMX_N;

    if (row_base >= row_high) {
        return;
    }

    const int  num_k_blocks = ncols_x / XMX_K;
    const bool active_sg    = (col_base < ncols_y);
    constexpr int Q8_0_QS_BYTES = QK8_0;
    constexpr int Q8_1_BLOCK_SIZE = sizeof(block_q8_1);
    const size_t act_row_stride = static_cast<size_t>(src1_stride_blocks) * Q8_1_BLOCK_SIZE;

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    int8_t *  my_B_tile   = slm_B + sg_id * (XMX_K * XMX_N);
    int32_t * my_C_tile   = slm_C + sg_id * (XMX_M * XMX_N);
    float *   my_scales_B = slm_scales_B + sg_id * XMX_N;

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        if (sg_id == 0 && lane_id < XMX_M) {
            const int row = row_base + lane_id;
            if (row < row_high) {
                const int64_t block_idx = static_cast<int64_t>(row) * num_k_blocks + k_block;
                const int8_t * qs = reinterpret_cast<const int8_t *>(weight_qs + block_idx * Q8_0_QS_BYTES);
                slm_scales_A[lane_id] = static_cast<float>(weight_d[block_idx]);

                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = qs[j];
                }
            } else {
                slm_scales_A[lane_id] = 0.0f;
                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        if (active_sg) {
            const int col = col_base + lane_id;
            if (col < ncols_y) {
                const uint8_t * act_row = act_base + static_cast<size_t>(col) * act_row_stride;
                const int8_t * qs = reinterpret_cast<const int8_t *>(act_row);
                const sycl::half2 * ds =
                    reinterpret_cast<const sycl::half2 *>(act_row + ncols_x);
                const sycl::half2 ds_vals = ds[k_block];

                my_scales_B[lane_id] = static_cast<float>(ds_vals.x());

#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    my_B_tile[k + lane_id * XMX_K] = qs[k_block * XMX_K + k];
                }
            } else {
                my_scales_B[lane_id] = 0.0f;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    my_B_tile[k + lane_id * XMX_K] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        if (active_sg) {
            auto A_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                    slm_A);
            auto B_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                    my_B_tile);
            auto C_ptr =
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                    my_C_tile);

            sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K,
                                   sycl_xmx::layout::row_major>
                matA;
            sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N,
                                   sycl_xmx::layout::col_major>
                matB;
            sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

            sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
            sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
            sycl_xmx::joint_matrix_fill(sg, matC, 0);
            sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
            sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

#pragma unroll
            for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
                const int   i    = idx / XMX_N;
                const int   j    = idx % XMX_N;
                const float d_A  = slm_scales_A[i];
                const float d_B  = my_scales_B[j];
                const float C_ij = static_cast<float>(my_C_tile[idx]);
                acc[acc_idx] += d_A * d_B * C_ij;
            }
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

    if (active_sg) {
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
}

static inline void q4_0_cf4_kernel_soa(const uint8_t * __restrict__ weight_qs,
                                       const sycl::half * __restrict__ weight_d,
                                       const uint8_t * __restrict__ act_base,
                                       float * __restrict__ dst,
                                       const int ncols_x,
                                       const int ncols_y,
                                       const int nrows_dst,
                                       const int src1_stride_blocks,
                                       const int row_low,
                                       const int row_high,
                                       int8_t * __restrict__ slm_A,
                                       int8_t * __restrict__ slm_B,
                                       int32_t * __restrict__ slm_C,
                                       float * __restrict__ slm_scales_A,
                                       float * __restrict__ slm_scales_B,
                                       float * __restrict__ slm_sums_B,
                                       sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];

    const int row_base  = item.get_group(0) * XMX_M + row_low;
    const int col_group = item.get_group(1);
    const int col_base  = col_group * (CF4_COL_TILES * XMX_N);

    if (row_base >= row_high) {
        return;
    }

    const int num_k_blocks = ncols_x / XMX_K;
    constexpr int Q4_0_QS_BYTES = QK4_0 / 2;
    constexpr int Q8_1_BLOCK_SIZE = sizeof(block_q8_1);
    const size_t act_row_stride = static_cast<size_t>(src1_stride_blocks) * Q8_1_BLOCK_SIZE;

    float acc[CF4_COL_TILES][8] = {};

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        if (lane_id < XMX_M) {
            const int row = row_base + lane_id;
            if (row < row_high) {
                const int64_t block_idx = static_cast<int64_t>(row) * num_k_blocks + k_block;
                const uint8_t * nibbles = weight_qs + block_idx * Q4_0_QS_BYTES;
                slm_scales_A[lane_id] = static_cast<float>(weight_d[block_idx]);

                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < 16; j++) {
                    const uint8_t packed = nibbles[j];
                    slm_A[base_idx + j]      = static_cast<int8_t>(packed & 0x0F);
                    slm_A[base_idx + j + 16] = static_cast<int8_t>(packed >> 4);
                }
            } else {
                slm_scales_A[lane_id] = 0.0f;
                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        for (int tile = 0; tile < CF4_COL_TILES; tile++) {
            const int col_tile = col_base + tile * XMX_N;
            const bool active_tile = (col_tile < ncols_y);
            const int col = col_tile + lane_id;
            if (col < ncols_y) {
                const uint8_t * act_row = act_base + static_cast<size_t>(col) * act_row_stride;
                const int8_t * qs = reinterpret_cast<const int8_t *>(act_row);
                const sycl::half2 * ds =
                    reinterpret_cast<const sycl::half2 *>(act_row + ncols_x);
                const sycl::half2 ds_vals = ds[k_block];

                slm_scales_B[lane_id] = static_cast<float>(ds_vals.x());
                slm_sums_B[lane_id]   = static_cast<float>(ds_vals.y());

#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = qs[k_block * XMX_K + k];
                }
            } else {
                slm_scales_B[lane_id] = 0.0f;
                slm_sums_B[lane_id]   = 0.0f;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = 0;
                }
            }

            item.barrier(sycl::access::fence_space::local_space);

            if (active_tile) {
                auto A_ptr =
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        slm_A);
                auto B_ptr =
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        slm_B);
                auto C_ptr =
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        slm_C);

                sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K,
                                       sycl_xmx::layout::row_major>
                    matA;
                sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N,
                                       sycl_xmx::layout::col_major>
                    matB;
                sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

                sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
                sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
                sycl_xmx::joint_matrix_fill(sg, matC, 0);
                sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
                sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

#pragma unroll
                for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
                    const int   i    = idx / XMX_N;
                    const int   j    = idx % XMX_N;
                    const float d_A  = slm_scales_A[i];
                    const float d_B  = slm_scales_B[j];
                    const float s_B  = slm_sums_B[j];
                    const float C_ij = static_cast<float>(slm_C[idx]);
                    acc[tile][acc_idx] += d_A * (C_ij * d_B - 8.0f * s_B);
                }
            }

            item.barrier(sycl::access::fence_space::local_space);
        }
    }

    for (int tile = 0; tile < CF4_COL_TILES; tile++) {
        const int col_tile = col_base + tile * XMX_N;
        if (col_tile >= ncols_y) {
            continue;
        }
#pragma unroll
        for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
            const int i   = idx / XMX_N;
            const int j   = idx % XMX_N;
            const int row = row_base + i;
            const int col = col_tile + j;
            if (row < row_high && col < ncols_y) {
                dst[col * nrows_dst + row] = acc[tile][acc_idx];
            }
        }
    }
}

static inline void q8_0_cf4_kernel_soa(const uint8_t * __restrict__ weight_qs,
                                       const sycl::half * __restrict__ weight_d,
                                       const uint8_t * __restrict__ act_base,
                                       float * __restrict__ dst,
                                       const int ncols_x,
                                       const int ncols_y,
                                       const int nrows_dst,
                                       const int src1_stride_blocks,
                                       const int row_low,
                                       const int row_high,
                                       int8_t * __restrict__ slm_A,
                                       int8_t * __restrict__ slm_B,
                                       int32_t * __restrict__ slm_C,
                                       float * __restrict__ slm_scales_A,
                                       float * __restrict__ slm_scales_B,
                                       sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];

    const int row_base  = item.get_group(0) * XMX_M + row_low;
    const int col_group = item.get_group(1);
    const int col_base  = col_group * (CF4_COL_TILES * XMX_N);

    if (row_base >= row_high) {
        return;
    }

    const int num_k_blocks = ncols_x / XMX_K;
    constexpr int Q8_0_QS_BYTES = QK8_0;
    constexpr int Q8_1_BLOCK_SIZE = sizeof(block_q8_1);
    const size_t act_row_stride = static_cast<size_t>(src1_stride_blocks) * Q8_1_BLOCK_SIZE;

    float acc[CF4_COL_TILES][8] = {};

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        if (lane_id < XMX_M) {
            const int row = row_base + lane_id;
            if (row < row_high) {
                const int64_t block_idx = static_cast<int64_t>(row) * num_k_blocks + k_block;
                const int8_t * qs = reinterpret_cast<const int8_t *>(weight_qs + block_idx * Q8_0_QS_BYTES);
                slm_scales_A[lane_id] = static_cast<float>(weight_d[block_idx]);

                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = qs[j];
                }
            } else {
                slm_scales_A[lane_id] = 0.0f;
                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
            }
        }

        for (int tile = 0; tile < CF4_COL_TILES; tile++) {
            const int col_tile = col_base + tile * XMX_N;
            const bool active_tile = (col_tile < ncols_y);
            const int col = col_tile + lane_id;
            if (col < ncols_y) {
                const uint8_t * act_row = act_base + static_cast<size_t>(col) * act_row_stride;
                const int8_t * qs = reinterpret_cast<const int8_t *>(act_row);
                const sycl::half2 * ds =
                    reinterpret_cast<const sycl::half2 *>(act_row + ncols_x);
                const sycl::half2 ds_vals = ds[k_block];

                slm_scales_B[lane_id] = static_cast<float>(ds_vals.x());

#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = qs[k_block * XMX_K + k];
                }
            } else {
                slm_scales_B[lane_id] = 0.0f;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[k + lane_id * XMX_K] = 0;
                }
            }

            item.barrier(sycl::access::fence_space::local_space);

            if (active_tile) {
                auto A_ptr =
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        slm_A);
                auto B_ptr =
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        slm_B);
                auto C_ptr =
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        slm_C);

                sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K,
                                       sycl_xmx::layout::row_major>
                    matA;
                sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N,
                                       sycl_xmx::layout::col_major>
                    matB;
                sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

                sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
                sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
                sycl_xmx::joint_matrix_fill(sg, matC, 0);
                sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
                sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

#pragma unroll
                for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
                    const int   i    = idx / XMX_N;
                    const int   j    = idx % XMX_N;
                    const float d_A  = slm_scales_A[i];
                    const float d_B  = slm_scales_B[j];
                    const float C_ij = static_cast<float>(slm_C[idx]);
                    acc[tile][acc_idx] += d_A * d_B * C_ij;
                }
            }

            item.barrier(sycl::access::fence_space::local_space);
        }
    }

    for (int tile = 0; tile < CF4_COL_TILES; tile++) {
        const int col_tile = col_base + tile * XMX_N;
        if (col_tile >= ncols_y) {
            continue;
        }
#pragma unroll
        for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
            const int i   = idx / XMX_N;
            const int j   = idx % XMX_N;
            const int row = row_base + i;
            const int col = col_tile + j;
            if (row < row_high && col < ncols_y) {
                dst[col * nrows_dst + row] = acc[tile][acc_idx];
            }
        }
    }
}

inline bool launch_xmx_tile(const ggml_sycl::mmvq_bench_args & args,
                            std::vector<sycl::event> * events,
                            std::string & error) {
    if (!validate_xmx_args(args, error)) {
        return false;
    }

    const int row_low  = static_cast<int>(args.row_low);
    const int row_high = static_cast<int>(args.row_high);
    const int rows_span = row_high - row_low;
    const int num_row_tiles = (rows_span + XMX_M - 1) / XMX_M;
    const int num_col_tiles = (static_cast<int>(args.batch) + XMX_N - 1) / XMX_N;

    const int ncols_x = static_cast<int>(args.ncols);
    const int ncols_y = static_cast<int>(args.batch);
    const int nrows_dst = static_cast<int>(args.dst_row_stride);
    const int src1_stride_blocks = static_cast<int>(args.src1_padded_col_size / XMX_K);
    const void * weights = args.weights;
    const void * activations = args.activations;
    float * output = args.output;

    sycl::queue & queue = *args.stream;
    const sycl::range<2> grid(num_row_tiles, num_col_tiles);
    const sycl::range<2> block(1, XMX_SG_SIZE);

    sycl::event ev;
    if (args.weight_type == GGML_TYPE_Q4_0) {
        constexpr int TOTAL_SLM_SIZE =
            SLM_A_SIZE + SLM_B_SIZE + SLM_C_SIZE * 4 + SLM_SCALES_A_SIZE * 4 + SLM_SCALES_B_SIZE * 4 +
            SLM_SUMS_B_SIZE * 4;
        ev = queue.submit([&](sycl::handler & h) {
            sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE), h);

            h.parallel_for(sycl::nd_range<2>(grid * block, block),
                           [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                               char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                               int offset = 0;
                               int8_t * slm_A = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_A_SIZE;
                               int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_B_SIZE;
                               int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                               offset += SLM_C_SIZE * 4;
                               float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                               offset += SLM_SCALES_A_SIZE * 4;
                               float * slm_scales_B = reinterpret_cast<float *>(shared + offset);
                               offset += SLM_SCALES_B_SIZE * 4;
                               float * slm_sums_B = reinterpret_cast<float *>(shared + offset);

                               q4_0_tile_kernel(weights, activations, output,
                                                ncols_x, ncols_y, nrows_dst,
                                                src1_stride_blocks, row_low, row_high,
                                                slm_A, slm_B, slm_C, slm_scales_A, slm_scales_B, slm_sums_B,
                                                item);
                           });
        });
    } else {
        constexpr int TOTAL_SLM_SIZE =
            SLM_A_SIZE + SLM_B_SIZE + SLM_C_SIZE * 4 + SLM_SCALES_A_SIZE * 4 + SLM_SCALES_B_SIZE * 4;
        ev = queue.submit([&](sycl::handler & h) {
            sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE), h);

            h.parallel_for(sycl::nd_range<2>(grid * block, block),
                           [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                               char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                               int offset = 0;
                               int8_t * slm_A = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_A_SIZE;
                               int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_B_SIZE;
                               int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                               offset += SLM_C_SIZE * 4;
                               float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                               offset += SLM_SCALES_A_SIZE * 4;
                               float * slm_scales_B = reinterpret_cast<float *>(shared + offset);

                               q8_0_tile_kernel(weights, activations, output,
                                                ncols_x, ncols_y, nrows_dst,
                                                src1_stride_blocks, row_low, row_high,
                                                slm_A, slm_B, slm_C, slm_scales_A, slm_scales_B,
                                                item);
                           });
        });
    }

    if (events != nullptr) {
        events->push_back(ev);
    }
    return true;
}

inline bool launch_xmx_tile_db(const ggml_sycl::mmvq_bench_args & args,
                               std::vector<sycl::event> * events,
                               std::string & error) {
    if (!validate_xmx_args(args, error)) {
        return false;
    }

    const int row_low  = static_cast<int>(args.row_low);
    const int row_high = static_cast<int>(args.row_high);
    const int rows_span = row_high - row_low;
    const int num_row_tiles = (rows_span + XMX_M - 1) / XMX_M;
    const int num_col_tiles = (static_cast<int>(args.batch) + XMX_N - 1) / XMX_N;

    const int ncols_x = static_cast<int>(args.ncols);
    const int ncols_y = static_cast<int>(args.batch);
    const int nrows_dst = static_cast<int>(args.dst_row_stride);
    const int src1_stride_blocks = static_cast<int>(args.src1_padded_col_size / XMX_K);
    const void * weights = args.weights;
    const void * activations = args.activations;
    float * output = args.output;

    sycl::queue & queue = *args.stream;
    const sycl::range<2> grid(num_row_tiles, num_col_tiles);
    const sycl::range<2> block(1, XMX_SG_SIZE);

    sycl::event ev;
    if (args.weight_type == GGML_TYPE_Q4_0) {
        constexpr int TOTAL_SLM_SIZE =
            SLM_A_SIZE_DB + SLM_B_SIZE_DB + SLM_C_SIZE * 4 +
            SLM_SCALES_A_SIZE_DB * 4 + SLM_SCALES_B_SIZE_DB * 4 + SLM_SUMS_B_SIZE_DB * 4;
        ev = queue.submit([&](sycl::handler & h) {
            sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE), h);

            h.parallel_for(sycl::nd_range<2>(grid * block, block),
                           [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                               char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                               int offset = 0;
                               int8_t * slm_A = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_A_SIZE_DB;
                               int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_B_SIZE_DB;
                               int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                               offset += SLM_C_SIZE * 4;
                               float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                               offset += SLM_SCALES_A_SIZE_DB * 4;
                               float * slm_scales_B = reinterpret_cast<float *>(shared + offset);
                               offset += SLM_SCALES_B_SIZE_DB * 4;
                               float * slm_sums_B = reinterpret_cast<float *>(shared + offset);

                               q4_0_tile_db_kernel(weights, activations, output,
                                                   ncols_x, ncols_y, nrows_dst,
                                                   src1_stride_blocks, row_low, row_high,
                                                   slm_A, slm_B, slm_C, slm_scales_A, slm_scales_B, slm_sums_B,
                                                   item);
                           });
        });
    } else {
        constexpr int TOTAL_SLM_SIZE =
            SLM_A_SIZE_DB + SLM_B_SIZE_DB + SLM_C_SIZE * 4 +
            SLM_SCALES_A_SIZE_DB * 4 + SLM_SCALES_B_SIZE_DB * 4;
        ev = queue.submit([&](sycl::handler & h) {
            sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE), h);

            h.parallel_for(sycl::nd_range<2>(grid * block, block),
                           [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                               char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                               int offset = 0;
                               int8_t * slm_A = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_A_SIZE_DB;
                               int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_B_SIZE_DB;
                               int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                               offset += SLM_C_SIZE * 4;
                               float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                               offset += SLM_SCALES_A_SIZE_DB * 4;
                               float * slm_scales_B = reinterpret_cast<float *>(shared + offset);

                               q8_0_tile_db_kernel(weights, activations, output,
                                                   ncols_x, ncols_y, nrows_dst,
                                                   src1_stride_blocks, row_low, row_high,
                                                   slm_A, slm_B, slm_C, slm_scales_A, slm_scales_B,
                                                   item);
                           });
        });
    }

    if (events != nullptr) {
        events->push_back(ev);
    }
    return true;
}

inline bool launch_xmx_cf2(const ggml_sycl::mmvq_bench_args & args,
                           std::vector<sycl::event> * events,
                           std::string & error) {
    if (!validate_xmx_args(args, error)) {
        return false;
    }

    const int row_low  = static_cast<int>(args.row_low);
    const int row_high = static_cast<int>(args.row_high);
    const int rows_span = row_high - row_low;
    const int num_row_tiles = (rows_span + XMX_M - 1) / XMX_M;
    const int num_col_groups =
        (static_cast<int>(args.batch) + CF2_COL_TILES * XMX_N - 1) / (CF2_COL_TILES * XMX_N);

    const int ncols_x = static_cast<int>(args.ncols);
    const int ncols_y = static_cast<int>(args.batch);
    const int nrows_dst = static_cast<int>(args.dst_row_stride);
    const int src1_stride_blocks = static_cast<int>(args.src1_padded_col_size / XMX_K);
    const void * weights = args.weights;
    const void * activations = args.activations;
    float * output = args.output;

    sycl::queue & queue = *args.stream;
    const sycl::range<2> grid(num_row_tiles, num_col_groups);
    const sycl::range<2> block(CF2_COL_TILES, XMX_SG_SIZE);

    sycl::event ev;
    if (args.weight_type == GGML_TYPE_Q4_0) {
        constexpr int CF2_SLM_B_SIZE = CF2_COL_TILES * SLM_B_SIZE;
        constexpr int CF2_SLM_C_SIZE = CF2_COL_TILES * SLM_C_SIZE;
        constexpr int CF2_SCALES_B_SIZE = CF2_COL_TILES * SLM_SCALES_B_SIZE;
        constexpr int CF2_SUMS_B_SIZE = CF2_COL_TILES * SLM_SUMS_B_SIZE;
        constexpr int TOTAL_SLM_SIZE =
            SLM_A_SIZE + CF2_SLM_B_SIZE + CF2_SLM_C_SIZE * 4 + SLM_SCALES_A_SIZE * 4 +
            CF2_SCALES_B_SIZE * 4 + CF2_SUMS_B_SIZE * 4;

        ev = queue.submit([&](sycl::handler & h) {
            sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE), h);

            h.parallel_for(sycl::nd_range<2>(grid * block, block),
                           [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                               char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                               int offset = 0;
                               int8_t * slm_A = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_A_SIZE;
                               int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                               offset += CF2_SLM_B_SIZE;
                               int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                               offset += CF2_SLM_C_SIZE * 4;
                               float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                               offset += SLM_SCALES_A_SIZE * 4;
                               float * slm_scales_B = reinterpret_cast<float *>(shared + offset);
                               offset += CF2_SCALES_B_SIZE * 4;
                               float * slm_sums_B = reinterpret_cast<float *>(shared + offset);

                               q4_0_cf2_kernel(weights, activations, output,
                                               ncols_x, ncols_y, nrows_dst,
                                               src1_stride_blocks, row_low, row_high,
                                               slm_A, slm_B, slm_C, slm_scales_A, slm_scales_B, slm_sums_B,
                                               item);
                           });
        });
    } else {
        constexpr int CF2_SLM_B_SIZE = CF2_COL_TILES * SLM_B_SIZE;
        constexpr int CF2_SLM_C_SIZE = CF2_COL_TILES * SLM_C_SIZE;
        constexpr int CF2_SCALES_B_SIZE = CF2_COL_TILES * SLM_SCALES_B_SIZE;
        constexpr int TOTAL_SLM_SIZE =
            SLM_A_SIZE + CF2_SLM_B_SIZE + CF2_SLM_C_SIZE * 4 + SLM_SCALES_A_SIZE * 4 +
            CF2_SCALES_B_SIZE * 4;

        ev = queue.submit([&](sycl::handler & h) {
            sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE), h);

            h.parallel_for(sycl::nd_range<2>(grid * block, block),
                           [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                               char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                               int offset = 0;
                               int8_t * slm_A = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_A_SIZE;
                               int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                               offset += CF2_SLM_B_SIZE;
                               int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                               offset += CF2_SLM_C_SIZE * 4;
                               float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                               offset += SLM_SCALES_A_SIZE * 4;
                               float * slm_scales_B = reinterpret_cast<float *>(shared + offset);

                               q8_0_cf2_kernel(weights, activations, output,
                                               ncols_x, ncols_y, nrows_dst,
                                               src1_stride_blocks, row_low, row_high,
                                               slm_A, slm_B, slm_C, slm_scales_A, slm_scales_B,
                                               item);
                           });
        });
    }

    if (events != nullptr) {
        events->push_back(ev);
    }
    return true;
}

inline bool launch_xmx_cf4(const ggml_sycl::mmvq_bench_args & args,
                           std::vector<sycl::event> * events,
                           std::string & error) {
    if (!validate_xmx_args(args, error)) {
        return false;
    }

    const int row_low  = static_cast<int>(args.row_low);
    const int row_high = static_cast<int>(args.row_high);
    const int rows_span = row_high - row_low;
    const int num_row_tiles = (rows_span + XMX_M - 1) / XMX_M;
    const int num_col_groups =
        (static_cast<int>(args.batch) + CF4_COL_TILES * XMX_N - 1) / (CF4_COL_TILES * XMX_N);

    const int ncols_x = static_cast<int>(args.ncols);
    const int ncols_y = static_cast<int>(args.batch);
    const int nrows_dst = static_cast<int>(args.dst_row_stride);
    const int src1_stride_blocks = static_cast<int>(args.src1_padded_col_size / XMX_K);
    const void * weights = args.weights;
    const void * activations = args.activations;
    float * output = args.output;

    sycl::queue & queue = *args.stream;
    const sycl::range<2> grid(num_row_tiles, num_col_groups);
    const sycl::range<2> block(1, XMX_SG_SIZE);

    sycl::event ev;
    if (args.weight_type == GGML_TYPE_Q4_0) {
        constexpr int TOTAL_SLM_SIZE =
            SLM_A_SIZE + SLM_B_SIZE + SLM_C_SIZE * 4 + SLM_SCALES_A_SIZE * 4 +
            SLM_SCALES_B_SIZE * 4 + SLM_SUMS_B_SIZE * 4;

        ev = queue.submit([&](sycl::handler & h) {
            sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE), h);

            h.parallel_for(sycl::nd_range<2>(grid * block, block),
                           [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                               char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                               int offset = 0;
                               int8_t * slm_A = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_A_SIZE;
                               int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_B_SIZE;
                               int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                               offset += SLM_C_SIZE * 4;
                               float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                               offset += SLM_SCALES_A_SIZE * 4;
                               float * slm_scales_B = reinterpret_cast<float *>(shared + offset);
                               offset += SLM_SCALES_B_SIZE * 4;
                               float * slm_sums_B = reinterpret_cast<float *>(shared + offset);

                               q4_0_cf4_kernel(weights, activations, output,
                                               ncols_x, ncols_y, nrows_dst,
                                               src1_stride_blocks, row_low, row_high,
                                               slm_A, slm_B, slm_C, slm_scales_A, slm_scales_B, slm_sums_B,
                                               item);
                           });
        });
    } else {
        constexpr int TOTAL_SLM_SIZE =
            SLM_A_SIZE + SLM_B_SIZE + SLM_C_SIZE * 4 + SLM_SCALES_A_SIZE * 4 + SLM_SCALES_B_SIZE * 4;

        ev = queue.submit([&](sycl::handler & h) {
            sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE), h);

            h.parallel_for(sycl::nd_range<2>(grid * block, block),
                           [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                               char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                               int offset = 0;
                               int8_t * slm_A = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_A_SIZE;
                               int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_B_SIZE;
                               int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                               offset += SLM_C_SIZE * 4;
                               float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                               offset += SLM_SCALES_A_SIZE * 4;
                               float * slm_scales_B = reinterpret_cast<float *>(shared + offset);

                               q8_0_cf4_kernel(weights, activations, output,
                                               ncols_x, ncols_y, nrows_dst,
                                               src1_stride_blocks, row_low, row_high,
                                               slm_A, slm_B, slm_C, slm_scales_A, slm_scales_B,
                                               item);
                           });
        });
    }

    if (events != nullptr) {
        events->push_back(ev);
    }
    return true;
}

inline bool launch_xmx_tile_soa(const ggml_sycl::mmvq_bench_args & args,
                                std::vector<sycl::event> * events,
                                std::string & error) {
    if (!validate_xmx_soa_args(args, error)) {
        return false;
    }

    const int row_low  = static_cast<int>(args.row_low);
    const int row_high = static_cast<int>(args.row_high);
    const int rows_span = row_high - row_low;
    const int num_row_tiles = (rows_span + XMX_M - 1) / XMX_M;
    const int num_col_tiles = (static_cast<int>(args.batch) + XMX_N - 1) / XMX_N;

    const int ncols_x = static_cast<int>(args.ncols);
    const int ncols_y = static_cast<int>(args.batch);
    const int nrows_dst = static_cast<int>(args.dst_row_stride);
    const int src1_stride_blocks = static_cast<int>(args.src1_padded_col_size / XMX_K);
    const uint8_t * activations = static_cast<const uint8_t *>(args.activations);
    float * output = args.output;

    const uint8_t * weight_qs = nullptr;
    const sycl::half * weight_d = nullptr;
    get_soa_weight_ptrs(args, weight_qs, weight_d);

    sycl::queue & queue = *args.stream;
    const sycl::range<2> grid(num_row_tiles, num_col_tiles);
    const sycl::range<2> block(1, XMX_SG_SIZE);

    sycl::event ev;
    if (args.weight_type == GGML_TYPE_Q4_0) {
        constexpr int TOTAL_SLM_SIZE =
            SLM_A_SIZE + SLM_B_SIZE + SLM_C_SIZE * 4 + SLM_SCALES_A_SIZE * 4 + SLM_SCALES_B_SIZE * 4 +
            SLM_SUMS_B_SIZE * 4;
        ev = queue.submit([&](sycl::handler & h) {
            sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE), h);

            h.parallel_for(sycl::nd_range<2>(grid * block, block),
                           [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                               char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                               int offset = 0;
                               int8_t * slm_A = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_A_SIZE;
                               int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_B_SIZE;
                               int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                               offset += SLM_C_SIZE * 4;
                               float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                               offset += SLM_SCALES_A_SIZE * 4;
                               float * slm_scales_B = reinterpret_cast<float *>(shared + offset);
                               offset += SLM_SCALES_B_SIZE * 4;
                               float * slm_sums_B = reinterpret_cast<float *>(shared + offset);

                               q4_0_tile_kernel_soa(weight_qs, weight_d, activations, output,
                                                    ncols_x, ncols_y, nrows_dst,
                                                    src1_stride_blocks, row_low, row_high,
                                                    slm_A, slm_B, slm_C, slm_scales_A, slm_scales_B, slm_sums_B,
                                                    item);
                           });
        });
    } else {
        constexpr int TOTAL_SLM_SIZE =
            SLM_A_SIZE + SLM_B_SIZE + SLM_C_SIZE * 4 + SLM_SCALES_A_SIZE * 4 + SLM_SCALES_B_SIZE * 4;
        ev = queue.submit([&](sycl::handler & h) {
            sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE), h);

            h.parallel_for(sycl::nd_range<2>(grid * block, block),
                           [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                               char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                               int offset = 0;
                               int8_t * slm_A = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_A_SIZE;
                               int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_B_SIZE;
                               int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                               offset += SLM_C_SIZE * 4;
                               float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                               offset += SLM_SCALES_A_SIZE * 4;
                               float * slm_scales_B = reinterpret_cast<float *>(shared + offset);

                               q8_0_tile_kernel_soa(weight_qs, weight_d, activations, output,
                                                    ncols_x, ncols_y, nrows_dst,
                                                    src1_stride_blocks, row_low, row_high,
                                                    slm_A, slm_B, slm_C, slm_scales_A, slm_scales_B,
                                                    item);
                           });
        });
    }

    if (events != nullptr) {
        events->push_back(ev);
    }
    return true;
}

inline bool launch_xmx_tile_db_soa(const ggml_sycl::mmvq_bench_args & args,
                                   std::vector<sycl::event> * events,
                                   std::string & error) {
    if (!validate_xmx_soa_args(args, error)) {
        return false;
    }

    const int row_low  = static_cast<int>(args.row_low);
    const int row_high = static_cast<int>(args.row_high);
    const int rows_span = row_high - row_low;
    const int num_row_tiles = (rows_span + XMX_M - 1) / XMX_M;
    const int num_col_tiles = (static_cast<int>(args.batch) + XMX_N - 1) / XMX_N;

    const int ncols_x = static_cast<int>(args.ncols);
    const int ncols_y = static_cast<int>(args.batch);
    const int nrows_dst = static_cast<int>(args.dst_row_stride);
    const int src1_stride_blocks = static_cast<int>(args.src1_padded_col_size / XMX_K);
    const uint8_t * activations = static_cast<const uint8_t *>(args.activations);
    float * output = args.output;

    const uint8_t * weight_qs = nullptr;
    const sycl::half * weight_d = nullptr;
    get_soa_weight_ptrs(args, weight_qs, weight_d);

    sycl::queue & queue = *args.stream;
    const sycl::range<2> grid(num_row_tiles, num_col_tiles);
    const sycl::range<2> block(1, XMX_SG_SIZE);

    sycl::event ev;
    if (args.weight_type == GGML_TYPE_Q4_0) {
        constexpr int TOTAL_SLM_SIZE =
            SLM_A_SIZE_DB + SLM_B_SIZE_DB + SLM_C_SIZE * 4 +
            SLM_SCALES_A_SIZE_DB * 4 + SLM_SCALES_B_SIZE_DB * 4 + SLM_SUMS_B_SIZE_DB * 4;
        ev = queue.submit([&](sycl::handler & h) {
            sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE), h);

            h.parallel_for(sycl::nd_range<2>(grid * block, block),
                           [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                               char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                               int offset = 0;
                               int8_t * slm_A = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_A_SIZE_DB;
                               int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_B_SIZE_DB;
                               int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                               offset += SLM_C_SIZE * 4;
                               float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                               offset += SLM_SCALES_A_SIZE_DB * 4;
                               float * slm_scales_B = reinterpret_cast<float *>(shared + offset);
                               offset += SLM_SCALES_B_SIZE_DB * 4;
                               float * slm_sums_B = reinterpret_cast<float *>(shared + offset);

                               q4_0_tile_db_kernel_soa(weight_qs, weight_d, activations, output,
                                                       ncols_x, ncols_y, nrows_dst,
                                                       src1_stride_blocks, row_low, row_high,
                                                       slm_A, slm_B, slm_C, slm_scales_A, slm_scales_B, slm_sums_B,
                                                       item);
                           });
        });
    } else {
        constexpr int TOTAL_SLM_SIZE =
            SLM_A_SIZE_DB + SLM_B_SIZE_DB + SLM_C_SIZE * 4 +
            SLM_SCALES_A_SIZE_DB * 4 + SLM_SCALES_B_SIZE_DB * 4;
        ev = queue.submit([&](sycl::handler & h) {
            sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE), h);

            h.parallel_for(sycl::nd_range<2>(grid * block, block),
                           [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                               char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                               int offset = 0;
                               int8_t * slm_A = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_A_SIZE_DB;
                               int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_B_SIZE_DB;
                               int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                               offset += SLM_C_SIZE * 4;
                               float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                               offset += SLM_SCALES_A_SIZE_DB * 4;
                               float * slm_scales_B = reinterpret_cast<float *>(shared + offset);

                               q8_0_tile_db_kernel_soa(weight_qs, weight_d, activations, output,
                                                       ncols_x, ncols_y, nrows_dst,
                                                       src1_stride_blocks, row_low, row_high,
                                                       slm_A, slm_B, slm_C, slm_scales_A, slm_scales_B,
                                                       item);
                           });
        });
    }

    if (events != nullptr) {
        events->push_back(ev);
    }
    return true;
}

inline bool launch_xmx_cf2_soa(const ggml_sycl::mmvq_bench_args & args,
                               std::vector<sycl::event> * events,
                               std::string & error) {
    if (!validate_xmx_soa_args(args, error)) {
        return false;
    }

    const int row_low  = static_cast<int>(args.row_low);
    const int row_high = static_cast<int>(args.row_high);
    const int rows_span = row_high - row_low;
    const int num_row_tiles = (rows_span + XMX_M - 1) / XMX_M;
    const int num_col_groups =
        (static_cast<int>(args.batch) + CF2_COL_TILES * XMX_N - 1) / (CF2_COL_TILES * XMX_N);

    const int ncols_x = static_cast<int>(args.ncols);
    const int ncols_y = static_cast<int>(args.batch);
    const int nrows_dst = static_cast<int>(args.dst_row_stride);
    const int src1_stride_blocks = static_cast<int>(args.src1_padded_col_size / XMX_K);
    const uint8_t * activations = static_cast<const uint8_t *>(args.activations);
    float * output = args.output;

    const uint8_t * weight_qs = nullptr;
    const sycl::half * weight_d = nullptr;
    get_soa_weight_ptrs(args, weight_qs, weight_d);

    sycl::queue & queue = *args.stream;
    const sycl::range<2> grid(num_row_tiles, num_col_groups);
    const sycl::range<2> block(CF2_COL_TILES, XMX_SG_SIZE);

    sycl::event ev;
    if (args.weight_type == GGML_TYPE_Q4_0) {
        constexpr int CF2_SLM_B_SIZE = CF2_COL_TILES * SLM_B_SIZE;
        constexpr int CF2_SLM_C_SIZE = CF2_COL_TILES * SLM_C_SIZE;
        constexpr int CF2_SCALES_B_SIZE = CF2_COL_TILES * SLM_SCALES_B_SIZE;
        constexpr int CF2_SUMS_B_SIZE = CF2_COL_TILES * SLM_SUMS_B_SIZE;
        constexpr int TOTAL_SLM_SIZE =
            SLM_A_SIZE + CF2_SLM_B_SIZE + CF2_SLM_C_SIZE * 4 + SLM_SCALES_A_SIZE * 4 +
            CF2_SCALES_B_SIZE * 4 + CF2_SUMS_B_SIZE * 4;

        ev = queue.submit([&](sycl::handler & h) {
            sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE), h);

            h.parallel_for(sycl::nd_range<2>(grid * block, block),
                           [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                               char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                               int offset = 0;
                               int8_t * slm_A = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_A_SIZE;
                               int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                               offset += CF2_SLM_B_SIZE;
                               int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                               offset += CF2_SLM_C_SIZE * 4;
                               float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                               offset += SLM_SCALES_A_SIZE * 4;
                               float * slm_scales_B = reinterpret_cast<float *>(shared + offset);
                               offset += CF2_SCALES_B_SIZE * 4;
                               float * slm_sums_B = reinterpret_cast<float *>(shared + offset);

                               q4_0_cf2_kernel_soa(weight_qs, weight_d, activations, output,
                                                   ncols_x, ncols_y, nrows_dst,
                                                   src1_stride_blocks, row_low, row_high,
                                                   slm_A, slm_B, slm_C, slm_scales_A, slm_scales_B, slm_sums_B,
                                                   item);
                           });
        });
    } else {
        constexpr int CF2_SLM_B_SIZE = CF2_COL_TILES * SLM_B_SIZE;
        constexpr int CF2_SLM_C_SIZE = CF2_COL_TILES * SLM_C_SIZE;
        constexpr int CF2_SCALES_B_SIZE = CF2_COL_TILES * SLM_SCALES_B_SIZE;
        constexpr int TOTAL_SLM_SIZE =
            SLM_A_SIZE + CF2_SLM_B_SIZE + CF2_SLM_C_SIZE * 4 + SLM_SCALES_A_SIZE * 4 +
            CF2_SCALES_B_SIZE * 4;

        ev = queue.submit([&](sycl::handler & h) {
            sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE), h);

            h.parallel_for(sycl::nd_range<2>(grid * block, block),
                           [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                               char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                               int offset = 0;
                               int8_t * slm_A = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_A_SIZE;
                               int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                               offset += CF2_SLM_B_SIZE;
                               int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                               offset += CF2_SLM_C_SIZE * 4;
                               float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                               offset += SLM_SCALES_A_SIZE * 4;
                               float * slm_scales_B = reinterpret_cast<float *>(shared + offset);

                               q8_0_cf2_kernel_soa(weight_qs, weight_d, activations, output,
                                                   ncols_x, ncols_y, nrows_dst,
                                                   src1_stride_blocks, row_low, row_high,
                                                   slm_A, slm_B, slm_C, slm_scales_A, slm_scales_B,
                                                   item);
                           });
        });
    }

    if (events != nullptr) {
        events->push_back(ev);
    }
    return true;
}

inline bool launch_xmx_cf4_soa(const ggml_sycl::mmvq_bench_args & args,
                               std::vector<sycl::event> * events,
                               std::string & error) {
    if (!validate_xmx_soa_args(args, error)) {
        return false;
    }

    const int row_low  = static_cast<int>(args.row_low);
    const int row_high = static_cast<int>(args.row_high);
    const int rows_span = row_high - row_low;
    const int num_row_tiles = (rows_span + XMX_M - 1) / XMX_M;
    const int num_col_groups =
        (static_cast<int>(args.batch) + CF4_COL_TILES * XMX_N - 1) / (CF4_COL_TILES * XMX_N);

    const int ncols_x = static_cast<int>(args.ncols);
    const int ncols_y = static_cast<int>(args.batch);
    const int nrows_dst = static_cast<int>(args.dst_row_stride);
    const int src1_stride_blocks = static_cast<int>(args.src1_padded_col_size / XMX_K);
    const uint8_t * activations = static_cast<const uint8_t *>(args.activations);
    float * output = args.output;

    const uint8_t * weight_qs = nullptr;
    const sycl::half * weight_d = nullptr;
    get_soa_weight_ptrs(args, weight_qs, weight_d);

    sycl::queue & queue = *args.stream;
    const sycl::range<2> grid(num_row_tiles, num_col_groups);
    const sycl::range<2> block(1, XMX_SG_SIZE);

    sycl::event ev;
    if (args.weight_type == GGML_TYPE_Q4_0) {
        constexpr int TOTAL_SLM_SIZE =
            SLM_A_SIZE + SLM_B_SIZE + SLM_C_SIZE * 4 + SLM_SCALES_A_SIZE * 4 +
            SLM_SCALES_B_SIZE * 4 + SLM_SUMS_B_SIZE * 4;

        ev = queue.submit([&](sycl::handler & h) {
            sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE), h);

            h.parallel_for(sycl::nd_range<2>(grid * block, block),
                           [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                               char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                               int offset = 0;
                               int8_t * slm_A = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_A_SIZE;
                               int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_B_SIZE;
                               int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                               offset += SLM_C_SIZE * 4;
                               float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                               offset += SLM_SCALES_A_SIZE * 4;
                               float * slm_scales_B = reinterpret_cast<float *>(shared + offset);
                               offset += SLM_SCALES_B_SIZE * 4;
                               float * slm_sums_B = reinterpret_cast<float *>(shared + offset);

                               q4_0_cf4_kernel_soa(weight_qs, weight_d, activations, output,
                                                   ncols_x, ncols_y, nrows_dst,
                                                   src1_stride_blocks, row_low, row_high,
                                                   slm_A, slm_B, slm_C, slm_scales_A, slm_scales_B, slm_sums_B,
                                                   item);
                           });
        });
    } else {
        constexpr int TOTAL_SLM_SIZE =
            SLM_A_SIZE + SLM_B_SIZE + SLM_C_SIZE * 4 + SLM_SCALES_A_SIZE * 4 + SLM_SCALES_B_SIZE * 4;

        ev = queue.submit([&](sycl::handler & h) {
            sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE), h);

            h.parallel_for(sycl::nd_range<2>(grid * block, block),
                           [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                               char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                               int offset = 0;
                               int8_t * slm_A = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_A_SIZE;
                               int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                               offset += SLM_B_SIZE;
                               int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                               offset += SLM_C_SIZE * 4;
                               float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                               offset += SLM_SCALES_A_SIZE * 4;
                               float * slm_scales_B = reinterpret_cast<float *>(shared + offset);

                               q8_0_cf4_kernel_soa(weight_qs, weight_d, activations, output,
                                                   ncols_x, ncols_y, nrows_dst,
                                                   src1_stride_blocks, row_low, row_high,
                                                   slm_A, slm_B, slm_C, slm_scales_A, slm_scales_B,
                                                   item);
                           });
        });
    }

    if (events != nullptr) {
        events->push_back(ev);
    }
    return true;
}

#else

inline bool launch_xmx_tile(const ggml_sycl::mmvq_bench_args &,
                            std::vector<sycl::event> *,
                            std::string & error) {
    error = "SYCL joint_matrix unavailable; XMX tier2 kernels disabled.";
    return false;
}

inline bool launch_xmx_tile_db(const ggml_sycl::mmvq_bench_args &,
                               std::vector<sycl::event> *,
                               std::string & error) {
    error = "SYCL joint_matrix unavailable; XMX tier2 kernels disabled.";
    return false;
}

inline bool launch_xmx_cf2(const ggml_sycl::mmvq_bench_args &,
                           std::vector<sycl::event> *,
                           std::string & error) {
    error = "SYCL joint_matrix unavailable; XMX tier2 kernels disabled.";
    return false;
}

inline bool launch_xmx_cf4(const ggml_sycl::mmvq_bench_args &,
                           std::vector<sycl::event> *,
                           std::string & error) {
    error = "SYCL joint_matrix unavailable; XMX tier2 kernels disabled.";
    return false;
}

inline bool launch_xmx_tile_soa(const ggml_sycl::mmvq_bench_args &,
                                std::vector<sycl::event> *,
                                std::string & error) {
    error = "SYCL joint_matrix unavailable; XMX tier2 kernels disabled.";
    return false;
}

inline bool launch_xmx_tile_db_soa(const ggml_sycl::mmvq_bench_args &,
                                   std::vector<sycl::event> *,
                                   std::string & error) {
    error = "SYCL joint_matrix unavailable; XMX tier2 kernels disabled.";
    return false;
}

inline bool launch_xmx_cf2_soa(const ggml_sycl::mmvq_bench_args &,
                               std::vector<sycl::event> *,
                               std::string & error) {
    error = "SYCL joint_matrix unavailable; XMX tier2 kernels disabled.";
    return false;
}

inline bool launch_xmx_cf4_soa(const ggml_sycl::mmvq_bench_args &,
                               std::vector<sycl::event> *,
                               std::string & error) {
    error = "SYCL joint_matrix unavailable; XMX tier2 kernels disabled.";
    return false;
}

#endif  // MMVQ_TIER2_XMX_AVAILABLE

}  // namespace mmvq_tier2
}  // namespace sycl_bench
