#ifndef GGML_SYCL_FUSED_GEMM_HPP
#define GGML_SYCL_FUSED_GEMM_HPP

#include "common.hpp"


// Shape and type gates for the kernels below. Device capability is separate: it needs a queue to ask.
static constexpr int GGML_SYCL_FG_MAX_N = 64; // 2 * FG_BN

// weight formats the fused A stage decodes; K must cover whole stored blocks
constexpr bool ggml_sycl_fused_dequant_gemm_f16_type_ok(ggml_type src0_type, int64_t K) {
    // iq4_nl stores 32 values per block; every other format here is a 256-value superblock that
    // the A stage walks in steps of 32, so K must cover whole superblocks.
    if (src0_type == GGML_TYPE_IQ4_NL) {
        return K % QK4_NL == 0;
    }
    const bool superblock =
           src0_type == GGML_TYPE_IQ3_S ||
           src0_type == GGML_TYPE_IQ4_XS ||
           src0_type == GGML_TYPE_IQ3_XXS ||
           src0_type == GGML_TYPE_IQ2_XXS ||
           src0_type == GGML_TYPE_IQ2_XS ||
           src0_type == GGML_TYPE_IQ2_S ||
           src0_type == GGML_TYPE_IQ1_S ||
           src0_type == GGML_TYPE_IQ1_M;
    return superblock && QK_K == 256 && K % QK_K == 0;
}

constexpr bool ggml_sycl_fused_dequant_gemm_f16_shape_ok(ggml_type src0_type, int64_t M, int64_t N, int64_t K,
                                                         int64_t ldd) {
    return ggml_sycl_fused_dequant_gemm_f16_type_ok(src0_type, K) && M > 0 && N > 0 && K > 0 &&
           N <= GGML_SYCL_FG_MAX_N &&
           M <= INT32_MAX && N <= INT32_MAX && K <= INT32_MAX && ldd <= INT32_MAX;
}

// grouped variant: the per-expert fused kernel is only worth it while each expert is narrow,
// so wider average slices are left to the per-expert library GEMM loop
constexpr bool ggml_sycl_grouped_dequant_gemm_f16_shape_ok(ggml_type src0_type, int64_t M, int64_t K,
                                                           int64_t total_rows, int64_t n_active) {
    return ggml_sycl_fused_dequant_gemm_f16_shape_ok(src0_type, M, 1, K, M) && total_rows > 0 &&
           total_rows <= INT32_MAX && total_rows <= n_active * GGML_SYCL_FG_MAX_N;
}

// Runtime type gate, kept out of the constexpr predicates above so those stay pure.
inline bool ggml_sycl_xmx_gather_type_enabled(ggml_type src0_type) {
    switch (src0_type) {
        case GGML_TYPE_IQ4_NL:  return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ4_NL  ) != 0;
        case GGML_TYPE_IQ3_S:   return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ3_S   ) != 0;
        case GGML_TYPE_IQ4_XS:  return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ4_XS  ) != 0;
        case GGML_TYPE_IQ3_XXS: return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ3_XXS ) != 0;
        case GGML_TYPE_IQ2_XXS: return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ2_XXS ) != 0;
        case GGML_TYPE_IQ2_XS:  return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ2_XS  ) != 0;
        case GGML_TYPE_IQ2_S:   return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ2_S   ) != 0;
        case GGML_TYPE_IQ1_S:   return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ1_S   ) != 0;
        case GGML_TYPE_IQ1_M:   return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_IQ1_M   ) != 0;
        default:          return false;
    }
}

// True if the device can run the kernel at all; cached, so it is cheap to ask per node.
bool ggml_sycl_fused_dequant_gemm_f16_device_ok(dpct::queue_ptr stream);

// dst[n*ldd + m] = sum_k dequant(src0)[m*K + k] * src1_f16[n*K + k]
// Returns false when the case is not handled (type, device, or shape).
bool ggml_sycl_fused_dequant_gemm_f16(ggml_type src0_type, const void * src0, const sycl::half * src1_f16, float * dst,
                                      int64_t M, int64_t N, int64_t K, int64_t ldd, ggml_sycl_pool & pool,
                                      dpct::queue_ptr stream);

// One launch for every expert of a MUL_MAT_ID: rows of src1/dst are grouped by expert, expert e
// owns rows [expert_row_offsets[e], expert_row_offsets[e+1]) and reads its weights at
// src0_base + e*expert_stride. tiles is host scratch that must stay alive until the queue drains.
// dst[n*M + m] = sum_k dequant(src0_e)[m*K + k] * src1[n*K + k]
// Returns false when the case is not handled (type, device, or shape).
bool ggml_sycl_grouped_dequant_gemm_f16(ggml_type src0_type, const void * src0_base, size_t expert_stride,
                                        const float * src1, float * dst, const int64_t * expert_row_offsets,
                                        int64_t n_as, int64_t M, int64_t K, int64_t total_rows,
                                        std::vector<ggml_sycl_gg_tile> & tiles, ggml_sycl_pool & pool,
                                        dpct::queue_ptr stream);

#endif // GGML_SYCL_FUSED_GEMM_HPP
