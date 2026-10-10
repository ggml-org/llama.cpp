#ifndef GGML_SYCL_FUSED_GEMM_HPP
#define GGML_SYCL_FUSED_GEMM_HPP

#include "common.hpp"


// Routed row n is at base + (map[n].i1 % ne1)*nb1 + map[n].i2*nb2.
struct ggml_sycl_gg_rows {
    char *                   base;
    const mmid_row_mapping * map;
    int64_t                  ne1;
    size_t                   nb1;
    size_t                   nb2;
};

static constexpr int GGML_SYCL_FG_BN = 32;
static constexpr int64_t GGML_SYCL_MMID_SCHED_MAX_EXPERTS = 4096;

// Shape and type gates for the kernels below. Device capability is separate: it needs a queue to ask.
static constexpr int GGML_SYCL_FG_MAX_N = 64; // widest N taken; each shape covers it in BN-wide tiles

// weight formats the fused A stage decodes; K must cover whole stored blocks
constexpr bool ggml_sycl_fused_dequant_gemm_type_ok(ggml_type src0_type, int64_t K) {
    // iq4_nl and q8_0 store 32 values per block; every other format here is a 256-value superblock
    // that the A stage walks in steps of 32, so K must cover whole superblocks.
    if (src0_type == GGML_TYPE_IQ4_NL || src0_type == GGML_TYPE_Q8_0) {
        return K % 32 == 0;
    }
    const bool superblock =
           src0_type == GGML_TYPE_Q2_K ||
           src0_type == GGML_TYPE_Q3_K ||
           src0_type == GGML_TYPE_Q4_K ||
           src0_type == GGML_TYPE_Q5_K ||
           src0_type == GGML_TYPE_Q6_K ||
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

constexpr bool ggml_sycl_fused_dequant_gemm_shape_ok(ggml_type src0_type, int64_t M, int64_t N, int64_t K,
                                                     int64_t ldd) {
    return ggml_sycl_fused_dequant_gemm_type_ok(src0_type, K) && M > 0 && N > 0 && K > 0 &&
           N <= GGML_SYCL_FG_MAX_N &&
           M <= INT32_MAX && N <= INT32_MAX && K <= INT32_MAX && ldd <= INT32_MAX;
}

// grouped variant: the per-expert fused kernel is only worth it while each expert is narrow,
// so wider average slices are left to the per-expert library GEMM loop
constexpr bool ggml_sycl_grouped_dequant_gemm_shape_ok(ggml_type src0_type, int64_t M, int64_t K,
                                                       int64_t total_rows, int64_t n_active, bool unbounded = false) {
    return ggml_sycl_fused_dequant_gemm_shape_ok(src0_type, M, 1, K, M) && total_rows > 0 &&
           total_rows <= INT32_MAX && (unbounded || total_rows <= n_active * GGML_SYCL_FG_MAX_N);
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
        case GGML_TYPE_Q8_0:    return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_Q8_0    ) != 0;
        case GGML_TYPE_Q2_K:    return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_Q2_K    ) != 0;
        case GGML_TYPE_Q3_K:    return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_Q3_K    ) != 0;
        case GGML_TYPE_Q4_K:    return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_Q4_K    ) != 0;
        case GGML_TYPE_Q5_K:    return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_Q5_K    ) != 0;
        case GGML_TYPE_Q6_K:    return (g_ggml_sycl_xmx_gather_types & GGML_SYCL_XMX_GATHER_Q6_K    ) != 0;
        default:          return false;
    }
}

// dst[n*ldd + m] = sum_k dequant(src0)[m*K + k] * src1[n*K + k], src1 is F32, F16 or BF16.
// The XMX combination is picked per call from the src1 type and its precision request src1_prec
// (op_params[3], [TAG_GGML_PREC]); the accumulator is f32, which meets any request.
// q8_0 and the k-quants are not handled here, only in the grouped path below.
// Returns false when the case is not handled (type, device, precision, or shape).
bool ggml_sycl_fused_dequant_gemm(ggml_type src0_type, const void * src0, const void * src1, ggml_type src1_type,
                                  int32_t src1_prec, float * dst, int64_t M, int64_t N, int64_t K, int64_t ldd,
                                  ggml_sycl_pool & pool, dpct::queue_ptr stream);

// One launch for every expert of a MUL_MAT_ID: rows of src1/dst are grouped by expert, expert e
// owns rows [expert_row_offsets[e], expert_row_offsets[e+1]) and reads its weights at
// src0_base + e*expert_stride. tiles is host scratch that must stay alive until the queue drains.
// reordered: each expert slice is in the reorder (SoA) layout of reorder_qw().
// dst[n*M + m] = sum_k dequant(src0_e)[m*K + k] * src1[n*K + k]
// Returns false when the case is not handled (type, layout, device, precision, or shape).
bool ggml_sycl_grouped_dequant_gemm(ggml_type src0_type, bool reordered, const void * src0_base, size_t expert_stride,
                                    const float * src1, int32_t src1_prec, float * dst,
                                    const int64_t * expert_row_offsets, int64_t n_as, int64_t M, int64_t K,
                                    int64_t total_rows, std::vector<ggml_sycl_gg_tile> & tiles,
                                    ggml_sycl_pool & pool, dpct::queue_ptr stream);

// Build routing and tile tables on the device; unused tile slots are empty.
bool ggml_sycl_build_mmid_schedule(const int32_t * ids_dev, size_t ids_token_stride, int64_t n_as, int64_t n_ids,
                                   int64_t n_tokens, int64_t n_tiles_max, mmid_row_mapping * row_mapping,
                                   ggml_sycl_gg_tile * tiles, uint32_t * expert_offsets, dpct::queue_ptr stream);

bool ggml_sycl_grouped_dequant_gemm_device_supported(ggml_type src0_type, bool reordered, int32_t src1_prec,
                                                      int64_t n_as, int64_t n_ids, int64_t n_tokens,
                                                      int64_t M, int64_t K, dpct::queue_ptr stream);

// Returns false before queuing work when the type, shape, layout or device is unsupported.
bool ggml_sycl_grouped_dequant_gemm_device(ggml_type src0_type, bool reordered, const void * src0_base,
                                           size_t expert_stride, const ggml_sycl_gg_rows & src1, int32_t src1_prec,
                                           const ggml_sycl_gg_rows & dst, const int32_t * ids_dev,
                                           size_t ids_token_stride, int64_t n_as, int64_t n_ids, int64_t n_tokens,
                                           int64_t M, int64_t K, ggml_sycl_pool & pool, dpct::queue_ptr stream);

#endif // GGML_SYCL_FUSED_GEMM_HPP
