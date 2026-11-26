#include <assert.h>
#include <HAP_compute_res.h>
#include <HAP_farf.h>
#include <math.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "dma-queue.h"
#include "hex-fastdiv.h"
#include "htp-trace.h"
#include "hmx-queue.h"
#include "hmx-utils.h"
#include "hvx-utils.h"
#include "hvx-dump.h"
#include "hvx-copy.h"
#include "hvx-reduce.h"
#include "hvx-flash-attn.h"
#include "htp-vtcm.h"
#include "work-queue.h"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-tensor.h"
#include "hvx-quant.h"

#include "flash-attn-ops.h"
#include "hvx-fa-kernels.h"
#include "hmx-fa-kernels.h"
#include "hmx-fa-w8a8-kernels.h"

// Must be multiple of 32
#define FLASH_ATTN_BLOCK_SIZE (32 * 2)

// Large scratch lives in ctx->ddr_spad to keep stack usage low.
// Ops can run on the dspqueue callback thread, which has a small stack.
struct hvx_fa_ddr_spad {
    __fp16   slopes[512];
    uint16_t active_blocks[FA_MAX_KV_BLOCKS];
    uint16_t thread_active_blocks[HTP_MAX_NHVX][FA_MAX_KV_BLOCKS] __attribute__((aligned(128)));
    float    split_m[HTP_MAX_NHVX][FA_HVX_G_MAX] __attribute__((aligned(128)));
    float    split_s[HTP_MAX_NHVX][FA_HVX_G_MAX] __attribute__((aligned(128)));
};

struct hmx_fa_ddr_spad {
    uint16_t active_blocks[FA_MAX_KV_BLOCKS];
    __fp16   slopes[] __attribute__((aligned(128))); // [g_br]
};

static inline void * fa_get_ddr_spad(const struct htp_ops_context * octx, size_t size) {
    if (!octx->ctx->ddr_spad_base || octx->ctx->ddr_spad_size < size) {
        return NULL;
    }
    return octx->ctx->ddr_spad_base;
}

struct htp_fa_context {
    const struct htp_ops_context * octx;

    struct fastdiv_values src0_div21;
    struct fastdiv_values src0_div1;

    struct fastdiv_values broadcast_rk2;
    struct fastdiv_values broadcast_rk3;
    struct fastdiv_values broadcast_rv2;
    struct fastdiv_values broadcast_rv3;

    struct fastdiv_values src3_div2;
    struct fastdiv_values src3_div3;

    float scale;
    float max_bias;
    bool  has_softcap;
    __fp16 logit_softcap;

    uint32_t n_head_log2;
    float m0;
    float m1;

    uint32_t n_blocks;
    uint32_t n_active_blocks;
    bool     shared_active_blocks;

    size_t size_q_row_padded;
    size_t size_k_row_padded;
    size_t size_v_row_padded;

    size_t size_k_block;
    size_t size_v_block;
    size_t size_m_block;

    uint32_t qrows;
    uint32_t qrows_per_hvx;
    uint32_t qrow_start;

    uint32_t kv_head_start;
    uint32_t n_kv_local;
    uint32_t G;
    uint32_t splits_per_kv;

    bool is_q_fp32;

    size_t size_q_block;
    size_t size_vkq_acc;

    uint8_t *  spad_q;
    uint8_t *  spad_k;
    uint8_t *  spad_v;
    uint8_t *  spad_m;
    uint8_t *  spad_a;
    float *    spad_sinks;

    struct hvx_fa_ddr_spad * ddr;
    dma_cache_dm *           m_cache; // [n_hvx]

    const struct htp_tensor * k;
    const struct htp_tensor * v;
};

struct hmx_fa_context {
    const struct htp_ops_context * octx;
    const struct htp_tensor *      sinks;  // attention sinks (src[4]), NULL if absent
    const struct htp_tensor *      k;
    const struct htp_tensor *      v;
    bool         pipeline;  // true when n_kv_blocks >= FA_MIN_KV_BLOCKS && n_hvx >= 2
    bool         is_w8a8;
    uint32_t     n_hvx;

    // Op parameters
    __fp16       scale;
    float        scale_f32;
    float        max_bias;
    bool         has_softcap;
    __fp16       logit_softcap;
    uint32_t     n_head_log2;
    float        m0, m1;

    // Dimensions
    uint32_t     DK, DV;
    uint32_t     DK_pad, DV_pad;  // head_dim rounded up to 64 for HMX tiling
    uint32_t     n_kv;        // kv_len
    uint32_t     n_kv_heads;  // number of KV heads
    uint32_t     n_heads;     // number of Q heads
    uint32_t     G;           // GQA factor = n_heads / n_kv_heads
    struct fastdiv_values div_G;
    struct fastdiv_values src3_div2;
    struct fastdiv_values src3_div3;
    uint32_t     n_kv_blocks;
    uint32_t     neq1;        // Q token count

    // Types
    bool         is_q_fp32;
    bool         is_dst_fp32;

    // Dynamic block sizes
    uint32_t     Br;    // Q tokens per block (before GQA expansion)
    uint32_t     Bc;
    uint32_t     g_br;  // hex_align_up(G * Br, 32) - actual tile row dim

    // VTCM buffers (allocated by vtcm_seq_alloc)
    __fp16 *     vtcm_q_dma;           // Q DMA fetch buffer
    __fp16 *     vtcm_q_tiles;         // Q tile format [g_br, D]
    __fp16 *     vtcm_o_tiles[2];      // O ping-pong [g_br, D]
    __fp16 *     vtcm_k_fp16[2];       // K DMA double-buffer [Bc, D]
    __fp16 *     vtcm_v_fp16[2];       // V DMA double-buffer [Bc, D]
    __fp16 *     vtcm_k_tiles[2];      // K tiles (transposed, double-buffered)
    __fp16 *     vtcm_v_tiles[2];      // V tiles (column-major, double-buffered)
    __fp16 *     vtcm_s_tiles[2];      // S = QK^T [g_br, Bc] (double-buffered)
    __fp16 *     vtcm_p_tiles[2];      // P = softmax(S) [g_br, Bc]
    __fp16 *     vtcm_d_tiles[2];      // Diagonal rescale, g_br/32 packed diagonal tiles (double-buffered)
    __fp16 *     vtcm_d_inv_l;         // Diagonal rescale (1/l), same packed layout
    HVX_Vector * vtcm_m_vec;           // Row max [g_br]
    HVX_Vector * vtcm_l_vec;           // Row sum [g_br]
    HVX_Vector * vtcm_s_rowmax;        // Softmax intermediate [g_br]
    HVX_Vector * vtcm_p_rowsum;        // Softmax intermediate [g_br]
    HVX_Vector * vtcm_row_bufs;        // Per-HVX softmax row scratch [n_hvx][2][Bc/64]
    uint8_t *    vtcm_hmx_scales_id;   // HMX output scales (identity)
    uint8_t *    vtcm_hmx_scales_qk;   // HMX output scales (qk_scale)
    __fp16 *     vtcm_mask_buf;        // VTCM mask buffer [Br * m_line], DMA'd per KV block
    __fp16 *     vtcm_slopes;          // ALiBi slopes [g_br]
    float *      vtcm_sinks;           // Attention sinks
    float *      vtcm_q_scales;        // W8A8 Q per-row scales [g_br]
    float *      vtcm_k_scales[2];     // W8A8 K per-col scales [Bc]
    uint8_t *    vtcm_k_bias[2];       // W8A8 K HMX bias banks [n_col_tiles * 2 * 256]
    float *      vtcm_o_f32;           // W8A8 FP32 O accumulator [g_br, DV]
    uint8_t *    vtcm_do_tiles;        // W8A8 PV output tiles [n_row_tiles_64, DV/32, 2]
    float *      vtcm_v_scales[2];     // W8A8 V per-col scales [DV]
    uint8_t *    vtcm_v_bias[2];       // W8A8 V HMX bias banks [DV/32 * 2 * 256]
    float *      vtcm_p_scales[2];     // W8A8 P per-row scales [g_br]
    float *      vtcm_exp_mdiff[2];    // W8A8 exp2(m_prev - m_curr) [g_br]
    size_t       k_raw_row_padded;     // W8A8 raw Q8_0 K row stride in VTCM
    size_t       v_raw_row_padded;     // W8A8 raw Q8_0 V row stride in VTCM
    size_t       o_f32_bytes;
    __fp16 *     ddr_slopes;           // ALiBi slopes staging buffer in DDR [g_br]
    size_t       row_buf_stride;       // HVX vectors per row buffer (Bc/64)
    size_t       mask_buf_row_stride;  // elements (__fp16) per row in mask buffer
    size_t       q_tile_bytes;
    size_t       k_tile_bytes;
    size_t       o_tile_bytes;
    size_t       col_vec_bytes;
    size_t       d_tile_bytes;
    bool         mask_broadcast;       // true when mask->ne[2] == 1 (head-independent, single 2D DMA)
    dma_cache_fa * m_cache;
};

// Scan mask rows [q_start, q_start + n_rows_q) and build compact list of KV blocks with any entry != -inf.
static uint32_t fa_scan_active_blocks(
        dma_queue *               dma_q,
        const struct htp_tensor * mask,
        uint32_t                  q_start,
        uint32_t                  n_rows_q,
        uint32_t                  im2,
        uint32_t                  im3,
        uint32_t                  nek1,
        uint32_t                  Bc,
        uint32_t                  n_kv_blocks,
        void *                    vtcm_scratch,
        size_t                    scratch_bytes,
        uint16_t *                active_blocks) {
    const size_t row_bytes  = (size_t) nek1 * sizeof(__fp16);
    const size_t row_stride = hex_round_up(row_bytes, 128);

    if (!mask || n_kv_blocks <= 1 || n_kv_blocks > FA_MAX_KV_BLOCKS ||
        (size_t) n_rows_q * row_stride > scratch_bytes) {
        #pragma clang loop vectorize(disable)
        for (uint32_t b = 0; b < n_kv_blocks; ++b) {
            active_blocks[b] = (uint16_t) b;
        }
        return n_kv_blocks;
    }

    const dma_addr_t m_src = mask->data + q_start * mask->nb[1] + im2 * mask->nb[2] + im3 * mask->nb[3];
    dma_queue_push(dma_q, dma_make_data(vtcm_scratch, m_src), row_stride, mask->nb[1], row_bytes, n_rows_q);
    const uint8_t * m_buf = (const uint8_t *) dma_queue_pop(dma_q).dst;

    const HVX_Vector v_ninf   = Q6_Vh_vsplat_R(0xFC00);
    uint32_t         n_active = 0;

    for (uint32_t b = 0; b < n_kv_blocks; ++b) {
        const uint32_t col0 = b * Bc;
        const uint32_t cols = MIN(Bc, nek1 - col0);

        for (uint32_t r = 0; r < n_rows_q; ++r) {
            const __fp16 * row_ptr = (const __fp16 *) (m_buf + r * row_stride) + col0;
            HVX_Vector v_any = Q6_V_vzero();
            uint32_t c = 0;
            for (; c + 64 <= cols; c += 64) {
                const HVX_Vector v_m = *(const HVX_Vector *) (row_ptr + c);
                v_any = Q6_V_vor_VV(v_any, Q6_V_vxor_VV(v_m, v_ninf));
            }
            if (c < cols) {
                const uint32_t rem = cols - c;
                const HVX_VectorPred q_rem = Q6_Q_vsetq2_R(rem * sizeof(__fp16));
                const HVX_Vector v_m = *(const HVX_Vector *) (row_ptr + c);
                const HVX_Vector v_diff = Q6_V_vand_QV(q_rem, Q6_V_vxor_VV(v_m, v_ninf));
                v_any = Q6_V_vor_VV(v_any, v_diff);
            }
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 64));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 32));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 16));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 8));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 4));
            if (Q6_R_vextract_VR(v_any, 0) != 0) {
                active_blocks[n_active++] = (uint16_t) b;
                break;
            }
        }
    }

    if (n_active == 0 && n_kv_blocks > 0) {
        active_blocks[0] = 0;
        n_active = 1;
    }
    return n_active;
}

static __attribute__((noinline)) void fa_hvx_zero_acc(float * dst, uint32_t n) {
    hvx_splat_f32_a((uint8_t *) dst, 0, n);
}

static __attribute__((noinline)) void fa_hvx_q_f32_to_f16(uint8_t * dst, uint32_t DK) {
    hvx_copy_f16_f32_aa(dst, dst, DK);
}

static __attribute__((noinline)) float fa_hvx_exp_scalar(float x) {
    return hvx_vec_get_f32(hvx_vec_exp_f32(hvx_vec_splat_f32(x)));
}

static __attribute__((noinline)) void fa_hvx_split_exp_weights(float * dst, const float * src) {
    *(HVX_Vector *) dst = hvx_vec_exp_f32(*(const HVX_Vector *) src);
}

static __attribute__((noinline)) void fa_hvx_dequant_q8_0_block(
        uint8_t * base,
        size_t    row_stride,
        uint32_t  n_rows,
        uint32_t  dim) {
    for (uint32_t r = 0; r < n_rows; ++r) {
        __fp16 * row = (__fp16 *) (base + r * row_stride);
        hvx_dequantize_row_flat_q8_0_f16(row, row, dim);
    }
}

// Accumulate PV in dealt HVX_VectorPair order.
static __attribute__((always_inline)) inline void fa_hvx_pv_accum_fixed(
        float * restrict         vkq,
        const uint8_t * restrict v_base,
        size_t                   v_stride,
        HVX_Vector               P,
        HVX_Vector               ms_vec,
        uint32_t                 block_size,
        const uint32_t           n_pairs) {
    HVX_VectorPair * restrict vp = (HVX_VectorPair *) vkq;
    HVX_VectorPair acc[4];

    #pragma clang loop unroll(full)
    for (uint32_t p = 0; p < n_pairs; ++p) {
        acc[p] = Q6_W_vcombine_VV(
            HVX_OP_MUL_F32(Q6_V_hi_W(vp[p]), ms_vec),
            HVX_OP_MUL_F32(Q6_V_lo_W(vp[p]), ms_vec)
        );
    }

    const uint8_t * v_ptr   = v_base;
    const size_t    stride2 = v_stride * 2;
    uint32_t j = 0;
    for (; j + 1 < block_size; j += 2) {
        const HVX_Vector   S0  = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
        const HVX_Vector   S1  = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
        const HVX_Vector * vx0 = (const HVX_Vector *) v_ptr;
        const HVX_Vector * vx1 = (const HVX_Vector *) (v_ptr + v_stride);

        #pragma clang loop unroll(full)
        for (uint32_t p = 0; p < n_pairs; ++p) {
            acc[p] = hvx_vec_mpyacc_f32_f16(acc[p], vx0[p], S0);
            acc[p] = hvx_vec_mpyacc_f32_f16(acc[p], vx1[p], S1);
        }
        v_ptr += stride2;
    }
    if (j < block_size) {
        const HVX_Vector   S0  = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
        const HVX_Vector * vx0 = (const HVX_Vector *) v_ptr;
        #pragma clang loop unroll(full)
        for (uint32_t p = 0; p < n_pairs; ++p) {
            acc[p] = hvx_vec_mpyacc_f32_f16(acc[p], vx0[p], S0);
        }
    }

    #pragma clang loop unroll(full)
    for (uint32_t p = 0; p < n_pairs; ++p) {
        vp[p] = acc[p];
    }
}

static __attribute__((noinline)) void fa_hvx_pv_accum_64(
        float * restrict vkq, const uint8_t * restrict v_base, size_t v_stride,
        HVX_Vector P, HVX_Vector ms_vec, uint32_t block_size) {
    fa_hvx_pv_accum_fixed(vkq, v_base, v_stride, P, ms_vec, block_size, 1);
}

static __attribute__((noinline)) void fa_hvx_pv_accum_128(
        float * restrict vkq, const uint8_t * restrict v_base, size_t v_stride,
        HVX_Vector P, HVX_Vector ms_vec, uint32_t block_size) {
    fa_hvx_pv_accum_fixed(vkq, v_base, v_stride, P, ms_vec, block_size, 2);
}

static __attribute__((noinline)) void fa_hvx_pv_accum_256(
        float * restrict vkq, const uint8_t * restrict v_base, size_t v_stride,
        HVX_Vector P, HVX_Vector ms_vec, uint32_t block_size) {
    fa_hvx_pv_accum_fixed(vkq, v_base, v_stride, P, ms_vec, block_size, 4);
}

static __attribute__((noinline)) void fa_hvx_pv_accum_generic(
        float * restrict         vkq,
        const uint8_t * restrict v_base,
        size_t                   v_stride,
        HVX_Vector               P,
        HVX_Vector               ms_vec,
        uint32_t                 block_size,
        uint32_t                 DV) {
    const uint32_t n_pairs = (DV + 63) / 64;
    HVX_VectorPair * restrict vp = (HVX_VectorPair *) vkq;
    for (uint32_t p = 0; p < n_pairs; ++p) {
        HVX_VectorPair v = vp[p];
        vp[p] = Q6_W_vcombine_VV(
            HVX_OP_MUL_F32(Q6_V_hi_W(v), ms_vec),
            HVX_OP_MUL_F32(Q6_V_lo_W(v), ms_vec)
        );
    }

    const uint8_t * v_ptr   = v_base;
    const size_t    stride2 = v_stride * 2;
    uint32_t j = 0;
    for (; j + 1 < block_size; j += 2) {
        const HVX_Vector   S0  = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
        const HVX_Vector   S1  = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
        const HVX_Vector * vx0 = (const HVX_Vector *) v_ptr;
        const HVX_Vector * vx1 = (const HVX_Vector *) (v_ptr + v_stride);
        for (uint32_t p = 0; p < n_pairs; ++p) {
            HVX_VectorPair acc = vp[p];
            acc = hvx_vec_mpyacc_f32_f16(acc, vx0[p], S0);
            acc = hvx_vec_mpyacc_f32_f16(acc, vx1[p], S1);
            vp[p] = acc;
        }
        v_ptr += stride2;
    }
    if (j < block_size) {
        const HVX_Vector   S0  = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
        const HVX_Vector * vx0 = (const HVX_Vector *) v_ptr;
        for (uint32_t p = 0; p < n_pairs; ++p) {
            vp[p] = hvx_vec_mpyacc_f32_f16(vp[p], vx0[p], S0);
        }
    }
}

static inline void fa_hvx_pv_accum(
        float * restrict         vkq,
        const uint8_t * restrict v_base,
        size_t                   v_stride,
        HVX_Vector               P,
        HVX_Vector               ms_vec,
        uint32_t                 block_size,
        uint32_t                 DV) {
    if (DV == 64) {
        fa_hvx_pv_accum_64(vkq, v_base, v_stride, P, ms_vec, block_size);
    } else if (DV == 128) {
        fa_hvx_pv_accum_128(vkq, v_base, v_stride, P, ms_vec, block_size);
    } else if (DV == 256) {
        fa_hvx_pv_accum_256(vkq, v_base, v_stride, P, ms_vec, block_size);
    } else {
        fa_hvx_pv_accum_generic(vkq, v_base, v_stride, P, ms_vec, block_size, DV);
    }
}

static __attribute__((noinline)) void fa_hvx_finalize_dealt_row_f32(
        void *        dst,
        const float * src,
        uint32_t      DV,
        float         scale_out) {
    const uint32_t n_pairs = (DV + 63) / 64;
    const HVX_Vector v_scale = hvx_vec_splat_f32(scale_out);
    const HVX_VectorPair * vp_src = (const HVX_VectorPair *) src;
    HVX_VectorPair * vp_dst = (HVX_VectorPair *) dst;

    for (uint32_t p = 0; p < n_pairs; ++p) {
        const HVX_VectorPair vp = vp_src[p];
        const HVX_Vector lo = HVX_OP_MUL_F32(Q6_V_lo_W(vp), v_scale);
        const HVX_Vector hi = HVX_OP_MUL_F32(Q6_V_hi_W(vp), v_scale);
        vp_dst[p] = Q6_W_vshuff_VVR(hi, lo, -4);
    }
}

static __attribute__((noinline)) void fa_hvx_finalize_dealt_row_f16(
        void *        dst,
        const float * src,
        uint32_t      DV,
        float         scale_out) {
    const uint32_t n_pairs = (DV + 63) / 64;
    const HVX_Vector v_scale = hvx_vec_splat_f32(scale_out);
    const HVX_VectorPair * vp_src = (const HVX_VectorPair *) src;
    HVX_Vector * v_dst = (HVX_Vector *) dst;

    for (uint32_t p = 0; p < n_pairs; ++p) {
        const HVX_VectorPair vp = vp_src[p];
        const HVX_Vector lo = HVX_OP_MUL_F32(Q6_V_lo_W(vp), v_scale);
        const HVX_Vector hi = HVX_OP_MUL_F32(Q6_V_hi_W(vp), v_scale);
        v_dst[p] = hvx_vec_f32_to_f16_shuff(lo, hi);
    }
}

static __attribute__((always_inline)) inline HVX_VectorPair fa_hvx_merge_dealt_pair(
        const uint8_t * restrict src_ptr,
        size_t                   worker_stride,
        uint32_t                 n_splits,
        const float * restrict   scales) {
    HVX_Vector v_lo = Q6_V_vzero();
    HVX_Vector v_hi = Q6_V_vzero();
    for (uint32_t s = 0; s < n_splits; ++s) {
        const HVX_VectorPair vp = *(const HVX_VectorPair *) src_ptr;
        const HVX_Vector vs = hvx_vec_splat_f32(scales[s]);
        v_lo = HVX_OP_ADD_F32(v_lo, HVX_OP_MUL_F32(Q6_V_lo_W(vp), vs));
        v_hi = HVX_OP_ADD_F32(v_hi, HVX_OP_MUL_F32(Q6_V_hi_W(vp), vs));
        src_ptr += worker_stride;
    }
    return Q6_W_vcombine_VV(v_hi, v_lo);
}

static __attribute__((noinline)) void fa_hvx_merge_and_store_row_f32(
        void * restrict          dst,
        const uint8_t * restrict spad_a_base,
        size_t                   worker_stride,
        uint32_t                 n_splits,
        const float * restrict   scales,
        uint32_t                 DV) {
    const uint32_t n_pairs = (DV + 63) / 64;
    HVX_VectorPair * restrict vp_dst = (HVX_VectorPair *) dst;

    for (uint32_t p = 0; p < n_pairs; ++p) {
        const HVX_VectorPair sum = fa_hvx_merge_dealt_pair(spad_a_base + p * sizeof(HVX_VectorPair), worker_stride, n_splits, scales);
        vp_dst[p] = Q6_W_vshuff_VVR(Q6_V_hi_W(sum), Q6_V_lo_W(sum), -4);
    }
}

static __attribute__((noinline)) void fa_hvx_merge_and_store_row_f16(
        void * restrict          dst,
        const uint8_t * restrict spad_a_base,
        size_t                   worker_stride,
        uint32_t                 n_splits,
        const float * restrict   scales,
        uint32_t                 DV) {
    const uint32_t n_pairs = (DV + 63) / 64;
    HVX_Vector * restrict v_dst = (HVX_Vector *) dst;

    for (uint32_t p = 0; p < n_pairs; ++p) {
        const HVX_VectorPair sum = fa_hvx_merge_dealt_pair(spad_a_base + p * sizeof(HVX_VectorPair), worker_stride, n_splits, scales);
        v_dst[p] = hvx_vec_f32_to_f16_shuff(Q6_V_lo_W(sum), Q6_V_hi_W(sum));
    }
}

static __attribute__((noinline)) void fa_hvx_qk_block(
        const struct htp_fa_context * factx,
        const uint8_t * restrict      q_ptr,
        const uint8_t * restrict      k_base,
        const __fp16 * restrict       m_base,
        uint32_t                      block_size,
        uint16_t                      slope_bits,
        void * restrict               out_scores) {
    const uint32_t DK       = factx->k->ne[0];
    const size_t   k_stride = factx->size_k_row_padded;

    HVX_Vector scores_f16 = Q6_V_vzero();
    if (block_size > 0) {
        HVX_Vector scores0 = hvx_dot_f16_f16_aa_rx32(q_ptr, k_base, k_stride, DK, factx->scale);
        HVX_Vector scores1 = (block_size > 32)
            ? hvx_dot_f16_f16_aa_rx32(q_ptr, k_base + 32 * k_stride, k_stride, DK, factx->scale)
            : Q6_V_vzero();
        scores_f16 = hvx_vec_f32_to_f16(scores0, scores1);
    }

    if (factx->has_softcap) {
        scores_f16 = hvx_vec_tanh_f16(scores_f16);
        scores_f16 = hvx_vec_mul_f16_f16(scores_f16, hvx_vec_splat_f16(factx->logit_softcap));
    }

    const HVX_Vector     v_neg_inf   = Q6_Vh_vsplat_R(0xfbff);
    const HVX_VectorPred q_tail_keep = Q6_Q_vsetq2_R(block_size * sizeof(__fp16));

    if (m_base) {
        const HVX_Vector vinf = Q6_Vh_vsplat_R(0xFC00);
        const HVX_Vector vmin = Q6_Vh_vsplat_R(0xFBFF);
        HVX_Vector m_vals_f16 = *(const HVX_UVector *) m_base;
        HVX_VectorPred is_inf = Q6_Q_vcmp_eq_VhVh(m_vals_f16, vinf);
        m_vals_f16 = Q6_V_vmux_QVV(is_inf, vmin, m_vals_f16);

        HVX_Vector m_scaled = hvx_vec_mul_f16_f16(m_vals_f16, Q6_Vh_vsplat_R(slope_bits));
        scores_f16 = Q6_V_vmux_QVV(q_tail_keep, hvx_vec_add_f16_f16(scores_f16, m_scaled), v_neg_inf);
    } else {
        scores_f16 = Q6_V_vmux_QVV(q_tail_keep, scores_f16, v_neg_inf);
    }

    *(HVX_Vector *) out_scores = scores_f16;
}

static __attribute__((noinline)) void fa_hvx_sfm_pv_block(
        const struct htp_fa_context * factx,
        float * restrict              vkq,
        const uint8_t * restrict      v_base,
        const void * restrict         in_scores,
        uint32_t                      block_size,
        float * restrict              io_m,
        float * restrict              io_s) {
    const HVX_Vector     scores_f16  = *(const HVX_Vector *) in_scores;
    const HVX_Vector     v_log2e     = hvx_vec_splat_f16(EXP_LOG2E_F);
    const HVX_VectorPred q_tail_keep = Q6_Q_vsetq2_R(block_size * sizeof(__fp16));

    const HVX_Vector v_max_f16 = hvx_vec_reduce_max_f16(scores_f16);
    const HVX_Vector v_max     = Q6_V_lo_W(hvx_vec_f16_to_f32(v_max_f16));

    const HVX_Vector m_prev    = hvx_vec_splat_f32(*io_m);
    const HVX_Vector s_prev    = hvx_vec_splat_f32(*io_s);
    const HVX_Vector m_new     = Q6_Vsf_vmax_VsfVsf(v_max, m_prev);
    const HVX_Vector diff_vec  = HVX_OP_SUB_F32(m_prev, m_new);
    const HVX_Vector diff_f16  = hvx_vec_f32_to_f16(diff_vec, diff_vec);
    const HVX_Vector ms_f16    = hvx_vec_exp2_neg_f16(hvx_vec_mul_f16_f16(diff_f16, v_log2e));
    const HVX_Vector ms_vec    = Q6_V_lo_W(hvx_vec_f16_to_f32(ms_f16));

    const HVX_Vector v_m_f16   = hvx_vec_f32_to_f16(m_new, m_new);
    const HVX_Vector v_sm      = Q6_Vqf16_vsub_VhfVhf(scores_f16, v_m_f16);
    const HVX_Vector v_sm_b2   = hvx_vec_mul_f16_f16(Q6_Vhf_equals_Vqf16(v_sm), v_log2e);
    HVX_Vector       P         = hvx_vec_exp2_neg_f16(v_sm_b2);
    P = Q6_V_vmux_QVV(q_tail_keep, P, Q6_V_vzero());

    const HVX_VectorPair P_pair    = hvx_vec_f16_to_f32(P);
    const HVX_Vector     p_sum_vec = hvx_vec_reduce_sum_f32(HVX_OP_ADD_F32(Q6_V_lo_W(P_pair), Q6_V_hi_W(P_pair)));
    const HVX_Vector     s_new     = HVX_OP_ADD_F32(HVX_OP_MUL_F32(s_prev, ms_vec), p_sum_vec);

    *io_m = hvx_vec_get_f32(m_new);
    *io_s = hvx_vec_get_f32(s_new);

    fa_hvx_pv_accum(vkq, v_base, factx->size_v_row_padded, P, ms_vec, block_size, factx->v->ne[0]);
}

static void flash_attn_ext_f16_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_fa_context * factx = (struct htp_fa_context *) data;
    const struct htp_ops_context * octx = factx->octx;
    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * mask  = octx->src[3];
    const struct htp_tensor * sinks = octx->src[4];
    const struct htp_tensor * dst   = octx->dst;

    const uint32_t neq1 = q->ne[1];
    const uint32_t neq2 = q->ne[2];

    const uint32_t nek0 = k->ne[0];
    const uint32_t nek1 = k->ne[1];

    const uint32_t nev0 = v->ne[0];

    const uint32_t nbq1 = q->nb[1];
    const uint32_t nbq2 = q->nb[2];
    const uint32_t nbq3 = q->nb[3];

    const uint32_t nbk1 = k->nb[1];
    const uint32_t nbk2 = k->nb[2];
    const uint32_t nbk3 = k->nb[3];

    const uint32_t nbv1 = v->nb[1];
    const uint32_t nbv2 = v->nb[2];
    const uint32_t nbv3 = v->nb[3];

    // total rows in q
    const uint32_t dr  = factx->qrows_per_hvx;
    const uint32_t ir0 = factx->qrow_start + dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, factx->qrow_start + factx->qrows);

    if (ir0 >= ir1) return;

    struct htp_thread_trace * tr = th->trace;

    dma_queue * dma_q = th->dma_q;

    const uint32_t DK = nek0;
    const uint32_t DV = nev0;

    const size_t size_q_row = htp_tensor_get_row_size(q->type, DK);
    const size_t size_k_row = htp_tensor_get_row_size(k->type, DK);
    const size_t size_v_row = htp_tensor_get_row_size(v->type, DV);

    // Scratchpad buffers for Q, K, V, Mask, and VKQ32 accumulator
    uint8_t * spad_q = factx->spad_q + factx->size_q_block * ith;
    uint8_t * spad_k = factx->spad_k + factx->size_k_block * 2 * ith;
    uint8_t * spad_v = factx->spad_v + factx->size_v_block * 2 * ith;
    uint8_t * spad_m = factx->spad_m + (mask ? factx->size_m_block * HVX_FA_DMA_CACHE_SIZE : 0) * ith;
    uint8_t * spad_a = factx->spad_a + factx->size_vkq_acc * ith;

    dma_cache_dm * m_cache = &factx->m_cache[ith];
    dma_cache_dm_init(m_cache, spad_m, factx->size_m_block, HVX_FA_DMA_CACHE_SIZE);

    uint16_t * thread_active_blocks = factx->ddr->thread_active_blocks[ith];
    const uint16_t * active_blocks = factx->shared_active_blocks ? factx->ddr->active_blocks : thread_active_blocks;
    uint32_t n_active_blocks       = factx->n_active_blocks;
    dma_addr_t prev_mp_base        = 0;

    if (!factx->shared_active_blocks && !mask) {
        #pragma clang loop vectorize(disable)
        for (uint32_t b = 0; b < factx->n_blocks; ++b) {
            thread_active_blocks[b] = (uint16_t) b;
        }
    }

    const size_t   size_vkq_acc_single = hex_round_up(DV * sizeof(float), 256);
    const uint32_t dv_zero_elems       = ((DV + 63) / 64) * 64;

    uint32_t ir = ir0;
    while (ir < ir1) {
        const uint32_t iq3 = fastdiv(ir, &factx->src0_div21);
        const uint32_t iq2 = fastdiv(ir - iq3*neq2*neq1, &factx->src0_div1);
        const uint32_t iq1 = (ir - iq3*neq2*neq1 - iq2 * neq1);

        const uint32_t ik3 = fastdiv(iq3, &factx->broadcast_rk3);
        const uint32_t ik2 = fastdiv(iq2, &factx->broadcast_rk2);

        const uint32_t iv3 = fastdiv(iq3, &factx->broadcast_rv3);
        const uint32_t iv2 = fastdiv(iq2, &factx->broadcast_rv2);

        uint32_t G_local = 1;
        if (neq1 == 1 && (mask == NULL || mask->ne[2] == 1)) {
            while (ir + G_local < ir1 && G_local < FA_HVX_G_MAX) {
                const uint32_t next_ir = ir + G_local;
                const uint32_t next_iq3 = fastdiv(next_ir, &factx->src0_div21);
                const uint32_t next_iq2 = fastdiv(next_ir - next_iq3*neq2*neq1, &factx->src0_div1);
                const uint32_t next_iq1 = (next_ir - next_iq3*neq2*neq1 - next_iq2 * neq1);

                const uint32_t next_ik3 = fastdiv(next_iq3, &factx->broadcast_rk3);
                const uint32_t next_ik2 = fastdiv(next_iq2, &factx->broadcast_rk2);

                const uint32_t next_iv3 = fastdiv(next_iq3, &factx->broadcast_rv3);
                const uint32_t next_iv2 = fastdiv(next_iq2, &factx->broadcast_rv2);

                if (next_ik2 != ik2 || next_ik3 != ik3 || next_iv2 != iv2 || next_iv3 != iv3 || next_iq1 != iq1 || next_iq3 != iq3) {
                    break;
                }
                G_local++;
            }
        }

        dma_addr_t mp_base = 0;
        if (mask) {
            const uint32_t im2 = fastmodulo(iq2, mask->ne[2], &factx->src3_div2);
            const uint32_t im3 = fastmodulo(iq3, mask->ne[3], &factx->src3_div3);
            mp_base = mask->data + iq1*mask->nb[1] + im2*mask->nb[2] + im3*mask->nb[3];
            if (!factx->shared_active_blocks && mp_base != prev_mp_base) {
                n_active_blocks = fa_scan_active_blocks(
                    dma_q, mask, iq1, 1, im2, im3, nek1, FLASH_ATTN_BLOCK_SIZE,
                    factx->n_blocks, spad_k, factx->size_k_block * 2, thread_active_blocks);
                prev_mp_base = mp_base;
            }
        }

        uint8_t   scores_buf[128] __attribute__((aligned(128)));
        uint32_t  heads[FA_HVX_G_MAX];
        uint16_t  slopes[FA_HVX_G_MAX];
        float     S_vals[FA_HVX_G_MAX];
        float     M_vals[FA_HVX_G_MAX];
        uint8_t * q_ptrs[FA_HVX_G_MAX];
        float *   vkq_ptrs[FA_HVX_G_MAX];

        for (uint32_t g = 0; g < G_local; ++g) {
            const uint32_t r = ir + g;
            const uint32_t r_iq3 = fastdiv(r, &factx->src0_div21);
            const uint32_t r_iq2 = fastdiv(r - r_iq3*neq2*neq1, &factx->src0_div1);
            const uint32_t r_iq1 = (r - r_iq3*neq2*neq1 - r_iq2 * neq1);

            heads[g]  = r_iq2;
            memcpy(&slopes[g], &factx->ddr->slopes[r_iq2], sizeof(uint16_t));
            S_vals[g] = 0.0f;
            M_vals[g] = HTP_FA_M_INITIAL_VAL;

            uint8_t * q_dst = spad_q + g * factx->size_q_row_padded;
            q_ptrs[g] = q_dst;

            float * vkq_dst = (float *)(spad_a + g * size_vkq_acc_single);
            vkq_ptrs[g] = vkq_dst;
            fa_hvx_zero_acc(vkq_dst, dv_zero_elems);

            // Fetch Q row g
            const dma_addr_t q_row_ptr = q->data + r_iq1*nbq1 + r_iq2*nbq2 + r_iq3*nbq3;
            dma_queue_push(dma_q, dma_make_data(q_dst, q_row_ptr), factx->size_q_row_padded, nbq1, size_q_row, 1);
        }

        // Prefetch first two active blocks
        for (uint32_t step = 0; step < MIN(n_active_blocks, 2); ++step) {
            const uint32_t ib = active_blocks[step];
            const uint32_t ic_start = ib * FLASH_ATTN_BLOCK_SIZE;
            const uint32_t current_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - ic_start);

            // K
            const dma_addr_t k_src = k->data + ic_start*nbk1 + ik2*nbk2 + ik3*nbk3;
            uint8_t * k_dst = spad_k + (step % 2) * factx->size_k_block;
            dma_queue_push(dma_q, dma_make_data(k_dst, k_src), factx->size_k_row_padded, nbk1, size_k_row, current_block_size);

            // V
            const dma_addr_t v_src = v->data + ic_start*nbv1 + iv2*nbv2 + iv3*nbv3;
            uint8_t * v_dst = spad_v + (step % 2) * factx->size_v_block;
            dma_queue_push(dma_q, dma_make_data(v_dst, v_src), factx->size_v_row_padded, nbv1, size_v_row, current_block_size);

            // Mask
            if (mask) {
                const dma_addr_t m_src = mp_base + ic_start * sizeof(__fp16);
                dma_cache_dm_push(dma_q, m_cache, ib, m_src, current_block_size * 2, current_block_size * 2, current_block_size * 2, 1);
            }
        }

        // Pop all Q rows
        for (uint32_t g = 0; g < G_local; ++g) {
            uint8_t * q_ptr_vtcm = (void *) dma_queue_pop(dma_q).dst;
            if (factx->is_q_fp32) {
                fa_hvx_q_f32_to_f16(q_ptr_vtcm, DK);
            }
        }

        for (uint32_t step = 0; step < n_active_blocks; ++step) {
            const uint32_t ib = active_blocks[step];
            const uint32_t ic_start = ib * FLASH_ATTN_BLOCK_SIZE;
            const uint32_t current_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - ic_start);

            // Wait for DMA
            uint8_t * k_base = (void *) dma_queue_pop(dma_q).dst; // K
            uint8_t * v_base = (void *) dma_queue_pop(dma_q).dst; // V
            __fp16  * m_base = mask ? (__fp16 *) dma_queue_pop(dma_q).dst : NULL; // M

            if (factx->k->type == HTP_TYPE_Q8_0) {
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, ir);
                fa_hvx_dequant_q8_0_block(k_base, factx->size_k_row_padded, current_block_size, DK);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, ir);
            }
            if (factx->v->type == HTP_TYPE_Q8_0) {
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, ir);
                fa_hvx_dequant_q8_0_block(v_base, factx->size_v_row_padded, current_block_size, DV);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, ir);
            }

            for (uint32_t g = 0; g < G_local; ++g) {
                const uint32_t head_ir = ir + g;

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_QK, head_ir);
                fa_hvx_qk_block(factx, q_ptrs[g], k_base, m_base, current_block_size, slopes[g], scores_buf);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_QK, head_ir);

                // prefetch K for step + 2 after last head finished QK
                if (g + 1 == G_local && step + 2 < n_active_blocks) {
                    const uint32_t next_ib = active_blocks[step + 2];
                    const uint32_t next_ic_start = next_ib * FLASH_ATTN_BLOCK_SIZE;
                    const uint32_t next_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - next_ic_start);

                    const dma_addr_t k_src = k->data + next_ic_start*nbk1 + ik2*nbk2 + ik3*nbk3;
                    dma_queue_push(dma_q, dma_make_data(k_base, k_src), factx->size_k_row_padded, nbk1, size_k_row, next_block_size);
                }

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_SFM, head_ir);
                fa_hvx_sfm_pv_block(factx, vkq_ptrs[g], v_base, scores_buf, current_block_size, &M_vals[g], &S_vals[g]);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, head_ir);

                // prefetch V and mask for step + 2 after last head finished V accumulation
                if (g + 1 == G_local && step + 2 < n_active_blocks) {
                    const uint32_t next_ib = active_blocks[step + 2];
                    const uint32_t next_ic_start = next_ib * FLASH_ATTN_BLOCK_SIZE;
                    const uint32_t next_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - next_ic_start);

                    // V
                    const dma_addr_t v_src = v->data + next_ic_start*nbv1 + iv2*nbv2 + iv3*nbv3;
                    dma_queue_push(dma_q, dma_make_data(v_base, v_src), factx->size_v_row_padded, nbv1, size_v_row, next_block_size);

                    // Mask
                    if (mask) {
                        const dma_addr_t m_src = mp_base + next_ic_start * sizeof(__fp16);
                        dma_cache_dm_push(dma_q, m_cache, next_ib, m_src, next_block_size * 2, next_block_size * 2, next_block_size * 2, 1);
                    }
                }
            } // end for g
        } // end for step

        for (uint32_t g = 0; g < G_local; ++g) {
            const uint32_t head_ir = ir + g;
            const uint32_t h = heads[g];
            float * VKQ32 = vkq_ptrs[g];

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, head_ir);

            float M = M_vals[g];
            float S = S_vals[g];
            float scale_out = 1.0f;

            if (sinks) {
                const float s = factx->spad_sinks[h];

                if (s > M) {
                    const float ms = fa_hvx_exp_scalar(M - s);
                    S = S * ms + 1.0f;
                    scale_out = ms;
                } else {
                    const float vs = fa_hvx_exp_scalar(s - M);
                    S += vs;
                }
            }

            scale_out = (S == 0.0f) ? 0.0f : (scale_out / S);

            const uint32_t r_iq3 = fastdiv(head_ir, &factx->src0_div21);
            const uint32_t r_iq2 = fastdiv(head_ir - r_iq3*neq2*neq1, &factx->src0_div1);
            const uint32_t r_iq1 = (head_ir - r_iq3*neq2*neq1 - r_iq2 * neq1);

            const dma_addr_t dst_addr = dst->data + r_iq2 * dst->nb[1] + r_iq1 * dst->nb[2] + r_iq3 * dst->nb[3];

            if (dst->type == HTP_TYPE_F32) {
                fa_hvx_finalize_dealt_row_f32(VKQ32, VKQ32, DV, scale_out);
                const size_t row_bytes = DV * sizeof(float);
                dma_queue_push(dma_q, dma_make_data(dst_addr, VKQ32), row_bytes, row_bytes, row_bytes, 1);
            } else if (dst->type == HTP_TYPE_F16) {
                fa_hvx_finalize_dealt_row_f16(VKQ32, VKQ32, DV, scale_out);
                const size_t row_bytes = DV * sizeof(__fp16);
                dma_queue_push(dma_q, dma_make_data(dst_addr, VKQ32), row_bytes, row_bytes, row_bytes, 1);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, head_ir);
        }

        dma_queue_flush(dma_q);

        ir += G_local;
    }
}

static void flash_attn_ext_f16_split_kv_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_fa_context * factx = (struct htp_fa_context *) data;
    const struct htp_ops_context * octx = factx->octx;
    const struct htp_tensor * q    = octx->src[0];
    const struct htp_tensor * k    = octx->src[1];
    const struct htp_tensor * v    = octx->src[2];
    const struct htp_tensor * mask = octx->src[3];

    const uint32_t splits_per_kv = factx->splits_per_kv;
    const uint32_t n_workers     = factx->n_kv_local * splits_per_kv;
    if (ith >= n_workers) {
        return;
    }

    const uint32_t kv_local  = ith / splits_per_kv;
    const uint32_t split_idx = ith % splits_per_kv;
    const uint32_t kv_head   = factx->kv_head_start + kv_local;
    const uint32_t G         = factx->G;
    const uint32_t q_base    = kv_head * G;

    const uint32_t base_blocks = factx->n_active_blocks / splits_per_kv;
    const uint32_t rem_blocks  = factx->n_active_blocks % splits_per_kv;
    const uint32_t step_start  = split_idx * base_blocks + MIN(split_idx, rem_blocks);
    const uint32_t step_count  = base_blocks + (split_idx < rem_blocks ? 1 : 0);
    const uint32_t step_end    = step_start + step_count;

    const uint32_t DK   = k->ne[0];
    const uint32_t DV   = v->ne[0];
    const uint32_t nek1 = k->ne[1];

    const uint32_t nbq1 = q->nb[1];
    const uint32_t nbq2 = q->nb[2];
    const uint32_t nbk1 = k->nb[1];
    const uint32_t nbk2 = k->nb[2];
    const uint32_t nbv1 = v->nb[1];
    const uint32_t nbv2 = v->nb[2];

    const size_t size_q_row = htp_tensor_get_row_size(q->type, DK);
    const size_t size_k_row = htp_tensor_get_row_size(k->type, DK);
    const size_t size_v_row = htp_tensor_get_row_size(v->type, DV);

    struct htp_thread_trace * tr = th->trace;
    dma_queue * dma_q = th->dma_q;

    uint8_t * spad_q = factx->spad_q + factx->size_q_block * ith;
    uint8_t * spad_k = factx->spad_k + factx->size_k_block * 2 * ith;
    uint8_t * spad_v = factx->spad_v + factx->size_v_block * 2 * ith;
    uint8_t * spad_m = factx->spad_m + (mask ? factx->size_m_block * HVX_FA_DMA_CACHE_SIZE : 0) * ith;
    uint8_t * spad_a = factx->spad_a + factx->size_vkq_acc * ith;

    dma_cache_dm * m_cache = &factx->m_cache[ith];
    dma_cache_dm_init(m_cache, spad_m, factx->size_m_block, HVX_FA_DMA_CACHE_SIZE);

    const uint16_t * active_blocks = factx->shared_active_blocks ? factx->ddr->active_blocks : NULL;
    const dma_addr_t mp_base       = mask ? mask->data : 0;

    const size_t   size_vkq_acc_single = hex_round_up(DV * sizeof(float), 256);
    const uint32_t dv_zero_elems       = ((DV + 63) / 64) * 64;

    uint8_t   scores_buf[128] __attribute__((aligned(128)));
    uint16_t  slopes[FA_HVX_G_MAX];
    float     S_vals[FA_HVX_G_MAX];
    float     M_vals[FA_HVX_G_MAX];
    uint8_t * q_ptrs[FA_HVX_G_MAX];
    float *   vkq_ptrs[FA_HVX_G_MAX];

    for (uint32_t g = 0; g < G; ++g) {
        const uint32_t qh = q_base + g;
        memcpy(&slopes[g], &factx->ddr->slopes[qh], sizeof(uint16_t));
        S_vals[g] = 0.0f;
        M_vals[g] = HTP_FA_M_INITIAL_VAL;

        uint8_t * q_dst = spad_q + g * factx->size_q_row_padded;
        q_ptrs[g] = q_dst;

        float * vkq_dst = (float *) (spad_a + g * size_vkq_acc_single);
        vkq_ptrs[g] = vkq_dst;
        fa_hvx_zero_acc(vkq_dst, dv_zero_elems);

        const dma_addr_t q_row_ptr = q->data + qh * nbq2;
        dma_queue_push(dma_q, dma_make_data(q_dst, q_row_ptr), factx->size_q_row_padded, nbq1, size_q_row, 1);
    }

    const uint32_t n_prefetch = MIN(step_count, 2);
    for (uint32_t p = 0; p < n_prefetch; ++p) {
        const uint32_t step               = step_start + p;
        const uint32_t ib                 = active_blocks ? active_blocks[step] : step;
        const uint32_t ic_start           = ib * FLASH_ATTN_BLOCK_SIZE;
        const uint32_t current_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - ic_start);

        const dma_addr_t k_src = k->data + ic_start * nbk1 + kv_head * nbk2;
        uint8_t *        k_dst = spad_k + (p & 1) * factx->size_k_block;
        dma_queue_push(dma_q, dma_make_data(k_dst, k_src), factx->size_k_row_padded, nbk1, size_k_row, current_block_size);

        const dma_addr_t v_src = v->data + ic_start * nbv1 + kv_head * nbv2;
        uint8_t *        v_dst = spad_v + (p & 1) * factx->size_v_block;
        dma_queue_push(dma_q, dma_make_data(v_dst, v_src), factx->size_v_row_padded, nbv1, size_v_row, current_block_size);

        if (mask) {
            const dma_addr_t m_src = mp_base + ic_start * sizeof(__fp16);
            dma_cache_dm_push(dma_q, m_cache, ib, m_src, current_block_size * 2, current_block_size * 2, current_block_size * 2, 1);
        }
    }

    for (uint32_t g = 0; g < G; ++g) {
        uint8_t * q_ptr_vtcm = (void *) dma_queue_pop(dma_q).dst;
        if (factx->is_q_fp32) {
            fa_hvx_q_f32_to_f16(q_ptr_vtcm, DK);
        }
    }

    for (uint32_t step = step_start; step < step_end; ++step) {
        const uint32_t ib                 = active_blocks ? active_blocks[step] : step;
        const uint32_t ic_start           = ib * FLASH_ATTN_BLOCK_SIZE;
        const uint32_t current_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - ic_start);

        uint8_t * k_base = (void *) dma_queue_pop(dma_q).dst;
        uint8_t * v_base = (void *) dma_queue_pop(dma_q).dst;
        __fp16 *  m_base = mask ? (__fp16 *) dma_queue_pop(dma_q).dst : NULL;

        if (factx->k->type == HTP_TYPE_Q8_0) {
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, q_base);
            fa_hvx_dequant_q8_0_block(k_base, factx->size_k_row_padded, current_block_size, DK);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, q_base);
        }
        if (factx->v->type == HTP_TYPE_Q8_0) {
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, q_base);
            fa_hvx_dequant_q8_0_block(v_base, factx->size_v_row_padded, current_block_size, DV);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, q_base);
        }

        for (uint32_t g = 0; g < G; ++g) {
            const uint32_t head_ir = q_base + g;

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_QK, head_ir);
            fa_hvx_qk_block(factx, q_ptrs[g], k_base, m_base, current_block_size, slopes[g], scores_buf);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_QK, head_ir);

            if (g + 1 == G && step + 2 < step_end) {
                const uint32_t next_ib         = active_blocks ? active_blocks[step + 2] : (step + 2);
                const uint32_t next_ic_start   = next_ib * FLASH_ATTN_BLOCK_SIZE;
                const uint32_t next_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - next_ic_start);

                const dma_addr_t k_src = k->data + next_ic_start * nbk1 + kv_head * nbk2;
                dma_queue_push(dma_q, dma_make_data(k_base, k_src), factx->size_k_row_padded, nbk1, size_k_row, next_block_size);
            }

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_SFM, head_ir);
            fa_hvx_sfm_pv_block(factx, vkq_ptrs[g], v_base, scores_buf, current_block_size, &M_vals[g], &S_vals[g]);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, head_ir);

            if (g + 1 == G && step + 2 < step_end) {
                const uint32_t next_ib         = active_blocks ? active_blocks[step + 2] : (step + 2);
                const uint32_t next_ic_start   = next_ib * FLASH_ATTN_BLOCK_SIZE;
                const uint32_t next_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - next_ic_start);

                const dma_addr_t v_src = v->data + next_ic_start * nbv1 + kv_head * nbv2;
                dma_queue_push(dma_q, dma_make_data(v_base, v_src), factx->size_v_row_padded, nbv1, size_v_row, next_block_size);

                if (mask) {
                    const dma_addr_t m_src = mp_base + next_ic_start * sizeof(__fp16);
                    dma_cache_dm_push(dma_q, m_cache, next_ib, m_src, next_block_size * 2, next_block_size * 2, next_block_size * 2, 1);
                }
            }
        }
    }

    for (uint32_t g = 0; g < G; ++g) {
        factx->ddr->split_m[ith][g] = M_vals[g];
        factx->ddr->split_s[ith][g] = S_vals[g];
    }

    (void) nth;
}

static void flash_attn_ext_f16_split_merge_thread(const htp_thread_t * th, unsigned int nth, unsigned int ith, void * data) {
    struct htp_fa_context * factx = (struct htp_fa_context *) data;
    const struct htp_ops_context * octx = factx->octx;
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * sinks = octx->src[4];
    const struct htp_tensor * dst   = octx->dst;

    const uint32_t G             = factx->G;
    const uint32_t splits_per_kv = factx->splits_per_kv;
    const uint32_t n_q_local     = factx->n_kv_local * G;
    const uint32_t h_per_t       = (n_q_local + nth - 1) / nth;
    const uint32_t h_start       = ith * h_per_t;
    const uint32_t h_end         = MIN(h_start + h_per_t, n_q_local);

    if (h_start >= h_end) {
        return;
    }

    const uint32_t DV         = v->ne[0];
    const size_t   row_bytes  = htp_tensor_get_row_size(dst->type, DV);
    const size_t   out_stride = hex_round_up(row_bytes, 256);
    const size_t   acc_stride = hex_round_up(DV * sizeof(float), 256);

    // spad_v is unused during merge and has 128 * size_v_row_padded bytes per thread.
    uint8_t * out_base = factx->spad_v + ith * factx->size_v_block * 2;

    struct htp_thread_trace * tr = th->trace;
    dma_queue * dma_q = th->dma_q;

    float diffs[32]   __attribute__((aligned(128)));
    float weights[32] __attribute__((aligned(128)));
    float scales[HTP_MAX_NHVX];

    uint32_t kv_local = h_start / G;
    uint32_t g        = h_start % G;

    for (uint32_t hl = h_start; hl < h_end; ++hl) {
        const uint32_t q_head = (factx->kv_head_start + kv_local) * G + g;
        const uint32_t w_base = kv_local * splits_per_kv;

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, q_head);

        bool  has_valid = false;
        float m_global  = HTP_FA_M_INITIAL_VAL;
        for (uint32_t s = 0; s < splits_per_kv; ++s) {
            if (factx->ddr->split_s[w_base + s][g] > 0.0f) {
                const float ms = factx->ddr->split_m[w_base + s][g];
                if (!has_valid || ms > m_global) {
                    m_global  = ms;
                    has_valid = true;
                }
            }
        }

        float sink_val = 0.0f;
        if (sinks) {
            sink_val = factx->spad_sinks[q_head];
            if (!has_valid || sink_val > m_global) {
                m_global  = sink_val;
                has_valid = true;
            }
        }

        memset(diffs, 0, sizeof(diffs));
        for (uint32_t s = 0; s < splits_per_kv; ++s) {
            if (factx->ddr->split_s[w_base + s][g] > 0.0f) {
                diffs[s] = factx->ddr->split_m[w_base + s][g] - m_global;
            }
        }
        if (sinks) {
            diffs[splits_per_kv] = sink_val - m_global;
        }

        fa_hvx_split_exp_weights(weights, diffs);

        float l_global = sinks ? weights[splits_per_kv] : 0.0f;
        for (uint32_t s = 0; s < splits_per_kv; ++s) {
            const float ls = factx->ddr->split_s[w_base + s][g];
            if (ls > 0.0f) {
                l_global += ls * weights[s];
            }
        }

        const float inv_l = (l_global == 0.0f) ? 0.0f : (1.0f / l_global);
        for (uint32_t s = 0; s < splits_per_kv; ++s) {
            scales[s] = (factx->ddr->split_s[w_base + s][g] > 0.0f) ? (weights[s] * inv_l) : 0.0f;
        }

        uint8_t *       out_ptr     = out_base + (hl - h_start) * out_stride;
        const uint8_t * spad_a_base = factx->spad_a + w_base * factx->size_vkq_acc + g * acc_stride;
        if (dst->type == HTP_TYPE_F32) {
            fa_hvx_merge_and_store_row_f32(out_ptr, spad_a_base, factx->size_vkq_acc, splits_per_kv, scales, DV);
        } else {
            fa_hvx_merge_and_store_row_f16(out_ptr, spad_a_base, factx->size_vkq_acc, splits_per_kv, scales, DV);
        }

        const dma_addr_t dst_addr = dst->data + q_head * dst->nb[1];
        dma_queue_push(dma_q, dma_make_data(dst_addr, out_ptr), row_bytes, row_bytes, row_bytes, 1);

        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, q_head);

        if (++g == G) {
            g = 0;
            ++kv_local;
        }
    }

    dma_queue_flush(dma_q);
}

// ============================================================================
// HMX Phase args and thread logic
// ============================================================================

typedef struct {
    struct hmx_fa_context * factx;
    uint32_t                kv_rows;
    size_t                  src_stride;
    void *                  curr_k;
    uint32_t                kv_start;
    uint32_t                rows_per_t;
    size_t                  buf_idx;
} fa_k_int_args_t;

static void fa_k_interleave_thread(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    fa_k_int_args_t *       args  = (fa_k_int_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const uint32_t total_rows = args->kv_rows;
    const uint32_t rows_per_t = args->rows_per_t;
    const uint32_t start      = i * rows_per_t;

    if (start >= total_rows) {
        return;
    }

    const uint32_t end_valid = (uint32_t) hex_smin(start + rows_per_t, total_rows);
    const uint32_t end_tiled = (uint32_t) hex_smin(start + rows_per_t, hex_align_up(total_rows, HMX_FP16_TILE_N_ROWS));

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, (uint16_t) (args->kv_start + start));
    if (factx->k->type == HTP_TYPE_Q8_0) {
        for (uint32_t r = start; r < end_valid; ++r) {
            __fp16 * row_k = (__fp16 *)((char *)args->curr_k + r * args->src_stride * sizeof(__fp16));
            hvx_dequantize_row_flat_q8_0_f16(row_k, row_k, factx->DK);
        }
    }
    hmx_interleave_rows_to_tiles(factx->vtcm_k_tiles[args->buf_idx], (const __fp16 *) args->curr_k, total_rows, factx->DK_pad,
                             args->src_stride, start, end_tiled);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, (uint16_t) (args->kv_start + start));
}

static void fa_w8a8_k_prep_thread(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    fa_k_int_args_t *       args  = (fa_k_int_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const uint32_t total_rows  = args->kv_rows;
    const uint32_t n_col_tiles = hmx_ceil_div(total_rows, HMX_W8A8_TILE_N_COLS);
    const uint32_t tiles_per_t = hmx_ceil_div(n_col_tiles, n);
    const uint32_t ct_start    = i * tiles_per_t;
    const uint32_t ct_end      = (uint32_t) hex_smin(ct_start + tiles_per_t, n_col_tiles);

    if (ct_start >= n_col_tiles) {
        return;
    }

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, (uint16_t) (args->kv_start + ct_start * HMX_W8A8_TILE_N_COLS));

    const uint32_t n_dot   = factx->DK / 32;
    const size_t   buf_idx = args->buf_idx;

    for (uint32_t ct = ct_start; ct < ct_end; ++ct) {
        const uint32_t valid_rows = (uint32_t) hex_smin(HMX_W8A8_TILE_N_COLS, total_rows - ct * HMX_W8A8_TILE_N_COLS);
        const uint8_t * k_raw     = (const uint8_t *) args->curr_k + (size_t) ct * HMX_W8A8_TILE_N_COLS * factx->k_raw_row_padded;
        uint8_t *       dst_wt    = (uint8_t *) factx->vtcm_k_tiles[buf_idx] + (size_t) ct * n_dot * HMX_W8A8_WT_TILE_SIZE;
        float *         dst_sc    = factx->vtcm_k_scales[buf_idx] + (size_t) ct * HMX_W8A8_TILE_N_COLS;
        HVX_Vector *    bias_out  = (HVX_Vector *) (factx->vtcm_k_bias[buf_idx] + (size_t) ct * 2 * HMX_W8A8_BIAS_BANK_SIZE);
        hmx_fa_w8a8_k_prep_tile(k_raw, dst_wt, dst_sc, bias_out, valid_rows, n_dot, factx->k_raw_row_padded);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, (uint16_t) (args->kv_start + ct_start * HMX_W8A8_TILE_N_COLS));
}

static void fa_phase_k_interleave(struct hmx_fa_context * factx, uint32_t kv_rows, size_t src_stride, void * curr_k, uint32_t kv_start, size_t buf_idx) {
    work_queue_t wp = factx->octx->ctx->hvx_queue;
    if (factx->is_w8a8) {
        const uint32_t n_col_tiles = hmx_ceil_div(kv_rows, HMX_W8A8_TILE_N_COLS);
        const uint32_t n           = (uint32_t) hex_smin(factx->n_hvx, n_col_tiles);
        fa_k_int_args_t args       = { factx, kv_rows, src_stride, curr_k, kv_start, 0, buf_idx };
        work_queue_run(wp, fa_w8a8_k_prep_thread, &args, n);
        return;
    }
    const uint32_t n_col_tiles = hmx_ceil_div(kv_rows, HMX_FP16_TILE_N_ROWS);
    const uint32_t n           = (uint32_t) hex_smin(factx->n_hvx, n_col_tiles);
    const uint32_t rows_per_t  = hmx_ceil_div(n_col_tiles, n) * HMX_FP16_TILE_N_ROWS;
    fa_k_int_args_t args = { factx, kv_rows, src_stride, curr_k, kv_start, rows_per_t, buf_idx };
    work_queue_run(wp, fa_k_interleave_thread, &args, n);
}

typedef struct {
    struct hmx_fa_context * factx;
    uint32_t                kv_rows;
    size_t                  src_stride;
    void *                  v_src;
    void *                  v_tiles_dst;
    size_t                  n_col_tiles;
    uint32_t                kv_start;
    uint32_t                rows_per_t;
    size_t                  buf_idx;
} fa_v_int_args_t;

static void fa_v_interleave_thread(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    fa_v_int_args_t *       args  = (fa_v_int_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const uint32_t total_rows = args->kv_rows;
    const uint32_t rows_per_t = args->rows_per_t;
    const uint32_t start      = i * rows_per_t;
    const uint32_t end        = (uint32_t) hex_smin(start + rows_per_t, total_rows);

    if (start >= total_rows) {
        return;
    }

    __fp16 * v_tiles_dst = (__fp16 *) args->v_tiles_dst;

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, (uint16_t) (args->kv_start + start));
    if (factx->v->type == HTP_TYPE_Q8_0) {
        for (uint32_t r = start; r < end; ++r) {
            __fp16 * row_v = (__fp16 *)((char *)args->v_src + r * args->src_stride * sizeof(__fp16));
            hvx_dequantize_row_flat_q8_0_f16(row_v, row_v, factx->DV);
        }
    }
    hmx_interleave_cols_to_tiles(v_tiles_dst, (const __fp16 *) args->v_src, total_rows, factx->DV_pad,
                             args->src_stride, (uint32_t) args->n_col_tiles, start, end);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, (uint16_t) (args->kv_start + start));
}

static void fa_w8a8_v_prep_thread(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    fa_v_int_args_t *       args  = (fa_v_int_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const uint32_t n_v_tiles   = factx->DV / HMX_W8A8_TILE_N_COLS;
    const uint32_t tiles_per_t = hmx_ceil_div(n_v_tiles, n);
    const uint32_t dv_start    = i * tiles_per_t;
    const uint32_t dv_end      = (uint32_t) hex_smin(dv_start + tiles_per_t, n_v_tiles);

    if (dv_start >= n_v_tiles) {
        return;
    }

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, (uint16_t) (args->kv_start + dv_start * HMX_W8A8_TILE_N_COLS));

    const uint32_t kv_rows     = args->kv_rows;
    const uint32_t n_col_tiles = hmx_ceil_div(kv_rows, HMX_W8A8_TILE_N_COLS);
    const size_t   buf_idx     = args->buf_idx;

    for (uint32_t dv_tile = dv_start; dv_tile < dv_end; ++dv_tile) {
        const uint8_t * v_base   = (const uint8_t *) args->v_src + (size_t) dv_tile * 34;
        uint8_t *       dst_wt   = (uint8_t *) args->v_tiles_dst + (size_t) dv_tile * n_col_tiles * HMX_W8A8_WT_TILE_SIZE;
        float *         dst_sc   = factx->vtcm_v_scales[buf_idx] + (size_t) dv_tile * HMX_W8A8_TILE_N_COLS;
        HVX_Vector *    bias_out = (HVX_Vector *) (factx->vtcm_v_bias[buf_idx] + (size_t) dv_tile * 2 * HMX_W8A8_BIAS_BANK_SIZE);
        hmx_fa_w8a8_v_prep_tile(v_base, dst_wt, dst_sc, bias_out, kv_rows, n_col_tiles, factx->v_raw_row_padded);
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, (uint16_t) (args->kv_start + dv_start * HMX_W8A8_TILE_N_COLS));
}

static void fa_phase_v_interleave(struct hmx_fa_context * factx,
                                  uint32_t                kv_rows,
                                  size_t                  src_stride,
                                  void *                  v_src,
                                  void *                  v_tiles_dst,
                                  size_t                  n_col_tiles,
                                  uint32_t                kv_start,
                                  size_t                  buf_idx) {
    work_queue_t wp = factx->octx->ctx->hvx_queue;
    if (factx->is_w8a8) {
        const uint32_t n_v_tiles = factx->DV / HMX_W8A8_TILE_N_COLS;
        const uint32_t n         = (uint32_t) hex_smin(factx->n_hvx, n_v_tiles);
        fa_v_int_args_t args     = { factx, kv_rows, src_stride, v_src, v_tiles_dst, n_col_tiles, kv_start, 0, buf_idx };
        work_queue_run(wp, fa_w8a8_v_prep_thread, &args, n);
        return;
    }
    uint32_t n = 1;
    if (factx->n_hvx > 1 && kv_rows >= factx->n_hvx * 2) {
        n = factx->n_hvx;
    }
    uint32_t rows_per_t = hex_align_up(hmx_ceil_div(kv_rows, n), 2);
    fa_v_int_args_t args = { factx, kv_rows, src_stride, v_src, v_tiles_dst, n_col_tiles, kv_start, rows_per_t, buf_idx };
    work_queue_run(wp, fa_v_interleave_thread, &args, n);
}

typedef struct {
    struct hmx_fa_context * factx;
    uint32_t                q_start;
    size_t                  n_rows_g;
    size_t                  rows_per_t;
    size_t                  n_rows_q;
    bool                    q_transposed;
    atomic_uint             barrier;
} fa_q_load_args_t;

static void fa_q_load_thread(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    fa_q_load_args_t *      args  = (fa_q_load_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const size_t n_rows_g = args->n_rows_g;
    const size_t G        = factx->G;
    const size_t DK       = factx->DK;

    // Partition the padded Q rows (g_br) across threads.
    // Keep start/end even so r and r+1 are always in the same thread's range.
    const size_t rows_per_t = args->rows_per_t;
    const size_t start      = (size_t) i * rows_per_t;
    const size_t end        = hex_smin(start + rows_per_t, factx->g_br);

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_Q_PREP, (uint16_t) (args->q_start * G + start));

    // Parallel initialization of per-block state
    {
        const size_t col_vec_bytes = factx->col_vec_bytes;
        const size_t d_tile_bytes  = factx->d_tile_bytes;

        // Initialize vtcm_l_vec & vtcm_m_vec
        const size_t l_bytes_per_t = hex_align_up(col_vec_bytes / n, 128);
        const size_t l_start       = i * l_bytes_per_t;
        const size_t l_end         = hex_smin(l_start + l_bytes_per_t, col_vec_bytes);

        const size_t m_bytes_per_t = hex_align_up(col_vec_bytes / n, 128);
        const size_t m_start       = i * m_bytes_per_t;
        const size_t m_end         = hex_smin(m_start + m_bytes_per_t, col_vec_bytes);

        if (l_start < col_vec_bytes) {
            hvx_splat_u8_a((char *) factx->vtcm_l_vec + l_start, 0, l_end - l_start);
        }
        if (m_start < col_vec_bytes) {
            hvx_splat_f32_a((char *) factx->vtcm_m_vec + m_start, HTP_FA_M_INITIAL_VAL, (m_end - m_start) / sizeof(float));
        }

        // Zero the whole rescale region: vtcm_d_tiles[0], the optional vtcm_d_tiles[1]
        // and vtcm_d_inv_l are equal-sized and allocated back to back, so one run covers
        // them all.  The scatter only ever writes the diagonal, ignore the rest.
        const size_t d_bytes_per_t = hex_align_up(d_tile_bytes / n, 128);
        const size_t d_start       = i * d_bytes_per_t;
        const size_t d_end         = hex_smin(d_start + d_bytes_per_t, d_tile_bytes);
        if (d_start < d_tile_bytes) {
            hvx_splat_u8_a((char *) factx->vtcm_d_tiles[0] + d_start, 0, d_end - d_start);
        }
    }

    if (start < factx->g_br) {
        __fp16 *     q_tiles    = factx->vtcm_q_tiles;
        const size_t DK_pad     = factx->DK_pad;
        const size_t g_rows_end = hex_smin(end, n_rows_g);
        const uint32_t d_limit  = factx->is_q_fp32 ? DK / 32 : DK / 64;

        uint8_t * q_flat = (uint8_t *) factx->vtcm_q_dma;
        if (DK_pad != DK) {
            if (factx->is_q_fp32) {
                hmx_fa_q_prep_fp32_pad(q_tiles, q_flat, start, end, g_rows_end, DK, DK_pad, G, args->n_rows_q, &factx->div_G, args->q_transposed);
            } else {
                hmx_fa_q_prep_fp16_pad(q_tiles, q_flat, start, end, g_rows_end, DK, DK_pad, G, args->n_rows_q, &factx->div_G, args->q_transposed);
            }
        } else if (factx->is_q_fp32) {
            switch (d_limit) {
            case 2:  hmx_fa_q_prep_fp32_d2(q_tiles, q_flat, start, end, g_rows_end, DK, G, args->n_rows_q, &factx->div_G, args->q_transposed); break;
            case 4:  hmx_fa_q_prep_fp32_d4(q_tiles, q_flat, start, end, g_rows_end, DK, G, args->n_rows_q, &factx->div_G, args->q_transposed); break;
            default: hmx_fa_q_prep_fp32(   q_tiles, q_flat, start, end, g_rows_end, DK, G, args->n_rows_q, &factx->div_G, d_limit, args->q_transposed); break;
            }
        } else {
            switch (d_limit) {
            case 1:  hmx_fa_q_prep_fp16_d1(q_tiles, q_flat, start, end, g_rows_end, DK, G, args->n_rows_q, &factx->div_G, args->q_transposed); break;
            case 2:  hmx_fa_q_prep_fp16_d2(q_tiles, q_flat, start, end, g_rows_end, DK, G, args->n_rows_q, &factx->div_G, args->q_transposed); break;
            default: hmx_fa_q_prep_fp16(   q_tiles, q_flat, start, end, g_rows_end, DK, G, args->n_rows_q, &factx->div_G, d_limit, args->q_transposed); break;
            }
        }
    }

    // Synchronize threads before zeroing out vtcm_o_tiles[0] to prevent race condition
    if (n > 1) {
        atomic_fetch_sub(&args->barrier, 1);
        while (atomic_load(&args->barrier) > 0) {
            // spin wait
        }
    }

    // Zero out vtcm_o_tiles[0] as it was used as temp_q_vtcm
    {
        const size_t o_tile_bytes  = factx->o_tile_bytes;
        const size_t o_bytes_per_t = hex_align_up(o_tile_bytes / n, 128);
        const size_t o_start       = i * o_bytes_per_t;
        const size_t o_end         = hex_smin(o_start + o_bytes_per_t, o_tile_bytes);
        if (o_start < o_tile_bytes) {
            hvx_splat_u8_a((char *) factx->vtcm_o_tiles[0] + o_start, 0, o_end - o_start);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_Q_PREP, (uint16_t) (args->q_start * G + start));
}

static void fa_w8a8_q_prep_thread(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    fa_q_load_args_t *      args  = (fa_q_load_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const size_t n_rows_g   = args->n_rows_g;
    const size_t G          = factx->G;
    const size_t DK         = factx->DK;
    const size_t rows_per_t = args->rows_per_t;
    const size_t start      = (size_t) i * rows_per_t;
    const size_t end        = hex_smin(start + rows_per_t, factx->g_br);

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_Q_PREP, (uint16_t) (args->q_start * G + start));

    // Parallel initialization of per-block state
    {
        const size_t col_vec_bytes = factx->col_vec_bytes;
        const size_t o_f32_bytes   = factx->o_f32_bytes;

        const size_t l_bytes_per_t = hex_align_up(col_vec_bytes / n, 128);
        const size_t l_start       = i * l_bytes_per_t;
        const size_t l_end         = hex_smin(l_start + l_bytes_per_t, col_vec_bytes);

        const size_t m_bytes_per_t = hex_align_up(col_vec_bytes / n, 128);
        const size_t m_start       = i * m_bytes_per_t;
        const size_t m_end         = hex_smin(m_start + m_bytes_per_t, col_vec_bytes);

        if (l_start < col_vec_bytes) {
            hvx_splat_u8_a((char *) factx->vtcm_l_vec + l_start, 0, l_end - l_start);
        }
        if (m_start < col_vec_bytes) {
            hvx_splat_f32_a((char *) factx->vtcm_m_vec + m_start, HTP_FA_M_INITIAL_VAL, (m_end - m_start) / sizeof(float));
        }

        const size_t o_bytes_per_t = hex_align_up(o_f32_bytes / n, 128);
        const size_t o_start       = i * o_bytes_per_t;
        const size_t o_end         = hex_smin(o_start + o_bytes_per_t, o_f32_bytes);
        if (o_start < o_f32_bytes) {
            hvx_splat_u8_a((char *) factx->vtcm_o_f32 + o_start, 0, o_end - o_start);
        }
    }

    if (start < factx->g_br) {
        const size_t    g_rows_end = hex_smin(end, n_rows_g);
        uint8_t *       q_tiles    = (uint8_t *) factx->vtcm_q_tiles;
        const uint8_t * q_flat     = (const uint8_t *) factx->vtcm_q_dma;
        if (factx->is_q_fp32) {
            hmx_fa_w8a8_q_prep_f32(q_tiles, factx->vtcm_q_scales, q_flat, start, end, g_rows_end,
                                   DK, G, args->n_rows_q, &factx->div_G, args->q_transposed, factx->scale_f32);
        } else {
            hmx_fa_w8a8_q_prep_f16(q_tiles, factx->vtcm_q_scales, q_flat, start, end, g_rows_end,
                                   DK, G, args->n_rows_q, &factx->div_G, args->q_transposed, factx->scale_f32);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_Q_PREP, (uint16_t) (args->q_start * G + start));
}

static void fa_phase_q_load(struct hmx_fa_context *   factx,
                            const struct htp_tensor * q,
                            uint32_t                  q_start,
                            size_t                    n_rows_g) {
    work_queue_t wp = factx->octx->ctx->hvx_queue;
    uint32_t n = 1;
    const size_t row_align = factx->is_w8a8 ? 4 : 2;
    if (factx->n_hvx > 1 && n_rows_g >= (size_t) (factx->n_hvx * row_align)) {
        n = factx->n_hvx;
    }
    size_t rows_per_t = hex_align_up(hmx_ceil_div(factx->g_br, n), row_align);
    const uint32_t n_rows_q = hex_smin(factx->Br, factx->neq1 - q_start);
    fa_q_load_args_t args;
    args.factx = factx;
    args.q_start = q_start;
    args.n_rows_g = n_rows_g;
    args.rows_per_t = rows_per_t;
    args.n_rows_q = n_rows_q;
    args.q_transposed = q->nb[1] < q->nb[2];
    atomic_init(&args.barrier, n);
    work_queue_run(wp, factx->is_w8a8 ? fa_w8a8_q_prep_thread : fa_q_load_thread, &args, n);
}

typedef struct {
    struct hmx_fa_context * factx;
    uint8_t *               vtcm_dst;
    const __fp16 *          o_tile_src;
    uint32_t                q_start;
    uint32_t                kv_head;
    size_t                  row_off;
    size_t                  n_rows;
    size_t                  rows_per_t;
} fa_o_store_args_t;

static void fa_o_store_thread_f32(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    fa_o_store_args_t *     args  = (fa_o_store_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const size_t n_rows     = args->n_rows;
    const size_t G          = factx->G;
    const size_t DV         = factx->DV;
    const size_t rows_per_t = args->rows_per_t;
    const size_t start      = (size_t) i * rows_per_t;
    const size_t end        = hex_smin(start + rows_per_t, n_rows);

    if (start >= n_rows) {
        return;
    }

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (args->q_start * G + args->row_off + start));

    uint8_t *      vtcm_dst   = args->vtcm_dst;
    const __fp16 * o_tile_src = args->o_tile_src;
    const size_t   row_off    = args->row_off;
    const size_t   DV_pad     = factx->DV_pad;
    const size_t   row_bytes  = DV * sizeof(float);

    for (size_t rl = start; rl < end; ++rl) {
        const size_t   r             = row_off + rl;
        float *        out           = (float *) (vtcm_dst + rl * row_bytes);
        const size_t   r0            = r / HMX_FP16_TILE_N_ROWS;
        const size_t   r1            = r % HMX_FP16_TILE_N_ROWS;
        const __fp16 * tile_row_base = o_tile_src + r0 * HMX_FP16_TILE_N_ROWS * DV_pad;

        for (uint32_t d = 0; d < DV / 32; ++d) {
            const HVX_Vector * in_tile = (const HVX_Vector *) (tile_row_base + d * HMX_FP16_TILE_N_ELMS);
            HVX_VectorPair     vp      = hvx_vec_f16_to_f32_shuff(in_tile[r1 / 2]);
            if (r1 % 2 == 0) {
                *(HVX_UVector *) (out + d * 32) = Q6_V_lo_W(vp);
            } else {
                *(HVX_UVector *) (out + d * 32) = Q6_V_hi_W(vp);
            }
        }
        const uint32_t d_tail = DV / 32;
        const uint32_t rem    = DV - d_tail * 32;
        if (rem) {
            const HVX_Vector * in_tile = (const HVX_Vector *) (tile_row_base + d_tail * HMX_FP16_TILE_N_ELMS);
            HVX_VectorPair     vp      = hvx_vec_f16_to_f32_shuff(in_tile[r1 / 2]);
            HVX_Vector         vd      = (r1 % 2 == 0) ? Q6_V_lo_W(vp) : Q6_V_hi_W(vp);
            hvx_vec_store_u((void *) (out + d_tail * 32), rem * sizeof(float), vd);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (args->q_start * G + args->row_off + start));
}

static void fa_o_store_thread_f16(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    fa_o_store_args_t *     args  = (fa_o_store_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const size_t n_rows     = args->n_rows;
    const size_t rows_per_t = args->rows_per_t;
    const size_t G          = factx->G;
    const size_t DV         = factx->DV;
    const size_t start      = (size_t) i * rows_per_t;
    const size_t end        = hex_smin(start + rows_per_t, n_rows);

    if (start >= n_rows) {
        return;
    }

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (args->q_start * G + args->row_off + start));

    uint8_t *      vtcm_dst   = args->vtcm_dst;
    const __fp16 * o_tile_src = args->o_tile_src;
    const size_t   row_off    = args->row_off;
    const size_t   DV_pad     = factx->DV_pad;
    const size_t   row_bytes  = DV * sizeof(__fp16);

    for (size_t rl = start; rl < end; ++rl) {
        const size_t   r             = row_off + rl;
        __fp16 *       out           = (__fp16 *) (vtcm_dst + rl * row_bytes);
        const size_t   r0            = r / HMX_FP16_TILE_N_ROWS;
        const size_t   r1            = r % HMX_FP16_TILE_N_ROWS;
        const __fp16 * tile_row_base = o_tile_src + r0 * HMX_FP16_TILE_N_ROWS * DV_pad;

        for (uint32_t d = 0; d < DV / 64; ++d) {
            const __fp16 *     in_dtile = tile_row_base + d * HMX_FP16_TILE_N_ELMS * 2;
            const HVX_Vector * pv_in0   = ((const HVX_Vector *) in_dtile) + r1 / 2;
            const HVX_Vector * pv_in1   = pv_in0 + 16;
            HVX_VectorPair     vp       = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);
            if (r1 % 2 == 0) {
                *(HVX_UVector *) (out + d * 64) = Q6_V_lo_W(vp);
            } else {
                *(HVX_UVector *) (out + d * 64) = Q6_V_hi_W(vp);
            }
        }
        const uint32_t d_tail = DV / 64;
        const uint32_t rem    = DV - d_tail * 64;
        if (rem) {
            const __fp16 *     in_dtile = tile_row_base + d_tail * HMX_FP16_TILE_N_ELMS * 2;
            const HVX_Vector * pv_in0   = ((const HVX_Vector *) in_dtile) + r1 / 2;
            const HVX_Vector * pv_in1   = pv_in0 + 16;
            HVX_VectorPair     vp       = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);
            HVX_Vector         vd       = (r1 % 2 == 0) ? Q6_V_lo_W(vp) : Q6_V_hi_W(vp);
            hvx_vec_store_u((void *) (out + d_tail * 64), rem * sizeof(__fp16), vd);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (args->q_start * G + args->row_off + start));
}

// Push [row_off, row_off + n_rows) from vtcm_src to dst->data via bypass DMA.
// Always leaves exactly 1 pending DMA descriptor on dma_q.
static void fa_push_o_dma(dma_queue *                   dma_q,
                          const struct htp_tensor *     dst,
                          const uint8_t *               vtcm_src,
                          size_t                        vtcm_row_stride,
                          size_t                        row_bytes,
                          uint32_t                      q_start,
                          uint32_t                      kv_head,
                          uint32_t                      ib3,
                          uint32_t                      G,
                          const struct fastdiv_values * div_G,
                          size_t                        row_off,
                          size_t                        n_rows) {
    const dma_addr_t dst_base = dst->data + ib3 * dst->nb[3] + (kv_head * G) * dst->nb[1];

    if (G == 1) {
        const dma_addr_t d = dst_base + (q_start + row_off) * dst->nb[2];
        dma_queue_push(dma_q, dma_make_data(d, vtcm_src), dst->nb[2], vtcm_row_stride, row_bytes, n_rows);
        return;
    }

    const size_t q0 = fastdiv(row_off, div_G);
    const size_t h0 = fastmodulo(row_off, G, div_G);
    if (h0 == 0 && (n_rows % G) == 0 && dst->nb[1] == row_bytes && vtcm_row_stride == row_bytes) {
        const size_t     n_q     = n_rows / G;
        const size_t     g_bytes = (size_t) G * row_bytes;
        const dma_addr_t d       = dst_base + (q_start + q0) * dst->nb[2];
        dma_queue_push(dma_q, dma_make_data(d, vtcm_src), dst->nb[2], g_bytes, g_bytes, n_q);
        return;
    }

    size_t q_idx = q0;
    size_t h_idx = h0;
    size_t done  = 0;
    while (done < n_rows) {
        const size_t     cnt = hex_smin(G - h_idx, n_rows - done);
        const dma_addr_t d   = dst_base + h_idx * dst->nb[1] + (q_start + q_idx) * dst->nb[2];
        const uint8_t *  s   = vtcm_src + done * vtcm_row_stride;
        if (!dma_queue_push(dma_q, dma_make_data(d, s), dst->nb[1], vtcm_row_stride, row_bytes, cnt)) {
            dma_queue_pop(dma_q);
            dma_queue_push(dma_q, dma_make_data(d, s), dst->nb[1], vtcm_row_stride, row_bytes, cnt);
        }
        done  += cnt;
        h_idx  = 0;
        q_idx += 1;
    }
    while (dma_queue_depth(dma_q) > 1) {
        dma_queue_pop(dma_q);
    }
}

static void fa_phase_o_store(struct hmx_fa_context *   factx,
                             dma_queue *               dma_q,
                             const struct htp_tensor * dst,
                             const __fp16 *            o_tile_src,
                             uint8_t *                 vtcm_dst,
                             uint32_t                  q_start,
                             uint32_t                  kv_head,
                             uint32_t                  ib3,
                             size_t                    n_rows_g) {
    work_queue_t      wp        = factx->octx->ctx->hvx_queue;
    const size_t      row_bytes = factx->DV * (factx->is_dst_fp32 ? sizeof(float) : sizeof(__fp16));
    work_queue_func_t store_fn  = factx->is_dst_fp32 ? fa_o_store_thread_f32 : fa_o_store_thread_f16;

    size_t max_rows = factx->o_tile_bytes / row_bytes;
    if (max_rows >= factx->G && max_rows < n_rows_g) {
        max_rows = (max_rows / factx->G) * factx->G;
    }
    if (max_rows == 0) {
        max_rows = 1;
    }

    for (size_t row_off = 0; row_off < n_rows_g; row_off += max_rows) {
        if (row_off > 0) {
            dma_queue_pop(dma_q);
        }
        const size_t n_rows = hex_smin(max_rows, n_rows_g - row_off);
        uint32_t     n      = 1;
        if (factx->n_hvx > 1 && n_rows >= (size_t) (factx->n_hvx * 2)) {
            n = factx->n_hvx;
        }
        const size_t      rows_per_t = hmx_ceil_div(n_rows, n);
        fa_o_store_args_t args       = { factx, vtcm_dst, o_tile_src, q_start, kv_head, row_off, n_rows, rows_per_t };
        work_queue_run(wp, store_fn, &args, n);

        fa_push_o_dma(dma_q, dst, vtcm_dst, row_bytes, row_bytes,
                      q_start, kv_head, ib3, factx->G, &factx->div_G, row_off, n_rows);
    }
}

typedef struct {
    struct hmx_fa_context *   factx;
    size_t                    buf_idx;
    size_t                    kv_rows;
    size_t                    n_rows_g;
    size_t                    n_col_tiles;
    size_t                    n_tiles_per_bc;
    size_t                    n_row_tiles;
    size_t                    n_row_tiles_g_br;
    uint32_t                  Bc;
    uint32_t                  G;
    uint32_t                  kv_head;
    uint32_t                  kv_start;
    uint32_t                  q_start;
    uint32_t                  ib3;
    bool                      has_alibi;  // true when max_bias != 0 (need slope * mask + add)
    __fp16 *                  slopes;
    const struct htp_tensor * mask;
    const __fp16 *            mask_vtcm;             // VTCM mask buffer base (NULL = DDR fallback)
    size_t                    mask_vtcm_row_stride;  // elements (__fp16) per row in VTCM mask buffer
    struct fastdiv_values     thread_div;
} fa_softmax_args_t;

static inline void fa_softmax_impl(
    const htp_thread_t * th, unsigned int n, unsigned int i, void * data,
    const bool has_mask,
    const bool mask_broadcast,
    const bool is_g1,
    const bool has_alibi,
    const bool has_softcap
) {
    fa_softmax_args_t *     args  = (fa_softmax_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const size_t n_rows_g       = args->n_rows_g;
    const size_t kv_rows        = args->kv_rows;
    const size_t Bc             = args->Bc;
    const size_t G              = args->G;
    const size_t n_row_vec_cnt  = hmx_ceil_div(n_rows_g, 64);

    size_t vec_start = 0;
    size_t vec_end   = n_row_vec_cnt;
    if (n > 1) {
        const size_t vecs_per_t = fastdiv(n_row_vec_cnt + n - 1, &args->thread_div);
        vec_start = i * vecs_per_t;
        vec_end   = hex_smin(vec_start + vecs_per_t, n_row_vec_cnt);
    }

    if (vec_start >= n_row_vec_cnt) {
        return;
    }

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_SFM, (uint16_t) (args->q_start * G + vec_start * 64));

    // Per-thread row scratch: thread i uses bufs at offset i * 2 * stride
    const size_t row_buf_stride = factx->row_buf_stride;
    HVX_Vector * my_row_buf0    = factx->vtcm_row_bufs + i * 2 * row_buf_stride;
    HVX_Vector * my_row_buf1    = my_row_buf0 + row_buf_stride;

    const HVX_Vector v_neg_inf = Q6_Vh_vsplat_R(0xfbff);

    for (size_t r_vec_idx = vec_start; r_vec_idx < vec_end; ++r_vec_idx) {
        HVX_Vector rowmax_acc_v = v_neg_inf;
        HVX_Vector rowsum_acc_v = Q6_V_vzero();
        HVX_Vector m_prev_v0    = factx->vtcm_m_vec[r_vec_idx * 2 + 0];
        HVX_Vector m_prev_v1    = factx->vtcm_m_vec[r_vec_idx * 2 + 1];

        HVX_Vector v_slopes = Q6_V_vzero();
        if (has_alibi) {
            v_slopes = hvx_vmem(args->slopes + r_vec_idx * 64);
        }

        for (uint32_t r_vec_off = 0; r_vec_off < 64; r_vec_off += 2) {
            uint32_t r = r_vec_idx * 64 + r_vec_off;
            if (r >= hex_align_up(n_rows_g, 2)) {
                break;
            }

            uint32_t r0 = r / HMX_FP16_TILE_N_ROWS;
            uint32_t r1 = r % HMX_FP16_TILE_N_ROWS;

            const __fp16 * s_ld_base = factx->vtcm_s_tiles[args->buf_idx] + r0 * HMX_FP16_TILE_N_ROWS * Bc;
            __fp16 *       p_st_base = factx->vtcm_p_tiles[args->buf_idx] + r0 * HMX_FP16_TILE_N_ROWS * Bc;

            // Decode 2 rows from S tiles into per-thread row buffers
            if (has_softcap) {
                const HVX_Vector v_cap = hvx_vec_splat_f16(factx->logit_softcap);
                for (size_t c = 0; c < kv_rows; c += 64) {
                    size_t             ci       = c / 64;
                    const __fp16 *     in_dtile = s_ld_base + ci * HMX_FP16_TILE_N_ELMS * 2;
                    const HVX_Vector * pv_s_in0 = ((const HVX_Vector *) in_dtile) + r1 / 2;
                    const HVX_Vector * pv_s_in1 = pv_s_in0 + 16;

                    HVX_VectorPair vp_s_drow = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
                    HVX_Vector     v_s_row0  = Q6_V_lo_W(vp_s_drow);
                    HVX_Vector     v_s_row1  = Q6_V_hi_W(vp_s_drow);

                    HVX_Vector t0   = hvx_vec_tanh_f16(v_s_row0);
                    my_row_buf0[ci] = hvx_vec_mul_f16_f16(t0, v_cap);

                    HVX_Vector t1   = hvx_vec_tanh_f16(v_s_row1);
                    my_row_buf1[ci] = hvx_vec_mul_f16_f16(t1, v_cap);
                }
            } else {
                size_t c = 0;
                for (; c + 64 < kv_rows; c += 128) {
                    size_t             ci0       = c / 64;
                    size_t             ci1       = ci0 + 1;
                    const __fp16 *     in_dtile0 = s_ld_base + ci0 * HMX_FP16_TILE_N_ELMS * 2;
                    const __fp16 *     in_dtile1 = s_ld_base + ci1 * HMX_FP16_TILE_N_ELMS * 2;
                    const HVX_Vector * pv_s_in0_0 = ((const HVX_Vector *) in_dtile0) + r1 / 2;
                    const HVX_Vector * pv_s_in1_0 = pv_s_in0_0 + 16;
                    const HVX_Vector * pv_s_in0_1 = ((const HVX_Vector *) in_dtile1) + r1 / 2;
                    const HVX_Vector * pv_s_in1_1 = pv_s_in0_1 + 16;

                    HVX_VectorPair vp_s_drow0 = Q6_W_vdeal_VVR(*pv_s_in1_0, *pv_s_in0_0, -2);
                    my_row_buf0[ci0]          = Q6_V_lo_W(vp_s_drow0);
                    my_row_buf1[ci0]          = Q6_V_hi_W(vp_s_drow0);

                    HVX_VectorPair vp_s_drow1 = Q6_W_vdeal_VVR(*pv_s_in1_1, *pv_s_in0_1, -2);
                    my_row_buf0[ci1]          = Q6_V_lo_W(vp_s_drow1);
                    my_row_buf1[ci1]          = Q6_V_hi_W(vp_s_drow1);
                }
                for (; c < kv_rows; c += 64) {
                    size_t             ci       = c / 64;
                    const __fp16 *     in_dtile = s_ld_base + ci * HMX_FP16_TILE_N_ELMS * 2;
                    const HVX_Vector * pv_s_in0 = ((const HVX_Vector *) in_dtile) + r1 / 2;
                    const HVX_Vector * pv_s_in1 = pv_s_in0 + 16;

                    HVX_VectorPair vp_s_drow = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
                    my_row_buf0[ci]          = Q6_V_lo_W(vp_s_drow);
                    my_row_buf1[ci]          = Q6_V_hi_W(vp_s_drow);
                }
            }

            // Apply mask & compute rowmax(S)
            HVX_Vector v_slope0 = Q6_V_vzero();
            HVX_Vector v_slope1 = Q6_V_vzero();
            if (has_alibi) {
                v_slope0 = hvx_vec_repl_f16(Q6_V_vror_VR(v_slopes, r_vec_off * 2));
                v_slope1 = (r + 1 < n_rows_g) ? hvx_vec_repl_f16(Q6_V_vror_VR(v_slopes, (r_vec_off + 1) * 2)) : Q6_V_vzero();
            }

            const HVX_Vector v_threshold = Q6_Vh_vsplat_R(0xcc00);  // fp16 -16.0

            HVX_Vector v_s_rowmax0 = v_neg_inf;
            HVX_Vector v_s_rowmax1 = v_neg_inf;
            if (has_mask) {
                for (size_t c = 0; c < kv_rows; c += 64) {
                    size_t         ci          = c / 64;
                    const size_t   ne          = hex_smin(kv_rows - c, 64);
                    HVX_VectorPred q_tail_keep = Q6_Q_vsetq2_R(ne * sizeof(__fp16));

                    HVX_Vector v_mask0, v_mask1;

                    if (mask_broadcast) {
                        if (is_g1) {
                            const size_t qi0 = r + 0;
                            v_mask0 = *(const HVX_Vector *) (args->mask_vtcm + qi0 * args->mask_vtcm_row_stride + c);
                            v_mask1 = v_neg_inf;
                            if (r + 1 < n_rows_g) {
                                const size_t qi1 = r + 1;
                                v_mask1 = *(const HVX_Vector *) (args->mask_vtcm + qi1 * args->mask_vtcm_row_stride + c);
                            }
                        } else {
                            const size_t qi0 = fastdiv(r + 0, &factx->div_G);
                            v_mask0 = *(const HVX_Vector *) (args->mask_vtcm + qi0 * args->mask_vtcm_row_stride + c);
                            v_mask1 = v_neg_inf;
                            if (r + 1 < n_rows_g) {
                                const size_t qi1 = fastdiv(r + 1, &factx->div_G);
                                if (qi1 == qi0) {
                                    v_mask1 = v_mask0;
                                } else {
                                    v_mask1 = *(const HVX_Vector *) (args->mask_vtcm + qi1 * args->mask_vtcm_row_stride + c);
                                }
                            }
                        }
                    } else {
                        // Head-dependent mask: pre-interleaved per row r.
                        const size_t r0 = r + 0;
                        v_mask0 = *(const HVX_Vector *) (args->mask_vtcm + r0 * args->mask_vtcm_row_stride + c);
                        v_mask1 = v_neg_inf;
                        if (r + 1 < n_rows_g) {
                            const size_t r1 = r + 1;
                            v_mask1 = *(const HVX_Vector *) (args->mask_vtcm + r1 * args->mask_vtcm_row_stride + c);
                        }
                    }

                    // Threshold: mask values below -16.0 are treated as -inf (causal mask).
                    HVX_VectorPred q_keep0 = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask0, v_threshold), q_tail_keep);
                    HVX_VectorPred q_keep1 = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask1, v_threshold), q_tail_keep);

                    // Scale mask values by log2(e) for base-2 calculations
                    const HVX_Vector v_log2e = hvx_vec_splat_f16(EXP_LOG2E_F);
                    HVX_Vector v_mask0_scaled = hvx_vec_mul_f16_f16(v_mask0, v_log2e);
                    HVX_Vector v_mask1_scaled = hvx_vec_mul_f16_f16(v_mask1, v_log2e);

                    if (has_alibi) {
                        HVX_Vector v_sm0 = hvx_vec_mul_f16_f16(v_mask0_scaled, v_slope0);
                        HVX_Vector v_sm1 = hvx_vec_mul_f16_f16(v_mask1_scaled, v_slope1);
                        my_row_buf0[ci]  = Q6_V_vmux_QVV(q_keep0, hvx_vec_add_f16_f16(my_row_buf0[ci], v_sm0), v_neg_inf);
                        my_row_buf1[ci]  = Q6_V_vmux_QVV(q_keep1, hvx_vec_add_f16_f16(my_row_buf1[ci], v_sm1), v_neg_inf);
                    } else {
                        my_row_buf0[ci] = Q6_V_vmux_QVV(q_keep0, hvx_vec_add_f16_f16(my_row_buf0[ci], v_mask0_scaled), v_neg_inf);
                        my_row_buf1[ci] = Q6_V_vmux_QVV(q_keep1, hvx_vec_add_f16_f16(my_row_buf1[ci], v_mask1_scaled), v_neg_inf);
                    }

                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_row_buf0[ci]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_row_buf1[ci]);
                }
            } else {
                size_t c = 0;
                for (; c + 128 <= kv_rows; c += 128) {
                    size_t ci0 = c / 64;
                    size_t ci1 = ci0 + 1;
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_row_buf0[ci0]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_row_buf1[ci0]);
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_row_buf0[ci1]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_row_buf1[ci1]);
                }
                for (; c < kv_rows; c += 64) {
                    size_t         ci          = c / 64;
                    const size_t   ne          = hex_smin(kv_rows - c, 64);
                    HVX_VectorPred q_tail_keep = Q6_Q_vsetq2_R(ne * sizeof(__fp16));
                    if (ne < 64) {
                        my_row_buf0[ci] = Q6_V_vmux_QVV(q_tail_keep, my_row_buf0[ci], v_neg_inf);
                        my_row_buf1[ci] = Q6_V_vmux_QVV(q_tail_keep, my_row_buf1[ci], v_neg_inf);
                    }
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_row_buf0[ci]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_row_buf1[ci]);
                }
            }

            v_s_rowmax0 = hvx_vec_reduce_max_f16(v_s_rowmax0);
            v_s_rowmax1 = hvx_vec_reduce_max_f16(v_s_rowmax1);

            // Splat m_prev[r], m_prev[r+1] from the float per-row accumulators and convert to fp16 vectors
            HVX_Vector v_m_prev0, v_m_prev1;
            if (r_vec_off < 32) {
                HVX_Vector v0 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v0, r_vec_off * 4));
                v_m_prev0 = hvx_vec_f32_to_f16(v0, v0);
                if (r + 1 < n_rows_g) {
                    HVX_Vector v1 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v0, (r_vec_off + 1) * 4));
                    v_m_prev1 = hvx_vec_f32_to_f16(v1, v1);
                } else {
                    v_m_prev1 = Q6_V_vzero();
                }
            } else {
                HVX_Vector v0 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v1, (r_vec_off - 32) * 4));
                v_m_prev0 = hvx_vec_f32_to_f16(v0, v0);
                if (r + 1 < n_rows_g) {
                    HVX_Vector v1 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v1, (r_vec_off + 1 - 32) * 4));
                    v_m_prev1 = hvx_vec_f32_to_f16(v1, v1);
                } else {
                    v_m_prev1 = Q6_V_vzero();
                }
            }

            HVX_Vector v_dup_m0 = Q6_Vhf_vmax_VhfVhf(v_m_prev0, v_s_rowmax0);
            HVX_Vector v_dup_m1 = Q6_Vhf_vmax_VhfVhf(v_m_prev1, v_s_rowmax1);

            // Insert row r, r+1 rowmax into rowmax_acc_v
            {
                HVX_VectorPred p_start = Q6_Q_vsetq_R(r_vec_off * 2);
                HVX_VectorPred p_mid   = Q6_Q_vsetq_R((r_vec_off + 1) * 2);
                HVX_VectorPred p_end   = Q6_Q_vsetq2_R((r_vec_off + 2) * 2);
                HVX_VectorPred p_lane0 = Q6_Q_and_QQn(p_mid, p_start);
                HVX_VectorPred p_lane1 = Q6_Q_and_QQn(p_end, p_mid);
                rowmax_acc_v           = Q6_V_vmux_QVV(p_lane0, v_dup_m0, rowmax_acc_v);
                rowmax_acc_v           = Q6_V_vmux_QVV(p_lane1, v_dup_m1, rowmax_acc_v);
            }

            // Compute P = exp(S - m_new)
            const HVX_Vector v_zero      = Q6_V_vzero();
            HVX_Vector       v_p_rowsum0 = v_zero;
            HVX_Vector       v_p_rowsum1 = v_zero;

            size_t c = 0;
            for (; c + 64 < kv_rows; c += 128) {
                size_t     ci0          = c / 64;
                size_t     ci1          = ci0 + 1;

                HVX_Vector v_s_minus_m0_0 = Q6_Vqf16_vsub_VhfVhf(my_row_buf0[ci0], v_dup_m0);
                HVX_Vector v_s_minus_m1_0 = Q6_Vqf16_vsub_VhfVhf(my_row_buf1[ci0], v_dup_m1);
                HVX_Vector v_s_minus_m0_1 = Q6_Vqf16_vsub_VhfVhf(my_row_buf0[ci1], v_dup_m0);
                HVX_Vector v_s_minus_m1_1 = Q6_Vqf16_vsub_VhfVhf(my_row_buf1[ci1], v_dup_m1);

                HVX_Vector v_p_row0_hf_0  = hvx_vec_exp2_neg_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0_0));
                HVX_Vector v_p_row1_hf_0  = hvx_vec_exp2_neg_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1_0));
                HVX_Vector v_p_row0_hf_1  = hvx_vec_exp2_neg_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0_1));
                HVX_Vector v_p_row1_hf_1  = hvx_vec_exp2_neg_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1_1));

                __fp16 *     out_dtile0  = p_st_base + ci0 * HMX_FP16_TILE_N_ELMS * 2;
                __fp16 *     out_dtile1  = p_st_base + ci1 * HMX_FP16_TILE_N_ELMS * 2;
                HVX_Vector * pv_p_out0_0  = ((HVX_Vector *) out_dtile0) + r1 / 2;
                HVX_Vector * pv_p_out1_0  = pv_p_out0_0 + 16;
                HVX_Vector * pv_p_out0_1  = ((HVX_Vector *) out_dtile1) + r1 / 2;
                HVX_Vector * pv_p_out1_1  = pv_p_out0_1 + 16;

                HVX_VectorPair vp_p_dual0 = Q6_W_vshuff_VVR(v_p_row1_hf_0, v_p_row0_hf_0, -2);
                *pv_p_out0_0               = Q6_V_lo_W(vp_p_dual0);
                *pv_p_out1_0               = Q6_V_hi_W(vp_p_dual0);

                HVX_VectorPair vp_p_dual1 = Q6_W_vshuff_VVR(v_p_row1_hf_1, v_p_row0_hf_1, -2);
                *pv_p_out0_1               = Q6_V_lo_W(vp_p_dual1);
                *pv_p_out1_1               = Q6_V_hi_W(vp_p_dual1);

                HVX_VectorPair vp_p0_0 = hvx_vec_f16_to_f32_shuff(v_p_row0_hf_0);
                HVX_VectorPair vp_p1_0 = hvx_vec_f16_to_f32_shuff(v_p_row1_hf_0);
                HVX_VectorPair vp_p0_1 = hvx_vec_f16_to_f32_shuff(v_p_row0_hf_1);
                HVX_VectorPair vp_p1_1 = hvx_vec_f16_to_f32_shuff(v_p_row1_hf_1);

                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0_0), Q6_V_hi_W(vp_p0_0)));
                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0_1), Q6_V_hi_W(vp_p0_1)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1_0), Q6_V_hi_W(vp_p1_0)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1_1), Q6_V_hi_W(vp_p1_1)));
            }
            for (size_t c_rem = c; c_rem < kv_rows; c_rem += 64) {
                size_t     ci           = c_rem / 64;
                HVX_Vector v_s_minus_m0 = Q6_Vqf16_vsub_VhfVhf(my_row_buf0[ci], v_dup_m0);
                HVX_Vector v_s_minus_m1 = Q6_Vqf16_vsub_VhfVhf(my_row_buf1[ci], v_dup_m1);

                HVX_Vector v_p_row0_hf  = hvx_vec_exp2_neg_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0));
                HVX_Vector v_p_row1_hf  = hvx_vec_exp2_neg_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1));
                __fp16 *     out_dtile  = p_st_base + ci * HMX_FP16_TILE_N_ELMS * 2;
                HVX_Vector * pv_p_out0  = ((HVX_Vector *) out_dtile) + r1 / 2;
                HVX_Vector * pv_p_out1  = pv_p_out0 + 16;

                HVX_VectorPair vp_p_dual = Q6_W_vshuff_VVR(v_p_row1_hf, v_p_row0_hf, -2);
                *pv_p_out0               = Q6_V_lo_W(vp_p_dual);
                *pv_p_out1               = Q6_V_hi_W(vp_p_dual);

                HVX_VectorPair vp_p0 = hvx_vec_f16_to_f32_shuff(v_p_row0_hf);
                HVX_VectorPair vp_p1 = hvx_vec_f16_to_f32_shuff(v_p_row1_hf);

                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0), Q6_V_hi_W(vp_p0)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1), Q6_V_hi_W(vp_p1)));
            }

            HVX_Vector rowsum0_sf = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(v_p_rowsum0));
            HVX_Vector rowsum1_sf = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(v_p_rowsum1));
            {
                HVX_Vector rv0_v = hvx_vec_f32_to_f16(rowsum0_sf, rowsum0_sf);
                HVX_Vector rv1_v = hvx_vec_f32_to_f16(rowsum1_sf, rowsum1_sf);

                HVX_VectorPred p_start = Q6_Q_vsetq_R(r_vec_off * 2);
                HVX_VectorPred p_mid   = Q6_Q_vsetq_R((r_vec_off + 1) * 2);
                HVX_VectorPred p_end   = Q6_Q_vsetq2_R((r_vec_off + 2) * 2);
                HVX_VectorPred p_lane0 = Q6_Q_and_QQn(p_mid, p_start);
                HVX_VectorPred p_lane1 = Q6_Q_and_QQn(p_end, p_mid);
                rowsum_acc_v           = Q6_V_vmux_QVV(p_lane0, rv0_v, rowsum_acc_v);
                rowsum_acc_v           = Q6_V_vmux_QVV(p_lane1, rv1_v, rowsum_acc_v);
            }
        }

        // Inline fa_ml_update_and_build_d for this vector (lock-free and in parallel)
        HVX_VectorPair rowmax_acc_pair    = hvx_vec_f16_to_f32(rowmax_acc_v);
        HVX_Vector     v_rowmax_acc_f32_0 = Q6_V_lo_W(rowmax_acc_pair);
        HVX_Vector     v_rowmax_acc_f32_1 = Q6_V_hi_W(rowmax_acc_pair);

        HVX_Vector v_m_curr0 = Q6_Vsf_vmax_VsfVsf(m_prev_v0, v_rowmax_acc_f32_0);
        HVX_Vector v_m_curr1 = Q6_Vsf_vmax_VsfVsf(m_prev_v1, v_rowmax_acc_f32_1);

        HVX_Vector v_m_diff0 = HVX_OP_SUB_F32(m_prev_v0, v_m_curr0);
        HVX_Vector v_m_diff1 = HVX_OP_SUB_F32(m_prev_v1, v_m_curr1);

        HVX_Vector v_m_diff_f16   = hvx_vec_f32_to_f16(v_m_diff0, v_m_diff1);
        HVX_Vector exp_m_diff_f16 = hvx_vec_exp2_neg_f16(v_m_diff_f16);

        HVX_VectorPair exp_m_diff_pair = hvx_vec_f16_to_f32(exp_m_diff_f16);
        HVX_Vector exp_m_diff0 = Q6_V_lo_W(exp_m_diff_pair);
        HVX_Vector exp_m_diff1 = Q6_V_hi_W(exp_m_diff_pair);

        HVX_VectorPair rowsum_acc_pair = hvx_vec_f16_to_f32(rowsum_acc_v);
        HVX_Vector     v_rowsum_acc_f32_0 = Q6_V_lo_W(rowsum_acc_pair);
        HVX_Vector     v_rowsum_acc_f32_1 = Q6_V_hi_W(rowsum_acc_pair);

        HVX_Vector l_prev_v0 = factx->vtcm_l_vec[r_vec_idx * 2 + 0];
        HVX_Vector l_prev_v1 = factx->vtcm_l_vec[r_vec_idx * 2 + 1];
        HVX_Vector v_l_curr0 = HVX_OP_ADD_F32(HVX_OP_MUL_F32(l_prev_v0, exp_m_diff0), v_rowsum_acc_f32_0);
        HVX_Vector v_l_curr1 = HVX_OP_ADD_F32(HVX_OP_MUL_F32(l_prev_v1, exp_m_diff1), v_rowsum_acc_f32_1);

        factx->vtcm_m_vec[r_vec_idx * 2 + 0] = v_m_curr0;
        factx->vtcm_m_vec[r_vec_idx * 2 + 1] = v_m_curr1;
        factx->vtcm_l_vec[r_vec_idx * 2 + 0] = v_l_curr0;
        factx->vtcm_l_vec[r_vec_idx * 2 + 1] = v_l_curr1;

        // Build diagonal tile D = diag(exp(m_diff))
        const HVX_Vector     v_offsets = *(const HVX_Vector *) d_tile_scatter_offsets;
        const HVX_VectorPred q_32_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));
        HVX_Vector           v_exp_m_diff = exp_m_diff_f16;

        __fp16 * const d_tiles_out = factx->vtcm_d_tiles[args->buf_idx];

        size_t t0 = r_vec_idx * 2;
        if (t0 < args->n_row_tiles) {
            const HVX_Vector v_content = v_exp_m_diff;
            __fp16 *         out_base  = d_tiles_out + t0 * HMX_FP16_TILE_N_ELMS;
            Q6_vscatter_QRMVhV(q_32_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
        }

        size_t t1 = r_vec_idx * 2 + 1;
        if (t1 < args->n_row_tiles) {
            const HVX_Vector v_content = Q6_V_vror_VR(v_exp_m_diff, 64);
            __fp16 *         out_base  = d_tiles_out + t1 * HMX_FP16_TILE_N_ELMS;
            Q6_vscatter_QRMVhV(q_32_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, (uint16_t) (args->q_start * G + vec_start * 64));
}

static void fa_softmax_thread_nomask(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    fa_softmax_impl(th, n, i, data,
                    /*has_mask=*/false,
                    /*mask_broadcast=*/false,
                    /*is_g1=*/false,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

static void fa_softmax_thread_mask_broadcast_g1(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    fa_softmax_impl(th, n, i, data,
                    /*has_mask=*/true,
                    /*mask_broadcast=*/true,
                    /*is_g1=*/true,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

static void fa_softmax_thread_mask_broadcast_gn(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    fa_softmax_impl(th, n, i, data,
                    /*has_mask=*/true,
                    /*mask_broadcast=*/true,
                    /*is_g1=*/false,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

static void fa_softmax_thread(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    fa_softmax_args_t *     args  = (fa_softmax_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const bool has_mask       = (args->mask != NULL);
    const bool mask_broadcast = factx->mask_broadcast;
    const bool is_g1          = (args->G == 1);
    const bool has_alibi      = args->has_alibi;
    const bool has_softcap    = factx->has_softcap;

    fa_softmax_impl(th, n, i, data, has_mask, mask_broadcast, is_g1, has_alibi, has_softcap);
}

static inline HVX_Vector hvx_fa_load_sinks(const float * sinks_data, size_t idx, size_t count) {
    const HVX_Vector * v_src    = (const HVX_Vector *) sinks_data + (idx >> 5);
    const size_t       lane_off = idx & 31;
    const size_t       byte_off = lane_off * sizeof(float);
    return (lane_off + count > 32)
        ? Q6_V_valign_VVR(v_src[1], v_src[0], byte_off)
        : Q6_V_vror_VR(v_src[0], byte_off);
}

static __attribute__((noinline)) void fa_build_d_diag_inv_l(struct hmx_fa_context * factx,
                                                            size_t                  n_row_tiles,
                                                            size_t                  n_row_tiles_g_br,
                                                            uint32_t                kv_head,
                                                            size_t                  n_rows_g) {
    (void) n_row_tiles_g_br;

    // sinks: l += exp(sink - m * ln2) in f32, m and l are in the log2 domain
    if (factx->sinks) {
        const float *    sinks_data   = factx->vtcm_sinks;
        const size_t     G            = factx->G;
        const size_t     head_base    = (size_t) kv_head * G;
        const HVX_Vector v_ln2        = Q6_V_vsplat_R(EXP_LOGN2);
        const HVX_Vector v_head_sinks = (G <= 32) ? hvx_fa_load_sinks(sinks_data, head_base, G) : Q6_V_vzero();
        HVX_Vector       v_sink       = Q6_V_vzero();
        size_t           g_prev       = SIZE_MAX;

        for (size_t i = 0; i < n_row_tiles; ++i) {
            // the sink pattern of a tile only depends on (i * 32) % G, rebuild it when that changes
            const size_t r0 = i * 32;
            const size_t g0 = fastmodulo(r0, G, &factx->div_G);
            if (g0 != g_prev) {
                const size_t rem_g = G - g0;
                if (G <= 32) {
                    const HVX_Vector v_lo = Q6_V_vror_VR(v_head_sinks, g0 * sizeof(float));
                    const HVX_Vector v_hi = Q6_V_vror_VR(v_head_sinks, 128 - rem_g * sizeof(float));
                    v_sink = Q6_V_vmux_QVV(Q6_Q_vsetq2_R(rem_g * sizeof(float)), v_lo, v_hi);
                    for (size_t len = G; len < 32; len <<= 1) {
                        const HVX_VectorPred q_keep    = Q6_Q_vsetq_R(len * sizeof(float));
                        const HVX_Vector     v_shifted = Q6_V_vror_VR(v_sink, 128 - len * sizeof(float));
                        v_sink = Q6_V_vmux_QVV(q_keep, v_sink, v_shifted);
                    }
                } else if (rem_g >= 32) {
                    v_sink = hvx_fa_load_sinks(sinks_data, head_base + g0, 32);
                } else {
                    const HVX_Vector v_lo = hvx_fa_load_sinks(sinks_data, head_base + g0, rem_g);
                    const HVX_Vector v_hi = hvx_fa_load_sinks(sinks_data, head_base, 32 - rem_g);
                    v_sink = Q6_V_vmux_QVV(Q6_Q_vsetq_R(rem_g * sizeof(float)), v_lo, Q6_V_vror_VR(v_hi, 128 - rem_g * sizeof(float)));
                }
                g_prev = g0;
            }

            HVX_Vector v_d = HVX_OP_SUB_F32(v_sink, HVX_OP_MUL_F32(factx->vtcm_m_vec[i], v_ln2));
            if (r0 + 32 > n_rows_g) {
                const HVX_VectorPred q_valid = Q6_Q_vsetq_R((n_rows_g - r0) * sizeof(float));
                v_d = Q6_V_vmux_QVV(q_valid, v_d, Q6_V_vzero());
            }
            factx->vtcm_l_vec[i] = HVX_OP_ADD_F32(factx->vtcm_l_vec[i], hvx_vec_exp_f32(v_d));
        }
    }

    const HVX_Vector     v_offsets = *(const HVX_Vector *) d_tile_scatter_offsets;
    const HVX_VectorPred q_32_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));
    const HVX_Vector     one       = hvx_vec_splat_f32(1.0f);

    HVX_Vector v_content = Q6_V_vzero();
    for (size_t i = 0; i < n_row_tiles; ++i) {
        if ((i % 2) == 0) {
            HVX_Vector inv_lo = HVX_OP_MUL_F32(one, hvx_vec_inverse_f32(factx->vtcm_l_vec[i]));
            HVX_Vector inv_hi = (i + 1 < n_row_tiles) ? HVX_OP_MUL_F32(one, hvx_vec_inverse_f32(factx->vtcm_l_vec[i + 1])) : Q6_V_vzero();
            v_content = hvx_vec_f32_to_f16(inv_lo, inv_hi);
        } else {
            v_content = Q6_V_vror_VR(v_content, 64);
        }

        __fp16 * out_base = factx->vtcm_d_inv_l + i * HMX_FP16_TILE_N_ELMS;
        Q6_vscatter_QRMVhV(q_32_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
    }
}

static __attribute__((always_inline)) inline void fa_w8a8_mask_and_rowmax_pair(
    struct hmx_fa_context * factx,
    fa_softmax_args_t *     args,
    HVX_Vector *            row_buf0,
    HVX_Vector *            row_buf1,
    uint32_t                r,
    uint32_t                r_vec_off,
    size_t                  n_rows_g,
    size_t                  kv_rows,
    const __fp16 *          slopes_row64,
    HVX_Vector *            v_rowmax0_out,
    HVX_Vector *            v_rowmax1_out,
    const bool              has_mask,
    const bool              mask_broadcast,
    const bool              is_g1,
    const bool              has_alibi
) {
    int32_t slope0_u16 = 0;
    int32_t slope1_u16 = 0;
    if (has_alibi) {
        const HVX_Vector v_sl = hvx_vmem(slopes_row64);
        slope0_u16 = Q6_R_vextract_VR(v_sl, (int) (r_vec_off * 2));
        if (r + 1 < n_rows_g) {
            slope1_u16 = Q6_R_vextract_VR(v_sl, (int) ((r_vec_off + 1) * 2));
        }
    }

    HVX_Vector v_s_rowmax0 = Q6_Vh_vsplat_R(0xfbff);
    HVX_Vector v_s_rowmax1 = Q6_Vh_vsplat_R(0xfbff);

    if (has_mask) {
        #pragma clang loop pipeline(disable) unroll(disable)
        for (size_t c = 0; c < kv_rows; c += 64) {
            const HVX_Vector v_neg_inf   = Q6_Vh_vsplat_R(0xfbff);
            const HVX_Vector v_threshold = Q6_Vh_vsplat_R(0xcc00); // fp16 -16.0
            size_t           ci          = c / 64;
            const size_t     ne          = hex_smin(kv_rows - c, 64);
            HVX_VectorPred   q_tail_keep = Q6_Q_vsetq2_R(ne * sizeof(__fp16));

            HVX_Vector v_mask0, v_mask1;
            if (mask_broadcast) {
                if (is_g1) {
                    const size_t qi0 = r + 0;
                    v_mask0 = *(const HVX_Vector *) (args->mask_vtcm + qi0 * args->mask_vtcm_row_stride + c);
                    v_mask1 = v_neg_inf;
                    if (r + 1 < n_rows_g) {
                        const size_t qi1 = r + 1;
                        v_mask1 = *(const HVX_Vector *) (args->mask_vtcm + qi1 * args->mask_vtcm_row_stride + c);
                    }
                } else {
                    const size_t qi0 = fastdiv(r + 0, &factx->div_G);
                    v_mask0 = *(const HVX_Vector *) (args->mask_vtcm + qi0 * args->mask_vtcm_row_stride + c);
                    v_mask1 = v_neg_inf;
                    if (r + 1 < n_rows_g) {
                        const size_t qi1 = fastdiv(r + 1, &factx->div_G);
                        if (qi1 == qi0) {
                            v_mask1 = v_mask0;
                        } else {
                            v_mask1 = *(const HVX_Vector *) (args->mask_vtcm + qi1 * args->mask_vtcm_row_stride + c);
                        }
                    }
                }
            } else {
                const size_t m0 = r + 0;
                v_mask0 = *(const HVX_Vector *) (args->mask_vtcm + m0 * args->mask_vtcm_row_stride + c);
                v_mask1 = v_neg_inf;
                if (r + 1 < n_rows_g) {
                    const size_t m1 = r + 1;
                    v_mask1 = *(const HVX_Vector *) (args->mask_vtcm + m1 * args->mask_vtcm_row_stride + c);
                }
            }

            const HVX_Vector v_log2e  = hvx_vec_splat_f16(EXP_LOG2E_F);
            {
                HVX_VectorPred q_keep0   = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask0, v_threshold), q_tail_keep);
                HVX_Vector     v_m0      = hvx_vec_mul_f16_f16(v_mask0, v_log2e);
                if (has_alibi) {
                    v_m0 = hvx_vec_mul_f16_f16(v_m0, Q6_Vh_vsplat_R(slope0_u16));
                }
                HVX_Vector v_r0 = Q6_V_vmux_QVV(q_keep0, hvx_vec_add_f16_f16(row_buf0[ci], v_m0), v_neg_inf);
                row_buf0[ci]    = v_r0;
                v_s_rowmax0     = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, v_r0);
            }
            {
                HVX_VectorPred q_keep1   = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask1, v_threshold), q_tail_keep);
                HVX_Vector     v_m1      = hvx_vec_mul_f16_f16(v_mask1, v_log2e);
                if (has_alibi) {
                    v_m1 = hvx_vec_mul_f16_f16(v_m1, Q6_Vh_vsplat_R(slope1_u16));
                }
                HVX_Vector v_r1 = Q6_V_vmux_QVV(q_keep1, hvx_vec_add_f16_f16(row_buf1[ci], v_m1), v_neg_inf);
                row_buf1[ci]    = v_r1;
                v_s_rowmax1     = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, v_r1);
            }
        }
    } else {
        size_t c = 0;
        for (; c + 128 <= kv_rows; c += 128) {
            size_t ci0 = c / 64;
            size_t ci1 = ci0 + 1;
            v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, row_buf0[ci0]);
            v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, row_buf1[ci0]);
            v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, row_buf0[ci1]);
            v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, row_buf1[ci1]);
        }
        for (; c < kv_rows; c += 64) {
            const HVX_Vector v_neg_inf   = Q6_Vh_vsplat_R(0xfbff);
            size_t           ci          = c / 64;
            const size_t     ne          = hex_smin(kv_rows - c, 64);
            HVX_VectorPred   q_tail_keep = Q6_Q_vsetq2_R(ne * sizeof(__fp16));
            if (ne < 64) {
                row_buf0[ci] = Q6_V_vmux_QVV(q_tail_keep, row_buf0[ci], v_neg_inf);
                row_buf1[ci] = Q6_V_vmux_QVV(q_tail_keep, row_buf1[ci], v_neg_inf);
            }
            v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, row_buf0[ci]);
            v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, row_buf1[ci]);
        }
    }

    const HVX_Vector v_neg_inf = Q6_Vh_vsplat_R(0xfbff);
    *v_rowmax0_out = (r + 0 < n_rows_g) ? hvx_vec_reduce_max_f16(v_s_rowmax0) : v_neg_inf;
    *v_rowmax1_out = (r + 1 < n_rows_g) ? hvx_vec_reduce_max_f16(v_s_rowmax1) : v_neg_inf;
}

static __attribute__((noinline)) void fa_w8a8_exp_quant_quad(
    HVX_Vector * restrict row_buf0,
    HVX_Vector * restrict row_buf1,
    HVX_Vector * restrict row_buf2,
    HVX_Vector * restrict row_buf3,
    uint8_t * restrict    p_row64_base,
    uint32_t              rg,
    size_t                kv_rows,
    uint32_t              n_col_tiles,
    int32_t               smax0,
    int32_t               smax1,
    int32_t               smax2,
    int32_t               smax3,
    uint32_t              valid_mask,
    float * restrict      rowsum_dst,
    uint32_t              rmp0_bits,
    uint32_t              rmp1_bits,
    uint32_t              rmp2_bits,
    uint32_t              rmp3_bits
) {
    HVX_Vector * bufs[4]  = { row_buf0, row_buf1, row_buf2, row_buf3 };
    int32_t      smaxs[4] = { smax0, smax1, smax2, smax3 };
    uint32_t     rmps[4]  = { rmp0_bits, rmp1_bits, rmp2_bits, rmp3_bits };
    int32_t      sum_bits[4];

    for (uint32_t k = 0; k < 4; ++k) {
        HVX_Vector *  rb    = bufs[k];
        const int32_t sm    = smaxs[k];
        HVX_Vector    v_sum = Q6_V_vzero();
        #pragma clang loop pipeline(disable) unroll(disable)
        for (size_t c = 0; c < kv_rows; c += 64) {
            const size_t   ci  = c / 64;
            HVX_Vector     v_p = hvx_vec_exp2_neg_f16(Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(rb[ci], Q6_Vh_vsplat_R(sm))));
            rb[ci]             = v_p;
            HVX_VectorPair vp  = hvx_vec_f16_to_f32_shuff(v_p);
            v_sum              = Q6_Vqf32_vadd_Vqf32Vqf32(v_sum, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp), Q6_V_hi_W(vp)));
        }
        HVX_Vector v_s = Q6_Vsf_equals_Vqf32(hvx_qf32_mul_sf_scalar(hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(v_sum)), rmps[k]));
        sum_bits[k]    = Q6_R_vextract_VR(v_s, 0);
    }

    {
        HVX_Vector v02 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(sum_bits[2]), Q6_V_vsplat_R(sum_bits[0]), -4));
        HVX_Vector v13 = Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_vsplat_R(sum_bits[3]), Q6_V_vsplat_R(sum_bits[1]), -4));
        hvx_vec_store_u(rowsum_dst, 4 * sizeof(float), Q6_V_lo_W(Q6_W_vshuff_VVR(v13, v02, -4)));
    }

    #pragma clang loop pipeline(disable) unroll(disable)
    for (size_t c = 0; c < kv_rows; c += 64) {
        const size_t     ci       = c / 64;
        const uint32_t   kt       = (uint32_t) (ci * 2);
        const bool       write_hi = (kt + 1 < n_col_tiles);
        const HVX_Vector v_255_hf = hvx_vec_splat_f16(255.0f);

        HVX_Vector hf0 = (valid_mask & 1u) ? hvx_vec_mul_f16_f16(row_buf0[ci], v_255_hf) : Q6_V_vzero();
        HVX_Vector hf1 = (valid_mask & 2u) ? hvx_vec_mul_f16_f16(row_buf1[ci], v_255_hf) : Q6_V_vzero();
        HVX_Vector hf2 = (valid_mask & 4u) ? hvx_vec_mul_f16_f16(row_buf2[ci], v_255_hf) : Q6_V_vzero();
        HVX_Vector hf3 = (valid_mask & 8u) ? hvx_vec_mul_f16_f16(row_buf3[ci], v_255_hf) : Q6_V_vzero();

        hmx_fa_w8a8_pack_p_quad_u8_sm(p_row64_base, kt, rg, write_hi, hf0, hf1, hf2, hf3);
    }
}

static __attribute__((noinline)) void fa_w8a8_decode_s_quad_rows_nocap(
    const uint8_t * restrict tile_base,
    const float * restrict   dw_base,
    HVX_Vector * restrict    row_buf0,
    HVX_Vector * restrict    row_buf1,
    HVX_Vector * restrict    row_buf2,
    HVX_Vector * restrict    row_buf3,
    size_t                   kv_rows,
    uint32_t                 da0,
    uint32_t                 da1,
    uint32_t                 da2,
    uint32_t                 da3
) {
    #pragma clang loop pipeline(disable) unroll(disable)
    for (size_t c = 0; c < kv_rows; c += 64) {
        const size_t     ci      = c / 64;
        const HVX_Vector sign_80 = Q6_Vb_vsplat_R(0x80);
        HVX_Vector s0_lo, s1_lo, s2_lo, s3_lo;
        hmx_fa_w8a8_eval_s_tile_quad(tile_base, dw_base, c, da0, da1, da2, da3, sign_80,
                                     &s0_lo, &s1_lo, &s2_lo, &s3_lo);

        HVX_Vector s0_hi = Q6_V_vzero();
        HVX_Vector s1_hi = Q6_V_vzero();
        HVX_Vector s2_hi = Q6_V_vzero();
        HVX_Vector s3_hi = Q6_V_vzero();
        if (c + 32 < kv_rows) {
            hmx_fa_w8a8_eval_s_tile_quad(tile_base, dw_base, c + 32, da0, da1, da2, da3, sign_80,
                                         &s0_hi, &s1_hi, &s2_hi, &s3_hi);
        }

        row_buf0[ci] = hvx_vec_f32_to_f16(s0_lo, s0_hi);
        row_buf1[ci] = hvx_vec_f32_to_f16(s1_lo, s1_hi);
        row_buf2[ci] = hvx_vec_f32_to_f16(s2_lo, s2_hi);
        row_buf3[ci] = hvx_vec_f32_to_f16(s3_lo, s3_hi);
    }
}

static __attribute__((noinline)) void fa_w8a8_decode_s_quad_rows_cap(
    const uint8_t * restrict tile_base,
    const float * restrict   dw_base,
    HVX_Vector * restrict    row_buf0,
    HVX_Vector * restrict    row_buf1,
    HVX_Vector * restrict    row_buf2,
    HVX_Vector * restrict    row_buf3,
    size_t                   kv_rows,
    uint32_t                 da0,
    uint32_t                 da1,
    uint32_t                 da2,
    uint32_t                 da3,
    const __fp16 *           logit_softcap_ptr
) {
    fa_w8a8_decode_s_quad_rows_nocap(
        tile_base, dw_base, row_buf0, row_buf1, row_buf2, row_buf3,
        kv_rows, da0, da1, da2, da3);

    HVX_Vector * bufs[4] = { row_buf0, row_buf1, row_buf2, row_buf3 };
    const __fp16 cap     = *logit_softcap_ptr;
    for (uint32_t k = 0; k < 4; ++k) {
        HVX_Vector * rb = bufs[k];
        #pragma clang loop pipeline(disable) unroll(disable)
        for (size_t c = 0; c < kv_rows; c += 64) {
            const size_t ci = c / 64;
            rb[ci] = hvx_vec_mul_f16_f16(hvx_vec_tanh_f16(rb[ci]), hvx_vec_splat_f16(cap));
        }
    }
}

static __attribute__((always_inline)) inline void fa_w8a8_softmax_impl(
    const htp_thread_t * th, unsigned int n, unsigned int i, void * data,
    const bool has_mask,
    const bool mask_broadcast,
    const bool is_g1,
    const bool has_alibi,
    const bool has_softcap
) {
    fa_softmax_args_t *     args  = (fa_softmax_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const size_t n_rows_g      = args->n_rows_g;
    const size_t kv_rows       = args->kv_rows;
    const size_t G             = args->G;
    const size_t n_row_vec_cnt = hmx_ceil_div(n_rows_g, 64);

    size_t vec_start = 0;
    size_t vec_end   = n_row_vec_cnt;
    if (n > 1) {
        const size_t vecs_per_t = fastdiv(n_row_vec_cnt + n - 1, &args->thread_div);
        vec_start = i * vecs_per_t;
        vec_end   = hex_smin(vec_start + vecs_per_t, n_row_vec_cnt);
    }

    if (vec_start >= n_row_vec_cnt) {
        return;
    }

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_SFM, (uint16_t) (args->q_start * G + vec_start * 64));

    const size_t row_buf_stride = factx->row_buf_stride;
    HVX_Vector * my_row_buf0    = factx->vtcm_row_bufs + (size_t) i * 4 * row_buf_stride;
    HVX_Vector * my_row_buf1    = my_row_buf0 + row_buf_stride;
    HVX_Vector * my_row_buf2    = my_row_buf1 + row_buf_stride;
    HVX_Vector * my_row_buf3    = my_row_buf2 + row_buf_stride;

    const float *   dw_base   = factx->vtcm_k_scales[args->buf_idx];
    const uint8_t * s_base    = (const uint8_t *) factx->vtcm_s_tiles[args->buf_idx];
    uint8_t *       p_base    = (uint8_t *) factx->vtcm_p_tiles[args->buf_idx];
    float *         p_scales  = factx->vtcm_p_scales[args->buf_idx];
    float *         s_rowmax  = (float *) factx->vtcm_s_rowmax;
    float *         p_rowsum  = (float *) factx->vtcm_p_rowsum;
    const uint32_t  n_ct      = (uint32_t) args->n_col_tiles;

    for (size_t r_vec_idx = vec_start; r_vec_idx < vec_end; ++r_vec_idx) {
        factx->vtcm_s_rowmax[r_vec_idx * 2 + 0] = factx->vtcm_m_vec[r_vec_idx * 2 + 0];
        factx->vtcm_s_rowmax[r_vec_idx * 2 + 1] = factx->vtcm_m_vec[r_vec_idx * 2 + 1];
        factx->vtcm_p_rowsum[r_vec_idx * 2 + 0] = Q6_V_vzero();
        factx->vtcm_p_rowsum[r_vec_idx * 2 + 1] = Q6_V_vzero();

        const uint8_t * s_row64_base = s_base + r_vec_idx * n_ct * 2 * HMX_W8A8_ACT_TILE_SIZE;
        uint8_t *       p_row64_base = p_base + r_vec_idx * n_ct * HMX_W8A8_ACT_TILE_SIZE;

        #pragma clang loop unroll(disable)
        for (uint32_t r_vec_off = 0; r_vec_off < 64; r_vec_off += 4) {
            const uint32_t r = (uint32_t) (r_vec_idx * 64 + r_vec_off);
            if (r >= n_rows_g) {
                break;
            }

            const uint32_t  rg        = r_vec_off >> 2;
            const uint8_t * tile_base = s_row64_base + (size_t) rg * 128;

            const HVX_Vector v_da = hvx_vmem(factx->vtcm_q_scales + r_vec_idx * 64 + (r_vec_off & 32u));
            const int        doff = (int) ((r_vec_off & 31u) * 4u);
            const uint32_t   da0  = (uint32_t) Q6_R_vextract_VR(v_da, doff + 0);
            const uint32_t   da1  = (uint32_t) Q6_R_vextract_VR(v_da, doff + 4);
            const uint32_t   da2  = (uint32_t) Q6_R_vextract_VR(v_da, doff + 8);
            const uint32_t   da3  = (uint32_t) Q6_R_vextract_VR(v_da, doff + 12);

            if (has_softcap) {
                fa_w8a8_decode_s_quad_rows_cap(
                    tile_base, dw_base,
                    my_row_buf0, my_row_buf1, my_row_buf2, my_row_buf3,
                    kv_rows, da0, da1, da2, da3, &factx->logit_softcap);
            } else {
                fa_w8a8_decode_s_quad_rows_nocap(
                    tile_base, dw_base,
                    my_row_buf0, my_row_buf1, my_row_buf2, my_row_buf3,
                    kv_rows, da0, da1, da2, da3);
            }

            const HVX_Vector v_neg_inf    = Q6_Vh_vsplat_R(0xfbff);
            const __fp16 *   slopes_row64 = args->slopes + r_vec_idx * 64;
            HVX_Vector v_smax0, v_smax1, v_smax2, v_smax3;
            fa_w8a8_mask_and_rowmax_pair(
                factx, args, my_row_buf0, my_row_buf1,
                r + 0, r_vec_off + 0, n_rows_g, kv_rows, slopes_row64,
                &v_smax0, &v_smax1, has_mask, mask_broadcast, is_g1, has_alibi);
            if (r + 2 < n_rows_g) {
                fa_w8a8_mask_and_rowmax_pair(
                    factx, args, my_row_buf2, my_row_buf3,
                    r + 2, r_vec_off + 2, n_rows_g, kv_rows, slopes_row64,
                    &v_smax2, &v_smax3, has_mask, mask_broadcast, is_g1, has_alibi);
            } else {
                v_smax2 = v_neg_inf;
                v_smax3 = v_neg_inf;
            }

            // Compute m_new = max(m_prev, s_rowmax) and store to vtcm_s_rowmax
            HVX_Vector m_src = factx->vtcm_m_vec[r_vec_idx * 2 + (r_vec_off >> 5)];
            uint32_t   m_off = r_vec_off & 31u;
            HVX_Vector mp0   = hvx_vec_repl_f32(Q6_V_vror_VR(m_src, (m_off + 0) * 4));
            HVX_Vector mp1   = hvx_vec_repl_f32(Q6_V_vror_VR(m_src, (m_off + 1) * 4));
            HVX_Vector mp2   = hvx_vec_repl_f32(Q6_V_vror_VR(m_src, (m_off + 2) * 4));
            HVX_Vector mp3   = hvx_vec_repl_f32(Q6_V_vror_VR(m_src, (m_off + 3) * 4));

            HVX_Vector v_dup_m0 = Q6_Vhf_vmax_VhfVhf(hvx_vec_f32_to_f16(mp0, mp0), v_smax0);
            HVX_Vector v_dup_m1 = Q6_Vhf_vmax_VhfVhf(hvx_vec_f32_to_f16(mp1, mp1), v_smax1);
            HVX_Vector v_dup_m2 = Q6_Vhf_vmax_VhfVhf(hvx_vec_f32_to_f16(mp2, mp2), v_smax2);
            HVX_Vector v_dup_m3 = Q6_Vhf_vmax_VhfVhf(hvx_vec_f32_to_f16(mp3, mp3), v_smax3);

            {
                HVX_Vector fm0 = Q6_V_lo_W(hvx_vec_f16_to_f32(v_dup_m0));
                HVX_Vector fm1 = Q6_V_lo_W(hvx_vec_f16_to_f32(v_dup_m1));
                HVX_Vector fm2 = Q6_V_lo_W(hvx_vec_f16_to_f32(v_dup_m2));
                HVX_Vector fm3 = Q6_V_lo_W(hvx_vec_f16_to_f32(v_dup_m3));
                HVX_Vector m02 = Q6_V_lo_W(Q6_W_vshuff_VVR(fm2, fm0, -4));
                HVX_Vector m13 = Q6_V_lo_W(Q6_W_vshuff_VVR(fm3, fm1, -4));
                hvx_vec_store_u(s_rowmax + r, 4 * sizeof(float), Q6_V_lo_W(Q6_W_vshuff_VVR(m13, m02, -4)));
            }

            // row_max_p = exp2(s_rowmax - m_new); zero if row is masked or out of bounds
            HVX_VectorPred q_ok0 = Q6_Q_vcmp_gt_VhfVhf(v_smax0, v_neg_inf);
            HVX_VectorPred q_ok1 = Q6_Q_vcmp_gt_VhfVhf(v_smax1, v_neg_inf);
            HVX_VectorPred q_ok2 = Q6_Q_vcmp_gt_VhfVhf(v_smax2, v_neg_inf);
            HVX_VectorPred q_ok3 = Q6_Q_vcmp_gt_VhfVhf(v_smax3, v_neg_inf);

            const HVX_Vector v_one_i = Q6_V_vsplat_R(1);
            uint32_t valid_mask = 0;
            valid_mask |= (Q6_R_vextract_VR(Q6_V_vmux_QVV(q_ok0, v_one_i, Q6_V_vzero()), 0) & 1u) << 0;
            valid_mask |= (Q6_R_vextract_VR(Q6_V_vmux_QVV(q_ok1, v_one_i, Q6_V_vzero()), 0) & 1u) << 1;
            valid_mask |= (Q6_R_vextract_VR(Q6_V_vmux_QVV(q_ok2, v_one_i, Q6_V_vzero()), 0) & 1u) << 2;
            valid_mask |= (Q6_R_vextract_VR(Q6_V_vmux_QVV(q_ok3, v_one_i, Q6_V_vzero()), 0) & 1u) << 3;

            const int32_t smax0 = Q6_R_vextract_VR(v_smax0, 0);
            const int32_t smax1 = Q6_R_vextract_VR(v_smax1, 0);
            const int32_t smax2 = Q6_R_vextract_VR(v_smax2, 0);
            const int32_t smax3 = Q6_R_vextract_VR(v_smax3, 0);

            HVX_Vector v_rmp_hf0 = Q6_V_vmux_QVV(q_ok0, hvx_vec_exp2_neg_f16(Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(v_smax0, v_dup_m0))), Q6_V_vzero());
            HVX_Vector v_rmp_hf1 = Q6_V_vmux_QVV(q_ok1, hvx_vec_exp2_neg_f16(Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(v_smax1, v_dup_m1))), Q6_V_vzero());
            HVX_Vector v_rmp_hf2 = Q6_V_vmux_QVV(q_ok2, hvx_vec_exp2_neg_f16(Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(v_smax2, v_dup_m2))), Q6_V_vzero());
            HVX_Vector v_rmp_hf3 = Q6_V_vmux_QVV(q_ok3, hvx_vec_exp2_neg_f16(Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(v_smax3, v_dup_m3))), Q6_V_vzero());

            const uint32_t rmp0_bits = (uint32_t) Q6_R_vextract_VR(Q6_V_lo_W(hvx_vec_f16_to_f32(v_rmp_hf0)), 0);
            const uint32_t rmp1_bits = (uint32_t) Q6_R_vextract_VR(Q6_V_lo_W(hvx_vec_f16_to_f32(v_rmp_hf1)), 0);
            const uint32_t rmp2_bits = (uint32_t) Q6_R_vextract_VR(Q6_V_lo_W(hvx_vec_f16_to_f32(v_rmp_hf2)), 0);
            const uint32_t rmp3_bits = (uint32_t) Q6_R_vextract_VR(Q6_V_lo_W(hvx_vec_f16_to_f32(v_rmp_hf3)), 0);

            {
                const HVX_Vector v_inv255 = hvx_vec_splat_f32(1.0f / 255.0f);
                HVX_Vector vd0   = hvx_vec_mul_f32_f32(Q6_V_vsplat_R((int32_t) rmp0_bits), v_inv255);
                HVX_Vector vd1   = hvx_vec_mul_f32_f32(Q6_V_vsplat_R((int32_t) rmp1_bits), v_inv255);
                HVX_Vector vd2   = hvx_vec_mul_f32_f32(Q6_V_vsplat_R((int32_t) rmp2_bits), v_inv255);
                HVX_Vector vd3   = hvx_vec_mul_f32_f32(Q6_V_vsplat_R((int32_t) rmp3_bits), v_inv255);
                HVX_Vector v_d02 = Q6_V_lo_W(Q6_W_vshuff_VVR(vd2, vd0, -4));
                HVX_Vector v_d13 = Q6_V_lo_W(Q6_W_vshuff_VVR(vd3, vd1, -4));
                HVX_Vector v_dp  = Q6_V_lo_W(Q6_W_vshuff_VVR(v_d13, v_d02, -4));
                hvx_vec_store_u(p_scales + r, 4 * sizeof(float), v_dp);
            }

            fa_w8a8_exp_quant_quad(
                my_row_buf0, my_row_buf1, my_row_buf2, my_row_buf3,
                p_row64_base, rg, kv_rows, n_ct,
                smax0, smax1, smax2, smax3, valid_mask,
                p_rowsum + r, rmp0_bits, rmp1_bits, rmp2_bits, rmp3_bits);
        }

        HVX_Vector m_prev_v0 = factx->vtcm_m_vec[r_vec_idx * 2 + 0];
        HVX_Vector m_prev_v1 = factx->vtcm_m_vec[r_vec_idx * 2 + 1];
        HVX_Vector v_m_curr0 = Q6_Vsf_vmax_VsfVsf(m_prev_v0, factx->vtcm_s_rowmax[r_vec_idx * 2 + 0]);
        HVX_Vector v_m_curr1 = Q6_Vsf_vmax_VsfVsf(m_prev_v1, factx->vtcm_s_rowmax[r_vec_idx * 2 + 1]);

        HVX_Vector v_m_diff0 = HVX_OP_SUB_F32(m_prev_v0, v_m_curr0);
        HVX_Vector v_m_diff1 = HVX_OP_SUB_F32(m_prev_v1, v_m_curr1);

        HVX_Vector     v_m_diff_f16    = hvx_vec_f32_to_f16(v_m_diff0, v_m_diff1);
        HVX_Vector     exp_m_diff_f16  = hvx_vec_exp2_neg_f16(v_m_diff_f16);
        HVX_VectorPair exp_m_diff_pair = hvx_vec_f16_to_f32(exp_m_diff_f16);
        HVX_Vector     exp_m_diff0     = Q6_V_lo_W(exp_m_diff_pair);
        HVX_Vector     exp_m_diff1     = Q6_V_hi_W(exp_m_diff_pair);

        HVX_Vector l_prev_v0 = factx->vtcm_l_vec[r_vec_idx * 2 + 0];
        HVX_Vector l_prev_v1 = factx->vtcm_l_vec[r_vec_idx * 2 + 1];
        HVX_Vector v_l_curr0 = HVX_OP_ADD_F32(HVX_OP_MUL_F32(l_prev_v0, exp_m_diff0), factx->vtcm_p_rowsum[r_vec_idx * 2 + 0]);
        HVX_Vector v_l_curr1 = HVX_OP_ADD_F32(HVX_OP_MUL_F32(l_prev_v1, exp_m_diff1), factx->vtcm_p_rowsum[r_vec_idx * 2 + 1]);

        factx->vtcm_m_vec[r_vec_idx * 2 + 0] = v_m_curr0;
        factx->vtcm_m_vec[r_vec_idx * 2 + 1] = v_m_curr1;
        factx->vtcm_l_vec[r_vec_idx * 2 + 0] = v_l_curr0;
        factx->vtcm_l_vec[r_vec_idx * 2 + 1] = v_l_curr1;

        HVX_Vector * exp_mdiff_out = (HVX_Vector *) (factx->vtcm_exp_mdiff[args->buf_idx] + r_vec_idx * 64);
        exp_mdiff_out[0] = exp_m_diff0;
        exp_mdiff_out[1] = exp_m_diff1;
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, (uint16_t) (args->q_start * G + vec_start * 64));
}

static void fa_w8a8_softmax_thread_nomask(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    fa_w8a8_softmax_impl(th, n, i, data, false, false, false, false, false);
}

static void fa_w8a8_softmax_thread_mask_broadcast_g1(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    fa_w8a8_softmax_impl(th, n, i, data, true, true, true, false, false);
}

static void fa_w8a8_softmax_thread_mask_broadcast_gn(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    fa_w8a8_softmax_impl(th, n, i, data, true, true, false, false, false);
}

static void fa_w8a8_softmax_thread(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    fa_softmax_args_t *     args  = (fa_softmax_args_t *) data;
    struct hmx_fa_context * factx = args->factx;
    fa_w8a8_softmax_impl(th, n, i, data,
                         args->mask != NULL, factx->mask_broadcast, args->G == 1,
                         args->has_alibi, factx->has_softcap);
}

static void fa_phase_softmax_and_build_d(struct hmx_fa_context * factx,
                                         fa_softmax_args_t *     sargs,
                                         size_t                  n_row_tiles,
                                         size_t                  n_row_tiles_g_br) {
    work_queue_t wp = factx->octx->ctx->hvx_queue;
    const size_t n_row_vec_cnt = hmx_ceil_div(sargs->n_rows_g, 64);

    work_queue_func_t softmax_fn;
    if (factx->is_w8a8) {
        softmax_fn = fa_w8a8_softmax_thread;
        if (sargs->mask == NULL && !factx->has_softcap && !sargs->has_alibi) {
            softmax_fn = fa_w8a8_softmax_thread_nomask;
        } else if (sargs->mask != NULL && factx->mask_broadcast && !factx->has_softcap && !sargs->has_alibi) {
            softmax_fn = (sargs->G == 1) ? fa_w8a8_softmax_thread_mask_broadcast_g1 : fa_w8a8_softmax_thread_mask_broadcast_gn;
        }
    } else {
        softmax_fn = fa_softmax_thread;
        if (sargs->mask == NULL && !factx->has_softcap && !sargs->has_alibi) {
            softmax_fn = fa_softmax_thread_nomask;
        } else if (sargs->mask != NULL && factx->mask_broadcast && !factx->has_softcap && !sargs->has_alibi) {
            softmax_fn = (sargs->G == 1) ? fa_softmax_thread_mask_broadcast_g1 : fa_softmax_thread_mask_broadcast_gn;
        }
    }

    uint32_t n_use = 1;
    if (factx->n_hvx > 1 && n_row_vec_cnt >= 2) {
        n_use = (uint32_t) hex_smin((size_t) factx->n_hvx, n_row_vec_cnt);
        sargs->thread_div = init_fastdiv_values(n_use);
    }
    work_queue_run(wp, softmax_fn, sargs, n_use);
}

typedef struct {
    struct hmx_fa_context * factx;
    size_t                  buf_idx;
    size_t                  n_rows_g;
    uint32_t                q_start;
    bool                    is_first;
} fa_w8a8_o_accum_args_t;

static __attribute__((noinline)) void fa_w8a8_o_accum_quad_first(
        const uint8_t * restrict tile_base,
        const float * restrict   dv_scales,
        float * restrict         o0,
        float * restrict         o1,
        float * restrict         o2,
        float * restrict         o3,
        size_t                   DV,
        uint32_t                 dp0,
        uint32_t                 dp1,
        uint32_t                 dp2,
        uint32_t                 dp3) {
    const HVX_Vector sign_80 = Q6_Vb_vsplat_R(0x80);
    #pragma clang loop pipeline(disable) unroll(disable)
    for (size_t c = 0; c < DV; c += 32) {
        HVX_Vector do0, do1, do2, do3;
        hmx_fa_w8a8_eval_s_tile_quad(tile_base, dv_scales, c, dp0, dp1, dp2, dp3, sign_80,
                                     &do0, &do1, &do2, &do3);
        *(HVX_Vector *) (o0 + c) = do0;
        *(HVX_Vector *) (o1 + c) = do1;
        *(HVX_Vector *) (o2 + c) = do2;
        *(HVX_Vector *) (o3 + c) = do3;
    }
}

static __attribute__((noinline)) void fa_w8a8_o_accum_quad_update(
        const uint8_t * restrict tile_base,
        const float * restrict   dv_scales,
        float * restrict         o0,
        float * restrict         o1,
        float * restrict         o2,
        float * restrict         o3,
        size_t                   DV,
        uint32_t                 dp0,
        uint32_t                 dp1,
        uint32_t                 dp2,
        uint32_t                 dp3,
        uint32_t                 em0,
        uint32_t                 em1,
        uint32_t                 em2,
        uint32_t                 em3) {
    const HVX_Vector sign_80 = Q6_Vb_vsplat_R(0x80);
    #pragma clang loop pipeline(disable) unroll(disable)
    for (size_t c = 0; c < DV; c += 32) {
        HVX_Vector do0, do1, do2, do3;
        hmx_fa_w8a8_eval_s_tile_quad(tile_base, dv_scales, c, dp0, dp1, dp2, dp3, sign_80,
                                     &do0, &do1, &do2, &do3);
        *(HVX_Vector *) (o0 + c) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(hvx_qf32_mul_sf_scalar(*(const HVX_Vector *) (o0 + c), em0), do0));
        *(HVX_Vector *) (o1 + c) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(hvx_qf32_mul_sf_scalar(*(const HVX_Vector *) (o1 + c), em1), do1));
        *(HVX_Vector *) (o2 + c) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(hvx_qf32_mul_sf_scalar(*(const HVX_Vector *) (o2 + c), em2), do2));
        *(HVX_Vector *) (o3 + c) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vsf(hvx_qf32_mul_sf_scalar(*(const HVX_Vector *) (o3 + c), em3), do3));
    }
}

static void fa_w8a8_o_accum_thread(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    fa_w8a8_o_accum_args_t * args  = (fa_w8a8_o_accum_args_t *) data;
    struct hmx_fa_context *  factx = args->factx;

    const size_t n_rows_g    = args->n_rows_g;
    const size_t n_quads     = hmx_ceil_div(n_rows_g, 4);
    const size_t quads_per_t = hmx_ceil_div(n_quads, n);
    const size_t q_start_idx = (size_t) i * quads_per_t;
    const size_t q_end_idx   = hex_smin(q_start_idx + quads_per_t, n_quads);

    if (q_start_idx >= n_quads) {
        return;
    }

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (args->q_start * factx->G + q_start_idx * 4));

    const size_t    DV        = factx->DV;
    const size_t    n_v_tiles = DV / HMX_W8A8_TILE_N_COLS;
    const uint8_t * do_tiles  = factx->vtcm_do_tiles;
    const float *   dv_scales = factx->vtcm_v_scales[args->buf_idx];
    const float *   p_scales  = factx->vtcm_p_scales[args->buf_idx];
    const float *   exp_mdiff = factx->vtcm_exp_mdiff[args->buf_idx];
    float *         o_f32     = factx->vtcm_o_f32;
    const bool      is_first  = args->is_first;

    for (size_t quad = q_start_idx; quad < q_end_idx; ++quad) {
        const size_t   r         = quad * 4;
        const size_t   r_vec_idx = r >> 6;
        const size_t   r_tile    = r >> 5;
        const uint32_t r_vec_off = (uint32_t) (r & 63);
        const uint32_t rg        = r_vec_off >> 2;
        const int      off       = (int) ((r & 31u) * 4u);

        const HVX_Vector v_dp = hvx_vmem(p_scales + r_tile * 32);
        const uint32_t   dp0  = (uint32_t) Q6_R_vextract_VR(v_dp, off + 0);
        const uint32_t   dp1  = (uint32_t) Q6_R_vextract_VR(v_dp, off + 4);
        const uint32_t   dp2  = (uint32_t) Q6_R_vextract_VR(v_dp, off + 8);
        const uint32_t   dp3  = (uint32_t) Q6_R_vextract_VR(v_dp, off + 12);

        const uint8_t * tile_base = do_tiles + r_vec_idx * n_v_tiles * 2 * HMX_W8A8_ACT_TILE_SIZE + (size_t) rg * 128;
        float * o0 = o_f32 + (r + 0) * DV;
        float * o1 = o_f32 + (r + 1) * DV;
        float * o2 = o_f32 + (r + 2) * DV;
        float * o3 = o_f32 + (r + 3) * DV;

        if (is_first) {
            fa_w8a8_o_accum_quad_first(tile_base, dv_scales, o0, o1, o2, o3, DV, dp0, dp1, dp2, dp3);
        } else {
            const HVX_Vector v_em = hvx_vmem(exp_mdiff + r_tile * 32);
            const uint32_t   em0  = (uint32_t) Q6_R_vextract_VR(v_em, off + 0);
            const uint32_t   em1  = (uint32_t) Q6_R_vextract_VR(v_em, off + 4);
            const uint32_t   em2  = (uint32_t) Q6_R_vextract_VR(v_em, off + 8);
            const uint32_t   em3  = (uint32_t) Q6_R_vextract_VR(v_em, off + 12);
            fa_w8a8_o_accum_quad_update(tile_base, dv_scales, o0, o1, o2, o3, DV, dp0, dp1, dp2, dp3, em0, em1, em2, em3);
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (args->q_start * factx->G + q_start_idx * 4));
}

static void fa_phase_w8a8_o_accum(struct hmx_fa_context * factx,
                                  size_t                  buf_idx,
                                  size_t                  n_rows_g,
                                  uint32_t                q_start,
                                  bool                    is_first) {
    work_queue_t wp    = factx->octx->ctx->hvx_queue;
    const size_t n_q   = hmx_ceil_div(n_rows_g, 4);
    const uint32_t n   = (uint32_t) hex_smin((size_t) factx->n_hvx, n_q);
    fa_w8a8_o_accum_args_t args = { factx, buf_idx, n_rows_g, q_start, is_first };
    work_queue_run(wp, fa_w8a8_o_accum_thread, &args, n);
}

static void fa_w8a8_o_store_thread(const htp_thread_t * th, unsigned int n, unsigned int i, void * data) {
    fa_o_store_args_t *     args  = (fa_o_store_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const size_t n_rows     = args->n_rows;
    const size_t rows_per_t = args->rows_per_t;
    const size_t start      = (size_t) i * rows_per_t;
    const size_t end        = hex_smin(start + rows_per_t, n_rows);

    if (start >= n_rows) {
        return;
    }

    const size_t G  = factx->G;
    const size_t DV = factx->DV;

    struct htp_thread_trace * tr = th->trace;
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (args->q_start * G + start));

    const uint32_t   kv_head      = args->kv_head;
    float *          o_f32        = factx->vtcm_o_f32;
    const bool       has_sinks    = (factx->sinks != NULL);
    const bool       is_f32       = factx->is_dst_fp32;
    const size_t     head_base    = (size_t) kv_head * G;
    const HVX_Vector v_ln2        = Q6_V_vsplat_R(EXP_LOGN2);
    const HVX_Vector v_one        = hvx_vec_splat_f32(1.0f);
    const HVX_Vector v_head_sinks = (has_sinks && G <= 32) ? hvx_fa_load_sinks(factx->vtcm_sinks, head_base, G) : Q6_V_vzero();

    size_t     prev_r_tile = SIZE_MAX;
    HVX_Vector v_sc        = Q6_V_vzero();

    for (size_t r = start; r < end; ++r) {
        const size_t r_tile = r >> 5;
        if (r_tile != prev_r_tile) {
            HVX_Vector v_l = factx->vtcm_l_vec[r_tile];
            if (has_sinks) {
                const size_t g0    = fastmodulo(r_tile * 32, G, &factx->div_G);
                const size_t rem_g = G - g0;
                HVX_Vector   v_sink;
                if (G <= 32) {
                    const HVX_Vector v_lo = Q6_V_vror_VR(v_head_sinks, g0 * sizeof(float));
                    const HVX_Vector v_hi = Q6_V_vror_VR(v_head_sinks, 128 - rem_g * sizeof(float));
                    v_sink = Q6_V_vmux_QVV(Q6_Q_vsetq2_R(rem_g * sizeof(float)), v_lo, v_hi);
                    for (size_t len = G; len < 32; len <<= 1) {
                        const HVX_VectorPred q_keep    = Q6_Q_vsetq_R(len * sizeof(float));
                        const HVX_Vector     v_shifted = Q6_V_vror_VR(v_sink, 128 - len * sizeof(float));
                        v_sink = Q6_V_vmux_QVV(q_keep, v_sink, v_shifted);
                    }
                } else if (rem_g >= 32) {
                    v_sink = hvx_fa_load_sinks(factx->vtcm_sinks, head_base + g0, 32);
                } else {
                    const HVX_Vector v_lo = hvx_fa_load_sinks(factx->vtcm_sinks, head_base + g0, rem_g);
                    const HVX_Vector v_hi = hvx_fa_load_sinks(factx->vtcm_sinks, head_base, 32 - rem_g);
                    v_sink = Q6_V_vmux_QVV(Q6_Q_vsetq_R(rem_g * sizeof(float)), v_lo, Q6_V_vror_VR(v_hi, 128 - rem_g * sizeof(float)));
                }

                HVX_Vector     v_me    = HVX_OP_MUL_F32(factx->vtcm_m_vec[r_tile], v_ln2);
                HVX_VectorPred q_gt    = Q6_Q_vcmp_gt_VsfVsf(v_sink, v_me);
                HVX_Vector     v_diff  = Q6_V_vmux_QVV(q_gt, HVX_OP_SUB_F32(v_me, v_sink), HVX_OP_SUB_F32(v_sink, v_me));
                HVX_Vector     v_exp   = hvx_vec_exp_f32(v_diff);
                HVX_Vector     v_l_new = Q6_V_vmux_QVV(q_gt, HVX_OP_ADD_F32(HVX_OP_MUL_F32(v_l, v_exp), v_one), HVX_OP_ADD_F32(v_l, v_exp));
                HVX_Vector     v_num   = Q6_V_vmux_QVV(q_gt, v_exp, v_one);
                HVX_VectorPred q_nz    = Q6_Q_vcmp_gt_VsfVsf(v_l_new, Q6_V_vzero());
                v_sc                   = Q6_V_vmux_QVV(q_nz, HVX_OP_MUL_F32(v_num, hvx_vec_inverse_f32(v_l_new)), Q6_V_vzero());
            } else {
                HVX_VectorPred q_nz = Q6_Q_vcmp_gt_VsfVsf(v_l, Q6_V_vzero());
                v_sc                = Q6_V_vmux_QVV(q_nz, hvx_vec_inverse_f32(v_l), Q6_V_vzero());
            }
            prev_r_tile = r_tile;
        }

        const uint32_t sc_bits = (uint32_t) Q6_R_vextract_VR(v_sc, (int) ((r & 31u) * 4u));
        float *        o_row   = o_f32 + r * DV;

        // Normalize in-place in VTCM; caller writes out via bypass DMA.
        if (is_f32) {
            for (size_t c = 0; c < DV; c += 32) {
                *(HVX_Vector *) (o_row + c) = Q6_Vsf_equals_Vqf32(hvx_qf32_mul_sf_scalar(*(const HVX_Vector *) (o_row + c), sc_bits));
            }
        } else {
            __fp16 * out = (__fp16 *) o_row;
            size_t   c   = 0;
            for (; c + 64 <= DV; c += 64) {
                HVX_Vector v0 = Q6_Vsf_equals_Vqf32(hvx_qf32_mul_sf_scalar(*(const HVX_Vector *) (o_row + c + 0),  sc_bits));
                HVX_Vector v1 = Q6_Vsf_equals_Vqf32(hvx_qf32_mul_sf_scalar(*(const HVX_Vector *) (o_row + c + 32), sc_bits));
                *(HVX_Vector *) (out + c) = hvx_vec_f32_to_f16(v0, v1);
            }
            if (c < DV) {
                HVX_Vector v0 = Q6_Vsf_equals_Vqf32(hvx_qf32_mul_sf_scalar(*(const HVX_Vector *) (o_row + c), sc_bits));
                hvx_vec_store_a(out + c, 32 * sizeof(__fp16), hvx_vec_f32_to_f16(v0, Q6_V_vzero()));
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (args->q_start * G + start));
}

static void fa_phase_w8a8_o_store(struct hmx_fa_context *   factx,
                                  dma_queue *               dma_q,
                                  const struct htp_tensor * dst,
                                  uint32_t                  q_start,
                                  uint32_t                  kv_head,
                                  uint32_t                  ib3,
                                  size_t                    n_rows_g) {
    work_queue_t wp = factx->octx->ctx->hvx_queue;
    uint32_t     n  = 1;
    if (factx->n_hvx > 1 && n_rows_g >= (size_t) (factx->n_hvx * 2)) {
        n = factx->n_hvx;
    }
    const size_t      rows_per_t = hmx_ceil_div(n_rows_g, n);
    fa_o_store_args_t args       = { factx, (uint8_t *) factx->vtcm_o_f32, NULL, q_start, kv_head, 0, n_rows_g, rows_per_t };
    work_queue_run(wp, fa_w8a8_o_store_thread, &args, n);

    const size_t vtcm_row_stride = factx->DV * sizeof(float);
    const size_t row_bytes       = factx->DV * (factx->is_dst_fp32 ? sizeof(float) : sizeof(__fp16));
    fa_push_o_dma(dma_q, dst, (const uint8_t *) factx->vtcm_o_f32, vtcm_row_stride, row_bytes,
                  q_start, kv_head, ib3, factx->G, &factx->div_G, 0, n_rows_g);
}

// ============================================================================
// HMX job structs and worker functions
// ============================================================================

typedef struct {
    const __fp16 * q_tiles;
    const __fp16 * k_tiles;
    __fp16 *       s_tiles;
    size_t         n_row_tiles;
    size_t         n_col_tiles;
    size_t         n_dot_tiles;  // DK / 32
    size_t         n_tiles_per_bc;
    uint8_t *      hmx_scales;
} hmx_fa_qk_job_t;

static void hmx_fa_qk_dot_worker(const htp_thread_t * th, void * data) {
    hmx_fa_qk_job_t * job            = (hmx_fa_qk_job_t *) data;
    const size_t      n_row_tiles    = job->n_row_tiles;
    const size_t      n_col_tiles    = job->n_col_tiles;
    const size_t      n_dot_tiles    = job->n_dot_tiles;
    const size_t      n_tiles_per_bc = job->n_tiles_per_bc;
    const __fp16 * restrict q_tiles  = job->q_tiles;
    const __fp16 * restrict k_tiles  = job->k_tiles;
    __fp16 * restrict s_tiles        = job->s_tiles;
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);

    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int)job->hmx_scales));
    const size_t dot_stride = n_dot_tiles * HMX_FP16_TILE_N_ELMS;
    for (size_t r = 0; r < n_row_tiles; ++r) {
        const __fp16 * row_tiles = q_tiles + r * dot_stride;
        const __fp16 * col_tiles = k_tiles;
        __fp16 *       out_tile  = s_tiles + r * n_tiles_per_bc * HMX_FP16_TILE_N_ELMS;

        for (size_t c = 0; c < n_col_tiles; ++c) {
            hmx_fa_qk_dot_tile(row_tiles, col_tiles, out_tile, n_dot_tiles);
            col_tiles += dot_stride;
            out_tile  += HMX_FP16_TILE_N_ELMS;
        }
    }
}

typedef struct {
    __fp16 *       o_curr;
    const __fp16 * o_prev;
    const __fp16 * p_tiles;
    const __fp16 * v_tiles;
    const __fp16 * d_tiles;
    uint8_t *      hmx_scales;
    size_t         n_row_tiles;
    size_t         n_col_tiles;
    size_t         n_row_tiles_g_br;
    size_t         n_tiles_per_bc;
    size_t         DV;
} hmx_fa_o_update_job_t;

static void hmx_fa_o_update_worker(const htp_thread_t * th, void * data) {
    hmx_fa_o_update_job_t * job              = (hmx_fa_o_update_job_t *) data;
    const size_t            n_row_tiles      = job->n_row_tiles;
    const size_t            n_col_tiles      = job->n_col_tiles;
    const size_t            n_row_tiles_g_br = job->n_row_tiles_g_br;
    const size_t            n_tiles_per_bc   = job->n_tiles_per_bc;
    const size_t            DV_tiles         = job->DV / 32;
    const __fp16 * restrict d_tiles          = job->d_tiles;
    const __fp16 * restrict p_tiles          = job->p_tiles;
    const __fp16 * restrict v_tiles          = job->v_tiles;
    const __fp16 * restrict o_prev           = job->o_prev;
    __fp16 * restrict o_curr                 = job->o_curr;
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(DV_tiles > 0);

    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int)job->hmx_scales));
    const size_t o_stride = n_row_tiles_g_br * HMX_FP16_TILE_N_ELMS;
    const size_t v_stride = n_tiles_per_bc * HMX_FP16_TILE_N_ELMS;
    for (size_t r = 0; r < n_row_tiles; ++r) {
        const __fp16 * d_diag     = d_tiles + r * HMX_FP16_TILE_N_ELMS;
        const __fp16 * p_tile_in  = p_tiles + (r * n_tiles_per_bc) * HMX_FP16_TILE_N_ELMS;
        const __fp16 * o_rc       = o_prev + r * HMX_FP16_TILE_N_ELMS;
        const __fp16 * v_tile_in  = v_tiles;
        __fp16       * o_tile_out = o_curr + r * HMX_FP16_TILE_N_ELMS;

        for (size_t c = 0; c < DV_tiles; ++c) {
            hmx_fa_o_update_tile(d_diag, o_rc, p_tile_in, v_tile_in, o_tile_out, n_col_tiles);
            o_rc       += o_stride;
            v_tile_in  += v_stride;
            o_tile_out += o_stride;
        }
    }
}

typedef struct {
    __fp16 *       o_curr;   // output (row-major tile layout)
    const __fp16 * o_prev;   // input (column-major tile layout)
    const __fp16 * d_tiles;  // diag(1/l) tiles
    uint8_t *      hmx_scales;
    size_t         n_row_tiles;
    size_t         n_row_tiles_g_br;
    size_t         DV;
} hmx_fa_o_norm_job_t;

static void hmx_fa_o_norm_worker(const htp_thread_t * th, void * data) {
    hmx_fa_o_norm_job_t * job              = (hmx_fa_o_norm_job_t *) data;
    const size_t          n_row_tiles      = job->n_row_tiles;
    const size_t          n_row_tiles_g_br = job->n_row_tiles_g_br;
    const size_t          DV_tiles         = job->DV / 32;
    const __fp16 * restrict d_tiles        = job->d_tiles;
    const __fp16 * restrict o_prev         = job->o_prev;
    __fp16 * restrict o_curr               = job->o_curr;
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(DV_tiles > 0);

    asm volatile(HMX_SET_BIAS("%0") :: "r"((unsigned int)job->hmx_scales));
    const size_t o_stride = n_row_tiles_g_br * HMX_FP16_TILE_N_ELMS;
    for (size_t r = 0; r < n_row_tiles; ++r) {
        const __fp16 * d_diag = d_tiles + r * HMX_FP16_TILE_N_ELMS;
        const __fp16 * o_rc = o_prev + r * HMX_FP16_TILE_N_ELMS;
        __fp16 *       o_out = o_curr + r * DV_tiles * HMX_FP16_TILE_N_ELMS;

        for (size_t c = 0; c < DV_tiles; ++c) {
            hmx_fa_o_norm_tile(d_diag, o_rc, o_out);
            o_rc  += o_stride;
            o_out += HMX_FP16_TILE_N_ELMS;
        }
    }
}

// Populate per-GQA-row ALiBi slopes for a given KV head.
static __attribute__((noinline)) void fa_compute_slopes(
                              const struct hmx_fa_context * factx,
                              uint32_t                      kv_head,
                              size_t                        n_rows_g) {
    __fp16 * slopes = factx->vtcm_slopes;
    if (factx->max_bias == 0.0f) {
        hvx_splat_f16_a(slopes, 1.0f, n_rows_g);
        return;
    }

    const uint32_t G           = factx->G;
    const uint32_t n_head_log2 = factx->n_head_log2;
    const float    m0          = factx->m0;
    const float    m1          = factx->m1;

    __fp16 * temp_slopes = factx->ddr_slopes;
    if (G <= 32) {
        // Fast path: Compute G unique slope values in vector registers
        HVX_Vector v_val = hvx_alibi_slopes(kv_head, G, n_head_log2, m0, m1);

        __fp16 temp_slopes_aligned[64] __attribute__((aligned(128)));
        hvx_vmem(temp_slopes_aligned) = hvx_vec_f32_to_f16(v_val, Q6_V_vzero());

        for (uint32_t i = 0; i < G; ++i) {
            temp_slopes[i] = temp_slopes_aligned[i];
        }
    } else {
        // Fallback path: G > 32 (rare configurations)
        for (uint32_t i = 0; i < G; ++i) {
            temp_slopes[i] = (__fp16)alibi_slope(kv_head * G + i, n_head_log2, m0, m1);
        }
    }

    // Replicate the G values in DDR to avoid scalar writes to VTCM (which generates L2 misses)
    for (size_t r = G; r < n_rows_g; ++r) {
        temp_slopes[r] = temp_slopes[r - G];
    }

    // Copy to VTCM slopes using HVX block copy (both are aligned to 128 bytes)
    hvx_copy_f16_aa((uint8_t *)slopes, (const uint8_t *)temp_slopes, n_rows_g);
}

static inline void fa_push_q_dma(dma_queue *               dma_q,
                                 const struct htp_tensor * q,
                                 void *                    vtcm_q_dma,
                                 uint32_t                  q_start,
                                 uint32_t                  kv_head,
                                 uint32_t                  ib3,
                                 uint32_t                  G,
                                 uint32_t                  n_rows_q,
                                 size_t                    q_elem_row_bytes,
                                 bool                      q_transposed) {
    const dma_addr_t q_ptr      = q->data + q_start * q->nb[1] + (kv_head * G) * q->nb[2] + ib3 * q->nb[3];
    const size_t     q_inner_nb = q_transposed ? q->nb[1] : q->nb[2];
    const size_t     q_outer_nb = q_transposed ? q->nb[2] : q->nb[1];
    const size_t     n_inner    = q_transposed ? n_rows_q : G;
    const size_t     n_outer    = q_transposed ? G : n_rows_q;

    if (__builtin_expect(q_inner_nb == q_elem_row_bytes, true)) {
        const size_t row_bytes = n_inner * q_elem_row_bytes;
        dma_queue_push(dma_q, dma_make_data(vtcm_q_dma, q_ptr), row_bytes, hex_smax(q_outer_nb, row_bytes), row_bytes, n_outer);
        return;
    }

    // Non-contiguous inner dim: one 2D DMA per outer slice, keeping dma_q depth at 2 (prev O + Q).
    uint8_t *    dst_ptr          = (uint8_t *) vtcm_q_dma;
    const size_t dst_outer_stride = n_inner * q_elem_row_bytes;
    const size_t src_inner_stride = hex_smax(q_inner_nb, q_elem_row_bytes);
    for (size_t o = 0; o < n_outer; ++o) {
        if (!dma_queue_push(dma_q, dma_make_data(dst_ptr + o * dst_outer_stride, q_ptr + o * q_outer_nb),
                            q_elem_row_bytes, src_inner_stride, q_elem_row_bytes, n_inner)) {
            dma_queue_pop(dma_q);
            dma_queue_push(dma_q, dma_make_data(dst_ptr + o * dst_outer_stride, q_ptr + o * q_outer_nb),
                           q_elem_row_bytes, src_inner_stride, q_elem_row_bytes, n_inner);
        }
    }
    while (dma_queue_depth(dma_q) > 2) {
        dma_queue_pop(dma_q);
    }
}

static void fa_push_mask_dma_gqa(
    dma_queue *               dma_q,
    const struct htp_tensor * mask,
    uint32_t                  q_start,
    uint32_t                  im3,
    uint32_t                  kv_start,
    uint32_t                  kv_head,
    uint32_t                  G,
    uint32_t                  m_line_bytes,
    uint32_t                  kv_rows,
    uint32_t                  n_rows_q,
    struct hmx_fa_context *   factx
) {
    for (uint32_t g = 0; g < G; ++g) {
        const uint32_t h_idx = kv_head * G + g;
        const uint32_t im2 = fastmodulo(h_idx, mask->ne[2], &factx->src3_div2);
        const dma_addr_t ms_src = mask->data + q_start * mask->nb[1] +
                                  im2 * mask->nb[2] + im3 * mask->nb[3] + kv_start * sizeof(__fp16);
        uint8_t * ms_dst = (uint8_t *) factx->vtcm_mask_buf + g * m_line_bytes;
        dma_queue_push(dma_q, dma_make_data(ms_dst, ms_src), G * m_line_bytes, mask->nb[1], kv_rows * sizeof(__fp16), n_rows_q);
    }
}

static void fa_pop_mask_dma_gqa(dma_queue * dma_q, uint32_t G) {
    for (uint32_t g = 0; g < G; ++g) {
        dma_queue_pop(dma_q);
    }
}

static inline void fa_prefetch_block(dma_queue * dma_q, const struct htp_tensor * k, const struct htp_tensor * v, const struct htp_tensor * mask,
                                     uint32_t b, size_t Bc, size_t size_k_row_padded, size_t size_k_row, size_t size_v_row_padded, size_t size_v_row,
                                     uint32_t ik2, uint32_t ik3, uint32_t iv2, uint32_t iv3, uint32_t q_start, uint32_t im3, uint32_t kv_head, uint32_t G,
                                     size_t m_line_bytes, size_t n_rows_q, size_t nek1, size_t prefetch_buf, struct hmx_fa_context * factx) {
    const uint32_t  prefetch_start = b * Bc;
    const uint32_t  prefetch_rows  = hex_smin(Bc, nek1 - prefetch_start);
    const dma_addr_t k_prefetch_src = k->data + prefetch_start * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
    dma_queue_push(dma_q, dma_make_data(factx->vtcm_k_fp16[prefetch_buf], k_prefetch_src), size_k_row_padded, k->nb[1], size_k_row, prefetch_rows);
    const dma_addr_t v_prefetch_src = v->data + prefetch_start * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
    dma_queue_push(dma_q, dma_make_data(factx->vtcm_v_fp16[prefetch_buf], v_prefetch_src), size_v_row_padded, v->nb[1], size_v_row, prefetch_rows);

    if (mask) {
        if (__builtin_expect(factx->mask_broadcast, true)) {
            const dma_addr_t ms_src = mask->data + q_start * mask->nb[1] + im3 * mask->nb[3] + prefetch_start * sizeof(__fp16);
            dma_cache_fa_push(dma_q, factx->m_cache, ms_src, m_line_bytes, mask->nb[1], prefetch_rows * sizeof(__fp16), n_rows_q);
        } else {
            fa_push_mask_dma_gqa(dma_q, mask, q_start, im3, prefetch_start, kv_head, G, m_line_bytes, prefetch_rows, n_rows_q, factx);
        }
    }
}

// ============================================================================
// Core HMX flash attention algorithm (GQA-merged)
// ============================================================================

int hmx_flash_attn_ext(struct htp_ops_context * octx) {
    struct htp_thread_trace * tr_hvx = &octx->ctx->trace[0];
    const struct htp_tensor * q    = octx->src[0];
    const struct htp_tensor * k    = octx->src[1];
    const struct htp_tensor * v    = octx->src[2];
    const struct htp_tensor * mask = (octx->src[3] && octx->src[3]->data) ? octx->src[3] : NULL;
    const struct htp_tensor * dst  = octx->dst;

    struct htp_context * const ctx = octx->ctx;

    if (!ctx->hmx_enabled) {
        return HTP_STATUS_NO_SUPPORT;
    }

    // Dimensions
    const uint32_t neq0 = q->ne[0];  // head_dim (DK)
    const uint32_t neq1 = q->ne[1];  // n_tokens
    const uint32_t neq2 = q->ne[2];  // n_heads
    const uint32_t neq3 = q->ne[3];  // n_seqs

    const uint32_t nek1 = k->ne[1];  // kv_len

    const uint32_t nev0 = v->ne[0];  // head_dim (DV)

    const uint32_t DK = neq0;
    const uint32_t DV = nev0;

    // HMX tiles head_dim in units of 64. head_dim need not be 64- (or 32-) aligned:
    // we can operate on DK/DV rounded up to 64 with tail lanes [D, D_pad) zero-filled.
    const uint32_t DK_pad = hex_round_up(DK, 64);
    const uint32_t DV_pad = hex_round_up(DV, 64);
    if (DK == 0 || DV == 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_fa_kernel_params * kparams = (const struct htp_fa_kernel_params *) octx->kernel_params;
    const uint32_t n_kv_heads = k->ne[2];

    // ======== Build context ========
    struct hmx_fa_context factx;
    memset(&factx, 0, sizeof(factx));
    factx.octx           = octx;
    factx.sinks          = octx->src[4];  // NULL if this op has no attention sinks
    factx.k              = k;
    factx.v              = v;
    factx.n_hvx          = kparams->n_hvx;
    factx.DK             = DK;
    factx.DV             = DV;
    factx.DK_pad         = DK_pad;
    factx.DV_pad         = DV_pad;
    factx.n_kv           = nek1;
    factx.n_kv_heads     = n_kv_heads;
    factx.n_heads        = neq2;
    factx.G              = kparams->G;
    factx.div_G          = kparams->u.hmx.div_G;
    factx.neq1           = neq1;
    factx.Br             = kparams->Br;
    factx.Bc             = kparams->Bc;
    factx.g_br           = kparams->u.hmx.g_br;
    factx.n_kv_blocks    = kparams->n_kv_blocks;
    factx.is_q_fp32      = (q->type == HTP_TYPE_F32);
    factx.is_dst_fp32    = (dst->type == HTP_TYPE_F32);
    factx.pipeline       = (kparams->u.hmx.pipeline != 0);
    factx.is_w8a8        = (kparams->kernel_type == HTP_FA_KERNEL_HMX_W8A8);
    factx.mask_broadcast = (kparams->u.hmx.mask_broadcast != 0);
    if (mask) {
        factx.src3_div2  = kparams->src3_div2;
        factx.src3_div3  = kparams->src3_div3;
    }

    factx.has_softcap   = (kparams->logit_softcap != 0.0f);
    factx.scale         = (__fp16) kparams->scale;
    factx.scale_f32     = kparams->scale;
    factx.max_bias      = kparams->max_bias;
    factx.logit_softcap = 0;
    if (factx.has_softcap) {
        factx.logit_softcap = (__fp16) kparams->logit_softcap;
    }

    factx.n_head_log2 = kparams->n_head_log2;
    factx.m0          = kparams->m0;
    factx.m1          = kparams->m1;

    const uint32_t Br = factx.Br;
    const uint32_t Bc = factx.Bc;
    const uint32_t g_br = factx.g_br;
    const bool pipeline = factx.pipeline;
    const uint32_t n_hvx = factx.n_hvx;
    const uint32_t G = factx.G;

    // Multi-device: prefer head-parallel partitioning (each core owns a disjoint head
    // shard), falling back to Q-block (token) split when heads don't divide evenly.
    const uint32_t n_q_blocks = (neq1 + Br - 1) / Br;
    uint32_t q_start_min  = 0;
    uint32_t q_start_max  = neq1;
    uint32_t kv_head_min  = 0;
    uint32_t kv_head_max  = n_kv_heads;

    if (octx->ctx->mdev.count > 1) {
        const uint32_t mdev_count = octx->ctx->mdev.count;
        const uint32_t mdev_idx   = octx->ctx->mdev.idx;

        if (kparams->flags & HTP_FA_FLAG_TEAM_SPLIT) {
            // KV heads split across teams, Q-blocks split within a team
            uint32_t n_teams, team_size;
            htp_fa_mdev_teams(n_kv_heads, mdev_count, kparams->head_split != 0, &n_teams, &team_size);

            const uint32_t team        = mdev_idx / team_size;
            const uint32_t member      = mdev_idx - team * team_size;
            const uint32_t kv_per_team = n_kv_heads / n_teams;
            kv_head_min = team * kv_per_team;
            kv_head_max = kv_head_min + kv_per_team;

            const uint32_t base  = n_q_blocks / team_size;
            const uint32_t rem   = n_q_blocks - base * team_size;
            const uint32_t b0    = member * base + MIN(member, rem);
            const uint32_t nblk  = base + (member < rem ? 1 : 0);

            if (nblk == 0) {
                return HTP_STATUS_OK;
            }

            q_start_min = b0 * Br;
            q_start_max = MIN((b0 + nblk) * Br, neq1);
        } else if (kparams->head_split && n_kv_heads >= mdev_count && n_kv_heads % mdev_count == 0) {
            const uint32_t kv_per_core = n_kv_heads / mdev_count;
            kv_head_min = mdev_idx * kv_per_core;
            kv_head_max = kv_head_min + kv_per_core;
        } else {
            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(n_q_blocks, 1, mdev_idx, mdev_count, &octx->ctx->mdev.count_div);
            const uint32_t block_start = range.start;
            const uint32_t block_end   = range.start + range.count;

            if (block_start >= block_end) {
                return HTP_STATUS_OK;
            }

            q_start_min = block_start * Br;
            q_start_max = MIN(block_end * Br, neq1);
        }
    }

    // ======== VTCM allocation (GQA-aware) ========
    // K/V row sizes drive the DMA descriptors (not the VTCM layout) and are used
    // throughout the KV loop below. The DMA copies only the real DK/DV columns; the
    // staging rows are padded to hold DK_pad/DV_pad columns (tail zero-filled below)
    // so the HMX interleave/tile logic can operate on 64-aligned head dims.
    // Build the VTCM layout once (shared with the host estimator) and place every
    // scratch buffer at its computed offset. Padded head dims size the HMX tiles.
    const struct hmx_fa_layout_params lp = {
        .kernel_type    = kparams->kernel_type,
        .q_type         = q->type,
        .gqa_factor     = G,
        .DK             = DK_pad,
        .DV             = DV_pad,
        .Br             = Br,
        .Bc             = Bc,
        .n_hvx          = n_hvx,
        .n_heads        = factx.n_heads,
        .pipeline       = pipeline,
        .has_mask       = mask != NULL,
        .mask_broadcast = factx.mask_broadcast,
        .has_sinks      = factx.sinks != NULL,
    };
    struct hmx_fa_vtcm_layout L;
    hmx_fa_vtcm_layout_build(&L, &lp);

    if (L.total_bytes > ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    const size_t size_k_row        = htp_tensor_get_row_size(k->type, DK);
    const size_t size_v_row        = htp_tensor_get_row_size(v->type, DV);
    const size_t size_k_row_padded = factx.is_w8a8 ? L.k_raw_row_padded : hex_round_up(DK_pad * sizeof(__fp16), 128);
    const size_t size_v_row_padded = factx.is_w8a8 ? L.v_raw_row_padded : hex_round_up(DV_pad * sizeof(__fp16), 128);

    uint8_t * const base = ctx->vtcm_base;

    factx.vtcm_q_dma          = VTCM_LAYOUT_PTR(__fp16, base, L.off_q_dma);
    factx.vtcm_q_tiles        = VTCM_LAYOUT_PTR(__fp16, base, L.off_q_tiles);
    factx.vtcm_o_tiles[0]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_o_tiles[0], !factx.is_w8a8);
    factx.vtcm_o_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_o_tiles[1], !factx.is_w8a8);
    factx.vtcm_k_fp16[0]      = VTCM_LAYOUT_PTR(__fp16, base, L.off_k_fp16[0]);
    factx.vtcm_k_fp16[1]      = VTCM_LAYOUT_PTR(__fp16, base, L.off_k_fp16[1]);
    factx.vtcm_v_fp16[0]      = VTCM_LAYOUT_PTR(__fp16, base, L.off_v_fp16[0]);
    factx.vtcm_v_fp16[1]      = VTCM_LAYOUT_PTR(__fp16, base, L.off_v_fp16[1]);
    factx.vtcm_k_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, base, L.off_k_tiles[0]);
    factx.vtcm_k_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_k_tiles[1], pipeline);
    factx.vtcm_v_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, base, L.off_v_tiles[0]);
    factx.vtcm_v_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_v_tiles[1], pipeline);
    factx.vtcm_s_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, base, L.off_s_tiles[0]);
    factx.vtcm_s_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_s_tiles[1], pipeline);
    factx.vtcm_p_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, base, L.off_p_tiles[0]);
    factx.vtcm_p_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_p_tiles[1], pipeline);
    factx.vtcm_d_tiles[0]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_d_tiles[0], !factx.is_w8a8);
    factx.vtcm_d_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_d_tiles[1], !factx.is_w8a8 && pipeline);
    factx.vtcm_d_inv_l        = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_d_inv_l, !factx.is_w8a8);
    factx.vtcm_m_vec          = VTCM_LAYOUT_PTR(HVX_Vector, base, L.off_m_vec);
    factx.vtcm_l_vec          = VTCM_LAYOUT_PTR(HVX_Vector, base, L.off_l_vec);
    factx.vtcm_s_rowmax       = VTCM_LAYOUT_PTR(HVX_Vector, base, L.off_s_rowmax);
    factx.vtcm_p_rowsum       = VTCM_LAYOUT_PTR(HVX_Vector, base, L.off_p_rowsum);
    factx.vtcm_row_bufs       = VTCM_LAYOUT_PTR(HVX_Vector, base, L.off_row_bufs);
    factx.row_buf_stride      = L.row_buf_stride;
    factx.vtcm_hmx_scales_id  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_hmx_scales_id);
    factx.vtcm_hmx_scales_qk  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_hmx_scales_qk);
    factx.vtcm_mask_buf       = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_mask_buf, mask != NULL);
    factx.mask_buf_row_stride = L.mask_buf_row_stride;
    factx.q_tile_bytes        = L.q_tile_bytes;
    factx.k_tile_bytes        = L.k_tile_bytes;
    factx.o_tile_bytes        = L.o_tile_bytes;
    factx.col_vec_bytes       = L.col_vec_bytes;
    factx.d_tile_bytes        = L.d_tile_bytes;
    factx.o_f32_bytes         = L.o_f32_bytes;
    factx.vtcm_slopes         = VTCM_LAYOUT_PTR(__fp16, base, L.off_slopes);
    factx.vtcm_sinks          = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_sinks, factx.sinks != NULL);
    factx.vtcm_q_scales       = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_q_scales, factx.is_w8a8);
    factx.vtcm_k_scales[0]    = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_k_scales[0], factx.is_w8a8);
    factx.vtcm_k_scales[1]    = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_k_scales[1], factx.is_w8a8 && pipeline);
    factx.vtcm_k_bias[0]      = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, base, L.off_k_bias[0], factx.is_w8a8);
    factx.vtcm_k_bias[1]      = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, base, L.off_k_bias[1], factx.is_w8a8 && pipeline);
    factx.vtcm_o_f32          = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_o_f32, factx.is_w8a8);
    factx.vtcm_do_tiles       = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, base, L.off_do_tiles, factx.is_w8a8);
    factx.vtcm_v_scales[0]    = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_v_scales[0], factx.is_w8a8);
    factx.vtcm_v_scales[1]    = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_v_scales[1], factx.is_w8a8 && pipeline);
    factx.vtcm_v_bias[0]      = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, base, L.off_v_bias[0], factx.is_w8a8);
    factx.vtcm_v_bias[1]      = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, base, L.off_v_bias[1], factx.is_w8a8 && pipeline);
    factx.vtcm_p_scales[0]    = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_p_scales[0], factx.is_w8a8);
    factx.vtcm_p_scales[1]    = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_p_scales[1], factx.is_w8a8 && pipeline);
    factx.vtcm_exp_mdiff[0]   = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_exp_mdiff[0], factx.is_w8a8);
    factx.vtcm_exp_mdiff[1]   = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_exp_mdiff[1], factx.is_w8a8 && pipeline);
    factx.k_raw_row_padded    = L.k_raw_row_padded;
    factx.v_raw_row_padded    = L.v_raw_row_padded;

    const size_t m_line_bytes = L.m_line_bytes;  // used by the mask DMAs in the KV loop

    struct hmx_fa_ddr_spad * ddr = (struct hmx_fa_ddr_spad *) fa_get_ddr_spad(octx, sizeof(struct hmx_fa_ddr_spad) + factx.g_br * sizeof(__fp16));
    if (!ddr) {
        return HTP_STATUS_INTERNAL_ERR;
    }

    factx.ddr_slopes = ddr->slopes;
    factx.m_cache    = &octx->dma_cache_fa[0];
    if (mask && factx.mask_broadcast) {
        dma_cache_fa_init(factx.m_cache, (uint8_t *) factx.vtcm_mask_buf, L.m_buf_slot_bytes, HMX_FA_DMA_CACHE_SIZE);
    }

    // Head-dim padding: the K/V DMA staging buffers and the flat-Q buffer are laid out
    // with padded row strides (size_{k,v,q}_row_padded, covering D_pad columns) but the
    // DMA only writes the real D columns per row. Zero the whole staging buffers once up
    // front so tail lanes [D, D_pad) stay zero for all KV blocks. No-op when already aligned.
    if (DK_pad != DK || DV_pad != DV) {
        const size_t k_buf_bytes = (size_t) factx.Bc * size_k_row_padded;
        const size_t v_buf_bytes = (size_t) factx.Bc * size_v_row_padded;
        hvx_splat_u8_a((char *) factx.vtcm_k_fp16[0], 0, k_buf_bytes);
        hvx_splat_u8_a((char *) factx.vtcm_k_fp16[1], 0, k_buf_bytes);
        hvx_splat_u8_a((char *) factx.vtcm_v_fp16[0], 0, v_buf_bytes);
        hvx_splat_u8_a((char *) factx.vtcm_v_fp16[1], 0, v_buf_bytes);
        // Flat-Q DMA scratch
        const size_t q_dma_bytes = hex_align_up(factx.g_br * DK * (factx.is_q_fp32 ? sizeof(float) : sizeof(__fp16)), 128);
        hvx_splat_u8_a((char *) factx.vtcm_q_dma, 0, q_dma_bytes);
    }

    // ======== Initialize HMX output scales ========
    hmx_init_column_scales(factx.vtcm_hmx_scales_id, Q6_V_vsplat_R(0x3c00)); // 1.0
    {
        union { __fp16 f; uint16_t u; } sc = { .f = factx.scale };
        hmx_init_column_scales(factx.vtcm_hmx_scales_qk, Q6_V_vsplat_R(sc.u));
    }

    // ======== DMA setup ========
    dma_queue * const dma_q = ctx->dma[0];

    if (factx.sinks) {
        dma_queue_push(dma_q, dma_make_data(factx.vtcm_sinks, factx.sinks->data), L.sinks_bytes, 0, factx.sinks->size, 1);
        dma_queue_pop(dma_q);
    }

    const size_t n_row_tiles_g_br = g_br / HMX_FP16_TILE_N_ROWS;
    const size_t n_tiles_per_bc   = Bc / HMX_FP16_TILE_N_COLS;

    const size_t qo_element_size  = factx.is_q_fp32 ? sizeof(float) : sizeof(__fp16);
    const bool   q_transposed     = q->nb[1] < q->nb[2];
    const size_t q_elem_row_bytes = factx.DK * qo_element_size;

    // ======== Reusable job descriptors for pipeline ========
    hmx_fa_qk_job_t       qk_job;
    hmx_fa_w8a8_qk_job_t  w8a8_qk_job;
    hmx_fa_w8a8_pv_job_t  w8a8_pv_job;
    hmx_fa_o_update_job_t ou_job;
    hmx_fa_o_norm_job_t   on_job;

    // ======== Main loop ========
    for (uint32_t ib3 = 0; ib3 < neq3; ++ib3) {
        const uint32_t im3 = mask ? fastmodulo(ib3, mask->ne[3], &factx.src3_div3) : 0;
        for (uint32_t q_start = q_start_min; q_start < q_start_max; q_start += Br) {
            const uint32_t n_rows_q       = hex_smin(Br, neq1 - q_start);
            const size_t   n_rows_g       = n_rows_q * G;
            const size_t   g_br_actual    = hex_align_up(n_rows_g, HMX_FP16_TILE_N_ROWS);
            const size_t   n_row_tiles    = g_br_actual / HMX_FP16_TILE_N_ROWS;
            const uint32_t n_row_tiles_64 = (uint32_t) hmx_ceil_div(n_rows_g, HMX_W8A8_ACT_TILE_N_ROWS);

            uint16_t * active_blocks = ddr->active_blocks;
            uint32_t n_active_blocks = factx.n_kv_blocks;
            if (mask && factx.mask_broadcast && factx.n_kv_blocks > 1) {
                const size_t scratch_bytes = factx.k_tile_bytes * (factx.pipeline ? 2 : 1);
                n_active_blocks = fa_scan_active_blocks(
                    dma_q, mask, q_start, n_rows_q, 0, im3, nek1, Bc,
                    factx.n_kv_blocks, factx.vtcm_k_tiles[0], scratch_bytes, active_blocks);
            } else {
                #pragma clang loop vectorize(disable)
                for (uint32_t b = 0; b < factx.n_kv_blocks; ++b) {
                    active_blocks[b] = (uint16_t) b;
                }
            }
            const uint32_t b0       = (n_active_blocks > 0) ? active_blocks[0] : 0;
            const uint32_t b0_start = b0 * Bc;
            const uint32_t b0_rows  = hex_smin(Bc, nek1 - b0_start);

            // Dummy O writeback so every kv_head iteration pops prev O then curr Q unconditionally.
            dma_queue_push(dma_q, dma_make_data(dst->data, factx.vtcm_o_tiles[0]), 0, 0, 0, 0);

            for (uint32_t kv_head = kv_head_min; kv_head < kv_head_max; ++kv_head) {
                const uint32_t ik2 = kv_head;
                const uint32_t ik3 = fastdiv(ib3, &kparams->broadcast_rk3);
                const uint32_t iv2 = kv_head;
                const uint32_t iv3 = fastdiv(ib3, &kparams->broadcast_rv3);

                // 1. Push Q and KV DMAs for the first kv_head of this q_start block.
                // Subsequent kv_head iterations are enqueued early at the end of the previous iteration.
                if (kv_head == kv_head_min) {
                    fa_push_q_dma(dma_q, q, factx.vtcm_q_dma, q_start, kv_head, ib3, factx.G, n_rows_q, q_elem_row_bytes, q_transposed);

                    if (n_active_blocks > 0) {
                        const dma_addr_t k_src = k->data + b0_start * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
                        dma_queue_push(dma_q, dma_make_data(factx.vtcm_k_fp16[0], k_src), size_k_row_padded, k->nb[1], size_k_row, b0_rows);

                        const dma_addr_t v_src = v->data + b0_start * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
                        dma_queue_push(dma_q, dma_make_data(factx.vtcm_v_fp16[0], v_src), size_v_row_padded, v->nb[1], size_v_row, b0_rows);

                        if (factx.pipeline && mask) {
                            if (__builtin_expect(factx.mask_broadcast, true)) {
                                const dma_addr_t ms_src = mask->data + q_start * mask->nb[1] + im3 * mask->nb[3] + b0_start * sizeof(__fp16);
                                dma_cache_fa_push(dma_q, factx.m_cache, ms_src, m_line_bytes, mask->nb[1], b0_rows * sizeof(__fp16), n_rows_q);
                            } else {
                                fa_push_mask_dma_gqa(dma_q, mask, q_start, im3, b0_start, kv_head, G, m_line_bytes, b0_rows, n_rows_q, &factx);
                            }
                        }
                    }
                }

                // 2. Pop previous kv_head's O writeback and current Q DMA.
                dma_queue_pop(dma_q);
                dma_queue_pop(dma_q);

                // ---- Load Q block & Initialize per-block state ----
                fa_phase_q_load(&factx, q, q_start, n_rows_g);

                __fp16 * o_tile_prev = factx.vtcm_o_tiles[0];
                __fp16 * o_tile_curr = factx.vtcm_o_tiles[1];

                // ---- KV block loop with DMA double-buffering ----
                size_t buf_idx = 0;

                htp_trace_event_start(tr_hvx, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) q_start);
                fa_compute_slopes(&factx, kv_head, n_rows_g);
                htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) q_start);

                const size_t k_src_stride = size_k_row_padded / sizeof(__fp16);
                const size_t v_src_stride = size_v_row_padded / sizeof(__fp16);

                hmx_queue_t hmx_q = ctx->hmx_queue;

                if (factx.pipeline) {
                    // Double-buffered job structs because HMX queue runs asynchronously
                    hmx_fa_qk_job_t qk_job[2];
                    hmx_fa_w8a8_qk_job_t w8a8_qk_job[2];
                    hmx_fa_w8a8_pv_job_t w8a8_pv_job[2];
                    hmx_fa_o_update_job_t ou_job[2];

                    // Prefetch second active block early if there are multiple active blocks
                    if (n_active_blocks > 1) {
                        fa_prefetch_block(dma_q, k, v, mask, active_blocks[1], Bc, size_k_row_padded, size_k_row, size_v_row_padded, size_v_row,
                                          ik2, ik3, iv2, iv3, q_start, im3, kv_head, G, m_line_bytes, n_rows_q, nek1, 1, &factx);
                    }

                    // Prep and start QK-dot(0)
                    void * curr_k0 = (void *) dma_queue_pop(dma_q).dst;
                    fa_phase_k_interleave(&factx, b0_rows, k_src_stride, curr_k0, b0_start, 0);

                    if (factx.is_w8a8) {
                        w8a8_qk_job[0].q_tiles        = (const uint8_t *) factx.vtcm_q_tiles;
                        w8a8_qk_job[0].k_tiles        = (const uint8_t *) factx.vtcm_k_tiles[0];
                        w8a8_qk_job[0].k_bias         = factx.vtcm_k_bias[0];
                        w8a8_qk_job[0].s_tiles        = (uint8_t *) factx.vtcm_s_tiles[0];
                        w8a8_qk_job[0].n_row_tiles_64 = n_row_tiles_64;
                        w8a8_qk_job[0].n_col_tiles    = (uint32_t) hmx_ceil_div(b0_rows, HMX_W8A8_TILE_N_COLS);
                        w8a8_qk_job[0].n_dot_tiles    = DK / 32;
                        hmx_queue_push(hmx_q, hmx_queue_make_desc(hmx_fa_w8a8_qk_dot_worker, &w8a8_qk_job[0]));
                    } else {
                        qk_job[0].q_tiles        = factx.vtcm_q_tiles;
                        qk_job[0].k_tiles        = factx.vtcm_k_tiles[0];
                        qk_job[0].s_tiles        = factx.vtcm_s_tiles[0];
                        qk_job[0].n_row_tiles    = n_row_tiles;
                        qk_job[0].n_col_tiles    = hmx_ceil_div(b0_rows, HMX_FP16_TILE_N_COLS);
                        qk_job[0].n_dot_tiles    = DK_pad / 32;
                        qk_job[0].n_tiles_per_bc = n_tiles_per_bc;
                        qk_job[0].hmx_scales     = factx.vtcm_hmx_scales_qk;
                        hmx_queue_push(hmx_q, hmx_queue_make_desc(hmx_fa_qk_dot_worker, &qk_job[0]));
                    }

                    for (uint32_t step = 0; step < n_active_blocks; ++step) {
                        const uint32_t kv_blk      = active_blocks[step];
                        const uint32_t kv_start    = kv_blk * Bc;
                        const uint32_t kv_rows     = hex_smin(Bc, nek1 - kv_start);
                        const size_t   n_col_tiles = hmx_ceil_div(kv_rows, HMX_FP16_TILE_N_COLS);

                        // ---- 1. Pop and run V-prep for current block ----
                        void * curr_v = (void *) dma_queue_pop(dma_q).dst;
                        fa_phase_v_interleave(&factx, kv_rows, v_src_stride, curr_v, factx.vtcm_v_tiles[buf_idx], n_tiles_per_bc, kv_start, buf_idx);

                        // ---- 2. Pop and run mask-prep for current block ----
                        __fp16 * current_mask_vtcm = NULL;
                        if (mask) {
                            if (__builtin_expect(factx.mask_broadcast, true)) {
                                current_mask_vtcm = (__fp16 *) dma_queue_pop(dma_q).dst;
                            } else {
                                fa_pop_mask_dma_gqa(dma_q, G);
                                current_mask_vtcm = factx.vtcm_mask_buf;
                            }
                        }

                        // ---- 3. Start HMX O update / PV for step - 1 (reads P[1 - buf_idx], V[1 - buf_idx]) ----
                        // MUST be pushed before the next block's QK-dot: hmx_queue_pop() retires the
                        // oldest descriptor, so push order alone decides which pop waits for which job.
                        if (step > 0) {
                            const uint32_t prev_blk = active_blocks[step - 1];
                            const size_t   prev_buf = 1 - buf_idx;
                            const size_t   prev_ct  = hmx_ceil_div(hex_smin(Bc, nek1 - prev_blk * Bc), HMX_FP16_TILE_N_COLS);
                            if (factx.is_w8a8) {
                                w8a8_pv_job[prev_buf].p_tiles        = (const uint8_t *) factx.vtcm_p_tiles[prev_buf];
                                w8a8_pv_job[prev_buf].v_tiles        = (const uint8_t *) factx.vtcm_v_tiles[prev_buf];
                                w8a8_pv_job[prev_buf].v_bias         = factx.vtcm_v_bias[prev_buf];
                                w8a8_pv_job[prev_buf].do_tiles       = factx.vtcm_do_tiles;
                                w8a8_pv_job[prev_buf].n_row_tiles_64 = n_row_tiles_64;
                                w8a8_pv_job[prev_buf].n_v_tiles      = DV / HMX_W8A8_TILE_N_COLS;
                                w8a8_pv_job[prev_buf].n_dot_tiles    = (uint32_t) prev_ct;
                                hmx_queue_push(hmx_q, hmx_queue_make_desc(hmx_fa_w8a8_pv_worker, &w8a8_pv_job[prev_buf]));
                            } else {
                                ou_job[prev_buf].o_curr           = o_tile_curr;
                                ou_job[prev_buf].o_prev           = o_tile_prev;
                                ou_job[prev_buf].p_tiles          = factx.vtcm_p_tiles[prev_buf];
                                ou_job[prev_buf].v_tiles          = factx.vtcm_v_tiles[prev_buf];
                                ou_job[prev_buf].d_tiles          = factx.vtcm_d_tiles[prev_buf];
                                ou_job[prev_buf].hmx_scales       = factx.vtcm_hmx_scales_id;
                                ou_job[prev_buf].n_row_tiles      = n_row_tiles;
                                ou_job[prev_buf].n_col_tiles      = prev_ct;
                                ou_job[prev_buf].n_row_tiles_g_br = n_row_tiles_g_br;
                                ou_job[prev_buf].n_tiles_per_bc   = n_tiles_per_bc;
                                ou_job[prev_buf].DV               = DV_pad;
                                hmx_queue_push(hmx_q, hmx_queue_make_desc(hmx_fa_o_update_worker, &ou_job[prev_buf]));
                            }
                        }

                        // ---- 4. Pop and run K-prep for next active block & push next QK-dot ----
                        if (step + 1 < n_active_blocks) {
                            const uint32_t next_blk   = active_blocks[step + 1];
                            const uint32_t next_start = next_blk * Bc;
                            const uint32_t next_rows  = hex_smin(Bc, nek1 - next_start);
                            const size_t   next_buf   = 1 - buf_idx;

                            void * next_k = (void *) dma_queue_pop(dma_q).dst;
                            fa_phase_k_interleave(&factx, next_rows, k_src_stride, next_k, next_start, next_buf);

                            if (factx.is_w8a8) {
                                w8a8_qk_job[next_buf].q_tiles        = (const uint8_t *) factx.vtcm_q_tiles;
                                w8a8_qk_job[next_buf].k_tiles        = (const uint8_t *) factx.vtcm_k_tiles[next_buf];
                                w8a8_qk_job[next_buf].k_bias         = factx.vtcm_k_bias[next_buf];
                                w8a8_qk_job[next_buf].s_tiles        = (uint8_t *) factx.vtcm_s_tiles[next_buf];
                                w8a8_qk_job[next_buf].n_row_tiles_64 = n_row_tiles_64;
                                w8a8_qk_job[next_buf].n_col_tiles    = (uint32_t) hmx_ceil_div(next_rows, HMX_W8A8_TILE_N_COLS);
                                w8a8_qk_job[next_buf].n_dot_tiles    = DK / 32;
                                hmx_queue_push(hmx_q, hmx_queue_make_desc(hmx_fa_w8a8_qk_dot_worker, &w8a8_qk_job[next_buf]));
                            } else {
                                qk_job[next_buf].q_tiles        = factx.vtcm_q_tiles;
                                qk_job[next_buf].k_tiles        = factx.vtcm_k_tiles[next_buf];
                                qk_job[next_buf].s_tiles        = factx.vtcm_s_tiles[next_buf];
                                qk_job[next_buf].n_row_tiles    = n_row_tiles;
                                qk_job[next_buf].n_col_tiles    = hmx_ceil_div(next_rows, HMX_FP16_TILE_N_COLS);
                                qk_job[next_buf].n_dot_tiles    = DK_pad / 32;
                                qk_job[next_buf].n_tiles_per_bc = n_tiles_per_bc;
                                qk_job[next_buf].hmx_scales     = factx.vtcm_hmx_scales_qk;
                                hmx_queue_push(hmx_q, hmx_queue_make_desc(hmx_fa_qk_dot_worker, &qk_job[next_buf]));
                            }
                        }

                        // ---- 5. Wait for current block's QK-dot to finish ----
                        hmx_queue_pop(hmx_q);

                        // ---- 6. Phase 2: softmax + build_D ----
                        fa_softmax_args_t sargs;
                        memset(&sargs, 0, sizeof(sargs));
                        sargs.factx                = &factx;
                        sargs.buf_idx              = buf_idx;
                        sargs.kv_rows              = kv_rows;
                        sargs.n_rows_g             = n_rows_g;
                        sargs.n_col_tiles          = n_col_tiles;
                        sargs.n_tiles_per_bc       = n_tiles_per_bc;
                        sargs.n_row_tiles          = n_row_tiles;
                        sargs.n_row_tiles_g_br     = n_row_tiles_g_br;
                        sargs.Bc                   = Bc;
                        sargs.G                    = G;
                        sargs.kv_head              = kv_head;
                        sargs.kv_start             = kv_start;
                        sargs.q_start              = q_start;
                        sargs.ib3                  = ib3;
                        sargs.has_alibi            = (factx.max_bias != 0.0f);
                        sargs.mask                 = mask;
                        sargs.mask_vtcm            = current_mask_vtcm;
                        sargs.mask_vtcm_row_stride = factx.mask_buf_row_stride;
                        sargs.slopes               = factx.vtcm_slopes;

                        // Run Softmax on HVX (blocking call)
                        fa_phase_softmax_and_build_d(&factx, &sargs, n_row_tiles, n_row_tiles_g_br);

                        // Wait for HMX O update / PV for step - 1 to finish
                        if (step > 0) {
                            hmx_queue_pop(hmx_q);
                            if (factx.is_w8a8) {
                                fa_phase_w8a8_o_accum(&factx, 1 - buf_idx, n_rows_g, q_start, step == 1);
                            } else {
                                hex_swap_ptr((void **) &o_tile_curr, (void **) &o_tile_prev);
                            }
                        }

                        // Prefetch block step + 2
                        if (step + 2 < n_active_blocks) {
                            fa_prefetch_block(dma_q, k, v, mask, active_blocks[step + 2], Bc, size_k_row_padded, size_k_row, size_v_row_padded, size_v_row,
                                              ik2, ik3, iv2, iv3, q_start, im3, kv_head, G, m_line_bytes, n_rows_q, nek1, buf_idx, &factx);
                        }

                        buf_idx = 1 - buf_idx;
                    }

                    // Epilogue
                    if (n_active_blocks > 0) {
                        const uint32_t last_blk  = active_blocks[n_active_blocks - 1];
                        const size_t   last_cols = hmx_ceil_div(hex_smin(Bc, nek1 - last_blk * Bc), HMX_FP16_TILE_N_COLS);
                        const size_t   last_buf  = 1 - buf_idx;
                        if (factx.is_w8a8) {
                            w8a8_pv_job[0].p_tiles        = (const uint8_t *) factx.vtcm_p_tiles[last_buf];
                            w8a8_pv_job[0].v_tiles        = (const uint8_t *) factx.vtcm_v_tiles[last_buf];
                            w8a8_pv_job[0].v_bias         = factx.vtcm_v_bias[last_buf];
                            w8a8_pv_job[0].do_tiles       = factx.vtcm_do_tiles;
                            w8a8_pv_job[0].n_row_tiles_64 = n_row_tiles_64;
                            w8a8_pv_job[0].n_v_tiles      = DV / HMX_W8A8_TILE_N_COLS;
                            w8a8_pv_job[0].n_dot_tiles    = (uint32_t) last_cols;
                            hmx_queue_push(hmx_q, hmx_queue_make_desc(hmx_fa_w8a8_pv_worker, &w8a8_pv_job[0]));
                            hmx_queue_pop(hmx_q);
                            fa_phase_w8a8_o_accum(&factx, last_buf, n_rows_g, q_start, n_active_blocks == 1);
                        } else {
                            ou_job[0].o_curr           = o_tile_curr;
                            ou_job[0].o_prev           = o_tile_prev;
                            ou_job[0].p_tiles          = factx.vtcm_p_tiles[last_buf];
                            ou_job[0].v_tiles          = factx.vtcm_v_tiles[last_buf];
                            ou_job[0].d_tiles          = factx.vtcm_d_tiles[last_buf];
                            ou_job[0].hmx_scales       = factx.vtcm_hmx_scales_id;
                            ou_job[0].n_row_tiles      = n_row_tiles;
                            ou_job[0].n_col_tiles      = last_cols;
                            ou_job[0].n_row_tiles_g_br = n_row_tiles_g_br;
                            ou_job[0].n_tiles_per_bc   = n_tiles_per_bc;
                            ou_job[0].DV               = DV_pad;
                            hmx_queue_push(hmx_q, hmx_queue_make_desc(hmx_fa_o_update_worker, &ou_job[0]));

                            // Overlapped: run HVX build diag inv L while HMX is busy executing the update
                            htp_trace_event_start(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_start);
                            fa_build_d_diag_inv_l(&factx, n_row_tiles, n_row_tiles_g_br, kv_head, n_rows_g);
                            htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_start);
                            hmx_queue_pop(hmx_q);

                            hex_swap_ptr((void **) &o_tile_curr, (void **) &o_tile_prev);
                        }
                    }

                } else {
                    // Fallback path
                    for (uint32_t step = 0; step < n_active_blocks; ++step) {
                        const uint32_t kv_blk      = active_blocks[step];
                        const uint32_t kv_start    = kv_blk * Bc;
                        const uint32_t kv_rows     = hex_smin(Bc, nek1 - kv_start);
                        const size_t   n_col_tiles = hmx_ceil_div(kv_rows, HMX_FP16_TILE_N_COLS);

                        if (mask) {
                            if (__builtin_expect(factx.mask_broadcast, true)) {
                                const dma_addr_t ms_src = mask->data + q_start * mask->nb[1] + im3 * mask->nb[3] + kv_start * sizeof(__fp16);
                                dma_cache_fa_push(dma_q, factx.m_cache, ms_src, m_line_bytes, mask->nb[1], kv_rows * sizeof(__fp16), n_rows_q);
                            } else {
                                fa_push_mask_dma_gqa(dma_q, mask, q_start, im3, kv_start, kv_head, G, m_line_bytes, kv_rows, n_rows_q, &factx);
                            }
                        }

                        if (step + 1 < n_active_blocks) {
                            const uint32_t  prefetch_blk   = active_blocks[step + 1];
                            const uint32_t  prefetch_start = prefetch_blk * Bc;
                            const uint32_t  prefetch_rows  = hex_smin(Bc, nek1 - prefetch_start);
                            const size_t    prefetch_buf   = 1 - buf_idx;
                            const dma_addr_t k_prefetch_src = k->data + prefetch_start * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
                            dma_queue_push(dma_q, dma_make_data(factx.vtcm_k_fp16[prefetch_buf], k_prefetch_src), size_k_row_padded, k->nb[1], size_k_row, prefetch_rows);
                            const dma_addr_t v_prefetch_src = v->data + prefetch_start * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
                            dma_queue_push(dma_q, dma_make_data(factx.vtcm_v_fp16[prefetch_buf], v_prefetch_src), size_v_row_padded, v->nb[1], size_v_row, prefetch_rows);
                        }

                        // Wait for current K DMA and interleave
                        void * curr_k = (void *) dma_queue_pop(dma_q).dst;
                        fa_phase_k_interleave(&factx, kv_rows, k_src_stride, curr_k, kv_start, 0);

                        if (factx.is_w8a8) {
                            w8a8_qk_job.q_tiles        = (const uint8_t *) factx.vtcm_q_tiles;
                            w8a8_qk_job.k_tiles        = (const uint8_t *) factx.vtcm_k_tiles[0];
                            w8a8_qk_job.k_bias         = factx.vtcm_k_bias[0];
                            w8a8_qk_job.s_tiles        = (uint8_t *) factx.vtcm_s_tiles[0];
                            w8a8_qk_job.n_row_tiles_64 = n_row_tiles_64;
                            w8a8_qk_job.n_col_tiles    = (uint32_t) n_col_tiles;
                            w8a8_qk_job.n_dot_tiles    = DK / 32;

                            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_fa_w8a8_qk_dot_worker, &w8a8_qk_job));
                            hmx_queue_pop(ctx->hmx_queue);
                        } else {
                            qk_job.q_tiles        = factx.vtcm_q_tiles;
                            qk_job.k_tiles        = factx.vtcm_k_tiles[0];
                            qk_job.s_tiles        = factx.vtcm_s_tiles[0];
                            qk_job.n_row_tiles    = n_row_tiles;
                            qk_job.n_col_tiles    = n_col_tiles;
                            qk_job.n_dot_tiles    = (size_t) (DK_pad / 32);
                            qk_job.n_tiles_per_bc = n_tiles_per_bc;
                            qk_job.hmx_scales     = factx.vtcm_hmx_scales_qk;

                            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_fa_qk_dot_worker, &qk_job));
                            hmx_queue_pop(ctx->hmx_queue);
                        }

                        // Wait for current V DMA and interleave
                        void * curr_v = (void *) dma_queue_pop(dma_q).dst;
                        fa_phase_v_interleave(&factx, kv_rows, v_src_stride, curr_v, factx.vtcm_v_tiles[0], n_tiles_per_bc, kv_start, 0);

                        // ---- Phase 3: softmax + build_D ----
                        __fp16 * current_mask_vtcm = NULL;
                        if (mask) {
                            if (__builtin_expect(factx.mask_broadcast, true)) {
                                current_mask_vtcm = (__fp16 *) dma_queue_pop(dma_q).dst;
                            } else {
                                fa_pop_mask_dma_gqa(dma_q, G);
                                current_mask_vtcm = factx.vtcm_mask_buf;
                            }
                        }

                        fa_softmax_args_t sargs;
                        memset(&sargs, 0, sizeof(sargs));
                        sargs.factx                = &factx;
                        sargs.kv_rows              = kv_rows;
                        sargs.n_rows_g             = n_rows_g;
                        sargs.n_col_tiles          = n_col_tiles;
                        sargs.n_tiles_per_bc       = n_tiles_per_bc;
                        sargs.n_row_tiles          = n_row_tiles;
                        sargs.n_row_tiles_g_br     = n_row_tiles_g_br;
                        sargs.Bc                   = Bc;
                        sargs.G                    = G;
                        sargs.kv_head              = kv_head;
                        sargs.kv_start             = kv_start;
                        sargs.q_start              = q_start;
                        sargs.ib3                  = ib3;
                        sargs.has_alibi            = (factx.max_bias != 0.0f);
                        sargs.mask                 = mask;
                        sargs.mask_vtcm            = current_mask_vtcm;
                        sargs.mask_vtcm_row_stride = factx.mask_buf_row_stride;
                        sargs.slopes               = factx.vtcm_slopes;
                        fa_phase_softmax_and_build_d(&factx, &sargs, n_row_tiles, n_row_tiles_g_br);

                        if (factx.is_w8a8) {
                            w8a8_pv_job.p_tiles        = (const uint8_t *) factx.vtcm_p_tiles[0];
                            w8a8_pv_job.v_tiles        = (const uint8_t *) factx.vtcm_v_tiles[0];
                            w8a8_pv_job.v_bias         = factx.vtcm_v_bias[0];
                            w8a8_pv_job.do_tiles       = factx.vtcm_do_tiles;
                            w8a8_pv_job.n_row_tiles_64 = n_row_tiles_64;
                            w8a8_pv_job.n_v_tiles      = DV / HMX_W8A8_TILE_N_COLS;
                            w8a8_pv_job.n_dot_tiles    = (uint32_t) n_col_tiles;

                            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_fa_w8a8_pv_worker, &w8a8_pv_job));
                            hmx_queue_pop(ctx->hmx_queue);
                            fa_phase_w8a8_o_accum(&factx, 0, n_rows_g, q_start, step == 0);
                        } else {
                            ou_job.o_curr           = o_tile_curr;
                            ou_job.o_prev           = o_tile_prev;
                            ou_job.p_tiles          = factx.vtcm_p_tiles[0];
                            ou_job.v_tiles          = factx.vtcm_v_tiles[0];
                            ou_job.d_tiles          = factx.vtcm_d_tiles[0];
                            ou_job.hmx_scales       = factx.vtcm_hmx_scales_id;
                            ou_job.n_row_tiles      = n_row_tiles;
                            ou_job.n_col_tiles      = n_col_tiles;
                            ou_job.n_row_tiles_g_br = n_row_tiles_g_br;
                            ou_job.n_tiles_per_bc   = n_tiles_per_bc;
                            ou_job.DV               = DV_pad;

                            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_fa_o_update_worker, &ou_job));
                            if (step + 1 == n_active_blocks) {
                                // Overlapped: run HVX build diag inv L while HMX is busy executing the update
                                htp_trace_event_start(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_start);
                                fa_build_d_diag_inv_l(&factx, n_row_tiles, n_row_tiles_g_br, kv_head, n_rows_g);
                                htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_start);
                            }
                            hmx_queue_pop(ctx->hmx_queue);

                            hex_swap_ptr((void **) &o_tile_curr, (void **) &o_tile_prev);
                        }

                        buf_idx = 1 - buf_idx;
                    }
                }

                // ---- Final normalization & Store O block (via bypass DMA) ----
                if (factx.is_w8a8) {
                    fa_phase_w8a8_o_store(&factx, dma_q, dst, q_start, kv_head, ib3, n_rows_g);
                } else {
                    on_job.o_curr           = o_tile_curr;
                    on_job.o_prev           = o_tile_prev;
                    on_job.d_tiles          = factx.vtcm_d_inv_l;
                    on_job.hmx_scales       = factx.vtcm_hmx_scales_id;
                    on_job.n_row_tiles      = n_row_tiles;
                    on_job.n_row_tiles_g_br = n_row_tiles_g_br;
                    on_job.DV               = DV_pad;
                    hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_fa_o_norm_worker, &on_job));
                    hmx_queue_pop(ctx->hmx_queue);

                    fa_phase_o_store(&factx, dma_q, dst, o_tile_curr, (uint8_t *) o_tile_prev, q_start, kv_head, ib3, n_rows_g);
                }

                // Enqueue DMAs for the next kv_head behind O writeback on dma_q.
                const uint32_t next_kv_head = kv_head + 1;
                const bool     has_next     = (next_kv_head < kv_head_max);

                if (has_next) {
                    fa_push_q_dma(dma_q, q, factx.vtcm_q_dma, q_start, next_kv_head, ib3, factx.G, n_rows_q, q_elem_row_bytes, q_transposed);

                    if (n_active_blocks > 0) {
                        const uint32_t next_ik2 = next_kv_head;
                        const uint32_t next_iv2 = next_kv_head;

                        const dma_addr_t next_k_src = k->data + b0_start * k->nb[1] + next_ik2 * k->nb[2] + ik3 * k->nb[3];
                        dma_queue_push(dma_q, dma_make_data(factx.vtcm_k_fp16[0], next_k_src), size_k_row_padded, k->nb[1], size_k_row, b0_rows);

                        const dma_addr_t next_v_src = v->data + b0_start * v->nb[1] + next_iv2 * v->nb[2] + iv3 * v->nb[3];
                        dma_queue_push(dma_q, dma_make_data(factx.vtcm_v_fp16[0], next_v_src), size_v_row_padded, v->nb[1], size_v_row, b0_rows);

                        if (factx.pipeline && mask) {
                            if (__builtin_expect(factx.mask_broadcast, true)) {
                                const dma_addr_t ms_src = mask->data + q_start * mask->nb[1] + im3 * mask->nb[3] + b0_start * sizeof(__fp16);
                                dma_cache_fa_push(dma_q, factx.m_cache, ms_src, m_line_bytes, mask->nb[1], b0_rows * sizeof(__fp16), n_rows_q);
                            } else {
                                fa_push_mask_dma_gqa(dma_q, mask, q_start, im3, b0_start, next_kv_head, G, m_line_bytes, b0_rows, n_rows_q, &factx);
                            }
                        }
                    }
                }
            }

            dma_queue_pop(dma_q);
        }
    }

    return HTP_STATUS_OK;
}

int op_flash_attn_ext(struct htp_ops_context * octx) {
    const struct htp_tensor * q    = octx->src[0];
    const struct htp_tensor * k    = octx->src[1];
    const struct htp_tensor * v    = octx->src[2];
    const struct htp_tensor * mask = octx->src[3];

    // Check support
    if ((q->type != HTP_TYPE_F16 && q->type != HTP_TYPE_F32) ||
        (k->type != HTP_TYPE_F16 && k->type != HTP_TYPE_Q8_0) ||
        (v->type != HTP_TYPE_F16 && v->type != HTP_TYPE_Q8_0)) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_fa_kernel_params * kparams = (const struct htp_fa_kernel_params *) octx->kernel_params;

    if (kparams->kernel_type == HTP_FA_KERNEL_UNSUPPORTED || kparams->n_kv_blocks > FA_MAX_KV_BLOCKS) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (!htp_ops_context_set_n_hvx(octx, kparams->n_hvx)) {
        return HTP_STATUS_INVAL_PARAMS;
    }

    if (kparams->kernel_type == HTP_FA_KERNEL_HMX_F16 || kparams->kernel_type == HTP_FA_KERNEL_HMX_W8A8) {
        return hmx_flash_attn_ext(octx);
    }

    struct htp_fa_context factx;
    factx.octx = octx;
    factx.k = k;
    factx.v = v;

    factx.src0_div21 = kparams->u.hvx.src0_div21;
    factx.src0_div1  = kparams->u.hvx.src0_div1;

    factx.broadcast_rk2 = kparams->broadcast_rk2;
    factx.broadcast_rk3 = kparams->broadcast_rk3;
    factx.broadcast_rv2 = kparams->broadcast_rv2;
    factx.broadcast_rv3 = kparams->broadcast_rv3;

    if (mask) {
        factx.src3_div2 = kparams->src3_div2;
        factx.src3_div3 = kparams->src3_div3;
    }

    factx.is_q_fp32 = (q->type == HTP_TYPE_F32);
    factx.size_q_row_padded = kparams->u.hvx.size_q_row_padded;
    factx.size_k_row_padded = kparams->u.hvx.size_k_row_padded;
    factx.size_v_row_padded = kparams->u.hvx.size_v_row_padded;

    factx.n_blocks = kparams->n_kv_blocks;

    factx.scale = kparams->scale;
    factx.max_bias = kparams->max_bias;
    factx.has_softcap = (kparams->logit_softcap != 0.0f);
    factx.logit_softcap = 0;
    if (factx.has_softcap) {
        factx.logit_softcap = (__fp16) kparams->logit_softcap;
    }

    factx.n_head_log2 = kparams->n_head_log2;
    factx.m0          = kparams->m0;
    factx.m1          = kparams->m1;

    const uint32_t n_head = q->ne[2];
    if (n_head > 512) {
        return HTP_STATUS_NO_SUPPORT;
    }

    factx.ddr = (struct hvx_fa_ddr_spad *) fa_get_ddr_spad(octx, sizeof(struct hvx_fa_ddr_spad));
    if (!factx.ddr) {
        return HTP_STATUS_INTERNAL_ERR;
    }
    factx.m_cache = octx->dma_cache_dm;

    for (uint32_t h = 0; h < n_head; ++h) {
        factx.ddr->slopes[h] = (__fp16) ((kparams->max_bias > 0.0f) ? alibi_slope(h, factx.n_head_log2, factx.m0, factx.m1) : 1.0f);
    }

    // total rows in q
    const uint32_t neq1 = q->ne[1];
    const uint32_t neq2 = q->ne[2];
    const uint32_t neq3 = q->ne[3];
    const uint32_t total_qrows = neq1 * neq2 * neq3;

    uint32_t qrow_start    = 0;
    uint32_t qrows         = total_qrows;
    uint32_t kv_head_start = 0;
    uint32_t n_kv_local    = k->ne[2];

    if (octx->ctx->mdev.count > 1) {
        const uint32_t mdev_count = octx->ctx->mdev.count;
        const uint32_t mdev_idx   = octx->ctx->mdev.idx;
        const uint32_t n_kv_heads = k->ne[2];

        // head range is contiguous in flat row space only when neq3 == 1
        if (kparams->head_split && neq3 == 1 && n_kv_heads >= mdev_count && n_kv_heads % mdev_count == 0) {
            const uint32_t G              = kparams->G;
            const uint32_t kv_per_core    = n_kv_heads / mdev_count;
            const uint32_t heads_per_core = kv_per_core * G;
            const uint32_t head_start     = mdev_idx * heads_per_core;
            qrow_start    = head_start * neq1;
            qrows         = heads_per_core * neq1;
            kv_head_start = mdev_idx * kv_per_core;
            n_kv_local    = kv_per_core;
        } else {
            const struct htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_qrows, 1, mdev_idx, mdev_count, &octx->ctx->mdev.count_div);
            qrow_start = range.start;
            qrows      = range.count;
            n_kv_local = 0;
        }
    }

    if (qrows == 0) {
        return HTP_STATUS_OK;
    }

    const uint32_t n_hvx = octx->n_hvx;

    factx.qrows         = qrows;
    factx.qrow_start    = qrow_start;
    factx.qrows_per_hvx = fastdiv(qrows + n_hvx - 1, &octx->n_hvx_div);
    factx.kv_head_start = kv_head_start;
    factx.n_kv_local    = n_kv_local;
    factx.G             = kparams->G;
    factx.splits_per_kv = 1;

    const bool has_mask  = (mask != NULL);
    const bool has_sinks = (octx->src[4] != NULL);
    const struct hvx_fa_layout_params lp = {
        .q_type    = q->type,
        .DK        = k->ne[0],
        .DV        = v->ne[0],
        .n_heads   = n_head,
        .n_hvx     = n_hvx,
        .has_mask  = has_mask,
        .has_sinks = has_sinks,
    };
    struct hvx_fa_vtcm_layout L;
    hvx_fa_vtcm_layout_build(&L, &lp);

    if (L.total_bytes > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    factx.size_q_block = L.size_q_block;
    factx.size_k_block = L.size_k_block;
    factx.size_v_block = L.size_v_block;
    factx.size_m_block = L.size_m_block;
    factx.size_vkq_acc = L.size_vkq_acc;

    uint8_t * const base = octx->ctx->vtcm_base;

    factx.spad_q               = VTCM_LAYOUT_PTR(uint8_t, base, L.off_q);
    factx.spad_k               = VTCM_LAYOUT_PTR(uint8_t, base, L.off_k);
    factx.spad_v               = VTCM_LAYOUT_PTR(uint8_t, base, L.off_v);
    factx.spad_m               = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, base, L.off_m, has_mask);
    factx.spad_a               = VTCM_LAYOUT_PTR(uint8_t, base, L.off_a);
    factx.spad_sinks           = VTCM_LAYOUT_PTR_OPTIONAL(float, base, L.off_sinks, has_sinks);
    factx.n_active_blocks      = factx.n_blocks;
    factx.shared_active_blocks = false;

    if (has_mask && neq1 == 1 && neq3 == 1 && mask->ne[2] == 1 && factx.n_blocks > 1 && factx.n_blocks <= FA_MAX_KV_BLOCKS) {
        factx.n_active_blocks = fa_scan_active_blocks(
            octx->ctx->dma[0], mask, 0, 1, 0, 0, k->ne[1], FLASH_ATTN_BLOCK_SIZE,
            factx.n_blocks, factx.spad_k, factx.size_k_block * 2, factx.ddr->active_blocks);
        factx.shared_active_blocks = true;
    }

    if (has_sinks) {
        const struct htp_tensor * sinks = octx->src[4];
        dma_queue * dma_q = octx->ctx->dma[0];
        dma_queue_push(dma_q, dma_make_data(factx.spad_sinks, sinks->data), L.size_sinks, 0, sinks->size, 1);
        dma_queue_pop(dma_q);
    }

    const bool can_split_kv =
        (neq1 == 1) &&
        (neq3 == 1) &&
        (k->ne[2] == v->ne[2]) &&
        (!has_mask || mask->ne[2] == 1) &&
        (n_kv_local > 0 && n_kv_local < n_hvx) &&
        (factx.G > 0 && factx.G <= FA_HVX_G_MAX) &&
        (factx.n_active_blocks >= 4);

    if (can_split_kv) {
        const uint32_t max_splits = MIN(n_hvx / n_kv_local, factx.n_active_blocks / 2);
        if (max_splits >= 2) {
            factx.splits_per_kv = max_splits;
            const uint32_t n_workers = n_kv_local * max_splits;
            const uint32_t n_merge   = MIN(n_hvx, n_kv_local * factx.G);
            work_queue_run(octx->ctx->hvx_queue, flash_attn_ext_f16_split_kv_thread, &factx, n_workers);
            work_queue_run(octx->ctx->hvx_queue, flash_attn_ext_f16_split_merge_thread, &factx, n_merge);
            return HTP_STATUS_OK;
        }
    }

    work_queue_run(octx->ctx->hvx_queue, flash_attn_ext_f16_thread, &factx, n_hvx);

    return HTP_STATUS_OK;
}
