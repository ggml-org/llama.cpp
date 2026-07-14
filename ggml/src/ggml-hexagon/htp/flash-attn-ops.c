#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <assert.h>
#include <HAP_compute_res.h>
#include <HAP_farf.h>
#include <HAP_perf.h>
#include <math.h>
#include <stdbool.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "hex-dma.h"
#include "hex-fastdiv.h"
#include "hex-profile.h"
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

#include "flash-attn-ops.h"
#include "hvx-fa-kernels.h"
#include "hmx-fa-kernels.h"

// Must be multiple of 32
#define FLASH_ATTN_BLOCK_SIZE (32 * 2)

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
    __fp16 logit_softcap;

    uint32_t n_head_log2;
    float m0;
    float m1;
    __fp16 slopes[512];

    uint32_t n_blocks;

    size_t size_q_row_padded;
    size_t size_k_row_padded;
    size_t size_v_row_padded;

    size_t size_k_block;
    size_t size_v_block;
    size_t size_m_block;

    uint32_t qrows;
    uint32_t qrows_per_thread;

    bool is_q_fp32;

    size_t size_q_block;
    size_t size_vkq_acc;

    uint8_t * spad_q;
    uint8_t * spad_k;
    uint8_t * spad_v;
    uint8_t * spad_m;
    uint8_t * spad_a;

    uint64_t t_start;
};

struct hmx_fa_context {
    const struct htp_ops_context * octx;
    const struct htp_tensor *      sinks;  // attention sinks (src[4]), NULL if absent
    bool         pipeline;  // true when n_kv_blocks >= FA_MIN_KV_BLOCKS && n_threads >= 2
    uint32_t     n_threads;

    // Op parameters
    __fp16       scale;
    float        max_bias;
    __fp16       logit_softcap;
    uint32_t     n_head_log2;
    float        m0, m1;

    // Dimensions
    uint32_t     DK, DV;
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
    __fp16 *     vtcm_q_tiles;         // Q tile format [g_br, D]
    __fp16 *     vtcm_o_tiles[2];      // O ping-pong [g_br, D]
    __fp16 *     vtcm_k_fp16[2];       // K DMA double-buffer [Bc, D]
    __fp16 *     vtcm_v_fp16[2];       // V DMA double-buffer [Bc, D]
    __fp16 *     vtcm_k_tiles;         // K tiles (transposed)
    __fp16 *     vtcm_v_tiles[2];      // V tiles (column-major, double-buffered)
    __fp16 *     vtcm_s_tiles;         // S = QK^T [g_br, Bc]
    __fp16 *     vtcm_p_tiles;         // P = softmax(S) [g_br, Bc]
    __fp16 *     vtcm_d_tiles;         // Diagonal rescale [g_br, g_br]
    HVX_Vector * vtcm_m_vec;           // Row max [g_br]
    HVX_Vector * vtcm_l_vec;           // Row sum [g_br]
    HVX_Vector * vtcm_s_rowmax;        // Softmax intermediate [g_br]
    HVX_Vector * vtcm_p_rowsum;        // Softmax intermediate [g_br]
    HVX_Vector * vtcm_row_bufs;        // Per-thread softmax row scratch [n_threads][2][Bc/64]
    uint8_t *    vtcm_hmx_scales_id;   // HMX output scales (identity)
    uint8_t *    vtcm_hmx_scales_qk;   // HMX output scales (qk_scale)
    __fp16 *     vtcm_mask_buf;        // VTCM mask buffer [Br * m_line], DMA'd per KV block
    __fp16 *     vtcm_slopes;          // ALiBi slopes [g_br]
    size_t       row_buf_stride;       // HVX vectors per row buffer (Bc/64)
    size_t       mask_buf_row_stride;  // elements (__fp16) per row in mask buffer
    size_t       q_tile_bytes;
    size_t       o_tile_bytes;
    size_t       col_vec_bytes;
    size_t       d_tile_bytes;
    bool         mask_broadcast;       // true when mask->ne[2] == 1 (head-independent, single 2D DMA)
    dma_cache    m_cache;
};

static void flash_attn_ext_f16_thread(unsigned int nth, unsigned int ith, void * data) {
    struct htp_fa_context * factx = (struct htp_fa_context *) data;
    const struct htp_ops_context * octx = factx->octx;
    const struct htp_tensor * q     = octx->src[0];
    const struct htp_tensor * k     = octx->src[1];
    const struct htp_tensor * v     = octx->src[2];
    const struct htp_tensor * mask  = octx->src[3];
    const struct htp_tensor * sinks = octx->src[4];
    const struct htp_tensor * dst   = octx->dst;

    const uint32_t neq0 = q->ne[0];
    const uint32_t neq1 = q->ne[1];
    const uint32_t neq2 = q->ne[2];
    const uint32_t neq3 = q->ne[3];

    const uint32_t nek0 = k->ne[0];
    const uint32_t nek1 = k->ne[1];
    const uint32_t nek2 = k->ne[2];
    const uint32_t nek3 = k->ne[3];

    const uint32_t nev0 = v->ne[0];
    const uint32_t nev1 = v->ne[1];
    const uint32_t nev2 = v->ne[2];
    const uint32_t nev3 = v->ne[3];

    const uint32_t nbq1 = q->nb[1];
    const uint32_t nbq2 = q->nb[2];
    const uint32_t nbq3 = q->nb[3];

    const uint32_t nbk1 = k->nb[1];
    const uint32_t nbk2 = k->nb[2];
    const uint32_t nbk3 = k->nb[3];

    const uint32_t nbv1 = v->nb[1];
    const uint32_t nbv2 = v->nb[2];
    const uint32_t nbv3 = v->nb[3];

    const uint32_t ne1 = dst->ne[1];
    const uint32_t ne2 = dst->ne[2];
    const uint32_t ne3 = dst->ne[3];

    const uint32_t nb1 = dst->nb[1];
    const uint32_t nb2 = dst->nb[2];
    const uint32_t nb3 = dst->nb[3];

    // total rows in q
    const uint32_t nr = factx->qrows;
    const uint32_t dr = factx->qrows_per_thread;
    const uint32_t ir0 = dr * ith;
    const uint32_t ir1 = MIN(ir0 + dr, nr);

    if (ir0 >= ir1) return;

    struct htp_thread_trace * tr = &octx->ctx->trace[ith];

    dma_queue * dma = octx->ctx->dma[ith];

    const uint32_t DK = nek0;
    const uint32_t DV = nev0;

    const size_t size_q_row = DK * ((q->type == HTP_TYPE_F32) ? 4 : 2);
    const size_t size_k_row = DK * sizeof(__fp16);
    const size_t size_v_row = DV * sizeof(__fp16);

    // Scratchpad buffers for Q, K, V, Mask, and VKQ32 accumulator
    uint8_t * spad_q = factx->spad_q + factx->size_q_block * ith;
    uint8_t * spad_k = factx->spad_k + factx->size_k_block * 2 * ith;
    uint8_t * spad_v = factx->spad_v + factx->size_v_block * 2 * ith;
    uint8_t * spad_m = factx->spad_m + (mask ? factx->size_m_block * HVX_FA_DMA_CACHE_SIZE : 0) * ith;
    uint8_t * spad_a = factx->spad_a + factx->size_vkq_acc * ith;

    dma_cache m_cache;
    dma_cache_init(&m_cache, spad_m, factx->size_m_block, HVX_FA_DMA_CACHE_SIZE);

    for (uint32_t ir = ir0; ir < ir1; ++ir) {
        const uint32_t iq3 = fastdiv(ir, &factx->src0_div21);
        const uint32_t iq2 = fastdiv(ir - iq3*neq2*neq1, &factx->src0_div1);
        const uint32_t iq1 = (ir - iq3*neq2*neq1 - iq2 * neq1);

        const uint32_t ik3 = fastdiv(iq3, &factx->broadcast_rk3);
        const uint32_t ik2 = fastdiv(iq2, &factx->broadcast_rk2);

        const uint32_t iv3 = fastdiv(iq3, &factx->broadcast_rv3);
        const uint32_t iv2 = fastdiv(iq2, &factx->broadcast_rv2);

        // Fetch Q row
        const uint8_t * q_row_ptr = (const uint8_t *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3);
        dma_queue_push(dma, dma_make_ptr(spad_q, q_row_ptr), factx->size_q_row_padded, nbq1, size_q_row, 1);

        const __fp16 * mp_base = NULL;
        if (mask) {
            const uint32_t im2 = fastmodulo(iq2, mask->ne[2], &factx->src3_div2);
            const uint32_t im3 = fastmodulo(iq3, mask->ne[3], &factx->src3_div3);
            mp_base = (const __fp16 *) ((const uint8_t *) mask->data + iq1*mask->nb[1] + im2*mask->nb[2] + im3*mask->nb[3]);
        }

        // Prefetch first two blocks
        for (uint32_t ib = 0; ib < MIN(factx->n_blocks, 2); ++ib) {
            const uint32_t ic_start = ib * FLASH_ATTN_BLOCK_SIZE;
            const uint32_t current_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - ic_start);

            // K
            const uint8_t * k_src = (const uint8_t *) k->data + (ic_start*nbk1 + ik2*nbk2 + ik3*nbk3);
            uint8_t * k_dst = spad_k + (ib % 2) * factx->size_k_block;
            dma_queue_push(dma, dma_make_ptr(k_dst, k_src), factx->size_k_row_padded, nbk1, size_k_row, current_block_size);

            // V
            const uint8_t * v_src = (const uint8_t *) v->data + (ic_start*nbv1 + iv2*nbv2 + iv3*nbv3);
            uint8_t * v_dst = spad_v + (ib % 2) * factx->size_v_block;
            dma_queue_push(dma, dma_make_ptr(v_dst, v_src), factx->size_v_row_padded, nbv1, size_v_row, current_block_size);

            // Mask
            if (mask) {
                const uint8_t * m_src = (const uint8_t *) (mp_base + ic_start);
                // Mask is 1D contiguous for this row
                dma_cache_push(dma, &m_cache, m_src, current_block_size * 2, current_block_size * 2, current_block_size * 2, 1);
            }
        }

        const uint32_t h = iq2; // head index
        const __fp16 slope = factx->slopes[h];

        HVX_Vector S_vec = hvx_vec_splat_f32(0.0f);
        HVX_Vector M_vec = hvx_vec_splat_f32(HTP_FA_M_INITIAL_VAL);

        // Clear accumulator
        hvx_splat_f32_a(spad_a, 0, DV);
        float * VKQ32 = (float *) (spad_a + 0);

        uint8_t * q_ptr_vtcm = dma_queue_pop(dma).dst;
        if (factx->is_q_fp32) {
            hvx_copy_f16_f32_aa(q_ptr_vtcm, q_ptr_vtcm, DK);  // inplace convert f32 to f16
        }

        const HVX_Vector slope_vec = hvx_vec_splat_f16(slope);
        const HVX_Vector v_neg_inf = Q6_Vh_vsplat_R(0xfbff);
        for (uint32_t ib = 0; ib < factx->n_blocks; ++ib) {
            const uint32_t ic_start = ib * FLASH_ATTN_BLOCK_SIZE;
            const uint32_t current_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - ic_start);

            // Wait for DMA
            uint8_t * k_base = dma_queue_pop(dma).dst; // K
            uint8_t * v_base = dma_queue_pop(dma).dst; // V
            __fp16  * m_base = mask ? dma_queue_pop(dma).dst : NULL; // M

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_QK, ir);

            // Inner loop processing the block from VTCM
            // 1. Compute scores (64 elements FP16)
            HVX_Vector scores_f16 = Q6_V_vzero();
            if (current_block_size > 0) {
                HVX_Vector scores0 = hvx_dot_f16_f16_aa_rx32(q_ptr_vtcm, k_base, factx->size_k_row_padded, DK, factx->scale);
                HVX_Vector scores1 = (current_block_size > 32) ? hvx_dot_f16_f16_aa_rx32(q_ptr_vtcm, k_base + 32 * factx->size_k_row_padded, factx->size_k_row_padded, DK, factx->scale) : Q6_V_vzero();
                scores_f16 = hvx_vec_f32_to_f16(scores0, scores1);
            }

            // 2. Softcap (in FP16)
            if (factx->logit_softcap != 0.0f) {
                const HVX_Vector v_cap = hvx_vec_splat_f16(factx->logit_softcap);
                scores_f16 = hvx_vec_tanh_f16(scores_f16);
                scores_f16 = hvx_vec_mul_f16_f16(scores_f16, v_cap);
            }

            HVX_VectorPred q_tail_keep = Q6_Q_vsetq2_R(current_block_size * sizeof(__fp16));

            // 3. Mask (in FP16)
            if (mask) {
                HVX_Vector m_vals_f16 = *(const HVX_UVector *) m_base;
                HVX_Vector vinf = Q6_Vh_vsplat_R(0xFC00);
                HVX_Vector vmin = Q6_Vh_vsplat_R(0xFBFF);
                HVX_VectorPred is_inf = Q6_Q_vcmp_eq_VhVh(m_vals_f16, vinf);
                m_vals_f16 = Q6_V_vmux_QVV(is_inf, vmin, m_vals_f16);

                HVX_Vector m_scaled = hvx_vec_mul_f16_f16(m_vals_f16, slope_vec);
                scores_f16 = Q6_V_vmux_QVV(q_tail_keep, hvx_vec_add_f16_f16(scores_f16, m_scaled), v_neg_inf);
            } else {
                scores_f16 = Q6_V_vmux_QVV(q_tail_keep, scores_f16, v_neg_inf);
            }

            // Compute block max in FP16
            HVX_Vector v_max_f16 = hvx_vec_reduce_max_f16(scores_f16);
            HVX_Vector v_max     = Q6_V_lo_W(hvx_vec_f16_to_f32(v_max_f16)); // splat block max in FP32
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_QK, ir);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_SFM, ir);
            {
                const HVX_Vector v_log2e = hvx_vec_splat_f16(EXP_LOG2E_F);

                // 4. Online Softmax Update
                HVX_Vector M_new_vec = Q6_Vsf_vmax_VsfVsf(v_max, M_vec);
                HVX_Vector diff_vec  = HVX_OP_SUB_F32(M_vec, M_new_vec);

                HVX_Vector diff_f16   = hvx_vec_f32_to_f16(diff_vec, diff_vec);
                HVX_Vector diff_base2 = hvx_vec_mul_f16_f16(diff_f16, v_log2e);
                HVX_Vector ms_f16     = hvx_vec_exp2_f16(diff_base2);
                HVX_Vector ms_vec     = Q6_V_lo_W(hvx_vec_f16_to_f32(ms_f16));

                M_vec = M_new_vec;

                hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (const uint8_t *) VKQ32, DV, ms_vec);

                // Compute P = exp2((S - M) * log2(e)) in FP16
                HVX_Vector v_m_vec_f16 = hvx_vec_f32_to_f16(M_vec, M_vec);
                HVX_Vector v_s_minus_m = Q6_Vqf16_vsub_VhfVhf(scores_f16, v_m_vec_f16);

                HVX_Vector v_s_minus_m_base2 = hvx_vec_mul_f16_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m), v_log2e);

                HVX_Vector P = hvx_vec_exp2_f16(v_s_minus_m_base2);
                P = Q6_V_vmux_QVV(q_tail_keep, P, Q6_V_vzero());

                // Convert P to FP32 to update the running sum S_vec
                HVX_VectorPair P_pair = hvx_vec_f16_to_f32(P);
                HVX_Vector P0 = Q6_V_lo_W(P_pair);
                HVX_Vector P1 = Q6_V_hi_W(P_pair);
                HVX_Vector p_sum_vec = hvx_vec_reduce_sum_f32(HVX_OP_ADD_F32(P0, P1));

                S_vec = HVX_OP_ADD_F32(HVX_OP_MUL_F32(S_vec, ms_vec), p_sum_vec);

                // 5. Accumulate V (F16 * F16 -> F32 accumulator)
                __fp16 __attribute__((aligned(128))) p_arr[VLEN_FP16];
                hvx_vec_store_a(p_arr, 128, P);

                for (uint32_t j = 0; j < current_block_size; j += 2) {
                    if (j + 1 == current_block_size) {
                        if (p_arr[j] != 0.0f) {
                            const uint8_t * v_ptr = v_base + j * factx->size_v_row_padded;
                            hvx_mad_f32_f16_aa(VKQ32, v_ptr, (p_arr + j), DV);
                        }
                        break;
                    }

                    if (p_arr[j] == 0.0f && p_arr[j + 1] == 0.0f) {
                        continue;
                    }

                    const uint8_t * v_ptr = v_base + j * factx->size_v_row_padded;
                    hvx_mad_f32_f16_aa_rx2(VKQ32, v_ptr, v_ptr + factx->size_v_row_padded, (p_arr + j), (p_arr + j + 1), DV);
                }
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, ir);

            // Issue DMA for next+1 block (if exists)
            if (ib + 2 < factx->n_blocks) {
                const uint32_t next_ib = ib + 2;
                const uint32_t next_ic_start = next_ib * FLASH_ATTN_BLOCK_SIZE;
                const uint32_t next_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - next_ic_start);

                // K
                const uint8_t * k_src = (const uint8_t *) k->data + (next_ic_start*nbk1 + ik2*nbk2 + ik3*nbk3);
                dma_queue_push(dma, dma_make_ptr(k_base, k_src), factx->size_k_row_padded, nbk1, size_k_row, next_block_size);

                // V
                const uint8_t * v_src = (const uint8_t *) v->data + (next_ic_start*nbv1 + iv2*nbv2 + iv3*nbv3);
                dma_queue_push(dma, dma_make_ptr(v_base, v_src), factx->size_v_row_padded, nbv1, size_v_row, next_block_size);

                // Mask
                if (mask) {
                    const uint8_t * m_src = (const uint8_t *) (mp_base + next_ic_start);
                    dma_cache_push(dma, &m_cache, m_src, next_block_size * 2, next_block_size * 2, next_block_size * 2, 1);
                }
            }
        }

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, ir);
        // sinks
        float M = hvx_vec_get_f32(M_vec);
        float S = hvx_vec_get_f32(S_vec);

        if (sinks) {
            const float s = ((float *)((char *) sinks->data))[h];

            float vs = 1.0f;

            if (s > M) {
                HVX_Vector diff_vec = hvx_vec_splat_f32(M - s);
                HVX_Vector ms_vec   = hvx_vec_exp_f32(diff_vec);
                hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (const uint8_t *) VKQ32, DV, ms_vec);

                float ms = hvx_vec_get_f32(ms_vec);
                S = S * ms + vs;
            } else {
                HVX_Vector diff_vec = hvx_vec_splat_f32(s - M);
                vs = hvx_vec_get_f32(hvx_vec_exp_f32(diff_vec));
                S += vs;
            }
        }

        const float S_inv = S == 0.0f ? 0.0f : 1.0f/S;
        hvx_scale_f32_aa((uint8_t *) VKQ32, (const uint8_t *) VKQ32, DV, S_inv);

        // Store result
        // dst indices
        const uint32_t i1 = iq1;
        const uint32_t i2 = iq2;
        const uint32_t i3 = iq3;

        // dst is permuted: [DV, n_heads, n_tokens, n_seq]
        // head stride is nb[1], token stride is nb[2], batch stride is nb[3]
        uint8_t * dst_ptr = (uint8_t *) dst->data + i2 * dst->nb[1] + i1 * dst->nb[2] + i3 * dst->nb[3];

        if (dst->type == HTP_TYPE_F32) {
            hvx_copy_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
        } else if (dst->type == HTP_TYPE_F16) {
            hvx_copy_f16_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, ir);
    }
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
} fa_k_int_args_t;

static void fa_k_interleave_thread(unsigned int n, unsigned int i, void * data) {
    fa_k_int_args_t *       args  = (fa_k_int_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const uint32_t total_rows = args->kv_rows;
    const uint32_t rows_per_t = args->rows_per_t;
    const uint32_t start      = i * rows_per_t;
    const uint32_t end        = (uint32_t) hex_smin(start + rows_per_t, total_rows);

    if (start >= total_rows) {
        return;
    }

    struct htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, (uint16_t) (args->kv_start + start));
    hmx_interleave_rows_to_tiles(factx->vtcm_k_tiles, (const __fp16 *) args->curr_k, total_rows, factx->DK,
                             args->src_stride, start, end);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, (uint16_t) (args->kv_start + start));
}

static void fa_phase_k_interleave(struct hmx_fa_context * factx, uint32_t kv_rows, size_t src_stride, void * curr_k, uint32_t kv_start) {
    work_queue_t wp = factx->octx->ctx->work_queue;
    uint32_t n = 1;
    if (factx->n_threads > 1 && kv_rows >= factx->n_threads * 2) {
        n = factx->n_threads;
    }
    uint32_t rows_per_t = hex_align_up(hmx_ceil_div(kv_rows, n), 2);
    fa_k_int_args_t args = { factx, kv_rows, src_stride, curr_k, kv_start, rows_per_t };
    if (n > 1) {
        work_queue_run(wp, fa_k_interleave_thread, &args, n);
    } else {
        fa_k_interleave_thread(1, 0, &args);
    }
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
} fa_v_int_args_t;

static void fa_v_interleave_thread(unsigned int n, unsigned int i, void * data) {
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

    struct htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, (uint16_t) (args->kv_start + start));
    hmx_interleave_cols_to_tiles(v_tiles_dst, (const __fp16 *) args->v_src, total_rows, factx->DV,
                             args->src_stride, (uint32_t) args->n_col_tiles, start, end);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, (uint16_t) (args->kv_start + start));
}

static void fa_phase_v_interleave(struct hmx_fa_context * factx,
                                  uint32_t                kv_rows,
                                  size_t                  src_stride,
                                  void *                  v_src,
                                  void *                  v_tiles_dst,
                                  size_t                  n_col_tiles,
                                  uint32_t                kv_start) {
    work_queue_t wp = factx->octx->ctx->work_queue;
    uint32_t n = 1;
    if (factx->n_threads > 1 && kv_rows >= factx->n_threads * 2) {
        n = factx->n_threads;
    }
    uint32_t rows_per_t = hex_align_up(hmx_ceil_div(kv_rows, n), 2);
    fa_v_int_args_t args = { factx, kv_rows, src_stride, v_src, v_tiles_dst, n_col_tiles, kv_start, rows_per_t };
    if (n > 1) {
        work_queue_run(wp, fa_v_interleave_thread, &args, n);
    } else {
        fa_v_interleave_thread(1, 0, &args);
    }
}

typedef struct {
    struct hmx_fa_context *   factx;
    const struct htp_tensor * q;
    uint32_t                  q_start;
    uint32_t                  kv_head;
    uint32_t                  ib3;
    size_t                    n_rows_g;
    size_t                    rows_per_t;
    size_t                    n_rows_q;
    bool                      q_transposed;
    atomic_uint               barrier;
} fa_q_load_args_t;

static void fa_q_load_thread(unsigned int n, unsigned int i, void * data) {
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

    struct htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_Q_PREP, (uint16_t) (args->q_start * G + start));

    // Parallel initialization of per-block state
    {
        const uint32_t g_br = factx->g_br;
        const uint32_t DV   = factx->DV;

        const size_t col_vec_bytes = factx->col_vec_bytes;
        const size_t d_tile_bytes  = factx->d_tile_bytes;

        // Initialize vtcm_l_vec & vtcm_m_vec
        const size_t l_bytes_per_t = hex_align_up(col_vec_bytes / n, 128);
        const size_t l_start       = i * l_bytes_per_t;
        const size_t l_end         = hex_smin(l_start + l_bytes_per_t, col_vec_bytes);

        const size_t m_bytes_per_t = hex_align_up(col_vec_bytes / n, 128);
        const size_t m_start       = i * m_bytes_per_t;
        const size_t m_end         = hex_smin(m_start + m_bytes_per_t, col_vec_bytes);

        if (factx->sinks) {
            const float * sinks_data = (const float *) (uintptr_t) factx->sinks->data;
            float *       m_vec      = (float *) factx->vtcm_m_vec;
            const size_t  r_start    = l_start / sizeof(float);
            const size_t  r_end      = l_end / sizeof(float);
            const float   scale_factor = EXP_LOG2E_F;

            const HVX_Vector v_scale = hvx_vec_splat_f32(scale_factor);

            for (size_t r = r_start; r < r_end; r += 32) {
                HVX_VectorAlias local_m;
                for (size_t j = 0; j < 32; ++j) {
                    size_t curr_r = r + j;
                    if (curr_r < n_rows_g) {
                        const size_t h_idx = fastmodulo(curr_r, G, &factx->div_G);
                        const size_t head  = args->kv_head * G + h_idx;
                        local_m.fp32[j] = sinks_data[head];
                    } else {
                        local_m.fp32[j] = HTP_FA_M_INITIAL_VAL;
                    }
                }
                HVX_Vector v_scaled = HVX_OP_MUL_F32(local_m.v, v_scale);
                *(HVX_Vector *) (m_vec + r) = v_scaled;
            }
            if (l_start < col_vec_bytes) {
                hvx_splat_u8_a((char *) factx->vtcm_l_vec + l_start, 0, l_end - l_start);
            }
        } else {
            if (l_start < col_vec_bytes) {
                hvx_splat_u8_a((char *) factx->vtcm_l_vec + l_start, 0, l_end - l_start);
            }
            if (m_start < col_vec_bytes) {
                hvx_splat_f32_a((char *) factx->vtcm_m_vec + m_start, HTP_FA_M_INITIAL_VAL, (m_end - m_start) / sizeof(float));
            }
        }

        // Initialize vtcm_d_tiles to 0
        const size_t d_bytes_per_t = hex_align_up(d_tile_bytes / n, 128);
        const size_t d_start       = i * d_bytes_per_t;
        const size_t d_end         = hex_smin(d_start + d_bytes_per_t, d_tile_bytes);
        if (d_start < d_tile_bytes) {
            hvx_splat_u8_a((char *) factx->vtcm_d_tiles + d_start, 0, d_end - d_start);
        }
    }

    if (start < factx->g_br) {
        const struct htp_tensor * q       = args->q;
        const uint32_t            q_start = args->q_start;
        const uint32_t            kv_head = args->kv_head;
        const uint32_t            ib3     = args->ib3;

        assert(factx->DK == factx->DV);

        const size_t o_tile_bytes = factx->o_tile_bytes;
        const bool use_q_dma = (2 * o_tile_bytes >= factx->g_br * DK * (factx->is_q_fp32 ? 4 : 2));

        __fp16 * q_tiles = factx->vtcm_q_tiles;
        if (use_q_dma) {
            const size_t g_rows_end = hex_smin(end, n_rows_g);
            const uint32_t d_limit = factx->is_q_fp32 ? DK / 32 : DK / 64;

            uint8_t * q_flat  = (uint8_t *) factx->vtcm_o_tiles[0];
            if (factx->is_q_fp32) {
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
        } else {
            // Fallback: direct-from-DDR/L2 path
            hmx_fa_q_prep_fallback(q_tiles, q->data, q->nb[1], q->nb[2], q->nb[3],
                                   q_start, kv_head, ib3, start, end, n_rows_g, G, DK, factx->is_q_fp32, &factx->div_G);
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
        const uint32_t g_br = factx->g_br;
        const uint32_t DV   = factx->DV;
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

static void fa_phase_q_load(struct hmx_fa_context *   factx,
                            const struct htp_tensor * q,
                            uint32_t                  q_start,
                            uint32_t                  kv_head,
                            uint32_t                  ib3,
                            size_t                    n_rows_g) {
    work_queue_t wp = factx->octx->ctx->work_queue;
    uint32_t n = 1;
    if (factx->n_threads > 1 && n_rows_g >= (size_t) (factx->n_threads * 2)) {
        n = factx->n_threads;
    }
    size_t rows_per_t = hex_align_up(hmx_ceil_div(factx->g_br, n), 2);
    const uint32_t n_rows_q = hex_smin(factx->Br, factx->neq1 - q_start);
    fa_q_load_args_t args;
    args.factx = factx;
    args.q = q;
    args.q_start = q_start;
    args.kv_head = kv_head;
    args.ib3 = ib3;
    args.n_rows_g = n_rows_g;
    args.rows_per_t = rows_per_t;
    args.n_rows_q = n_rows_q;
    args.q_transposed = q->nb[1] < q->nb[2];
    atomic_init(&args.barrier, n);
    if (n > 1) {
        work_queue_run(wp, fa_q_load_thread, &args, n);
    } else {
        fa_q_load_thread(1, 0, &args);
    }
}

typedef struct {
    struct hmx_fa_context *   factx;
    const struct htp_tensor * dst;
    const __fp16 *            o_tile_src;
    uint32_t                  q_start;
    uint32_t                  kv_head;
    uint32_t                  ib3;
    size_t                    n_rows_g;
    size_t                    rows_per_t;
} fa_o_store_args_t;

static void fa_o_store_thread_f32(unsigned int n, unsigned int i, void * data) {
    fa_o_store_args_t *     args  = (fa_o_store_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const size_t n_rows_g = args->n_rows_g;
    const size_t G        = factx->G;
    const size_t DV       = factx->DV;

    const size_t rows_per_t = args->rows_per_t;
    const size_t start      = (size_t) i * rows_per_t;
    const size_t end        = hex_smin(start + rows_per_t, n_rows_g);

    if (start >= n_rows_g) {
        return;
    }

    struct htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (args->q_start * G + start));

    const struct htp_tensor * dst        = args->dst;
    const __fp16 *            o_tile_src = args->o_tile_src;
    const uint32_t            q_start    = args->q_start;
    const uint32_t            kv_head    = args->kv_head;
    const uint32_t            ib3        = args->ib3;

    for (size_t r = start; r < end; ++r) {
        const size_t q_idx = fastdiv(r, &factx->div_G);
        const size_t h_idx = fastmodulo(r, G, &factx->div_G);

        float * out = (float *) ((uint8_t *) dst->data + (kv_head * G + h_idx) * dst->nb[1] +
                                 (q_start + q_idx) * dst->nb[2] + ib3 * dst->nb[3]);

        size_t         r0            = r / HMX_FP16_TILE_N_ROWS;
        size_t         r1            = r % HMX_FP16_TILE_N_ROWS;
        const __fp16 * tile_row_base = o_tile_src + r0 * HMX_FP16_TILE_N_ROWS * DV;

        for (uint32_t d = 0; d < DV / 32; ++d) {
            const HVX_Vector * in_tile = (const HVX_Vector *) (tile_row_base + d * HMX_FP16_TILE_N_ELMS);
            HVX_VectorPair     vp      = hvx_vec_f16_to_f32_shuff(in_tile[r1 / 2]);
            if (r1 % 2 == 0) {
                *(HVX_UVector *) (out + d * 32) = Q6_V_lo_W(vp);
            } else {
                *(HVX_UVector *) (out + d * 32) = Q6_V_hi_W(vp);
            }
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (args->q_start * G + start));
}

static void fa_o_store_thread_f16(unsigned int n, unsigned int i, void * data) {
    fa_o_store_args_t *     args  = (fa_o_store_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const size_t n_rows_g   = args->n_rows_g;
    const size_t rows_per_t = args->rows_per_t;
    const size_t G          = factx->G;
    const size_t DV         = factx->DV;
    const size_t start      = (size_t) i * rows_per_t;
    const size_t end        = hex_smin(start + rows_per_t, n_rows_g);

    if (start >= n_rows_g) {
        return;
    }

    struct htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (args->q_start * G + start));

    const struct htp_tensor * dst        = args->dst;
    const __fp16 *            o_tile_src = args->o_tile_src;
    const uint32_t            q_start    = args->q_start;
    const uint32_t            kv_head    = args->kv_head;
    const uint32_t            ib3        = args->ib3;

    for (size_t r = start; r < end; ++r) {
        const size_t q_idx = fastdiv(r, &factx->div_G);
        const size_t h_idx = fastmodulo(r, G, &factx->div_G);

        __fp16 * out = (__fp16 *) ((uint8_t *) dst->data + (kv_head * G + h_idx) * dst->nb[1] +
                                   (q_start + q_idx) * dst->nb[2] + ib3 * dst->nb[3]);

        size_t         r0            = r / HMX_FP16_TILE_N_ROWS;
        size_t         r1            = r % HMX_FP16_TILE_N_ROWS;
        const __fp16 * tile_row_base = o_tile_src + r0 * HMX_FP16_TILE_N_ROWS * DV;

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
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (args->q_start * G + start));
}

static void fa_phase_o_store(struct hmx_fa_context *   factx,
                             const struct htp_tensor * dst,
                             const __fp16 *            o_tile_src,
                             uint32_t                  q_start,
                             uint32_t                  kv_head,
                             uint32_t                  ib3,
                             size_t                    n_rows_g) {
    work_queue_t wp = factx->octx->ctx->work_queue;
    uint32_t n = 1;
    if (factx->n_threads > 1 && n_rows_g >= (size_t) (factx->n_threads * 2)) {
        n = factx->n_threads;
    }
    size_t rows_per_t = hmx_ceil_div(n_rows_g, n);
    fa_o_store_args_t args = { factx, dst, o_tile_src, q_start, kv_head, ib3, n_rows_g, rows_per_t };
    worker_callback_t store_fn = factx->is_dst_fp32 ? fa_o_store_thread_f32 : fa_o_store_thread_f16;
    if (n > 1) {
        work_queue_run(wp, store_fn, &args, n);
    } else {
        store_fn(1, 0, &args);
    }
}

typedef struct {
    struct hmx_fa_context *   factx;
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
    unsigned int n, unsigned int i, void * data,
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
    const size_t n_tiles_per_bc = args->n_tiles_per_bc;
    const size_t n_row_vec_cnt  = hmx_ceil_div(n_rows_g, 64);
    const uint32_t im3          = has_mask ? fastmodulo(args->ib3, args->mask->ne[3], &factx->src3_div3) : 0;

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

    struct htp_thread_trace * tr = &factx->octx->ctx->trace[i];
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

            const __fp16 * s_ld_base = factx->vtcm_s_tiles + r0 * HMX_FP16_TILE_N_ROWS * Bc;
            __fp16 *       p_st_base = factx->vtcm_p_tiles + r0 * HMX_FP16_TILE_N_ROWS * Bc;

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
                for (size_t c = 0; c < kv_rows; c += 64) {
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
            for (size_t c = 0; c < kv_rows; c += 64) {
                size_t         ci          = c / 64;
                const size_t   ne          = hex_smin(kv_rows - c, 64);
                HVX_VectorPred q_tail_keep = Q6_Q_vsetq2_R(ne * sizeof(__fp16));

                if (has_mask) {
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
                } else {
                    if (ne < 64) {
                        my_row_buf0[ci] = Q6_V_vmux_QVV(q_tail_keep, my_row_buf0[ci], v_neg_inf);
                        my_row_buf1[ci] = Q6_V_vmux_QVV(q_tail_keep, my_row_buf1[ci], v_neg_inf);
                    }
                }

                v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_row_buf0[ci]);
                v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_row_buf1[ci]);
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

            for (size_t c = 0; c < kv_rows; c += 64) {
                size_t     ci           = c / 64;
                HVX_Vector v_s_minus_m0 = Q6_Vqf16_vsub_VhfVhf(my_row_buf0[ci], v_dup_m0);
                HVX_Vector v_s_minus_m1 = Q6_Vqf16_vsub_VhfVhf(my_row_buf1[ci], v_dup_m1);

                HVX_Vector v_p_row0_hf  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0));
                HVX_Vector v_p_row1_hf  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1));
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
        HVX_Vector exp_m_diff_f16 = hvx_vec_exp2_f16(v_m_diff_f16);

        HVX_VectorPair exp_m_diff_pair = hvx_vec_f16_to_f32(exp_m_diff_f16);
        HVX_Vector exp_m_diff0 = Q6_V_lo_W(exp_m_diff_pair);
        HVX_Vector exp_m_diff1 = Q6_V_hi_W(exp_m_diff_pair);

        HVX_VectorPair rowsum_acc_pair = hvx_vec_f16_to_f32(rowsum_acc_v);
        HVX_Vector     v_rowsum_acc_f32_0 = Q6_V_lo_W(rowsum_acc_pair);
        HVX_Vector     v_rowsum_acc_f32_1 = Q6_V_hi_W(rowsum_acc_pair);

        HVX_Vector v_l_curr0;
        HVX_Vector v_l_curr1;
        if (args->kv_start == 0 && factx->sinks != NULL) {
            // First KV block with sinks: m_prev holds the seeded sink value (not -inf),
            // so exp_m_diff = exp2(sink - m_curr) is the sink's contribution to the
            // denominator. l_prev is 0 here, so add exp_m_diff directly instead of
            // multiplying the (uninitialized) l_prev term.
            v_l_curr0 = HVX_OP_ADD_F32(exp_m_diff0, v_rowsum_acc_f32_0);
            v_l_curr1 = HVX_OP_ADD_F32(exp_m_diff1, v_rowsum_acc_f32_1);
        } else {
            HVX_Vector l_prev_v0 = factx->vtcm_l_vec[r_vec_idx * 2 + 0];
            HVX_Vector l_prev_v1 = factx->vtcm_l_vec[r_vec_idx * 2 + 1];
            v_l_curr0 = HVX_OP_ADD_F32(HVX_OP_MUL_F32(l_prev_v0, exp_m_diff0), v_rowsum_acc_f32_0);
            v_l_curr1 = HVX_OP_ADD_F32(HVX_OP_MUL_F32(l_prev_v1, exp_m_diff1), v_rowsum_acc_f32_1);
        }

        factx->vtcm_m_vec[r_vec_idx * 2 + 0] = v_m_curr0;
        factx->vtcm_m_vec[r_vec_idx * 2 + 1] = v_m_curr1;
        factx->vtcm_l_vec[r_vec_idx * 2 + 0] = v_l_curr0;
        factx->vtcm_l_vec[r_vec_idx * 2 + 1] = v_l_curr1;

        // Build diagonal tile D = diag(exp(m_diff))
        const HVX_Vector     v_offsets = *(const HVX_Vector *) d_tile_scatter_offsets;
        const HVX_VectorPred q_32_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));
        HVX_Vector           v_exp_m_diff = exp_m_diff_f16;

        size_t t0 = r_vec_idx * 2;
        if (t0 < args->n_row_tiles) {
            const HVX_Vector v_content = v_exp_m_diff;
            __fp16 *         out_base  = factx->vtcm_d_tiles + t0 * (args->n_row_tiles_g_br + 1) * HMX_FP16_TILE_N_ELMS;
            Q6_vscatter_QRMVhV(q_32_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
        }

        size_t t1 = r_vec_idx * 2 + 1;
        if (t1 < args->n_row_tiles) {
            const HVX_Vector v_content = Q6_V_vror_VR(v_exp_m_diff, 64);
            __fp16 *         out_base  = factx->vtcm_d_tiles + t1 * (args->n_row_tiles_g_br + 1) * HMX_FP16_TILE_N_ELMS;
            Q6_vscatter_QRMVhV(q_32_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, (uint16_t) (args->q_start * G + vec_start * 64));
}

static void fa_softmax_thread_nomask(unsigned int n, unsigned int i, void * data) {
    fa_softmax_impl(n, i, data,
                    /*has_mask=*/false,
                    /*mask_broadcast=*/false,
                    /*is_g1=*/false,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

static void fa_softmax_thread_mask_broadcast_g1(unsigned int n, unsigned int i, void * data) {
    fa_softmax_impl(n, i, data,
                    /*has_mask=*/true,
                    /*mask_broadcast=*/true,
                    /*is_g1=*/true,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

static void fa_softmax_thread_mask_broadcast_gn(unsigned int n, unsigned int i, void * data) {
    fa_softmax_impl(n, i, data,
                    /*has_mask=*/true,
                    /*mask_broadcast=*/true,
                    /*is_g1=*/false,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

static void fa_softmax_thread(unsigned int n, unsigned int i, void * data) {
    fa_softmax_args_t *     args  = (fa_softmax_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const bool has_mask       = (args->mask != NULL);
    const bool mask_broadcast = factx->mask_broadcast;
    const bool is_g1          = (args->G == 1);
    const bool has_alibi      = args->has_alibi;
    const bool has_softcap    = (factx->logit_softcap != 0.0f);

    fa_softmax_impl(n, i, data, has_mask, mask_broadcast, is_g1, has_alibi, has_softcap);
}

static __attribute__((noinline)) void fa_build_d_diag_inv_l(struct hmx_fa_context * factx,
                                                            size_t                  n_row_tiles,
                                                            size_t                  n_row_tiles_g_br) {
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

        __fp16 * out_base = factx->vtcm_d_tiles + i * (n_row_tiles_g_br + 1) * HMX_FP16_TILE_N_ELMS;
        Q6_vscatter_QRMVhV(q_32_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
    }
}

static void fa_phase_softmax_and_build_d(struct hmx_fa_context * factx,
                                         fa_softmax_args_t *     sargs,
                                         size_t                  n_row_tiles,
                                         size_t                  n_row_tiles_g_br) {
    work_queue_t wp = factx->octx->ctx->work_queue;
    const size_t n_row_vec_cnt = hmx_ceil_div(sargs->n_rows_g, 64);

    worker_callback_t softmax_fn = fa_softmax_thread;
    if (sargs->mask == NULL && factx->logit_softcap == 0.0f && !sargs->has_alibi) {
        softmax_fn = fa_softmax_thread_nomask;
    } else if (sargs->mask != NULL && factx->mask_broadcast && factx->logit_softcap == 0.0f && !sargs->has_alibi) {
        if (sargs->G == 1) {
            softmax_fn = fa_softmax_thread_mask_broadcast_g1;
        } else {
            softmax_fn = fa_softmax_thread_mask_broadcast_gn;
        }
    }

    if (factx->n_threads > 1 && n_row_vec_cnt >= 2) {
        uint32_t n_use = (uint32_t) hex_smin((size_t) factx->n_threads, n_row_vec_cnt);
        sargs->thread_div = init_fastdiv_values(n_use);
        work_queue_run(wp, softmax_fn, sargs, n_use);
    } else {
        softmax_fn(1, 0, sargs);
    }
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

static void hmx_fa_qk_dot_worker(void * data) {
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

static void hmx_fa_o_update_worker(void * data) {
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
        const __fp16 * d_diag     = d_tiles + r * (n_row_tiles_g_br + 1) * HMX_FP16_TILE_N_ELMS;
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

static void hmx_fa_o_norm_worker(void * data) {
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
        const __fp16 * d_diag = d_tiles + r * (n_row_tiles_g_br + 1) * HMX_FP16_TILE_N_ELMS;
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

    __fp16 temp_slopes[512] __attribute__((aligned(128)));
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

    // Allocate stack buffer to avoid scalar writes to VTCM (which generates L2 misses)
    __fp16 local_slopes[n_rows_g] __attribute__((aligned(128)));
    for (size_t r = 0; r < n_rows_g; ++r) {
        local_slopes[r] = temp_slopes[fastmodulo(r, G, &factx->div_G)];
    }

    // Copy to VTCM slopes using HVX block copy (both are aligned to 128 bytes)
    hvx_copy_f16_aa((uint8_t *)slopes, (const uint8_t *)local_slopes, n_rows_g);
}

static void fa_push_mask_dma_gqa(
    dma_queue *               dma,
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
        const uint8_t * ms_src = (const uint8_t *) mask->data + q_start * mask->nb[1] +
                                 im2 * mask->nb[2] + im3 * mask->nb[3] + kv_start * sizeof(__fp16);
        uint8_t * ms_dst = (uint8_t *) factx->vtcm_mask_buf + g * m_line_bytes;
        dma_queue_push(dma, dma_make_ptr(ms_dst, ms_src), G * m_line_bytes, mask->nb[1], kv_rows * sizeof(__fp16), n_rows_q);
    }
}

static void fa_pop_mask_dma_gqa(dma_queue * dma, uint32_t G) {
    for (uint32_t g = 0; g < G; ++g) {
        dma_queue_pop(dma);
    }
}

// ============================================================================
// Core HMX flash attention algorithm (GQA-merged)
// ============================================================================

int hmx_flash_attn_ext(struct htp_ops_context * octx) {
    struct htp_thread_trace * tr_hvx = &octx->ctx->trace[0];
    struct htp_thread_trace * tr_hmx = &octx->ctx->trace[HTP_MAX_NTHREADS];
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

    const uint32_t nek0 = k->ne[0];  // head_dim
    const uint32_t nek1 = k->ne[1];  // kv_len

    const uint32_t nev0 = v->ne[0];  // head_dim (DV)

    const uint32_t DK = neq0;
    const uint32_t DV = nev0;

    // HMX requires head_dim to be multiple of 32
    if (DK % 32 != 0 || DV % 32 != 0) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_fa_kernel_params * kparams = (const struct htp_fa_kernel_params *) octx->kernel_params;
    const uint32_t n_kv_heads = k->ne[2];

    // ======== Build context ========
    struct hmx_fa_context factx;
    memset(&factx, 0, sizeof(factx));
    factx.octx           = octx;
    factx.sinks          = octx->src[4];  // NULL if this op has no attention sinks
    factx.n_threads      = kparams->n_threads;
    factx.DK             = DK;
    factx.DV             = DV;
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
    factx.is_q_fp32      = (kparams->is_q_fp32 != 0);
    factx.is_dst_fp32    = (kparams->is_dst_fp32 != 0);
    factx.pipeline       = (kparams->u.hmx.pipeline != 0);
    factx.mask_broadcast = (kparams->u.hmx.mask_broadcast != 0);
    if (mask) {
        factx.src3_div2  = kparams->src3_div2;
        factx.src3_div3  = kparams->src3_div3;
    }

    if (kparams->logit_softcap == 0.0f) {
        factx.scale = (__fp16) (kparams->scale * EXP_LOG2E_F);  // log2(e)
    } else {
        factx.scale = (__fp16) kparams->scale;
    }
    factx.max_bias      = kparams->max_bias;
    factx.logit_softcap = (__fp16) (kparams->logit_softcap * EXP_LOG2E_F);

    factx.n_head_log2 = kparams->n_head_log2;
    factx.m0          = kparams->m0;
    factx.m1          = kparams->m1;

    const uint32_t Br = factx.Br;
    const uint32_t Bc = factx.Bc;
    const uint32_t g_br = factx.g_br;
    const bool pipeline = factx.pipeline;
    const uint32_t n_threads = factx.n_threads;
    const uint32_t G = factx.G;

    // ======== VTCM allocation (GQA-aware) ========
    // K/V row sizes drive the DMA descriptors (not the VTCM layout) and are used
    // throughout the KV loop below.
    const size_t size_k_row        = DK * sizeof(__fp16);
    const size_t size_v_row        = DV * sizeof(__fp16);
    const size_t size_k_row_padded = hex_round_up(size_k_row, 128);
    const size_t size_v_row_padded = hex_round_up(size_v_row, 128);

    // Build the VTCM layout once (shared with the host estimator) and place every
    // scratch buffer at its computed offset.
    struct hmx_fa_vtcm_layout L;
    hmx_fa_vtcm_layout_build(&L, G, DK, DV, Br, Bc, n_threads, pipeline);

    if (L.total_bytes > ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * const base = ctx->vtcm_base;

    factx.vtcm_q_tiles        = VTCM_LAYOUT_PTR(__fp16, base, L.off_q_tiles);
    factx.vtcm_o_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, base, L.off_o_tiles[0]);
    factx.vtcm_o_tiles[1]     = VTCM_LAYOUT_PTR(__fp16, base, L.off_o_tiles[1]);
    factx.vtcm_k_fp16[0]      = VTCM_LAYOUT_PTR(__fp16, base, L.off_k_fp16[0]);
    factx.vtcm_k_fp16[1]      = VTCM_LAYOUT_PTR(__fp16, base, L.off_k_fp16[1]);
    factx.vtcm_v_fp16[0]      = VTCM_LAYOUT_PTR(__fp16, base, L.off_v_fp16[0]);
    factx.vtcm_v_fp16[1]      = VTCM_LAYOUT_PTR(__fp16, base, L.off_v_fp16[1]);
    factx.vtcm_k_tiles        = VTCM_LAYOUT_PTR(__fp16, base, L.off_k_tiles);
    factx.vtcm_v_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, base, L.off_v_tiles[0]);
    factx.vtcm_v_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_v_tiles[1], pipeline);
    factx.vtcm_s_tiles        = VTCM_LAYOUT_PTR(__fp16, base, L.off_s_tiles);
    factx.vtcm_p_tiles        = VTCM_LAYOUT_PTR(__fp16, base, L.off_p_tiles);
    factx.vtcm_d_tiles        = VTCM_LAYOUT_PTR(__fp16, base, L.off_d_tiles);
    factx.vtcm_m_vec          = VTCM_LAYOUT_PTR(HVX_Vector, base, L.off_m_vec);
    factx.vtcm_l_vec          = VTCM_LAYOUT_PTR(HVX_Vector, base, L.off_l_vec);
    factx.vtcm_s_rowmax       = VTCM_LAYOUT_PTR(HVX_Vector, base, L.off_s_rowmax);
    factx.vtcm_p_rowsum       = VTCM_LAYOUT_PTR(HVX_Vector, base, L.off_p_rowsum);
    factx.vtcm_row_bufs       = VTCM_LAYOUT_PTR(HVX_Vector, base, L.off_row_bufs);
    factx.row_buf_stride      = L.row_buf_stride;
    factx.vtcm_hmx_scales_id  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_hmx_scales_id);
    factx.vtcm_hmx_scales_qk  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_hmx_scales_qk);
    factx.vtcm_mask_buf       = VTCM_LAYOUT_PTR(__fp16, base, L.off_mask_buf);
    factx.mask_buf_row_stride = L.mask_buf_row_stride;
    factx.q_tile_bytes        = L.q_tile_bytes;
    factx.o_tile_bytes        = L.o_tile_bytes;
    factx.col_vec_bytes       = L.col_vec_bytes;
    factx.d_tile_bytes        = L.d_tile_bytes;
    factx.vtcm_slopes         = VTCM_LAYOUT_PTR(__fp16, base, L.off_slopes);

    const size_t m_line_bytes = L.m_line_bytes;  // used by the mask DMAs in the KV loop

    dma_cache_init(&factx.m_cache, (uint8_t *) factx.vtcm_mask_buf, L.m_buf_slot_bytes, HMX_FA_DMA_CACHE_SIZE);

    // ======== Initialize HMX output scales ========
    hmx_init_column_scales(factx.vtcm_hmx_scales_id, Q6_V_vsplat_R(0x3c00)); // 1.0
    hmx_init_column_scales(factx.vtcm_hmx_scales_qk, hvx_vec_splat_f16(factx.scale));

    // ======== Skip compute if profiling ========
    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    // ======== DMA setup ========
    dma_queue * const dma = ctx->dma[0];

    const size_t n_row_tiles_g_br = g_br / HMX_FP16_TILE_N_ROWS;
    const size_t n_tiles_per_bc   = Bc / HMX_FP16_TILE_N_COLS;

    const size_t qo_element_size = factx.is_q_fp32 ? sizeof(float) : sizeof(__fp16);

    // ======== Reusable job descriptors for pipeline ========
    hmx_fa_qk_job_t       qk_job;
    hmx_fa_o_update_job_t ou_job;
    hmx_fa_o_norm_job_t   on_job;

    // ======== Main loop ========
    for (uint32_t ib3 = 0; ib3 < neq3; ++ib3) {
        const uint32_t im3 = mask ? fastmodulo(ib3, mask->ne[3], &factx.src3_div3) : 0;
        for (uint32_t q_start = 0; q_start < neq1; q_start += Br) {
            const uint32_t n_rows_q    = hex_smin(Br, neq1 - q_start);
            const size_t   n_rows_g    = n_rows_q * G;
            const size_t   g_br_actual = hex_align_up(n_rows_g, HMX_FP16_TILE_N_ROWS);
            const size_t   n_row_tiles = g_br_actual / HMX_FP16_TILE_N_ROWS;

            for (uint32_t kv_head = 0; kv_head < n_kv_heads; ++kv_head) {
                const uint32_t ik2 = kv_head;
                const uint32_t ik3 = fastdiv(ib3, &kparams->broadcast_rk3);
                const uint32_t iv2 = kv_head;
                const uint32_t iv3 = fastdiv(ib3, &kparams->broadcast_rv3);

                // 1. Push Q DMA (if Q DMA is used)
                const size_t o_tile_bytes = factx.o_tile_bytes;
                const bool use_q_dma = (2 * o_tile_bytes >= factx.g_br * factx.DK * (factx.is_q_fp32 ? 4 : 2));
                if (use_q_dma) {
                    const bool q_transposed = q->nb[1] < q->nb[2];
                    const uint8_t * q_ptr = (const uint8_t *) q->data + q_start * q->nb[1] + (kv_head * factx.G) * q->nb[2] + ib3 * q->nb[3];
                    const size_t el_size = factx.is_q_fp32 ? sizeof(float) : sizeof(__fp16);
                    const size_t q_row_bytes = q_transposed ? n_rows_q * factx.DK * el_size : factx.G * factx.DK * el_size;
                    const size_t src_stride  = q_transposed ? q->nb[2] : q->nb[1];
                    const size_t n_rows      = q_transposed ? factx.G : n_rows_q;
                    dma_queue_push(dma, dma_make_ptr(factx.vtcm_o_tiles[0], q_ptr), q_row_bytes, hex_smax(src_stride, q_row_bytes), q_row_bytes, n_rows);
                }

                // 2. Prefetch first KV block
                if (factx.n_kv_blocks > 0) {
                    const uint32_t kv_rows0 = hex_smin(Bc, nek1);

                    const uint8_t * k_src = (const uint8_t *) k->data + ik2 * k->nb[2] + ik3 * k->nb[3];
                    dma_queue_push(dma, dma_make_ptr(factx.vtcm_k_fp16[0], k_src), size_k_row_padded, k->nb[1], size_k_row, kv_rows0);

                    const uint8_t * v_src = (const uint8_t *) v->data + iv2 * v->nb[2] + iv3 * v->nb[3];
                    dma_queue_push(dma, dma_make_ptr(factx.vtcm_v_fp16[0], v_src), size_v_row_padded, v->nb[1], size_v_row, kv_rows0);
                }

                // 3. Pop Q DMA (blocks until Q is loaded)
                if (use_q_dma) {
                    dma_queue_pop(dma);
                }

                // ---- Load Q block & Initialize per-block state ----
                fa_phase_q_load(&factx, q, q_start, kv_head, ib3, n_rows_g);

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
                    // Pipeline path
                    for (uint32_t kv_blk = 0; kv_blk < factx.n_kv_blocks; ++kv_blk) {
                        const uint32_t kv_start    = kv_blk * Bc;
                        const uint32_t kv_rows     = hex_smin(Bc, nek1 - kv_start);
                        const size_t   n_col_tiles = hmx_ceil_div(kv_rows, HMX_FP16_TILE_N_COLS);

                        // Push mask DMA
                        if (mask) {
                            if (__builtin_expect(factx.mask_broadcast, true)) {
                                const uint8_t * ms_src = (const uint8_t *) mask->data + q_start * mask->nb[1] + im3 * mask->nb[3] + kv_start * sizeof(__fp16);
                                dma_cache_push(dma, &factx.m_cache, ms_src, m_line_bytes, mask->nb[1], kv_rows * sizeof(__fp16), n_rows_q);
                            } else {
                                fa_push_mask_dma_gqa(dma, mask, q_start, im3, kv_start, kv_head, G, m_line_bytes, kv_rows, n_rows_q, &factx);
                            }
                        }

                        // Prefetch next KV block early
                        if (kv_blk + 1 < factx.n_kv_blocks) {
                            const uint32_t  prefetch_start = (kv_blk + 1) * Bc;
                            const uint32_t  prefetch_rows  = hex_smin(Bc, nek1 - prefetch_start);
                            const size_t    prefetch_buf   = 1 - buf_idx;
                            const uint8_t * k_prefetch_src = (const uint8_t *) k->data + prefetch_start * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
                            dma_queue_push(dma, dma_make_ptr(factx.vtcm_k_fp16[prefetch_buf], k_prefetch_src), size_k_row_padded, k->nb[1], size_k_row, prefetch_rows);
                            const uint8_t * v_prefetch_src = (const uint8_t *) v->data + prefetch_start * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
                            dma_queue_push(dma, dma_make_ptr(factx.vtcm_v_fp16[prefetch_buf], v_prefetch_src), size_v_row_padded, v->nb[1], size_v_row, prefetch_rows);
                        }

                        // ---- Phase 1: K_int ----
                        if (kv_blk > 0) {
                            ou_job.o_curr           = o_tile_curr;
                            ou_job.o_prev           = o_tile_prev;
                            ou_job.p_tiles          = factx.vtcm_p_tiles;
                            ou_job.v_tiles          = factx.vtcm_v_tiles[1 - buf_idx];
                            ou_job.d_tiles          = factx.vtcm_d_tiles;
                            ou_job.hmx_scales       = factx.vtcm_hmx_scales_id;
                            ou_job.n_row_tiles      = n_row_tiles;
                            ou_job.n_col_tiles      = hmx_ceil_div(hex_smin(Bc, nek1 - (kv_blk - 1) * Bc), HMX_FP16_TILE_N_COLS);
                            ou_job.n_row_tiles_g_br = n_row_tiles_g_br;
                            ou_job.n_tiles_per_bc   = n_tiles_per_bc;
                            ou_job.DV               = DV;
                            hmx_queue_push(hmx_q, hmx_queue_make_desc(hmx_fa_o_update_worker, &ou_job));
                        }

                        // Wait for current K DMA and interleave
                        void * curr_k = dma_queue_pop(dma).dst;
                        fa_phase_k_interleave(&factx, kv_rows, k_src_stride, curr_k, kv_start);

                        // ---- Phase 2: qk_dot ----
                        qk_job.q_tiles        = factx.vtcm_q_tiles;
                        qk_job.k_tiles        = factx.vtcm_k_tiles;
                        qk_job.s_tiles        = factx.vtcm_s_tiles;
                        qk_job.n_row_tiles    = n_row_tiles;
                        qk_job.n_col_tiles    = n_col_tiles;
                        qk_job.n_dot_tiles    = DK / 32;
                        qk_job.n_tiles_per_bc = n_tiles_per_bc;
                        qk_job.hmx_scales     = factx.vtcm_hmx_scales_qk;
                        hmx_queue_push(hmx_q, hmx_queue_make_desc(hmx_fa_qk_dot_worker, &qk_job));

                        // Wait for current V DMA and interleave
                        void * curr_v = dma_queue_pop(dma).dst;
                        fa_phase_v_interleave(&factx, kv_rows, v_src_stride, curr_v, factx.vtcm_v_tiles[buf_idx], n_tiles_per_bc, kv_start);

                        if (kv_blk > 0) {
                            hmx_queue_pop(hmx_q);
                            hex_swap_ptr((void **) &o_tile_curr, (void **) &o_tile_prev);
                        }

                        hmx_queue_pop(hmx_q);

                        // ---- Phase 3: softmax + build_D ----
                        __fp16 * current_mask_vtcm = NULL;
                        if (mask) {
                            if (__builtin_expect(factx.mask_broadcast, true)) {
                                current_mask_vtcm = (__fp16 *) dma_queue_pop(dma).dst;
                            } else {
                                fa_pop_mask_dma_gqa(dma, G);
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

                        buf_idx = 1 - buf_idx;
                    }

                    // Epilogue
                    if (factx.n_kv_blocks > 0) {
                        const uint32_t last_blk = factx.n_kv_blocks - 1;
                        const size_t last_cols  = hmx_ceil_div(hex_smin(Bc, nek1 - last_blk * Bc), HMX_FP16_TILE_N_COLS);
                        ou_job.o_curr           = o_tile_curr;
                        ou_job.o_prev           = o_tile_prev;
                        ou_job.p_tiles          = factx.vtcm_p_tiles;
                        ou_job.v_tiles          = factx.vtcm_v_tiles[1 - buf_idx];
                        ou_job.d_tiles          = factx.vtcm_d_tiles;
                        ou_job.hmx_scales       = factx.vtcm_hmx_scales_id;
                        ou_job.n_row_tiles      = n_row_tiles;
                        ou_job.n_col_tiles      = last_cols;
                        ou_job.n_row_tiles_g_br = n_row_tiles_g_br;
                        ou_job.n_tiles_per_bc   = n_tiles_per_bc;
                        ou_job.DV               = DV;
                        hmx_queue_push(hmx_q, hmx_queue_make_desc(hmx_fa_o_update_worker, &ou_job));
                        hmx_queue_pop(hmx_q);

                        hex_swap_ptr((void **) &o_tile_curr, (void **) &o_tile_prev);
                    }

                } else {
                    // Fallback path
                    for (uint32_t kv_blk = 0; kv_blk < factx.n_kv_blocks; ++kv_blk) {
                        const uint32_t kv_start    = kv_blk * Bc;
                        const uint32_t kv_rows     = hex_smin(Bc, nek1 - kv_start);
                        const size_t   n_col_tiles = hmx_ceil_div(kv_rows, HMX_FP16_TILE_N_COLS);

                        if (mask) {
                            if (__builtin_expect(factx.mask_broadcast, true)) {
                                const uint8_t * ms_src = (const uint8_t *) mask->data + q_start * mask->nb[1] + im3 * mask->nb[3] + kv_start * sizeof(__fp16);
                                dma_cache_push(dma, &factx.m_cache, ms_src, m_line_bytes, mask->nb[1], kv_rows * sizeof(__fp16), n_rows_q);
                            } else {
                                fa_push_mask_dma_gqa(dma, mask, q_start, im3, kv_start, kv_head, G, m_line_bytes, kv_rows, n_rows_q, &factx);
                            }
                        }

                        if (kv_blk + 1 < factx.n_kv_blocks) {
                            const uint32_t  prefetch_start = (kv_blk + 1) * Bc;
                            const uint32_t  prefetch_rows  = hex_smin(Bc, nek1 - prefetch_start);
                            const size_t    prefetch_buf   = 1 - buf_idx;
                            const uint8_t * k_prefetch_src = (const uint8_t *) k->data + prefetch_start * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
                            dma_queue_push(dma, dma_make_ptr(factx.vtcm_k_fp16[prefetch_buf], k_prefetch_src), size_k_row_padded, k->nb[1], size_k_row, prefetch_rows);
                            const uint8_t * v_prefetch_src = (const uint8_t *) v->data + prefetch_start * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
                            dma_queue_push(dma, dma_make_ptr(factx.vtcm_v_fp16[prefetch_buf], v_prefetch_src), size_v_row_padded, v->nb[1], size_v_row, prefetch_rows);
                        }

                        // Wait for current K DMA and interleave
                        void * curr_k = dma_queue_pop(dma).dst;
                        fa_phase_k_interleave(&factx, kv_rows, k_src_stride, curr_k, kv_start);

                        {
                            qk_job.q_tiles        = factx.vtcm_q_tiles;
                            qk_job.k_tiles        = factx.vtcm_k_tiles;
                            qk_job.s_tiles        = factx.vtcm_s_tiles;
                            qk_job.n_row_tiles    = n_row_tiles;
                            qk_job.n_col_tiles    = n_col_tiles;
                            qk_job.n_dot_tiles    = (size_t) (DK / 32);
                            qk_job.n_tiles_per_bc = n_tiles_per_bc;
                            qk_job.hmx_scales     = factx.vtcm_hmx_scales_qk;

                            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_fa_qk_dot_worker, &qk_job));
                            hmx_queue_pop(ctx->hmx_queue);
                        }

                        // Wait for current V DMA and interleave
                        void * curr_v = dma_queue_pop(dma).dst;
                        fa_phase_v_interleave(&factx, kv_rows, v_src_stride, curr_v, factx.vtcm_v_tiles[0], n_tiles_per_bc, kv_start);

                        // ---- Phase 3: softmax + build_D ----
                        __fp16 * current_mask_vtcm = NULL;
                        if (mask) {
                            if (__builtin_expect(factx.mask_broadcast, true)) {
                                current_mask_vtcm = (__fp16 *) dma_queue_pop(dma).dst;
                            } else {
                                fa_pop_mask_dma_gqa(dma, G);
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

                        {
                            ou_job.o_curr           = o_tile_curr;
                            ou_job.o_prev           = o_tile_prev;
                            ou_job.p_tiles          = factx.vtcm_p_tiles;
                            ou_job.v_tiles          = factx.vtcm_v_tiles[0];
                            ou_job.d_tiles          = factx.vtcm_d_tiles;
                            ou_job.hmx_scales       = factx.vtcm_hmx_scales_id;
                            ou_job.n_row_tiles      = n_row_tiles;
                            ou_job.n_col_tiles      = n_col_tiles;
                            ou_job.n_row_tiles_g_br = n_row_tiles_g_br;
                            ou_job.n_tiles_per_bc   = n_tiles_per_bc;
                            ou_job.DV               = DV;

                            hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_fa_o_update_worker, &ou_job));
                            hmx_queue_pop(ctx->hmx_queue);

                            hex_swap_ptr((void **) &o_tile_curr, (void **) &o_tile_prev);
                        }

                        buf_idx = 1 - buf_idx;
                    }
                }

                // ---- Final normalization ----
                {
                    htp_trace_event_start(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_start);
                    fa_build_d_diag_inv_l(&factx, n_row_tiles, n_row_tiles_g_br);
                    htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_start);

                    on_job.o_curr           = o_tile_curr;
                    on_job.o_prev           = o_tile_prev;
                    on_job.d_tiles          = factx.vtcm_d_tiles;
                    on_job.hmx_scales       = factx.vtcm_hmx_scales_id;
                    on_job.n_row_tiles      = n_row_tiles;
                    on_job.n_row_tiles_g_br = n_row_tiles_g_br;
                    on_job.DV               = DV;
                    hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_fa_o_norm_worker, &on_job));
                    hmx_queue_pop(ctx->hmx_queue);
                }

                // ---- Store O block ----
                fa_phase_o_store(&factx, dst, o_tile_curr, q_start, kv_head, ib3, n_rows_g);
            }
        }
    }

    return HTP_STATUS_OK;
}

int op_flash_attn_ext(struct htp_ops_context * octx) {
    const struct htp_tensor * q    = octx->src[0];
    const struct htp_tensor * k    = octx->src[1];
    const struct htp_tensor * v    = octx->src[2];
    const struct htp_tensor * mask = octx->src[3];
    const struct htp_tensor * dst  = octx->dst;

    // Check support
    if ((q->type != HTP_TYPE_F16 && q->type != HTP_TYPE_F32) || k->type != HTP_TYPE_F16 || v->type != HTP_TYPE_F16) {
        return HTP_STATUS_NO_SUPPORT;
    }

    const struct htp_fa_kernel_params * kparams = (const struct htp_fa_kernel_params *) octx->kernel_params;

    if (kparams->kernel_type == HTP_FA_KERNEL_UNSUPPORTED) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (kparams->kernel_type == HTP_FA_KERNEL_HMX) {
        return hmx_flash_attn_ext(octx);
    }

    struct htp_fa_context factx;
    factx.octx = octx;

    factx.t_start = HAP_perf_get_qtimer_count();

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

    factx.is_q_fp32 = (kparams->is_q_fp32 != 0);
    factx.size_q_row_padded = kparams->u.hvx.size_q_row_padded;
    factx.size_k_row_padded = kparams->u.hvx.size_k_row_padded;
    factx.size_v_row_padded = kparams->u.hvx.size_v_row_padded;

    size_t size_q_block = factx.size_q_row_padded * 1; // single row for now
    factx.size_k_block = factx.size_k_row_padded * FLASH_ATTN_BLOCK_SIZE;
    factx.size_v_block = factx.size_v_row_padded * FLASH_ATTN_BLOCK_SIZE;
    factx.size_m_block = hex_round_up(FLASH_ATTN_BLOCK_SIZE * sizeof(__fp16), 128);

    factx.n_blocks = kparams->n_kv_blocks;

    factx.scale = kparams->scale;
    factx.max_bias = kparams->max_bias;
    factx.logit_softcap = (__fp16) kparams->logit_softcap;

    factx.n_head_log2 = kparams->n_head_log2;
    factx.m0          = kparams->m0;
    factx.m1          = kparams->m1;

    const uint32_t n_head = q->ne[2];
    if (n_head > 512) {
        return HTP_STATUS_NO_SUPPORT;
    }
    for (uint32_t h = 0; h < n_head; ++h) {
        factx.slopes[h] = (__fp16) ((kparams->max_bias > 0.0f) ? alibi_slope(h, factx.n_head_log2, factx.m0, factx.m1) : 1.0f);
    }

    // total rows in q
    factx.qrows = kparams->qrows;
    factx.qrows_per_thread = kparams->qrows_per_thread;

    size_t size_vkq_acc = hex_round_up(v->ne[0] * sizeof(float), 128); // VKQ32

    factx.size_q_block = size_q_block;
    factx.size_vkq_acc = size_vkq_acc;

    uint8_t * vtcm_cur = octx->ctx->vtcm_base;

    factx.spad_q = vtcm_seq_alloc(&vtcm_cur, size_q_block * octx->n_threads);
    factx.spad_k = vtcm_seq_alloc(&vtcm_cur, factx.size_k_block * 2 * octx->n_threads);
    factx.spad_v = vtcm_seq_alloc(&vtcm_cur, factx.size_v_block * 2 * octx->n_threads);
    factx.spad_m = vtcm_seq_alloc(&vtcm_cur, (mask ? factx.size_m_block * HVX_FA_DMA_CACHE_SIZE : 0) * octx->n_threads);
    factx.spad_a = vtcm_seq_alloc(&vtcm_cur, size_vkq_acc * octx->n_threads);

    if ((size_t) (vtcm_cur - octx->ctx->vtcm_base) > octx->ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    if (!(octx->flags & HTP_OPFLAGS_SKIP_COMPUTE)) {
        work_queue_run(octx->ctx->work_queue, flash_attn_ext_f16_thread, &factx, octx->n_threads);
    }

    return HTP_STATUS_OK;
}
