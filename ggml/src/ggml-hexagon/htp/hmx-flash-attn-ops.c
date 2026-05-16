// HMX-accelerated Flash Attention for prefill (neq1 >= 32).
// Ported from htp-ops-lib/src/dsp/ops/flash_attn.c, adapted to the htp/ codebase.

#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <assert.h>
#include <HAP_compute_res.h>
#include <HAP_farf.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "hex-dma.h"
#include "hmx-profile.h"
#include "hmx-queue.h"
#include "hmx-utils.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "hvx-dump.h"
#include "hvx-reduce.h"
#include "hvx-utils.h"
#include "vtcm-utils.h"
#include "worker-pool.h"

// ============================================================================
// Constants
// ============================================================================

// Tile constants from hmx-utils.h
// HMX_FP16_TILE_N_ROWS  = 32
// HMX_FP16_TILE_N_COLS  = 32
// HMX_FP16_TILE_N_ELMS  = 1024
// HMX_FP16_TILE_SIZE    = 2048

// ============================================================================
// Dynamic block size computation (GQA-aware)
// ============================================================================

// Exact VTCM usage for a given (gqa_factor, DK, DV, Br, Bc) configuration.
// g_br = hex_align_up(gqa_factor * Br, 32) replaces Br for all Q/O/S/P/D dimensions.
// Layout: Q + O_ping + O_pong + K_dma*2 + V_dma*2 + K_tile + V_tile + S + P + D + vectors + scales
// Mask is DMA'd into a VTCM buffer (Br rows per KV block) to avoid DDR reads in softmax.
static size_t hmx_fa_compute_vtcm_usage(size_t gqa_factor, size_t DK, size_t DV, size_t Br, size_t Bc, size_t n_threads) {
    const size_t g_br         = hex_align_up(gqa_factor * Br, HMX_FP16_TILE_N_ROWS);
    const size_t q_tile_size  = hex_align_up(g_br * DK * sizeof(__fp16), 4096);    // Q:  [g_br, DK]
    const size_t o_tile_size  = hex_align_up(g_br * DV * sizeof(__fp16), 4096);    // O:  [g_br, DV] x2 ping-pong
    const size_t k_dma_size   = hex_align_up(Bc * DK * sizeof(__fp16), 4096);      // K DMA: [Bc, DK] x2 double-buf
    const size_t v_dma_size   = hex_align_up(Bc * DV * sizeof(__fp16), 4096);      // V DMA: [Bc, DV] x2 double-buf
    const size_t k_tile_size  = hex_align_up(Bc * DK * sizeof(__fp16), 4096);      // K tiles: [Bc, DK] interleaved
    const size_t v_tile_size  = hex_align_up(Bc * DV * sizeof(__fp16), 4096);      // V tiles: [Bc, DV] interleaved
    const size_t s_tile_size  = hex_align_up(g_br * Bc * sizeof(__fp16), 4096);    // S/P:[g_br, Bc]
    const size_t d_tile_size  = hex_align_up(g_br * g_br * sizeof(__fp16), 4096);  // D:  [g_br, g_br]
    const size_t col_vec_size = hex_align_up(g_br * sizeof(__fp16), 256);          // m, l, etc.
    const size_t row_vec_size = hex_align_up(Bc * sizeof(__fp16), 256);
    const size_t m_line_size  = hex_align_up(Bc * sizeof(__fp16), 128);
    const size_t m_buf_size   = hex_align_up(Br * m_line_size, 4096);
    const size_t slopes_size  = hex_align_up(g_br * sizeof(__fp16), 128);

    return   q_tile_size * 1               // Q tiles
           + o_tile_size * 2               // O ping-pong
           + k_dma_size  * 2               // K DMA x2
           + v_dma_size  * 2               // V DMA x2
           + k_tile_size * 1               // K tiles
           + v_tile_size * 1               // V tiles
           + s_tile_size * 2               // S + P
           + d_tile_size * 1               // D (diagonal matrix)
           + col_vec_size * 4              // m_vec, l_vec, s_rowmax, p_rowsum
           + row_vec_size * 2 * n_threads  // per-thread softmax row scratch
           + m_buf_size * 1                // mask VTCM buffer [Br rows]
           + slopes_size                   // Slopes
           + 256 * 2;                      // HMX scales (id + qk)
}

// ============================================================================
// FP16 exp2 polynomial (ported from htp-ops-lib/include/dsp/hvx_math.h)
// ============================================================================
// 5th-order Horner polynomial for exp2(x) in qf16/hf16 domain.  Input must be
// ≤ 0 (safe softmax invariant — overflow handling omitted).  ~18 ALU ops per
// 64 fp16 lanes, fully parallel across HVX threads (no scatter/gather engine).
// Replaces the F32 round-trip (qf16→f32→exp→f32→f16, ~44 ops for 2×32 lanes).
static inline HVX_Vector hvx_exp2_hf(HVX_Vector x_v) {
    const HVX_Vector zero_v    = Q6_V_vzero();
    const HVX_Vector half_hf_v = Q6_Vh_vsplat_R(0x3800);  // fp16 0.5

    // k = round_toward_neg_inf(x);  f = (float)k;  frac = x - f
    HVX_Vector x_minus_half = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vsub_VhfVhf(x_v, half_hf_v));
    HVX_Vector k_v          = Q6_Vh_equals_Vhf(x_minus_half);  // truncate to int16
    HVX_Vector f_v          = Q6_Vhf_equals_Vh(k_v);           // back to fp16

    HVX_Vector x_qf16 = Q6_Vqf16_vsub_VhfVhf(x_v, f_v);        // fractional part in qf16

    // Horner: y = ((((E5*x + E4)*x + E3)*x + E2)*x + E1)*x + E0
    HVX_Vector y = Q6_Vqf16_vmpy_Vqf16Vqf16(Q6_Vh_vsplat_R(0x5082), x_qf16); // E5*x
    y            = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x157d));        // + E4
    y            = Q6_Vqf16_vmpy_Vqf16Vqf16(y, x_qf16);
    y            = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x20ed));        // + E3
    y            = Q6_Vqf16_vmpy_Vqf16Vqf16(y, x_qf16);
    y            = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x2b1b));        // + E2
    y            = Q6_Vqf16_vmpy_Vqf16Vqf16(y, x_qf16);
    y            = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x33b0));        // + E1
    y            = Q6_Vqf16_vmpy_Vqf16Vqf16(y, x_qf16);
    y            = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x398c));        // + E0
    y            = Q6_Vqf16_vmpy_Vqf16Vqf16(y, x_qf16);                      // y = y * x
    y            = Q6_Vqf16_vadd_Vqf16Vhf(y, Q6_Vh_vsplat_R(0x3c00));        // + 1.0

    // Combine polynomial (mantissa) with integer part (exponent): result = y * 2^k
    y                          = Q6_Vhf_equals_Vqf16(y);
    HVX_Vector y_exp           = Q6_Vuh_vlsr_VuhR(Q6_Vh_vasl_VhR(y, 1), 11);
    y_exp                      = Q6_Vh_vadd_VhVh(k_v, y_exp);
    HVX_VectorPred q_underflow = Q6_Q_vcmp_gt_VhVh(zero_v, y_exp);
    y                          = Q6_Vh_vaslacc_VhVhR(y, k_v, 10);
    return Q6_V_vmux_QVV(q_underflow, zero_v, y);
}

#define FA_MIN_KV_BLOCKS 3

// Cost-based (Br, Bc) search for flash attention with pipeline constraint.
//
// VTCM model (same as before):
//   overhead + g_br * per_gbr + g_br² * per_gbr2 + Bc * per_bc + g_br * Bc * per_gbr_bc
//
// Cost model (minimization objective):
//   Q * (c_q_fixed + K * c_iter_fixed),  where Q = ceil(qo/Br), K = ceil(kv/Bc)
static int hmx_fa_find_chunk_size(size_t * Br_out,
                                  size_t * Bc_out,
                                  size_t   gqa_factor,
                                  size_t   DK,
                                  size_t   DV,
                                  size_t   qo_len,
                                  size_t   kv_len,
                                  size_t   vtcm_budget,
                                  size_t   n_threads) {
    const size_t T       = HMX_FP16_TILE_N_ROWS;  // 32
    const size_t br_unit = hmx_ceil_div(T, gqa_factor);
    // Bc must be a multiple of 64 so that n_tiles_per_bc is even.  The softmax
    // P-tile write uses a dual-tile pattern (vshuff + two stores 16 slots apart)
    // that would race across r0 blocks if the last dual-tile is half-occupied.
    // See .cursor/todos/hmx-flash-attn-bc-search-space.md for the perf trade-off.
    const size_t bc_unit = HMX_FP16_TILE_N_COLS * 2;  // 64
    const size_t fp16    = sizeof(__fp16);

    // Approximate per-unit VTCM costs (without per-buffer alignment padding).
    const size_t per_gbr  = (DK + 2 * DV) * fp16 + 4 * fp16;  // Q + O×2 + 4 col vectors
    const size_t per_gbr2 = fp16;                             // D diagonal matrix
    const size_t per_bc =
        3 * (DK + DV) * fp16 + 2 * n_threads * fp16;          // K_dma×2 + V_dma×2 + K_tile + V_tile + row bufs
    const size_t per_gbr_bc = 2 * fp16;                       // S + P

    const size_t overhead = 256 * 2 + 13 * 4096;

    if (vtcm_budget <= overhead) {
        return -1;
    }
    const size_t usable = vtcm_budget - overhead;

    // Br_max: largest Br aligned to br_unit that does not exceed qo_len.
    const size_t Br_max = qo_len >= br_unit ? hex_align_down(qo_len, br_unit) : br_unit;

    // Pipeline constraint: cap Bc so n_kv_blocks >= FA_MIN_KV_BLOCKS.
    // Only relax when kv_len is too short to form enough blocks.
    const bool   can_pipeline = (kv_len >= FA_MIN_KV_BLOCKS * bc_unit && n_threads >= 2);
    const size_t Bc_limit     = can_pipeline ? hex_align_down(kv_len / FA_MIN_KV_BLOCKS, bc_unit) :
                                               (kv_len >= bc_unit ? hex_align_down(kv_len, bc_unit) : bc_unit);
    // Cost coefficients calibrated from profiling
    const size_t c_q_fixed    = 1400;  // per-Q-block: q_load + epilogue o_update + o_norm + o_store
    const size_t c_iter_fixed = 200;   // per-KV-iter: HMX queue push/pop + DMA pop + barriers

    size_t best_cost = SIZE_MAX, best_mn = 0;
    size_t best_Br = 0, best_Bc = 0;

    for (size_t Br = Br_max; Br >= br_unit; Br -= br_unit) {
        const size_t g_br = hex_align_up(gqa_factor * Br, T);

        // g_br-dependent VTCM cost: g_br * per_gbr + g_br² * per_gbr2
        const size_t gbr_cost = g_br * per_gbr + g_br * g_br * per_gbr2;
        if (gbr_cost >= usable) {
            if (Br == br_unit) {
                break;
            }
            continue;
        }

        // Analytically solve for max Bc:
        //   remain >= Bc * (per_bc + g_br * per_gbr_bc + Br * fp16_mask)
        // The Br * fp16 term accounts for the VTCM mask buffer [Br × Bc].
        const size_t remain   = usable - gbr_cost;
        const size_t bc_denom = per_bc + g_br * per_gbr_bc + Br * fp16;
        size_t       Bc       = hex_smin(hex_align_down(remain / bc_denom, bc_unit), Bc_limit);
        if (Bc < bc_unit) {
            if (Br == br_unit) {
                break;
            }
            continue;
        }

        // Exact VTCM verification (alignment padding may push over budget)
        while (Bc >= bc_unit && hmx_fa_compute_vtcm_usage(gqa_factor, DK, DV, Br, Bc, n_threads) > vtcm_budget) {
            Bc -= bc_unit;
        }
        if (Bc < bc_unit) {
            if (Br == br_unit) {
                break;
            }
            continue;
        }

        const size_t q_blocks  = (qo_len + Br - 1) / Br;
        const size_t kv_blocks = (kv_len + Bc - 1) / Bc;
        const size_t cost      = q_blocks * (c_q_fixed + kv_blocks * c_iter_fixed);
        const size_t mn        = Br * Bc;

        if (cost < best_cost || (cost == best_cost && mn > best_mn)) {
            best_cost = cost;
            best_mn   = mn;
            best_Br   = Br;
            best_Bc   = Bc;
        }

        if (Br == br_unit) {
            break;
        }
    }

    if (best_Br == 0) {
        return -1;
    }

    *Br_out = best_Br;
    *Bc_out = best_Bc;
    return 0;
}

// ============================================================================
// Tile interleave / extract helpers
// ============================================================================

// transpose scatter offsets moved to hmx-utils.h as hmx_transpose_scatter_offsets

// Scatter offsets for diagonal tile: entry[2i] = i*136, entry[2i+1] = i*136+6
// 136 = 4 * 32 + 8 = byte offset to diagonal in a 32x32 fp16 interleaved tile
static const int16_t d_tile_scatter_offsets[64] __attribute__((aligned(128))) = {
    0 * 136,  0 * 136 + 6,
    1 * 136,  1 * 136 + 6,
    2 * 136,  2 * 136 + 6,
    3 * 136,  3 * 136 + 6,
    4 * 136,  4 * 136 + 6,
    5 * 136,  5 * 136 + 6,
    6 * 136,  6 * 136 + 6,
    7 * 136,  7 * 136 + 6,
    8 * 136,  8 * 136 + 6,
    9 * 136,  9 * 136 + 6,
    10 * 136, 10 * 136 + 6,
    11 * 136, 11 * 136 + 6,
    12 * 136, 12 * 136 + 6,
    13 * 136, 13 * 136 + 6,
    14 * 136, 14 * 136 + 6,
    15 * 136, 15 * 136 + 6,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
};

// hmx_interleave_rows_to_tiles and hmx_interleave_cols_to_tiles are in hmx-utils.h

// ============================================================================
// HMX Flash Attention context (GQA-merged)
// ============================================================================

struct hmx_fa_context {
    const struct htp_ops_context * octx;
    bool         use_pipeline;  // true when n_kv_blocks >= FA_MIN_KV_BLOCKS && n_threads >= 2
    uint32_t     n_threads;

    // Op parameters
    float        scale;
    float        max_bias;
    float        logit_softcap;
    uint32_t     n_head_log2;
    float        m0, m1;

    // Dimensions
    uint32_t     DK, DV;
    uint32_t     n_kv;        // kv_len
    uint32_t     n_kv_heads;  // number of KV heads
    uint32_t     n_heads;     // number of Q heads
    uint32_t     G;           // GQA factor = n_heads / n_kv_heads
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
    __fp16 *     vtcm_v_tiles;         // V tiles (column-major)
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
    __fp16 *     vtcm_mask_buf;        // VTCM mask buffer [Br × m_line], DMA'd per KV block
    __fp16 *     vtcm_slopes;          // ALiBi slopes [g_br]
    size_t       row_buf_stride;       // HVX vectors per row buffer (Bc/64)
    size_t       mask_buf_row_stride;  // elements (__fp16) per row in mask buffer
    bool         mask_broadcast;       // true when mask->ne[2] == 1 (head-independent, single 2D DMA)
};

// ============================================================================
// Multi-thread K interleave phase
// ============================================================================

typedef struct {
    struct hmx_fa_context * factx;
    int                     kv_rows;
    size_t                  src_stride;
    size_t                  buf_idx;
} fa_k_int_args_t;

static void fa_k_interleave_thread(unsigned int n, unsigned int i, void * data) {
    fa_k_int_args_t *       args  = (fa_k_int_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const int total_rows = args->kv_rows;
    const int rows_per_t = hex_align_up(hmx_ceil_div(total_rows, n), 2);  // ensure even (row pairs)
    const int start      = i * rows_per_t;
    const int end        = hex_smin(start + rows_per_t, total_rows);

    if (start >= total_rows) {
        return;
    }

    hmx_interleave_rows_to_tiles(factx->vtcm_k_tiles, factx->vtcm_k_fp16[args->buf_idx], total_rows, (int) factx->DK,
                             (int) args->src_stride, start, end);
}

static void fa_phase_k_interleave(struct hmx_fa_context * factx, int kv_rows, size_t src_stride, size_t buf_idx) {
    worker_pool_context_t wp = factx->octx->ctx->worker_pool;
    fa_k_int_args_t args = { factx, kv_rows, src_stride, buf_idx };
    if (factx->n_threads > 1 && kv_rows >= (int) (factx->n_threads * 2)) {
        worker_pool_run_func(wp, fa_k_interleave_thread, &args, factx->n_threads);
    } else {
        fa_k_interleave_thread(1, 0, &args);
    }
}

// ============================================================================
// Multi-thread V interleave phase
// ============================================================================

typedef struct {
    struct hmx_fa_context * factx;
    int                     kv_rows;
    size_t                  src_stride;
    size_t                  buf_idx;
    size_t                  n_col_tiles;
} fa_v_int_args_t;

static void fa_v_interleave_thread(unsigned int n, unsigned int i, void * data) {
    fa_v_int_args_t *       args  = (fa_v_int_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const int total_rows = args->kv_rows;
    const int rows_per_t = hex_align_up(hmx_ceil_div(total_rows, n), 2);
    const int start      = i * rows_per_t;
    const int end        = hex_smin(start + rows_per_t, total_rows);

    if (start >= total_rows) {
        return;
    }

    hmx_interleave_cols_to_tiles(factx->vtcm_v_tiles, factx->vtcm_v_fp16[args->buf_idx], total_rows, (int) factx->DV,
                             (int) args->src_stride, (int) args->n_col_tiles, start, end);
}

static void fa_phase_v_interleave(struct hmx_fa_context * factx,
                                  int                     kv_rows,
                                  size_t                  src_stride,
                                  size_t                  buf_idx,
                                  size_t                  n_col_tiles) {
    worker_pool_context_t wp = factx->octx->ctx->worker_pool;
    fa_v_int_args_t args = { factx, kv_rows, src_stride, buf_idx, n_col_tiles };
    if (factx->n_threads > 1 && kv_rows >= (int) (factx->n_threads * 2)) {
        worker_pool_run_func(wp, fa_v_interleave_thread, &args, factx->n_threads);
    } else {
        fa_v_interleave_thread(1, 0, &args);
    }
}

// ============================================================================
// Multi-thread Q load phase: read Q[G × neq1, DK] from DDR, convert F32→F16
// (or deal F16 pairs), and write interleaved into vtcm_q_tiles.
// Each thread owns a disjoint range of row pairs; writes target distinct tile
// slots (r0 selects tile row, r1 selects intra-tile slot), so there is no
// write conflict.  Padding fill (when n_rows_g < g_br) is done single-threaded
// by the caller before dispatching.
// ============================================================================

typedef struct {
    struct hmx_fa_context *   factx;
    const struct htp_tensor * q;
    uint32_t                  q_start;
    uint32_t                  kv_head;
    uint32_t                  ib3;
    size_t                    n_rows_g;
} fa_q_load_args_t;

static void fa_q_load_thread(unsigned int n, unsigned int i, void * data) {
    fa_q_load_args_t *      args  = (fa_q_load_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const size_t n_rows_g = args->n_rows_g;
    const size_t G        = factx->G;
    const size_t DK       = factx->DK;

    // Partition row pairs across threads.  Keep each thread's start even so r/r+1
    // are always in the same thread's range.
    const size_t rows_per_t = hex_align_up(hmx_ceil_div(n_rows_g, n), 2);
    const size_t start      = (size_t) i * rows_per_t;
    const size_t end        = hex_smin(start + rows_per_t, n_rows_g);

    if (start >= n_rows_g) {
        return;
    }

    const struct htp_tensor * q       = args->q;
    const uint32_t            q_start = args->q_start;
    const uint32_t            kv_head = args->kv_head;
    const uint32_t            ib3     = args->ib3;

    for (size_t r = start; r < end; r += 2) {
        const bool next_row_valid = (r + 1) < n_rows_g;

        const size_t q_idx0 = (r + 0) / G;
        const size_t h_idx0 = (r + 0) % G;
        const size_t q_idx1 = (r + 1) / G;
        const size_t h_idx1 = (r + 1) % G;

        const uint8_t * q_ptr0 = (const uint8_t *) q->data + (q_start + q_idx0) * q->nb[1] +
                                                  (kv_head * G + h_idx0) * q->nb[2] + ib3 * q->nb[3];
        const uint8_t * q_ptr1 = next_row_valid ? ((const uint8_t *) q->data + (q_start + q_idx1) * q->nb[1] +
                                                  (kv_head * G + h_idx1) * q->nb[2] + ib3 * q->nb[3]) :
                                                  NULL;

        size_t   r0       = r / HMX_FP16_TILE_N_ROWS;
        size_t   r1       = r % HMX_FP16_TILE_N_ROWS;
        __fp16 * out_base = factx->vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ROWS * DK;

        if (factx->is_q_fp32) {
            const HVX_Vector * pv_in0 = (const HVX_Vector *) q_ptr0;
            const HVX_Vector * pv_in1 = q_ptr1 ? (const HVX_Vector *) q_ptr1 : NULL;

            for (uint32_t d = 0; d < DK / 32; ++d) {
                HVX_Vector v0   = pv_in0[d];
                HVX_Vector v1   = pv_in1 ? pv_in1[d] : Q6_V_vzero();
                HVX_Vector v_hf = hvx_vec_f32_to_f16_shuff(v0, v1);

                HVX_Vector * out_tile = (HVX_Vector *) (out_base + d * HMX_FP16_TILE_N_ELMS);
                out_tile[r1 / 2]      = v_hf;
            }
        } else {
            const HVX_Vector * pv_in0 = (const HVX_Vector *) q_ptr0;
            const HVX_Vector * pv_in1 = q_ptr1 ? (const HVX_Vector *) q_ptr1 : NULL;

            for (uint32_t d = 0; d < DK / 64; ++d) {
                HVX_Vector     v0 = pv_in0[d];
                HVX_Vector     v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
                HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

                __fp16 *     out_dual_tile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_Vector * pv_out0       = ((HVX_Vector *) out_dual_tile) + r1 / 2;
                HVX_Vector * pv_out1       = pv_out0 + 16;

                *pv_out0 = Q6_V_lo_W(vp);
                *pv_out1 = Q6_V_hi_W(vp);
            }
        }
    }
}

static void fa_phase_q_load(struct hmx_fa_context *   factx,
                            const struct htp_tensor * q,
                            uint32_t                  q_start,
                            uint32_t                  kv_head,
                            uint32_t                  ib3,
                            size_t                    n_rows_g) {
    worker_pool_context_t wp = factx->octx->ctx->worker_pool;
    fa_q_load_args_t args = { factx, q, q_start, kv_head, ib3, n_rows_g };
    // Require >= 2 row pairs per thread so partitioning is worthwhile.
    if (factx->n_threads > 1 && n_rows_g >= (size_t) (factx->n_threads * 2)) {
        worker_pool_run_func(wp, fa_q_load_thread, &args, factx->n_threads);
    } else {
        fa_q_load_thread(1, 0, &args);
    }
}

// ============================================================================
// Multi-thread O store phase: read O tiles from VTCM, convert F16->F32 (or
// deal F16 pairs), and write to strided DDR dst tensor.  Each thread owns a
// disjoint row range; writes target distinct dst rows (different q_idx/h_idx
// pairs produced by r/G and r%G), so there is no write conflict.
// ============================================================================

typedef struct {
    struct hmx_fa_context *   factx;
    const struct htp_tensor * dst;
    const __fp16 *            o_tile_src;
    uint32_t                  q_start;
    uint32_t                  kv_head;
    uint32_t                  ib3;
    size_t                    n_rows_g;
} fa_o_store_args_t;

static void fa_o_store_thread(unsigned int n, unsigned int i, void * data) {
    fa_o_store_args_t *     args  = (fa_o_store_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const size_t n_rows_g = args->n_rows_g;
    const size_t G        = factx->G;
    const size_t DV       = factx->DV;

    const size_t rows_per_t = hmx_ceil_div(n_rows_g, n);
    const size_t start      = (size_t) i * rows_per_t;
    const size_t end        = hex_smin(start + rows_per_t, n_rows_g);

    if (start >= n_rows_g) {
        return;
    }

    const struct htp_tensor * dst        = args->dst;
    const __fp16 *            o_tile_src = args->o_tile_src;
    const uint32_t            q_start    = args->q_start;
    const uint32_t            kv_head    = args->kv_head;
    const uint32_t            ib3        = args->ib3;

    for (size_t r = start; r < end; ++r) {
        const size_t q_idx = r / G;
        const size_t h_idx = r % G;

        // FIX(dst-indexing): ggml_flash_attn_ext() creates dst as permute(0,2,1,3) ->
        // [DV, n_heads, n_tokens, n_seq], so head stride is nb[1] and token stride is nb[2].
        uint8_t * dst_row = (uint8_t *) dst->data + (kv_head * G + h_idx) * dst->nb[1] +
                            (q_start + q_idx) * dst->nb[2] + ib3 * dst->nb[3];

        size_t         r0            = r / HMX_FP16_TILE_N_ROWS;
        size_t         r1            = r % HMX_FP16_TILE_N_ROWS;
        const __fp16 * tile_row_base = o_tile_src + r0 * HMX_FP16_TILE_N_ROWS * DV;

        if (factx->is_dst_fp32) {
            float * out = (float *) dst_row;
            for (uint32_t d = 0; d < DV / 32; ++d) {
                const HVX_Vector * in_tile = (const HVX_Vector *) (tile_row_base + d * HMX_FP16_TILE_N_ELMS);
                HVX_VectorPair     vp      = hvx_vec_f16_to_f32_shuff(in_tile[r1 / 2]);
                if (r1 % 2 == 0) {
                    *(HVX_UVector *) (out + d * 32) = Q6_V_lo_W(vp);
                } else {
                    *(HVX_UVector *) (out + d * 32) = Q6_V_hi_W(vp);
                }
            }
        } else {
            __fp16 * out = (__fp16 *) dst_row;
            for (uint32_t d = 0; d < DV / 64; ++d) {
                const __fp16 *     in_dual_tile = tile_row_base + d * HMX_FP16_TILE_N_ELMS * 2;
                const HVX_Vector * pv_in0       = ((const HVX_Vector *) in_dual_tile) + r1 / 2;
                const HVX_Vector * pv_in1       = pv_in0 + 16;
                HVX_VectorPair     vp           = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);
                if (r1 % 2 == 0) {
                    *(HVX_UVector *) (out + d * 64) = Q6_V_lo_W(vp);
                } else {
                    *(HVX_UVector *) (out + d * 64) = Q6_V_hi_W(vp);
                }
            }
        }
    }
}

static void fa_phase_o_store(struct hmx_fa_context *   factx,
                             const struct htp_tensor * dst,
                             const __fp16 *            o_tile_src,
                             uint32_t                  q_start,
                             uint32_t                  kv_head,
                             uint32_t                  ib3,
                             size_t                    n_rows_g) {
    worker_pool_context_t wp = factx->octx->ctx->worker_pool;
    fa_o_store_args_t args = { factx, dst, o_tile_src, q_start, kv_head, ib3, n_rows_g };
    if (factx->n_threads > 1 && n_rows_g >= (size_t) (factx->n_threads * 2)) {
        worker_pool_run_func(wp, fa_o_store_thread, &args, factx->n_threads);
    } else {
        fa_o_store_thread(1, 0, &args);
    }
}

// ============================================================================
// Multi-thread softmax phase + serial m/l update + build_D
// ============================================================================

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

    // ALiBi per-head slopes (indexed by GQA-merged row: slope[r] for r in [0, n_rows_g))
    // slope[r] = 1.0 when max_bias == 0 (no ALiBi)
    // Pointer into hmx_fa_context.vtcm_slopes (sized to g_br)
    __fp16 *                  slopes;

    // Mask info (preloaded before softmax)
    const struct htp_tensor * mask;
    const __fp16 *            mask_vtcm;             // VTCM mask buffer base (NULL = DDR fallback)
    size_t                    mask_vtcm_row_stride;  // elements (__fp16) per row in VTCM mask buffer
} fa_softmax_args_t;

static void fa_softmax_thread(unsigned int n, unsigned int i, void * data) {
    fa_softmax_args_t *     args  = (fa_softmax_args_t *) data;
    struct hmx_fa_context * factx = args->factx;

    const size_t n_rows_g       = args->n_rows_g;
    const size_t kv_rows        = args->kv_rows;
    const size_t Bc             = args->Bc;
    const size_t G              = args->G;
    const size_t n_tiles_per_bc = args->n_tiles_per_bc;
    const size_t n_row_vec_cnt  = hmx_ceil_div(n_rows_g, 64);

    // Partition r_vec_idx across threads
    const size_t vecs_per_t = hmx_ceil_div(n_row_vec_cnt, n);
    const size_t vec_start  = i * vecs_per_t;
    const size_t vec_end    = hex_smin(vec_start + vecs_per_t, n_row_vec_cnt);

    if (vec_start >= n_row_vec_cnt) {
        return;
    }

    // Per-thread row scratch: thread i uses bufs at offset i * 2 * stride
    const size_t row_buf_stride = factx->row_buf_stride;
    HVX_Vector * my_row_buf0    = factx->vtcm_row_bufs + i * 2 * row_buf_stride;
    HVX_Vector * my_row_buf1    = my_row_buf0 + row_buf_stride;

    const HVX_Vector v_neg_inf = Q6_Vh_vsplat_R(0xfbff);

    // Per-row accumulators: each fp16 lane in a 64-lane vector holds one row's scalar.
    // CONTRACT: lane bits must be IEEE fp16 (hf), never qf16 — qf16 uses a different
    // bit layout, so a later hf-domain read would silently produce wrong values.
    // Convert first via Q6_Vhf_equals_Vqf16(). For reference: vtcm_m_vec/vtcm_s_rowmax
    // are hf; vtcm_l_vec is qf16 — don't mix them up.

    for (size_t r_vec_idx = vec_start; r_vec_idx < vec_end; ++r_vec_idx) {
        HVX_Vector rowmax_acc_v = v_neg_inf;
        HVX_Vector rowsum_acc_v = Q6_V_vzero();
        HVX_Vector m_prev_v     = factx->vtcm_m_vec[r_vec_idx];

        for (int r_vec_off = 0; r_vec_off < 64; r_vec_off += 2) {
            int r = r_vec_idx * 64 + r_vec_off;
            if (r >= (int) hex_align_up(n_rows_g, 2)) {
                break;
            }

            int r0 = r / HMX_FP16_TILE_N_ROWS;
            int r1 = r % HMX_FP16_TILE_N_ROWS;

            const __fp16 * s_ld_base = factx->vtcm_s_tiles + r0 * HMX_FP16_TILE_N_ROWS * Bc;
            __fp16 *       p_st_base = factx->vtcm_p_tiles + r0 * HMX_FP16_TILE_N_ROWS * Bc;

            // Decode 2 rows from S tiles into per-thread row buffers
            HVX_Vector * pv_row_buf0 = my_row_buf0;
            HVX_Vector * pv_row_buf1 = my_row_buf1;
            for (size_t c = 0; c < kv_rows; c += 64) {
                const __fp16 *     in_dual_tile = s_ld_base + (c / 64) * HMX_FP16_TILE_N_ELMS * 2;
                const HVX_Vector * pv_s_in0     = ((const HVX_Vector *) in_dual_tile) + r1 / 2;
                const HVX_Vector * pv_s_in1     = pv_s_in0 + 16;

                HVX_VectorPair vp_s_dual_row = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
                *pv_row_buf0++               = Q6_V_lo_W(vp_s_dual_row);
                *pv_row_buf1++               = Q6_V_hi_W(vp_s_dual_row);
            }

            // Apply softcap if enabled (in F32 precision)
            if (factx->logit_softcap != 0.0f) {
                // When EXP2_HF is on, fold log2(e) into v_cap so the output lands in
                // log2(e)-scaled space for the downstream exp2.  log2(e) is kept OUT
                // of qk_scale in this configuration (see scale setup) so tanh sees
                // the physical QK/(√d·c) argument.
                float cap = factx->logit_softcap;
#ifdef HMX_FA_USE_EXP2_HF
                cap *= 1.44269504f;  // log2(e)
#endif
                const HVX_Vector v_cap = hvx_vec_splat_f32(cap);
                for (size_t c = 0; c < kv_rows; c += 64) {
                    size_t ci = c / 64;

                    HVX_VectorPair r0_f32 = hvx_vec_f16_to_f32(my_row_buf0[ci]);
                    HVX_Vector     t0_lo  = hvx_vec_tanh_f32(Q6_V_lo_W(r0_f32));
                    HVX_Vector     t0_hi  = hvx_vec_tanh_f32(Q6_V_hi_W(r0_f32));
                    t0_lo                 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(t0_lo, v_cap));
                    t0_hi                 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(t0_hi, v_cap));
                    my_row_buf0[ci]       = hvx_vec_f32_to_f16(t0_lo, t0_hi);

                    HVX_VectorPair r1_f32 = hvx_vec_f16_to_f32(my_row_buf1[ci]);
                    HVX_Vector     t1_lo  = hvx_vec_tanh_f32(Q6_V_lo_W(r1_f32));
                    HVX_Vector     t1_hi  = hvx_vec_tanh_f32(Q6_V_hi_W(r1_f32));
                    t1_lo                 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(t1_lo, v_cap));
                    t1_hi                 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(t1_hi, v_cap));
                    my_row_buf1[ci]       = hvx_vec_f32_to_f16(t1_lo, t1_hi);
                }
            }

            // Apply mask & compute rowmax(S)
            //
            // Optimizations over baseline:
            //   A. No-ALiBi fast path: when max_bias==0 (slope≡1.0), skip the
            //      slope multiplication — still add mask (additive bias) but
            //      avoid the mul_f16_f16.  Saves 2 ops/dual-row vs ALiBi path.
            //   B. GQA mask row dedup: G consecutive Q rows share one mask row
            //      (qi = r / G).  Reuse mask vector when qi is unchanged between
            //      row0 and row1 (saves ~75% of VTCM loads for G=4).

            // ALiBi slopes — only needed when has_alibi (scheme A)
            HVX_Vector v_slope0, v_slope1;
            if (args->has_alibi) {
                HVX_Vector v_s = hvx_vmemu(args->slopes + r);
                v_slope0 = hvx_vec_repl_f16(v_s);
                v_slope1 = (r + 1 < (int) n_rows_g) ? hvx_vec_repl_f16(Q6_V_vror_VR(v_s, 2)) : Q6_V_vzero();
            }

            const HVX_Vector v_threshold = Q6_Vh_vsplat_R(0xcc00);  // fp16 -16.0 (hoisted outside for-c)

            HVX_Vector v_s_rowmax0 = v_neg_inf;
            HVX_Vector v_s_rowmax1 = v_neg_inf;
            for (size_t c = 0; c < kv_rows; c += 64) {
                size_t         ci          = c / 64;
                const size_t   ne          = hex_smin(kv_rows - c, 64);
                HVX_VectorPred q_tail_keep = Q6_Q_vsetq2_R(ne * sizeof(__fp16));

                if (args->mask) {
                    HVX_Vector v_mask0, v_mask1;

                    if (args->mask_vtcm) {
                        // Read mask from VTCM buffer (DMA'd per KV block).
                        // GQA dedup (scheme B): skip load when qi unchanged.
                        const size_t qi0 = (r + 0) / G;
                        v_mask0 = *(const HVX_UVector *) (args->mask_vtcm + qi0 * args->mask_vtcm_row_stride + c);
                        v_mask1 = v_neg_inf;
                        if (r + 1 < (int) n_rows_g) {
                            const size_t qi1 = (r + 1) / G;
                            if (qi1 == qi0) {
                                v_mask1 = v_mask0;  // scheme B: reuse — same mask row
                            } else {
                                v_mask1 = *(const HVX_UVector *) (args->mask_vtcm + qi1 * args->mask_vtcm_row_stride + c);
                            }
                        }
                    } else {
                        // Fallback: read mask directly from DDR (when mask->ne[2] > 1).
                        const struct htp_tensor * mask   = args->mask;
                        const size_t              q_idx0 = args->q_start + ((r + 0) / G);
                        const size_t              h_idx0 = args->kv_head * G + (r + 0) % G;
                        const uint32_t            im2_0  = h_idx0 % mask->ne[2];
                        const uint32_t            im3_0  = args->ib3 % mask->ne[3];

                        const __fp16 * m0_ptr = (const __fp16 *) ((const uint8_t *) mask->data + q_idx0 * mask->nb[1] +
                                                        im2_0 * mask->nb[2] + im3_0 * mask->nb[3]) + args->kv_start + c;
                        v_mask0 = *(const HVX_UVector *) m0_ptr;
                        v_mask1 = v_neg_inf;

                        if (r + 1 < (int) n_rows_g) {
                            const size_t q_idx1 = args->q_start + ((r + 1) / G);
                            if (q_idx1 == q_idx0) {
                                // scheme B: same mask row in DDR path
                                v_mask1 = v_mask0;
                            } else {
                                const size_t   h_idx1 = args->kv_head * G + (r + 1) % G;
                                const uint32_t im2_1  = h_idx1 % mask->ne[2];
                                const uint32_t im3_1  = args->ib3 % mask->ne[3];
                                const __fp16 * m1_ptr = (const __fp16 *) ((const uint8_t *) mask->data + q_idx1 * mask->nb[1] +
                                                                im2_1 * mask->nb[2] + im3_1 * mask->nb[3]) + args->kv_start + c;
                                v_mask1 = *(const HVX_UVector *) m1_ptr;
                            }
                        }
                    }

                    // Threshold: mask values below -16.0 are treated as -inf (causal mask).
                    HVX_VectorPred q_keep0 = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask0, v_threshold), q_tail_keep);
                    HVX_VectorPred q_keep1 = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask1, v_threshold), q_tail_keep);

                    if (args->has_alibi) {
                        // ALiBi path: S += slope * mask (full mul + add)
                        HVX_Vector v_sm0 = hvx_vec_mul_f16_f16(v_mask0, v_slope0);
                        HVX_Vector v_sm1 = hvx_vec_mul_f16_f16(v_mask1, v_slope1);
                        my_row_buf0[ci]  = Q6_V_vmux_QVV(q_keep0, hvx_vec_add_f16_f16(my_row_buf0[ci], v_sm0), v_neg_inf);
                        my_row_buf1[ci]  = Q6_V_vmux_QVV(q_keep1, hvx_vec_add_f16_f16(my_row_buf1[ci], v_sm1), v_neg_inf);
                    } else {
                        // No-ALiBi fast path (scheme A): slope≡1.0, skip the mul
                        // but still add mask (additive positional bias).  vmux
                        // clamps mask < -16 to -inf as a numerical safeguard.
                        my_row_buf0[ci] = Q6_V_vmux_QVV(q_keep0, hvx_vec_add_f16_f16(my_row_buf0[ci], v_mask0), v_neg_inf);
                        my_row_buf1[ci] = Q6_V_vmux_QVV(q_keep1, hvx_vec_add_f16_f16(my_row_buf1[ci], v_mask1), v_neg_inf);
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

            // Splat m_prev[r], m_prev[r+1] from the per-row accumulator.
            // vror brings the target lane to lane 0, then extract + re-splat.
            HVX_Vector v_m_prev0 = hvx_vec_splat_f16(hvx_vec_get_f16(Q6_V_vror_VR(m_prev_v, r_vec_off * 2)));
            HVX_Vector v_m_prev1 = hvx_vec_splat_f16(hvx_vec_get_f16(Q6_V_vror_VR(m_prev_v, (r_vec_off + 1) * 2)));

            // HVX max — both operands are splats, so result is splat of m_new.
            HVX_Vector v_dup_m0 = Q6_Vhf_vmax_VhfVhf(v_m_prev0, v_s_rowmax0);
            HVX_Vector v_dup_m1 = Q6_Vhf_vmax_VhfVhf(v_m_prev1, v_s_rowmax1);

            // Insert row r, r+1 rowmax into rowmax_acc_v via 2-byte-wide vmux.
            // Byte ranges: lane0 = [r_vec_off*2 .. r_vec_off*2+1], lane1 shifted by 2.
            // vsetq2 handles the n=128 corner case when r_vec_off reaches 62.
            {
                HVX_VectorPred p_start = Q6_Q_vsetq_R(r_vec_off * 2);
                HVX_VectorPred p_mid   = Q6_Q_vsetq_R((r_vec_off + 1) * 2);
                HVX_VectorPred p_end   = Q6_Q_vsetq2_R((r_vec_off + 2) * 2);
                HVX_VectorPred p_lane0 = Q6_Q_and_QQn(p_mid, p_start);
                HVX_VectorPred p_lane1 = Q6_Q_and_QQn(p_end, p_mid);
                rowmax_acc_v           = Q6_V_vmux_QVV(p_lane0, v_dup_m0, rowmax_acc_v);
                rowmax_acc_v           = Q6_V_vmux_QVV(p_lane1, v_dup_m1, rowmax_acc_v);
            }

            // Compute P = exp(S - m_new), using HVX exp
            const HVX_Vector v_zero      = Q6_V_vzero();
            HVX_Vector       v_p_rowsum0 = v_zero;
            HVX_Vector       v_p_rowsum1 = v_zero;

#ifdef HMX_FA_USE_EXP2_HF
            // FP16 exp2 polynomial path (matches htp-ops-lib flash_attn.c):
            // P = exp2(S - m_new)
            for (size_t c = 0; c < kv_rows; c += 64) {
                size_t     ci           = c / 64;
                HVX_Vector v_s_minus_m0 = Q6_Vqf16_vsub_VhfVhf(my_row_buf0[ci], v_dup_m0);
                HVX_Vector v_s_minus_m1 = Q6_Vqf16_vsub_VhfVhf(my_row_buf1[ci], v_dup_m1);

                HVX_Vector v_p_row0_hf  = hvx_exp2_hf(Q6_Vhf_equals_Vqf16(v_s_minus_m0));
                HVX_Vector v_p_row1_hf  = hvx_exp2_hf(Q6_Vhf_equals_Vqf16(v_s_minus_m1));
#else
            // F32 exp path: qf16 → f32 → exp → f32 → f16.  Higher precision,
            for (size_t c = 0; c < kv_rows; c += 64) {
                size_t     ci           = c / 64;
                HVX_Vector v_s_minus_m0 = Q6_Vqf16_vsub_VhfVhf(my_row_buf0[ci], v_dup_m0);
                HVX_Vector v_s_minus_m1 = Q6_Vqf16_vsub_VhfVhf(my_row_buf1[ci], v_dup_m1);

                HVX_VectorPair vp0         = hvx_vec_f16_to_f32_shuff(Q6_Vhf_equals_Vqf16(v_s_minus_m0));
                HVX_Vector     p0_lo       = hvx_vec_exp_f32(Q6_V_lo_W(vp0));
                HVX_Vector     p0_hi       = hvx_vec_exp_f32(Q6_V_hi_W(vp0));
                HVX_Vector     v_p_row0_hf = hvx_vec_f32_to_f16_shuff(p0_lo, p0_hi);

                HVX_VectorPair vp1         = hvx_vec_f16_to_f32_shuff(Q6_Vhf_equals_Vqf16(v_s_minus_m1));
                HVX_Vector     p1_lo       = hvx_vec_exp_f32(Q6_V_lo_W(vp1));
                HVX_Vector     p1_hi       = hvx_vec_exp_f32(Q6_V_hi_W(vp1));
                HVX_Vector     v_p_row1_hf = hvx_vec_f32_to_f16_shuff(p1_lo, p1_hi);
#endif
                // Write P to tile format.  Dual-tile pattern assumes Bc is a
                // multiple of 64 (enforced by bc_unit=64 in hmx_fa_find_chunk_size),
                // so both tile halves are always in the current r0 block.
                __fp16 *     out_dual_tile = p_st_base + (c / 64) * HMX_FP16_TILE_N_ELMS * 2;
                HVX_Vector * pv_p_out0     = ((HVX_Vector *) out_dual_tile) + r1 / 2;
                HVX_Vector * pv_p_out1     = pv_p_out0 + 16;

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
                // Both inputs are f32 splats, so the f32->f16 output is an fp16 splat.
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

        factx->vtcm_s_rowmax[r_vec_idx] = rowmax_acc_v;
        factx->vtcm_p_rowsum[r_vec_idx] = rowsum_acc_v;
    }
}

// Serial m/l update + build_D.  Must run after softmax barrier (s_rowmax written by all threads).
//
// noinline: function boundary acts as a hard compiler barrier so the (size_t)addr scatter
// intrinsics inside cannot be hoisted past the call site.  Mirrors the structural protection
// matmul gets for free via worker_pool function-pointer dispatch.  Without this, the compiler
// can reorder the scatter past the subsequent hmx_queue_push and the HMX-queue worker thread
// reads stale VTCM (PPL → ~vocab-size).
static __attribute__((noinline)) void fa_ml_update_and_build_d(struct hmx_fa_context * factx,
                                                               size_t                  n_rows_g,
                                                               size_t                  n_row_tiles,
                                                               size_t                  n_row_tiles_g_br) {
    // Reuse s_rowmax buffer for exp(m_diff) — safe because softmax is fully complete
    HVX_Vector * const mvec_exp_m_diff = factx->vtcm_s_rowmax;

    const size_t n_row_vec_cnt = hmx_ceil_div(n_rows_g, 64);
    for (size_t i = 0; i < n_row_vec_cnt; ++i) {
        HVX_Vector v_m_prev = factx->vtcm_m_vec[i];
        HVX_Vector v_m_curr = Q6_Vhf_vmax_VhfVhf(v_m_prev, factx->vtcm_s_rowmax[i]);
        HVX_Vector v_m_diff = Q6_Vqf16_vsub_VhfVhf(v_m_prev, v_m_curr);

#ifdef HMX_FA_USE_EXP2_HF
        // Base-2 path: must match P = exp2(S - m_new) in fa_softmax_thread.
        HVX_Vector v_exp_m_diff      = hvx_exp2_hf(Q6_Vhf_equals_Vqf16(v_m_diff));
#else
        HVX_VectorPair vp_diff       = hvx_vec_f16_to_f32_shuff(Q6_Vhf_equals_Vqf16(v_m_diff));
        HVX_Vector     exp_lo        = hvx_vec_exp_f32(Q6_V_lo_W(vp_diff));
        HVX_Vector     exp_hi        = hvx_vec_exp_f32(Q6_V_hi_W(vp_diff));
        HVX_Vector     v_exp_m_diff  = hvx_vec_f32_to_f16_shuff(exp_lo, exp_hi);
#endif

        HVX_Vector v_l_curr = Q6_Vqf16_vmpy_Vqf16Vhf(factx->vtcm_l_vec[i], v_exp_m_diff);
        v_l_curr            = Q6_Vqf16_vadd_Vqf16Vhf(v_l_curr, factx->vtcm_p_rowsum[i]);

        factx->vtcm_m_vec[i] = v_m_curr;
        factx->vtcm_l_vec[i] = v_l_curr;
        mvec_exp_m_diff[i]   = v_exp_m_diff;
    }

    // Build diagonal tile D = diag(exp(m_diff))
    const HVX_Vector     v_offsets = *(const HVX_Vector *) d_tile_scatter_offsets;
    const HVX_VectorPred q_32_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));
    for (size_t i = 0; i < n_row_tiles; ++i) {
        const HVX_Vector v_content = Q6_V_vror_VR(mvec_exp_m_diff[i / 2], (i % 2) * 64);
        __fp16 *         out_base  = factx->vtcm_d_tiles + i * (n_row_tiles_g_br + 1) * HMX_FP16_TILE_N_ELMS;
        Q6_vscatter_QRMVhV(q_32_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
        // Compiler barrier — Q6_vscatter takes (size_t)addr; without this the
        // compiler may not recognize the volatile read below as aliasing and
        // could reorder it before the scatter, defeating the HW drain.
        __asm__ __volatile__("" ::: "memory");
        // Per-tile drain: scatter regions are disjoint (stride > tile size),
        // so a single drain at tile 0 does NOT retire later tiles' entries.
        (void) *(volatile HVX_Vector *) out_base;
    }
}

// Build D = diag(1/l) tile for the final O = D @ O normalization.
//
// noinline: same rationale as fa_ml_update_and_build_d — keeps Q6_vscatter from
// being hoisted past the subsequent hmx_queue_push at the o_norm call site.
static __attribute__((noinline)) void fa_build_d_diag_inv_l(struct hmx_fa_context * factx,
                                                            size_t                  n_row_tiles,
                                                            size_t                  n_row_tiles_g_br) {
    const HVX_Vector     v_offsets = *(const HVX_Vector *) d_tile_scatter_offsets;
    const HVX_VectorPred q_32_mask = Q6_Q_vsetq_R(32 * sizeof(__fp16));
    const HVX_Vector     one       = hvx_vec_splat_f32(1.0f);

    HVX_Vector v_content = Q6_V_vzero();
    for (size_t i = 0; i < n_row_tiles; ++i) {
        if ((i % 2) == 0) {
            HVX_Vector     v_l_hf = Q6_Vhf_equals_Vqf16(factx->vtcm_l_vec[i / 2]);
            HVX_VectorPair vp_l   = hvx_vec_f16_to_f32_shuff(v_l_hf);
            HVX_Vector     inv_lo = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(one, hvx_vec_inverse_f32(Q6_V_lo_W(vp_l))));
            HVX_Vector     inv_hi = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(one, hvx_vec_inverse_f32(Q6_V_hi_W(vp_l))));
            v_content = hvx_vec_f32_to_f16_shuff(inv_lo, inv_hi);
        } else {
            v_content = Q6_V_vror_VR(v_content, 64);
        }

        __fp16 * out_base = factx->vtcm_d_tiles + i * (n_row_tiles_g_br + 1) * HMX_FP16_TILE_N_ELMS;
        Q6_vscatter_QRMVhV(q_32_mask, (size_t) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
        // Compiler barrier — see fa_ml_update_and_build_d for rationale.
        __asm__ __volatile__("" ::: "memory");
        (void) *(volatile HVX_Vector *) out_base;
    }
}

// Combined: multi-thread softmax -> barrier -> serial m/l update + build_D
static void fa_phase_softmax_and_build_d(struct hmx_fa_context * factx,
                                         fa_softmax_args_t *     sargs,
                                         size_t                  n_row_tiles,
                                         size_t                  n_row_tiles_g_br) {
    worker_pool_context_t wp = factx->octx->ctx->worker_pool;
    const size_t n_row_vec_cnt = hmx_ceil_div(sargs->n_rows_g, 64);

    if (factx->n_threads > 1 && n_row_vec_cnt >= 2) {
        uint32_t n_use = (uint32_t) hex_smin((size_t) factx->n_threads, n_row_vec_cnt);
        worker_pool_run_func(wp, fa_softmax_thread, sargs, n_use);
    } else {
        fa_softmax_thread(1, 0, sargs);
    }
    // barrier implicit in worker_pool_run_func return

    fa_ml_update_and_build_d(factx, sargs->n_rows_g, n_row_tiles, n_row_tiles_g_br);
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

    Q6_bias_mxmem2_A((void *) job->hmx_scales);
    for (size_t r = 0; r < n_row_tiles; ++r) {
        for (size_t c = 0; c < n_col_tiles; ++c) {
            const __fp16 * row_tiles = q_tiles + r * HMX_FP16_TILE_N_ROWS * n_dot_tiles * HMX_FP16_TILE_N_COLS;
            const __fp16 * col_tiles = k_tiles + c * HMX_FP16_TILE_N_COLS * n_dot_tiles * HMX_FP16_TILE_N_COLS;
            __fp16 *       out_tile  = s_tiles + (r * n_tiles_per_bc + c) * HMX_FP16_TILE_N_ELMS;

            for (size_t k = 0; k < n_dot_tiles; ++k) {
                Q6_activation_hf_mxmem_RR((unsigned int) row_tiles, 2047);
                Q6_weight_hf_mxmem_RR((unsigned int) col_tiles, 2047);
                row_tiles += HMX_FP16_TILE_N_ELMS;
                col_tiles += HMX_FP16_TILE_N_ELMS;
            }
            Q6_mxmem_AR_after_hf(out_tile, 0);
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

    Q6_bias_mxmem2_A((void *) job->hmx_scales);
    for (size_t r = 0; r < n_row_tiles; ++r) {
        for (size_t c = 0; c < DV_tiles; ++c) {
            // D[r,r] @ O_prev[r,c] — only the diagonal tile
            const __fp16 * d_diag = d_tiles + r * (n_row_tiles_g_br + 1) * HMX_FP16_TILE_N_ELMS;
            const __fp16 * o_rc   = o_prev + (c * n_row_tiles_g_br + r) * HMX_FP16_TILE_N_ELMS;
            Q6_activation_hf_mxmem_RR((unsigned int) d_diag, 2047);
            Q6_weight_hf_mxmem_RR((unsigned int) o_rc, 2047);

            // P @ V (accumulate on same accumulator)
            const __fp16 * p_tile_in = p_tiles + (r * n_tiles_per_bc) * HMX_FP16_TILE_N_ELMS;
            const __fp16 * v_tile_in = v_tiles + (c * n_tiles_per_bc) * HMX_FP16_TILE_N_ELMS;
            for (size_t k = 0; k < n_col_tiles; ++k) {
                Q6_activation_hf_mxmem_RR((unsigned int) p_tile_in, 2047);
                Q6_weight_hf_mxmem_RR((unsigned int) v_tile_in, 2047);
                p_tile_in += HMX_FP16_TILE_N_ELMS;
                v_tile_in += HMX_FP16_TILE_N_ELMS;
            }

            __fp16 * o_tile_out = o_curr + (c * n_row_tiles_g_br + r) * HMX_FP16_TILE_N_ELMS;
            Q6_mxmem_AR_after_hf(o_tile_out, 0);
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

    Q6_bias_mxmem2_A((void *) job->hmx_scales);
    for (size_t r = 0; r < n_row_tiles; ++r) {
        for (size_t c = 0; c < DV_tiles; ++c) {
            const __fp16 * d_diag = d_tiles + r * (n_row_tiles_g_br + 1) * HMX_FP16_TILE_N_ELMS;
            const __fp16 * o_rc   = o_prev + (c * n_row_tiles_g_br + r) * HMX_FP16_TILE_N_ELMS;
            __fp16 *       o_out  = o_curr + (r * DV_tiles + c) * HMX_FP16_TILE_N_ELMS;

            Q6_activation_hf_mxmem_RR((unsigned int) d_diag, 2047);
            Q6_weight_hf_mxmem_RR((unsigned int) o_rc, 2047);
            Q6_mxmem_AR_after_hf(o_out, 0);
        }
    }
}

// Populate per-GQA-row ALiBi slopes for a given KV head.
// Row r in the GQA-merged block maps to Q head h = kv_head * G + r % G.
// slope(h) = m0^(h+1) when h < n_head_log2, else m1^(2*(h-n_head_log2)+1).
// When max_bias == 0, all slopes are 1.0 (no ALiBi).
static __attribute__((noinline)) void fa_compute_slopes(fa_softmax_args_t * sargs,
                              const struct hmx_fa_context * factx,
                              uint32_t                      kv_head,
                              size_t                        n_rows_g) {
    if (factx->max_bias == 0.0f) {
        for (size_t r = 0; r < n_rows_g; ++r) {
            sargs->slopes[r] = 1.0f;
        }
        return;
    }

    const uint32_t G           = factx->G;
    const uint32_t n_head_log2 = factx->n_head_log2;
    const float    m0          = factx->m0;
    const float    m1          = factx->m1;

    for (size_t r = 0; r < n_rows_g; ++r) {
        const uint32_t h = kv_head * G + r % G;
        sargs->slopes[r] = (h < n_head_log2) ? powf(m0, h + 1) : powf(m1, 2 * (h - n_head_log2) + 1);
    }
}

// ============================================================================
// Core HMX flash attention algorithm (GQA-merged)
// ============================================================================

int hmx_flash_attn_ext(struct htp_ops_context * octx) {
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
    if (neq1 < 32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    // GQA factor
    const uint32_t n_kv_heads = k->ne[2];
    const uint32_t G          = neq2 / n_kv_heads;

    // Thread count for multi-thread HVX phases
    const uint32_t n_threads = octx->n_threads;

    // Compute dynamic block sizes (GQA-aware, accounting for per-thread row bufs)
    size_t       Br, Bc;
    const size_t vtcm_budget = ctx->vtcm_size;
    if (hmx_fa_find_chunk_size(&Br, &Bc, G, DK, DV, neq1, nek1, vtcm_budget, n_threads) != 0) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    const size_t g_br = hex_align_up(G * Br, HMX_FP16_TILE_N_ROWS);

    const uint32_t n_kv_blocks  = (nek1 + Bc - 1) / Bc;
    const bool     use_pipeline = (n_kv_blocks >= FA_MIN_KV_BLOCKS && n_threads >= 2);

    FARF(HIGH, "hmx-fa: neq1=%u nek1=%u DK=%u DV=%u G=%u Br=%zu Bc=%zu g_br=%zu n_kv_blocks=%u pipeline=%d vtcm=%zu",
         neq1, nek1, DK, DV, G, Br, Bc, g_br, n_kv_blocks, use_pipeline, vtcm_budget);

    // ======== Build context ========
    struct hmx_fa_context factx;
    memset(&factx, 0, sizeof(factx));
    factx.octx           = octx;
    factx.n_threads      = octx->ctx->n_threads;
    factx.DK             = DK;
    factx.DV             = DV;
    factx.n_kv           = nek1;
    factx.n_kv_heads     = n_kv_heads;
    factx.n_heads        = neq2;
    factx.G              = G;
    factx.neq1           = neq1;
    factx.Br             = (uint32_t) Br;
    factx.Bc             = (uint32_t) Bc;
    factx.g_br           = (uint32_t) g_br;
    factx.n_kv_blocks    = n_kv_blocks;
    factx.is_q_fp32      = (q->type == HTP_TYPE_F32);
    factx.is_dst_fp32    = (dst->type == HTP_TYPE_F32);
    factx.use_pipeline   = use_pipeline;
    factx.mask_broadcast = (mask != NULL && mask->ne[2] == 1);

    // Extract op parameters (mutable during softcap adjustment, then stored as const in factx)
    float scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&scale, (float *) octx->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) octx->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (float *) octx->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

#ifdef HMX_FA_USE_EXP2_HF
    // Pre-bake log2(e) into qk_scale so HMX-produced S tiles are in log2(e)-scaled
    // space.  Then exp2(S - m) in the softmax equals base-e exp((S - m) / log2(e)),
    // preserving ggml's base-e softmax semantics.  Matches htp-ops-lib flash_attn.c.
    //
    // When softcap is active we cannot pre-bake log2(e) here — it would land inside
    // the tanh argument and shift the softcap knee from x≈c to x≈c/log2(e), giving
    // numerically wrong softcapped values.  Instead fold log2(e) into the post-tanh
    // multiplier (see softcap block: v_cap absorbs log2(e)).
    if (logit_softcap == 0.0f) {
        scale *= 1.44269504f;  // log2(e)
    }
#endif

    factx.scale         = scale;
    factx.max_bias      = max_bias;
    factx.logit_softcap = logit_softcap;

    factx.n_head_log2 = 1u << (uint32_t) floor(log2(neq2));
    factx.m0          = powf(2.0f, -(max_bias) / factx.n_head_log2);
    factx.m1          = powf(2.0f, -(max_bias / 2.0f) / factx.n_head_log2);

    // ======== VTCM allocation (GQA-aware) ========
    const size_t q_tile_bytes  = hex_align_up(g_br * DK * sizeof(__fp16), 4096);
    const size_t o_tile_bytes  = hex_align_up(g_br * DV * sizeof(__fp16), 4096);
    const size_t k_dma_bytes   = hex_align_up(Bc * DK * sizeof(__fp16), 4096);
    const size_t v_dma_bytes   = hex_align_up(Bc * DV * sizeof(__fp16), 4096);
    const size_t k_tile_bytes  = hex_align_up(Bc * DK * sizeof(__fp16), 4096);
    const size_t v_tile_bytes  = hex_align_up(Bc * DV * sizeof(__fp16), 4096);
    const size_t s_tile_bytes  = hex_align_up(g_br * Bc * sizeof(__fp16), 4096);
    const size_t d_tile_bytes  = hex_align_up(g_br * g_br * sizeof(__fp16), 4096);
    const size_t col_vec_bytes = hex_align_up(g_br * sizeof(__fp16), 256);
    const size_t row_vec_bytes = hex_align_up(Bc * sizeof(__fp16), 256);
    const size_t m_line_bytes  = hex_align_up(Bc * sizeof(__fp16), 128);
    const size_t m_buf_bytes   = hex_align_up(Br * m_line_bytes, 4096);
    const size_t slopes_bytes  = hex_align_up(g_br * sizeof(__fp16), 128);

    uint8_t * vtcm_cur = ctx->vtcm_base;

    factx.vtcm_q_tiles        = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, q_tile_bytes);
    factx.vtcm_o_tiles[0]     = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, o_tile_bytes);
    factx.vtcm_o_tiles[1]     = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, o_tile_bytes);
    factx.vtcm_k_fp16[0]      = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, k_dma_bytes);
    factx.vtcm_k_fp16[1]      = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, k_dma_bytes);
    factx.vtcm_v_fp16[0]      = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, v_dma_bytes);
    factx.vtcm_v_fp16[1]      = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, v_dma_bytes);
    factx.vtcm_k_tiles        = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, k_tile_bytes);
    factx.vtcm_v_tiles        = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, v_tile_bytes);
    factx.vtcm_s_tiles        = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, s_tile_bytes);
    factx.vtcm_p_tiles        = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, s_tile_bytes);
    factx.vtcm_d_tiles        = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, d_tile_bytes);
    factx.vtcm_m_vec          = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_bytes);
    factx.vtcm_l_vec          = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_bytes);
    factx.vtcm_s_rowmax       = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_bytes);
    factx.vtcm_p_rowsum       = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, col_vec_bytes);
    factx.vtcm_row_bufs       = (HVX_Vector *) vtcm_seq_alloc(&vtcm_cur, row_vec_bytes * 2 * n_threads);
    factx.row_buf_stride      = row_vec_bytes / sizeof(HVX_Vector);
    factx.vtcm_hmx_scales_id  = vtcm_seq_alloc(&vtcm_cur, 256);
    factx.vtcm_hmx_scales_qk  = vtcm_seq_alloc(&vtcm_cur, 256);
    factx.vtcm_mask_buf       = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, m_buf_bytes);
    factx.mask_buf_row_stride = m_line_bytes / sizeof(__fp16);
    factx.vtcm_slopes         = (__fp16 *) vtcm_seq_alloc(&vtcm_cur, slopes_bytes);

    if ((size_t) (vtcm_cur - ctx->vtcm_base) > ctx->vtcm_size) {
        return HTP_STATUS_VTCM_TOO_SMALL;
    }

    // ======== Initialize HMX output scales ========
    // Identity scale (1.0) for O updates and normalization
    hmx_init_column_scales(factx.vtcm_hmx_scales_id, Q6_V_vsplat_R(0x3c00)); // 1.0

    // QK scale embedded in HMX output
    hmx_init_column_scales(factx.vtcm_hmx_scales_qk, hvx_vec_splat_f16(factx.scale));

    // ======== Skip compute if profiling ========
    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    // Profiling timers
    TIMER_DEFINE(total);
    TIMER_DEFINE(q_load);
    TIMER_DEFINE(kv_dma);
    TIMER_DEFINE(k_interleave);
    TIMER_DEFINE(v_interleave);
    TIMER_DEFINE(qk_dot);
    TIMER_DEFINE(softmax);
    TIMER_DEFINE(o_update);
    TIMER_DEFINE(o_norm);
    TIMER_DEFINE(o_store);

    TIMER_START(total);

    // ======== DMA setup ========
    dma_queue * const dma = ctx->dma[0];

    // Padded row sizes for DMA
    const size_t size_k_row        = nek0 * sizeof(__fp16);
    const size_t size_v_row        = nev0 * sizeof(__fp16);
    const size_t size_k_row_padded = hex_round_up(nek0 * sizeof(__fp16), 128);
    const size_t size_v_row_padded = hex_round_up(nev0 * sizeof(__fp16), 128);

    const size_t n_row_tiles_g_br = g_br / HMX_FP16_TILE_N_ROWS;
    const size_t n_tiles_per_bc   = Bc / HMX_FP16_TILE_N_COLS;

    // Q/O element size for Q load and O store
    const size_t qo_element_size = factx.is_q_fp32 ? sizeof(float) : sizeof(__fp16);

    // ======== HMX lock strategy ========
    // Pipeline: queue thread auto-acquires HMX lock on first push; released by suspend.
    // Fallback: main thread holds the lock (original behavior).
    if (!factx.use_pipeline) {
        HAP_compute_res_hmx_lock(ctx->vtcm_rctx);
    }

    // ======== Reusable job descriptors for pipeline ========
    hmx_fa_qk_job_t       qk_job;
    hmx_fa_o_update_job_t ou_job;
    hmx_fa_o_norm_job_t   on_job;

    // ======== Main loop: per batch, per KV head, per Q block ========
    for (uint32_t ib3 = 0; ib3 < neq3; ++ib3) {
        for (uint32_t kv_head = 0; kv_head < n_kv_heads; ++kv_head) {
            const uint32_t ik2 = kv_head;
            const uint32_t ik3 = ib3 / (neq3 / k->ne[3]);
            const uint32_t iv2 = kv_head;
            const uint32_t iv3 = ib3 / (neq3 / v->ne[3]);

            for (uint32_t q_start = 0; q_start < neq1; q_start += Br) {
                const uint32_t n_q_rows    = hex_smin(Br, neq1 - q_start);
                const size_t   n_rows_g    = n_q_rows * G;
                const size_t   g_br_actual = hex_align_up(n_rows_g, HMX_FP16_TILE_N_ROWS);
                const size_t   n_row_tiles = g_br_actual / HMX_FP16_TILE_N_ROWS;

                // ---- Load Q block [g_br, D] -> tiles, interleaving G heads ----
                TIMER_START(q_load);
                if (n_rows_g < g_br) {
                    hvx_splat_u8_a(factx.vtcm_q_tiles, 0, q_tile_bytes);
                }
                fa_phase_q_load(&factx, q, q_start, kv_head, ib3, n_rows_g);
                TIMER_STOP(q_load);

                // ---- Initialize per-block state ----
                hvx_splat_u8_a(factx.vtcm_l_vec,   0,      col_vec_bytes);
                hvx_splat_u8_a(factx.vtcm_d_tiles, 0,      d_tile_bytes);
                hvx_splat_u16_a(factx.vtcm_m_vec,  0xfbff, col_vec_bytes/2);

                __fp16 * o_tile_prev = factx.vtcm_o_tiles[0];
                __fp16 * o_tile_curr = factx.vtcm_o_tiles[1];
                hvx_splat_u8_a(o_tile_prev, 0, o_tile_bytes);

                // ---- KV block loop with DMA double-buffering ----
                size_t buf_idx = 0;

                // Prefetch first KV block
                if (factx.n_kv_blocks > 0) {
                    const uint32_t kv_rows0 = hex_smin(Bc, nek1);

                    const uint8_t * k_src = (const uint8_t *) k->data + ik2 * k->nb[2] + ik3 * k->nb[3];
                    dma_queue_push(dma, dma_make_ptr(factx.vtcm_k_fp16[0], k_src), size_k_row_padded, k->nb[1],
                                   size_k_row, kv_rows0);

                    const uint8_t * v_src = (const uint8_t *) v->data + iv2 * v->nb[2] + iv3 * v->nb[3];
                    dma_queue_push(dma, dma_make_ptr(factx.vtcm_v_fp16[0], v_src), size_v_row_padded, v->nb[1],
                                   size_v_row, kv_rows0);
                }

                // Mask DMA: single 2D transfer of n_q_rows unique mask rows into VTCM buffer.
                // Only when mask is head-broadcast (ne[2]==1); otherwise softmax reads DDR directly.
                #define MASK_DMA_PUSH(kv_start_val, kv_rows_val, has_mask_dma_var)                                             \
                    do {                                                                                                       \
                        has_mask_dma_var = false;                                                                              \
                        if (mask && factx.mask_broadcast) {                                                                    \
                            const uint32_t  _im3 = ib3 % mask->ne[3];                                                          \
                            const uint8_t * _ms  = (const uint8_t *) mask->data + q_start * mask->nb[1] + _im3 * mask->nb[3] + \
                                                  (kv_start_val) * sizeof(__fp16);                                             \
                            dma_queue_push(dma, dma_make_ptr(factx.vtcm_mask_buf, _ms), m_line_bytes, mask->nb[1],             \
                                           (kv_rows_val) * sizeof(__fp16), n_q_rows);                                          \
                            has_mask_dma_var = true;                                                                           \
                        }                                                                                                      \
                    } while (0)

                #define MASK_DMA_POP(has_mask_dma_var) \
                    do {                               \
                        if (has_mask_dma_var) {        \
                            dma_queue_pop(dma);        \
                        }                              \
                    } while (0)

                #define DMA_PREFETCH_KV(blk_val)                                                                                          \
                    do {                                                                                                                  \
                        if ((blk_val) < factx.n_kv_blocks) {                                                                              \
                            const uint32_t  _ns = (blk_val) * Bc;                                                                         \
                            const uint32_t  _nr = hex_smin(Bc, nek1 - _ns);                                                               \
                            size_t          _nb = 1 - buf_idx;                                                                            \
                            const uint8_t * _ks = (const uint8_t *) k->data + _ns * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];           \
                            dma_queue_push(dma, dma_make_ptr(factx.vtcm_k_fp16[_nb], _ks), size_k_row_padded, k->nb[1], size_k_row, _nr); \
                            const uint8_t * _vs = (const uint8_t *) v->data + _ns * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];           \
                            dma_queue_push(dma, dma_make_ptr(factx.vtcm_v_fp16[_nb], _vs), size_v_row_padded, v->nb[1], size_v_row, _nr); \
                        }                                                                                                                 \
                    } while (0)

                const size_t k_src_stride = size_k_row_padded / sizeof(__fp16);
                const size_t v_src_stride = size_v_row_padded / sizeof(__fp16);

                if (factx.use_pipeline) {
                    // ==================================================================
                    // Pipeline path: HVX phases ‖ HMX queue worker
                    // ==================================================================
                    struct hmx_queue * hmx_q = ctx->hmx_queue;

                    for (uint32_t kv_blk = 0; kv_blk < factx.n_kv_blocks; ++kv_blk) {
                        const uint32_t kv_start    = kv_blk * Bc;
                        const uint32_t kv_rows     = hex_smin(Bc, nek1 - kv_start);
                        const size_t   n_col_tiles = hmx_ceil_div(kv_rows, HMX_FP16_TILE_N_COLS);

                        // Wait for current KV DMA
                        TIMER_START(kv_dma);
                        dma_queue_pop(dma);  // K
                        dma_queue_pop(dma);  // V
                        TIMER_STOP(kv_dma);

                        // Push mask DMA for this block (single 2D DMA when broadcast)
                        bool has_mask_dma = false;
                        MASK_DMA_PUSH(kv_start, kv_rows, has_mask_dma);

                        // ---- Phase 1: K_int(blk) ‖ O_update(blk-1) ----
                        if (kv_blk > 0) {
                            // Submit O_update for previous block (HMX worker)
                            ou_job.o_curr           = o_tile_curr;
                            ou_job.o_prev           = o_tile_prev;
                            ou_job.p_tiles          = factx.vtcm_p_tiles;
                            ou_job.v_tiles          = factx.vtcm_v_tiles;
                            ou_job.d_tiles          = factx.vtcm_d_tiles;
                            ou_job.hmx_scales       = factx.vtcm_hmx_scales_id;
                            ou_job.n_row_tiles      = n_row_tiles;
                            ou_job.n_col_tiles      = hmx_ceil_div(hex_smin(Bc, nek1 - (kv_blk - 1) * Bc), HMX_FP16_TILE_N_COLS);
                            ou_job.n_row_tiles_g_br = n_row_tiles_g_br;
                            ou_job.n_tiles_per_bc   = n_tiles_per_bc;
                            ou_job.DV               = DV;
                            hmx_queue_push(hmx_q, hmx_queue_make_desc(hmx_fa_o_update_worker, &ou_job));
                        }

                        TIMER_START(k_interleave);
                        fa_phase_k_interleave(&factx, kv_rows, k_src_stride, buf_idx);
                        TIMER_STOP(k_interleave);

                        if (kv_blk > 0) {
                            hmx_queue_pop(hmx_q);
                            hex_swap_ptr((void **) &o_tile_curr, (void **) &o_tile_prev);
                        }

                        // ---- Phase 2: qk_dot(blk) on HMX ‖ V_int(blk) + DMA prefetch on HVX ----
                        qk_job.q_tiles        = factx.vtcm_q_tiles;
                        qk_job.k_tiles        = factx.vtcm_k_tiles;
                        qk_job.s_tiles        = factx.vtcm_s_tiles;
                        qk_job.n_row_tiles    = n_row_tiles;
                        qk_job.n_col_tiles    = n_col_tiles;
                        qk_job.n_dot_tiles    = DK / 32;
                        qk_job.n_tiles_per_bc = n_tiles_per_bc;
                        qk_job.hmx_scales     = factx.vtcm_hmx_scales_qk;
                        TIMER_START(qk_dot);
                        hmx_queue_push(hmx_q, hmx_queue_make_desc(hmx_fa_qk_dot_worker, &qk_job));

                        // DMA push next block (non-blocking, before worker_pool)
                        DMA_PREFETCH_KV(kv_blk + 1);

                        TIMER_START(v_interleave);
                        fa_phase_v_interleave(&factx, kv_rows, v_src_stride, buf_idx, n_tiles_per_bc);
                        TIMER_STOP(v_interleave);

                        hmx_queue_pop(hmx_q);
                        TIMER_STOP(qk_dot);

                        // ---- Phase 3: softmax(blk) + build_D(blk) | HMX idle ----
                        // Pop mask DMA before softmax (ensures VTCM buffer is ready)
                        MASK_DMA_POP(has_mask_dma);

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
                        sargs.mask_vtcm            = has_mask_dma ? (const __fp16 *) factx.vtcm_mask_buf : NULL;
                        sargs.mask_vtcm_row_stride = factx.mask_buf_row_stride;
                        sargs.slopes               = factx.vtcm_slopes;
                        fa_compute_slopes(&sargs, &factx, kv_head, n_rows_g);

                        TIMER_START(softmax);
                        fa_phase_softmax_and_build_d(&factx, &sargs, n_row_tiles, n_row_tiles_g_br);
                        TIMER_STOP(softmax);

                        buf_idx = 1 - buf_idx;
                    }  // end KV block loop (pipeline)

                    // Epilogue: O_update for last block
                    if (factx.n_kv_blocks > 0) {
                        const uint32_t last_blk = factx.n_kv_blocks - 1;
                        const size_t last_cols  = hmx_ceil_div(hex_smin(Bc, nek1 - last_blk * Bc), HMX_FP16_TILE_N_COLS);
                        ou_job.o_curr           = o_tile_curr;
                        ou_job.o_prev           = o_tile_prev;
                        ou_job.p_tiles          = factx.vtcm_p_tiles;
                        ou_job.v_tiles          = factx.vtcm_v_tiles;
                        ou_job.d_tiles          = factx.vtcm_d_tiles;
                        ou_job.hmx_scales       = factx.vtcm_hmx_scales_id;
                        ou_job.n_row_tiles      = n_row_tiles;
                        ou_job.n_col_tiles      = last_cols;
                        ou_job.n_row_tiles_g_br = n_row_tiles_g_br;
                        ou_job.n_tiles_per_bc   = n_tiles_per_bc;
                        ou_job.DV               = DV;

                        TIMER_START(o_update);
                        hmx_queue_push(hmx_q, hmx_queue_make_desc(hmx_fa_o_update_worker, &ou_job));
                        hmx_queue_pop(hmx_q);
                        TIMER_STOP(o_update);

                        hex_swap_ptr((void **) &o_tile_curr, (void **) &o_tile_prev);
                    }

                } else {
                    // ==================================================================
                    // Fallback path: sequential with multi-thread HVX phases
                    // Main thread holds HMX lock, runs HMX inline.
                    // ==================================================================

                    for (uint32_t kv_blk = 0; kv_blk < factx.n_kv_blocks; ++kv_blk) {
                        const uint32_t kv_start    = kv_blk * Bc;
                        const uint32_t kv_rows     = hex_smin(Bc, nek1 - kv_start);
                        const size_t   n_col_tiles = hmx_ceil_div(kv_rows, HMX_FP16_TILE_N_COLS);

                        TIMER_START(kv_dma);
                        dma_queue_pop(dma);  // K
                        dma_queue_pop(dma);  // V
                        TIMER_STOP(kv_dma);

                        bool has_mask_dma = false;
                        MASK_DMA_PUSH(kv_start, kv_rows, has_mask_dma);
                        DMA_PREFETCH_KV(kv_blk + 1);

                        // K interleave (multi-thread HVX)
                        TIMER_START(k_interleave);
                        fa_phase_k_interleave(&factx, kv_rows, k_src_stride, buf_idx);
                        TIMER_STOP(k_interleave);

                        // QK dot (inline HMX on main thread)
                        TIMER_START(qk_dot);
                        {
                            const size_t n_dot_tiles       = (size_t) (DK / 32);
                            const __fp16 * restrict q_base = factx.vtcm_q_tiles;
                            const __fp16 * restrict k_base = factx.vtcm_k_tiles;
                            __fp16 * restrict s_base       = factx.vtcm_s_tiles;
                            __builtin_assume(n_row_tiles > 0);
                            __builtin_assume(n_col_tiles > 0);
                            __builtin_assume(n_dot_tiles > 0);

                            Q6_bias_mxmem2_A((void *) factx.vtcm_hmx_scales_qk);
                            for (size_t r = 0; r < n_row_tiles; ++r) {
                                for (size_t c = 0; c < n_col_tiles; ++c) {
                                    const __fp16 * row_tiles = q_base + r * HMX_FP16_TILE_N_ROWS * DK;
                                    const __fp16 * col_tiles = k_base + c * HMX_FP16_TILE_N_COLS * DK;
                                    __fp16 *       out_tile  = s_base + (r * n_tiles_per_bc + c) * HMX_FP16_TILE_N_ELMS;
                                    for (size_t k = 0; k < n_dot_tiles; ++k) {
                                        Q6_activation_hf_mxmem_RR((unsigned int) row_tiles, 2047);
                                        Q6_weight_hf_mxmem_RR((unsigned int) col_tiles, 2047);
                                        row_tiles += HMX_FP16_TILE_N_ELMS;
                                        col_tiles += HMX_FP16_TILE_N_ELMS;
                                    }
                                    Q6_mxmem_AR_after_hf(out_tile, 0);
                                }
                            }
                        }
                        TIMER_STOP(qk_dot);

                        // Pop mask DMA
                        MASK_DMA_POP(has_mask_dma);

                        // Softmax + build_D (multi-thread HVX + serial m/l update)
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
                        sargs.mask_vtcm            = has_mask_dma ? (const __fp16 *) factx.vtcm_mask_buf : NULL;
                        sargs.mask_vtcm_row_stride = factx.mask_buf_row_stride;
                        sargs.slopes               = factx.vtcm_slopes;
                        fa_compute_slopes(&sargs, &factx, kv_head, n_rows_g);

                        TIMER_START(softmax);
                        fa_phase_softmax_and_build_d(&factx, &sargs, n_row_tiles, n_row_tiles_g_br);
                        TIMER_STOP(softmax);

                        // V interleave (multi-thread HVX)
                        TIMER_START(v_interleave);
                        // FIX(v-stride): use n_tiles_per_bc (block-invariant) as V tile layout
                        // stride to match o_update's v_tile access.  Using per-block n_col_tiles
                        // misplaces DV_tile 1..3 in the last partial KV block.
                        fa_phase_v_interleave(&factx, kv_rows, v_src_stride, buf_idx, n_tiles_per_bc);
                        TIMER_STOP(v_interleave);

                        // O update (inline HMX on main thread)
                        TIMER_START(o_update);
                        {
                            const size_t DV_tiles           = (size_t) (DV / 32);
                            const __fp16 * restrict d_base  = factx.vtcm_d_tiles;
                            const __fp16 * restrict p_base  = factx.vtcm_p_tiles;
                            const __fp16 * restrict v_base  = factx.vtcm_v_tiles;
                            const __fp16 * restrict op_base = o_tile_prev;
                            __fp16 * restrict oc_base       = o_tile_curr;
                            __builtin_assume(n_row_tiles > 0);
                            __builtin_assume(n_col_tiles > 0);
                            __builtin_assume(DV_tiles > 0);

                            Q6_bias_mxmem2_A((void *) factx.vtcm_hmx_scales_id);
                            for (size_t r = 0; r < n_row_tiles; ++r) {
                                for (size_t c = 0; c < DV_tiles; ++c) {
                                    const __fp16 * d_diag = d_base  + r * (n_row_tiles_g_br + 1) * HMX_FP16_TILE_N_ELMS;
                                    const __fp16 * o_rc   = op_base + (c * n_row_tiles_g_br + r) * HMX_FP16_TILE_N_ELMS;
                                    Q6_activation_hf_mxmem_RR((unsigned int) d_diag, 2047);
                                    Q6_weight_hf_mxmem_RR((unsigned int) o_rc, 2047);

                                    const __fp16 * p_tile_in = p_base + (r * n_tiles_per_bc) * HMX_FP16_TILE_N_ELMS;
                                    const __fp16 * v_tile_in = v_base + (c * n_tiles_per_bc) * HMX_FP16_TILE_N_ELMS;
                                    for (size_t k = 0; k < n_col_tiles; ++k) {
                                        Q6_activation_hf_mxmem_RR((unsigned int) p_tile_in, 2047);
                                        Q6_weight_hf_mxmem_RR((unsigned int) v_tile_in, 2047);
                                        p_tile_in += HMX_FP16_TILE_N_ELMS;
                                        v_tile_in += HMX_FP16_TILE_N_ELMS;
                                    }

                                    __fp16 * o_tile_out = oc_base + (c * n_row_tiles_g_br + r) * HMX_FP16_TILE_N_ELMS;
                                    Q6_mxmem_AR_after_hf(o_tile_out, 0);
                                }
                            }
                            hex_swap_ptr((void **) &o_tile_curr, (void **) &o_tile_prev);
                        }
                        TIMER_STOP(o_update);

                        buf_idx = 1 - buf_idx;
                    }  // end KV block loop (fallback)
                }

                // ---- Final normalization: O = diag(1/l) @ O ----
                TIMER_START(o_norm);
                {
                    fa_build_d_diag_inv_l(&factx, n_row_tiles, n_row_tiles_g_br);

                    // HMX: O_final = diag(1/l) @ O_prev
                    if (factx.use_pipeline) {
                        on_job.o_curr           = o_tile_curr;
                        on_job.o_prev           = o_tile_prev;
                        on_job.d_tiles          = factx.vtcm_d_tiles;
                        on_job.hmx_scales       = factx.vtcm_hmx_scales_id;
                        on_job.n_row_tiles      = n_row_tiles;
                        on_job.n_row_tiles_g_br = n_row_tiles_g_br;
                        on_job.DV               = DV;
                        hmx_queue_push(ctx->hmx_queue, hmx_queue_make_desc(hmx_fa_o_norm_worker, &on_job));
                        hmx_queue_pop(ctx->hmx_queue);
                    } else {
                        const size_t DV_tiles           = (size_t) (DV / 32);
                        const __fp16 * restrict d_base  = factx.vtcm_d_tiles;
                        const __fp16 * restrict op_base = o_tile_prev;
                        __fp16 * restrict oc_base       = o_tile_curr;
                        __builtin_assume(n_row_tiles > 0);
                        __builtin_assume(DV_tiles > 0);

                        Q6_bias_mxmem2_A((void *) factx.vtcm_hmx_scales_id);
                        for (size_t r = 0; r < n_row_tiles; ++r) {
                            for (size_t c = 0; c < DV_tiles; ++c) {
                                const __fp16 * d_diag = d_base  + r * (n_row_tiles_g_br + 1) * HMX_FP16_TILE_N_ELMS;
                                const __fp16 * o_rc   = op_base + (c * n_row_tiles_g_br + r) * HMX_FP16_TILE_N_ELMS;
                                __fp16 *       o_out  = oc_base + (r * DV_tiles + c) * HMX_FP16_TILE_N_ELMS;

                                Q6_activation_hf_mxmem_RR((unsigned int) d_diag, 2047);
                                Q6_weight_hf_mxmem_RR((unsigned int) o_rc, 2047);
                                Q6_mxmem_AR_after_hf(o_out, 0);
                            }
                        }
                    }
                }
                TIMER_STOP(o_norm);

                // ---- Store O block ----
                TIMER_START(o_store);
                fa_phase_o_store(&factx, dst, o_tile_curr, q_start, kv_head, ib3, n_rows_g);
                TIMER_STOP(o_store);

#undef MASK_DMA_PUSH
#undef MASK_DMA_POP
#undef DMA_PREFETCH_KV

            }  // end Q block loop
        }  // end KV head loop
    }  // end batch loop

    if (factx.use_pipeline) {
        hmx_queue_suspend(ctx->hmx_queue);
    } else {
        HAP_compute_res_hmx_unlock(ctx->vtcm_rctx);
    }

    TIMER_STOP(total);

#if defined(ENABLE_PROFILE_TIMERS)
    FARF(HIGH, "hmx-fa: %lld us, q_load=%lld kv_dma=%lld k_interleave=%lld v_interleave=%lld", TIMER_US(total),
         TIMER_US(q_load), TIMER_US(kv_dma), TIMER_US(k_interleave), TIMER_US(v_interleave));
    FARF(HIGH, "  qk_dot=%lld softmax=%lld o_update=%lld o_norm=%lld o_store=%lld", TIMER_US(qk_dot), TIMER_US(softmax),
         TIMER_US(o_update), TIMER_US(o_norm), TIMER_US(o_store));
#endif

    return HTP_STATUS_OK;
}
