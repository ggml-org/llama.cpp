#ifndef HTP_FLASH_ATTN_OPS_H
#define HTP_FLASH_ATTN_OPS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "hex-fastdiv.h"
#include "hex-common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Tile constants (mirrored from hmx-utils.h for use on host side if needed)
#define HMX_FP16_TILE_N_ROWS   32
#define HMX_FP16_TILE_N_COLS   32
#define HMX_FP16_TILE_N_ELMS   1024
#define HMX_FP16_TILE_SIZE     2048
#define HVX_FA_DMA_CACHE_SIZE  128
#define HMX_FA_DMA_CACHE_SIZE  4

#define HTP_FA_M_INITIAL_VAL  -10000.0f

enum htp_fa_kernel_type {
    HTP_FA_KERNEL_UNSUPPORTED = 0,
    HTP_FA_KERNEL_HVX,
    HTP_FA_KERNEL_HMX
};

struct htp_fa_kernel_params {
    uint8_t  kernel_type;        // enum htp_fa_kernel_type
    uint8_t  is_q_fp32;          // 1 = Q type is F32, 0 = F16
    uint8_t  is_dst_fp32;        // 1 = dst type is F32, 0 = F16
    uint8_t  n_threads;          // Number of threads to run

    // Common parameters
    uint16_t Br;
    uint16_t Bc;
    uint16_t n_kv_blocks;        // also HVX's n_blocks
    uint16_t G;                  // GQA factor (n_heads / n_kv_heads)

    float    scale;
    float    max_bias;
    float    logit_softcap;
    uint32_t vtcm_size;

    uint32_t qrows;
    uint32_t qrows_per_thread;
    float    m0;
    float    m1;
    uint32_t n_head_log2;

    struct fastdiv_values src3_div2;
    struct fastdiv_values src3_div3;

    union {
        struct {
            uint32_t g_br;
            uint32_t row_buf_stride;
            uint32_t mask_buf_row_stride;
            int32_t  mask_broadcast;
            int32_t  pipeline;
            struct fastdiv_values div_G;
        } hmx;
        struct {
            uint32_t size_q_row_padded;
            uint32_t size_k_row_padded;
            uint32_t size_v_row_padded;
            struct fastdiv_values src0_div21;
            struct fastdiv_values src0_div1;
            struct fastdiv_values broadcast_rk2;
            struct fastdiv_values broadcast_rk3;
            struct fastdiv_values broadcast_rv2;
            struct fastdiv_values broadcast_rv3;
        } hvx;
    } u;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_fa_kernel_params) <= 128, "htp_fa_kernel_params is too large for kernel_params blob");
#endif

// Exact VTCM usage for a given (gqa_factor, DK, DV, Br, Bc) configuration.
// g_br = hex_align_up(gqa_factor * Br, 32) replaces Br for all Q/O/S/P/D dimensions.
// Layout: Q + O_ping + O_pong + K_dma*2 + V_dma*2 + K_tile + V_tile + S + P + D + vectors + scales
// Mask is DMA'd into a VTCM buffer (Br rows per KV block) to avoid DDR reads in softmax.
static inline size_t hmx_fa_compute_vtcm_usage(size_t gqa_factor, size_t DK, size_t DV, size_t Br, size_t Bc, size_t n_threads, bool pipeline) {
    const size_t g_br         = hex_align_up(gqa_factor * Br, HMX_FP16_TILE_N_ROWS);
    const size_t q_tile_size  = hex_align_up(g_br * DK * sizeof(__fp16), 4096);    // Q:  [g_br, DK]
    const size_t o_tile_size  = hex_align_up(g_br * DV * sizeof(__fp16), 4096);    // O:  [g_br, DV] x2 ping-pong
    const size_t k_dma_size   = hex_align_up(Bc * hex_round_up(DK * sizeof(__fp16), 128), 4096);      // K DMA: [Bc, DK] x2 double-buf
    const size_t v_dma_size   = hex_align_up(Bc * hex_round_up(DV * sizeof(__fp16), 128), 4096);      // V DMA: [Bc, DV] x2 double-buf
    const size_t k_tile_size  = hex_align_up(Bc * DK * sizeof(__fp16), 4096);      // K tiles: [Bc, DK] interleaved
    const size_t v_tile_size  = hex_align_up(Bc * DV * sizeof(__fp16), 4096);      // V tiles: [Bc, DV] interleaved
    const size_t s_tile_size  = hex_align_up(g_br * Bc * sizeof(__fp16), 4096);    // S/P:[g_br, Bc]
    const size_t d_tile_size  = hex_align_up(g_br * g_br * sizeof(__fp16), 4096);  // D:  [g_br, g_br]
    const size_t col_vec_size = hex_align_up(g_br * sizeof(float), 256);          // m, l, etc.
    const size_t row_vec_size = hex_align_up(Bc * sizeof(__fp16), 256);
    const size_t m_line_size  = hex_align_up(Bc * sizeof(__fp16), 128);
    const size_t m_buf_size   = hex_align_up(Br * m_line_size, 4096) * HMX_FA_DMA_CACHE_SIZE;
    const size_t slopes_size  = hex_align_up(g_br * sizeof(__fp16), 128);

    return   q_tile_size * 1               // Q tiles
           + o_tile_size * 2               // O ping-pong
           + k_dma_size  * 2               // K DMA x2
           + v_dma_size  * 2               // V DMA x2
           + k_tile_size * 1               // K tiles
           + v_tile_size * (pipeline ? 2 : 1) // V tiles (double-buffered if pipelining)
           + s_tile_size * 2               // S + P
           + d_tile_size * 1               // D (diagonal matrix)
           + col_vec_size * 4              // m_vec, l_vec, s_rowmax, p_rowsum
           + row_vec_size * 2 * n_threads  // per-thread softmax row scratch
           + m_buf_size * 1                // mask VTCM buffer [Br rows]
           + slopes_size                   // Slopes
           + 256 * 2;                      // HMX scales (id + qk)
}

#define FA_HVX_BLOCK_SIZE 64

static inline size_t hvx_fa_compute_vtcm_usage(size_t DK, size_t DV, bool is_q_fp32, bool has_mask, size_t n_threads) {
    const size_t size_q_row_padded = hex_round_up(DK * (is_q_fp32 ? 4 : 2), 128);
    const size_t size_k_row_padded = hex_round_up(DK * sizeof(__fp16), 128);
    const size_t size_v_row_padded = hex_round_up(DV * sizeof(__fp16), 128);

    const size_t size_q_block = size_q_row_padded * 1;
    const size_t size_k_block = size_k_row_padded * FA_HVX_BLOCK_SIZE;
    const size_t size_v_block = size_v_row_padded * FA_HVX_BLOCK_SIZE;
    const size_t size_m_block = hex_round_up(FA_HVX_BLOCK_SIZE * sizeof(__fp16), 128);
    const size_t size_vkq_acc = hex_round_up(DV * sizeof(float), 128);

    const size_t size_per_thread = size_q_block * 1
                                 + size_k_block * 2
                                 + size_v_block * 2
                                 + (has_mask ? size_m_block * HVX_FA_DMA_CACHE_SIZE : 0)
                                 + size_vkq_acc;

    return size_per_thread * n_threads;
}

#define FA_MIN_KV_BLOCKS 3

// Cost-based (Br, Bc) search for flash attention with pipeline constraint.
static inline int hmx_fa_find_chunk_size(size_t * Br_out,
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
    const size_t bc_unit = HMX_FP16_TILE_N_COLS * 2;  // 64
    const size_t fp16    = sizeof(__fp16);
    const bool   can_pipeline = (kv_len >= FA_MIN_KV_BLOCKS * bc_unit && n_threads >= 2);

    // Approximate per-unit VTCM costs (without per-buffer alignment padding).
    const size_t per_gbr  = (DK + 2 * DV) * fp16 + 4 * sizeof(float);  // Q + O*2 + 4 col vectors
    const size_t per_gbr2 = fp16;                             // D diagonal matrix
    const size_t per_bc =
        3 * DK * fp16 + (can_pipeline ? 4 : 3) * DV * fp16 + 2 * n_threads * fp16;          // K/V DMA x2 + tiles + row bufs
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
    const size_t Bc_limit     = can_pipeline ? hex_align_down(kv_len / FA_MIN_KV_BLOCKS, bc_unit) :
                                               (kv_len >= bc_unit ? hex_align_down(kv_len, bc_unit) : bc_unit);
    // Cost coefficients calibrated from profiling
    const size_t c_q_fixed    = 1400;  // per-Q-block: q_load + epilogue o_update + o_norm + o_store
    const size_t c_iter_fixed = 200;   // per-KV-iter: HMX queue push/pop + DMA pop + barriers

    size_t best_cost = SIZE_MAX, best_mn = 0;
    size_t best_Br = 0, best_Bc = 0;

    for (size_t Br = Br_max; Br >= br_unit; Br -= br_unit) {
        const size_t g_br = hex_align_up(gqa_factor * Br, T);

        // g_br-dependent VTCM cost: g_br * per_gbr + g_br*g_br * per_gbr2
        const size_t gbr_cost = g_br * per_gbr + g_br * g_br * per_gbr2;
        if (gbr_cost >= usable) {
            if (Br == br_unit) {
                break;
            }
            continue;
        }

        // Analytically solve for max Bc:
        //   remain >= Bc * (per_bc + g_br * per_gbr_bc + Br * fp16 * HMX_FA_DMA_CACHE_SIZE)
        // The Br * fp16 term accounts for the VTCM mask buffer [Br * Bc].
        const size_t remain   = usable - gbr_cost;
        const size_t bc_denom = per_bc + g_br * per_gbr_bc + Br * fp16 * HMX_FA_DMA_CACHE_SIZE;
        size_t       Bc       = hex_smin(hex_align_down(remain / bc_denom, bc_unit), Bc_limit);
        if (Bc < bc_unit) {
            if (Br == br_unit) {
                break;
            }
            continue;
        }

        // Exact VTCM verification (alignment padding may push over budget)
        while (Bc >= bc_unit && hmx_fa_compute_vtcm_usage(gqa_factor, DK, DV, Br, Bc, n_threads, can_pipeline) > vtcm_budget) {
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

#ifdef __cplusplus
}
#endif

#endif /* HTP_FLASH_ATTN_OPS_H */
