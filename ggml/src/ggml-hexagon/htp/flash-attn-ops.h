#ifndef HTP_FLASH_ATTN_OPS_H
#define HTP_FLASH_ATTN_OPS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "hex-fastdiv.h"
#include "hex-common.h"
#include "htp-vtcm.h"

#ifdef __cplusplus
extern "C" {
#endif

// Tile constants (mirrored from hmx-utils.h for use on host side if needed)
#define HTP_FA_HMX_TILE_SIZE   2048
#define HMX_FP16_TILE_SIZE     2048
#define HMX_FP16_TILE_N_ROWS   32
#define HMX_FP16_TILE_N_COLS   32
#define HMX_FP16_TILE_N_ELMS   1024

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

    struct fastdiv_values broadcast_rk2;
    struct fastdiv_values broadcast_rk3;
    struct fastdiv_values broadcast_rv2;
    struct fastdiv_values broadcast_rv3;

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
        } hvx;
    } u;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_fa_kernel_params) <= 128, "htp_fa_kernel_params is too large for kernel_params blob");
#endif

// VTCM region layout for the HMX flash-attention kernel.
//
// Single source of truth for both the host (which needs the total size to pick a
// (Br, Bc) tiling that fits the VTCM budget) and the device (which needs the actual
// byte offsets to place each scratch buffer). Building the layout once and reading
// offsets/total from it makes host estimate and device allocation impossible to
// desync -- previously they were duplicated formulas in two files and drifted.
//
// All fields are byte offsets / byte sizes -- no HVX_Vector type is named here so the
// header stays host-includable. The device casts (base + off_*) to the proper type.
// An offset of 0 marks a region that is not allocated for this configuration (only
// off_v_tiles[1], which exists only when pipelining); the device sets such pointers NULL.
struct hmx_fa_vtcm_layout {
    // Byte offsets from vtcm_base for each region.
    size_t off_q_tiles;
    size_t off_o_tiles[2];
    size_t off_k_fp16[2];
    size_t off_v_fp16[2];
    size_t off_k_tiles;
    size_t off_v_tiles[2];     // [1] allocated only when pipeline, else 0
    size_t off_s_tiles;
    size_t off_p_tiles;
    size_t off_d_tiles;
    size_t off_m_vec;
    size_t off_l_vec;
    size_t off_s_rowmax;
    size_t off_p_rowsum;
    size_t off_row_bufs;
    size_t off_hmx_scales_id;
    size_t off_hmx_scales_qk;
    size_t off_mask_buf;
    size_t off_slopes;

    // Region byte sizes reused by the device at runtime (not just for allocation).
    size_t q_tile_bytes;
    size_t o_tile_bytes;
    size_t s_tile_bytes;       // S and P tiles (same size)
    size_t d_tile_bytes;
    size_t m_line_bytes;       // one mask row
    size_t m_buf_slot_bytes;   // one dma_cache slot = align_up(Br * m_line_bytes, 4096)
    size_t col_vec_bytes;

    // Derived strides.
    size_t row_buf_stride;       // HVX vectors (128B) per row buffer
    size_t mask_buf_row_stride;  // __fp16 elements per row in the mask buffer

    bool   pipeline;
    size_t total_bytes;
};

// Build the VTCM layout.

static inline void hmx_fa_vtcm_layout_build(struct hmx_fa_vtcm_layout * L,
                                       size_t gqa_factor, size_t DK, size_t DV,
                                       size_t Br, size_t Bc, size_t n_threads, bool pipeline) {
    const size_t g_br         = hex_align_up(gqa_factor * Br, HMX_FP16_TILE_N_ROWS);
    const size_t q_tile_size  = hex_align_up(g_br * DK   * sizeof(__fp16), HTP_FA_HMX_TILE_SIZE);
    const size_t o_tile_size  = hex_align_up(g_br * DV   * sizeof(__fp16), HTP_FA_HMX_TILE_SIZE);
    const size_t k_tile_size  = hex_align_up(Bc   * DK   * sizeof(__fp16), HTP_FA_HMX_TILE_SIZE);
    const size_t v_tile_size  = hex_align_up(Bc   * DV   * sizeof(__fp16), HTP_FA_HMX_TILE_SIZE);
    const size_t s_tile_size  = hex_align_up(g_br * Bc   * sizeof(__fp16), HTP_FA_HMX_TILE_SIZE);
    const size_t d_tile_size  = hex_align_up(g_br * g_br * sizeof(__fp16), HTP_FA_HMX_TILE_SIZE);

    const size_t k_dma_size   = hex_align_up(Bc * hex_round_up(DK * sizeof(__fp16), 128), 128);
    const size_t v_dma_size   = hex_align_up(Bc * hex_round_up(DV * sizeof(__fp16), 128), 128);
    const size_t col_vec_size = hex_align_up(g_br * sizeof(float),  256);
    const size_t row_vec_size = hex_align_up(Bc   * sizeof(__fp16), 256);
    const size_t m_line_size  = hex_align_up(Bc   * sizeof(__fp16), 128);
    const size_t m_buf_slot   = hex_align_up(Br * m_line_size, 256);
    const size_t m_buf_size   = m_buf_slot * HMX_FA_DMA_CACHE_SIZE;
    const size_t slopes_size  = hex_align_up(g_br * sizeof(__fp16), 128);

    size_t off = 0;

    // Section 1: HMX Tiled Buffers (FA_HMX_TILE_SIZE = 2KB Aligned)
    VTCM_LAYOUT_ALLOC(off, off_q_tiles,       q_tile_size);
    VTCM_LAYOUT_ALLOC(off, off_o_tiles[0],    o_tile_size);
    VTCM_LAYOUT_ALLOC(off, off_o_tiles[1],    o_tile_size);
    VTCM_LAYOUT_ALLOC(off, off_k_tiles,       k_tile_size);
    VTCM_LAYOUT_ALLOC(off, off_v_tiles[0],    v_tile_size);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_v_tiles[1], v_tile_size, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_s_tiles,       s_tile_size);
    VTCM_LAYOUT_ALLOC(off, off_p_tiles,       s_tile_size);
    VTCM_LAYOUT_ALLOC(off, off_d_tiles,       d_tile_size);

    // Section 2: HVX/DMA flat and vector buffers (128B / 256B Aligned)
    VTCM_LAYOUT_ALLOC(off, off_k_fp16[0],     k_dma_size);
    VTCM_LAYOUT_ALLOC(off, off_k_fp16[1],     k_dma_size);
    VTCM_LAYOUT_ALLOC(off, off_v_fp16[0],     v_dma_size);
    VTCM_LAYOUT_ALLOC(off, off_v_fp16[1],     v_dma_size);
    VTCM_LAYOUT_ALLOC(off, off_m_vec,         col_vec_size);
    VTCM_LAYOUT_ALLOC(off, off_l_vec,         col_vec_size);
    VTCM_LAYOUT_ALLOC(off, off_s_rowmax,      col_vec_size);
    VTCM_LAYOUT_ALLOC(off, off_p_rowsum,      col_vec_size);
    VTCM_LAYOUT_ALLOC(off, off_row_bufs,      row_vec_size * 2 * n_threads);
    VTCM_LAYOUT_ALLOC(off, off_hmx_scales_id, 256);
    VTCM_LAYOUT_ALLOC(off, off_hmx_scales_qk, 256);
    VTCM_LAYOUT_ALLOC(off, off_mask_buf,      m_buf_size);
    VTCM_LAYOUT_ALLOC(off, off_slopes,        slopes_size);

    L->q_tile_bytes        = q_tile_size;
    L->o_tile_bytes        = o_tile_size;
    L->col_vec_bytes       = col_vec_size;
    L->s_tile_bytes        = s_tile_size;
    L->d_tile_bytes        = d_tile_size;
    L->m_line_bytes        = m_line_size;
    L->m_buf_slot_bytes    = m_buf_slot;
    L->row_buf_stride      = row_vec_size / 128;
    L->mask_buf_row_stride = m_line_size / sizeof(__fp16);
    L->pipeline            = pipeline;
    L->total_bytes         = off;
}

// Exact VTCM usage for a given (gqa_factor, DK, DV, Br, Bc) configuration.
static inline size_t hmx_fa_compute_vtcm_usage(size_t gqa_factor, size_t DK, size_t DV, size_t Br, size_t Bc, size_t n_threads, bool pipeline) {
    struct hmx_fa_vtcm_layout L;
    hmx_fa_vtcm_layout_build(&L, gqa_factor, DK, DV, Br, Bc, n_threads, pipeline);
    return L.total_bytes;
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
    const bool   can_pipeline = (kv_len >= FA_MIN_KV_BLOCKS * bc_unit && n_threads >= 2);

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
        // Try all Bc candidates from Bc_limit down to bc_unit
        for (size_t Bc = Bc_limit; Bc >= bc_unit; Bc -= bc_unit) {
            size_t vtcm_needed = hmx_fa_compute_vtcm_usage(gqa_factor, DK, DV, Br, Bc, n_threads, can_pipeline);
            if (vtcm_needed <= vtcm_budget) {
                // This Bc fits for this Br!
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
                // Since we iterate Bc from largest to smallest, this is the largest Bc that fits
                // for this Br. We can break to the next Br.
                break;
            }
        }

        if (Br == br_unit) {
            break;
        }
    }

    if (best_Br == 0 || best_Bc == 0) {
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
