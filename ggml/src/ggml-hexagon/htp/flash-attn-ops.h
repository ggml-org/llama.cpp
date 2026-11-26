#ifndef HTP_FLASH_ATTN_OPS_H
#define HTP_FLASH_ATTN_OPS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "hex-fastdiv.h"
#include "hex-common.h"
#include "htp-tensor.h"
#include "htp-vtcm.h"

#ifdef __cplusplus
extern "C" {
#endif

// Tile constants (mirrored from hmx-utils.h for use on host side if needed)
#define HTP_FA_HMX_TILE_SIZE     2048
#define HMX_FP16_TILE_SIZE       2048
#define HMX_FP16_TILE_N_ROWS     32
#define HMX_FP16_TILE_N_COLS     32
#define HMX_FP16_TILE_N_ELMS     1024

#define HMX_W8A8_ACT_TILE_N_ROWS 64
#define HMX_W8A8_TILE_N_COLS     32
#define HMX_W8A8_ACT_TILE_SIZE   2048
#define HMX_W8A8_WT_TILE_SIZE    1024
#define HMX_W8A8_BIAS_BANK_SIZE  256

#define HVX_FA_DMA_CACHE_SIZE  128
#define HMX_FA_DMA_CACHE_SIZE  4


#define HTP_FA_M_INITIAL_VAL  -10000.0f

enum htp_fa_kernel_type {
    HTP_FA_KERNEL_UNSUPPORTED = 0,
    HTP_FA_KERNEL_HVX,
    HTP_FA_KERNEL_HMX_F16,
    HTP_FA_KERNEL_HMX_W8A8,
};

// Multicore: gcd(n_kv_heads, N) teams own the KV heads, team members share the Q-blocks
#define HTP_FA_FLAG_TEAM_SPLIT 0x01

static inline void htp_fa_mdev_teams(uint32_t n_kv_heads, uint32_t mdev_count, bool head_split,
                                     uint32_t * n_teams, uint32_t * team_size) {
    const uint32_t t = (head_split && n_kv_heads > 0 && mdev_count > 1) ? hex_gcd_u32(n_kv_heads, mdev_count) : 1;
    *n_teams   = t;
    *team_size = mdev_count > 1 ? mdev_count / t : 1;
}

struct htp_fa_kernel_params {
    uint8_t  kernel_type;        // enum htp_fa_kernel_type
    uint8_t  head_split;         // 1 = partition by KV heads in multicore, 0 = token partition
    uint8_t  flags;              // HTP_FA_FLAG_*
    uint8_t  n_hvx;              // Number of threads to run

    // Common parameters
    uint16_t Br;
    uint16_t Bc;
    uint16_t n_kv_blocks;        // also HVX's n_blocks
    uint16_t G;                  // GQA factor (n_heads / n_kv_heads)

    float    scale;
    float    max_bias;
    float    logit_softcap;
    uint32_t vtcm_size;

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
#else
_Static_assert(sizeof(struct htp_fa_kernel_params) <= 128, "htp_fa_kernel_params is too large for kernel_params blob");
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
    size_t off_q_dma;
    size_t off_o_tiles[2];
    size_t off_k_fp16[2];
    size_t off_v_fp16[2];
    size_t off_k_tiles[2];
    size_t off_v_tiles[2];
    size_t off_s_tiles[2];
    size_t off_p_tiles[2];
    size_t off_d_tiles[2];
    size_t off_d_inv_l;
    size_t off_m_vec;
    size_t off_l_vec;
    size_t off_s_rowmax;
    size_t off_p_rowsum;
    size_t off_row_bufs;
    size_t off_hmx_scales_id;
    size_t off_hmx_scales_qk;
    size_t off_mask_buf;
    size_t off_slopes;
    size_t off_sinks;
    size_t off_q_scales;
    size_t off_k_scales[2];
    size_t off_k_bias[2];
    size_t off_o_f32;
    size_t off_do_tiles;
    size_t off_v_scales[2];
    size_t off_v_bias[2];
    size_t off_p_scales[2];
    size_t off_exp_mdiff[2];

    // Region byte sizes reused by the device at runtime (not just for allocation).
    size_t q_tile_bytes;
    size_t o_tile_bytes;
    size_t k_tile_bytes;
    size_t s_tile_bytes;       // S tiles (2x in W8A8 for hi+lo)
    size_t p_tile_bytes;       // P tiles
    size_t d_tile_bytes;       // d_tiles[0..1] + d_inv_l, allocated back to back
    size_t o_f32_bytes;        // W8A8 FP32 O accumulator
    size_t m_line_bytes;       // one mask row
    size_t m_buf_slot_bytes;   // one dma_cache slot = align_up(Br * m_line_bytes, 4096)
    size_t col_vec_bytes;
    size_t sinks_bytes;
    size_t k_raw_row_padded;
    size_t v_raw_row_padded;

    // Derived strides.
    size_t row_buf_stride;       // HVX vectors (128B) per row buffer
    size_t mask_buf_row_stride;  // __fp16 elements per row in the mask buffer

    bool   pipeline;
    bool   is_w8a8;
    size_t total_bytes;
};

#define FA_MAX_KV_BLOCKS 1024

struct hmx_fa_layout_params {
    int    kernel_type;
    int    q_type;
    size_t gqa_factor;
    size_t DK;
    size_t DV;
    size_t Br;
    size_t Bc;
    size_t n_hvx;
    size_t n_heads;
    bool   pipeline;
    bool   has_mask;
    bool   mask_broadcast;
    bool   has_sinks;
};

// Build the VTCM layout.

static inline void hmx_fa_vtcm_layout_build(struct hmx_fa_vtcm_layout * L,
                                            const struct hmx_fa_layout_params * p) {
    const bool   is_w8a8        = (p->kernel_type == HTP_FA_KERNEL_HMX_W8A8);
    const size_t gqa_factor     = p->gqa_factor;
    const size_t DK             = p->DK;
    const size_t DV             = p->DV;
    const size_t Br             = p->Br;
    const size_t Bc             = p->Bc;
    const size_t n_hvx          = p->n_hvx;
    const size_t n_heads        = p->n_heads;
    const bool   pipeline       = p->pipeline;
    const bool   has_mask       = p->has_mask;
    const bool   mask_broadcast = p->mask_broadcast;
    const bool   has_sinks      = p->has_sinks;

    const size_t g_br           = hex_align_up(gqa_factor * Br, is_w8a8 ? HMX_W8A8_ACT_TILE_N_ROWS : HMX_FP16_TILE_N_ROWS);
    const size_t n_row_tiles_64 = g_br / HMX_W8A8_ACT_TILE_N_ROWS;
    const size_t n_col_tiles    = Bc   / HMX_W8A8_TILE_N_COLS;
    const size_t n_k_tiles      = DK   / HMX_W8A8_TILE_N_COLS;
    const size_t n_v_tiles      = DV   / HMX_W8A8_TILE_N_COLS;

    const size_t q_tile_size = is_w8a8
        ? (n_row_tiles_64 * n_k_tiles * HMX_W8A8_ACT_TILE_SIZE)
        : hex_align_up(g_br * DK * sizeof(__fp16), HTP_FA_HMX_TILE_SIZE);
    const size_t o_tile_size = is_w8a8
        ? 0
        : hex_align_up(g_br * DV * sizeof(__fp16), HTP_FA_HMX_TILE_SIZE);
    const size_t k_tile_size = is_w8a8
        ? hex_align_up(n_col_tiles * n_k_tiles * HMX_W8A8_WT_TILE_SIZE, HTP_FA_HMX_TILE_SIZE)
        : hex_align_up(Bc * DK * sizeof(__fp16), HTP_FA_HMX_TILE_SIZE);
    const size_t v_tile_size = is_w8a8
        ? hex_align_up(n_v_tiles * n_col_tiles * HMX_W8A8_WT_TILE_SIZE, HTP_FA_HMX_TILE_SIZE)
        : hex_align_up(Bc * DV * sizeof(__fp16), HTP_FA_HMX_TILE_SIZE);
    const size_t s_tile_size = is_w8a8
        ? (n_row_tiles_64 * n_col_tiles * 2 * HMX_W8A8_ACT_TILE_SIZE)
        : hex_align_up(g_br * Bc * sizeof(__fp16), HTP_FA_HMX_TILE_SIZE);
    const size_t p_tile_size = is_w8a8
        ? (n_row_tiles_64 * n_col_tiles * HMX_W8A8_ACT_TILE_SIZE)
        : hex_align_up(g_br * Bc * sizeof(__fp16), HTP_FA_HMX_TILE_SIZE);

    // The rescale matrices are diagonal: the HMX kernels only ever load the g_br/32
    // tiles that sit on the diagonal, so store just those, packed back to back with
    // a stride of one tile.  The old [g_br, g_br] square layout allocated g_br/32
    // times more than it used, which is also why a second D buffer was unaffordable.
    const size_t d_tile_size  = is_w8a8 ? 0 : (g_br / HMX_FP16_TILE_N_ROWS) * HTP_FA_HMX_TILE_SIZE;
    const size_t o_f32_size   = hex_align_up(g_br * DV * sizeof(float), 128);
    const size_t do_tile_size = n_row_tiles_64 * n_v_tiles * 2 * HMX_W8A8_ACT_TILE_SIZE;

    const size_t k_raw_row_padded = is_w8a8
        ? hex_round_up((DK / 32) * 34, 128)
        : hex_round_up(DK * sizeof(__fp16), 128);
    const size_t v_raw_row_padded = is_w8a8
        ? hex_round_up((DV / 32) * 34, 128)
        : hex_round_up(DV * sizeof(__fp16), 128);

    const size_t q_dma_size     = hex_align_up(g_br * htp_tensor_get_row_size(p->q_type, DK), 128);
    const size_t k_dma_size     = hex_align_up(Bc * k_raw_row_padded, 128);
    const size_t v_dma_size     = hex_align_up(Bc * v_raw_row_padded, 128);
    const size_t col_vec_size   = hex_align_up(g_br * sizeof(float),  256);
    const size_t row_vec_size   = hex_align_up(Bc   * sizeof(__fp16), 256);
    const size_t m_line_size    = hex_align_up(Bc   * sizeof(__fp16), 128);
    const size_t m_buf_slot     = hex_align_up(Br * m_line_size, 256);
    const size_t m_buf_slots    = mask_broadcast ? HMX_FA_DMA_CACHE_SIZE : gqa_factor;
    const size_t m_buf_size     = m_buf_slot * m_buf_slots;
    const size_t slopes_size    = hex_align_up(g_br * sizeof(__fp16), 128);
    const size_t sinks_size     = hex_round_up(n_heads * sizeof(float), 128);
    const size_t q_scales_size  = hex_align_up(g_br * sizeof(float), 128);
    const size_t k_scales_size  = hex_align_up(Bc   * sizeof(float), 128);
    const size_t k_bias_size    = hex_align_up(n_col_tiles * 2 * HMX_W8A8_BIAS_BANK_SIZE, HMX_W8A8_BIAS_BANK_SIZE);
    const size_t v_scales_size  = hex_align_up(DV   * sizeof(float), 128);
    const size_t v_bias_size    = hex_align_up(n_v_tiles * 2 * HMX_W8A8_BIAS_BANK_SIZE, HMX_W8A8_BIAS_BANK_SIZE);
    const size_t p_scales_size  = hex_align_up(g_br * sizeof(float), 128);
    const size_t exp_mdiff_size = hex_align_up(g_br * sizeof(float), 128);

    size_t off = 0;

    // Group A (Part 1 - HMX Tiled buffers)
    VTCM_LAYOUT_ALLOC(off, off_q_tiles,                 q_tile_size);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_o_tiles[0],     o_tile_size, !is_w8a8);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_o_tiles[1],     o_tile_size, !is_w8a8);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_d_tiles[0],     d_tile_size, !is_w8a8);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_d_tiles[1],     d_tile_size, !is_w8a8 && pipeline);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_d_inv_l,        d_tile_size, !is_w8a8);

    // Group B & C share start offset (Group B tiles must be 2KB aligned)
    size_t off_group_b_c = hex_align_up(off, HTP_FA_HMX_TILE_SIZE);

    // Group B: Compute-only buffers
    size_t off_group_b = off_group_b_c;
    VTCM_LAYOUT_ALLOC(off_group_b, off_k_tiles[0],    k_tile_size);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_k_tiles[1], k_tile_size, pipeline);
    VTCM_LAYOUT_ALLOC(off_group_b, off_v_tiles[0],    v_tile_size);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_v_tiles[1], v_tile_size, pipeline);
    VTCM_LAYOUT_ALLOC(off_group_b, off_s_tiles[0],    s_tile_size);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_s_tiles[1], s_tile_size, pipeline);
    VTCM_LAYOUT_ALLOC(off_group_b, off_p_tiles[0],    p_tile_size);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_p_tiles[1], p_tile_size, pipeline);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_do_tiles,   do_tile_size, is_w8a8);
    VTCM_LAYOUT_ALLOC(off_group_b, off_s_rowmax,      col_vec_size);
    VTCM_LAYOUT_ALLOC(off_group_b, off_p_rowsum,      col_vec_size);
    VTCM_LAYOUT_ALLOC(off_group_b, off_row_bufs,      row_vec_size * (is_w8a8 ? 4 : 2) * n_hvx);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_k_scales[0],  k_scales_size,  is_w8a8);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_k_scales[1],  k_scales_size,  is_w8a8 && pipeline);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_v_scales[0],  v_scales_size,  is_w8a8);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_v_scales[1],  v_scales_size,  is_w8a8 && pipeline);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_p_scales[0],  p_scales_size,  is_w8a8);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_p_scales[1],  p_scales_size,  is_w8a8 && pipeline);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_exp_mdiff[0], exp_mdiff_size, is_w8a8);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_exp_mdiff[1], exp_mdiff_size, is_w8a8 && pipeline);
    if (is_w8a8) {
        off_group_b = hex_align_up(off_group_b, HMX_W8A8_BIAS_BANK_SIZE);
    }
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_k_bias[0],   k_bias_size, is_w8a8);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_k_bias[1],   k_bias_size, is_w8a8 && pipeline);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_v_bias[0],   v_bias_size, is_w8a8);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_v_bias[1],   v_bias_size, is_w8a8 && pipeline);

    const size_t group_b_size = off_group_b - off_group_b_c;

    // Group C: Q fetch DMA buffer
    size_t off_group_c = off_group_b_c;
    VTCM_LAYOUT_ALLOC(off_group_c, off_q_dma,         q_dma_size);

    const size_t group_c_size = off_group_c - off_group_b_c;

    off = off_group_b_c + hex_smax(group_b_size, group_c_size);

    // Group A (Part 2 - remaining non-HMX buffers)
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_o_f32, o_f32_size, is_w8a8);
    VTCM_LAYOUT_ALLOC(off, off_k_fp16[0],     k_dma_size);
    VTCM_LAYOUT_ALLOC(off, off_k_fp16[1],     k_dma_size);
    VTCM_LAYOUT_ALLOC(off, off_v_fp16[0],     v_dma_size);
    VTCM_LAYOUT_ALLOC(off, off_v_fp16[1],     v_dma_size);
    VTCM_LAYOUT_ALLOC(off, off_m_vec,         col_vec_size);
    VTCM_LAYOUT_ALLOC(off, off_l_vec,         col_vec_size);
    VTCM_LAYOUT_ALLOC(off, off_hmx_scales_id, 256);
    VTCM_LAYOUT_ALLOC(off, off_hmx_scales_qk, 256);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_mask_buf, m_buf_size, has_mask);
    VTCM_LAYOUT_ALLOC(off, off_slopes,        slopes_size);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_q_scales, q_scales_size, is_w8a8);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_sinks, sinks_size, has_sinks);

    L->q_tile_bytes        = q_tile_size;
    L->o_tile_bytes        = o_tile_size;
    L->k_tile_bytes        = k_tile_size;
    L->col_vec_bytes       = col_vec_size;
    L->s_tile_bytes        = s_tile_size;
    L->p_tile_bytes        = p_tile_size;
    // Measured from the actual offsets rather than assumed to be N * d_tile_size, so
    // that inserting a region between them (or adding padding to VTCM_LAYOUT_ALLOC)
    // cannot silently leave the tail of the run unzeroed.
    L->d_tile_bytes        = is_w8a8 ? 0 : ((L->off_d_inv_l + d_tile_size) - L->off_d_tiles[0]);
    L->o_f32_bytes         = is_w8a8 ? o_f32_size : 0;
    L->m_line_bytes        = m_line_size;
    L->m_buf_slot_bytes    = m_buf_slot;
    L->k_raw_row_padded    = k_raw_row_padded;
    L->v_raw_row_padded    = v_raw_row_padded;
    L->row_buf_stride      = row_vec_size / 128;
    L->mask_buf_row_stride = m_line_size / sizeof(__fp16);
    L->sinks_bytes         = has_sinks ? sinks_size : 0;
    L->pipeline            = pipeline;
    L->is_w8a8             = is_w8a8;
    L->total_bytes         = off;
}

// Exact VTCM usage for a given HMX layout configuration.
static inline size_t hmx_fa_compute_vtcm_usage(const struct hmx_fa_layout_params * p) {
    struct hmx_fa_vtcm_layout L;
    hmx_fa_vtcm_layout_build(&L, p);
    return L.total_bytes;
}

#define FA_HVX_BLOCK_SIZE 64
#define FA_HVX_G_MAX      8

struct hvx_fa_layout_params {
    int    q_type;
    size_t DK;
    size_t DV;
    size_t n_heads;
    size_t n_hvx;
    bool   has_mask;
    bool   has_sinks;
};

struct hvx_fa_vtcm_layout {
    size_t off_q;
    size_t off_k;
    size_t off_v;
    size_t off_m;
    size_t off_a;
    size_t off_sinks;

    size_t size_q_block;
    size_t size_k_block;
    size_t size_v_block;
    size_t size_m_block;
    size_t size_vkq_acc;
    size_t size_sinks;

    size_t total_bytes;
};

static inline void hvx_fa_vtcm_layout_build(struct hvx_fa_vtcm_layout * L,
                                            const struct hvx_fa_layout_params * p) {
    const size_t size_q_row_padded = hex_round_up(htp_tensor_get_row_size(p->q_type, p->DK), 128);
    const size_t size_k_row_padded = hex_round_up(p->DK * sizeof(__fp16), 128);
    const size_t size_v_row_padded = hex_round_up(p->DV * sizeof(__fp16), 128);

    const size_t size_q_block = size_q_row_padded * FA_HVX_G_MAX;
    const size_t size_k_block = size_k_row_padded * FA_HVX_BLOCK_SIZE;
    const size_t size_v_block = size_v_row_padded * FA_HVX_BLOCK_SIZE;
    const size_t size_m_block = hex_round_up(FA_HVX_BLOCK_SIZE * sizeof(__fp16), 128);
    const size_t size_vkq_acc = hex_round_up(p->DV * sizeof(float), 256) * FA_HVX_G_MAX;
    const size_t size_sinks   = hex_round_up(p->n_heads * sizeof(float), 128);

    size_t off = 0;

    VTCM_LAYOUT_ALLOC(off, off_q, size_q_block * p->n_hvx);
    VTCM_LAYOUT_ALLOC(off, off_k, size_k_block * 2 * p->n_hvx);
    VTCM_LAYOUT_ALLOC(off, off_v, size_v_block * 2 * p->n_hvx);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_m, size_m_block * HVX_FA_DMA_CACHE_SIZE * p->n_hvx, p->has_mask);
    VTCM_LAYOUT_ALLOC(off, off_a, size_vkq_acc * p->n_hvx);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_sinks, size_sinks, p->has_sinks);

    L->size_q_block = size_q_block;
    L->size_k_block = size_k_block;
    L->size_v_block = size_v_block;
    L->size_m_block = size_m_block;
    L->size_vkq_acc = size_vkq_acc;
    L->size_sinks   = p->has_sinks ? size_sinks : 0;
    L->total_bytes  = off;
}

static inline size_t hvx_fa_compute_vtcm_usage(const struct hvx_fa_layout_params * p) {
    struct hvx_fa_vtcm_layout L;
    hvx_fa_vtcm_layout_build(&L, p);
    return L.total_bytes;
}

#define FA_MIN_KV_BLOCKS 3

// Cost-based (Br, Bc) search for flash attention with pipeline constraint.
static inline int hmx_fa_find_chunk_size(struct hmx_fa_layout_params * p,
                                         size_t qo_len,
                                         size_t kv_len,
                                         size_t vtcm_budget,
                                         size_t q_split) {
    const size_t gqa_factor   = p->gqa_factor;
    const size_t n_hvx        = p->n_hvx;
    const bool   is_w8a8      = (p->kernel_type == HTP_FA_KERNEL_HMX_W8A8);
    const size_t T            = HMX_FP16_TILE_N_ROWS;  // 32
    const size_t br_unit      = hmx_ceil_div(T, gqa_factor);
    const size_t bc_unit      = HMX_FP16_TILE_N_COLS * 2;  // 64
    const bool   can_pipeline = (kv_len >= FA_MIN_KV_BLOCKS * bc_unit && n_hvx >= 2 && (!p->has_mask || p->mask_broadcast));

    // Br_max: largest Br aligned to br_unit that does not exceed qo_len.
    const size_t Br_max = qo_len >= br_unit ? hex_align_down(qo_len, br_unit) : br_unit;

    // Pipeline constraint: cap Bc so n_kv_blocks >= FA_MIN_KV_BLOCKS.
    // Only relax when kv_len is too short to form enough blocks.
    size_t Bc_limit     = can_pipeline ? hex_align_down(kv_len / FA_MIN_KV_BLOCKS, bc_unit) :
                                         (kv_len >= bc_unit ? hex_align_down(kv_len, bc_unit) : bc_unit);
    // For small-Q SWA-sized KV caches (qo_len <= 16, kv_len <= 2048), cap Bc at 256 for block-skipping granularity
    if (qo_len <= 16 && kv_len <= 2048 && Bc_limit > 256) {
        Bc_limit = 256;
    }
    // Cap Bc at 2048 for W8A8 so P * V always runs in 1 superblock (<= 64 tiles of 32)
    if (is_w8a8 && Bc_limit > 2048) {
        Bc_limit = 2048;
    }
    // Cost coefficients calibrated from profiling
    const size_t c_q_fixed    = 800;   // per-Q-block: q_load + epilogue o_update + o_norm + o_store
    const size_t c_iter_base  = 200;   // per-KV-iter base (HMX dot/update + DMA)
    const size_t c_softmax    = 600;   // per 64-row vector chunk on HVX

    size_t best_cost = SIZE_MAX, best_mn = 0;
    size_t best_Br = 0, best_Bc = 0;

    struct hmx_fa_layout_params cand = *p;
    cand.pipeline = can_pipeline;

    for (size_t Br = Br_max; Br >= br_unit; Br -= br_unit) {
        cand.Br = Br;
        // Try all Bc candidates from Bc_limit down to bc_unit
        for (size_t Bc = Bc_limit; Bc >= bc_unit; Bc -= bc_unit) {
            cand.Bc = Bc;
            size_t vtcm_needed = hmx_fa_compute_vtcm_usage(&cand);
            if (vtcm_needed <= vtcm_budget) {
                // This Bc fits for this Br!
                const size_t q_blocks       = (qo_len + Br - 1) / Br;
                const size_t kv_blocks      = (kv_len + Bc - 1) / Bc;
                const size_t actual_hvx     = (kv_blocks >= 3 && n_hvx >= 2) ? n_hvx : 1;
                const size_t n_rows_g       = Br * gqa_factor;
                const size_t n_row_vec_cnt  = (n_rows_g + 63) / 64;
                const size_t n_use          = n_row_vec_cnt < actual_hvx ? n_row_vec_cnt : actual_hvx;
                const size_t vecs_per_t     = n_use > 0 ? (n_row_vec_cnt + n_use - 1) / n_use : 1;

                const size_t c_iter_actual  = c_iter_base + c_softmax * vecs_per_t;
                const size_t q_blocks_nsp   = (q_blocks + q_split - 1) / (q_split ? q_split : 1);
                const size_t cost           = q_blocks_nsp * (c_q_fixed + kv_blocks * c_iter_actual);
                const size_t mn             = Br * Bc;

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

    p->Br = best_Br;
    p->Bc = best_Bc;
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* HTP_FLASH_ATTN_OPS_H */
