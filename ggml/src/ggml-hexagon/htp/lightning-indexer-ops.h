#ifndef HTP_LIGHTNING_INDEXER_OPS_H
#define HTP_LIGHTNING_INDEXER_OPS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "hex-common.h"
#include "htp-vtcm.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HTP_LID_KV_BLOCK   64 // kv rows per block (HVX and HMX Bc)
#define HTP_LID_TOKENS_MAX 32 // max tokens per Q tile (HVX splat path)
#define HTP_LID_HMX_TOKENS_MAX 512 // HMX can keep more tokens; VTCM decides

// Mirrored from hmx-utils.h so the host can size VTCM without pulling DSP headers
#ifndef HMX_FP16_TILE_N_ROWS
#define HMX_FP16_TILE_N_ROWS 32
#define HMX_FP16_TILE_N_COLS 32
#define HMX_FP16_TILE_N_ELMS 1024
#define HMX_FP16_TILE_SIZE   2048
#endif

enum htp_lid_kernel_type {
    HTP_LID_KERNEL_HVX = 0,
    HTP_LID_KERNEL_HMX = 1,
};

struct htp_lid_kernel_params {
    uint32_t n_threads;
    uint32_t n_tokens_tile;     // tokens per Q tile
    uint32_t n_kv_blocks;       // per stream
    uint32_t blocks_per_thread;
    uint32_t vtcm_size;
    uint32_t kernel_type;       // htp_lid_kernel_type
    uint32_t n_row_pad;         // HMX: pad(n_tok * n_head, 32)
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_lid_kernel_params) <= 128, "htp_lid_kernel_params is too large");
#else
_Static_assert(sizeof(struct htp_lid_kernel_params) <= 128, "htp_lid_kernel_params is too large");
#endif

// VTCM layout shared by host (fit check) and device (buffer offsets).
// Shared regions: Q splats (one f16 vector per q element) and head weight splats for the current token tile.
// Per-thread regions: Q prep staging, K / mask / output double buffers, transposed K and transpose scratch.
struct htp_lid_vtcm_layout {
    size_t off_q;
    size_t off_w;
    size_t off_qs;
    size_t off_k;
    size_t off_kt;
    size_t off_m;
    size_t off_o;

    size_t size_q_row;   // f16 Q row, padded
    size_t size_qs_row;  // f32 Q staging row, padded
    size_t size_k_row;   // f16 K row, padded (also holds a Q8_0 row before dequant)
    size_t size_m_row;   // f16 mask row of one kv block, padded
    size_t size_o_row;   // f32 output row of one kv block, padded

    size_t size_qs;      // per thread
    size_t size_k_buf;   // one of two K buffers
    size_t size_kt;      // per thread: transposed K block plus two 64x64 scratch tiles
    size_t size_m_buf;   // one of two mask buffers
    size_t size_o_buf;   // one of two output buffers

    size_t total_bytes;
};

static inline void htp_lid_vtcm_layout_build(struct htp_lid_vtcm_layout * L, size_t D, size_t n_head, size_t n_tok, size_t n_threads) {
    L->size_q_row  = hex_round_up(D * sizeof(uint16_t), 128);
    L->size_qs_row = hex_round_up(D * sizeof(float), 128);
    L->size_k_row  = hex_round_up(D * sizeof(uint16_t), 128);
    L->size_m_row  = hex_round_up(HTP_LID_KV_BLOCK * sizeof(uint16_t), 128);
    L->size_o_row  = hex_round_up(HTP_LID_KV_BLOCK * sizeof(float), 128);

    L->size_qs    = n_head * L->size_qs_row + hex_round_up(n_head * sizeof(float), 128) + L->size_q_row;
    L->size_k_buf = HTP_LID_KV_BLOCK * L->size_k_row;
    L->size_kt    = (hex_round_up(D, HTP_LID_KV_BLOCK) + 2 * HTP_LID_KV_BLOCK) * 128;
    L->size_m_buf = n_tok * L->size_m_row;
    L->size_o_buf = n_tok * L->size_o_row;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_q,  n_tok * n_head * D * 128);
    VTCM_LAYOUT_ALLOC(off, off_w,  n_tok * n_head * 128);
    VTCM_LAYOUT_ALLOC(off, off_qs, n_threads * L->size_qs);
    VTCM_LAYOUT_ALLOC(off, off_k,  n_threads * 2 * L->size_k_buf);
    VTCM_LAYOUT_ALLOC(off, off_kt, n_threads * L->size_kt);
    VTCM_LAYOUT_ALLOC(off, off_m,  n_threads * 2 * L->size_m_buf);
    VTCM_LAYOUT_ALLOC(off, off_o,  n_threads * 2 * L->size_o_buf);

    L->total_bytes = off;
}

static inline size_t htp_lid_compute_vtcm_usage(size_t D, size_t n_head, size_t n_tok, size_t n_threads) {
    struct htp_lid_vtcm_layout L;
    htp_lid_vtcm_layout_build(&L, D, n_head, n_tok, n_threads);
    return L.total_bytes;
}

// HMX VTCM: Q/K/S as 32x32 f16 tiles. Bc = HTP_LID_KV_BLOCK (64 = 2 col tiles).
struct htp_lid_hmx_vtcm_layout {
    size_t off_q_tiles;
    size_t off_w;
    size_t off_qs;
    size_t off_k_raw[2];
    size_t off_k_tiles[2];
    size_t off_s_tiles[2];
    size_t off_m[2];
    size_t off_o[2];
    size_t off_hmx_scales;

    size_t size_q_row;
    size_t size_qs_row;
    size_t size_k_row;
    size_t size_m_row;
    size_t size_o_row;

    size_t size_q_tiles;
    size_t size_k_raw;
    size_t size_k_tiles;
    size_t size_s_tiles;
    size_t size_m_buf;
    size_t size_o_buf;
    size_t size_qs;

    uint32_t n_row_tiles;
    uint32_t n_col_tiles;
    uint32_t n_dot_tiles;

    size_t total_bytes;
};

static inline void htp_lid_hmx_vtcm_layout_build(struct htp_lid_hmx_vtcm_layout * L,
                                                 size_t D, size_t n_head, size_t n_tok, size_t n_threads) {
    const size_t M           = n_tok * n_head;
    const size_t M_pad       = hex_round_up(M, (size_t) HMX_FP16_TILE_N_ROWS);
    const size_t Bc          = HTP_LID_KV_BLOCK;
    const size_t n_row_tiles = M_pad / HMX_FP16_TILE_N_ROWS;
    const size_t n_col_tiles = Bc / HMX_FP16_TILE_N_COLS;
    const size_t n_dot_tiles = D / HMX_FP16_TILE_N_COLS;

    L->n_row_tiles = (uint32_t) n_row_tiles;
    L->n_col_tiles = (uint32_t) n_col_tiles;
    L->n_dot_tiles = (uint32_t) n_dot_tiles;

    L->size_q_row  = hex_round_up(D * sizeof(uint16_t), 128);
    L->size_qs_row = hex_round_up(D * sizeof(float), 128);
    L->size_k_row  = hex_round_up(D * sizeof(uint16_t), 128);
    L->size_m_row  = hex_round_up(Bc * sizeof(uint16_t), 128);
    L->size_o_row  = hex_round_up(Bc * sizeof(float), 128);

    L->size_q_tiles = n_row_tiles * n_dot_tiles * HMX_FP16_TILE_SIZE;
    L->size_k_raw   = Bc * L->size_k_row;
    L->size_k_tiles = n_col_tiles * n_dot_tiles * HMX_FP16_TILE_SIZE;
    L->size_s_tiles = n_row_tiles * n_col_tiles * HMX_FP16_TILE_SIZE;
    L->size_m_buf   = n_tok * L->size_m_row;
    L->size_o_buf   = n_tok * L->size_o_row;
    L->size_qs      = n_head * L->size_qs_row; // per-thread f32 Q staging

    size_t off = 0;
    const size_t size_w_row = hex_round_up(n_head * sizeof(float), 128);

    // HMX scatter needs each tile buffer 2KB-aligned (vtcm_base is). Pad between
    // non-tile regions so odd n_tok weight sizes cannot shift tile bases.
    VTCM_LAYOUT_ALLOC(off, off_q_tiles, L->size_q_tiles);
    VTCM_LAYOUT_ALLOC(off, off_w,       n_tok * size_w_row);
    VTCM_LAYOUT_ALLOC(off, off_qs,      n_threads * L->size_qs);
    off = hex_round_up(off, (size_t) HMX_FP16_TILE_SIZE);

    VTCM_LAYOUT_ALLOC(off, off_k_raw[0],   L->size_k_raw);
    VTCM_LAYOUT_ALLOC(off, off_k_raw[1],   L->size_k_raw);
    off = hex_round_up(off, (size_t) HMX_FP16_TILE_SIZE);

    VTCM_LAYOUT_ALLOC(off, off_k_tiles[0], L->size_k_tiles);
    VTCM_LAYOUT_ALLOC(off, off_k_tiles[1], L->size_k_tiles);
    VTCM_LAYOUT_ALLOC(off, off_s_tiles[0], L->size_s_tiles);
    VTCM_LAYOUT_ALLOC(off, off_s_tiles[1], L->size_s_tiles);
    off = hex_round_up(off, (size_t) HMX_FP16_TILE_SIZE);

    VTCM_LAYOUT_ALLOC(off, off_m[0],       L->size_m_buf);
    VTCM_LAYOUT_ALLOC(off, off_m[1],       L->size_m_buf);
    VTCM_LAYOUT_ALLOC(off, off_o[0],       L->size_o_buf);
    VTCM_LAYOUT_ALLOC(off, off_o[1],       L->size_o_buf);
    VTCM_LAYOUT_ALLOC(off, off_hmx_scales, 256);

    L->total_bytes = off;
}

static inline size_t htp_lid_hmx_compute_vtcm_usage(size_t D, size_t n_head, size_t n_tok, size_t n_threads) {
    struct htp_lid_hmx_vtcm_layout L;
    htp_lid_hmx_vtcm_layout_build(&L, D, n_head, n_tok, n_threads);
    return L.total_bytes;
}

#ifdef __cplusplus
}
#endif

#endif /* HTP_LIGHTNING_INDEXER_OPS_H */
