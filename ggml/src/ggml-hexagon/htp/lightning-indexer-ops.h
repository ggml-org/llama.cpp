#ifndef HTP_LIGHTNING_INDEXER_OPS_H
#define HTP_LIGHTNING_INDEXER_OPS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "hex-common.h"
#include "hex-fastdiv.h"
#include "htp-vtcm.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HTP_LID_KV_BLOCK   64 // kv rows per block (HVX and HMX Bc)
#define HTP_LID_TOKENS_MAX 32 // max tokens per Q tile (HVX splat path)
#define HTP_LID_HMX_TOKENS_MAX 512 // HMX can keep more tokens; VTCM decides
#define HTP_LID_HMX_KV_BLOCK_MAX 2048
#define HTP_LID_HMX_MAX_QSETS    8    // streams in one HMX pipeline run
#define HTP_LID_N_BUF      4  // K / mask / output buffers per thread (two blocks computed, two in flight)

#ifndef HMX_FP16_TILE_N_COLS
#define HMX_FP16_TILE_N_COLS 32
#endif
#ifndef HMX_FP16_TILE_SIZE
#define HMX_FP16_TILE_SIZE 2048
#endif

enum htp_lid_kernel_type {
    HTP_LID_KERNEL_HVX = 0,
    HTP_LID_KERNEL_HMX = 1,
};

struct htp_lid_kernel_params {
    uint32_t n_threads;
    uint32_t n_tokens_tile;     // tokens per Q tile
    uint32_t n_kv_blocks;       // HVX: per stream, HMX: per run of n_qsets sets
    uint32_t blocks_per_thread;
    uint32_t vtcm_size;
    uint32_t kernel_type;       // htp_lid_kernel_type
    uint32_t k_row_size;        // K row bytes
    uint32_t kv_block;          // HMX: kv rows per block (multiple of 64)
    uint32_t n_qsets;           // HMX: Q sets kept in VTCM (streams run in one pipeline)
    uint32_t n_qbuf;            // HMX: Q buffers, 2: the next run's Q is prepped during the current run
    uint32_t kv_seg;            // HMX: rows per set, n_kv padded to 64
    uint32_t n_tt;              // HMX: token tiles per stream

    struct fastdiv_values div_n_head;
    struct fastdiv_values div_m_ne3;  // mask streams (broadcast)
    struct fastdiv_values div_kv_seg; // HMX
    struct fastdiv_values div_n_tt;   // HMX
    struct fastdiv_values div_n_ch;   // HMX: channel tiles per Q set
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_lid_kernel_params) <= 128, "htp_lid_kernel_params is too large");
#else
_Static_assert(sizeof(struct htp_lid_kernel_params) <= 128, "htp_lid_kernel_params is too large");
#endif

// VTCM layout shared by host (fit check) and device (buffer offsets).
// Shared regions: Q splats (one f16 vector per q element) and head weight splats for the current token tile.
// Per-thread regions: Q prep staging, K / mask / output buffers, two transposed K blocks and transpose scratch.
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
    size_t size_k_row;   // f16 K row, padded
    size_t size_k_raw_row;
    size_t size_m_row;   // f16 mask row of one kv block, padded
    size_t size_o_row;   // f32 output row of one kv block, padded

    size_t size_qs;      // per thread
    size_t size_k_buf;   // one K buffer
    size_t size_kt;      // per thread: two transposed K blocks, two 64x64 scratch tiles, Q8_0 block scales
    size_t size_m_buf;   // one mask buffer
    size_t size_o_buf;   // one output buffer

    size_t total_bytes;
};

static inline void htp_lid_vtcm_layout_build(struct htp_lid_vtcm_layout * L, size_t D, size_t n_head, size_t n_tok, size_t n_threads, size_t k_row_size) {
    L->size_q_row  = hex_round_up(D * sizeof(uint16_t), 128);
    L->size_qs_row = hex_round_up(D * sizeof(float), 128);
    L->size_k_row  = hex_round_up(D * sizeof(uint16_t), 128);
    L->size_k_raw_row = L->size_k_row > hex_round_up(k_row_size, 128) ? L->size_k_row : hex_round_up(k_row_size, 128);
    L->size_m_row  = hex_round_up(HTP_LID_KV_BLOCK * sizeof(uint16_t), 128);
    L->size_o_row  = hex_round_up(HTP_LID_KV_BLOCK * sizeof(float), 128);

    L->size_qs    = n_head * L->size_qs_row + hex_round_up(n_head * sizeof(float), 128) + L->size_q_row;
    L->size_k_buf = HTP_LID_KV_BLOCK * L->size_k_raw_row;
    L->size_kt    = (2 * hex_round_up(D, HTP_LID_KV_BLOCK) + 2 * HTP_LID_KV_BLOCK + D / 32) * 128;
    L->size_m_buf = n_tok * L->size_m_row;
    L->size_o_buf = n_tok * L->size_o_row;

    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_q,  n_tok * n_head * D * 128);
    VTCM_LAYOUT_ALLOC(off, off_w,  n_tok * n_head * 128);
    VTCM_LAYOUT_ALLOC(off, off_qs, n_threads * L->size_qs);
    VTCM_LAYOUT_ALLOC(off, off_k,  n_threads * HTP_LID_N_BUF * L->size_k_buf);
    VTCM_LAYOUT_ALLOC(off, off_kt, n_threads * L->size_kt);
    VTCM_LAYOUT_ALLOC(off, off_m,  n_threads * HTP_LID_N_BUF * L->size_m_buf);
    VTCM_LAYOUT_ALLOC(off, off_o,  n_threads * HTP_LID_N_BUF * L->size_o_buf);

    L->total_bytes = off;
}

static inline size_t htp_lid_compute_vtcm_usage(size_t D, size_t n_head, size_t n_tok, size_t n_threads, size_t k_row_size) {
    struct htp_lid_vtcm_layout L;
    htp_lid_vtcm_layout_build(&L, D, n_head, n_tok, n_threads, k_row_size);
    return L.total_bytes;
}

// HMX path. K is the activation (kv rows), Q the weights (one output channel per (token, head)).
// mm1: S = relu(K Q^T) * |w| per channel (HMX convert scale + pos clamp)
// mm2: O = S G, G[(t,h), t] = sign(w[t,h]). O is [kv, token], f16.
// Q weights, G, scales and w are per Q set (one per stream when all tokens fit one tile).
struct htp_lid_hmx_vtcm_layout {
    size_t off_qw;       // Q weight tiles [n_qsets][n_ch][n_dot]
    size_t off_g;        // G weight tiles [n_qsets][n_ch]
    size_t off_sc;       // per channel scales, 256 bytes per channel tile, [n_qsets][n_ch]
    size_t off_sc_id;    // identity scales
    size_t off_s;        // S tiles of one block and token tile, [n_rt][n_sg]
    size_t off_w;        // w of the token tile, [n_qsets][n_tok][n_head] f32
    size_t off_qs;       // per thread: f32 and f16 staging for 32 Q rows, gather scratch
    size_t off_k_raw[2]; // K rows, [Bc][size_k_row]
    size_t off_k_t[2];   // K tiles [n_rt][n_dot]
    size_t off_o[2];     // O tiles [n_rt][n_u]
    size_t off_m[4];     // mask rows [n_tok][size_m_row]
    size_t off_out[2];   // f32 output rows [n_tok][size_o_row]

    size_t size_qs_row;  // f32 Q row, padded
    size_t size_qh_row;  // f16 Q row, padded
    size_t size_k_row;   // f16 K row, padded
    size_t size_k_raw_row;
    size_t size_m_row;
    size_t size_o_row;
    size_t size_qs;      // per thread
    size_t size_w;       // per Q set

    uint32_t n_ch;       // channel tiles: ceil(n_tok * n_head / 32)
    uint32_t n_u;        // token tiles: ceil(n_tok / 32)
    uint32_t n_dot;      // D / 32
    uint32_t n_rt;       // kv row tiles per block: Bc / 32
    uint32_t n_sg;       // S tiles per (row tile, token tile): min(n_head, n_ch)

    size_t total_bytes;
};

static inline void htp_lid_hmx_vtcm_layout_build(struct htp_lid_hmx_vtcm_layout * L,
                                                  size_t D, size_t n_head, size_t n_tok, size_t Bc, size_t n_qsets, size_t n_threads, size_t k_row_size) {
    L->n_ch  = (uint32_t) ((n_tok * n_head + 31) / 32);
    L->n_u   = (uint32_t) ((n_tok + 31) / 32);
    L->n_dot = (uint32_t) (D / 32);
    L->n_rt  = (uint32_t) (Bc / 32);
    L->n_sg  = (uint32_t) (n_head < L->n_ch ? n_head : L->n_ch);

    L->size_qs_row = hex_round_up(D * sizeof(float), 128);
    L->size_qh_row = hex_round_up(D * sizeof(uint16_t), 128);
    L->size_k_row  = hex_round_up(D * sizeof(uint16_t), 128);
    L->size_k_raw_row = L->size_k_row > hex_round_up(k_row_size, 128) ? L->size_k_row : hex_round_up(k_row_size, 128);
    L->size_m_row  = hex_round_up(Bc * sizeof(uint16_t), 128);
    L->size_o_row  = hex_round_up(Bc * sizeof(float), 128);
    L->size_qs     = 32 * (L->size_qs_row + L->size_qh_row) + hex_round_up(Bc / 64 * 128, 128);
    L->size_w      = hex_round_up(n_tok * n_head * sizeof(float), 128);

    const size_t tile = HMX_FP16_TILE_SIZE;

    // tile regions first: they need 2 KB alignment
    size_t off = 0;
    VTCM_LAYOUT_ALLOC(off, off_qw,     n_qsets * L->n_ch * L->n_dot * tile);
    VTCM_LAYOUT_ALLOC(off, off_g,      n_qsets * L->n_ch * tile);
    VTCM_LAYOUT_ALLOC(off, off_s,      (size_t) L->n_rt * L->n_sg * tile);
    VTCM_LAYOUT_ALLOC(off, off_k_t[0], (size_t) L->n_rt * L->n_dot * tile);
    VTCM_LAYOUT_ALLOC(off, off_k_t[1], (size_t) L->n_rt * L->n_dot * tile);
    VTCM_LAYOUT_ALLOC(off, off_o[0],   (size_t) L->n_rt * L->n_u * tile);
    VTCM_LAYOUT_ALLOC(off, off_o[1],   (size_t) L->n_rt * L->n_u * tile);
    VTCM_LAYOUT_ALLOC(off, off_sc,     n_qsets * L->n_ch * 256);
    VTCM_LAYOUT_ALLOC(off, off_sc_id,  256);
    VTCM_LAYOUT_ALLOC(off, off_w,      n_qsets * L->size_w);
    VTCM_LAYOUT_ALLOC(off, off_qs,     n_threads * L->size_qs);
    for (int i = 0; i < 2; i++) {
        VTCM_LAYOUT_ALLOC(off, off_k_raw[i], Bc * L->size_k_raw_row);
    }
    for (int i = 0; i < 4; i++) {
        VTCM_LAYOUT_ALLOC(off, off_m[i], n_tok * L->size_m_row);
    }
    for (int i = 0; i < 2; i++) {
        VTCM_LAYOUT_ALLOC(off, off_out[i], n_tok * L->size_o_row);
    }

    L->total_bytes = off;
}

static inline size_t htp_lid_hmx_compute_vtcm_usage(size_t D, size_t n_head, size_t n_tok, size_t Bc, size_t n_qsets, size_t n_threads, size_t k_row_size) {
    struct htp_lid_hmx_vtcm_layout L;
    htp_lid_hmx_vtcm_layout_build(&L, D, n_head, n_tok, Bc, n_qsets, n_threads, k_row_size);
    return L.total_bytes;
}

#ifdef __cplusplus
}
#endif

#endif /* HTP_LIGHTNING_INDEXER_OPS_H */
