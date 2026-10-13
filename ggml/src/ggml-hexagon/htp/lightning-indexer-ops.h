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

#define HTP_LID_KV_BLOCK   64 // kv rows per block, must be 64 (one f16 vector per transposed K row)
#define HTP_LID_TOKENS_MAX 32 // max tokens per Q tile

struct htp_lid_kernel_params {
    uint32_t n_threads;
    uint32_t n_tokens_tile;     // tokens per Q tile
    uint32_t n_kv_blocks;       // per stream
    uint32_t blocks_per_thread;
    uint32_t vtcm_size;
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

#ifdef __cplusplus
}
#endif

#endif /* HTP_LIGHTNING_INDEXER_OPS_H */
