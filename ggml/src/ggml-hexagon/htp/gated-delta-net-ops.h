#ifndef HTP_GATED_DELTA_NET_OPS_H
#define HTP_GATED_DELTA_NET_OPS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "hex-fastdiv.h"
#include "hex-common.h"
#include "htp-vtcm.h"

#define HTP_GDN_MAX_SV     128
#define HTP_GDN_CHUNK_SIZE 64

#ifndef HMX_FP16_TILE_SIZE
#define HMX_FP16_TILE_SIZE 2048
#endif

enum htp_gdn_kernel_type {
    HTP_GDN_KERNEL_HVX_RECURRENT = 0,
    HTP_GDN_KERNEL_HMX_CHUNKED   = 1,
};

struct htp_gdn_kernel_params {
    uint8_t  kernel_type;
    uint8_t  pipeline;
    uint16_t chunk_size;
    uint16_t n_chunks;
    uint16_t n_heads_batch;

    uint32_t n_threads;
    uint32_t S_v;
    uint32_t H;
    uint32_t n_tokens;
    uint32_t n_seqs;
    uint32_t K;

    uint32_t total_rows;
    uint32_t row_start;
    uint32_t nrows;
    uint32_t rows_per_thread;

    uint32_t kda;
    uint32_t state_aligned;
    uint32_t vtcm_per_thread;
    uint32_t vtcm_size;
    uint32_t state_seq_stride;
    uint32_t state_size_per_snap;

    float    scale;

    struct fastdiv_values div_H;
    struct fastdiv_values div_q1;
    struct fastdiv_values div_k1;
    struct fastdiv_values div_rq3;
    struct fastdiv_values div_rk3;
    struct fastdiv_values div_n_threads;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_gdn_kernel_params) <= 128, "htp_gdn_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_gdn_kernel_params) <= 128, "htp_gdn_kernel_params is too large for kernel_params blob");
#endif

struct htp_gdn_vtcm_layout {
    size_t state_aligned;
    size_t bytes_per_thread;
    size_t total_bytes;
};

static inline void htp_gdn_vtcm_layout_build(
    struct htp_gdn_vtcm_layout * layout,
    uint32_t S_v,
    uint32_t n_threads
) {
    size_t state_bytes = (size_t) S_v * S_v * sizeof(float);
    layout->state_aligned = hex_round_up(state_bytes, 128);
    layout->bytes_per_thread = 2 * layout->state_aligned;
    layout->total_bytes = layout->bytes_per_thread * n_threads;
}

struct htp_gdn_hmx_vtcm_layout {
    size_t off_s_state;
    size_t off_s_f16;
    size_t off_s_col_tiles;
    size_t off_s_update_f32;
    size_t off_s_update_tiles;

    size_t off_q_f32[2];
    size_t off_k_f32[2];
    size_t off_v_f32[2];
    size_t off_g_f32[2];
    size_t off_b_f32[2];
    size_t off_g_raw[2];
    size_t off_b_raw[2];
    size_t off_o_f32[2];

    size_t off_v_inter_f32;
    size_t off_o_inter_f32;
    size_t off_o_intra_f32;
    size_t off_k_f16;
    size_t off_v_prime_f16;
    size_t off_delta_f16;
    size_t off_d_f16;

    size_t off_q_row_tiles;
    size_t off_q_prime_row_tiles;
    size_t off_k_row_tiles;
    size_t off_k_col_tiles;
    size_t off_k_prime_row_tiles;
    size_t off_k_col_tiles_64x128;
    size_t off_kk_tiles;
    size_t off_qk_tiles;
    size_t off_v_inter_tiles;
    size_t off_o_inter_tiles;
    size_t off_inv_row_tiles;
    size_t off_a_row_tiles;
    size_t off_v_prime_col_tiles;
    size_t off_delta_tiles;
    size_t off_delta_col_tiles;
    size_t off_o_intra_tiles;
    size_t off_d_row_tiles;

    size_t off_gamma;
    size_t off_lambda_init;
    size_t off_decay_m;
    size_t off_decay_a;
    size_t off_rows_kk;
    size_t off_rows_qk;
    size_t off_rows_inv;
    size_t off_rows_a;

    size_t off_thread_scratch;
    size_t off_attn_rem;
    size_t off_scales_1;

    size_t state_f32_bytes;
    size_t state_f16_bytes;
    size_t state_tiles_bytes;
    size_t dma_chunk_bytes;
    size_t act_f16_bytes;
    size_t tile_64xSv_bytes;
    size_t tile_64x64_bytes;

    uint32_t n_heads_batch;
    uint32_t n_threads;
    bool     pipeline;
    size_t   total_bytes;
};

static inline void htp_gdn_hmx_vtcm_layout_build(
    struct htp_gdn_hmx_vtcm_layout * L,
    uint32_t S_v,
    uint32_t chunk_size,
    uint32_t n_heads_batch,
    uint32_t n_threads,
    bool     pipeline
) {
    memset(L, 0, sizeof(*L));
    L->n_heads_batch = n_heads_batch;
    L->n_threads     = n_threads;
    L->pipeline      = pipeline;

    const size_t bh = (size_t) n_heads_batch;
    const size_t nth = (size_t) (n_threads > 0 ? n_threads : 1);

    const size_t state_f32_sz   = hex_round_up(S_v * S_v * sizeof(float), 2048);
    const size_t state_f16_sz   = hex_round_up(S_v * S_v * sizeof(__fp16), 2048);
    const size_t n_sv_tiles     = S_v / 32;
    const size_t state_tiles_sz = n_sv_tiles * n_sv_tiles * 2048;

    const size_t dma_chunk_sz   = hex_round_up(chunk_size * S_v * sizeof(float), 2048);
    const size_t dma_scalar_sz  = hex_round_up(chunk_size * sizeof(float), 128);

    const size_t act_f16_sz     = hex_round_up(chunk_size * S_v * sizeof(__fp16), 2048);
    const size_t tile_64xSv_sz  = 2 * n_sv_tiles * 2048;
    const size_t tile_64x64_sz  = 4 * 2048;

    const size_t decay_sz       = 64 * 64 * sizeof(__fp16);
    const size_t row_vecs_sz    = 64 * 128;

    L->state_f32_bytes   = state_f32_sz;
    L->state_f16_bytes   = state_f16_sz;
    L->state_tiles_bytes = state_tiles_sz;
    L->dma_chunk_bytes   = dma_chunk_sz;
    L->act_f16_bytes     = act_f16_sz;
    L->tile_64xSv_bytes  = tile_64xSv_sz;
    L->tile_64x64_bytes  = tile_64x64_sz;

    size_t off = 0;

    VTCM_LAYOUT_ALLOC(off, off_s_state,        bh * state_f32_sz);
    VTCM_LAYOUT_ALLOC(off, off_s_f16,          bh * state_f16_sz);
    VTCM_LAYOUT_ALLOC(off, off_s_col_tiles,    bh * state_tiles_sz);
    VTCM_LAYOUT_ALLOC(off, off_s_update_f32,   bh * state_f32_sz);
    VTCM_LAYOUT_ALLOC(off, off_s_update_tiles, bh * state_tiles_sz);

    VTCM_LAYOUT_ALLOC(off, off_q_f32[0], bh * dma_chunk_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_q_f32[1], bh * dma_chunk_sz, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_k_f32[0], bh * dma_chunk_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_k_f32[1], bh * dma_chunk_sz, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_v_f32[0], bh * dma_chunk_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_v_f32[1], bh * dma_chunk_sz, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_g_f32[0], bh * dma_scalar_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_g_f32[1], bh * dma_scalar_sz, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_b_f32[0], bh * dma_scalar_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_b_f32[1], bh * dma_scalar_sz, pipeline);
    const size_t raw_gb_sz = hex_round_up(bh * chunk_size * sizeof(float), 128);
    VTCM_LAYOUT_ALLOC(off, off_g_raw[0], raw_gb_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_g_raw[1], raw_gb_sz, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_b_raw[0], raw_gb_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_b_raw[1], raw_gb_sz, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_o_f32[0], bh * dma_chunk_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_o_f32[1], bh * dma_chunk_sz, pipeline);

    VTCM_LAYOUT_ALLOC(off, off_v_inter_f32, bh * dma_chunk_sz);
    VTCM_LAYOUT_ALLOC(off, off_o_inter_f32, bh * dma_chunk_sz);
    VTCM_LAYOUT_ALLOC(off, off_o_intra_f32, bh * dma_chunk_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_f16,       bh * act_f16_sz);
    VTCM_LAYOUT_ALLOC(off, off_v_prime_f16, bh * act_f16_sz);
    VTCM_LAYOUT_ALLOC(off, off_delta_f16,   bh * act_f16_sz);
    VTCM_LAYOUT_ALLOC(off, off_d_f16,       bh * act_f16_sz);

    VTCM_LAYOUT_ALLOC(off, off_q_row_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_q_prime_row_tiles,  bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_row_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_col_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_prime_row_tiles,  bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_k_col_tiles_64x128, bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_kk_tiles,           bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_qk_tiles,           bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_v_inter_tiles,      bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_o_inter_tiles,      bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_inv_row_tiles,      bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_a_row_tiles,        bh * tile_64x64_sz);
    VTCM_LAYOUT_ALLOC(off, off_v_prime_col_tiles,  bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_delta_tiles,        bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_delta_col_tiles,    bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_o_intra_tiles,      bh * tile_64xSv_sz);
    VTCM_LAYOUT_ALLOC(off, off_d_row_tiles,        bh * tile_64xSv_sz);

    VTCM_LAYOUT_ALLOC(off, off_gamma,       bh * hex_round_up(chunk_size * sizeof(float), 128));
    VTCM_LAYOUT_ALLOC(off, off_lambda_init, bh * hex_round_up(chunk_size * sizeof(float), 128));
    VTCM_LAYOUT_ALLOC(off, off_decay_m,     bh * decay_sz);
    VTCM_LAYOUT_ALLOC(off, off_decay_a,     bh * decay_sz);
    VTCM_LAYOUT_ALLOC(off, off_rows_kk,     bh * row_vecs_sz);
    VTCM_LAYOUT_ALLOC(off, off_rows_qk,     bh * row_vecs_sz);
    VTCM_LAYOUT_ALLOC(off, off_rows_inv,    bh * row_vecs_sz);
    VTCM_LAYOUT_ALLOC(off, off_rows_a,      bh * row_vecs_sz);

    const size_t thread_scratch_sz = 64 * 128;
    VTCM_LAYOUT_ALLOC(off, off_thread_scratch, nth * thread_scratch_sz);
    VTCM_LAYOUT_ALLOC(off, off_attn_rem,       nth * (128 * sizeof(float)));
    VTCM_LAYOUT_ALLOC(off, off_scales_1,       256);

    L->total_bytes = off;
}

static inline bool htp_gdn_hmx_solve_layout(
    struct htp_gdn_hmx_vtcm_layout * layout_out,
    uint32_t S_v,
    uint32_t chunk_size,
    uint32_t total_rows,
    size_t   vtcm_budget,
    uint32_t n_threads,
    bool     pipeline,
    uint32_t * n_heads_batch_out
) {
    uint32_t max_batch = 8;
    if (max_batch > total_rows) {
        max_batch = total_rows;
    }
    if (max_batch > n_threads) {
        max_batch = n_threads;
    }
    static const uint32_t candidates[] = { 8, 6, 4, 2, 1 };
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        uint32_t bh = candidates[i];
        if (bh > max_batch) {
            continue;
        }
        struct htp_gdn_hmx_vtcm_layout L;
        htp_gdn_hmx_vtcm_layout_build(&L, S_v, chunk_size, bh, n_threads, pipeline);
        if (L.total_bytes <= vtcm_budget) {
            *layout_out = L;
            *n_heads_batch_out = bh;
            return true;
        }
    }
    if (pipeline) {
        return htp_gdn_hmx_solve_layout(layout_out, S_v, chunk_size, total_rows, vtcm_budget, n_threads, false, n_heads_batch_out);
    }
    return false;
}

#endif // HTP_GATED_DELTA_NET_OPS_H
