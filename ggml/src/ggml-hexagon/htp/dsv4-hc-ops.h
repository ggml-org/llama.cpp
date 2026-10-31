#ifndef HTP_DSV4_HC_OPS_H
#define HTP_DSV4_HC_OPS_H

#include "hex-fastdiv.h"
#include "htp-ops.h"

// one job is a 32-element (one HVX vector) chunk of a token row, dma blocks group up to 128 chunks of one token
#define HTP_DSV4_HC_CHUNK_ELEMS 32
#define HTP_DSV4_HC_DMA_CHUNKS  128

enum htp_dsv4_hc_pre_kernel_type {
    HTP_DSV4_HC_PRE_KERNEL_PLAIN = 0, // per-token weights [n_hc, n_tokens]
    HTP_DSV4_HC_PRE_KERNEL_GATED = 1, // per-element sigmoid gate [n_embd, n_hc, n_tokens]
};

// dma buffer: x rows 0..n_hc-1, gate rows at dma_g_offset (gated only), dst row at dma_d_offset
struct htp_dsv4_hc_pre_kernel_params {
    uint32_t kernel_type;
    uint32_t n_threads;
    uint32_t n_hc;
    uint32_t n_chunks;
    uint32_t row_size;
    uint32_t total_jobs;
    uint32_t jobs_per_thread;

    uint32_t dma_block_chunks;
    uint32_t dma_row_stride;
    uint32_t dma_g_offset;
    uint32_t dma_d_offset;
    uint32_t dma_buffer_size;
    uint32_t dma_vtcm_size;

    struct fastdiv_values div_n_chunks;
    struct fastdiv_values div_n_threads;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_dsv4_hc_pre_kernel_params) <= 128, "htp_dsv4_hc_pre_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_dsv4_hc_pre_kernel_params) <= 128, "htp_dsv4_hc_pre_kernel_params is too large for kernel_params blob");
#endif

// dma buffer: residual rows 0..3, x row 4, dst rows 5..8
#define HTP_DSV4_HC_POST_DMA_ROWS 9

struct htp_dsv4_hc_post_kernel_params {
    uint32_t n_threads;
    uint32_t n_chunks;
    uint32_t row_size;
    uint32_t total_jobs;
    uint32_t jobs_per_thread;
    uint32_t has_comb;

    uint32_t dma_block_chunks;
    uint32_t dma_row_stride;
    uint32_t dma_buffer_size;
    uint32_t dma_vtcm_size;

    struct fastdiv_values div_n_chunks;
    struct fastdiv_values div_n_threads;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_dsv4_hc_post_kernel_params) <= 128, "htp_dsv4_hc_post_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_dsv4_hc_post_kernel_params) <= 128, "htp_dsv4_hc_post_kernel_params is too large for kernel_params blob");
#endif

int op_dsv4_hc_pre(struct htp_ops_context * octx);
int op_dsv4_hc_post(struct htp_ops_context * octx);

#endif // HTP_DSV4_HC_OPS_H
