#ifndef HTP_DSV4_HC_OPS_H
#define HTP_DSV4_HC_OPS_H

#include "hex-fastdiv.h"
#include "htp-ops.h"

#define HTP_DSV4_HC_CHUNK_ELEMS 32
#define HTP_DSV4_HC_DMA_CHUNKS 128

struct htp_dsv4_hc_pre_kernel_params {
    uint32_t n_threads;
    uint32_t n_chunks;
    uint32_t total_jobs;
    uint32_t jobs_per_thread;

    uint32_t dma_block_chunks;
    uint32_t dma_row_stride;
    uint32_t dma_buffer_size;
    uint32_t dma_input_size;
    uint32_t dma_vtcm_size;
    uint32_t dma_eligible;

    struct fastdiv_values div_n_chunks;
    struct fastdiv_values div_n_threads;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_dsv4_hc_pre_kernel_params) <= 128, "htp_dsv4_hc_pre_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_dsv4_hc_pre_kernel_params) <= 128, "htp_dsv4_hc_pre_kernel_params is too large for kernel_params blob");
#endif

// dma buffer holds residual rows 0..3, x at row 4, dst rows 5..8
#define HTP_DSV4_HC_POST_DMA_ROWS 9

struct htp_dsv4_hc_post_kernel_params {
    uint32_t n_threads;
    uint32_t n_chunks;
    uint32_t total_jobs;
    uint32_t jobs_per_thread;
    uint32_t has_comb;

    uint32_t dma_block_chunks;
    uint32_t dma_row_stride;
    uint32_t dma_buffer_size;
    uint32_t dma_vtcm_size;
    uint32_t dma_eligible;

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
