#ifndef HTP_SOLVE_TRI_OPS_H
#define HTP_SOLVE_TRI_OPS_H

#include "hex-fastdiv.h"
#include <stdint.h>

struct htp_solve_tri_kernel_params {
    uint32_t n_threads;
    uint32_t total_batches;
    uint32_t total_jobs;
    uint32_t k_chunks;
    uint32_t col_block;
    uint32_t batched;
    uint32_t batches_per_chunk;

    struct fastdiv_values div_ne02;
    struct fastdiv_values div_k_chunks;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_solve_tri_kernel_params) <= 128, "htp_solve_tri_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_solve_tri_kernel_params) <= 128, "htp_solve_tri_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_SOLVE_TRI_OPS_H
