#ifndef ALLREDUCE_OPS_H
#define ALLREDUCE_OPS_H

#include <stdint.h>

#define HTP_ALLREDUCE_MAX_RANKS 4

#ifdef __cplusplus
extern "C" {
#endif

enum htp_allreduce_kernel_type {
    HTP_ALLREDUCE_KERNEL_UNSUPPORTED = 0,
    HTP_ALLREDUCE_KERNEL_DMA_1D,
    HTP_ALLREDUCE_KERNEL_DMA_2D,
};

struct htp_allreduce_kernel_params {
    int32_t rank;
    int32_t n_ranks;
    int32_t n_threads;
    int32_t block_elems;          // 1D: block_elems, 2D: block_rows
    int32_t elems_per_thread;     // 1D: nelem_per_thread, 2D: nrows_per_thread
    int32_t vtcm_size_per_thread;
    int32_t vtcm_size;
    int32_t kernel_type;
    int32_t ne0;
    int32_t ne1;
    int32_t row_size_aligned;
    int32_t rank_elem_start;
    int32_t rank_nelem;
    int32_t n_dsts;
    int32_t is_row_bcast;
};

#ifdef __cplusplus
}
#endif

#endif /* ALLREDUCE_OPS_H */
