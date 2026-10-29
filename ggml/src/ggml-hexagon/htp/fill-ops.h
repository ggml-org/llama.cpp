#ifndef HTP_FILL_OPS_H
#define HTP_FILL_OPS_H

#include "hex-fastdiv.h"
#include <stdint.h>

enum htp_fill_kernel_type {
    HTP_FILL_KERNEL_CONTIGUOUS = 0,
    HTP_FILL_KERNEL_STRIDED    = 1,
};

struct htp_fill_kernel_params {
    uint8_t  kernel_type;
    uint32_t n_threads;
    uint32_t total_rows;
    uint32_t elem_size;
    uint32_t rows_per_chunk;

    struct fastdiv_values div_ne1;
    struct fastdiv_values div_ne1_ne2;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_fill_kernel_params) <= 128, "htp_fill_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_fill_kernel_params) <= 128, "htp_fill_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_FILL_OPS_H
