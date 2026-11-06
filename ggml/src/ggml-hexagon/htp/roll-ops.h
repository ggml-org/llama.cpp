#ifndef HTP_ROLL_OPS_H
#define HTP_ROLL_OPS_H

#include "hex-fastdiv.h"
#include <stdint.h>

enum htp_roll_kernel_type {
    HTP_ROLL_KERNEL_DMA_CONTIGUOUS = 1,
    HTP_ROLL_KERNEL_DMA_STRIDED    = 2,
    HTP_ROLL_KERNEL_HVX            = 3,
};

struct htp_roll_kernel_params {
    uint8_t  kernel_type;
    uint32_t n_threads;
    uint32_t total_rows;
    uint32_t i0_src0;
    uint32_t n0;

    struct fastdiv_values div_ne1;
    struct fastdiv_values div_ne2_ne1;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_roll_kernel_params) <= 128, "htp_roll_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_roll_kernel_params) <= 128, "htp_roll_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_ROLL_OPS_H
