#ifndef HTP_REPEAT_OPS_H
#define HTP_REPEAT_OPS_H

#include "hex-fastdiv.h"
#include <stdint.h>

struct htp_repeat_kernel_params {
    uint32_t n_threads;
    uint32_t nr0;
    uint32_t nr1;
    uint32_t nr2;
    uint32_t nr3;

    struct fastdiv_values div_ne1;
    struct fastdiv_values div_ne12;
    struct fastdiv_values div_ne01;
    struct fastdiv_values div_ne02;
    struct fastdiv_values div_ne03;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_repeat_kernel_params) <= 128, "htp_repeat_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_repeat_kernel_params) <= 128, "htp_repeat_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_REPEAT_OPS_H
