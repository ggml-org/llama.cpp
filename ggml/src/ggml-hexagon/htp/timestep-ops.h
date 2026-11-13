#ifndef HTP_TIMESTEP_OPS_H
#define HTP_TIMESTEP_OPS_H

#include "hex-fastdiv.h"
#include <stdint.h>

struct htp_timestep_kernel_params {
    uint32_t n_hvx;
    uint32_t n_rows;
    uint32_t dim;
    uint32_t half;
    uint32_t total_blocks;

    struct fastdiv_values div_rows;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_timestep_kernel_params) <= 128, "htp_timestep_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_timestep_kernel_params) <= 128, "htp_timestep_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_TIMESTEP_OPS_H
