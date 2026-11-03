#ifndef HTP_SUM_ROWS_OPS_H
#define HTP_SUM_ROWS_OPS_H

#include <stdint.h>

struct htp_sum_rows_kernel_params {
    uint32_t n_threads;
    uint32_t elems_per_thread;
};

#if defined(__cplusplus)
static_assert(sizeof(struct htp_sum_rows_kernel_params) <= 128, "htp_sum_rows_kernel_params is too large for kernel_params blob");
#else
_Static_assert(sizeof(struct htp_sum_rows_kernel_params) <= 128, "htp_sum_rows_kernel_params is too large for kernel_params blob");
#endif

#endif // HTP_SUM_ROWS_OPS_H
