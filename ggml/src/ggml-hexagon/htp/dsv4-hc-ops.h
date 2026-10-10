#ifndef HTP_DSV4_HC_OPS_H
#define HTP_DSV4_HC_OPS_H

#include "htp-ops.h"

struct htp_dsv4_hc_pre_kernel_params {
    uint32_t n_threads;
};

int op_dsv4_hc_pre(struct htp_ops_context * octx);

#endif // HTP_DSV4_HC_OPS_H
