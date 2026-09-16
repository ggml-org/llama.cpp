#ifndef HTP_GROUPNORM_OPS_H
#define HTP_GROUPNORM_OPS_H

#include <stdint.h>

enum htp_group_norm_flags {
    HTP_GROUP_NORM_AFFINE = 1u << 0,
    HTP_GROUP_NORM_SILU   = 1u << 1,
};

struct htp_group_norm_kernel_params {
    uint32_t flags;
};


#endif
