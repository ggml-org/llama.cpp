#pragma once

#include "ggml-et-common.h"

struct ggml_et_binary_params {
    ggml_tensor src0;
    ggml_tensor src1;
    ggml_tensor dst;
};

bool ggml_et_op_mul(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
