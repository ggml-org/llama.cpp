#pragma once

#include "ggml-cpu-impl.h"

#ifdef __cplusplus
extern "C" {
#endif
size_t ggml_fp16_mkl_workspace_size(const struct ggml_tensor * op);
bool ggml_fp16_mkl_compute(const struct ggml_compute_params * params, struct ggml_tensor * op);
#ifdef __cplusplus
}
#endif
