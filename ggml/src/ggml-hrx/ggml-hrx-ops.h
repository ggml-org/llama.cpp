#pragma once

#include "ggml-hrx-common.h"

// supports_X must return true only for nodes that op_X can run. op_X returns false on failure.

bool ggml_hrx_supports_scale(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_scale(ggml_backend_hrx_context * ctx, const ggml_tensor * node);

bool ggml_hrx_supports_cpy(const ggml_hrx_device_context * dev, const ggml_tensor * op); // CPY DUP CONT
bool ggml_hrx_op_cpy(ggml_backend_hrx_context * ctx, const ggml_tensor * node);

bool ggml_hrx_supports_set_rows(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_set_rows(ggml_backend_hrx_context * ctx, const ggml_tensor * node);

bool ggml_hrx_supports_mul_mat(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_mul_mat(ggml_backend_hrx_context * ctx, const ggml_tensor * node);
