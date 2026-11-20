#pragma once

#include "ggml-hrx-common.h"

// supports_X must return true only for nodes that op_X can run. op_X returns false on failure.
// ops/binary.cpp is the reference for this pattern.

bool ggml_hrx_supports_binary(const ggml_hrx_device_context * dev, const ggml_tensor * op); // ADD MUL
bool ggml_hrx_op_binary(ggml_backend_hrx_context * ctx, const ggml_tensor * node);

bool ggml_hrx_supports_scale(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_scale(ggml_backend_hrx_context * ctx, const ggml_tensor * node);
bool ggml_hrx_supports_glu(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_glu(ggml_backend_hrx_context * ctx, const ggml_tensor * node);

bool ggml_hrx_supports_cpy(const ggml_hrx_device_context * dev, const ggml_tensor * op); // CPY DUP CONT
bool ggml_hrx_op_cpy(ggml_backend_hrx_context * ctx, const ggml_tensor * node);

bool ggml_hrx_supports_get_rows(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_get_rows(ggml_backend_hrx_context * ctx, const ggml_tensor * node);
bool ggml_hrx_supports_set_rows(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_set_rows(ggml_backend_hrx_context * ctx, const ggml_tensor * node);

bool ggml_hrx_supports_rms_norm(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_rms_norm(ggml_backend_hrx_context * ctx, const ggml_tensor * node);

bool ggml_hrx_supports_rope(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_rope(ggml_backend_hrx_context * ctx, const ggml_tensor * node);

bool ggml_hrx_supports_soft_max(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_soft_max(ggml_backend_hrx_context * ctx, const ggml_tensor * node);

bool ggml_hrx_supports_mul_mat(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_mul_mat(ggml_backend_hrx_context * ctx, const ggml_tensor * node);

bool ggml_hrx_supports_flash_attn_ext(const ggml_hrx_device_context * dev, const ggml_tensor * op);
bool ggml_hrx_op_flash_attn_ext(ggml_backend_hrx_context * ctx, const ggml_tensor * node);
