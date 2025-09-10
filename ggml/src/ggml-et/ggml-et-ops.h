#pragma once

#include "ggml-et-common.h"

struct ggml_et_binary_params {
    ggml_tensor src0;
    ggml_tensor src1;
    ggml_tensor dst;
};

// Element map parameters for embarrassingly parallel binary operations (MUL, ADD, etc.)
// Operation type is determined by dst->op (GGML_OP_MUL, GGML_OP_ADD, etc.)
struct ggml_et_elmap_params {
    ggml_tensor src0;
    ggml_tensor src1;
    ggml_tensor dst;
};

typedef struct {
    int32_t n_past;
    int32_t n_dims;        // Number of dimensions to apply ROPE to (must be even)
    int32_t mode;          // ROPE mode (0=normal, 1=neox, 2=glm)
    int32_t n_ctx;
    int32_t n_ctx_orig;
    float   freq_base;     // Base frequency (usually 10000.0f)
    float   freq_scale;    // Frequency scaling factor
    float   ext_factor;    // Extension factor for YaRN
    float   attn_factor;   // Attention factor for YaRN
    float   beta_fast;     // Fast beta for YaRN
    float   beta_slow;     // Slow beta for YaRN
    int32_t sections[4];   // Sections for multi-modal ROPE
} rope_params_t;

struct ggml_et_rope_params {
    ggml_tensor src0;
    ggml_tensor src1;
    ggml_tensor src2;
    ggml_tensor dst;
    rope_params_t rope_params;
};

struct ggml_et_rms_norm_params {
    ggml_tensor src0;  // F32 input tensor
    ggml_tensor dst;   // F32 output tensor
    float eps;         // Epsilon parameter for numerical stability
};

struct ggml_et_glu_params {
    ggml_tensor src0;     // F32 input tensor A (or combined tensor if src1 is null)
    ggml_tensor src1;     // F32 input tensor B (null for single tensor mode)
    ggml_tensor dst;      // F32 output tensor (n/2 columns)
    int32_t glu_op_type;  // GLU operation type (REGLU=0, GEGLU=1, SWIGLU=2, etc.)
    int32_t swapped;      // Whether gate and value are swapped
};

bool ggml_et_op_mul(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_add(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_mul_mat(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_rope(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_rms_norm(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_glu(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_elmap(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
