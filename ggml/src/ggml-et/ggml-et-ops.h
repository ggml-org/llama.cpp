#pragma once

#include "ggml-et-common.h"
#include "ggml.h"
#include <inttypes.h>

// Performance logging macros for ET ops
// Logs in machine-parseable pipe-delimited format: ET_PERF|field=value|...
#ifdef ET_PERF_RECORD
#define ET_PERF_START() int64_t _et_perf_start = ggml_time_us()

#define ET_PERF_END(op_name, kernel_name, node) do { \
    int64_t _et_perf_end = ggml_time_us(); \
    int64_t _et_perf_duration = _et_perf_end - _et_perf_start; \
    GGML_LOG_DEBUG("ET_PERF|op=%s|kernel=%s|duration_us=%" PRId64 "|tensor=%s|shape=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]|start_us=%" PRId64 "|end_us=%" PRId64 "\n", \
        op_name, kernel_name, _et_perf_duration, (node)->name, \
        (node)->ne[0], (node)->ne[1], (node)->ne[2], (node)->ne[3], \
        _et_perf_start, _et_perf_end); \
} while(0)

#define ET_PERF_END_EXT(op_name, kernel_name, node, fmt, ...) do { \
    int64_t _et_perf_end = ggml_time_us(); \
    int64_t _et_perf_duration = _et_perf_end - _et_perf_start; \
    GGML_LOG_DEBUG("ET_PERF|op=%s|kernel=%s|duration_us=%" PRId64 "|tensor=%s|shape=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]|start_us=%" PRId64 "|end_us=%" PRId64 "|" fmt "\n", \
        op_name, kernel_name, _et_perf_duration, (node)->name, \
        (node)->ne[0], (node)->ne[1], (node)->ne[2], (node)->ne[3], \
        _et_perf_start, _et_perf_end, ##__VA_ARGS__); \
} while(0)
#else

#define ET_PERF_START() do {} while(0)
#define ET_PERF_END_EXT(op_name, kernel_name, node, fmt, ...) do {(void)(node); } while(0)
#define ET_PERF_END(op_name, kernel_name, node) do {(void)(node);} while(0)

#endif

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

struct ggml_et_norm_params {
    ggml_tensor src0;  // F32 input tensor
    ggml_tensor dst;   // F32 output tensor
    float eps;         // Epsilon parameter for numerical stability
};

struct ggml_et_l2_norm_params {
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

struct ggml_et_softmax_params {
    ggml_tensor src0;     // F32 input tensor
    ggml_tensor src1;     // F32 mask tensor (optional, may be zeroed if not used)
    ggml_tensor src2;     // F32 sinks tensor (optional, may be zeroed if not used)
    ggml_tensor dst;      // F32 output tensor
    float scale;          // Scale factor
    float max_bias;       // Max bias for ALiBi (0.0f if not used)
};

struct ggml_et_get_rows_params {
    ggml_tensor src0;     // Data tensor (F32 or Q8_0)
    ggml_tensor src1;     // Row indices tensor (I32)
    ggml_tensor dst;      // Output tensor (F32)
};

struct ggml_et_cont_params {
    ggml_tensor src0;     // F32 input tensor (non-contiguous)
    ggml_tensor dst;      // F32 output tensor (contiguous)
};

struct ggml_et_concat_params {
    ggml_tensor src0;     // F32 input tensor 0
    ggml_tensor src1;     // F32 input tensor 1
    ggml_tensor dst;      // F32 output tensor
    int32_t dim;          // Concatenation dimension
};

struct ggml_et_repeat_params {
    ggml_tensor src0;     // F32 input tensor (tile)
    ggml_tensor dst;      // F32 output tensor (tiled result)
};

struct ggml_et_rwkv_wkv6_params {
    float* k;           // src[0]: [S, H, T]  key
    float* v;           // src[1]: [S, H, T]  value
    float* r;           // src[2]: [S, H, T]  receptance
    float* tf;          // src[3]: [S, H]     time_faaaa (per-head)
    float* td;          // src[4]: [S, H, T]  time_decay
    float* state_in;    // src[5]: [S*S*H, n_seqs]  initial state
    float* dst;         // [C, T + S*n_seqs]  output + state_out
    int32_t C;          // total channels (S * H)
    int32_t H;          // number of heads
    int32_t S;          // head size
    int32_t T;          // number of tokens
    int32_t n_seqs;     // number of sequences
};

struct ggml_et_rwkv_wkv7_params {
    float* r;           // [S, H, T]  receptance
    float* w;           // [S, H, T]  decay
    float* k;           // [S, H, T]  key
    float* v;           // [S, H, T]  value
    float* a;           // [S, H, T]  bonus gate
    float* b;           // [S, H, T]  bonus key
    float* state_in;    // [S*S*H, n_seqs]  initial state
    float* dst;         // [C, T + S*n_seqs]  output + state_out
    int32_t C;          // total channels (S * H)
    int32_t H;          // number of heads
    int32_t S;          // head size
    int32_t T;          // number of tokens
    int32_t n_seqs;     // number of sequences
};

struct ggml_et_set_rows_params {
    ggml_tensor src0;     // F32 source data tensor
    ggml_tensor src1;     // I64 row indices tensor
    ggml_tensor dst;      // F32/F16 destination tensor
};

struct ggml_et_rms_norm_mul_params {
    ggml_tensor src0;      // F32 input tensor (to be normalized)
    ggml_tensor src1;      // F32 weights tensor (element-wise multiply)
    ggml_tensor dst;       // F32 output tensor
    float eps;             // Epsilon for numerical stability
};

struct ggml_et_mul_mat_id_params {
    ggml_tensor src0;     // Expert weight matrices (Q8_0/F16/F32) [K, M, n_expert]
    ggml_tensor src1;     // Activations (F32) [K, n_expert_used, batch]
    ggml_tensor src2;     // Expert indices (I32) [n_expert_used, batch]
    ggml_tensor dst;      // Output (F32) [M, n_expert_used, batch, 1]
};

struct ggml_et_sqr_params {
    ggml_tensor src0;     // F32 input tensor
    ggml_tensor dst;      // F32 output tensor
};

struct ggml_et_unary_params {
    ggml_tensor src0;     // F32 input tensor
    ggml_tensor dst;      // F32 output tensor
    int32_t unary_op;     // ggml_unary_op enum value
};

struct ggml_et_sum_rows_params {
    ggml_tensor src0;     // F32 input tensor [ne00, ne01, ne02, ne03]
    ggml_tensor dst;      // F32 output tensor [1, ne01, ne02, ne03]
};

struct ggml_et_scale_params {
    ggml_tensor src0;     // F32 input tensor
    ggml_tensor dst;      // F32 output tensor
    float scale;          // Scale factor
    float bias;           // Bias (additive offset)
};

bool ggml_et_op_sqr(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_unary(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_sum_rows(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_scale(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_mul(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_add(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_sub(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_mul_mat(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_mul_mat_id(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_rope(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_rms_norm(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_norm(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_l2_norm(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_glu(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_softmax(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_get_rows(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_set_rows(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_cont(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_concat(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_repeat(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_rwkv_wkv6(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_rwkv_wkv7(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_elmap(ggml_backend_et_device_context* dev_ctx, const ggml_tensor* node);
bool ggml_et_op_rms_norm_mul(ggml_backend_et_device_context* dev_ctx,
                             const ggml_tensor* rms_norm_node,
                             const ggml_tensor* mul_node);
