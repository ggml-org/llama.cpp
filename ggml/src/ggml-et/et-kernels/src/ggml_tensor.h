//******************************************************************************
// GGML Tensor Definitions for Bare Metal Kernels
// Must match the full GGML structure used by ggml-et backend
//******************************************************************************

#ifndef GGML_TENSOR_H
#define GGML_TENSOR_H

#include <stdint.h>
#include <stddef.h> // for size_t

#define GGML_MAX_DIMS 4
#define GGML_MAX_SRC 10
#define GGML_MAX_NAME 64
#define GGML_MAX_OP_PARAMS 64

// Data types supported by GGML
enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
    // GGML_TYPE_Q4_0_4_8 = 32,
    // GGML_TYPE_Q4_0_8_8 = 33,
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    // GGML_TYPE_IQ4_NL_4_4 = 36,
    // GGML_TYPE_IQ4_NL_4_8 = 37,
    // GGML_TYPE_IQ4_NL_8_8 = 38,
    GGML_TYPE_MXFP4   = 39, // MXFP4 (1 block)
    GGML_TYPE_COUNT   = 40,
};

// Operations supported by GGML
enum ggml_op {
    GGML_OP_NONE = 0,

    GGML_OP_DUP,
    GGML_OP_ADD,
    GGML_OP_ADD_ID,
    GGML_OP_ADD1,
    GGML_OP_ACC,
    GGML_OP_SUB,
    GGML_OP_MUL,
    GGML_OP_DIV,
    GGML_OP_SQR,
    GGML_OP_SQRT,
    GGML_OP_LOG,
    GGML_OP_SIN,
    GGML_OP_COS,
    GGML_OP_SUM,
    GGML_OP_SUM_ROWS,
    GGML_OP_MEAN,
    GGML_OP_ARGMAX,
    GGML_OP_COUNT_EQUAL,
    GGML_OP_REPEAT,
    GGML_OP_REPEAT_BACK,
    GGML_OP_CONCAT,
    GGML_OP_SILU_BACK,
    GGML_OP_NORM,
    GGML_OP_RMS_NORM,
    GGML_OP_RMS_NORM_BACK,
    GGML_OP_GROUP_NORM,
    GGML_OP_L2_NORM,

    GGML_OP_MUL_MAT,
    GGML_OP_MUL_MAT_ID,
    GGML_OP_OUT_PROD,

    GGML_OP_SCALE,
    GGML_OP_SET,
    GGML_OP_CPY,
    GGML_OP_CONT,
    GGML_OP_RESHAPE,
    GGML_OP_VIEW,
    GGML_OP_PERMUTE,
    GGML_OP_TRANSPOSE,
    GGML_OP_GET_ROWS,
    GGML_OP_GET_ROWS_BACK,
    GGML_OP_SET_ROWS,
    GGML_OP_DIAG,
    GGML_OP_DIAG_MASK_INF,
    GGML_OP_DIAG_MASK_ZERO,
    GGML_OP_SOFT_MAX,
    GGML_OP_SOFT_MAX_BACK,
    GGML_OP_ROPE,
    GGML_OP_ROPE_BACK,
    GGML_OP_CLAMP,
    GGML_OP_CONV_TRANSPOSE_1D,
    GGML_OP_IM2COL,
    GGML_OP_IM2COL_BACK,
    GGML_OP_IM2COL_3D,
    GGML_OP_CONV_2D,
    GGML_OP_CONV_3D,
    GGML_OP_CONV_2D_DW,
    GGML_OP_CONV_TRANSPOSE_2D,
    GGML_OP_POOL_1D,
    GGML_OP_POOL_2D,
    GGML_OP_POOL_2D_BACK,
    GGML_OP_UPSCALE,
    GGML_OP_PAD,
    GGML_OP_PAD_REFLECT_1D,
    GGML_OP_ROLL,
    GGML_OP_ARANGE,
    GGML_OP_TIMESTEP_EMBEDDING,
    GGML_OP_ARGSORT,
    GGML_OP_LEAKY_RELU,

    GGML_OP_FLASH_ATTN_EXT,
    GGML_OP_FLASH_ATTN_BACK,
    GGML_OP_SSM_CONV,
    GGML_OP_SSM_SCAN,
    GGML_OP_WIN_PART,
    GGML_OP_WIN_UNPART,
    GGML_OP_GET_REL_POS,
    GGML_OP_ADD_REL_POS,
    GGML_OP_RWKV_WKV6,
    GGML_OP_GATED_LINEAR_ATTN,
    GGML_OP_RWKV_WKV7,

    GGML_OP_UNARY,

    GGML_OP_MAP_CUSTOM1,
    GGML_OP_MAP_CUSTOM2,
    GGML_OP_MAP_CUSTOM3,

    GGML_OP_CUSTOM,

    GGML_OP_CROSS_ENTROPY_LOSS,
    GGML_OP_CROSS_ENTROPY_LOSS_BACK,
    GGML_OP_OPT_STEP_ADAMW,
    GGML_OP_OPT_STEP_SGD,

    GGML_OP_GLU,

    GGML_OP_COUNT,
};

enum ggml_unary_op {
    GGML_UNARY_OP_ABS,
    GGML_UNARY_OP_SGN,
    GGML_UNARY_OP_NEG,
    GGML_UNARY_OP_STEP,
    GGML_UNARY_OP_TANH,
    GGML_UNARY_OP_ELU,
    GGML_UNARY_OP_RELU,
    GGML_UNARY_OP_SIGMOID,
    GGML_UNARY_OP_GELU,
    GGML_UNARY_OP_GELU_QUICK,
    GGML_UNARY_OP_SILU,
    GGML_UNARY_OP_HARDSWISH,
    GGML_UNARY_OP_HARDSIGMOID,
    GGML_UNARY_OP_EXP,
    GGML_UNARY_OP_EXPM1,
    GGML_UNARY_OP_SOFTPLUS,
    GGML_UNARY_OP_GELU_ERF,
    GGML_UNARY_OP_XIELU,
    GGML_UNARY_OP_FLOOR,
    GGML_UNARY_OP_CEIL,
    GGML_UNARY_OP_ROUND,
    GGML_UNARY_OP_TRUNC,

    GGML_UNARY_OP_COUNT,
};

enum ggml_glu_op {
    GGML_GLU_OP_REGLU,
    GGML_GLU_OP_GEGLU,
    GGML_GLU_OP_SWIGLU,
    GGML_GLU_OP_SWIGLU_OAI,
    GGML_GLU_OP_GEGLU_ERF,
    GGML_GLU_OP_GEGLU_QUICK,

    GGML_GLU_OP_COUNT,
};

// Forward declarations
struct ggml_backend_buffer;
struct ggml_tensor;

// Main tensor structure - matches the GGML definition exactly
struct ggml_tensor {
    enum ggml_type type;                          // Data type
    struct ggml_backend_buffer * buffer;          // Memory buffer

    int64_t ne[GGML_MAX_DIMS];                   // Number of elements in each dimension
    size_t  nb[GGML_MAX_DIMS];                   // Stride in bytes for each dimension
                                                  // nb[0] = ggml_type_size(type)
                                                  // nb[1] = nb[0] * (ne[0] / ggml_blck_size(type)) + padding
                                                  // nb[i] = nb[i-1] * ne[i-1]

    // Compute data
    enum ggml_op op;                              // Operation type
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)]; // Operation parameters
    int32_t flags;                                // Tensor flags

    struct ggml_tensor * src[GGML_MAX_SRC];       // Source tensors

    // View data
    struct ggml_tensor * view_src;                // Source tensor for views
    size_t               view_offs;               // Offset for views

    void * data;                                  // Pointer to tensor data

    char name[GGML_MAX_NAME];                     // Tensor name

    void * extra;                                 // Extra data (backend-specific)

    char padding[8];                              // Padding for alignment
};

// Binary operation parameters (for MUL, ADD, etc.)
struct ggml_et_binary_params {
    struct ggml_tensor src0;
    struct ggml_tensor src1;
    struct ggml_tensor dst;
};

// MUL_MAT_ID operation parameters (Mixture of Experts)
struct ggml_et_mul_mat_id_params {
    struct ggml_tensor src0;  // Expert weight matrices [K, M, n_expert]
    struct ggml_tensor src1;  // Activations [K, n_expert_used, batch]
    struct ggml_tensor src2;  // Expert indices [n_expert_used, batch] (I32)
    struct ggml_tensor dst;   // Output [M, n_expert_used, batch, 1]
};

// Check whether a tensor's data is physically contiguous in memory.
// When ne[i] == 1, the stride nb[i] is irrelevant (no second element along
// that axis), so we skip it and only check strides that are actually walked.
static inline int ggml_tensor_is_contiguous(const struct ggml_tensor * t, int type_size) {
    int64_t expected = type_size;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (t->ne[i] > 1 && (int64_t)t->nb[i] != expected) {
            return 0;
        }
        expected *= t->ne[i];
    }
    return 1;
}

#endif // GGML_TENSOR_H