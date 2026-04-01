#ifndef HTP_MSG_H
#define HTP_MSG_H

#include <assert.h>

// ggml-common.h must be included prio to this header

// Mask to enable various stages of the Ops.
// Used for debugging and profiling.
enum htp_op_mask {
    HTP_OPMASK_QUEUE    = (1 << 0),  // Enable Queueing (ie calls into the DSP)
    HTP_OPMASK_QUANTIZE = (1 << 1),  // Enable Quantize
    HTP_OPMASK_COMPUTE  = (1 << 2),  // Enable Compute
};

enum htp_status {
    HTP_STATUS_OK             = 1,
    HTP_STATUS_INTERNAL_ERR   = 2,
    HTP_STATUS_NO_SUPPORT     = 3,
    HTP_STATUS_INVAL_PARAMS   = 4,
    HTP_STATUS_VTCM_TOO_SMALL = 5,
};

// The values must match the ggml_type.
// Duplicated here because we can't include full ggml.h in the htp build.
// We have some static_asserts in the cpp code to ensure things are in sync.
enum htp_data_type {
    HTP_TYPE_F32    = 0,
    HTP_TYPE_F16    = 1,
    HTP_TYPE_Q4_0   = 2,
    HTP_TYPE_Q8_0   = 8,
    HTP_TYPE_IQ4_NL = 20,
    HTP_TYPE_I32    = 26,
    HTP_TYPE_I64    = 27,
    HTP_TYPE_MXFP4  = 39,
    HTP_TYPE_INVALID
};

// Do not reorder first 4 (used as an index)
enum htp_op_code {
    HTP_OP_MUL = 0,
    HTP_OP_ADD = 1,
    HTP_OP_SUB = 2,
    HTP_OP_DIV = 3,
    HTP_OP_MUL_MAT,
    HTP_OP_MUL_MAT_ID,
    HTP_OP_RMS_NORM,
    HTP_OP_UNARY_SILU,
    HTP_OP_UNARY_GELU,
    HTP_OP_UNARY_SIGMOID,
    HTP_OP_UNARY_EXP,
    HTP_OP_UNARY_NEG,
    HTP_OP_UNARY_SOFTPLUS,
    HTP_OP_GLU_SWIGLU,
    HTP_OP_GLU_SWIGLU_OAI,
    HTP_OP_GLU_GEGLU,
    HTP_OP_SOFTMAX,
    HTP_OP_ADD_ID,
    HTP_OP_ROPE,
    HTP_OP_FLASH_ATTN_EXT,
    HTP_OP_SET_ROWS,
    HTP_OP_GET_ROWS,
    HTP_OP_SCALE,
    HTP_OP_CPY,
    HTP_OP_ARGSORT,
    HTP_OP_SQR,
    HTP_OP_SQRT,
    HTP_OP_SUM_ROWS,
    HTP_OP_SSM_CONV,
    HTP_OP_REPEAT,
    HTP_OP_CUMSUM,
    HTP_OP_INVALID
};

// Internal types
#define QK_Q4_0x4x2  256  // 4x Q4_0  blocks packed with next 4x Q4_0 blocks (size in bytes 128)
#define QK_Q8_0x4x2  256  // 4x Q8_0  blocks concat with next 4x Q8_0 blocks
#define QK_MXFP4x4x2 256  // 4x MXFP4 blocks concat with next 4x MXFP4 blocks

#define HTP_OP_MAX_DIMS    4    // aka GGML_MAX_DIMS
#define HTP_OP_MAX_INPUTS  6    // aka GGML_MAX_SRCS
#define HTP_OP_MAX_PARAMS  64   // ala GGML_MAX_OP_PARAMS
#define HTP_OP_MAX_BUFS    8
#define HTP_OP_MAX_REQS    128

struct htp_tensor {
    uint32_t data;                 // Buffer offset in the messages, and data pointer on the NPU
    uint32_t size;                 // Data size in bytes
    uint16_t type;                 // Data type
    uint16_t bi;                   // Buffer index
    uint32_t ne[HTP_OP_MAX_DIMS];  // Number of elements
    uint32_t nb[HTP_OP_MAX_DIMS];  // Stride in bytes (see ggml.h ggml_tensor)
};

enum htp_op_buf_flags {
    HTP_OP_BUF_WEIGHT  = (1U << 0),
    HTP_OP_BUF_COMPUTE = (1U << 1)
};

struct htp_op_buf {
    uint64_t base;     // base address
    uint64_t size;     // total size
    uint32_t flags;    // buffer flags
    uint32_t fd;       // file descriptor
};

enum htp_op_flags {
    HTP_OPFLAGS_SKIP_QUANTIZE = (1U << 0),  // Skip dynamic quantization (reuse quantized tensors)
    HTP_OPFLAGS_SKIP_COMPUTE  = (1U << 1),  // Skip actual computation   (used for profiling)
    HTP_OPFLAGS_EARLY_WAKEUP  = (1U << 2)   // Send early wakeup notification
};

struct htp_op_req {
    struct htp_tensor src[HTP_OP_MAX_INPUTS]; // Input tensors
    struct htp_tensor dst;                    // Output tensor
    uint32_t          opcode; // GGML/HTP Op
    uint32_t          flags;  // OPFLAGS
    int32_t           params[HTP_OP_MAX_PARAMS / sizeof(int32_t)]; // Params for the op, e.g. epsilon of RMS norm
};

struct htp_general_req {
    uint32_t n_bufs;       // Number of buffers
    uint32_t n_ops;        // Number of ops
    // struct htp_op_buf  bufs[0];
    // struct htp_op_req  ops[0];
};

struct htp_general_rsp {
    uint32_t status;       // HTP_STATUS_...
    uint32_t prof_usecs;   // Number of usec per request
    uint32_t prof_cycles;  // Number of cycles per request
    uint32_t prof_pkts;    // Number of instruction packets per request
};

#endif /* HTP_MSG_H */
