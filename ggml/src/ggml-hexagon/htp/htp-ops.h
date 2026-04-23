#ifndef HTP_OPS_H
#define HTP_OPS_H

#include <assert.h>

// ggml-common.h must be included prio to this header

enum htp_status {
    HTP_STATUS_OK             = 1,
    HTP_STATUS_INTERNAL_ERR   = 2,
    HTP_STATUS_NO_SUPPORT     = 3,
    HTP_STATUS_INVAL_PARAMS   = 4,
    HTP_STATUS_VTCM_TOO_SMALL = 5,
};

// First set of values must match the ggml_type.
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

    // types used internally for repack, dyn.quant, etc
    HTP_TYPE_Q4_0x4x2 = 200,
    HTP_TYPE_Q8_0x4x2,
    HTP_TYPE_MXFP4x4x2,

    HTP_TYPE_INVALID
};

// Constats for internal types
#define QK_Q4_0x4x2  256  // 4x Q4_0  blocks packed with next 4x Q4_0 blocks (size in bytes 128)
#define QK_Q8_0x4x2  256  // 4x Q8_0  blocks concat with next 4x Q8_0 blocks
#define QK_MXFP4x4x2 256  // 4x MXFP4 blocks concat with next 4x MXFP4 blocks


// Mask to enable various stages of the Ops.
// Used for debugging and profiling.
enum htp_op_stage {
    HTP_OPSTAGE_QUEUE    = (1 << 0),  // Enable Queueing (ie calls into NPU)
    HTP_OPSTAGE_COMPUTE  = (1 << 1),  // Enable Compute
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
    HTP_OP_FILL,
    HTP_OP_DIAG,

    HTP_OP_INVALID
};

#define HTP_OP_MAX_DIMS    4    // aka GGML_MAX_DIMS
#define HTP_OP_MAX_INPUTS  6    // aka GGML_MAX_SRCS
#define HTP_OP_MAX_PARAMS  16   // aka GGML_MAX_OP_PARAMS

#define HTP_OP_MAX_BUFS    8
#define HTP_OP_MAX_REQS    256
#define HTP_OP_MAX_TENSORS (HTP_OP_MAX_REQS * HTP_OP_MAX_INPUTS + HTP_OP_MAX_REQS)

#if __HVX_ARCH__ < 75
#define HTP_OP_MAX_VMEM    (3167538380u)
#else
#define HTP_OP_MAX_VMEM    (3221225472u)
#endif

#define HTP_MMAP_MAX_VMEM  (2147483648u)

enum htp_tensor_flags {
    HTP_TENSOR_COMPUTE = (1U << 0), // Tensor buffer temporal compute data (not weights)
    HTP_TENSOR_FLUSHED = (1U << 1)  // Tensor buffer has been flushed (set by the NPU)
};

// Tensor descriptor
struct htp_tensor {
    uint32_t data;                 // Buffer offset in the messages, and data pointer on the NPU
    uint32_t size;                 // Data size in bytes
    uint32_t flags;                // Buffer / tensor flags
    uint16_t type;                 // Data type
    uint16_t bi;                   // Buffer index
    uint32_t ne[HTP_OP_MAX_DIMS];  // Number of elements
    uint32_t nb[HTP_OP_MAX_DIMS];  // Stride in bytes (see ggml.h ggml_tensor)
};

// Buffer descriptor
struct htp_buf_desc {
    uint64_t base;     // base address
    uint64_t size;     // total size
    uint32_t flags;    // buffer flags (unused)
    uint32_t fd;       // file descriptor
};

enum htp_op_flags {
    HTP_OPFLAGS_SKIP_COMPUTE  = (1U << 0), // Skip actual computation (used for profiling)
};

// Op descriptor
struct htp_op_desc {
    uint32_t opcode;                    // GGML/HTP Op
    uint32_t flags;                     // Op flags
    int32_t  params[HTP_OP_MAX_PARAMS]; // Params for the op, e.g. epsilon of RMS norm
    uint16_t src[HTP_OP_MAX_INPUTS];    // Input tensors indices
    uint16_t dst;                       // Output tensor index
};

enum htp_profiler_mode {
    HTP_PROF_DISABLED = 0,
    HTP_PROF_BASIC    = 1,
    HTP_PROF_PMU      = 2,
};

#define HTP_PROF_PMU_NCNT 8

// Profile descriptor
struct htp_prof_desc {
    uint32_t opcode;                 // GGML/HTP Op
    uint32_t usecs;                  // Number of usec
    uint32_t cycles;                 // Number of cycles
    uint32_t pad;                    // Unused
    uint32_t pmu[HTP_PROF_PMU_NCNT]; // PMU counters
};

struct htp_opbatch_req {
    uint32_t id;          // Batch id
    uint32_t n_bufs;      // Number of buffers
    uint32_t n_tensors;   // Number of tensors
    uint32_t n_ops;       // Number of ops
    uint32_t flags;       // unused
    uint32_t pad;         // unused
    // struct htp_buf_desc  bufs[];    -- dspqueue buf 0
    // struct htp_tensor    tensors[]; -- dspqueue buf 0
    // struct htp_op_desc   ops[];     -- dspqueue buf 0
};

struct htp_opbatch_rsp {
    uint32_t id;         // Batch id
    uint32_t status;     // HTP_STATUS_...
    uint32_t n_bufs;     // Number of buffers
    uint32_t n_tensors;  // Number of tensors
    uint32_t n_ops;      // Number of op profile descriptors
    uint32_t pad;        // unused
    // struct htp_prof_desc profs[];  -- dspqueue buf 0
};

#endif /* HTP_OPS_H */
