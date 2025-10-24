#ifndef HTP_MSG_H
#define HTP_MSG_H

#include <assert.h>

// ggml-common.h must be included prio to this header

// Mask to enable various stages of the Ops.
// Used for debugging and profiling.
enum {
    HTP_OPMASK_QUEUE    = (1 << 0),  // Enable Queueing (ie calls into the DSP)
    HTP_OPMASK_QUANTIZE = (1 << 1),  // Enable Quantize
    HTP_OPMASK_COMPUTE  = (1 << 2),  // Enable Compute
};

// Op flags
enum {
    HTP_OPFLAGS_SKIP_QUANTIZE = (1 << 0),  // Skip dynamic quantization (reuse quantized tensors)
    HTP_OPFLAGS_SKIP_COMPUTE  = (1 << 1),  // Skip actual computation (used for profiling)
    HTP_OPFLAGS_EARLY_WAKEUP  = (1 << 2)   // Send early wakeup notification
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
    HTP_TYPE_F32   = 0,
    HTP_TYPE_F16   = 1,
    HTP_TYPE_Q4_0  = 2,
    HTP_TYPE_Q8_0  = 8,
    HTP_TYPE_MXFP4 = 39,
    HTP_TYPE_COUNT
};

// These values are manually translated over to HTP
// !!!! DO NOT ALTER THE ORDER OF THE FIRST FOUR ENUMS !!!!
enum htp_op {
    HTP_OP_MUL            = 0,
    HTP_OP_ADD            = 1,
    HTP_OP_SUB            = 2,
    HTP_OP_DIV            = 3,
    HTP_OP_MUL_MAT        = 4,
    HTP_OP_MUL_MAT_ID     = 5,
    HTP_OP_RMS_NORM       = 6,
    HTP_OP_UNARY_SILU     = 7,
    HTP_OP_GLU_SWIGLU     = 8,
    HTP_OP_GLU_SWIGLU_OAI = 9,
    HTP_OP_SOFTMAX        = 10,
    HTP_OP_ADD_ID         = 11,
    HTP_OP_ROPE           = 12,
    INVALID
};

static inline size_t htp_type_block_size(uint32_t t) {
    switch (t) {
        case HTP_TYPE_F32:
            return 1;
        case HTP_TYPE_F16:
            return 1;
        case HTP_TYPE_Q4_0:
            return QK4_0;
        case HTP_TYPE_Q8_0:
            return QK8_0;
        case HTP_TYPE_MXFP4:
            return QK_MXFP4;
        default:
            assert(0 && "unsupported HTP data type");
    }
    return 0;
}

static inline size_t htp_type_nbytes(uint32_t t) {
    switch (t) {
        case HTP_TYPE_F32:
            return 4;
        case HTP_TYPE_F16:
            return 2;
        case HTP_TYPE_Q4_0:
            return sizeof(block_q4_0);
        case HTP_TYPE_Q8_0:
            return sizeof(block_q8_0);
        case HTP_TYPE_MXFP4:
            return sizeof(block_mxfp4);
        default:
            assert(0 && "unsupported HTP data type");
    }
    return 0;
}

static const char * htp_type_name(uint32_t t) {
    switch (t) {
        case HTP_TYPE_F32:
            return "fp32";
        case HTP_TYPE_F16:
            return "fp16";
        case HTP_TYPE_Q4_0:
            return "q4_0";
        case HTP_TYPE_Q8_0:
            return "q8_0";
        case HTP_TYPE_MXFP4:
            return "mxfp4";
    }
    return 0;
}

// Internal types
#define QK_Q4_0x4x2  256  // 4x Q4_0 blocks packed with next 4x Q4_0 blocks (size in bytes 128)
#define QK_Q8_0x4x2  256  // 4x Q8_0 blocks concat with next 4x Q8_0 blocks
#define QK_MXFP4x4x2 256  // 4x MXFP4 blocks concat with next 4x MXFP4 blocks

#define HTP_MAX_DIMS 4

struct htp_tensor {
    uint32_t data;              // Buffer offset in the messages, and data pointer on the NSP
    uint32_t type;              // Data type
    uint32_t ne[HTP_MAX_DIMS];  // Number of elements
    uint32_t nb[HTP_MAX_DIMS];  // Stride in bytes (see ggml.h ggml_tensor)
};

#define HTP_MAX_OP_PARAMS 64

struct htp_general_req {
    uint32_t op;  // GGML/HTP Op
    int32_t  op_params[HTP_MAX_OP_PARAMS / sizeof(int32_t)];
    // Params for the op, e.g. epsilon of RMS norm
    uint32_t flags;          // Request flags

    struct htp_tensor src0;  // Input0 tensor
    struct htp_tensor src1;  // Input1 tensor
    struct htp_tensor src2;  // Input2 tensor
    struct htp_tensor dst;   // Output tensor

    // should be multiple of 64 bytes (cacheline)
};

struct htp_general_rsp {
    uint32_t op;           // GGML/HTP Op
    uint32_t status;       // HTP_STATUS_...
    uint32_t prof_usecs;   // Number of usec per request
    uint32_t prof_cycles;  // Number of cycles per request
    uint32_t prof_pkts;    // Number of instruction packets per request
    uint8_t  unused[44];   // Pad to 64 bytes
};

#define HTP_MAX_MESSAGE_SIZE   sizeof(struct htp_general_req)
#define HTP_MAX_PACKET_BUFFERS 4

#endif /* HTP_MSG_H */
