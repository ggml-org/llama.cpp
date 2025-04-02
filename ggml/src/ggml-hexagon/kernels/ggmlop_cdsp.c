/*
* Copyright (c) 2023-2025 The ggml authors
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to
* deal in the Software without restriction, including without limitation the
* rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
* sell copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
* IN THE SOFTWARE.
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>

#include "HAP_perf.h"
#include "HAP_farf.h"
#include "HAP_power.h"
#include "HAP_vtcm_mgr.h"
#include "HAP_compute_res.h"

#include "AEEStdErr.h"
#include "hexagon_types.h"
#include "hexagon_protos.h"

#include "ggmlop_ap_skel.h"

// =================================================================================================
//  section-1: forward/prototype declaration,global vars,macros,data structures
// =================================================================================================
#define ggml_tensor         dsptensor

#define GGML_MAX_DIMS       4

#define ALIGN_128_BYTE      128

#define GGML_UNUSED(x)      (void)(x)

#define UNUSED              GGML_UNUSED

#define GGML_PAD(x, n)      (((x) + (n) - 1) & ~((n) - 1))

#define GGML_ABORT(...)     ggml_abort(__FILE__, __LINE__, __VA_ARGS__)

#define GGML_ASSERT(x)      if (!(x)) GGML_ABORT("GGML_ASSERT(%s) failed", #x)

#define MIN(a, b)           ((a) < (b) ? (a) : (b))
#define MAX(a, b)           ((a) > (b) ? (a) : (b))

#if UINTPTR_MAX == 0xFFFFFFFF
#define GGML_MEM_ALIGN      4
#else
#define GGML_MEM_ALIGN      16
#endif

#define GGML_RESTRICT

#define static_assert(a, b) do { } while (0)

#define GROUP_MAX_EPS 1e-15f

// QK = number of values after dequantization
// QK_K = super-block size
#define QK_K 256
#define K_SCALE_SIZE 12

#define GGML_COMPUTE_FP16_TO_FP32(x)    ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x)    ggml_compute_fp32_to_fp16(x)
#define GGML_FP32_TO_FP16(x)            GGML_COMPUTE_FP32_TO_FP16(x)
#define GGML_FP16_TO_FP32(x)            ggml_lookup_fp16_to_fp32(x)

#if 0//def NDEBUG
#define GGMLQNN_DEBUG                                       0
#else
#define GGMLQNN_DEBUG                                       1
#endif

#define GGMLHEXAGON_LOGBUF_LEN                              4096
#define GGML_QNN_TMPBUF_LEN                                 256
#if GGMLQNN_DEBUG
#define GGMLHEXAGON_LOG_DEBUG(...)                          ggmlhexagon_log_internal(GGMLHEXAGON_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#else
#define GGMLHEXAGON_LOG_DEBUG(...)
#endif
#define GGMLQNN_DUMP_TENSOR(tensor)                         ggmlhexagon_dump_tensor(tensor, #tensor)

#define GGML_TENSOR_LOCALS_1(type, prefix, pointer, array) \
    const type prefix##0 = (pointer)->array[0]; \
    GGML_UNUSED(prefix##0);
#define GGML_TENSOR_LOCALS_2(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_1    (type, prefix, pointer, array) \
    const type prefix##1 = (pointer)->array[1]; \
    GGML_UNUSED(prefix##1);
#define GGML_TENSOR_LOCALS_3(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_2    (type, prefix, pointer, array) \
    const type prefix##2 = (pointer)->array[2]; \
    GGML_UNUSED(prefix##2);
#define GGML_TENSOR_LOCALS(type, prefix, pointer, array) \
    GGML_TENSOR_LOCALS_3  (type, prefix, pointer, array) \
    const type prefix##3 = (pointer)->array[3]; \
    GGML_UNUSED(prefix##3);

#define GGML_TENSOR_UNARY_OP_LOCALS \
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

#define GGML_TENSOR_BINARY_OP_LOCALS \
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne,  dst,  ne) \
    GGML_TENSOR_LOCALS(size_t,  nb,  dst,  nb)

#define GGML_TENSOR_BINARY_OP_LOCALS01 \
    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb0, src0, nb) \
    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne) \
    GGML_TENSOR_LOCALS(size_t,  nb1, src1, nb)

enum ggmlhexagon_log_level {
    GGMLHEXAGON_LOG_LEVEL_NONE  = 0,
    GGMLHEXAGON_LOG_LEVEL_DEBUG = 1,
    GGMLHEXAGON_LOG_LEVEL_INFO  = 2,
    GGMLHEXAGON_LOG_LEVEL_WARN  = 3,
    GGMLHEXAGON_LOG_LEVEL_ERROR = 4,
    GGMLHEXAGON_LOG_LEVEL_CONT  = 5,
};

enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
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
    GGML_TYPE_COUNT   = 39,
};

typedef double      ggml_float;
typedef uint16_t    ggml_fp16_t;
typedef uint16_t    ggml_half;
typedef uint32_t    ggml_half2;
typedef void        (*ggml_vec_dot_t)  (int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT x, size_t bx,
                                        const void * GGML_RESTRICT y, size_t by, int nrc);
typedef void        (*ggml_from_float_t)(const float * GGML_RESTRICT x, void  * GGML_RESTRICT y, int64_t k);

typedef void        (*ggml_to_float_t)  (const void  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
typedef void        (*ggml_from_float_t)(const float * GGML_RESTRICT x, void  * GGML_RESTRICT y, int64_t k);

struct ggml_compute_params {
    // ith = thread index, nth = number of threads
    int ith, nth;

    // work buffer for all threads
    size_t wsize;
    void * wdata;
};

#define QK4_0 32
typedef struct {
    ggml_half d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

#define QK4_1 32
typedef struct {
    union {
        struct {
            ggml_half d; // delta
            ggml_half m; // min
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t qs[QK4_1 / 2]; // nibbles / quants
} block_q4_1;

#define QK5_0 32
typedef struct {
    ggml_half d;           // delta
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_0 / 2]; // nibbles / quants
} block_q5_0;

#define QK5_1 32
typedef struct {
    union {
        struct {
            ggml_half d; // delta
            ggml_half m; // min
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t qh[4];         // 5-th bit of quants
    uint8_t qs[QK5_1 / 2]; // nibbles / quants
} block_q5_1;

#define QK8_0 32
typedef struct {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;

#define QK8_1 32
typedef struct {
    union {
        struct {
            ggml_half d; // delta
            ggml_half s; // d * sum(qs[i])
        } GGML_COMMON_AGGR_S;
        ggml_half2 ds;
    } GGML_COMMON_AGGR_U;
    int8_t qs[QK8_1]; // quants
} block_q8_1;

// 2-bit quantization
// weight is represented as x = a * q + b
// 16 blocks of 16 elements each
// Effectively 2.625 bits per weight
typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
} block_q2_K;

// 3-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 3.4375 bits per weight
typedef struct {
    uint8_t hmask[QK_K/8]; // quants - high bit
    uint8_t qs[QK_K/4];    // quants - low 2 bits
    uint8_t scales[12];    // scales, quantized with 6 bits
    ggml_half d;           // super-block scale
} block_q3_K;

// 4-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
typedef struct {
    union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;

// 5-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 5.5 bits per weight
typedef struct {
    union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR_S;
        ggml_half2 dm;
    } GGML_COMMON_AGGR_U;
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit
    uint8_t qs[QK_K/2];           // quants, low 4 bits
} block_q5_K;

// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    ggml_half d;             // super-block scale
} block_q6_K;

typedef struct {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
} block_q8_K;

struct ggml_type_traits {
    const char             * type_name;
    int64_t                  blck_size;
    int64_t                  blck_size_interleave; // interleave elements in blocks
    size_t                   type_size;
    bool                     is_quantized;
    ggml_to_float_t          to_float;
    ggml_from_float_t        from_float_ref;
};

struct ggml_type_traits_cpu {
    ggml_from_float_t        from_float;
    ggml_vec_dot_t           vec_dot;
    enum ggml_type           vec_dot_type;
    int64_t                  nrows; // number of rows to process simultaneously
};

static size_t ggml_nbytes(const struct ggml_tensor * tensor);
static void   ggmlhexagon_log_internal(int level, const char * file, const char * func, int line, const char * format, ...);
static void   ggml_vec_dot_f32(int n, float * GGML_RESTRICT s, size_t bs, const float * GGML_RESTRICT x, size_t bx, const float * GGML_RESTRICT y, size_t by, int nrc);

static void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
static void quantize_row_q6_K_ref(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
static void quantize_row_q6_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k);
static void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

static float ggml_table_f32_f16[1 << 16];

static const struct ggml_type_traits_cpu type_traits_cpu[GGML_TYPE_COUNT] = {
        [GGML_TYPE_F32] = {
                .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f32,
                .vec_dot_type             = GGML_TYPE_F32,
                .nrows                    = 1,
        },
        [GGML_TYPE_F16] = {
                .from_float               = NULL,
                .vec_dot                  = NULL,
                .vec_dot_type             = GGML_TYPE_F16,
                .nrows                    = 1,
        },
        [GGML_TYPE_Q4_0] = {
                .from_float               = NULL,
                .vec_dot                  = NULL,
                .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
                .nrows                    = 2,
#else
                .nrows                    = 1,
#endif
        },
        [GGML_TYPE_Q4_1] = {
                .from_float               = NULL,
                .vec_dot                  = NULL,
                .vec_dot_type             = GGML_TYPE_Q8_1,
#if defined (__ARM_FEATURE_MATMUL_INT8)
                .nrows                    = 2,
#else
                .nrows                    = 1,
#endif
        },
        [GGML_TYPE_Q5_0] = {
                .from_float               = NULL,
                .vec_dot                  = NULL,
                .vec_dot_type             = GGML_TYPE_Q8_0,
                .nrows                    = 1,
        },
        [GGML_TYPE_Q5_1] = {
                .from_float               = NULL,
                .vec_dot                  = NULL,
                .vec_dot_type             = GGML_TYPE_Q8_1,
                .nrows                    = 1,
        },
        [GGML_TYPE_Q8_0] = {
                .from_float               = NULL,
                .vec_dot                  = NULL,
                .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
                .nrows                    = 2,
#else
                .nrows                    = 1,
#endif
        },
        [GGML_TYPE_Q8_1] = {
                .from_float               = NULL,
                .vec_dot_type             = GGML_TYPE_Q8_1,
                .nrows                    = 1,
        },
        [GGML_TYPE_Q2_K] = {
                .from_float               = NULL,
                .vec_dot                  = NULL,
                .vec_dot_type             = GGML_TYPE_Q8_K,
                .nrows                    = 1,
        },
        [GGML_TYPE_Q3_K] = {
                .from_float               = NULL,
                .vec_dot                  = NULL,
                .vec_dot_type             = GGML_TYPE_Q8_K,
                .nrows                    = 1,
        },
        [GGML_TYPE_Q4_K] = {
                .from_float               = NULL,
                .vec_dot                  = NULL,
                .vec_dot_type             = GGML_TYPE_Q8_K,
                .nrows                    = 1,
        },
        [GGML_TYPE_Q5_K] = {
                .from_float               = NULL,
                .vec_dot                  = NULL,
                .vec_dot_type             = GGML_TYPE_Q8_K,
                .nrows                    = 1,
        },
        [GGML_TYPE_Q6_K] = {
                .from_float               = quantize_row_q6_K,
                .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
                .vec_dot_type             = GGML_TYPE_Q8_K,
                .nrows                    = 1,
        },
};

static const struct ggml_type_traits type_traits[GGML_TYPE_COUNT] = {
        [GGML_TYPE_I8] = {
                .type_name                = "i8",
                .blck_size                = 1,
                .type_size                = sizeof(int8_t),
                .is_quantized             = false,
        },
        [GGML_TYPE_I16] = {
                .type_name                = "i16",
                .blck_size                = 1,
                .type_size                = sizeof(int16_t),
                .is_quantized             = false,
        },
        [GGML_TYPE_I32] = {
                .type_name                = "i32",
                .blck_size                = 1,
                .type_size                = sizeof(int32_t),
                .is_quantized             = false,
        },
        [GGML_TYPE_I64] = {
                .type_name                = "i64",
                .blck_size                = 1,
                .type_size                = sizeof(int64_t),
                .is_quantized             = false,
        },
        [GGML_TYPE_F64] = {
                .type_name                = "f64",
                .blck_size                = 1,
                .type_size                = sizeof(double),
                .is_quantized             = false,
        },
        [GGML_TYPE_F32] = {
                .type_name                = "f32",
                .blck_size                = 1,
                .type_size                = sizeof(float),
                .is_quantized             = false,
        },
        [GGML_TYPE_F16] = {
                .type_name                = "f16",
                .blck_size                = 1,
                .type_size                = sizeof(ggml_fp16_t),
                .is_quantized             = false,
                .to_float                 = NULL,
                .from_float_ref           = NULL,
        },
        [GGML_TYPE_Q4_0] = {
                .type_name                = "q4_0",
                .blck_size                = QK4_0,
                .type_size                = sizeof(block_q4_0),
                .is_quantized             = true,
                .to_float                 = NULL,
                .from_float_ref           = NULL,
        },
        [GGML_TYPE_Q4_1] = {
                .type_name                = "q4_1",
                .blck_size                = QK4_1,
                .type_size                = sizeof(block_q4_1),
                .is_quantized             = true,
                .to_float                 = NULL,
                .from_float_ref           = NULL,
        },
        [4] = { // GGML_TYPE_Q4_2
                .type_name                = "DEPRECATED",
                .blck_size                = 0,
                .type_size                = 0,
                .is_quantized             = false,
        },
        [5] = { // GGML_TYPE_Q4_3
                .type_name                = "DEPRECATED",
                .blck_size                = 0,
                .type_size                = 0,
                .is_quantized             = false,
        },
        [GGML_TYPE_Q5_0] = {
                .type_name                = "q5_0",
                .blck_size                = QK5_0,
                .type_size                = sizeof(block_q5_0),
                .is_quantized             = true,
                .to_float                 = NULL,
                .from_float_ref           = NULL,
        },
        [GGML_TYPE_Q5_1] = {
                .type_name                = "q5_1",
                .blck_size                = QK5_1,
                .type_size                = sizeof(block_q5_1),
                .is_quantized             = true,
                .to_float                 = NULL,
                .from_float_ref           = NULL,
        },
        [GGML_TYPE_Q8_0] = {
                .type_name                = "q8_0",
                .blck_size                = QK8_0,
                .type_size                = sizeof(block_q8_0),
                .is_quantized             = true,
                .to_float                 = NULL,
                .from_float_ref           = NULL,
        },
        [GGML_TYPE_Q8_1] = {
                .type_name                = "q8_1",
                .blck_size                = QK8_1,
                .type_size                = sizeof(block_q8_1),
                .is_quantized             = true,
                .from_float_ref           = NULL,
        },
        [GGML_TYPE_Q2_K] = {
                .type_name                = "q2_K",
                .blck_size                = QK_K,
                .type_size                = sizeof(block_q2_K),
                .is_quantized             = true,
                .to_float                 = NULL,
                .from_float_ref           = NULL,
        },
        [GGML_TYPE_Q3_K] = {
                .type_name                = "q3_K",
                .blck_size                = QK_K,
                .type_size                = sizeof(block_q3_K),
                .is_quantized             = true,
                .to_float                 = NULL,
                .from_float_ref           = NULL,
        },
        [GGML_TYPE_Q4_K] = {
                .type_name                = "q4_K",
                .blck_size                = QK_K,
                .type_size                = sizeof(block_q4_K),
                .is_quantized             = true,
                .to_float                 = NULL,
                .from_float_ref           = NULL,
        },
        [GGML_TYPE_Q5_K] = {
                .type_name                = "q5_K",
                .blck_size                = QK_K,
                .type_size                = sizeof(block_q5_K),
                .is_quantized             = true,
                .to_float                 = NULL,
                .from_float_ref           = NULL,
        },
        [GGML_TYPE_Q6_K] = {
                .type_name                = "q6_K",
                .blck_size                = QK_K,
                .type_size                = sizeof(block_q6_K),
                .is_quantized             = true,
                .to_float                 = (ggml_to_float_t) dequantize_row_q6_K,
                .from_float_ref           = (ggml_from_float_t) quantize_row_q6_K_ref,
        },

};

// =================================================================================================
//  section-2: ggml-hexagon kernel's internal troubleshooting function
// =================================================================================================
static void ggmlhexagon_log_internal(int level, const char *file, const char *func, int line, const char *format, ...) {
    return;
    static char s_ggmlhexagon_log_internal_buf[GGMLHEXAGON_LOGBUF_LEN];
    va_list args;
    va_start(args, format);
    int len_prefix = snprintf(s_ggmlhexagon_log_internal_buf, GGMLHEXAGON_LOGBUF_LEN, "[%s, %d]: ",
                              func, line);
    int len = vsnprintf(s_ggmlhexagon_log_internal_buf + len_prefix,
                        GGMLHEXAGON_LOGBUF_LEN - len_prefix, format, args);
    if (len < (GGMLHEXAGON_LOGBUF_LEN - len_prefix)) {
        FARF(ALWAYS, "%s\n", s_ggmlhexagon_log_internal_buf);
    }
    va_end(args);
}

static void ggmlhexagon_dump_tensor_elements(const ggml_tensor * tensor) {
    //return;
    float value = 0;
    char tmpbuf[GGMLHEXAGON_LOGBUF_LEN];
    size_t buflen = 0;
    if (tensor->type == GGML_TYPE_F32) {
        memset(tmpbuf, 0, GGMLHEXAGON_LOG_LEVEL_DEBUG);
        for (int h = 0; h < tensor->ne[3]; h++) {
            for (int i = 0; i < tensor->ne[2]; i++) {
                for (int j = 0; j < tensor->ne[1]; j++) {
                    for (int k = 0; k < tensor->ne[0]; k++) {
                        value = ((float *) tensor->data)[h * tensor->ne[2] + i * tensor->ne[1] +
                                                         j * tensor->ne[0] + k];
                        buflen += snprintf(tmpbuf + buflen, GGMLHEXAGON_LOGBUF_LEN - buflen, "%-4.2f\t", value);
                    }
                    buflen += snprintf(tmpbuf + buflen, GGMLHEXAGON_LOGBUF_LEN - buflen, "\n");
                }
            }
        }
        GGMLHEXAGON_LOG_DEBUG("\n%s\n", tmpbuf);
    }

    GGMLHEXAGON_LOG_DEBUG("\n");
}

static void ggmlhexagon_dump_tensor(const ggml_tensor * tensor, int dump_tensor_data) {
    GGMLHEXAGON_LOG_DEBUG("ne = %5d x %5d x %5d x %5d , nb = (%5zi, %5zi, %5zi, %5zi)\n",
         tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
         tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);

    if ((1 == dump_tensor_data) && (ggml_nbytes(tensor) < 320)) {
        ggmlhexagon_dump_tensor_elements(tensor);
    }
}

// =================================================================================================
//  section-3: tiny ggml-dsp(ggml on Hexagon cDSP, ported from original ggml)
// =================================================================================================
static const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type) {
    return &type_traits_cpu[type];
}

static void ggml_vec_dot_f32(int n, float * GGML_RESTRICT s, size_t bs, const float * GGML_RESTRICT x,
                             size_t bx, const float *GGML_RESTRICT y, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float) (x[i] * y[i]);
    }
    *s = sumf;
}

inline static void ggml_vec_mul_f32 (const int n, float * z, const float * x, const float * y) {
    for (int i = 0; i < n; ++i) z[i]  = x[i]*y[i];
}

inline static void ggml_vec_div_f32 (const int n, float * z, const float * x, const float * y) {
    for (int i = 0; i < n; ++i) z[i]  = x[i]/y[i];
}

inline static void ggml_vec_sub_f32 (const int n, float * z, const float * x, const float * y) {
    for (int i = 0; i < n; ++i) z[i]  = x[i] - y[i];
}

static const struct ggml_type_traits * ggml_get_type_traits(enum ggml_type type) {
    return &type_traits[type];
}

static int64_t ggml_blck_size(enum ggml_type type) {
    return type_traits[type].blck_size;
}

static size_t ggml_type_size(enum ggml_type type) {
    return type_traits[type].type_size;
}

static size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    assert(ne % ggml_blck_size(type) == 0);
    return ggml_type_size(type)*ne/ggml_blck_size(type);
}

static size_t ggml_nbytes(const struct ggml_tensor * tensor) {
    size_t nbytes;
    const size_t blck_size = ggml_blck_size(tensor->type);
    if (blck_size == 1) {
        nbytes = ggml_type_size(tensor->type);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    else {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_size;
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    return nbytes;
}

static size_t ggml_nbytes_pad(const struct ggml_tensor * tensor) {
    return GGML_PAD(ggml_nbytes(tensor), GGML_MEM_ALIGN);
}

static double ggml_type_sizef(enum ggml_type type) {
    return ((double)(type_traits[type].type_size))/type_traits[type].blck_size;
}

static const char * ggml_type_name(enum ggml_type type) {
    return type < GGML_TYPE_COUNT ? type_traits[type].type_name : "NONE";
}

static bool ggml_is_quantized(enum ggml_type type) {
    return type_traits[type].is_quantized;
}

static bool ggml_is_empty(const struct ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] == 0) {
            return true;
        }
    }
    return false;
}

static bool ggml_can_repeat(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return ggml_is_empty(t0) ? ggml_is_empty(t1) :
           (t1->ne[0]%t0->ne[0] == 0) &&
           (t1->ne[1]%t0->ne[1] == 0) &&
           (t1->ne[2]%t0->ne[2] == 0) &&
           (t1->ne[3]%t0->ne[3] == 0);
}

static bool ggml_are_same_shape(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");
    return
            (t0->ne[0] == t1->ne[0]) &&
            (t0->ne[1] == t1->ne[1]) &&
            (t0->ne[2] == t1->ne[2]) &&
            (t0->ne[3] == t1->ne[3]);
}

static int64_t ggml_nrows(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

static bool ggml_is_transposed(const struct ggml_tensor * tensor) {
    return tensor->nb[0] > tensor->nb[1];
}

static bool ggml_is_contiguous_n(const struct ggml_tensor * tensor, int n) {
    size_t next_nb = ggml_type_size(tensor->type);
    if (tensor->ne[0] != ggml_blck_size(tensor->type) && tensor->nb[0] != next_nb) {
        return false;
    }
    next_nb *= tensor->ne[0]/ggml_blck_size(tensor->type);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        if (tensor->ne[i] != 1) {
            if (i > n) {
                if (tensor->nb[i] != next_nb) {
                    return false;
                }
                next_nb *= tensor->ne[i];
            } else {
                // this dimension does not need to be contiguous
                next_nb = tensor->ne[i]*tensor->nb[i];
            }
        }
    }
    return true;
}

static int64_t ggml_nelements(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

static bool ggml_is_contiguous_0(const struct ggml_tensor * tensor) {
    return ggml_is_contiguous_n(tensor, 0);
}

static bool ggml_is_contiguous(const struct ggml_tensor * tensor) {
    return ggml_is_contiguous_0(tensor);
}

inline static void ggml_vec_add_f32 (const int n, float * z, const float * x, const float * y) {
    for (int i = 0; i < n; ++i) z[i]  = x[i] + y[i];
}

static void ggml_abort(const char * file, int line, const char * fmt, ...) {
    GGMLHEXAGON_LOG_DEBUG("enter ggml_abort");
    abort();
    return;
}

// FP16 <-> FP32
static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;
}

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
                            (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
        const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

static inline float ggml_lookup_fp16_to_fp32(ggml_fp16_t f) {
    uint16_t s;
    memcpy(&s, &f, sizeof(uint16_t));
    return ggml_table_f32_f16[s];
}

static inline void ggml_init(void) {
    for (int i = 0; i < (1 << 16); ++i) {
        union {
            uint16_t u16;
            ggml_fp16_t fp16;
        } u = {i};
        ggml_table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(u.fp16);
    }
}

static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static float make_qx_quants(int n, int nmax, const float * GGML_RESTRICT x, int8_t * GGML_RESTRICT L, int rmse_type,
                            const float * GGML_RESTRICT qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < GROUP_MAX_EPS) { // all zero
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
        }
        return 1/iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    float sumlx = 0;
    float suml2 = 0;
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 0; i < n; ++i) {
#else
    for (int i = 0; i < n; ++i) {
#endif
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
        float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    if (return_early) return suml2 > 0 ? 0.5f*(scale + 1/iscale) : 1/iscale;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) {
            continue;
        }
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}

static void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * GGML_RESTRICT ql = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const int8_t  * GGML_RESTRICT sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

static void quantize_row_q6_K_ref(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    int8_t L[QK_K];
    float   scales[QK_K/16];

    for (int i = 0; i < nb; i++) {

        float max_scale = 0;
        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K/16; ++ib) {

            const float scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, NULL);
            scales[ib] = scale;

            const float abs_scale = fabsf(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }

        }

        if (max_abs_scale < GROUP_MAX_EPS) {
            memset(&y[i], 0, sizeof(block_q6_K));
            y[i].d = GGML_FP32_TO_FP16(0.f);
            x += QK_K;
            continue;
        }

        float iscale = -128.f/max_scale;
        y[i].d = GGML_FP32_TO_FP16(1/iscale);
        for (int ib = 0; ib < QK_K/16; ++ib) {
            y[i].scales[ib] = MIN(127, nearest_int(iscale*scales[ib]));
        }

        for (int j = 0; j < QK_K/16; ++j) {
            float d = GGML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = MAX(-32, MIN(31, l));
                L[16*j + ii] = l + 32;
            }
        }

        uint8_t * GGML_RESTRICT ql = y[i].ql;
        uint8_t * GGML_RESTRICT qh = y[i].qh;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = L[j + l +  0] & 0xF;
                const uint8_t q2 = L[j + l + 32] & 0xF;
                const uint8_t q3 = L[j + l + 64] & 0xF;
                const uint8_t q4 = L[j + l + 96] & 0xF;
                ql[l+ 0] = q1 | (q3 << 4);
                ql[l+32] = q2 | (q4 << 4);
                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
            }
            ql += 64;
            qh += 32;
        }

        x += QK_K;
    }
}

static void quantize_row_q6_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    block_q6_K * GGML_RESTRICT y = vy;
    quantize_row_q6_K_ref(x, y, k);
}

static void ggml_vec_dot_q6_K_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
            UNUSED(nrc);
            UNUSED(bx);
            UNUSED(by);
            UNUSED(bs);

    const block_q6_K * GGML_RESTRICT x = vx;
    const block_q8_K * GGML_RESTRICT y = vy;

    const int nb = n / QK_K;

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * GGML_RESTRICT q4 = x[i].ql;
        const uint8_t * GGML_RESTRICT qh = x[i].qh;
        const  int8_t * GGML_RESTRICT q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * GGML_RESTRICT a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a  += 128;
            q4 += 64;
            qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;

}

static inline uint64 hexagon_perf_get_time_us(void)
{
    unsigned long long count;
    asm volatile (" %0 = c31:30 " : "=r"(count));
    return (uint64)(count) * 10ull / 192ull;
}

static void ggml_time_init(void) {

}

static int64_t ggml_time_ms(void) {
    return hexagon_perf_get_time_us() * 1000;
}

int64_t ggml_time_us(void) {
    return hexagon_perf_get_time_us();
}

// =================================================================================================
//  section-4: ggml-hexagon kernel helper function
// =================================================================================================
int ggmlop_dsp_open(const char*uri, remote_handle64* handle) {
    void *tptr = NULL;
    FARF(HIGH, "uri %s", uri);
    tptr = (void *)malloc(1);
    *handle = (remote_handle64)tptr;
    assert(*handle);

    ggml_init();

    return 0;
}

int ggmlop_dsp_close(remote_handle64 handle) {
    if (handle)
        free((void*)handle);
    return 0;
}

AEEResult ggmlop_dsp_setclocks(remote_handle64 handle, int32 power_level, int32 latency, int32 dcvs_enabled) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    HAP_power_request_t request;
    memset(&request, 0, sizeof(HAP_power_request_t));
    request.type = HAP_power_set_apptype;
    request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;

    void * ggmop_ctx = (void*)(handle);
    int retval = HAP_power_set(ggmop_ctx, &request);
    if (retval)  {
        GGMLHEXAGON_LOG_DEBUG("failed first power vote");
        return AEE_EFAILED;
    }

    //configure clocks & DCVS mode
    memset(&request, 0, sizeof(HAP_power_request_t));
    request.type = HAP_power_set_DCVS_v2;
    request.dcvs_v2.dcvs_enable = TRUE;
    request.dcvs_v2.dcvs_params.target_corner = (HAP_dcvs_voltage_corner_t)power_level;
    if (dcvs_enabled) {
        request.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_DISABLE;
        request.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_DISABLE;
    } else {
        request.dcvs_v2.dcvs_params.min_corner = request.dcvs_v2.dcvs_params.target_corner;
        request.dcvs_v2.dcvs_params.max_corner = request.dcvs_v2.dcvs_params.target_corner;
    }
    request.dcvs_v2.dcvs_option     = HAP_DCVS_V2_PERFORMANCE_MODE;
    request.dcvs_v2.set_dcvs_params = TRUE;
    request.dcvs_v2.set_latency     = TRUE;
    request.dcvs_v2.latency         = latency;
    retval = HAP_power_set(ggmop_ctx, &request);
    if (retval) {
        GGMLHEXAGON_LOG_DEBUG("failed to vote for performance mode");
        return AEE_EFAILED;
    }

    memset(&request, 0, sizeof(HAP_power_request_t));
    request.type = HAP_power_set_HVX;
    request.hvx.power_up = TRUE;
    retval = HAP_power_set(ggmop_ctx, &request);
    if (retval) {
        GGMLHEXAGON_LOG_DEBUG("failed to vote for HVX power");
        return AEE_EFAILED;
    }
    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return AEE_SUCCESS;
}

// =================================================================================================
//  section-5: ggml-hexagon kernel function: offload ggmlop to cDSP through Hexagon C API and SIMD instructions
// =================================================================================================
inline static void ggmlhexagon_dsp_add_f32 (const int n, float * z, const float * x, const float * y) {
    HVX_Vector * va;
    HVX_Vector * vb;
    HVX_Vector * vc;
    HVX_Vector qf32;
    const int FLOATS_PER_VECTOR = 128 / sizeof(float);
    const int block  = n / FLOATS_PER_VECTOR;
    const int left   = n % FLOATS_PER_VECTOR;
    const int blocks = block * FLOATS_PER_VECTOR;

    if (0 == block) {
        for (size_t i = 0; i < n; ++i)
            z[i] = x[i] + y[i];

        return;
    }

    if ((((uintptr_t)z | (uintptr_t)x | (uintptr_t)y) % ALIGN_128_BYTE) != 0) {
        GGMLHEXAGON_LOG_DEBUG("memaddress mismatch alignment 128 bytes z:%p x:%p y:%p", z, x, y);
        for (size_t i = 0; i < n; ++i)
            z[i] = x[i] + y[i];

        return;
    }

    va = (HVX_Vector *)x;
    vb = (HVX_Vector *)y;
    vc = (HVX_Vector *)z;
    for (size_t i = 0; i < block; ++i) {
        qf32 = Q6_Vqf32_vadd_VsfVsf(*va++, *vb++);
        *vc = Q6_Vsf_equals_Vqf32(qf32);
        vc++;
    }

    if (left > 0) {
        for (size_t i = 0; i < left; ++i)
            z[i + blocks] = x[i + blocks] + y[i + blocks];
    }
}

static void ggml_compute_forward_add_f32(
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    uint64_t start_time = ggml_time_us();

    memcpy(dst->ne, src1->ne, 16);
    memcpy(dst->nb, src1->nb, 16);
    ggmlhexagon_dump_tensor(src0, 1);
    ggmlhexagon_dump_tensor(src1, 1);
    ggmlhexagon_dump_tensor(dst, 1);

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    const int ith = 0;
    const int nth = 1;

    const int nr  = ggml_nrows(src0);
    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(float));
    GGML_ASSERT(nb00 == sizeof(float));

    const int dr = (nr + nth - 1)/nth;
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);
    if (nb10 == sizeof(float)) {
        for (int ir = ir0; ir < ir1; ++ir) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int32_t i03 = ir/(ne02*ne01);
            const int32_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int32_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int32_t i13 = i03 % ne13;
            const int32_t i12 = i02 % ne12;
            const int32_t i11 = i01 % ne11;
            const int32_t nr0 = ne00 / ne10;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
            float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);
            for (int32_t r = 0; r < nr0; ++r) {
                ggmlhexagon_dsp_add_f32(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
            }
        }
    } else {
        // src1 is not contiguous
        for (int ir = ir0; ir < ir1; ++ir) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int32_t i03 = ir/(ne02*ne01);
            const int32_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int32_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int32_t i13 = i03 % ne13;
            const int32_t i12 = i02 % ne12;
            const int32_t i11 = i01 % ne11;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

            for (int32_t i0 = 0; i0 < ne0; ++i0) {
                const int32_t i10 = i0 % ne10;
                float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + i10*nb10);

                dst_ptr[i0] = src0_ptr[i0] + *src1_ptr;
            }
        }
    }

    uint64_t end_time = ggml_time_us();
    uint64_t duration = (end_time - start_time);
    GGMLHEXAGON_LOG_DEBUG("duration %llu us", duration);

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
}

int ggmlop_dsp_add(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst)
{
    GGMLHEXAGON_LOG_DEBUG("enter %s\n", __func__);
    switch (src0->type) {
        case GGML_TYPE_F32:
        {
            if (src1->type == GGML_TYPE_F32) {
                ggml_compute_forward_add_f32(src0, src1, dst);
            } else {
                GGML_ABORT("fatal error");
            }
            break;
        }
        default:
        {
            GGML_ABORT("fatal error");
        }
    }
    GGMLHEXAGON_LOG_DEBUG("leave %s\n", __func__);
    return 0;
}

static void ggml_compute_forward_sub_f32(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        struct ggml_tensor * dst) {

    memcpy(dst->ne, src1->ne, 16);
    memcpy(dst->nb, src1->nb, 16);

    assert(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    const int ith = 0;
    const int nth = 1;

    const int nr  = ggml_nrows(src0);

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(float));
    GGML_ASSERT(nb00 == sizeof(float));

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    if (nb10 == sizeof(float)) {
        for (int ir = ir0; ir < ir1; ++ir) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;
            const int64_t nr0 = ne00 / ne10;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
            float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);

            for (int64_t r = 0; r < nr0; ++r) {
#ifdef GGML_USE_ACCELERATE
                vDSP_vsub(src1_ptr, 1, src0_ptr + r*ne10, 1, dst_ptr + r*ne10, 1, ne10);
#else
                ggml_vec_sub_f32(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
#endif
            }
        }
    } else {
        // src1 is not contiguous
        for (int ir = ir0; ir < ir1; ++ir) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

            for (int64_t i0 = 0; i0 < ne0; ++i0) {
                const int64_t i10 = i0 % ne10;
                float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + i10*nb10);

                dst_ptr[i0] = src0_ptr[i0] - *src1_ptr;
            }
        }
    }
}
int ggmlop_dsp_sub(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F32:
        {
            ggml_compute_forward_sub_f32(src0, src1, dst);
        } break;
        default:
        {
            GGML_ABORT("fatal error");
        }
    }
    return 0;
}

static void ggml_compute_forward_mul_f32(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        struct ggml_tensor * dst) {

    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    memcpy(dst->ne, src1->ne, 16);
    memcpy(dst->nb, src1->nb, 16);


    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    const int ith = 0;
    const int nth = 1;

    const int64_t nr = ggml_nrows(src0);

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(float));
    GGML_ASSERT(nb00 == sizeof(float));

    if (nb10 == sizeof(float)) {
        for (int64_t ir = ith; ir < nr; ir += nth) {
            // src0 and dst are same shape => same indices
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;
            const int64_t nr0 = ne00 / ne10;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
            float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);

            for (int64_t r = 0 ; r < nr0; ++r) {
#ifdef GGML_USE_ACCELERATE
                UNUSED(ggml_vec_mul_f32);

                vDSP_vmul(src0_ptr + r*ne10, 1, src1_ptr, 1, dst_ptr + r*ne10, 1, ne10);
#else
                ggml_vec_mul_f32(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
#endif
            }
        }
    } else {
        // src1 is not contiguous
        for (int64_t ir = ith; ir < nr; ir += nth) {
            // src0 and dst are same shape => same indices
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

            for (int64_t i0 = 0; i0 < ne00; ++i0) {
                const int64_t i10 = i0 % ne10;
                float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + i10*nb10);

                dst_ptr[i0] = src0_ptr[i0] * (*src1_ptr);
            }
        }
    }
}

int ggmlop_dsp_mul(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s\n", __func__);
    switch (src0->type) {
        case GGML_TYPE_F32:
        {
            if (src1->type == GGML_TYPE_F32) {
                ggml_compute_forward_mul_f32(src0, src1, dst);
            } else {
                GGML_ABORT("fatal error");
            }
            break;
        }
        default:
        {
            GGML_ABORT("fatal error");
        }
    }
    GGMLHEXAGON_LOG_DEBUG("leave %s\n", __func__);
    return 0;
}
static void ggml_compute_forward_div_f32(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        struct ggml_tensor * dst) {

    memcpy(dst->ne, src1->ne, 16);
    memcpy(dst->nb, src1->nb, 16);

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    const int ith = 0;
    const int nth = 1;

    const int64_t nr = ggml_nrows(src0);

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(float));
    GGML_ASSERT(nb00 == sizeof(float));

    if (nb10 == sizeof(float)) {
        for (int64_t ir = ith; ir < nr; ir += nth) {
            // src0 and dst are same shape => same indices
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;
            const int64_t nr0 = ne00 / ne10;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
            float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);

            for (int64_t r = 0; r < nr0; ++r) {
#ifdef GGML_USE_ACCELERATE
                UNUSED(ggml_vec_div_f32);

                vDSP_vdiv(src1_ptr, 1, src0_ptr + r*ne10, 1, dst_ptr + r*ne10, 1, ne10);
#else
                ggml_vec_div_f32(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
#endif
            }
        }
    } else {
        // src1 is not contiguous
        for (int64_t ir = ith; ir < nr; ir += nth) {
            // src0 and dst are same shape => same indices
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t i03 = ir/(ne02*ne01);
            const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
            const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int64_t i13 = i03 % ne13;
            const int64_t i12 = i02 % ne12;
            const int64_t i11 = i01 % ne11;

            float * dst_ptr  = (float *) ((char *) dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
            float * src0_ptr = (float *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);

            for (int64_t i0 = 0; i0 < ne00; ++i0) {
                const int64_t i10 = i0 % ne10;
                float * src1_ptr = (float *) ((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + i10*nb10);

                dst_ptr[i0] = src0_ptr[i0] / (*src1_ptr);
            }
        }
    }
}

int ggmlop_dsp_div(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    switch (src0->type) {
        case GGML_TYPE_F32:
        {
            ggml_compute_forward_div_f32(src0, src1, dst);
        } break;

        default:
        {
            GGML_ABORT("fatal error");
        }
    }

    return 0;
}

static void ggml_compute_forward_mul_mat_one_chunk(
        const struct ggml_compute_params * params,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        struct ggml_tensor * dst,
        const enum ggml_type type,
        const int32_t num_rows_per_vec_dot,
        const int32_t ir0_start,
        const int32_t ir0_end,
        const int32_t ir1_start,
        const int32_t ir1_end) {
    ggmlhexagon_dump_tensor(src0, 0);
    ggmlhexagon_dump_tensor(src1, 0);
    ggmlhexagon_dump_tensor(dst, 0);

    dst->ne[0] = src0->ne[1];
    dst->ne[1] = src1->ne[1];
    dst->ne[2] = src1->ne[2];
    dst->ne[3] = src1->ne[3];

    dst->nb[0] = ggml_type_size(src1->type);
    dst->nb[1] = dst->nb[0] * (dst->ne[0] / ggml_blck_size(src1->type));
    dst->nb[2] = dst->nb[1] * dst->ne[1];
    dst->nb[3] = dst->nb[2] * dst->ne[2];
    ggmlhexagon_dump_tensor(dst, 0);

    GGML_TENSOR_BINARY_OP_LOCALS

    const bool src1_cont = ggml_is_contiguous(src1);

    ggml_vec_dot_t const vec_dot      = type_traits_cpu[type].vec_dot;
    enum ggml_type const vec_dot_type = type_traits_cpu[type].vec_dot_type;

    // broadcast factors
    const int32_t r2 = ne12 / ne02;
    const int32_t r3 = ne13 / ne03;

    if (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        return;
    }

    const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    const size_t row_size = ggml_row_size(vec_dot_type, ne10);

    assert(ne12 % ne02 == 0);
    assert(ne13 % ne03 == 0);

    // block-tiling attempt
    const int32_t blck_0 = 16;
    const int32_t blck_1 = 16;

    const size_t src1_col_stride = src1_cont || src1->type != vec_dot_type ? row_size : nb11;

    // attempt to reduce false-sharing (does not seem to make a difference)
    // 16 * 2, accounting for mmla kernels
    float tmp[32];

    for (int32_t iir1 = ir1_start; iir1 < ir1_end; iir1 += blck_1) {
        for (int32_t iir0 = ir0_start; iir0 < ir0_end; iir0 += blck_0) {
            for (int32_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir1_end; ir1 += num_rows_per_vec_dot) {
                const int32_t i13 = (ir1 / (ne12 * ne1));
                const int32_t i12 = (ir1 - i13 * ne12 * ne1) / ne1;
                const int32_t i11 = (ir1 - i13 * ne12 * ne1 - i12 * ne1);

                // broadcast src0 into src1
                const int32_t i03 = i13 / r3;
                const int32_t i02 = i12 / r2;

                const int32_t i1 = i11;
                const int32_t i2 = i12;
                const int32_t i3 = i13;

                const char * src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

                // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
                //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
                //       the original src1 data pointer, so we should index using the indices directly
                // TODO: this is a bit of a hack, we should probably have a better way to handle this
                const char * src1_col = (const char*)wdata +
                                        (src1_cont || src1->type != vec_dot_type
                                         ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                                         : (i11 * nb11 + i12 * nb12 + i13 * nb13));
                float * dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

                //for (int32_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ++ir0) {
                //    vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
                //}

                for (int32_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir0_end; ir0 += num_rows_per_vec_dot) {
                    vec_dot(ne00, &tmp[ir0 - iir0], (num_rows_per_vec_dot > 1 ? 16 : 0), src0_row + ir0 * nb01, (num_rows_per_vec_dot > 1 ? nb01 : 0), src1_col, (num_rows_per_vec_dot > 1 ? src1_col_stride : 0), num_rows_per_vec_dot);
                }

                for (int cn = 0; cn < num_rows_per_vec_dot; ++cn) {
                    memcpy(&dst_col[iir0 + cn * nb1 / nb0], tmp + (cn * 16), (MIN(iir0 + blck_0, ir0_end) - iir0) * sizeof(float));
                }
            }
        }
    }
}

int ggmlop_dsp_mulmat(remote_handle64 h, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    ggmlhexagon_dump_tensor(src0, 0);
    ggmlhexagon_dump_tensor(src1, 0);
    ggmlhexagon_dump_tensor(dst, 0);

    dst->ne[0] = src0->ne[1];
    dst->ne[1] = src1->ne[1];
    dst->ne[2] = src1->ne[2];
    dst->ne[3] = src1->ne[3];

    dst->nb[0] = ggml_type_size(src1->type);
    dst->nb[1] = dst->nb[0] * (dst->ne[0] / ggml_blck_size(src1->type));
    dst->nb[2] = dst->nb[1] * dst->ne[1];
    dst->nb[3] = dst->nb[2] * dst->ne[2];
    ggmlhexagon_dump_tensor(dst, 0);

    GGML_TENSOR_BINARY_OP_LOCALS

    enum ggml_type           const vec_dot_type         = type_traits_cpu[src0->type].vec_dot_type;
    ggml_from_float_t        const from_float           = type_traits_cpu[vec_dot_type].from_float;
    int32_t                  const vec_dot_num_rows     = type_traits_cpu[src0->type].nrows;
    const int ith = 0;
    const int nth = 1;

    struct ggml_compute_params params;
    params.ith = 0;
    params.nth = 1;
    params.wsize = 0;
    params.wdata = NULL;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

#if 0 //naive algorithm for fp32, can pass various case in UT
    {
        //ggml_dump_tensor(src0);
        //ggml_dump_tensor(src1);

        float * a = (float*)src0->data;
        float * b = (float*)src1->data;
        float * c = (float*)dst->data;
        int M = src0->ne[1];
        int K = src0->ne[0];
        int N = src1->ne[1];
        float sum = 0;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                sum = 0;
                for (int h = 0; h < K; h++) {
                    sum += a[i * K + h] * b[h * N + j];
                }
                c[i * N + j] = sum;
            }
        }
        return 0;
    }
#endif

    if (src1->type != vec_dot_type) {
        params.wsize = ggml_row_size(vec_dot_type, ggml_nelements(src1));
        params.wdata = (char*)malloc(params.wsize);
    }

    if (src1->type != vec_dot_type) {
        char * wdata = params.wdata;

        const size_t nbw0 = ggml_type_size(vec_dot_type);
        const size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        assert(params.wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        for (int64_t i13 = 0; i13 < ne13; ++i13) {
            for (int64_t i12 = 0; i12 < ne12; ++i12) {
                for (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + ne10_block_start*bs*nb10),
                               (void *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
    }


    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    const int32_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    const int32_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    int chunk_size = 16;

    // We need to step up the size if it's small
    if (nr0 == 1 || nr1 == 1) {
        chunk_size = 64;
    }

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim.
    // CEIL(nr0/chunk_size)
    int32_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int32_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // If the chunking is poor for the number of threads on this setup, scrap the whole plan.  Re-chunk it by thread.
    //   Also, chunking by thread was measured to have perform better on NUMA systems.  See https://github.com/ggml-org/llama.cpp/pull/6915
    //   In theory, chunking should be just as useful on NUMA and non NUMA systems, but testing disagreed with that.
    if (nchunk0 * nchunk1 <  4) {
        // distribute the thread work across the inner or outer loop based on which one is larger
        nchunk0 =  1; // parallelize by src0 rows
        nchunk1 =  1; // parallelize by src1 rows
    }

    // The number of elements in each chunk
    const int32_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int32_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    int current_chunk = 0;

    while (current_chunk < nchunk0 * nchunk1) {
        const int32_t ith0 = current_chunk % nchunk0;
        const int32_t ith1 = current_chunk / nchunk0;

        const int32_t ir0_start = dr0 * ith0;
        const int32_t ir0_end = MIN(ir0_start + dr0, nr0);

        const int32_t ir1_start = dr1 * ith1;
        const int32_t ir1_end = MIN(ir1_start + dr1, nr1);

        // dot kernels can handle 1 row and col at a time, but mmla kernels can process 2 rows and cols
        int32_t num_rows_per_vec_dot = vec_dot_num_rows;

        // these checks are needed to avoid crossing dim1 boundaries
        // can be optimized, but the logic would become more complicated, so keeping it like this for simplicity
        if ((nr0 % 2 != 0) || (ne11 % 2 != 0) || ((ir0_end - ir0_start) % 2 != 0) || ((ir1_end - ir1_start) % 2 != 0)) {
            num_rows_per_vec_dot = 1;
        }
        ggml_compute_forward_mul_mat_one_chunk(&params, src0, src1, dst, src0->type, num_rows_per_vec_dot, ir0_start, ir0_end, ir1_start, ir1_end);

        if (1 >= nchunk0 * nchunk1) {
            break;
        }
        current_chunk++;
    }
    if (src1->type != vec_dot_type) {
        free(params.wdata);
    }

    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return 0;
}
