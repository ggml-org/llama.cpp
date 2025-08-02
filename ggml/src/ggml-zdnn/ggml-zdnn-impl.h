#ifndef GGML_ZDNN_IMPL
#define GGML_ZDNN_IMPL

#include "zdnn.h"
#include "ggml.h"
#include "ggml-zdnn.h"

#include <vector>
#include <memory>
#include <vecintrin.h>

#define GGML_ZDNN_NAME    "zDNN"
#define GGML_ZDNN_VERSION ZDNN_VERNUM

#define vec_neg(a)    (-(a))                // Vector Negate
#define vec_add(a, b) ((a) + (b))           // Vector Add
#define vec_sub(a, b) ((a) - (b))           // Vector Subtract
#define vec_mul(a, b) ((a) * (b))           // Vector Multiply
#define vec_div(a, b) ((a) / (b))           // Vector Divide
#define vec_sl(a, b)  ((a) << (b))          // Vector Shift Left
#define vec_sra(a, b) ((a) >> (b))          // Vector Shift Right
#define vec_sr(a, b)  ((a) >> (b))          // Vector Shift Right Algebraic
#define vec_slo(a, b) vec_slb(a, (b) << 64) // Vector Shift Left by Octet
#define vec_sro(a, b) vec_srb(a, (b) << 64) // Vector Shift Right by Octet

#ifndef vec_and
#define vec_and(a, b) ((a) & (b)) // Vector AND
#endif

#ifndef vec_or
#define vec_or(a, b)  ((a) | (b)) // Vector OR
#endif

#ifndef vec_xor
#define vec_xor(a, b) ((a) ^ (b)) // Vector XOR
#endif

typedef   signed char char8x16_t  __attribute__((vector_size(16)));
typedef unsigned char uchar8x16_t __attribute__((vector_size(16)));

typedef int8_t   int8x16_t  __attribute__((vector_size(16)));
typedef int16_t  int16x8_t  __attribute__((vector_size(16)));
typedef int32_t  int32x4_t  __attribute__((vector_size(16)));
typedef uint8_t  uint8x16_t __attribute__((vector_size(16)));
typedef uint16_t uint16x8_t __attribute__((vector_size(16)));
typedef uint32_t uint32x4_t __attribute__((vector_size(16)));

typedef float float32x4_t   __attribute__((vector_size(16)));
typedef double double64x2_t __attribute__((vector_size(16)));

typedef   signed long long long64x2_t  __attribute__((vector_size(16)));
typedef unsigned long long ulong64x2_t __attribute__((vector_size(16)));

#define ZDNN_CHECK(stmt)                \
    do {                                \
        zdnn_status status = (stmt);    \
        GGML_ASSERT(status == ZDNN_OK); \
    } while (0);

struct ggml_backend_zdnn_device_context {
    int zdnn_device;
    int zdnn_device_ref_count;

    bool has_parmblkformat_0;
    bool has_parmblkformat_1;

    size_t max_size;

    char name[128];
};

struct ggml_backend_zdnn_context {
    int device;
    ggml_cgraph * gf;
};

struct ggml_backend_zdnn_buffer {
    void * data;
    size_t size;

    zdnn_tensor_desc pre_tfm_desc;
    zdnn_tensor_desc tfm_desc;
    zdnn_ztensor     ztensor;

    char name[GGML_MAX_NAME];
};

struct ggml_backend_zdnn_buffer_context {
    void * all_data;
    size_t all_size;
    bool owned;

    int n_buffers;
    std::vector<std::unique_ptr<ggml_backend_zdnn_buffer>> buffers;
};

#endif  // GGML_ZDNN_IMPL
