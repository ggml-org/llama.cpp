#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "HAP_perf.h"
#include "HAP_farf.h"
#include "HAP_power.h"
#include "HAP_vtcm_mgr.h"
#include "HAP_compute_res.h"

#include "qurt.h"
#include "AEEStdErr.h"
#include "hexagon_types.h"
#include "hexagon_protos.h"

#include "skel.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define ggml_tensor         dsptensor

#define GGML_MAX_DIMS       4

#define ALIGN_128_BYTE      128

#define VLEN                128

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

#define GGML_API            extern

#ifdef __cplusplus
// restrict not standard in C++
#    if defined(__GNUC__)
#        define GGML_RESTRICT       __restrict__
#    elif defined(__clang__)
#        define GGML_RESTRICT       __restrict
#    elif defined(_MSC_VER)
#        define GGML_RESTRICT       __restrict
#    else
#        define GGML_RESTRICT
#    endif
#else
#    if defined (_MSC_VER) && (__STDC_VERSION__ < 201112L)
#        define GGML_RESTRICT       __restrict
#    else
#        define GGML_RESTRICT       restrict
#    endif
#endif

#ifndef __cplusplus
#ifndef static_assert
        #if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
            #define static_assert(cond, msg) _Static_assert(cond, msg)
        #else
            #define static_assert(cond, msg) struct global_scope_noop_trick
        #endif
#endif
#endif // __cplusplus


//NPU performance will be slower when enable GGMLHEXAGON_DEBUG
#ifdef NDEBUG
#define GGMLHEXAGON_DEBUG                                   0
#else
#define GGMLHEXAGON_DEBUG                                   1
#endif

#define GGMLHEXAGON_LOGBUF_LEN                              4096
#define GGMLHEXAGON_TMPBUF_LEN                              256
#if GGMLHEXAGON_DEBUG
#define GGMLHEXAGON_LOG_DEBUG(...)                          ggmlhexagon_log_internal(GGMLHEXAGON_LOG_LEVEL_DEBUG, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)
#else
#define GGMLHEXAGON_LOG_DEBUG(...)
#endif

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
};

enum ggml_type {
    GGML_TYPE_F32     = 0,
};

typedef double      ggml_float;

GGML_API int64_t ggml_time_ms(void);
GGML_API int64_t ggml_time_us(void);

GGML_API size_t ggml_nbytes(const struct ggml_tensor * tensor);
GGML_API int64_t ggml_nrows(const struct ggml_tensor * tensor);
GGML_API int ggml_n_dims(const struct ggml_tensor * tensor);
GGML_API bool ggml_is_contiguous(const struct ggml_tensor * tensor);
GGML_API void ggml_abort(const char * file, int line, const char * fmt, ...);
GGML_API bool ggml_can_repeat(const struct ggml_tensor * t0, const struct ggml_tensor * t1);
GGML_API bool ggml_are_same_shape(const struct ggml_tensor * t0, const struct ggml_tensor * t1);

GGML_API void ggmlhexagon_dump_tensor_elements(const ggml_tensor * tensor);
GGML_API void ggmlhexagon_dump_tensor(const ggml_tensor * tensor, int dump_tensor_data);
GGML_API void ggmlhexagon_log_internal(int level, const char *file, const char *func, int line, const char *format, ...);

GGML_API int ggmlop_get_thread_counts(void);

#ifdef  __cplusplus
}
#endif
