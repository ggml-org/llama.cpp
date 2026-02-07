#pragma once

#if !defined(QK_IFAIRY_GROUPS_PER_BLOCK)
#    ifdef __cplusplus
#        define GGML_COMMON_DECL_CPP
#    else
#        define GGML_COMMON_DECL_C
#    endif
#    include "ggml-common.h"
#endif

#include "ggml-backend.h"
#include "ggml.h"

#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// Common env helpers shared by ggml-cpu.c and ggml-ifairy-lut*.cpp.
// "enabled" means: env is set and not equal to "0".
static inline bool ggml_ifairy_env_enabled(const char * name) {
    const char * env = getenv(name);
    return env && strcmp(env, "0") != 0;
}

// Parse an integer env var if it is set and not equal to "0", otherwise return default_value.
// If parsing fails, also returns default_value.
static inline int ggml_ifairy_env_get_int_nonzero(const char * name, int default_value) {
    const char * env = getenv(name);
    if (!env || strcmp(env, "0") == 0) {
        return default_value;
    }

    errno          = 0;
    char *     end = NULL;
    const long v   = strtol(env, &end, 10);
    if (end == env) {
        return default_value;
    }

    if (v > (long) INT_MAX) {
        return INT_MAX;
    }
    if (v < (long) INT_MIN) {
        return INT_MIN;
    }
    if (errno == ERANGE) {
        return v < 0 ? INT_MIN : INT_MAX;
    }
    return (int) v;
}

struct ifairy_lut_extra {
    uint8_t *             indexes;
    size_t                size;
    void *                packed_w;
    size_t                packed_w_size;
    struct ggml_tensor *  index_tensor;
    ggml_backend_buffer_t index_buffer;
};

#if defined(_MSC_VER)
#    define GGML_IFAIRY_LUT_ALIGN(n) __declspec(align(n))
#else
#    define GGML_IFAIRY_LUT_ALIGN(n) __attribute__((aligned(n)))
#endif

#define GGML_IFAIRY_LUT_WTILE_ALIGNMENT 64

// Packed iFairy 3-weight codes for lut_c-style kernels:
// - per ggml QK_IFAIRY block has QK_IFAIRY_GROUPS_PER_BLOCK groups
// - each group stores 16 lanes (16 output rows) of 1-byte codes (idx16 + flags)
// - d_real/d_imag are per-row/per-block weight scales, stored as float for fast use in kernels
struct GGML_IFAIRY_LUT_ALIGN(GGML_IFAIRY_LUT_WTILE_ALIGNMENT) ifairy_lut_wtile_16 {
    uint8_t qs[QK_IFAIRY_GROUPS_PER_BLOCK][16];
    float   d_real[16];
    float   d_imag[16];
};

// iFairy 3-weight LUT API
//
// Current state:
// - CPU-only iFairy LUT path integrated into ggml mul_mat (guarded by GGML_IFAIRY_ARM_LUT + GGML_IFAIRY_LUT env).
// - Correctness matches ggml_vec_dot_ifairy_q16_K_generic semantics (w * conj(x)).
// - Index encoding is 6-bit pattern per 3 weights: pat = c0 | (c1<<2) | (c2<<4).
// - V2 core path keeps a single production layout/kernel:
//   - LUT: 16 entries × 4 channels × int8 per group (64B/group), lut_c-style
//   - Weights: packed 16-row tiles (`struct ifairy_lut_wtile_16`), lut_c-style
// - Runtime env:
//   - `GGML_IFAIRY_LUT=0/1` (enable/disable)
//   - `GGML_IFAIRY_LUT_DEBUG=0/1` (debug logging)
//   - `GGML_IFAIRY_LUT_IMPL=auto|lut16|lut_c` (optional impl selection; `lut_c` uses 42.6-scaled Q8 activations
//     when src1 is F32; otherwise falls back to lut16)

void   ggml_ifairy_lut_init(void);
void   ggml_ifairy_lut_free(void);
bool   ggml_ifairy_lut_can_mul_mat(const struct ggml_tensor * src0,
                                   const struct ggml_tensor * src1,
                                   const struct ggml_tensor * dst);
size_t ggml_ifairy_lut_get_wsize(const struct ggml_tensor * src0,
                                 const struct ggml_tensor * src1,
                                 const struct ggml_tensor * dst,
                                 int                        n_threads);
bool   ggml_ifairy_lut_transform_tensor(struct ggml_tensor * tensor, struct ggml_tensor ** index_tensor_out);
void   ggml_ifairy_lut_preprocess_ex_lut16(int          m,
                                           int          k,
                                           int          n,
                                           const void * act,
                                           size_t       act_stride,
                                           void *       lut_scales,
                                           void *       lut_buf,
                                           int          ith,
                                           int          nth);
void   ggml_ifairy_lut_preprocess_ex_lut_c(int          m,
                                           int          k,
                                           int          n,
                                           const void * act,
                                           size_t       act_stride,
                                           void *       lut_scales,
                                           void *       lut_buf,
                                           int          ith,
                                           int          nth);
void   ggml_ifairy_lut_qgemm_lut16(int          m,
                                   int          k,
                                   int          n,
                                   const void * packed_wtiles,
                                   const void * lut,
                                   const void * lut_scales,
                                   float *      dst,
                                   size_t       dst_col_stride,
                                   size_t       dst_row_stride,
                                   bool         pack_bf16,
                                   bool         add);
void   ggml_ifairy_lut_qgemm_lut_c(int          m,
                                   int          k,
                                   int          n,
                                   const void * packed_wtiles,
                                   const void * lut,
                                   const void * lut_scales,
                                   float *      dst,
                                   size_t       dst_col_stride,
                                   size_t       dst_row_stride,
                                   bool         pack_bf16,
                                   bool         add);
void   ggml_ifairy_lut_mul_mat_scalar(int          m,
                                      int          k,
                                      int          n,
                                      const void * qweights,
                                      const void * act,
                                      size_t       act_stride,
                                      float *      dst);

#ifdef __cplusplus
}
#endif
