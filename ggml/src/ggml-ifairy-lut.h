#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ifairy_lut_extra {
    uint8_t * indexes;
    size_t    size;
    struct ggml_tensor * index_tensor;
    ggml_backend_buffer_t index_buffer;
};

// iFairy 3-weight LUT API
//
// Current state:
// - CPU-only scalar LUT path integrated into ggml mul_mat (guarded by GGML_IFAIRY_ARM_LUT + GGML_IFAIRY_LUT env).
// - Correctness matches ggml_vec_dot_ifairy_q16_K_generic semantics (w * conj(x)).
// - Index encoding is direct 6-bit pattern per 3 weights: pat = c0 | (c1<<2) | (c2<<4).
// - Two LUT layouts are supported (selected by env `GGML_IFAIRY_LUT_LAYOUT=auto|legacy|compact`):
//   - legacy : 4x64 int16 tables per group (fast for small N, larger workspace)
//   - compact: int8 "3 positions × 4 codes × 4 channels" tables per group (48 B / group), NEON uses 32-bit loads + widen

void   ggml_ifairy_lut_init(void);
void   ggml_ifairy_lut_free(void);
bool   ggml_ifairy_lut_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst);
size_t ggml_ifairy_lut_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst, int n_threads);
bool   ggml_ifairy_lut_transform_tensor(struct ggml_tensor * tensor, struct ggml_tensor ** index_tensor_out);
void   ggml_ifairy_lut_preprocess(int m, int k, int n, const void * act, size_t act_stride, void * lut_scales, void * lut_buf);
void   ggml_ifairy_lut_qgemm(int m, int k, int n, const void * qweights, const uint8_t * indexes, const void * lut, const void * lut_scales, const void * act, size_t act_stride, float * dst, size_t dst_col_stride, size_t dst_row_stride, bool pack_bf16, bool strict);
void   ggml_ifairy_lut_qgemm_ex(int m, int k, int n, const void * qweights, const uint8_t * indexes, const void * lut, const void * lut_scales, const void * act, size_t act_stride, float * dst, size_t dst_col_stride, size_t dst_row_stride, bool pack_bf16, bool strict, bool add);
void   ggml_ifairy_lut_accum4_ex(int k, int n, const uint8_t * indexes, const void * lut, const void * lut_scales, float * dst, size_t dst_col_stride, bool add);
void   ggml_ifairy_lut_mul_mat_scalar(int m, int k, int n, const void * qweights, const void * act, size_t act_stride, float * dst);

#ifdef __cplusplus
}
#endif
