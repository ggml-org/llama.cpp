#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// iFairy 3-weight LUT API skeleton
// Note: current implementation is a stub; real functionality will be added in subsequent steps.

void   ggml_ifairy_lut_init(void);
void   ggml_ifairy_lut_free(void);
bool   ggml_ifairy_lut_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst);
size_t ggml_ifairy_lut_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst, int n_threads);
bool   ggml_ifairy_lut_transform_tensor(struct ggml_tensor * tensor, struct ggml_tensor ** index_tensor_out);
void   ggml_ifairy_lut_preprocess(int m, int k, int n, const void * act, size_t act_stride, void * lut_scales, void * lut_buf);
void   ggml_ifairy_lut_qgemm(int m, int k, int n, const void * qweights, const uint8_t * indexes, const void * lut, const void * lut_scales, float * dst, size_t dst_stride);

#ifdef __cplusplus
}
#endif

