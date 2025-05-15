#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
typedef float32_t tmac_float_type;
#else
typedef float tmac_float_type;
#endif

#ifdef  __cplusplus
extern "C" {
#endif

struct tmac_tensor_extra {
    int lut_scales_size;
    int scales_size;
    int n_tile_num;
    uint8_t * qweights;
    tmac_float_type * scales;
};

GGML_API void ggml_tmac_init(void);
GGML_API void ggml_tmac_free(void);
// src0->type == Q4_0/IQ2_XXS/IQ3_XXS
// T-MAC currently only supports BitNet quantization or GPTQ-like quantization (only scales, without zeros)
// If use i-quantization gguf models, the results will be wrong
// TODO: add customized block types Q2_0/Q3_0
GGML_API bool ggml_tmac_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst);
GGML_API size_t ggml_tmac_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst);
GGML_API void ggml_tmac_mul_mat_task_init(void * src1, void * qlut, void * lut_scales, void * lut_biases, int n, int k, int m, int bits);
GGML_API void ggml_tmac_mul_mat_task_compute(void * src0, void * scales, void * qlut, void * lut_scales, void * lut_biases, void * dst, int n, int k, int m, int bits);
GGML_API void ggml_tmac_transform_tensor(struct ggml_tensor * tensor);
GGML_API int ggml_tmac_get_type_bits(enum ggml_type type);
GGML_API void ggml_tmac_set_n_threads(int n_threads);
#if defined(GGML_BITNET_ARM_TL1)
GGML_API void ggml_qgemm_lut(int k, void* A, void* LUT, void* Scales, void* LUT_Scales, void* C);
GGML_API void ggml_preprocessor(int k, void* B, void* LUT_Scales, void* QLUT);
#endif
#if defined(GGML_BITNET_X86_TL2)
GGML_API void ggml_qgemm_lut(int bs, int k, void* A, void* sign, void* LUT, void* Scales, void* LUT_Scales, void* C);
GGML_API void ggml_preprocessor(int bs, int three_k, int two_k, void* B, void* LUT_Scales, void* Three_QLUT, void* Two_QLUT);
#endif
<<<<<<< HEAD

=======
>>>>>>> upstream/release-dev
#ifdef  __cplusplus
}
#endif
