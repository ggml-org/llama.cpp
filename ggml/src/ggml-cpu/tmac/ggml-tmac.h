#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "tbl.h"
#include "lut_ctor.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
typedef float16_t tmac_float_type;
#else
typedef float tmac_float_type;
#endif

#ifdef  __cplusplus
extern "C" {
#endif

struct llama_model_tmac_meta {
    int bits;
    int q_group_size;
    bool has_scale;
    bool has_zero_point;
    bool one_scale;
    char * quant_method;

    int g;
    int ngroups_per_elem;
    int act_group_size;
    int actk;
};

struct tmac_run_single_kernel_settings {
    int32_t test_time_ms;
    int32_t M;
    int32_t N;
    int32_t K;

    int32_t n;

    struct tmac_kernel_config * kernel_config;
};



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
GGML_API struct tmac_kernel_config * find_tmac_kernel_config(int M, int K, int bits);
GGML_API bool ggml_tmac_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst);
GGML_API size_t ggml_tmac_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst);
GGML_API void ggml_tmac_mul_mat_task_init(void * src1, void * qlut, void * lut_scales, void * lut_biases, int n, int k, int m, int bits);
GGML_API void ggml_tmac_mul_mat_task_compute(void * src0, void * scales, void * qlut, void * lut_scales, void * lut_biases, void * dst, int n, int k, int m, int bits);
GGML_API int ggml_tmac_get_scales_size(const struct tmac_kernel_config * kernel_config, int m, int k);
GGML_API void ggml_tmac_transform_tensor(struct ggml_tensor * tensor);
GGML_API int ggml_tmac_get_type_bits(enum ggml_type type);
GGML_API size_t ggml_tmac_get_nbytes(const struct ggml_tensor * tensor);
GGML_API void ggml_tmac_tune_kernel_config(int M, int K);

// GGML_API bool qtype_has_tmac_kernels(enum ggml_type type);

GGML_API bool tmac_meta_init(const char * tmac_meta_fname);

#ifdef  __cplusplus
}
#endif
