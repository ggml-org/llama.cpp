#pragma once

/* Please do not include this header file outside ggml-cpu/tmac */

#ifndef INTRINSIC_TYPES_H
#define INTRINSIC_TYPES_H

#ifdef __ARM_NEON
#include <arm_neon.h>
#elif defined __AVX2__
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
typedef float16_t tmac_float_type;
#else
#include <stdbool.h>
#include <stdint.h>
typedef float tmac_float_type;
#endif

#endif


#ifdef __ARM_NEON
#define vaddvq_f16(v) \
    ((v)[0] + (v)[1] + (v)[2] + (v)[3] + (v)[4] + (v)[5] + (v)[6] + (v)[7])
#elif defined __AVX2__
static inline float _mm256_addv_ps(const __m256 v);
#endif

#define my_fputs(s) fputs(s, stderr); fflush(stderr);
#define my_fputsf(buf, s, ...) snprintf(buf, sizeof(buf), s, __VA_ARGS__); my_fputs(buf);


struct tmac_kernel_config {
    int32_t g;
    int32_t ngroups_per_elem;
    int32_t q_group_size;
    int32_t act_group_size;

    bool has_scale;
    int kfactor;
    int bits;
    int actk;   // should be equal to (act_group_size / g).
    bool has_zero_point;
    bool one_scale;

    int32_t bm;
    uint32_t simd_n_in;
    uint32_t simd_n_out;

    int32_t chunk_n;
};



#ifdef __cplusplus
extern "C" {
#endif

int32_t partial_max_g4_int8_k8(void* lut_scales_, void* b_);

int32_t partial_max_reset(void* lut_scales_);

void lut_ctor_int8_g4(void* B, void* LUT_Scales, void* LUT_Biases, void* QLUT, int K, const struct tmac_kernel_config * const kernel_config);

#ifdef __cplusplus
}
#endif


