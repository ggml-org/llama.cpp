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


#ifndef TMAC_HALF_TYPEDEF_H
#define TMAC_HALF_TYPEDEF_H

#ifndef __AVX2__
typedef _Float16 half;
#endif
#endif

#include "lut_ctor.h"


#ifdef __cplusplus
extern "C" {
#endif

int32_t tbl_int8_reset(int32_t m, int8_t* c);

int32_t tbl_float_reset(int32_t m, void* c);

int32_t tbl_int32_reset(int32_t m, int32_t* c);

int32_t tbl_int16_reset(int32_t m, int16_t* c);


void qgemm_lut_int8_g4(
        void* A, void* LUT, void* Scales, void* LUT_Scales, void* LUT_Biases, void* C,
        int bm, int K, int N, const struct tmac_kernel_config * const kernel_config);

#ifdef __cplusplus
}
#endif








