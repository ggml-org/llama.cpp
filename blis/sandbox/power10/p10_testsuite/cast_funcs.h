#include "blis.h"

#define EASY_CAST_FUNC_NAME_(ch_src, ch_dst) cast_ ## ch_src ## _to_ ## ch_dst
#define EASY_CAST_FUNC_NAME(ch_src, ch_dst) EASY_CAST_FUNC_NAME_(ch_src, ch_dst)

#define CAST_MATRIX_FUNC_NAME_(ch_src, ch_dst) cast_ ## ch_src ## _to_ ## ch_dst ## m
#define CAST_MATRIX_FUNC_NAME(ch_src, ch_dst) CAST_MATRIX_FUNC_NAME_(ch_src, ch_dst)

#define CAST_MATRIX_FUNC_PROTO(ch_src, ch_dst, src_t, dst_t) \
void CAST_MATRIX_FUNC_NAME(ch_src, ch_dst) (src_t *, dst_t *, int)

#define EASY_CAST_FUNC_PROTO(ch_src, ch_dst, src_t, dst_t) \
dst_t EASY_CAST_FUNC_NAME(ch_src, ch_dst) (src_t)

#define EASY_CAST_FUNC(ch_src, ch_dst, src_t, dst_t) \
dst_t EASY_CAST_FUNC_NAME(ch_src, ch_dst) \
(src_t val) { \
    return (dst_t) val; \
}

#define CASTING_MATRIX_FUNC(ch_src, ch_dst, src_t, dst_t, cast_func) \
void CAST_MATRIX_FUNC_NAME(ch_src, ch_dst) \
(src_t *m1, dst_t *m2, int num_elems) { \
    for(int i=0;i<num_elems;i++) \
        m2[i] = cast_func (m1[i]); \
}

float cast_bf16_to_f32(bfloat16 val);
float cast_f16_to_f32(float16 val);

float16 cast_f32_to_f16(const float in);
bfloat16 cast_f32_to_bf16 (float val);

void cast_i4_to_f32(float *fvals, nibbles val);
nibbles cast_f32_to_i4(float val0, float val1);

void cast_f32_to_i4m(float *a_float, nibbles *a, int num_elems);
void cast_i4_to_f32m(nibbles *a, float *a_float, int num_elems);

EASY_CAST_FUNC_PROTO(f32, f32, float,   float);

EASY_CAST_FUNC_PROTO(f32, i32, float, int32_t);
EASY_CAST_FUNC_PROTO(f32, i16, float, int16_t);
EASY_CAST_FUNC_PROTO(f32,  i8, float,  int8_t);

EASY_CAST_FUNC_PROTO(i32, f32, int32_t, float);
EASY_CAST_FUNC_PROTO(i16, f32, int16_t, float);
EASY_CAST_FUNC_PROTO( i8, f32,  int8_t, float);

CAST_MATRIX_FUNC_PROTO(f32, f32, float, float);
CAST_MATRIX_FUNC_PROTO(f32, bf16, float, bfloat16);
CAST_MATRIX_FUNC_PROTO(f32,  f16, float, float16);

CAST_MATRIX_FUNC_PROTO(f32,  i32, float, int32_t);
CAST_MATRIX_FUNC_PROTO(f32,  i16, float, int16_t);
CAST_MATRIX_FUNC_PROTO(f32,   i8, float, int8_t);

CAST_MATRIX_FUNC_PROTO(bf16, f32, bfloat16, float);
CAST_MATRIX_FUNC_PROTO( f16, f32,  float16, float);
CAST_MATRIX_FUNC_PROTO( i32, f32,  int32_t, float);
CAST_MATRIX_FUNC_PROTO( i16, f32,  int16_t, float);
CAST_MATRIX_FUNC_PROTO(  i8, f32,   int8_t, float);
