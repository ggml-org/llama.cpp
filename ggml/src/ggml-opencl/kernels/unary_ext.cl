#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// Extended elementwise unary ops, same variant shape as abs.cl:
//   f32, f32_4 (vec4), f16, f16_4 (vec4), f32_nc, f16_nc (stride-addressed).
//
//   sgn, step, elu, hardswish, hardsigmoid, floor, ceil, round, trunc.
//
// Semantics match the ggml CPU reference (ggml.c). Values are computed in float
// (the f16 variants read/write half and convert), so the conditional ops match
// the CPU bit-for-bit within tolerance. SEXPR is the scalar form, VEXPR the
// float4 form (vector ternaries need select()).
//------------------------------------------------------------------------------

#define UNARY_EXT(NAME, SEXPR, VEXPR)                                           \
kernel void kernel_##NAME##_f32(                                               \
        global const float * src0, ulong offset0,                             \
        global       float * dst,  ulong offsetd) {                           \
    src0 = (global float*)((global char*)src0 + offset0);                     \
    dst  = (global float*)((global char*)dst  + offsetd);                     \
    float x = src0[get_global_id(0)];                                         \
    dst[get_global_id(0)] = (SEXPR);                                          \
}                                                                             \
kernel void kernel_##NAME##_f32_4(                                            \
        global const float4 * src0, ulong offset0,                           \
        global       float4 * dst,  ulong offsetd) {                         \
    src0 = (global float4*)((global char*)src0 + offset0);                   \
    dst  = (global float4*)((global char*)dst  + offsetd);                   \
    float4 x = src0[get_global_id(0)];                                       \
    dst[get_global_id(0)] = (VEXPR);                                         \
}                                                                             \
kernel void kernel_##NAME##_f16(                                             \
        global const half * src0, ulong offset0,                            \
        global       half * dst,  ulong offsetd) {                          \
    src0 = (global half*)((global char*)src0 + offset0);                    \
    dst  = (global half*)((global char*)dst  + offsetd);                    \
    float x = src0[get_global_id(0)];                                       \
    dst[get_global_id(0)] = (SEXPR);                                        \
}                                                                            \
kernel void kernel_##NAME##_f16_4(                                          \
        global const half4 * src0, ulong offset0,                          \
        global       half4 * dst,  ulong offsetd) {                        \
    src0 = (global half4*)((global char*)src0 + offset0);                  \
    dst  = (global half4*)((global char*)dst  + offsetd);                  \
    float4 x = convert_float4(src0[get_global_id(0)]);                     \
    dst[get_global_id(0)] = convert_half4(VEXPR);                          \
}                                                                           \
kernel void kernel_##NAME##_f32_nc(                                         \
        global const char * src0, ulong offset0,                          \
        global       char * dst,  ulong offsetd,                          \
        int ne00, ulong nb00, ulong nb01, ulong nb02, ulong nb03,         \
        ulong nb0, ulong nb1, ulong nb2, ulong nb3) {                     \
    src0 = src0 + offset0; dst = dst + offsetd;                            \
    const int i3 = get_group_id(2);                                       \
    const int i2 = get_group_id(1);                                       \
    const int i1 = get_group_id(0);                                       \
    for (int i0 = get_local_id(0); i0 < ne00; i0 += get_local_size(0)) {  \
        float x = *(global const float *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00); \
        *(global float *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0) = (SEXPR);            \
    }                                                                     \
}                                                                         \
kernel void kernel_##NAME##_f16_nc(                                       \
        global const char * src0, ulong offset0,                        \
        global       char * dst,  ulong offsetd,                        \
        int ne00, ulong nb00, ulong nb01, ulong nb02, ulong nb03,       \
        ulong nb0, ulong nb1, ulong nb2, ulong nb3) {                   \
    src0 = src0 + offset0; dst = dst + offsetd;                          \
    const int i3 = get_group_id(2);                                     \
    const int i2 = get_group_id(1);                                     \
    const int i1 = get_group_id(0);                                     \
    for (int i0 = get_local_id(0); i0 < ne00; i0 += get_local_size(0)) {\
        float x = *(global const half *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00); \
        *(global half *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0) = (SEXPR);            \
    }                                                                   \
}

UNARY_EXT(sgn,          sign(x),                                           sign(x))
UNARY_EXT(step,         x > 0.0f ? 1.0f : 0.0f,                            select((float4)0.0f, (float4)1.0f, x > 0.0f))
UNARY_EXT(elu,          x > 0.0f ? x : expm1(x),                           select(expm1(x), x, x > 0.0f))
UNARY_EXT(hardswish,    x * fmin(1.0f, fmax(0.0f, (x + 3.0f) / 6.0f)),     x * fmin((float4)1.0f, fmax((float4)0.0f, (x + 3.0f) / 6.0f)))
UNARY_EXT(hardsigmoid,  fmin(1.0f, fmax(0.0f, (x + 3.0f) / 6.0f)),         fmin((float4)1.0f, fmax((float4)0.0f, (x + 3.0f) / 6.0f)))
UNARY_EXT(floor,        floor(x),                                          floor(x))
UNARY_EXT(ceil,         ceil(x),                                           ceil(x))
UNARY_EXT(round,        round(x),                                          round(x))
UNARY_EXT(trunc,        trunc(x),                                          trunc(x))
