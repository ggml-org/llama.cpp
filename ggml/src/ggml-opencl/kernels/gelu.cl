#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// gelu
//------------------------------------------------------------------------------
#define GELU_COEF_A     0.044715f
#define GELU_QUICK_COEF -1.702f
#define SQRT_2_OVER_PI  0.79788456080286535587989211986876f
#define SQRT_2_INV      0.70710678118654752440084436210484f

kernel void kernel_gelu(
    global float * src0,
    ulong offset0,
    global float * dst,
    ulong offsetd
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    float x = src0[get_global_id(0)];

    dst[get_global_id(0)] = 0.5f*x*(1.0f + tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

kernel void kernel_gelu_4(
    global float4 * src0,
    ulong offset0,
    global float4 * dst,
    ulong offsetd
) {
    src0 = (global float4*)((global char*)src0 + offset0);
    dst = (global float4*)((global char*)dst + offsetd);

    float4 x = src0[get_global_id(0)];

    dst[get_global_id(0)] = 0.5f*x*(1.0f + tanh(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
}

kernel void kernel_gelu_erf(
    global float * src0,
    ulong offset0,
    global float * dst,
    ulong offsetd
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    float x = src0[get_global_id(0)];
    dst[get_global_id(0)] = 0.5f*x*(1.0f + erf(x*SQRT_2_INV));
}

kernel void kernel_gelu_erf_4(
    global float4 * src0,
    ulong offset0,
    global float4 * dst,
    ulong offsetd
) {
    src0 = (global float4*)((global char*)src0 + offset0);
    dst = (global float4*)((global char*)dst + offsetd);

    float4 x = src0[get_global_id(0)];
    dst[get_global_id(0)] = 0.5f*x*(1.0f + erf(x*SQRT_2_INV));
}

kernel void kernel_gelu_quick(
    global float * src0,
    ulong offset0,
    global float * dst,
    ulong offsetd
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    float x = src0[get_global_id(0)];
    dst[get_global_id(0)] = x*(1.0f/(1.0f+exp(GELU_QUICK_COEF*x)));
}

kernel void kernel_gelu_quick_4(
    global float4 * src0,
    ulong offset0,
    global float4 * dst,
    ulong offsetd
) {
    src0 = (global float4*)((global char*)src0 + offset0);
    dst = (global float4*)((global char*)dst + offsetd);

    float4 x = src0[get_global_id(0)];
    dst[get_global_id(0)] = x*(1.0f/(1.0f+exp(GELU_QUICK_COEF*x)));
}
