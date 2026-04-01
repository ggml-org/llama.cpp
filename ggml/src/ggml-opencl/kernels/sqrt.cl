#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void kernel_sqrt_cont_f32(
    global float * src0,
    ulong          offset0,
    global float * dst,
    ulong          offsetd
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst  = (global float*)((global char*)dst + offsetd);

    uint gid = get_global_id(0);
    dst[gid] = sqrt(src0[gid]);
}

kernel void kernel_sqrt_cont_f32_4(
    global float4 * src0,
    ulong           offset0,
    global float4 * dst,
    ulong           offsetd
) {
    src0 = (global float4*)((global char*)src0 + offset0);
    dst  = (global float4*)((global char*)dst + offsetd);

    uint gid = get_global_id(0);
    dst[gid] = sqrt(src0[gid]);
}

kernel void kernel_sqrt_cont_f16(
    global half * src0,
    ulong         offset0,
    global half * dst,
    ulong         offsetd
) {
    src0 = (global half*)((global char*)src0 + offset0);
    dst  = (global half*)((global char*)dst + offsetd);

    uint gid = get_global_id(0);
#if GGML_OPENCL_USE_NATIVE_FP16_MATH
    dst[gid] = sqrt(src0[gid]);
#else
    dst[gid] = (half) sqrt((float) src0[gid]);
#endif
}

kernel void kernel_sqrt_cont_f16_4(
    global half4 * src0,
    ulong          offset0,
    global half4 * dst,
    ulong          offsetd
) {
    src0 = (global half4*)((global char*)src0 + offset0);
    dst  = (global half4*)((global char*)dst + offsetd);

    uint gid = get_global_id(0);
#if GGML_OPENCL_USE_NATIVE_FP16_MATH
    dst[gid] = sqrt(src0[gid]);
#else
    half4 h = src0[gid];
    float4 f = (float4)((float)h.s0, (float)h.s1, (float)h.s2, (float)h.s3);
    float4 r = sqrt(f);
    dst[gid] = (half4)((half)r.s0, (half)r.s1, (half)r.s2, (half)r.s3);
#endif
}
