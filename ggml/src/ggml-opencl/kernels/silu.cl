#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// silu
//------------------------------------------------------------------------------
kernel void kernel_silu(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    float x = src0[get_global_id(0)];
    const float xf = (float)x;
    dst[get_global_id(0)] = xf / (1.0f + exp(-xf));
}

kernel void kernel_silu_4(
        global float4 * src0,
        ulong offset0,
        global float4 * dst,
        ulong offsetd
) {
    src0 = (global float4*)((global char*)src0 + offset0);
    dst = (global float4*)((global char*)dst + offsetd);

    float4 x = src0[get_global_id(0)];
    const float4 xf = x;
    dst[get_global_id(0)] = xf / (1.0f + exp(-xf));
}
