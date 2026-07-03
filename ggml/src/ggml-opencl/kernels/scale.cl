#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void kernel_scale_f32(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        float scale,
        float bias
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);
    dst[get_global_id(0)] = src0[get_global_id(0)] * scale + bias;
}

kernel void kernel_scale_f32_4(
        global float4 * src0,
        ulong offset0,
        global float4 * dst,
        ulong offsetd,
        float scale,
        float bias
) {
    src0 = (global float4*)((global char*)src0 + offset0);
    dst = (global float4*)((global char*)dst + offsetd);
    dst[get_global_id(0)] = src0[get_global_id(0)] * scale + bias;
}
