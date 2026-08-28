#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// clamp
//------------------------------------------------------------------------------
kernel void kernel_clamp(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        float min,
        float max
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    dst[get_global_id(0)] = src0[get_global_id(0)] < min ?
        min :
        (src0[get_global_id(0)] > max ? max : src0[get_global_id(0)]);
}
