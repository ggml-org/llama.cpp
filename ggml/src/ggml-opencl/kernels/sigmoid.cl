#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// sigmoid
//------------------------------------------------------------------------------

kernel void kernel_sigmoid_f32(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    dst[get_global_id(0)] = 1.0f / (1.0f + exp(-src0[get_global_id(0)]));
}

kernel void kernel_sigmoid_f16(
        global half * src0,
        ulong offset0,
        global half * dst,
        ulong offsetd
) {
    src0 = (global half*)((global char*)src0 + offset0);
    dst = (global half*)((global char*)dst + offsetd);

    dst[get_global_id(0)] = 1.0f / (1.0f + exp(-src0[get_global_id(0)]));
}
