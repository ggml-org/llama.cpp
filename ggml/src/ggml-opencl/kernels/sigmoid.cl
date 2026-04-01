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

    const float x = src0[get_global_id(0)];
    dst[get_global_id(0)] = 1.0f / (1.0f + exp(-x));
}

kernel void kernel_sigmoid_f16(
        global half * src0,
        ulong offset0,
        global half * dst,
        ulong offsetd
) {
    src0 = (global half*)((global char*)src0 + offset0);
    dst = (global half*)((global char*)dst + offsetd);

#if GGML_OPENCL_USE_NATIVE_FP16_MATH
    dst[get_global_id(0)] = 1.0h / (1.0h + exp(-src0[get_global_id(0)]));
#else
    const float x = (float) src0[get_global_id(0)];
    dst[get_global_id(0)] = (half) (1.0f / (1.0f + exp(-x)));
#endif
}
