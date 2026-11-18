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
    dst[get_global_id(0)] = x / (1.0f + exp(-x));
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
    dst[get_global_id(0)] = x / (1.0f + exp(-x));
}

kernel void kernel_silu_nc(
        global const char * src0,
        ulong               offset0,
        global       char * dst,
        ulong               offsetd,
        int   ne00,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = src0 + offset0;
    dst  = dst + offsetd;

    const int i3 = get_group_id(2);
    const int i2 = get_group_id(1);
    const int i1 = get_group_id(0);

    for (int i0 = get_local_id(0); i0 < ne00; i0 += get_local_size(0)) {
        global const float * x = (global const float *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
        global       float * y = (global       float *)(dst  + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

        *y = *x / (1.0f + exp(-*x));
    }
}

kernel void kernel_silu_f16(
        global half * src0,
        ulong offset0,
        global half * dst,
        ulong offsetd
) {
    src0 = (global half*)((global char*)src0 + offset0);
    dst = (global half*)((global char*)dst + offsetd);

    float x = src0[get_global_id(0)];
    dst[get_global_id(0)] = x / (1.0f + exp(-x));
}

kernel void kernel_silu_f16_4(
        global half4 * src0,
        ulong offset0,
        global half4 * dst,
        ulong offsetd
) {
    src0 = (global half4*)((global char*)src0 + offset0);
    dst = (global half4*)((global char*)dst + offsetd);

    float4 x = convert_float4(src0[get_global_id(0)]);
    dst[get_global_id(0)] = convert_half4(x / (1.0f + exp(-x)));
}

kernel void kernel_silu_f16_nc(
        global const char * src0,
        ulong               offset0,
        global       char * dst,
        ulong               offsetd,
        int   ne00,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = src0 + offset0;
    dst  = dst + offsetd;

    const int i3 = get_group_id(2);
    const int i2 = get_group_id(1);
    const int i1 = get_group_id(0);

    for (int i0 = get_local_id(0); i0 < ne00; i0 += get_local_size(0)) {
        float x = *(global const half *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);

        *(global half *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0) = x / (1.0f + exp(-x));
    }
}
