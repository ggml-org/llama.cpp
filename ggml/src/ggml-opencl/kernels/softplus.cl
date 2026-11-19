#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// softplus
//------------------------------------------------------------------------------

kernel void kernel_softplus_f32(
        global const float * src0,
        ulong                offset0,
        global       float * dst,
        ulong                offsetd
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst  = (global float*)((global char*)dst + offsetd);

    dst[get_global_id(0)] = (src0[get_global_id(0)] > 20.0f) ? src0[get_global_id(0)] : log(1.0f + exp(src0[get_global_id(0)]));
}

kernel void kernel_softplus_f32_4(
        global const float4 * src0,
        ulong                 offset0,
        global       float4 * dst,
        ulong                 offsetd
) {
    src0 = (global float4*)((global char*)src0 + offset0);
    dst  = (global float4*)((global char*)dst + offsetd);

    dst[get_global_id(0)] = (src0[get_global_id(0)] > 20.0f) ? src0[get_global_id(0)] : log(1.0f + exp(src0[get_global_id(0)]));
}

kernel void kernel_softplus_f16(
        global const half * src0,
        ulong               offset0,
        global       half * dst,
        ulong               offsetd
) {
    src0 = (global half*)((global char*)src0 + offset0);
    dst  = (global half*)((global char*)dst + offsetd);

    const float x = convert_float(src0[get_global_id(0)]);
    dst[get_global_id(0)] = convert_half_rte((x > 20.0f) ? x : log(1.0f + exp(x)));
}

kernel void kernel_softplus_f16_4(
        global const half4 * src0,
        ulong                offset0,
        global       half4 * dst,
        ulong                offsetd
) {
    src0 = (global half4*)((global char*)src0 + offset0);
    dst  = (global half4*)((global char*)dst + offsetd);

    const float4 x = convert_float4(src0[get_global_id(0)]);
    dst[get_global_id(0)] = convert_half4_rte((x > 20.0f) ? x : log(1.0f + exp(x)));
}

kernel void kernel_softplus_f32_nc(
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

        *y = (*x > 20.0f) ? *x : log(1.0f + exp(*x));
    }
}

kernel void kernel_softplus_f16_nc(
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
        global const half * hx = (global const half *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
        global       half * hy = (global       half *)(dst  + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

        const float x = convert_float(*hx);
        *hy = convert_half_rte((x > 20.0f) ? x : log(1.0f + exp(x)));
    }
}
