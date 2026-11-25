#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void kernel_scale_f32(
        global const float * src0,
        ulong                offset0,
        global       float * dst,
        ulong                offsetd,
        float                scale,
        float                bias
) {
    src0 = (global const float *)((global const char *)src0 + offset0);
    dst  = (global       float *)((global       char *)dst  + offsetd);
    const size_t i = get_global_id(0);
    dst[i] = src0[i] * scale + bias;
}

kernel void kernel_scale_f32_4(
        global const float4 * src0,
        ulong                 offset0,
        global       float4 * dst,
        ulong                 offsetd,
        float                 scale,
        float                 bias
) {
    src0 = (global const float4 *)((global const char *)src0 + offset0);
    dst  = (global       float4 *)((global       char *)dst  + offsetd);
    const size_t i = get_global_id(0);
    dst[i] = src0[i] * scale + bias;
}

kernel void kernel_scale_f16(
        global const half * src0,
        ulong               offset0,
        global       half * dst,
        ulong               offsetd,
        float               scale,
        float               bias
) {
    src0 = (global const half *)((global const char *)src0 + offset0);
    dst  = (global       half *)((global       char *)dst  + offsetd);
    const size_t i = get_global_id(0);
    dst[i] = convert_half(convert_float(src0[i]) * scale + bias);
}

kernel void kernel_scale_f16_4(
        global const half4 * src0,
        ulong                offset0,
        global       half4 * dst,
        ulong                offsetd,
        float                scale,
        float                bias
) {
    src0 = (global const half4 *)((global const char *)src0 + offset0);
    dst  = (global       half4 *)((global       char *)dst  + offsetd);
    const size_t i = get_global_id(0);
    dst[i] = convert_half4(convert_float4(src0[i]) * scale + bias);
}

kernel void kernel_scale_f32_nc(
        global const char * src0,
        ulong               offset0,
        global       char * dst,
        ulong               offsetd,
        float               scale,
        float               bias,
        ulong ne00,
        ulong nb00, ulong nb01, ulong nb02, ulong nb03,
        ulong nb0,  ulong nb1,  ulong nb2,  ulong nb3
) {
    const size_t i1 = get_group_id(0);
    const size_t i2 = get_group_id(1);
    const size_t i3 = get_group_id(2);
    src0 += offset0 + i1*nb01 + i2*nb02 + i3*nb03;
    dst  += offsetd + i1*nb1  + i2*nb2  + i3*nb3;

    for (size_t i0 = get_local_id(0); i0 < ne00; i0 += get_local_size(0)) {
        global const float * x = (global const float *)(src0 + i0*nb00);
        global       float * y = (global       float *)(dst  + i0*nb0);
        *y = *x * scale + bias;
    }
}

kernel void kernel_scale_f16_nc(
        global const char * src0,
        ulong               offset0,
        global       char * dst,
        ulong               offsetd,
        float               scale,
        float               bias,
        ulong ne00,
        ulong nb00, ulong nb01, ulong nb02, ulong nb03,
        ulong nb0,  ulong nb1,  ulong nb2,  ulong nb3
) {
    const size_t i1 = get_group_id(0);
    const size_t i2 = get_group_id(1);
    const size_t i3 = get_group_id(2);
    src0 += offset0 + i1*nb01 + i2*nb02 + i3*nb03;
    dst  += offsetd + i1*nb1  + i2*nb2  + i3*nb3;

    for (size_t i0 = get_local_id(0); i0 < ne00; i0 += get_local_size(0)) {
        global const half * x = (global const half *)(src0 + i0*nb00);
        global       half * y = (global       half *)(dst  + i0*nb0);
        *y = convert_half(convert_float(*x) * scale + bias);
    }
}
