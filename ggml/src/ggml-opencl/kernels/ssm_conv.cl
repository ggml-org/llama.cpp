kernel void kernel_ssm_conv_f32_f32(
    global char * src0,
    ulong         offset0,
    global char * src1,
    ulong         offset1,
    global char * dst,
    ulong         offsetd,
    ulong         nb00,
    ulong         nb01,
    ulong         nb02,
    int           ne10,
    ulong         nb11,
    ulong         nb0,
    ulong         nb1,
    ulong         nb2
){
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int ir = get_global_id(0);
    int i2 = get_global_id(1);
    int i3 = get_global_id(2);

    int nc  = ne10;

    global float * s = (global float *) (src0 + ir*nb01 + i2*nb00 + i3*nb02);
    global float * c = (global float *) (src1 + ir*nb11);
    global float * d = (global float *) (dst  + ir*nb0  + i2*nb1  + i3*nb2);

    float sumf = 0.0f;

    for (int i0 = 0; i0 < nc; ++i0) {
        sumf += s[i0] * c[i0];
    }

    d[0] = sumf;
}

kernel void kernel_ssm_conv_f32_f32_4(
    global char * src0,
    ulong         offset0,
    global char * src1,
    ulong         offset1,
    global char * dst,
    ulong         offsetd,
    ulong         nb00,
    ulong         nb01,
    ulong         nb02,
    int           ne10,
    ulong         nb11,
    ulong         nb0,
    ulong         nb1,
    ulong         nb2
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int ir = get_global_id(0);
    int i2 = get_global_id(1);
    int i3 = get_global_id(2);

    int nc = ne10;

    global float4 * s = (global float4 *) (src0 + ir*nb01 + i2*nb00 + i3*nb02);
    global float4 * c = (global float4 *) (src1 + ir*nb11);
    global float  * d = (global float  *) (dst  + ir*nb0  + i2*nb1  + i3*nb2);

    float sumf = 0.0f;

    for (int i0 = 0; i0 < nc/4; ++i0) {
        sumf += dot(s[i0], c[i0]);
    }

    d[0] = sumf;
}
