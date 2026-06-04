#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// ifairy_split
//------------------------------------------------------------------------------

static inline float ifairy_bf16_to_f32(ushort x) {
    return as_float(((uint) x) << 16);
}

kernel void kernel_ifairy_split(
        global char * src0,
        ulong offset0,
        global char * dst,
        ulong offsetd,
        int n_dims,
        int ne1,
        int ne2,
        int ne3,
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
    dst  = dst  + offsetd;

    const int i1 = get_group_id(0);
    const int i2 = get_group_id(1);
    const int i3 = get_group_id(2);

    if (i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    global char * src_row = src0 + i3 * nb03 + i2 * nb02 + i1 * nb01;
    global char * dst_row = dst  + i3 * nb3  + i2 * nb2  + i1 * nb1;

    for (int i0 = get_local_id(0); i0 < n_dims; i0 += get_local_size(0)) {
        const uint v = *((global uint *) (src_row + i0 * nb00));

        const float real_v = ifairy_bf16_to_f32((ushort) (v & 0xFFFFu));
        const float imag_v = ifairy_bf16_to_f32((ushort) (v >> 16));

        *((global float *) (dst_row + i0 * nb0))          = real_v;
        *((global float *) (dst_row + (i0 + n_dims) * nb0)) = imag_v;
    }
}
