#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// ifairy_merge
//------------------------------------------------------------------------------

static inline ushort ifairy_f32_to_bf16(float x) {
    const uint u = as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        // Quiet NaNs to match ggml_compute_fp32_to_bf16.
        return (ushort) ((u >> 16) | 64u);
    }
    return (ushort) ((u + (0x7fffu + ((u >> 16) & 1u))) >> 16);
}

static inline uint ifairy_pack_bf16_pair(float real_v, float imag_v) {
    const uint r = (uint) ifairy_f32_to_bf16(real_v);
    const uint i = (uint) ifairy_f32_to_bf16(imag_v);
    return r | (i << 16);
}

kernel void kernel_ifairy_merge(
        global char * src0,
        ulong offset0,
        global char * dst,
        ulong offsetd,
        int half_dims,
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

    for (int i0 = get_local_id(0); i0 < half_dims; i0 += get_local_size(0)) {
        const float real_v = *((global float *) (src_row + i0 * nb00));
        const float imag_v = *((global float *) (src_row + (i0 + half_dims) * nb00));

        *((global uint *) (dst_row + i0 * nb0)) = ifairy_pack_bf16_pair(real_v, imag_v);
    }
}
