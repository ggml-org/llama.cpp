static inline float ifairy_bf16_to_f32(ushort h) {
    return as_float(((uint) h) << 16);
}

static inline ushort ifairy_f32_to_bf16(float x) {
    uint bits = as_uint(x);
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        return (ushort) ((bits >> 16) | 64U);
    }
    return (ushort) ((bits + (0x7fffU + ((bits >> 16) & 1U))) >> 16);
}

static inline uint ifairy_add_pair(uint a, uint b) {
    const float ar = ifairy_bf16_to_f32((ushort) (a & 0xffffU));
    const float ai = ifairy_bf16_to_f32((ushort) (a >> 16));
    const float br = ifairy_bf16_to_f32((ushort) (b & 0xffffU));
    const float bi = ifairy_bf16_to_f32((ushort) (b >> 16));

    const uint r = (uint) ifairy_f32_to_bf16(ar + br);
    const uint i = (uint) ifairy_f32_to_bf16(ai + bi);
    return r | (i << 16);
}

/**
 * Adds two tensors whose elements are packed iFairy complex values:
 * low 16 bits are real BF16, high 16 bits are imaginary BF16.
 */
kernel void kernel_ifairy_add(
        global char * src0,
        ulong  offset0,
        global char * src1,
        ulong  offset1,
        global char * dst,
        ulong  offsetd,
        int   ne00,
        int   ne01,
        int   ne02,
        int   ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int   ne10,
        int   ne11,
        int   ne12,
        int   ne13,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int   ne0,
        int   ne1,
        int   ne2,
        int   ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst = dst + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int i13 = i03 % ne13;
    int i12 = i02 % ne12;
    int i11 = i01 % ne11;

    global char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    global char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    global char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1;

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const int i10 = i0 % ne10;
        const uint a = *((global uint *)(src0_ptr + i0*nb00));
        const uint b = *((global uint *)(src1_ptr + i10*nb10));
        *((global uint *)(dst_ptr + i0*nb0)) = ifairy_add_pair(a, b);
    }
}
