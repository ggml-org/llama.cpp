kernel void kernel_repeat_f32(
        global const char * src0,
        ulong               offset0,
        global       char * dst,
        ulong               offsetd,
        int     ne00,
        int     ne01,
        int     ne02,
        int     ne03,
        ulong   nb00,
        ulong   nb01,
        ulong   nb02,
        ulong   nb03,
        int     ne0,
        ulong   nb0,
        ulong   nb1,
        ulong   nb2,
        ulong   nb3
) {
    src0 = src0 + offset0;
    dst  = dst  + offsetd;

    const int i3 = get_group_id(2);
    const int i2 = get_group_id(1);
    const int i1 = get_group_id(0);

    const int i03 = i3%ne03;
    const int i02 = i2%ne02;
    const int i01 = i1%ne01;

    global const char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    global       char * dst_ptr  = dst  +  i3*nb3  +  i2*nb2  +  i1*nb1;

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const int i00 = i0%ne00;
        *((global float *)(dst_ptr + i0*nb0)) = *((global float *)(src0_ptr + i00*nb00));
    }
}
