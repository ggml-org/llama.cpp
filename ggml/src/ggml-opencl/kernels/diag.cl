kernel void kernel_diag_f32(
    global const char * src0,
    ulong               offset0,
    global       char * dst,
    ulong               offsetd,
    ulong               nb01,
    ulong               nb02,
    ulong               nb03,
    int                 ne0,
    ulong               nb0,
    ulong               nb2,
    ulong               nb3
) {
    src0 = src0 + offset0;
    dst  = dst + offsetd;

    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    global const float * src0_ptr = (global const float *)(src0 +           i2*nb02 + i3*nb03);
    global       float * dst_ptr  = (global       float *)(dst  + i1*nb01 + i2*nb2  + i3*nb3);

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        dst_ptr[i0] = i0 == i1 ? src0_ptr[i0] : 0.0f;
    }
}
