kernel void kernel_pad(
        global void * src0,
        ulong offset0,
        global void * dst,
        ulong offsetd,
        int ne00, int ne01, int ne02, int ne03,
        ulong nb00, ulong nb01, ulong nb02, ulong nb03,
        int ne0, int ne1, int ne2, int ne3,
        ulong nb0, ulong nb1, ulong nb2, ulong nb3,
        int lp0, int rp0,
        int lp1, int rp1,
        int lp2, int rp2,
        int lp3, int rp3
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst  = (global float*)((global char*)dst  + offsetd);

    int i0 = get_global_id(0);
    int i1 = get_group_id(1);
    int i2 = get_group_id(2) % ne2;
    int i3 = get_group_id(2) / ne2;

    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    uint src0_idx = (i3 - lp3)*nb03 + (i2 - lp2)*nb02 + (i1 - lp1)*nb01 + (i0 - lp0)*nb00;
    uint dst_idx  =         i3*nb3  +         i2*nb2  +         i1*nb1  +         i0*nb0;

    global float * src0_ptr = (global float *)((global char *)src0 + src0_idx);
    global float * dst_ptr  = (global float *)((global char *)dst  + dst_idx);

    bool in_src_bounds = (i0 >= lp0 && i0 < ne0 - rp0) &&
                         (i1 >= lp1 && i1 < ne1 - rp1) &&
                         (i2 >= lp2 && i2 < ne2 - rp2) &&
                         (i3 >= lp3 && i3 < ne3 - rp3);

    *dst_ptr = in_src_bounds ? *src0_ptr : 0.0f;
}
