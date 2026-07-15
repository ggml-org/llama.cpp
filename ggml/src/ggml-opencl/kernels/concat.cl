kernel void kernel_concat_f32(
    global  const char * src0,
    ulong                offset0,
    global  const char * src1,
    ulong                offset1,
    global        char * dst,
    ulong                offsetd,
    int             ne00,
    int             ne01,
    int             ne02,
    int             ne03,
    ulong           nb00,
    ulong           nb01,
    ulong           nb02,
    ulong           nb03,
    ulong           nb10,
    ulong           nb11,
    ulong           nb12,
    ulong           nb13,
    int             ne0,
    ulong           nb0,
    ulong           nb1,
    ulong           nb2,
    ulong           nb3,
    int             dim
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    const int i3 = get_group_id(2);
    const int i2 = get_group_id(1);
    const int i1 = get_group_id(0);

    int o[4] = {0, 0, 0, 0};
    o[dim] = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));

    global const float * x;

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (global const float *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00);
        } else {
            x = (global const float *)(src1 + (i3 - o[3])*nb13 + (i2 - o[2])*nb12 + (i1 - o[1])*nb11 + (i0 - o[0])*nb10);
        }

        global float * y = (global float *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}

kernel void kernel_concat_f32_pack(
    global  const char * src0,
    ulong                offset0,
    global  const char * src1,
    ulong                offset1,
    global        char * dst,
    ulong                offsetd,
    int             ne00,
    int             ne01,
    int             ne02,
    int             ne03,
    ulong           nb00,
    ulong           nb01,
    ulong           nb02,
    ulong           nb03,
    ulong           nb10,
    ulong           nb11,
    ulong           nb12,
    ulong           nb13,
    int             ne0,
    ulong           nb0,
    ulong           nb1,
    ulong           nb2,
    ulong           nb3,
    int             dim,
    int             ne1,
    int             ne2,
    int             ne3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int lsz = get_local_size(0);
    int tpr = min(ne0, lsz);          // threads per row
    int rpw = lsz / tpr;              // rows per workgroup
    int lid = get_local_id(0);
    int row = get_group_id(0)*rpw + lid / tpr;
    int lane = lid - (lid / tpr) * tpr;

    int nrows = ne1*ne2*ne3;
    if (row >= nrows) {
        return;
    }

    int i1 = row % ne1;
    int t  = row / ne1;
    int i2 = t % ne2;
    int i3 = t / ne2;

    int o[4] = {0, 0, 0, 0};
    o[dim] = dim == 0 ? ne00 : (dim == 1 ? ne01 : (dim == 2 ? ne02 : ne03));

    for (int i0 = lane; i0 < ne0; i0 += tpr) {
        global const float * x;
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (global const float *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00);
        } else {
            x = (global const float *)(src1 + (i3 - o[3])*nb13 + (i2 - o[2])*nb12 + (i1 - o[1])*nb11 + (i0 - o[0])*nb10);
        }

        global float * y = (global float *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}
