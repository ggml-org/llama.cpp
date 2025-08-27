
kernel void kernel_sum_rows_f32(
    global float *  src0,
    ulong           offset0,
    global float *  dst,
    ulong           offsetd,
    int             ne00,
    int             ne01,
    int             ne02,
    int             ne03,
    ulong           nb01,
    ulong           nb02,
    ulong           nb03,
    ulong           nb1,
    ulong           nb2,
    ulong           nb3
) {
    src0 = (global float *)((global char *)src0 + offset0);
    dst  = (global float *)((global char *)dst  + offsetd);

    int i3 = get_global_id(2);
    int i2 = get_global_id(1);
    int i1 = get_global_id(0);

    if (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        return;
    }

    global float * src_row = (global float *) ((global char *) src0 + i1*nb01 + i2*nb02 + i3*nb03);
    global float * dst_row = (global float *) ((global char *) dst  + i1*nb1  + i2*nb2  + i3*nb3);

    float row_sum = 0;

    for (int i0 = 0; i0 < ne00; i0++) {
        row_sum += src_row[i0];
    }

    dst_row[0] = row_sum;
}
