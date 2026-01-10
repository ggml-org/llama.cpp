#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// cumsum
//------------------------------------------------------------------------------
kernel void kernel_cumsum_f32(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        int axis,
        int exclusive,
        int reverse,
        int lines
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    const int gid = get_global_id(0);

    int i0 = 0, i1 = 0, i2 = 0, i3 = 0;
    int t = gid;

    if (axis != 3) { i3 = t % ne3; t /= ne3; }
    if (axis != 2) { i2 = t % ne2; t /= ne2; }
    if (axis != 1) { i1 = t % ne1; t /= ne1; }
    if (axis != 0) { i0 = t % ne0; t /= ne0; }

    const int axis_len = (axis == 0 ? ne0 : axis == 1 ? ne1 : axis == 2 ? ne2 : ne3);

    float acc = 0.0f;

    for (int pos = 0; pos < axis_len; pos++) {
        const int a = reverse ? (axis_len - 1 - pos) : pos;

        int j0 = i0, j1 = i1, j2 = i2, j3 = i3;
        if (axis == 0) j0 = a;
        else if (axis == 1) j1 = a;
        else if (axis == 2) j2 = a;
        else j3 = a;

        int idx = j0 + ne0 * (j1 + ne1 * (j2 + ne2 * j3));

        if (exclusive) {
            dst[idx] = acc;
            acc += src0[idx];
        } else {
            acc += src0[idx];
            dst[idx] = acc;
        }
    }
}
