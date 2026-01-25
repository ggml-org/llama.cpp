#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// tri
//------------------------------------------------------------------------------
__kernel void kernel_tri_f32(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int n,
        int ne0,
        int ne1,
        int tri_type
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int idx = get_global_id(0);
    if (idx >= n) return;

    int i0 = idx % ne0;
    int i1 = (idx / ne0) % ne1;

    int keep = 0;
    if (tri_type == 0) keep = (i0 >= i1);
    else if (tri_type == 1) keep = (i0 >  i1);
    else if (tri_type == 2) keep = (i0 <= i1);
    else                    keep = (i0 <  i1);

    dst[idx] = keep ? src0[idx] : 0.0f;
}
