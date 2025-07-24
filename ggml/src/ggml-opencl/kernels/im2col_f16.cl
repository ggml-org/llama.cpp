#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void kernel_im2col_f16(
        global float * src1,
        ulong offset1,
        global half  * dst,
        ulong offsetd,
        ulong batch_offset,
        ulong delta_offset,
        long IW,
        long IH,
        long IC,
        long OW,
        long OH,
        long KW,
        long KH,
        long pelements,
        long CHW,
        int  s0,
        int  s1,
        int  p0,
        int  p1,
        int  d0,
        int  d1
) {
    long i = get_global_id(0);
    if (i >= pelements) {
        return;
    }

    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global half*)((global char*)dst + offsetd);

    long  ksize = OW * KH;
    long  kx = i / ksize;
    long  kd = kx * ksize;
    long  ky = (i - kd) / OW;
    long  ix = i % OW;

    long  oh = get_group_id(1);
    long  batch = get_group_id(2) / IC;
    long  ic = get_group_id(2) % IC;

    long iiw = ix * s0 + kx * d0 - p0;
    long iih = oh * s1 + ky * d1 - p1;

    long offset_dst =
        ((batch * OH + oh) * OW + ix) * CHW +
        (ic * (KW * KH) + ky * KW + kx);

    if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
        dst[offset_dst] = 0.0f;
    } else {
        long offset_src = ic * delta_offset + batch * batch_offset;
        dst[offset_dst] = src1[offset_src + iih * IW + iiw];
    }
}
