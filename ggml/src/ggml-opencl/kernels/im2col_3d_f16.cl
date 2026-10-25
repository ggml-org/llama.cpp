#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void kernel_im2col_3d_f16(
        global float * src1,
        ulong offset1,
        global half * dst,
        ulong offsetd,
        long IC,
        long ID,
        long IH,
        long IW,
        long KD,
        long KH,
        long KW,
        long OD,
        long OH,
        long OW,
        long N,
        long stride_x,
        long stride_y,
        long stride_z,
        long stride_q,
        int s0,
        int s1,
        int s2,
        int p0,
        int p1,
        int p2,
        int d0,
        int d1,
        int d2
) {
    const int ic = (int)IC;
    const int kd = (int)KD;
    const int kh = (int)KH;
    const int kw = (int)KW;
    const int od = (int)OD;
    const int oh = (int)OH;
    const int ow = (int)OW;
    const int n  = (int)N;

    const int KD_KH_KW    = kd * kh * kw;
    const int KH_KW       = kh * kw;
    const int IC_KD_KH_KW = ic * KD_KH_KW;

    const int i = (int)get_global_id(0);
    if (i >= IC_KD_KH_KW) {
        return;
    }

    const int iow = (int)get_global_id(1);
    if (iow >= ow) {
        return;
    }

    const int OD_OH = od * oh;
    const int iz = (int)get_global_id(2);
    if (iz >= n * OD_OH) {
        return;
    }

    src1 = (global float*)((global char*)src1 + offset1);
    dst  = (global half*)((global char*)dst + offsetd);

    const int iic = i / KD_KH_KW;
    const int ikd = (i - iic * KD_KH_KW) / KH_KW;
    const int ikh = (i - iic * KD_KH_KW - ikd * KH_KW) / kw;
    const int ikw = i % kw;

    const int in  = iz / OD_OH;
    const int iod = (iz - in * OD_OH) / oh;
    const int ioh = iz % oh;

    const int iiw = iow * s0 + ikw * d0 - p0;
    const int iih = ioh * s1 + ikh * d1 - p1;
    const int iid = iod * s2 + ikd * d2 - p2;

    const long offset_dst =
        (long)(iz * ow + iow) * (long)IC_KD_KH_KW +
        (long)(iic * KD_KH_KW + ikd * KH_KW + ikh * kw + ikw);

    if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
        dst[offset_dst] = 0.0f;
    } else {
        const long offset_src =
            (long)(in * ic + iic) * stride_q + (long)iid * stride_z +
            (long)iih * stride_y + (long)iiw * stride_x;
        dst[offset_dst] = convert_half_rte(src1[offset_src]);
    }
}
