#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

//------------------------------------------------------------------------------
// kernel_mul_mv_q6_K_f32_flat
//------------------------------------------------------------------------------
#define Q6_K_MASK1 0x03
#define Q6_K_MASK2 0x0C
#define Q6_K_MASK3 0x30
#define Q6_K_MASK4 0xC0

#define QK_K       256

inline float block_q_6_K_dot_y_flat(
    global uchar * blk_ql,
    global uchar * blk_qh,
    global char  * blk_scales,
    global half  * blk_d,
    global float * yy,
    int ib,
    int ip,
    int is,
    int l0
) {
    int y_offset   = 128*ip + l0;
    int q_offset_l =  64*ip + l0;
    int q_offset_h =  32*ip + l0;

    global uchar * q1 = blk_ql     + ib*128 + q_offset_l;
    global uchar * q2 = q1         + QK_K/8;
    global uchar * qh = blk_qh     + ib*64 + q_offset_h;
    global char  * sc = blk_scales + ib*16 + is;

    global float * y = yy + ib * QK_K + y_offset;

    float dall = blk_d[ib];

    float  sumf = 0;
    float4 sums = {0.f, 0.f, 0.f, 0.f};

    sums.s0 += y[0+ 0] * ((float)((q1[0] & 0xF) | ((qh[0] & Q6_K_MASK1) << 4)) - 32.f);
    sums.s1 += y[0+32] * ((float)((q2[0] & 0xF) | ((qh[0] & Q6_K_MASK2) << 2)) - 32.f);
    sums.s2 += y[0+64] * ((float)((q1[0]  >> 4) | ((qh[0] & Q6_K_MASK3) << 0)) - 32.f);
    sums.s3 += y[0+96] * ((float)((q2[0]  >> 4) | ((qh[0] & Q6_K_MASK4) >> 2)) - 32.f);

    sums.s0 += y[1+ 0] * ((float)((q1[1] & 0xF) | ((qh[1] & Q6_K_MASK1) << 4)) - 32.f);
    sums.s1 += y[1+32] * ((float)((q2[1] & 0xF) | ((qh[1] & Q6_K_MASK2) << 2)) - 32.f);
    sums.s2 += y[1+64] * ((float)((q1[1]  >> 4) | ((qh[1] & Q6_K_MASK3) << 0)) - 32.f);
    sums.s3 += y[1+96] * ((float)((q2[1]  >> 4) | ((qh[1] & Q6_K_MASK4) >> 2)) - 32.f);

    sums.s0 += y[2+ 0] * ((float)((q1[2] & 0xF) | ((qh[2] & Q6_K_MASK1) << 4)) - 32.f);
    sums.s1 += y[2+32] * ((float)((q2[2] & 0xF) | ((qh[2] & Q6_K_MASK2) << 2)) - 32.f);
    sums.s2 += y[2+64] * ((float)((q1[2]  >> 4) | ((qh[2] & Q6_K_MASK3) << 0)) - 32.f);
    sums.s3 += y[2+96] * ((float)((q2[2]  >> 4) | ((qh[2] & Q6_K_MASK4) >> 2)) - 32.f);

    sums.s0 += y[3+ 0] * ((float)((q1[3] & 0xF) | ((qh[3] & Q6_K_MASK1) << 4)) - 32.f);
    sums.s1 += y[3+32] * ((float)((q2[3] & 0xF) | ((qh[3] & Q6_K_MASK2) << 2)) - 32.f);
    sums.s2 += y[3+64] * ((float)((q1[3]  >> 4) | ((qh[3] & Q6_K_MASK3) << 0)) - 32.f);
    sums.s3 += y[3+96] * ((float)((q2[3]  >> 4) | ((qh[3] & Q6_K_MASK4) >> 2)) - 32.f);

    sumf += dall * (sums.s0 * sc[0] + sums.s1 * sc[2] + sums.s2 * sc[4] + sums.s3 * sc[6]);

    return sumf;
}

#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define N_DST 4
#define N_SIMDGROUP 2
#define N_SIMDWIDTH 16
#elif defined (ADRENO_GPU)
#define N_DST 4
#define N_SIMDGROUP 2
#define N_SIMDWIDTH 64
#endif

#define BLOCK_STRIDE (N_SIMDWIDTH/16) // number of blocks each subgroup processes

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q6_K_f32_flat(
        global uchar * src0_ql,
        global uchar * src0_qh,
        global char  * src0_s,
        global half  * src0_d,
        global float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int nb = ne00/QK_K;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int i12 = im%ne12;
    int i13 = im/ne12;

    int first_row = (N_SIMDGROUP * r0 + get_sub_group_id()) * N_DST;

    ulong offset_src0    = first_row*nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    ulong offset_src0_ql = offset_src0 * 128;
    ulong offset_src0_qh = offset_src0 * 64;
    ulong offset_src0_s  = offset_src0 * 16;
    ulong offset_src0_d  = offset_src0;

    global uchar * blk_ql     = (global uchar *) src0_ql + offset_src0_ql;
    global uchar * blk_qh     = (global uchar *) src0_qh + offset_src0_qh;
    global char  * blk_scales = (global char  *) src0_s  + offset_src0_s;
    global half  * blk_d      = (global half  *) src0_d  + offset_src0_d;
    global float * yy         = (global float *) src1    + r1*ne10 + im*ne00*ne1;

    int tid = get_sub_group_local_id()/BLOCK_STRIDE; // first block_stride groups have tid=0
    int ix  = get_sub_group_local_id()%BLOCK_STRIDE; // first block is 0..block_stride-1
    int ip  = tid/8;   // first or second half of (super) block (0 or 1)
    int il  = tid%8;   // each half has 8 parts, one per scale
    int n   = 4;       // 4 scales at a time (and 4 sums)
    int l0  = n*il;    // offset into half-block, 0..28
    int is  = 8*ip + l0/16; // 0, 1, 8, 9

    float4 sumf = 0;

    for (int ib = ix; ib < nb; ib += BLOCK_STRIDE) {
        if (first_row + 0 < ne01) {
            sumf.s0 += block_q_6_K_dot_y_flat(blk_ql + 0*nb*128, blk_qh + 0*nb*64, blk_scales + 0*nb*16, blk_d + 0*nb, yy, ib, ip, is, l0);
        }
        if (first_row + 1 < ne01) {
            sumf.s1 += block_q_6_K_dot_y_flat(blk_ql + 1*nb*128, blk_qh + 1*nb*64, blk_scales + 1*nb*16, blk_d + 1*nb, yy, ib, ip, is, l0);
        }
        if (first_row + 2 < ne01) {
            sumf.s2 += block_q_6_K_dot_y_flat(blk_ql + 2*nb*128, blk_qh + 2*nb*64, blk_scales + 2*nb*16, blk_d + 2*nb, yy, ib, ip, is, l0);
        }
        if (first_row + 3 < ne01) {
            sumf.s3 += block_q_6_K_dot_y_flat(blk_ql + 3*nb*128, blk_qh + 3*nb*64, blk_scales + 3*nb*16, blk_d + 3*nb, yy, ib, ip, is, l0);
        }
    }

    float4 tot = (float4)(
        sub_group_reduce_add(sumf.s0),
        sub_group_reduce_add(sumf.s1),
        sub_group_reduce_add(sumf.s2),
        sub_group_reduce_add(sumf.s3)
    );
    if (get_sub_group_local_id() == 0) {
        if (first_row + 0 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 0] = tot.s0;
        }
        if (first_row + 1 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 1] = tot.s1;
        }
        if (first_row + 2 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 2] = tot.s2;
        }
        if (first_row + 3 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 3] = tot.s3;
        }
    }
}
