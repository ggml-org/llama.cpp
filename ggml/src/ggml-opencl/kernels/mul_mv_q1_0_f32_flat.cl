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

#define QK1_0 128
#define QK1_0_BYTES (QK1_0/8)              // 16 quant bytes per block
#define QK1_0_BLK_BYTES (QK1_0_BYTES + 2)  // d + qs in original tensor = 18

#define NB_Q1_0 16 // quants handled per thread (two qs bytes)

#ifdef INTEL_GPU
#define N_R0_Q1_0 4 // number of rows each subgroup works on
#define N_SG_Q1_0 2 // number of subgroups in a work group
#define N_SIMDWIDTH 16 // subgroup size
#elif defined (ADRENO_GPU)
#define N_R0_Q1_0 4
#define N_SG_Q1_0 2
#define N_SIMDWIDTH 64
#endif

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q1_0_f32_flat(
    global char * src0_q,
    global half * src0_d,
    global char * src1,
    ulong         offset1,
    global char * dst,
    ulong         offsetd,
    int           ne00,
    int           ne01,
    ulong         nb01,
    ulong         nb02,
    ulong         nb03,
    int           ne12,
    ulong         nb11,
    ulong         nb12,
    ulong         nb13,
    int           ne0,
    int           ne1,
    int           r2,
    int           r3
) {
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    int nb = ne00/QK1_0;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int first_row = (r0*N_SG_Q1_0 + get_sub_group_id()) * N_R0_Q1_0;

    uint i12 = im%ne12;
    uint i13 = im/ne12;

    ulong offset_src1 = r1*nb11 + i12*nb12 + i13*nb13;
    global float * y  = (global float *) (src1 + offset_src1);

    // pointers to src0 rows (flat: q bytes + scales)
    uint offset_src0_base = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;

    global uchar * ax0, * ax1, * ax2, * ax3;
    global half  * ad0, * ad1, * ad2, * ad3;
    uint offset_src0;

    offset_src0 = (offset_src0_base + 0*nb01) / QK1_0_BLK_BYTES;
    ax0 = (global uchar *) ((global char *) src0_q + offset_src0*QK1_0_BYTES);
    ad0 = (global half  *) ((global char *) src0_d + offset_src0*sizeof(half));

    offset_src0 = (offset_src0_base + 1*nb01) / QK1_0_BLK_BYTES;
    ax1 = (global uchar *) ((global char *) src0_q + offset_src0*QK1_0_BYTES);
    ad1 = (global half  *) ((global char *) src0_d + offset_src0*sizeof(half));

    offset_src0 = (offset_src0_base + 2*nb01) / QK1_0_BLK_BYTES;
    ax2 = (global uchar *) ((global char *) src0_q + offset_src0*QK1_0_BYTES);
    ad2 = (global half  *) ((global char *) src0_d + offset_src0*sizeof(half));

    offset_src0 = (offset_src0_base + 3*nb01) / QK1_0_BLK_BYTES;
    ax3 = (global uchar *) ((global char *) src0_q + offset_src0*QK1_0_BYTES);
    ad3 = (global half  *) ((global char *) src0_d + offset_src0*sizeof(half));

    const short ix = get_sub_group_local_id()/8;
    const short il = get_sub_group_local_id()%8;

    global float * yb = y + ix*QK1_0 + il*NB_Q1_0;

    float8 yl_lo;
    float8 yl_hi;
    float4 sumf = 0.f;

    // each thread handles NB_Q1_0 = 16 quants (two qs bytes) at a time
    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/8) {
        yl_lo = vload8(0, yb);
        yl_hi = vload8(0, yb + 8);
        float sumy = yl_lo.s0 + yl_lo.s1 + yl_lo.s2 + yl_lo.s3
                   + yl_lo.s4 + yl_lo.s5 + yl_lo.s6 + yl_lo.s7
                   + yl_hi.s0 + yl_hi.s1 + yl_hi.s2 + yl_hi.s3
                   + yl_hi.s4 + yl_hi.s5 + yl_hi.s6 + yl_hi.s7;

        uint b0, b1;
        float acc;

        b0 = ax0[ib*QK1_0_BYTES + il*2 + 0];
        b1 = ax0[ib*QK1_0_BYTES + il*2 + 1];
        acc  = yl_lo.s0*(float)((b0 >> 0) & 1) + yl_lo.s1*(float)((b0 >> 1) & 1)
             + yl_lo.s2*(float)((b0 >> 2) & 1) + yl_lo.s3*(float)((b0 >> 3) & 1)
             + yl_lo.s4*(float)((b0 >> 4) & 1) + yl_lo.s5*(float)((b0 >> 5) & 1)
             + yl_lo.s6*(float)((b0 >> 6) & 1) + yl_lo.s7*(float)((b0 >> 7) & 1)
             + yl_hi.s0*(float)((b1 >> 0) & 1) + yl_hi.s1*(float)((b1 >> 1) & 1)
             + yl_hi.s2*(float)((b1 >> 2) & 1) + yl_hi.s3*(float)((b1 >> 3) & 1)
             + yl_hi.s4*(float)((b1 >> 4) & 1) + yl_hi.s5*(float)((b1 >> 5) & 1)
             + yl_hi.s6*(float)((b1 >> 6) & 1) + yl_hi.s7*(float)((b1 >> 7) & 1);
        sumf.s0 += (float)ad0[ib] * (2.0f*acc - sumy);

        b0 = ax1[ib*QK1_0_BYTES + il*2 + 0];
        b1 = ax1[ib*QK1_0_BYTES + il*2 + 1];
        acc  = yl_lo.s0*(float)((b0 >> 0) & 1) + yl_lo.s1*(float)((b0 >> 1) & 1)
             + yl_lo.s2*(float)((b0 >> 2) & 1) + yl_lo.s3*(float)((b0 >> 3) & 1)
             + yl_lo.s4*(float)((b0 >> 4) & 1) + yl_lo.s5*(float)((b0 >> 5) & 1)
             + yl_lo.s6*(float)((b0 >> 6) & 1) + yl_lo.s7*(float)((b0 >> 7) & 1)
             + yl_hi.s0*(float)((b1 >> 0) & 1) + yl_hi.s1*(float)((b1 >> 1) & 1)
             + yl_hi.s2*(float)((b1 >> 2) & 1) + yl_hi.s3*(float)((b1 >> 3) & 1)
             + yl_hi.s4*(float)((b1 >> 4) & 1) + yl_hi.s5*(float)((b1 >> 5) & 1)
             + yl_hi.s6*(float)((b1 >> 6) & 1) + yl_hi.s7*(float)((b1 >> 7) & 1);
        sumf.s1 += (float)ad1[ib] * (2.0f*acc - sumy);

        b0 = ax2[ib*QK1_0_BYTES + il*2 + 0];
        b1 = ax2[ib*QK1_0_BYTES + il*2 + 1];
        acc  = yl_lo.s0*(float)((b0 >> 0) & 1) + yl_lo.s1*(float)((b0 >> 1) & 1)
             + yl_lo.s2*(float)((b0 >> 2) & 1) + yl_lo.s3*(float)((b0 >> 3) & 1)
             + yl_lo.s4*(float)((b0 >> 4) & 1) + yl_lo.s5*(float)((b0 >> 5) & 1)
             + yl_lo.s6*(float)((b0 >> 6) & 1) + yl_lo.s7*(float)((b0 >> 7) & 1)
             + yl_hi.s0*(float)((b1 >> 0) & 1) + yl_hi.s1*(float)((b1 >> 1) & 1)
             + yl_hi.s2*(float)((b1 >> 2) & 1) + yl_hi.s3*(float)((b1 >> 3) & 1)
             + yl_hi.s4*(float)((b1 >> 4) & 1) + yl_hi.s5*(float)((b1 >> 5) & 1)
             + yl_hi.s6*(float)((b1 >> 6) & 1) + yl_hi.s7*(float)((b1 >> 7) & 1);
        sumf.s2 += (float)ad2[ib] * (2.0f*acc - sumy);

        b0 = ax3[ib*QK1_0_BYTES + il*2 + 0];
        b1 = ax3[ib*QK1_0_BYTES + il*2 + 1];
        acc  = yl_lo.s0*(float)((b0 >> 0) & 1) + yl_lo.s1*(float)((b0 >> 1) & 1)
             + yl_lo.s2*(float)((b0 >> 2) & 1) + yl_lo.s3*(float)((b0 >> 3) & 1)
             + yl_lo.s4*(float)((b0 >> 4) & 1) + yl_lo.s5*(float)((b0 >> 5) & 1)
             + yl_lo.s6*(float)((b0 >> 6) & 1) + yl_lo.s7*(float)((b0 >> 7) & 1)
             + yl_hi.s0*(float)((b1 >> 0) & 1) + yl_hi.s1*(float)((b1 >> 1) & 1)
             + yl_hi.s2*(float)((b1 >> 2) & 1) + yl_hi.s3*(float)((b1 >> 3) & 1)
             + yl_hi.s4*(float)((b1 >> 4) & 1) + yl_hi.s5*(float)((b1 >> 5) & 1)
             + yl_hi.s6*(float)((b1 >> 6) & 1) + yl_hi.s7*(float)((b1 >> 7) & 1);
        sumf.s3 += (float)ad3[ib] * (2.0f*acc - sumy);

        yb += N_SIMDWIDTH*NB_Q1_0;
    }

    global float * dst_f32 = (global float *) dst + (ulong)im*ne0*ne1 + (ulong)r1*ne0;

    float4 tot = (float4)(
        sub_group_reduce_add(sumf.s0),
        sub_group_reduce_add(sumf.s1),
        sub_group_reduce_add(sumf.s2),
        sub_group_reduce_add(sumf.s3)
    );

    if (get_sub_group_local_id() == 0) {
        if (first_row + 0 < ne01) dst_f32[first_row + 0] = tot.s0;
        if (first_row + 1 < ne01) dst_f32[first_row + 1] = tot.s1;
        if (first_row + 2 < ne01) dst_f32[first_row + 2] = tot.s2;
        if (first_row + 3 < ne01) dst_f32[first_row + 3] = tot.s3;
    }
}
