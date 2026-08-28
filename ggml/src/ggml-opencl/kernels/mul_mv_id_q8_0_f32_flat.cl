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

#define QK8_0 32
typedef struct {
    half d;       // delta
    char qs[QK8_0]; // quants
} block_q8_0;

#define NB_Q8_0 8

#ifdef INTEL_GPU
#define N_R0_Q8_0 4 // number of rows each subgroup works on
#define N_SG_Q8_0 2 // number of subgroups in a work group
#define N_SIMDWIDTH 16 // subgroup size
#elif defined (ADRENO_GPU)
#define N_R0_Q8_0 4
#define N_SG_Q8_0 2
#define N_SIMDWIDTH 64
#endif

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_id_q8_0_f32_flat(
    global char * src0_q,
    global half * src0_d,
    global char * src1,
    ulong         offset1,
    global char * src2,
    ulong         offset2,
    global char * dst,
    ulong         offsetd,
    int           ne00,
    int           ne01,
    ulong         nb01,
    ulong         nb02,
    int           ne11,
    int           ne12,
    ulong         nb11,
    ulong         nb12,
    int           ne20,
    int           ne21,
    ulong         nb21,
    int           ne0,
    int           ne1
) {
    src1 = (global char *)((global char *)src1 + offset1);
    src2 = (global char *)((global char *)src2 + offset2);
    dst  = (global char *)((global char *)dst  + offsetd);

    int iid1 = (int)get_group_id(2)/ne20;
    int idx  = (int)get_group_id(2)%ne20;

    int i02 = ((global int *) (src2 + iid1*nb21))[idx];

    int i11_ = idx % ne11;
    int i12_ = iid1;

    int i1 = idx;
    int i2 = i12_;

    // 34 == sizeof(block_q8_0)
    uint src0_off = i02*nb02;
    src0_off /= 34;

    global char * src0_q_cur = src0_q + src0_off*sizeof(char)*QK8_0;
    global half * src0_d_cur = src0_d + src0_off;
    global char * src1_cur   = src1 + i11_*nb11 + i12_*nb12;

    global char * dst_cur = dst + (i1*ne0 + i2*ne1*ne0)*sizeof(float);

    int nb = ne00/QK8_0;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);

    int first_row = (r0*N_SG_Q8_0 + get_sub_group_id()) * N_R0_Q8_0;

    ulong offset_src1 = r1*nb11;
    global float * y  = (global float *) (src1_cur + offset_src1);

    // pointers to src0 rows
    uint offset_src0_base = first_row*nb01;

    global char * ax0, * ax1, * ax2, * ax3;
    global half * ad0, * ad1, * ad2, * ad3;
    uint offset_src0;

    offset_src0 = offset_src0_base + 0*nb01;
    offset_src0 = offset_src0/34;
    ax0 = (global char *) ((global char *) src0_q_cur + offset_src0*sizeof(char)*QK8_0);
    ad0 = (global half *) ((global char *) src0_d_cur + offset_src0*sizeof(half));

    offset_src0 = offset_src0_base + 1*nb01;
    offset_src0 = offset_src0/34;
    ax1 = (global char *) ((global char *) src0_q_cur + offset_src0*sizeof(char)*QK8_0);
    ad1 = (global half *) ((global char *) src0_d_cur + offset_src0*sizeof(half));

    offset_src0 = offset_src0_base + 2*nb01;
    offset_src0 = offset_src0/34;
    ax2 = (global char *) ((global char *) src0_q_cur + offset_src0*sizeof(char)*QK8_0);
    ad2 = (global half *) ((global char *) src0_d_cur + offset_src0*sizeof(half));

    offset_src0 = offset_src0_base + 3*nb01;
    offset_src0 = offset_src0/34;
    ax3 = (global char *) ((global char *) src0_q_cur + offset_src0*sizeof(char)*QK8_0);
    ad3 = (global half *) ((global char *) src0_d_cur + offset_src0*sizeof(half));

    const short ix = get_sub_group_local_id()/4;
    const short il = get_sub_group_local_id()%4;

    global float * yb = y + ix*QK8_0 + il*NB_Q8_0;

    float8 yl;
    float8 qv;
    float4 sumf = 0.f;
    float  sumq = 0.f;
    global char * qs;

    // each thread handles NB_Q8_0 quants at a time
    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/4) {
        yl = vload8(0, yb);

        qs = ax0 + ib*sizeof(char)*QK8_0 + il*NB_Q8_0;
        qv = convert_float8(vload8(0, qs));
        sumq = 0;
        sumq += qv.s0*yl.s0;
        sumq += qv.s1*yl.s1;
        sumq += qv.s2*yl.s2;
        sumq += qv.s3*yl.s3;
        sumq += qv.s4*yl.s4;
        sumq += qv.s5*yl.s5;
        sumq += qv.s6*yl.s6;
        sumq += qv.s7*yl.s7;
        sumf.s0 += sumq*ad0[ib];

        qs = ax1 + ib*sizeof(char)*QK8_0 + il*NB_Q8_0;
        qv = convert_float8(vload8(0, qs));
        sumq = 0;
        sumq += qv.s0*yl.s0;
        sumq += qv.s1*yl.s1;
        sumq += qv.s2*yl.s2;
        sumq += qv.s3*yl.s3;
        sumq += qv.s4*yl.s4;
        sumq += qv.s5*yl.s5;
        sumq += qv.s6*yl.s6;
        sumq += qv.s7*yl.s7;
        sumf.s1 += sumq*ad1[ib];

        qs = ax2 + ib*sizeof(char)*QK8_0 + il*NB_Q8_0;
        qv = convert_float8(vload8(0, qs));
        sumq = 0;
        sumq += qv.s0*yl.s0;
        sumq += qv.s1*yl.s1;
        sumq += qv.s2*yl.s2;
        sumq += qv.s3*yl.s3;
        sumq += qv.s4*yl.s4;
        sumq += qv.s5*yl.s5;
        sumq += qv.s6*yl.s6;
        sumq += qv.s7*yl.s7;
        sumf.s2 += sumq*ad2[ib];

        qs = ax3 + ib*sizeof(char)*QK8_0 + il*NB_Q8_0;
        qv = convert_float8(vload8(0, qs));
        sumq = 0;
        sumq += qv.s0*yl.s0;
        sumq += qv.s1*yl.s1;
        sumq += qv.s2*yl.s2;
        sumq += qv.s3*yl.s3;
        sumq += qv.s4*yl.s4;
        sumq += qv.s5*yl.s5;
        sumq += qv.s6*yl.s6;
        sumq += qv.s7*yl.s7;
        sumf.s3 += sumq*ad3[ib];

        yb += N_SIMDWIDTH*NB_Q8_0;
    }

    global float * dst_f32 = (global float *) dst_cur + (ulong)r1*ne0;

    float4 tot = (float4)(
        sub_group_reduce_add(sumf.s0),
        sub_group_reduce_add(sumf.s1),
        sub_group_reduce_add(sumf.s2),
        sub_group_reduce_add(sumf.s3)
    );

    if (get_sub_group_local_id() == 0) {
        if (first_row + 0 < ne01) {
            dst_f32[first_row + 0] = tot.s0;
        }
        if (first_row + 1 < ne01) {
            dst_f32[first_row + 1] = tot.s1;
        }
        if (first_row + 2 < ne01) {
            dst_f32[first_row + 2] = tot.s2;
        }
        if (first_row + 3 < ne01) {
            dst_f32[first_row + 3] = tot.s3;
        }
    }
}
