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
typedef struct {
    half d;
    uchar qs[QK1_0/8];
} block_q1_0;

#define NB_Q1_0 16

#ifdef INTEL_GPU
#define N_R0_Q1_0 4 // number of rows each subgroup works on
#define N_SG_Q1_0 2 // number of subgroups in a work group
#define N_SIMDWIDTH 16 // subgroup size
#elif defined (ADRENO_GPU)
#define N_R0_Q1_0 4
#define N_SG_Q1_0 2
#define N_SIMDWIDTH 64
#endif

inline float block_q_1_0_dot_y(global block_q1_0 * qb, float sumy, float yl[NB_Q1_0], short il) {
    global uchar * qs = qb->qs + il*2;
    uint b0 = qs[0];
    uint b1 = qs[1];

    float acc = 0.f;
    acc += yl[ 0]*(float)((b0 >> 0) & 1) + yl[ 1]*(float)((b0 >> 1) & 1);
    acc += yl[ 2]*(float)((b0 >> 2) & 1) + yl[ 3]*(float)((b0 >> 3) & 1);
    acc += yl[ 4]*(float)((b0 >> 4) & 1) + yl[ 5]*(float)((b0 >> 5) & 1);
    acc += yl[ 6]*(float)((b0 >> 6) & 1) + yl[ 7]*(float)((b0 >> 7) & 1);

    acc += yl[ 8]*(float)((b1 >> 0) & 1) + yl[ 9]*(float)((b1 >> 1) & 1);
    acc += yl[10]*(float)((b1 >> 2) & 1) + yl[11]*(float)((b1 >> 3) & 1);
    acc += yl[12]*(float)((b1 >> 4) & 1) + yl[13]*(float)((b1 >> 5) & 1);
    acc += yl[14]*(float)((b1 >> 6) & 1) + yl[15]*(float)((b1 >> 7) & 1);

    return qb->d * (2.0f*acc - sumy);
}

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q1_0_f32(
    global char * src0,
    ulong         offset0,
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
    src0 = (global char*)((global char*)src0 + offset0);
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

    // pointers to src0 rows
    global block_q1_0 * ax[N_R0_Q1_0];
    for (int row = 0; row < N_R0_Q1_0; ++row) {
        ulong offset_src0 = (first_row + row)*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
        ax[row] = (global block_q1_0 *) ((global char *) src0 + offset_src0);
    }

    float yl[NB_Q1_0];
    float sumf[N_R0_Q1_0] = { 0.f };

    const short ix = get_sub_group_local_id()/8;
    const short il = get_sub_group_local_id()%8;

    global float * yb = y + ix*QK1_0 + il*NB_Q1_0;

    // each thread handles NB_Q1_0 quants at a time
    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/8) {
        float sumy = 0.f;
        for (short i = 0; i < NB_Q1_0; ++i) {
            yl[i] = yb[i];
            sumy += yb[i];
        }

        for (short row = 0; row < N_R0_Q1_0; row++) {
            sumf[row] += block_q_1_0_dot_y(ax[row] + ib, sumy, yl, il);
        }

        yb += N_SIMDWIDTH*NB_Q1_0;
    }

    global float * dst_f32 = (global float *) dst + (ulong)im*ne0*ne1 + (ulong)r1*ne0;

    for (int row = 0; row < N_R0_Q1_0; ++row) {
        float tot = sub_group_reduce_add(sumf[row]);

        if (get_sub_group_local_id() == 0 && first_row + row < ne01) {
            dst_f32[first_row + row] = tot;
        }
    }
}
