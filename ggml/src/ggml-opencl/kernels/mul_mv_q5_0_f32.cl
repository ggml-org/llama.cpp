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

#define QK5_0                   32

struct block_q5_0 {
    half d;
    uchar qh[4];
    uchar qs[QK5_0 / 2];
};

inline float block_q5_0_dot_y(
    global const struct block_q5_0 * qb_curr,
    float sumy,
    float16 yl,
    int il,
    global const float * yb
) {
    float d = qb_curr->d;

    float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    global const ushort * qs = ((global const ushort *)((global const uchar *) qb_curr + 6 + il));

    acc.s0 += yl.s0 * (qs[0] & 0x000F);
    acc.s0 += yl.s1 * (qs[0] & 0x0F00);
    acc.s0 += yl.s8 * (qs[0] & 0x00F0);
    acc.s3 += yl.s9 * (qs[0] & 0xF000);

    acc.s0 += yl.s2 * (qs[1] & 0x000F);
    acc.s1 += yl.s3 * (qs[1] & 0x0F00);
    acc.s2 += yl.sa * (qs[1] & 0x00F0);
    acc.s3 += yl.sb * (qs[1] & 0xF000);

    acc.s0 += yl.s4 * (qs[2] & 0x000F);
    acc.s1 += yl.s5 * (qs[2] & 0x0F00);
    acc.s2 += yl.sc * (qs[2] & 0x00F0);
    acc.s3 += yl.sd * (qs[2] & 0xF000);

    acc.s0 += yl.s6 * (qs[3] & 0x000F);
    acc.s1 += yl.s7 * (qs[3] & 0x0F00);
    acc.s2 += yl.se * (qs[3] & 0x00F0);
    acc.s3 += yl.sf * (qs[3] & 0xF000);

    uint qh_val = *((global const uint *)((global const uchar *) qb_curr + 2));
    uchar qh_lo = (uchar)((qh_val >> il) & 0xFF);
    uchar qh_hi = (uchar)((qh_val >> (il + 16)) & 0xFF);

    float qh_sum = 0.0f;
    qh_sum += yb[0]  * (float)((qh_lo >> 0) & 1);
    qh_sum += yb[1]  * (float)((qh_lo >> 1) & 1);
    qh_sum += yb[2]  * (float)((qh_lo >> 2) & 1);
    qh_sum += yb[3]  * (float)((qh_lo >> 3) & 1);
    qh_sum += yb[4]  * (float)((qh_lo >> 4) & 1);
    qh_sum += yb[5]  * (float)((qh_lo >> 5) & 1);
    qh_sum += yb[6]  * (float)((qh_lo >> 6) & 1);
    qh_sum += yb[7]  * (float)((qh_lo >> 7) & 1);
    qh_sum += yb[16] * (float)((qh_hi >> 0) & 1);
    qh_sum += yb[17] * (float)((qh_hi >> 1) & 1);
    qh_sum += yb[18] * (float)((qh_hi >> 2) & 1);
    qh_sum += yb[19] * (float)((qh_hi >> 3) & 1);
    qh_sum += yb[20] * (float)((qh_hi >> 4) & 1);
    qh_sum += yb[21] * (float)((qh_hi >> 5) & 1);
    qh_sum += yb[22] * (float)((qh_hi >> 6) & 1);
    qh_sum += yb[23] * (float)((qh_hi >> 7) & 1);

    return d * (acc.s0 + acc.s1 + acc.s2 + acc.s3 + 16.0f * qh_sum - 16.0f * sumy);
}

#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define N_DST 4 // each subgroup works on 4 rows
#define N_SIMDGROUP 1 // number of subgroups in a thread group
#define N_SIMDWIDTH 16 // assuming subgroup size is 16
#elif defined (ADRENO_GPU)
#define N_DST 4
#define N_SIMDGROUP 1
#define N_SIMDWIDTH 64
#endif

inline void mul_vec_q_n_f32(
        global void * src0,
        global float * src1,
        global float * dst,
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
    const ulong nb = ne00/QK5_0;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset0 = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    global struct block_q5_0 * x = (global struct block_q5_0 *) src0 + offset0;
    global float             * y = (global float             *) src1 + r1*ne10 + im*ne00*ne1;

    float16 yl;
    float4 sumf = (float4)(0.f, 0.f, 0.f, 0.f);

    int ix = get_sub_group_local_id()/2;
    int il = 8*(get_sub_group_local_id()%2);

    global float * yb = y + ix * QK5_0 + il;

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/2) {
        float sumy = 0;

        sumy += yb[0];
        sumy += yb[1];
        sumy += yb[2];
        sumy += yb[3];
        sumy += yb[4];
        sumy += yb[5];
        sumy += yb[6];
        sumy += yb[7];

        sumy += yb[16];
        sumy += yb[17];
        sumy += yb[18];
        sumy += yb[19];
        sumy += yb[20];
        sumy += yb[21];
        sumy += yb[22];
        sumy += yb[23];


        yl.s0 = yb[0];
        yl.s1 = yb[1]/256.f;

        yl.s2 = yb[2];
        yl.s3 = yb[3]/256.f;

        yl.s4 = yb[4];
        yl.s5 = yb[5]/256.f;

        yl.s6 = yb[6];
        yl.s7 = yb[7]/256.f;

        yl.s8 = yb[16]/16.f;
        yl.s9 = yb[17]/4096.f;

        yl.sa = yb[18]/16.f;
        yl.sb = yb[19]/4096.f;

        yl.sc = yb[20]/16.f;
        yl.sd = yb[21]/4096.f;

        yl.se = yb[22]/16.f;
        yl.sf = yb[23]/4096.f;

        sumf.s0 += block_q5_0_dot_y(x+ib+0*nb, sumy, yl, il, yb);
        sumf.s1 += block_q5_0_dot_y(x+ib+1*nb, sumy, yl, il, yb);
        sumf.s2 += block_q5_0_dot_y(x+ib+2*nb, sumy, yl, il, yb);
        sumf.s3 += block_q5_0_dot_y(x+ib+3*nb, sumy, yl, il, yb);

        yb += QK5_0 * (N_SIMDWIDTH/2);
    }

    float4 tot = (float4)(
        sub_group_reduce_add(sumf.s0), sub_group_reduce_add(sumf.s1),
        sub_group_reduce_add(sumf.s2), sub_group_reduce_add(sumf.s3)
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

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q5_0_f32(
        global void * src0,
        ulong offset0,
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
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    mul_vec_q_n_f32(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3);
}
