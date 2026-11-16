#pragma OPENCL EXTENSION cl_khr_fp16 : enable

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

#define QK_K 256

typedef struct {
    half d;
    ushort scales_h;
    uchar scales_l[QK_K/64];
    uchar qs[QK_K/2];
} block_iq4_xs;

#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define N_DST 4
#define N_SIMDGROUP 1
#define N_SIMDWIDTH 16
#elif defined (ADRENO_GPU)
#define N_DST 4
#define N_SIMDGROUP 1
#define N_SIMDWIDTH 64
#endif

#undef  BLOCK_STRIDE
// 8 threads cover one super block, one 32 element sub block each
#define BLOCK_STRIDE (N_SIMDWIDTH/8)

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_iq4_xs_f32_flat(
        global uchar * src0_qs,
        global uchar * src0_sl,
        global ushort * src0_sh,
        global half * src0_d,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global char * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne12,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int ix = get_sub_group_local_id()/8;  // super block index
    int it = get_sub_group_local_id()%8;  // sub block inside the super block

    int nb = ne00/QK_K;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);
    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset_src0 = (offset0 + first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03)/sizeof(block_iq4_xs);
    ulong row_stride = nb01/sizeof(block_iq4_xs);
    ulong offset_src1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    global float         * y = (global float         *) (src1 + offset_src1);

    float yl[32];
    float sumf[N_DST] = {0.f};
    float all_sum;

    global float * y4 = y + ix * QK_K + 32 * it;

    for (int ib = ix; ib < nb; ib += BLOCK_STRIDE) {
        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        // only ne01 output rows exist; reading past them can pull in an inf scale
        for (int row = 0; row < N_DST && first_row + row < ne01; row++) {
            const ulong b = offset_src0 + ib + row*row_stride;

            int ls = ((src0_sl[4*b + it/2] >> (4*(it%2))) & 15) | (((src0_sh[b] >> (2*it)) & 3) << 4);
            float db = (float)src0_d[b] * (float)(ls - 32);
            global uchar * qs = src0_qs + 128*b + 16*it;

            float acc = 0.f;
            for (int j = 0; j < 16; ++j) {
                acc += yl[j]    * (float)iq4_xs_values[qs[j] & 15];
                acc += yl[j+16] * (float)iq4_xs_values[qs[j] >>  4];
            }

            sumf[row] += db * acc;
        }

        y4 += BLOCK_STRIDE * QK_K;
    }

    global float * dst_f32 = (global float *) dst + im*ne0*ne1 + r1*ne0;

    for (int row = 0; row < N_DST; ++row) {
        all_sum = sub_group_reduce_add(sumf[row]);
        if (first_row + row < ne01) {
            if (get_sub_group_local_id() == 0) {
                dst_f32[first_row + row] = all_sum;
            }
        }
    }
}
