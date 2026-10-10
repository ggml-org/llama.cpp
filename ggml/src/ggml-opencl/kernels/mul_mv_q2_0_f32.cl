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

#define QK2_0 64

// 00=-1, 01=0, 10=+1, 11=+2
typedef struct {
    half  d;
    uchar qs[QK2_0/4];
} block_q2_0;

#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define N_DST 4 // number of rows each SIMD group works on
#define N_SIMDGROUP 1 // number of SIMD groups in a thread group
#define N_SIMDWIDTH 16 // SIMD group size
#elif defined (ADRENO_GPU)
#define N_DST 4
#define N_SIMDGROUP 1
#define N_SIMDWIDTH 64
#endif

#undef  BLOCK_STRIDE
// 4 threads cover one block
#define BLOCK_STRIDE (N_SIMDWIDTH/4)

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q2_0_f32(
        global char * src0,
        int offset0,
        global char * src1,
        int offset1,
        global char * dst,
        int offsetd,
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
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int ix = get_sub_group_local_id()/4;  // block index
    int it = get_sub_group_local_id()%4;  // part of the block

    int nb = ne00/QK2_0;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);
    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    int i12 = im%ne12;
    int i13 = im/ne12;

    int offset_src0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    int offset_src1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    global block_q2_0 * x = (global block_q2_0 *) (src0 + offset_src0);
    global float   * y = (global float   *) (src1 + offset_src1);

    float yl[16];
    float sumf[N_DST] = {0.f};
    float all_sum;

    global float * y4 = y + ix * QK2_0 + 16 * it;

    for (int ib = ix; ib < nb; ib += BLOCK_STRIDE) {
        for (int i = 0; i < 16; ++i) {
            yl[i] = y4[i];
        }

        global char * xrow = (global char *)(x + ib);

        // only ne01 output rows exist; reading past them can pull in an inf scale
        for (int row = 0; row < N_DST && first_row + row < ne01; row++) {
            global block_q2_0 * xb = (global block_q2_0 *)(xrow + row*nb01);

            global uchar * q = xb->qs + 4*it;

            float acc = 0.f;
            for (int i = 0; i < 4; ++i) {
                uchar b = q[i];
                acc += yl[4*i+0] * (float)(((b >> 0) & 3) - 1);
                acc += yl[4*i+1] * (float)(((b >> 2) & 3) - 1);
                acc += yl[4*i+2] * (float)(((b >> 4) & 3) - 1);
                acc += yl[4*i+3] * (float)(((b >> 6) & 3) - 1);
            }

            sumf[row] += (float)xb->d * acc;
        }

        y4 += BLOCK_STRIDE * QK2_0;
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
