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
// block_q4_K
//------------------------------------------------------------------------------
#define QK_K            256
#define K_SCALE_SIZE    12

// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
typedef struct {
    half d;    // super-block scale for quantized scales
    half dmin; // super-block scale for quantized mins

    uchar scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uchar qs[QK_K/2];           // 4-bit quants
} block_q4_K;

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
// number of (super) blocks each subgroup processes
// each thread in a subgroup processes a block (32 weights)
#define BLOCK_STRIDE (N_SIMDWIDTH/8)

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q4_K_f32(
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

    ushort kmask1 = 0x3f3f;
    ushort kmask2 = 0x0f0f;
    ushort kmask3 = 0xc0c0;

    int ix = get_sub_group_local_id()/8;  // super block index
    int it = get_sub_group_local_id()%8;  // block index (inside super block)
    int iq = it/4;     // 0 or 1 - first or second half of the super block
    int ir = it%4;     // 0...3 - block index in the half super block

    int nb = ne00/QK_K;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);
    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    int i12 = im%ne12;
    int i13 = im/ne12;

    int offset_src0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    int offset_src1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    global block_q4_K * x = (global block_q4_K *) (src0 + offset_src0);
    global float      * y = (global float      *) (src1 + offset_src1);

    float yl[16];
    float yh[16];
    float sumf[N_DST] = {0.f};
    float all_sum;

    global float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    ushort  sc16[4];
    uchar * sc8 = (uchar *)sc16;

    for (int ib = ix; ib < nb; ib += BLOCK_STRIDE) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (int i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+0];
            sumy.s0 += yl[i+0];

            yl[i+8] = y4[i+32];
            sumy.s1 += yl[i+8];

            yh[i+0] = y4[i+128];
            sumy.s2 += yh[i+0];

            yh[i+8] = y4[i+160];
            sumy.s3 += yh[i+8];
        }

        global ushort * sc = (global ushort *)x[ib].scales + iq;
        global ushort * q1 = (global ushort *)x[ib].qs + 16 * iq + 4 * ir;
        global half     * dh = &x[ib].d;

        for (int row = 0; row < N_DST; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            global ushort * q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            for (int i = 0; i < 8; i += 2) {
                acc1.s0 += yl[i+0] * (q1[i/2] & 0x000F);
                acc1.s1 += yl[i+1] * (q1[i/2] & 0x0F00);
                acc1.s2 += yl[i+8] * (q1[i/2] & 0x00F0);
                acc1.s3 += yl[i+9] * (q1[i/2] & 0xF000);
                acc2.s0 += yh[i+0] * (q2[i/2] & 0x000F);
                acc2.s1 += yh[i+1] * (q2[i/2] & 0x0F00);
                acc2.s2 += yh[i+8] * (q2[i/2] & 0x00F0);
                acc2.s3 += yh[i+9] * (q2[i/2] & 0xF000);
            }

            float dall = dh[0];
            float dmin = dh[1];
            sumf[row] += dall * ((acc1.s0 + 1.f/256.f * acc1.s1) * sc8[0] +
                                 (acc1.s2 + 1.f/256.f * acc1.s3) * sc8[1] * 1.f/16.f +
                                 (acc2.s0 + 1.f/256.f * acc2.s1) * sc8[4] +
                                 (acc2.s2 + 1.f/256.f * acc2.s3) * sc8[5] * 1.f/16.f) -
                         dmin * (sumy.s0 * sc8[2] + sumy.s1 * sc8[3] + sumy.s2 * sc8[6] + sumy.s3 * sc8[7]);

            q1 += nb01/2;
            sc += nb01/2;
            dh += nb01/2;
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
