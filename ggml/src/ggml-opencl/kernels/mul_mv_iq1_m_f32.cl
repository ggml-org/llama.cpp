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
#define IQ1_M_DELTA 0.125f

// no d field: the super block scale is 4 nibbles spread over scales[]
typedef struct {
    uchar qs[QK_K/8];
    uchar qh[QK_K/16];
    uchar scales[QK_K/32];
} block_iq1_m;

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
kernel void kernel_mul_mv_iq1_m_f32(
        global char * src0,
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
    src0 = src0 + offset0;
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

    ulong offset_src0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    ulong offset_src1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    global block_iq1_m * x = (global block_iq1_m *) (src0 + offset_src0);
    global float         * y = (global float         *) (src1 + offset_src1);

    float yl[32];
    float sumf[N_DST] = {0.f};
    float all_sum;

    global float * y4 = y + ix * QK_K + 32 * it;

    for (int ib = ix; ib < nb; ib += BLOCK_STRIDE) {
        for (int i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        global char * xrow = (global char *)(x + ib);

        // only ne01 output rows exist; reading past them can pull in an inf scale
        for (int row = 0; row < N_DST && first_row + row < ne01; row++) {
            global block_iq1_m * xb = (global block_iq1_m *)(xrow + row*nb01);

            global ushort * sc = (global ushort *)xb->scales;
            ushort u16 = (ushort)(((uint)sc[0] >> 12) | (((uint)sc[1] >> 8) & 0x00f0)
                                | (((uint)sc[2] >> 4) & 0x0f00) | ((uint)sc[3] & 0xf000));
            float dsuper = (float)as_half(u16);

            float acc = 0.f;
            for (int il = 0; il < 4; ++il) {
                int   ib16 = 2*it + il/2;
                float dl   = dsuper * (float)(2*(((uint)sc[ib16/4] >> (3*(ib16%4))) & 7) + 1);

                uchar qhb = xb->qh[2*it + il/2];
                float dlt = (qhb & (0x08 << (4*(il%2)))) ? (-1.f - IQ1_M_DELTA) : (-1.f + IQ1_M_DELTA);

                uint gi = (uint)xb->qs[4*it+il] | ((((uint)qhb >> (4*(il%2))) & 7) << 8);
                uint g  = iq1_s_grid[gi];

                float a = 0.f;
                for (int j = 0; j < 4; ++j) {
                    a += yl[8*il+j+0] * ((float)((g >> (8*j+0)) & 0xF) + dlt);
                    a += yl[8*il+j+4] * ((float)((g >> (8*j+4)) & 0xF) + dlt);
                }
                acc += dl * a;
            }

            sumf[row] += acc;
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
