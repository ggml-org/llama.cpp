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
    half   d;
    ushort qs[QK_K/8];
    uchar  scales[QK_K/32];
} block_iq2_xs;

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
kernel void kernel_mul_mv_iq2_xs_f32(
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

    global block_iq2_xs * x = (global block_iq2_xs *) (src0 + offset_src0);
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
            global block_iq2_xs * xb = (global block_iq2_xs *)(xrow + row*nb01);

            uchar scb = xb->scales[it];
            float dlo = (float)xb->d * (0.5f + (float)(scb & 0xf)) * 0.25f;
            float dhi = (float)xb->d * (0.5f + (float)(scb >>  4)) * 0.25f;

            global ushort * q16 = xb->qs + 4*it;

            float alo = 0.f, ahi = 0.f;
            for (int l = 0; l < 4; ++l) {
                ushort q  = q16[l];
                uchar  sg = iq2_xs_signs(q >> 9);
                uint   lo = iq2_xs_grid[2*(q & 511)+0];
                uint   hi = iq2_xs_grid[2*(q & 511)+1];
                float  a  = 0.f;
                for (int j = 0; j < 4; ++j) {
                    a += yl[8*l+j+0] * (float)((lo >> (8*j)) & 0xFF) * ((sg & (1 << j))     ? -1.f : 1.f);
                    a += yl[8*l+j+4] * (float)((hi >> (8*j)) & 0xFF) * ((sg & (1 << (j+4))) ? -1.f : 1.f);
                }
                if (l < 2) alo += a; else ahi += a;
            }

            sumf[row] += dlo * alo + dhi * ahi;
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
