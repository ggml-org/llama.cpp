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

//------------------------------------------------------------------------------
// block_q3_K
//------------------------------------------------------------------------------
#define QK_K 256

// 16 blocks of 16 elements each
// weight is represented as x = a * q
typedef struct {
    uchar hmask[QK_K/8]; // quants - high bit
    uchar qs[QK_K/4];    // quants - low 2 bits
    uchar scales[12];    // scales, quantized with 6 bits
    half  d;             // super-block scale
} block_q3_K;

#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define N_DST 2 // number of rows each SIMD group works on
#define N_SIMDGROUP 1 // number of SIMD groups in a thread group
#define N_SIMDWIDTH 16 // SIMD group size
#elif defined (ADRENO_GPU)
#define N_DST 2
#define N_SIMDGROUP 1
#define N_SIMDWIDTH 64
#endif

#undef  BLOCK_STRIDE
// 8 threads cover one super block, so a wave covers N_SIMDWIDTH/8 of them
#define BLOCK_STRIDE (N_SIMDWIDTH/8)

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q3_K_f32(
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

    int nb = ne00/QK_K;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);
    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    int i12 = im%ne12;
    int i13 = im/ne12;

    int offset_src0 = first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    int offset_src1 =        r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    global block_q3_K * x  = (global block_q3_K *) (src0 + offset_src0);
    global float      * yy = (global float      *) (src1 + offset_src1);

    float yl[32];

    // ix picks the super block, tid the position inside it. Keep tid in 0..7 so
    // ip/il/ir stay in range when the sub group is wider than 32.
    int ix  = get_sub_group_local_id() % BLOCK_STRIDE;
    int tid = get_sub_group_local_id() / BLOCK_STRIDE;

    int ip = tid/4;         // 0 or 1
    int il = 2*((tid%4)/2); // 0 or 2
    int ir = tid%2;
    int l0 = 8*ir;

    // Masks for the high bit and for the low 2 bits. Both tables in the metal
    // kernel are one base vector shifted left, so shift instead of indexing: a
    // private array indexed at runtime spills to scratch on Adreno.
    ushort4 hm = (ushort4)(0x0001, 0x0100, 0x0002, 0x0200) << (ushort4)(2*(2*ip + il/2));
    int4    qm = (int4)   (0x0003, 0x0300, 0x000c, 0x0c00) << (int4)   (4*(il/2));

    int shift = 2*il;

    float v1 = il == 0 ? 4.f : 64.f;
    float v2 = 4.f * v1;

    int s_shift1 = 4*ip;
    int s_shift2 = s_shift1 + il;

    int q_offset = 32*ip + l0;
    int y_offset = 128*ip + 32*il + l0;

    global float * y1 = yy + ix*QK_K + y_offset;

    float sumf1[N_DST] = {0.f};
    float sumf2[N_DST] = {0.f};
    float all_sum;

    for (int i = ix; i < nb; i += BLOCK_STRIDE) {
        for (int l = 0; l < 8; ++l) {
            yl[l+ 0] = y1[l+ 0];
            yl[l+ 8] = y1[l+16];
            yl[l+16] = y1[l+32];
            yl[l+24] = y1[l+48];
        }

        global ushort * q = (global ushort *)(x[i].qs + q_offset);
        global ushort * h = (global ushort *)(x[i].hmask + l0);
        global ushort * a = (global ushort *)(x[i].scales);
        global half   * dh = &x[i].d;

        // only ne01 output rows exist; reading past them can pull in an inf scale
        for (int row = 0; row < N_DST && first_row + row < ne01; ++row) {
            float d_all = dh[0];

            uint s32   = (uint)a[4] | ((uint)a[5] << 16);
            uint aux32 = ((s32 >> s_shift2) << 4) & 0x30303030u;
            s32 = (uint)a[il+0] | ((uint)a[il+1] << 16);
            s32 = ((s32 >> s_shift1) & 0x0f0f0f0fu) | aux32;
            char4 sc = as_char4(s32);

            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
            for (int l = 0; l < 8; l += 2) {
                int qs = q[l/2];
                s1 += yl[l+0] * (qs & qm.s0);
                s2 += yl[l+1] * (qs & qm.s1);
                s3 += ((h[l/2] & hm.s0) ? 0.f : yl[l+0]) + ((h[l/2] & hm.s1) ? 0.f : yl[l+1]);
                s4 += yl[l+16] * (qs & qm.s2);
                s5 += yl[l+17] * (qs & qm.s3);
                s6 += ((h[l/2] & hm.s2) ? 0.f : yl[l+16]) + ((h[l/2] & hm.s3) ? 0.f : yl[l+17]);
            }
            float d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            float d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (sc.s0 - 32);
            sumf2[row] += d2 * (sc.s2 - 32);

            s1 = s2 = s3 = s4 = s5 = s6 = 0;
            for (int l = 0; l < 8; l += 2) {
                int qs = q[l/2+8];
                s1 += yl[l+8] * (qs & qm.s0);
                s2 += yl[l+9] * (qs & qm.s1);
                s3 += ((h[l/2+8] & hm.s0) ? 0.f : yl[l+8]) + ((h[l/2+8] & hm.s1) ? 0.f : yl[l+9]);
                s4 += yl[l+24] * (qs & qm.s2);
                s5 += yl[l+25] * (qs & qm.s3);
                s6 += ((h[l/2+8] & hm.s2) ? 0.f : yl[l+24]) + ((h[l/2+8] & hm.s3) ? 0.f : yl[l+25]);
            }
            d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (sc.s1 - 32);
            sumf2[row] += d2 * (sc.s3 - 32);

            q  += nb01/2;
            h  += nb01/2;
            a  += nb01/2;
            dh += nb01/2;
        }

        y1 += BLOCK_STRIDE * QK_K;
    }

    global float * dst_f32 = (global float *) dst + im*ne0*ne1 + r1*ne0;

    for (int row = 0; row < N_DST; ++row) {
        all_sum = sub_group_reduce_add((sumf1[row] + 0.25f * sumf2[row]) / (1 << shift));
        if (first_row + row < ne01) {
            if (get_sub_group_local_id() == 0) {
                dst_f32[first_row + row] = all_sum;
            }
        }
    }
}
