#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef NVIDIA_GPU

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

//------------------------------------------------------------------------------
// block_q4_K
//------------------------------------------------------------------------------
#define QK_K            256
#define BLOCK_Q4K_SIZE  144
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
#define N_DST 16
#define N_SIMDGROUP 2
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
kernel void kernel_mul_mv_q4_K_f32_flat(
    global uchar * src0_q,
    global uchar * src0_s,
    global half  * src0_d,
    global half  * src0_dm,
    global char  * src1,
    int offset1,
    global char  * dst,
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
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    ushort kmask1 = 0x3f3f;
    ushort kmask2 = 0x0f0f;
    ushort kmask3 = 0xc0c0;

    int ix = get_sub_group_local_id()/8;
    int it = get_sub_group_local_id()%8;
    int iq = it/4;
    int ir = it%4;

    int nb = ne00/QK_K;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);
    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    int i12 = im%ne12;
    int i13 = im/ne12;

    int offset_src0 = (first_row*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03)/BLOCK_Q4K_SIZE;
    uint blk = nb01 / BLOCK_Q4K_SIZE;
    global uchar * blk_q     = (global uchar *)src0_q  + offset_src0*(QK_K/2);
    global uchar * blk_s     = (global uchar *)src0_s  + offset_src0*K_SCALE_SIZE;
    global half  * blk_d     = (global half  *)src0_d  + offset_src0;
    global half  * blk_dm    = (global half  *)src0_dm + offset_src0;

    int offset_src1 = r1*nb11 + (i12)*nb12 + (i13)*nb13;
    global float * y = (global float *)(src1 + offset_src1);

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

        global ushort * q1 = (global ushort *)(blk_q + ib * (QK_K/2)) + (16 * iq + 4 * ir);
        global ushort * sc = (global ushort *)(blk_s + ib * K_SCALE_SIZE) + iq;
        global half   * d  = blk_d + ib;
        global half   * dm = blk_dm + ib;

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

            float dall = *d;
            float dmin = *dm;
            sumf[row] += dall * ((acc1.s0 + 1.f/256.f * acc1.s1) * sc8[0] +
                                 (acc1.s2 + 1.f/256.f * acc1.s3) * sc8[1] * 1.f/16.f +
                                 (acc2.s0 + 1.f/256.f * acc2.s1) * sc8[4] +
                                 (acc2.s2 + 1.f/256.f * acc2.s3) * sc8[5] * 1.f/16.f) -
                         dmin * (sumy.s0 * sc8[2] + sumy.s1 * sc8[3] + sumy.s2 * sc8[6] + sumy.s3 * sc8[7]);

            q1 += blk*64;
            sc += blk*6;
            d  += blk;
            dm += blk;
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

#endif // !NVIDIA_GPU

// NVIDIA OpenCL does not expose cl_khr_subgroups as a compile-time extension
// macro, so we skip the #pragma here and use __local tree-reduction instead.

// SOA flat kernel for Q4_K x f32 matrix-vector multiply.
// NVIDIA only: N_SIMDWIDTH=32 (warp), BLOCK_STRIDE=2, __local tree-reduction.
//
// Q4_K qs byte layout (SOA, 128 bytes per super-block = QK_K/2):
//   qs[ 0..31]: lo nibble = element   0..31  (sg=0),  hi nibble = element  32..63  (sg=1)
//   qs[32..63]: lo nibble = element  64..95  (sg=2),  hi nibble = element  96..127 (sg=3)
//   qs[64..95]: lo nibble = element 128..159 (sg=4),  hi nibble = element 160..191 (sg=5)
//   qs[96..127]:lo nibble = element 192..223 (sg=6),  hi nibble = element 224..255 (sg=7)
//
// Thread mapping (16 threads per super-block):
//   tid = lid / BLOCK_STRIDE  (0..15)   role within super-block
//   ix  = lid % BLOCK_STRIDE  (0..BLOCK_STRIDE-1)  which super-block this thread handles
//   ip  = tid / 8  (0..1)   first or second 64-byte qs section
//   il  = tid % 8  (0..7)   8-byte segment within section
//   sg_lo = 4*ip + 2*(il/4)   scale-group for lo nibbles (0,2,4,6)
//   sg_hi = sg_lo + 1          scale-group for hi nibbles (1,3,5,7)
//   q_off    = 64*ip + 8*il   byte offset into qs[0..127]
//   y_lo_off = 32*sg_lo + 8*(il&3)
//   y_hi_off = y_lo_off + 32
//
// Scale decode follows get_scale_min_k4 from ggml-quants.c.

#define QK_K         256
#define K_SCALE_SIZE 12
#define N_DST        4

#define N_SIMDWIDTH 32

// Apple M1 no-subgroups compat mode defines BOTH NVIDIA_GPU and INTEL_GPU.
// Override N_SIMDWIDTH to 32 to match the nth0=32 workgroup dispatch used for
// this GPU — with nth0=16 (INTEL path default), lm[N_SIMDWIDTH] would be
// under-sized and threads lid=16..31 would write out of bounds.
// BLOCK_STRIDE = 32/16 = 2 so ix ∈ {0,1} covers all super-blocks correctly.
#if defined(NVIDIA_GPU) && defined(INTEL_GPU)
#undef  N_SIMDWIDTH
#define N_SIMDWIDTH 32
#endif

// 16 threads collaborate on one super-block, BLOCK_STRIDE super-blocks per iteration
#define BLOCK_STRIDE (N_SIMDWIDTH/16)

kernel void kernel_mul_mv_q4_K_f32_flat(
    global uchar * src0_qs,
    global uchar * src0_scales,
    global half  * src0_d,
    global half  * src0_dmin,
    global char  * src1,
    int            offset1,
    global char  * dst,
    int            offsetd,
    int            ne00,
    int            ne01,
    int            ne02,
    int            ne12,
    ulong          nb11,
    ulong          nb12,
    ulong          nb13,
    int            ne0,
    int            ne1,
    int            r2,
    int            r3
) {
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int nb = ne00 / QK_K;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int first_row = r0 * N_DST;
    int i12 = im % ne12;
    int i13 = im / ne12;

    ulong offset_src1 = (ulong)r1*nb11 + (ulong)i12*nb12 + (ulong)i13*nb13;
    global float * y = (global float *)(src1 + offset_src1);

    // Flat block index base for src0 (in units of super-blocks).
    // Layout: [ne02][ne01][nb] super-blocks, with GQA broadcasting via r2/r3.
    ulong offset_src0 = (ulong)first_row * nb
                      + (ulong)(i12/r2) * ((ulong)nb * ne01)
                      + (ulong)(i13/r3) * ((ulong)nb * ne01 * ne02);

    int lid = get_local_id(0);
    int tid = lid / BLOCK_STRIDE;  // 0..15: role within a super-block
    int ix  = lid % BLOCK_STRIDE;  // 0..BLOCK_STRIDE-1: which super-block per iter

    int ip  = tid / 8;             // 0 or 1 (which 64-byte qs section)
    int il  = tid % 8;             // 0..7  (8-byte segment within section)

    // Scale groups for lo and hi nibbles of this thread's 8 qs bytes.
    int sg_lo = 4*ip + 2*(il/4);  // in {0,2,4,6}
    int sg_hi = sg_lo + 1;        // in {1,3,5,7}
    // Clamped versions for safe speculative evaluation in ternary false-branch
    // (GPU evaluates both branches; negative index -> page fault).
    int sg_lo_m4 = max(sg_lo - 4, 0);  // safe substitute for sg_lo-4
    int sg_hi_m4 = max(sg_hi - 4, 0);  // safe substitute for sg_hi-4

    // Byte offset into qs[0..127] for this thread's 8 bytes.
    int q_off = 64*ip + 8*il;

    // Y element offsets.
    int y_lo_off = 32*sg_lo + 8*(il & 3);
    int y_hi_off = y_lo_off + 32;

    float4 sumf = (float4)(0.f);

    for (int ib = ix; ib < nb; ib += BLOCK_STRIDE) {
        global float * yb = y + (ulong)ib * QK_K;

        float4 ylo0 = vload4(0, yb + y_lo_off);      // y[y_lo_off..y_lo_off+3]
        float4 ylo1 = vload4(0, yb + y_lo_off + 4);  // y[y_lo_off+4..y_lo_off+7]
        float4 yhi0 = vload4(0, yb + y_hi_off);
        float4 yhi1 = vload4(0, yb + y_hi_off + 4);

        float sumy_lo = dot(ylo0 + ylo1, (float4)(1.f));
        float sumy_hi = dot(yhi0 + yhi1, (float4)(1.f));

        if (first_row + 0 < ne01) {
            ulong bi = offset_src0 + (ulong)0*nb + ib;
            uchar4 qa = vload4(0, src0_qs + bi*(ulong)128 + q_off);
            uchar4 qb = vload4(0, src0_qs + bi*(ulong)128 + q_off + 4);
            float dotq_lo = dot(ylo0, convert_float4(qa & (uchar4)0x0F))
                          + dot(ylo1, convert_float4(qb & (uchar4)0x0F));
            float dotq_hi = dot(yhi0, convert_float4(qa >> (uchar)4))
                          + dot(yhi1, convert_float4(qb >> (uchar)4));
            global uchar * sc = src0_scales + bi*(ulong)K_SCALE_SIZE;
            float scale_lo = sg_lo < 4 ? (float)(sc[sg_lo]   & 63)
                                       : (float)((sc[sg_lo+4] & 0x0F) | ((sc[sg_lo_m4] >> 6) << 4));
            float scale_hi = sg_hi < 4 ? (float)(sc[sg_hi]   & 63)
                                       : (float)((sc[sg_hi+4] & 0x0F) | ((sc[sg_hi_m4] >> 6) << 4));
            float smin_lo  = sg_lo < 4 ? (float)(sc[sg_lo+4] & 63)
                                       : (float)((sc[sg_lo+4] >>    4) | ((sc[sg_lo  ] >> 6) << 4));
            float smin_hi  = sg_hi < 4 ? (float)(sc[sg_hi+4] & 63)
                                       : (float)((sc[sg_hi+4] >>    4) | ((sc[sg_hi  ] >> 6) << 4));
            sumf.s0 += vload_half(0, src0_d    + bi) * (scale_lo * dotq_lo + scale_hi * dotq_hi)
                     - vload_half(0, src0_dmin  + bi) * (smin_lo  * sumy_lo + smin_hi  * sumy_hi);
        }
        if (first_row + 1 < ne01) {
            ulong bi = offset_src0 + (ulong)1*nb + ib;
            uchar4 qa = vload4(0, src0_qs + bi*(ulong)128 + q_off);
            uchar4 qb = vload4(0, src0_qs + bi*(ulong)128 + q_off + 4);
            float dotq_lo = dot(ylo0, convert_float4(qa & (uchar4)0x0F))
                          + dot(ylo1, convert_float4(qb & (uchar4)0x0F));
            float dotq_hi = dot(yhi0, convert_float4(qa >> (uchar)4))
                          + dot(yhi1, convert_float4(qb >> (uchar)4));
            global uchar * sc = src0_scales + bi*(ulong)K_SCALE_SIZE;
            float scale_lo = sg_lo < 4 ? (float)(sc[sg_lo]   & 63)
                                       : (float)((sc[sg_lo+4] & 0x0F) | ((sc[sg_lo_m4] >> 6) << 4));
            float scale_hi = sg_hi < 4 ? (float)(sc[sg_hi]   & 63)
                                       : (float)((sc[sg_hi+4] & 0x0F) | ((sc[sg_hi_m4] >> 6) << 4));
            float smin_lo  = sg_lo < 4 ? (float)(sc[sg_lo+4] & 63)
                                       : (float)((sc[sg_lo+4] >>    4) | ((sc[sg_lo  ] >> 6) << 4));
            float smin_hi  = sg_hi < 4 ? (float)(sc[sg_hi+4] & 63)
                                       : (float)((sc[sg_hi+4] >>    4) | ((sc[sg_hi  ] >> 6) << 4));
            sumf.s1 += vload_half(0, src0_d    + bi) * (scale_lo * dotq_lo + scale_hi * dotq_hi)
                     - vload_half(0, src0_dmin  + bi) * (smin_lo  * sumy_lo + smin_hi  * sumy_hi);
        }
        if (first_row + 2 < ne01) {
            ulong bi = offset_src0 + (ulong)2*nb + ib;
            uchar4 qa = vload4(0, src0_qs + bi*(ulong)128 + q_off);
            uchar4 qb = vload4(0, src0_qs + bi*(ulong)128 + q_off + 4);
            float dotq_lo = dot(ylo0, convert_float4(qa & (uchar4)0x0F))
                          + dot(ylo1, convert_float4(qb & (uchar4)0x0F));
            float dotq_hi = dot(yhi0, convert_float4(qa >> (uchar)4))
                          + dot(yhi1, convert_float4(qb >> (uchar)4));
            global uchar * sc = src0_scales + bi*(ulong)K_SCALE_SIZE;
            float scale_lo = sg_lo < 4 ? (float)(sc[sg_lo]   & 63)
                                       : (float)((sc[sg_lo+4] & 0x0F) | ((sc[sg_lo_m4] >> 6) << 4));
            float scale_hi = sg_hi < 4 ? (float)(sc[sg_hi]   & 63)
                                       : (float)((sc[sg_hi+4] & 0x0F) | ((sc[sg_hi_m4] >> 6) << 4));
            float smin_lo  = sg_lo < 4 ? (float)(sc[sg_lo+4] & 63)
                                       : (float)((sc[sg_lo+4] >>    4) | ((sc[sg_lo  ] >> 6) << 4));
            float smin_hi  = sg_hi < 4 ? (float)(sc[sg_hi+4] & 63)
                                       : (float)((sc[sg_hi+4] >>    4) | ((sc[sg_hi  ] >> 6) << 4));
            sumf.s2 += vload_half(0, src0_d    + bi) * (scale_lo * dotq_lo + scale_hi * dotq_hi)
                     - vload_half(0, src0_dmin  + bi) * (smin_lo  * sumy_lo + smin_hi  * sumy_hi);
        }
        if (first_row + 3 < ne01) {
            ulong bi = offset_src0 + (ulong)3*nb + ib;
            uchar4 qa = vload4(0, src0_qs + bi*(ulong)128 + q_off);
            uchar4 qb = vload4(0, src0_qs + bi*(ulong)128 + q_off + 4);
            float dotq_lo = dot(ylo0, convert_float4(qa & (uchar4)0x0F))
                          + dot(ylo1, convert_float4(qb & (uchar4)0x0F));
            float dotq_hi = dot(yhi0, convert_float4(qa >> (uchar)4))
                          + dot(yhi1, convert_float4(qb >> (uchar)4));
            global uchar * sc = src0_scales + bi*(ulong)K_SCALE_SIZE;
            float scale_lo = sg_lo < 4 ? (float)(sc[sg_lo]   & 63)
                                       : (float)((sc[sg_lo+4] & 0x0F) | ((sc[sg_lo_m4] >> 6) << 4));
            float scale_hi = sg_hi < 4 ? (float)(sc[sg_hi]   & 63)
                                       : (float)((sc[sg_hi+4] & 0x0F) | ((sc[sg_hi_m4] >> 6) << 4));
            float smin_lo  = sg_lo < 4 ? (float)(sc[sg_lo+4] & 63)
                                       : (float)((sc[sg_lo+4] >>    4) | ((sc[sg_lo  ] >> 6) << 4));
            float smin_hi  = sg_hi < 4 ? (float)(sc[sg_hi+4] & 63)
                                       : (float)((sc[sg_hi+4] >>    4) | ((sc[sg_hi  ] >> 6) << 4));
            sumf.s3 += vload_half(0, src0_d    + bi) * (scale_lo * dotq_lo + scale_hi * dotq_hi)
                     - vload_half(0, src0_dmin  + bi) * (smin_lo  * sumy_lo + smin_hi  * sumy_hi);
        }
    }

    global float * dst_f32 = (global float *)dst + (ulong)im*ne0*ne1 + (ulong)r1*ne0;

    // NVIDIA: cl_khr_subgroups is not exposed as a compile-time extension macro,
    // so use __local tree-reduction instead of sub_group_reduce_add.
    __local float4 lm[N_SIMDWIDTH];
    lm[lid] = sumf;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = N_SIMDWIDTH/2; s > 0; s >>= 1) {
        if (lid < s) lm[lid] += lm[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        if (first_row + 0 < ne01) dst_f32[first_row + 0] = lm[0].s0;
        if (first_row + 1 < ne01) dst_f32[first_row + 1] = lm[0].s1;
        if (first_row + 2 < ne01) dst_f32[first_row + 2] = lm[0].s2;
        if (first_row + 3 < ne01) dst_f32[first_row + 3] = lm[0].s3;
    }
}
