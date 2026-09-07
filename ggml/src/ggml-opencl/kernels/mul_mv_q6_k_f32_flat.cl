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

//------------------------------------------------------------------------------
// kernel_mul_mv_q6_K_f32_flat
//------------------------------------------------------------------------------
#define Q6_K_MASK1 0x03
#define Q6_K_MASK2 0x0C
#define Q6_K_MASK3 0x30
#define Q6_K_MASK4 0xC0

#define QK_K       256

// ADRENO_OLD_COMPILER is defined by the host (-D) only for the Adreno E031
// compilers older than E031.45, which miscompile several constructs this kernel
// used (confirmed on E031.38 and E031.41; E031.45 is clean). Every other
// compiler -- newer E031, E17, DX, Intel, and every non-Adreno device that
// builds this program -- takes the #else branches, which are the original
// source: the workarounds below cost ~13% on the q6_K flat n=1 GEMV where they
// are not needed.
inline float block_q_6_K_dot_y_flat(
    global uchar * blk_ql,
    global uchar * blk_qh,
    global char  * blk_scales,
    global half  * blk_d,
    int ib,
    int ip,
    int is,
    int l0,
#if defined(ADRENO_OLD_COMPILER)
    int dbg,
#endif
    float4 y0,
    float4 y1,
    float4 y2,
    float4 y3
) {
    int q_offset_l =  64*ip + l0;
    int q_offset_h =  32*ip + l0;

    global uchar * q1 = blk_ql     + ib*128 + q_offset_l;
    global uchar * q2 = q1         + QK_K/8;
    global uchar * qh = blk_qh     + ib*64 + q_offset_h;

    float dall = blk_d[ib];

#if defined(ADRENO_OLD_COMPILER)
    // The vectorized dequant (int4/float4 bit-ops, convert_*4, dot()) and vload4
    // are miscompiled here -> garbage weights. Reconstruct the 6-bit weights and
    // take the dot product scalar. q4_K/q5_K flat already use scalar paths, which
    // is why q6_K was the only flat GEMV that failed.
    // Scales are SIGNED int8; read as uchar and sign-extend arithmetically so the
    // result does not depend on whether the compiler treats `char` as signed.
    global uchar * sc = (global uchar *)(blk_scales + ib*16 + is);

    int s0 = (int)sc[0] - 256*(sc[0] >> 7);
    int s2 = (int)sc[2] - 256*(sc[2] >> 7);
    int s4 = (int)sc[4] - 256*(sc[4] >> 7);
    int s6 = (int)sc[6] - 256*(sc[6] >> 7);

    // one 6-bit weight: low/high nibble of a ql byte OR'd with a 2-bit qh plane
    // (plane p in {0,1,2,3} selects qh bits 2p..2p+1) placed at bits 4-5, minus 32.
    #define Q6W(qb, sh, hb, p) ((float)((((int)(qb) >> (sh)) & 15) | ((((int)(hb) >> (2*(p))) & 3) << 4)) - 32.f)

    float d0 = y0.s0*Q6W(q1[0],0,qh[0],0) + y0.s1*Q6W(q1[1],0,qh[1],0) + y0.s2*Q6W(q1[2],0,qh[2],0) + y0.s3*Q6W(q1[3],0,qh[3],0);
    float d1 = y1.s0*Q6W(q2[0],0,qh[0],1) + y1.s1*Q6W(q2[1],0,qh[1],1) + y1.s2*Q6W(q2[2],0,qh[2],1) + y1.s3*Q6W(q2[3],0,qh[3],1);
    float d2 = y2.s0*Q6W(q1[0],4,qh[0],2) + y2.s1*Q6W(q1[1],4,qh[1],2) + y2.s2*Q6W(q1[2],4,qh[2],2) + y2.s3*Q6W(q1[3],4,qh[3],2);
    float d3 = y3.s0*Q6W(q2[0],4,qh[0],3) + y3.s1*Q6W(q2[1],4,qh[1],3) + y3.s2*Q6W(q2[2],4,qh[2],3) + y3.s3*Q6W(q2[3],4,qh[3],3);
    #undef Q6W

    if (dbg) printf("HELPER dall=%f s=[%d %d %d %d] d=[%f %f %f %f] ql0=%d qh0=%d y00=%f\n",
        dall, s0, s2, s4, s6, d0, d1, d2, d3, (int)q1[0], (int)qh[0], y0.s0);

    return dall * (d0 * s0 + d1 * s2 + d2 * s4 + d3 * s6);
#else
    global char * sc = blk_scales + ib*16 + is;

    // Vectorized loads: 3 uchar4 weight loads instead of 12 scalar byte reads.
    // q_offset_l/h are 4-aligned, so these are aligned vector loads.
    uchar4 q1v = vload4(0, q1);
    uchar4 q2v = vload4(0, q2);
    uchar4 qhv = vload4(0, qh);

    int4 q1i = convert_int4(q1v);
    int4 q2i = convert_int4(q2v);
    int4 qhi = convert_int4(qhv);

    // Reconstruct the four 6-bit weight groups (low/high nibble of ql OR'd with the
    // matching 2-bit plane of qh), same arithmetic as the scalar version, then dot()
    // against the cached activation lanes.
    float4 w0 = convert_float4((q1i & 0xF) | ((qhi & Q6_K_MASK1) << 4)) - 32.f;
    float4 w1 = convert_float4((q2i & 0xF) | ((qhi & Q6_K_MASK2) << 2)) - 32.f;
    float4 w2 = convert_float4((q1i >> 4)  | ((qhi & Q6_K_MASK3)     )) - 32.f;
    float4 w3 = convert_float4((q2i >> 4)  | ((qhi & Q6_K_MASK4) >> 2)) - 32.f;

    return dall * (dot(y0, w0) * sc[0] + dot(y1, w1) * sc[2] +
                   dot(y2, w2) * sc[4] + dot(y3, w3) * sc[6]);
#endif
}

#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define N_DST 4
#define N_SIMDGROUP 2
#define N_SIMDWIDTH 16
#elif defined (ADRENO_GPU)
#define N_DST 16
#define N_SIMDGROUP 2
#define N_SIMDWIDTH 64
#endif

#define BLOCK_STRIDE (N_SIMDWIDTH/16) // number of blocks each subgroup processes

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q6_K_f32_flat(
        global uchar * src0_ql,
        global uchar * src0_qh,
        global char  * src0_s,
        global half  * src0_d,
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
#if defined(ADRENO_OLD_COMPILER)
        ,
        uchar q6k_mask   // runtime 0xFF; the host passes it so the compiler cannot
                         // constant-fold the printf guards below into nothing
#endif
) {
    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int nb = ne00/QK_K;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int i12 = im%ne12;
    int i13 = im/ne12;

    int first_row = (N_SIMDGROUP * r0 + get_sub_group_id()) * N_DST;

#if defined(ADRENO_OLD_COMPILER)
    // 64-bit `ulong` integer arithmetic is miscompiled here -> the base-pointer byte
    // offsets came out wrong, so EVERY weight/scale read hit the wrong address. This
    // was the primary cause of the q6_K flat failure (q5_K uses int offsets and is
    // unaffected). Compute the block index in `int` and widen to `ulong` only inside
    // the pointer expression: the byte offset stays 64-bit, but there is no ulong
    // arithmetic chain to miscompile. The int index would overflow past ~2^31 blocks,
    // which no realistic weight reaches -- but that is a narrowing, so keep it off the
    // conformant path, which retains full ulong arithmetic.
    int offset_src0 = first_row*nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    global uchar * blk_ql     = (global uchar *) src0_ql + (ulong)offset_src0 * 128;
    global uchar * blk_qh     = (global uchar *) src0_qh + (ulong)offset_src0 * 64;
    global char  * blk_scales = (global char  *) src0_s  + (ulong)offset_src0 * 16;
    global half  * blk_d      = (global half  *) src0_d  + offset_src0;
#else
    ulong offset_src0    = first_row*nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    ulong offset_src0_ql = offset_src0 * 128;
    ulong offset_src0_qh = offset_src0 * 64;
    ulong offset_src0_s  = offset_src0 * 16;
    ulong offset_src0_d  = offset_src0;

    global uchar * blk_ql     = (global uchar *) src0_ql + offset_src0_ql;
    global uchar * blk_qh     = (global uchar *) src0_qh + offset_src0_qh;
    global char  * blk_scales = (global char  *) src0_s  + offset_src0_s;
    global half  * blk_d      = (global half  *) src0_d  + offset_src0_d;
#endif
    global float * yy         = (global float *) src1    + r1*ne10 + im*ne00*ne1;

    int tid = get_sub_group_local_id()%(N_SIMDWIDTH/BLOCK_STRIDE); // within-super-block part, 0..15
    int ix  = get_sub_group_local_id()/(N_SIMDWIDTH/BLOCK_STRIDE); // super-block selector, 0..BLOCK_STRIDE-1
    int ip  = tid/8;   // first or second half of (super) block (0 or 1)
    int il  = tid%8;   // each half has 8 parts, one per scale
    int n   = 4;       // 4 scales at a time (and 4 sums)
    int l0  = n*il;    // offset into half-block, 0..28
    int is  = 8*ip + l0/16; // 0, 1, 8, 9

    float sumf[N_DST];
    for (int row = 0; row < N_DST; row++) {
        sumf[row] = 0.f;
    }

    for (int ib = ix; ib < nb; ib += BLOCK_STRIDE) {
        global float * y = yy + ib * QK_K + 128*ip + l0;
#if defined(ADRENO_OLD_COMPILER)
        // vload4 of f32 is miscompiled here; index the lanes scalar instead.
        float4 y0 = (float4)(y[ 0], y[ 1], y[ 2], y[ 3]);
        float4 y1 = (float4)(y[32], y[33], y[34], y[35]);
        float4 y2 = (float4)(y[64], y[65], y[66], y[67]);
        float4 y3 = (float4)(y[96], y[97], y[98], y[99]);
#else
        float4 y0 = vload4(0, y +  0);
        float4 y1 = vload4(0, y + 32);
        float4 y2 = vload4(0, y + 64);
        float4 y3 = vload4(0, y + 96);
#endif

        for (int row = 0; row < N_DST; row++) {
            if (first_row + row < ne01) {
#if defined(ADRENO_OLD_COMPILER)
                int dbg = (q6k_mask==0xFE && r0==0 && r1==0 && im==0 && row==0 && ib==0 &&
                           ne00==256 && ne01==16 && get_sub_group_local_id()==0) ? 1 : 0;
                sumf[row] += block_q_6_K_dot_y_flat(
                    blk_ql + row*nb*128, blk_qh + row*nb*64, blk_scales + row*nb*16, blk_d + row*nb,
                    ib, ip, is, l0, dbg, y0, y1, y2, y3);
#else
                sumf[row] += block_q_6_K_dot_y_flat(
                    blk_ql + row*nb*128, blk_qh + row*nb*64, blk_scales + row*nb*16, blk_d + row*nb,
                    ib, ip, is, l0, y0, y1, y2, y3);
#endif
            }
        }
    }

#if defined(ADRENO_OLD_COMPILER)
    // Optimizer barrier. This compiler drops the sumf partials unless a side effect
    // forces them to materialize. q6k_mask is a kernel arg the compiler cannot prove
    // is never 0xFE (the host always passes 0xFF), so the printf survives compilation
    // but never executes. FRAGILE: the exact set and placement of these guarded
    // printfs is load-bearing on E031.41 -- removing any one re-breaks q6_K.
    if (q6k_mask==0xFE && r0==0 && r1==0 && im==0 && ne00==256 && ne01==16 && get_sub_group_local_id()<16) {
        printf("Q6KLANE lane=%d ip=%d il=%d is=%d l0=%d sumf0=%f\n",
            get_sub_group_local_id(), ip, il, is, l0, sumf[0]);
    }
#endif
    for (int row = 0; row < N_DST; row++) {
        float tot = sub_group_reduce_add(sumf[row]);
        if (get_sub_group_local_id() == 0 && first_row + row < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = tot;
#if defined(ADRENO_OLD_COMPILER)
            if (q6k_mask==0xFE && r0==0 && r1==0 && im==0 && row==0 && ne00==256 && ne01==16)
                printf("Q6KTOT tot=%f\n", tot);
#endif
        }
    }
}
