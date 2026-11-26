// Cooperative-K DP4A GEMM for 2 to 4 columns with Q6_K weights and Q8_1 activations.
// Named vector components avoid dynamic private-array indexing, which can cause register spills.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#ifndef COOPK_NSG
#define COOPK_NSG 4
#endif
#define COOPK_SG  64

#ifndef COOPK_ROWS
#define COOPK_ROWS 4
#endif
#ifndef COOPK_COLS
#define COOPK_COLS 4
#endif

#if COOPK_COLS == 4
typedef float4 coopk_accv;
typedef int4   coopk_dotv;
#define COOPK_CONVF convert_float4
#else
typedef float2 coopk_accv;
typedef int2   coopk_dotv;
#define COOPK_CONVF convert_float2
#endif

// Combine low nibbles and high bits, subtract 32, and pack four signed bytes for DP4A.
#define EXP6(l, h, mf, mc)                                                                    \
    ( (((uint)((int)((( (l)        & 0x000Fu) | (((h) & 0x03u) << 4)) - 32) & 0xFF)))         \
    | (((uint)((int)(((((l) & 0x00F0u) >> 4)  | (((h) & 0x0Cu) << 2)) - 32) & 0xFF)) <<  8)   \
    | (((uint)((int)(((((l) & 0x0F00u) >> 8)  |  ((h) & 0x30u))       - 32) & 0xFF)) << 16)   \
    | (((uint)((int)(((((l) & (mf))   >> 12)  | (((h) & (mc)) >> 2))  - 32) & 0xFF)) << 24) )

#if COOPK_ROWS == 4
#define COOPK_DOT(ci, t)                                               \
    d0.s##ci = dot_acc_sat_4x8packed_ss_int(w0, A##ci.s##t, d0.s##ci); \
    d1.s##ci = dot_acc_sat_4x8packed_ss_int(w1, A##ci.s##t, d1.s##ci); \
    d2.s##ci = dot_acc_sat_4x8packed_ss_int(w2, A##ci.s##t, d2.s##ci); \
    d3.s##ci = dot_acc_sat_4x8packed_ss_int(w3, A##ci.s##t, d3.s##ci);
#else
#define COOPK_DOT(ci, t)                                               \
    d0.s##ci = dot_acc_sat_4x8packed_ss_int(w0, A##ci.s##t, d0.s##ci); \
    d1.s##ci = dot_acc_sat_4x8packed_ss_int(w1, A##ci.s##t, d1.s##ci);
#endif

#if COOPK_COLS == 4
#define COOPK_DOTS_AT(t)  COOPK_DOT(0,t) COOPK_DOT(1,t) COOPK_DOT(2,t) COOPK_DOT(3,t)
#define COOPK_FOR_COLS(F) F(0) F(1) F(2) F(3)
#else
#define COOPK_DOTS_AT(t)  COOPK_DOT(0,t) COOPK_DOT(1,t)
#define COOPK_FOR_COLS(F) F(0) F(1)
#endif

#if COOPK_ROWS == 4
#define COOPK_KSTEP(t)                                      \
    {                                                       \
    ushort4 bl = vload4(0, src0_ql + row0 + (ku0 + t) * m); \
    uchar4  bh = vload4(0, src0_qh + row0 + (ku0 + t) * m); \
    const uint w0 = EXP6(bl.s0, bh.s0, mask_f000, mask_c0); \
    const uint w1 = EXP6(bl.s1, bh.s1, mask_f000, mask_c0); \
    const uint w2 = EXP6(bl.s2, bh.s2, mask_f000, mask_c0); \
    const uint w3 = EXP6(bl.s3, bh.s3, mask_f000, mask_c0); \
    COOPK_DOTS_AT(t)                                        \
    }
#else
#define COOPK_KSTEP(t)                                      \
    {                                                       \
    ushort2 bl = vload2(0, src0_ql + row0 + (ku0 + t) * m); \
    uchar2  bh = vload2(0, src0_qh + row0 + (ku0 + t) * m); \
    const uint w0 = EXP6(bl.s0, bh.s0, mask_f000, mask_c0); \
    const uint w1 = EXP6(bl.s1, bh.s1, mask_f000, mask_c0); \
    COOPK_DOTS_AT(t)                                        \
    }
#endif

// Weights use the untiled Adreno noshuffle SoA layout, with the output row contiguous in each plane.
// Q8_1 activation planes are K-major (K contiguous); no activation transpose is used.
kernel void kernel_gemm_coopk_q6_k_q8_1_dp4a(
    global const ushort * src0_ql,    // four consecutive low nibbles per ushort, [k/4][m]
    global const uchar  * src0_qh,    // four consecutive high 2-bit codes per uchar, [k/4][m]
    global const ushort * src0_s,     // two signed int8 scales per ushort, [k/32][m]
    global const half   * src0_d,     // super-block scales, [k/256][m]
    global const uint   * src1_qa,    // four signed activation bytes per uint, [n][k/4]
    global const half   * src1_da,    // activation scales, [n][k/32]
    global float * dst,
    ulong offsetd,
    int m,
    int n,
    int k,
    int n_no_padding,
    ushort mask_f000,
    uchar  mask_c0
) {
    dst = (global float *)((global char *)dst + offsetd);

    const int gx   = get_global_id(0);
    const int sg   = get_local_id(1);
    const int lane = get_local_id(0);

    const int row0 = gx * COOPK_ROWS;
    const int num_32blk = k / 32;
    const int k_u = k >> 2;
    const int k_b = k >> 5;

    // The host pads activations to COOPK_COLS; stores mask columns beyond n_no_padding.
    const int c0 = 0, c1 = 1;
#if COOPK_COLS == 4
    const int c2 = 2, c3 = 3;
#endif

    coopk_accv acc0 = (coopk_accv)(0.0f), acc1 = (coopk_accv)(0.0f);
#if COOPK_ROWS == 4
    coopk_accv acc2 = (coopk_accv)(0.0f), acc3 = (coopk_accv)(0.0f);
#endif

    for (int blk = sg; blk < num_32blk; blk += COOPK_NSG) {
        const int i = blk << 5;

        // Each ushort packs two signed scale bytes, one for each 16-value half of this 32-value block.
        // The half-precision super-block scale is shared by 256 K values.
        ushort4 spk = vload4(0, src0_s + row0 + blk * m);
        half4   scd = vload4(0, src0_d + row0 + (i >> 8) * m);

        coopk_accv da;
        da.s0 = (float)src1_da[c0*k_b + blk];
        da.s1 = (float)src1_da[c1*k_b + blk];
#if COOPK_COLS == 4
        da.s2 = (float)src1_da[c2*k_b + blk];
        da.s3 = (float)src1_da[c3*k_b + blk];
#endif

        // Apply each signed weight scale to its 16-value dot product.
        for (int half_idx = 0; half_idx < 2; ++half_idx) {
            const int ku0 = (i >> 2) + half_idx * 4;

            coopk_dotv d0 = (coopk_dotv)(0), d1 = (coopk_dotv)(0);
#if COOPK_ROWS == 4
            coopk_dotv d2 = (coopk_dotv)(0), d3 = (coopk_dotv)(0);
#endif
            uint4 A0 = vload4(0, src1_qa + (uint)c0 * k_u + ku0);
            uint4 A1 = vload4(0, src1_qa + (uint)c1 * k_u + ku0);
#if COOPK_COLS == 4
            uint4 A2 = vload4(0, src1_qa + (uint)c2 * k_u + ku0);
            uint4 A3 = vload4(0, src1_qa + (uint)c3 * k_u + ku0);
#endif
            COOPK_KSTEP(0)
            COOPK_KSTEP(1)
            COOPK_KSTEP(2)
            COOPK_KSTEP(3)

            const float s0f = (float)(half_idx ? as_char2(spk.s0).s1 : as_char2(spk.s0).s0) * (float)scd.s0;
            const float s1f = (float)(half_idx ? as_char2(spk.s1).s1 : as_char2(spk.s1).s0) * (float)scd.s1;
            acc0 += s0f * da * COOPK_CONVF(d0);
            acc1 += s1f * da * COOPK_CONVF(d1);
#if COOPK_ROWS == 4
            const float s2f = (float)(half_idx ? as_char2(spk.s2).s1 : as_char2(spk.s2).s0) * (float)scd.s2;
            const float s3f = (float)(half_idx ? as_char2(spk.s3).s1 : as_char2(spk.s3).s0) * (float)scd.s3;
            acc2 += s2f * da * COOPK_CONVF(d2);
            acc3 += s3f * da * COOPK_CONVF(d3);
#endif
        }
    }

    // Reuse one local buffer for all rows; the first barrier protects the previous row's reads.
    local coopk_accv reduceLM[COOPK_SG * (COOPK_NSG - 1)];
    coopk_accv out0 = (coopk_accv)(0.0f), out1 = (coopk_accv)(0.0f);
#if COOPK_ROWS == 4
    coopk_accv out2 = (coopk_accv)(0.0f), out3 = (coopk_accv)(0.0f);
#endif

#define COOPK_REDUCE(accv, outv)                                   \
    barrier(CLK_LOCAL_MEM_FENCE);                                  \
    if (sg > 0) { reduceLM[(sg - 1) * COOPK_SG + lane] = (accv); } \
    barrier(CLK_LOCAL_MEM_FENCE);                                  \
    if (sg == 0) {                                                 \
        coopk_accv sum = (accv);                                   \
        for (int s = 0; s < COOPK_NSG - 1; s++) {                  \
            sum += reduceLM[s * COOPK_SG + lane];                  \
        }                                                          \
        (outv) = sum;                                              \
    }

    COOPK_REDUCE(acc0, out0)
    COOPK_REDUCE(acc1, out1)
#if COOPK_ROWS == 4
    COOPK_REDUCE(acc2, out2)
    COOPK_REDUCE(acc3, out3)
#endif

#undef COOPK_REDUCE

#if COOPK_ROWS == 4
#define COOPK_STORE_COL(ci)                                                              \
    if (idx < m*n_no_padding) {                                                          \
        vstore4((float4)(out0.s##ci, out1.s##ci, out2.s##ci, out3.s##ci), 0, dst + idx); \
        idx += m;                                                                        \
    }
#else
#define COOPK_STORE_COL(ci)                                      \
    if (idx < m*n_no_padding) {                                  \
        vstore2((float2)(out0.s##ci, out1.s##ci), 0, dst + idx); \
        idx += m;                                                \
    }
#endif

    if (sg == 0) {
        int idx = row0;
        COOPK_FOR_COLS(COOPK_STORE_COL)
    }

#undef COOPK_STORE_COL
}
