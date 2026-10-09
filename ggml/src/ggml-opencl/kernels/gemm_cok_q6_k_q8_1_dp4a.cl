// Cooperative-K q6_K GEMM with dp4a for narrow batches (ne1 = 2..4). A lane folds COK_ROWS rows and
// computes COK_COLS (4 or 2) columns; COK_NSG subgroups split K, reduced in local memory. The -32 is
// folded into the byte packing; ss changes every 16 K, so the dot is flushed twice per 32-K block.
// Accumulators and column indices are named, never dynamically indexed private arrays.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#ifndef COK_NSG
#define COK_NSG 4
#endif
#define COK_SG  64

#ifndef COK_ROWS
#define COK_ROWS 4
#endif
#ifndef COK_COLS
#define COK_COLS 4
#endif

#if COK_COLS == 4
typedef float4 cok_accv;
typedef int4   cok_dotv;
#define COK_CONVF convert_float4
#else
typedef float2 cok_accv;
typedef int2   cok_dotv;
#define COK_CONVF convert_float2
#endif

// Four consecutive-K q6_K weights of one row as signed bytes in a uint, for dp4a. Low
// nibble from the ushort plane, high two bits from the uchar plane, -32 bias folded in
// ((q - 32) in [-32, 31]).
#define EXP6(l, h, mf, mc)                                                            \
    ( (((uint)((int)((( (l)        & 0x000Fu) | (((h) & 0x03u) << 4)) - 32) & 0xFF)))       \
    | (((uint)((int)(((((l) & 0x00F0u) >> 4)  | (((h) & 0x0Cu) << 2)) - 32) & 0xFF)) <<  8) \
    | (((uint)((int)(((((l) & 0x0F00u) >> 8)  |  ((h) & 0x30u))       - 32) & 0xFF)) << 16) \
    | (((uint)((int)(((((l) & (mf))   >> 12)  | (((h) & (mc)) >> 2))  - 32) & 0xFF)) << 24) )

#if COK_ROWS == 4
#define COK_DOT(ci, t)                                                 \
    d0.s##ci = dot_acc_sat_4x8packed_ss_int(w0, A##ci.s##t, d0.s##ci); \
    d1.s##ci = dot_acc_sat_4x8packed_ss_int(w1, A##ci.s##t, d1.s##ci); \
    d2.s##ci = dot_acc_sat_4x8packed_ss_int(w2, A##ci.s##t, d2.s##ci); \
    d3.s##ci = dot_acc_sat_4x8packed_ss_int(w3, A##ci.s##t, d3.s##ci);
#else
#define COK_DOT(ci, t)                                                 \
    d0.s##ci = dot_acc_sat_4x8packed_ss_int(w0, A##ci.s##t, d0.s##ci); \
    d1.s##ci = dot_acc_sat_4x8packed_ss_int(w1, A##ci.s##t, d1.s##ci);
#endif

#if COK_COLS == 4
#define COK_DOTS_AT(t)  COK_DOT(0,t) COK_DOT(1,t) COK_DOT(2,t) COK_DOT(3,t)
#define COK_FOR_COLS(F) F(0) F(1) F(2) F(3)
#else
#define COK_DOTS_AT(t)  COK_DOT(0,t) COK_DOT(1,t)
#define COK_FOR_COLS(F) F(0) F(1)
#endif

// One K-group (4 K values): unpack the folded rows' weights, then dot every column.
#if COK_ROWS == 4
#define COK_KSTEP(t)                                                        \
    {                                                                       \
    ushort4 bl = vload4(0, src0_ql + row0 + (ku0 + t) * m);                 \
    uchar4  bh = vload4(0, src0_qh + row0 + (ku0 + t) * m);                 \
    const uint w0 = EXP6(bl.s0, bh.s0, mask_f000, mask_c0);                 \
    const uint w1 = EXP6(bl.s1, bh.s1, mask_f000, mask_c0);                 \
    const uint w2 = EXP6(bl.s2, bh.s2, mask_f000, mask_c0);                 \
    const uint w3 = EXP6(bl.s3, bh.s3, mask_f000, mask_c0);                 \
    COK_DOTS_AT(t)                                                          \
    }
#else
#define COK_KSTEP(t)                                                        \
    {                                                                       \
    ushort2 bl = vload2(0, src0_ql + row0 + (ku0 + t) * m);                 \
    uchar2  bh = vload2(0, src0_qh + row0 + (ku0 + t) * m);                 \
    const uint w0 = EXP6(bl.s0, bh.s0, mask_f000, mask_c0);                 \
    const uint w1 = EXP6(bl.s1, bh.s1, mask_f000, mask_c0);                 \
    COK_DOTS_AT(t)                                                          \
    }
#endif

kernel void kernel_gemm_cok_q6_k_q8_1_dp4a(
    global const ushort * src0_ql,    // low nibbles   [row + (K/4)*m]
    global const uchar  * src0_qh,    // high 2 bits   [row + (K/4)*m]
    global const ushort * src0_s,     // two int8 scales per 32 K
    global const half   * src0_d,     // super-block scale, per 256 K
    global const uint   * src1_qa,    // q8_1 activations [col*k_u + K/4]
    global const half   * src1_da,    // activation scale [col*k_b + blk]
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

    const int row0 = gx * COK_ROWS;
    const int num_32blk = k / 32;
    const int k_u = k >> 2;
    const int k_b = k >> 5;

    // Column indices are compile-time constants: the activation is allocated at the kernel's
    // width, so nothing is clamped and the address arithmetic stays affine.
    const int c0 = 0, c1 = 1;
#if COK_COLS == 4
    const int c2 = 2, c3 = 3;
#endif

    cok_accv acc0 = (cok_accv)(0.0f), acc1 = (cok_accv)(0.0f);
#if COK_ROWS == 4
    cok_accv acc2 = (cok_accv)(0.0f), acc3 = (cok_accv)(0.0f);
#endif

    for (int blk = sg; blk < num_32blk; blk += COK_NSG) {
        const int i = blk << 5;

        // one ushort per row holds BOTH int8 scales for this 32-K block: low half is the
        // first 16 K, high half the second. One half per row per 256 K on top.
        ushort4 spk = vload4(0, src0_s + row0 + blk * m);
        half4   scd = vload4(0, src0_d + row0 + (i >> 8) * m);

        cok_accv da;
        da.s0 = (float)src1_da[c0*k_b + blk];
        da.s1 = (float)src1_da[c1*k_b + blk];
#if COK_COLS == 4
        da.s2 = (float)src1_da[c2*k_b + blk];
        da.s3 = (float)src1_da[c3*k_b + blk];
#endif

        // Two 16-K halves, each with its own weight scale, so the raw dot is flushed twice.
        for (int half_idx = 0; half_idx < 2; ++half_idx) {
            const int ku0 = (i >> 2) + half_idx * 4;

            cok_dotv d0 = (cok_dotv)(0), d1 = (cok_dotv)(0);
#if COK_ROWS == 4
            cok_dotv d2 = (cok_dotv)(0), d3 = (cok_dotv)(0);
#endif
            uint4 A0 = vload4(0, src1_qa + (uint)c0 * k_u + ku0);
            uint4 A1 = vload4(0, src1_qa + (uint)c1 * k_u + ku0);
#if COK_COLS == 4
            uint4 A2 = vload4(0, src1_qa + (uint)c2 * k_u + ku0);
            uint4 A3 = vload4(0, src1_qa + (uint)c3 * k_u + ku0);
#endif
            COK_KSTEP(0)
            COK_KSTEP(1)
            COK_KSTEP(2)
            COK_KSTEP(3)

            const float s0f = (float)(half_idx ? as_char2(spk.s0).s1 : as_char2(spk.s0).s0) * (float)scd.s0;
            const float s1f = (float)(half_idx ? as_char2(spk.s1).s1 : as_char2(spk.s1).s0) * (float)scd.s1;
            acc0 += s0f * da * COK_CONVF(d0);
            acc1 += s1f * da * COK_CONVF(d1);
#if COK_ROWS == 4
            const float s2f = (float)(half_idx ? as_char2(spk.s2).s1 : as_char2(spk.s2).s0) * (float)scd.s2;
            const float s3f = (float)(half_idx ? as_char2(spk.s3).s1 : as_char2(spk.s3).s0) * (float)scd.s3;
            acc2 += s2f * da * COK_CONVF(d2);
            acc3 += s3f * da * COK_CONVF(d3);
#endif
        }
    }

    // Cross-subgroup reduction over the K split, one row at a time, in named registers.
    local cok_accv reduceLM[COK_SG * (COK_NSG - 1)];
    cok_accv out0 = (cok_accv)(0.0f), out1 = (cok_accv)(0.0f);
#if COK_ROWS == 4
    cok_accv out2 = (cok_accv)(0.0f), out3 = (cok_accv)(0.0f);
#endif

#define COK_REDUCE(accv, outv)                                       \
    barrier(CLK_LOCAL_MEM_FENCE);                                    \
    if (sg > 0) { reduceLM[(sg - 1) * COK_SG + lane] = (accv); }     \
    barrier(CLK_LOCAL_MEM_FENCE);                                    \
    if (sg == 0) {                                                   \
        cok_accv sum = (accv);                                       \
        for (int s = 0; s < COK_NSG - 1; s++) {                      \
            sum += reduceLM[s * COK_SG + lane];                      \
        }                                                            \
        (outv) = sum;                                                \
    }

    COK_REDUCE(acc0, out0)
    COK_REDUCE(acc1, out1)
#if COK_ROWS == 4
    COK_REDUCE(acc2, out2)
    COK_REDUCE(acc3, out3)
#endif

#undef COK_REDUCE

#if COK_ROWS == 4
#define COK_STORE_COL(ci)                                                                 \
    if (idx < m*n_no_padding) {                                                           \
        vstore4((float4)(out0.s##ci, out1.s##ci, out2.s##ci, out3.s##ci), 0, dst + idx);  \
        idx += m;                                                                         \
    }
#else
#define COK_STORE_COL(ci)                                                                 \
    if (idx < m*n_no_padding) {                                                           \
        vstore2((float2)(out0.s##ci, out1.s##ci), 0, dst + idx);                          \
        idx += m;                                                                         \
    }
#endif

    if (sg == 0) {
        int idx = row0;
        COK_FOR_COLS(COK_STORE_COL)
    }

#undef COK_STORE_COL
}
