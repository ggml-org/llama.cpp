// Cooperative-K DP4A GEMM for 2 to 4 columns with Q4_0 weights and Q8_1 activations.
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

// Pack four Q4_0 weights as signed bytes after subtracting 8.
#define EXP40(l)                                                                      \
    ( (((uint)((int)(( (l)         & 0x000Fu)       - 8) & 0xFF)))                    \
    | (((uint)((int)((((l) & 0x00F0u) >>  4)        - 8) & 0xFF)) <<  8)              \
    | (((uint)((int)((((l) & 0x0F00u) >>  8)        - 8) & 0xFF)) << 16)              \
    | (((uint)((int)((((l) & 0xF000u) >> 12)        - 8) & 0xFF)) << 24) )

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
#define COOPK_KSTEP(t)                                     \
    {                                                      \
    ushort4 bl = vload4(0, src0_q + row0 + (ku0 + t) * m); \
    const uint w0 = EXP40(bl.s0);                          \
    const uint w1 = EXP40(bl.s1);                          \
    const uint w2 = EXP40(bl.s2);                          \
    const uint w3 = EXP40(bl.s3);                          \
    COOPK_DOTS_AT(t)                                       \
    }
#else
#define COOPK_KSTEP(t)                                     \
    {                                                      \
    ushort2 bl = vload2(0, src0_q + row0 + (ku0 + t) * m); \
    const uint w0 = EXP40(bl.s0);                          \
    const uint w1 = EXP40(bl.s1);                          \
    COOPK_DOTS_AT(t)                                       \
    }
#endif

// Weights use the untiled Adreno noshuffle SoA layout, with the output row contiguous in each plane.
// Q8_1 activation planes are K-major (K contiguous); no activation transpose is used.
kernel void kernel_gemm_coopk_q4_0_q8_1_dp4a(
    global const ushort * src0_q,     // four consecutive K nibbles per ushort, [k/4][m]
    global const half   * src0_d,     // block scales, [k/32][m]
    global const uint   * src1_qa,    // four signed activation bytes per uint, [n][k/4]
    global const half   * src1_da,    // activation scales, [n][k/32]
    global float * dst,
    ulong offsetd,
    int m,
    int n,
    int k,
    int n_no_padding
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
#if COOPK_ROWS == 4
        half4 scale = vload4(0, src0_d + row0 + blk * m);
#else
        half2 scale = vload2(0, src0_d + row0 + blk * m);
#endif

        coopk_accv da;
        da.s0 = (float)src1_da[c0*k_b + blk];
        da.s1 = (float)src1_da[c1*k_b + blk];
#if COOPK_COLS == 4
        da.s2 = (float)src1_da[c2*k_b + blk];
        da.s3 = (float)src1_da[c3*k_b + blk];
#endif

        // Q4_0 uses one weight scale per 32 K values.
        coopk_dotv d0 = (coopk_dotv)(0), d1 = (coopk_dotv)(0);
#if COOPK_ROWS == 4
        coopk_dotv d2 = (coopk_dotv)(0), d3 = (coopk_dotv)(0);
#endif
        for (int half_idx = 0; half_idx < 2; ++half_idx) {
            const int ku0 = (blk << 3) + half_idx * 4;

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
        }

        acc0 += (float)scale.s0 * da * COOPK_CONVF(d0);
        acc1 += (float)scale.s1 * da * COOPK_CONVF(d1);
#if COOPK_ROWS == 4
        acc2 += (float)scale.s2 * da * COOPK_CONVF(d2);
        acc3 += (float)scale.s3 * da * COOPK_CONVF(d3);
#endif
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
