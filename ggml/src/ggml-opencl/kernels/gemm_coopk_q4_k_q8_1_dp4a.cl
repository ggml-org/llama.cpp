// Cooperative-K DP4A GEMM for 2 to 4 columns with Q4_K weights and Q8_1 activations.
// Named vector components avoid dynamic private-array indexing, which can cause register spills.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#define QK_K          256
#define K_SCALE_SIZE   12

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

// Expand four Q4_K nibbles into bytes for DP4A.
#define EXP4(u)  ( ((uint)((u) & 0x000Fu))        | \
                  (((uint)((u) & 0x00F0u)) << 4)  | \
                  (((uint)((u) & 0x0F00u)) << 8)  | \
                  (((uint)((u) & 0xF000u)) << 12) )

#if COOPK_ROWS == 4
#define COOPK_DOT(ci, t)                                               \
    s0.s##ci = dot_acc_sat_4x8packed_ss_int(w0, A##ci.s##t, s0.s##ci); \
    s1.s##ci = dot_acc_sat_4x8packed_ss_int(w1, A##ci.s##t, s1.s##ci); \
    s2.s##ci = dot_acc_sat_4x8packed_ss_int(w2, A##ci.s##t, s2.s##ci); \
    s3.s##ci = dot_acc_sat_4x8packed_ss_int(w3, A##ci.s##t, s3.s##ci);
#else
#define COOPK_DOT(ci, t)                                               \
    s0.s##ci = dot_acc_sat_4x8packed_ss_int(w0, A##ci.s##t, s0.s##ci); \
    s1.s##ci = dot_acc_sat_4x8packed_ss_int(w1, A##ci.s##t, s1.s##ci);
#endif

#if COOPK_COLS == 4
#define COOPK_DOTS_AT(t)  COOPK_DOT(0,t) COOPK_DOT(1,t) COOPK_DOT(2,t) COOPK_DOT(3,t)
#define COOPK_FOR_COLS(F) F(0) F(1) F(2) F(3)
#else
#define COOPK_DOTS_AT(t)  COOPK_DOT(0,t) COOPK_DOT(1,t)
#define COOPK_FOR_COLS(F) F(0) F(1)
#endif

// The bin layout packs eight K values in each uint, with four nibbles in each 16-bit half.
// Since ku0 is a multiple of four, t / 2 selects the uint and t & 1 selects its half.
#ifdef COOPK_Q_BIN
#define COOPK_QBITS4(t) convert_ushort4(((t) & 1) ? (vload4(0, src0_q + row0 + ((ku0 >> 1) + ((t) >> 1)) * m) >> 16) \
                                                : (vload4(0, src0_q + row0 + ((ku0 >> 1) + ((t) >> 1)) * m) & 0xFFFFu))
#define COOPK_QBITS2(t) convert_ushort2(((t) & 1) ? (vload2(0, src0_q + row0 + ((ku0 >> 1) + ((t) >> 1)) * m) >> 16) \
                                                : (vload2(0, src0_q + row0 + ((ku0 >> 1) + ((t) >> 1)) * m) & 0xFFFFu))
#else
#define COOPK_QBITS4(t) vload4(0, src0_q + row0 + (ku0 + t) * m)
#define COOPK_QBITS2(t) vload2(0, src0_q + row0 + (ku0 + t) * m)
#endif

#if COOPK_ROWS == 4
#define COOPK_KSTEP(t)              \
    {                               \
    ushort4 bits = COOPK_QBITS4(t); \
    const uint w0 = EXP4(bits.s0);  \
    const uint w1 = EXP4(bits.s1);  \
    const uint w2 = EXP4(bits.s2);  \
    const uint w3 = EXP4(bits.s3);  \
    COOPK_DOTS_AT(t)                \
    }
#else
#define COOPK_KSTEP(t)              \
    {                               \
    ushort2 bits = COOPK_QBITS2(t); \
    const uint w0 = EXP4(bits.s0);  \
    const uint w1 = EXP4(bits.s1);  \
    COOPK_DOTS_AT(t)                \
    }
#endif

inline void get_scale_min_k4_c(int j, global const uchar * q, int stride,
                               uchar * d, uchar * m,
                               uchar mask_d6, uchar mask_d4, uchar mask_hi2) {
    if (j < 4) {
        *d = q[j*stride] & mask_d6;
        *m = q[(j + 4)*stride] & mask_d6;
    } else {
        *d = (q[(j + 4)*stride] & mask_d4) | ((q[(j - 4)*stride] >> 6) << 4);
        *m = (q[(j + 4)*stride] >>   4)    | ((q[(j    )*stride] >> 6) << 4);
    }
}

// Weights use the untiled Adreno noshuffle SoA layout; COOPK_Q_BIN selects uint-transposed q instead of ushort-transposed q.
// Q8_1 activation planes are K-major (K contiguous); no activation transpose is used.
kernel void kernel_gemm_coopk_q4_k_q8_1_dp4a(
#ifdef COOPK_Q_BIN
    global const uint   * src0_q,     // eight consecutive K nibbles per uint, [k/8][m]
#else
    global const ushort * src0_q,     // four consecutive K nibbles per ushort, [k/4][m]
#endif
    global const uchar  * src0_s,     // packed 6-bit scales/mins, [k/256][12][m]
    global const half   * src0_d,     // super-block scales, [k/256][m]
    global const half   * src0_dm,    // super-block mins, [k/256][m]
    global const uint   * src1_qa,    // four signed activation bytes per uint, [n][k/4]
    global const half   * src1_da,    // activation scales, [n][k/32]
    global const half   * src1_sa,    // scaled activation sums, [n][k/32]
    global float * dst,
    ulong offsetd,
    int m,
    int n,
    int k,
    int n_no_padding,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2
) {
    dst = (global float *)((global char *)dst + offsetd);

    const int gx   = get_global_id(0);   // row group
    const int sg   = get_local_id(1);    // K-split subgroup
    const int lane = get_local_id(0);

    const int row0 = gx * COOPK_ROWS;
    const int num_32blk = k / 32;
    const int k_u = k >> 2;              // K in uint (int8x4) units
    const int k_b = k >> 5;              // 32-blocks along K

    // The host pads activations to COOPK_COLS; stores mask columns beyond n_no_padding.
    const int nl = n - 1;
    const int c0 = 0;
    const int c1 = (1 < n) ? 1 : nl;
#if COOPK_COLS >= 4
    const int c2 = (2 < n) ? 2 : nl;
    const int c3 = (3 < n) ? 3 : nl;
#endif

    coopk_accv acc0 = (coopk_accv)(0.0f), acc1 = (coopk_accv)(0.0f);
#if COOPK_ROWS == 4
    coopk_accv acc2 = (coopk_accv)(0.0f), acc3 = (coopk_accv)(0.0f);
#endif

    for (int blk = sg; blk < num_32blk; blk += COOPK_NSG) {
        const int i       = blk << 5;
        const int sb_idx  = blk >> 3;
        const int sub_idx = blk & 7;

        global const uchar * sc = src0_s + sb_idx * K_SCALE_SIZE * m + row0;
        uchar sv0, mn0, sv1, mn1;
        get_scale_min_k4_c(sub_idx, sc + 0, m, &sv0, &mn0, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4_c(sub_idx, sc + 1, m, &sv1, &mn1, mask_d6, mask_d4, mask_hi2);
#if COOPK_ROWS == 4
        half4 dd  = vload4(0, src0_d  + row0 + sb_idx * m);
        half4 dmm = vload4(0, src0_dm + row0 + sb_idx * m);
        uchar sv2, mn2, sv3, mn3;
        get_scale_min_k4_c(sub_idx, sc + 2, m, &sv2, &mn2, mask_d6, mask_d4, mask_hi2);
        get_scale_min_k4_c(sub_idx, sc + 3, m, &sv3, &mn3, mask_d6, mask_d4, mask_hi2);
        const float sc2 = (float)dd.s2 * (float)sv2, mv2 = (float)dmm.s2 * (float)mn2;
        const float sc3 = (float)dd.s3 * (float)sv3, mv3 = (float)dmm.s3 * (float)mn3;
#else
        half2 dd  = vload2(0, src0_d  + row0 + sb_idx * m);
        half2 dmm = vload2(0, src0_dm + row0 + sb_idx * m);
#endif
        const float sc0 = (float)dd.s0 * (float)sv0, mv0 = (float)dmm.s0 * (float)mn0;
        const float sc1 = (float)dd.s1 * (float)sv1, mv1 = (float)dmm.s1 * (float)mn1;

        // Q4_K scale and min values change every 32 K values.
        coopk_dotv s0 = (coopk_dotv)(0), s1 = (coopk_dotv)(0);
#if COOPK_ROWS == 4
        coopk_dotv s2 = (coopk_dotv)(0), s3 = (coopk_dotv)(0);
#endif

        for (int uq = 0; uq < 2; ++uq) {
            const int ku0 = (i >> 2) + uq * 4;

            uint4 A0 = vload4(0, src1_qa + (uint)c0 * k_u + ku0);
            uint4 A1 = vload4(0, src1_qa + (uint)c1 * k_u + ku0);
#if COOPK_COLS >= 4
            uint4 A2 = vload4(0, src1_qa + (uint)c2 * k_u + ku0);
            uint4 A3 = vload4(0, src1_qa + (uint)c3 * k_u + ku0);
#endif
            COOPK_KSTEP(0)
            COOPK_KSTEP(1)
            COOPK_KSTEP(2)
            COOPK_KSTEP(3)
        }

        // Q4_K stores q * scale - min for each 32-value block.
        // src1_sa already includes the activation scale, so the min term needs no extra factor of da.
        coopk_accv da, sa;
        da.s0 = (float)src1_da[c0*k_b + blk];  sa.s0 = (float)src1_sa[c0*k_b + blk];
        da.s1 = (float)src1_da[c1*k_b + blk];  sa.s1 = (float)src1_sa[c1*k_b + blk];
#if COOPK_COLS >= 4
        da.s2 = (float)src1_da[c2*k_b + blk];  sa.s2 = (float)src1_sa[c2*k_b + blk];
        da.s3 = (float)src1_da[c3*k_b + blk];  sa.s3 = (float)src1_sa[c3*k_b + blk];
#endif

        acc0 += sc0 * da * COOPK_CONVF(s0) - mv0 * sa;
        acc1 += sc1 * da * COOPK_CONVF(s1) - mv1 * sa;
#if COOPK_ROWS == 4
        acc2 += sc2 * da * COOPK_CONVF(s2) - mv2 * sa;
        acc3 += sc3 * da * COOPK_CONVF(s3) - mv3 * sa;
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
