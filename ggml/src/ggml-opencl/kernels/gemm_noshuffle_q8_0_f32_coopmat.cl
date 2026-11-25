#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_kernel_cooperative_matrix : enable

// Q8_0 x F32 GEMM with cl_qcom_kernel_cooperative_matrix: one 64x64 f16 C tile per wave,
// A dequantized in registers, B packed by kernel_pack_b_f16_coopmat.

typedef half mat_a_t __attribute__((qcom_coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP_QCOM, 64, 16, CLK_COOPERATIVE_MATRIX_A_QCOM)));
typedef half mat_b_t __attribute__((qcom_coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP_QCOM, 16, 64, CLK_COOPERATIVE_MATRIX_B_QCOM)));
typedef half mat_c_t __attribute__((qcom_coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP_QCOM, 64, 64, CLK_COOPERATIVE_MATRIX_ACCUMULATOR_QCOM)));

typedef half half1024 __attribute__((ext_vector_type(1024)));
typedef half half4096 __attribute__((ext_vector_type(4096)));

typedef union {
    half1024 v;
    half16   h;
} a_row_t;

typedef union {
    half4096 v;
    struct {
        half16 h0;
        half16 h1;
        half16 h2;
        half16 h3;
    } s;
} c_row_t;

inline half4 dequant4(uint v, half d) {
    return convert_half4(as_char4(v)) * d;
}

inline void store_c16(global float * p, half16 h, int j0, int nv, int M) {
    float16 f = convert_float16(h);
    if (j0 +  0 < nv) p[(ulong)(j0 +  0) * M] = f.s0;
    if (j0 +  1 < nv) p[(ulong)(j0 +  1) * M] = f.s1;
    if (j0 +  2 < nv) p[(ulong)(j0 +  2) * M] = f.s2;
    if (j0 +  3 < nv) p[(ulong)(j0 +  3) * M] = f.s3;
    if (j0 +  4 < nv) p[(ulong)(j0 +  4) * M] = f.s4;
    if (j0 +  5 < nv) p[(ulong)(j0 +  5) * M] = f.s5;
    if (j0 +  6 < nv) p[(ulong)(j0 +  6) * M] = f.s6;
    if (j0 +  7 < nv) p[(ulong)(j0 +  7) * M] = f.s7;
    if (j0 +  8 < nv) p[(ulong)(j0 +  8) * M] = f.s8;
    if (j0 +  9 < nv) p[(ulong)(j0 +  9) * M] = f.s9;
    if (j0 + 10 < nv) p[(ulong)(j0 + 10) * M] = f.sa;
    if (j0 + 11 < nv) p[(ulong)(j0 + 11) * M] = f.sb;
    if (j0 + 12 < nv) p[(ulong)(j0 + 12) * M] = f.sc;
    if (j0 + 13 < nv) p[(ulong)(j0 + 13) * M] = f.sd;
    if (j0 + 14 < nv) p[(ulong)(j0 + 14) * M] = f.se;
    if (j0 + 15 < nv) p[(ulong)(j0 + 15) * M] = f.sf;
}

#define LOAD_Q(qv, dd) { \
        qv.s0 = qp[0]; \
        qv.s1 = qp[M]; \
        qv.s2 = qp[2 * M]; \
        qv.s3 = qp[3 * M]; \
        qv.s4 = qp[4 * M]; \
        qv.s5 = qp[5 * M]; \
        qv.s6 = qp[6 * M]; \
        qv.s7 = qp[7 * M]; \
        dd    = dp[0]; \
        qp += 8 * M; \
        dp += M; }

#define MMA_K32(qv, dd, b) { \
        a_row_t ua0; \
        a_row_t ua1; \
        ua0.h = (half16)(dequant4(qv.s0, dd), dequant4(qv.s1, dd), dequant4(qv.s2, dd), dequant4(qv.s3, dd)); \
        ua1.h = (half16)(dequant4(qv.s4, dd), dequant4(qv.s5, dd), dequant4(qv.s6, dd), dequant4(qv.s7, dd)); \
        mat_a_t a0 = __builtin_astype(ua0.v, mat_a_t); \
        mat_a_t a1 = __builtin_astype(ua1.v, mat_a_t); \
        mat_b_t b0 = qcom_coop_mat_load((b),        CLK_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR_QCOM, 16); \
        mat_b_t b1 = qcom_coop_mat_load((b) + 1024, CLK_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR_QCOM, 16); \
        acc = qcom_coop_mat_mulAdd(a0, b0, acc, CLK_COOPERATIVE_MATRIX_OPERAND_NONE_QCOM); \
        acc = qcom_coop_mat_mulAdd(a1, b1, acc, CLK_COOPERATIVE_MATRIX_OPERAND_NONE_QCOM); }

kernel void kernel_gemm_noshuffle_q8_0_f32_coopmat(
        global const uint * src0_q,  // A: q8_0 quants, [K/4][M] u32
        global const half * src0_d,  // A: q8_0 scales, [K/32][M] f16
        global const half * src1,    // B: packed f16 tiles
        global       char * dst,     // C: [N][M] f32
        ulong               offsetd,
        int                 K,
        int                 M,
        int                 N
) {
    const int lane    = get_local_id(0);
    const int row     = get_global_id(1) * 64;
    const int tn      = get_global_id(2);
    const int n_tiles = (N + 63) / 64;

    if (tn >= n_tiles) {
        return;
    }

    global const uint * qp = src0_q + row + lane;
    global const half * dp = src0_d + row + lane;
    global const half * bp = src1 + (ulong)tn * (K / 16) * 1024;

    mat_c_t acc = qcom_coop_mat_init(0.0);

    {
        const int odd = (K / 32) & 1;
        const int kb  = odd ? 0 : 32;
        uint8 qa;
        uint8 qb;
        half  da;
        half  db;
        qa.s0 = qp[0];
        qa.s1 = qp[M];
        qa.s2 = qp[2 * M];
        qa.s3 = qp[3 * M];
        qa.s4 = qp[4 * M];
        qa.s5 = qp[5 * M];
        qa.s6 = qp[6 * M];
        qa.s7 = qp[7 * M];
        da    = odd ? 0.0h : dp[0];
        qb.s0 = qp[(kb / 4 + 0) * M];
        qb.s1 = qp[(kb / 4 + 1) * M];
        qb.s2 = qp[(kb / 4 + 2) * M];
        qb.s3 = qp[(kb / 4 + 3) * M];
        qb.s4 = qp[(kb / 4 + 4) * M];
        qb.s5 = qp[(kb / 4 + 5) * M];
        qb.s6 = qp[(kb / 4 + 6) * M];
        qb.s7 = qp[(kb / 4 + 7) * M];
        db    = dp[(kb / 32) * M];
        MMA_K32(qa, da, bp)
        MMA_K32(qb, db, bp + (kb / 16) * 1024)
        qp += (kb / 4 + 8) * M;
        dp += (kb / 32 + 1) * M;
        bp += (kb / 16 + 2) * 1024;
    }

#pragma unroll 1
    for (int k0 = (K / 32) & 1 ? 32 : 64; k0 < K; k0 += 64) {
        uint8 qa;
        uint8 qb;
        half  da;
        half  db;
        LOAD_Q(qa, da)
        LOAD_Q(qb, db)
        MMA_K32(qa, da, bp)
        MMA_K32(qb, db, bp + 2048)
        bp += 4096;
    }

    c_row_t uc;
    uc.v = __builtin_astype(acc, half4096);

    global float * p = (global float *)(dst + offsetd) + (ulong)tn * 64 * M + row + lane;
    const int nv = N - tn * 64;
    store_c16(p, uc.s.h0,  0, nv, M);
    store_c16(p, uc.s.h1, 16, nv, M);
    store_c16(p, uc.s.h2, 32, nv, M);
    store_c16(p, uc.s.h3, 48, nv, M);
}
