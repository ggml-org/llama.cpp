#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_kernel_cooperative_matrix : enable

// Q5_0 x F32 GEMM with cl_qcom_kernel_cooperative_matrix: one 64x64 f16 C tile per wave,
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

inline half8 dequant8(uint v, uint h, half d) {
    uint4 t;
    t.s0 = ( v        & 0x000F) | ((v << 12) & 0x000F0000) | ((h << 4) & 0x0010) | ((h << 19) & 0x00100000);
    t.s1 = ((v >>  8) & 0x000F) | ((v <<  4) & 0x000F0000) | ((h << 2) & 0x0010) | ((h << 17) & 0x00100000);
    t.s2 = ((v >> 16) & 0x000F) | ((v >>  4) & 0x000F0000) | ( h       & 0x0010) | ((h << 15) & 0x00100000);
    t.s3 = ((v >> 24) & 0x000F) | ((v >> 12) & 0x000F0000) | ((h >> 2) & 0x0010) | ((h << 13) & 0x00100000);
    t |= (uint4)(0x64006400);

    return (as_half8(t) - (half8)(1040.0h)) * d;
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

#define LOAD_Q(qv, hv, dd, o) { \
        qv.s0 = (uint)qp[(8 * (o) + 0) * M] | ((uint)qp[(8 * (o) + 1) * M] << 16); \
        qv.s1 = (uint)qp[(8 * (o) + 2) * M] | ((uint)qp[(8 * (o) + 3) * M] << 16); \
        qv.s2 = (uint)qp[(8 * (o) + 4) * M] | ((uint)qp[(8 * (o) + 5) * M] << 16); \
        qv.s3 = (uint)qp[(8 * (o) + 6) * M] | ((uint)qp[(8 * (o) + 7) * M] << 16); \
        hv.s0 = hp[(4 * (o) + 0) * M]; \
        hv.s1 = hp[(4 * (o) + 1) * M]; \
        hv.s2 = hp[(4 * (o) + 2) * M]; \
        hv.s3 = hp[(4 * (o) + 3) * M]; \
        dd    = dp[(o) * M]; }

#define MMA_K32(qv, hv, dd, b) { \
        a_row_t ua0; \
        a_row_t ua1; \
        ua0.h = (half16)(dequant8(qv.s0, hv.s0, dd), dequant8(qv.s1, hv.s1, dd)); \
        ua1.h = (half16)(dequant8(qv.s2, hv.s2, dd), dequant8(qv.s3, hv.s3, dd)); \
        mat_a_t a0 = __builtin_astype(ua0.v, mat_a_t); \
        mat_a_t a1 = __builtin_astype(ua1.v, mat_a_t); \
        mat_b_t b0 = qcom_coop_mat_load((b),        CLK_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR_QCOM, 16); \
        mat_b_t b1 = qcom_coop_mat_load((b) + 1024, CLK_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR_QCOM, 16); \
        acc = qcom_coop_mat_mulAdd(a0, b0, acc, CLK_COOPERATIVE_MATRIX_OPERAND_NONE_QCOM); \
        acc = qcom_coop_mat_mulAdd(a1, b1, acc, CLK_COOPERATIVE_MATRIX_OPERAND_NONE_QCOM); }

kernel void kernel_gemm_noshuffle_q5_0_f32_coopmat(
        global const ushort * src0_q,  // A: q5_0 quants, [K/4][M] u16
        global const uchar  * src0_qh, // A: q5_0 high bits, [K/8][M] u8
        global const half   * src0_d,  // A: q5_0 scales, [K/32][M] f16
        global const half   * src1,    // B: packed f16 tiles
        global       char   * dst,     // C: [N][M] f32
        ulong                 offsetd,
        int                   K,
        int                   M,
        int                   N
) {
    const int lane    = get_local_id(0);
    const int row     = get_global_id(1) * 64;
    const int tn      = get_global_id(2);
    const int n_tiles = (N + 63) / 64;

    if (tn >= n_tiles) {
        return;
    }

    global const ushort * qp = src0_q  + row + lane;
    global const uchar  * hp = src0_qh + row + lane;
    global const half   * dp = src0_d  + row + lane;
    global const half   * bp = src1 + (ulong)tn * (K / 16) * 1024;

    mat_c_t acc = qcom_coop_mat_init(0.0);

    {
        const int odd = (K / 32) & 1;
        const int ob  = odd ? 0 : 1;
        uint4 qa;
        uint4 qb;
        uint4 ha;
        uint4 hb;
        half  da;
        half  db;
        LOAD_Q(qa, ha, da, 0)
        LOAD_Q(qb, hb, db, ob)
        da = odd ? 0.0h : da;
        MMA_K32(qa, ha, da, bp)
        MMA_K32(qb, hb, db, bp + ob * 2048)
        qp += (ob + 1) * 8 * M;
        hp += (ob + 1) * 4 * M;
        dp += (ob + 1) * M;
        bp += (ob + 1) * 2048;
    }

#pragma unroll 1
    for (int k0 = (K / 32) & 1 ? 32 : 64; k0 < K; k0 += 64) {
        uint4 qa;
        uint4 qb;
        uint4 ha;
        uint4 hb;
        half  da;
        half  db;
        LOAD_Q(qa, ha, da, 0)
        LOAD_Q(qb, hb, db, 1)
        MMA_K32(qa, ha, da, bp)
        MMA_K32(qb, hb, db, bp + 2048)
        qp += 16 * M;
        hp +=  8 * M;
        dp +=  2 * M;
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
