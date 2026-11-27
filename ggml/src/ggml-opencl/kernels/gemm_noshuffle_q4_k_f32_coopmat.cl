#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_kernel_cooperative_matrix : enable

// Q4_K x F32 GEMM with cl_qcom_kernel_cooperative_matrix: one 64x64 f16 C tile per wave,
// A dequantized in registers, B packed by kernel_pack_b_f16_coopmat.

#define K_SCALE_SIZE 12

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

inline half8 dequant8(uint v, half scale, half nminv) {
    uint4 t;
    t.s0 = ( v        & 0x000F) | ((v << 12) & 0x000F0000);
    t.s1 = ((v >>  8) & 0x000F) | ((v <<  4) & 0x000F0000);
    t.s2 = ((v >> 16) & 0x000F) | ((v >>  4) & 0x000F0000);
    t.s3 = ((v >> 24) & 0x000F) | ((v >> 12) & 0x000F0000);
    t |= (uint4)(0x64006400);

    return (as_half8(t) - (half8)(1024.0h)) * scale + nminv;
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

#define MMA_K32(q0, q1, q2, q3, sc, nm, b) { \
        a_row_t ua0; \
        a_row_t ua1; \
        ua0.h = (half16)(dequant8(q0, sc, nm), dequant8(q1, sc, nm)); \
        ua1.h = (half16)(dequant8(q2, sc, nm), dequant8(q3, sc, nm)); \
        mat_a_t a0 = __builtin_astype(ua0.v, mat_a_t); \
        mat_a_t a1 = __builtin_astype(ua1.v, mat_a_t); \
        mat_b_t b0 = qcom_coop_mat_load((b),        CLK_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR_QCOM, 16); \
        mat_b_t b1 = qcom_coop_mat_load((b) + 1024, CLK_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR_QCOM, 16); \
        acc = qcom_coop_mat_mulAdd(a0, b0, acc, CLK_COOPERATIVE_MATRIX_OPERAND_NONE_QCOM); \
        acc = qcom_coop_mat_mulAdd(a1, b1, acc, CLK_COOPERATIVE_MATRIX_OPERAND_NONE_QCOM); }

#define STEP64(sa, na, sb_, nb_) { \
        uint8 qv; \
        qv.s0 = qp[0]; \
        qv.s1 = qp[M]; \
        qv.s2 = qp[2 * M]; \
        qv.s3 = qp[3 * M]; \
        qv.s4 = qp[4 * M]; \
        qv.s5 = qp[5 * M]; \
        qv.s6 = qp[6 * M]; \
        qv.s7 = qp[7 * M]; \
        qp += 8 * M; \
        MMA_K32(qv.s0, qv.s1, qv.s2, qv.s3, sa, na, bp) \
        MMA_K32(qv.s4, qv.s5, qv.s6, qv.s7, sb_, nb_, bp + 2048) \
        bp += 4096; }

kernel void kernel_gemm_noshuffle_q4_k_f32_coopmat(
        global const uint  * src0_q,   // A: q4_K quants,         [K/8][M] u32
        global const uchar * src0_s,   // A: 6-bit scale/min codes, [K/256*12][M] u8
        global const half  * src0_d,   // A: per-superblock scale, [K/256][M] f16
        global const half  * src0_dm,  // A: per-superblock min,   [K/256][M] f16
        global const half  * src1,     // B: packed f16 tiles
        global       char  * dst,      // C: [N][M] f32
        ulong                offsetd,
        int                  K,
        int                  M,
        int                  N
) {
    const int lane    = get_local_id(0);
    const int row     = get_global_id(1) * 64;
    const int tn      = get_global_id(2);
    const int n_tiles = (N + 63) / 64;

    if (tn >= n_tiles) {
        return;
    }

    const int lrow = row + lane;

    global const uint * qp = src0_q + lrow;
    global const half * bp = src1 + (ulong)tn * (K / 16) * 1024;

    mat_c_t acc = qcom_coop_mat_init(0.0);

#pragma unroll 1
    for (int k0 = 0; k0 < K; k0 += 128) {
        const int  sb  = k0 >> 8;
        const half dd  = src0_d [(ulong)sb * M + lrow];
        const half dmm = src0_dm[(ulong)sb * M + lrow];

        global const uchar * sp = src0_s + (ulong)(sb * K_SCALE_SIZE) * M + lrow;
        const uchar4 c0 = (uchar4)(sp[0],     sp[M],     sp[2 * M],  sp[3 * M]);
        const uchar4 c1 = (uchar4)(sp[4 * M], sp[5 * M], sp[6 * M],  sp[7 * M]);
        const uchar4 c2 = (uchar4)(sp[8 * M], sp[9 * M], sp[10 * M], sp[11 * M]);
        const uchar4 sl = c0 & (uchar)0x3F;
        const uchar4 ml = c1 & (uchar)0x3F;
        const uchar4 sh = (c2 & (uchar)0x0F) | ((c0 >> (uchar)2) & (uchar)0x30);
        const uchar4 mh = (c2 >> (uchar)4)   | ((c1 >> (uchar)2) & (uchar)0x30);
        const int    hi = (k0 >> 7) & 1;
        const half4  S  = dd * convert_half4(hi ? sh : sl);
        const half4  Nm = -(dmm * convert_half4(hi ? mh : ml));

        STEP64(S.s0, Nm.s0, S.s1, Nm.s1)
        STEP64(S.s2, Nm.s2, S.s3, Nm.s3)
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
