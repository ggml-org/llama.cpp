#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_kernel_cooperative_matrix : enable

// MoE q6_k x f32 GEMM with cl_qcom_kernel_cooperative_matrix: m64n32k16, f16 in, f32 accumulate.
// One wave computes 64 weight rows x one tile of 32 token slots, like kernel_gemm_moe_q6_k_f32_ns.
// A is the transposed q6_k weight (same layout as the ns kernel), dequantized in registers.
// B is the f16 copy of the reordered activations written by kernel_moe_reorder_b_f16 (moe_reorder_b.cl).
// A is dequantized in registers, staged through local memory (one row per lane) and loaded with qcom_coop_mat_load;
// C is stored with qcom_coop_mat_store to local memory and read back one row per lane. B is loaded from global memory.
// Keep this kernel alone in the program: with another kernel before it, the compiler gave wrong results.

typedef half  mat_a_t __attribute__((qcom_coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP_QCOM, 64, 16, CLK_COOPERATIVE_MATRIX_A_QCOM)));
typedef half  mat_b_t __attribute__((qcom_coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP_QCOM, 16, 32, CLK_COOPERATIVE_MATRIX_B_QCOM)));
typedef float mat_c_t __attribute__((qcom_coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP_QCOM, 64, 32, CLK_COOPERATIVE_MATRIX_ACCUMULATOR_QCOM)));

// 8 q6 values (low nibbles in v, 2 high bits each in the low 16 bits of h) -> (q - 32) * scale in k order 0 4 1 5 2 6 3 7.
// Computed in f32 and rounded once to f16.
inline half8 q6_k_to_f16x8(uint v, uint h, float scale) {
    uint4 t;
    t.s0 = ( v        & 0x000F000F) | ((h << 4) & 0x30) | ((h << 12) & 0x00300000);
    t.s1 = ((v >>  4) & 0x000F000F) | ((h << 2) & 0x30) | ((h << 10) & 0x00300000);
    t.s2 = ((v >>  8) & 0x000F000F) | ( h       & 0x30) | ((h <<  8) & 0x00300000);
    t.s3 = ((v >> 12) & 0x000F000F) | ((h >> 2) & 0x30) | ((h <<  6) & 0x00300000);
    t |= (uint4)(0x64006400);
    return convert_half8((convert_float8(as_half8(t)) - 1056.0f) * scale);
}

// loads of one K step of 32 for row rl into the *n variables
#define LOAD_A(kk) { \
        const uint sub = (kk) / 32; \
        const uint sb  = sub / 8; \
        const uint jj  = sub % 8; \
        const uint nsb = ne00 / 256; \
        const uint dof = rl + sb * ne01 + expert * nsb * ne01; \
        const uint qo = expert * (ne00 / 8) * ne01 + rl + ((kk) / 8) * ne01; \
        qn.s0 = read_imageui(src0_ql, (int)(qo           )).x; \
        qn.s1 = read_imageui(src0_ql, (int)(qo +     ne01)).x; \
        qn.s2 = read_imageui(src0_ql, (int)(qo + 2 * ne01)).x; \
        qn.s3 = read_imageui(src0_ql, (int)(qo + 3 * ne01)).x; \
        const float dv = (float)src0_d[dof]; \
        global const char * scp = src0_s + (expert * ne01 + rl) * (nsb * 16) + sb * 16; \
        s0n = dv * (float)scp[jj * 2]; \
        s1n = dv * (float)scp[jj * 2 + 1]; \
        const uint qhb = rl + (sub * 2) * ne01 + expert * (nsb * 16) * ne01; \
        h0n = src0_qh[qhb]; \
        h1n = src0_qh[qhb + ne01]; }

#define STORE_C(v, idx) { \
        if ((idx) != 0xFFFFFFFFu && wr) { \
            write_imagef(dst, (int)((idx) * ne01 + row + lane), (v)); \
        } }

#define STORE_C16(u, ix) \
    STORE_C(u.s0, ix.s0) STORE_C(u.s1, ix.s1) STORE_C(u.s2, ix.s2) STORE_C(u.s3, ix.s3) \
    STORE_C(u.s4, ix.s4) STORE_C(u.s5, ix.s5) STORE_C(u.s6, ix.s6) STORE_C(u.s7, ix.s7) \
    STORE_C(u.s8, ix.s8) STORE_C(u.s9, ix.s9) STORE_C(u.sa, ix.sa) STORE_C(u.sb, ix.sb) \
    STORE_C(u.sc, ix.sc) STORE_C(u.sd, ix.sd) STORE_C(u.se, ix.se) STORE_C(u.sf, ix.sf)

// Same arguments as kernel_gemm_moe_q6_k_f32_ns, except src1 is the f16 buffer of kernel_moe_reorder_b_f16.
// global size (64, ne01/64 rounded up, max tiles), local size (64, 1, 1)
__attribute__((reqd_work_group_size(64, 1, 1)))
kernel void kernel_gemm_moe_q6_k_f32_ns_coopmat(
        __read_only  image1d_buffer_t src0_ql,   // A: [E][K/8][M] u32, 8 low nibbles in natural k order
        __global     const uint *     src0_qh,   // A: [E][K/16][M] u32, 2 high bits of each of the 16 k
        __global     const char *     src0_s,    // A: [E][M][K/256][16] int8 scales, one per 16 k
        __global     const half *     src0_d,    // A: [E][K/256][M] f16
        __global     const half *     src1,      // B: [slot][K] f16
        __global     uint *           src2,      // token slot -> dst row, 0xFFFFFFFF for padding
        __global     ushort *         src2_emap, // tile -> expert
        __write_only image1d_buffer_t dst,
        __global     int *            total_tiles,
        uint ne00,
        uint ne01,
        uint is_ragged,
        uint skip_gran
) {
    const uint lane = get_local_id(0);
    const uint row  = get_global_id(1) * 64;
    const uint tile = get_global_id(2);

    if (tile >= (uint)total_tiles[0]) {
        return;
    }

    const uint expert = src2_emap[tile];
    const uint col    = tile * 32;
    // rows past ne01 (ne01 % 64 == 32) load the last row and are not stored
    const uint rl     = min(row + lane, ne01 - 1);

    // dst rows of the tile, read back as vectors at the store (uniform global loads there are slow)
    local uint out_idx[32];
    if (lane < 32) {
        out_idx[lane] = src2[col + lane];
    }

    global const half * bp = src1 + (ulong)col * ne00;

    local half  la[2][64 * 16];  // A tiles, row-major, ld 16
    local float lc[64 * 32];     // C tile, row-major, ld 32

    mat_c_t acc = qcom_coop_mat_init(0.0f);

    // the loads of the next K step are issued before the dequant and MMA of the current one
    uint4 qn; float s0n; float s1n; uint h0n; uint h1n;
    LOAD_A(0)

#pragma unroll 1
    for (uint k0 = 0; k0 < ne00; k0 += 32) {
        const uint4 q = qn; const float sc0 = s0n; const float sc1 = s1n; const uint hq0 = h0n; const uint hq1 = h1n;
        if (k0 + 32 < ne00) {
            LOAD_A(k0 + 32)
        }

        half16 h0 = (half16)(q6_k_to_f16x8(q.s0, hq0, sc0), q6_k_to_f16x8(q.s1, hq0 >> 16, sc0));
        half16 h1 = (half16)(q6_k_to_f16x8(q.s2, hq1, sc1), q6_k_to_f16x8(q.s3, hq1 >> 16, sc1));
        vstore16(h0, 0, la[0] + lane * 16);
        vstore16(h1, 0, la[1] + lane * 16);
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        mat_a_t a0 = qcom_coop_mat_load(la[0], CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_QCOM, 16);
        mat_a_t a1 = qcom_coop_mat_load(la[1], CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_QCOM, 16);
        sub_group_barrier(CLK_LOCAL_MEM_FENCE);
        mat_b_t b0 = qcom_coop_mat_load(bp + k0,      CLK_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR_QCOM, ne00);
        mat_b_t b1 = qcom_coop_mat_load(bp + k0 + 16, CLK_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR_QCOM, ne00);
        acc = qcom_coop_mat_mulAdd(a0, b0, acc, CLK_COOPERATIVE_MATRIX_OPERAND_NONE_QCOM);
        acc = qcom_coop_mat_mulAdd(a1, b1, acc, CLK_COOPERATIVE_MATRIX_OPERAND_NONE_QCOM);
    }

    qcom_coop_mat_store(lc, acc, CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_QCOM, 32);

    // the work-group is one subgroup; a full barrier is ~5% slower
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    const uint16 ix0 = vload16(0, out_idx);
    const uint16 ix1 = vload16(1, out_idx);

    const bool wr = row + lane < ne01;
    const float16 c0 = vload16(0, lc + lane * 32);
    const float16 c1 = vload16(1, lc + lane * 32);
    STORE_C16(c0, ix0)
    STORE_C16(c1, ix1)
}
