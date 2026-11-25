#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_kernel_cooperative_matrix : enable

// MoE q4_0 x f32 GEMM with cl_qcom_kernel_cooperative_matrix: m64n32k16, f16 in, f32 accumulate.
// One wave computes 64 weight rows x one tile of 32 token slots, like kernel_gemm_moe_q4_0_f32_ns.
// A is the transposed q4_0 weight of kernel_convert_block_q4_0_trans4_ns, dequantized in registers.
// B is the f16 copy of the reordered activations written by kernel_moe_reorder_b_f16 (moe_reorder_b.cl).
// A is dequantized in registers, staged through local memory (one row per lane) and loaded with qcom_coop_mat_load;
// C is stored with qcom_coop_mat_store to local memory and read back one row per lane. B is loaded from global memory.
// Keep this kernel alone in the program: with another kernel before it, the compiler gave wrong results.

typedef half  mat_a_t __attribute__((qcom_coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP_QCOM, 64, 16, CLK_COOPERATIVE_MATRIX_A_QCOM)));
typedef half  mat_b_t __attribute__((qcom_coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP_QCOM, 16, 32, CLK_COOPERATIVE_MATRIX_B_QCOM)));
typedef float mat_c_t __attribute__((qcom_coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP_QCOM, 64, 32, CLK_COOPERATIVE_MATRIX_ACCUMULATOR_QCOM)));

// 8 nibbles -> (q - 8) * d in k order 0 4 1 5 2 6 3 7 (kernel_moe_reorder_b_f16 writes B in the same order).
// 0x6400 | q is f16 1024 + q, so the subtract is exact.
inline half8 q4_0_to_f16x8(uint v, half d) {
    uint4 t;
    t.s0 =  v        & 0x000F000F;
    t.s1 = (v >>  4) & 0x000F000F;
    t.s2 = (v >>  8) & 0x000F000F;
    t.s3 = (v >> 12) & 0x000F000F;
    t |= (uint4)(0x64006400);
    return (as_half8(t) - (half8)(1032.0h)) * d;
}

// loads of one K step of 32 (one q4_0 block) for row rl: 4 u32 of quants and 1 scale
#define LOAD_A(qv, dv) { \
        qv.s0 = read_imageui(src0_q, (int)(qoff           )).x; \
        qv.s1 = read_imageui(src0_q, (int)(qoff +     ne01)).x; \
        qv.s2 = read_imageui(src0_q, (int)(qoff + 2 * ne01)).x; \
        qv.s3 = read_imageui(src0_q, (int)(qoff + 3 * ne01)).x; \
        dv    = src0_d[doff]; \
        qoff += 4 * ne01; \
        doff += ne01; }

#define STORE_C(v, idx) { \
        if ((idx) != 0xFFFFFFFFu && wr) { \
            write_imagef(dst, (int)((idx) * ne01 + row + lane), (v)); \
        } }

#define STORE_C16(u, ix) \
    STORE_C(u.s0, ix.s0) STORE_C(u.s1, ix.s1) STORE_C(u.s2, ix.s2) STORE_C(u.s3, ix.s3) \
    STORE_C(u.s4, ix.s4) STORE_C(u.s5, ix.s5) STORE_C(u.s6, ix.s6) STORE_C(u.s7, ix.s7) \
    STORE_C(u.s8, ix.s8) STORE_C(u.s9, ix.s9) STORE_C(u.sa, ix.sa) STORE_C(u.sb, ix.sb) \
    STORE_C(u.sc, ix.sc) STORE_C(u.sd, ix.sd) STORE_C(u.se, ix.se) STORE_C(u.sf, ix.sf)

// Same arguments as kernel_gemm_moe_q4_0_f32_ns, except src1 is the f16 buffer of kernel_moe_reorder_b_f16.
// global size (64, ne01/64 rounded up, max tiles), local size (64, 1, 1)
__attribute__((reqd_work_group_size(64, 1, 1)))
kernel void kernel_gemm_moe_q4_0_f32_ns_coopmat(
        __read_only  image1d_buffer_t src0_q,    // A: [E][K/8][M] u32, 8 nibbles in natural k order
        __global     const half *     src0_d,    // A: [E][K/32][M] f16
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

    uint qoff = expert * (ne00 / 8)  * ne01 + rl;
    uint doff = expert * (ne00 / 32) * ne01 + rl;
    global const half * bp = src1 + (ulong)col * ne00;

    local half  la[2][64 * 16];  // A tiles, row-major, ld 16
    local float lc[64 * 32];     // C tile, row-major, ld 32

    mat_c_t acc = qcom_coop_mat_init(0.0f);

    // the loads of the next K step are issued before the dequant and MMA of the current one
    uint4 qn;
    half  dn;
    LOAD_A(qn, dn)

#pragma unroll 1
    for (uint k0 = 0; k0 < ne00; k0 += 32) {
        const uint4 q = qn;
        const half  d = dn;
        if (k0 + 32 < ne00) {
            LOAD_A(qn, dn)
        }

        vstore16((half16)(q4_0_to_f16x8(q.s0, d), q4_0_to_f16x8(q.s1, d)), 0, la[0] + lane * 16);
        vstore16((half16)(q4_0_to_f16x8(q.s2, d), q4_0_to_f16x8(q.s3, d)), 0, la[1] + lane * 16);
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
