#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_qcom_kernel_cooperative_matrix : enable

// MoE mxfp4 x f32 GEMM with cl_qcom_kernel_cooperative_matrix: m64n32k16, f16 in, f32 accumulate.
// One wave computes 64 weight rows x one tile of 32 token slots, like kernel_gemm_moe_mxfp4_f32_ns.
// A is the transposed mxfp4 weight of kernel_convert_block_mxfp4_trans4_ns, dequantized in registers.
// B is the f16 copy of the reordered activations written by kernel_moe_reorder_b_f16 (moe_reorder_b.cl).
// A is dequantized in registers, staged through local memory (one row per lane) and loaded with qcom_coop_mat_load;
// C is stored with qcom_coop_mat_store to local memory and read back one row per lane. B is loaded from global memory.
// Keep this kernel alone in the program: with another kernel before it, the compiler gave wrong results.

typedef half  mat_a_t __attribute__((qcom_coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP_QCOM, 64, 16, CLK_COOPERATIVE_MATRIX_A_QCOM)));
typedef half  mat_b_t __attribute__((qcom_coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP_QCOM, 16, 32, CLK_COOPERATIVE_MATRIX_B_QCOM)));
typedef float mat_c_t __attribute__((qcom_coop_mat(CLK_COOPERATIVE_MATRIX_SCOPE_SUBGROUP_QCOM, 64, 32, CLK_COOPERATIVE_MATRIX_ACCUMULATOR_QCOM)));

// 8 E2M1 nibbles -> f16 in k order 0 4 1 5 2 6 3 7 (kernel_moe_reorder_b_f16 writes B in the same order).
// The bits s e1 e0 m go to f16 bits 15, 11, 10, 9: this is the E2M1 value times 2^-14 for all codes
// (e1 e0 = 0 gives the f16 subnormal), d holds the 2^14.
inline half8 mxfp4_to_f16x8(uint v, half d) {
    uint4 t;
    t.s0 = ((v << 9) & 0x0E000E00) | ((v << 12) & 0x80008000);
    t.s1 = ((v << 5) & 0x0E000E00) | ((v <<  8) & 0x80008000);
    t.s2 = ((v << 1) & 0x0E000E00) | ((v <<  4) & 0x80008000);
    t.s3 = ((v >> 3) & 0x0E000E00) | ( v        & 0x80008000);
    return as_half8(t) * d;
}

#define STORE_C(t, v) { \
        const uint idx = src2[col + (t)]; \
        if (idx != 0xFFFFFFFFu && wr) { \
            write_imagef(dst, (int)(idx * ne01 + row + lane), (v)); \
        } }

// Same arguments as kernel_gemm_moe_mxfp4_f32_ns, except src1 is the f16 buffer of kernel_moe_reorder_b_f16.
// global size (64, ne01/64 rounded up, max tiles), local size (64, 1, 1)
__attribute__((reqd_work_group_size(64, 1, 1)))
kernel void kernel_gemm_moe_mxfp4_f32_ns_coopmat(
        __read_only  image1d_buffer_t src0_q,    // A: [E][K/8][M] u32, 8 E2M1 in natural k order
        __global     uchar *          src0_d,    // A: [E][K/32][M] e8m0
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

    uint qoff = expert * (ne00 / 8)  * ne01 + rl;
    uint eoff = expert * (ne00 / 32) * ne01 + rl;
    global const half * bp = src1 + (ulong)col * ne00;

    local half  la[2][64 * 16];  // A tiles, row-major, ld 16
    local float lc[64 * 32];     // C tile, row-major, ld 32

    mat_c_t acc = qcom_coop_mat_init(0.0f);

#pragma unroll 1
    for (uint k0 = 0; k0 < ne00; k0 += 32) {
        uint4 q;
        q.s0 = read_imageui(src0_q, (int)(qoff           )).x;
        q.s1 = read_imageui(src0_q, (int)(qoff +     ne01)).x;
        q.s2 = read_imageui(src0_q, (int)(qoff + 2 * ne01)).x;
        q.s3 = read_imageui(src0_q, (int)(qoff + 3 * ne01)).x;
        const uint e = src0_d[eoff];
        qoff += 4 * ne01;
        eoff += ne01;

        // 2^(e - 127 + 14) overflows f16 for e > 128 (gpt-oss ffn_down_exps has e up to 136): second factor
        const half d = (half)(as_float(min(e, 128u) << 23) * 16384.0f);
        half16 h0 = (half16)(mxfp4_to_f16x8(q.s0, d), mxfp4_to_f16x8(q.s1, d));
        half16 h1 = (half16)(mxfp4_to_f16x8(q.s2, d), mxfp4_to_f16x8(q.s3, d));
        if (e > 128) {
            const half d2 = (half)as_float((e - 1) << 23);
            h0 *= d2;
            h1 *= d2;
        }
        vstore16(h0, 0, la[0] + lane * 16);
        vstore16(h1, 0, la[1] + lane * 16);
        barrier(CLK_LOCAL_MEM_FENCE);
        mat_a_t a0 = qcom_coop_mat_load(la[0], CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_QCOM, 16);
        mat_a_t a1 = qcom_coop_mat_load(la[1], CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_QCOM, 16);
        barrier(CLK_LOCAL_MEM_FENCE);
        mat_b_t b0 = qcom_coop_mat_load(bp + k0,      CLK_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR_QCOM, ne00);
        mat_b_t b1 = qcom_coop_mat_load(bp + k0 + 16, CLK_COOPERATIVE_MATRIX_LAYOUT_COLUMN_MAJOR_QCOM, ne00);
        acc = qcom_coop_mat_mulAdd(a0, b0, acc, CLK_COOPERATIVE_MATRIX_OPERAND_NONE_QCOM);
        acc = qcom_coop_mat_mulAdd(a1, b1, acc, CLK_COOPERATIVE_MATRIX_OPERAND_NONE_QCOM);
    }

    qcom_coop_mat_store(lc, acc, CLK_COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_QCOM, 32);
    barrier(CLK_LOCAL_MEM_FENCE);

    const bool wr = row + lane < ne01;
    const local float * cr = lc + lane * 32;
#pragma unroll
    for (uint t = 0; t < 32; t++) {
        STORE_C(t, cr[t])
    }
}
