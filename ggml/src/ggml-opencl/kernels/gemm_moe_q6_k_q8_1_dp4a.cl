#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#define TILESIZE_N 32
#define QK_K 256

// 4 nibbles in the low 16 bits of `u` -> 4 bytes (value 0..15, in bits 0-3).
#define EXP4(u)  ( ((uint)((u) & 0x000Fu))        | \
                  (((uint)((u) & 0x00F0u)) << 4)  | \
                  (((uint)((u) & 0x0F00u)) << 8)  | \
                  (((uint)((u) & 0xF000u)) << 12) )

// 4 2-bit highs in byte `b` (8 bits) -> 4 bytes, value 0..3 in bits 4-5
// (pre-multiplied by 16 so it ORs with the EXP4 nibble to form q6 in 0..63).
#define EXP2(b)  ( (((uint)((b) & 0x03u)) << 4)   | \
                  (((uint)((b) & 0x0Cu)) << 10)  | \
                  (((uint)((b) & 0x30u)) << 16)  | \
                  (((uint)((b) & 0xC0u)) << 22) )

// q6 (0..63, bits 0-5 of each byte) -> (q6-32) as a signed int8 per byte.
// Flipping bit5 subtracts 32 in 6-bit two's complement; then replicate bit5
// into bits 6-7 to sign-extend to int8. Per-byte, no inter-byte carry.
inline uint SIGN6(uint q6p) {
    uint x = q6p ^ 0x20202020u;
    uint s = x & 0x20202020u;
    return x | (s << 1) | (s << 2);
}

inline int dp4a_q6(uint qw0, uint qw1, uint qw2, uint qw3,
                   uint a0, uint a1, uint a2, uint a3) {
    int raw = 0;
    raw = dot_acc_sat_4x8packed_ss_int(qw0, a0, raw);
    raw = dot_acc_sat_4x8packed_ss_int(qw1, a1, raw);
    raw = dot_acc_sat_4x8packed_ss_int(qw2, a2, raw);
    raw = dot_acc_sat_4x8packed_ss_int(qw3, a3, raw);
    return raw;
}

// One token's q6_K dp4a dot (two halves, per-16 scales) + epilogue into acc[t].
#define MOE_Q6K_DP4A_T(t) do {                                                                            \
        const int raw1 = dp4a_q6(qw[0], qw[1], qw[2], qw[3], sh_qa[t][0], sh_qa[t][1], sh_qa[t][2], sh_qa[t][3]); \
        const int raw2 = dp4a_q6(qw[4], qw[5], qw[6], qw[7], sh_qa[t][4], sh_qa[t][5], sh_qa[t][6], sh_qa[t][7]); \
        const float a_d = (float)sh_d[t];                                                                 \
        acc[t] += scale0 * a_d * (float)raw1 + scale1 * a_d * (float)raw2;                                \
    } while (0)

__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_gemm_moe_q6_k_q8_1_dp4a(
        __read_only  image1d_buffer_t src0_ql,   // q6_K low nibbles (image, q4_K-style layout)
        __global     uint *           src0_qh,   // q6_K high 2-bit (16 elems/uint)
        __global     char *           src0_s,    // int8 scales (one per 16 elems)
        __global     half *           src0_d,    // per-superblock scale
        __global     uint *           src1_qa,   // q8_1 activations int8 (as uint, 4/elem)
        __global     half *           src1_da,   // q8_1 per-block scale [tok_slot * ne00/32]
        __global     uint *           src2,      // post-router (orig out positions)
        __global     ushort *         src2_emap, // tile -> expert id
        __write_only image1d_buffer_t dst,
        __global     int *            total_tiles,
        uint ne00,
        uint ne01,
        int  is_ragged                         // 1: compute only real tokens per tile
) {
    const uint block_id_m = get_global_id(1);
    const uint block_id_n = get_global_id(2);

    if (block_id_n >= total_tiles[0]) {
        return;
    }

    const uint lid = get_local_id(0);          // 0..63 -> row within M-tile

    const ushort expert_id = src2_emap[block_id_n];
    const uint   row = block_id_m * 64;
    const uint   col = block_id_n * TILESIZE_N;

    const uint num_superblocks = ne00 / QK_K;
    const uint scales_per_row  = num_superblocks * 16;
    const uint row_idx         = row + lid;

    const uint ne00_u = ne00 >> 2;
    const uint ne00_b = ne00 >> 5;

    __local uint sh_qa[TILESIZE_N][8];
    __local half sh_d[TILESIZE_N];

    // Real token count for this tile
    __local uint sh_src2[TILESIZE_N];
    __local int  sh_nreal;
    if (lid < TILESIZE_N) {
        sh_src2[lid] = src2[col + lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        int nr = TILESIZE_N;
        if (is_ragged) {
            nr = 0;
            #pragma unroll
            for (int t = 0; t < TILESIZE_N; ++t) {
                if (sh_src2[t] != 0xFFFFFFFFu) ++nr;
            }
        }
        sh_nreal = nr;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int n_real = sh_nreal;

    float acc[TILESIZE_N];
    #pragma unroll
    for (int t = 0; t < TILESIZE_N; ++t) acc[t] = 0.0f;

    for (uint step = 0; step < ne00; step += 32) {
        const uint sub = step >> 5;
        const uint sb  = sub >> 3;
        const uint j   = sub & 7;

        const float d_val = (float)src0_d[row + sb * ne01 + expert_id * num_superblocks * ne01 + lid];
        global const char * sc = src0_s + (expert_id * ne01 + row_idx) * scales_per_row + sb * 16;
        const float scale0 = d_val * (float)sc[j * 2];
        const float scale1 = d_val * (float)sc[j * 2 + 1];

        // high bits: one uint covers 16 elems; first/second 16 of this 32-block
        const uint qh_base = row + (sub * 2) * ne01 + expert_id * (num_superblocks * 16) * ne01 + lid;
        const uint qh1 = src0_qh[qh_base];
        const uint qh2 = src0_qh[qh_base + ne01];

        // low nibbles: same image layout as q4_K (8 ushorts over the 32 K)
        const uint qoff0 = row + ((ne01 * step) >> 3)        + ((expert_id * ne00 * ne01) >> 3);
        const uint qoff1 = row + ((ne01 * (step + 16)) >> 3) + ((expert_id * ne00 * ne01) >> 3);
        const uint r0 = read_imageui(src0_ql, qoff0 + lid).x;
        const uint r1 = read_imageui(src0_ql, qoff0 + lid + ne01).x;
        const uint r2 = read_imageui(src0_ql, qoff1 + lid).x;
        const uint r3 = read_imageui(src0_ql, qoff1 + lid + ne01).x;

        uint qw[8];
        qw[0] = SIGN6(EXP4(r0)       | EXP2((qh1)       & 0xFFu));
        qw[1] = SIGN6(EXP4(r0 >> 16) | EXP2((qh1 >> 8)  & 0xFFu));
        qw[2] = SIGN6(EXP4(r1)       | EXP2((qh1 >> 16) & 0xFFu));
        qw[3] = SIGN6(EXP4(r1 >> 16) | EXP2((qh1 >> 24) & 0xFFu));
        qw[4] = SIGN6(EXP4(r2)       | EXP2((qh2)       & 0xFFu));
        qw[5] = SIGN6(EXP4(r2 >> 16) | EXP2((qh2 >> 8)  & 0xFFu));
        qw[6] = SIGN6(EXP4(r3)       | EXP2((qh2 >> 16) & 0xFFu));
        qw[7] = SIGN6(EXP4(r3 >> 16) | EXP2((qh2 >> 24) & 0xFFu));

        const uint stage_lim = (uint)n_real * 8;
        for (uint idx = lid; idx < stage_lim; idx += 64) {
            const uint t = idx >> 3;
            const uint u = idx & 7;
            sh_qa[t][u] = src1_qa[(col + t) * ne00_u + (step >> 2) + u];
        }
        if (lid < (uint)n_real) {
            sh_d[lid] = src1_da[(col + lid) * ne00_b + sub];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Full tiles keep the fully-unrolled 32-wide loop; partial tiles run n_real.
        if (n_real == TILESIZE_N) {
            #pragma unroll
            for (int t = 0; t < TILESIZE_N; ++t) { MOE_Q6K_DP4A_T(t); }
        } else {
            #pragma unroll 4
            for (int t = 0; t < n_real; ++t) { MOE_Q6K_DP4A_T(t); }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row_idx >= ne01) {
        return;
    }

    __local uint out_idx[TILESIZE_N];
    if (lid < TILESIZE_N) {
        uint idx = sh_src2[lid];
        if (idx == 0xFFFFFFFF) {
            idx = sh_src2[0];
        }
        out_idx[lid] = idx * ne01;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint m_offset = row + lid;
    if (n_real == TILESIZE_N) {
        #pragma unroll
        for (int t = 1; t < TILESIZE_N; ++t) {
            write_imagef(dst, out_idx[t] + m_offset, acc[t]);
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        write_imagef(dst, out_idx[0] + m_offset, acc[0]);
    } else {
        for (int t = 0; t < n_real; ++t) {
            write_imagef(dst, out_idx[t] + m_offset, acc[t]);
        }
    }
}
