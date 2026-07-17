#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#define TILESIZE_M 64
#define TILESIZE_N 32

// Expand the 4 nibbles held in the low 16 bits of `u` into 4 bytes (one nibble
// per byte, value 0..15), packed for the int8 dp4a. The -8 zero-point is applied
// in the epilogue via the activation sum term (cheaper than biasing every byte).
#define EXP4(u)  ( ((uint)((u) & 0x000Fu))        | \
                  (((uint)((u) & 0x00F0u)) << 4)  | \
                  (((uint)((u) & 0x0F00u)) << 8)  | \
                  (((uint)((u) & 0xF000u)) << 12) )

// One token's dp4a dot (8 uints = 32 K elems) + q4_0 scale/zero-point epilogue.
#define MOE_Q40_DP4A_T(t) do {                                       \
        uint4 a0 = vload4(0, &sh_qa[t][0]);                          \
        uint4 a1 = vload4(0, &sh_qa[t][4]);                          \
        int raw = 0;                                                 \
        raw = dot_acc_sat_4x8packed_ss_int(qw[0], a0.s0, raw);       \
        raw = dot_acc_sat_4x8packed_ss_int(qw[1], a0.s1, raw);       \
        raw = dot_acc_sat_4x8packed_ss_int(qw[2], a0.s2, raw);       \
        raw = dot_acc_sat_4x8packed_ss_int(qw[3], a0.s3, raw);       \
        raw = dot_acc_sat_4x8packed_ss_int(qw[4], a1.s0, raw);       \
        raw = dot_acc_sat_4x8packed_ss_int(qw[5], a1.s1, raw);       \
        raw = dot_acc_sat_4x8packed_ss_int(qw[6], a1.s2, raw);       \
        raw = dot_acc_sat_4x8packed_ss_int(qw[7], a1.s3, raw);       \
        acc[t] += d_val * ((float)sh_d[t] * (float)raw - 8.0f * (float)sh_s[t]); \
    } while (0)

__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_gemm_moe_q4_0_q8_1_dp4a(
        __read_only  image1d_buffer_t src0_q,   // q4_0 weights (transposed, packed nibbles)
        __global     half *           src0_d,   // per-32-block scale
        __global     uint *           src1_qa,  // q8_1 activations: int8 quants (as uint, 4/elem)
        __global     half *           src1_da,  // q8_1 per-block scale  [tok_slot * ne00/32]
        __global     half *           src1_sa,  // q8_1 per-block sum*d  [tok_slot * ne00/32]
        __global     uint *           src2,     // post-router (orig out positions)
        __global     ushort *         src2_emap,// tile -> expert id
        __write_only image1d_buffer_t dst,
        __global     int *            total_tiles,
        uint ne00,
        uint ne01,
        int  is_ragged                          // 1: compute only real tokens per tile
) {
    const uint block_id_m = get_global_id(1); // m_tile
    const uint block_id_n = get_global_id(2); // n_tile

    if (block_id_n >= total_tiles[0]) {
        return;
    }

    const uint lid = get_local_id(0);          // 0..63, == this WI's output row in the M-tile

    const ushort expert_id = src2_emap[block_id_n];
    const uint   row = block_id_m * TILESIZE_M;
    const uint   col = block_id_n * TILESIZE_N;

    const uint num_blocks = ne00 >> 5;          // blocks-of-32 per token
    const uint row_idx    = row + lid;

    const uint ne00_u = ne00 >> 2;   // ne00 in uint (int8x4) units

    __local uint sh_qa[TILESIZE_N][8]; // 32 tokens x 8 uints (32 int8) = 1 KiB
    __local half sh_d[TILESIZE_N];
    __local half sh_s[TILESIZE_N];

    // Real-token count for this tile
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
        const uint sub = step >> 5;        // 32-block index along K

        // per-32-block scale for this WI's row
        const uint d_offset = row_idx + sub * ne01 + expert_id * num_blocks * ne01;
        const float d_val = (float)src0_d[d_offset];

        // repack this WI's 32 weight nibbles into 8 dp4a uints
        const uint qoff0 = row + ((ne01 * step) >> 3)        + ((expert_id * ne00 * ne01) >> 3);
        const uint qoff1 = row + ((ne01 * (step + 16)) >> 3) + ((expert_id * ne00 * ne01) >> 3);
        const uint r0 = read_imageui(src0_q, qoff0 + lid).x;
        const uint r1 = read_imageui(src0_q, qoff0 + lid + ne01).x;
        const uint r2 = read_imageui(src0_q, qoff1 + lid).x;
        const uint r3 = read_imageui(src0_q, qoff1 + lid + ne01).x;
        uint qw[8];
        qw[0] = EXP4(r0);        qw[1] = EXP4(r0 >> 16);
        qw[2] = EXP4(r1);        qw[3] = EXP4(r1 >> 16);
        qw[4] = EXP4(r2);        qw[5] = EXP4(r2 >> 16);
        qw[6] = EXP4(r3);        qw[7] = EXP4(r3 >> 16);

        // cooperatively stage the n_real-token x 32-K int8 activations
        // Stage each token's 8 activation uints as two 128-bit uint4 loads/stores.
        const uint vlim = (uint)n_real * 2;
        for (uint idx = lid; idx < vlim; idx += 64) {
            const uint t = idx >> 1;
            const uint h = (idx & 1) << 2;   // 0 or 4
            uint4 v = vload4(0, &src1_qa[(col + t) * ne00_u + (step >> 2) + h]);
            vstore4(v, 0, &sh_qa[t][h]);
        }
        if (lid < (uint)n_real) {
            sh_d[lid] = src1_da[(col + lid) * num_blocks + sub];
            sh_s[lid] = src1_sa[(col + lid) * num_blocks + sub];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (n_real == TILESIZE_N) {
            #pragma unroll
            for (int t = 0; t < TILESIZE_N; ++t) { MOE_Q40_DP4A_T(t); }
        } else {
            #pragma unroll 4
            for (int t = 0; t < n_real; ++t) { MOE_Q40_DP4A_T(t); }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row_idx >= ne01) {
        return;
    }

    // scatter results to original output rows (reuse sh_src2 from the top)
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
