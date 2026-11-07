#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

// q4_K MoE prefill GEMM with a dp4a inner loop (dot_acc_sat_4x8packed_ss_int); same tiling as
// kernel_gemm_moe_q4_k_f32_ns, with the activation tile pre-quantized to q8_1
// (kernel_moe_quant_a_q8_1).
//
// q4_K subblock (32 elems): w_i = scale*q_i - minv, q_i in [0,15], scale =
// d_super*sv6, minv = dmin_super*mn6. With activation block (a_d, a_s, qa[32]):
//   Sum_i w_i * a_i = scale * a_d * dp4a(q, qa) - minv * a_s
// where a_s = a_d * Sum(qa) (the q8_1 "s" field). Mirrors vec_dot_q4_K_q8_1.
//
// Accumulators are 8 float4 groups of 4 tokens; a ragged tile runs whole groups up to its
// real-token count. The superblock's d/dmin and 12 scale bytes are loaded once per 256 K as
// three uints. -DMOE_RB2 gives each lane rows lid and lid+64 of a 128-row tile.

#define TILESIZE_M 64
#define TILESIZE_N 32
#define NGROUPS    (TILESIZE_N / 4)
#define QK_K 256
#define K_SCALE_SIZE 12

// 6-bit scale and min of subblock j (0..7) from the superblock's 12 scale bytes,
// held as three little-endian uints q0 (bytes 0-3), q1 (4-7), q2 (8-11). Same
// decode as get_scale_min_k4.
#define BYTE_OF(u, i) (((u) >> (8 * (i))) & 0xFFu)
inline void q4k_scale_min(uint j, uint q0, uint q1, uint q2, float * sv, float * mn) {
    if (j < 4) {
        *sv = (float)(BYTE_OF(q0, j) & 63u);
        *mn = (float)(BYTE_OF(q1, j) & 63u);
    } else {
        const uint i = j - 4;
        *sv = (float)((BYTE_OF(q2, i) & 0x0Fu) | ((BYTE_OF(q0, i) & 0xC0u) >> 2));
        *mn = (float)((BYTE_OF(q2, i) >> 4)    | ((BYTE_OF(q1, i) & 0xC0u) >> 2));
    }
}

// Expand the 4 nibbles held in the low 16 bits of `u` into 4 bytes (one nibble
// per byte, value 0..15), packed for the int8 dp4a.
#define EXP4(u)  ( ((uint)((u) & 0x000Fu))        | \
                  (((uint)((u) & 0x00F0u)) << 4)  | \
                  (((uint)((u) & 0x0F00u)) << 8)  | \
                  (((uint)((u) & 0xF000u)) << 12) )

// R = 32-K dp4a dot of this WI's 8 weight uints (qw) with token T's 8 activation uints, read as two
// 128-bit local loads (a __local operand passed straight to the dp4a builtin can miscompile).
#define MOE_DOT8(T, R) do {                                            \
        const uint4 a0 = vload4(0, &sh_qa[T][0]);                      \
        const uint4 a1 = vload4(0, &sh_qa[T][4]);                      \
        int r_ = 0;                                                    \
        r_ = dot_acc_sat_4x8packed_ss_int(qw[0], a0.s0, r_);           \
        r_ = dot_acc_sat_4x8packed_ss_int(qw[1], a0.s1, r_);           \
        r_ = dot_acc_sat_4x8packed_ss_int(qw[2], a0.s2, r_);           \
        r_ = dot_acc_sat_4x8packed_ss_int(qw[3], a0.s3, r_);           \
        r_ = dot_acc_sat_4x8packed_ss_int(qw[4], a1.s0, r_);           \
        r_ = dot_acc_sat_4x8packed_ss_int(qw[5], a1.s1, r_);           \
        r_ = dot_acc_sat_4x8packed_ss_int(qw[6], a1.s2, r_);           \
        r_ = dot_acc_sat_4x8packed_ss_int(qw[7], a1.s3, r_);           \
        R = r_;                                                        \
    } while (0)

#define MOE_DOT8X2(T, R, R2) do {                                      \
        const uint4 a0 = vload4(0, &sh_qa[T][0]);                      \
        const uint4 a1 = vload4(0, &sh_qa[T][4]);                      \
        int r_ = 0, s_ = 0;                                            \
        r_ = dot_acc_sat_4x8packed_ss_int(qw[0], a0.s0, r_);           \
        s_ = dot_acc_sat_4x8packed_ss_int(qv[0], a0.s0, s_);           \
        r_ = dot_acc_sat_4x8packed_ss_int(qw[1], a0.s1, r_);           \
        s_ = dot_acc_sat_4x8packed_ss_int(qv[1], a0.s1, s_);           \
        r_ = dot_acc_sat_4x8packed_ss_int(qw[2], a0.s2, r_);           \
        s_ = dot_acc_sat_4x8packed_ss_int(qv[2], a0.s2, s_);           \
        r_ = dot_acc_sat_4x8packed_ss_int(qw[3], a0.s3, r_);           \
        s_ = dot_acc_sat_4x8packed_ss_int(qv[3], a0.s3, s_);           \
        r_ = dot_acc_sat_4x8packed_ss_int(qw[4], a1.s0, r_);           \
        s_ = dot_acc_sat_4x8packed_ss_int(qv[4], a1.s0, s_);           \
        r_ = dot_acc_sat_4x8packed_ss_int(qw[5], a1.s1, r_);           \
        s_ = dot_acc_sat_4x8packed_ss_int(qv[5], a1.s1, s_);           \
        r_ = dot_acc_sat_4x8packed_ss_int(qw[6], a1.s2, r_);           \
        s_ = dot_acc_sat_4x8packed_ss_int(qv[6], a1.s2, s_);           \
        r_ = dot_acc_sat_4x8packed_ss_int(qw[7], a1.s3, r_);           \
        s_ = dot_acc_sat_4x8packed_ss_int(qv[7], a1.s3, s_);           \
        R = r_; R2 = s_;                                               \
    } while (0)

#ifdef MOE_RB2
#define MOE_ROWS_PER_WG 128
#else
#define MOE_ROWS_PER_WG TILESIZE_M
#endif

// Token t's accumulator; t is a compile-time constant after unrolling.
#define ACC(t) ((((t) & 3) == 0) ? acc[(t) >> 2].s0 : (((t) & 3) == 1) ? acc[(t) >> 2].s1 : \
                (((t) & 3) == 2) ? acc[(t) >> 2].s2 : acc[(t) >> 2].s3)

__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_gemm_moe_q4_k_q8_1_dp4a(
        __read_only  image1d_buffer_t src0_q,   // q4_K weights (transposed, packed nibbles)
        __global     half *           src0_d,   // per-superblock scale
        __global     half *           src0_dm,  // per-superblock min
        __global     uchar *          src0_s,   // 6-bit scale/min codes
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
    const uint   row = block_id_m * MOE_ROWS_PER_WG;
    const uint   col = block_id_n * TILESIZE_N;

    const uint num_superblocks = ne00 / QK_K;
    const uint scales_per_row  = num_superblocks * K_SCALE_SIZE;
    const uint row_idx         = row + lid;

    const uint ne00_u  = ne00 >> 2;   // ne00 in uint (int8x4) units
    const uint ne00_b  = ne00 >> 5;   // blocks-of-32 per token

#ifdef MOE_LM_PAD
    // E17 (Adreno 850) returns wrong values for the start of these tiles; starting
    // each one MOE_LM_PAD x 8 bytes in avoids it.
    __local uint sh_qa_store[TILESIZE_N * 8 + 2 * MOE_LM_PAD];
    __local half sh_d_store[TILESIZE_N + 4 * MOE_LM_PAD];
    __local half sh_s_store[TILESIZE_N + 4 * MOE_LM_PAD];
    __local uint (*sh_qa)[8] = (__local uint (*)[8])(sh_qa_store + 2 * MOE_LM_PAD);
    __local half * sh_d = sh_d_store + 4 * MOE_LM_PAD;
    __local half * sh_s = sh_s_store + 4 * MOE_LM_PAD;
#else
    __local uint sh_qa[TILESIZE_N][8]; // 32 tokens x 8 uints (32 int8) = 1 KiB
    __local half sh_d[TILESIZE_N];
    __local half sh_s[TILESIZE_N];
#endif

    // Real-token count for this tile. Real tokens (original out positions) are
    // packed contiguously at the tile start by kernel_moe_scatter; padded slots
    // hold 0xFFFFFFFF (only the last tile of each expert is partial). When
    // is_ragged, skip the dp4a/staging/scatter for the padded slots.
    // is_ragged==0 forces n_real=32 == the full-tile path.
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

    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc[g] = (float4)(0.0f);
#ifdef MOE_RB2
    float4 acc2[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc2[g] = (float4)(0.0f);
#endif

    const uint q_expert = (expert_id * ne00 * ne01) >> 3;
    // 12 scale bytes per superblock, so every row's and every superblock's codes
    // start on a 4-byte boundary.
    global const uint * sc_row  = (global const uint *)(src0_s + (expert_id * ne01 + row_idx) * scales_per_row);
#ifdef MOE_RB2
    global const uint * sc_row2 = (global const uint *)(src0_s + (expert_id * ne01 + row_idx + 64) * scales_per_row);
#endif

    for (uint sb = 0; sb < num_superblocks; ++sb) {
        // --- this superblock's d/dmin and 12 scale bytes for this WI's row(s) ---
        const uint d_offset = row_idx + sb * ne01 + expert_id * num_superblocks * ne01;
        const float d_val  = (float)src0_d[d_offset];
        const float dm_val = (float)src0_dm[d_offset];
        const uint3 sc3    = vload3(sb, sc_row);
#ifdef MOE_RB2
        const float d_val2  = (float)src0_d[d_offset + 64];
        const float dm_val2 = (float)src0_dm[d_offset + 64];
        const uint3 sc3b    = vload3(sb, sc_row2);
#endif

        for (uint j = 0; j < 8; ++j) {
            const uint step = sb * QK_K + j * 32;
            const uint sub  = step >> 5;

            float sv, mn;
            q4k_scale_min(j, sc3.x, sc3.y, sc3.z, &sv, &mn);
            const float scale = d_val  * sv;
            const float minv  = dm_val * mn;

            // --- repack this WI's 32 weight nibbles into 8 dp4a uints ---
            const uint qoff0 = row + ((ne01 * step) >> 3)        + q_expert;
            const uint qoff1 = row + ((ne01 * (step + 16)) >> 3) + q_expert;
            uint qw[8];
            {
                const uint r0 = read_imageui(src0_q, qoff0 + lid).x;
                const uint r1 = read_imageui(src0_q, qoff0 + lid + ne01).x;
                const uint r2 = read_imageui(src0_q, qoff1 + lid).x;
                const uint r3 = read_imageui(src0_q, qoff1 + lid + ne01).x;
                qw[0] = EXP4(r0);        qw[1] = EXP4(r0 >> 16);
                qw[2] = EXP4(r1);        qw[3] = EXP4(r1 >> 16);
                qw[4] = EXP4(r2);        qw[5] = EXP4(r2 >> 16);
                qw[6] = EXP4(r3);        qw[7] = EXP4(r3 >> 16);
            }
#ifdef MOE_RB2
            float sv2, mn2;
            q4k_scale_min(j, sc3b.x, sc3b.y, sc3b.z, &sv2, &mn2);
            const float scale2 = d_val2  * sv2;
            const float minv2  = dm_val2 * mn2;
            uint qv[8];
            {
                const uint t0 = read_imageui(src0_q, qoff0 + lid + 64).x;
                const uint t1 = read_imageui(src0_q, qoff0 + lid + 64 + ne01).x;
                const uint t2 = read_imageui(src0_q, qoff1 + lid + 64).x;
                const uint t3 = read_imageui(src0_q, qoff1 + lid + 64 + ne01).x;
                qv[0] = EXP4(t0);    qv[1] = EXP4(t0 >> 16);
                qv[2] = EXP4(t1);    qv[3] = EXP4(t1 >> 16);
                qv[4] = EXP4(t2);    qv[5] = EXP4(t2 >> 16);
                qv[6] = EXP4(t3);    qv[7] = EXP4(t3 >> 16);
            }
#endif

            // --- cooperatively stage the n_real-token x 32-K int8 activations to LDS ---
            const uint vlim = (uint)n_real * 2;
            for (uint idx = lid; idx < vlim; idx += 64) {
                const uint t = idx >> 1;
                const uint h = (idx & 1) << 2;   // 0 or 4
                uint4 v = vload4(0, &src1_qa[(col + t) * ne00_u + (step >> 2) + h]);
                vstore4(v, 0, &sh_qa[t][h]);
            }
            if (lid < (uint)n_real) {
                sh_d[lid] = src1_da[(col + lid) * ne00_b + sub];
                sh_s[lid] = src1_sa[(col + lid) * ne00_b + sub];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            // Whole groups up to n_real. Lanes past n_real in the last group read
            // unstaged LDS, but their results are never stored.
            #pragma unroll
            for (int g = 0; g < NGROUPS; ++g) {
                if (g * 4 < n_real) {
                    const int b = g * 4;
                    int4 raw;
#ifdef MOE_RB2
                    int4 raw2;
                    MOE_DOT8X2(b + 0, raw.s0, raw2.s0); MOE_DOT8X2(b + 1, raw.s1, raw2.s1);
                    MOE_DOT8X2(b + 2, raw.s2, raw2.s2); MOE_DOT8X2(b + 3, raw.s3, raw2.s3);
                    const float4 rf2 = convert_float4(raw2);
#else
                    MOE_DOT8(b + 0, raw.s0); MOE_DOT8(b + 1, raw.s1);
                    MOE_DOT8(b + 2, raw.s2); MOE_DOT8(b + 3, raw.s3);
#endif
                    const float4 rf = convert_float4(raw);
                    const float4 ad = (float4)((float)sh_d[b + 0], (float)sh_d[b + 1], (float)sh_d[b + 2], (float)sh_d[b + 3]);
                    const float4 as = (float4)((float)sh_s[b + 0], (float)sh_s[b + 1], (float)sh_s[b + 2], (float)sh_s[b + 3]);
                    acc[g] += scale * ad * rf - minv * as;
#ifdef MOE_RB2
                    acc2[g] += scale2 * ad * rf2 - minv2 * as;
#endif
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (row_idx >= ne01) {
        return;
    }

    // --- scatter results to original output rows (reuse sh_src2 from the top) ---
    // The full tile keeps the 0xFFFFFFFF->slot0 fallback + t=0-written-last
    // ordering (padded slots alias slot 0; the last write wins). The ragged path
    // writes only the n_real real tokens (distinct positions).
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
#ifdef MOE_RB2
#define ACC2(t) ((((t) & 3) == 0) ? acc2[(t) >> 2].s0 : (((t) & 3) == 1) ? acc2[(t) >> 2].s1 : \
                 (((t) & 3) == 2) ? acc2[(t) >> 2].s2 : acc2[(t) >> 2].s3)
#endif
    if (n_real == TILESIZE_N) {
        #pragma unroll
        for (int t = 1; t < TILESIZE_N; ++t) {
            write_imagef(dst, out_idx[t] + m_offset, ACC(t));
#ifdef MOE_RB2
            write_imagef(dst, out_idx[t] + m_offset + 64, ACC2(t));
#endif
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        write_imagef(dst, out_idx[0] + m_offset, ACC(0));
#ifdef MOE_RB2
        write_imagef(dst, out_idx[0] + m_offset + 64, ACC2(0));
#endif
    } else {
        #pragma unroll
        for (int t = 0; t < TILESIZE_N; ++t) {
            if (t < n_real) {
                write_imagef(dst, out_idx[t] + m_offset, ACC(t));
#ifdef MOE_RB2
                write_imagef(dst, out_idx[t] + m_offset + 64, ACC2(t));
#endif
            }
        }
    }
}
