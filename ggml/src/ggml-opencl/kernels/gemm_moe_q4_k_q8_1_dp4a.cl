#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

// q4_K subblock (32 elems): w_i = scale*q_i - minv, q_i in [0,15], scale =
// d_super*sv6, minv = dmin_super*mn6. With activation block (a_d, a_s, qa[32]):
//   Sum_i w_i * a_i = scale * a_d * dp4a(q, qa) - minv * a_s
// where a_s = a_d * Sum(qa) (the q8_1 "s" field)

#define TILESIZE_M 64
#define TILESIZE_N 32
#define QK_K 256
#define K_SCALE_SIZE 12

inline void get_scale_min_k4(
    int j,
    global const uchar * q,
    uchar * d,
    uchar * m
) {
    if (j < 4) {
        *d = q[j]   & 63;
        *m = q[j+4] & 63;
    } else {
        *d = (q[j+4] & 0x0F) | ((q[j-4] & 0xC0) >> 2);
        *m = ((q[j+4] >> 4) & 0x0F) | ((q[j]   & 0xC0) >> 2);
    }
}

// Expand the 4 nibbles held in the low 16 bits of `u` into 4 bytes (one nibble
// per byte, value 0..15), packed for the int8 dp4a.
#define EXP4(u)  ( ((uint)((u) & 0x000Fu))        | \
                  (((uint)((u) & 0x00F0u)) << 4)  | \
                  (((uint)((u) & 0x0F00u)) << 8)  | \
                  (((uint)((u) & 0xF000u)) << 12) )

// One token's dp4a dot (8 uints = 32 K elems) + q4_K scale/min epilogue into acc[t].
#define MOE_Q4K_DP4A_T(t) do {                                       \
        int raw = 0;                                                 \
        raw = dot_acc_sat_4x8packed_ss_int(qw[0], sh_qa[t][0], raw); \
        raw = dot_acc_sat_4x8packed_ss_int(qw[1], sh_qa[t][1], raw); \
        raw = dot_acc_sat_4x8packed_ss_int(qw[2], sh_qa[t][2], raw); \
        raw = dot_acc_sat_4x8packed_ss_int(qw[3], sh_qa[t][3], raw); \
        raw = dot_acc_sat_4x8packed_ss_int(qw[4], sh_qa[t][4], raw); \
        raw = dot_acc_sat_4x8packed_ss_int(qw[5], sh_qa[t][5], raw); \
        raw = dot_acc_sat_4x8packed_ss_int(qw[6], sh_qa[t][6], raw); \
        raw = dot_acc_sat_4x8packed_ss_int(qw[7], sh_qa[t][7], raw); \
        acc[t] += scale * (float)sh_d[t] * (float)raw - minv * (float)sh_s[t]; \
    } while (0)

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
    const uint   row = block_id_m * TILESIZE_M;
    const uint   col = block_id_n * TILESIZE_N;

    const uint num_superblocks = ne00 / QK_K;
    const uint scales_per_row  = num_superblocks * K_SCALE_SIZE;
    const uint row_idx         = row + lid;

    const uint ne00_u  = ne00 >> 2;   // ne00 in uint (int8x4) units
    const uint ne00_b  = ne00 >> 5;   // blocks-of-32 per token

    __local uint sh_qa[TILESIZE_N][8]; // 32 tokens x 8 uints (32 int8) = 1 KiB
    __local half sh_d[TILESIZE_N];
    __local half sh_s[TILESIZE_N];

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
        const uint sub = step >> 5;        // subblock index along K
        const uint sb  = sub >> 3;         // superblock index
        const uint j   = sub & 7;          // subblock within superblock

        // --- weight scale / min for this WI's row, this subblock ---
        const uint d_offset = row + sb * ne01 + expert_id * num_superblocks * ne01 + lid;
        const float d_val  = (float)src0_d[d_offset];
        const float dm_val = (float)src0_dm[d_offset];

        global const uchar * sc = src0_s + (expert_id * ne01 + row_idx) * scales_per_row + sb * K_SCALE_SIZE;
        uchar sv, mn;
        get_scale_min_k4(j, sc, &sv, &mn);
        const float scale = d_val  * (float)sv;
        const float minv  = dm_val * (float)mn;

        // --- repack this WI's 32 weight nibbles into 8 dp4a uints ---
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

        // --- cooperatively stage the n_real-token x 32-K int8 activations to LDS ---
        const uint stage_lim = (uint)n_real * 8;
        for (uint idx = lid; idx < stage_lim; idx += 64) {
            const uint t = idx >> 3;
            const uint u = idx & 7;
            sh_qa[t][u] = src1_qa[(col + t) * ne00_u + (step >> 2) + u];
        }
        if (lid < (uint)n_real) {
            sh_d[lid] = src1_da[(col + lid) * ne00_b + sub];
            sh_s[lid] = src1_sa[(col + lid) * ne00_b + sub];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // dp4a - each real token sum over 8 uints (32 K), then scale/min
        // Full tiles keep the fully-unrolled 32-wide loop;
        // partial tiles run only n_real (saves the padded-slot dp4a + staging).
        if (n_real == TILESIZE_N) {
            #pragma unroll
            for (int t = 0; t < TILESIZE_N; ++t) { MOE_Q4K_DP4A_T(t); }
        } else {
            #pragma unroll 4
            for (int t = 0; t < n_real; ++t) { MOE_Q4K_DP4A_T(t); }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row_idx >= ne01) {
        return;
    }

    // scatter results to original output rows
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
