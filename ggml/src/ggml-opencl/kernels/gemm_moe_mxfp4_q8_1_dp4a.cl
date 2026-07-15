#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#define TILESIZE_M 64
#define TILESIZE_N 32

// 2*mxfp4_value as signed int8, packed 4 codes per uint. Divergent nibble
// lookups read a __constant *uint* array + shift, never a byte array
// (byte-indexed __constant loads serialize on Adreno and are far slower).
//   idx 0-3:   0,  1,  2,  3   = 0x03020100
//   idx 4-7:   4,  6,  8, 12   = 0x0C080604
//   idx 8-11:  0, -1, -2, -3   = 0xFDFEFF00   (-1=0xFF,-2=0xFE,-3=0xFD)
//   idx 12-15:-4, -6, -8,-12   = 0xF4F8FAFC   (-4=0xFC,-6=0xFA,-8=0xF8,-12=0xF4)
__constant uint mxfp4_i8x4[4] = {
    0x03020100u, 0x0C080604u, 0xFDFEFF00u, 0xF4F8FAFCu
};
inline uint mxfp4_code(uint n) {
    return (mxfp4_i8x4[n >> 2] >> ((n & 3u) * 8u)) & 0xFFu;
}
// 4 nibbles in the low 16 bits of u -> 4 codebook int8, packed for dp4a.
inline uint mxfp4_pack(ushort u) {
    return  mxfp4_code((uint)( u        & 0xF))
         | (mxfp4_code((uint)((u >>  4) & 0xF)) <<  8)
         | (mxfp4_code((uint)((u >>  8) & 0xF)) << 16)
         | (mxfp4_code((uint)((u >> 12) & 0xF)) << 24);
}

static inline float e8m0_to_fp32(uchar x) {
    int bits;
    bits = (x == 0) ? 0x00400000 : ((uint) x << 23);
    return as_float(bits);
}

// One token's dp4a dot (8 uints = 32 K elems) + mxfp4 block-scale epilogue.
// blk_scale already carries the 0.5 factor (== 0.5 * 2^e).
#define MOE_MXFP4_DP4A_T(t) do {                                     \
        int raw = 0;                                                 \
        raw = dot_acc_sat_4x8packed_ss_int(qw[0], sh_qa[t][0], raw); \
        raw = dot_acc_sat_4x8packed_ss_int(qw[1], sh_qa[t][1], raw); \
        raw = dot_acc_sat_4x8packed_ss_int(qw[2], sh_qa[t][2], raw); \
        raw = dot_acc_sat_4x8packed_ss_int(qw[3], sh_qa[t][3], raw); \
        raw = dot_acc_sat_4x8packed_ss_int(qw[4], sh_qa[t][4], raw); \
        raw = dot_acc_sat_4x8packed_ss_int(qw[5], sh_qa[t][5], raw); \
        raw = dot_acc_sat_4x8packed_ss_int(qw[6], sh_qa[t][6], raw); \
        raw = dot_acc_sat_4x8packed_ss_int(qw[7], sh_qa[t][7], raw); \
        acc[t] += blk_scale * (float)sh_d[t] * (float)raw;           \
    } while (0)

__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_gemm_moe_mxfp4_q8_1_dp4a(
        __read_only  image1d_buffer_t src0_q,    // mxfp4 codes (transposed, packed nibbles)
        __global     uchar *          src0_e,    // e8m0 per-32-block scale
        __global     uint *           src1_qa,   // q8_1 activations: int8 quants (as uint, 4/elem)
        __global     half *           src1_da,   // q8_1 per-block scale  [tok_slot * ne00/32]
        __global     uint *           src2,      // post-router (orig out positions)
        __global     ushort *         src2_emap, // tile -> expert id
        __write_only image1d_buffer_t dst,
        __global     int *            total_tiles,
        uint ne00,
        uint ne01,
        int  is_ragged                           // 1: compute only real tokens per tile
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

    // Real token count for this tile.
    // Real tokens are packed contiguously at the tile start; padded slots hold
    // 0xFFFFFFFF (only the last tile of each expert is partial). is_ragged skips
    // the dp4a/staging/scatter for padded slots; is_ragged==0 forces n_real=32.
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

        // e8m0 block scale for this WI's row, this 32-block (folded x0.5)
        const uint e_offset = row_idx + sub * ne01 + expert_id * num_blocks * ne01;
        const float blk_scale = 0.5f * e8m0_to_fp32(src0_e[e_offset]);

        // repack this WI's 32 weight nibbles into 8 dp4a uints
        const uint qoff0 = row + ((ne01 * step) >> 3)        + ((expert_id * ne00 * ne01) >> 3);
        const uint qoff1 = row + ((ne01 * (step + 16)) >> 3) + ((expert_id * ne00 * ne01) >> 3);
        const uint r0 = read_imageui(src0_q, qoff0 + lid).x;
        const uint r1 = read_imageui(src0_q, qoff0 + lid + ne01).x;
        const uint r2 = read_imageui(src0_q, qoff1 + lid).x;
        const uint r3 = read_imageui(src0_q, qoff1 + lid + ne01).x;
        uint qw[8];
        qw[0] = mxfp4_pack((ushort)(r0));        qw[1] = mxfp4_pack((ushort)(r0 >> 16));
        qw[2] = mxfp4_pack((ushort)(r1));        qw[3] = mxfp4_pack((ushort)(r1 >> 16));
        qw[4] = mxfp4_pack((ushort)(r2));        qw[5] = mxfp4_pack((ushort)(r2 >> 16));
        qw[6] = mxfp4_pack((ushort)(r3));        qw[7] = mxfp4_pack((ushort)(r3 >> 16));

        // cooperatively stage the n_real-token x 32-K int8 activations
        const uint stage_lim = (uint)n_real * 8;
        for (uint idx = lid; idx < stage_lim; idx += 64) {
            const uint t = idx >> 3;
            const uint u = idx & 7;
            sh_qa[t][u] = src1_qa[(col + t) * ne00_u + (step >> 2) + u];
        }
        if (lid < (uint)n_real) {
            sh_d[lid] = src1_da[(col + lid) * num_blocks + sub];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Full tiles keep the fully-unrolled 32-wide loop; partial tiles run only n_real
        if (n_real == TILESIZE_N) {
            #pragma unroll
            for (int t = 0; t < TILESIZE_N; ++t) { MOE_MXFP4_DP4A_T(t); }
        } else {
            #pragma unroll 4
            for (int t = 0; t < n_real; ++t) { MOE_MXFP4_DP4A_T(t); }
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
