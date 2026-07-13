#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#ifndef TILESIZE_N
#define TILESIZE_N 32
#endif
#define QK_K 256
#define K_SCALE_SIZE 12

inline void get_scale_min_k4(
    int j,
    global const uchar * q,
    uchar * d,
    uchar * m,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2
) {
    if (j < 4) {
        *d = q[j]   & mask_d6;
        *m = q[j+4] & mask_d6;
    } else {
        *d = (q[j+4] & mask_d4) | ((q[j-4] & mask_hi2) >> 2);
        *m = ((q[j+4] >> 4) & mask_d4) | ((q[j]   & mask_hi2) >> 2);
    }
}

// Expand the 4 nibbles in the low 16 bits of `u` into 4 bytes (one nibble per
// byte, value 0..15), packed for the int8 dp4a.
#define EXP4(u)  ( ((uint)((u) & 0x000Fu))        | \
                  (((uint)((u) & 0x00F0u)) << 4)  | \
                  (((uint)((u) & 0x0F00u)) << 8)  | \
                  (((uint)((u) & 0xF000u)) << 12) )

// 32-K dp4a dot of one token's int8 activations (8 packed uints in lm) against the
// row's 8 packed weight uints. qw passed by value as a uint8 (register), not an array.
inline int dot8_q8a(uint8 qw, __local const uint * a) {
    int r = 0;
    r = dot_acc_sat_4x8packed_ss_int(qw.s0, a[0], r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s1, a[1], r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s2, a[2], r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s3, a[3], r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s4, a[4], r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s5, a[5], r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s6, a[6], r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s7, a[7], r);
    return r;
}

__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_gemm_noshuffle_q4_k_q8_1_dp4a(
        __global const ushort * src0_q,    // q4_K weights (noshuffle, packed nibbles)
        __global const uchar  * src0_s,    // 6-bit scale/min codes
        __global const half   * src0_d,    // per-superblock scale
        __global const half   * src0_dm,   // per-superblock min
        __global const uint   * src1_qa,   // q8_1 activations int8 (as uint, 4/elem) [N, K]
        __global const half   * src1_da,   // q8_1 per-block scale [N, K/32]
        __global const half   * src1_sa,   // q8_1 per-block sum*d [N, K/32]
        __global       float  * dst,
        ulong  offsetd,
        int    m,                          // output features (rows)
        int    n_no_padding,               // tokens (cols)
        int    k,                          // K (== ne00)
        uchar  mask_d6,
        uchar  mask_d4,
        uchar  mask_hi2
) {
    dst = (global float *)((global char *)dst + offsetd);

    const uint lid = get_local_id(0);          // 0..63 -> row within the M-tile
    const uint block_id_m = get_global_id(1);
    const uint block_id_n = get_global_id(2);

    const uint row      = block_id_m * 64 + lid;
    const uint col_base = block_id_n * TILESIZE_N;
    const bool row_valid = row < (uint)m;
    const uint rrow     = row_valid ? row : 0;  // clamp OOB rows; their writes are masked

    const uint num_superblocks = (uint)k / QK_K;
    const uint k_u = (uint)k >> 2;   // K in uint (int8x4) units
    const uint k_b = (uint)k >> 5;   // blocks-of-32 along K

    __local uint sh_qa[TILESIZE_N][8];
    __local half sh_d[TILESIZE_N];
    __local half sh_s[TILESIZE_N];

    // One float4 vector-register accumulator per group of 4 tokens (NGROUPS = TILESIZE_N/4).
#define NGROUPS (TILESIZE_N / 4)
    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) { acc[g] = (float4)(0.0f); }

    for (uint step = 0; step < (uint)k; step += 32) {
        const uint sub     = step >> 5;
        const uint sb_idx  = step / QK_K;
        const uint sub_idx = sub & 7;

        // weight scale/min for this WI's row, this subblock
        const float dd  = (float)src0_d [rrow + sb_idx * m];
        const float dmm = (float)src0_dm[rrow + sb_idx * m];
        global const uchar * sc = src0_s + rrow * num_superblocks * K_SCALE_SIZE + sb_idx * K_SCALE_SIZE;
        uchar sv, mn;
        get_scale_min_k4(sub_idx, sc, &sv, &mn, mask_d6, mask_d4, mask_hi2);
        const float scale = dd  * (float)sv;
        const float minv  = dmm * (float)mn;

        // repack this row's 32 weight nibbles into 8 dp4a uints. The packed q4_K
        // layout stores one ushort = 4 consecutive-K nibbles for a row at
        // src0_q[row + (K_group)*m], K_group = step/4 + u.
        const uint wbase = rrow + (step >> 2) * (uint)m;
        uint8 qw;
        qw.s0 = EXP4(src0_q[wbase + 0 * m]);
        qw.s1 = EXP4(src0_q[wbase + 1 * m]);
        qw.s2 = EXP4(src0_q[wbase + 2 * m]);
        qw.s3 = EXP4(src0_q[wbase + 3 * m]);
        qw.s4 = EXP4(src0_q[wbase + 4 * m]);
        qw.s5 = EXP4(src0_q[wbase + 5 * m]);
        qw.s6 = EXP4(src0_q[wbase + 6 * m]);
        qw.s7 = EXP4(src0_q[wbase + 7 * m]);

        // cooperatively stage the 32-token x 32-K int8 activations to lm
        for (uint idx = lid; idx < TILESIZE_N * 8; idx += 64) {
            const uint t = idx >> 3;
            const uint u = idx & 7;
            const uint c = col_base + t;
            sh_qa[t][u] = (c < (uint)n_no_padding) ? src1_qa[c * k_u + (step >> 2) + u] : 0u;
        }
        if (lid < TILESIZE_N) {
            const uint c = col_base + lid;
            sh_d[lid] = (c < (uint)n_no_padding) ? src1_da[c * k_b + sub] : (half)0;
            sh_s[lid] = (c < (uint)n_no_padding) ? src1_sa[c * k_b + sub] : (half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

#define LD4(arr, b) ((float4)((float)arr[(b)+0], (float)arr[(b)+1], (float)arr[(b)+2], (float)arr[(b)+3]))
        #pragma unroll
        for (int g = 0; g < NGROUPS; ++g) {
            const int b = g * 4;
            float4 rf;
            rf.s0 = (float)dot8_q8a(qw, sh_qa[b+0]);  rf.s1 = (float)dot8_q8a(qw, sh_qa[b+1]);
            rf.s2 = (float)dot8_q8a(qw, sh_qa[b+2]);  rf.s3 = (float)dot8_q8a(qw, sh_qa[b+3]);
            acc[g] += scale * LD4(sh_d, b) * rf - minv * LD4(sh_s, b);
        }
#undef LD4
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!row_valid) {
        return;
    }

    // dst is [token, feature] row-major (stride m): dst[col*m + row]. Scatter each
    // lane with a per-token padding guard (dst is non-contiguous in token).
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) {
        const uint b = (uint)(g * 4);
        const float4 a = acc[g];
        const uint c0 = col_base + b;
        if (c0 + 0 < (uint)n_no_padding) dst[(c0 + 0) * (uint)m + row] = a.s0;
        if (c0 + 1 < (uint)n_no_padding) dst[(c0 + 1) * (uint)m + row] = a.s1;
        if (c0 + 2 < (uint)n_no_padding) dst[(c0 + 2) * (uint)m + row] = a.s2;
        if (c0 + 3 < (uint)n_no_padding) dst[(c0 + 3) * (uint)m + row] = a.s3;
    }
#undef NGROUPS
}

__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_gemm_noshuffle_q4_k_q8_1_dp4a_wimg(
        __read_only image1d_buffer_t src0_q_img, // q4_K weights as uint32 texels (2 ushorts/texel)
        __global const uchar  * src0_s,    // 6-bit scale/min codes
        __global const half   * src0_d,    // per-superblock scale
        __global const half   * src0_dm,   // per-superblock min
        __global const uint   * src1_qa,   // q8_1 activations int8 (as uint, 4/elem) [N, K]
        __global const half   * src1_da,   // q8_1 per-block scale [N, K/32]
        __global const half   * src1_sa,   // q8_1 per-block sum*d [N, K/32]
        __global       float  * dst,
        ulong  offsetd,
        int    m,                          // output features (rows)
        int    n_no_padding,               // tokens (cols)
        int    k,                          // K (== ne00)
        uchar  mask_d6,
        uchar  mask_d4,
        uchar  mask_hi2
) {
    dst = (global float *)((global char *)dst + offsetd);

    const uint lid = get_local_id(0);          // 0..63 -> row within the M-tile
    const uint block_id_m = get_global_id(1);
    const uint block_id_n = get_global_id(2);

    const uint row      = block_id_m * 64 + lid;
    const uint col_base = block_id_n * TILESIZE_N;
    const bool row_valid = row < (uint)m;
    const uint rrow     = row_valid ? row : 0;  // clamp OOB rows; their writes are masked

    // Constant per WI: the ushort the row needs always sits in the same half of
    // its uint32 texel (m even => index parity == rrow parity). Hoist the shift.
    const uint sel = (rrow & 1u) * 16u;

    const uint k_u = (uint)k >> 2;   // K in uint (int8x4) units
    const uint k_b = (uint)k >> 5;   // blocks-of-32 along K
    const uint num_superblocks = (uint)k / QK_K;

    __local uint sh_qa[TILESIZE_N][8];
    __local half sh_d[TILESIZE_N];
    __local half sh_s[TILESIZE_N];

#define NGROUPS (TILESIZE_N / 4)
    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc[g] = (float4)(0.0f);

    for (uint step = 0; step < (uint)k; step += 32) {
        const uint sub     = step >> 5;
        const uint sb_idx  = step / QK_K;
        const uint sub_idx = sub & 7;

        const float dd  = (float)src0_d [rrow + sb_idx * m];
        const float dmm = (float)src0_dm[rrow + sb_idx * m];
        global const uchar * sc = src0_s + rrow * num_superblocks * K_SCALE_SIZE + sb_idx * K_SCALE_SIZE;
        uchar sv, mn;
        get_scale_min_k4(sub_idx, sc, &sv, &mn, mask_d6, mask_d4, mask_hi2);
        const float scale = dd  * (float)sv;
        const float minv  = dmm * (float)mn;

        const uint wbase = rrow + (step >> 2) * (uint)m;
        uint8 qw;
        qw.s0 = EXP4(read_imageui(src0_q_img, (int)((wbase + 0 * m) >> 1)).x >> sel);
        qw.s1 = EXP4(read_imageui(src0_q_img, (int)((wbase + 1 * m) >> 1)).x >> sel);
        qw.s2 = EXP4(read_imageui(src0_q_img, (int)((wbase + 2 * m) >> 1)).x >> sel);
        qw.s3 = EXP4(read_imageui(src0_q_img, (int)((wbase + 3 * m) >> 1)).x >> sel);
        qw.s4 = EXP4(read_imageui(src0_q_img, (int)((wbase + 4 * m) >> 1)).x >> sel);
        qw.s5 = EXP4(read_imageui(src0_q_img, (int)((wbase + 5 * m) >> 1)).x >> sel);
        qw.s6 = EXP4(read_imageui(src0_q_img, (int)((wbase + 6 * m) >> 1)).x >> sel);
        qw.s7 = EXP4(read_imageui(src0_q_img, (int)((wbase + 7 * m) >> 1)).x >> sel);

        for (uint idx = lid; idx < TILESIZE_N * 8; idx += 64) {
            const uint t = idx >> 3;
            const uint u = idx & 7;
            const uint c = col_base + t;
            sh_qa[t][u] = (c < (uint)n_no_padding) ? src1_qa[c * k_u + (step >> 2) + u] : 0u;
        }
        if (lid < TILESIZE_N) {
            const uint c = col_base + lid;
            sh_d[lid] = (c < (uint)n_no_padding) ? src1_da[c * k_b + sub] : (half)0;
            sh_s[lid] = (c < (uint)n_no_padding) ? src1_sa[c * k_b + sub] : (half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

#define LD4(arr, b) ((float4)((float)arr[(b)+0], (float)arr[(b)+1], (float)arr[(b)+2], (float)arr[(b)+3]))
        #pragma unroll
        for (int g = 0; g < NGROUPS; ++g) {
            const int b = g * 4;
            float4 rf;
            rf.s0 = (float)dot8_q8a(qw, sh_qa[b+0]);  rf.s1 = (float)dot8_q8a(qw, sh_qa[b+1]);
            rf.s2 = (float)dot8_q8a(qw, sh_qa[b+2]);  rf.s3 = (float)dot8_q8a(qw, sh_qa[b+3]);
            acc[g] += scale * LD4(sh_d, b) * rf - minv * LD4(sh_s, b);
        }
#undef LD4
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!row_valid) {
        return;
    }

    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) {
        const uint b = (uint)(g * 4);
        const float4 a = acc[g];
        const uint c0 = col_base + b;
        if (c0 + 0 < (uint)n_no_padding) dst[(c0 + 0) * (uint)m + row] = a.s0;
        if (c0 + 1 < (uint)n_no_padding) dst[(c0 + 1) * (uint)m + row] = a.s1;
        if (c0 + 2 < (uint)n_no_padding) dst[(c0 + 2) * (uint)m + row] = a.s2;
        if (c0 + 3 < (uint)n_no_padding) dst[(c0 + 3) * (uint)m + row] = a.s3;
    }
#undef NGROUPS
}
