#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

// Weight layout
//   src0_qs[row + (k/4)*m]  ushort = 4 low nibbles (K = 4*grp .. +3)
//   src0_qh[row + (k/8)*m]  uchar  = 8 high bits  (one per element)
//   src0_d [row + (k/32)*m] half   = per-32-block scale

#define TILESIZE_N 32

// 4 nibbles in low 16 bits of u -> 4 bytes (value 0..15)
#define EXP4(u)  ( ((uint)((u) & 0x000Fu))        | \
                  (((uint)((u) & 0x00F0u)) << 4)  | \
                  (((uint)((u) & 0x0F00u)) << 8)  | \
                  (((uint)((u) & 0xF000u)) << 12) )
// 4 high bits (one per element, in bits 0..3 of h) -> bit4 of each of 4 bytes
#define EXP1(h)  ( (((uint)((h) & 0x1u)) << 4)   | \
                  (((uint)((h) & 0x2u)) << 11)  | \
                  (((uint)((h) & 0x4u)) << 18)  | \
                  (((uint)((h) & 0x8u)) << 25) )

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
kernel void kernel_gemm_noshuffle_q5_0_q8_1_dp4a(
        __global const ushort * src0_qs,   // q5_0 low nibbles (4/ushort, feature-major)
        __global const uchar  * src0_qh,   // q5_0 high-bit plane (8/uchar, feature-major)
        __global const half   * src0_d,    // per-32-block scale, feature-major
        __global const uint   * src1_qa,   // q8_1 activations int8 (as uint, 4/elem) [N, K]
        __global const half   * src1_da,   // q8_1 per-block scale [N, K/32]
        __global const half   * src1_sa,   // q8_1 per-block sum*d [N, K/32]
        __global       float  * dst,
        ulong  offsetd,
        int    m,                          // output features (rows)
        int    n_no_padding,               // tokens (cols)
        int    k                           // K (== ne00)
) {
    dst = (global float *)((global char *)dst + offsetd);

    const uint lid = get_local_id(0);          // 0..63 -> row within the M-tile
    const uint block_id_m = get_global_id(1);
    const uint block_id_n = get_global_id(2);

    const uint row      = block_id_m * 64 + lid;
    const uint col_base = block_id_n * TILESIZE_N;
    const bool row_valid = row < (uint)m;
    const uint rrow     = row_valid ? row : 0;  // clamp OOB rows; their writes are masked

    const uint k_u = (uint)k >> 2;   // K in uint (int8x4) units
    const uint k_b = (uint)k >> 5;   // blocks-of-32 along K

    __local uint sh_qa[TILESIZE_N][8];
    __local half sh_d[TILESIZE_N];
    __local half sh_s[TILESIZE_N];

#define NGROUPS (TILESIZE_N / 4)
    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc[g] = (float4)(0.0f);

    for (uint step = 0; step < (uint)k; step += 32) {
        const uint sub = step >> 5;

        const float d_w  = (float)src0_d[rrow + sub * (uint)m];
        const float minv = d_w * 16.0f;     // -16 centering -> subtract via q8_1 sum

        // 8 weight uints (32 elements) for this row, this 32-block.
        // nibbles: src0_qs[row + (step/4 + u)*m]; high bits: src0_qh[row + (step/8 + u/2)*m],
        // 4-bit group selected by (u&1)*4.
        const uint qsbase = rrow + (step >> 2) * (uint)m;
        const uint qhbase = rrow + (step >> 3) * (uint)m;
        uint8 qw;
        #define QW(u) (EXP4(src0_qs[qsbase + (u) * m]) | \
                       EXP1((uint)(src0_qh[qhbase + ((u) >> 1) * m] >> (((u) & 1u) * 4u)) & 0xFu))
        qw.s0 = QW(0); qw.s1 = QW(1); qw.s2 = QW(2); qw.s3 = QW(3);
        qw.s4 = QW(4); qw.s5 = QW(5); qw.s6 = QW(6); qw.s7 = QW(7);
        #undef QW

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
            acc[g] += d_w * LD4(sh_d, b) * rf - minv * LD4(sh_s, b);
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

__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_gemm_noshuffle_q5_0_q8_1_dp4a_wimg(
        __read_only image1d_buffer_t src0_qs_img, // q5_0 low nibbles as uint32 texels (2 ushorts/texel)
        __global const uchar  * src0_qh,
        __global const half   * src0_d,
        __global const uint   * src1_qa,
        __global const half   * src1_da,
        __global const half   * src1_sa,
        __global       float  * dst,
        ulong  offsetd,
        int    m,
        int    n_no_padding,
        int    k
) {
    dst = (global float *)((global char *)dst + offsetd);

    const uint lid = get_local_id(0);
    const uint block_id_m = get_global_id(1);
    const uint block_id_n = get_global_id(2);

    const uint row      = block_id_m * 64 + lid;
    const uint col_base = block_id_n * TILESIZE_N;
    const bool row_valid = row < (uint)m;
    const uint rrow     = row_valid ? row : 0;

    const uint sel = (rrow & 1u) * 16u;   // constant per WI: qs ushort half in its uint32 texel

    const uint k_u = (uint)k >> 2;
    const uint k_b = (uint)k >> 5;

    __local uint sh_qa[TILESIZE_N][8];
    __local half sh_d[TILESIZE_N];
    __local half sh_s[TILESIZE_N];

#define NGROUPS (TILESIZE_N / 4)
    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc[g] = (float4)(0.0f);

    for (uint step = 0; step < (uint)k; step += 32) {
        const uint sub = step >> 5;

        const float d_w  = (float)src0_d[rrow + sub * (uint)m];
        const float minv = d_w * 16.0f;

        const uint qsbase = rrow + (step >> 2) * (uint)m;   // ushort index
        const uint qhbase = rrow + (step >> 3) * (uint)m;
        uint8 qw;
        // qs ushort via texture: uint32 texel = ushort_index>>1, half = sel.
        #define QSU(u) ((read_imageui(src0_qs_img, (int)((qsbase + (u) * m) >> 1)).x >> sel) & 0xFFFFu)
        #define QW(u) (EXP4(QSU(u)) | \
                       EXP1((uint)(src0_qh[qhbase + ((u) >> 1) * m] >> (((u) & 1u) * 4u)) & 0xFu))
        qw.s0 = QW(0); qw.s1 = QW(1); qw.s2 = QW(2); qw.s3 = QW(3);
        qw.s4 = QW(4); qw.s5 = QW(5); qw.s6 = QW(6); qw.s7 = QW(7);
        #undef QW
        #undef QSU

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
            acc[g] += d_w * LD4(sh_d, b) * rf - minv * LD4(sh_s, b);
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
