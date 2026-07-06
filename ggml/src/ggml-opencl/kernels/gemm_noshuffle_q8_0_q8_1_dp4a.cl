#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

// Dense q8_0 prefill GEMM, dp4a (int8) inner loop.
//
// dp4a alternative to kernel_gemm_noshuffle_q8_0_f32 (the f16 half-dot GEMM used
// for the dense q8_0 matmuls — attention / ffn projections in a full-Q8_0 model).
// The activations are pre-quantized to q8_1 (kernel_quant_a_q8_1) straight from
// the original [N, K] token-major buffer (no transpose); the per-32-block dot
// uses the qcom int8 dp4a.
//
// q8_0 is the simplest case: the stored weights are already signed int8 (4 per
// uint, feature-major: src0_q[row + (k/4)*m]) — no nibble unpack — with one fp16
// scale per 32-block per row (src0_d[row + (k/32)*m]) and NO min/offset term
// (symmetric quant). So per 32-K step: qw = 8 raw weight uints, and
//   Sum w*a = d_w * a_d * dp4a(qw, qa)
// with no q8_1 sum-correction term. Mirrors kernel_gemm_noshuffle_q4_k_q8_1_dp4a
// (one WI per output row, a TILESIZE_N-token tile) without the q4_K scale decode.
//
// Large-batch (prefill) only; ne1<=8 keeps the f16 / bin small-batch path.

#define TILESIZE_N 32

// 32-K dp4a dot of one token's int8 activations (8 packed uints in LDS) against
// 8 packed weight uints. q8_0 weights are already dp4a-format signed int8.
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
kernel void kernel_gemm_noshuffle_q8_0_q8_1_dp4a(
        __global const uint  * src0_q,     // q8_0 weights: signed int8, 4/uint, feature-major
        __global const half  * src0_d,     // per-32-block scale, feature-major [row + (k/32)*m]
        __global const uint  * src1_qa,    // q8_1 activations int8 (as uint, 4/elem) [N, K]
        __global const half  * src1_da,    // q8_1 per-block scale [N, K/32]
        __global       float * dst,
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

#define NGROUPS (TILESIZE_N / 4)
    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc[g] = (float4)(0.0f);

    for (uint step = 0; step < (uint)k; step += 32) {
        const uint sub = step >> 5;

        const float d_w = (float)src0_d[rrow + sub * (uint)m];

        // 8 weight uints (32 int8) for this row, this 32-block. Feature-major:
        // src0_q[row + (k/4 + u)*m], k/4 = step/4 (= step>>2).
        const uint wbase = rrow + (step >> 2) * (uint)m;
        uint8 qw;
        qw.s0 = src0_q[wbase + 0 * m];
        qw.s1 = src0_q[wbase + 1 * m];
        qw.s2 = src0_q[wbase + 2 * m];
        qw.s3 = src0_q[wbase + 3 * m];
        qw.s4 = src0_q[wbase + 4 * m];
        qw.s5 = src0_q[wbase + 5 * m];
        qw.s6 = src0_q[wbase + 6 * m];
        qw.s7 = src0_q[wbase + 7 * m];

        // cooperatively stage the 32-token x 32-K int8 activations to LDS
        for (uint idx = lid; idx < TILESIZE_N * 8; idx += 64) {
            const uint t = idx >> 3;
            const uint u = idx & 7;
            const uint c = col_base + t;
            sh_qa[t][u] = (c < (uint)n_no_padding) ? src1_qa[c * k_u + (step >> 2) + u] : 0u;
        }
        if (lid < TILESIZE_N) {
            const uint c = col_base + lid;
            sh_d[lid] = (c < (uint)n_no_padding) ? src1_da[c * k_b + sub] : (half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

#define LD4(arr, b) ((float4)((float)arr[(b)+0], (float)arr[(b)+1], (float)arr[(b)+2], (float)arr[(b)+3]))
        #pragma unroll
        for (int g = 0; g < NGROUPS; ++g) {
            const int b = g * 4;
            float4 rf;
            rf.s0 = (float)dot8_q8a(qw, sh_qa[b+0]);  rf.s1 = (float)dot8_q8a(qw, sh_qa[b+1]);
            rf.s2 = (float)dot8_q8a(qw, sh_qa[b+2]);  rf.s3 = (float)dot8_q8a(qw, sh_qa[b+3]);
            acc[g] += d_w * LD4(sh_d, b) * rf;
        }
#undef LD4
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!row_valid) {
        return;
    }

    // dst is [token, feature] row-major (stride m): dst[col*m + row].
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

// Weight-as-texture variant of kernel_gemm_noshuffle_q8_0_q8_1_dp4a.
//
// Byte-identical math; the only change is that the q8_0 int8 weight plane is read
// through an image1d_buffer (read_imageui -> texture/L1 cache) instead of a plain
// __global buffer. Same motivation as the q4_K _wimg variant: keep the int8 dp4a
// ALU win AND the texture-cache bandwidth the f16 path relies on (the lever for
// Adreno X1, where the buffer dp4a path regresses). Cleanest case of all: the q8_0
// weight is already 4 int8/uint, so the CL_R/UINT32 image (1 texel = 4 int8, width
// M*K/4 -- the same view the GEMV path builds) maps 1:1 to the buffer index, no
// parity / half-select. Opt-in: GGML_OPENCL_Q8_DENSE_DP4A_WIMG.
__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_gemm_noshuffle_q8_0_q8_1_dp4a_wimg(
        __read_only image1d_buffer_t src0_q_img,  // q8_0 weights as uint32 texels (4 int8/texel)
        __global const half  * src0_d,
        __global const uint  * src1_qa,
        __global const half  * src1_da,
        __global       float * dst,
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

    const uint k_u = (uint)k >> 2;
    const uint k_b = (uint)k >> 5;

    __local uint sh_qa[TILESIZE_N][8];
    __local half sh_d[TILESIZE_N];

#define NGROUPS (TILESIZE_N / 4)
    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc[g] = (float4)(0.0f);

    for (uint step = 0; step < (uint)k; step += 32) {
        const uint sub = step >> 5;

        const float d_w = (float)src0_d[rrow + sub * (uint)m];

        const uint wbase = rrow + (step >> 2) * (uint)m;
        uint8 qw;
        qw.s0 = read_imageui(src0_q_img, (int)(wbase + 0 * m)).x;
        qw.s1 = read_imageui(src0_q_img, (int)(wbase + 1 * m)).x;
        qw.s2 = read_imageui(src0_q_img, (int)(wbase + 2 * m)).x;
        qw.s3 = read_imageui(src0_q_img, (int)(wbase + 3 * m)).x;
        qw.s4 = read_imageui(src0_q_img, (int)(wbase + 4 * m)).x;
        qw.s5 = read_imageui(src0_q_img, (int)(wbase + 5 * m)).x;
        qw.s6 = read_imageui(src0_q_img, (int)(wbase + 6 * m)).x;
        qw.s7 = read_imageui(src0_q_img, (int)(wbase + 7 * m)).x;

        for (uint idx = lid; idx < TILESIZE_N * 8; idx += 64) {
            const uint t = idx >> 3;
            const uint u = idx & 7;
            const uint c = col_base + t;
            sh_qa[t][u] = (c < (uint)n_no_padding) ? src1_qa[c * k_u + (step >> 2) + u] : 0u;
        }
        if (lid < TILESIZE_N) {
            const uint c = col_base + lid;
            sh_d[lid] = (c < (uint)n_no_padding) ? src1_da[c * k_b + sub] : (half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

#define LD4(arr, b) ((float4)((float)arr[(b)+0], (float)arr[(b)+1], (float)arr[(b)+2], (float)arr[(b)+3]))
        #pragma unroll
        for (int g = 0; g < NGROUPS; ++g) {
            const int b = g * 4;
            float4 rf;
            rf.s0 = (float)dot8_q8a(qw, sh_qa[b+0]);  rf.s1 = (float)dot8_q8a(qw, sh_qa[b+1]);
            rf.s2 = (float)dot8_q8a(qw, sh_qa[b+2]);  rf.s3 = (float)dot8_q8a(qw, sh_qa[b+3]);
            acc[g] += d_w * LD4(sh_d, b) * rf;
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
