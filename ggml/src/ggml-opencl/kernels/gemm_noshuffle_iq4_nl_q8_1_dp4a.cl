#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

// Weight layout, feature-major:
//   src0_q[row + (k/4)*m]  ushort = 4 nibbles (K = 4*grp .. +3)
//   src0_d[row + (k/32)*m] half   = per-32-block scale

#define TILESIZE_N 32

// IQ4_NL non-linear codebook as signed int8, packed 4 codes per uint.
// divergent nibble lookups read a small __constant uint array + shift,
// never a byte array because byte-indexed __constant loads serialize on Adreno and tank perf
//   idx 0-3:  -127,-104,-83,-65 = 0x81,0x98,0xAD,0xBF
//   idx 4-7:  -49,-35,-22,-10   = 0xCF,0xDD,0xEA,0xF6
//   idx 8-11:  1, 13, 25, 38    = 0x01,0x0D,0x19,0x26
//   idx 12-15: 53, 69, 89,113   = 0x35,0x45,0x59,0x71
__constant uint kvalues_iq4nl_i8x4[4] = {
    0xBFAD9881u, 0xF6EADDCFu, 0x26190D01u, 0x71594535u
};

// nibble (0..15) -> its codebook byte in the low 8 bits.
inline uint iq4nl_code(uint n) {
    return (kvalues_iq4nl_i8x4[n >> 2] >> ((n & 3u) * 8u)) & 0xFFu;
}

// 4 nibbles in low 16 bits of u -> 4 codebook int8, packed for dp4a.
inline uint iq4nl_pack(ushort u) {
    return  iq4nl_code((uint)( u        & 0xF))
         | (iq4nl_code((uint)((u >>  4) & 0xF)) <<  8)
         | (iq4nl_code((uint)((u >>  8) & 0xF)) << 16)
         | (iq4nl_code((uint)((u >> 12) & 0xF)) << 24);
}

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
kernel void kernel_gemm_noshuffle_iq4_nl_q8_1_dp4a(
        __global const ushort * src0_q,    // IQ4_NL nibbles (4/ushort, feature-major)
        __global const half   * src0_d,    // per-32-block scale, feature-major
        __global const uint   * src1_qa,   // q8_1 activations int8 (as uint, 4/elem) [N, K]
        __global const half   * src1_da,   // q8_1 per-block scale [N, K/32]
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

#define NGROUPS (TILESIZE_N / 4)
    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc[g] = (float4)(0.0f);

    for (uint step = 0; step < (uint)k; step += 32) {
        const uint sub = step >> 5;

        const float d_w = (float)src0_d[rrow + sub * (uint)m];

        // 8 weight uints (32 codebook int8) for this row, this 32-block.
        const uint qsbase = rrow + (step >> 2) * (uint)m;
        uint8 qw;
        qw.s0 = iq4nl_pack(src0_q[qsbase + 0 * m]);
        qw.s1 = iq4nl_pack(src0_q[qsbase + 1 * m]);
        qw.s2 = iq4nl_pack(src0_q[qsbase + 2 * m]);
        qw.s3 = iq4nl_pack(src0_q[qsbase + 3 * m]);
        qw.s4 = iq4nl_pack(src0_q[qsbase + 4 * m]);
        qw.s5 = iq4nl_pack(src0_q[qsbase + 5 * m]);
        qw.s6 = iq4nl_pack(src0_q[qsbase + 6 * m]);
        qw.s7 = iq4nl_pack(src0_q[qsbase + 7 * m]);

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
