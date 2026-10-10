#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

// Dense IQ4_XS prefill GEMM, dp4a inner loop: the IQ4_NL dp4a GEMM with the per-32 scale rebuilt
// from the packed super-block scales, ls = scales_l nibble | ((scales_h >> 2*sb) & 3) << 4,
// d_w = d * (ls - 32). The scales stay packed so the plane split is size preserving.
// Planes: src0_q [row + (k/4)*m] ushort (4 nibbles); [row + (k/256)*m] for src0_d (half),
// src0_sh (ushort scales_h) and src0_sl (uint, the four scales_l bytes).

// TILESIZE_N is the token tile: it fixes the accumulator count and the LDS staging width, so it is
// compile-time. Overridable because the right value is per device. The activation staging loop is
// strided, so any tile is correct.
#ifndef TILESIZE_N
#define TILESIZE_N 32
#endif

// IQ4_NL codebook as signed int8, packed 4 codes per uint: divergent nibble lookups read a small
// __constant uint array and shift, since byte-indexed __constant loads serialize on Adreno.
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

// The activation tile is staged as uint4: the eight uints a token needs for one 32-K step are
// contiguous, which cuts __local loads 4x and widens the staging load to 16 bytes per lane.
// The uint4s are copied into private temps at the call site: dp4a with a __local operand in an
// unrolled loop miscompiles on X2.
inline int dot8_q8a_v(uint8 qw, uint4 a0, uint4 a1) {
    int r = 0;
    r = dot_acc_sat_4x8packed_ss_int(qw.s0, a0.x, r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s1, a0.y, r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s2, a0.z, r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s3, a0.w, r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s4, a1.x, r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s5, a1.y, r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s6, a1.z, r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s7, a1.w, r);
    return r;
}

__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_gemm_noshuffle_iq4_xs_q8_1_dp4a(
        __global const ushort * src0_q,    // 4 codebook indices per ushort, feature-major
        __global const half   * src0_d,    // per-256 super-block scale, feature-major
        __global const ushort * src0_sh,   // scales_h, feature-major
        __global const uint   * src0_sl,   // scales_l (4 bytes packed), feature-major
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

    __local uint4 sh_qa4[TILESIZE_N][2];
    __local half sh_d[TILESIZE_N];

#define NGROUPS (TILESIZE_N / 4)
    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc[g] = (float4)(0.0f);

    for (uint step = 0; step < (uint)k; step += 32) {
        const uint sub = step >> 5;

        // rebuild the 6-bit sub-scale for this 32-block
        const uint   ib  = sub >> 3;              // super-block along K
        const uint   sb  = sub & 7u;              // sub-block inside it
        const uint   sl  = src0_sl[rrow + ib * (uint)m];
        const uint   shv = (uint)src0_sh[rrow + ib * (uint)m];
        const int    ls  = (int)(((sl >> (8u * (sb >> 1))) >> (4u * (sb & 1u))) & 0xFu)
                         | (int)(((shv >> (2u * sb)) & 3u) << 4);
        const float d_w = (float)src0_d[rrow + ib * (uint)m] * (float)(ls - 32);

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

        // Stage the activation tile to LDS as uint4s; (c*k_u + step/4) is a multiple of 8, so
        // vload4 is aligned.
        for (uint idx = lid; idx < TILESIZE_N * 2; idx += 64) {
            const uint t = idx >> 1;
            const uint v = idx & 1;
            const uint c = col_base + t;
            sh_qa4[t][v] = (c < (uint)n_no_padding)
                         ? vload4(0, src1_qa + c * k_u + (step >> 2) + (v << 2))
                         : (uint4)(0u);
        }
        if (lid < TILESIZE_N) {
            const uint c = col_base + lid;
            sh_d[lid] = (c < (uint)n_no_padding) ? src1_da[c * k_b + sub] : (half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

#define LD4(arr, b) ((float4)((float)arr[(b)+0], (float)arr[(b)+1], (float)arr[(b)+2], (float)arr[(b)+3]))
#define TDOT4(T) dot8_q8a_v(qw, (uint4)(sh_qa4[T][0]), (uint4)(sh_qa4[T][1]))
        #pragma unroll
        for (int g = 0; g < NGROUPS; ++g) {
            const int b = g * 4;
            const float4 rf = (float4)((float)TDOT4(b+0), (float)TDOT4(b+1),
                                       (float)TDOT4(b+2), (float)TDOT4(b+3));
            acc[g] += d_w * LD4(sh_d, b) * rf;
        }
#undef TDOT4
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
