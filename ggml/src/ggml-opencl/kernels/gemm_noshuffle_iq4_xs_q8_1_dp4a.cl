#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#define QK_K 256

#ifndef TILESIZE_N
#define TILESIZE_N 32
#endif

inline uint iq4_xs_pack(ushort q) {
    return iq4_xs_code(q & 15)
         | (iq4_xs_code((q >> 4) & 15) << 8)
         | (iq4_xs_code((q >> 8) & 15) << 16)
         | (iq4_xs_code(q >> 12) << 24);
}

// Pass staged operands by value; local dp4a operands miscompile on X2.
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
        __global const ushort * src0_qs,
        __global const uint   * src0_sl,
        __global const ushort * src0_sh,
        __global const half   * src0_d,
        __global const uint   * src1_qa,
        __global const half   * src1_da,
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

    const uint k_u = (uint)k >> 2;
    const uint k_b = (uint)k >> 5;

    __local uint4 sh_qa4[TILESIZE_N][2];
    __local half sh_d[TILESIZE_N];

#define NGROUPS (TILESIZE_N / 4)
    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc[g] = (float4)(0.0f);

    for (uint step = 0; step < (uint)k; step += 32) {
        const uint sub = step >> 5;

        const uint ib = sub >> 3;
        const uint sb = sub & 7u;
        const uint sl = src0_sl[rrow + ib*(uint)m];
        const uint sh = src0_sh[rrow + ib*(uint)m];
        const int ls = (int)(((sl >> (4*sb)) & 15u) | (((sh >> (2*sb)) & 3u) << 4)) - 32;
        const float d_w = (float)src0_d[rrow + ib*(uint)m] * (float)ls;
        const uint qsb = rrow + sub*8u*(uint)m;

        uint8 qw;
        qw.s0 = iq4_xs_pack(src0_qs[qsb + 0u*(uint)m]);
        qw.s1 = iq4_xs_pack(src0_qs[qsb + 1u*(uint)m]);
        qw.s2 = iq4_xs_pack(src0_qs[qsb + 2u*(uint)m]);
        qw.s3 = iq4_xs_pack(src0_qs[qsb + 3u*(uint)m]);
        qw.s4 = iq4_xs_pack(src0_qs[qsb + 4u*(uint)m]);
        qw.s5 = iq4_xs_pack(src0_qs[qsb + 5u*(uint)m]);
        qw.s6 = iq4_xs_pack(src0_qs[qsb + 6u*(uint)m]);
        qw.s7 = iq4_xs_pack(src0_qs[qsb + 7u*(uint)m]);

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
        #pragma unroll
        for (int g = 0; g < NGROUPS; ++g) {
            const int b = g * 4;
            float4 rf;
#define TDOT4(T) dot8_q8a_v(qw, (uint4)(sh_qa4[T][0]), (uint4)(sh_qa4[T][1]))
            rf.s0 = (float)TDOT4(b+0);  rf.s1 = (float)TDOT4(b+1);
            rf.s2 = (float)TDOT4(b+2);  rf.s3 = (float)TDOT4(b+3);
            acc[g] += d_w * LD4(sh_d, b) * rf;
        }
#undef TDOT4
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
