#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#define QK_K 256

#ifndef TILESIZE_N
#define TILESIZE_N 32
#endif

inline uint iq2_xxs_pack(uint gv, uint sg, uint base) {
    int v0 = (int)((gv >>  0) & 0xFF); if (sg & (1u << (base + 0))) { v0 = -v0; }
    int v1 = (int)((gv >>  8) & 0xFF); if (sg & (1u << (base + 1))) { v1 = -v1; }
    int v2 = (int)((gv >> 16) & 0xFF); if (sg & (1u << (base + 2))) { v2 = -v2; }
    int v3 = (int)((gv >> 24) & 0xFF); if (sg & (1u << (base + 3))) { v3 = -v3; }
    return ((uint)v0 & 0xFFu) | (((uint)v1 & 0xFFu) <<  8)
         | (((uint)v2 & 0xFFu) << 16) | (((uint)v3 & 0xFFu) << 24);
}

// Pass staged operands by value; local dp4a operands miscompile on X2.
inline int dot4_q8a_v(uint4 qw, uint4 a) {
    int r = 0;
    r = dot_acc_sat_4x8packed_ss_int(qw.s0, a.x, r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s1, a.y, r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s2, a.z, r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s3, a.w, r);
    return r;
}

__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_gemm_noshuffle_iq2_xxs_q8_1_dp4a(
        __global const uchar  * src0_qs,
        __global const uint   * src0_sas,
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

#define IQ2_XXS_GRID(i) iq2_xxs_grid[(i)]

#define NGROUPS (TILESIZE_N / 4)
    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc[g] = (float4)(0.0f);

    for (uint step = 0; step < (uint)k; step += 32) {
        const uint sub = step >> 5;
        const uint ib  = sub >> 3;

        const uint  aux = src0_sas[rrow + sub * (uint)m];
        const float dv  = (float)src0_d[rrow + ib * (uint)m];
        const float dl  = dv * (0.5f + (float)(aux >> 28)) * 0.25f;

        const uint gb = rrow + (step >> 3) * (uint)m;

        uint4 qlo, qhi;
        {
            const uint gi0 = (uint)src0_qs[gb + 0u * (uint)m] << 1;
            const uint gi1 = (uint)src0_qs[gb + 1u * (uint)m] << 1;
            const uint gi2 = (uint)src0_qs[gb + 2u * (uint)m] << 1;
            const uint gi3 = (uint)src0_qs[gb + 3u * (uint)m] << 1;
            const uint s0  = iq2_xxs_signs((aux >>  0) & 127u);
            const uint s1  = iq2_xxs_signs((aux >>  7) & 127u);
            const uint s2  = iq2_xxs_signs((aux >> 14) & 127u);
            const uint s3  = iq2_xxs_signs((aux >> 21) & 127u);
            qlo.s0 = iq2_xxs_pack(IQ2_XXS_GRID(gi0 + 0u), s0, 0u);
            qlo.s1 = iq2_xxs_pack(IQ2_XXS_GRID(gi0 + 1u), s0, 4u);
            qlo.s2 = iq2_xxs_pack(IQ2_XXS_GRID(gi1 + 0u), s1, 0u);
            qlo.s3 = iq2_xxs_pack(IQ2_XXS_GRID(gi1 + 1u), s1, 4u);
            qhi.s0 = iq2_xxs_pack(IQ2_XXS_GRID(gi2 + 0u), s2, 0u);
            qhi.s1 = iq2_xxs_pack(IQ2_XXS_GRID(gi2 + 1u), s2, 4u);
            qhi.s2 = iq2_xxs_pack(IQ2_XXS_GRID(gi3 + 0u), s3, 0u);
            qhi.s3 = iq2_xxs_pack(IQ2_XXS_GRID(gi3 + 1u), s3, 4u);
        }

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
#define IQ2_XXS_COL(b) (dl * (float)(dot4_q8a_v(qlo, (uint4)(sh_qa4[b][0]))   \
                                  + dot4_q8a_v(qhi, (uint4)(sh_qa4[b][1]))))
        #pragma unroll
        for (int g = 0; g < NGROUPS; ++g) {
            const int b = g * 4;
            float4 rf;
            rf.s0 = IQ2_XXS_COL(b+0);  rf.s1 = IQ2_XXS_COL(b+1);
            rf.s2 = IQ2_XXS_COL(b+2);  rf.s3 = IQ2_XXS_COL(b+3);
            acc[g] += LD4(sh_d, b) * rf;
        }
#undef IQ2_XXS_COL
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
#undef IQ2_XXS_GRID
}
