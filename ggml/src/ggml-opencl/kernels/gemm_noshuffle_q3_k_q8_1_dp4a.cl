#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#define QK_K 256

#ifndef TILESIZE_N
#define TILESIZE_N 32
#endif

inline int q3_k_scale(uint sc0, uint sc1, uint sc2, uint is) {
    const uint w  = is >> 2;
    const uint sh = 8u * (is & 3u);
    const uint tb = (sc2 >> sh) & 0xFFu;
    uint v;
    if      (w == 0u) { v = ( (sc0 >> sh)       & 0xFu) | (((tb >> 0) & 3u) << 4); }
    else if (w == 1u) { v = ( (sc1 >> sh)       & 0xFu) | (((tb >> 2) & 3u) << 4); }
    else if (w == 2u) { v = (((sc0 >> sh) >> 4) & 0xFu) | (((tb >> 4) & 3u) << 4); }
    else              { v = (((sc1 >> sh) >> 4) & 0xFu) | (((tb >> 6) & 3u) << 4); }
    return (int)v - 32;
}

inline uint q3_k_pack(uint pk, uint hb) {
    int v0 = (int)((pk      ) & 3u) - 4 + 4*(int)((hb     ) & 1u);
    int v1 = (int)((pk >> 2) & 3u) - 4 + 4*(int)((hb >> 1) & 1u);
    int v2 = (int)((pk >> 4) & 3u) - 4 + 4*(int)((hb >> 2) & 1u);
    int v3 = (int)((pk >> 6) & 3u) - 4 + 4*(int)((hb >> 3) & 1u);
    return ((uint)v0 & 0xFFu) | (((uint)v1 & 0xFFu) <<  8)
         | (((uint)v2 & 0xFFu) << 16) | (((uint)v3 & 0xFFu) << 24);
}

// Pass staged operands by value; local dp4a operands miscompile on X2.
#define KQ_STAGE(p) vload4(0, (p))

inline int dot4_q8a_v(uint4 qw, uint4 a) {
    int r = 0;
    r = dot_acc_sat_4x8packed_ss_int(qw.s0, a.x, r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s1, a.y, r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s2, a.z, r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s3, a.w, r);
    return r;
}

__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_gemm_noshuffle_q3_k_q8_1_dp4a(
        __global const uchar  * src0_qs,
        __global const uchar  * src0_hm,
        __global const uint   * src0_sc,
        __global const half   * src0_d,
        __global const uint   * src1_qa,   // q8_1 activations int8 (as uint, 4/elem) [N, K]
        __global const half   * src1_da,   // q8_1 per-block scale [N, K/32]
        __global       float  * dst,
        ulong  offsetd,
        int    m,                          // output features (rows)
        int    n_no_padding,               // tokens (cols)
        int    k                           // K (== ne00)
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
        const uint ib  = sub >> 3;                // super-block along K
        const uint sb  = sub & 7u;

        const uint scb = rrow + (3u * ib) * (uint)m;
        const uint sc0 = src0_sc[scb + 0u * (uint)m];
        const uint sc1 = src0_sc[scb + 1u * (uint)m];
        const uint sc2 = src0_sc[scb + 2u * (uint)m];

        const float d_w = (float)src0_d[rrow + ib * (uint)m];
        const float dl0 = d_w * (float)q3_k_scale(sc0, sc1, sc2, 2u*sb + 0u);
        const float dl1 = d_w * (float)q3_k_scale(sc0, sc1, sc2, 2u*sb + 1u);

        const uint qsb = rrow + (step >> 2) * (uint)m;
        const uint hmb = rrow + (step >> 3) * (uint)m;

        uint4 qlo, qhi;
        {
            const uint h0 = (uint)src0_hm[hmb + 0u * (uint)m];
            const uint h1 = (uint)src0_hm[hmb + 1u * (uint)m];
            const uint h2 = (uint)src0_hm[hmb + 2u * (uint)m];
            const uint h3 = (uint)src0_hm[hmb + 3u * (uint)m];
            qlo.s0 = q3_k_pack((uint)src0_qs[qsb + 0u * (uint)m],  h0       & 0xFu);
            qlo.s1 = q3_k_pack((uint)src0_qs[qsb + 1u * (uint)m], (h0 >> 4) & 0xFu);
            qlo.s2 = q3_k_pack((uint)src0_qs[qsb + 2u * (uint)m],  h1       & 0xFu);
            qlo.s3 = q3_k_pack((uint)src0_qs[qsb + 3u * (uint)m], (h1 >> 4) & 0xFu);
            qhi.s0 = q3_k_pack((uint)src0_qs[qsb + 4u * (uint)m],  h2       & 0xFu);
            qhi.s1 = q3_k_pack((uint)src0_qs[qsb + 5u * (uint)m], (h2 >> 4) & 0xFu);
            qhi.s2 = q3_k_pack((uint)src0_qs[qsb + 6u * (uint)m],  h3       & 0xFu);
            qhi.s3 = q3_k_pack((uint)src0_qs[qsb + 7u * (uint)m], (h3 >> 4) & 0xFu);
        }

        for (uint idx = lid; idx < TILESIZE_N * 2; idx += 64) {
            const uint t = idx >> 1;
            const uint v = idx & 1;
            const uint c = col_base + t;
            sh_qa4[t][v] = (c < (uint)n_no_padding)
                         ? KQ_STAGE(src1_qa + c * k_u + (step >> 2) + (v << 2))
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
#define Q3_K_COL(T) (dl0 * (float)dot4_q8a_v(qlo, (uint4)(sh_qa4[T][0]))  \
                  + dl1 * (float)dot4_q8a_v(qhi, (uint4)(sh_qa4[T][1])))
            rf.s0 = Q3_K_COL(b+0);  rf.s1 = Q3_K_COL(b+1);
            rf.s2 = Q3_K_COL(b+2);  rf.s3 = Q3_K_COL(b+3);
#undef Q3_K_COL
            acc[g] += LD4(sh_d, b) * rf;
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
