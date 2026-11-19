#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#define QK_K 256

#ifndef TILESIZE_N
#define TILESIZE_N 32
#endif

inline uint iq1_m_pack(uint g, uint shift, uint negative_delta) {
    // 8*(grid + delta) is an exact signed integer in [-9, 9].
    const int bias = negative_delta ? -9 : -7;
    const int4 values = 8*convert_int4(as_uchar4((g >> shift) & 0x0F0F0F0Fu)) + bias;
    return as_uint(convert_char4(values));
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
kernel void kernel_gemm_noshuffle_iq1_m_q8_1_dp4a(
        __global const uchar  * src0_qs,
        __global const uchar  * src0_qh,
        __global const ushort * src0_sc,
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

#define IQ1_M_GRID(i) iq1_s_grid[(i)]

#define NGROUPS (TILESIZE_N / 4)
    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc[g] = (float4)(0.0f);

    float dsuper = 0.f;
    for (uint step = 0; step < (uint)k; step += 32) {
        const uint sub = step >> 5;
        const uint ib  = sub >> 3;

        if ((sub & 7u) == 0u) {
            const uint sb = rrow + 4u*ib*(uint)m;
            const uint w0 = src0_sc[sb + 0u*(uint)m];
            const uint w1 = src0_sc[sb + 1u*(uint)m];
            const uint w2 = src0_sc[sb + 2u*(uint)m];
            const uint w3 = src0_sc[sb + 3u*(uint)m];
            const ushort bits = (ushort)((w0 >> 12) | ((w1 >> 8) & 0x00f0u)
                                       | ((w2 >> 4) & 0x0f00u) | (w3 & 0xf000u));
            dsuper = (float)as_half(bits);
        }
        const uint sc = src0_sc[rrow + (4u*ib + (sub & 7u)/2u)*(uint)m];
        const uint sh = 6u*(sub & 1u);
        const float dl0 = dsuper * (float)(2u*((sc >> sh) & 7u) + 1u) * 0.125f;
        const float dl1 = dsuper * (float)(2u*((sc >> (sh + 3u)) & 7u) + 1u) * 0.125f;
        const uint h0 = src0_qh[rrow + (2u*sub + 0u)*(uint)m];
        const uint h1 = src0_qh[rrow + (2u*sub + 1u)*(uint)m];
        const uint gb = rrow + 4u*sub*(uint)m;
        const uint g0 = IQ1_M_GRID((uint)src0_qs[gb + 0u*(uint)m] | (( h0       & 7u) << 8));
        const uint g1 = IQ1_M_GRID((uint)src0_qs[gb + 1u*(uint)m] | (((h0 >> 4) & 7u) << 8));
        const uint g2 = IQ1_M_GRID((uint)src0_qs[gb + 2u*(uint)m] | (( h1       & 7u) << 8));
        const uint g3 = IQ1_M_GRID((uint)src0_qs[gb + 3u*(uint)m] | (((h1 >> 4) & 7u) << 8));
        const uint4 qlo = (uint4)(iq1_m_pack(g0, 0, h0 & 0x08u), iq1_m_pack(g0, 4, h0 & 0x08u),
                                 iq1_m_pack(g1, 0, h0 & 0x80u), iq1_m_pack(g1, 4, h0 & 0x80u));
        const uint4 qhi = (uint4)(iq1_m_pack(g2, 0, h1 & 0x08u), iq1_m_pack(g2, 4, h1 & 0x08u),
                                 iq1_m_pack(g3, 0, h1 & 0x80u), iq1_m_pack(g3, 4, h1 & 0x80u));

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
#define IQ1_M_COL(b) (dl0 * (float)dot4_q8a_v(qlo, (uint4)(sh_qa4[b][0])) \
                           + dl1 * (float)dot4_q8a_v(qhi, (uint4)(sh_qa4[b][1])))
        #pragma unroll
        for (int g = 0; g < NGROUPS; ++g) {
            const int b = g * 4;
            float4 rf;
            rf.s0 = IQ1_M_COL(b+0);  rf.s1 = IQ1_M_COL(b+1);
            rf.s2 = IQ1_M_COL(b+2);  rf.s3 = IQ1_M_COL(b+3);
            acc[g] += LD4(sh_d, b) * rf;
        }
#undef IQ1_M_COL
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
#undef IQ1_M_GRID
}
