#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#define QK_K 256

#ifndef TILESIZE_N
#define TILESIZE_N 32
#endif

inline uint q2_k_pack(uint pk) {
    return ( (pk      ) & 3u)
         | (((pk >> 2) & 3u) <<  8)
         | (((pk >> 4) & 3u) << 16)
         | (((pk >> 6) & 3u) << 24);
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
kernel void kernel_gemm_noshuffle_q2_k_q8_1_dp4a(
        __global const uchar  * src0_qs,
        __global const uchar  * src0_sc,
        __global const half   * src0_dm,   // half pairs: d, dmin
        __global const uint   * src1_qa,   // q8_1 activations int8 (as uint, 4/elem) [N, K]
        __global const half   * src1_da,   // q8_1 per-block scale [N, K/32]
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
    __local float sh_s [TILESIZE_N][2];   // per-16 activation sums, in qa units
    __local half  sh_d [TILESIZE_N];

#define NGROUPS (TILESIZE_N / 4)
    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc[g] = (float4)(0.0f);

    for (uint step = 0; step < (uint)k; step += 32) {
        const uint sub = step >> 5;
        const uint ib  = sub >> 3;

        const half2 dm  = vload2(rrow + ib * (uint)m, src0_dm);
        const float dv  = (float)dm.s0;
        const float dmv = (float)dm.s1;

        const uint  scb = rrow + (sub << 1) * (uint)m;
        const uint  sc0 = (uint)src0_sc[scb + 0u * (uint)m];
        const uint  sc1 = (uint)src0_sc[scb + 1u * (uint)m];
        const float dl0 = dv  * (float)( sc0       & 0xFu);
        const float ml0 = dmv * (float)( sc0 >> 4);
        const float dl1 = dv  * (float)( sc1       & 0xFu);
        const float ml1 = dmv * (float)( sc1 >> 4);

        const uint qsb = rrow + (step >> 2) * (uint)m;
        uint4 qlo, qhi;
        qlo.s0 = q2_k_pack((uint)src0_qs[qsb + 0u * (uint)m]);
        qlo.s1 = q2_k_pack((uint)src0_qs[qsb + 1u * (uint)m]);
        qlo.s2 = q2_k_pack((uint)src0_qs[qsb + 2u * (uint)m]);
        qlo.s3 = q2_k_pack((uint)src0_qs[qsb + 3u * (uint)m]);
        qhi.s0 = q2_k_pack((uint)src0_qs[qsb + 4u * (uint)m]);
        qhi.s1 = q2_k_pack((uint)src0_qs[qsb + 5u * (uint)m]);
        qhi.s2 = q2_k_pack((uint)src0_qs[qsb + 6u * (uint)m]);
        qhi.s3 = q2_k_pack((uint)src0_qs[qsb + 7u * (uint)m]);

        for (uint idx = lid; idx < TILESIZE_N * 2u; idx += 64u) {
            const uint t  = idx >> 1;
            const uint h  = idx & 1u;
            const uint c  = col_base + t;
            const bool ok = c < (uint)n_no_padding;
            const uint4 w = ok ? KQ_STAGE(src1_qa + c * k_u + (step >> 2) + (h << 2))
                               : (uint4)(0u);
            sh_qa4[t][h] = w;
            int s = 0;
            s = dot_acc_sat_4x8packed_ss_int(w.x, 0x01010101u, s);
            s = dot_acc_sat_4x8packed_ss_int(w.y, 0x01010101u, s);
            s = dot_acc_sat_4x8packed_ss_int(w.z, 0x01010101u, s);
            s = dot_acc_sat_4x8packed_ss_int(w.w, 0x01010101u, s);
            sh_s[t][h] = (float)s;
            if (h == 0u) {
                sh_d[t] = ok ? src1_da[c * k_b + sub] : (half)0;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

#define LD4(arr, b) ((float4)((float)arr[(b)+0], (float)arr[(b)+1], (float)arr[(b)+2], (float)arr[(b)+3]))
#define Q2_K_COL(b) (dl0 * (float)dot4_q8a_v(qlo, (uint4)(sh_qa4[b][0])) - ml0 * sh_s[b][0] + \
                    dl1 * (float)dot4_q8a_v(qhi, (uint4)(sh_qa4[b][1])) - ml1 * sh_s[b][1])
        #pragma unroll
        for (int g = 0; g < NGROUPS; ++g) {
            const int b = g * 4;
            float4 rf;
            rf.s0 = Q2_K_COL(b+0);  rf.s1 = Q2_K_COL(b+1);
            rf.s2 = Q2_K_COL(b+2);  rf.s3 = Q2_K_COL(b+3);
            acc[g] += LD4(sh_d, b) * rf;
        }
#undef Q2_K_COL
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
