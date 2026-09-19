#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

// Dense Q2_K prefill GEMM, dp4a (int8) inner loop, over the feature-major plane
// split produced by kernel_convert_block_q2_k_ns:
//
//   src0_qs[row + (k/4)*m]     uchar  four 2-bit values (0..3)
//   src0_sc[row + (k/16)*m]    uchar  low nibble scale, high nibble min
//   src0_dm[row + (k/256)*m]   half2  d, dmin
//
// Unlike q3_K this type carries a MIN, so per 16-weight run
//
//   sum_k a_k*(dl*q_k - ml) = da * ( dl*<q,qa> - ml*sum(qa) )
//
// with a_k = da*qa_k for q8_1 activations. The kernel therefore needs a per-16
// activation sum. q8_1's own `sa` plane is per-32 and cannot supply it, and
// recomputing it per row (as CUDA's MMQ does, by dp4a'ing a replicated min) would
// double the dp4a count. It does not depend on the row, so instead the LDS
// staging is restructured: each of the 64 lanes owns exactly one (column, half)
// = 4 uints, and sums them as it stores them. The sum costs 4 dp4a per column
// half per WORKGROUP rather than per row, and adds no barrier.
//
// That mapping is why TILESIZE_N must stay 32 with a 64-lane workgroup:
// 2 halves * 32 columns == 64 lanes exactly.

#define QK_K 256

// TILESIZE_N is the token tile: it fixes the accumulator count and the LDS
// staging width, so it is compile-time, and the right value is PER DEVICE.
// This kernel used to be locked at 32 because it mapped one (column, half) per
// lane; that staging is now a strided loop, so any tile is correct. The
// dispatch must pass the SAME value -- see ggml_cl_lowbit_dp4a_ts.
#ifndef TILESIZE_N
#define TILESIZE_N 32
#endif

// Four weights of one group as packed int8. Q2_K values are UNSIGNED 0..3; the
// offset lives in the separate min term, not in the quant.
inline uint q2k_pack(uint pk) {
    return ( (pk      ) & 3u)
         | (((pk >> 2) & 3u) <<  8)
         | (((pk >> 4) & 3u) << 16)
         | (((pk >> 6) & 3u) << 24);
}

// The activation tile is staged as uint4, not uint: the eight uints a token needs
// for one 32-K step are contiguous, so they are two uint4s. That cuts the inner
// loop's __local load count 4x and widens the cooperative staging load from 4 to
// 16 bytes per lane. Measured on the IQ4_XS twin of this kernel: 3B pp512
// 675 -> 780 (+15.6%), 27B 72.2 -> 78.2.
//
// The uint4s are copied into private temps at the call site -- dp4a with a
// __local operand inside an unrolled loop is a documented miscompile on X2.
// This kernel is miscompiled on A7X (E031.41) and on E17; both are declined in
// ggml_cl_kquant_plane_dp4a_gemm_on.

#define KQ_STAGE(p) vload4(0, (p))

inline int dot4_q8a_v(uint4 qw, uint4 a) {
    int r = 0;
    r = dot_acc_sat_4x8packed_ss_int(qw.s0, a.x, r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s1, a.y, r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s2, a.z, r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s3, a.w, r);
    return r;
}

// One output column. The staged operands arrive by value, so they are
// private here whatever the caller passed.
inline float q2k_col(uint4 qlo, uint4 qhi, float dl0, float dl1,
                     float ml0, float ml1, uint4 a0, uint4 a1,
                     float s0, float s1) {
    return dl0 * (float)dot4_q8a_v(qlo, a0) - ml0 * s0
         + dl1 * (float)dot4_q8a_v(qhi, a1) - ml1 * s1;
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
        qlo.s0 = q2k_pack((uint)src0_qs[qsb + 0u * (uint)m]);
        qlo.s1 = q2k_pack((uint)src0_qs[qsb + 1u * (uint)m]);
        qlo.s2 = q2k_pack((uint)src0_qs[qsb + 2u * (uint)m]);
        qlo.s3 = q2k_pack((uint)src0_qs[qsb + 3u * (uint)m]);
        qhi.s0 = q2k_pack((uint)src0_qs[qsb + 4u * (uint)m]);
        qhi.s1 = q2k_pack((uint)src0_qs[qsb + 5u * (uint)m]);
        qhi.s2 = q2k_pack((uint)src0_qs[qsb + 6u * (uint)m]);
        qhi.s3 = q2k_pack((uint)src0_qs[qsb + 7u * (uint)m]);

        // one (column, half) per lane, so the half's activation sum falls out of
        // the same four loads -- see the header note
        // Strided over (column, half) rather than one slot per lane. At
        // TILESIZE_N=32 that is TILESIZE_N*2 == 64 == the workgroup, so idx==lid
        // once and this is byte-identical to the direct map it replaces -- but it
        // is now correct at any tile, which is what lets this kernel take the
        // per-device TILESIZE_N the others already take.
        for (uint idx = lid; idx < TILESIZE_N * 2u; idx += 64u) {
            const uint t  = idx >> 1;
            const uint h  = idx & 1u;
            const uint c  = col_base + t;
            const bool ok = c < (uint)n_no_padding;
            // the lane's whole half is ONE uint4, so the sum now falls out of a
            // single 16-byte load rather than four 4-byte ones
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
#define Q2K_COL(b) (dl0 * (float)dot4_q8a_v(qlo, (uint4)(sh_qa4[b][0])) - ml0 * sh_s[b][0] + \
                    dl1 * (float)dot4_q8a_v(qhi, (uint4)(sh_qa4[b][1])) - ml1 * sh_s[b][1])
        #pragma unroll
        for (int g = 0; g < NGROUPS; ++g) {
            const int b = g * 4;
            float4 rf;
            rf.s0 = Q2K_COL(b+0);  rf.s1 = Q2K_COL(b+1);
            rf.s2 = Q2K_COL(b+2);  rf.s3 = Q2K_COL(b+3);
            acc[g] += LD4(sh_d, b) * rf;
        }
#undef Q2K_COL
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
