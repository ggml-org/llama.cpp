#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#define KQ_TN 32
#define KQ_MB 8
#define KQ_TM 64
#define KQ_WG 64
#define KQ_DKC 256

__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_mul_mm_q8_kq_p8_dk512(
        global const uint  * kq,        // K int8, [head_kv][n_kv][dk]   as packed uints
        global const half  * kd,        // K scales, [head_kv][n_kv][dk/32]
        global const uint  * qq,        // Q int8, [head][n_q][dk]       as packed uints
        global const half  * qd,        // Q scales, [head][n_q][dk/32]
        global       uchar * qp,        // u8 P, [head][n_q][p_pitch]
        global       float * bmax,      // per 32-row block max,          [head][n_q][p_pitch/32]
        global       float * bsum,      // per block sum of exp(s - bmax), [head][n_q][p_pitch/32]
        global const char  * mask,      // f16 [>=n_kv][n_q] view, row stride mask_nb1 bytes
        ulong                offset_mask,
        ulong                mask_nb1,
        float                scale,
        float                max_bias,
        float                m0,
        float                m1,
        int                  n_head_log2,
        int                  dk,
        int                  n_kv,
        int                  n_q,
        int                  n_head,
        int                  n_head_kv,
        int                  p_pitch     // bytes per P row, >= n_kv, multiple of 32
) {
    const int lid  = get_local_id(0);
    const int gsz  = n_head / n_head_kv;
    const int g0   = get_group_id(0);
    const int head_kv = get_group_id(2);
    const int head = head_kv*gsz + (g0 % gsz);
    const int qn0  = (g0 / gsz) * KQ_TN;
    const int kv0  = get_group_id(1)*KQ_TM*KQ_MB + lid;

    const int nblk = dk / 32;
    const int nu   = dk / 4;

    // Reuse the 256-wide Q slice for the F32 softmax scores after the contraction.
    __local uint  sh_q [KQ_TN*KQ_DKC/4];
    __local half  sh_qd[KQ_TN][KQ_DKC/32];
    __local float * p8_lds = (__local float *)sh_q;

    const size_t qbase = (size_t)head*n_q;

    float slope = 1.0f;
    if (max_bias > 0.0f) {
        const float base = head < n_head_log2 ? m0 : m1;
        const int   ex   = head < n_head_log2 ? head + 1 : 2*(head - n_head_log2) + 1;
        slope = pow(base, ex);
    }
    global const half * mrow = (global const half *)(mask + offset_mask);
    const uint mstride = (uint)(mask_nb1 / 2);
    const int  pblk    = p_pitch / 32;

    for (int m = 0; m < KQ_MB; ++m) {
        const int kv = kv0 + m*KQ_TM;
        if (kv - lid >= n_kv) {
            break; // uniform across the workgroup
        }

        float acc[KQ_TN];
        #pragma unroll
        for (int t = 0; t < KQ_TN; ++t) {
            acc[t] = 0.0f;
        }

        const int    kl    = min(kv, n_kv - 1);
        const size_t kbase = ((size_t)head_kv*n_kv + kl)*nu;
        const size_t kdbas = ((size_t)head_kv*n_kv + kl)*nblk;

        for (int c0 = 0; c0 < nblk; c0 += KQ_DKC/32) {
            const int cb = min(KQ_DKC/32, nblk - c0);
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int i = lid; i < KQ_TN*(KQ_DKC/16); i += KQ_WG) {
                const int t  = i / (KQ_DKC/16);
                const int u  = (i % (KQ_DKC/16)) * 4;
                const int qi = qn0 + t;
                uint4 v = (uint4)(0u);
                if (qi < n_q && u < cb*8) {
                    v = vload4(0, &qq[(qbase + qi)*nu + c0*8 + u]);
                }
                vstore4(v, 0, &sh_q[t*(KQ_DKC/4) + u]);
            }
            for (int i = lid; i < KQ_TN*(KQ_DKC/32); i += KQ_WG) {
                const int t  = i / (KQ_DKC/32);
                const int b  = i % (KQ_DKC/32);
                const int qi = qn0 + t;
                sh_qd[t][b] = (qi < n_q && b < cb) ? qd[(qbase + qi)*nblk + c0 + b] : (half)0.0f;
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            for (int b = 0; b < cb; ++b) {
                const uint4 w0 = vload4(0, &kq[kbase + (c0 + b)*8]);
                const uint4 w1 = vload4(0, &kq[kbase + (c0 + b)*8 + 4]);
                const float dks = (float)kd[kdbas + c0 + b];

                #pragma unroll
                for (int t = 0; t < KQ_TN; ++t) {
                    const uint4 a0 = vload4(0, &sh_q[t*(KQ_DKC/4) + b*8]);
                    const uint4 a1 = vload4(0, &sh_q[t*(KQ_DKC/4) + b*8 + 4]);
                    int raw = 0;
                    raw = dot_acc_sat_4x8packed_ss_int(w0.s0, a0.s0, raw);
                    raw = dot_acc_sat_4x8packed_ss_int(w0.s1, a0.s1, raw);
                    raw = dot_acc_sat_4x8packed_ss_int(w0.s2, a0.s2, raw);
                    raw = dot_acc_sat_4x8packed_ss_int(w0.s3, a0.s3, raw);
                    raw = dot_acc_sat_4x8packed_ss_int(w1.s0, a1.s0, raw);
                    raw = dot_acc_sat_4x8packed_ss_int(w1.s1, a1.s1, raw);
                    raw = dot_acc_sat_4x8packed_ss_int(w1.s2, a1.s2, raw);
                    raw = dot_acc_sat_4x8packed_ss_int(w1.s3, a1.s3, raw);
                    acc[t] += dks * (float)sh_qd[t][b] * (float)raw;
                }
            }
        }

        // Finish reading the Q slice before writing scores to the same local memory.
        const int kvblk0 = kv - lid;
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int t = 0; t < KQ_TN; ++t) {
            p8_lds[t*64 + lid] = acc[t];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        {
            const int j = lid >> 1;
            const int b = lid & 1;
            const int n = qn0 + j;
            if (n < n_q) {
                const int kvb = kvblk0 + b*32;
                __local float * src = p8_lds + j*64 + b*32;
                const global half * mp = mrow + (size_t)n*mstride + kvb;
                float4 s = vload4(0, src)*scale + slope*convert_float4(vload4(0, mp));
                vstore4(s, 0, src);
                float amax = fmax(fmax(s.x, s.y), fmax(s.z, s.w));
                for (int c = 1; c < 8; ++c) {
                    s = vload4(c, src)*scale + slope*convert_float4(vload4(c, mp));
                    vstore4(s, c, src);
                    amax = fmax(amax, fmax(fmax(s.x, s.y), fmax(s.z, s.w)));
                }
                const size_t prow = qbase + n;
                const int    blk  = kvb >> 5;
                global uchar * qdst = qp + prow*(size_t)p_pitch + (size_t)kvb;
                float sum = 0.0f;
                // finite sentinel: the program builds with -cl-finite-math-only
                if (amax > -1.0e30f) {
                    for (int c = 0; c < 8; ++c) {
                        const float4 e = exp(vload4(c, src) - amax);
                        sum += (e.x + e.y) + (e.z + e.w);
                        vstore4(convert_uchar4_sat_rte(e*255.0f), c, qdst);
                    }
                } else {
                    for (int c = 0; c < 8; ++c) {
                        vstore4((uchar4)(0), c, qdst);
                    }
                }
                bmax[prow*pblk + blk] = amax;
                bsum[prow*pblk + blk] = sum;
            }
        }
    }
}
