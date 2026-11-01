#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

// int8 KQ for the decomposed prefill: out[kv, q, h] = sum_d K[d, kv, h_kv] * Q[d, q, h], with K
// and Q quantised per call to symmetric int8, one half scale per 32 along d.
// One lane owns one kv row and KQ_TN queries and a workgroup walks KQ_MB 64-row kv blocks. The
// separate kernel stages the Q tile once; the fused one stages it per KQ_DKC-wide head slice. Dispatch order (matched with the host): query
// tile, then kv block, then KV head, so each K block is reused from L2 across its GQA group.
// The packed dot takes the K word (registers) first, the Q word (local) second; see mul_mm_q8_kqv.cl.

#ifndef KQ_TN
#define KQ_TN 32        // queries per workgroup
#endif
#ifndef KQ_MB
#define KQ_MB 8         // 64-row kv blocks per workgroup, sharing one staged Q tile
#endif
#define KQ_TM 64        // kv rows per block, one per lane
#define KQ_WG 64
#ifndef KQ_DK_MAX
#define KQ_DK_MAX 256   // local memory is sized for this; a second build at 512 serves larger heads
#endif
#define KQ_DKC 256      // head-size slice the fused kernel stages per kv block

// ---- row quantisation: [nhead][nrow][ne0] -> int8 + per-32 half scale --------------------
// One lane per 32-block. src rows are strided (nb1 bytes per row, nb2 per head) so a KV-cache
// view is read in place; the outputs are dense.

kernel void kernel_fa_q8_rows_f16(
        global const char * src, ulong off_src, ulong nb1, ulong nb2,
        global char * dq, global half * dd,
        int ne0, int nrow, int nhead
) {
    const int nblk = ne0 / 32;
    const int gid  = get_global_id(0);
    if (gid >= nblk*nrow*nhead) {
        return;
    }
    const int b = gid % nblk;
    const int r = (gid / nblk) % nrow;
    const int h = gid / (nblk*nrow);

    global const half * p = (global const half *)(src + off_src + (ulong)h*nb2 + (ulong)r*nb1) + b*32;
    float v[32];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float4 x = convert_float4(vload4(i, p));
        v[i*4+0] = x.s0; v[i*4+1] = x.s1; v[i*4+2] = x.s2; v[i*4+3] = x.s3;
        amax = fmax(amax, fmax(fmax(fabs(x.s0), fabs(x.s1)), fmax(fabs(x.s2), fabs(x.s3))));
    }
    const float d  = amax / 127.0f;
    const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
    const size_t row = (size_t)h*nrow + r;
    dd[row*nblk + b] = (half)d;
    global char * q = dq + row*ne0 + b*32;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        q[i] = (char)clamp((int)rint(v[i]*id), -127, 127);
    }
}

// q8_0 KV cache -> the same planes: a q8_0 block {half d; char qs[32]} already is this
// quantisation, so this splits the cached blocks into the two dense planes without requantising.
kernel void kernel_fa_q8_rows_q8_0(
        global const char * src, ulong off_src, ulong nb1, ulong nb2,
        global char * dq, global half * dd,
        int ne0, int nrow, int nhead
) {
    const int nblk = ne0 / 32;
    const int gid  = get_global_id(0);
    if (gid >= nblk*nrow*nhead) {
        return;
    }
    const int b = gid % nblk;
    const int r = (gid / nblk) % nrow;
    const int h = gid / (nblk*nrow);

    global const char * blk = src + off_src + (ulong)h*nb2 + (ulong)r*nb1 + (ulong)b*34;
    const size_t row = (size_t)h*nrow + r;
    dd[row*nblk + b] = *(global const half *)blk;
    global char * q = dq + row*ne0 + b*32;
    vstore16(vload16(0, blk + 2), 0, q);
    vstore16(vload16(1, blk + 2), 1, q);
}

kernel void kernel_fa_q8_rows_f32(
        global const char * src, ulong off_src, ulong nb1, ulong nb2,
        global char * dq, global half * dd,
        int ne0, int nrow, int nhead
) {
    const int nblk = ne0 / 32;
    const int gid  = get_global_id(0);
    if (gid >= nblk*nrow*nhead) {
        return;
    }
    const int b = gid % nblk;
    const int r = (gid / nblk) % nrow;
    const int h = gid / (nblk*nrow);

    global const float * p = (global const float *)(src + off_src + (ulong)h*nb2 + (ulong)r*nb1) + b*32;
    float v[32];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float4 x = vload4(i, p);
        v[i*4+0] = x.s0; v[i*4+1] = x.s1; v[i*4+2] = x.s2; v[i*4+3] = x.s3;
        amax = fmax(amax, fmax(fmax(fabs(x.s0), fabs(x.s1)), fmax(fabs(x.s2), fabs(x.s3))));
    }
    const float d  = amax / 127.0f;
    const float id = amax > 0.0f ? 127.0f / amax : 0.0f;
    const size_t row = (size_t)h*nrow + r;
    dd[row*nblk + b] = (half)d;
    global char * q = dq + row*ne0 + b*32;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        q[i] = (char)clamp((int)rint(v[i]*id), -127, 127);
    }
}

// ---- the GEMM -----------------------------------------------------------------------------

__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_mul_mm_q8_kq(
        global const uint  * kq,        // K int8, [head_kv][n_kv][dk]   as packed uints
        global const half  * kd,        // K scales, [head_kv][n_kv][dk/32]
        global const uint  * qq,        // Q int8, [head][n_q][dk]       as packed uints
        global const half  * qd,        // Q scales, [head][n_q][dk/32]
        global       float * dst,       // [head][n_q][n_kv]
        ulong                off_dst,
        int                  dk,
        int                  n_kv,
        int                  n_q,
        int                  n_head,
        int                  n_head_kv
) {
    dst = (global float *)((global char *)dst + off_dst);

    const int lid  = get_local_id(0);
    const int gsz  = n_head / n_head_kv;
    const int g0   = get_group_id(0);
    const int head_kv = get_group_id(2);
    const int head = head_kv*gsz + (g0 % gsz);
    const int qn0  = (g0 / gsz) * KQ_TN;
    const int kv0  = get_group_id(1)*KQ_TM*KQ_MB + lid;

    const int nblk = dk / 32;
    const int nu   = dk / 4;

    __local uint sh_q [KQ_TN][KQ_DK_MAX/4];
    __local half sh_qd[KQ_TN][KQ_DK_MAX/32];

    const size_t qbase = (size_t)head*n_q;
    for (int i = lid; i < KQ_TN*(KQ_DK_MAX/16); i += KQ_WG) {
        const int t = i / (KQ_DK_MAX/16);
        const int u = (i % (KQ_DK_MAX/16)) * 4;
        const int qi = qn0 + t;
        uint4 v = (uint4)(0u);
        if (qi < n_q && u < nu) {
            v = vload4(0, &qq[(qbase + qi)*nu + u]);
        }
        vstore4(v, 0, &sh_q[t][u]);
    }
    for (int i = lid; i < KQ_TN*(KQ_DK_MAX/32); i += KQ_WG) {
        const int t = i / (KQ_DK_MAX/32);
        const int b = i % (KQ_DK_MAX/32);
        const int qi = qn0 + t;
        sh_qd[t][b] = (qi < n_q && b < nblk) ? qd[(qbase + qi)*nblk + b] : (half)0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int m = 0; m < KQ_MB; ++m) {
        const int kv = kv0 + m*KQ_TM;
        if (kv - lid >= n_kv) {
            break;                                  // whole block past the end, uniform per workgroup
        }

        float acc[KQ_TN];
        #pragma unroll
        for (int t = 0; t < KQ_TN; ++t) {
            acc[t] = 0.0f;
        }

        const int    kl    = min(kv, n_kv - 1);     // clamp so a tail lane loads in bounds
        const size_t kbase = ((size_t)head_kv*n_kv + kl)*nu;
        const size_t kdbas = ((size_t)head_kv*n_kv + kl)*nblk;

        for (int b = 0; b < nblk; ++b) {
            const uint4 w0 = vload4(0, &kq[kbase + b*8]);
            const uint4 w1 = vload4(0, &kq[kbase + b*8 + 4]);
            const float dks = (float)kd[kdbas + b];

            #pragma unroll
            for (int t = 0; t < KQ_TN; ++t) {
                const uint4 a0 = vload4(0, &sh_q[t][b*8]);
                const uint4 a1 = vload4(0, &sh_q[t][b*8 + 4]);
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

        if (kv < n_kv) {
            // dst is [head][n_q][n_kv] with n_kv contiguous, so the 64 lanes of a kv block write contiguously
            #pragma unroll
            for (int t = 0; t < KQ_TN; ++t) {
                const int qi = qn0 + t;
                if (qi < n_q) {
                    dst[(qbase + qi)*n_kv + kv] = acc[t];
                }
            }
        }
    }
}

// ---- the GEMM with the block softmax folded in (see mul_mm_f16_f32_kq_p8) ----------------
// Same tile and dispatch as kernel_mul_mm_q8_kq; instead of the f32 score row each 64-row kv
// block is transposed through local memory and each (query, 32-row block) is scaled, masked,
// quantised to u8 against its block max and summed, with the block max and sum written to
// side arrays for kernel_fa_p8_fixup. Requires KQ_TN == 32 (two 16-column halves).
#if KQ_TN == 32
#if KQ_DK_MAX > 256
// Head sizes past 256: Q is staged per KQ_DKC-wide slice and the scores reuse its local memory,
// so the footprint does not grow with the head size.
__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_mul_mm_q8_kq_p8(
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

    // Local memory is what bounds residency here, so it is sized for one KQ_DKC-wide slice of the
    // Q tile, restaged per kv block (the tile stays in L2), and the softmax scores of a block reuse
    // the same words once its contraction is done: 8.5 KB at any head size.
    __local uint  sh_q [KQ_TN*KQ_DKC/4];               // also the [KQ_TN][64] f32 scores
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
            break;                                  // whole block past the end, uniform per workgroup
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
            const int cb = min(KQ_DKC/32, nblk - c0);   // 32-blocks in this slice
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

        // ---- fused block softmax epilogue (n_kv is a multiple of 64: whole block valid) ----
        // All accumulators go to local memory before any softmax math, so none is live across it,
        // and each of the 64 lanes then takes one (query, 32-row block) pair.
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
#else
__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_mul_mm_q8_kq_p8(
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

    __local uint  sh_q [KQ_TN][KQ_DK_MAX/4];
    __local half  sh_qd[KQ_TN][KQ_DK_MAX/32];
    __local float p8_lds[1024];

    const size_t qbase = (size_t)head*n_q;
    for (int i = lid; i < KQ_TN*(KQ_DK_MAX/16); i += KQ_WG) {
        const int t = i / (KQ_DK_MAX/16);
        const int u = (i % (KQ_DK_MAX/16)) * 4;
        const int qi = qn0 + t;
        uint4 v = (uint4)(0u);
        if (qi < n_q && u < nu) {
            v = vload4(0, &qq[(qbase + qi)*nu + u]);
        }
        vstore4(v, 0, &sh_q[t][u]);
    }
    for (int i = lid; i < KQ_TN*(KQ_DK_MAX/32); i += KQ_WG) {
        const int t = i / (KQ_DK_MAX/32);
        const int b = i % (KQ_DK_MAX/32);
        const int qi = qn0 + t;
        sh_qd[t][b] = (qi < n_q && b < nblk) ? qd[(qbase + qi)*nblk + b] : (half)0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

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
            break;                                  // whole block past the end, uniform per workgroup
        }

        float acc[KQ_TN];
        #pragma unroll
        for (int t = 0; t < KQ_TN; ++t) {
            acc[t] = 0.0f;
        }

        const int    kl    = min(kv, n_kv - 1);
        const size_t kbase = ((size_t)head_kv*n_kv + kl)*nu;
        const size_t kdbas = ((size_t)head_kv*n_kv + kl)*nblk;

        for (int b = 0; b < nblk; ++b) {
            const uint4 w0 = vload4(0, &kq[kbase + b*8]);
            const uint4 w1 = vload4(0, &kq[kbase + b*8 + 4]);
            const float dks = (float)kd[kdbas + b];

            #pragma unroll
            for (int t = 0; t < KQ_TN; ++t) {
                const uint4 a0 = vload4(0, &sh_q[t][b*8]);
                const uint4 a1 = vload4(0, &sh_q[t][b*8 + 4]);
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

        // ---- fused block softmax epilogue (n_kv is a multiple of 64: whole block valid) ----
        const int kvblk0 = kv - lid;
        #pragma unroll
        for (int half_ = 0; half_ < 2; ++half_) {
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                p8_lds[j*64 + lid] = acc[half_*16 + j];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if (lid < 32) {
                const int j = lid >> 1;
                const int b = lid & 1;
                const int n = qn0 + half_*16 + j;
                if (n < n_q) {
                    const int kvb = kvblk0 + b*32;
                    const __local float * src = p8_lds + j*64 + b*32;
                    float16 s0 = vload16(0, src);
                    float16 s1 = vload16(1, src);
                    const global half * mp = mrow + (size_t)n*mstride + kvb;
                    const float16 m0v = convert_float16(vload16(0, mp));
                    const float16 m1v = convert_float16(vload16(1, mp));
                    s0 = s0*scale + slope*m0v;
                    s1 = s1*scale + slope*m1v;
                    const float16 mx16 = fmax(s0, s1);
                    const float8  mx8  = fmax(mx16.lo, mx16.hi);
                    const float4  mx4  = fmax(mx8.lo, mx8.hi);
                    const float2  mx2  = fmax(mx4.lo, mx4.hi);
                    const float   amax = fmax(mx2.x, mx2.y);
                    const size_t prow = qbase + n;
                    const int    blk  = kvb >> 5;
                    float sum = 0.0f;
                    uchar16 q0 = (uchar16)(0);
                    uchar16 q1 = (uchar16)(0);
                    // finite sentinel: the program builds with -cl-finite-math-only
                    if (amax > -1.0e30f) {
                        const float16 e0 = exp(s0 - amax);
                        const float16 e1 = exp(s1 - amax);
                        const float16 a16 = e0 + e1;
                        const float8  a8  = a16.lo + a16.hi;
                        const float4  a4  = a8.lo + a8.hi;
                        const float2  a2  = a4.lo + a4.hi;
                        sum = a2.x + a2.y;
                        q0 = convert_uchar16_sat_rte(e0*255.0f);
                        q1 = convert_uchar16_sat_rte(e1*255.0f);
                    }
                    global uchar * qdst = qp + prow*(size_t)p_pitch + (size_t)kvb;
                    vstore16(q0, 0, qdst);
                    vstore16(q1, 1, qdst);
                    bmax[prow*pblk + blk] = amax;
                    bsum[prow*pblk + blk] = sum;
                }
            }
        }
    }
}
#endif // KQ_DK_MAX > 256
#endif // KQ_TN == 32
