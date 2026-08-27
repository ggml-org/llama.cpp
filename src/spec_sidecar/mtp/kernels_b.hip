// SPDX-License-Identifier: MIT
// batched variants (catch-up: ncols rows in one launch per op)
// packed layouts: x[col*stride_in + i], y[col*stride_out + i]

__global__ void gemv_q4_0_b(const block_q4_0 * __restrict__ W, const float * __restrict__ x,
                            float * __restrict__ y, int n_in, int n_out) {
    const int row = blockIdx.x, col = blockIdx.y, lane = threadIdx.x;
    if (row >= n_out) return;
    const float * xc = x + (size_t) col * n_in;
    const int nblk = n_in / 32;
    const block_q4_0 * wr = W + (size_t) row * nblk;
    float acc = 0.0f;
    for (int b = lane; b < nblk; b += 32) {
        const block_q4_0 * blk = &wr[b];
        const float d = f16_to_f32(blk->d);
        const float * xb = xc + b * 32;
        float s = 0.0f;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const uint8_t byte = blk->qs[j];
            s += (float) ((byte & 0x0F) - 8) * xb[j];
            s += (float) ((byte >> 4)   - 8) * xb[j + 16];
        }
        acc += d * s;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off, 32);
    if (lane == 0) y[(size_t) col * n_out + row] = acc;
}

__global__ void deq_rows_q4_0_b8(const block_q4_0 * __restrict__ W,
                                  int t0, int t1, int t2, int t3,
                                  int t4, int t5, int t6, int t7,
                                  float * __restrict__ y, int n_in, int n_rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, col = blockIdx.y;
    if (i >= n_in || col >= n_rows) return;
    const int toks[8] = { t0, t1, t2, t3, t4, t5, t6, t7 };
    const int nblk = n_in / 32;
    const block_q4_0 * b = W + (size_t) toks[col] * nblk + (i >> 5);
    y[(size_t) col * n_in + i] = q4_0_get(b, i & 31);
}

__global__ void rmsnorm_b(const float * __restrict__ x, const float * __restrict__ w,
                          float * __restrict__ y, int n) {
    __shared__ float ssum[256];
    const float * xc = x + (size_t) blockIdx.x * n;
    float * yc = y + (size_t) blockIdx.x * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xc[i] * xc[i];
    ssum[threadIdx.x] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if ((int) threadIdx.x < off) ssum[threadIdx.x] += ssum[threadIdx.x + off];
        __syncthreads();
    }
    const float scale = rsqrtf(ssum[0] / n + RMS_EPS);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yc[i] = xc[i] * scale * w[i];
}

// per-head norm over ncols rows: grid (nh, ncols); row stride = nh*HEAD_DIM
__global__ void rmsnorm_heads_b(const float * __restrict__ x, const float * __restrict__ w,
                                float * __restrict__ y, int nh) {
    __shared__ float ssum[256];
    const float * xh = x + ((size_t) blockIdx.y * nh + blockIdx.x) * HEAD_DIM;
    float * yh = y + ((size_t) blockIdx.y * nh + blockIdx.x) * HEAD_DIM;
    float s = 0.0f;
    for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) s += xh[i] * xh[i];
    ssum[threadIdx.x] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if ((int) threadIdx.x < off) ssum[threadIdx.x] += ssum[threadIdx.x + off];
        __syncthreads();
    }
    const float scale = rsqrtf(ssum[0] / HEAD_DIM + RMS_EPS);
    for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) yh[i] = xh[i] * scale * w[i];
}

__global__ void rope_neox_b_base(float * __restrict__ x, int nh, int pos_base) {
    const int h = blockIdx.x, col = blockIdx.y, i = threadIdx.x;
    if (i >= N_ROT / 2) return;
    float * xh = x + ((size_t) col * nh + h) * HEAD_DIM;
    const float theta = (pos_base + col) * powf(FREQ_BASE, -2.0f * i / N_ROT);
    const float c = cosf(theta), s = sinf(theta);
    const float x0 = xh[i], x1 = xh[i + N_ROT / 2];
    xh[i]             = x0 * c - x1 * s;
    xh[i + N_ROT / 2] = x0 * s + x1 * c;
}

__global__ void kv_store_b_base(const float * __restrict__ k, const float * __restrict__ v,
                                __half * __restrict__ Kc, __half * __restrict__ Vc,
                                int pos_base, int pending_base) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, col = blockIdx.y;
    const int n = N_HEAD_KV * HEAD_DIM;
    if (i < n) {
        const int pos = pos_base + col - pending_base;
        Kc[(size_t) pos * n + i] = __float2half(k[(size_t) col * n + i]);
        Vc[(size_t) pos * n + i] = __float2half(v[(size_t) col * n + i]);
    }
}

__global__ void split_q_gate_b(const float * __restrict__ qfull, float * __restrict__ q, float * __restrict__ gate) {
    const int h = blockIdx.x, col = blockIdx.y, i = threadIdx.x;
    const size_t src = ((size_t) col * N_HEAD * 2 + (size_t) h * 2) * HEAD_DIM;
    const size_t dst = ((size_t) col * N_HEAD + h) * HEAD_DIM;
    q[dst + i]    = qfull[src + i];
    gate[dst + i] = qfull[src + HEAD_DIM + i];
}

// concat pairs per col: cat[col] = [a[col] | b[col]]
__global__ void concat2_b(const float * a, const float * b, float * y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, col = blockIdx.y;
    if (i < n) y[(size_t) col * 2 * n + i] = a[(size_t) col * n + i];
    else if (i < 2 * n) y[(size_t) col * 2 * n + i] = b[(size_t) col * n + i - n];
}

// batched attention: grid (head, col); per-col length = poss[col]+1
__global__ void attn_1q_b_base(const float * __restrict__ q, const __half * __restrict__ Kc,
                               const __half * __restrict__ Vc, const __half * __restrict__ Kp,
                               const __half * __restrict__ Vp, int pending_base,
                               float * __restrict__ out, int pos_base) {
    const int h = blockIdx.x, col = blockIdx.y;
    const int kh = h / (N_HEAD / N_HEAD_KV);
    const int lane = threadIdx.x;
    const int n_pos = pos_base + col + 1;
    __shared__ float scores[ATTN_CHUNK];
    __shared__ float red[256];
    const float * qh = q + ((size_t) col * N_HEAD + h) * HEAD_DIM;
    const float kq_scale = rsqrtf((float) HEAD_DIM);
    float acc = 0.0f, m_s = -1e30f, l_s = 0.0f;
    for (int c0 = 0; c0 < n_pos; c0 += ATTN_CHUNK) {
        const int cn = min(ATTN_CHUNK, n_pos - c0);
        for (int p = lane; p < cn; p += blockDim.x) {
            const int pos = c0 + p;
            const __half * kp = (pos >= pending_base ? Kp + (size_t) (pos - pending_base) * N_HEAD_KV * HEAD_DIM : Kc + (size_t) pos * N_HEAD_KV * HEAD_DIM) + (size_t) kh * HEAD_DIM;
            float s = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < HEAD_DIM; ++i) s += qh[i] * __half2float(kp[i]);
            scores[p] = s * kq_scale;
        }
        __syncthreads();
        float m = -1e30f;
        for (int p = lane; p < cn; p += blockDim.x) m = fmaxf(m, scores[p]);
        red[lane] = m;
        __syncthreads();
        for (int off = blockDim.x / 2; off > 0; off >>= 1) {
            if (lane < off) red[lane] = fmaxf(red[lane], red[lane + off]);
            __syncthreads();
        }
        const float m_chunk = red[0];
        __syncthreads();
        float ls = 0.0f;
        for (int p = lane; p < cn; p += blockDim.x) {
            const float w = expf(scores[p] - m_chunk);
            scores[p] = w;
            ls += w;
        }
        red[lane] = ls;
        __syncthreads();
        for (int off = blockDim.x / 2; off > 0; off >>= 1) {
            if (lane < off) red[lane] += red[lane + off];
            __syncthreads();
        }
        const float l_chunk = red[0];
        __syncthreads();
        const float m_new = fmaxf(m_s, m_chunk);
        const float scale_old = expf(m_s - m_new);
        const float scale_new = expf(m_chunk - m_new);
        acc *= scale_old;
        if (lane < HEAD_DIM) {
            float vsum = 0.0f;
            for (int p = 0; p < cn; ++p) {
                const int pos = c0 + p;
                const __half * vp = (pos >= pending_base ? Vp + (size_t) (pos - pending_base) * N_HEAD_KV * HEAD_DIM : Vc + (size_t) pos * N_HEAD_KV * HEAD_DIM) + (size_t) kh * HEAD_DIM;
                vsum += scores[p] * __half2float(vp[lane]);
            }
            acc += scale_new * vsum;
        }
        l_s = l_s * scale_old + l_chunk * scale_new;
        m_s = m_new;
        __syncthreads();
    }
    if (lane < HEAD_DIM) out[((size_t) col * N_HEAD + h) * HEAD_DIM + lane] = acc / l_s;
}

// split-K single-query attention: grid (head, pos_chunk) so deep contexts use the
// whole GPU; each block emits a partial online-softmax state, merged by attn_merge.
#define ATTN_SPLIT 4096
#define ATTN_MAX_PARTS 128
__global__ void attn_1q_split(const float * __restrict__ q, const __half * __restrict__ Kc,
                              const __half * __restrict__ Vc, int n_pos,
                              float * __restrict__ part_m, float * __restrict__ part_l,
                              float * __restrict__ part_acc) {
    const int h = blockIdx.x, part = blockIdx.y;
    const int kh = h / (N_HEAD / N_HEAD_KV);
    const int lane = threadIdx.x;
    const int p0 = part * ATTN_SPLIT;
    if (p0 >= n_pos) {
        if (lane == 0) { part_m[h * ATTN_MAX_PARTS + part] = -1e30f; part_l[h * ATTN_MAX_PARTS + part] = 0.0f; }
        if (lane < HEAD_DIM) part_acc[((size_t) h * ATTN_MAX_PARTS + part) * HEAD_DIM + lane] = 0.0f;
        return;
    }
    const int pn = min(ATTN_SPLIT, n_pos - p0);
    __shared__ float scores[ATTN_SPLIT];
    __shared__ float red[256];
    const float * qh = q + h * HEAD_DIM;
    const float kq_scale = rsqrtf((float) HEAD_DIM);

    for (int p = lane; p < pn; p += blockDim.x) {
        const __half * kp = Kc + ((size_t) (p0 + p) * N_HEAD_KV + kh) * HEAD_DIM;
        float s = 0.0f;
        #pragma unroll 8
        for (int i = 0; i < HEAD_DIM; ++i) s += qh[i] * __half2float(kp[i]);
        scores[p] = s * kq_scale;
    }
    __syncthreads();
    float m = -1e30f;
    for (int p = lane; p < pn; p += blockDim.x) m = fmaxf(m, scores[p]);
    red[lane] = m;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (lane < off) red[lane] = fmaxf(red[lane], red[lane + off]);
        __syncthreads();
    }
    const float m_c = red[0];
    __syncthreads();
    float ls = 0.0f;
    for (int p = lane; p < pn; p += blockDim.x) {
        const float w = expf(scores[p] - m_c);
        scores[p] = w;
        ls += w;
    }
    red[lane] = ls;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (lane < off) red[lane] += red[lane + off];
        __syncthreads();
    }
    if (lane == 0) { part_m[h * ATTN_MAX_PARTS + part] = m_c; part_l[h * ATTN_MAX_PARTS + part] = red[0]; }
    __syncthreads();
    if (lane < HEAD_DIM) {
        float vsum = 0.0f;
        for (int p = 0; p < pn; ++p) {
            vsum += scores[p] * __half2float(Vc[((size_t) (p0 + p) * N_HEAD_KV + kh) * HEAD_DIM + lane]);
        }
        part_acc[((size_t) h * ATTN_MAX_PARTS + part) * HEAD_DIM + lane] = vsum;
    }
}

__global__ void attn_merge(const float * __restrict__ part_m, const float * __restrict__ part_l,
                           const float * __restrict__ part_acc, float * __restrict__ out, int n_parts) {
    const int h = blockIdx.x, lane = threadIdx.x; // lane over HEAD_DIM
    float m = -1e30f;
    for (int c = 0; c < n_parts; ++c) m = fmaxf(m, part_m[h * ATTN_MAX_PARTS + c]);
    float l = 0.0f, acc = 0.0f;
    for (int c = 0; c < n_parts; ++c) {
        const float w = expf(part_m[h * ATTN_MAX_PARTS + c] - m);
        l   += w * part_l[h * ATTN_MAX_PARTS + c];
        acc += w * part_acc[((size_t) h * ATTN_MAX_PARTS + c) * HEAD_DIM + lane];
    }
    out[h * HEAD_DIM + lane] = acc / l;
}

// GQA-aware split-K attention: one workgroup per (kv-head, chunk) computes ALL
// q-heads sharing that kv-head (6 for 24/4) against a single pass over K/V —
// cuts deep-context K/V traffic 6x vs one-workgroup-per-q-head.
#define GQA_RATIO (N_HEAD / N_HEAD_KV)
#define ATTN_SPLIT2 1024
__global__ void attn_1q_gqa(const float * __restrict__ q, const __half * __restrict__ Kc,
                            const __half * __restrict__ Vc, int n_pos,
                            float * __restrict__ part_m, float * __restrict__ part_l,
                            float * __restrict__ part_acc) {
    const int kh = blockIdx.x, part = blockIdx.y;
    const int lane = threadIdx.x; // 256
    const int p0 = part * ATTN_SPLIT2;
    __shared__ float scores[GQA_RATIO][ATTN_SPLIT2];
    __shared__ float qs[GQA_RATIO][HEAD_DIM];
    __shared__ float red[256];
    const float kq_scale = rsqrtf((float) HEAD_DIM);
    const int pn = min(ATTN_SPLIT2, n_pos - p0);

    // stage the 6 q vectors in LDS
    for (int g = 0; g < GQA_RATIO; ++g) {
        if (lane < HEAD_DIM) qs[g][lane] = q[((size_t) (kh * GQA_RATIO + g)) * HEAD_DIM + lane];
    }
    __syncthreads();

    if (pn > 0) {
        // one pass over K: each thread owns positions, dots against all 6 heads
        for (int p = lane; p < pn; p += blockDim.x) {
            const __half * kp = Kc + ((size_t) (p0 + p) * N_HEAD_KV + kh) * HEAD_DIM;
            float s[GQA_RATIO] = {};
            #pragma unroll 4
            for (int i = 0; i < HEAD_DIM; ++i) {
                const float kv = __half2float(kp[i]);
                #pragma unroll
                for (int g = 0; g < GQA_RATIO; ++g) s[g] += qs[g][i] * kv;
            }
            #pragma unroll
            for (int g = 0; g < GQA_RATIO; ++g) scores[g][p] = s[g] * kq_scale;
        }
    }
    __syncthreads();

    for (int g = 0; g < GQA_RATIO; ++g) {
        const int h = kh * GQA_RATIO + g;
        if (pn <= 0) {
            if (lane == 0) { part_m[h * ATTN_MAX_PARTS + part] = -1e30f; part_l[h * ATTN_MAX_PARTS + part] = 0.0f; }
            if (lane < HEAD_DIM) part_acc[((size_t) h * ATTN_MAX_PARTS + part) * HEAD_DIM + lane] = 0.0f;
            continue;
        }
        float m = -1e30f;
        for (int p = lane; p < pn; p += blockDim.x) m = fmaxf(m, scores[g][p]);
        red[lane] = m;
        __syncthreads();
        for (int off = blockDim.x / 2; off > 0; off >>= 1) {
            if (lane < off) red[lane] = fmaxf(red[lane], red[lane + off]);
            __syncthreads();
        }
        const float m_c = red[0];
        __syncthreads();
        float ls = 0.0f;
        for (int p = lane; p < pn; p += blockDim.x) {
            const float w = expf(scores[g][p] - m_c);
            scores[g][p] = w;
            ls += w;
        }
        red[lane] = ls;
        __syncthreads();
        for (int off = blockDim.x / 2; off > 0; off >>= 1) {
            if (lane < off) red[lane] += red[lane + off];
            __syncthreads();
        }
        if (lane == 0) { part_m[h * ATTN_MAX_PARTS + part] = m_c; part_l[h * ATTN_MAX_PARTS + part] = red[0]; }
        __syncthreads();
    }

    // one pass over V: thread owns dim `lane`, accumulates all 6 heads
    if (pn > 0 && lane < HEAD_DIM) {
        float acc[GQA_RATIO] = {};
        for (int p = 0; p < pn; ++p) {
            const float vv = __half2float(Vc[((size_t) (p0 + p) * N_HEAD_KV + kh) * HEAD_DIM + lane]);
            #pragma unroll
            for (int g = 0; g < GQA_RATIO; ++g) acc[g] += scores[g][p] * vv;
        }
        #pragma unroll
        for (int g = 0; g < GQA_RATIO; ++g) {
            part_acc[((size_t) (kh * GQA_RATIO + g) * ATTN_MAX_PARTS + part) * HEAD_DIM + lane] = acc[g];
        }
    }
}
