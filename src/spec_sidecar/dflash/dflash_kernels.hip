// SPDX-License-Identifier: MIT
// HIP kernels for the DFlash2 block drafter (Qwen3.8-27B controller).
// Dims: n_embd 5120, 5 layers, 32 q-heads / 8 kv-heads x 128, non-causal SWA-2048,
// block 8 (anchor + 7 masks), conv kernel 2 group 16, selector rank 256 top-16.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>

#define DF_EMBD    5120
#define DF_LAYERS  5
#define DF_HEADS   32
#define DF_KV      8
#define DF_HDIM    128
#define DF_FF      17408
#define DF_BLOCK   8
#define DF_GROUPS  320   // 5120/16
#define DF_CONVDYN 1280  // 320*2*2
#define DF_WINDOW  2048
#define DF_VOCAB   248320
#define DF_RANK    256
#define DF_TOPK    16
#define DF_EPS     1e-6f
#define DF_FBASE   10000000.0f

// ---------------- quant blocks ----------------
struct block_q4_0d { uint16_t d; uint8_t qs[16]; };
struct block_q4_K { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; }; // 256 elems
struct block_q6_K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; }; // 256 elems
static_assert(sizeof(block_q4_K) == 144, "q4_K");
static_assert(sizeof(block_q6_K) == 210, "q6_K");

__device__ __host__ __forceinline__ float df_f16(uint16_t h) {
#ifdef __HIP_DEVICE_COMPILE__
    __half hv; memcpy(&hv, &h, 2); return __half2float(hv);
#else
    // host-side minimal f16->f32
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v;
    if (e == 0) v = ldexpf((float) m, -24);
    else if (e == 31) v = m ? nanf("") : INFINITY;
    else v = ldexpf((float) (m + 1024), (int) e - 25);
    return s ? -v : v;
#endif
}

__device__ __host__ __forceinline__ void q4k_scale_min(int j, const uint8_t * q, float * sc, float * mn) {
    if (j < 4) { *sc = (float) (q[j] & 63); *mn = (float) (q[j + 4] & 63); }
    else { *sc = (float) ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
           *mn = (float) ((q[j + 4] >> 4) | ((q[j] >> 6) << 4)); }
}

// ---------------- weights-stationary GEMV for the 8-row block ----------------
// y[c][row] = sum_k W[row,k] x[c][k]; each wave reads a weight block ONCE and
// accumulates all NCOLS columns (x vectors stay hot in L2).

template<int NCOLS>
__global__ void gemv8_q4_K(const block_q4_K * __restrict__ W, const float * __restrict__ x,
                           float * __restrict__ y, int n_in, int n_out) {
    const int row = blockIdx.x, lane = threadIdx.x; // 32 lanes
    if (row >= n_out) return;
    const int nblk = n_in / 256;
    const block_q4_K * wr = W + (size_t) row * nblk;
    float acc[NCOLS] = {};
    for (int b = lane; b < nblk; b += 32) {
        const block_q4_K * bk = &wr[b];
        const float d = df_f16(bk->d), dmin = df_f16(bk->dmin);
        const int kbase = b * 256;
        for (int g64 = 0; g64 < 4; ++g64) {           // four 64-elem groups
            float sc0, mn0, sc1, mn1;
            q4k_scale_min(g64 * 2 + 0, bk->scales, &sc0, &mn0);
            q4k_scale_min(g64 * 2 + 1, bk->scales, &sc1, &mn1);
            const uint8_t * qs = bk->qs + g64 * 32;
            const int k0 = kbase + g64 * 64;
            for (int l = 0; l < 32; ++l) {
                const float w0 = d * sc0 * (float) (qs[l] & 0xF) - dmin * mn0;
                const float w1 = d * sc1 * (float) (qs[l] >> 4) - dmin * mn1;
                #pragma unroll
                for (int c = 0; c < NCOLS; ++c) {
                    acc[c] += w0 * x[(size_t) c * n_in + k0 + l] + w1 * x[(size_t) c * n_in + k0 + 32 + l];
                }
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < NCOLS; ++c) {
        float a = acc[c];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down(a, off, 32);
        if (lane == 0) y[(size_t) c * n_out + row] = a;
    }
}

template<int NCOLS>
__global__ void gemv8_q6_K(const block_q6_K * __restrict__ W, const float * __restrict__ x,
                           float * __restrict__ y, int n_in, int n_out) {
    const int row = blockIdx.x, lane = threadIdx.x;
    if (row >= n_out) return;
    const int nblk = n_in / 256;
    const block_q6_K * wr = W + (size_t) row * nblk;
    float acc[NCOLS] = {};
    for (int b = lane; b < nblk; b += 32) {
        const block_q6_K * bk = &wr[b];
        const float d = df_f16(bk->d);
        const int kbase = b * 256;
        for (int half = 0; half < 2; ++half) {
            const uint8_t * ql = bk->ql + half * 64;
            const uint8_t * qh = bk->qh + half * 32;
            const int8_t  * sc = bk->scales + half * 8;
            const int k0 = kbase + half * 128;
            for (int l = 0; l < 32; ++l) {
                const int q0 = (int) ((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q1 = (int) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q2 = (int) ((ql[l]      >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q3 = (int) ((ql[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                const float w0 = d * sc[(l)      / 16] * q0;
                const float w1 = d * sc[(l + 32) / 16] * q1;
                const float w2 = d * sc[(l + 64) / 16] * q2;
                const float w3 = d * sc[(l + 96) / 16] * q3;
                #pragma unroll
                for (int c = 0; c < NCOLS; ++c) {
                    const float * xc = x + (size_t) c * n_in + k0;
                    acc[c] += w0 * xc[l] + w1 * xc[l + 32] + w2 * xc[l + 64] + w3 * xc[l + 96];
                }
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < NCOLS; ++c) {
        float a = acc[c];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down(a, off, 32);
        if (lane == 0) y[(size_t) c * n_out + row] = a;
    }
}

// token embedding rows from the target's Q4_0 token_embd (pinned host, mapped)
__global__ void df_embed_rows(const block_q4_0d * __restrict__ W, const int * __restrict__ toks,
                              float * __restrict__ y, int n_in) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, col = blockIdx.y;
    if (i >= n_in) return;
    const block_q4_0d * b = W + (size_t) toks[col] * (n_in / 32) + (i >> 5);
    const int j = i & 31;
    const uint8_t byte = b->qs[j & 15];
    const int q = (j < 16) ? (byte & 0x0F) : (byte >> 4);
    y[(size_t) col * n_in + i] = (q - 8) * df_f16(b->d);
}

// ---------------- norms / rope / elementwise ----------------
__global__ void df_rmsnorm(const float * __restrict__ x, const float * __restrict__ w,
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
    const float scale = rsqrtf(ssum[0] / n + DF_EPS);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yc[i] = xc[i] * scale * w[i];
}

// per-head rmsnorm over 128 dims: grid (nheads, ncols), 128 threads
__global__ void df_rmsnorm_head(float * __restrict__ x, const float * __restrict__ w, int nh) {
    __shared__ float ssum[128];
    float * xh = x + ((size_t) blockIdx.y * nh + blockIdx.x) * DF_HDIM;
    const int i = threadIdx.x;
    ssum[i] = xh[i] * xh[i];
    __syncthreads();
    for (int off = 64; off > 0; off >>= 1) {
        if (i < off) ssum[i] += ssum[i + off];
        __syncthreads();
    }
    const float scale = rsqrtf(ssum[0] / DF_HDIM + DF_EPS);
    xh[i] = xh[i] * scale * w[i];
}

// neox rope over n_rot dims of each 128-dim head (degenerate mrope, equal pos)
__global__ void df_rope(float * __restrict__ x, int nh, const int * __restrict__ poss, int n_rot) {
    const int h = blockIdx.x, col = blockIdx.y, i = threadIdx.x;
    if (i >= n_rot / 2) return;
    float * xh = x + ((size_t) col * nh + h) * DF_HDIM;
    const float theta = poss[col] * powf(DF_FBASE, -2.0f * i / n_rot);
    const float c = cosf(theta), s = sinf(theta);
    const float x0 = xh[i], x1 = xh[i + n_rot / 2];
    xh[i]             = x0 * c - x1 * s;
    xh[i + n_rot / 2] = x0 * s + x1 * c;
}

// conv gate: out[d,t] = (dyn[g(d)+320*0+640*side, t] + base[d + 5120*0 + 10240*side]) * h[d,t]
//                     + (dyn[g(d)+320*1+640*side, t] + base[d + 5120*1 + 10240*side]) * h_prev(d,t)
// h_prev = h[d,t-1] within the 8-block, zero at t==0. One kernel per gate call.
__global__ void df_conv_gate(const float * __restrict__ h, const float * __restrict__ dyn,
                             const float * __restrict__ base, float * __restrict__ out,
                             int side, int ncols) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (d >= DF_EMBD || t >= ncols) return;
    const int g = d / 16;
    const float w0 = dyn[(size_t) t * DF_CONVDYN + g + 640 * side]           + base[d + 10240 * side];
    const float w1 = dyn[(size_t) t * DF_CONVDYN + g + 320 + 640 * side]     + base[d + 5120 + 10240 * side];
    const float hp = (t == 0) ? 0.0f : h[(size_t) (t - 1) * DF_EMBD + d];
    out[(size_t) t * DF_EMBD + d] = w0 * h[(size_t) t * DF_EMBD + d] + w1 * hp;
}

__global__ void df_add(float * y, const float * a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a[i];
}
__global__ void df_silu_mul(float * gate, const float * up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float g = gate[i]; gate[i] = (g / (1.0f + expf(-g))) * up[i]; }
}

// KV store: k/v [ncols][8kv*128] -> cache[layer][pos][kv][128] (f32)
__global__ void df_kv_store(const float * __restrict__ k, const float * __restrict__ v,
                            __half * __restrict__ Kc, __half * __restrict__ Vc,
                            const int * __restrict__ poss, int ncols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, col = blockIdx.y;
    const int n = DF_KV * DF_HDIM;
    if (i < n && col < ncols) {
        const int pos = poss[col];
        Kc[(size_t) pos * n + i] = __float2half(k[(size_t) col * n + i]);
        Vc[(size_t) pos * n + i] = __float2half(v[(size_t) col * n + i]);
    }
}

__global__ void df_rope_base(float * __restrict__ x, int nh, int pos_base, int n_rot) {
    const int h = blockIdx.x, col = blockIdx.y, i = threadIdx.x;
    if (i >= n_rot / 2) return;
    float * xh = x + ((size_t) col * nh + h) * DF_HDIM;
    const float theta = (pos_base + col) * powf(DF_FBASE, -2.0f * i / n_rot);
    const float c = cosf(theta), s = sinf(theta);
    const float x0 = xh[i], x1 = xh[i + n_rot / 2];
    xh[i]             = x0 * c - x1 * s;
    xh[i + n_rot / 2] = x0 * s + x1 * c;
}

__global__ void df_kv_store_base(const float * __restrict__ k, const float * __restrict__ v,
                                 __half * __restrict__ Kc, __half * __restrict__ Vc,
                                 int pos_base, int pending_base, int ncols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, col = blockIdx.y;
    const int n = DF_KV * DF_HDIM;
    if (i < n && col < ncols) {
        const int pos = pos_base + col - pending_base;
        Kc[(size_t) pos * n + i] = __float2half(k[(size_t) col * n + i]);
        Vc[(size_t) pos * n + i] = __float2half(v[(size_t) col * n + i]);
    }
}

// Interleave the five target layer-input device rows into the DFlash encoder
// layout. Pointers are kernel arguments, not a mutable shared table, so batched
// multi-sequence calls cannot race while earlier rows are still in flight.
__global__ void df_pack_layers5(const float * l0, const float * l1, const float * l2,
                                const float * l3, const float * l4, float * out,
                                int layer_width, int ncols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x, col = blockIdx.y;
    if (col >= ncols || i >= layer_width) return;
    const size_t dst = (size_t) col * 5 * layer_width;
    out[dst + i] = l0[(size_t) col * layer_width + i];
    out[dst + layer_width + i] = l1[(size_t) col * layer_width + i];
    out[dst + 2 * layer_width + i] = l2[(size_t) col * layer_width + i];
    out[dst + 3 * layer_width + i] = l3[(size_t) col * layer_width + i];
    out[dst + 4 * layer_width + i] = l4[(size_t) col * layer_width + i];
}

__global__ void df_sanitize(float * values, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = values[i];
    values[i] = v != v ? 0.0f : (isinf(v) ? (v > 0.0f ? 65504.0f : -65504.0f) : v);
}

// non-causal SWA attention for the 8-row block. Cache holds injected positions
// 0..n_past-1 (valid) plus the block rows are provided separately (kb/vb) so all
// 8 rows attend to: cache positions in (q_pos-2047 .. n_past-1) AND all 8 block
// rows (non-causal within block). grid (32 q-heads, 8 rows); 256 threads.
__global__ void df_attn(const float * __restrict__ q, const __half * __restrict__ Kc,
                        const __half * __restrict__ Vc, const float * __restrict__ kb,
                        const float * __restrict__ vb, float * __restrict__ out,
                        int n_past_unused, const int * __restrict__ q_poss) {
    (void) n_past_unused;
    const int n_past = q_poss[0]; // block row 0 sits at n_past (graph-capture safe)
    const int h = blockIdx.x, row = blockIdx.y, lane = threadIdx.x;
    const int kh = h / (DF_HEADS / DF_KV);
    __shared__ float red[256];
    __shared__ float sc_blk[DF_BLOCK];
    const float * qh = q + ((size_t) row * DF_HEADS + h) * DF_HDIM;
    const float kq_scale = rsqrtf((float) DF_HDIM);
    const int qpos = q_poss[row];
    const int lo = max(0, qpos - (DF_WINDOW - 1));

    // pass 1: cache scores -> streaming max/sum with block rows
    float m = -1e30f, l = 0.0f, accd = 0.0f; // acc for dim `lane` if lane<128
    // block-row scores first (small, shared)
    if (lane < DF_BLOCK) {
        const float * kp = kb + ((size_t) lane * DF_KV + kh) * DF_HDIM;
        float s = 0.0f;
        for (int i = 0; i < DF_HDIM; ++i) s += qh[i] * kp[i];
        sc_blk[lane] = s * kq_scale;
    }
    __syncthreads();
    for (int i2 = 0; i2 < DF_BLOCK; ++i2) m = fmaxf(m, sc_blk[i2]);

    // chunked cache pass
    const int CH = 1024;
    __shared__ float scores[1024];
    for (int c0 = lo; c0 < n_past; c0 += CH) {
        const int cn = min(CH, n_past - c0);
        for (int p = lane; p < cn; p += blockDim.x) {
            const __half * kp = Kc + ((size_t) (c0 + p) * DF_KV + kh) * DF_HDIM;
            float s = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < DF_HDIM; ++i) s += qh[i] * __half2float(kp[i]);
            scores[p] = s * kq_scale;
        }
        __syncthreads();
        float mc = -1e30f;
        for (int p = lane; p < cn; p += blockDim.x) mc = fmaxf(mc, scores[p]);
        red[lane] = mc;
        __syncthreads();
        for (int off = 128; off > 0; off >>= 1) {
            if (lane < off) red[lane] = fmaxf(red[lane], red[lane + off]);
            __syncthreads();
        }
        mc = red[0];
        __syncthreads();
        const float m_new = fmaxf(m, mc);
        float ls = 0.0f;
        for (int p = lane; p < cn; p += blockDim.x) {
            const float w = expf(scores[p] - m_new);
            scores[p] = w;
            ls += w;
        }
        red[lane] = ls;
        __syncthreads();
        for (int off = 128; off > 0; off >>= 1) {
            if (lane < off) red[lane] += red[lane + off];
            __syncthreads();
        }
        const float l_chunk = red[0];
        __syncthreads();
        const float scale_old = expf(m - m_new);
        if (lane < DF_HDIM) {
            float vs = 0.0f;
            for (int p = 0; p < cn; ++p) vs += scores[p] * __half2float(Vc[((size_t) (c0 + p) * DF_KV + kh) * DF_HDIM + lane]);
            accd = accd * scale_old + vs;
        }
        l = l * scale_old + l_chunk;
        m = m_new;
        __syncthreads();
    }

    // block rows fold-in
    {
        float lb = 0.0f;
        __shared__ float wblk[DF_BLOCK];
        const float m_new = m; // block max already folded into m at start
        if (lane == 0) {
            for (int i2 = 0; i2 < DF_BLOCK; ++i2) wblk[i2] = expf(sc_blk[i2] - m_new);
        }
        __syncthreads();
        for (int i2 = 0; i2 < DF_BLOCK; ++i2) lb += wblk[i2];
        if (lane < DF_HDIM) {
            float vs = 0.0f;
            for (int i2 = 0; i2 < DF_BLOCK; ++i2) vs += wblk[i2] * vb[((size_t) i2 * DF_KV + kh) * DF_HDIM + lane];
            accd += vs;
        }
        l += lb;
    }
    if (lane < DF_HDIM) out[((size_t) row * DF_HEADS + h) * DF_HDIM + lane] = accd / l;
}

// ---- GPU top-16: 16 rounds of (partial argmax -> final+mask) per row ----
// partials: grid (nparts, nrows); each block covers 2048 elems of its row.
__global__ void dfa_argmax_partial(const float * __restrict__ x, int n,
                                   float * __restrict__ pmax, int * __restrict__ pidx, int nparts) {
    __shared__ float sm[256]; __shared__ int si[256];
    const int tid = threadIdx.x, part = blockIdx.x, row = blockIdx.y;
    const float * xr = x + (size_t) row * n;
    const int base = part * 2048;
    float m = -1e30f; int mi = -1;
    for (int k = 0; k < 8; ++k) {
        const int i = base + k * 256 + tid;
        const float value = i < n ? xr[i] : 0.0f;
        if (i < n && value == value && (mi < 0 || value > m)) { m = value; mi = i; }
    }
    sm[tid] = m; si[tid] = mi;
    __syncthreads();
    for (int off = 128; off > 0; off >>= 1) {
        if (tid < off && si[tid + off] >= 0 && (si[tid] < 0 || sm[tid + off] > sm[tid])) {
            sm[tid] = sm[tid + off]; si[tid] = si[tid + off];
        }
        __syncthreads();
    }
    if (tid == 0) { pmax[(size_t) row * nparts + part] = sm[0]; pidx[(size_t) row * nparts + part] = si[0]; }
}

// final: pick global max per row, record into slot s, mask it out of logits
__global__ void dfa_argmax_final_mask(float * __restrict__ x, int n,
                                      const float * __restrict__ pmax, const int * __restrict__ pidx,
                                      int nparts, int * __restrict__ out_ids, float * __restrict__ out_vals,
                                      int slot, int topk) {
    const int row = blockIdx.x;
    float m = -1e30f; int mi = -1;
    const float * pm = pmax + (size_t) row * nparts;
    const int * pi = pidx + (size_t) row * nparts;
    for (int i = 0; i < nparts; ++i) {
        const int candidate = pi[i];
        const float value = pm[i];
        if (candidate >= 0 && candidate < n && value == value && (mi < 0 || value > m)) {
            m = value; mi = candidate;
        }
    }
    out_ids[(size_t) row * topk + slot] = mi;
    out_vals[(size_t) row * topk + slot] = m;
    if (mi >= 0) x[(size_t) row * n + mi] = -1e30f;
}

// ---- tiled GEMV8: 8 rows per workgroup (8 waves), x-tile staged in LDS ----
// x-traffic divided by 8 vs the naive weights-stationary form.
#define GT_TILE 512

template<int NCOLS>
__global__ void gemv8t_q4_K(const block_q4_K * __restrict__ W, const float * __restrict__ x,
                            float * __restrict__ y, int n_in, int n_out) {
    __shared__ float xt[NCOLS][GT_TILE];
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + wave;
    float acc[NCOLS] = {};
    for (int t0 = 0; t0 < n_in; t0 += GT_TILE) {
        const int tn = min(GT_TILE, n_in - t0);
        // cooperative x-tile load (256 threads)
        for (int c = 0; c < NCOLS; ++c) {
            for (int i = threadIdx.x; i < tn; i += 256) xt[c][i] = x[(size_t) c * n_in + t0 + i];
        }
        __syncthreads();
        if (row < n_out) {
            const int b0 = t0 / 256, bn = tn / 256;
            const block_q4_K * wr = W + (size_t) row * (n_in / 256);
            for (int b = 0; b < bn; ++b) {
                const block_q4_K * bk = &wr[b0 + b];
                const float d = df_f16(bk->d), dmin = df_f16(bk->dmin);
                // lane l covers elems l*8 .. l*8+7: one 32-run => one scale pair
                const int e0 = lane * 8;
                const int g64 = e0 >> 6, rem0 = e0 & 63;
                float sc, mn;
                q4k_scale_min(g64 * 2 + (rem0 >= 32 ? 1 : 0), bk->scales, &sc, &mn);
                const float ds = d * sc, dm = dmin * mn;
                const uint8_t * qs = bk->qs + g64 * 32 + (rem0 & 31);
                const bool hi = rem0 >= 32;
                const float * xtb = &xt[0][b * 256 + e0];
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const uint8_t byte = qs[e];
                    const float w = ds * (hi ? (byte >> 4) : (byte & 0xF)) - dm;
                    #pragma unroll
                    for (int c = 0; c < NCOLS; ++c) acc[c] += w * xtb[(size_t) c * GT_TILE + e];
                }
            }
        }
        __syncthreads();
    }
    if (row < n_out) {
        #pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
            float a = acc[c];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) a += __shfl_down(a, off, 32);
            if (lane == 0) y[(size_t) c * n_out + row] = a;
        }
    }
}

template<int NCOLS>
__global__ void gemv8t_q6_K(const block_q6_K * __restrict__ W, const float * __restrict__ x,
                            float * __restrict__ y, int n_in, int n_out) {
    __shared__ float xt[NCOLS][GT_TILE];
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + wave;
    float acc[NCOLS] = {};
    for (int t0 = 0; t0 < n_in; t0 += GT_TILE) {
        const int tn = min(GT_TILE, n_in - t0);
        for (int c = 0; c < NCOLS; ++c) {
            for (int i = threadIdx.x; i < tn; i += 256) xt[c][i] = x[(size_t) c * n_in + t0 + i];
        }
        __syncthreads();
        if (row < n_out) {
            const int b0 = t0 / 256, bn = tn / 256;
            const block_q6_K * wr = W + (size_t) row * (n_in / 256);
            for (int b = 0; b < bn; ++b) {
                const block_q6_K * bk = &wr[b0 + b];
                const float d = df_f16(bk->d);
                // lane's 8 elems: one half, one grp, one scale group
                const int e0 = lane * 8;
                const int half = e0 >> 7, r0 = e0 & 127;
                const int grp = r0 >> 5, l0 = r0 & 31;
                const float dsc = d * bk->scales[half * 8 + (r0 >> 4)];
                const uint8_t * qlp = bk->ql + half * 64 + (grp & 1) * 32 + l0;
                const uint8_t * qhp = bk->qh + half * 32 + l0;
                const int sh_h = grp * 2;
                const bool hi = grp >= 2;
                const float * xtb = &xt[0][b * 256 + e0];
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int qv = (hi ? (qlp[e] >> 4) : (qlp[e] & 0xF)) | (((qhp[e] >> sh_h) & 3) << 4);
                    const float w = dsc * (qv - 32);
                    #pragma unroll
                    for (int c = 0; c < NCOLS; ++c) acc[c] += w * xtb[(size_t) c * GT_TILE + e];
                }
            }
        }
        __syncthreads();
    }
    if (row < n_out) {
        #pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
            float a = acc[c];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) a += __shfl_down(a, off, 32);
            if (lane == 0) y[(size_t) c * n_out + row] = a;
        }
    }
}
