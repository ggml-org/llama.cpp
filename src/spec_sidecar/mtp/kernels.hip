// SPDX-License-Identifier: MIT
// HIP draft sidecar prototype: the Qwen3.8-27B nextn (MTP) block as HIP kernels,
// replaying the golden event log for token-exact parity with the Vulkan drafter.
//
// Model facts (from drafter_manifest.json):
//   n_embd 5120, n_head 24, n_head_kv 4, head_dim 256, n_rot 64 (mrope sections 11,11,10,0
//   -> equal text positions reduce to NEOX rope on the first 64 dims), ffn 17408,
//   sliced head 40960 rows, all mats Q4_0, norms f32, rms_eps 1e-6, freq_base 1e7.
//   attn_q output is q/gate interleaved per head: [q 256 | gate 256] x 24.
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

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { printf("HIP fail %s at line %d\n", hipGetErrorString(e_), __LINE__); exit(1); } } while (0)

// ---------------- model dims ----------------
static const int N_EMBD    = 5120;
static const int N_HEAD    = 24;
static const int N_HEAD_KV = 4;
static const int HEAD_DIM  = 256;
static const int N_ROT     = 64;
static const int N_FF      = 17408;
static const int N_HEAD_ROWS = 40960;   // sliced draft head
static constexpr int MTP_DRAFT_TOP_K = 32;
static const float RMS_EPS   = 1e-6f;
static const float FREQ_BASE = 10000000.0f;
static const int MAX_POS   = 131072;    // matches production context ceiling (f16 KV: 512MB)

// ---------------- q4_0 ----------------
struct block_q4_0 { uint16_t d; uint8_t qs[16]; }; // f16 scale + 32 4-bit quants
static_assert(sizeof(block_q4_0) == 18, "q4_0 block size");

__device__ __forceinline__ float f16_to_f32(uint16_t h) {
    __half hv;
    memcpy(&hv, &h, sizeof(hv));
    return __half2float(hv);
}

// dequant elem j (0..31) of a block: low nibble = j<16, high nibble = j-16
__device__ __forceinline__ float q4_0_get(const block_q4_0 * b, int j) {
    const uint8_t byte = b->qs[j & 15];
    const int q = (j < 16) ? (byte & 0x0F) : (byte >> 4);
    return (q - 8) * f16_to_f32(b->d);
}

// ---------------- kernels ----------------

// y[row] = sum_k W[row,k] * x[k]; W is q4_0, row-major rows of n_in elems.
// one wave32 (gfx1100) per row; each lane accumulates whole blocks.
__global__ void gemv_q4_0(const block_q4_0 * __restrict__ W, const float * __restrict__ x,
                          float * __restrict__ y, int n_in, int n_out) {
    const int row  = blockIdx.x;
    const int lane = threadIdx.x; // 0..31
    if (row >= n_out) return;
    const int nblk = n_in / 32;
    const block_q4_0 * wr = W + (size_t) row * nblk;
    float acc = 0.0f;
    for (int b = lane; b < nblk; b += 32) {
        const block_q4_0 * blk = &wr[b];
        const float d = f16_to_f32(blk->d);
        const float * xb = x + b * 32;
        float s = 0.0f;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const uint8_t byte = blk->qs[j];
            s += (float) ((byte & 0x0F) - 8) * xb[j];
            s += (float) ((byte >> 4)   - 8) * xb[j + 16];
        }
        acc += d * s;
    }
    // wave32 reduction
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down(acc, off, 32);
    }
    if (lane == 0) y[row] = acc;
}

// dequant one q4_0 row (token embedding lookup); row index read from device memory
// so the draft chain never returns to the host between steps
__global__ void deq_row_q4_0(const block_q4_0 * __restrict__ W, const int * __restrict__ row_ptr,
                             float * __restrict__ y, int n_in, int n_rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_in) return;
    const int row = *row_ptr;
    if (row < 0 || row >= n_rows) {
        y[i] = 0.0f;
        return;
    }
    const int nblk = n_in / 32;
    const block_q4_0 * b = W + (size_t) row * nblk + (i >> 5);
    y[i] = q4_0_get(b, i & 31);
}

// rms-norm: y = x / rms(x) * w   (one block, x length n; supports n up to 17408)
__global__ void rmsnorm(const float * __restrict__ x, const float * __restrict__ w,
                        float * __restrict__ y, int n) {
    __shared__ float ssum[256];
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i] * x[i];
    ssum[threadIdx.x] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if ((int) threadIdx.x < off) ssum[threadIdx.x] += ssum[threadIdx.x + off];
        __syncthreads();
    }
    const float scale = rsqrtf(ssum[0] / n + RMS_EPS);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * scale * w[i];
}

// per-head rms-norm over HEAD_DIM elems, nh heads: x[nh*HEAD_DIM] in place-to-y, w[HEAD_DIM]
__global__ void rmsnorm_heads(const float * __restrict__ x, const float * __restrict__ w,
                              float * __restrict__ y, int nh) {
    __shared__ float ssum[256];
    const int h = blockIdx.x;
    if (h >= nh) return;
    const float * xh = x + h * HEAD_DIM;
    float s = 0.0f;
    for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) s += xh[i] * xh[i];
    ssum[threadIdx.x] = s;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if ((int) threadIdx.x < off) ssum[threadIdx.x] += ssum[threadIdx.x + off];
        __syncthreads();
    }
    const float scale = rsqrtf(ssum[0] / HEAD_DIM + RMS_EPS);
    for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) y[h * HEAD_DIM + i] = xh[i] * scale * w[i];
}

// NEOX rope on the first N_ROT dims of each head (pairs (i, i+N_ROT/2)), rest pass-through.
// mrope with equal text positions == neox rope with theta = pos * base^(-2i/n_rot).
__global__ void rope_neox(float * __restrict__ x, int nh, int pos) {
    const int h = blockIdx.x;
    const int i = threadIdx.x; // 0..N_ROT/2-1
    if (h >= nh || i >= N_ROT / 2) return;
    float * xh = x + h * HEAD_DIM;
    const float theta = pos * powf(FREQ_BASE, -2.0f * i / N_ROT);
    const float c = cosf(theta), s = sinf(theta);
    const float x0 = xh[i], x1 = xh[i + N_ROT / 2];
    xh[i]             = x0 * c - x1 * s;
    xh[i + N_ROT / 2] = x0 * s + x1 * c;
}

// single-query attention, one block per q-head, GQA 24/4, KV cache by position.
// Positions-parallel: per chunk, threads compute scores in parallel (shared),
// block-reduce max/sum once, then dim-parallel weighted-V accumulation.
// Online-softmax merge across chunks. out[nh*HEAD_DIM].
#define ATTN_CHUNK 2048
__global__ void attn_1q(const float * __restrict__ q, const __half * __restrict__ Kc,
                        const __half * __restrict__ Vc, float * __restrict__ out, int n_pos) {
    const int h  = blockIdx.x;           // q head
    const int kh = h / (N_HEAD / N_HEAD_KV);
    const int lane = threadIdx.x;        // 256 threads
    __shared__ float scores[ATTN_CHUNK];
    __shared__ float red[256];

    const float * qh = q + h * HEAD_DIM;
    const float kq_scale = rsqrtf((float) HEAD_DIM);

    // per-thread copy of q dim `lane` for the V phase; whole q vector is read
    // repeatedly in the score phase (L2/L0-cached, 1KB)
    float acc  = 0.0f;   // accumulator for out dim `lane`
    float m_s  = -1e30f;
    float l_s  = 0.0f;

    for (int c0 = 0; c0 < n_pos; c0 += ATTN_CHUNK) {
        const int cn = min(ATTN_CHUNK, n_pos - c0);

        // phase 1: scores for this chunk, positions parallel across threads
        for (int p = lane; p < cn; p += blockDim.x) {
            const __half * kp = Kc + ((size_t) (c0 + p) * N_HEAD_KV + kh) * HEAD_DIM;
            float s = 0.0f;
            #pragma unroll 8
            for (int i = 0; i < HEAD_DIM; ++i) s += qh[i] * __half2float(kp[i]);
            scores[p] = s * kq_scale;
        }
        __syncthreads();

        // phase 2: chunk max
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

        // phase 3: exp in place + chunk sum
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

        // phase 4: online merge + weighted V, dim-parallel (thread = dim `lane`)
        const float m_new = fmaxf(m_s, m_chunk);
        const float scale_old = expf(m_s - m_new);
        const float scale_new = expf(m_chunk - m_new);
        acc *= scale_old;
        if (lane < HEAD_DIM) {
            float vsum = 0.0f;
            for (int p = 0; p < cn; ++p) {
                vsum += scores[p] * __half2float(Vc[((size_t) (c0 + p) * N_HEAD_KV + kh) * HEAD_DIM + lane]);
            }
            acc += scale_new * vsum;
        }
        l_s = l_s * scale_old + l_chunk * scale_new;
        m_s = m_new;
        __syncthreads();
    }
    if (lane < HEAD_DIM) out[h * HEAD_DIM + lane] = acc / l_s;
}

// misc elementwise
__global__ void concat2(const float * a, const float * b, float * y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i];
    else if (i < 2 * n) y[i] = b[i - n];
}
__global__ void add_inplace(float * y, const float * a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a[i];
}
__global__ void sigmoid_gate(float * attn, const float * gate, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) attn[i] *= 1.0f / (1.0f + expf(-gate[i]));
}
__global__ void silu_mul(float * gate, const float * up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float g = gate[i]; gate[i] = (g / (1.0f + expf(-g))) * up[i]; }
}
// deinterleave attn_q output: qfull[24*(256+256)] -> q[24*256], gate[24*256]
__global__ void split_q_gate(const float * qfull, float * q, float * gate) {
    int h = blockIdx.x, i = threadIdx.x;
    q[h * HEAD_DIM + i]    = qfull[h * 2 * HEAD_DIM + i];
    gate[h * HEAD_DIM + i] = qfull[h * 2 * HEAD_DIM + HEAD_DIM + i];
}
__global__ void kv_store(const float * k, const float * v, __half * Kc, __half * Vc, int pos) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = N_HEAD_KV * HEAD_DIM;
    if (i < n) {
        Kc[(size_t) pos * n + i] = __float2half(k[i]);
        Vc[(size_t) pos * n + i] = __float2half(v[i]);
    }
}
// argmax over n rows -> idx_out[0]; two-stage
__global__ void argmax_partial(const float * x, int n, float * pmax, int * pidx) {
    __shared__ float sm[256]; __shared__ int si[256];
    int tid = threadIdx.x, base = blockIdx.x * blockDim.x * 8;
    float m = -1e30f; int mi = -1;
    for (int k = 0; k < 8; ++k) {
        int i = base + k * blockDim.x + tid;
        const float value = i < n ? x[i] : 0.0f;
        if (i < n && value == value && (mi < 0 || value > m)) { m = value; mi = i; }
    }
    sm[tid] = m; si[tid] = mi;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off && si[tid + off] >= 0 && (si[tid] < 0 || sm[tid + off] > sm[tid])) {
            sm[tid] = sm[tid + off]; si[tid] = si[tid + off];
        }
        __syncthreads();
    }
    if (tid == 0) { pmax[blockIdx.x] = sm[0]; pidx[blockIdx.x] = si[0]; }
}
__global__ void argmax_final(const float * pmax, const int * pidx, int nparts, int n,
                             const int * remap, int vocab, int * out) {
    float m = -1e30f; int mi = -1;
    for (int i = 0; i < nparts; ++i) {
        const int candidate = pidx[i];
        const float value = pmax[i];
        if (candidate >= 0 && candidate < n && value == value && (mi < 0 || value > m)) {
            m = value; mi = candidate;
        }
    }
    if (mi < 0) {
        out[0] = -1;
        return;
    }
    const int token = remap ? remap[mi] : mi;
    out[0] = token >= 0 && token < vocab ? token : -1;
}

// Select the top MTP_DRAFT_TOP_K logits without returning the
// sliced head to the host. Each partition keeps its local top-k; the final
// one-thread merge is exact because every global top-k item is present in its
// partition's local top-k.
__global__ void topk_partial(const float * __restrict__ x, int n,
                             float * __restrict__ pvals, int * __restrict__ pidx) {
    __shared__ float sm[256];
    __shared__ int si[256];
    __shared__ int selected[MTP_DRAFT_TOP_K];
    const int tid = threadIdx.x;
    const int base = blockIdx.x * blockDim.x * 8;

    if (tid < MTP_DRAFT_TOP_K) {
        selected[tid] = -1;
    }
    __syncthreads();

    for (int slot = 0; slot < MTP_DRAFT_TOP_K; ++slot) {
        float best = -1e30f;
        int best_i = -1;
        for (int k = 0; k < 8; ++k) {
            const int i = base + k * blockDim.x + tid;
            if (i >= n) {
                continue;
            }
            bool already_selected = false;
            for (int j = 0; j < slot; ++j) {
                already_selected = already_selected || selected[j] == i;
            }
            const float value = x[i];
            if (!already_selected && value == value &&
                    (best_i < 0 || value > best || (value == best && i < best_i))) {
                best = value;
                best_i = i;
            }
        }
        sm[tid] = best;
        si[tid] = best_i;
        __syncthreads();
        for (int off = blockDim.x / 2; off > 0; off >>= 1) {
            if (tid < off && si[tid + off] >= 0 &&
                    (si[tid] < 0 || sm[tid + off] > sm[tid] ||
                     (sm[tid + off] == sm[tid] && si[tid + off] < si[tid]))) {
                sm[tid] = sm[tid + off];
                si[tid] = si[tid + off];
            }
            __syncthreads();
        }
        if (tid == 0) {
            selected[slot] = si[0];
            pvals[blockIdx.x * MTP_DRAFT_TOP_K + slot] = sm[0];
            pidx[blockIdx.x * MTP_DRAFT_TOP_K + slot] = si[0];
        }
        __syncthreads();
    }
}

__global__ void topk_final(const float * __restrict__ pvals, const int * __restrict__ pidx,
                           int nparts, const int * __restrict__ remap, int vocab,
                           int * __restrict__ out_ids, float * __restrict__ out_vals) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int selected[MTP_DRAFT_TOP_K];
    for (int slot = 0; slot < MTP_DRAFT_TOP_K; ++slot) {
        float best = -1e30f;
        int best_i = -1;
        for (int i = 0; i < nparts * MTP_DRAFT_TOP_K; ++i) {
            const int candidate = pidx[i];
            const float value = pvals[i];
            bool already_selected = false;
            for (int j = 0; j < slot; ++j) {
                already_selected = already_selected || selected[j] == candidate;
            }
            if (!already_selected && candidate >= 0 && value == value &&
                    (best_i < 0 || value > best || (value == best && candidate < best_i))) {
                best = value;
                best_i = candidate;
            }
        }
        selected[slot] = best_i;
        out_vals[slot] = best;
        if (best_i < 0) {
            out_ids[slot] = -1;
        } else {
            const int token = remap ? remap[best_i] : best_i;
            out_ids[slot] = token >= 0 && token < vocab ? token : -1;
        }
    }
}

// Normalize a top-k row and sample it using the caller-provided deterministic
// counter-based uniform. The active flag makes a p_min stop sticky while the
// enclosing sidecar continues issuing the already-pipelined fixed number of
// kernel launches without any host synchronization.
__global__ void sample_topk(const int * __restrict__ ids, const float * __restrict__ logits,
                            int topk, float temperature, float p_min, double uniform,
                            int step, int * __restrict__ output_id,
                            float * __restrict__ output_probs, int * __restrict__ output_count,
                            int * __restrict__ active) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || *active == 0) {
        return;
    }
    float max_score = -1e30f;
    int valid = 0;
    for (int k = 0; k < topk; ++k) {
        if (ids[k] >= 0 && logits[k] == logits[k] && logits[k] > max_score) {
            max_score = logits[k];
        }
        valid += ids[k] >= 0 && logits[k] == logits[k];
    }
    if (valid == 0 || !(temperature > 0.0f) || !(max_score == max_score)) {
        *active = 0;
        return;
    }
    float probs[MTP_DRAFT_TOP_K] = {};
    float sum = 0.0f;
    for (int k = 0; k < topk; ++k) {
        const float p = ids[k] >= 0 && logits[k] == logits[k]
                ? expf((logits[k] - max_score) / temperature) : 0.0f;
        probs[k] = p;
        sum += p;
    }
    if (!(sum > 0.0f) || !(sum == sum)) {
        *active = 0;
        return;
    }
    int selected = topk - 1;
    double cumulative = 0.0;
    const double u = uniform < 1.0 ? (uniform > 0.0 ? uniform : 0.0) : 0.9999999999999999;
    for (int k = 0; k < topk; ++k) {
        probs[k] /= sum;
        cumulative += probs[k];
        if (u < cumulative) {
            selected = k;
            break;
        }
    }
    if (probs[selected] < p_min || ids[selected] < 0) {
        *active = 0;
        return;
    }
    output_id[step] = ids[selected];
    for (int k = 0; k < topk; ++k) {
        output_probs[step * topk + k] = probs[k];
    }
    *output_count = step + 1;
}
