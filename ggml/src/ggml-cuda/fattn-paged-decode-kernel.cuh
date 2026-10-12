#pragma once

// Decode attention over contiguous or paged K/V for Q8_0 K / Q4_0 V, head size 256, GQA 6, 1-4 query tokens, RDNA4
// WMMA. Plain HIP (no ggml dependencies); ggml's dispatch lives in fattn-paged-decode.cu.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#if defined(__gfx1200__) || defined(__gfx1201__)
#define PD_RDNA4 1
#else
#define PD_RDNA4 0
#endif

// Split-K flash decoding over 64-token chunks. One block = one KV head x a range of chunks, 8 waves of 32.
// Rows are (query token, GQA head) pairs, packed into one or two 16-row WMMA tiles (nb * 6 <= 32).
//
// Per chunk:
//   1. raw Q8_0 K rows and Q4_0 V rows go to LDS (coalesced 16 B loads; the next chunk is prefetched into registers);
//   2. S = Q K^T with WMMA, K dequantized to f16 while building the B fragments; wave w owns 16 tokens of one m-tile;
//   3. online softmax by row through LDS (P stored as f16, per-row rescale factor);
//   4. O += P V with WMMA, V dequantized while gathering B fragments; wave w owns 32 output dims.
// A second kernel combines the splits.

static constexpr int PD_D      = 256;
static constexpr int PD_GQA    = 6;
static constexpr int PD_CHUNK  = 64;
static constexpr int PD_PAGE   = 256;
static constexpr int PD_NWARPS = 8;
static constexpr int PD_QK     = 32;                       // Q8_0 and Q4_0 block size
static constexpr int PD_Q8_BLK = 34;                       // Q8_0 block bytes
static constexpr int PD_Q4_BLK = 18;                       // Q4_0 block bytes
static constexpr int PD_K_ROW  = PD_D / PD_QK * PD_Q8_BLK; // 272 B
static constexpr int PD_V_ROW  = PD_D / PD_QK * PD_Q4_BLK; // 144 B
static constexpr int PD_K_VEC  = PD_K_ROW / 16;                           // 17 x 16 B
static constexpr int PD_V_VEC  = PD_V_ROW / 16;                           //  9 x 16 B
static constexpr int PD_K_LOADS = (PD_CHUNK * PD_K_VEC + PD_NWARPS * 32 - 1) / (PD_NWARPS * 32); // 5
static constexpr int PD_V_LOADS = (PD_CHUNK * PD_V_VEC + PD_NWARPS * 32 - 1) / (PD_NWARPS * 32); // 3

static_assert(PD_K_ROW % 16 == 0 && PD_V_ROW % 16 == 0, "rows must allow 16 B loads");

// Rows of one KV head. Without a table, token t lives at base + t * nb1 + head * nb2. With one, entry e = table[t / 256]
// is either a pool page (e >= 0: row e * 256 + t % 256 of base) or a ring page (e < 0: row (-1 - e) * 256 + t % 256 of
// ring). Pool and ring share the row layout (nb1, nb2).
struct pd_kv {
    const char *    base;
    const int32_t * table = nullptr;
    const char *    ring  = nullptr;
    int64_t         nb1;
    int64_t         nb2;

    // Address of token t0's row for `head`, where t0 starts a chunk: a chunk never straddles a page, so the page is
    // resolved once per chunk (a uniform load); rows within it are t0's row + r * nb1.
    __device__ __forceinline__ const char * chunk(const int t0, const int head) const {
        const int e = table[t0 / PD_PAGE];
        const char * src = e >= 0 ? base : ring;
        const int64_t page = e >= 0 ? e : -1 - e;
        return src + (page * PD_PAGE + t0 % PD_PAGE) * nb1 + head * nb2;
    }

    // Per-row form for contiguous K/V (the form the nb = 3-4 variant compiled best with at the VGPR limit).
    __device__ __forceinline__ const char * row(const int t, const int head) const {
        return base + (int64_t) t * nb1 + head * nb2;
    }
};

// K/V written by another queue (ggml_flash_attn_ext_set_sync): the copy queue publishes *ready >= tag once the data has
// landed; the combine kernel publishes *consumed = tag once every block has read it. Flags are monotonic and compared
// with >=, so a producer running ahead never aliases a later step's value onto an earlier one. The tag is a graph input
// in device memory, so one captured graph serves every step. Disabled when `ready` is null.
struct pd_sync {
    const int32_t *  tag      = nullptr;
    const uint32_t * ready    = nullptr;
    uint32_t *       consumed = nullptr;

    __host__ __device__ __forceinline__ bool enabled() const { return ready != nullptr; }

    __device__ __forceinline__ void wait() const {
        const uint32_t t = (uint32_t) *tag;
        while (__hip_atomic_load(ready, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < t) {
            __builtin_amdgcn_s_sleep(2);
        }
    }

    __device__ __forceinline__ void release() const {
        __hip_atomic_store(consumed, (uint32_t) *tag, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
};

#if PD_RDNA4
typedef _Float16 pd_half8  __attribute__((ext_vector_type(8)));
typedef float    pd_float8 __attribute__((ext_vector_type(8)));

static __device__ __forceinline__ pd_float8 pd_wmma(const pd_half8 a, const pd_half8 b, const pd_float8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}
#endif

// Packed dequantization: 0x6400 is half(1024); OR-ing a byte b < 1024 into the mantissa gives half(1024 + b).
static __device__ __forceinline__ half2 pd_i8x2(const uint32_t b0, const uint32_t b1) { // two int8 as half2
    const uint32_t u = 0x64006400u | ((b0 ^ 0x80u) & 0xFFu) | (((b1 ^ 0x80u) & 0xFFu) << 16);
    return __hsub2(*(const half2 *) &u, __halves2half2(__float2half(1152.0f), __float2half(1152.0f)));
}
static __device__ __forceinline__ half2 pd_u4x2(const uint32_t n0, const uint32_t n1) { // two nibbles - 8 as half2
    const uint32_t u = 0x64006400u | n0 | (n1 << 16);
    return __hsub2(*(const half2 *) &u, __halves2half2(__float2half(1032.0f), __float2half(1032.0f)));
}

// PAGED selects table addressing at compile time, so contiguous K/V carries no table logic.
template <int M_TILES, bool PAGED>
#ifndef PD_MIN_WAVES_PER_SIMD
#define PD_MIN_WAVES_PER_SIMD 6 // keeps every variant at <= 240 VGPRs: 3 blocks of 8 waves per WGP (P-3: the ring variant
                                 // otherwise reached 256 VGPRs, occupancy 5, and lost about 10%)
#endif
// HIP reads the second argument as minimum waves per SIMD (EU), which caps VGPRs at 1536 / waves.
__launch_bounds__(PD_NWARPS * 32, PD_MIN_WAVES_PER_SIMD)
static __global__ void pd_partial(
        const char * __restrict__ Q, const int64_t q_nb1, const int64_t q_nb2,
        const pd_kv K, const pd_kv V, const pd_sync sync,
        const char * __restrict__ mask, const int64_t mask_nb1,
        const float scale, const int n_kv, const int nb, const int n_chunks, const int chunks_per_split,
        float * __restrict__ O_part, float2 * __restrict__ ML_part, const int ablate) {
#if PD_RDNA4
    constexpr int R   = M_TILES * 16;          // rows
    constexpr int TPR = PD_NWARPS * 32 / R;    // softmax threads per row
    constexpr int TOK = PD_CHUNK / TPR;        // tokens per softmax thread

    __shared__ __align__(16) char      Kr[PD_CHUNK * PD_K_ROW];
    __shared__ __align__(16) char      Vr[PD_CHUNK * PD_V_ROW];
    __shared__ __align__(16) float     Ss[R * PD_CHUNK];
    __shared__ __align__(16) _Float16  Ps[R * PD_CHUNK];
    __shared__                float    alpha_s[R];
    __shared__ __align__(16) half      Ms[4 * PD_CHUNK]; // mask rows of the chunk, one per query token

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int head = blockIdx.y;                // KV head
    const int split = blockIdx.x;
    const int c0 = split * chunks_per_split;
    const int c1 = min(n_chunks, c0 + chunks_per_split);
    const int n_rows = nb * PD_GQA;

    // S phase: wave -> (m-tile, 16-token n-tile). Q fragments stay in registers.
    const int s_mt = warp / 4;
    const int s_nt = warp % 4;
    const bool s_active = s_mt < M_TILES;
    pd_half8 Qa[PD_D / 16];
    {
        const int r = s_mt * 16 + lane % 16;
        const bool valid = s_active && r < n_rows;
        const float4 * q = valid ? (const float4 *) (Q + (r / PD_GQA) * q_nb1 + (head * PD_GQA + r % PD_GQA) * q_nb2) : nullptr;
#pragma unroll
        for (int ks = 0; ks < PD_D / 16; ++ks) {
            const int d0 = ks * 16 + 8 * (lane / 16);
            const float4 lo = valid ? q[d0 / 4 + 0] : make_float4(0, 0, 0, 0);
            const float4 hi = valid ? q[d0 / 4 + 1] : make_float4(0, 0, 0, 0);
            Qa[ks] = pd_half8{(_Float16) (lo.x * scale), (_Float16) (lo.y * scale), (_Float16) (lo.z * scale), (_Float16) (lo.w * scale),
                              (_Float16) (hi.x * scale), (_Float16) (hi.y * scale), (_Float16) (hi.z * scale), (_Float16) (hi.w * scale)};
        }
    }

    // Softmax phase: TPR consecutive threads per row.
    const int sm_row = tid / TPR;
    const int sm_sub = tid % TPR;
    float m_run = -INFINITY;
    float l_run = 0.0f;

    // PV phase: wave -> 32 output dims.
    pd_float8 O[M_TILES][2];
#pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt) {
#pragma unroll
        for (int dt = 0; dt < 2; ++dt) {
            O[mt][dt] = pd_float8{0, 0, 0, 0, 0, 0, 0, 0};
        }
    }

    uint4 kreg[PD_K_LOADS];
    uint4 vreg[PD_V_LOADS];
    // Mask values travel with K/V: a separate global read in the S phase would wait for the whole prefetch
    // (vmcnt waits are in order).
    constexpr int M_LOADS = 4 * PD_CHUNK / (PD_NWARPS * 32);
    half mreg[M_LOADS];
    // Blocks that read data from another queue wait once, before any load: with a table only those whose chunks reach
    // a ring page. (Waiting inside the chunk loop put a conditional barrier and a system-scope fence between the
    // prefetch and its use, which cost about 6% on every layer even when nothing was streamed.)
    if (sync.enabled()) {
        bool need = !PAGED;
        if (PAGED) {
            for (int p = c0 * PD_CHUNK / PD_PAGE; c0 < c1 && p <= ((c1 - 1) * PD_CHUNK) / PD_PAGE; ++p) {
                need |= K.table[p] < 0;
            }
        }
        if (need) {
            if (tid == 0) {
                sync.wait();
            }
            __syncthreads();
            __atomic_thread_fence(__ATOMIC_ACQUIRE);
        }
    }
    auto fetch = [&](const int c) {
        const char * kc = PAGED ? K.chunk(c * PD_CHUNK, head) : nullptr;
        const char * vc = PAGED ? V.chunk(c * PD_CHUNK, head) : nullptr;
#pragma unroll
        for (int j = 0; j < PD_K_LOADS; ++j) {
            const int i = tid + j * PD_NWARPS * 32;
            const int r = i / PD_K_VEC, col = i % PD_K_VEC;
            const int t = c * PD_CHUNK + r;
            kreg[j] = i < PD_CHUNK * PD_K_VEC && t < n_kv ? ((const uint4 *) (PAGED ? kc + r * K.nb1 : K.row(t, head)))[col] : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int j = 0; j < PD_V_LOADS; ++j) {
            const int i = tid + j * PD_NWARPS * 32;
            const int r = i / PD_V_VEC, col = i % PD_V_VEC;
            const int t = c * PD_CHUNK + r;
            vreg[j] = i < PD_CHUNK * PD_V_VEC && t < n_kv ? ((const uint4 *) (PAGED ? vc + r * V.nb1 : V.row(t, head)))[col] : make_uint4(0, 0, 0, 0);
        }
#pragma unroll
        for (int j = 0; j < M_LOADS; ++j) {
            const int i = tid + j * PD_NWARPS * 32;
            const int q = i / PD_CHUNK, t = c * PD_CHUNK + i % PD_CHUNK;
            mreg[j] = mask && q < nb && t < n_kv ? ((const half *) (mask + q * mask_nb1))[t] : __float2half(0.0f);
        }
    };

    if (c0 < c1) {
        fetch(c0);
    }
    for (int c = c0; c < c1; ++c) {
        // 1. registers -> LDS, then prefetch the next chunk.
#pragma unroll
        for (int j = 0; j < PD_K_LOADS; ++j) {
            const int i = tid + j * PD_NWARPS * 32;
            if (i < PD_CHUNK * PD_K_VEC) ((uint4 *) Kr)[i] = kreg[j];
        }
#pragma unroll
        for (int j = 0; j < PD_V_LOADS; ++j) {
            const int i = tid + j * PD_NWARPS * 32;
            if (i < PD_CHUNK * PD_V_VEC) ((uint4 *) Vr)[i] = vreg[j];
        }
#pragma unroll
        for (int j = 0; j < M_LOADS; ++j) {
            Ms[tid + j * PD_NWARPS * 32] = mreg[j];
        }
        __syncthreads();
        if (c + 1 < c1 && !(ablate & 4)) {
            fetch(c + 1);
        }

        // 2. S = Q K^T for this wave's tile.
        if (s_active) {
            pd_float8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
            const int n = s_nt * 16 + lane % 16;
            const char * krow = Kr + n * PD_K_ROW;
#pragma unroll
            for (int ks = 0; ks < PD_D / 16; ++ks) {
                if (ablate & 1) break;
                const int d0 = ks * 16 + 8 * (lane / 16);
                const char * blk = krow + (d0 / PD_QK) * PD_Q8_BLK;
                const half d = *(const half *) blk;
                const half2 d2 = __halves2half2(d, d);
                const uint16_t * qs = (const uint16_t *) (blk + 2 + d0 % PD_QK);
                half2 b2[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t w = qs[j];
                    b2[j] = __hmul2(pd_i8x2(w, w >> 8), d2);
                }
                acc = pd_wmma(Qa[ks], *(const pd_half8 *) b2, acc);
            }
            const int t_kv = c * PD_CHUNK + n;
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int r = s_mt * 16 + 8 * (lane / 16) + l;
                Ss[r * PD_CHUNK + n] = r < n_rows && t_kv < n_kv ? acc[l] + __half2float(Ms[(r / PD_GQA) * PD_CHUNK + n]) : -INFINITY;
            }
        }
        __syncthreads();

        // 3. Online softmax by row.
        {
            const float * srow = Ss + sm_row * PD_CHUNK + sm_sub * TOK;
            float s[TOK];
            float mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < TOK; ++j) {
                s[j] = srow[j];
                mx = fmaxf(mx, s[j]);
            }
#pragma unroll
            for (int o = TPR / 2; o > 0; o /= 2) {
                mx = fmaxf(mx, __shfl_xor(mx, o, 32));
            }
            const float m_new = fmaxf(m_run, mx);
            float sum = 0.0f;
            _Float16 * prow = Ps + sm_row * PD_CHUNK + sm_sub * TOK;
#pragma unroll
            for (int j = 0; j < TOK; ++j) {
                const float p = m_new == -INFINITY ? 0.0f : __expf(s[j] - m_new);
                sum += p;
                prow[j] = (_Float16) p;
            }
#pragma unroll
            for (int o = TPR / 2; o > 0; o /= 2) {
                sum += __shfl_xor(sum, o, 32);
            }
            const float alpha = m_new == -INFINITY ? 1.0f : __expf(m_run - m_new);
            l_run = l_run * alpha + sum;
            m_run = m_new;
            if (sm_sub == 0) {
                alpha_s[sm_row] = alpha;
            }
        }
        __syncthreads();

        // 4. O = O * alpha + P V for this wave's 32 dims.
#pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const float a = alpha_s[mt * 16 + 8 * (lane / 16) + l];
                O[mt][0][l] *= a;
                O[mt][1][l] *= a;
            }
        }
#pragma unroll
        for (int ks = 0; ks < PD_CHUNK / 16; ++ks) {
            if (ablate & 2) break;
            const int k0 = ks * 16 + 8 * (lane / 16);
            pd_half8 a[M_TILES];
#pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                a[mt] = *(const pd_half8 *) (Ps + (mt * 16 + lane % 16) * PD_CHUNK + k0);
            }
#pragma unroll
            for (int dt = 0; dt < 2; ++dt) {
                const int dim = warp * 32 + dt * 16 + lane % 16;
                const int boff = (dim / PD_QK) * PD_Q4_BLK;
                const int shift = (dim % PD_QK) < 16 ? 0 : 4;
                half2 b2[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const char * blk0 = Vr + (k0 + 2*j + 0) * PD_V_ROW + boff;
                    const char * blk1 = Vr + (k0 + 2*j + 1) * PD_V_ROW + boff;
                    const uint32_t q0 = ((uint8_t) blk0[2 + dim % 16] >> shift) & 0x0F;
                    const uint32_t q1 = ((uint8_t) blk1[2 + dim % 16] >> shift) & 0x0F;
                    b2[j] = __hmul2(pd_u4x2(q0, q1), __halves2half2(*(const half *) blk0, *(const half *) blk1));
                }
#pragma unroll
                for (int mt = 0; mt < M_TILES; ++mt) {
                    O[mt][dt] = pd_wmma(a[mt], *(const pd_half8 *) b2, O[mt][dt]);
                }
            }
        }
        __syncthreads();
    }

    // Partial results for valid rows: unnormalized O plus (max, sum) per row.
    const int64_t base = ((int64_t) split * gridDim.y + head) * R;
#pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt) {
#pragma unroll
        for (int dt = 0; dt < 2; ++dt) {
            const int dim = warp * 32 + dt * 16 + lane % 16;
#pragma unroll
            for (int l = 0; l < 8; ++l) {
                const int r = mt * 16 + 8 * (lane / 16) + l;
                if (r < n_rows) {
                    O_part[(base + r) * PD_D + dim] = O[mt][dt][l];
                }
            }
        }
    }
    if (sm_sub == 0 && sm_row < n_rows) {
        ML_part[base + sm_row] = make_float2(m_run, l_run);
    }
#endif // PD_RDNA4
}

// One block per (row, KV head), one thread per output dim.
static __global__ void pd_combine(
        const float * __restrict__ O_part, const float2 * __restrict__ ML_part, const int n_splits, const int R,
        char * __restrict__ dst, const int64_t dst_nb1, const int64_t dst_nb2, const pd_sync sync) {
    // Every partial block has finished (stream order), so the data they read is free again.
    if (sync.consumed && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        sync.release();
    }
    const int r = blockIdx.x;
    const int head = blockIdx.y;
    const int n_heads = gridDim.y;
    float M = -INFINITY;
    for (int s = 0; s < n_splits; ++s) {
        M = fmaxf(M, ML_part[((int64_t) s * n_heads + head) * R + r].x);
    }
    float num = 0.0f, den = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        const int64_t i = ((int64_t) s * n_heads + head) * R + r;
        const float2 ml = ML_part[i];
        const float w = ml.x == -INFINITY ? 0.0f : expf(ml.x - M);
        num += w * O_part[i * PD_D + threadIdx.x];
        den += w * ml.y;
    }
    float * out = (float *) (dst + (r / PD_GQA) * dst_nb2 + (head * PD_GQA + r % PD_GQA) * dst_nb1);
    out[threadIdx.x] = num / den;
}
