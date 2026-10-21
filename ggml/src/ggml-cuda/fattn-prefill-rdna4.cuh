#pragma once

// Prefill attention for head size 256, GQA 6, on RDNA4 WMMA (gfx1201), over f16 K/V: one part of the chunked paged
// prefill (fattn-paged-decode.cu), which writes the normalized output and the per-row log-sum-exp for pd_merge.
// Plain HIP (no ggml dependencies).
//
// The design packs the GQA heads and transposes the score product:
//   * Rows are (query token, head) pairs of one KV head, 16 per wave: a block of PP_NW waves covers 16 * PP_NW rows,
//     PP_NW * 16 / 6 query tokens, and each K/V tile staged in LDS serves all of them (the stock kernel packs two heads
//     per tile, so it stages each tile three times per KV head).
//   * S^T = K Q^T: the WMMA takes 16 keys of the tile as A and 16 rows' queries as B, so a lane's accumulator holds 8
//     keys of one row (lane l: row l % 16, keys 8 * (l / 16) + 0..7). A row's softmax needs one exchange between the
//     two half-waves, and the lane's P values are already the B operand of O^T += V^T P^T (same lane layout).
//   * Q stays in registers as f16 fragments, scaled by scale * log2(e), so the softmax works in base 2.
//   * The output accumulator is O^T, 16 tiles of 16 dims x 16 rows, accumulated in f16 as the stock kernel does on
//     RDNA4 (64 VGPRs; f32 would take 128 next to Q's 64 and spill Q): lane l holds row l % 16, dims
//     16 * dt + 8 * (l / 16) + 0..7, so normalizing a row is lane-local too. The chunked prefill's parts bound the
//     f16 sums (6144 keys), as for the stock kernel.
//   * The accumulator is rescaled only when some row's maximum grows (a wave-wide vote), and p <= 1 keeps the f16
//     sums in range.
// K is staged as [key][dim], V transposed as [dim][key], both padded so that a fragment is one conflict-free 16-byte
// LDS read. Fragments are read in groups of PP_PF ahead of their WMMAs, with a scheduling barrier between groups:
// otherwise the compiler hoists all 16 reads (64 VGPRs) next to Q (64) and O (128) and spills Q. Barriers are
// LDS-scoped (they only order LDS).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#if defined(__gfx1200__) || defined(__gfx1201__)
#define PP_RDNA4 1
#else
#define PP_RDNA4 0
#endif

static constexpr int   PP_D    = 256;
static constexpr int   PP_GQA  = 6;
static constexpr int   PP_NW   = 12;               // waves per block: 192 rows = 32 query tokens x 6 heads
static constexpr int   PP_ROWS = PP_NW * 16;
static constexpr int   PP_BQ   = PP_ROWS / PP_GQA; // query tokens per block
static constexpr int   PP_TILE = 32;               // keys per staged tile
static constexpr int   PP_KSTR = PP_D + 8;         // sK row stride (halves)
static constexpr int   PP_VSTR = PP_TILE + 8;      // sV row stride (halves)
static constexpr int   PP_PF   = 4;                // fragments read ahead of their WMMAs (a group; see below)

static_assert(PP_ROWS % PP_GQA == 0, "a block covers whole query tokens");

#if PP_RDNA4
typedef _Float16 pp_half8  __attribute__((ext_vector_type(8)));
typedef float    pp_float8 __attribute__((ext_vector_type(8)));

static __device__ __forceinline__ void pp_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
#endif

// Q: f32, token t head h at Q + t * q_nb1 + h * q_nb2 (bytes). K, V: f16 rows of PP_D, key k of KV head g at
// base + k * nb1 + g * nb2 (bytes). mask: f16, token t key k at mask + t * mask_nb1 (bytes) + 2 k; may be null.
// dst: f32 [n_q][n_heads][PP_D]; lse: f32 [n_q][n_heads], natural log.
__launch_bounds__(PP_NW * 32, 1)
static __global__ void pp_attn_f16(
        const char * __restrict__ Q, const int64_t q_nb1, const int64_t q_nb2,
        const char * __restrict__ K, const int64_t k_nb1, const int64_t k_nb2,
        const char * __restrict__ V, const int64_t v_nb1, const int64_t v_nb2,
        const char * __restrict__ mask, const int64_t mask_nb1,
        float * __restrict__ dst, float * __restrict__ lse,
        const int n_q, const int n_kv, const int n_heads, const float scale) {
#if PP_RDNA4
    __shared__ __align__(16) _Float16 sK[PP_TILE * PP_KSTR];
    __shared__ __align__(16) _Float16 sV[PP_D * PP_VSTR];
    __shared__ __align__(16) _Float16 sM[PP_BQ * PP_TILE];

    const int tid  = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int c    = lane % 16;
    const int h    = lane / 16;
    const int kvh  = blockIdx.y;
    const int q0   = blockIdx.x * PP_BQ;

    const int  row   = warp * 16 + c;
    const int  qi    = row / PP_GQA;
    const int  qtok  = q0 + qi;
    const int  qhead = kvh * PP_GQA + row % PP_GQA;
    const bool live  = qtok < n_q;

    // Q fragments: dims 16 * ks + 8 * h + 0..7 of this lane's row, in base-2 logit units
    pp_half8 qf[PP_D / 16];
    {
        const float mul = scale * 1.4426950408889634f;
        const float * q = (const float *) (Q + (int64_t) (live ? qtok : 0) * q_nb1 + (int64_t) qhead * q_nb2);
#pragma unroll
        for (int ks = 0; ks < PP_D / 16; ++ks) {
            const float4 lo = live ? *(const float4 *) (q + 16 * ks + 8 * h)     : make_float4(0, 0, 0, 0);
            const float4 hi = live ? *(const float4 *) (q + 16 * ks + 8 * h + 4) : make_float4(0, 0, 0, 0);
            qf[ks] = pp_half8{(_Float16) (lo.x * mul), (_Float16) (lo.y * mul), (_Float16) (lo.z * mul), (_Float16) (lo.w * mul),
                              (_Float16) (hi.x * mul), (_Float16) (hi.y * mul), (_Float16) (hi.z * mul), (_Float16) (hi.w * mul)};
        }
    }

    pp_half8 acc[PP_D / 16];
#pragma unroll
    for (int dt = 0; dt < PP_D / 16; ++dt) {
        acc[dt] = pp_half8{0, 0, 0, 0, 0, 0, 0, 0};
    }
    float m_run = -INFINITY; // the reference maximum of this lane's row (both half-waves agree)
    float l_run = 0.0f;      // the sum of p over this lane's keys

    const char * Kh = K + (int64_t) kvh * k_nb2;
    const char * Vh = V + (int64_t) kvh * v_nb2;

    for (int kv0 = 0; kv0 < n_kv; kv0 += PP_TILE) {
        // K: [key][dim], 32 pieces of 8 dims per key (not unrolled: the loads would stay live next to Q and O)
#pragma unroll 1
        for (int i = tid; i < PP_TILE * (PP_D / 8); i += PP_NW * 32) {
            const int key = i / (PP_D / 8), p = i % (PP_D / 8);
            const int k = kv0 + key;
            const uint4 v = k < n_kv ? *(const uint4 *) (Kh + (int64_t) k * k_nb1 + 16 * p) : make_uint4(0, 0, 0, 0);
            *(uint4 *) (sK + key * PP_KSTR + 8 * p) = v;
        }
        // V: transposed to [dim][key], a key pair per dword
#pragma unroll 1
        for (int i = tid; i < (PP_TILE / 2) * (PP_D / 8); i += PP_NW * 32) {
            const int kp = i % (PP_TILE / 2), g = i / (PP_TILE / 2);
            const int k = kv0 + 2 * kp;
            const uint4 a = k     < n_kv ? *(const uint4 *) (Vh + (int64_t) (k    ) * v_nb1 + 16 * g) : make_uint4(0, 0, 0, 0);
            const uint4 b = k + 1 < n_kv ? *(const uint4 *) (Vh + (int64_t) (k + 1) * v_nb1 + 16 * g) : make_uint4(0, 0, 0, 0);
            const uint32_t wa[4] = { a.x, a.y, a.z, a.w };
            const uint32_t wb[4] = { b.x, b.y, b.z, b.w };
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                *(uint32_t *) (sV + (8 * g + 2 * e    ) * PP_VSTR + 2 * kp) = (wa[e] & 0xFFFFu) | (wb[e] << 16);
                *(uint32_t *) (sV + (8 * g + 2 * e + 1) * PP_VSTR + 2 * kp) = (wa[e] >> 16) | (wb[e] & 0xFFFF0000u);
            }
        }
        // mask rows of the block's query tokens; keys past n_kv are masked
#pragma unroll 1
        for (int i = tid; i < PP_BQ * PP_TILE; i += PP_NW * 32) {
            const int t = i / PP_TILE, key = i % PP_TILE;
            const int k = kv0 + key;
            _Float16 m = (_Float16) 0.0f;
            if (k >= n_kv) {
                m = (_Float16) -INFINITY;
            } else if (mask && q0 + t < n_q) {
                m = ((const _Float16 *) (mask + (int64_t) (q0 + t) * mask_nb1))[k];
            }
            sM[i] = m;
        }
        pp_lds_barrier();

#pragma unroll
        for (int mt = 0; mt < PP_TILE / 16; ++mt) {
            // S^T for keys mt * 16 + 8 * h + 0..7 of row c
            pp_float8 s = {0, 0, 0, 0, 0, 0, 0, 0};
            const _Float16 * kr = sK + (mt * 16 + c) * PP_KSTR + 8 * h;
#pragma unroll
            for (int g = 0; g < PP_D / 16; g += PP_PF) {
                pp_half8 kf[PP_PF];
#pragma unroll
                for (int j = 0; j < PP_PF; ++j) {
                    kf[j] = *(const pp_half8 *) (kr + 16 * (g + j));
                }
                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int j = 0; j < PP_PF; ++j) {
                    s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kf[j], qf[g + j], s);
                }
            }
            const _Float16 * mr = sM + qi * PP_TILE + mt * 16 + 8 * h;
            float mx = -INFINITY;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                s[e] += 1.4426950408889634f * (float) mr[e];
                mx = fmaxf(mx, s[e]);
            }
            mx = fmaxf(mx, __shfl_xor(mx, 16, 32));

            // rescale only when some row's maximum grows
            if (__any(mx > m_run)) {
                const float m_new = fmaxf(m_run, mx);
                const float alpha = m_new == -INFINITY ? 1.0f : __builtin_amdgcn_exp2f(m_run - m_new);
                const _Float16 a16 = (_Float16) alpha;
#pragma unroll
                for (int dt = 0; dt < PP_D / 16; ++dt) {
                    acc[dt] *= a16;
                }
                l_run *= alpha;
                m_run  = m_new;
            }

            pp_half8 p16;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float p = s[e] == -INFINITY ? 0.0f : __builtin_amdgcn_exp2f(s[e] - m_run);
                l_run += p;
                p16[e] = (_Float16) p;
            }

            // O^T += V^T P^T: A = dims 16 * dt + c, keys mt * 16 + 8 * h + 0..7
            const _Float16 * vr = sV + c * PP_VSTR + mt * 16 + 8 * h;
#pragma unroll
            for (int g = 0; g < PP_D / 16; g += PP_PF) {
                pp_half8 vf[PP_PF];
#pragma unroll
                for (int j = 0; j < PP_PF; ++j) {
                    vf[j] = *(const pp_half8 *) (vr + 16 * (g + j) * PP_VSTR);
                }
                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for (int j = 0; j < PP_PF; ++j) {
                    acc[g + j] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(vf[j], p16, acc[g + j]);
                }
            }
        }
        pp_lds_barrier();
    }

    if (!live) {
        return;
    }
    const float l = l_run + __shfl_xor(l_run, 16, 32);
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    float * o = dst + ((int64_t) qtok * n_heads + qhead) * PP_D;
#pragma unroll
    for (int dt = 0; dt < PP_D / 16; ++dt) {
        *(float4 *) (o + 16 * dt + 8 * h)     = make_float4((float) acc[dt][0] * inv, (float) acc[dt][1] * inv,
                                                        (float) acc[dt][2] * inv, (float) acc[dt][3] * inv);
        *(float4 *) (o + 16 * dt + 8 * h + 4) = make_float4((float) acc[dt][4] * inv, (float) acc[dt][5] * inv,
                                                        (float) acc[dt][6] * inv, (float) acc[dt][7] * inv);
    }
    if (lse && h == 0) {
        lse[(int64_t) qtok * n_heads + qhead] = l > 0.0f ? (m_run + log2f(l)) * 0.6931471805599453f : -INFINITY;
    }
#endif // PP_RDNA4
}
