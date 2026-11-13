#include "common.cuh"
#include "convert.cuh"
#include "fattn-banded.cuh"
#include "fattn.cuh"

#include <algorithm>
#include <cstdint>

static __device__ __forceinline__ float fattn_banded_load(
        const char * ptr, const int type) {
    switch (type) {
        case GGML_TYPE_F32:
            return *(const float *) ptr;
        case GGML_TYPE_F16:
            return __half2float(*(const half *) ptr);
        case GGML_TYPE_BF16: {
            // Read BF16 as raw bits so this kernel needs no native BF16 support; all math stays FP32.
            const uint32_t bits = uint32_t(*(const uint16_t *) ptr) << 16;
            return __uint_as_float(bits);
        }
        default:
            return 0.0f;
    }
}

template<int D, int WARPS_PER_BLOCK>
static __global__ void flash_attn_ext_banded_f32(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ v,
        const char * __restrict__ mask,
        const char * __restrict__ rel,
        float      * __restrict__ dst,
        float scale,
        int type_k,
        int type_v,
        int type_rel,
        int64_t n_q,
        int64_t n_kv,
        int64_t n_head_q,
        int64_t n_head_kv,
        int64_t n_batch,
        int64_t rel_extent,
        int64_t mask_ne2,
        int64_t mask_ne3,
        uint64_t q_nb1,
        uint64_t q_nb2,
        uint64_t q_nb3,
        uint64_t k_nb0,
        uint64_t k_nb1,
        uint64_t k_nb2,
        uint64_t k_nb3,
        uint64_t v_nb0,
        uint64_t v_nb1,
        uint64_t v_nb2,
        uint64_t v_nb3,
        uint64_t m_nb1,
        uint64_t m_nb2,
        uint64_t m_nb3,
        const float * __restrict__ sinks,
        int64_t sinks_s1,
        int64_t sinks_s2,
        uint64_t r_nb0,
        uint64_t r_nb1,
        uint64_t r_nb2,
        uint64_t r_nb3,
        int64_t rel_ne3) {
    constexpr int values_per_lane = D / WARP_SIZE;
    static_assert(D == 64 || D == 128, "banded FA supports head dimensions 64 and 128");
    static_assert(D % WARP_SIZE == 0, "head dimension must be divisible by warp size");

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int64_t iq = int64_t(blockIdx.x) * WARPS_PER_BLOCK + warp;
    const int64_t ih = blockIdx.y;
    const int64_t ib = blockIdx.z;

    if (iq >= n_q || ih >= n_head_q || ib >= n_batch) {
        return;
    }

    const int64_t ih_kv = ih / (n_head_q / n_head_kv);
    const char * q_row = q + uint64_t(iq)*q_nb1 + uint64_t(ih)*q_nb2 + uint64_t(ib)*q_nb3;

    float q_reg[values_per_lane];
    float out[values_per_lane];
#pragma unroll
    for (int j = 0; j < values_per_lane; ++j) {
        const int d = lane + j*WARP_SIZE;
        q_reg[j] = *(const float *)(q_row + uint64_t(d)*sizeof(float));
        out[j] = 0.0f;
    }

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    const int64_t q_offset = n_kv - n_q;

    for (int64_t ik = 0; ik < n_kv; ++ik) {
        if (mask && __half2float(*(const half *) (mask + uint64_t(ik)*sizeof(half) +
                uint64_t(iq)*m_nb1 + uint64_t(ih % mask_ne2)*m_nb2 +
                uint64_t(ib % mask_ne3)*m_nb3)) == -INFINITY) {
            continue;
        }

        const char * k_row = k + uint64_t(ik)*k_nb1 + uint64_t(ih_kv)*k_nb2 + uint64_t(ib)*k_nb3;
        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < values_per_lane; ++j) {
            const int d = lane + j*WARP_SIZE;
            dot += q_reg[j] * fattn_banded_load(k_row + uint64_t(d)*k_nb0, type_k);
        }
        dot = warp_reduce_sum(dot);

        float score = dot * scale;
        if (lane == 0) {
            const int64_t rel_dist = iq + q_offset - ik;
            if (rel_dist >= 0 && rel_dist < rel_extent) {
                const char * rel_value = rel +
                    uint64_t(rel_dist)*r_nb0 + uint64_t(ih)*r_nb1 +
                    uint64_t(iq)*r_nb2 + uint64_t(ib % rel_ne3)*r_nb3;
                score += fattn_banded_load(rel_value, type_rel);
            }
            if (mask) {
                const char * mask_value = mask + uint64_t(ik)*sizeof(half) +
                    uint64_t(iq)*m_nb1 + uint64_t(ih % mask_ne2)*m_nb2 +
                    uint64_t(ib % mask_ne3)*m_nb3;
                score += __half2float(*(const half *) mask_value);
            }
        }
        score = __shfl_sync(0xffffffff, score, 0, WARP_SIZE);

        if (score == -INFINITY) {
            continue;
        }

        float old_scale = 1.0f;
        float value_scale = 1.0f;
        if (score > row_max) {
            old_scale = expf(row_max - score);
            row_max = score;
        } else {
            value_scale = expf(score - row_max);
        }

        const char * v_row = v + uint64_t(ik)*v_nb1 + uint64_t(ih_kv)*v_nb2 + uint64_t(ib)*v_nb3;
#pragma unroll
        for (int j = 0; j < values_per_lane; ++j) {
            const int d = lane + j*WARP_SIZE;
            const float vv = fattn_banded_load(v_row + uint64_t(d)*v_nb0, type_v);
            out[j] = out[j]*old_scale + vv*value_scale;
        }
        row_sum = row_sum*old_scale + value_scale;
    }

    // attention sink: an extra logit with a zero value
    if (sinks) {
        const float sink = sinks[ih + iq*sinks_s1 + ib*sinks_s2];
        if (sink > row_max) {
            const float old_scale = expf(row_max - sink);
#pragma unroll
            for (int j = 0; j < values_per_lane; ++j) {
                out[j] *= old_scale;
            }
            row_sum = row_sum*old_scale + 1.0f;
            row_max = sink;
        } else {
            row_sum += expf(sink - row_max);
        }
    }

    const float inv_sum = row_sum == 0.0f ? 0.0f : 1.0f/row_sum;
    float * dst_row = dst + ((ib*n_q + iq)*n_head_q + ih)*D;
#pragma unroll
    for (int j = 0; j < values_per_lane; ++j) {
        const int d = lane + j*WARP_SIZE;
        dst_row[d] = out[j]*inv_sum;
    }
}

template<int D, int NW>
static __global__ void flash_attn_ext_banded_f32_split(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ v,
        const char * __restrict__ mask,
        const char * __restrict__ rel,
        float      * __restrict__ part,
        float scale,
        int type_k,
        int type_v,
        int type_rel,
        int64_t n_q,
        int64_t n_kv,
        int64_t n_head_q,
        int64_t n_head_kv,
        int64_t n_chunks,
        int64_t chunk,
        int64_t rel_extent,
        int64_t mask_ne2,
        int64_t mask_ne3,
        uint64_t q_nb1,
        uint64_t q_nb2,
        uint64_t q_nb3,
        uint64_t k_nb0,
        uint64_t k_nb1,
        uint64_t k_nb2,
        uint64_t k_nb3,
        uint64_t v_nb0,
        uint64_t v_nb1,
        uint64_t v_nb2,
        uint64_t v_nb3,
        uint64_t m_nb1,
        uint64_t m_nb2,
        uint64_t m_nb3,
        uint64_t r_nb0,
        uint64_t r_nb1,
        uint64_t r_nb2,
        uint64_t r_nb3,
        int64_t rel_ne3) {
    constexpr int values_per_lane = D / WARP_SIZE;
    static_assert(D == 64 || D == 128, "banded FA supports head dimensions 64 and 128");

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int64_t ic = blockIdx.x;
    const int64_t ih = blockIdx.y;
    const int64_t iq = int64_t(blockIdx.z) % n_q;
    const int64_t ib = int64_t(blockIdx.z) / n_q;

    const int64_t ih_kv = ih / (n_head_q / n_head_kv);
    const char * q_row = q + uint64_t(iq)*q_nb1 + uint64_t(ih)*q_nb2 + uint64_t(ib)*q_nb3;

    float q_reg[values_per_lane];
    float out[values_per_lane];
#pragma unroll
    for (int j = 0; j < values_per_lane; ++j) {
        const int d = lane + j*WARP_SIZE;
        q_reg[j] = *(const float *)(q_row + uint64_t(d)*sizeof(float));
        out[j] = 0.0f;
    }

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    const int64_t q_offset = n_kv - n_q;
    const int64_t ik0 = ic*chunk;
    const int64_t ik1 = ik0 + chunk < n_kv ? ik0 + chunk : n_kv;

    for (int64_t ik = ik0 + warp; ik < ik1; ik += NW) {
        if (mask && __half2float(*(const half *) (mask + uint64_t(ik)*sizeof(half) +
                uint64_t(iq)*m_nb1 + uint64_t(ih % mask_ne2)*m_nb2 +
                uint64_t(ib % mask_ne3)*m_nb3)) == -INFINITY) {
            continue;
        }

        const char * k_row = k + uint64_t(ik)*k_nb1 + uint64_t(ih_kv)*k_nb2 + uint64_t(ib)*k_nb3;
        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < values_per_lane; ++j) {
            const int d = lane + j*WARP_SIZE;
            dot += q_reg[j] * fattn_banded_load(k_row + uint64_t(d)*k_nb0, type_k);
        }
        dot = warp_reduce_sum(dot);

        float score = dot * scale;
        if (lane == 0) {
            const int64_t rel_dist = iq + q_offset - ik;
            if (rel_dist >= 0 && rel_dist < rel_extent) {
                const char * rel_value = rel +
                    uint64_t(rel_dist)*r_nb0 + uint64_t(ih)*r_nb1 +
                    uint64_t(iq)*r_nb2 + uint64_t(ib % rel_ne3)*r_nb3;
                score += fattn_banded_load(rel_value, type_rel);
            }
            if (mask) {
                const char * mask_value = mask + uint64_t(ik)*sizeof(half) +
                    uint64_t(iq)*m_nb1 + uint64_t(ih % mask_ne2)*m_nb2 +
                    uint64_t(ib % mask_ne3)*m_nb3;
                score += __half2float(*(const half *) mask_value);
            }
        }
        score = __shfl_sync(0xffffffff, score, 0, WARP_SIZE);

        if (score == -INFINITY) {
            continue;
        }

        float old_scale = 1.0f;
        float value_scale = 1.0f;
        if (score > row_max) {
            old_scale = expf(row_max - score);
            row_max = score;
        } else {
            value_scale = expf(score - row_max);
        }

        const char * v_row = v + uint64_t(ik)*v_nb1 + uint64_t(ih_kv)*v_nb2 + uint64_t(ib)*v_nb3;
#pragma unroll
        for (int j = 0; j < values_per_lane; ++j) {
            const int d = lane + j*WARP_SIZE;
            const float vv = fattn_banded_load(v_row + uint64_t(d)*v_nb0, type_v);
            out[j] = out[j]*old_scale + vv*value_scale;
        }
        row_sum = row_sum*old_scale + value_scale;
    }

    __shared__ float s_max[NW];
    __shared__ float s_sum[NW];
    __shared__ float s_out[NW][D];
    if (lane == 0) {
        s_max[warp] = row_max;
        s_sum[warp] = row_sum;
    }
#pragma unroll
    for (int j = 0; j < values_per_lane; ++j) {
        s_out[warp][lane + j*WARP_SIZE] = out[j];
    }
    __syncthreads();

    if (warp == 0) {
        float m = -INFINITY;
#pragma unroll
        for (int w = 0; w < NW; ++w) {
            m = fmaxf(m, s_max[w]);
        }
        float acc[values_per_lane];
#pragma unroll
        for (int j = 0; j < values_per_lane; ++j) {
            acc[j] = 0.0f;
        }
        float sum = 0.0f;
        if (m != -INFINITY) {
#pragma unroll
            for (int w = 0; w < NW; ++w) {
                if (s_max[w] == -INFINITY) {
                    continue;
                }
                const float f = expf(s_max[w] - m);
                sum += s_sum[w]*f;
#pragma unroll
                for (int j = 0; j < values_per_lane; ++j) {
                    acc[j] += s_out[w][lane + j*WARP_SIZE]*f;
                }
            }
        }
        float * p = part + ((((ib*n_q + iq)*n_head_q + ih)*n_chunks) + ic)*(D + 2);
        if (lane == 0) {
            p[0] = m;
            p[1] = sum;
        }
#pragma unroll
        for (int j = 0; j < values_per_lane; ++j) {
            p[2 + lane + j*WARP_SIZE] = acc[j];
        }
    }
}

template<int D>
static __global__ void flash_attn_ext_banded_f32_merge(
        const float * __restrict__ part,
        float       * __restrict__ dst,
        int64_t n_rows,
        int64_t n_head_q,
        int64_t n_q,
        int64_t n_chunks,
        const float * __restrict__ sinks,
        int64_t sinks_s1,
        int64_t sinks_s2) {
    constexpr int values_per_lane = D / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int64_t row = int64_t(blockIdx.x)*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    if (row >= n_rows) {
        return;
    }
    const int64_t ih = row % n_head_q;
    const int64_t iq = (row / n_head_q) % n_q;
    const int64_t ib = row / (n_head_q*n_q);
    const float * p = part + row*n_chunks*(D + 2);

    float m = -INFINITY;
    for (int64_t c = 0; c < n_chunks; ++c) {
        m = fmaxf(m, p[c*(D + 2)]);
    }
    float out[values_per_lane];
#pragma unroll
    for (int j = 0; j < values_per_lane; ++j) {
        out[j] = 0.0f;
    }
    float row_sum = 0.0f;
    if (m != -INFINITY) {
        for (int64_t c = 0; c < n_chunks; ++c) {
            const float * pc = p + c*(D + 2);
            if (pc[0] == -INFINITY) {
                continue;
            }
            const float f = expf(pc[0] - m);
            row_sum += pc[1]*f;
#pragma unroll
            for (int j = 0; j < values_per_lane; ++j) {
                out[j] += pc[2 + lane + j*WARP_SIZE]*f;
            }
        }
    }
    float row_max = m;

    if (sinks) {
        const float sink = sinks[ih + iq*sinks_s1 + ib*sinks_s2];
        if (sink > row_max) {
            const float old_scale = expf(row_max - sink);
#pragma unroll
            for (int j = 0; j < values_per_lane; ++j) {
                out[j] *= old_scale;
            }
            row_sum = row_sum*old_scale + 1.0f;
            row_max = sink;
        } else {
            row_sum += expf(sink - row_max);
        }
    }

    const float inv_sum = row_sum == 0.0f ? 0.0f : 1.0f/row_sum;
    float * dst_row = dst + row*D;
#pragma unroll
    for (int j = 0; j < values_per_lane; ++j) {
        dst_row[lane + j*WARP_SIZE] = out[j]*inv_sum;
    }
}

#if defined(TURING_MMA_AVAILABLE)
#define FATTN_BANDED_MMA_AVAILABLE
#endif

static constexpr int fattn_banded_mma_bk = 64;

#ifdef FATTN_BANDED_MMA_AVAILABLE
static __device__ __forceinline__ void fattn_banded_mma_16816(
        float (&c)[4], const uint32_t a0, const uint32_t a1, const uint32_t a2, const uint32_t a3,
        const uint32_t b0, const uint32_t b1) {
#ifdef AMPERE_MMA_AVAILABLE
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(b0));
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a2), "r"(a3), "r"(b1));
#endif // AMPERE_MMA_AVAILABLE
}

static __device__ __forceinline__ void fattn_banded_ldmatrix_x4(uint32_t (&r)[4], const half * p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

static __device__ __forceinline__ void fattn_banded_ldmatrix_x4_trans(uint32_t (&r)[4], const half * p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

static __device__ __forceinline__ uint32_t fattn_banded_pack_h2(const float lo, const float hi) {
    const half2 h = __floats2half2_rn(lo, hi);
    return *(const uint32_t *) &h;
}

template<int D, int NW>
static __device__ __forceinline__ void fattn_banded_mma_load_tile(
        half * __restrict__ tile, const char * __restrict__ src, const uint64_t nb1, const int64_t kv0, const int64_t n_kv) {
    constexpr int stride   = D + 8;
    constexpr int chunks   = D/8;
    constexpr int nthreads = NW*WARP_SIZE;
#pragma unroll
    for (int c0 = 0; c0 < fattn_banded_mma_bk*chunks; c0 += nthreads) {
        const int c   = c0 + threadIdx.x;
        const int row = c / chunks;
        const int col = (c % chunks)*8;
        half * dst = tile + row*stride + col;
        if (kv0 + row < n_kv) {
            const char * s = src + uint64_t(kv0 + row)*nb1 + uint64_t(col)*sizeof(half);
#ifdef CP_ASYNC_AVAILABLE
            const uint32_t d = (uint32_t) __cvta_generic_to_shared(dst);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" : : "r"(d), "l"(s));
#else
            *(int4 *) dst = *(const int4 *) s;
#endif // CP_ASYNC_AVAILABLE
        } else {
            *(int4 *) dst = make_int4(0, 0, 0, 0);
        }
    }
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;");
#endif // CP_ASYNC_AVAILABLE
}

template<int n>
static __device__ __forceinline__ void fattn_banded_mma_wait() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;" : : "n"(n));
#endif // CP_ASYNC_AVAILABLE
}
#endif // FATTN_BANDED_MMA_AVAILABLE

static __global__ void flash_attn_ext_banded_mma_tile_flags(
        const char * __restrict__ mask,
        uint8_t    * __restrict__ flags,
        int64_t n_q,
        int64_t n_kv,
        int bq,
        int64_t mask_ne2,
        uint64_t m_nb1,
        uint64_t m_nb2,
        uint64_t m_nb3) {
    const int64_t kt   = blockIdx.x;
    const int64_t qt   = blockIdx.y;
    const int64_t z    = blockIdx.z;
    const int64_t z2   = z % mask_ne2;
    const int64_t z3   = z / mask_ne2;
    const char  * mrow = mask + z2*m_nb2 + z3*m_nb3;

    bool any_visible = false;
    bool any_nonzero = false;
    for (int e = threadIdx.x; e < bq*fattn_banded_mma_bk; e += blockDim.x) {
        const int64_t iq = qt*bq + e / fattn_banded_mma_bk;
        const int64_t ik = kt*fattn_banded_mma_bk + e % fattn_banded_mma_bk;
        if (iq < n_q && ik < n_kv) {
            const float m = __half2float(*(const half *) (mrow + uint64_t(iq)*m_nb1 + uint64_t(ik)*sizeof(half)));
            any_visible |= m != -INFINITY;
            any_nonzero |= m != 0.0f;
        }
    }
    any_visible = __syncthreads_or(any_visible);
    any_nonzero = __syncthreads_or(any_nonzero);
    if (threadIdx.x == 0) {
        const int64_t n_kt = gridDim.x;
        const int64_t n_qt = gridDim.y;
        flags[(z*n_qt + qt)*n_kt + kt] = !any_visible ? 0 : (any_nonzero ? 2 : 1);
    }
}

template<int D, int NW>
__launch_bounds__(NW*WARP_SIZE, 1)
static __global__ void flash_attn_ext_banded_mma(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ v,
        const char * __restrict__ mask,
        const uint8_t * __restrict__ flags,
        const char * __restrict__ rel,
        const float * __restrict__ sinks,
        float      * __restrict__ dst,
        float      * __restrict__ part,
        float scale,
        int type_rel,
        int64_t n_q,
        int64_t n_kv,
        int64_t n_head_q,
        int64_t n_head_kv,
        int64_t n_chunks,
        int64_t tiles_per_chunk,
        int64_t rel_extent,
        int64_t mask_ne2,
        int64_t mask_ne3,
        uint64_t q_nb1,
        uint64_t q_nb2,
        uint64_t q_nb3,
        uint64_t k_nb1,
        uint64_t k_nb2,
        uint64_t k_nb3,
        uint64_t v_nb1,
        uint64_t v_nb2,
        uint64_t v_nb3,
        uint64_t m_nb1,
        uint64_t m_nb2,
        uint64_t m_nb3,
        uint64_t r_nb0,
        uint64_t r_nb1,
        uint64_t r_nb2,
        uint64_t r_nb3,
        int64_t rel_ne3,
        int64_t sinks_s1,
        int64_t sinks_s2) {
#ifdef FATTN_BANDED_MMA_AVAILABLE
    static_assert(D == 64 || D == 128, "banded FA supports head dimensions 64 and 128");
    constexpr int BK     = fattn_banded_mma_bk;
    constexpr int BQ     = NW*16;
    constexpr int stride = D + 8; // padded smem row (halfs): conflict-free ldmatrix
    constexpr int ND     = D/8;
    constexpr int NK     = D/16;

    __shared__ __align__(16) half K_s[BK*stride];
    __shared__ __align__(16) half V_s[BK*stride];

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int g    = lane / 4;
    const int t    = lane % 4;

    const int64_t n_qt = (n_q + BQ - 1)/BQ;
    const int64_t qt   = int64_t(blockIdx.x) / n_chunks;
    const int64_t ic   = int64_t(blockIdx.x) % n_chunks;
    const int64_t ih   = blockIdx.y;
    const int64_t ib   = blockIdx.z;
    const int64_t ih_kv = ih / (n_head_q / n_head_kv);

    const int64_t n_kt     = (n_kv + BK - 1)/BK;
    const int64_t kt_begin = ic*tiles_per_chunk;
    const int64_t kt_end   = kt_begin + tiles_per_chunk < n_kt ? kt_begin + tiles_per_chunk : n_kt;

    const int64_t iq_w = qt*BQ + warp*16;
    const int64_t iq_r[2] = {iq_w + g, iq_w + g + 8};
    const int64_t q_offset = n_kv - n_q;

    const char * k_base = k + uint64_t(ih_kv)*k_nb2 + uint64_t(ib)*k_nb3;
    const char * v_base = v + uint64_t(ih_kv)*v_nb2 + uint64_t(ib)*v_nb3;
    const char * m_base = mask ? mask + uint64_t(ih % mask_ne2)*m_nb2 + uint64_t(ib % mask_ne3)*m_nb3 : nullptr;
    const uint8_t * f_base = flags ? flags + ((ih % mask_ne2) + mask_ne2*(ib % mask_ne3))*n_qt*n_kt + qt*n_kt : nullptr;
    const char * r_base = rel + uint64_t(ih)*r_nb1 + uint64_t(ib % rel_ne3)*r_nb3;

    uint32_t Q_a[NK][4];
#pragma unroll
    for (int kk = 0; kk < NK; ++kk) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float lo0 = 0.0f, hi0 = 0.0f, lo1 = 0.0f, hi1 = 0.0f;
            if (iq_r[r] < n_q) {
                const float * qr = (const float *) (q + uint64_t(iq_r[r])*q_nb1 + uint64_t(ih)*q_nb2 + uint64_t(ib)*q_nb3);
                lo0 = qr[kk*16 + 2*t + 0];
                hi0 = qr[kk*16 + 2*t + 1];
                lo1 = qr[kk*16 + 2*t + 8];
                hi1 = qr[kk*16 + 2*t + 9];
            }
            Q_a[kk][r + 0] = fattn_banded_pack_h2(lo0, hi0);
            Q_a[kk][r + 2] = fattn_banded_pack_h2(lo1, hi1);
        }
    }

    float O[ND][4];
#pragma unroll
    for (int n = 0; n < ND; ++n) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            O[n][e] = 0.0f;
        }
    }
    float row_max[2] = {-INFINITY, -INFINITY};
    float row_sum[2] = {0.0f, 0.0f};

    auto next_tile = [&](int64_t kt) {
        if (f_base) {
            while (kt < kt_end && f_base[kt] == 0) {
                ++kt;
            }
        }
        return kt;
    };

    int64_t kt = next_tile(kt_begin);
    if (kt < kt_end) {
        fattn_banded_mma_load_tile<D, NW>(K_s, k_base, k_nb1, kt*BK, n_kv);
        fattn_banded_mma_load_tile<D, NW>(V_s, v_base, v_nb1, kt*BK, n_kv);
    }

    while (kt < kt_end) {
        const int64_t kv0  = kt*BK;
        const int     flag = f_base ? f_base[kt] : 1;

        fattn_banded_mma_wait<1>(); // K tile landed, V may still be in flight
        __syncthreads();

        float S[BK/8][4];
#pragma unroll
        for (int j = 0; j < BK/8; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                S[j][e] = 0.0f;
            }
        }
#pragma unroll
        for (int kk = 0; kk < NK; ++kk) {
#pragma unroll
            for (int j = 0; j < BK/8; j += 2) {
                const int mi = lane / 8;
                uint32_t b[4];
                fattn_banded_ldmatrix_x4(b, K_s + ((j + mi/2)*8 + lane % 8)*stride + kk*16 + (mi % 2)*8);
                fattn_banded_mma_16816(S[j + 0], Q_a[kk][0], Q_a[kk][1], Q_a[kk][2], Q_a[kk][3], b[0], b[1]);
                fattn_banded_mma_16816(S[j + 1], Q_a[kk][0], Q_a[kk][1], Q_a[kk][2], Q_a[kk][3], b[2], b[3]);
            }
        }

        const int64_t dist_hi = iq_w + 15 + q_offset - kv0;
        const int64_t dist_lo = iq_w + q_offset - (kv0 + BK - 1);
        const bool in_band = dist_hi >= 0 && dist_lo < rel_extent;

#pragma unroll
        for (int j = 0; j < BK/8; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int     r  = e / 2;
                const int64_t ik = kv0 + j*8 + 2*t + (e % 2);
                float s = S[j][e]*scale;
                if (ik >= n_kv) {
                    s = -INFINITY;
                } else if (iq_r[r] < n_q) {
                    if (in_band) {
                        const int64_t dist = iq_r[r] + q_offset - ik;
                        if (dist >= 0 && dist < rel_extent) {
                            s += fattn_banded_load(r_base + uint64_t(dist)*r_nb0 + uint64_t(iq_r[r])*r_nb2, type_rel);
                        }
                    }
                    if (flag == 2) {
                        s += __half2float(*(const half *) (m_base + uint64_t(iq_r[r])*m_nb1 + uint64_t(ik)*sizeof(half)));
                    }
                }
                S[j][e] = s;
            }
        }

        float alpha[2];
        float msub[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float m = row_max[r];
#pragma unroll
            for (int j = 0; j < BK/8; ++j) {
                m = fmaxf(m, fmaxf(S[j][2*r + 0], S[j][2*r + 1]));
            }
            m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1, WARP_SIZE));
            m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2, WARP_SIZE));
            msub[r]    = m == -INFINITY ? 0.0f : m;
            alpha[r]   = expf(row_max[r] - msub[r]);
            row_max[r] = m;
        }
        uint32_t P_a[BK/8][2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            row_sum[r] *= alpha[r];
        }
#pragma unroll
        for (int j = 0; j < BK/8; ++j) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const float p0 = expf(S[j][2*r + 0] - msub[r]);
                const float p1 = expf(S[j][2*r + 1] - msub[r]);
                row_sum[r] += p0 + p1;
                P_a[j][r] = fattn_banded_pack_h2(p0, p1);
            }
        }
#pragma unroll
        for (int n = 0; n < ND; ++n) {
            O[n][0] *= alpha[0];
            O[n][1] *= alpha[0];
            O[n][2] *= alpha[1];
            O[n][3] *= alpha[1];
        }

        __syncthreads(); // every warp is done with K_s
        const int64_t kt_next = next_tile(kt + 1);
        if (kt_next < kt_end) {
            fattn_banded_mma_load_tile<D, NW>(K_s, k_base, k_nb1, kt_next*BK, n_kv);
            fattn_banded_mma_wait<1>(); // V tile of this iteration landed
        } else {
            fattn_banded_mma_wait<0>();
        }
        __syncthreads();

#pragma unroll
        for (int s = 0; s < BK/16; ++s) {
#pragma unroll
            for (int n = 0; n < ND; n += 2) {
                const int mi = lane / 8;
                uint32_t b[4];
                fattn_banded_ldmatrix_x4_trans(b, V_s + (s*16 + (mi % 2)*8 + lane % 8)*stride + (n + mi/2)*8);
                fattn_banded_mma_16816(O[n + 0], P_a[2*s][0], P_a[2*s][1], P_a[2*s + 1][0], P_a[2*s + 1][1], b[0], b[1]);
                fattn_banded_mma_16816(O[n + 1], P_a[2*s][0], P_a[2*s][1], P_a[2*s + 1][0], P_a[2*s + 1][1], b[2], b[3]);
            }
        }

        __syncthreads(); // every warp is done with V_s
        if (kt_next < kt_end) {
            fattn_banded_mma_load_tile<D, NW>(V_s, v_base, v_nb1, kt_next*BK, n_kv);
        }
        kt = kt_next;
    }

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        row_sum[r] += __shfl_xor_sync(0xffffffff, row_sum[r], 1, WARP_SIZE);
        row_sum[r] += __shfl_xor_sync(0xffffffff, row_sum[r], 2, WARP_SIZE);
    }

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int64_t iq = iq_r[r];
        if (iq >= n_q) {
            continue;
        }
        const int64_t row = (ib*n_q + iq)*n_head_q + ih;
        if (n_chunks == 1) {
            float m   = row_max[r];
            float sum = row_sum[r];
            float f   = 1.0f;
            if (sinks) {
                const float sink = sinks[ih + iq*sinks_s1 + ib*sinks_s2];
                const float m_new = fmaxf(m, sink);
                f   = expf(m - m_new);
                sum = sum*f + expf(sink - m_new);
            }
            const float inv = sum == 0.0f ? 0.0f : f/sum;
            float2 * d = (float2 *) (dst + row*D);
#pragma unroll
            for (int n = 0; n < ND; ++n) {
                d[n*4 + t] = make_float2(O[n][2*r + 0]*inv, O[n][2*r + 1]*inv);
            }
        } else {
            float * p = part + (row*n_chunks + ic)*(D + 2);
            if (t == 0) {
                p[0] = row_max[r];
                p[1] = row_sum[r];
            }
#pragma unroll
            for (int n = 0; n < ND; ++n) {
                p[2 + n*8 + 2*t + 0] = O[n][2*r + 0];
                p[2 + n*8 + 2*t + 1] = O[n][2*r + 1];
            }
        }
    }
#else
    GGML_UNUSED_VARS(q, k, v, mask, flags, rel, sinks, dst, part, scale, type_rel, n_q, n_kv, n_head_q, n_head_kv,
        n_chunks, tiles_per_chunk, rel_extent, mask_ne2, mask_ne3, q_nb1, q_nb2, q_nb3, k_nb1, k_nb2, k_nb3,
        v_nb1, v_nb2, v_nb3, m_nb1, m_nb2, m_nb3, r_nb0, r_nb1, r_nb2, r_nb3, rel_ne3, sinks_s1, sinks_s2);
    NO_DEVICE_CODE;
#endif // FATTN_BANDED_MMA_AVAILABLE
}

static bool fattn_banded_type_supported(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

bool ggml_cuda_flash_attn_ext_banded_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
#if defined(GGML_USE_MUSA)
    GGML_UNUSED(dst);
    return false;
#else
    if (dst->op != GGML_OP_FLASH_ATTN_EXT_BANDED) {
        return false;
    }

    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * v   = dst->src[2];
    const ggml_tensor * m   = dst->src[3];
    const ggml_tensor * rel = dst->src[6];
    if (!q || !k || !v || !rel || q->type != GGML_TYPE_F32) {
        return false;
    }
    if (!fattn_banded_type_supported(k->type) ||
        !fattn_banded_type_supported(v->type) ||
        !fattn_banded_type_supported(rel->type)) {
        return false;
    }
    if ((q->ne[0] != 64 && q->ne[0] != 128) || v->ne[0] != q->ne[0] || k->ne[0] != q->ne[0]) {
        return false;
    }
    if (q->ne[2] % k->ne[2] != 0 || q->ne[2] % v->ne[2] != 0 || k->ne[2] != v->ne[2]) {
        return false;
    }
    if (q->ne[3] != k->ne[3] || q->ne[3] != v->ne[3]) {
        return false;
    }
    if (q->nb[0] != sizeof(float) || k->nb[0] != ggml_type_size(k->type) ||
        v->nb[0] != ggml_type_size(v->type) || rel->nb[0] != ggml_type_size(rel->type)) {
        return false;
    }
    if (rel->ne[0] <= 0 || rel->ne[1] != q->ne[2] || rel->ne[2] != q->ne[1] ||
        (rel->ne[3] != 1 && rel->ne[3] != q->ne[3])) {
        return false;
    }
    const ggml_tensor * sinks = dst->src[4];
    if (sinks && (sinks->type != GGML_TYPE_F32 || sinks->nb[0] != sizeof(float))) {
        return false;
    }
    return !m || (m->type == GGML_TYPE_F16 && ggml_is_contiguous(m) &&
        q->ne[2] % m->ne[2] == 0 && q->ne[3] % m->ne[3] == 0);
#endif
}

static constexpr int64_t fattn_banded_mma_min_q = 16;

static bool fattn_banded_mma_aligned(const ggml_tensor * t) {
    return (uintptr_t) t->data % 16 == 0 && t->nb[1] % 16 == 0 && t->nb[2] % 16 == 0 && t->nb[3] % 16 == 0;
}

static bool fattn_banded_mma_usable(const int device, const ggml_tensor * dst) {
    static const bool disabled = [] {
        const char * env = getenv("GGML_CUDA_DISABLE_BANDED_MMA");
        return env != nullptr && atoi(env) != 0;
    }();
    if (disabled) {
        return false;
    }
    if (!turing_mma_available(ggml_cuda_info().devices[device].cc)) {
        return false;
    }
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    if (q->ne[1] < fattn_banded_mma_min_q) {
        return false;
    }
    for (const ggml_tensor * t : {k, v}) {
        if (t->type == GGML_TYPE_F16 ? !fattn_banded_mma_aligned(t) : t->type != GGML_TYPE_BF16) {
            return false;
        }
    }
    return true;
}

static void fattn_banded_mma_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const float scale, const int64_t rel_extent) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * m     = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * rel   = dst->src[6];
    cudaStream_t stream = ctx.stream();

    const int64_t D    = q->ne[0];
    const int64_t n_q  = q->ne[1];
    const int64_t n_kv = k->ne[1];

    const char * k_data = (const char *) k->data;
    const char * v_data = (const char *) v->data;
    uint64_t k_nb[4] = {k->nb[0], k->nb[1], k->nb[2], k->nb[3]};
    uint64_t v_nb[4] = {v->nb[0], v->nb[1], v->nb[2], v->nb[3]};
    ggml_cuda_pool_alloc<half> k_f16(ctx.pool());
    ggml_cuda_pool_alloc<half> v_f16(ctx.pool());
    auto to_f16 = [&](const ggml_tensor * t, ggml_cuda_pool_alloc<half> & buf, const char *& data, uint64_t (&nb)[4]) {
        if (t->type == GGML_TYPE_F16) {
            return;
        }
        const size_t ts = ggml_type_size(t->type);
        buf.alloc(ggml_nelements(t));
        ggml_get_to_fp16_nc_cuda(t->type)(t->data, buf.get(), t->ne[0], t->ne[1], t->ne[2], t->ne[3],
            t->nb[1]/ts, t->nb[2]/ts, t->nb[3]/ts, stream);
        data  = (const char *) buf.get();
        nb[0] = sizeof(half);
        nb[1] = t->ne[0]*nb[0];
        nb[2] = t->ne[1]*nb[1];
        nb[3] = t->ne[2]*nb[2];
    };
    to_f16(k, k_f16, k_data, k_nb);
    to_f16(v, v_f16, v_data, v_nb);

    const int nw = n_q >= 64 ? 4 : (n_q >= 32 ? 2 : 1);
    const int bq = nw*16;
    const int64_t n_qt = (n_q + bq - 1)/bq;
    const int64_t n_kt = (n_kv + fattn_banded_mma_bk - 1)/fattn_banded_mma_bk;

    ggml_cuda_pool_alloc<uint8_t> flags(ctx.pool());
    if (m) {
        const int64_t n_z = m->ne[2]*m->ne[3];
        flags.alloc(n_z*n_qt*n_kt);
        flash_attn_ext_banded_mma_tile_flags<<<dim3(n_kt, n_qt, n_z), 256, 0, stream>>>(
            (const char *) m->data, flags.get(), n_q, n_kv, bq, m->ne[2], m->nb[1], m->nb[2], m->nb[3]);
    }

    int occ = 0;
#define OCC_BANDED_MMA(DD, NW) CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, flash_attn_ext_banded_mma<DD, NW>, NW*WARP_SIZE, 0))
    if (D == 64) {
        if (nw == 4) { OCC_BANDED_MMA(64, 4); } else if (nw == 2) { OCC_BANDED_MMA(64, 2); } else { OCC_BANDED_MMA(64, 1); }
    } else {
        if (nw == 4) { OCC_BANDED_MMA(128, 4); } else if (nw == 2) { OCC_BANDED_MMA(128, 2); } else { OCC_BANDED_MMA(128, 1); }
    }
#undef OCC_BANDED_MMA
    const int64_t resident    = int64_t(std::max(occ, 1))*ggml_cuda_info().devices[ctx.device].nsm;
    const int64_t base_blocks = n_qt*q->ne[2]*q->ne[3];
    constexpr int64_t min_tiles_per_chunk = 4;
    const int64_t max_chunks  = std::max<int64_t>(1, std::min<int64_t>(n_kt/min_tiles_per_chunk, (4*resident + base_blocks - 1)/base_blocks));
    int64_t n_chunks = 1;
    double  best_eff = 0.0;
    for (int64_t c = 1; c <= max_chunks; ++c) {
        const int64_t blocks = base_blocks*c;
        const double  eff    = double(blocks) / double(((blocks + resident - 1)/resident)*resident);
        if (eff > best_eff + 1e-3) {
            best_eff = eff;
            n_chunks = c;
        }
    }
    const int64_t tiles_per_chunk = (n_kt + n_chunks - 1)/n_chunks;
    n_chunks = (n_kt + tiles_per_chunk - 1)/tiles_per_chunk;

    const int64_t n_rows = n_q*q->ne[2]*q->ne[3];
    ggml_cuda_pool_alloc<float> part(ctx.pool());
    if (n_chunks > 1) {
        part.alloc(n_rows*n_chunks*(D + 2));
    }

    const float * sinks_d  = (const float *) sinks->data;
    const int64_t sinks_s1 = sinks->ne[1] > 1 ? int64_t(sinks->nb[1] / sizeof(float)) : 0;
    const int64_t sinks_s2 = sinks->ne[2] > 1 ? int64_t(sinks->nb[2] / sizeof(float)) : 0;

    const dim3 blocks(n_qt*n_chunks, q->ne[2], q->ne[3]);
#define LAUNCH_BANDED_MMA(DD, NW) \
    flash_attn_ext_banded_mma<DD, NW><<<blocks, NW*WARP_SIZE, 0, stream>>>( \
        (const char *) q->data, k_data, v_data, m ? (const char *) m->data : nullptr, m ? flags.get() : nullptr, \
        (const char *) rel->data, n_chunks == 1 ? sinks_d : nullptr, (float *) dst->data, part.get(), \
        scale, rel->type, n_q, n_kv, q->ne[2], k->ne[2], n_chunks, tiles_per_chunk, rel_extent, \
        m ? m->ne[2] : 1, m ? m->ne[3] : 1, q->nb[1], q->nb[2], q->nb[3], \
        k_nb[1], k_nb[2], k_nb[3], v_nb[1], v_nb[2], v_nb[3], \
        m ? m->nb[1] : 0, m ? m->nb[2] : 0, m ? m->nb[3] : 0, \
        rel->nb[0], rel->nb[1], rel->nb[2], rel->nb[3], rel->ne[3], sinks_s1, sinks_s2)
#define LAUNCH_BANDED_MMA_NW(DD) \
    switch (nw) { \
        case 4:  LAUNCH_BANDED_MMA(DD, 4); break; \
        case 2:  LAUNCH_BANDED_MMA(DD, 2); break; \
        default: LAUNCH_BANDED_MMA(DD, 1); break; \
    }
    if (D == 64) {
        LAUNCH_BANDED_MMA_NW(64);
    } else {
        LAUNCH_BANDED_MMA_NW(128);
    }
#undef LAUNCH_BANDED_MMA_NW
#undef LAUNCH_BANDED_MMA
    CUDA_CHECK(cudaGetLastError());

    if (n_chunks > 1) {
        constexpr int merge_warps = 4;
        const dim3 mblocks((n_rows + merge_warps - 1)/merge_warps, 1, 1);
        const dim3 mthreads(merge_warps*WARP_SIZE, 1, 1);
        if (D == 64) {
            flash_attn_ext_banded_f32_merge<64><<<mblocks, mthreads, 0, stream>>>(
                part.get(), (float *) dst->data, n_rows, q->ne[2], n_q, n_chunks, sinks_d, sinks_s1, sinks_s2);
        } else {
            flash_attn_ext_banded_f32_merge<128><<<mblocks, mthreads, 0, stream>>>(
                part.get(), (float *) dst->data, n_rows, q->ne[2], n_q, n_chunks, sinks_d, sinks_s1, sinks_s2);
        }
    }
}

void ggml_cuda_flash_attn_ext_banded(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_flash_attn_ext_banded_supported(ctx.device, dst));

    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * v   = dst->src[2];
    const ggml_tensor * m   = dst->src[3];
    const ggml_tensor * rel = dst->src[6];
    const ggml_tensor * sinks = dst->src[4];

    // only the MMA kernel reads src[6]; delegating otherwise drops the bias
    if (ggml_cuda_flash_attn_ext_banded_use_mma(ctx.device, dst)) {
        ggml_cuda_flash_attn_ext(ctx, dst);
        return;
    }

    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));
    // the tensor extent (not op_params) is authoritative after graph cloning
    const int64_t rel_extent = rel->ne[0];

    if (sinks && fattn_banded_mma_usable(ctx.device, dst)) {
        fattn_banded_mma_launch(ctx, dst, scale, rel_extent);
        return;
    }

    {
        constexpr int split_warps   = 4;
        constexpr int min_chunk     = 64;
        const int64_t n_rows        = q->ne[1]*q->ne[2]*q->ne[3];
        const int64_t n_kv          = k->ne[1];
        const int64_t target_blocks = int64_t(ggml_cuda_info().devices[ctx.device].nsm)*16;
        int64_t n_chunks = n_rows < target_blocks ? (target_blocks + n_rows - 1)/n_rows : 1;
        n_chunks = std::min<int64_t>(n_chunks, (n_kv + min_chunk - 1)/min_chunk);
        if (n_chunks > 1 && q->ne[1]*q->ne[3] <= 65535) {
            const int64_t chunk = (n_kv + n_chunks - 1)/n_chunks;
            n_chunks = (n_kv + chunk - 1)/chunk;
            const int64_t D = q->ne[0];
            ggml_cuda_pool_alloc<float> part(ctx.pool(), n_rows*n_chunks*(D + 2));
            const dim3 sblocks(n_chunks, q->ne[2], q->ne[1]*q->ne[3]);
            const dim3 sthreads(split_warps*WARP_SIZE, 1, 1);
            constexpr int merge_warps = 4;
            const dim3 mblocks((n_rows + merge_warps - 1)/merge_warps, 1, 1);
            const dim3 mthreads(merge_warps*WARP_SIZE, 1, 1);
            cudaStream_t stream = ctx.stream();
#define LAUNCH_SPLIT(DD) \
            flash_attn_ext_banded_f32_split<DD, split_warps><<<sblocks, sthreads, 0, stream>>>( \
                (const char *) q->data, (const char *) k->data, (const char *) v->data, \
                m ? (const char *) m->data : nullptr, (const char *) rel->data, part.get(), \
                scale, k->type, v->type, rel->type, q->ne[1], n_kv, q->ne[2], k->ne[2], n_chunks, chunk, \
                rel_extent, m ? m->ne[2] : 1, m ? m->ne[3] : 1, \
                q->nb[1], q->nb[2], q->nb[3], k->nb[0], k->nb[1], k->nb[2], k->nb[3], \
                v->nb[0], v->nb[1], v->nb[2], v->nb[3], \
                m ? m->nb[1] : 0, m ? m->nb[2] : 0, m ? m->nb[3] : 0, \
                rel->nb[0], rel->nb[1], rel->nb[2], rel->nb[3], rel->ne[3]); \
            flash_attn_ext_banded_f32_merge<DD><<<mblocks, mthreads, 0, stream>>>( \
                part.get(), (float *) dst->data, n_rows, q->ne[2], q->ne[1], n_chunks, \
                sinks ? (const float *) sinks->data : nullptr, \
                sinks && sinks->ne[1] > 1 ? int64_t(sinks->nb[1] / sizeof(float)) : 0, \
                sinks && sinks->ne[2] > 1 ? int64_t(sinks->nb[2] / sizeof(float)) : 0)
            if (D == 64) {
                LAUNCH_SPLIT(64);
            } else {
                LAUNCH_SPLIT(128);
            }
#undef LAUNCH_SPLIT
            return;
        }
    }

    constexpr int warps_per_block = 4;
    const dim3 blocks((q->ne[1] + warps_per_block - 1) / warps_per_block, q->ne[2], q->ne[3]);
    const dim3 threads(warps_per_block * WARP_SIZE, 1, 1);
    cudaStream_t stream = ctx.stream();

#define LAUNCH_BANDED(D) \
    flash_attn_ext_banded_f32<D, warps_per_block><<<blocks, threads, 0, stream>>>( \
        (const char *) q->data, (const char *) k->data, (const char *) v->data, \
        m ? (const char *) m->data : nullptr, (const char *) rel->data, (float *) dst->data, \
        scale, k->type, v->type, rel->type, q->ne[1], k->ne[1], q->ne[2], k->ne[2], q->ne[3], \
        rel_extent, m ? m->ne[2] : 1, m ? m->ne[3] : 1, \
        q->nb[1], q->nb[2], q->nb[3], k->nb[0], k->nb[1], k->nb[2], k->nb[3], \
        v->nb[0], v->nb[1], v->nb[2], v->nb[3], \
        m ? m->nb[1] : 0, m ? m->nb[2] : 0, m ? m->nb[3] : 0, \
        sinks ? (const float *) sinks->data : nullptr, \
        sinks && sinks->ne[1] > 1 ? int64_t(sinks->nb[1] / sizeof(float)) : 0, \
        sinks && sinks->ne[2] > 1 ? int64_t(sinks->nb[2] / sizeof(float)) : 0, \
        rel->nb[0], rel->nb[1], rel->nb[2], rel->nb[3], rel->ne[3])

    if (q->ne[0] == 64) {
        LAUNCH_BANDED(64);
    } else {
        LAUNCH_BANDED(128);
    }
#undef LAUNCH_BANDED
}
