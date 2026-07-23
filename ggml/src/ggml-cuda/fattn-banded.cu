#include "common.cuh"
#include "fattn-banded.cuh"
#include "fattn.cuh"

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

    const float inv_sum = row_sum == 0.0f ? 0.0f : 1.0f/row_sum;
    float * dst_row = dst + ((ib*n_q + iq)*n_head_q + ih)*D;
#pragma unroll
    for (int j = 0; j < values_per_lane; ++j) {
        const int d = lane + j*WARP_SIZE;
        dst_row[d] = out[j]*inv_sum;
    }
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
    const ggml_tensor * rel = dst->src[5];
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
    return !m || (m->type == GGML_TYPE_F16 && ggml_is_contiguous(m) &&
        q->ne[2] % m->ne[2] == 0 && q->ne[3] % m->ne[3] == 0);
#endif
}

void ggml_cuda_flash_attn_ext_banded(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_flash_attn_ext_banded_supported(ctx.device, dst));

    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * v   = dst->src[2];
    const ggml_tensor * m   = dst->src[3];
    const ggml_tensor * rel = dst->src[5];

    // route F16/BF16 K/V to the MMA kernel; keep this FP32 kernel for mixed types and strided rel
    if (k->type != GGML_TYPE_F32 && v->type != GGML_TYPE_F32 &&
        rel->type == GGML_TYPE_F32 && ggml_is_contiguous(rel) &&
        // MMA ABI indexes rel by Q's batch: a singleton rel batch must take the stride-aware fallback
        rel->ne[3] == q->ne[3] && rel->ne[0] <= (1 << 20)) {
        ggml_cuda_flash_attn_ext(ctx, dst);
        return;
    }

    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));
    // the tensor extent (not op_params) is authoritative after graph cloning
    const int64_t rel_extent = rel->ne[0];

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
        rel->nb[0], rel->nb[1], rel->nb[2], rel->nb[3], rel->ne[3])

    if (q->ne[0] == 64) {
        LAUNCH_BANDED(64);
    } else {
        LAUNCH_BANDED(128);
    }
#undef LAUNCH_BANDED
}
