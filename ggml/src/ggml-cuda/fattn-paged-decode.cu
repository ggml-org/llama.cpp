#include "fattn-paged-decode.cuh"

#if defined(GGML_USE_HIP)

#include "fattn-paged-decode-kernel.cuh"

#include <cstdlib>

static bool pd_disabled() {
    static const bool disabled = getenv("GGML_CUDA_DISABLE_PAGED_DECODE") != nullptr;
    return disabled;
}

bool ggml_cuda_fattn_paged_decode_enabled(const ggml_tensor * dst) {
    if (pd_disabled()) {
        return false;
    }
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    float params[3];
    memcpy(params, dst->op_params, sizeof(params)); // scale, max_bias, logit_softcap
    if (!GGML_CUDA_CC_IS_RDNA4(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) return false;
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_Q8_0 || V->type != GGML_TYPE_Q4_0) return false;
    if (Q->ne[0] != PD_D || K->ne[0] != PD_D || V->ne[0] != PD_D) return false;
    if (Q->ne[1] < 1 || Q->ne[1] > 4 || Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1) return false;
    if (K->ne[2] < 1 || Q->ne[2] != PD_GQA * K->ne[2] || V->ne[2] != K->ne[2] || V->ne[1] != K->ne[1]) return false;
    if (dst->src[4] || params[1] != 0.0f || params[2] != 0.0f) return false;
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->ne[3] != 1)) return false;
    if (K->nb[1] % 16 || K->nb[2] % 16 || V->nb[1] % 16 || V->nb[2] % 16) return false;
    if (Q->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) return false;
    return true;
}

void ggml_cuda_flash_attn_ext_paged_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    cudaStream_t stream = ctx.stream();

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    const int n_kv = (int) K->ne[1];
    const int nb = (int) Q->ne[1];
    const int n_heads = (int) K->ne[2];
    const int M_TILES = nb * PD_GQA <= 16 ? 1 : 2;
    const int R = M_TILES * 16;
    const int n_chunks = (n_kv + PD_CHUNK - 1) / PD_CHUNK;

    // One resident wave of blocks: three 8-wave blocks per WGP (nsm counts WGPs on RDNA4). More splits add partial
    // traffic, fewer leave WGPs idle; this was fastest from 8k to 262k tokens.
    const int want = 3 * ggml_cuda_info().devices[ctx.device].nsm;
    int n_splits = std::max(1, std::min(n_chunks, (want + n_heads - 1) / n_heads));
    const int chunks_per_split = (n_chunks + n_splits - 1) / n_splits;
    n_splits = (n_chunks + chunks_per_split - 1) / chunks_per_split;

    // Contiguous KV is the identity page table.
    const pd_kv kv_k = { (const char *) K->data, nullptr, (int64_t) K->nb[1], (int64_t) K->nb[2] };
    const pd_kv kv_v = { (const char *) V->data, nullptr, (int64_t) V->nb[1], (int64_t) V->nb[2] };

    ggml_cuda_pool_alloc<float>  O_part(ctx.pool(), (size_t) n_splits * n_heads * R * PD_D);
    ggml_cuda_pool_alloc<float2> ML_part(ctx.pool(), (size_t) n_splits * n_heads * R);

    const dim3 grid(n_splits, n_heads);
    const char * mask_data = mask ? (const char *) mask->data : nullptr;
    const int64_t mask_nb1 = mask ? mask->nb[1] : 0;
    if (M_TILES == 1) {
        pd_partial<1, false><<<grid, PD_NWARPS * 32, 0, stream>>>((const char *) Q->data, Q->nb[1], Q->nb[2], kv_k, kv_v, pd_ring{},
            mask_data, mask_nb1, scale, n_kv, nb, n_chunks, chunks_per_split, O_part.get(), ML_part.get(), 0);
    } else {
        pd_partial<2, false><<<grid, PD_NWARPS * 32, 0, stream>>>((const char *) Q->data, Q->nb[1], Q->nb[2], kv_k, kv_v, pd_ring{},
            mask_data, mask_nb1, scale, n_kv, nb, n_chunks, chunks_per_split, O_part.get(), ML_part.get(), 0);
    }
    CUDA_CHECK(cudaGetLastError());
    pd_combine<<<dim3(nb * PD_GQA, n_heads), PD_D, 0, stream>>>(O_part.get(), ML_part.get(), n_splits, R,
        (char *) dst->data, dst->nb[1], dst->nb[2], pd_ring{});
    CUDA_CHECK(cudaGetLastError());
}

#else

bool ggml_cuda_fattn_paged_decode_enabled(const ggml_tensor * dst) {
    GGML_UNUSED(dst);
    return false;
}

void ggml_cuda_flash_attn_ext_paged_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, dst);
    GGML_ABORT("paged decode attention is only built for HIP");
}

#endif // defined(GGML_USE_HIP)
