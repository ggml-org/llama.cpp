#include "common.cuh"
#include "fattn-wmma-dpart.cuh"

// Shallower KV or a single token: the existing kernels are as fast or faster.
static constexpr int64_t ggml_cuda_fattn_wmma_dpart_min_kv = 8192;
static constexpr int64_t ggml_cuda_fattn_wmma_dpart_min_nb = 2;

bool ggml_cuda_fattn_wmma_dpart_supported(const int cc, const ggml_tensor * dst, const bool gqa_opt) {
    // gqa_opt: a mask, no ALiBi, K/V padded to FATTN_KQ_STRIDE, every stride a multiple of 16 bytes
    if (!GGML_CUDA_CC_IS_RDNA3(cc) || !gqa_opt) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    if (mask == nullptr || sinks != nullptr) {
        return false;
    }
    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        return false;
    }
    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_F16 ||
            dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256 || Q->ne[3] != 1 || K->ne[3] != 1 || V->ne[3] != 1) {
        return false;
    }
    // GQA ratios that are not a power of 2 (the other kernels stream such a KV head more than once); all rows of a KV head
    // in one block: gqa_ratio*ne01 <= nrows
    const int64_t gqa_ratio = Q->ne[2] / K->ne[2];
    if (gqa_ratio < 3 || gqa_ratio > 7 || gqa_ratio == 4) {
        return false;
    }
    if (Q->ne[1] < ggml_cuda_fattn_wmma_dpart_min_nb || Q->ne[1]*gqa_ratio > ggml_cuda_fattn_wmma_dpart_nrows ||
            K->ne[1] < ggml_cuda_fattn_wmma_dpart_min_kv || K->ne[1] % ggml_cuda_fattn_wmma_dpart_nbatch_fa != 0 ||
            V->ne[1] != K->ne[1]) {
        return false;
    }
    // one mask for all heads, 16-byte loads of the Q, K, V and mask rows
    if (mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1] || mask->ne[2] != 1 || mask->ne[3] != 1) {
        return false;
    }
    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if ((uintptr_t) t->data % 16 != 0) {
            return false;
        }
    }
    return Q->nb[0] == sizeof(float) && K->nb[0] == sizeof(half) && V->nb[0] == sizeof(half) && ggml_is_contiguous(dst);
}

void ggml_cuda_flash_attn_ext_wmma_dpart(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    GGML_ASSERT(Q->ne[0] == 256 && K->ne[0] == 256 && V->ne[0] == 256);
    GGML_ASSERT(Q->ne[1]*(Q->ne[2]/K->ne[2]) <= ggml_cuda_fattn_wmma_dpart_nrows && Q->ne[2]/K->ne[2] <= 8 && Q->ne[3] == 1);

    constexpr int DKQ       = 256;
    constexpr int DV        = 256;
    constexpr int nwarps    = ggml_cuda_fattn_wmma_dpart_nwarps;
    constexpr int nbatch_fa = ggml_cuda_fattn_wmma_dpart_nbatch_fa;

    // launch_fattn sizes the grid from ncols1 x ncols2: ncols1 = nrows >= ne01 and ncols2 = 8 >= gqa_ratio give one block per
    // KV head and KV split (blockIdx.x = 0, blockIdx.z = KV head). No stream-k: gridDim.y blocks over interleaved nbatch_fa
    // chunks, then flash_attn_combine_results. SRAM is static, K/V are f16 already.
    fattn_kernel_t fattn_kernel = flash_attn_ext_wmma_dpart<DKQ, DV>;
    launch_fattn<DV, ggml_cuda_fattn_wmma_dpart_nrows, 8>(ctx, dst, fattn_kernel, nwarps, 0, nbatch_fa,
        /*need_f16_K =*/ false, /*need_f16_V =*/ false, /*stream_k =*/ false, /*use_sparse =*/ false, /*warp_size =*/ 32);
}
