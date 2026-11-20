#include "penalties.cuh"

#define CUDA_PENALTIES_BLOCK_SIZE 256

static __global__ void k_penalties_fused(
        float * __restrict__ logits, const int32_t * __restrict__ ids, const float * __restrict__ coefs,
        const int n, const int64_t s1) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const float pos_scale = coefs[0*n + i];
    const float neg_scale = coefs[1*n + i];
    const float offset    = coefs[2*n + i];

    // padding entries are the identity and share one token id, skip them so each logit has one writer
    if (pos_scale == 1.0f && neg_scale == 1.0f && offset == 0.0f) {
        return;
    }

    float * dst = logits + ids[i]*s1;
    const float x = *dst;

    // no contraction, match the unfused mul -> sub
    *dst = __fsub_rn(__fmul_rn(x, x > 0.0f ? pos_scale : neg_scale), offset);
}

void ggml_cuda_op_penalties_fused(ggml_backend_cuda_context & ctx,
                                  const ggml_tensor * ids,
                                  const ggml_tensor * coefs,
                                  ggml_tensor *       dst) {
    GGML_ASSERT(ids->type   == GGML_TYPE_I32);
    GGML_ASSERT(coefs->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);

    const int     n  = (int) ids->ne[0];
    const int64_t s1 = dst->nb[1] / sizeof(float);

    const int n_blocks = (n + CUDA_PENALTIES_BLOCK_SIZE - 1) / CUDA_PENALTIES_BLOCK_SIZE;

    k_penalties_fused<<<n_blocks, CUDA_PENALTIES_BLOCK_SIZE, 0, ctx.stream()>>>(
            (float *) dst->data, (const int32_t *) ids->data, (const float *) coefs->data, n, s1);
}
