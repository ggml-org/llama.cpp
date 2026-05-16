#include "dsv4-hc-expand.cuh"

// out[i_embd, i_hc, i_tok] = post[i_hc, i_tok] * block_out[i_embd, i_tok]
//                          + sum_{hc'} comb[i_hc, hc', i_tok] * residual[i_embd, hc', i_tok]
//
// block_out is 2D (no hc axis); the post*block_out term is broadcast across hc.
// See ggml/src/ggml-cpu/ops.cpp:11218-11231 for the CPU reference loop body.
static __global__ void dsv4_hc_expand_f32(
        const float * __restrict__ block_out,
        const float * __restrict__ residual,
        const float * __restrict__ post,
        const float * __restrict__ comb,
        float       * __restrict__ dst,
        const int n_embd, const int n_hc, const int n_tokens,
        // block_out strides (2D -- no hc axis)
        const int nb_b0, const int nb_b1,
        // residual strides (3D)
        const int nb_r0, const int nb_r1, const int nb_r2,
        // post strides (2D)
        const int nb_p0, const int nb_p1,
        // comb strides (3D)
        const int nb_c0, const int nb_c1, const int nb_c2,
        // dst strides (3D)
        const int nb0,   const int nb1,   const int nb2) {
    const int64_t gid   = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t)n_embd * n_hc * n_tokens;
    if (gid >= total) {
        return;
    }

    const int i_embd = gid % n_embd;
    const int rest   = gid / n_embd;
    const int i_hc   = rest % n_hc;
    const int i_tok  = rest / n_hc;

    // post * block_out  (block_out is 2D: indexed by (i_embd, i_tok) only)
    const float p = *(const float *)((const char *)post
        + i_hc * nb_p0 + i_tok * nb_p1);
    const float b = *(const float *)((const char *)block_out
        + i_embd * nb_b0 + i_tok * nb_b1);
    float acc = p * b;

    // comb @ residual: sum over hc'
    for (int hc_p = 0; hc_p < n_hc; ++hc_p) {
        const float c = *(const float *)((const char *)comb
            + i_hc * nb_c0 + hc_p * nb_c1 + i_tok * nb_c2);
        const float r = *(const float *)((const char *)residual
            + i_embd * nb_r0 + hc_p * nb_r1 + i_tok * nb_r2);
        acc += c * r;
    }

    float * d = (float *)((char *)dst
        + i_embd * nb0 + i_hc * nb1 + i_tok * nb2);
    *d = acc;
}

void ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * block_out = dst->src[0];
    const ggml_tensor * residual  = dst->src[1];
    const ggml_tensor * post      = dst->src[2];
    const ggml_tensor * comb      = dst->src[3];

    GGML_ASSERT(block_out->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type  == GGML_TYPE_F32);
    GGML_ASSERT(post->type      == GGML_TYPE_F32);
    GGML_ASSERT(comb->type      == GGML_TYPE_F32);
    GGML_ASSERT(dst->type       == GGML_TYPE_F32);

    const int n_embd   = (int) dst->ne[0];
    const int n_hc     = (int) dst->ne[1];
    const int n_tokens = (int) dst->ne[2];

    const int64_t total = (int64_t)n_embd * n_hc * n_tokens;
    constexpr int blk   = 256;
    const dim3 grid((total + blk - 1) / blk);
    const dim3 block(blk);

    cudaStream_t stream = ctx.stream();
    dsv4_hc_expand_f32<<<grid, block, 0, stream>>>(
        (const float *) block_out->data,
        (const float *) residual->data,
        (const float *) post->data,
        (const float *) comb->data,
        (float *)       dst->data,
        n_embd, n_hc, n_tokens,
        (int) block_out->nb[0], (int) block_out->nb[1],
        (int) residual->nb[0],  (int) residual->nb[1],  (int) residual->nb[2],
        (int) post->nb[0],      (int) post->nb[1],
        (int) comb->nb[0],      (int) comb->nb[1],      (int) comb->nb[2],
        (int) dst->nb[0],       (int) dst->nb[1],       (int) dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
}
