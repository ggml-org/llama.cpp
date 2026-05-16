#include "dsv4-hc-weighted-sum.cuh"

// CUDA port of kernel_dsv4_hc_weighted_sum (ggml-metal.metal:2278-2327).
//
// Layout (FP32 throughout):
//   x       : {n_embd, n_hc,    n_tokens}
//   weights : {n_hc,   n_tokens}
//   dst     : {n_embd, n_tokens}
// Output[d, t] = sum_{h=0..n_hc-1} x[d, h, t] * weights[h, t].
//
// One thread per output element. Total threads = n_embd * n_tokens.
// Strides are passed in BYTES (matching ggml's nb[] convention); element
// access is via `(const char *) base + d*nb0 + h*nb1 + t*nb2` reinterpret
// as `const float *`, identical to the Metal kernel and CPU reference.

static __global__ void dsv4_hc_weighted_sum_f32(
        const char * __restrict__ x,
        const char * __restrict__ weights,
              char * __restrict__ dst,
        const int      n_embd,
        const int      n_hc,
        const int      n_tokens,
        const int64_t  nb_x0,
        const int64_t  nb_x1,
        const int64_t  nb_x2,
        const int64_t  nb_w0,
        const int64_t  nb_w1,
        const int64_t  nb0,
        const int64_t  nb1) {
    const int64_t gid   = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = (int64_t) n_embd * n_tokens;
    if (gid >= total) {
        return;
    }

    const int64_t d = gid % n_embd;
    const int64_t t = gid / n_embd;

    float acc = 0.0f;
    for (int h = 0; h < n_hc; ++h) {
        const float xv = *((const float *) (x       + d*nb_x0 + h*nb_x1 + t*nb_x2));
        const float wv = *((const float *) (weights + h*nb_w0 + t*nb_w1));
        acc += xv * wv;
    }

    *((float *) (dst + d*nb0 + t*nb1)) = acc;
}

void ggml_cuda_op_dsv4_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type       == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type     == GGML_TYPE_F32);

    // Shape contract: see ggml.c:6335-6339 and the CPU reference asserts at
    // ggml-cpu/ops.cpp:11129-11140.
    GGML_ASSERT(x->ne[0]       == dst->ne[0]);
    GGML_ASSERT(x->ne[1]       == weights->ne[0]);
    GGML_ASSERT(x->ne[2]       == dst->ne[1]);
    GGML_ASSERT(weights->ne[1] == dst->ne[1]);
    GGML_ASSERT(x->ne[3]       == 1);
    GGML_ASSERT(weights->ne[2] == 1);
    GGML_ASSERT(weights->ne[3] == 1);
    GGML_ASSERT(dst->ne[2]     == 1);
    GGML_ASSERT(dst->ne[3]     == 1);

    const int n_embd   = (int) dst->ne[0];
    const int n_hc     = (int) x->ne[1];
    const int n_tokens = (int) dst->ne[1];

    const int64_t nb_x0 = (int64_t) x->nb[0];
    const int64_t nb_x1 = (int64_t) x->nb[1];
    const int64_t nb_x2 = (int64_t) x->nb[2];
    const int64_t nb_w0 = (int64_t) weights->nb[0];
    const int64_t nb_w1 = (int64_t) weights->nb[1];
    const int64_t nb0   = (int64_t) dst->nb[0];
    const int64_t nb1   = (int64_t) dst->nb[1];

    const int64_t total = (int64_t) n_embd * n_tokens;
    if (total == 0) {
        return;
    }

    constexpr int CUDA_DSV4_HC_WEIGHTED_SUM_BLOCK_SIZE = 256;
    const dim3 block_dims(CUDA_DSV4_HC_WEIGHTED_SUM_BLOCK_SIZE, 1, 1);
    const dim3 grid_dims((unsigned) ((total + CUDA_DSV4_HC_WEIGHTED_SUM_BLOCK_SIZE - 1) /
                                     CUDA_DSV4_HC_WEIGHTED_SUM_BLOCK_SIZE),
                         1, 1);

    cudaStream_t stream = ctx.stream();

    dsv4_hc_weighted_sum_f32<<<grid_dims, block_dims, 0, stream>>>(
        (const char *) x->data,
        (const char *) weights->data,
              (char *) dst->data,
        n_embd, n_hc, n_tokens,
        nb_x0, nb_x1, nb_x2,
        nb_w0, nb_w1,
        nb0, nb1);

    CUDA_CHECK(cudaGetLastError());
}
