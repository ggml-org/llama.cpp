#include "moe-weighted-reduction.cuh"

#include <climits>

static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  int64_t n_embd,
                                                  int64_t n_tokens,
                                                  int     n_expert_used) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = n_embd * n_tokens;
    if (index >= total) {
        return;
    }

    const int64_t token     = index / n_embd;
    const int64_t col       = index - token * n_embd;
    const int64_t first_row = token * n_expert_used;
    const float   first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float         sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const int64_t row   = token * n_expert_used + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }
    dst[index] = sum;
}

static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    const int64_t blocks = (n_embd * n_tokens + threads - 1) / threads;
    GGML_ASSERT(blocks <= INT_MAX);
    moe_weighted_reduction_f32
        <<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
            experts, expert_scale, weights, dst, n_embd, n_tokens, n_expert_used);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    const float * weights_data      = (const float *) weights->data;
    const float * expert_scale_data = expert_scale ? (const float *) expert_scale->data : nullptr;
    const uintptr_t weights_begin = (uintptr_t) weights->data;
    const uintptr_t weights_end   = weights_begin + ggml_nbytes(weights);
    const uintptr_t dst_begin     = (uintptr_t) dst->data;
    const uintptr_t dst_end       = dst_begin + ggml_nbytes(dst);
    ggml_cuda_pool_alloc<float> weights_copy(ctx.pool());
    if (weights_begin < dst_end && dst_begin < weights_end) {
        // The graph allocator may reuse weights for dst after the original MUL. Fusion reads both at once.
        weights_data = weights_copy.alloc(ggml_nelements(weights));
        CUDA_CHECK(cudaMemcpyAsync((void *) weights_data, weights->data, ggml_nbytes(weights),
                                   cudaMemcpyDeviceToDevice, stream));
    }

    ggml_cuda_pool_alloc<float> expert_scale_copy(ctx.pool());
    if (expert_scale != nullptr) {
        const uintptr_t scale_begin = (uintptr_t) expert_scale->data;
        const uintptr_t scale_end   = scale_begin + ggml_nbytes(expert_scale);
        if (scale_begin < dst_end && dst_begin < scale_end) {
            expert_scale_data = expert_scale_copy.alloc(ggml_nelements(expert_scale));
            CUDA_CHECK(cudaMemcpyAsync((void *) expert_scale_data, expert_scale->data, ggml_nbytes(expert_scale),
                                       cudaMemcpyDeviceToDevice, stream));
        }
    }

    launch_moe_weighted_reduction((const float *) experts->data, expert_scale_data, weights_data,
                                  (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    CUDA_CHECK(cudaGetLastError());
}
