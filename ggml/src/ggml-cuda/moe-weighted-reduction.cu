#include "moe-weighted-reduction.cuh"
#include "convert.cuh"

// T: type of the experts and of dst (float or nv_bfloat16), the sum is F32
template <typename T>
static __global__ void moe_weighted_reduction_f32(const T *     __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  T *           __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (ggml_cuda_cast<float>(experts[first_row * n_embd + col]) * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (ggml_cuda_cast<float>(experts[row * n_embd + col]) * scale) * weights[row];
    }
    dst[token * n_embd + col] = ggml_cuda_cast<T>(sum);
}

template <typename T>
static void launch_moe_weighted_reduction(const T *     experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          T *           dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          int           n_expert_used,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32<T>
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32 || experts->type == GGML_TYPE_BF16);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == experts->type);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    cudaStream_t  stream        = ctx.stream();

    const float * expert_scale_d = expert_scale ? (const float *) expert_scale->data : nullptr;
    if (experts->type == GGML_TYPE_BF16) {
        launch_moe_weighted_reduction((const nv_bfloat16 *) experts->data, expert_scale_d, (const float *) weights->data,
                                      (nv_bfloat16 *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    } else {
        launch_moe_weighted_reduction((const float *) experts->data, expert_scale_d, (const float *) weights->data,
                                      (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}
