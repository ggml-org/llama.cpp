#include "moe-weighted-reduction.cuh"

static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  const int64_t n_embd,
                                                  const int     n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    float          sum         = (experts[first_row * n_embd + col] * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        sum += (experts[row * n_embd + col] * scale) * weights[row];
    }
    dst[token * n_embd + col] = sum;
}

static __global__ void moe_weighted_reduction_bias_f32(const float * __restrict__ experts,
                                                       const float * __restrict__ expert_scale,
                                                       const float * __restrict__ weights,
                                                       const float * __restrict__ bias,
                                                       const int32_t * __restrict__ ids,
                                                       const int64_t ids_stride,
                                                       float * __restrict__ dst,
                                                       const int64_t n_embd,
                                                       const int n_expert_used) {
    const int64_t token = blockIdx.x;
    const int64_t col   = (int64_t) blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    const uint64_t first_row   = (uint64_t) token * n_expert_used;
    const float    first_scale = expert_scale != nullptr ? expert_scale[first_row] : 1.0f;
    const float    first_val   = experts[first_row * n_embd + col] + bias[(int64_t) ids[token * ids_stride] * n_embd + col];
    float          sum         = (first_val * first_scale) * weights[first_row];

    for (int expert = 1; expert < n_expert_used; ++expert) {
        const uint64_t row   = first_row + expert;
        const float   scale = expert_scale != nullptr ? expert_scale[row] : 1.0f;
        const float   val   = experts[row * n_embd + col] + bias[(int64_t) ids[token * ids_stride + expert] * n_embd + col];
        sum += (val * scale) * weights[row];
    }
    dst[token * n_embd + col] = sum;
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
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, dst, n_embd, n_expert_used);
}

static void launch_moe_weighted_reduction_bias(const float *   experts,
                                               const float *   expert_scale,
                                               const float *   weights,
                                               const float *   bias,
                                               const int32_t * ids,
                                               int64_t         ids_stride,
                                               float *         dst,
                                               int64_t         n_embd,
                                               int64_t         n_tokens,
                                               int             n_expert_used,
                                               cudaStream_t    stream) {
    constexpr int threads = 256;
    const dim3 blocks(n_tokens, (n_embd + threads - 1) / threads, 1);
    moe_weighted_reduction_bias_f32
        <<<blocks, threads, 0, stream>>>(experts, expert_scale, weights, bias, ids, ids_stride, dst, n_embd, n_expert_used);
}

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         const ggml_tensor *         bias,
                                         const ggml_tensor *         ids,
                                         ggml_tensor *               dst) {
    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(expert_scale == nullptr || expert_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(expert_scale == nullptr || ggml_is_contiguous(expert_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT((bias == nullptr) == (ids == nullptr));
    GGML_ASSERT(bias == nullptr || (bias->type == GGML_TYPE_F32 && ggml_is_contiguous(bias)));
    // Rows may have padding, as in add_id_kernel.
    GGML_ASSERT(ids == nullptr || (ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t)));

    const int64_t n_embd        = experts->ne[0];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t n_tokens      = experts->ne[2] * experts->ne[3];
    const int64_t ids_stride    = ids ? ids->nb[1] / sizeof(int32_t) : 0;
    cudaStream_t  stream        = ctx.stream();

    if (bias) {
        launch_moe_weighted_reduction_bias((const float *) experts->data,
                                           expert_scale ? (const float *) expert_scale->data : nullptr,
                                           (const float *) weights->data,
                                           (const float *) bias->data,
                                           (const int32_t *) ids->data,
                                           ids_stride,
                                           (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    } else {
        launch_moe_weighted_reduction((const float *) experts->data,
                                      expert_scale ? (const float *) expert_scale->data : nullptr,
                                      (const float *) weights->data,
                                      (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}
