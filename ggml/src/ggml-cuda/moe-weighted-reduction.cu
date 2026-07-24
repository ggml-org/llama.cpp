#include "moe-weighted-reduction.cuh"

#include <climits>

template <int n_expert_used, bool has_expert_scale>
static __global__ void moe_weighted_reduction_f32(const float * __restrict__ experts,
                                                  const float * __restrict__ expert_scale,
                                                  const float * __restrict__ weights,
                                                  float * __restrict__ dst,
                                                  int64_t n_embd,
                                                  int64_t n_tokens) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = n_embd * n_tokens;
    if (index >= total) {
        return;
    }

    const int64_t token = index / n_embd;
    const int64_t col   = index - token * n_embd;
    const int64_t first_row = token * n_expert_used;
    float         value     = experts[first_row * n_embd + col];
    if constexpr (has_expert_scale) {
        value = __fmul_rn(value, expert_scale[first_row]);
    }
    float sum = __fmul_rn(value, weights[first_row]);

#pragma unroll
    for (int expert = 1; expert < n_expert_used; ++expert) {
        const int64_t row     = token * n_expert_used + expert;
        float         value   = experts[row * n_embd + col];
        if constexpr (has_expert_scale) {
            value = __fmul_rn(value, expert_scale[row]);
        }
        const float   product = __fmul_rn(value, weights[row]);
        sum                   = __fadd_rn(sum, product);
    }
    dst[index] = sum;
}

template <int n_expert_used, bool has_expert_scale>
static __global__ void moe_weighted_reduction_f32x4(const float4 * __restrict__ experts,
                                                    const float * __restrict__ expert_scale,
                                                    const float * __restrict__ weights,
                                                    float4 * __restrict__ dst,
                                                    int64_t n_embd4,
                                                    int64_t n_tokens) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = n_embd4 * n_tokens;
    if (index >= total) {
        return;
    }

    const int64_t token        = index / n_embd4;
    const int64_t col4         = index - token * n_embd4;
    const int64_t first_row    = token * n_expert_used;
    float4        first        = experts[first_row * n_embd4 + col4];
    if constexpr (has_expert_scale) {
        const float scale = expert_scale[first_row];
        first.x = __fmul_rn(first.x, scale);
        first.y = __fmul_rn(first.y, scale);
        first.z = __fmul_rn(first.z, scale);
        first.w = __fmul_rn(first.w, scale);
    }
    const float   first_weight = weights[first_row];
    float4        sum          = make_float4(__fmul_rn(first.x, first_weight), __fmul_rn(first.y, first_weight),
                                             __fmul_rn(first.z, first_weight), __fmul_rn(first.w, first_weight));

#pragma unroll
    for (int expert = 1; expert < n_expert_used; ++expert) {
        const int64_t row    = token * n_expert_used + expert;
        float4        value  = experts[row * n_embd4 + col4];
        if constexpr (has_expert_scale) {
            const float scale = expert_scale[row];
            value.x = __fmul_rn(value.x, scale);
            value.y = __fmul_rn(value.y, scale);
            value.z = __fmul_rn(value.z, scale);
            value.w = __fmul_rn(value.w, scale);
        }
        const float   weight = weights[row];
        sum.x                = __fadd_rn(sum.x, __fmul_rn(value.x, weight));
        sum.y                = __fadd_rn(sum.y, __fmul_rn(value.y, weight));
        sum.z                = __fadd_rn(sum.z, __fmul_rn(value.z, weight));
        sum.w                = __fadd_rn(sum.w, __fmul_rn(value.w, weight));
    }
    dst[index] = sum;
}

template <int n_expert_used, bool has_expert_scale>
static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          cudaStream_t  stream) {
    constexpr int threads = 256;
    const bool aligned_f32x4 = (uintptr_t) experts % alignof(float4) == 0 &&
                               (uintptr_t) dst     % alignof(float4) == 0;
    if (n_embd % 4 == 0 && aligned_f32x4) {
        const int64_t n_embd4 = n_embd / 4;
        const int64_t blocks  = (n_embd4 * n_tokens + threads - 1) / threads;
        GGML_ASSERT(blocks <= INT_MAX);
        moe_weighted_reduction_f32x4<n_expert_used, has_expert_scale>
            <<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
                (const float4 *) experts, expert_scale, weights, (float4 *) dst, n_embd4, n_tokens);
    } else {
        const int64_t blocks = (n_embd * n_tokens + threads - 1) / threads;
        GGML_ASSERT(blocks <= INT_MAX);
        moe_weighted_reduction_f32<n_expert_used, has_expert_scale>
            <<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
                experts, expert_scale, weights, dst, n_embd, n_tokens);
    }
}

template <int n_expert_used>
static void launch_moe_weighted_reduction(const float * experts,
                                          const float * expert_scale,
                                          const float * weights,
                                          float *       dst,
                                          int64_t       n_embd,
                                          int64_t       n_tokens,
                                          cudaStream_t  stream) {
    if (expert_scale != nullptr) {
        launch_moe_weighted_reduction<n_expert_used, true>(
            experts, expert_scale, weights, dst, n_embd, n_tokens, stream);
    } else {
        launch_moe_weighted_reduction<n_expert_used, false>(
            experts, nullptr, weights, dst, n_embd, n_tokens, stream);
    }
}

// --- Runtime-k path ---
// A single kernel that takes the expert count as a runtime argument, used only when
// n_expert_used > 8 (beyond the per-k templated kernels above). The arithmetic and the
// __fmul_rn/__fadd_rn ordering are identical to the templated kernels, so the result is
// bit-identical. The hot small-k cases keep their fully-unrolled kernels.
template <bool has_expert_scale>
static __global__ void moe_weighted_reduction_f32_dynk(const float * __restrict__ experts,
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
    float         value     = experts[first_row * n_embd + col];
    if constexpr (has_expert_scale) {
        value = __fmul_rn(value, expert_scale[first_row]);
    }
    float sum = __fmul_rn(value, weights[first_row]);

    for (int expert = 1; expert < n_expert_used; ++expert) {   // runtime trip count -> not unrolled
        const int64_t row = token * n_expert_used + expert;
        float         v   = experts[row * n_embd + col];
        if constexpr (has_expert_scale) {
            v = __fmul_rn(v, expert_scale[row]);
        }
        sum = __fadd_rn(sum, __fmul_rn(v, weights[row]));
    }
    dst[index] = sum;
}

template <bool has_expert_scale>
static __global__ void moe_weighted_reduction_f32x4_dynk(const float4 * __restrict__ experts,
                                                         const float * __restrict__ expert_scale,
                                                         const float * __restrict__ weights,
                                                         float4 * __restrict__ dst,
                                                         int64_t n_embd4,
                                                         int64_t n_tokens,
                                                         int     n_expert_used) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = n_embd4 * n_tokens;
    if (index >= total) {
        return;
    }

    const int64_t token     = index / n_embd4;
    const int64_t col4      = index - token * n_embd4;
    const int64_t first_row = token * n_expert_used;
    float4        first     = experts[first_row * n_embd4 + col4];
    if constexpr (has_expert_scale) {
        const float scale = expert_scale[first_row];
        first.x = __fmul_rn(first.x, scale);
        first.y = __fmul_rn(first.y, scale);
        first.z = __fmul_rn(first.z, scale);
        first.w = __fmul_rn(first.w, scale);
    }
    const float first_weight = weights[first_row];
    float4      sum          = make_float4(__fmul_rn(first.x, first_weight), __fmul_rn(first.y, first_weight),
                                           __fmul_rn(first.z, first_weight), __fmul_rn(first.w, first_weight));

    for (int expert = 1; expert < n_expert_used; ++expert) {   // runtime trip count -> not unrolled
        const int64_t row   = token * n_expert_used + expert;
        float4        value = experts[row * n_embd4 + col4];
        if constexpr (has_expert_scale) {
            const float scale = expert_scale[row];
            value.x = __fmul_rn(value.x, scale);
            value.y = __fmul_rn(value.y, scale);
            value.z = __fmul_rn(value.z, scale);
            value.w = __fmul_rn(value.w, scale);
        }
        const float weight = weights[row];
        sum.x = __fadd_rn(sum.x, __fmul_rn(value.x, weight));
        sum.y = __fadd_rn(sum.y, __fmul_rn(value.y, weight));
        sum.z = __fadd_rn(sum.z, __fmul_rn(value.z, weight));
        sum.w = __fadd_rn(sum.w, __fmul_rn(value.w, weight));
    }
    dst[index] = sum;
}

template <bool has_expert_scale>
static void launch_moe_weighted_reduction_dynk(const float * experts,
                                               const float * expert_scale,
                                               const float * weights,
                                               float *       dst,
                                               int64_t       n_embd,
                                               int64_t       n_tokens,
                                               int           n_expert_used,
                                               cudaStream_t  stream) {
    constexpr int threads = 256;
    const bool aligned_f32x4 = (uintptr_t) experts % alignof(float4) == 0 &&
                               (uintptr_t) dst     % alignof(float4) == 0;
    if (n_embd % 4 == 0 && aligned_f32x4) {
        const int64_t n_embd4 = n_embd / 4;
        const int64_t blocks  = (n_embd4 * n_tokens + threads - 1) / threads;
        GGML_ASSERT(blocks <= INT_MAX);
        moe_weighted_reduction_f32x4_dynk<has_expert_scale>
            <<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
                (const float4 *) experts, expert_scale, weights, (float4 *) dst, n_embd4, n_tokens, n_expert_used);
    } else {
        const int64_t blocks = (n_embd * n_tokens + threads - 1) / threads;
        GGML_ASSERT(blocks <= INT_MAX);
        moe_weighted_reduction_f32_dynk<has_expert_scale>
            <<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
                experts, expert_scale, weights, dst, n_embd, n_tokens, n_expert_used);
    }
}

static void launch_moe_weighted_reduction_dynk(const float * experts,
                                               const float * expert_scale,
                                               const float * weights,
                                               float *       dst,
                                               int64_t       n_embd,
                                               int64_t       n_tokens,
                                               int           n_expert_used,
                                               cudaStream_t  stream) {
    if (expert_scale != nullptr) {
        launch_moe_weighted_reduction_dynk<true>(
            experts, expert_scale, weights, dst, n_embd, n_tokens, n_expert_used, stream);
    } else {
        launch_moe_weighted_reduction_dynk<false>(
            experts, nullptr, weights, dst, n_embd, n_tokens, n_expert_used, stream);
    }
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

    const float * weights_data    = (const float *) weights->data;
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

    switch (n_expert_used) {
        case 2:
            launch_moe_weighted_reduction<2>((const float *) experts->data, expert_scale_data, weights_data,
                                             (float *) dst->data, n_embd, n_tokens, stream);
            break;
        case 3:
            launch_moe_weighted_reduction<3>((const float *) experts->data, expert_scale_data, weights_data,
                                             (float *) dst->data, n_embd, n_tokens, stream);
            break;
        case 4:
            launch_moe_weighted_reduction<4>((const float *) experts->data, expert_scale_data, weights_data,
                                             (float *) dst->data, n_embd, n_tokens, stream);
            break;
        case 5:
            launch_moe_weighted_reduction<5>((const float *) experts->data, expert_scale_data, weights_data,
                                             (float *) dst->data, n_embd, n_tokens, stream);
            break;
        case 6:
            launch_moe_weighted_reduction<6>((const float *) experts->data, expert_scale_data, weights_data,
                                             (float *) dst->data, n_embd, n_tokens, stream);
            break;
        case 7:
            launch_moe_weighted_reduction<7>((const float *) experts->data, expert_scale_data, weights_data,
                                             (float *) dst->data, n_embd, n_tokens, stream);
            break;
        case 8:
            launch_moe_weighted_reduction<8>((const float *) experts->data, expert_scale_data, weights_data,
                                             (float *) dst->data, n_embd, n_tokens, stream);
            break;
        default:
            // n_expert_used > 8: use the runtime-k kernel (no per-k template). Rare in practice;
            // the common small-k cases above stay on the faster fully-unrolled kernels.
            launch_moe_weighted_reduction_dynk((const float *) experts->data, expert_scale_data, weights_data,
                                               (float *) dst->data, n_embd, n_tokens, (int) n_expert_used, stream);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}
