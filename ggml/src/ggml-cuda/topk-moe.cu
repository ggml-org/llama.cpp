#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "topk-moe.cuh"

#include <cmath>
#include <initializer_list>

// Kernel config struct - passed by value to CUDA kernel
struct topk_moe_config {
    bool use_sigmoid;
    bool use_sqrt_softplus;
    bool with_norm;
    bool delayed_softmax;
    bool grouped_experts;

    int  n_expert_groups;
    int  n_exp_per_group;
    int  n_group_used;
};

// Warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path.
template <int experts_per_thread, bool use_limit>
__device__ void softmax_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    float max_val = -INFINITY;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            max_val = max(max_val, vals[i]);
        }
    }

    max_val = warp_reduce_max(max_val);

    float sum = 0.f;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = expf(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } else {
            vals[i] = 0.f;
        }
    }

    sum = warp_reduce_sum(sum);

    const float inv_sum = 1.0f / sum;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            vals[i] *= inv_sum;
        }
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sigmoid_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? 1.f / (1.f + expf(-vals[i])) : -INFINITY;
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sqrt_softplus_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? sqrtf(vals[i] > 20.0f ? vals[i] : logf(1.0f + expf(vals[i]))) : -INFINITY;
    }
}

__device__ __forceinline__ void topk_moe_top2_merge(float &a1, float &a2, float b1, float b2) {
    // Merge two sorted top-2 pairs:
    //   a1 >= a2
    //   b1 >= b2
    if (b1 > a1) {
        a2 = max(a1, b2);
        a1 = b1;
    } else if (b1 > a2) {
        a2 = b1;
    }

    if (b2 > a2) {
        a2 = b2;
    }
}

template <int experts_per_thread, bool has_bias>
__device__ __forceinline__ float topk_moe_group_score(const float (&wt)[experts_per_thread],
                                                      const float * bias,
                                                      const int     n_experts,
                                                      const int     n_exp_per_group,
                                                      const int     group,
                                                      const int     lane) {
    const int start = group * n_exp_per_group;
    const int end   = start + n_exp_per_group;

    float top1 = -INFINITY;
    float top2 = -INFINITY;

#pragma unroll
    for (int i = 0; i < experts_per_thread; ++i) {
        const int e = i * WARP_SIZE + lane;

        if (e >= start && e < end && e < n_experts) {
            float v = wt[i];

            if constexpr (has_bias) {
                v += bias[e];
            }

            if (v > top1) {
                top2 = top1;
                top1 = v;
            } else if (v > top2) {
                top2 = v;
            }
        }
    }

    // Reduce the top-2 pair across the whole warp.
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
        const float o1 = __shfl_xor_sync(0xFFFFFFFF, top1, mask, WARP_SIZE);
        const float o2 = __shfl_xor_sync(0xFFFFFFFF, top2, mask, WARP_SIZE);

        topk_moe_top2_merge(top1, top2, o1, o2);
    }

    return (top2 == -INFINITY) ? top1 : (top1 + top2);
}

/*
    This kernel does the following:
    1. optionally softmax over the logits per token [n_experts, n_tokens]
    2. argmax reduce over the top-k (n_experts_used) logits
    3. write weights + ids to global memory
    4. optionally normalize the weights or apply softmax over the selected logits
    5. optionally select top-k grouped experts (n_group_used)

    It is intended as fusion of softmax->top-k->get_rows pipeline for MoE models
*/
template <int n_experts, bool has_bias>
__launch_bounds__(TOPK_MOE_ROWS_PER_BLOCK * WARP_SIZE, 1)
__global__ void topk_moe_cuda(const float *         logits,
                              float *               weights,
                              int32_t *             ids,
                              float *               bias,
                              const int             n_rows,
                              const int             n_expert_used,
                              const float           clamp_val,
                              const float           scale_val,
                              const topk_moe_config config) {
#if defined(GGML_USE_MUSA)
    // MUSA: every warp of a partially filled block must reach the barrier below.
    const int row = MIN(blockIdx.x * blockDim.y + threadIdx.y, n_rows - 1);
#else
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
#endif // defined(GGML_USE_MUSA)
    if (row >= n_rows) {
        return;
    }

    logits += n_experts * row;
    weights += n_expert_used * row;
    ids += n_experts * row;

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[experts_per_thread];

    // Initialize all slots to -INFINITY
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert  = i + threadIdx.x;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    // Weights and IDs can alias logits, so wait until every row in the block reads its logits.
    __syncthreads();

    if (!config.delayed_softmax) {
        if (config.use_sigmoid) {
           sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else if (config.use_sqrt_softplus) {
           sqrt_softplus_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else {
           softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs.
    // NaN comparisons always return false, which would cause the same expert to be
    // selected repeatedly. -FLT_MAX compares normally and is still excluded by the
    // -INFINITY sentinel used after each selection round.
    // More relevant for the cuBLAS path. See https://github.com/ggml-org/llama.cpp/issues/19659
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // selection_wt is only needed when bias is present (selection uses wt + bias)
    // when no bias, we use wt directly for both selection and weight values
    [[maybe_unused]] float selection_wt[has_bias ? experts_per_thread : 1];

    if constexpr (has_bias) {
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            selection_wt[i] = -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_experts; i += WARP_SIZE) {
            const int expert = i + threadIdx.x;
            selection_wt[i / WARP_SIZE] =
                (n_experts % WARP_SIZE == 0 || expert < n_experts) ? wt[i / WARP_SIZE] + bias[expert] : -INFINITY;
        }
    }

    //at this point, each thread holds either a portion of the softmax distribution
    //or the raw logits. We do the argmax reduce over n_expert_used, each time marking
    //the expert weight as -inf to exclude from the next iteration

    float wt_sum = 0.f;

    float output_weights[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    if (config.grouped_experts) {
        const int G = config.n_expert_groups;
        const int P = config.n_exp_per_group;

        // The score for each group must be reduced across the whole warp because
        // a group is distributed over many lanes.
        float gval = -INFINITY;

        for (int g = 0; g < G; ++g) {
            const float score = topk_moe_group_score<experts_per_thread, has_bias>(
                wt, bias, n_experts, P, g, threadIdx.x
            );

            if (threadIdx.x == g) {
                gval = score;
            }
        }

        unsigned selected_mask = 0;

        // Select top n_group_used groups.
        for (int k = 0; k < config.n_group_used; ++k) {
            float best = -INFINITY;
            int   best_group = WARP_SIZE;

            int   my_group = WARP_SIZE;
            float my_score = -INFINITY;

            if (threadIdx.x < G) {
                // Skip groups that were already selected in a previous round.
                if ((selected_mask & (1u << threadIdx.x)) == 0) {
                    my_group = threadIdx.x;
                    my_score = gval;
                }
            }

            if (my_score > best || (my_score == best && my_group < best_group)) {
                best = my_score;
                best_group = my_group;
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float other       = __shfl_xor_sync(0xFFFFFFFF, best, mask, WARP_SIZE);
                const int   other_group = __shfl_xor_sync(0xFFFFFFFF, best_group, mask, WARP_SIZE);

                if (other > best || (other == best && other_group < best_group)) {
                    best = other;
                    best_group = other_group;
                }
            }

            if (best_group < G) {
                selected_mask |= (1u << best_group);

                if (threadIdx.x == best_group) {
                    gval = -INFINITY;
                }
            }
        }

        // Build the masked expert selection scores.
        // If bias is present, selection uses wt + bias, but output weights use wt.
        float sel[experts_per_thread];

#pragma unroll
        for (int i = 0; i < experts_per_thread; ++i) {
            const int e = i * WARP_SIZE + threadIdx.x;

            if ((n_experts % WARP_SIZE == 0 || e < n_experts) && P > 0) {
                const int group = e / P;

                if (group < G && ((selected_mask >> group) & 1u)) {
                    float v = wt[i];

                    if constexpr (has_bias) {
                        v += bias[e];
                    }

                    sel[i] = v;
                } else {
                    sel[i] = -INFINITY;
                }
            } else {
                sel[i] = -INFINITY;
            }
        }

        ggml_cuda_pdl_lc();

        // Final top-k expert selection over masked scores.
        for (int k = 0; k < n_expert_used; ++k) {
            float max_w      = -INFINITY;
            float max_s      = -INFINITY;
            int   max_expert = threadIdx.x;

#pragma unroll
            for (int i = 0; i < experts_per_thread; ++i) {
                const int expert = threadIdx.x + i * WARP_SIZE;

                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && sel[i] > max_s) {
                    max_s      = sel[i];
                    max_w      = wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float other_s   = __shfl_xor_sync(0xFFFFFFFF, max_s, mask, WARP_SIZE);
                const float other_w   = __shfl_xor_sync(0xFFFFFFFF, max_w, mask, WARP_SIZE);
                const int   other_exp = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);

                if (other_s > max_s || (other_s == max_s && other_exp < max_expert)) {
                    max_s      = other_s;
                    max_w      = other_w;
                    max_expert = other_exp;
                }
            }

            if ((k & (WARP_SIZE - 1)) == threadIdx.x) {
                output_weights[k / WARP_SIZE] = max_w;
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                ids[k] = max_expert;

                if (config.with_norm) {
                    wt_sum += max_w;
                }

                sel[max_expert / WARP_SIZE] = -INFINITY;
            }
        }
    } else {
        ggml_cuda_pdl_lc();
        for (int k = 0; k < n_expert_used; k++) {
            float max_val    = wt[0];
            int   max_expert = threadIdx.x;

            if constexpr (has_bias) {
                float max_val_s = selection_wt[0];

#pragma unroll
                for (int i = 1; i < experts_per_thread; i++) {
                    const int expert = threadIdx.x + i * WARP_SIZE;
                    if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && selection_wt[i] > max_val_s) {
                        max_val    = wt[i];
                        max_val_s  = selection_wt[i];
                        max_expert = expert;
                    }
                }

#pragma unroll
                for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                    const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                    const float val_s  = __shfl_xor_sync(0xFFFFFFFF, max_val_s, mask, WARP_SIZE);
                    const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                    if (val_s > max_val_s || (val_s == max_val_s && expert < max_expert)) {
                        max_val    = val;
                        max_val_s  = val_s;
                        max_expert = expert;
                    }
                }

                if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                    selection_wt[max_expert / WARP_SIZE] = -INFINITY;
                }
            } else {
#pragma unroll
                for (int i = 1; i < experts_per_thread; i++) {
                    const int expert = threadIdx.x + i * WARP_SIZE;
                    if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                        max_val    = wt[i];
                        max_expert = expert;
                    }
                }

#pragma unroll
                for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                    const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                    const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                    if (val > max_val || (val == max_val && expert < max_expert)) {
                        max_val    = val;
                        max_expert = expert;
                    }
                }

                if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                    wt[max_expert / WARP_SIZE] = -INFINITY;
                }
            }

            if ((k & (WARP_SIZE - 1)) == threadIdx.x) {
                output_weights[k / WARP_SIZE] = max_val;
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                ids[k] = max_expert;
                if (config.with_norm) {
                    wt_sum += max_val;
                }
            }
        }
    }

    if (config.with_norm) {
        wt_sum              = warp_reduce_sum(wt_sum);
        wt_sum              = max(wt_sum, clamp_val);
        const float inv_sum = 1.0f / wt_sum;

        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv_sum;
        }
    }

    if (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, threadIdx.x);
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + threadIdx.x;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

template<bool has_bias>
static void launch_topk_moe_cuda(ggml_backend_cuda_context & ctx,
                                 const float *               logits,
                                 float *                     weights,
                                 int32_t *                   ids,
                                 float *                     bias,
                                 const int                   n_rows,
                                 const int                   n_expert,
                                 const int                   n_expert_used,
                                 const float                 clamp_val,
                                 const float                 scale_val,
                                 const topk_moe_config       config) {
    GGML_ASSERT(!(config.with_norm && config.delayed_softmax) &&
                "delayed softmax is not supported with weight normalization");
    const int    rows_per_block = TOPK_MOE_ROWS_PER_BLOCK;
    dim3         grid_dims((n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    dim3         block_dims(WARP_SIZE, rows_per_block, 1);
    cudaStream_t stream = ctx.stream();
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    switch (n_expert) {
        case 1:
            ggml_cuda_kernel_launch(topk_moe_cuda<1, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 2:
            ggml_cuda_kernel_launch(topk_moe_cuda<2, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 4:
            ggml_cuda_kernel_launch(topk_moe_cuda<4, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 8:
            ggml_cuda_kernel_launch(topk_moe_cuda<8, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 16:
            ggml_cuda_kernel_launch(topk_moe_cuda<16, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 32:
            ggml_cuda_kernel_launch(topk_moe_cuda<32, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 64:
            ggml_cuda_kernel_launch(topk_moe_cuda<64, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 128:
            ggml_cuda_kernel_launch(topk_moe_cuda<128, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 256:
            ggml_cuda_kernel_launch(topk_moe_cuda<256, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 288: // StepFun 3.7
            ggml_cuda_kernel_launch(topk_moe_cuda<288, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 512:
            ggml_cuda_kernel_launch(topk_moe_cuda<512, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 576:
            ggml_cuda_kernel_launch(topk_moe_cuda<576, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        default:
            GGML_ASSERT(false && "fatal error");
            break;
    }
}

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const int n_experts = logits->ne[0];
    const int n_rows    = logits->ne[1];

    const float * logits_d  = (const float *) logits->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;
    float *       bias_d    = bias ? (float *) bias->data : nullptr;

    float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    const int n_expert_used = weights->ne[1];

    const bool with_norm = clamp != nullptr;

    float clamp_val = -INFINITY;
    if (clamp) {
        clamp_val = ggml_get_op_params_f32(clamp, 0);
    }

    topk_moe_config config;
    config.use_sigmoid       = args.sigmoid;
    config.use_sqrt_softplus = args.sqrt_softplus;
    config.with_norm         = with_norm;
    config.delayed_softmax   = args.delayed_softmax;
    config.grouped_experts   = args.grouped_experts;
    config.n_expert_groups   = args.n_expert_groups;
    config.n_exp_per_group   = args.n_exp_per_group;
    config.n_group_used      = args.n_group_used;

    GGML_ASSERT(!(config.grouped_experts && config.delayed_softmax));

    if (bias) {
        launch_topk_moe_cuda<true>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    } else {
        launch_topk_moe_cuda<false>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    }
}

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids) {
    // must match an instantiation of launch_topk_moe_cuda: a power of 2 up to 512,
    // or one of the non-power-of-2 expert counts of supported models
    const int n_expert = ids->nb[1] / ids->nb[0];
    if (((n_expert & (n_expert - 1)) != 0 || n_expert > 512) && n_expert != 288 && n_expert != 576) {
        return false;
    }

    if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        return false;
    }

    if (gating_op->op == GGML_OP_SOFT_MAX) {
        const ggml_tensor * softmax  = gating_op;
        float               scale    = 1.0f;
        float               max_bias = 0.0f;

        memcpy(&scale, (const float *) softmax->op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) softmax->op_params + 1, sizeof(float));

        if (!ggml_is_contiguous(softmax->src[0])) {
            return false;
        }

        if (scale != 1.0f || max_bias != 0.0f) {
            return false;
        }

        // don't fuse when masks or sinks are present
        if (softmax->src[1] || softmax->src[2]) {
            return false;
        }
    } else if (gating_op->op == GGML_OP_UNARY) {
        ggml_unary_op op = ggml_get_unary_op(gating_op);

        if (op != GGML_UNARY_OP_SIGMOID && op != GGML_UNARY_OP_SOFTPLUS) {
            return false;
        }
    }

    return true;
}
