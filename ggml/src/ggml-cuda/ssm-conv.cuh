#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr,
                           ggml_tensor * res_add_node = nullptr, ggml_tensor * res_scale_dst = nullptr);

// CONCAT(history, x) -> VIEW -> CPY(state) -> SSM_CONV -> RESHAPE -> ADD(residual) -> SCALE, for one token and sequence
void ggml_cuda_op_ssm_conv_step(ggml_backend_cuda_context & ctx, const ggml_tensor * concat, ggml_tensor * cpy,
        const ggml_tensor * conv, const ggml_tensor * add, ggml_tensor * scale,
        const ggml_tensor * res2 = nullptr, ggml_tensor * add2 = nullptr); // add2 = res2 + scale, fused after the step

// the MoE weighted sum of one token (experts [n, n_used] * scale * weights, summed) feeding a one-token conv step as
// its new column and residual, in one kernel; add2 as in ggml_cuda_op_ssm_conv_step
void ggml_cuda_op_ssm_conv_step_moe(ggml_backend_cuda_context & ctx,
        const ggml_tensor * experts, const ggml_tensor * expert_scale, const ggml_tensor * weights, ggml_tensor * moe_dst,
        const ggml_tensor * concat, ggml_tensor * cpy, const ggml_tensor * conv, ggml_tensor * scale,
        const ggml_tensor * res2, ggml_tensor * add2);
