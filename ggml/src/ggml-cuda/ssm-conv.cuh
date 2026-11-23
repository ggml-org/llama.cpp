#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr,
                           ggml_tensor * res_add_node = nullptr, ggml_tensor * res_scale_dst = nullptr);

// CONCAT(history, x) -> VIEW -> CPY(state) -> SSM_CONV -> RESHAPE -> ADD(residual) -> SCALE, for one token and sequence
void ggml_cuda_op_ssm_conv_step(ggml_backend_cuda_context & ctx, const ggml_tensor * concat, ggml_tensor * cpy,
        const ggml_tensor * conv, const ggml_tensor * add, ggml_tensor * scale,
        const ggml_tensor * res2 = nullptr, ggml_tensor * add2 = nullptr); // add2 = res2 + scale, fused after the step
