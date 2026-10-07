#include "common.cuh"

void ggml_cuda_op_moe_weighted_reduction(ggml_backend_cuda_context & ctx,
                                         const ggml_tensor *         experts,
                                         const ggml_tensor *         expert_scale,
                                         const ggml_tensor *         weights,
                                         const ggml_tensor *         bias,
                                         const ggml_tensor *         ids,
                                         ggml_tensor *               dst);
