#include "common.cuh"
#include "ggml.h"

#include <initializer_list>

struct ggml_cuda_topk_moe_args {
    bool sigmoid{};
    bool sqrt_softplus{};
    bool softmax{};
    bool delayed_softmax{};
    bool prob_bias{};
    bool norm{};
    bool scale{};
};

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args);

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids);

// raw-pointer entry for GGML_OP_MOE_FFN: softmax gating, top-n_expert_used, normalized weights
// ids are written with a row stride of n_expert
void ggml_cuda_topk_moe_softmax_norm(ggml_backend_cuda_context & ctx,
                                     const float *               logits,
                                     float *                     weights,
                                     int32_t *                   ids,
                                     int                         n_rows,
                                     int                         n_expert,
                                     int                         n_expert_used,
                                     float                       clamp_val);
