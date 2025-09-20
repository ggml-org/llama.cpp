#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Delta-Net linear layer activation
// Implements the complete Delta-Net gated linear attention mechanism
// This includes causal convolution preprocessing and gated delta rule computation
// k, v, q, g: [S, H, n_tokens, n_seqs] - key, value, query, gate tensors
// conv_weight: [conv_dim, 1, conv_kernel_size] - convolution kernel weights
// conv_bias: [conv_dim] - convolution bias (optional, can be NULL)
// beta: [H, n_tokens, n_seqs] - beta parameter for delta rule
// state: [S, S, H, n_seqs] - recurrent state tensor
// chunk_size: chunk size for chunked computation (0 for recurrent mode)
// use_qk_l2norm: whether to apply L2 normalization to query and key
// scale: attention scaling factor
GGML_API struct ggml_tensor * ggml_delta_net(struct ggml_context * ctx,
                                             struct ggml_tensor *  k,
                                             struct ggml_tensor *  v,
                                             struct ggml_tensor *  q,
                                             struct ggml_tensor *  g,
                                             struct ggml_tensor *  conv_weight,
                                             struct ggml_tensor *  conv_bias,
                                             struct ggml_tensor *  beta,
                                             struct ggml_tensor *  state,
                                             bool                  use_qk_l2norm,
                                             float                 scale);

GGML_API struct ggml_tensor * ggml_delta_net_op(struct ggml_context * ctx,
                                                struct ggml_tensor *  q,
                                                struct ggml_tensor *  k,
                                                struct ggml_tensor *  v,
                                                struct ggml_tensor *  g,
                                                struct ggml_tensor *  beta,
                                                struct ggml_tensor *  state,
                                                bool                  use_qk_l2norm,
                                                float                 scale);

#ifdef __cplusplus
}
#endif
