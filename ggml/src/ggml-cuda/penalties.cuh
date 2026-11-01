#include "common.cuh"

// Fused sampling penalties: logits[ids[i]] = x*(x > 0 ? pos_scale[i] : neg_scale[i]) - offset[i]
// ids: [n] I32, coefs: [n, 3] F32 (pos_scale, neg_scale, offset), dst: [1, n_vocab] F32 updated in place
void ggml_cuda_op_penalties_fused(ggml_backend_cuda_context & ctx,
                                  const ggml_tensor * ids,
                                  const ggml_tensor * coefs,
                                  ggml_tensor *       dst);
