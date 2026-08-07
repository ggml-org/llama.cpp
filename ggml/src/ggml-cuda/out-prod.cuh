#include "common.cuh"

void ggml_cuda_out_prod(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Scattered outer-product for MUL_MAT_ID backward (gradient w.r.t. expert weight matrices).
// src0: activations [cols, n_expert_used, n_tokens]  F32
// src1: grad_output [rows, n_expert_used, n_tokens]  F32
// src2: expert ids  [n_expert_used, n_tokens]        I32 (may be CPU-resident)
// dst:  grad_weight [cols, rows, n_expert, 1]         F32
void ggml_cuda_out_prod_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
