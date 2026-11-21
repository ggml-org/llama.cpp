#include "common.cuh"

// A gated mix of two vectors driven by two small matrix-vector products (Inkling 2 loop controllers), one token:
// MUL_MAT(W_f, g), MUL_MAT(W_g, g), TANH, MUL(., cur), SUB(cur, .), TANH, MUL(msg, .), ADD
// = cur - tanh(W_f g)*cur + msg*tanh(W_g g). Returns the index of the ADD, or -1.
int  ggml_cuda_match_ctrl_mix(const ggml_cgraph * cgraph, int i);
void ggml_cuda_op_ctrl_mix(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int last);
