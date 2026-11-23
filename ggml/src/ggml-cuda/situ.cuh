#include "common.cuh"

// The SITU activation of Inkling 2's experts as one element-wise kernel:
//   SCALE(g) -> TANH -> SCALE, SIGMOID(g), MUL, SCALE(u) -> TANH -> SCALE, MUL
//   = (tanh(g/beta)*beta * sigmoid(g)) * (tanh(u/lb)*lb)
// g and u may be strided views (the halves of a fused gate_up product). Returns the index of the last MUL, or -1.
int  ggml_cuda_match_situ(const ggml_cgraph * cgraph, int i, bool check_memory = true);
void ggml_cuda_op_situ(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int last);
