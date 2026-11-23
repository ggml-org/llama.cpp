#include "common.cuh"

// Inkling 2 loop controllers, one token, 1-4 of them in a row in the graph:
//   CONCAT(a, b) -> MUL_MAT(W_down, .) -> the gated mix of ggml_cuda_match_ctrl_mix
// as two launches for all of them: the down products read [a; b] from the two sources (no concatenated copy), with
// the arithmetic of mul_mat_vec_f, then the gated mixes. Returns the index of the last node, or -1.
int  ggml_cuda_match_ctrl_batch(const ggml_cgraph * cgraph, int i, bool check_memory = true);
void ggml_cuda_op_ctrl_batch(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, int last);
