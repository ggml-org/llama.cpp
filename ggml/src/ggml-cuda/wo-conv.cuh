#include "common.cuh"

// Inkling 2, one token: the attention output projection (a BF16 MUL_MAT on one F32 vector) followed by the one-token
// conv step on its result (the step's new column and its residual), optionally with the layer residual ADD after the
// step, as one kernel: a block per output row with the arithmetic of mul_mat_vec_f (same block size, same reduction,
// the same two idle blocks on RDNA4), then that row's conv step. The matcher lives in ggml-cuda.cu beside the conv-step
// matchers it builds on.
void ggml_cuda_op_wo_conv(ggml_backend_cuda_context & ctx, const ggml_tensor * mm,
        const ggml_tensor * past, const ggml_tensor * conv_w, float conv_scale, ggml_tensor * state_cell,
        ggml_tensor * dst, const ggml_tensor * res2, ggml_tensor * dst2);
