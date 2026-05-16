#pragma once

// V4 hyper-connection weighted-sum: collapses the hc dimension.
//
//   out[embd, token] = sum over hc of weights[hc, token] * x[embd, hc, token]
//
// Inputs (all GGML_TYPE_F32):
//   dst->src[0] = x       shape {n_embd, n_hc,    n_tokens, 1}
//   dst->src[1] = weights shape {n_hc,   n_tokens, 1,       1}
// Output (GGML_TYPE_F32):
//   dst                   shape {n_embd, n_tokens, 1,       1}
//
// Reference Metal kernel: ggml/src/ggml-metal/ggml-metal.metal:2278-2327
// Reference Metal dispatch: ggml/src/ggml-metal/ggml-metal-ops.cpp:1440-1486
// CPU reference: ggml/src/ggml-cpu/ops.cpp:11121 (ggml_compute_forward_dsv4_hc_weighted_sum)
// Public API: ggml/include/ggml.h:2574 (ggml_dsv4_hc_weighted_sum)
//
// Implementation: embarrassingly parallel; one thread per output element
// (n_embd * n_tokens total), each thread loops over n_hc to accumulate.
// Strides are kept in bytes (matching the Metal kernel + the ggml tensor
// nb[] convention) and applied via (const char *) base + offset casts.

#include "common.cuh"

void ggml_cuda_op_dsv4_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
