#pragma once

// V4 hyperconnection expand: per-token mix of block_out and residual.
//
// out[i, hc, tok] = post[hc, tok] * block_out[i, tok]
//                 + sum_{hc'} comb[hc, hc', tok] * residual[i, hc', tok]
//
// Shapes:
//   block_out: 2D {n_embd, n_tokens}            -- no hc axis
//   residual:  3D {n_embd, n_hc,    n_tokens}
//   post:      2D {n_hc,  n_tokens}
//   comb:      3D {n_hc,  n_hc,     n_tokens}
//   dst:       3D {n_embd, n_hc,    n_tokens}
//
// Reference Metal kernel: ggml/src/ggml-metal/ggml-metal.metal:2247-2276
// CPU reference:          ggml/src/ggml-cpu/ops.cpp:11200+
// Public API:             ggml/include/ggml.h:2581 (ggml_dsv4_hc_expand)
// Shape constructor:      ggml/src/ggml.c:6363-6366
//
// Embarrassingly parallel: one thread per output element (i_embd, i_hc, i_tok).
// Each thread does an n_hc-wide accumulation for the comb*residual term plus
// one fused multiply-add for the post*block_out term.

#include "common.cuh"

void ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
