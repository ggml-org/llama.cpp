#pragma once

// V4 hyperconnection splitter with Sinkhorn normalization.
//
// Splits the mix vector [mix_hc, n_rows] into three sections:
//   - out[0:n_hc]       = sigmoid(mix[i] * scale[0] + base[i]) + eps     ("pre")
//   - out[n_hc:2*n_hc]  = 2 * sigmoid(mix[off] * scale[1] + base[off])    ("post")
//   - out[2*n_hc:]      = Sinkhorn-normalized n_hc x n_hc comb matrix     ("comb")
//
// The comb section starts as logits (mix * scale[2] + base), then a
// per-dst_hc row softmax (max-subtract + exp + normalize) with `eps` added,
// then alternating column / row normalizations for sinkhorn_iters - 1 more
// iterations. The result is doubly-stochastic up to `eps`-stabilization.
//
// Expected shape:
//   mixes  : [mix_hc, n_rows]    float32, contiguous along ne[0]
//   scale  : [3]                 float32 (pre, post, comb scales)
//   base   : [mix_hc]            float32, matches the mix layout
//   dst    : [mix_hc, n_rows]    float32, same shape as mixes
// where  mix_hc == (2 + n_hc) * n_hc  and  n_hc in [1, 16].
//
// Op params (i32, i32, f32): n_hc, sinkhorn_iters, eps.
//
// CUDA kernel design:
//   - One CUDA block per output row.
//   - Block size rounded up to a warp multiple (>= 32) so __syncthreads()
//     and any future block-wide reductions are well-formed even when the
//     natural row width (mix_hc = 24 or 80 for n_hc = 4 or 8) is not a
//     warp multiple. Excess threads do no memory work; loops guard `i < n`.
//   - Sections 1, 2, and the final copy parallelize across the block.
//   - Section 3 (Sinkhorn iterations on the n_hc x n_hc comb matrix) is
//     serialized on `tid == 0`; n_hc <= 16 makes this trivially cheap
//     (O(n_hc^2 * sinkhorn_iters) per row) and avoids the complexity of
//     warp-cooperative reductions over a 4-or-8-wide inner dimension.
//
// Reference Metal kernel:  ggml/src/ggml-metal/ggml-metal.metal:2076-2245
// CPU reference:           ggml/src/ggml-cpu/ops.cpp:11037-11117
// Public API:              ggml/include/ggml.h (ggml_dsv4_hc_split_sinkhorn)

#include "common.cuh"

void ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
