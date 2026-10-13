#pragma once

#include "common.cuh"

// RMS norm fusions for narrow rows (32..256 columns, at least 4096 rows), such as per-head norms: one wave per row and
// 8 rows per 256-thread block instead of one block per row, which leaves the stock kernel block-scheduling bound. The
// sum uses the same per-warp xor trees and 8-partial tree as rms_norm_f32<256>, so results are bit-identical.
// Ported from stew675/llama-cpp-rdna-boosts (block 08, norm-gated.cu; MIT). GGML_CUDA_DISABLE_NORM_ROWS=1 turns it off.
//
// Patterns at node i (RMS_NORM):
//   RMS_NORM -> MUL(w)                       (weight broadcast over rows)
//   RMS_NORM -> MUL(w) -> UNARY(silu, z) -> MUL   (gated norm, e.g. qwen35's linear-attention output norm)
//   RMS_NORM -> SCALE(s)                     (L2 norm built from RMS norm)
// Returns the number of nodes after i that it covered, or 0 if none applies.
int ggml_cuda_try_fuse_norm_rows(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
