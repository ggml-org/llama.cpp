#pragma once

#include "common.cuh"

// Fused conv for the linear-attention layers built by build_conv_state (qwen35, qwen35moe, qwen3next):
//   CONCAT(conv state [3, C], TRANSPOSE(x [C, T])) -> SSM_CONV(w [4, C]) -> UNARY(silu)
// The conv reads the state and x directly, so the concat only materializes the columns that the conv-state snapshot
// copies read. Bit-identical to the unfused ops (same 4-tap sum, same silu). Ported from stew675/llama-cpp-rdna-boosts
// (block 08, gdn-conv.cu; MIT). GGML_CUDA_DISABLE_CONV_FUSION=1 turns it off.
//
// At a CONCAT or SSM_CONV node of that pattern, runs its part and returns the number of nodes after i it covered
// (or 0 if the pattern does not apply).
int ggml_cuda_try_fuse_gdn_conv(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
