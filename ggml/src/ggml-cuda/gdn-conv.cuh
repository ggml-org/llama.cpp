#pragma once

#include "common.cuh"

// Fused conv for the linear-attention layers built by build_conv_state (qwen35, qwen35moe, qwen3next):
//   CONCAT(conv state [3, C], TRANSPOSE(x [C, T])) -> SSM_CONV(w [4, C]) -> UNARY(silu)
// One kernel writes the conv-state snapshot copies that follow the concat (CPY of 3-column views of it into the
// recurrent cache) from the state and x. Prefill ubatches (T >= 256): the conv reads the state and x directly too, so
// the concat is not written at all (ported from stew675/llama-cpp-rdna-boosts, block 08, gdn-conv.cu; MIT). Shorter
// batches (the verify of a speculative step): the kernel also writes the concat, and the conv runs at its own node.
// Both are bit-identical to the unfused ops. GGML_CUDA_DISABLE_CONV_FUSION=1 turns them off.
//
// At a CONCAT or SSM_CONV node of that pattern, runs its part and returns the number of nodes after i it covered
// (or 0 if the pattern does not apply).
int ggml_cuda_try_fuse_gdn_conv(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
