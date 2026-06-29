#pragma once

#include "common.cuh"

// Stage-1 dense device sampler for DiffusionGemma. Reads per-position canvas logits straight from a
// device tensor [n_vocab, n_tokens] (row-major per position) and returns the small per-position arrays,
// removing the per-step full-canvas D2H logits download + host full-vocab reductions.
//   logits   : device tensor, F32, contiguous, ne[0]=n_vocab, nrows>=n_tokens
//   u_host   : host [n_tokens] pre-drawn uniforms (kept on the host RNG stream for reproducibility)
//   *_host   : host outputs [n_tokens] (argmax, entropy, sampled)
// Returns false on a non-CUDA / unsupported tensor (caller falls back to the host path).
bool ggml_cuda_diffusion_sample(
        struct ggml_tensor * logits,
        const float        * u_host,
        int                * argmax_host,
        float              * entropy_host,
        int                * sampled_host,
        int                  n_tokens,
        float                inv_temp);
