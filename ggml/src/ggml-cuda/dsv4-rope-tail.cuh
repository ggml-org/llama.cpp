#pragma once

// V4 partial-RoPE: applies RoPE rotation to the last n_dims elements of each
// row, leaving the non-RoPE prefix (i.e. the first ne00 - n_dims elements)
// unchanged. The rotation math is the same as ggml_rope_ext (with YaRN
// extrapolation when ext_factor != 0), restricted to the tail.
//
// Reference Metal kernel: ggml/src/ggml-metal/ggml-metal.metal:4906-4997
// CPU reference:          ggml/src/ggml-cpu/ops.cpp:5961
// Public API:             ggml/include/ggml.h:2599 (ggml_dsv4_rope_tail)
//
// The dispatch function extracts op_params (i32 slots 0..3:
// n_dims, mode, n_ctx_orig, inverse; f32 slots 4..9: freq_base, freq_scale,
// ext_factor, attn_factor, beta_fast, beta_slow) from the destination
// tensor, precomputes YaRN corr_dims host-side, and launches the kernel
// with grid = (ne01, ne02, ne03), block.x = min(256, ne00), matching the
// Metal dispatch at ggml/src/ggml-metal/ggml-metal-ops.cpp:1670.
//
// Supports the two RoPE modes the public V4 API allows
// (ggml/src/ggml.c:6426 ASSERT mode == NORMAL || mode == NEOX). All
// other modes are rejected via ggml_backend_cuda_device_supports_op so
// the framework falls back to CPU rather than producing wrong output.

#include "common.cuh"

void ggml_cuda_op_dsv4_rope_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
