#pragma once

// V4 FP8 KV-cache simulation: quantizes/dequantizes the non-RoPE prefix
// of each row in 64-element blocks through E4M3FN representation with
// per-block scaling; leaves the RoPE tail unchanged.
//
// Block-scaled algorithm (must match CPU reference for the
// test-backend-ops NMSE check):
//   for each row (n_rows = ne01 * ne02 * ne03):
//     for off in [0, n_nope) step 64:
//       amax  = max(|src[off..off+64)|, 1e-4)
//       scale = 2^ceil(log2(amax / 448))
//       dst[off+i] = dequant_e4m3fn(clamp(src[off+i]/scale, -448, 448)) * scale
//     copy src[n_nope..ne00) to dst unchanged   // RoPE tail
//
// References:
//   CPU reference:        ggml/src/ggml-cpu/ops.cpp:11235-11313
//   Metal kernel:         ggml/src/ggml-metal/ggml-metal.metal:2302-2376
//   Metal dispatch:       ggml/src/ggml-metal/ggml-metal-ops.cpp:1550-1594
//   Public API:           ggml/include/ggml.h:2591 (ggml_dsv4_fp8_kv_quantize)
//
// Dual-path implementation:
//   - __CUDA_ARCH__ >= 890 (Ada/Hopper/Blackwell): native FP8 via the
//     __nv_fp8_e4m3 class wrapper from <cuda_fp8.h> (round-to-nearest-even,
//     saturate-to-finite to +/-448).
//   - __CUDA_ARCH__ <  890 (Volta/Turing/Ampere): software emulation by
//     nearest-even E4M3FN code search, mirroring the CPU reference.
//
// Both paths produce numerically equivalent output (subject to FP8's
// inherent lossiness). The four test_dsv4_fp8_kv_quantize cases from
// Stream A (tests/test-backend-ops.cpp:8868-8871) validate with
// max_nmse_err = 1e-3.

#include "common.cuh"

void ggml_cuda_op_dsv4_fp8_kv_quantize(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
