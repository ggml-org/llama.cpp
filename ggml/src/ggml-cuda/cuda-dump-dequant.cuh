#pragma once

#include "common.cuh"

struct ggml_tensor;

// Tessera Layer 1 (kernel dequant fidelity, see
// docs/runtime-aware-pipeline.md). When the dequant sidecar is enabled
// (--tessera-dequant-dir / LLAMA_TILE640_DEBUG_DEQUANT_DIR), this helper
// materializes the full dequantized F32 weight of `src0` on the device,
// copies it back to the host, and writes the rows to the sidecar via
// common/tessera-debug.h.
//
// The hook is a no-op when the dequant dir is not configured. The
// matmul output is byte-identical to a stock build in that case.
//
// Call once per matmul at the operator level (ggml_cuda_mul_mat /
// ggml_cuda_mul_mat_id). Per-process dedup ensures a single dump per
// tensor; batched matmul dumps the first slice only. See the .cu file
// for the open questions and limitations.
void ggml_cuda_dump_dequant(ggml_backend_cuda_context & ctx, const ggml_tensor * src0);
