#pragma once

//
// cpu-dump-dequant.h
//
// CPU-side helper for the Tessera runtime-aware calibration pipeline
// (Layer 1, see docs/runtime-aware-pipeline.md).
//
// When the dequant sidecar is enabled (via `LLAMA_TILE640_DEBUG_DEQUANT_DIR`
// or `--tessera-dequant-dir`), the matmul operator calls
// `cpu_dump_dequant` once per quantized weight to materialize the
// dequantized F32 weight to disk. The hook is a no-op when the sidecar
// is disabled or the source tensor is not quantized.
//
// In v3 (this commit) the hook also captures per-row wall-clock timing
// around the dequant call and writes it to the v3 per-row meta strip
// (timing_ns, kernel_id, dispatch_count). This is consumed by the L6
// kernel-direct fitness and the GA orchestrator.
//
// In W4A4 mode (LLAMA_TILE640_DEBUG_DEQUANT_MODE=w4a4) the L1.5
// FP16-reference sidecar (.act.dequant.f32) is written alongside the
// L1 dequant sidecar (.dequant.f32). The two are written from the
// same F32 data block; a future refactor will pass the original FP16
// weight to the hook so the L1.5 reference captures the actual
// ground-truth instead of the dequantized quantized weight.
//
// Each tensor is dumped at most once per process; the helper maintains
// a small dedup set keyed on the tensor name.
//

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;

void cpu_dump_dequant(
        const struct ggml_tensor * src0,
        int64_t                   ne0,
        int64_t                   ne1,
        const char              * tensor_name);

#ifdef __cplusplus
}
#endif
