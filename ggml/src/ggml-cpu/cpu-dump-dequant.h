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
