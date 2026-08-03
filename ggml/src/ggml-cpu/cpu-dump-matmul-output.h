#pragma once

//
// cpu-dump-matmul-output.h
//
// CPU-side helper for the Tessera runtime-aware calibration pipeline
// (Layer 2, see docs/runtime-aware-pipeline.md). See cpu-dump-matmul-output.cpp
// for the implementation.
//
// When the matmul-output sidecar is enabled (via
// `LLAMA_TILE640_DEBUG_MATMUL_OUTPUT_DIR` or `--tessera-matmul-output-dir`),
// the matmul operator calls `cpu_dump_matmul_output` once per matmul
// invocation to materialize the F32 dst tensor (the kernel's actual
// matmul output, distinct from the dequantized weight the L1 sidecar
// captures) to a sidecar file. The hook is a no-op when the sidecar
// is disabled.
//
// The on-disk file uses the v3 TDQT header layout (magic = "TPMO",
// shared v3 version) and an F32 data block. One file per tensor name;
// the first call opens, subsequent calls for the same name append to
// the same stream until it is closed.
//
// Each tensor's matmul output is captured at most once per call from
// thread 0 of the matmul op (the existing L1 hook follows the same
// pattern via a per-process dedup set; the L2 hook intentionally does
// NOT dedup because the matmul is invoked once per chunk/token-batch
// and we want every invocation's output to land in the sidecar).
//

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;
struct ggml_compute_params;

void cpu_dump_matmul_output(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * dst,
        const char              * tensor_name);

#ifdef __cplusplus
}
#endif
