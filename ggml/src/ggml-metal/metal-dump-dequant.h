#pragma once

//
// metal-dump-dequant.h
//
// Tessera Layer 1 dequant sidecar helper for the Metal backend
// (see docs/runtime-aware-pipeline.md).
//
// Called from ggml_metal_op_mul_mat / ggml_metal_op_mul_mat_id before the
// actual matmul dispatch. Issues a one-shot quant->F32 cpy kernel on a
// fresh command buffer, copies the dequantized weight out via
// `[id<MTLBuffer> contents]` (Apple Silicon shared memory), and streams it
// to the sidecar writer. No-op when the dequant debug hook is not
// enabled or src0 is not quantized.
//

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

// Materialize the dequantized F32 form of `src0` and stream it to the
// sidecar writer. The dump is laid out as `rows` rows of `cols` F32
// values, in row-major order. For a typical matmul weight this is
// (src0->ne[1], src0->ne[0]); for batched weights the caller may pass
// the product of ne1*ne2*ne3 in `rows`.
//
// `name` identifies the tensor in the sidecar directory
// (`<dequant_dir>/<name>.dequant.f32`).
//
// The function allocates a fresh destination buffer per call to avoid
// races with in-flight Metal command buffers; it synchronously waits
// for the dequant kernel to complete before reading the result back.
void metal_dump_dequant(
        ggml_metal_device_t dev,
        const struct ggml_tensor * src0,
        int64_t rows,
        int64_t cols,
        const char * name);

#ifdef __cplusplus
}
#endif
