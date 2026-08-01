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

// Tile640 variant: encodes the row-aware `kernel_TILE640_DEQUANT` Metal
// kernel (one thread per element, no row overflow) into the backend's
// active command encoder, writing the dequantized weight into a shared
// device buffer. `op` is the GGML_OP_TILE640_MATMUL / _MATMUL_ID node;
// its src[0..5] are the six Tile640 weight components (packed,
// page_scales, lane_scales, outlier row offsets/cols/vals). `row_width`
// is in_dim; `n_rows` is the number of weight rows to dequant (out_dim
// for MATMUL, n_experts*out_dim for MATMUL_ID). `name` identifies the
// tensor in the sidecar directory.
//
// `enc` is the backend's live compute encoder for `cmd_buf`; the dequant
// is encoded into the same command buffer as the matmul (no mid-graph
// waitUntilCompleted, which would deadlock the shared queue). A completed
// handler is added to `cmd_buf` so the readback + sidecar write happen
// after the GPU finishes the dequant. The handler runs asynchronously,
// so the sidecar file may appear shortly after the matmul returns.
//
// This path is the runtime-faithful L1 producer for Tile640: it captures
// the GPU dequant (including the sparse outlier addback the CPU
// `dequantize_row_tessera_t640` trait omits), so the L1 fitness reflects
// what the matmul actually consumes. No-op when the debug hook is off or
// any of the six weight components is missing.
void metal_dump_dequant_tile640(
        ggml_metal_device_t dev,
        ggml_metal_encoder_t enc,
        ggml_metal_cmd_buf_t cmd_buf,
        const struct ggml_tensor * op,
        int64_t row_width,
        int64_t n_rows,
        const char * name);

#ifdef __cplusplus
}
#endif
