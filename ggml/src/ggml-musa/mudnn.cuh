#pragma once

#include <mudnncxx/mudnn.h>

#include "ggml-cuda/common.cuh"
#include "ggml.h"

// Returns a human-readable error string for mudnn::Status
const char * mudnn_status_to_string(musa::dnn::Status err);

// Error checking macro for MUDNN calls
#define MUDNN_CHECK(err) CUDA_CHECK_GEN(err, musa::dnn::Status::SUCCESS, mudnn_status_to_string)

// Thread-safe cache of mudnn::Handle objects per device
musa::dnn::Handle * get_cached_handle(int device_id);

// Asynchronously copies data from src tensor to dst tensor using the provided context.
// Returns a musaError_t indicating success or failure.
musaError_t mudnnMemcpyAsync(
    ggml_backend_cuda_context &ctx,
    const ggml_tensor *dst,
    const ggml_tensor *src
);

// Returns true for the quantized types whose matrix multiplication has a measured win through the
// muDNN INT8 path.
bool mudnnQuantMulMatSupported(ggml_type type);

// Computes dst (ne11 x ne01, F32) = src0 (ne01 x ne10, quantized) * src1 (ne11 x ne10, transposed,
// F16 or F32) by re-encoding both operands as INT8 inside the MUSA backend and running a muDNN
// low-precision MatMul. Returns false when the shape or the type is not handled, so the caller can
// fall back.
bool mudnnMulMatQuant(
    ggml_backend_cuda_context &ctx,
    const ggml_tensor *src0,
    const ggml_tensor *src1,
    ggml_tensor *dst
);

#if defined(GGML_MUSA_MUDNN_MM_Q)
size_t mudnnQuantMulMatCacheSize(const struct ggml_tensor * src0);
bool   mudnnQuantMulMatCacheHit(const struct ggml_tensor * src0, int8_t ** wq, float ** scale_w);
bool   mudnnQuantMulMatCacheRepack(int device, musaStream_t stream, struct ggml_tensor * src0);
#endif // GGML_MUSA_MUDNN_MM_Q
