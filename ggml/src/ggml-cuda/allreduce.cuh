#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>

// Maximum chunk size (bytes per GPU) handled by one internal kernel launch.
// Larger tensors are reduced by issuing multiple chunked launches.
static constexpr size_t GGML_CUDA_AR_MAX_BYTES = 256 * 1024; // 256 KB

// Opaque pipeline context — owns all pinned buffers, streams, and events.
struct ggml_cuda_ar_pipeline;

// Allocate a pipeline for n_devices GPUs.
// devices[] holds the CUDA device IDs in rank order.
// max_bytes is the staging buffer size per device; must be at least as large
// as the largest tensor that will be reduced.
// Returns nullptr on allocation failure.
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(
    const int * devices, int n_devices, size_t max_bytes);

// Release all resources owned by the pipeline.
void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * pipeline);

// Execute an in-place AllReduce (sum) across tensors[0..n_devices-1].
// tensors[i] must live on the device managed by backends[i] and be
// contiguous F32, F16, or BF16.
// Returns true on success. Returns false when the tensor type or size is
// outside the currently supported range; the caller should fall back to
// another provider (NCCL or the meta-backend CPU reduce).
bool ggml_cuda_ar_allreduce(
    ggml_cuda_ar_pipeline * pipeline,
    ggml_backend_t        * backends,
    ggml_tensor           ** tensors);
