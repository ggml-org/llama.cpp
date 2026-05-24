#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// TMA transfer between pinned system RAM and GPU VRAM.
// Falls back to cudaMemcpyAsync when TMA is unavailable.

typedef struct ggml_tma_transfer * ggml_tma_transfer_t;

// Initialize a TMA transfer descriptor.
// src_pinned: pinned RAM address (from ggml_backend_cpu_pinned_buffer_type)
// dst_vram:   GPU VRAM address (from cudaMalloc)
// num_elements: element count (e.g. float16/bf16 elements to transfer)
// elem_size:    element size in bytes (2 for float16/bf16, 4 for float32)
// stream:       CUDA stream for the transfer
// Returns true on success, false if TMA unavailable (will use memcpy fallback).
bool ggml_tma_init_transfer(ggml_tma_transfer_t * out,
    void * src_pinned,
    void * dst_vram,
    size_t num_elements,
    size_t elem_size,
    void * stream);  // cudaStream_t passed as void* to keep C-compatible

// Launch the transfer asynchronously on the configured stream.
void ggml_tma_launch_transfer(ggml_tma_transfer_t transfer);

// Free TMA transfer resources (descriptor device memory, etc).
void ggml_tma_free_transfer(ggml_tma_transfer_t transfer);

#ifdef __cplusplus
}
#endif
