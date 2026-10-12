#pragma once

#include "common.cuh"

// Decode attention for Q8_0 K / Q4_0 V with head size 256, GQA 6 and 1-4 query tokens (MTP verify) on RDNA4.
// Split-K flash decoding with WMMA and in-register dequantization; much faster than the stock vec/tile paths for
// quantized KV at these shapes. Used automatically where it applies; GGML_CUDA_DISABLE_PAGED_DECODE turns it off for
// contiguous K/V. It is the only CUDA path for paged K/V (ggml_flash_attn_ext_set_pages).
bool ggml_cuda_fattn_paged_decode_enabled(int device, const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_paged_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Cross-queue sync (ggml_flash_attn_ext_set_sync) around the stock kernels: wait for the producer before them and
// release the data after them. No-ops without sync.
void ggml_cuda_fattn_sync_wait(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
void ggml_cuda_fattn_sync_release(ggml_backend_cuda_context & ctx, const ggml_tensor * dst);
