//
// cpu-dispatch.hpp — CPU compute path for data-local inference
// When unified cache evicts weights to host pinned memory, this dispatches
// layer computation to a SYCL CPU device instead of streaming to GPU.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_CPU_DISPATCH_HPP
#define GGML_SYCL_CPU_DISPATCH_HPP

#include "common.hpp"

// Dispatch a single ggml operation to the CPU SYCL device.
// Returns true if handled, false if the op is unsupported on CPU.
bool ggml_sycl_compute_forward_cpu(ggml_backend_sycl_context & ctx, struct ggml_tensor * dst);

// Drain any pending async staging events (call at boundary sync points).
void ggml_sycl_cpu_staging_drain();

// Clear persistent staging cache for leaf tensors.
// Call on graph shape change (new token count changes masks).
void ggml_sycl_cpu_staging_cache_clear();

// Register the original host (mmap) pointer for a weight tensor.
// Called from set_tensor when weight data is uploaded to the device.
// The CPU dispatch path uses this to access quantized weight data directly
// from the mmap'd GGUF file, avoiding dequantization when using vec_dot.
void ggml_sycl_cpu_dispatch_register_host_ptr(const char * name, const void * host_ptr, size_t size);

// Fused CPU operation handlers — eliminate intermediate staging round-trips.
// Returns true if fusion was applied, false to fall back to individual dispatch.
bool ggml_sycl_compute_fused_rms_norm_mul(ggml_backend_sycl_context & ctx,
                                           ggml_tensor * rms_dst, ggml_tensor * mul_dst);
bool ggml_sycl_compute_fused_add_rms_norm(ggml_backend_sycl_context & ctx,
                                            ggml_tensor * add_dst, ggml_tensor * rms_dst);

// Retained activation API — eliminates per-op staging overhead
// by keeping intermediate results in host scratch memory between
// consecutive CPU-dispatched ops within a layer block.
void   ggml_sycl_cpu_retained_init(int device, sycl::queue * gpu_q);
void   ggml_sycl_cpu_retained_cleanup();
bool   ggml_sycl_cpu_retained_active();
void * ggml_sycl_cpu_retained_alloc_output(const ggml_tensor * dst);
void   ggml_sycl_cpu_retained_flush_all(int device, sycl::queue * gpu_q);
void   ggml_sycl_cpu_retained_deactivate();

#endif // GGML_SYCL_CPU_DISPATCH_HPP
