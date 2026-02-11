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

// Register the original host (mmap) pointer for a weight tensor.
// Called from set_tensor when weight data is uploaded to the device.
// The CPU dispatch path uses this to access quantized weight data directly
// from the mmap'd GGUF file, avoiding dequantization when using vec_dot.
void ggml_sycl_cpu_dispatch_register_host_ptr(const char * name, const void * host_ptr, size_t size);

#endif // GGML_SYCL_CPU_DISPATCH_HPP
