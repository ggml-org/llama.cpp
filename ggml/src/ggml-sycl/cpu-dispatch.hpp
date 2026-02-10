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

#endif // GGML_SYCL_CPU_DISPATCH_HPP
