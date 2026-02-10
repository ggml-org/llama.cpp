//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_CPU_DISPATCH_HPP
#define GGML_SYCL_CPU_DISPATCH_HPP

#include "common.hpp"

//
// CPU compute path for data-local execution.
//
// When weights reside in host memory (PINNED_HOST or MMAP tier), these
// functions execute the operation on a CPU SYCL queue instead of
// transferring data to the GPU.  Phase 1 supports F32 and F16 types only.
//
// Each function returns true if the operation was handled on the CPU,
// or false if the tensor types / shapes are unsupported and the caller
// should fall back to the GPU path.
//

// MUL_MAT via oneDNN on the CPU queue.
// Supports F32 and F16 src0 (weights) with F32 src1 (activations).
bool cpu_compute_mul_mat(ggml_backend_sycl_context & ctx,
                         ggml_tensor * dst,
                         sycl::queue & cpu_queue);

// RMS_NORM via SYCL parallel_for on the CPU queue.
// F32 only.
bool cpu_compute_rms_norm(ggml_backend_sycl_context & ctx,
                          ggml_tensor * dst,
                          sycl::queue & cpu_queue);

// Element-wise ADD via SYCL parallel_for on the CPU queue.
// F32 only, contiguous tensors, matching shapes.
bool cpu_compute_add(ggml_backend_sycl_context & ctx,
                     ggml_tensor * dst,
                     sycl::queue & cpu_queue);

// Element-wise MUL via SYCL parallel_for on the CPU queue.
// F32 only, contiguous tensors, matching shapes.
bool cpu_compute_mul(ggml_backend_sycl_context & ctx,
                     ggml_tensor * dst,
                     sycl::queue & cpu_queue);

// Check whether a tensor's type combination is supported by the CPU path.
// Returns true if the op + types can be handled by cpu_compute_*.
bool cpu_dispatch_supported(const ggml_tensor * dst);

#endif // GGML_SYCL_CPU_DISPATCH_HPP
