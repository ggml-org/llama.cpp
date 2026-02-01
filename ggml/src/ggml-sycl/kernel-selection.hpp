// Kernel selection types and functions for SYCL MUL_MAT operations.
// This header exposes the kernel selection interface for testing.

#pragma once

#include <optional>
#include "ggml.h"

// Kernel types for MUL_MAT operations
enum class ggml_sycl_mul_mat_kernel {
    DMMV_SOA,
    DMMV_COALESCED,
    MMVQ_COALESCED,
    MMVQ_SOA,
    MMVQ_AOS,
    XMX_GEMM_TILED,
    XMX_GEMM_AOS,
    MMQ_COALESCED,
    MMQ_SOA,
    MMQ_AOS,
    ONEDNN_AOS,
    UNIFIED_MATMUL,
};

// Parse GGML_SYCL_FORCE_KERNEL environment variable
// Returns nullopt if unset or invalid value
std::optional<ggml_sycl_mul_mat_kernel> ggml_sycl_parse_force_kernel();

// Unified kernel selection for MUL_MAT operations
// Both layout selection (at finalization time) and dispatch (at runtime)
// should call this function to ensure consistent priority logic.
//
// Parameters:
//   src0: Weight tensor (quantized type like Q4_0, Q8_0)
//   src1: Activation tensor (typically F32)
//   device: Device ID for the operation
//   force_kernel: Optional override from env var or benchmark
//
// Returns:
//   The preferred kernel, or nullopt if no eligible kernel found
std::optional<ggml_sycl_mul_mat_kernel> ggml_sycl_select_preferred_kernel(
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    int device,
    std::optional<ggml_sycl_mul_mat_kernel> force_kernel = std::nullopt);

// Get the kernel name as a string (for debugging/logging)
const char * ggml_sycl_mul_mat_kernel_name(ggml_sycl_mul_mat_kernel kernel);
