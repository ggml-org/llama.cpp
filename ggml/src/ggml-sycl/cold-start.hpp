//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Cold-start hardware heuristics for SYCL kernel auto-tuning
//
// This module provides initial kernel configuration based on GPU hardware
// capabilities detected at runtime, before any runtime profiling data is
// available. These heuristics provide a reasonable starting point for
// auto-tuning.
//

#pragma once

#include <cstdint>
#include <string>

// Only include SYCL when building with SYCL support
// The detect_gpu_capabilities() function requires SYCL
#ifdef SYCL_LANGUAGE_VERSION
#    include <sycl/sycl.hpp>
#    define GGML_SYCL_COLD_START_HAS_SYCL 1
#endif

namespace ggml_sycl {

// ============================================================================
// GPU Family enumeration
// ============================================================================

enum class GPUFamily {
    UNKNOWN,         // Unknown or unsupported GPU
    ARC_ALCHEMIST,   // Intel Arc Alchemist (A-series: A770, A750, A580, A380, etc.)
    ARC_BATTLEMAGE,  // Intel Arc Battlemage (B-series: B580, B570, etc.)
};

// ============================================================================
// GPU Capabilities structure
// ============================================================================

struct GPUCapabilities {
    int         eu_count           = 0;      // Number of Execution Units (EUs)
    int         slm_size_kb        = 0;      // Shared Local Memory size in KB
    int         max_workgroup_size = 0;      // Maximum work-group size
    bool        has_dpas           = false;  // Device supports dpas (XMX) instructions
    std::string device_name;                 // Full device name string
    int         device_id = 0;               // Device index in the system

    // Derived properties (computed from hardware queries)
    int xecore_count() const {
        // Intel Arc has 8 EUs per XeCore (Vector Engine)
        constexpr int EUS_PER_XECORE = 8;
        return eu_count > 0 ? (eu_count / EUS_PER_XECORE) : 0;
    }
};

// ============================================================================
// Kernel Configuration structure
// ============================================================================

struct KernelConfig {
    // Tile dimensions for matrix operations
    int tile_m = 16;  // M dimension tile size
    int tile_n = 16;  // N dimension tile size
    int tile_k = 32;  // K dimension tile size

    // Hardware feature usage
    bool use_dpas = false;  // Use XMX/dpas instructions

    // Work distribution
    int wgs_per_xecore = 2;  // Work-groups per XeCore
};

// ============================================================================
// Function declarations
// ============================================================================

#ifdef GGML_SYCL_COLD_START_HAS_SYCL
/**
 * Detect GPU capabilities from a SYCL device.
 *
 * Queries hardware properties including:
 * - Execution unit count
 * - SLM size
 * - Maximum work-group size
 * - XMX/dpas support
 *
 * @param dev SYCL device to query
 * @return GPUCapabilities structure with detected values
 */
GPUCapabilities detect_gpu_capabilities(sycl::device & dev);
#endif

/**
 * Detect GPU family from capabilities.
 *
 * Identifies the GPU architecture family based on device name and
 * hardware characteristics.
 *
 * @param caps GPU capabilities (must have device_name populated)
 * @return GPUFamily enum value
 */
GPUFamily detect_gpu_family(const GPUCapabilities & caps);

/**
 * Derive initial kernel configuration from GPU capabilities.
 *
 * Uses hardware heuristics to determine a reasonable starting
 * configuration for kernel auto-tuning. The returned config
 * balances performance with safety (avoiding OOM, occupancy issues).
 *
 * Configuration table:
 * - Arc A770/A750 (512+ EUs): tile_m=64, tile_n=64, tile_k=32, use_dpas=true
 * - Arc A580 (256 EUs):       tile_m=32, tile_n=32, tile_k=32, use_dpas=true
 * - Arc B580 (Xe2):           tile_m=64, tile_n=64, tile_k=32, use_dpas=true
 * - Unknown:                  tile_m=16, tile_n=16, tile_k=32, use_dpas=false
 *
 * @param caps GPU capabilities from detect_gpu_capabilities()
 * @return KernelConfig with recommended initial settings
 */
KernelConfig derive_initial_config(const GPUCapabilities & caps);

/**
 * Get a human-readable name for a GPU family.
 *
 * @param family GPU family enum value
 * @return String name (e.g., "Arc Alchemist", "Arc Battlemage", "Unknown")
 */
const char * gpu_family_name(GPUFamily family);

}  // namespace ggml_sycl
