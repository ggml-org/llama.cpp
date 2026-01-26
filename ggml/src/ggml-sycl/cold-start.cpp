//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Cold-start hardware heuristics implementation
//

#include "cold-start.hpp"

#include <algorithm>
#include <cctype>

namespace ggml_sycl {

// ============================================================================
// Helper functions
// ============================================================================

// Check if device name contains a substring (case-insensitive)
static bool name_contains(const std::string & name, const char * substr) {
    std::string lower_name   = name;
    std::string lower_substr = substr;

    // Convert to lowercase for case-insensitive comparison
    for (char & c : lower_name) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    for (char & c : lower_substr) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    return lower_name.find(lower_substr) != std::string::npos;
}

// ============================================================================
// GPU Family detection
// ============================================================================

GPUFamily detect_gpu_family(const GPUCapabilities & caps) {
    const std::string & name = caps.device_name;

    // Check for Arc Battlemage (B-series)
    // B580, B570, B50 Pro, etc.
    if (name_contains(name, "B580") || name_contains(name, "B570") || name_contains(name, "B50") ||
        (name_contains(name, "Arc") && name_contains(name, "Battlemage"))) {
        return GPUFamily::ARC_BATTLEMAGE;
    }

    // Check for Arc Alchemist (A-series)
    // A770, A750, A580, A380, A310, etc.
    if (name_contains(name, "A770") || name_contains(name, "A750") || name_contains(name, "A580") ||
        name_contains(name, "A380") || name_contains(name, "A310") ||
        (name_contains(name, "Arc") && name_contains(name, "Graphics"))) {
        return GPUFamily::ARC_ALCHEMIST;
    }

    return GPUFamily::UNKNOWN;
}

const char * gpu_family_name(GPUFamily family) {
    switch (family) {
        case GPUFamily::ARC_ALCHEMIST:
            return "Arc Alchemist";
        case GPUFamily::ARC_BATTLEMAGE:
            return "Arc Battlemage";
        case GPUFamily::UNKNOWN:
        default:
            return "Unknown";
    }
}

// ============================================================================
// GPU Capabilities detection (requires SYCL)
// ============================================================================

#ifdef GGML_SYCL_COLD_START_HAS_SYCL
GPUCapabilities detect_gpu_capabilities(sycl::device & dev) {
    GPUCapabilities caps;

    // Query device name
    caps.device_name = dev.get_info<sycl::info::device::name>();

    // Query compute units (EUs on Intel)
    caps.eu_count = static_cast<int>(dev.get_info<sycl::info::device::max_compute_units>());

    // Query SLM size (local memory)
    size_t slm_bytes = dev.get_info<sycl::info::device::local_mem_size>();
    caps.slm_size_kb = static_cast<int>(slm_bytes / 1024);

    // Query max work-group size
    caps.max_workgroup_size = static_cast<int>(dev.get_info<sycl::info::device::max_work_group_size>());

    // Check for XMX/dpas support
    caps.has_dpas = dev.has(sycl::aspect::ext_intel_matrix);

    return caps;
}
#endif

// ============================================================================
// Initial configuration derivation
// ============================================================================

KernelConfig derive_initial_config(const GPUCapabilities & caps) {
    KernelConfig config;

    // Default conservative config for unknown devices
    config.tile_m         = 16;
    config.tile_n         = 16;
    config.tile_k         = 32;
    config.use_dpas       = false;
    config.wgs_per_xecore = 2;

    GPUFamily family = detect_gpu_family(caps);

    switch (family) {
        case GPUFamily::ARC_BATTLEMAGE:
            // Arc Battlemage (B580, B570, etc.)
            // Xe2 architecture with improved XMX
            // Use aggressive tiles - Battlemage has efficient matrix engines
            config.tile_m         = 64;
            config.tile_n         = 64;
            config.tile_k         = 32;
            config.use_dpas       = caps.has_dpas;
            config.wgs_per_xecore = 2;
            break;

        case GPUFamily::ARC_ALCHEMIST:
            // Arc Alchemist (A770, A750, A580, etc.)
            // Tile size based on EU count
            if (caps.eu_count >= 512) {
                // High-end: A770, A750 (512 EUs)
                config.tile_m   = 64;
                config.tile_n   = 64;
                config.tile_k   = 32;
                config.use_dpas = caps.has_dpas;
            } else if (caps.eu_count >= 256) {
                // Mid-range: A580 (256 EUs)
                config.tile_m   = 32;
                config.tile_n   = 32;
                config.tile_k   = 32;
                config.use_dpas = caps.has_dpas;
            } else {
                // Entry-level: A380, A310 (128 EUs or less)
                config.tile_m   = 16;
                config.tile_n   = 32;
                config.tile_k   = 32;
                config.use_dpas = caps.has_dpas;
            }
            config.wgs_per_xecore = 2;
            break;

        case GPUFamily::UNKNOWN:
        default:
            // Keep conservative defaults
            // Don't enable dpas for unknown devices
            break;
    }

    // SLM constraint check
    // Ensure tile configuration fits in available SLM
    // For XMX: need space for A tile (M*K), B tile (K*N), and accumulators
    if (caps.slm_size_kb > 0) {
        size_t slm_bytes = static_cast<size_t>(caps.slm_size_kb) * 1024;

        // Estimate SLM usage: A tile + B tile + accumulators + scales
        // A tile: tile_m * tile_k * sizeof(int8_t)
        // B tile: tile_k * tile_n * sizeof(int8_t)
        // Accumulators: tile_m * tile_n * sizeof(int32_t)
        // Scale buffer: ~256 bytes
        auto estimate_slm_usage = [](int tm, int tn, int tk) -> size_t {
            size_t a_tile = static_cast<size_t>(tm) * tk;
            size_t b_tile = static_cast<size_t>(tk) * tn;
            size_t acc    = static_cast<size_t>(tm) * tn * sizeof(int32_t);
            size_t scales = 256;
            return a_tile + b_tile + acc + scales;
        };

        size_t needed = estimate_slm_usage(config.tile_m, config.tile_n, config.tile_k);

        // If current config doesn't fit, reduce tile sizes
        while (needed > slm_bytes && (config.tile_m > 16 || config.tile_n > 16)) {
            if (config.tile_m >= config.tile_n && config.tile_m > 16) {
                config.tile_m /= 2;
            } else if (config.tile_n > 16) {
                config.tile_n /= 2;
            } else {
                break;
            }
            needed = estimate_slm_usage(config.tile_m, config.tile_n, config.tile_k);
        }
    }

    return config;
}

}  // namespace ggml_sycl
