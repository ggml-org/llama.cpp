//
// XMX ESIMD Common Infrastructure
//
// Shared utilities and configuration for XMX ESIMD kernels.
// All XMX kernels MUST use XMXConfig::from_capabilities() to get
// hardware-queried tile dimensions. NO hardcoded 8/16/32 values.
//
// MIT license
// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_XMX_ESIMD_COMMON_HPP
#define GGML_SYCL_XMX_ESIMD_COMMON_HPP

#include <sycl/sycl.hpp>

// Check for ESIMD support
#if __has_include(<sycl/ext/intel/esimd.hpp>)
#    define SYCL_XMX_ESIMD_AVAILABLE 1
#    include <sycl/ext/intel/esimd.hpp>
#else
#    define SYCL_XMX_ESIMD_AVAILABLE 0
#endif

// =============================================================================
// Standalone mode for unit tests
// When XMX_TEST_STANDALONE is defined, we provide minimal definitions
// instead of including common.hpp which has heavy dependencies.
// =============================================================================

#ifdef XMX_TEST_STANDALONE

// Minimal ggml_type enum for testing (matches ggml.h)
enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_COUNT,
};

// Minimal XMXCapabilities struct for testing
struct XMXCapabilities {
    bool supported = false;

    // Tile dimensions (queried from hardware)
    size_t M = 0;  // Expected: 8
    size_t N = 0;  // Expected: 16
    size_t K = 0;  // Expected: 32

    // Supported types
    bool supports_int8 = false;
    bool supports_fp16 = false;

    // Device memory info
    size_t slm_size = 0;  // Shared local memory per work-group

    // Derived optimal config
    int optimal_tiles_m = 1;
    int optimal_tiles_n = 1;
};

// Standalone query_xmx_capabilities implementation for tests
inline XMXCapabilities query_xmx_capabilities(sycl::device & dev) {
    XMXCapabilities caps;

    if (!dev.has(sycl::aspect::ext_intel_matrix)) {
        return caps;
    }
    caps.supported = true;

    // Query SLM size
    caps.slm_size = dev.get_info<sycl::info::device::local_mem_size>();

#    if defined(SYCL_EXT_ONEAPI_MATRIX_VERSION) && SYCL_EXT_ONEAPI_MATRIX_VERSION >= 1
    using namespace sycl::ext::oneapi::experimental;

    try {
        auto combinations = dev.get_info<info::device::matrix_combinations>();

        for (const auto & combo : combinations) {
            // Find int8 configuration (for Q8_0)
            if (combo.atype == matrix_type::sint8 && combo.btype == matrix_type::sint8) {
                caps.supports_int8 = true;
                caps.M             = combo.msize;
                caps.N             = combo.nsize;
                caps.K             = combo.ksize;
            }

            if (combo.atype == matrix_type::fp16 && combo.btype == matrix_type::fp16) {
                caps.supports_fp16 = true;
            }
        }
    } catch (const sycl::exception &) {
        // Query failed, use defaults
    }
#    else
    // Fallback: assume Intel Arc defaults
    caps.supports_int8 = true;
    caps.supports_fp16 = true;
    caps.M             = 8;
    caps.N             = 16;
    caps.K             = 32;
#    endif

    // Compute optimal tile counts
    if (caps.M > 0 && caps.N > 0) {
        caps.optimal_tiles_m = std::min(4, static_cast<int>(32 / caps.M));
        caps.optimal_tiles_n = std::min(4, static_cast<int>(64 / caps.N));
    }

    return caps;
}

#else  // !XMX_TEST_STANDALONE

// Include full common.hpp for production use
#    include "common.hpp"

#endif  // XMX_TEST_STANDALONE

namespace ggml_sycl_xmx {

// =============================================================================
// XMXConfig - Hardware-aware XMX configuration
//
// CRITICAL: All values come from XMXCapabilities queried from hardware.
// DO NOT add hardcoded fallbacks here - use query_xmx_capabilities() instead.
// =============================================================================

class XMXConfig {
  public:
    // Factory method - the ONLY way to create XMXConfig
    // Uses hardware-queried capabilities, no hardcoding
    static XMXConfig from_capabilities(const XMXCapabilities & caps) {
        XMXConfig config;
        config.m_supported       = caps.supported;
        config.m_tile_m          = caps.M;
        config.m_tile_n          = caps.N;
        config.m_tile_k          = caps.K;
        config.m_supports_int8   = caps.supports_int8;
        config.m_supports_fp16   = caps.supports_fp16;
        config.m_slm_size        = caps.slm_size;
        config.m_optimal_tiles_m = caps.optimal_tiles_m;
        config.m_optimal_tiles_n = caps.optimal_tiles_n;
        return config;
    }

    // Accessors for tile dimensions
    [[nodiscard]] size_t tile_m() const { return m_tile_m; }

    [[nodiscard]] size_t tile_n() const { return m_tile_n; }

    [[nodiscard]] size_t tile_k() const { return m_tile_k; }

    // Hardware support checks
    [[nodiscard]] bool is_supported() const { return m_supported; }

    [[nodiscard]] bool supports_int8() const { return m_supports_int8; }

    [[nodiscard]] bool supports_fp16() const { return m_supports_fp16; }

    // Quantization type support check
    // Maps ggml_type to XMX hardware capability
    [[nodiscard]] bool supports_qtype(ggml_type type) const {
        if (!m_supported) {
            return false;
        }

        switch (type) {
            // Q4_0 and Q8_0 use int8 XMX path
            // Q4_0 is dequantized to int8 before XMX
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q8_0:
                return m_supports_int8;

            // MXFP4 (IQ4_NL) uses int8 XMX with special dequantization
            // The 4-bit values are unpacked to int8 for XMX accumulation
            case GGML_TYPE_IQ4_NL:
                return m_supports_int8;

            // F16 uses native fp16 XMX path
            case GGML_TYPE_F16:
                return m_supports_fp16;

            // Other types not supported via XMX
            default:
                return false;
        }
    }

    // SLM and tiling info
    [[nodiscard]] size_t slm_size() const { return m_slm_size; }

    [[nodiscard]] int optimal_tiles_m() const { return m_optimal_tiles_m; }

    [[nodiscard]] int optimal_tiles_n() const { return m_optimal_tiles_n; }

    // Compute effective tile sizes for a given quantization type
    // Returns the number of elements in the K dimension that map to one XMX K tile
    [[nodiscard]] size_t k_elements_per_tile(ggml_type type) const {
        if (!m_supported || m_tile_k == 0) {
            return 0;
        }

        switch (type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_IQ4_NL:
                // For quantized types, K dimension is in elements
                return m_tile_k;

            case GGML_TYPE_F16:
                // For fp16, K tile is in fp16 elements
                return m_tile_k;

            default:
                return 0;
        }
    }

  private:
    XMXConfig() = default;

    bool   m_supported       = false;
    size_t m_tile_m          = 0;
    size_t m_tile_n          = 0;
    size_t m_tile_k          = 0;
    bool   m_supports_int8   = false;
    bool   m_supports_fp16   = false;
    size_t m_slm_size        = 0;
    int    m_optimal_tiles_m = 1;
    int    m_optimal_tiles_n = 1;
};

// =============================================================================
// ESIMD Helper Functions
// =============================================================================

#if SYCL_XMX_ESIMD_AVAILABLE

namespace esimd = sycl::ext::intel::esimd;

// -----------------------------------------------------------------------------
// Horizontal sum reduction for simd<float, N>
// Uses tree reduction pattern for efficiency
// -----------------------------------------------------------------------------

template <int N> SYCL_ESIMD_FUNCTION float esimd_hsum(esimd::simd<float, N> v) {
    static_assert(N > 0 && (N & (N - 1)) == 0, "N must be a power of 2");

    // Tree reduction
    if constexpr (N == 1) {
        return v[0];
    } else if constexpr (N == 2) {
        return v[0] + v[1];
    } else if constexpr (N == 4) {
        esimd::simd<float, 2> v2 = v.template select<2, 1>(0) + v.template select<2, 1>(2);
        return v2[0] + v2[1];
    } else if constexpr (N == 8) {
        esimd::simd<float, 4> v4 = v.template select<4, 1>(0) + v.template select<4, 1>(4);
        esimd::simd<float, 2> v2 = v4.template select<2, 1>(0) + v4.template select<2, 1>(2);
        return v2[0] + v2[1];
    } else if constexpr (N == 16) {
        esimd::simd<float, 8> v8 = v.template select<8, 1>(0) + v.template select<8, 1>(8);
        esimd::simd<float, 4> v4 = v8.template select<4, 1>(0) + v8.template select<4, 1>(4);
        esimd::simd<float, 2> v2 = v4.template select<2, 1>(0) + v4.template select<2, 1>(2);
        return v2[0] + v2[1];
    } else if constexpr (N == 32) {
        esimd::simd<float, 16> v16 = v.template select<16, 1>(0) + v.template select<16, 1>(16);
        esimd::simd<float, 8>  v8  = v16.template select<8, 1>(0) + v16.template select<8, 1>(8);
        esimd::simd<float, 4>  v4  = v8.template select<4, 1>(0) + v8.template select<4, 1>(4);
        esimd::simd<float, 2>  v2  = v4.template select<2, 1>(0) + v4.template select<2, 1>(2);
        return v2[0] + v2[1];
    } else if constexpr (N == 64) {
        esimd::simd<float, 32> v32 = v.template select<32, 1>(0) + v.template select<32, 1>(32);
        esimd::simd<float, 16> v16 = v32.template select<16, 1>(0) + v32.template select<16, 1>(16);
        esimd::simd<float, 8>  v8  = v16.template select<8, 1>(0) + v16.template select<8, 1>(8);
        esimd::simd<float, 4>  v4  = v8.template select<4, 1>(0) + v8.template select<4, 1>(4);
        esimd::simd<float, 2>  v2  = v4.template select<2, 1>(0) + v4.template select<2, 1>(2);
        return v2[0] + v2[1];
    } else {
        // Generic fallback for larger sizes
        float sum = 0.0f;
#    pragma unroll
        for (int i = 0; i < N; i++) {
            sum += v[i];
        }
        return sum;
    }
}

// -----------------------------------------------------------------------------
// MXFP4 Nibble Unpacking
//
// MXFP4 format: 16 packed bytes -> 32 4-bit values
// Each byte contains 2 nibbles (low and high)
// Output: unpacked[0..15] = low nibbles, unpacked[16..31] = high nibbles
// -----------------------------------------------------------------------------

SYCL_ESIMD_FUNCTION esimd::simd<int8_t, 32> unpack_nibbles_mxfp4(esimd::simd<uint8_t, 16> packed) {
    // Extract low nibbles (bits 0-3)
    esimd::simd<uint8_t, 16> lo = packed & 0x0F;

    // Extract high nibbles (bits 4-7)
    esimd::simd<uint8_t, 16> hi = packed >> 4;

    // Combine into result: [lo0..lo15, hi0..hi15]
    esimd::simd<int8_t, 32> result;

// Copy low nibbles to result[0..15]
#    pragma unroll
    for (int i = 0; i < 16; i++) {
        result[i] = static_cast<int8_t>(lo[i]);
    }

// Copy high nibbles to result[16..31]
#    pragma unroll
    for (int i = 0; i < 16; i++) {
        result[i + 16] = static_cast<int8_t>(hi[i]);
    }

    return result;
}

// -----------------------------------------------------------------------------
// MXFP4 Nibble Unpacking with Zero-Point Adjustment
//
// For MXFP4 dequantization, values need zero-point adjustment:
// dequant_value = (nibble - 8) * scale
// This version returns nibbles adjusted by -8
// -----------------------------------------------------------------------------

SYCL_ESIMD_FUNCTION esimd::simd<int8_t, 32> unpack_nibbles_mxfp4_zp(esimd::simd<uint8_t, 16> packed) {
    // Extract low nibbles (bits 0-3) and adjust by -8
    esimd::simd<int8_t, 16> lo;
#    pragma unroll
    for (int i = 0; i < 16; i++) {
        lo[i] = static_cast<int8_t>((packed[i] & 0x0F)) - 8;
    }

    // Extract high nibbles (bits 4-7) and adjust by -8
    esimd::simd<int8_t, 16> hi;
#    pragma unroll
    for (int i = 0; i < 16; i++) {
        hi[i] = static_cast<int8_t>((packed[i] >> 4)) - 8;
    }

    // Combine into result
    esimd::simd<int8_t, 32> result;

#    pragma unroll
    for (int i = 0; i < 16; i++) {
        result[i]      = lo[i];
        result[i + 16] = hi[i];
    }

    return result;
}

// -----------------------------------------------------------------------------
// Q4_0 Dequantization Helper
//
// Converts packed Q4_0 nibbles to float with scale and zero-point
// Input: 16 packed bytes (32 4-bit values)
// Output: 32 float values
// -----------------------------------------------------------------------------

SYCL_ESIMD_FUNCTION esimd::simd<float, 32> dequant_q4_0_esimd(esimd::simd<uint8_t, 16> packed, float scale) {
    esimd::simd<int8_t, 32> unpacked = unpack_nibbles_mxfp4_zp(packed);

    esimd::simd<float, 32> result;
#    pragma unroll
    for (int i = 0; i < 32; i++) {
        result[i] = static_cast<float>(unpacked[i]) * scale;
    }

    return result;
}

// -----------------------------------------------------------------------------
// Q8_0 Block Load Helper
//
// Loads a Q8_0 block's quantized values as int8
// -----------------------------------------------------------------------------

SYCL_ESIMD_FUNCTION esimd::simd<int8_t, 32> load_q8_0_qs(const int8_t * qs_ptr) {
    esimd::simd<int8_t, 32> result;
    result.copy_from(qs_ptr);
    return result;
}

#endif  // SYCL_XMX_ESIMD_AVAILABLE

// =============================================================================
// Dispatch Helpers
// =============================================================================

// Check if XMX ESIMD path should be used for a given type and capabilities
inline bool should_use_xmx_esimd(const XMXCapabilities & caps, ggml_type type) {
#if SYCL_XMX_ESIMD_AVAILABLE
    if (!caps.supported) {
        return false;
    }

    XMXConfig config = XMXConfig::from_capabilities(caps);
    return config.supports_qtype(type);
#else
    (void) caps;
    (void) type;
    return false;
#endif
}

}  // namespace ggml_sycl_xmx

#endif  // GGML_SYCL_XMX_ESIMD_COMMON_HPP
