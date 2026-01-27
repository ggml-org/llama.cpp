//
// MIT license
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// XMX Quantization Loader Infrastructure
//
// Template-based quantization loader for XMX-optimized matrix operations.
// Provides:
//   - QuantTraits<T>: Compile-time type traits for quantization formats
//   - XMXTileLoader<T, N, K>: Template class for tile-based weight loading
//   - dequant_block_*: Dequantization functions for each format
//
// Supported quantization types: Q4_0, Q8_0, Q6_K, Q4_K
//

#ifndef GGML_SYCL_XMX_QUANT_LOADERS_HPP
#define GGML_SYCL_XMX_QUANT_LOADERS_HPP

#include <cstdint>
#include <cstring>
#include <type_traits>

// Include GGML common for block type definitions
// Ensure these are defined before including this header
#ifndef GGML_COMMON_DECL
#    define GGML_COMMON_DECL_CPP
#    define GGML_COMMON_IMPL_CPP
#    include "ggml-common.h"
#endif

#include "ggml.h"

namespace xmx {

// =============================================================================
// XMX Hardware Constants
// =============================================================================

/**
 * @brief Intel XMX hardware tile dimensions
 *
 * These are the native tile sizes supported by Intel Xe Matrix eXtensions:
 *   - M: 8 (output rows per dpas instruction, can be extended with repeat count)
 *   - N: 16 (output columns per dpas instruction)
 *   - K: 32 (reduction dimension per dpas instruction for INT8/FP16)
 */
namespace hw {
static constexpr int XMX_TILE_M = 8;   // M dimension (can use repeat count 1-8)
static constexpr int XMX_TILE_N = 16;  // N dimension
static constexpr int XMX_TILE_K = 32;  // K dimension for INT8/FP16

// Common repeat counts for dpas instruction
static constexpr int DPAS_REPEAT_1 = 1;
static constexpr int DPAS_REPEAT_2 = 2;
static constexpr int DPAS_REPEAT_4 = 4;
static constexpr int DPAS_REPEAT_8 = 8;
}  // namespace hw

// =============================================================================
// Type Trait Helpers
// =============================================================================

/**
 * @brief Check if a quantization type is supported
 */
template <int QuantType> struct is_supported_quant_type : std::false_type {};

template <> struct is_supported_quant_type<GGML_TYPE_Q4_0> : std::true_type {};

template <> struct is_supported_quant_type<GGML_TYPE_Q8_0> : std::true_type {};

template <> struct is_supported_quant_type<GGML_TYPE_Q6_K> : std::true_type {};

template <> struct is_supported_quant_type<GGML_TYPE_Q4_K> : std::true_type {};

template <int QuantType> static constexpr bool is_supported_quant_type_v = is_supported_quant_type<QuantType>::value;

// =============================================================================
// QuantTraits - Compile-time type traits for quantization formats
// =============================================================================

/**
 * @brief Primary template for quantization type traits (undefined for unsupported types)
 *
 * Specializations provide:
 *   - block_size: Number of values per block after dequantization
 *   - bytes_per_block: Size of quantized block in bytes
 *   - bits_per_weight: Average bits per weight value
 *   - has_super_block: Whether this is a K-quant type with super-blocks
 *   - block_type: The underlying C struct type for the block
 */
template <int QuantType> struct QuantTraits;

/**
 * @brief Q4_0 quantization traits
 *
 * Q4_0: 4-bit quantization, 32 values per block
 * Block layout: [d: fp16 (2 bytes)][qs: 16 bytes of packed 4-bit values]
 * Total: 18 bytes per 32 values = 4.5 bits/weight
 */
template <> struct QuantTraits<GGML_TYPE_Q4_0> {
    static constexpr int  block_size      = QK4_0;               // 32
    static constexpr int  bytes_per_block = sizeof(block_q4_0);  // 18
    static constexpr int  bits_per_weight = 4;
    static constexpr bool has_super_block = false;

    using block_type = block_q4_0;
};

/**
 * @brief Q8_0 quantization traits
 *
 * Q8_0: 8-bit quantization, 32 values per block
 * Block layout: [d: fp16 (2 bytes)][qs: 32 bytes of int8 values]
 * Total: 34 bytes per 32 values = 8.5 bits/weight
 */
template <> struct QuantTraits<GGML_TYPE_Q8_0> {
    static constexpr int  block_size      = QK8_0;               // 32
    static constexpr int  bytes_per_block = sizeof(block_q8_0);  // 34
    static constexpr int  bits_per_weight = 8;
    static constexpr bool has_super_block = false;

    using block_type = block_q8_0;
};

/**
 * @brief Q6_K quantization traits
 *
 * Q6_K: 6-bit K-quant, 256 values per super-block
 * Block layout: [ql: 128 bytes][qh: 64 bytes][scales: 16 bytes][d: fp16]
 * Total: 210 bytes per 256 values = 6.5625 bits/weight
 */
template <> struct QuantTraits<GGML_TYPE_Q6_K> {
    static constexpr int  block_size      = QK_K;                // 256
    static constexpr int  bytes_per_block = sizeof(block_q6_K);  // 210
    static constexpr int  bits_per_weight = 6;
    static constexpr bool has_super_block = true;

    using block_type = block_q6_K;
};

/**
 * @brief Q4_K quantization traits
 *
 * Q4_K: 4-bit K-quant, 256 values per super-block
 * Block layout: [d: fp16][dmin: fp16][scales: 12 bytes][qs: 128 bytes]
 * Total: 144 bytes per 256 values = 4.5 bits/weight
 */
template <> struct QuantTraits<GGML_TYPE_Q4_K> {
    static constexpr int  block_size      = QK_K;                // 256
    static constexpr int  bytes_per_block = sizeof(block_q4_K);  // 144
    static constexpr int  bits_per_weight = 4;
    static constexpr bool has_super_block = true;

    using block_type = block_q4_K;
};

// =============================================================================
// Half-precision conversion utilities
// =============================================================================

/**
 * @brief Convert fp16 (stored as uint16_t) to float
 *
 * Uses standard IEEE 754 half-precision format.
 */
inline float fp16_to_float(uint16_t h) {
    // IEEE 754 half-precision to single-precision conversion
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            // Zero
            f = sign << 31;
        } else {
            // Denormal - normalize it
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            exp++;
            mant &= ~0x400;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        // Inf or NaN
        f = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        // Normal number
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

// =============================================================================
// Dequantization Functions
// =============================================================================

/**
 * @brief Dequantize a single Q4_0 block to float values
 *
 * Q4_0 dequantization:
 *   - Each byte contains 2 packed 4-bit values (low and high nibble)
 *   - Values are offset by 8 (range 0-15 -> -8 to +7)
 *   - Scale factor d is applied after offset
 *
 * @param block  Pointer to Q4_0 block
 * @param output Output array of 32 float values
 */
inline void dequant_block_q4_0(const block_q4_0 * block, float * output) {
    // Get scale factor as float
    uint16_t d_bits;
    memcpy(&d_bits, &block->d, sizeof(uint16_t));
    const float d = fp16_to_float(d_bits);

    // Unpack and dequantize
    for (int i = 0; i < 16; i++) {
        const uint8_t qs = block->qs[i];
        const int     lo = (qs & 0x0F) - 8;
        const int     hi = (qs >> 4) - 8;

        output[i]      = static_cast<float>(lo) * d;
        output[i + 16] = static_cast<float>(hi) * d;
    }
}

/**
 * @brief Dequantize a single Q8_0 block to float values
 *
 * Q8_0 dequantization:
 *   - Each int8 value is already in signed form (-128 to +127)
 *   - Scale factor d is applied directly
 *
 * @param block  Pointer to Q8_0 block
 * @param output Output array of 32 float values
 */
inline void dequant_block_q8_0(const block_q8_0 * block, float * output) {
    // Get scale factor as float
    uint16_t d_bits;
    memcpy(&d_bits, &block->d, sizeof(uint16_t));
    const float d = fp16_to_float(d_bits);

    // Dequantize
    for (int i = 0; i < 32; i++) {
        output[i] = static_cast<float>(block->qs[i]) * d;
    }
}

// =============================================================================
// XMXTileLoader - Template class for tile-based weight loading
// =============================================================================

/**
 * @brief XMX-optimized tile loader for quantized weights
 *
 * Template parameters:
 *   - QuantType: GGML_TYPE_* constant for the quantization format
 *   - TileN_: Number of rows (N dimension) in the tile
 *   - TileK_: Number of columns (K dimension) in the tile
 *
 * The TileK dimension should be a multiple of the block_size for efficient loading.
 *
 * Usage:
 *   using Loader = XMXTileLoader<GGML_TYPE_Q4_0, 16, 32>;
 *   Loader::dequant_block(block_ptr, output);
 */
template <int QuantType, int TileN_, int TileK_> class XMXTileLoader {
  public:
    using Traits     = QuantTraits<QuantType>;
    using block_type = typename Traits::block_type;

    // Tile dimensions exposed as static constants
    static constexpr int TileN = TileN_;
    static constexpr int TileK = TileK_;

    // Derived constants
    static constexpr int blocks_per_tile_k     = TileK / Traits::block_size;
    static constexpr int total_blocks_per_tile = TileN * blocks_per_tile_k;

    // Compile-time validation: TileK must be a multiple of block_size
    static_assert(TileK % Traits::block_size == 0, "TileK must be a multiple of block_size for aligned loading");

    // Compile-time validation: TileN and TileK must be positive
    static_assert(TileN > 0, "TileN must be positive");
    static_assert(TileK > 0, "TileK must be positive");

    /**
     * @brief Dequantize a single block using this loader's format
     *
     * Delegates to the appropriate dequant_block_* function based on QuantType.
     *
     * @param block  Pointer to quantized block
     * @param output Output array (size = Traits::block_size floats)
     */
    static void dequant_block(const block_type * block, float * output);
};

// Specialization for Q4_0
template <int TileN_, int TileK_> class XMXTileLoader<GGML_TYPE_Q4_0, TileN_, TileK_> {
  public:
    using Traits     = QuantTraits<GGML_TYPE_Q4_0>;
    using block_type = typename Traits::block_type;

    // Tile dimensions exposed as static constants
    static constexpr int TileN = TileN_;
    static constexpr int TileK = TileK_;

    static constexpr int blocks_per_tile_k     = TileK / Traits::block_size;
    static constexpr int total_blocks_per_tile = TileN * blocks_per_tile_k;

    static_assert(TileK_ % Traits::block_size == 0, "TileK must be a multiple of block_size for aligned loading");
    static_assert(TileN_ > 0, "TileN must be positive");
    static_assert(TileK_ > 0, "TileK must be positive");

    static void dequant_block(const block_type * block, float * output) { dequant_block_q4_0(block, output); }
};

// Specialization for Q8_0
template <int TileN_, int TileK_> class XMXTileLoader<GGML_TYPE_Q8_0, TileN_, TileK_> {
  public:
    using Traits     = QuantTraits<GGML_TYPE_Q8_0>;
    using block_type = typename Traits::block_type;

    // Tile dimensions exposed as static constants
    static constexpr int TileN = TileN_;
    static constexpr int TileK = TileK_;

    static constexpr int blocks_per_tile_k     = TileK / Traits::block_size;
    static constexpr int total_blocks_per_tile = TileN * blocks_per_tile_k;

    static_assert(TileK_ % Traits::block_size == 0, "TileK must be a multiple of block_size for aligned loading");
    static_assert(TileN_ > 0, "TileN must be positive");
    static_assert(TileK_ > 0, "TileK must be positive");

    static void dequant_block(const block_type * block, float * output) { dequant_block_q8_0(block, output); }
};

// Specialization for Q6_K
template <int TileN_, int TileK_> class XMXTileLoader<GGML_TYPE_Q6_K, TileN_, TileK_> {
  public:
    using Traits     = QuantTraits<GGML_TYPE_Q6_K>;
    using block_type = typename Traits::block_type;

    // Tile dimensions exposed as static constants
    static constexpr int TileN = TileN_;
    static constexpr int TileK = TileK_;

    static constexpr int blocks_per_tile_k     = TileK / Traits::block_size;
    static constexpr int total_blocks_per_tile = TileN * blocks_per_tile_k;

    static_assert(TileK_ % Traits::block_size == 0, "TileK must be a multiple of block_size for aligned loading");
    static_assert(TileN_ > 0, "TileN must be positive");
    static_assert(TileK_ > 0, "TileK must be positive");

    // Q6_K dequantization is more complex - placeholder for now
    static void dequant_block(const block_type * block, float * output) {
        // TODO: Implement Q6_K dequantization
        (void) block;
        (void) output;
    }
};

// Specialization for Q4_K
template <int TileN_, int TileK_> class XMXTileLoader<GGML_TYPE_Q4_K, TileN_, TileK_> {
  public:
    using Traits     = QuantTraits<GGML_TYPE_Q4_K>;
    using block_type = typename Traits::block_type;

    // Tile dimensions exposed as static constants
    static constexpr int TileN = TileN_;
    static constexpr int TileK = TileK_;

    static constexpr int blocks_per_tile_k     = TileK / Traits::block_size;
    static constexpr int total_blocks_per_tile = TileN * blocks_per_tile_k;

    static_assert(TileK_ % Traits::block_size == 0, "TileK must be a multiple of block_size for aligned loading");
    static_assert(TileN_ > 0, "TileN must be positive");
    static_assert(TileK_ > 0, "TileK must be positive");

    // Q4_K dequantization is more complex - placeholder for now
    static void dequant_block(const block_type * block, float * output) {
        // TODO: Implement Q4_K dequantization
        (void) block;
        (void) output;
    }
};

}  // namespace xmx

#endif  // GGML_SYCL_XMX_QUANT_LOADERS_HPP
