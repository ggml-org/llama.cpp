#pragma once

#include "ggml-impl.h"

#include <stdint.h>
#include <stddef.h>

/**
 * @brief A collection of quantization and data type conversion utilities for the RKNPU backend.
 *
 * This namespace provides optimized functions for converting between floating-point
 * and various integer formats (FP16, INT8, INT4) required by the Rockchip NPU.
 * It also serves as a centralized place for dequantization routines.
 */
namespace rknpu2_quantization {

// --- Conversion from FP32 ---

/**
 * @brief Converts a row of FP32 values to FP16.
 * @param src Pointer to the source FP32 data.
 * @param dst Pointer to the destination FP16 (uint16_t) data.
 * @param n_elements The number of elements to convert.
 */
void convert_fp32_to_fp16(const float * src, uint16_t * dst, size_t n_elements);

/**
 * @brief Symmetrically quantizes a row of FP32 values to INT8.
 *
 * The quantization formula is: `dst[i] = round(src[i] / scale)`.
 * The caller is responsible for calculating the appropriate scale
 * (e.g., `amax / 127.0f`).
 *
 * @param src Pointer to the source FP32 data.
 * @param dst Pointer to the destination INT8 data.
 * @param n_elements The number of elements to quantize.
 * @param scale The quantization scale factor.
 */
void quantize_fp32_to_int8(const float * src, int8_t * dst, size_t n_elements, float scale);

/**
 * @brief Symmetrically quantizes a row of FP32 values to INT4 and packs them.
 *
 * The quantization formula is: `v = round(src[i] / scale)`. Values are clamped to [-8, 7].
 * Two INT4 values are packed into a single uint8_t.
 * The caller is responsible for calculating the appropriate scale
 * (e.g., `amax / 7.0f`).
 *
 * @param src Pointer to the source FP32 data.
 * @param dst Pointer to the destination packed INT4 (uint8_t) data.
 * @param n_elements The number of elements to quantize (must be a multiple of 2).
 * @param scale The quantization scale factor.
 */
void quantize_fp32_to_int4_packed(const float * src, uint8_t * dst, size_t n_elements, float scale);


// --- Dequantization to FP32 ---

/**
 * @brief Dequantizes a row of INT16 values to FP32.
 *
 * The dequantization formula is: `dst[i] = src[i] * scale`.
 *
 * @param src Pointer to the source INT16 data.
 * @param dst Pointer to the destination FP32 data.
 * @param n_elements The number of elements to dequantize.
 * @param scale The dequantization scale factor.
 */
void dequantize_int16_to_fp32(const int16_t * src, float * dst, size_t n_elements, float scale);

/**
 * @brief Dequantizes a row of INT32 values to FP32.
 *
 * The dequantization formula is: `dst[i] = src[i] * scale`.
 *
 * @param src Pointer to the source INT32 data.
 * @param dst Pointer to the destination FP32 data.
 * @param n_elements The number of elements to dequantize.
 * @param scale The dequantization scale factor.
 */
void dequantize_int32_to_fp32(const int32_t * src, float * dst, size_t n_elements, float scale);

} // namespace rknpu2_quantization