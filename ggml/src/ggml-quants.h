#pragma once

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

// NOTE: these functions are defined as GGML_API because they used by the CPU backend

// Quantization
GGML_API void quantize_row_q4_0_ref(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q4_1_ref(const float * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_0_ref(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_1_ref(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_0_ref(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_1_ref(const float * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_mxfp4_ref(const float * GGML_RESTRICT x, block_mxfp4 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_nvfp4_ref(const float * GGML_RESTRICT x, block_nvfp4 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_mxfp8_ref(const float * GGML_RESTRICT x, block_mxfp8 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_mxfp6_ref(const float * GGML_RESTRICT x, block_mxfp6 * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_q2_K_ref(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q3_K_ref(const float * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q4_K_ref(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_K_ref(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q6_K_ref(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_K_ref(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_tq1_0_ref(const float * GGML_RESTRICT x, block_tq1_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tq2_0_ref(const float * GGML_RESTRICT x, block_tq2_0 * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_iq3_xxs_ref(const float * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq4_nl_ref (const float * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq4_xs_ref (const float * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq3_s_ref  (const float * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq2_s_ref  (const float * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);

// Dequantization
GGML_API void dequantize_row_q4_0(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_1(const block_q4_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_0(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_1(const block_q5_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
//GGML_API void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_mxfp4(const block_mxfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_nvfp4(const block_nvfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_mxfp8(const block_mxfp8 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_mxfp6(const block_mxfp6 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

// SoA (Struct-of-Arrays) quantize/dequantize — canonical reference for flash attention.
// Layout: [qs contiguous][e8m0 contiguous] per row. Same total bytes as AoS.
GGML_API void quantize_row_mxfp4_soa    (const float * GGML_RESTRICT x, void * GGML_RESTRICT dst, int64_t k);
GGML_API void dequantize_row_mxfp4_soa  (const void * GGML_RESTRICT src, float * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_mxfp8_soa    (const float * GGML_RESTRICT x, void * GGML_RESTRICT dst, int64_t k);
GGML_API void dequantize_row_mxfp8_soa  (const void * GGML_RESTRICT src, float * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_mxfp6_soa(const float * GGML_RESTRICT x, void * GGML_RESTRICT dst, int64_t k);
GGML_API void dequantize_row_mxfp6_soa(const void * GGML_RESTRICT src, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_q2_K(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q3_K(const block_q3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_K(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_K(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q8_K(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_tq1_0(const block_tq1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tq2_0(const block_tq2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_iq2_xxs(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq2_xs (const block_iq2_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq2_s  (const block_iq2_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq3_xxs(const block_iq3_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq1_s  (const block_iq1_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq1_m  (const block_iq1_m   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq4_nl (const block_iq4_nl  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq4_xs (const block_iq4_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq3_s  (const block_iq3_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
GGML_API size_t quantize_iq2_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq2_xs (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq2_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq3_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq1_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq1_m  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq4_nl (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq4_xs (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq3_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API size_t quantize_tq1_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tq2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API size_t quantize_q2_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q3_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q4_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q5_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q6_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q4_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q5_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q5_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q8_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API size_t quantize_mxfp4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_nvfp4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_mxfp8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_mxfp6(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

//
// MXFP element-level conversion functions (reference implementations)
//
// These implement the OCP Microscaling (MX) format element types as defined in:
//   OCP Microscaling Formats (MX) Specification v1.0, Sep 2023
//   https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
//
// Each MX block contains 32 elements sharing a single E8M0 exponent. The element
// types define the per-element mantissa format within each block.
//
// All converters use IEEE-754 bit manipulation for exact results (no floating-point
// rounding in the conversion itself). Quantization functions use round-to-nearest-even
// (RNE) per the MX specification.
//
// GPU backends (CUDA, Metal, Vulkan) provide their own optimized versions — these
// functions serve as the canonical reference and are used by the CPU backend.
//

// FP8 E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa bits
// Range: ±[2^-9, 448], NaN: exp=15 mant=7 (only NaN encoding)
// Ref: OCP MX v1.0 §4.2
GGML_API float   fp8_e4m3_to_float(uint8_t v);
GGML_API uint8_t float_to_fp8_e4m3_rn(float x);

// FP8 E5M2: 1 sign, 5 exponent (bias 15), 2 mantissa bits
// Range: ±[2^-16, 57344], NaN/Inf: exp=31 (standard IEEE-like)
// Ref: OCP MX v1.0 §4.2
GGML_API float   fp8_e5m2_to_float(uint8_t v);
GGML_API uint8_t float_to_fp8_e5m2_rn(float x);

// FP6 E2M3: 1 sign, 2 exponent (bias 1), 3 mantissa bits
// Range: ±[2^-3, 7.5], stored as low 6 bits of a byte (00xxxxxx)
// MX format: NO NaN/Inf — all bit patterns are valid numbers
// Ref: OCP MX v1.0 §4.2
GGML_API float   fp6_e2m3_to_float(uint8_t v);
GGML_API uint8_t float_to_fp6_e2m3_rn(float x);

// FP6 E3M2: 1 sign, 3 exponent (bias 3), 2 mantissa bits
// Range: ±[2^-4, 28.0], stored as low 6 bits of a byte (00xxxxxx)
// MX format: NO NaN/Inf — exp=7 is a valid normal value (unlike IEEE-754)
// CRITICAL: subnormal scale is 2^(1-bias-m) = 2^(-4) = 1/16, NOT 1/4
// Ref: OCP MX v1.0 §4.2
GGML_API float   fp6_e3m2_to_float(uint8_t v);
GGML_API uint8_t float_to_fp6_e3m2_rn(float x);

// FP6 tight packing: pack/unpack 4 six-bit values into/from 3 bytes
// Layout: v[0]=bits[5:0], v[1]=bits[11:6], v[2]=bits[17:12], v[3]=bits[23:18]
// Saves 25% memory vs byte-padded storage (24B vs 32B per MX block)
GGML_API void pack_fp6x4(const uint8_t v[4], uint8_t out[3]);
GGML_API void unpack_fp6x4(const uint8_t in[3], uint8_t v[4]);

//
// Hadamard rotation (reference scalar implementation)
//
// 32-element Walsh-Hadamard transform applied to MX blocks before quantization.
// Distributes outlier energy uniformly across the block, dramatically improving
// quantization quality for types with shared exponents.
//
// Mathematical property: H^T·H = I (orthogonal), so H(K)·H(Q) = K·Q.
// Flash attention applies matching rotation to Q, preserving attention scores exactly.
//
// Implementation: 5 butterfly stages (log2(32) = 5) + normalization by 1/sqrt(32).
// This is the standard "fast Walsh-Hadamard transform" with O(n log n) operations.
//
// Applied in set_rows (K cache quantization) and flash_attn (Q quantization).
// Skipped for MLA models (DK != DV) where V is a view of K — rotation would corrupt V.
//
// Empirical impact (PPL degradation WITHOUT rotation, Qwen3-Coder-30B):
//   MXFP8 E4M3: +0.22, MXFP8 E5M2: +1.38, MXFP6 E2M3: +3.34, MXFP6 E3M2: +4.60
//
// Prior art: QuIP# (Tseng et al. 2024), BRQ (Huang et al. 2024) use Hadamard for
// weight quantization. Our contribution applies it to KV cache quantization at the
// MX block boundary, where block-32 is optimal because it matches the shared exponent
// group size exactly.
//
// GPU backends provide optimized versions (CUDA warp shuffles, Metal SIMD groups).
//
GGML_API void ggml_hadamard_32_inplace(float vals[32]);

GGML_API void iq2xs_init_impl(enum ggml_type type);
GGML_API void iq2xs_free_impl(enum ggml_type type);
GGML_API void iq3xs_init_impl(int grid_size);
GGML_API void iq3xs_free_impl(int grid_size);

#ifdef __cplusplus
}
#endif
