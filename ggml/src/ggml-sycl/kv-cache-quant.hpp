//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_KV_CACHE_QUANT_HPP
#define GGML_SYCL_KV_CACHE_QUANT_HPP

#include <sycl/sycl.hpp>
#include <cstdint>
#include <cmath>
#include <cfloat>

// =============================================================================
// FP8 KV Cache Quantization
// =============================================================================
//
// This implements online quantization of K and V cache to 8-bit format,
// providing 2x memory reduction compared to FP16.
//
// Two FP8 formats are supported:
// - E5M2: 5-bit exponent, 2-bit mantissa (IEEE-compatible, larger range)
// - E4M3: 4-bit exponent, 3-bit mantissa (higher precision, smaller range)
//
// The quantization scheme uses per-head scaling factors:
// - Each head has its own scale factor for K and V
// - Scale = max(abs(values)) / max_fp8_value
// - Quantized value = round(original / scale)
// - Dequantized value = quantized * scale
//
// Memory layout:
// - k_cache_fp8: [num_blocks, num_kv_heads, block_size, D] uint8_t
// - v_cache_fp8: [num_blocks, num_kv_heads, block_size, D] uint8_t
// - k_scales: [num_blocks, num_kv_heads] float (one scale per head per block)
// - v_scales: [num_blocks, num_kv_heads] float
//

// =============================================================================
// FP8 Configuration
// =============================================================================

// FP8 E4M3 format (default - better precision)
constexpr float FP8_E4M3_MAX = 448.0f;      // Maximum representable value
constexpr float FP8_E4M3_MIN = 1.0f / 512;  // Minimum positive value

// FP8 E5M2 format (alternative - larger range)
constexpr float FP8_E5M2_MAX = 57344.0f;    // Maximum representable value
constexpr float FP8_E5M2_MIN = 1.0f / 16384; // Minimum positive value

// Work-group configuration
constexpr int FP8_NTHREADS = 256;
constexpr int FP8_WARP_SIZE = 32;

// =============================================================================
// FP8 Quantization Helpers
// =============================================================================

// Clamp value to FP8 E4M3 range and round to nearest
inline uint8_t float_to_fp8_e4m3(float val, float inv_scale) {
    float scaled = val * inv_scale;
    // Clamp to [-FP8_E4M3_MAX, FP8_E4M3_MAX]
    scaled = sycl::fmax(-FP8_E4M3_MAX, sycl::fmin(FP8_E4M3_MAX, scaled));
    // Convert to signed int8, then to uint8
    int8_t quantized = static_cast<int8_t>(sycl::round(scaled));
    return static_cast<uint8_t>(quantized);
}

// Dequantize FP8 E4M3 to float
inline float fp8_e4m3_to_float(uint8_t val, float scale) {
    int8_t signed_val = static_cast<int8_t>(val);
    return static_cast<float>(signed_val) * scale;
}

// =============================================================================
// Quantize K/V to FP8 - Kernel
// =============================================================================
//
// Quantizes a batch of K or V vectors to FP8 with per-head scaling.
// Each work-group handles one token's K or V vector.
//

template <int D>
static void quantize_kv_fp8_kernel(
    uint8_t * __restrict__ kv_fp8,      // Output: quantized KV [block_idx, head, offset, D]
    float * __restrict__ scales,         // Output: scales [block_idx, head]
    const sycl::half * __restrict__ kv,  // Input: FP16 KV values [token_idx, head, D]
    const int64_t * __restrict__ slot_mapping,  // Maps token_idx to slot in cache
    const int num_tokens,
    const int num_kv_heads,
    const int block_size,
    const sycl::nd_item<3> & item,
    float * shared_absmax) {

    const int token_idx = item.get_group(2);
    const int head_idx = item.get_group(1);
    const int tid = item.get_local_linear_id();
    auto sg = item.get_sub_group();
    const int lane_id = tid % FP8_WARP_SIZE;

    if (token_idx >= num_tokens) return;

    const int64_t slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) return;  // Padding token

    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;

    // Input pointer
    const sycl::half * kv_in = kv + token_idx * num_kv_heads * D + head_idx * D;

    // Output pointer
    const int64_t out_offset = block_idx * num_kv_heads * block_size * D +
                               head_idx * block_size * D +
                               block_offset * D;
    uint8_t * kv_out = kv_fp8 + out_offset;

    // Step 1: Find max absolute value for this head
    float local_absmax = 0.0f;
    for (int d = tid; d < D; d += FP8_NTHREADS) {
        float val = static_cast<float>(kv_in[d]);
        local_absmax = sycl::fmax(local_absmax, sycl::fabs(val));
    }

    // Warp-level reduction
    #pragma unroll
    for (int mask = FP8_WARP_SIZE / 2; mask >= 1; mask /= 2) {
        local_absmax = sycl::fmax(local_absmax,
            sycl::permute_group_by_xor(sg, local_absmax, mask));
    }

    // Store warp results
    const int warp_id = tid / FP8_WARP_SIZE;
    const int num_warps = FP8_NTHREADS / FP8_WARP_SIZE;
    if (lane_id == 0) {
        shared_absmax[warp_id] = local_absmax;
    }
    sycl::group_barrier(item.get_group());

    // Final reduction
    if (tid < num_warps) {
        local_absmax = shared_absmax[tid];
    } else {
        local_absmax = 0.0f;
    }
    #pragma unroll
    for (int mask = num_warps / 2; mask >= 1; mask /= 2) {
        local_absmax = sycl::fmax(local_absmax,
            sycl::permute_group_by_xor(sg, local_absmax, mask));
    }
    const float absmax = sycl::select_from_group(sg, local_absmax, 0);

    // Compute scale: scale = absmax / 127 (for symmetric int8 quantization)
    const float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
    const float inv_scale = 1.0f / scale;

    // Store scale (only thread 0)
    if (tid == 0) {
        scales[block_idx * num_kv_heads + head_idx] = scale;
    }

    sycl::group_barrier(item.get_group());

    // Step 2: Quantize values
    for (int d = tid; d < D; d += FP8_NTHREADS) {
        float val = static_cast<float>(kv_in[d]);
        kv_out[d] = float_to_fp8_e4m3(val, inv_scale);
    }
}

// =============================================================================
// Dequantize K/V from FP8 - Kernel
// =============================================================================
//
// Dequantizes a tile of K or V values from FP8 back to FP16.
// This can be fused with attention computation for better performance.
//

template <int D>
static void dequantize_kv_fp8_kernel(
    sycl::half * __restrict__ kv_out,       // Output: dequantized [tokens, heads, D]
    const uint8_t * __restrict__ kv_fp8,    // Input: quantized cache
    const float * __restrict__ scales,       // Scales [blocks, heads]
    const int * __restrict__ block_tables,   // [seqs, max_blocks]
    const int * __restrict__ positions,      // KV positions to dequantize
    const int num_positions,
    const int num_kv_heads,
    const int block_size,
    const int max_blocks_per_seq,
    const int seq_idx,
    const sycl::nd_item<3> & item) {

    const int pos_idx = item.get_group(2);
    const int head_idx = item.get_group(1);
    const int tid = item.get_local_linear_id();

    if (pos_idx >= num_positions) return;

    const int kv_pos = positions[pos_idx];
    const int logical_block = kv_pos / block_size;
    const int offset_in_block = kv_pos % block_size;
    const int physical_block = block_tables[seq_idx * max_blocks_per_seq + logical_block];

    // Get scale for this block/head
    const float scale = scales[physical_block * num_kv_heads + head_idx];

    // Input offset
    const int64_t in_offset = physical_block * num_kv_heads * block_size * D +
                              head_idx * block_size * D +
                              offset_in_block * D;
    const uint8_t * kv_in = kv_fp8 + in_offset;

    // Output offset
    sycl::half * out_ptr = kv_out + pos_idx * num_kv_heads * D + head_idx * D;

    // Dequantize
    for (int d = tid; d < D; d += FP8_NTHREADS) {
        float val = fp8_e4m3_to_float(kv_in[d], scale);
        out_ptr[d] = static_cast<sycl::half>(val);
    }
}

// =============================================================================
// Fused Attention with FP8 Dequantization
// =============================================================================
//
// This kernel computes attention while dequantizing K/V on-the-fly,
// avoiding the need to store full FP16 K/V tiles.
//

template <int D>
static void attention_fp8_kernel(
    float * __restrict__ out,               // [seqs, heads, D]
    const sycl::half * __restrict__ Q,      // [seqs, heads, D]
    const uint8_t * __restrict__ K_fp8,     // Quantized K cache
    const uint8_t * __restrict__ V_fp8,     // Quantized V cache
    const float * __restrict__ K_scales,    // K scales [blocks, kv_heads]
    const float * __restrict__ V_scales,    // V scales [blocks, kv_heads]
    const float scale,                       // Attention scale
    const int * __restrict__ block_tables,   // [seqs, max_blocks]
    const int * __restrict__ context_lens,   // [seqs]
    const int num_heads,
    const int num_kv_heads,
    const int block_size,
    const int max_blocks_per_seq,
    const sycl::nd_item<3> & item,
    float * shared_mem) {

    const int head_idx = item.get_group(2);
    const int seq_idx = item.get_group(1);
    const int tid = item.get_local_linear_id();
    auto sg = item.get_sub_group();
    const int warp_id = tid / FP8_WARP_SIZE;
    const int lane_id = tid % FP8_WARP_SIZE;

    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const int context_len = context_lens[seq_idx];

    // Shared memory layout
    float * q_shared = shared_mem;
    float * logits = q_shared + D;
    float * output_acc = logits + context_len;
    const int num_warps = FP8_NTHREADS / FP8_WARP_SIZE;
    float * warp_reduce = output_acc + D;

    // Load Q (scaled)
    const sycl::half * q_ptr = Q + seq_idx * num_heads * D + head_idx * D;
    for (int d = tid; d < D; d += FP8_NTHREADS) {
        q_shared[d] = static_cast<float>(q_ptr[d]) * scale;
    }

    for (int d = tid; d < D; d += FP8_NTHREADS) {
        output_acc[d] = 0.0f;
    }

    sycl::group_barrier(item.get_group());

    // Compute Q @ K^T with on-the-fly FP8 dequantization
    float qk_max = -FLT_MAX;

    for (int kv_pos = tid; kv_pos < context_len; kv_pos += FP8_NTHREADS) {
        const int logical_block = kv_pos / block_size;
        const int offset_in_block = kv_pos % block_size;
        const int physical_block = block_tables[seq_idx * max_blocks_per_seq + logical_block];

        // Get K scale
        const float k_scale = K_scales[physical_block * num_kv_heads + kv_head_idx];

        // K offset
        const int64_t k_offset = physical_block * num_kv_heads * block_size * D +
                                 kv_head_idx * block_size * D +
                                 offset_in_block * D;
        const uint8_t * k_ptr = K_fp8 + k_offset;

        // Compute dot product with on-the-fly dequantization
        float qk = 0.0f;
        for (int d = 0; d < D; ++d) {
            float k_val = fp8_e4m3_to_float(k_ptr[d], k_scale);
            qk += q_shared[d] * k_val;
        }

        logits[kv_pos] = qk;
        qk_max = sycl::fmax(qk_max, qk);
    }

    // Reduce qk_max
    #pragma unroll
    for (int mask = FP8_WARP_SIZE / 2; mask >= 1; mask /= 2) {
        qk_max = sycl::fmax(qk_max, sycl::permute_group_by_xor(sg, qk_max, mask));
    }
    if (lane_id == 0) warp_reduce[warp_id] = qk_max;
    sycl::group_barrier(item.get_group());

    if (tid < num_warps) qk_max = warp_reduce[tid];
    else qk_max = -FLT_MAX;
    #pragma unroll
    for (int mask = num_warps / 2; mask >= 1; mask /= 2) {
        qk_max = sycl::fmax(qk_max, sycl::permute_group_by_xor(sg, qk_max, mask));
    }
    qk_max = sycl::select_from_group(sg, qk_max, 0);

    sycl::group_barrier(item.get_group());

    // Compute softmax and exp_sum
    float exp_sum = 0.0f;
    for (int kv_pos = tid; kv_pos < context_len; kv_pos += FP8_NTHREADS) {
        // Use IEEE-compliant exp instead of native::exp for determinism
        float exp_qk = sycl::exp(logits[kv_pos] - qk_max);
        logits[kv_pos] = exp_qk;
        exp_sum += exp_qk;
    }

    // Reduce exp_sum
    #pragma unroll
    for (int mask = FP8_WARP_SIZE / 2; mask >= 1; mask /= 2) {
        exp_sum += sycl::permute_group_by_xor(sg, exp_sum, mask);
    }
    if (lane_id == 0) warp_reduce[warp_id] = exp_sum;
    sycl::group_barrier(item.get_group());

    if (tid < num_warps) exp_sum = warp_reduce[tid];
    else exp_sum = 0.0f;
    #pragma unroll
    for (int mask = num_warps / 2; mask >= 1; mask /= 2) {
        exp_sum += sycl::permute_group_by_xor(sg, exp_sum, mask);
    }
    exp_sum = sycl::select_from_group(sg, exp_sum, 0);

    sycl::group_barrier(item.get_group());

    // Compute weighted V with on-the-fly dequantization
    for (int d = tid; d < D; d += FP8_NTHREADS) {
        float acc = 0.0f;

        for (int kv_pos = 0; kv_pos < context_len; ++kv_pos) {
            const int logical_block = kv_pos / block_size;
            const int offset_in_block = kv_pos % block_size;
            const int physical_block = block_tables[seq_idx * max_blocks_per_seq + logical_block];

            // Get V scale
            const float v_scale = V_scales[physical_block * num_kv_heads + kv_head_idx];

            // V offset
            const int64_t v_offset = physical_block * num_kv_heads * block_size * D +
                                     kv_head_idx * block_size * D +
                                     offset_in_block * D;
            const uint8_t * v_ptr = V_fp8 + v_offset;

            float v_val = fp8_e4m3_to_float(v_ptr[d], v_scale);
            acc += logits[kv_pos] * v_val;
        }

        output_acc[d] = acc;
    }

    sycl::group_barrier(item.get_group());

    // Write output
    const float inv_exp_sum = 1.0f / (exp_sum + 1e-6f);
    float * out_ptr = out + seq_idx * num_heads * D + head_idx * D;

    for (int d = tid; d < D; d += FP8_NTHREADS) {
        out_ptr[d] = output_acc[d] * inv_exp_sum;
    }
}

// =============================================================================
// Launch Functions
// =============================================================================

template <int D>
void launch_quantize_kv_fp8(
    uint8_t * kv_fp8,
    float * scales,
    const sycl::half * kv,
    const int64_t * slot_mapping,
    const int num_tokens,
    const int num_kv_heads,
    const int block_size,
    sycl::queue * stream) {

    const int num_warps = FP8_NTHREADS / FP8_WARP_SIZE;

    sycl::range<3> grid(1, num_kv_heads, num_tokens);
    sycl::range<3> block(1, 1, FP8_NTHREADS);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> shared_acc(sycl::range<1>(num_warps), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(FP8_WARP_SIZE)]] {
                float * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                quantize_kv_fp8_kernel<D>(
                    kv_fp8, scales, kv, slot_mapping,
                    num_tokens, num_kv_heads, block_size,
                    item, shared);
            });
    });
}

template <int D>
void launch_attention_fp8(
    float * out,
    const sycl::half * Q,
    const uint8_t * K_fp8,
    const uint8_t * V_fp8,
    const float * K_scales,
    const float * V_scales,
    const float scale,
    const int * block_tables,
    const int * context_lens,
    const int num_seqs,
    const int num_heads,
    const int num_kv_heads,
    const int max_context_len,
    const int block_size,
    const int max_blocks_per_seq,
    sycl::queue * stream) {

    const int num_warps = FP8_NTHREADS / FP8_WARP_SIZE;
    const size_t shared_size = D + max_context_len + D + num_warps;

    sycl::range<3> grid(1, num_seqs, num_heads);
    sycl::range<3> block(1, 1, FP8_NTHREADS);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> shared_acc(sycl::range<1>(shared_size), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(FP8_WARP_SIZE)]] {
                float * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                attention_fp8_kernel<D>(
                    out, Q, K_fp8, V_fp8, K_scales, V_scales, scale,
                    block_tables, context_lens,
                    num_heads, num_kv_heads, block_size, max_blocks_per_seq,
                    item, shared);
            });
    });
}

// =============================================================================
// Memory Utilities
// =============================================================================

// Calculate FP8 cache size for given configuration
inline size_t fp8_kv_cache_size(
    int num_blocks, int num_kv_heads, int block_size, int head_dim) {

    // K and V cache (uint8_t per element instead of half)
    const size_t kv_size = num_blocks * num_kv_heads * block_size * head_dim * sizeof(uint8_t);

    // Scales (one float per head per block, for K and V)
    const size_t scales_size = num_blocks * num_kv_heads * sizeof(float) * 2;

    return kv_size * 2 + scales_size;  // K + V + scales
}

// Compare with FP16 cache size
inline size_t fp16_kv_cache_size(
    int num_blocks, int num_kv_heads, int block_size, int head_dim) {

    return num_blocks * num_kv_heads * block_size * head_dim * sizeof(sycl::half) * 2;
}

// Memory savings ratio
inline float fp8_memory_savings(
    int num_blocks, int num_kv_heads, int block_size, int head_dim) {

    const size_t fp16_size = fp16_kv_cache_size(num_blocks, num_kv_heads, block_size, head_dim);
    const size_t fp8_size = fp8_kv_cache_size(num_blocks, num_kv_heads, block_size, head_dim);

    return static_cast<float>(fp16_size) / static_cast<float>(fp8_size);
}

#endif // GGML_SYCL_KV_CACHE_QUANT_HPP
