#pragma once

/**
 * PagedAttention CUDA Kernel Header
 *
 * Based on the PagedAttention implementation from vLLM:
 * https://github.com/vllm-project/vllm/blob/main/csrc/attention/attention_kernels.cuh
 *
 * Copyright (c) 2023-2024 vLLM Project
 * SPDX-License-Identifier: Apache-2.0
 *
 * Adapted for GGML by llama.cpp contributors
 */

#include "common.cuh"

// WARP_SIZE is already defined in common.cuh
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

namespace ggml_cuda_paged_attention {

// Partition size for PagedAttention V2
constexpr int PARTITION_SIZE = 512;

// Supported head sizes
constexpr int SUPPORTED_HEAD_SIZES[] = {32, 64, 80, 96, 112, 120, 128, 192, 256};

//
// Helper structures and functions
//

// Vector types for efficient memory access
template<typename T, int N>
struct Vec {
    using Type = T;
};

template<> struct Vec<half, 1> { using Type = half; };
template<> struct Vec<half, 2> { using Type = half2; };
template<> struct Vec<half, 4> { using Type = uint2; };  // 4 halfs = 64 bits
template<> struct Vec<half, 8> { using Type = uint4; };  // 8 halfs = 128 bits

template<> struct Vec<float, 1> { using Type = float; };
template<> struct Vec<float, 2> { using Type = float2; };
template<> struct Vec<float, 4> { using Type = float4; };

// Float vector type conversion
template<typename L_vec>
struct FloatVec {
    using Type = L_vec;
};

// Warp shuffle utilities
#define SHFL_XOR_SYNC(var, lane_mask) __shfl_xor_sync(uint32_t(-1), var, lane_mask, WARP_SIZE)
#define SHFL_SYNC(var, src_lane) __shfl_sync(uint32_t(-1), var, src_lane, WARP_SIZE)

// Block-level reduction
template <int NUM_WARPS>
__inline__ __device__ float block_sum(float* red_smem, float sum) {
    // Decompose thread index into warp / lane
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // Warp-level reduction
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        sum += SHFL_XOR_SYNC(sum, mask);
    }

    // Warp leaders store to shared memory
    if (lane == 0) {
        red_smem[warp] = sum;
    }
    __syncthreads();

    // Final reduction across warps
    if (lane < NUM_WARPS) {
        sum = red_smem[lane];
    }
    #pragma unroll
    for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
        sum += SHFL_XOR_SYNC(sum, mask);
    }

    // Broadcast result
    return SHFL_SYNC(sum, 0);
}

// Dot product helpers
template<typename T>
__inline__ __device__ float dot(T a, T b) {
    // Default implementation for scalar types
    return float(a) * float(b);
}

__inline__ __device__ float dot(half2 a, half2 b) {
    float2 a_f = __half22float2(a);
    float2 b_f = __half22float2(b);
    return a_f.x * b_f.x + a_f.y * b_f.y;
}

// Convert from float
template<typename T>
__inline__ __device__ void from_float(T& dst, float src) {
    dst = T(src);
}

__inline__ __device__ void from_float(half& dst, float src) {
    dst = __float2half(src);
}

__inline__ __device__ void from_float(half2& dst, float src) {
    dst = __float2half2_rn(src);
}

// Zero initialization
template<typename T>
__inline__ __device__ void zero(T& val) {
    val = T(0);
}

//
// PagedAttention V1 Kernel
//
// For shorter sequences (≤8192 tokens)
// Each thread block processes one head of one sequence
//

template <typename scalar_t,      // Output/query data type (e.g., half)
          typename cache_t,        // KV cache data type (e.g., half)
          int HEAD_SIZE,           // Head dimension (e.g., 128)
          int BLOCK_SIZE,          // Block size in tokens (e.g., 16)
          int NUM_THREADS>         // Threads per block (e.g., 128)
__global__ void paged_attention_v1_kernel(
    scalar_t* __restrict__ out,              // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,          // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,     // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,     // [num_blocks, num_kv_heads, head_size, block_size]
    const int num_kv_heads,
    const float scale,
    const int* __restrict__ block_tables,    // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,        // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads] or nullptr
    const int q_stride,                      // stride for q
    const int kv_block_stride,               // stride between blocks in cache
    const int kv_head_stride);               // stride between heads in cache

//
// PagedAttention V2 Kernel
//
// For longer sequences (>8192 tokens)
// Uses partitioning to avoid shared memory limits
//

template <typename scalar_t,
          typename cache_t,
          int HEAD_SIZE,
          int BLOCK_SIZE,
          int NUM_THREADS,
          int PARTITION_SIZE = 512>
__global__ void paged_attention_v2_kernel(
    float* __restrict__ exp_sums,            // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,          // [num_seqs, num_heads, max_num_partitions]
    scalar_t* __restrict__ tmp_out,          // [num_seqs, num_heads, max_num_partitions, head_size]
    const scalar_t* __restrict__ q,
    const cache_t* __restrict__ k_cache,
    const cache_t* __restrict__ v_cache,
    const int num_kv_heads,
    const float scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride);

//
// PagedAttention V2 Reduce Kernel
//
// Combines partial results from V2 main kernel
//

template <typename scalar_t,
          int HEAD_SIZE,
          int NUM_THREADS,
          int PARTITION_SIZE = 512>
__global__ void paged_attention_v2_reduce_kernel(
    scalar_t* __restrict__ out,               // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,       // [num_seqs, num_heads, max_num_partitions]
    const float* __restrict__ max_logits,     // [num_seqs, num_heads, max_num_partitions]
    const scalar_t* __restrict__ tmp_out,     // [num_seqs, num_heads, max_num_partitions, head_size]
    const int* __restrict__ seq_lens,         // [num_seqs]
    const int max_num_partitions);

//
// Launcher functions (to be called from GGML backend)
//

// Launch PagedAttention V1
void paged_attention_v1_launcher(
    void* out,                    // Output tensor
    const void* query,            // Query tensor
    const void* key_cache,        // Key cache (paged)
    const void* value_cache,      // Value cache (paged)
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_size,
    int block_size,
    int max_num_blocks_per_seq,
    const int* block_tables,
    const int* seq_lens,
    int max_seq_len,
    float scale,
    const float* alibi_slopes,    // Can be nullptr
    ggml_type q_type,             // Query data type
    ggml_type kv_cache_type,      // KV cache data type
    cudaStream_t stream);

// Launch PagedAttention V2
void paged_attention_v2_launcher(
    void* out,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_size,
    int block_size,
    int max_num_blocks_per_seq,
    const int* block_tables,
    const int* seq_lens,
    int max_seq_len,
    float scale,
    const float* alibi_slopes,
    ggml_type q_type,
    ggml_type kv_cache_type,
    ggml_cuda_pool & pool,
    cudaStream_t stream);

// Helper: Decide which version to use
inline bool should_use_v1(int max_seq_len, int num_seqs, int num_heads) {
    const int max_num_partitions = (max_seq_len + PARTITION_SIZE - 1) / PARTITION_SIZE;

    // Use V1 if:
    // - Sequence is short enough (≤8192) AND
    // - Either we have only 1 partition OR we have lots of parallelism
    return max_seq_len <= 8192 && (max_num_partitions == 1 || num_seqs * num_heads > 512);
}

} // namespace ggml_cuda_paged_attention
