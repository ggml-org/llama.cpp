// PagedAttention is not yet supported on MUSA
#ifndef GGML_USE_MUSA

#include "paged-attention.cuh"
#include "common.cuh"

/**
 * PagedAttention V2 CUDA Kernel Implementation
 *
 * Based on the PagedAttention implementation from vLLM:
 * https://github.com/vllm-project/vllm/blob/main/csrc/attention/attention_kernels.cuh
 *
 * Copyright (c) 2023-2024 vLLM Project
 * SPDX-License-Identifier: Apache-2.0
 *
 * Adapted for GGML by llama.cpp contributors
 */

namespace ggml_cuda_paged_attention {

//
// Main PagedAttention V2 Kernel
//
// This kernel computes partial attention for one partition.
// The main difference from V1 is that it processes only a subset of the sequence
// and stores intermediate results (max_logits, exp_sums, partial outputs).
//

template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, int PARTITION_SIZE>
__global__ void paged_attention_v2_kernel(
    float* __restrict__ exp_sums,
    float* __restrict__ max_logits,
    scalar_t* __restrict__ tmp_out,
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
    const int kv_head_stride) {

    const int seq_idx = blockIdx.y;
    const int partition_idx = blockIdx.z;
    const int head_idx = blockIdx.x;
    const int num_heads = gridDim.x;
    const int max_num_partitions = gridDim.z;
    const int thread_idx = threadIdx.x;

    const int seq_len = seq_lens[seq_idx];
    if (partition_idx * PARTITION_SIZE >= seq_len) {
        // This partition is beyond the sequence length
        return;
    }

    // Calculate range of blocks to process in this partition
    const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
    const int num_blocks_per_partition = PARTITION_SIZE / BLOCK_SIZE;
    const int start_block_idx = partition_idx * num_blocks_per_partition;
    const int end_block_idx = MIN(start_block_idx + num_blocks_per_partition, num_seq_blocks);
    const int num_blocks = end_block_idx - start_block_idx;

    const int start_token_idx = start_block_idx * BLOCK_SIZE;
    const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
    const int num_tokens = end_token_idx - start_token_idx;

    const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

    // Shared memory for partial logits and reduction
    extern __shared__ char shared_mem[];
    float* logits = reinterpret_cast<float*>(shared_mem);
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    __shared__ float red_smem[2 * NUM_WARPS];

    const int warp_idx = thread_idx / WARP_SIZE;
    const int lane = thread_idx % WARP_SIZE;

    // Get KV head index (for GQA/MQA)
    const int num_queries_per_kv = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / num_queries_per_kv;

    // ALiBi bias (if applicable)
    const float alibi_slope = alibi_slopes ? alibi_slopes[head_idx] : 0.0f;

    // Query pointer for this sequence and head
    const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;

    // Compute Q·K for tokens in this partition only
    float qk_max = -FLT_MAX;

    for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx; block_idx += NUM_WARPS) {
        const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);

        // Load K vectors and compute dot products
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            const int token_idx = block_idx * BLOCK_SIZE + i;
            if (token_idx >= end_token_idx) break;

            // Compute Q·K dot product
            // K cache layout: [num_blocks, num_kv_heads, head_size, block_size]
            const cache_t* k_ptr = k_cache +
                physical_block_number * kv_block_stride +
                kv_head_idx * kv_head_stride +
                i;  // token position within block

            // Compute dot product between Q and K
            // Each thread computes part of the dot product
            float qk = 0.0f;
            for (int elem_idx = thread_idx; elem_idx < HEAD_SIZE; elem_idx += NUM_THREADS) {
                // Load K element for this token
                // K is stored as [head_size, block_size], so offset is elem_idx * BLOCK_SIZE
                const cache_t k_val = k_ptr[elem_idx * BLOCK_SIZE];

                // Load Q element (from scalar_t array)
                const scalar_t q_val = q_ptr[elem_idx];

                // Accumulate dot product
                qk += float(q_val) * float(k_val);
            }

            // Reduce across all threads in the block
            #pragma unroll
            for (int mask = NUM_THREADS / 2; mask >= 1; mask /= 2) {
                qk += SHFL_XOR_SYNC(qk, mask);
            }

            // Apply scale
            qk *= scale;

            // Add ALiBi bias if applicable
            if (alibi_slope != 0.0f) {
                qk += alibi_slope * (token_idx - seq_len + 1);
            }

            // Store logit (only thread 0 writes after full reduction)
            if (thread_idx == 0) {
                logits[token_idx - start_token_idx] = qk;
            }

            qk_max = fmaxf(qk_max, qk);
        }
    }

    // Warp and block level reduction to find max (same as V1)
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        qk_max = fmaxf(qk_max, SHFL_XOR_SYNC(qk_max, mask));
    }
    if (lane == 0) {
        red_smem[warp_idx] = qk_max;
    }
    __syncthreads();

    qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
    #pragma unroll
    for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
        qk_max = fmaxf(qk_max, SHFL_XOR_SYNC(qk_max, mask));
    }
    qk_max = SHFL_SYNC(qk_max, 0);

    // Compute softmax (for this partition only)
    float exp_sum = 0.0f;
    for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
        float val = __expf(logits[i] - qk_max);
        logits[i] = val;
        exp_sum += val;
    }
    exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

    // Store max_logit and exp_sum for this partition (for reduce kernel)
    if (thread_idx == 0) {
        const int idx = seq_idx * num_heads * max_num_partitions +
                        head_idx * max_num_partitions + partition_idx;
        max_logits[idx] = qk_max;
        exp_sums[idx] = exp_sum;
    }

    // Don't normalize yet - will be done in reduce kernel

    // Compute partial attention output (softmax · V)
    constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
    constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
    constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
    constexpr int NUM_ROWS_PER_THREAD = DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

    float accs[NUM_ROWS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        accs[i] = 0.0f;
    }

    // Compute partial attention output by multiplying softmax weights with V
    // V cache layout: [num_blocks, num_kv_heads, head_size, block_size]
    for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx; block_idx += NUM_WARPS) {
        const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);

        for (int i = 0; i < BLOCK_SIZE; ++i) {
            const int token_idx = block_idx * BLOCK_SIZE + i;
            if (token_idx >= end_token_idx) break;

            // Get attention weight for this token
            const float attn_weight = logits[token_idx - start_token_idx];

            // Accumulate V vectors weighted by attention
            #pragma unroll
            for (int j = 0; j < NUM_ROWS_PER_THREAD; j++) {
                const int row_idx = lane / NUM_V_VECS_PER_ROW + j * NUM_ROWS_PER_ITER;
                if (row_idx < HEAD_SIZE) {
                    // V cache pointer for this token and head dimension
                    const cache_t* v_ptr = v_cache +
                        physical_block_number * kv_block_stride +
                        kv_head_idx * kv_head_stride +
                        row_idx * BLOCK_SIZE + i;

                    accs[j] += attn_weight * float(*v_ptr);
                }
            }
        }
    }

    // Warp-level reduction of attention output
    #pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        float acc = accs[i];
        #pragma unroll
        for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
            acc += SHFL_XOR_SYNC(acc, mask);
        }
        accs[i] = acc;
    }

    __syncthreads();

    // Block-level reduction and output write (to temporary buffer)
    float* out_smem = reinterpret_cast<float*>(shared_mem);

    // Each warp writes its partial results to shared memory
    #pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
            out_smem[warp_idx * HEAD_SIZE + row_idx] = accs[i];
        }
    }
    __syncthreads();

    // Final reduction across warps and write to temporary output
    scalar_t* tmp_out_ptr = tmp_out +
        seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
        head_idx * max_num_partitions * HEAD_SIZE +
        partition_idx * HEAD_SIZE;

    for (int i = thread_idx; i < HEAD_SIZE; i += NUM_THREADS) {
        float acc = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; w++) {
            acc += out_smem[w * HEAD_SIZE + i];
        }
        from_float(tmp_out_ptr[i], acc);
    }
}

//
// PagedAttention V2 Reduce Kernel
//
// Combines partial results from all partitions
//

template <typename scalar_t, int HEAD_SIZE, int NUM_THREADS, int PARTITION_SIZE>
__global__ void paged_attention_v2_reduce_kernel(
    scalar_t* __restrict__ out,
    const float* __restrict__ exp_sums,
    const float* __restrict__ max_logits,
    const scalar_t* __restrict__ tmp_out,
    const int* __restrict__ seq_lens,
    const int max_num_partitions) {

    const int seq_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int num_heads = gridDim.x;
    const int thread_idx = threadIdx.x;

    const int seq_len = seq_lens[seq_idx];
    const int num_partitions = DIVIDE_ROUND_UP(seq_len, PARTITION_SIZE);

    // Find global max logit across all partitions
    float global_max_logit = -FLT_MAX;
    for (int i = 0; i < num_partitions; ++i) {
        const int idx = seq_idx * num_heads * max_num_partitions +
                        head_idx * max_num_partitions + i;
        global_max_logit = fmaxf(global_max_logit, max_logits[idx]);
    }

    // Share global max across threads
    __shared__ float shared_global_max;
    if (thread_idx == 0) {
        shared_global_max = global_max_logit;
    }
    __syncthreads();
    global_max_logit = shared_global_max;

    // Compute rescaled exp_sum
    float global_exp_sum = 0.0f;
    for (int i = 0; i < num_partitions; ++i) {
        const int idx = seq_idx * num_heads * max_num_partitions +
                        head_idx * max_num_partitions + i;
        float rescale = __expf(max_logits[idx] - global_max_logit);
        global_exp_sum += exp_sums[idx] * rescale;
    }

    // Share global exp_sum
    __shared__ float shared_global_exp_sum;
    if (thread_idx == 0) {
        shared_global_exp_sum = global_exp_sum;
    }
    __syncthreads();
    global_exp_sum = shared_global_exp_sum;

    const float inv_global_sum = __fdividef(1.0f, global_exp_sum + 1e-6f);

    // Combine partial outputs with proper rescaling
    for (int elem_idx = thread_idx; elem_idx < HEAD_SIZE; elem_idx += NUM_THREADS) {
        float acc = 0.0f;

        for (int i = 0; i < num_partitions; ++i) {
            const int idx = seq_idx * num_heads * max_num_partitions +
                            head_idx * max_num_partitions + i;

            const scalar_t* tmp_out_ptr = tmp_out +
                seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
                head_idx * max_num_partitions * HEAD_SIZE +
                i * HEAD_SIZE;

            float rescale = __expf(max_logits[idx] - global_max_logit);
            float partial_val = float(tmp_out_ptr[elem_idx]);
            acc += partial_val * rescale;
        }

        // Normalize and write final output
        scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
        from_float(out_ptr[elem_idx], acc * inv_global_sum);
    }
}

//
// Launcher function for V2
//

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
    cudaStream_t stream) {

    constexpr int NUM_THREADS = 128;
    const int max_num_partitions = DIVIDE_ROUND_UP(max_seq_len, PARTITION_SIZE);

    // Allocate temporary buffers using GGML's pool allocator
    // These will be automatically freed when they go out of scope at function exit
    const size_t exp_sum_count = num_seqs * num_heads * max_num_partitions;
    const size_t max_logit_count = num_seqs * num_heads * max_num_partitions;
    const size_t tmp_out_count = num_seqs * num_heads * max_num_partitions * head_size;

    ggml_cuda_pool_alloc<float> exp_sums_alloc(pool, exp_sum_count);
    ggml_cuda_pool_alloc<float> max_logits_alloc(pool, max_logit_count);

    // Use uint8_t for tmp_out to handle both half and float types generically
    const size_t tmp_out_bytes = tmp_out_count * (q_type == GGML_TYPE_F16 ? sizeof(half) : sizeof(float));
    ggml_cuda_pool_alloc<uint8_t> tmp_out_alloc(pool, tmp_out_bytes);

    float* exp_sums = exp_sums_alloc.get();
    float* max_logits = max_logits_alloc.get();
    void* tmp_out = tmp_out_alloc.get();

    // Launch main V2 kernel
    {
        dim3 grid(num_heads, num_seqs, max_num_partitions);
        dim3 block(NUM_THREADS);

        const int logits_size = PARTITION_SIZE * sizeof(float);
        const int outputs_size = (NUM_THREADS / WARP_SIZE / 2) * head_size * sizeof(float);
        const int shared_mem_size = max(logits_size, outputs_size);

        const int q_stride = num_heads * head_size;
        const int kv_block_stride = num_kv_heads * head_size * block_size;
        const int kv_head_stride = head_size * block_size;

        // Macro to dispatch V2 main kernel based on head size and block size
        #define LAUNCH_PAGED_ATTENTION_V2(SCALAR_T, CACHE_T, HEAD_SIZE, BLOCK_SIZE) \
            paged_attention_v2_kernel<SCALAR_T, CACHE_T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, PARTITION_SIZE> \
                <<<grid, block, shared_mem_size, stream>>>( \
                    exp_sums, max_logits, (SCALAR_T*)tmp_out, \
                    (const SCALAR_T*)query, (const CACHE_T*)key_cache, (const CACHE_T*)value_cache, \
                    num_kv_heads, scale, block_tables, seq_lens, \
                    max_num_blocks_per_seq, alibi_slopes, \
                    q_stride, kv_block_stride, kv_head_stride)

        // Dispatch for head size
        #define DISPATCH_V2_HEAD_SIZE(SCALAR_T, CACHE_T, BLOCK_SIZE) \
            switch (head_size) { \
                case 32:  LAUNCH_PAGED_ATTENTION_V2(SCALAR_T, CACHE_T, 32,  BLOCK_SIZE); break; \
                case 64:  LAUNCH_PAGED_ATTENTION_V2(SCALAR_T, CACHE_T, 64,  BLOCK_SIZE); break; \
                case 80:  LAUNCH_PAGED_ATTENTION_V2(SCALAR_T, CACHE_T, 80,  BLOCK_SIZE); break; \
                case 96:  LAUNCH_PAGED_ATTENTION_V2(SCALAR_T, CACHE_T, 96,  BLOCK_SIZE); break; \
                case 112: LAUNCH_PAGED_ATTENTION_V2(SCALAR_T, CACHE_T, 112, BLOCK_SIZE); break; \
                case 120: LAUNCH_PAGED_ATTENTION_V2(SCALAR_T, CACHE_T, 120, BLOCK_SIZE); break; \
                case 128: LAUNCH_PAGED_ATTENTION_V2(SCALAR_T, CACHE_T, 128, BLOCK_SIZE); break; \
                case 192: LAUNCH_PAGED_ATTENTION_V2(SCALAR_T, CACHE_T, 192, BLOCK_SIZE); break; \
                case 256: LAUNCH_PAGED_ATTENTION_V2(SCALAR_T, CACHE_T, 256, BLOCK_SIZE); break; \
                default: \
                    fprintf(stderr, "Unsupported head size: %d\n", head_size); \
                    GGML_ABORT("fatal error"); \
            }

        // Dispatch for block size
        #define DISPATCH_V2_BLOCK_SIZE(SCALAR_T, CACHE_T) \
            switch (block_size) { \
                case 8:  DISPATCH_V2_HEAD_SIZE(SCALAR_T, CACHE_T, 8);  break; \
                case 16: DISPATCH_V2_HEAD_SIZE(SCALAR_T, CACHE_T, 16); break; \
                case 32: DISPATCH_V2_HEAD_SIZE(SCALAR_T, CACHE_T, 32); break; \
                default: \
                    fprintf(stderr, "Unsupported block size: %d\n", block_size); \
                    GGML_ABORT("fatal error"); \
            }

        // Type dispatch based on q_type and kv_cache_type
        if (q_type == GGML_TYPE_F16 && kv_cache_type == GGML_TYPE_F16) {
            DISPATCH_V2_BLOCK_SIZE(half, half);
        } else if (q_type == GGML_TYPE_F32 && kv_cache_type == GGML_TYPE_F32) {
            DISPATCH_V2_BLOCK_SIZE(float, float);
        } else if (q_type == GGML_TYPE_F16 && kv_cache_type == GGML_TYPE_F32) {
            DISPATCH_V2_BLOCK_SIZE(half, float);
        } else if (q_type == GGML_TYPE_F32 && kv_cache_type == GGML_TYPE_F16) {
            DISPATCH_V2_BLOCK_SIZE(float, half);
        } else {
            fprintf(stderr, "Unsupported data type combination: q_type=%d, kv_cache_type=%d\n",
                    q_type, kv_cache_type);
            GGML_ABORT("fatal error");
        }

        #undef LAUNCH_PAGED_ATTENTION_V2
        #undef DISPATCH_V2_HEAD_SIZE
        #undef DISPATCH_V2_BLOCK_SIZE

        CUDA_CHECK(cudaGetLastError());
    }

    // Launch reduce kernel
    {
        dim3 reduce_grid(num_heads, num_seqs);
        dim3 reduce_block(NUM_THREADS);
        const int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);

        // Macro to dispatch V2 reduce kernel based on head size
        #define LAUNCH_PAGED_ATTENTION_V2_REDUCE(SCALAR_T, HEAD_SIZE) \
            paged_attention_v2_reduce_kernel<SCALAR_T, HEAD_SIZE, NUM_THREADS, PARTITION_SIZE> \
                <<<reduce_grid, reduce_block, reduce_shared_mem_size, stream>>>( \
                    (SCALAR_T*)out, exp_sums, max_logits, (const SCALAR_T*)tmp_out, \
                    seq_lens, max_num_partitions)

        // Dispatch reduce kernel for different head sizes and data types
        if (q_type == GGML_TYPE_F16) {
            switch (head_size) {
                case 32:  LAUNCH_PAGED_ATTENTION_V2_REDUCE(half, 32);  break;
                case 64:  LAUNCH_PAGED_ATTENTION_V2_REDUCE(half, 64);  break;
                case 80:  LAUNCH_PAGED_ATTENTION_V2_REDUCE(half, 80);  break;
                case 96:  LAUNCH_PAGED_ATTENTION_V2_REDUCE(half, 96);  break;
                case 112: LAUNCH_PAGED_ATTENTION_V2_REDUCE(half, 112); break;
                case 120: LAUNCH_PAGED_ATTENTION_V2_REDUCE(half, 120); break;
                case 128: LAUNCH_PAGED_ATTENTION_V2_REDUCE(half, 128); break;
                case 192: LAUNCH_PAGED_ATTENTION_V2_REDUCE(half, 192); break;
                case 256: LAUNCH_PAGED_ATTENTION_V2_REDUCE(half, 256); break;
                default:
                    fprintf(stderr, "Unsupported head size: %d\n", head_size);
                    GGML_ABORT("fatal error");
            }
        } else if (q_type == GGML_TYPE_F32) {
            switch (head_size) {
                case 32:  LAUNCH_PAGED_ATTENTION_V2_REDUCE(float, 32);  break;
                case 64:  LAUNCH_PAGED_ATTENTION_V2_REDUCE(float, 64);  break;
                case 80:  LAUNCH_PAGED_ATTENTION_V2_REDUCE(float, 80);  break;
                case 96:  LAUNCH_PAGED_ATTENTION_V2_REDUCE(float, 96);  break;
                case 112: LAUNCH_PAGED_ATTENTION_V2_REDUCE(float, 112); break;
                case 120: LAUNCH_PAGED_ATTENTION_V2_REDUCE(float, 120); break;
                case 128: LAUNCH_PAGED_ATTENTION_V2_REDUCE(float, 128); break;
                case 192: LAUNCH_PAGED_ATTENTION_V2_REDUCE(float, 192); break;
                case 256: LAUNCH_PAGED_ATTENTION_V2_REDUCE(float, 256); break;
                default:
                    fprintf(stderr, "Unsupported head size: %d\n", head_size);
                    GGML_ABORT("fatal error");
            }
        } else {
            fprintf(stderr, "Unsupported query data type: %d\n", q_type);
            GGML_ABORT("fatal error");
        }

        #undef LAUNCH_PAGED_ATTENTION_V2_REDUCE

        CUDA_CHECK(cudaGetLastError());
    }

    // Temporary buffers are automatically freed when pool_alloc objects go out of scope
}

} // namespace ggml_cuda_paged_attention

#endif // GGML_USE_MUSA
