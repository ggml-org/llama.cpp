/**
 * GGML CUDA Backend for PagedCopy (GGML_OP_PAGED_CPY)
 *
 * This file implements the CUDA kernel for copying K/V data to paged cache blocks.
 * Similar to vLLM's reshape_and_cache kernel.
 */

#ifndef GGML_USE_MUSA

#include "common.cuh"

// CUDA kernel for copying K/V data to paged blocks
// Inspired by vLLM's reshape_and_cache kernel
template<typename T>
__global__ void paged_cpy_kernel(
    const T* __restrict__ kv_cur,      // [head_size, n_heads, n_tokens]
    T* __restrict__ kv_cache,          // [num_blocks, n_kv_heads, head_size, block_size]
    const int32_t* __restrict__ slot_idxs,  // [n_tokens] - slot index for each token
    int head_size,
    int n_heads,
    int n_kv_heads,
    int n_tokens,
    int block_size) {

    // Each block processes one token
    const int token_idx = blockIdx.x;
    if (token_idx >= n_tokens) return;

    // Get the cache slot for this token
    const int slot_idx = slot_idxs[token_idx];
    const int block_id = slot_idx / block_size;
    const int block_offset = slot_idx % block_size;

    // GQA: map query head to kv head
    const int head_ratio = n_heads / n_kv_heads;
    const int head_idx = threadIdx.y;  // which head
    const int kv_head_idx = head_idx / head_ratio;

    // Each thread copies one element of head_size
    const int elem_idx = threadIdx.x;
    if (elem_idx >= head_size) return;

    // Source: kv_cur[elem_idx, head_idx, token_idx]
    // Layout: [head_size, n_heads, n_tokens]
    const int src_idx = elem_idx + head_idx * head_size + token_idx * head_size * n_heads;
    const T value = kv_cur[src_idx];

    // Destination: kv_cache[block_id, kv_head_idx, elem_idx, block_offset]
    // Layout: [num_blocks, n_kv_heads, head_size, block_size]
    const int dst_idx = block_id * (n_kv_heads * head_size * block_size) +
                        kv_head_idx * (head_size * block_size) +
                        elem_idx * block_size +
                        block_offset;

    kv_cache[dst_idx] = value;
}

// Launcher function
void ggml_cuda_op_paged_cpy(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst) {

    const ggml_tensor * kv_cur    = dst->src[0];  // [head_size, n_heads, n_tokens]
    const ggml_tensor * kv_cache  = dst->src[1];  // [num_blocks, n_kv_heads, head_size, block_size]
    const ggml_tensor * slot_idxs = dst->src[2];  // [n_tokens] (int32)

    // Get dimensions
    const int head_size = kv_cur->ne[0];
    const int n_heads = kv_cur->ne[1];
    const int n_tokens = kv_cur->ne[2];

    const int num_blocks = kv_cache->ne[0];
    const int n_kv_heads = kv_cache->ne[1];
    const int block_size = kv_cache->ne[3];

    GGML_ASSERT(head_size == kv_cache->ne[2]);
    GGML_ASSERT(n_tokens == slot_idxs->ne[0]);
    GGML_ASSERT(slot_idxs->type == GGML_TYPE_I32);

    // Skip if there are no tokens to copy
    if (n_tokens == 0) {
        return;
    }

    // Get pointers
    const void * kv_cur_ptr = kv_cur->data;
    void * kv_cache_ptr = kv_cache->data;
    const int32_t * slot_idxs_ptr = (const int32_t *)slot_idxs->data;

    // Get CUDA stream
    cudaStream_t stream = ctx.stream();

    // Launch kernel
    // Grid: one block per token
    // Block: head_size threads in x, n_heads threads in y
    dim3 grid(n_tokens);
    dim3 block(head_size, n_heads);

    // Ensure block dimensions are valid
    GGML_ASSERT(head_size * n_heads <= 1024);  // max threads per block

    // Debug logging
    fprintf(stderr, "paged_cpy: head_size=%d, n_heads=%d, n_kv_heads=%d, n_tokens=%d, block_size=%d\n",
            head_size, n_heads, n_kv_heads, n_tokens, block_size);
    fprintf(stderr, "paged_cpy: kv_cur dims=[%lld,%lld,%lld,%lld], kv_cache dims=[%lld,%lld,%lld,%lld]\n",
            kv_cur->ne[0], kv_cur->ne[1], kv_cur->ne[2], kv_cur->ne[3],
            kv_cache->ne[0], kv_cache->ne[1], kv_cache->ne[2], kv_cache->ne[3]);
    fprintf(stderr, "paged_cpy: pointers: kv_cur=%p, kv_cache=%p, slot_idxs=%p\n",
            kv_cur_ptr, kv_cache_ptr, slot_idxs_ptr);

    if (kv_cur->type == GGML_TYPE_F16) {
        paged_cpy_kernel<half><<<grid, block, 0, stream>>>(
            (const half *)kv_cur_ptr,
            (half *)kv_cache_ptr,
            slot_idxs_ptr,
            head_size,
            n_heads,
            n_kv_heads,
            n_tokens,
            block_size);
    } else if (kv_cur->type == GGML_TYPE_F32) {
        paged_cpy_kernel<float><<<grid, block, 0, stream>>>(
            (const float *)kv_cur_ptr,
            (float *)kv_cache_ptr,
            slot_idxs_ptr,
            head_size,
            n_heads,
            n_kv_heads,
            n_tokens,
            block_size);
    } else {
        GGML_ABORT("Unsupported type for paged_cpy");
    }

    CUDA_CHECK(cudaGetLastError());
}

#else // GGML_USE_MUSA

// Stub for MUSA
#include "common.cuh"

void ggml_cuda_op_paged_cpy(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("PagedCopy is not yet supported on MUSA");
}

#endif // GGML_USE_MUSA
