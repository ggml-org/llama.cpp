#include "common.cuh"

__global__ void paged_attention_write_kernel(const float * k_new,        // [batch_size, n_heads_kv, head_dim]
                                             const float * v_new,        // [batch_size, n_heads_kv, head_dim]
                                             half *        kv_cache,     // The paged cache
                                             const int *   write_slots,  // Global slot index for each token
                                             const int *   batch_offsets,
                                             const int *   batch_lens,
                                             const size_t  stride_token,  // Elements between tokens in a block (nb1)
                                             const size_t  stride_head,   // Elements between heads (nb2)
                                             const size_t  stride_block,  // Elements between physical blocks (nb3)
                                             const int     n_heads_kv,
                                             const int     block_size);

__global__ void paged_attention_decode_kernel(const float * __restrict__ q,
                                              const half * __restrict__ kv_cache,
                                              const int * __restrict__ block_table,
                                              const int * __restrict__ context_lens,
                                              const int * __restrict__ batch_offsets,
                                              const int * __restrict__ batch_lens,
                                              const size_t stride_token,
                                              const size_t stride_head,
                                              const size_t stride_block,
                                              const int    n_heads_kv,
                                              const int    block_size,
                                              const int    max_blocks,
                                              const float  scale,
                                              float * __restrict__ out);

void ggml_cuda_op_paged_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
