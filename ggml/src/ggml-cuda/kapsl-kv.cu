#include "kapsl-kv.cuh"

template<typename src_t>
static __device__ half kapsl_to_half(src_t v);

template<>
__device__ half kapsl_to_half<float>(float v) {
    return __float2half(v);
}

template<>
__device__ half kapsl_to_half<half>(half v) {
    return v;
}

template<typename src_t>
static __device__ float kapsl_to_float(src_t v);

template<>
__device__ float kapsl_to_float<float>(float v) {
    return v;
}

template<>
__device__ float kapsl_to_float<half>(half v) {
    return __half2float(v);
}

template<typename src_t, typename pos_t>
static __global__ void kapsl_kv_write_kernel(
        half        * __restrict__ pool,
        const src_t * __restrict__ cur,
        const pos_t * __restrict__ positions,
        const int   * __restrict__ block_table,
        int32_t layer_id,
        int32_t block_size,
        int32_t block_table_layer_stride,
        int64_t head_dim,
        int64_t n_heads,
        int64_t n_tokens,
        int64_t cur_nb0,
        int64_t cur_nb1,
        int64_t cur_nb2,
        int64_t pool_nb0,
        int64_t pool_nb1,
        int64_t pool_nb2,
        int64_t pool_nb3) {
    const int64_t token = blockIdx.x;
    const int64_t head  = blockIdx.y;
    const int64_t d     = threadIdx.x;

    if (token >= n_tokens || head >= n_heads || d >= head_dim) {
        return;
    }

    const int64_t pos = positions[token];
    if (pos < 0) {
        return;
    }

    const int64_t logical_block = pos / block_size;
    const int64_t pos_in_block  = pos - logical_block * block_size;
    const int64_t phys_block    = block_table[layer_id * block_table_layer_stride + logical_block];

    const int64_t src_off = d * cur_nb0 + head * cur_nb1 + token * cur_nb2;
    const int64_t dst_off = d * pool_nb0 + pos_in_block * pool_nb1 + head * pool_nb2 + phys_block * pool_nb3;

    pool[dst_off] = kapsl_to_half(cur[src_off]);
}

template<typename src_t, typename pos_t>
static void kapsl_kv_write_cuda(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * cur         = dst->src[0];
    const ggml_tensor * positions   = dst->src[1];
    const ggml_tensor * block_table = dst->src[2];

    const int32_t layer_id                 = ggml_get_op_params_i32(dst, 0);
    const int32_t block_size               = ggml_get_op_params_i32(dst, 1);
    const int32_t block_table_layer_stride = ggml_get_op_params_i32(dst, 2);

    const int64_t head_dim = cur->ne[0];
    const int64_t n_heads  = cur->ne[1];
    const int64_t n_tokens = cur->ne[2];

    const dim3 block((uint32_t) head_dim);
    const dim3 grid((uint32_t) n_tokens, (uint32_t) n_heads);

    if (head_dim > 0 && n_heads > 0 && n_tokens > 0) {
        kapsl_kv_write_kernel<<<grid, block, 0, ctx.stream()>>>(
                (half *) dst->data,
                (const src_t *) cur->data,
                (const pos_t *) positions->data,
                (const int *) block_table->data,
                layer_id,
                block_size,
                block_table_layer_stride,
                head_dim,
                n_heads,
                n_tokens,
                cur->nb[0] / (int64_t) sizeof(src_t),
                cur->nb[1] / (int64_t) sizeof(src_t),
                cur->nb[2] / (int64_t) sizeof(src_t),
                dst->nb[0] / (int64_t) sizeof(half),
                dst->nb[1] / (int64_t) sizeof(half),
                dst->nb[2] / (int64_t) sizeof(half),
                dst->nb[3] / (int64_t) sizeof(half));
    }
}

void ggml_cuda_op_kapsl_kv_write(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I32 || dst->src[1]->type == GGML_TYPE_I64);
    GGML_ASSERT(dst->src[2]->type == GGML_TYPE_I32);

    if (dst->src[0]->type == GGML_TYPE_F32 && dst->src[1]->type == GGML_TYPE_I32) {
        kapsl_kv_write_cuda<float, int32_t>(ctx, dst);
    } else if (dst->src[0]->type == GGML_TYPE_F32 && dst->src[1]->type == GGML_TYPE_I64) {
        kapsl_kv_write_cuda<float, int64_t>(ctx, dst);
    } else if (dst->src[0]->type == GGML_TYPE_F16 && dst->src[1]->type == GGML_TYPE_I32) {
        kapsl_kv_write_cuda<half, int32_t>(ctx, dst);
    } else {
        kapsl_kv_write_cuda<half, int64_t>(ctx, dst);
    }
}

// Tiled paged attention with online softmax.
//
// Each thread block handles one (token, q_head) pair. The context is walked
// in tiles of KAPSL_ATTN_TILE positions: threads cooperatively score the tile
// into shared memory, block-reduce the tile max and sum, rescale the running
// accumulator when the max moves, then accumulate V with each thread owning a
// strided subset of head dims. Shared memory usage is constant in context
// length, so long contexts never exceed the per-block shared memory limit.

#define KAPSL_ATTN_THREADS 256
#define KAPSL_ATTN_TILE    256
// Per-thread output accumulator registers: supports head_dim up to
// KAPSL_ATTN_THREADS * KAPSL_ATTN_MAX_DIMS_PER_THREAD.
#define KAPSL_ATTN_MAX_DIMS_PER_THREAD 4

template<typename q_t, typename pos_t>
static __global__ void kapsl_paged_attn_kernel(
        float     * __restrict__ out,
        const q_t * __restrict__ q,
        const half * __restrict__ kv_pool,
        const pos_t * __restrict__ positions,
        const int * __restrict__ block_table,
        int32_t layer_id,
        int32_t block_size,
        int32_t block_table_layer_stride,
        int32_t n_kv_heads,
        float scale,
        int64_t head_dim,
        int64_t n_q_heads,
        int64_t n_tokens,
        int64_t q_nb0,
        int64_t q_nb1,
        int64_t q_nb2,
        int64_t out_nb0,
        int64_t out_nb1,
        int64_t out_nb2) {
    const int64_t token  = blockIdx.x;
    const int64_t q_head = blockIdx.y;
    const int     tid      = threadIdx.x;
    const int     nthreads = blockDim.x;

    if (token >= n_tokens || q_head >= n_q_heads) {
        return;
    }

    const int64_t kv_head        = q_head * n_kv_heads / n_q_heads;
    const int64_t kv_head_stride = block_size * head_dim;
    const int64_t kv_type_stride = n_kv_heads * kv_head_stride;
    const int64_t block_stride   = 2 * kv_type_stride;

    const int * bt = block_table + layer_id * block_table_layer_stride;

    // Dynamic shared memory: head_dim floats of query + one tile of scores.
    extern __shared__ float smem[];
    float * q_smem    = smem;
    float * tile_smem = smem + head_dim;
    __shared__ float red_smem[KAPSL_ATTN_THREADS];

    for (int64_t d = tid; d < head_dim; d += nthreads) {
        q_smem[d] = kapsl_to_float(q[d * q_nb0 + q_head * q_nb1 + token * q_nb2]);
    }
    __syncthreads();

    float acc[KAPSL_ATTN_MAX_DIMS_PER_THREAD];
#pragma unroll
    for (int j = 0; j < KAPSL_ATTN_MAX_DIMS_PER_THREAD; ++j) {
        acc[j] = 0.0f;
    }
    float running_max = -3.402823466e+38F;
    float running_sum = 0.0f;

    const int64_t ctx_len = (int64_t) positions[token] + 1;

    for (int64_t tile_start = 0; tile_start < ctx_len; tile_start += KAPSL_ATTN_TILE) {
        const int tile_len = (int) min((int64_t) KAPSL_ATTN_TILE, ctx_len - tile_start);

        // Score this tile: each thread computes full Q·K dots for a strided
        // set of positions.
        for (int i = tid; i < tile_len; i += nthreads) {
            const int64_t pos          = tile_start + i;
            const int64_t pos_in_block = pos % block_size;
            const int64_t phys_block   = bt[pos / block_size];

            const half * k_ptr = kv_pool
                + phys_block * block_stride
                + kv_head * kv_head_stride
                + pos_in_block * head_dim;

            float dot = 0.0f;
            for (int64_t d = 0; d < head_dim; ++d) {
                dot += q_smem[d] * __half2float(k_ptr[d]);
            }
            tile_smem[i] = dot * scale;
        }
        __syncthreads();

        // Block-reduce the tile max.
        float tile_max = -3.402823466e+38F;
        for (int i = tid; i < tile_len; i += nthreads) {
            tile_max = fmaxf(tile_max, tile_smem[i]);
        }
        red_smem[tid] = tile_max;
        __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) {
                red_smem[tid] = fmaxf(red_smem[tid], red_smem[tid + s]);
            }
            __syncthreads();
        }
        const float new_max = fmaxf(running_max, red_smem[0]);
        __syncthreads();

        // Rescale the running state to the new max before folding in the tile.
        const float correction = running_sum > 0.0f ? expf(running_max - new_max) : 0.0f;
        running_sum *= correction;
#pragma unroll
        for (int j = 0; j < KAPSL_ATTN_MAX_DIMS_PER_THREAD; ++j) {
            acc[j] *= correction;
        }

        // Exponentiate the tile in place and block-reduce its sum.
        float tile_sum = 0.0f;
        for (int i = tid; i < tile_len; i += nthreads) {
            const float w = expf(tile_smem[i] - new_max);
            tile_smem[i] = w;
            tile_sum += w;
        }
        red_smem[tid] = tile_sum;
        __syncthreads();
        for (int s = nthreads / 2; s > 0; s >>= 1) {
            if (tid < s) {
                red_smem[tid] += red_smem[tid + s];
            }
            __syncthreads();
        }
        running_sum += red_smem[0];

        // Accumulate V: threads own head dims, positions read coalesced.
        for (int i = 0; i < tile_len; ++i) {
            const int64_t pos          = tile_start + i;
            const int64_t pos_in_block = pos % block_size;
            const int64_t phys_block   = bt[pos / block_size];

            const half * v_ptr = kv_pool
                + phys_block * block_stride
                + kv_type_stride
                + kv_head * kv_head_stride
                + pos_in_block * head_dim;

            const float w = tile_smem[i];
            int j = 0;
            for (int64_t d = tid; d < head_dim; d += nthreads, ++j) {
                acc[j] += w * __half2float(v_ptr[d]);
            }
        }
        __syncthreads();

        running_max = new_max;
    }

    const float inv_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
    int j = 0;
    for (int64_t d = tid; d < head_dim; d += nthreads, ++j) {
        out[d * out_nb0 + q_head * out_nb1 + token * out_nb2] = acc[j] * inv_sum;
    }
}

template<typename q_t, typename pos_t>
static void kapsl_paged_attn_cuda(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q           = dst->src[0];
    const ggml_tensor * kv_pool     = dst->src[1];
    const ggml_tensor * positions   = dst->src[2];
    const ggml_tensor * block_table = dst->src[3];

    const int32_t layer_id                 = ggml_get_op_params_i32(dst, 0);
    const int32_t block_size               = ggml_get_op_params_i32(dst, 1);
    const int32_t block_table_layer_stride = ggml_get_op_params_i32(dst, 2);
    const int32_t n_kv_heads               = ggml_get_op_params_i32(dst, 3);
    const float scale                      = ggml_get_op_params_f32(dst, 4);

    const int64_t head_dim = q->ne[0];
    const int64_t n_heads  = q->ne[1];
    const int64_t n_tokens = q->ne[2];

    GGML_ASSERT(head_dim <= KAPSL_ATTN_THREADS * KAPSL_ATTN_MAX_DIMS_PER_THREAD);

    const dim3 block(KAPSL_ATTN_THREADS);
    const dim3 grid((uint32_t) n_tokens, (uint32_t) n_heads);
    const size_t shared_bytes = (size_t) (head_dim + KAPSL_ATTN_TILE) * sizeof(float);

    if (head_dim > 0 && n_heads > 0 && n_tokens > 0) {
        kapsl_paged_attn_kernel<<<grid, block, shared_bytes, ctx.stream()>>>(
                (float *) dst->data,
                (const q_t *) q->data,
                (const half *) kv_pool->data,
                (const pos_t *) positions->data,
                (const int *) block_table->data,
                layer_id,
                block_size,
                block_table_layer_stride,
                n_kv_heads,
                scale,
                head_dim,
                n_heads,
                n_tokens,
                q->nb[0] / (int64_t) sizeof(q_t),
                q->nb[1] / (int64_t) sizeof(q_t),
                q->nb[2] / (int64_t) sizeof(q_t),
                dst->nb[0] / (int64_t) sizeof(float),
                dst->nb[1] / (int64_t) sizeof(float),
                dst->nb[2] / (int64_t) sizeof(float));
    }
}

void ggml_cuda_op_kapsl_paged_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[2]->type == GGML_TYPE_I32 || dst->src[2]->type == GGML_TYPE_I64);
    GGML_ASSERT(dst->src[3]->type == GGML_TYPE_I32);

    if (dst->src[0]->type == GGML_TYPE_F32 && dst->src[2]->type == GGML_TYPE_I32) {
        kapsl_paged_attn_cuda<float, int32_t>(ctx, dst);
    } else if (dst->src[0]->type == GGML_TYPE_F32 && dst->src[2]->type == GGML_TYPE_I64) {
        kapsl_paged_attn_cuda<float, int64_t>(ctx, dst);
    } else if (dst->src[0]->type == GGML_TYPE_F16 && dst->src[2]->type == GGML_TYPE_I32) {
        kapsl_paged_attn_cuda<half, int32_t>(ctx, dst);
    } else {
        kapsl_paged_attn_cuda<half, int64_t>(ctx, dst);
    }
}
