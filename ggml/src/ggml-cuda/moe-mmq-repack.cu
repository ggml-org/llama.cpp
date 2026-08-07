#include "moe-mmq-repack.cuh"

#include <cstring>

static __device__ __forceinline__ int64_t moe_mmq_cutlass_scale_offset(
        int row, int k_block, int padded_k_blocks) {
    const int inner_k       = k_block % 4;
    const int inner_m       = (row % 128) / 32;
    const int outer_m       = row % 32;
    const int k_tile        = k_block / 4;
    const int m_tile        = row / 128;
    const int k_tile_stride = 512;
    const int m_tile_stride = (padded_k_blocks / 4) * k_tile_stride;
    return (int64_t) m_tile * m_tile_stride + (int64_t) k_tile * k_tile_stride + outer_m * 16 +
           inner_m * 4 + inner_k;
}

static __global__ void moe_mmq_repack_cutlass(
        const block_mxfp4 * src,
        char *              values,
        uint8_t *           scales,
        int                 k_blocks,
        int                 padded_k_blocks,
        int                 rows,
        int                 padded_rows,
        int                 experts) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n     = (int64_t) experts * rows * padded_k_blocks;
    if (index >= n) {
        return;
    }

    const int     k_block = index % padded_k_blocks;
    const int64_t row_all = index / padded_k_blocks;
    const int     row      = row_all % rows;
    const int     expert   = row_all / rows;
    uint8_t * values_dst   = (uint8_t *) values + index * (QK_MXFP4 / 2);
    if (k_block >= k_blocks) {
        memset(values_dst, 0, QK_MXFP4 / 2);
        return;
    }

    const block_mxfp4 block = src[row_all * k_blocks + k_block];
#pragma unroll
    for (int i = 0; i < QK_MXFP4 / 2; ++i) {
        const int e0 = 2 * i;
        const int e1 = e0 + 1;
        const uint8_t q0 = e0 < QK_MXFP4 / 2 ? block.qs[e0] & 0x0F : block.qs[e0 - QK_MXFP4 / 2] >> 4;
        const uint8_t q1 = e1 < QK_MXFP4 / 2 ? block.qs[e1] & 0x0F : block.qs[e1 - QK_MXFP4 / 2] >> 4;
        values_dst[i] = q0 | (q1 << 4);
    }

    const int64_t scale_expert_stride = (int64_t) padded_rows * padded_k_blocks;
    scales[(int64_t) expert * scale_expert_stride +
           moe_mmq_cutlass_scale_offset(row, k_block, padded_k_blocks)] = block.e;
}

static __global__ void moe_mmq_repack_cutlass_nvfp4(
        const block_nvfp4 * src,
        char *              values,
        uint8_t *           scales,
        int                 k_blocks,
        int                 padded_scale_blocks,
        int                 rows,
        int                 padded_rows,
        int                 experts) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n     = (int64_t) experts * rows * padded_scale_blocks;
    if (index >= n) {
        return;
    }

    const int scale_block = index % padded_scale_blocks;
    const int64_t row_all = index / padded_scale_blocks;
    const int row          = row_all % rows;
    const int expert       = row_all / rows;
    uint8_t * values_dst   = (uint8_t *) values + row_all * padded_scale_blocks * (QK_NVFP4_SUB / 2) +
                           scale_block * (QK_NVFP4_SUB / 2);
    const int k_base       = scale_block * QK_NVFP4_SUB;
    if (k_base >= k_blocks * QK_NVFP4) {
        memset(values_dst, 0, QK_NVFP4_SUB / 2);
        return;
    }

    const block_nvfp4 block = src[row_all * k_blocks + k_base / QK_NVFP4];
    const int sub = (k_base % QK_NVFP4) / QK_NVFP4_SUB;
#pragma unroll
    for (int i = 0; i < QK_NVFP4_SUB / 2; ++i) {
        const int e0 = 2 * i;
        const int e1 = e0 + 1;
        const uint8_t v0 = block.qs[sub * (QK_NVFP4_SUB / 2) + e0 % (QK_NVFP4_SUB / 2)];
        const uint8_t v1 = block.qs[sub * (QK_NVFP4_SUB / 2) + e1 % (QK_NVFP4_SUB / 2)];
        const uint8_t q0 = e0 < QK_NVFP4_SUB / 2 ? v0 & 0x0F : v0 >> 4;
        const uint8_t q1 = e1 < QK_NVFP4_SUB / 2 ? v1 & 0x0F : v1 >> 4;
        values_dst[i] = q0 | (q1 << 4);
    }

    const int64_t scale_expert_stride = (int64_t) padded_rows * padded_scale_blocks;
    scales[(int64_t) expert * scale_expert_stride +
           moe_mmq_cutlass_scale_offset(row, scale_block, padded_scale_blocks)] = block.d[sub];
}

static __global__ void moe_mmq_repack_cutlass_nvfp4_pair(
        const block_nvfp4 * first,
        const block_nvfp4 * second,
        char *              values,
        uint8_t *           scales,
        int                 k_blocks,
        int                 padded_scale_blocks,
        int                 rows,
        int                 padded_rows,
        int                 experts) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int rows_pair = 2 * rows;
    const int64_t n = (int64_t) experts * rows_pair * padded_scale_blocks;
    if (index >= n) {
        return;
    }

    const int scale_block = index % padded_scale_blocks;
    const int64_t row_all = index / padded_scale_blocks;
    const int row          = row_all % rows_pair;
    const int expert       = row_all / rows_pair;
    const int source_row   = row < rows ? row : row - rows;
    const block_nvfp4 * source = row < rows ? first : second;
    uint8_t * values_dst = (uint8_t *) values + row_all * padded_scale_blocks * (QK_NVFP4_SUB / 2) +
                           scale_block * (QK_NVFP4_SUB / 2);
    const int k_base = scale_block * QK_NVFP4_SUB;
    if (k_base >= k_blocks * QK_NVFP4) {
        memset(values_dst, 0, QK_NVFP4_SUB / 2);
        return;
    }

    const int64_t source_row_all = (int64_t) expert * rows + source_row;
    const block_nvfp4 block = source[source_row_all * k_blocks + k_base / QK_NVFP4];
    const int sub = (k_base % QK_NVFP4) / QK_NVFP4_SUB;
#pragma unroll
    for (int i = 0; i < QK_NVFP4_SUB / 2; ++i) {
        const int e0 = 2 * i;
        const int e1 = e0 + 1;
        const uint8_t v0 = block.qs[sub * (QK_NVFP4_SUB / 2) + e0 % (QK_NVFP4_SUB / 2)];
        const uint8_t v1 = block.qs[sub * (QK_NVFP4_SUB / 2) + e1 % (QK_NVFP4_SUB / 2)];
        const uint8_t q0 = e0 < QK_NVFP4_SUB / 2 ? v0 & 0x0F : v0 >> 4;
        const uint8_t q1 = e1 < QK_NVFP4_SUB / 2 ? v1 & 0x0F : v1 >> 4;
        values_dst[i] = q0 | (q1 << 4);
    }

    const int64_t scale_expert_stride = (int64_t) padded_rows * padded_scale_blocks;
    scales[(int64_t) expert * scale_expert_stride +
           moe_mmq_cutlass_scale_offset(row, scale_block, padded_scale_blocks)] = block.d[sub];
}

static ggml_cuda_moe_weight_cache_entry * moe_mmq_find_weight(
        ggml_backend_cuda_context & ctx, const ggml_tensor * tensor, bool preserves_source) {
    const uint64_t buffer_generation = ggml_cuda_buffer_get_generation(tensor->buffer);
    for (auto & entry : ctx.moe_weight_cache) {
        if (entry.source == tensor && entry.source_data == tensor->data && entry.source_buffer == tensor->buffer &&
            entry.source_buffer_generation == buffer_generation && entry.source_secondary == nullptr &&
            entry.layout == (int) ggml_cuda_moe_weight_layout::cutlass &&
            entry.preserves_source == preserves_source && entry.ne[0] == tensor->ne[0] &&
            entry.ne[1] == tensor->ne[1] && entry.ne[2] == tensor->ne[2]) {
            return &entry;
        }
    }
    return nullptr;
}

static ggml_cuda_moe_weight_cache_entry * moe_mmq_repack_weight_cutlass(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *         tensor,
        cudaStream_t                stream,
        bool                        preserve_source) {
#if CUDART_VERSION >= 12080
    const bool nvfp4               = tensor->type == GGML_TYPE_NVFP4;
    const int qk                   = nvfp4 ? QK_NVFP4 : QK_MXFP4;
    const int scale_vector_size    = nvfp4 ? QK_NVFP4_SUB : QK_MXFP4;
    const int k_blocks             = tensor->ne[0] / qk;
    const int padded_k             = GGML_PAD(tensor->ne[0], 128);
    const int padded_scale_blocks  = padded_k / scale_vector_size;
    const int rows                 = tensor->ne[1];
    const int padded_rows          = GGML_PAD(rows, 128);
    const int experts              = tensor->ne[2];
    if (tensor->ne[3] != 1) {
        return nullptr;
    }

    const size_t values_size = (size_t) experts * rows * padded_k / 2;
    const size_t scales_size = (size_t) experts * padded_rows * padded_scale_blocks;
    if (values_size > ggml_nbytes(tensor)) {
        return nullptr;
    }

    void * values = nullptr;
    void * scales = nullptr;
    CUDA_CHECK(cudaMallocAsync(&values, values_size, stream));
    CUDA_CHECK(cudaMallocAsync(&scales, scales_size, stream));
    CUDA_CHECK(cudaMemsetAsync(scales, 0, scales_size, stream));

    constexpr int threads = 256;
    const int64_t n_blocks = (int64_t) experts * rows * padded_scale_blocks;
    const int grid = (int) ((n_blocks + threads - 1) / threads);
    if (nvfp4) {
        moe_mmq_repack_cutlass_nvfp4<<<grid, threads, 0, stream>>>(
            (const block_nvfp4 *) tensor->data, (char *) values, (uint8_t *) scales, k_blocks,
            padded_scale_blocks, rows, padded_rows, experts);
    } else {
        moe_mmq_repack_cutlass<<<grid, threads, 0, stream>>>(
            (const block_mxfp4 *) tensor->data, (char *) values, (uint8_t *) scales, k_blocks,
            padded_scale_blocks, rows, padded_rows, experts);
    }
    CUDA_CHECK(cudaGetLastError());
    if (!preserve_source) {
        CUDA_CHECK(cudaMemcpyAsync(tensor->data, values, values_size, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaFreeAsync(values, stream));
        values = tensor->data;
    }

    ctx.moe_weight_cache.emplace_back();
    ggml_cuda_moe_weight_cache_entry * entry = &ctx.moe_weight_cache.back();
    entry->source                   = tensor;
    entry->source_data              = tensor->data;
    entry->source_buffer            = tensor->buffer;
    entry->source_buffer_generation = ggml_cuda_buffer_get_generation(tensor->buffer);
    entry->layout                   = (int) ggml_cuda_moe_weight_layout::cutlass;
    entry->preserves_source         = preserve_source;
    entry->ne[0]                    = tensor->ne[0];
    entry->ne[1]                    = tensor->ne[1];
    entry->ne[2]                    = tensor->ne[2];
    entry->data                     = values;
    entry->owns_data                = preserve_source;
    entry->scales_data              = scales;
    entry->owns_scales              = true;
    entry->ncols                    = padded_k;
    entry->stride_row               = padded_k / qk;
    entry->stride_channel           = (int64_t) rows * entry->stride_row;
    entry->scale_stride             = padded_rows * padded_scale_blocks;
    entry->rows_padded              = padded_rows;
    CUDA_CHECK(cudaEventCreateWithFlags(&entry->ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&entry->last_use, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(entry->ready, stream));
    return entry;
#else
    GGML_UNUSED_VARS(ctx, tensor, stream, preserve_source);
    return nullptr;
#endif
}

bool ggml_cuda_moe_repack_weight_pair(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *         first,
        const ggml_tensor *         second,
        ggml_cuda_moe_weight_view & view,
        cudaStream_t                stream) {
#if CUDART_VERSION >= 12080
    if (first->type != GGML_TYPE_NVFP4 || second->type != GGML_TYPE_NVFP4 ||
        !ggml_is_contiguous(first) || !ggml_is_contiguous(second) || !ggml_are_same_shape(first, second) ||
        first->ne[3] != 1 || first->ne[0] % QK_NVFP4 != 0) {
        return false;
    }
    if (stream == nullptr) {
        stream = ctx.stream();
    }

    const uint64_t first_generation  = ggml_cuda_buffer_get_generation(first->buffer);
    const uint64_t second_generation = ggml_cuda_buffer_get_generation(second->buffer);
    ggml_cuda_moe_weight_cache_entry * entry = nullptr;
    for (auto & candidate : ctx.moe_weight_cache) {
        if (candidate.source == first && candidate.source_data == first->data &&
            candidate.source_buffer == first->buffer && candidate.source_buffer_generation == first_generation &&
            candidate.source_secondary == second && candidate.source_secondary_data == second->data &&
            candidate.source_secondary_buffer == second->buffer &&
            candidate.source_secondary_buffer_generation == second_generation &&
            candidate.layout == (int) ggml_cuda_moe_weight_layout::cutlass && candidate.ne[0] == first->ne[0] &&
            candidate.ne[1] == 2 * first->ne[1] && candidate.ne[2] == first->ne[2]) {
            entry = &candidate;
            break;
        }
    }

    if (entry == nullptr) {
        const int k_blocks            = first->ne[0] / QK_NVFP4;
        const int padded_k            = GGML_PAD(first->ne[0], 128);
        const int padded_scale_blocks = padded_k / QK_NVFP4_SUB;
        const int rows                = first->ne[1];
        const int rows_pair           = 2 * rows;
        const int padded_rows         = GGML_PAD(rows_pair, 128);
        const int experts             = first->ne[2];
        const size_t values_size = (size_t) experts * rows_pair * padded_k / 2;
        const size_t scales_size = (size_t) experts * padded_rows * padded_scale_blocks;

        ctx.moe_weight_cache.emplace_back();
        entry = &ctx.moe_weight_cache.back();
        CUDA_CHECK(cudaMallocAsync(&entry->data, values_size, stream));
        CUDA_CHECK(cudaMallocAsync(&entry->scales_data, scales_size, stream));
        CUDA_CHECK(cudaMemsetAsync(entry->scales_data, 0, scales_size, stream));

        constexpr int threads = 256;
        const int64_t n_blocks = (int64_t) experts * rows_pair * padded_scale_blocks;
        const int grid = (int) ((n_blocks + threads - 1) / threads);
        moe_mmq_repack_cutlass_nvfp4_pair<<<grid, threads, 0, stream>>>(
            (const block_nvfp4 *) first->data, (const block_nvfp4 *) second->data, (char *) entry->data,
            (uint8_t *) entry->scales_data, k_blocks, padded_scale_blocks, rows, padded_rows, experts);
        CUDA_CHECK(cudaGetLastError());

        entry->source                             = first;
        entry->source_data                        = first->data;
        entry->source_buffer                      = first->buffer;
        entry->source_buffer_generation           = first_generation;
        entry->source_secondary                   = second;
        entry->source_secondary_data              = second->data;
        entry->source_secondary_buffer            = second->buffer;
        entry->source_secondary_buffer_generation = second_generation;
        entry->layout                             = (int) ggml_cuda_moe_weight_layout::cutlass;
        entry->ne[0]                              = first->ne[0];
        entry->ne[1]                              = rows_pair;
        entry->ne[2]                              = experts;
        entry->owns_data                          = true;
        entry->owns_scales                        = true;
        entry->ncols                              = padded_k;
        entry->stride_row                         = padded_k / QK_NVFP4;
        entry->stride_channel                     = (int64_t) rows_pair * entry->stride_row;
        entry->scale_stride                       = padded_rows * padded_scale_blocks;
        entry->rows_padded                        = padded_rows;
        CUDA_CHECK(cudaEventCreateWithFlags(&entry->ready, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&entry->last_use, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(entry->ready, stream));
    }

    view = {
        (const char *) entry->data,
        (const uint8_t *) entry->scales_data,
        entry->ncols,
        entry->stride_row,
        entry->stride_channel,
        entry->scale_stride,
        ggml_cuda_moe_weight_layout::cutlass,
    };
    view.rows_padded = entry->rows_padded;
    view.ready       = entry->ready;
    view.last_use    = entry->last_use;
    view.type        = GGML_TYPE_NVFP4;
    return true;
#else
    GGML_UNUSED_VARS(ctx, first, second, view, stream);
    return false;
#endif
}

bool ggml_cuda_moe_repack_weight(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor *         tensor,
        ggml_cuda_moe_weight_layout layout,
        ggml_cuda_moe_weight_view & view,
        cudaStream_t                stream,
        size_t                      cache_entries,
        bool                        wait_ready,
        bool                        preserve_source) {
    GGML_ASSERT(layout == ggml_cuda_moe_weight_layout::cutlass);
    GGML_ASSERT(tensor->type == GGML_TYPE_MXFP4 || tensor->type == GGML_TYPE_NVFP4);
    GGML_ASSERT(ggml_is_contiguous(tensor));
    GGML_UNUSED(cache_entries);

    const int qk = tensor->type == GGML_TYPE_NVFP4 ? QK_NVFP4 : QK_MXFP4;
    GGML_ASSERT(tensor->ne[0] % qk == 0);
    if (stream == nullptr) {
        stream = ctx.stream();
    }

    ggml_cuda_moe_weight_cache_entry * entry = moe_mmq_find_weight(ctx, tensor, preserve_source);
    if (entry == nullptr) {
        entry = moe_mmq_repack_weight_cutlass(ctx, tensor, stream, preserve_source);
        if (entry == nullptr) {
            return false;
        }
    }

    if (wait_ready && entry->ready != nullptr) {
        CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), entry->ready, 0));
    }
    view = {
        (const char *) entry->data,
        (const uint8_t *) entry->scales_data,
        entry->ncols,
        entry->stride_row,
        entry->stride_channel,
        entry->scale_stride,
        ggml_cuda_moe_weight_layout::cutlass,
    };
    view.rows_padded = entry->rows_padded;
    view.ready       = entry->ready;
    view.last_use    = entry->last_use;
    view.type        = tensor->type;
    return true;
}


void ggml_cuda_moe_weight_wait_ready(const ggml_cuda_moe_weight_view & view, cudaStream_t stream) {
    if (view.ready != nullptr) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, view.ready, 0));
    }
}

void ggml_cuda_moe_weight_mark_used(const ggml_cuda_moe_weight_view & view, cudaStream_t stream) {
    if (view.last_use != nullptr) {
        CUDA_CHECK(cudaEventRecord(view.last_use, stream));
    }
}
