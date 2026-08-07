#include "mmq-cutlass-repack.cuh"

#include <climits>
#include <cstring>

#ifdef GGML_CUDA_CUTLASS

static __device__ __forceinline__ int64_t cutlass_scale_offset(int row, int k_block, int padded_k_blocks) {
    const int inner_k       = k_block % 4;
    const int inner_m       = (row % 128) / 32;
    const int outer_m       = row % 32;
    const int k_tile        = k_block / 4;
    const int m_tile        = row / 128;
    const int k_tile_stride = 512;
    const int m_tile_stride = (padded_k_blocks / 4) * k_tile_stride;
    return (int64_t) m_tile * m_tile_stride + (int64_t) k_tile * k_tile_stride + outer_m * 16 + inner_m * 4 + inner_k;
}

static __global__ void cutlass_repack_mxfp4(const block_mxfp4 * src,
                                            char *              values,
                                            uint8_t *           scales,
                                            int                 k_blocks,
                                            int                 padded_k_blocks,
                                            int                 rows,
                                            int                 padded_rows,
                                            int                 groups) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n     = (int64_t) groups * rows * padded_k_blocks;
    if (index >= n) {
        return;
    }

    const int     k_block    = index % padded_k_blocks;
    const int64_t row_all    = index / padded_k_blocks;
    const int     row        = row_all % rows;
    const int     group      = row_all / rows;
    uint8_t *     values_dst = (uint8_t *) values + index * (QK_MXFP4 / 2);
    if (k_block >= k_blocks) {
        memset(values_dst, 0, QK_MXFP4 / 2);
        return;
    }

    const block_mxfp4 block = src[row_all * k_blocks + k_block];
#    pragma unroll
    for (int i = 0; i < QK_MXFP4 / 2; ++i) {
        const int     e0 = 2 * i;
        const int     e1 = e0 + 1;
        const uint8_t q0 = e0 < QK_MXFP4 / 2 ? block.qs[e0] & 0x0F : block.qs[e0 - QK_MXFP4 / 2] >> 4;
        const uint8_t q1 = e1 < QK_MXFP4 / 2 ? block.qs[e1] & 0x0F : block.qs[e1 - QK_MXFP4 / 2] >> 4;
        values_dst[i]    = q0 | (q1 << 4);
    }

    const int64_t scale_group_stride = (int64_t) padded_rows * padded_k_blocks;
    scales[(int64_t) group * scale_group_stride + cutlass_scale_offset(row, k_block, padded_k_blocks)] = block.e;
}

static __global__ void cutlass_repack_nvfp4(const block_nvfp4 * src,
                                            char *              values,
                                            uint8_t *           scales,
                                            int                 k_blocks,
                                            int                 padded_scale_blocks,
                                            int                 rows,
                                            int                 padded_rows,
                                            int                 groups) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n     = (int64_t) groups * rows * padded_scale_blocks;
    if (index >= n) {
        return;
    }

    const int     scale_block = index % padded_scale_blocks;
    const int64_t row_all     = index / padded_scale_blocks;
    const int     row         = row_all % rows;
    const int     group       = row_all / rows;
    uint8_t *     values_dst =
        (uint8_t *) values + row_all * padded_scale_blocks * (QK_NVFP4_SUB / 2) + scale_block * (QK_NVFP4_SUB / 2);
    const int k_base = scale_block * QK_NVFP4_SUB;
    if (k_base >= k_blocks * QK_NVFP4) {
        memset(values_dst, 0, QK_NVFP4_SUB / 2);
        return;
    }

    const block_nvfp4 block = src[row_all * k_blocks + k_base / QK_NVFP4];
    const int         sub   = (k_base % QK_NVFP4) / QK_NVFP4_SUB;
#    pragma unroll
    for (int i = 0; i < QK_NVFP4_SUB / 2; ++i) {
        const int     e0 = 2 * i;
        const int     e1 = e0 + 1;
        const uint8_t v0 = block.qs[sub * (QK_NVFP4_SUB / 2) + e0 % (QK_NVFP4_SUB / 2)];
        const uint8_t v1 = block.qs[sub * (QK_NVFP4_SUB / 2) + e1 % (QK_NVFP4_SUB / 2)];
        const uint8_t q0 = e0 < QK_NVFP4_SUB / 2 ? v0 & 0x0F : v0 >> 4;
        const uint8_t q1 = e1 < QK_NVFP4_SUB / 2 ? v1 & 0x0F : v1 >> 4;
        values_dst[i]    = q0 | (q1 << 4);
    }

    const int64_t scale_group_stride = (int64_t) padded_rows * padded_scale_blocks;
    scales[(int64_t) group * scale_group_stride + cutlass_scale_offset(row, scale_block, padded_scale_blocks)] =
        block.d[sub];
}

static __global__ void cutlass_repack_nvfp4_pair(const block_nvfp4 * first,
                                                 const block_nvfp4 * second,
                                                 char *              values,
                                                 uint8_t *           scales,
                                                 int                 k_blocks,
                                                 int                 padded_scale_blocks,
                                                 int                 rows,
                                                 int                 padded_rows,
                                                 int                 groups) {
    const int64_t index     = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int     rows_pair = 2 * rows;
    const int64_t n         = (int64_t) groups * rows_pair * padded_scale_blocks;
    if (index >= n) {
        return;
    }

    const int           scale_block = index % padded_scale_blocks;
    const int64_t       row_all     = index / padded_scale_blocks;
    const int           row         = row_all % rows_pair;
    const int           group       = row_all / rows_pair;
    const int           source_row  = row < rows ? row : row - rows;
    const block_nvfp4 * source      = row < rows ? first : second;
    uint8_t *           values_dst =
        (uint8_t *) values + row_all * padded_scale_blocks * (QK_NVFP4_SUB / 2) + scale_block * (QK_NVFP4_SUB / 2);
    const int k_base = scale_block * QK_NVFP4_SUB;
    if (k_base >= k_blocks * QK_NVFP4) {
        memset(values_dst, 0, QK_NVFP4_SUB / 2);
        return;
    }

    const int64_t     source_row_all = (int64_t) group * rows + source_row;
    const block_nvfp4 block          = source[source_row_all * k_blocks + k_base / QK_NVFP4];
    const int         sub            = (k_base % QK_NVFP4) / QK_NVFP4_SUB;
#    pragma unroll
    for (int i = 0; i < QK_NVFP4_SUB / 2; ++i) {
        const int     e0 = 2 * i;
        const int     e1 = e0 + 1;
        const uint8_t v0 = block.qs[sub * (QK_NVFP4_SUB / 2) + e0 % (QK_NVFP4_SUB / 2)];
        const uint8_t v1 = block.qs[sub * (QK_NVFP4_SUB / 2) + e1 % (QK_NVFP4_SUB / 2)];
        const uint8_t q0 = e0 < QK_NVFP4_SUB / 2 ? v0 & 0x0F : v0 >> 4;
        const uint8_t q1 = e1 < QK_NVFP4_SUB / 2 ? v1 & 0x0F : v1 >> 4;
        values_dst[i]    = q0 | (q1 << 4);
    }

    const int64_t scale_group_stride = (int64_t) padded_rows * padded_scale_blocks;
    scales[(int64_t) group * scale_group_stride + cutlass_scale_offset(row, scale_block, padded_scale_blocks)] =
        block.d[sub];
}

static __global__ void cutlass_repack_mxfp4_pair(const block_mxfp4 * first,
                                                 const block_mxfp4 * second,
                                                 char *              values,
                                                 uint8_t *           scales,
                                                 int                 k_blocks,
                                                 int                 padded_k_blocks,
                                                 int                 rows,
                                                 int                 padded_rows,
                                                 int                 groups) {
    const int64_t index     = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int     rows_pair = 2 * rows;
    const int64_t n         = (int64_t) groups * rows_pair * padded_k_blocks;
    if (index >= n) {
        return;
    }

    const int           k_block    = index % padded_k_blocks;
    const int64_t       row_all    = index / padded_k_blocks;
    const int           row        = row_all % rows_pair;
    const int           group      = row_all / rows_pair;
    const int           source_row = row < rows ? row : row - rows;
    const block_mxfp4 * source     = row < rows ? first : second;
    uint8_t *           values_dst = (uint8_t *) values + index * (QK_MXFP4 / 2);
    if (k_block >= k_blocks) {
        memset(values_dst, 0, QK_MXFP4 / 2);
        return;
    }

    const block_mxfp4 block = source[((int64_t) group * rows + source_row) * k_blocks + k_block];
#    pragma unroll
    for (int i = 0; i < QK_MXFP4 / 2; ++i) {
        const int     e0 = 2 * i;
        const int     e1 = e0 + 1;
        const uint8_t q0 = e0 < QK_MXFP4 / 2 ? block.qs[e0] & 0x0F : block.qs[e0 - QK_MXFP4 / 2] >> 4;
        const uint8_t q1 = e1 < QK_MXFP4 / 2 ? block.qs[e1] & 0x0F : block.qs[e1 - QK_MXFP4 / 2] >> 4;
        values_dst[i]    = q0 | (q1 << 4);
    }

    const int64_t scale_group_stride = (int64_t) padded_rows * padded_k_blocks;
    scales[(int64_t) group * scale_group_stride + cutlass_scale_offset(row, k_block, padded_k_blocks)] = block.e;
}

static ggml_cuda_cutlass_weight_cache_entry * cutlass_find_weight(ggml_backend_cuda_context & ctx,
                                                                  const ggml_tensor *         tensor,
                                                                  bool                        preserve_source) {
    const uint64_t buffer_generation = ggml_cuda_buffer_get_generation(tensor->buffer);
    for (auto & entry : ctx.cutlass_weight_cache) {
        if (entry.source == tensor && entry.source_data == tensor->data && entry.source_buffer == tensor->buffer &&
            entry.source_buffer_generation == buffer_generation && entry.source_secondary == nullptr &&
            entry.preserves_source == preserve_source &&
            entry.ne[0] == tensor->ne[0] && entry.ne[1] == tensor->ne[1] && entry.ne[2] == tensor->ne[2]) {
            return &entry;
        }
    }
    return nullptr;
}

static ggml_cuda_cutlass_weight_cache_entry * cutlass_repack_weight(ggml_backend_cuda_context & ctx,
                                                                    const ggml_tensor *         tensor,
                                                                    cudaStream_t                stream,
                                                                    bool                        preserve_source) {
#    if CUDART_VERSION >= 12080
    cudaStreamCaptureStatus capture_status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    if (capture_status != cudaStreamCaptureStatusNone) {
        return nullptr;
    }

    if ((tensor->type != GGML_TYPE_NVFP4 && tensor->type != GGML_TYPE_MXFP4) || !ggml_is_contiguous(tensor) ||
        tensor->ne[0] <= 0 || tensor->ne[1] <= 0 || tensor->ne[2] <= 0 || tensor->ne[0] > INT_MAX - 127 ||
        tensor->ne[1] > INT_MAX - 127 || tensor->ne[2] > INT_MAX || tensor->ne[3] != 1) {
        return nullptr;
    }

    const bool nvfp4               = tensor->type == GGML_TYPE_NVFP4;
    const int  qk                  = nvfp4 ? QK_NVFP4 : QK_MXFP4;
    const int  scale_vector_size   = nvfp4 ? QK_NVFP4_SUB : QK_MXFP4;
    const int  k_blocks            = tensor->ne[0] / qk;
    const int  padded_k            = GGML_PAD(tensor->ne[0], 128);
    const int  padded_scale_blocks = padded_k / scale_vector_size;
    const int  rows                = tensor->ne[1];
    const int  padded_rows         = GGML_PAD(rows, 128);
    const int  groups              = tensor->ne[2];
    if (tensor->ne[0] % qk != 0 || (int64_t) padded_rows * padded_scale_blocks > INT_MAX) {
        return nullptr;
    }

    const size_t values_size = (size_t) groups * rows * padded_k / 2;
    const size_t scales_size = (size_t) groups * padded_rows * padded_scale_blocks;
    if (!preserve_source && values_size > ggml_nbytes(tensor)) {
        return nullptr;
    }
    void *       values      = nullptr;
    void *       scales      = nullptr;
    CUDA_CHECK(cudaMallocAsync(&values, values_size, stream));
    CUDA_CHECK(cudaMallocAsync(&scales, scales_size, stream));
    CUDA_CHECK(cudaMemsetAsync(scales, 0, scales_size, stream));

    constexpr int threads  = 256;
    const int64_t n_blocks = (int64_t) groups * rows * padded_scale_blocks;
    if ((n_blocks + threads - 1) / threads > INT_MAX) {
        CUDA_CHECK(cudaFreeAsync(values, stream));
        CUDA_CHECK(cudaFreeAsync(scales, stream));
        return nullptr;
    }
    const int grid = (int) ((n_blocks + threads - 1) / threads);
    if (nvfp4) {
        cutlass_repack_nvfp4<<<grid, threads, 0, stream>>>((const block_nvfp4 *) tensor->data, (char *) values,
                                                           (uint8_t *) scales, k_blocks, padded_scale_blocks, rows,
                                                           padded_rows, groups);
    } else {
        cutlass_repack_mxfp4<<<grid, threads, 0, stream>>>((const block_mxfp4 *) tensor->data, (char *) values,
                                                           (uint8_t *) scales, k_blocks, padded_scale_blocks, rows,
                                                           padded_rows, groups);
    }
    CUDA_CHECK(cudaGetLastError());
    if (!preserve_source) {
        CUDA_CHECK(cudaMemcpyAsync(tensor->data, values, values_size, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaFreeAsync(values, stream));
        values = tensor->data;
    }
    ctx.cutlass_weight_cache.emplace_back();
    ggml_cuda_cutlass_weight_cache_entry * entry = &ctx.cutlass_weight_cache.back();
    entry->source                                = tensor;
    entry->source_data                           = tensor->data;
    entry->source_buffer                         = tensor->buffer;
    entry->source_buffer_generation              = ggml_cuda_buffer_get_generation(tensor->buffer);
    entry->ne[0]                                 = tensor->ne[0];
    entry->ne[1]                                 = tensor->ne[1];
    entry->ne[2]                                 = tensor->ne[2];
    entry->data                                  = values;
    entry->scales_data                           = scales;
    entry->owns_data                             = preserve_source;
    entry->preserves_source                      = preserve_source;
    entry->k                                     = padded_k;
    entry->scale_stride                          = padded_rows * padded_scale_blocks;
    CUDA_CHECK(cudaEventCreateWithFlags(&entry->ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(entry->ready, stream));
    return entry;
#    else
    GGML_UNUSED_VARS(ctx, tensor, stream, preserve_source);
    return nullptr;
#    endif
}

bool ggml_cuda_cutlass_repack_weight_pair(ggml_backend_cuda_context & ctx,
                                          const ggml_tensor *         first,
                                          const ggml_tensor *         second,
                                          ggml_cuda_cutlass_weight &  weight,
                                          cudaStream_t                stream) {
#    if CUDART_VERSION >= 12080
    if (first == nullptr || second == nullptr || first->buffer == nullptr || second->buffer == nullptr ||
        first->data == nullptr || second->data == nullptr ||
        (first->type != GGML_TYPE_NVFP4 && first->type != GGML_TYPE_MXFP4) || second->type != first->type ||
        !ggml_is_contiguous(first) || !ggml_is_contiguous(second) || !ggml_are_same_shape(first, second) ||
        first->ne[0] <= 0 || first->ne[1] <= 0 || first->ne[2] <= 0 || first->ne[0] > INT_MAX - 127 ||
        first->ne[1] > (INT_MAX - 127) / 2 || first->ne[2] > INT_MAX || first->ne[3] != 1) {
        return false;
    }
    const bool nvfp4             = first->type == GGML_TYPE_NVFP4;
    const int  qk                = nvfp4 ? QK_NVFP4 : QK_MXFP4;
    const int  scale_vector_size = nvfp4 ? QK_NVFP4_SUB : QK_MXFP4;
    if (first->ne[0] % qk != 0) {
        return false;
    }
    if (stream == nullptr) {
        stream = ctx.stream();
    }

    const uint64_t                         first_generation  = ggml_cuda_buffer_get_generation(first->buffer);
    const uint64_t                         second_generation = ggml_cuda_buffer_get_generation(second->buffer);
    ggml_cuda_cutlass_weight_cache_entry * entry             = nullptr;
    for (auto & candidate : ctx.cutlass_weight_cache) {
        if (candidate.source == first && candidate.source_data == first->data &&
            candidate.source_buffer == first->buffer && candidate.source_buffer_generation == first_generation &&
            candidate.source_secondary == second && candidate.source_secondary_data == second->data &&
            candidate.source_secondary_buffer == second->buffer &&
            candidate.source_secondary_buffer_generation == second_generation && candidate.ne[0] == first->ne[0] &&
            candidate.ne[1] == 2 * first->ne[1] && candidate.ne[2] == first->ne[2]) {
            entry = &candidate;
            break;
        }
    }

    if (entry == nullptr) {
        cudaStreamCaptureStatus capture_status;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
        if (capture_status != cudaStreamCaptureStatusNone) {
            return false;
        }

        const int k_blocks            = first->ne[0] / qk;
        const int padded_k            = GGML_PAD(first->ne[0], 128);
        const int padded_scale_blocks = padded_k / scale_vector_size;
        const int rows                = first->ne[1];
        const int rows_pair           = 2 * rows;
        const int padded_rows         = GGML_PAD(rows_pair, 128);
        const int groups              = first->ne[2];
        if ((int64_t) padded_rows * padded_scale_blocks > INT_MAX) {
            return false;
        }
        const size_t values_size = (size_t) groups * rows_pair * padded_k / 2;
        const size_t scales_size = (size_t) groups * padded_rows * padded_scale_blocks;

        ctx.cutlass_weight_cache.emplace_back();
        entry = &ctx.cutlass_weight_cache.back();
        CUDA_CHECK(cudaMallocAsync(&entry->data, values_size, stream));
        CUDA_CHECK(cudaMallocAsync(&entry->scales_data, scales_size, stream));
        CUDA_CHECK(cudaMemsetAsync(entry->scales_data, 0, scales_size, stream));

        constexpr int threads  = 256;
        const int64_t n_blocks = (int64_t) groups * rows_pair * padded_scale_blocks;
        const int64_t grid_64  = (n_blocks + threads - 1) / threads;
        if (grid_64 > INT_MAX) {
            CUDA_CHECK(cudaFreeAsync(entry->data, stream));
            CUDA_CHECK(cudaFreeAsync(entry->scales_data, stream));
            ctx.cutlass_weight_cache.pop_back();
            return false;
        }
        const int grid = (int) grid_64;
        if (nvfp4) {
            cutlass_repack_nvfp4_pair<<<grid, threads, 0, stream>>>(
                (const block_nvfp4 *) first->data, (const block_nvfp4 *) second->data, (char *) entry->data,
                (uint8_t *) entry->scales_data, k_blocks, padded_scale_blocks, rows, padded_rows, groups);
        } else {
            cutlass_repack_mxfp4_pair<<<grid, threads, 0, stream>>>(
                (const block_mxfp4 *) first->data, (const block_mxfp4 *) second->data, (char *) entry->data,
                (uint8_t *) entry->scales_data, k_blocks, padded_scale_blocks, rows, padded_rows, groups);
        }
        CUDA_CHECK(cudaGetLastError());

        entry->source                             = first;
        entry->source_data                        = first->data;
        entry->source_buffer                      = first->buffer;
        entry->source_buffer_generation           = first_generation;
        entry->source_secondary                   = second;
        entry->source_secondary_data              = second->data;
        entry->source_secondary_buffer            = second->buffer;
        entry->source_secondary_buffer_generation = second_generation;
        entry->ne[0]                              = first->ne[0];
        entry->ne[1]                              = rows_pair;
        entry->ne[2]                              = groups;
        entry->owns_data                          = true;
        entry->preserves_source                   = true;
        entry->k                                  = padded_k;
        entry->scale_stride                       = padded_rows * padded_scale_blocks;
        CUDA_CHECK(cudaEventCreateWithFlags(&entry->ready, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(entry->ready, stream));
    }

    weight = {
        (const char *) entry->data,
        (const uint8_t *) entry->scales_data,
        entry->k,
        entry->scale_stride,
        entry->ready,
        first->type,
    };
    return true;
#    else
    GGML_UNUSED_VARS(ctx, first, second, weight, stream);
    return false;
#    endif
}

bool ggml_cuda_cutlass_repack_weight(ggml_backend_cuda_context & ctx,
                                     const ggml_tensor *         tensor,
                                     ggml_cuda_cutlass_weight &  weight,
                                     cudaStream_t                stream,
                                     bool                        wait_ready,
                                     bool                        preserve_source) {
    if (tensor == nullptr || tensor->buffer == nullptr || tensor->data == nullptr ||
        (tensor->type != GGML_TYPE_MXFP4 && tensor->type != GGML_TYPE_NVFP4) || !ggml_is_contiguous(tensor)) {
        return false;
    }

    const int qk = tensor->type == GGML_TYPE_NVFP4 ? QK_NVFP4 : QK_MXFP4;
    if (tensor->ne[0] % qk != 0) {
        return false;
    }
    if (stream == nullptr) {
        stream = ctx.stream();
    }

    ggml_cuda_cutlass_weight_cache_entry * entry = cutlass_find_weight(ctx, tensor, preserve_source);
    if (entry == nullptr) {
        entry = cutlass_repack_weight(ctx, tensor, stream, preserve_source);
        if (entry == nullptr) {
            return false;
        }
    }

    if (wait_ready && entry->ready != nullptr) {
        CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), entry->ready, 0));
    }
    weight = {
        (const char *) entry->data,
        (const uint8_t *) entry->scales_data,
        entry->k,
        entry->scale_stride,
        entry->ready,
        tensor->type,
    };
    return true;
}

void ggml_cuda_cutlass_weight_wait_ready(const ggml_cuda_cutlass_weight & weight, cudaStream_t stream) {
    if (weight.ready != nullptr) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, weight.ready, 0));
    }
}

bool ggml_cuda_cutlass_get_inplace_weight(ggml_backend_cuda_context & ctx,
                                          const ggml_tensor *         tensor,
                                          ggml_cuda_cutlass_weight &  weight) {
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return false;
    }
    ggml_cuda_cutlass_weight_cache_entry * entry = cutlass_find_weight(ctx, tensor, false);
    if (entry == nullptr || entry->owns_data || entry->data != tensor->data) {
        return false;
    }
    weight = {
        (const char *) entry->data,
        (const uint8_t *) entry->scales_data,
        entry->k,
        entry->scale_stride,
        entry->ready,
        tensor->type,
    };
    return true;
}

#else

bool ggml_cuda_cutlass_repack_weight(ggml_backend_cuda_context & ctx,
                                     const ggml_tensor *         tensor,
                                     ggml_cuda_cutlass_weight &  weight,
                                     cudaStream_t                stream,
                                     bool                        wait_ready,
                                     bool                        preserve_source) {
    GGML_UNUSED_VARS(ctx, tensor, weight, stream, wait_ready, preserve_source);
    return false;
}

bool ggml_cuda_cutlass_repack_weight_pair(ggml_backend_cuda_context & ctx,
                                          const ggml_tensor *         first,
                                          const ggml_tensor *         second,
                                          ggml_cuda_cutlass_weight &  weight,
                                          cudaStream_t                stream) {
    GGML_UNUSED_VARS(ctx, first, second, weight, stream);
    return false;
}

void ggml_cuda_cutlass_weight_wait_ready(const ggml_cuda_cutlass_weight & weight, cudaStream_t stream) {
    GGML_UNUSED_VARS(weight, stream);
}

bool ggml_cuda_cutlass_get_inplace_weight(ggml_backend_cuda_context & ctx,
                                          const ggml_tensor *         tensor,
                                          ggml_cuda_cutlass_weight &  weight) {
    GGML_UNUSED_VARS(ctx, tensor, weight);
    return false;
}

#endif
