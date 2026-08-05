#include "moe-mmq-repack.cuh"

#include <algorithm>
#include <cstring>

static constexpr size_t moe_mmq_repack_alignment = 256;

static __global__ void moe_mmq_repack_split(const block_mxfp4 * src,
                                            char *              qs,
                                            uint8_t *           scales,
                                            int                 src_k_blocks,
                                            int                 dst_k_blocks,
                                            int                 scale_stride,
                                            int64_t             n_rows) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n     = n_rows * dst_k_blocks;
    if (index >= n) {
        return;
    }

    const int     k_block = index % dst_k_blocks;
    const int64_t row     = index / dst_k_blocks;
    char *        qs_dst  = qs + index * (QK_MXFP4 / 2);
    if (k_block >= src_k_blocks) {
        memset(qs_dst, 0, QK_MXFP4 / 2);
        scales[row * scale_stride + k_block] = 0;
        return;
    }

    const block_mxfp4 * value = src + row * src_k_blocks + k_block;
    memcpy(qs_dst, value->qs, QK_MXFP4 / 2);
    scales[row * scale_stride + k_block] = value->e;
}

static __device__ __forceinline__ int64_t moe_mmq_cutlass_scale_offset(int row,
                                                                       int k_block,
                                                                       int padded_k_blocks) {
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

static __global__ void moe_mmq_repack_cutlass(const block_mxfp4 * src,
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
    uint8_t *     values_dst = (uint8_t *) values + index * (QK_MXFP4 / 2);
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

static __global__ void moe_mmq_repack_interleaved(const block_mxfp4 * src,
                                                  char *              dst,
                                                  int                 src_k_blocks,
                                                  int                 dst_k_blocks,
                                                  int64_t             n_rows) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n     = n_rows * dst_k_blocks;
    if (index >= n) {
        return;
    }

    const int     k_block = index % dst_k_blocks;
    const int64_t row     = index / dst_k_blocks;
    const int64_t group =
        row * (dst_k_blocks / ggml_cuda_moe_repack_group_blocks) + k_block / ggml_cuda_moe_repack_group_blocks;
    const int block_in_group = k_block % ggml_cuda_moe_repack_group_blocks;
    char *    record         = dst + group * ggml_cuda_moe_repack_group_bytes;
    char *    qs_dst         = record + block_in_group * (QK_MXFP4 / 2);
    uint8_t * scale_dst      = (uint8_t *) record + ggml_cuda_moe_repack_group_blocks * (QK_MXFP4 / 2) + block_in_group;
    if (k_block >= src_k_blocks) {
        memset(qs_dst, 0, QK_MXFP4 / 2);
        *scale_dst = 0;
        return;
    }

    const block_mxfp4 * value = src + row * src_k_blocks + k_block;
    memcpy(qs_dst, value->qs, QK_MXFP4 / 2);
    *scale_dst = value->e;
}

static __global__ void moe_mmq_repack_tma(const block_mxfp4 * src,
                                          char *              dst,
                                          int                 src_k_blocks,
                                          int                 rows,
                                          int                 rows_padded,
                                          int                 k_tiles,
                                          int                 experts) {
    const int64_t index     = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n_records = (int64_t) experts * k_tiles * rows_padded;
    const int64_t n         = n_records * ggml_cuda_moe_tma_k_blocks;
    if (index >= n) {
        return;
    }

    const int     k_block_in_tile = index % ggml_cuda_moe_tma_k_blocks;
    const int64_t record          = index / ggml_cuda_moe_tma_k_blocks;
    const int     row             = record % rows_padded;
    const int64_t group           = record / rows_padded;
    const int     k_tile          = group % k_tiles;
    const int     expert          = group / k_tiles;
    const int     k_block         = k_tile * ggml_cuda_moe_tma_k_blocks + k_block_in_tile;
    char *        record_dst      = dst + record * ggml_cuda_moe_tma_record_bytes;
    char *        qs_dst          = record_dst + k_block_in_tile * (QK_MXFP4 / 2);
    uint8_t *     scale_dst       = (uint8_t *) record_dst + ggml_cuda_moe_tma_data_bytes + k_block_in_tile;
    if (row >= rows || k_block >= src_k_blocks) {
        memset(qs_dst, 0, QK_MXFP4 / 2);
        *scale_dst = 0;
        return;
    }

    const block_mxfp4 * value = src + ((int64_t) expert * rows + row) * src_k_blocks + k_block;
    memcpy(qs_dst, value->qs, QK_MXFP4 / 2);
    *scale_dst = value->e;
}

static __global__ void moe_mmq_repack_tma_inplace(const block_mxfp4 * src,
                                                char *              dst,
                                                int                 k_blocks,
                                                int                 rows,
                                                int                 tma_tiles,
                                                int                 tail_blocks,
                                                int64_t             expert_stride,
                                                int64_t             tail_offset,
                                                int                 experts) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n     = (int64_t) experts * rows * k_blocks;
    if (index >= n) {
        return;
    }

    const int k_block = index % k_blocks;
    const int64_t row_global = index / k_blocks;
    const int row = row_global % rows;
    const int expert = row_global / rows;
    const block_mxfp4 * value = src + index;
    char * expert_dst = dst + expert * expert_stride;

    if (k_block < tma_tiles * ggml_cuda_moe_tma_k_blocks) {
        const int k_tile = k_block / ggml_cuda_moe_tma_k_blocks;
        const int block_in_tile = k_block % ggml_cuda_moe_tma_k_blocks;
        char * record = expert_dst + ((int64_t) k_tile * rows + row) * ggml_cuda_moe_tma_record_bytes;
        memcpy(record + block_in_tile * (QK_MXFP4 / 2), value->qs, QK_MXFP4 / 2);
        record[ggml_cuda_moe_tma_data_bytes + block_in_tile] = value->e;
        return;
    }

    const int tail_block = k_block - tma_tiles * ggml_cuda_moe_tma_k_blocks;
    block_mxfp4 * tail = reinterpret_cast<block_mxfp4 *>(
        expert_dst + tail_offset + ((int64_t) row * tail_blocks + tail_block) * sizeof(block_mxfp4));
    *tail = *value;
}

static void moe_mmq_encode_tma_map(uint64_t * dst, void * data, int64_t n_records, int tile_rows) {
#if CUDART_VERSION >= 12080
    static_assert(sizeof(CUtensorMap) == ggml_cuda_moe_tma_map_qwords * sizeof(uint64_t),
                  "unexpected CUtensorMap size");

    CUtensorMap      map;
    const cuuint64_t global_dim[2] = {
        ggml_cuda_moe_tma_record_bytes / sizeof(uint16_t),
        (cuuint64_t) n_records,
    };
    const cuuint64_t global_stride[1] = { ggml_cuda_moe_tma_record_bytes };
    const cuuint32_t box_dim[2]       = {
        ggml_cuda_moe_tma_record_bytes / sizeof(uint16_t),
        (cuuint32_t) tile_rows,
    };
    const cuuint32_t element_stride[2] = { 1, 1 };
    CU_CHECK(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT16, 2, data, global_dim, global_stride, box_dim,
                                    element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                    CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    memcpy(dst, &map, sizeof(map));
#else
    GGML_UNUSED_VARS(dst, data, n_records, tile_rows);
    GGML_ABORT("TMA MoE weights require CUDA 12.8 or newer");
#endif
}

static void moe_mmq_encode_tma_inplace_map(uint64_t * dst,
                                         void *     data,
                                         int        rows,
                                         int        tma_tiles,
                                         int        experts,
                                         int64_t    expert_stride,
                                         int        tile_rows) {
#if CUDART_VERSION >= 12080
    static_assert(sizeof(CUtensorMap) == ggml_cuda_moe_tma_map_qwords * sizeof(uint64_t),
                  "unexpected CUtensorMap size");

    CUtensorMap      map;
    const cuuint64_t global_dim[4] = {
        ggml_cuda_moe_tma_record_bytes / sizeof(uint16_t),
        (cuuint64_t) rows,
        (cuuint64_t) tma_tiles,
        (cuuint64_t) experts,
    };
    const cuuint64_t global_stride[3] = {
        ggml_cuda_moe_tma_record_bytes,
        (cuuint64_t) rows * ggml_cuda_moe_tma_record_bytes,
        (cuuint64_t) expert_stride,
    };
    const cuuint32_t box_dim[4] = {
        ggml_cuda_moe_tma_record_bytes / sizeof(uint16_t),
        (cuuint32_t) tile_rows,
        1,
        1,
    };
    const cuuint32_t element_stride[4] = { 1, 1, 1, 1 };
    CU_CHECK(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT16, 4, data, global_dim, global_stride, box_dim,
                                    element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                    CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    memcpy(dst, &map, sizeof(map));
#else
    GGML_UNUSED_VARS(dst, data, rows, tma_tiles, experts, expert_stride, tile_rows);
    GGML_ABORT("TMA MoE weights require CUDA 12.8 or newer");
#endif
}

static size_t moe_mmq_align(size_t value) {
    return (value + moe_mmq_repack_alignment - 1) & ~(moe_mmq_repack_alignment - 1);
}

static ggml_cuda_moe_weight_cache_entry * moe_mmq_find_weight(ggml_backend_cuda_context & ctx,
                                                              const ggml_tensor *         tensor,
                                                              ggml_cuda_moe_weight_layout layout) {
    for (auto & entry : ctx.moe_weight_cache) {
        if (entry.source == tensor && entry.source_data == tensor->data && entry.layout == (int) layout &&
            entry.ne[0] == tensor->ne[0] && entry.ne[1] == tensor->ne[1] && entry.ne[2] == tensor->ne[2]) {
            entry.last_used = ++ctx.moe_weight_cache_clock;
            return &entry;
        }
    }
    return nullptr;
}

static ggml_cuda_moe_weight_cache_entry * moe_mmq_acquire_weight(ggml_backend_cuda_context & ctx,
                                                                 size_t                      allocation_size,
                                                                 size_t                      cache_entries,
                                                                 cudaStream_t                stream) {
    const size_t owned_entries = std::count_if(ctx.moe_weight_cache.begin(), ctx.moe_weight_cache.end(),
                                               [](const auto & entry) { return entry.owns_data; });
    if (owned_entries < cache_entries) {
        ctx.moe_weight_cache.emplace_back();
        ctx.moe_weight_cache.back().owns_data = true;
        return &ctx.moe_weight_cache.back();
    }

    auto best = ctx.moe_weight_cache.end();
    for (auto it = ctx.moe_weight_cache.begin(); it != ctx.moe_weight_cache.end(); ++it) {
        if (!it->owns_data || it->allocation_size < allocation_size) {
            continue;
        }
        if (best == ctx.moe_weight_cache.end() || it->last_used < best->last_used) {
            best = it;
        }
    }
    if (best == ctx.moe_weight_cache.end()) {
        for (auto it = ctx.moe_weight_cache.begin(); it != ctx.moe_weight_cache.end(); ++it) {
            if (it->owns_data && (best == ctx.moe_weight_cache.end() || it->last_used < best->last_used)) {
                best = it;
            }
        }
    }

    GGML_ASSERT(best != ctx.moe_weight_cache.end());

    if (best->last_use != nullptr) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, best->last_use, 0));
    }
    return &*best;
}

static ggml_cuda_moe_weight_cache_entry * moe_mmq_repack_weight_inplace(ggml_backend_cuda_context & ctx,
                                                                      const ggml_tensor *         tensor,
                                                                      cudaStream_t                stream) {
#if CUDART_VERSION >= 12080
    const int k_blocks    = tensor->ne[0] / QK_MXFP4;
    const int rows        = tensor->ne[1];
    const int experts     = tensor->ne[2];
    const int tma_tiles   = k_blocks / ggml_cuda_moe_tma_k_blocks;
    const int tail_blocks = k_blocks - tma_tiles * ggml_cuda_moe_tma_k_blocks;
    if (tma_tiles <= 0 || tensor->ne[3] != 1) {
        return nullptr;
    }

    const int64_t tail_offset = (int64_t) tma_tiles * rows * ggml_cuda_moe_tma_record_bytes;
    const int64_t expert_stride = tail_offset + (int64_t) rows * tail_blocks * sizeof(block_mxfp4);
    const size_t tensor_size = ggml_nbytes(tensor);
    if ((size_t) expert_stride * experts != tensor_size || expert_stride % 16 != 0) {
        return nullptr;
    }

    void * temporary = nullptr;
    CUDA_CHECK(cudaMalloc(&temporary, tensor_size));
    constexpr int threads = 256;
    const int64_t n_blocks = (int64_t) experts * rows * k_blocks;
    const int grid = (int) ((n_blocks + threads - 1) / threads);
    moe_mmq_repack_tma_inplace<<<grid, threads, 0, stream>>>(
        (const block_mxfp4 *) tensor->data, (char *) temporary, k_blocks, rows, tma_tiles, tail_blocks, expert_stride,
        tail_offset, experts);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(tensor->data, temporary, tensor_size, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(temporary));

    ctx.moe_weight_cache.emplace_back();
    ggml_cuda_moe_weight_cache_entry * entry = &ctx.moe_weight_cache.back();
    entry->source         = tensor;
    entry->source_data    = tensor->data;
    entry->layout         = (int) ggml_cuda_moe_weight_layout::tma_inplace;
    entry->ne[0]          = tensor->ne[0];
    entry->ne[1]          = tensor->ne[1];
    entry->ne[2]          = tensor->ne[2];
    entry->data           = tensor->data;
    entry->owns_data      = false;
    entry->ncols          = tensor->ne[0];
    entry->stride_row     = k_blocks;
    entry->stride_channel = (int64_t) rows * k_blocks;
    entry->last_used      = ++ctx.moe_weight_cache_clock;
    entry->rows_padded    = rows;
    entry->k_tiles        = tma_tiles + (tail_blocks > 0);
    entry->tma_tiles      = tma_tiles;
    entry->tail_blocks    = tail_blocks;
    entry->expert_stride  = expert_stride;
    entry->tail_offset    = tail_offset;
    moe_mmq_encode_tma_inplace_map(entry->tma_map[0], entry->data, rows, tma_tiles, experts, expert_stride, 128);
    moe_mmq_encode_tma_inplace_map(entry->tma_map[1], entry->data, rows, tma_tiles, experts, expert_stride, 96);
    moe_mmq_encode_tma_inplace_map(entry->tma_map[2], entry->data, rows, tma_tiles, experts, expert_stride, 64);
    entry->tma_valid[0] = true;
    entry->tma_valid[1] = true;
    entry->tma_valid[2] = true;
    return entry;
#else
    GGML_UNUSED_VARS(ctx, tensor, stream);
    return nullptr;
#endif
}

static ggml_cuda_moe_weight_cache_entry * moe_mmq_repack_weight_cutlass(ggml_backend_cuda_context & ctx,
                                                                        const ggml_tensor *         tensor,
                                                                        cudaStream_t                stream) {
#if CUDART_VERSION >= 12080
    const int k_blocks        = tensor->ne[0] / QK_MXFP4;
    const int padded_k_blocks = GGML_PAD(k_blocks, 4);
    const int rows            = tensor->ne[1];
    const int padded_rows     = GGML_PAD(rows, 128);
    const int experts         = tensor->ne[2];
    if (tensor->ne[3] != 1) {
        return nullptr;
    }

    const size_t values_size = (size_t) experts * rows * padded_k_blocks * (QK_MXFP4 / 2);
    const size_t scales_size = (size_t) experts * padded_rows * padded_k_blocks;
    if (values_size > ggml_nbytes(tensor)) {
        return nullptr;
    }

    void * values = nullptr;
    void * scales = nullptr;
    CUDA_CHECK(cudaMallocAsync(&values, values_size, stream));
    CUDA_CHECK(cudaMallocAsync(&scales, scales_size, stream));
    CUDA_CHECK(cudaMemsetAsync(scales, 0, scales_size, stream));

    constexpr int threads = 256;
    const int64_t n_blocks = (int64_t) experts * rows * padded_k_blocks;
    const int grid = (int) ((n_blocks + threads - 1) / threads);
    moe_mmq_repack_cutlass<<<grid, threads, 0, stream>>>((const block_mxfp4 *) tensor->data, (char *) values,
                                                         (uint8_t *) scales, k_blocks, padded_k_blocks, rows,
                                                         padded_rows, experts);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(tensor->data, values, values_size, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaFreeAsync(values, stream));

    ctx.moe_weight_cache.emplace_back();
    ggml_cuda_moe_weight_cache_entry * entry = &ctx.moe_weight_cache.back();
    entry->source         = tensor;
    entry->source_data    = tensor->data;
    entry->layout         = (int) ggml_cuda_moe_weight_layout::cutlass;
    entry->ne[0]          = tensor->ne[0];
    entry->ne[1]          = tensor->ne[1];
    entry->ne[2]          = tensor->ne[2];
    entry->data           = tensor->data;
    entry->owns_data      = false;
    entry->scales_data    = scales;
    entry->owns_scales    = true;
    CUDA_CHECK(cudaEventCreateWithFlags(&entry->ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(entry->ready, stream));
    entry->ncols          = (int64_t) padded_k_blocks * QK_MXFP4;
    entry->stride_row     = padded_k_blocks;
    entry->stride_channel = (int64_t) rows * padded_k_blocks;
    entry->scale_stride   = padded_rows * padded_k_blocks;
    entry->last_used      = ++ctx.moe_weight_cache_clock;
    entry->rows_padded    = padded_rows;
    return entry;
#else
    GGML_UNUSED_VARS(ctx, tensor, stream);
    return nullptr;
#endif
}

bool ggml_cuda_moe_repack_weight(ggml_backend_cuda_context & ctx,
                                 const ggml_tensor *         tensor,
                                 ggml_cuda_moe_weight_layout layout,
                                 ggml_cuda_moe_weight_view & view,
                                 cudaStream_t                stream,
                                 size_t                      cache_entries,
                                 bool                        wait_ready) {
    GGML_ASSERT(tensor->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(ggml_is_contiguous(tensor));
    GGML_ASSERT(tensor->ne[0] % QK_MXFP4 == 0);
    GGML_ASSERT(cache_entries >= 2);

    if (stream == nullptr) {
        stream = ctx.stream();
    }

    const int src_k_blocks = tensor->ne[0] / QK_MXFP4;
    const int dst_k_blocks = GGML_PAD(src_k_blocks, ggml_cuda_moe_repack_group_blocks);

    if (layout == ggml_cuda_moe_weight_layout::canonical) {
        view = {
            (const char *) tensor->data, nullptr, tensor->ne[0], src_k_blocks, tensor->ne[1] * src_k_blocks, 0, layout,
        };
        return true;
    }

#if CUDART_VERSION < 12080
    if (layout == ggml_cuda_moe_weight_layout::tma || layout == ggml_cuda_moe_weight_layout::tma_inplace) {
        return false;
    }
#endif

    ggml_cuda_moe_weight_cache_entry * entry = moe_mmq_find_weight(ctx, tensor, layout);
    if (entry == nullptr && layout == ggml_cuda_moe_weight_layout::tma_inplace) {
        entry = moe_mmq_repack_weight_inplace(ctx, tensor, stream);
        if (entry == nullptr) {
            return false;
        }
    }
    if (entry == nullptr && layout == ggml_cuda_moe_weight_layout::cutlass) {
        entry = moe_mmq_repack_weight_cutlass(ctx, tensor, stream);
        if (entry == nullptr) {
            return false;
        }
    }
    if (entry == nullptr) {
        const int     rows         = tensor->ne[1];
        const int     experts      = tensor->ne[2];
        const int64_t n_rows       = (int64_t) rows * experts;
        const int     scale_stride = layout == ggml_cuda_moe_weight_layout::split ? GGML_PAD(dst_k_blocks, 32) : 0;
        const int     rows_padded  = layout == ggml_cuda_moe_weight_layout::tma ? GGML_PAD(rows, 128) : rows;
        const int     k_tiles      = layout == ggml_cuda_moe_weight_layout::tma ?
                                         (src_k_blocks + ggml_cuda_moe_tma_k_blocks - 1) / ggml_cuda_moe_tma_k_blocks :
                                         0;
        const int64_t tma_records  = (int64_t) experts * k_tiles * rows_padded;
        const size_t  qs_size =
            layout == ggml_cuda_moe_weight_layout::split ?
                 n_rows * dst_k_blocks * (QK_MXFP4 / 2) :
             layout == ggml_cuda_moe_weight_layout::tma ?
                 tma_records * ggml_cuda_moe_tma_record_bytes :
                 n_rows * (dst_k_blocks / ggml_cuda_moe_repack_group_blocks) * ggml_cuda_moe_repack_group_bytes;
        const size_t scales_offset = layout == ggml_cuda_moe_weight_layout::split ? moe_mmq_align(qs_size) : 0;
        const size_t scales_size   = layout == ggml_cuda_moe_weight_layout::split ? n_rows * scale_stride : 0;
        const size_t allocation_size =
            layout == ggml_cuda_moe_weight_layout::split ? scales_offset + scales_size : qs_size;

        entry = moe_mmq_acquire_weight(ctx, allocation_size, cache_entries, stream);
        if (entry->allocation_size < allocation_size) {
            if (entry->data != nullptr) {
                GGML_ASSERT(entry->owns_data);
                CUDA_CHECK(cudaFree(entry->data));
            }
            CUDA_CHECK(cudaMalloc(&entry->data, allocation_size));
            entry->owns_data      = true;
            entry->allocation_size = allocation_size;
        }
        if (entry->ready == nullptr) {
            CUDA_CHECK(cudaEventCreateWithFlags(&entry->ready, cudaEventDisableTiming));
        }
        if (entry->last_use == nullptr) {
            CUDA_CHECK(cudaEventCreateWithFlags(&entry->last_use, cudaEventDisableTiming));
        }
        constexpr int threads  = 256;
        const int64_t n_blocks = n_rows * dst_k_blocks;
        const int     blocks   = (int) ((n_blocks + threads - 1) / threads);
        if (layout == ggml_cuda_moe_weight_layout::split) {
            moe_mmq_repack_split<<<blocks, threads, 0, stream>>>(
                (const block_mxfp4 *) tensor->data, (char *) entry->data, (uint8_t *) entry->data + scales_offset,
                src_k_blocks, dst_k_blocks, scale_stride, n_rows);
        } else if (layout == ggml_cuda_moe_weight_layout::tma) {
            const int64_t tma_blocks = tma_records * ggml_cuda_moe_tma_k_blocks;
            const int     tma_grid   = (int) ((tma_blocks + threads - 1) / threads);
            moe_mmq_repack_tma<<<tma_grid, threads, 0, stream>>>((const block_mxfp4 *) tensor->data,
                                                                 (char *) entry->data, src_k_blocks, rows, rows_padded,
                                                                 k_tiles, experts);
        } else {
            GGML_ASSERT(layout == ggml_cuda_moe_weight_layout::interleaved);
            moe_mmq_repack_interleaved<<<blocks, threads, 0, stream>>>(
                (const block_mxfp4 *) tensor->data, (char *) entry->data, src_k_blocks, dst_k_blocks, n_rows);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(entry->ready, stream));

        entry->source         = tensor;
        entry->source_data    = tensor->data;
        entry->layout         = (int) layout;
        entry->ne[0]          = tensor->ne[0];
        entry->ne[1]          = tensor->ne[1];
        entry->ne[2]          = tensor->ne[2];
        entry->scales_offset  = scales_offset;
        entry->ncols          = (int64_t) dst_k_blocks * QK_MXFP4;
        entry->stride_row     = dst_k_blocks;
        entry->stride_channel = tensor->ne[1] * dst_k_blocks;
        entry->scale_stride   = scale_stride;
        entry->last_used      = ++ctx.moe_weight_cache_clock;
        entry->rows_padded    = 0;
        entry->k_tiles        = 0;
        entry->tma_tiles      = 0;
        entry->tail_blocks    = 0;
        entry->expert_stride  = 0;
        entry->tail_offset    = 0;
        memset(entry->tma_valid, 0, sizeof(entry->tma_valid));
        memset(entry->tma_map, 0, sizeof(entry->tma_map));
        if (layout == ggml_cuda_moe_weight_layout::tma) {
            entry->rows_padded = rows_padded;
            entry->k_tiles     = k_tiles;
            entry->tma_tiles   = k_tiles;
            moe_mmq_encode_tma_map(entry->tma_map[0], entry->data, tma_records, 128);
            moe_mmq_encode_tma_map(entry->tma_map[1], entry->data, tma_records, 96);
            moe_mmq_encode_tma_map(entry->tma_map[2], entry->data, tma_records, 64);
            entry->tma_valid[0] = true;
            entry->tma_valid[1] = GGML_PAD(rows, 96) <= rows_padded;
            entry->tma_valid[2] = GGML_PAD(rows, 64) <= rows_padded;
        }
    }

    if (wait_ready && entry->ready != nullptr) {
        CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), entry->ready, 0));
    }
    view = {
        (const char *) entry->data,
        entry->scales_data != nullptr ? (const uint8_t *) entry->scales_data :
        entry->scales_offset == 0 ? nullptr : (const uint8_t *) entry->data + entry->scales_offset,
        entry->ncols,
        entry->stride_row,
        entry->stride_channel,
        entry->scale_stride,
        layout,
    };
    view.rows_padded = entry->rows_padded;
    view.k_tiles     = entry->k_tiles;
    view.tma_tiles   = entry->tma_tiles;
    view.tail_blocks = entry->tail_blocks;
    view.expert_stride = entry->expert_stride;
    view.tail_offset = entry->tail_offset;
    for (int mode = 0; mode < ggml_cuda_moe_tma_modes; ++mode) {
        view.tma_valid[mode] = entry->tma_valid[mode];
        memcpy(view.tma_map[mode], entry->tma_map[mode], sizeof(view.tma_map[mode]));
    }
    view.ready    = entry->ready;
    view.last_use = entry->last_use;
    return true;
}

bool ggml_cuda_moe_weight_is_inplace_repacked(const ggml_backend_cuda_context & ctx, const ggml_tensor * tensor) {
    for (const auto & entry : ctx.moe_weight_cache) {
        if (entry.source == tensor && entry.source_data == tensor->data &&
            (entry.layout == (int) ggml_cuda_moe_weight_layout::tma_inplace ||
             entry.layout == (int) ggml_cuda_moe_weight_layout::cutlass)) {
            return true;
        }
    }
    return false;
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
