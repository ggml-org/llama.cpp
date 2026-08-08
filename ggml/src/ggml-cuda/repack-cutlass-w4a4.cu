#include "repack-cutlass-w4a4.cuh"

#include <climits>
#include <cstring>

#ifdef GGML_CUDA_CUTLASS

template <typename block_t_, int block_values_, int scale_values_>
struct cutlass_repack_format_base {
    using block_t = block_t_;

    static constexpr int block_values = block_values_;
    static constexpr int scale_values = scale_values_;
    static constexpr int packed_bytes = scale_values / 2;

    static __device__ __forceinline__ uint8_t value(const block_t & block, int subblock, int element) {
        const uint8_t packed = block.qs[subblock * packed_bytes + element % packed_bytes];
        return element < packed_bytes ? packed & 0x0F : packed >> 4;
    }
};

struct cutlass_repack_mxfp4_traits : cutlass_repack_format_base<block_mxfp4, QK_MXFP4, QK_MXFP4> {
    static __device__ __forceinline__ uint8_t scale(const block_t & block, int) {
        return block.e;
    }

    static __device__ __forceinline__ void set_scale(block_t & block, int, uint8_t scale) {
        block.e = scale;
    }
};

struct cutlass_repack_nvfp4_traits : cutlass_repack_format_base<block_nvfp4, QK_NVFP4, QK_NVFP4_SUB> {
    static __device__ __forceinline__ uint8_t scale(const block_t & block, int subblock) {
        return block.d[subblock];
    }

    static __device__ __forceinline__ void set_scale(block_t & block, int subblock, uint8_t scale) {
        block.d[subblock] = scale;
    }
};

template <typename traits>
static __global__ void cutlass_repack_w4a4(const typename traits::block_t * source,
                                           char *                           values,
                                           uint8_t *                        scales,
                                           int                              k_blocks,
                                           int                              padded_scale_blocks,
                                           int                              rows,
                                           int                              padded_rows,
                                           int                              groups) {
    const int64_t index    = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n        = (int64_t) groups * rows * padded_scale_blocks;
    if (index >= n) {
        return;
    }

    const int     scale_block = index % padded_scale_blocks;
    const int64_t row_all     = index / padded_scale_blocks;
    const int     row         = row_all % rows;
    const int     group       = row_all / rows;
    uint8_t *     values_dst  = (uint8_t *) values + index * traits::packed_bytes;
    const int     k_base      = scale_block * traits::scale_values;
    if (k_base >= k_blocks * traits::block_values) {
        memset(values_dst, 0, traits::packed_bytes);
        return;
    }

    const int64_t source_row_all = (int64_t) group * rows + row;
    const int     subblock                  = (k_base % traits::block_values) / traits::scale_values;
    const typename traits::block_t block = source[source_row_all * k_blocks + k_base / traits::block_values];
#    pragma unroll
    for (int i = 0; i < traits::packed_bytes; ++i) {
        const uint8_t q0 = traits::value(block, subblock, 2 * i);
        const uint8_t q1 = traits::value(block, subblock, 2 * i + 1);
        values_dst[i]    = q0 | (q1 << 4);
    }

    const int64_t scale_group_stride = (int64_t) padded_rows * padded_scale_blocks;
    scales[(int64_t) group * scale_group_stride +
           ggml_cuda_cutlass_blockscaled_scale_offset(row, scale_block, padded_scale_blocks)] =
        traits::scale(block, subblock);
}

static void cutlass_repack_w4a4_launch(ggml_type     type,
                                       const void *  source,
                                       char *        values,
                                       uint8_t *     scales,
                                       int           k_blocks,
                                       int           padded_scale_blocks,
                                       int           rows,
                                       int           padded_rows,
                                       int           groups,
                                       int           grid,
                                       int           threads,
                                       cudaStream_t  stream) {
    if (type == GGML_TYPE_NVFP4) {
        cutlass_repack_w4a4<cutlass_repack_nvfp4_traits><<<grid, threads, 0, stream>>>(
            (const block_nvfp4 *) source, values, scales, k_blocks,
            padded_scale_blocks, rows, padded_rows, groups);
    } else {
        GGML_ASSERT(type == GGML_TYPE_MXFP4);
        cutlass_repack_w4a4<cutlass_repack_mxfp4_traits><<<grid, threads, 0, stream>>>(
            (const block_mxfp4 *) source, values, scales, k_blocks,
            padded_scale_blocks, rows, padded_rows, groups);
    }
}

template <typename traits>
static __global__ void cutlass_unpack_w4a4(const char *             values,
                                           const uint8_t *          scales,
                                           typename traits::block_t * dst,
                                           int                      k_blocks,
                                           int                      padded_scale_blocks,
                                           int                      rows,
                                           int                      padded_rows,
                                           int                      groups) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n     = (int64_t) groups * rows * k_blocks;
    if (index >= n) {
        return;
    }

    const int     k_block = index % k_blocks;
    const int64_t row_all = index / k_blocks;
    const int     row     = row_all % rows;
    const int     group   = row_all / rows;
    typename traits::block_t block = {};

    constexpr int subblocks = traits::block_values / traits::scale_values;
#    pragma unroll
    for (int subblock = 0; subblock < subblocks; ++subblock) {
        const int scale_block = k_block * subblocks + subblock;
        const uint8_t * values_src = (const uint8_t *) values +
            ((int64_t) row_all * padded_scale_blocks + scale_block) * traits::packed_bytes;
#        pragma unroll
        for (int i = 0; i < traits::packed_bytes; ++i) {
            const int first_element  = i;
            const int second_element = i + traits::packed_bytes;
            const uint8_t first_byte = values_src[first_element / 2];
            const uint8_t second_byte = values_src[second_element / 2];
            const uint8_t q0 = first_element % 2 == 0 ? first_byte & 0x0F : first_byte >> 4;
            const uint8_t q1 = second_element % 2 == 0 ? second_byte & 0x0F : second_byte >> 4;
            block.qs[subblock * traits::packed_bytes + i] = q0 | (q1 << 4);
        }
        const uint8_t scale = scales[(int64_t) group * padded_rows * padded_scale_blocks +
            ggml_cuda_cutlass_blockscaled_scale_offset(row, scale_block, padded_scale_blocks)];
        traits::set_scale(block, subblock, scale);
    }
    dst[index] = block;
}

static bool cutlass_size_mul(size_t a, size_t b, size_t & result) {
    if (a != 0 && b > SIZE_MAX / a) {
        return false;
    }
    result = a * b;
    return true;
}

bool ggml_cuda_cutlass_get_weight_layout(
        const ggml_tensor * tensor, ggml_cuda_cutlass_weight_layout & layout) {
    layout = {};
    if (tensor == nullptr || (tensor->type != GGML_TYPE_MXFP4 && tensor->type != GGML_TYPE_NVFP4) ||
        tensor->view_src != nullptr || !ggml_is_contiguous(tensor) || tensor->ne[0] <= 0 || tensor->ne[1] <= 0 ||
        tensor->ne[2] <= 0 || tensor->ne[3] != 1 || tensor->ne[0] > INT_MAX - 127 ||
        tensor->ne[1] > INT_MAX - 127 || tensor->ne[2] > INT_MAX) {
        return false;
    }

    const int qk = tensor->type == GGML_TYPE_NVFP4 ? QK_NVFP4 : QK_MXFP4;
    const int scale_values = tensor->type == GGML_TYPE_NVFP4 ? QK_NVFP4_SUB : QK_MXFP4;
    if (tensor->ne[0] % qk != 0) {
        return false;
    }

    layout.padded_k            = GGML_PAD((int) tensor->ne[0], 128);
    layout.padded_rows         = GGML_PAD((int) tensor->ne[1], 128);
    layout.padded_scale_blocks = layout.padded_k / scale_values;
    layout.k_blocks            = (int) tensor->ne[0] / qk;
    layout.rows                = (int) tensor->ne[1];
    layout.groups              = (int) tensor->ne[2];
    if ((int64_t) layout.padded_rows * layout.padded_scale_blocks > INT_MAX) {
        return false;
    }
    layout.scale_stride = layout.padded_rows * layout.padded_scale_blocks;

    size_t rows_all;
    size_t values_elements;
    size_t repack_elements;
    size_t unpack_elements;
    if (!cutlass_size_mul((size_t) layout.groups, (size_t) layout.rows, rows_all) ||
        !cutlass_size_mul(rows_all, (size_t) layout.padded_k, values_elements) ||
        !cutlass_size_mul((size_t) layout.groups, (size_t) layout.scale_stride, layout.scales_size) ||
        !cutlass_size_mul(rows_all, (size_t) layout.padded_scale_blocks, repack_elements) ||
        !cutlass_size_mul(rows_all, (size_t) layout.k_blocks, unpack_elements) ||
        repack_elements > (size_t) INT_MAX * 256 || unpack_elements > (size_t) INT_MAX * 256) {
        return false;
    }
    layout.values_size = values_elements / 2;
    if (layout.values_size > SIZE_MAX - 127) {
        return false;
    }
    layout.scales_offset = GGML_PAD(layout.values_size, (size_t) 128);
    if (layout.scales_size > SIZE_MAX - layout.scales_offset) {
        return false;
    }
    layout.allocation_size = layout.scales_offset + layout.scales_size;
    return layout.allocation_size >= ggml_nbytes(tensor);
}

bool ggml_cuda_cutlass_weight_from_tensor(
        const ggml_tensor * tensor, ggml_cuda_cutlass_weight & weight) {
    ggml_cuda_cutlass_weight_layout layout;
    if (tensor == nullptr || tensor->buffer == nullptr || tensor->data == nullptr ||
        !ggml_backend_buft_is_cuda_cutlass(ggml_backend_buffer_get_type(tensor->buffer)) ||
        !ggml_cuda_cutlass_get_weight_layout(tensor, layout)) {
        return false;
    }
    weight = {
        (const char *) tensor->data,
        (const uint8_t *) tensor->data + layout.scales_offset,
        layout.padded_k,
        layout.scale_stride,
        tensor->type,
    };
    return true;
}

bool ggml_cuda_cutlass_weight_supported(const ggml_tensor * tensor) {
    ggml_cuda_cutlass_weight_layout layout;
    return tensor != nullptr && tensor->buffer != nullptr &&
        ggml_backend_buft_is_cuda_cutlass(ggml_backend_buffer_get_type(tensor->buffer)) &&
        ggml_cuda_cutlass_get_weight_layout(tensor, layout);
}

bool ggml_cuda_cutlass_pack_weight(
        ggml_tensor * tensor, const void * canonical, cudaStream_t stream) {
    ggml_cuda_cutlass_weight_layout layout;
    if (tensor == nullptr || tensor->data == nullptr || canonical == nullptr ||
        !ggml_cuda_cutlass_get_weight_layout(tensor, layout)) {
        return false;
    }

    uint8_t * scales = (uint8_t *) tensor->data + layout.scales_offset;
    CUDA_CHECK(cudaMemsetAsync(scales, 0, layout.scales_size, stream));
    constexpr int threads = 256;
    const int64_t n = (int64_t) layout.groups * layout.rows * layout.padded_scale_blocks;
    const int64_t grid_64 = (n + threads - 1) / threads;
    if (grid_64 > INT_MAX) {
        return false;
    }
    cutlass_repack_w4a4_launch(
        tensor->type, canonical, (char *) tensor->data, scales, layout.k_blocks,
        layout.padded_scale_blocks, layout.rows, layout.padded_rows, layout.groups, (int) grid_64, threads, stream);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_cutlass_unpack_weight(
        const ggml_tensor * tensor, void * canonical, cudaStream_t stream) {
    ggml_cuda_cutlass_weight_layout layout;
    if (tensor == nullptr || tensor->data == nullptr || canonical == nullptr ||
        !ggml_cuda_cutlass_get_weight_layout(tensor, layout)) {
        return false;
    }

    constexpr int threads = 256;
    const int64_t n = (int64_t) layout.groups * layout.rows * layout.k_blocks;
    const int64_t grid_64 = (n + threads - 1) / threads;
    if (grid_64 > INT_MAX) {
        return false;
    }
    const uint8_t * scales = (const uint8_t *) tensor->data + layout.scales_offset;
    if (tensor->type == GGML_TYPE_NVFP4) {
        cutlass_unpack_w4a4<cutlass_repack_nvfp4_traits><<<(int) grid_64, threads, 0, stream>>>(
            (const char *) tensor->data, scales, (block_nvfp4 *) canonical, layout.k_blocks,
            layout.padded_scale_blocks, layout.rows, layout.padded_rows, layout.groups);
    } else {
        cutlass_unpack_w4a4<cutlass_repack_mxfp4_traits><<<(int) grid_64, threads, 0, stream>>>(
            (const char *) tensor->data, scales, (block_mxfp4 *) canonical, layout.k_blocks,
            layout.padded_scale_blocks, layout.rows, layout.padded_rows, layout.groups);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

#else

bool ggml_cuda_cutlass_get_weight_layout(
        const ggml_tensor * tensor, ggml_cuda_cutlass_weight_layout & layout) {
    GGML_UNUSED(tensor);
    layout = {};
    return false;
}

bool ggml_cuda_cutlass_weight_from_tensor(
        const ggml_tensor * tensor, ggml_cuda_cutlass_weight & weight) {
    GGML_UNUSED_VARS(tensor, weight);
    return false;
}

bool ggml_cuda_cutlass_weight_supported(const ggml_tensor * tensor) {
    GGML_UNUSED(tensor);
    return false;
}

bool ggml_cuda_cutlass_pack_weight(
        ggml_tensor * tensor, const void * canonical, cudaStream_t stream) {
    GGML_UNUSED_VARS(tensor, canonical, stream);
    return false;
}

bool ggml_cuda_cutlass_unpack_weight(
        const ggml_tensor * tensor, void * canonical, cudaStream_t stream) {
    GGML_UNUSED_VARS(tensor, canonical, stream);
    return false;
}

#endif
