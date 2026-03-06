// MoE Q8_1 gather kernel - gathers selected rows from cached quantized data

#include "gather-q8.cuh"

constexpr int64_t BYTES_PER_BLOCK = 144;  // sizeof(block_q8_1_mmq)
constexpr int64_t INT4S_PER_BLOCK = 9;    // 144 / 16

template<int BLOCK_SIZE>
__global__ void gather_q8_1_kernel(
    const char* __restrict__ src,
    const int* __restrict__ ids,
    char* __restrict__ dst,
    int64_t n_blocks,
    int64_t n_src_rows_per_slice,
    int64_t n_dst_rows
) {
    const int64_t out_row = blockIdx.x;
    if (out_row >= n_dst_rows) return;

    const int src_flat_row = ids[out_row];
    const int64_t src_slice = src_flat_row / n_src_rows_per_slice;
    const int64_t src_row_in_slice = src_flat_row % n_src_rows_per_slice;
    const int64_t src_slice_base = src_slice * n_src_rows_per_slice * n_blocks;
    const int64_t total_int4s = n_blocks * INT4S_PER_BLOCK;

    for (int64_t i = threadIdx.x; i < total_int4s; i += BLOCK_SIZE) {
        const int64_t block_idx = i / INT4S_PER_BLOCK;
        const int64_t int4_offset = i - block_idx * INT4S_PER_BLOCK;

        const int64_t src_block_idx = src_slice_base + block_idx * n_src_rows_per_slice + src_row_in_slice;
        const int64_t dst_block_idx = block_idx * n_dst_rows + out_row;

        const int4* src_ptr = reinterpret_cast<const int4*>(src + src_block_idx * BYTES_PER_BLOCK) + int4_offset;
        int4* dst_ptr = reinterpret_cast<int4*>(dst + dst_block_idx * BYTES_PER_BLOCK) + int4_offset;
        *dst_ptr = *src_ptr;
    }
}

template<int BLOCK_SIZE>
__global__ void gather_q8_1_kernel_generic(
    const char* __restrict__ src,
    const int* __restrict__ ids,
    char* __restrict__ dst,
    int64_t block_size,
    int64_t n_blocks,
    int64_t n_src_rows_per_slice,
    int64_t n_dst_rows
) {
    const int64_t out_row = blockIdx.x;
    if (out_row >= n_dst_rows) return;

    const int src_flat_row = ids[out_row];
    const int64_t src_slice = src_flat_row / n_src_rows_per_slice;
    const int64_t src_row_in_slice = src_flat_row % n_src_rows_per_slice;
    const int64_t src_slice_base = src_slice * n_src_rows_per_slice * n_blocks;
    const int64_t int4s_per_block = block_size / 16;
    const int64_t total_int4s = n_blocks * int4s_per_block;

    for (int64_t i = threadIdx.x; i < total_int4s; i += BLOCK_SIZE) {
        const int64_t block_idx = i / int4s_per_block;
        const int64_t int4_offset = i % int4s_per_block;

        const int64_t src_block_idx = src_slice_base + block_idx * n_src_rows_per_slice + src_row_in_slice;
        const int64_t dst_block_idx = block_idx * n_dst_rows + out_row;

        const int4* src_ptr = reinterpret_cast<const int4*>(src + src_block_idx * block_size) + int4_offset;
        int4* dst_ptr = reinterpret_cast<int4*>(dst + dst_block_idx * block_size) + int4_offset;
        *dst_ptr = *src_ptr;
    }
}

void gather_q8_1_rows_cuda(
    const void* src,
    const int* ids,
    void* dst,
    int64_t block_size,
    int64_t n_blocks,
    int64_t n_src_rows_per_slice,
    int64_t n_dst_rows,
    cudaStream_t stream
) {
    if (n_dst_rows == 0) return;
    GGML_ASSERT(block_size % 16 == 0);

    constexpr int THREADS = 256;
    const int n_cuda_blocks = static_cast<int>(n_dst_rows);

    if (block_size == 144) {
        gather_q8_1_kernel<THREADS><<<n_cuda_blocks, THREADS, 0, stream>>>(
            static_cast<const char*>(src), ids, static_cast<char*>(dst),
            n_blocks, n_src_rows_per_slice, n_dst_rows);
    } else {
        gather_q8_1_kernel_generic<THREADS><<<n_cuda_blocks, THREADS, 0, stream>>>(
            static_cast<const char*>(src), ids, static_cast<char*>(dst),
            block_size, n_blocks, n_src_rows_per_slice, n_dst_rows);
    }
}
