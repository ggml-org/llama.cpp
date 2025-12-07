#pragma once

#include "common.hpp"

//
// Block-addressed KV write for Paged Attention V2
//
// This kernel writes KV data from a contiguous source tensor to a 4D paged destination tensor.
// The destination layout is [D, block_size, n_heads, num_blocks] which enables efficient
// block-based memory access patterns for long-context attention.
//
// Source layout: [D * n_heads, n_tokens] - contiguous KV data
// Destination layout: [D, block_size, n_heads, num_blocks] - paged blocked layout
//
// The block_table maps logical block indices to physical block indices, enabling
// non-contiguous memory allocation for different sequences.
//
// Supports type conversion: F32->F16, F16->F32, F32->F32, F16->F16
//

template<typename TIn, typename TOut, typename IdxT>
static void set_rows_paged_kernel(
    const TIn * __restrict__ src,             // [D * n_heads, n_tokens]
    TOut * __restrict__ dst,                  // [D, block_size, n_heads, num_blocks]
    const IdxT * __restrict__ indices,        // [n_tokens] - logical positions in KV cache
    const int32_t * __restrict__ block_table, // [max_blocks_per_seq, n_seqs]
    const int64_t D,                          // Head dimension
    const int64_t n_heads,                    // Number of KV heads
    const int64_t n_tokens,                   // Number of tokens to write
    const int32_t block_size,                 // Tokens per block (typically 16)
    const int32_t seq_idx,                    // Sequence index for block table lookup
    const int64_t max_blocks_per_seq,         // Max blocks per sequence in block table
    const sycl::nd_item<1> & item)
{
    // Global thread ID
    const int64_t tid = item.get_global_linear_id();

    // Total elements to copy: n_tokens * n_heads * D
    const int64_t total_elements = n_tokens * n_heads * D;
    if (tid >= total_elements) {
        return;
    }

    // Decompose tid into (token, head, d)
    const int64_t d = tid % D;
    const int64_t head = (tid / D) % n_heads;
    const int64_t token = tid / (D * n_heads);

    // Get the logical position for this token in the KV cache
    const int32_t logical_pos = (int32_t)indices[token];

    // Compute logical block index and offset within block
    const int32_t logical_block = logical_pos / block_size;
    const int32_t offset_in_block = logical_pos % block_size;

    // Look up physical block from block table or use identity mapping
    // block_table layout: [max_blocks_per_seq, n_seqs]
    // Note: For identity mapping (block_table == nullptr), physical = logical
    const int32_t physical_block = block_table ?
        block_table[seq_idx * max_blocks_per_seq + logical_block] : logical_block;

    // Compute source offset: src is [D * n_heads, n_tokens]
    // src[head * D + d, token]
    const int64_t src_offset = token * (D * n_heads) + head * D + d;

    // Compute destination offset: dst is [D, block_size, n_heads, num_blocks]
    // dst[d, offset_in_block, head, physical_block]
    // Strides: 1, D, D*block_size, D*block_size*n_heads
    const int64_t dst_offset = physical_block * (n_heads * block_size * D) +
                               head * (block_size * D) +
                               offset_in_block * D +
                               d;

    // Copy with type conversion
    const TIn val = src[src_offset];
    dst[dst_offset] = static_cast<TOut>(val);
}

// Dispatch function for set_rows_paged with type conversion
template<typename TIn, typename TOut, typename IdxT>
static void set_rows_paged_sycl(
    const TIn * src,
    TOut * dst,
    const IdxT * indices,
    const int32_t * block_table,
    const int64_t D,
    const int64_t n_heads,
    const int64_t n_tokens,
    const int32_t block_size,
    const int32_t seq_idx,
    const int64_t max_blocks_per_seq,
    queue_ptr stream)
{
    const int64_t total_elements = n_tokens * n_heads * D;
    constexpr int work_group_size = 256;
    const int64_t n_work_groups = (total_elements + work_group_size - 1) / work_group_size;

    stream->parallel_for(
        sycl::nd_range<1>(n_work_groups * work_group_size, work_group_size),
        [=](sycl::nd_item<1> item) {
            set_rows_paged_kernel<TIn, TOut, IdxT>(
                src, dst, indices, block_table,
                D, n_heads, n_tokens, block_size, seq_idx, max_blocks_per_seq,
                item);
        });
}

// Main dispatch function that handles type dispatch
static void ggml_sycl_op_set_rows_paged(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->op == GGML_OP_SET_ROWS_PAGED);

    const ggml_tensor * src = dst->src[0];         // Source data [D * n_heads, n_tokens]
    const ggml_tensor * indices = dst->src[1];     // Indices [n_tokens]
    const ggml_tensor * block_table = dst->src[2]; // Block table [max_blocks, n_seqs]
    const ggml_tensor * dst_orig = dst->src[3];    // Original destination tensor

    const int32_t block_size = ((const int32_t *) dst->op_params)[0];
    const int32_t seq_idx = ((const int32_t *) dst->op_params)[1];
    const int32_t use_identity = ((const int32_t *) dst->op_params)[2];

    // Get tensor dimensions
    // dst (the view) is [D, block_size, n_heads, num_blocks]
    const int64_t D = dst->ne[0];
    const int64_t n_heads = dst->ne[2];

    // src is [D * n_heads, n_tokens]
    const int64_t n_tokens = src->ne[1];

    // block_table is [max_blocks, n_seqs]
    const int64_t max_blocks_per_seq = block_table->ne[0];

    GGML_SYCL_DEBUG("%s: src_type=%s dst_type=%s D=%lld n_heads=%lld n_tokens=%lld block_size=%d seq_idx=%d max_blocks=%lld idx_type=%s\n",
                    __func__, ggml_type_name(src->type), ggml_type_name(dst_orig->type),
                    (long long)D, (long long)n_heads, (long long)n_tokens,
                    block_size, seq_idx, (long long)max_blocks_per_seq, ggml_type_name(indices->type));

    // Get SYCL stream
    queue_ptr stream = ctx.stream();

    // Get data pointers (using dst_orig for the actual destination)
    const void * src_d = src->data;
    void * dst_d = dst_orig->data;
    // Use nullptr for block_table when use_identity flag is set (identity mapping: physical = logical)
    const int32_t * block_table_d = use_identity ? nullptr : (const int32_t *) block_table->data;

    // Dispatch based on source and destination types
    const bool use_i64 = (indices->type == GGML_TYPE_I64);
    const ggml_type src_type = src->type;
    const ggml_type dst_type = dst_orig->type;

    // Dispatch: (src_type, dst_type, idx_type) -> kernel instantiation
    if (src_type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F32) {
        if (use_i64) {
            set_rows_paged_sycl<float, float, int64_t>(
                (const float *) src_d, (float *) dst_d,
                (const int64_t *) indices->data, block_table_d,
                D, n_heads, n_tokens, block_size, seq_idx, max_blocks_per_seq, stream);
        } else {
            set_rows_paged_sycl<float, float, int32_t>(
                (const float *) src_d, (float *) dst_d,
                (const int32_t *) indices->data, block_table_d,
                D, n_heads, n_tokens, block_size, seq_idx, max_blocks_per_seq, stream);
        }
    } else if (src_type == GGML_TYPE_F32 && dst_type == GGML_TYPE_F16) {
        // Common case: model computes in F32, KV cache stores in F16
        if (use_i64) {
            set_rows_paged_sycl<float, sycl::half, int64_t>(
                (const float *) src_d, (sycl::half *) dst_d,
                (const int64_t *) indices->data, block_table_d,
                D, n_heads, n_tokens, block_size, seq_idx, max_blocks_per_seq, stream);
        } else {
            set_rows_paged_sycl<float, sycl::half, int32_t>(
                (const float *) src_d, (sycl::half *) dst_d,
                (const int32_t *) indices->data, block_table_d,
                D, n_heads, n_tokens, block_size, seq_idx, max_blocks_per_seq, stream);
        }
    } else if (src_type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F16) {
        if (use_i64) {
            set_rows_paged_sycl<sycl::half, sycl::half, int64_t>(
                (const sycl::half *) src_d, (sycl::half *) dst_d,
                (const int64_t *) indices->data, block_table_d,
                D, n_heads, n_tokens, block_size, seq_idx, max_blocks_per_seq, stream);
        } else {
            set_rows_paged_sycl<sycl::half, sycl::half, int32_t>(
                (const sycl::half *) src_d, (sycl::half *) dst_d,
                (const int32_t *) indices->data, block_table_d,
                D, n_heads, n_tokens, block_size, seq_idx, max_blocks_per_seq, stream);
        }
    } else if (src_type == GGML_TYPE_F16 && dst_type == GGML_TYPE_F32) {
        if (use_i64) {
            set_rows_paged_sycl<sycl::half, float, int64_t>(
                (const sycl::half *) src_d, (float *) dst_d,
                (const int64_t *) indices->data, block_table_d,
                D, n_heads, n_tokens, block_size, seq_idx, max_blocks_per_seq, stream);
        } else {
            set_rows_paged_sycl<sycl::half, float, int32_t>(
                (const sycl::half *) src_d, (float *) dst_d,
                (const int32_t *) indices->data, block_table_d,
                D, n_heads, n_tokens, block_size, seq_idx, max_blocks_per_seq, stream);
        }
    } else {
        GGML_ABORT("GGML_OP_SET_ROWS_PAGED: unsupported type combination src=%s dst=%s\n",
                   ggml_type_name(src_type), ggml_type_name(dst_type));
    }
}
