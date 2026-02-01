//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_FATTN_XMX_F16_HPP
#define GGML_SYCL_FATTN_XMX_F16_HPP

#include "fattn-common.hpp"
#include <sycl/sycl.hpp>
#include <cfloat>

// Check for joint_matrix support
#if __has_include(<sycl/ext/oneapi/matrix/matrix.hpp>)
#define SYCL_XMX_AVAILABLE 1
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#else
#define SYCL_XMX_AVAILABLE 0
#endif

#if SYCL_XMX_AVAILABLE

namespace sycl_xmx = sycl::ext::oneapi::experimental::matrix;

// =============================================================================
// FP8 E4M3 Dequantization for KV Cache
// =============================================================================
// Converts FP8 E4M3 (1 sign, 4 exponent, 3 mantissa, bias=7) to FP16
// This enables 2x memory savings for KV cache with on-the-fly dequantization

inline sycl::half fp8_e4m3_to_half(uint8_t bits) {
    uint32_t sign = (bits >> 7) & 0x1;
    uint32_t exp = (bits >> 3) & 0xF;
    uint32_t mant = bits & 0x7;

    float result;

    if (exp == 0) {
        if (mant == 0) {
            // Zero
            result = sign ? -0.0f : 0.0f;
        } else {
            // Subnormal: value = (-1)^sign * 2^(-6) * (0.mant)
            // = (-1)^sign * mant * 2^(-9)
            result = (sign ? -1.0f : 1.0f) * (float)mant * (1.0f / 512.0f);  // 2^-9
        }
    } else if (exp == 15 && mant == 7) {
        // NaN in E4M3
        result = sycl::nan(0u);
    } else {
        // Normal: value = (-1)^sign * 2^(exp-7) * (1.mant)
        // Rebias: E4M3 bias=7, FP32 bias=127
        int32_t fp32_exp = (int32_t)exp - 7 + 127;
        uint32_t fp32_mant = mant << 20;  // 3 bits -> 23 bits
        uint32_t fp32_bits = (sign << 31) | (fp32_exp << 23) | fp32_mant;
        // Use union for type punning (SYCL-safe)
        union { uint32_t u; float f; } pun;
        pun.u = fp32_bits;
        result = pun.f;
    }

    return sycl::half(result);
}

// =============================================================================
// PagedAttention Helper Functions
// =============================================================================
// These functions translate logical KV positions to physical memory addresses
// using the block table. The block table maps logical block indices to physical
// block indices, enabling non-contiguous K/V storage.

// Compute physical memory offset for a K/V position in paged mode
// block_table: [batch_size, max_blocks_per_seq] - maps logical->physical blocks
// seq_idx: sequence index in the batch
// kv_pos: logical KV position within the sequence
// block_size: tokens per block (typically 16)
// max_blocks: max blocks per sequence
// D: head dimension
// Returns: byte offset into K/V block storage
inline int64_t paged_kv_offset(
    const int32_t * block_table,
    int seq_idx,
    int kv_pos,
    int block_size,
    int max_blocks,
    int D) {
    const int logical_block = kv_pos / block_size;
    const int offset_in_block = kv_pos % block_size;
    const int physical_block = block_table[seq_idx * max_blocks + logical_block];
    // K/V block layout: [n_blocks, block_size, D] in half precision
    // Physical offset = physical_block * block_size * D + offset_in_block * D
    return (int64_t)(physical_block * block_size + offset_in_block) * D * sizeof(sycl::half);
}

// =============================================================================
// XMX Configuration for Intel Arc GPUs
// =============================================================================

// Debug flag - set to 1 to enable debug output
#define FATTN_XMX_DEBUG 0

// Intel Arc XMX tile dimensions (verified working)
constexpr int XMX_TM = 8;    // Tile rows (queries per XMX op)
constexpr int XMX_TN = 16;   // Tile cols (KV positions per XMX op)
constexpr int XMX_TK = 16;   // Reduction dimension
constexpr int XMX_SG = 16;   // Sub-group size

// Work-group configuration
// 512 threads provides optimal balance of occupancy vs SLM pressure
constexpr int XMX_NTHREADS = 512;
constexpr int XMX_N_SG = XMX_NTHREADS / XMX_SG;  // 32 sub-groups

// Number of KV positions to process per main loop iteration
// Optimal: 32 balances loop overhead vs XMX efficiency
constexpr int XMX_BATCH_KV = 32;

// Shared memory padding to reduce bank conflicts (32 banks on Intel)
// IMPORTANT: joint_matrix_load requires stride to be divisible by 8!
// With XMX_BATCH_KV=16, stride is already 16 (divisible by 8), no pad needed.
constexpr int XMX_PAD = 0;

// =============================================================================
// Flash Attention XMX Kernel - With Double Buffering for K
// =============================================================================
// Template parameters:
//   D: head dimension (64, 80, 96, 128, 256)
//   ncols: number of query columns processed per work-group
//   use_logit_softcap: enable logit softcapping
//   Q_type: query tensor type (float or sycl::half)
//   kv_is_fp8: if true, K/V are FP8 E4M3 and need on-the-fly dequantization

template <int D, int ncols, bool use_logit_softcap, typename Q_type, bool kv_is_fp8 = false>
static void flash_attn_xmx_f16_kernel(
    const char * __restrict__ Q,
    const char * __restrict__ K,
    const char * __restrict__ V,
    const char * __restrict__ mask,
    const char * __restrict__ sinks,
    float * __restrict__ dst,
    float scale,
    float max_bias,
    float m0,
    float m1,
    uint32_t n_head_log2,
    float logit_softcap,
    int ne00, int ne01, int ne02, int ne03,
    int nb01, int nb02, int nb03,
    int ne10, int ne11, int ne12, int ne13,
    int nb11, int nb12, int64_t nb13,
    int nb21, int nb22, int64_t nb23,
    int ne30, int ne31, int ne32, int ne33,
    int nb31, int nb32, int64_t nb33,
    // Multi-sequence batching parameters (can be null for single-sequence)
    // Legacy offset-based approach:
    int n_seqs,
    const int32_t * __restrict__ seq_q_offsets,
    const int32_t * __restrict__ seq_kv_offsets,
    // New per-token/per-position sequence ID approach:
    const int32_t * __restrict__ q_seq_ids,   // [ne01] Sequence ID for each query
    const int32_t * __restrict__ kv_seq_ids,  // [ne11] Sequence ID for each KV position
    // PagedAttention parameters:
    bool use_paged_attn,
    int32_t block_size,
    int32_t max_blocks_per_seq,
    const int32_t * __restrict__ block_table,  // [batch, max_blocks] logical->physical block mapping
    const int32_t * __restrict__ seq_lens,     // [batch] sequence lengths
    // Multi-token decode parameters (speculative decoding / multi-step generation):
    bool multi_token_decode,
    const int32_t * __restrict__ q_positions,  // [ne01] Position for each query (for per-query causal boundary)
    int32_t kv_base_pos,                       // Base position of KV cache
    const sycl::nd_item<3> & item,
    sycl::half * shared_mem) {

    static_assert(D % XMX_TK == 0, "Head dimension D must be divisible by XMX_TK (16)");
    static_assert(XMX_BATCH_KV % XMX_TN == 0, "BATCH_KV must be divisible by XMX_TN (16)");

    // PagedAttention note: seq_lens is used to get per-sequence KV length when needed
    // Currently the kernel uses mask-based KV bounds, but seq_lens can be used for optimization
    (void)seq_lens;

    auto sg = item.get_sub_group();
    const int sg_id = sg.get_group_linear_id();
    const int tid = item.get_local_linear_id();

    // Work-group indices
    const int ic0 = item.get_group(2) * ncols;
    const int sequence = item.get_group(0) / ne02;
    const int head = item.get_group(0) % ne02;
    const int gqa_ratio = ne02 / ne12;

    // Pointers to this work-group's data
    const char * Q_base = Q + nb03 * sequence + nb02 * head;
    const int kv_head = head / gqa_ratio;

    // In unified KV mode (ne13=1), all sequences share the same K/V buffer.
    // Only use sequence index 0 for K/V when ne13=1, otherwise each sequence
    // has its own K/V slice at nb13*sequence offset.
    const int kv_sequence = (ne13 == 1) ? 0 : sequence;
    const char * K_base = K + nb13 * kv_sequence + nb12 * kv_head;
    const char * V_base = V + nb23 * kv_sequence + nb22 * kv_head;

    // Mask setup - use nb31/sizeof(half) as the stride (matches CUDA implementation)
    // This is critical because the mask tensor may have padding (nb31 != ne30 * sizeof(half))
    const int stride_mask = nb31 / sizeof(sycl::half);
    const int mask_head = ne32 > 1 ? head % ne32 : 0;
    const sycl::half * maskh = mask ?
        reinterpret_cast<const sycl::half*>(mask + nb33 * (sequence % ne33) + nb32 * mask_head + nb31 * ic0) : nullptr;
    // Always apply ALiBi slope when enabled to match CPU semantics; get_alibi_slope returns 1.0f when max_bias <= 0.
    const float slope = get_alibi_slope(max_bias, head, n_head_log2, m0, m1);

    // =========================================================================
    // Multi-sequence KV bounds computation
    // =========================================================================
    // When n_seqs > 1, queries in this work-group may belong to different sequences.
    // We need to determine the valid KV range for all queries in this work-group.
    // For simplicity, we find the union of all sequences' KV ranges that overlap
    // with queries in this work-group.
    int seq_kv_start = 0;
    int seq_kv_end = ne11;  // Default: full KV range

    if (n_seqs > 1 && seq_q_offsets != nullptr && seq_kv_offsets != nullptr) {
        // Find which sequences have queries in this work-group [ic0, ic0 + ncols)
        // Binary search could be used, but for small n_seqs linear search is fine
        int first_seq = -1, last_seq = -1;

        for (int s = 0; s < n_seqs; ++s) {
            int q_start = seq_q_offsets[s];
            int q_end = seq_q_offsets[s + 1];

            // Check if any query in this work-group belongs to sequence s
            // Work-group processes queries [ic0, ic0 + ncols)
            if (ic0 < q_end && (ic0 + ncols) > q_start) {
                if (first_seq < 0) first_seq = s;
                last_seq = s;
            }
        }

        if (first_seq >= 0) {
            // KV range is the union of all overlapping sequences
            seq_kv_start = seq_kv_offsets[first_seq];
            seq_kv_end = seq_kv_offsets[last_seq + 1];
        }
    }

    // Causal skip optimization: compute the last query position in this work-group
    // For causal attention, query at position P can attend to KV positions 0..P
    // So we can skip KV batches entirely beyond the last query's causal boundary
    const int last_q_pos = sycl::min(ic0 + ncols - 1, ne01 - 1);

    // =========================================================================
    // Shared memory layout - WITH DOUBLE BUFFERING for K^T AND XMX S@V
    // =========================================================================
    // XMX requires at least XMX_TM rows for Q tile, even if ncols < XMX_TM
    constexpr int ncols_padded = (ncols < XMX_TM) ? XMX_TM : ncols;

    // tile_Q:      [ncols_padded][D] half - padded to XMX_TM for XMX loads
    // tile_KT[2]:  [D][XMX_BATCH_KV + PAD] half x 2 - K transposed, DOUBLE BUFFERED
    // tile_V:      [XMX_BATCH_KV][D + PAD] half - V tile with padding for XMX stride
    // tile_S:      [ncols_padded][XMX_BATCH_KV + PAD] half - Softmax weights for XMX S@V
    // QK_acc:      [ncols_padded][XMX_BATCH_KV] float - QK scores before softmax
    // SV_acc:      [ncols_padded][D] float - S@V result for current batch
    constexpr int KT_STRIDE = XMX_BATCH_KV + XMX_PAD;  // Padded stride (must be divisible by 8!)
    constexpr int KT_SIZE = D * KT_STRIDE;
    constexpr int V_STRIDE = D + XMX_PAD;  // V stride with padding for XMX
    constexpr int S_STRIDE = XMX_BATCH_KV + XMX_PAD;  // S stride with padding for XMX

    sycl::half * tile_Q = shared_mem;
    sycl::half * tile_KT[2];
    tile_KT[0] = tile_Q + ncols_padded * D;
    tile_KT[1] = tile_KT[0] + KT_SIZE;  // Second buffer for double buffering
    sycl::half * tile_V = tile_KT[1] + KT_SIZE;
    sycl::half * tile_S = tile_V + XMX_BATCH_KV * V_STRIDE;
    float * QK_acc = reinterpret_cast<float*>(tile_S + ncols_padded * S_STRIDE);
    float * SV_acc = QK_acc + ncols_padded * XMX_BATCH_KV;

    // =========================================================================
    // Load Q into shared memory (scaled) - ALL threads participate
    // Zero-pad to ncols_padded for XMX tile alignment
    // Use vectorized loads (half4) for better memory bandwidth
    // =========================================================================
    // Phase 6.4: Vectorized Q loading with half4 when possible
    constexpr int Q_VEC_SIZE = 4;  // Load 4 halfs at a time
    constexpr int Q_TOTAL_VECS = (ncols_padded * D) / Q_VEC_SIZE;

    // Vectorized path: process 4 elements at a time
    for (int vec_idx = tid; vec_idx < Q_TOTAL_VECS; vec_idx += XMX_NTHREADS) {
        const int elem_idx = vec_idx * Q_VEC_SIZE;
        const int j = elem_idx / D;
        const int d = elem_idx % D;

        sycl::half4 q_vec;
        if (j < ncols && ic0 + j < ne01 && d + 3 < D) {
            const Q_type * Q_ptr = reinterpret_cast<const Q_type*>(Q_base + nb01 * (ic0 + j));
            // Load and scale 4 elements
            q_vec = sycl::half4(
                static_cast<sycl::half>(static_cast<float>(Q_ptr[d]) * scale),
                static_cast<sycl::half>(static_cast<float>(Q_ptr[d + 1]) * scale),
                static_cast<sycl::half>(static_cast<float>(Q_ptr[d + 2]) * scale),
                static_cast<sycl::half>(static_cast<float>(Q_ptr[d + 3]) * scale));
        } else {
            // Padding or boundary - load zeros
            q_vec = sycl::half4(0.0f);
        }
        *reinterpret_cast<sycl::half4*>(&tile_Q[j * D + d]) = q_vec;
    }

    // Handle remainder (if D not divisible by 4)
    constexpr int Q_REMAINDER = (ncols_padded * D) % Q_VEC_SIZE;
    if (Q_REMAINDER > 0) {
        for (int idx = tid; idx < Q_REMAINDER; idx += XMX_NTHREADS) {
            const int elem_idx = Q_TOTAL_VECS * Q_VEC_SIZE + idx;
            const int j = elem_idx / D;
            const int d = elem_idx % D;
            if (j < ncols && ic0 + j < ne01) {
                const Q_type * Q_ptr = reinterpret_cast<const Q_type*>(Q_base + nb01 * (ic0 + j));
                tile_Q[j * D + d] = static_cast<sycl::half>(static_cast<float>(Q_ptr[d]) * scale);
            } else {
                tile_Q[j * D + d] = sycl::half(0.0f);
            }
        }
    }

    sycl::group_barrier(item.get_group());

    // =========================================================================
    // Phase 2b: Pre-compute per-query KV bounds for tile-level skip optimization
    // =========================================================================
    // Each query j belongs to a sequence with a specific KV range.
    // By pre-computing these bounds, we can skip Q@K tiles where no query
    // in the tile has valid KV positions, saving wasted XMX computation.
    int q_kv_start_arr[ncols];
    int q_kv_end_arr[ncols];

    if (n_seqs > 1 && seq_kv_offsets != nullptr && q_seq_ids != nullptr) {
        // Multi-sequence mode: compute per-query KV bounds
        for (int j = 0; j < ncols; ++j) {
            if (ic0 + j < ne01) {
                const int q_seq = q_seq_ids[ic0 + j];
                if (q_seq >= 0 && q_seq < n_seqs) {
                    q_kv_start_arr[j] = seq_kv_offsets[q_seq];
                    q_kv_end_arr[j] = seq_kv_offsets[q_seq + 1];
                } else {
                    // Invalid sequence, allow full range
                    q_kv_start_arr[j] = 0;
                    q_kv_end_arr[j] = ne11;
                }
            } else {
                // Padding query, no valid KV
                q_kv_start_arr[j] = 0;
                q_kv_end_arr[j] = 0;
            }
        }
    } else {
        // Single-sequence mode: all queries use full KV range
        for (int j = 0; j < ncols; ++j) {
            q_kv_start_arr[j] = 0;
            q_kv_end_arr[j] = ne11;
        }
    }

    // =========================================================================
    // Per-thread accumulators for online softmax
    // =========================================================================
    constexpr int D_per_thread = (D + XMX_NTHREADS - 1) / XMX_NTHREADS;
    float VKQ[ncols][D_per_thread];
    float KQ_max[ncols];
    float KQ_sum[ncols];

    #pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ_max[j] = -FLT_MAX / 2.0f;
        KQ_sum[j] = 0.0f;
        #pragma unroll
        for (int i = 0; i < D_per_thread; ++i) {
            VKQ[j][i] = 0.0f;
        }
    }

    // =========================================================================
    // Prefetch first K batch into buffer 0
    // =========================================================================
    // Start from seq_kv_start when multi-sequence batching is enabled
    const int kv_loop_start = seq_kv_start;
    int kv_count_prefetch = sycl::min(XMX_BATCH_KV, seq_kv_end - kv_loop_start);

    // Load K[kv_loop_start:kv_loop_start+BATCH_KV] transposed into tile_KT[0]
    for (int idx = tid; idx < kv_count_prefetch * D; idx += XMX_NTHREADS) {
        const int k = idx / D;
        const int d = idx % D;
        const int kv_pos = kv_loop_start + k;

        const char * K_row_base;
        if (use_paged_attn && block_table != nullptr) {
            const int logical_block = kv_pos / block_size;
            const int offset_in_block = kv_pos % block_size;
            const int physical_block = block_table[sequence * max_blocks_per_seq + logical_block];
            const int token_pos = physical_block * block_size + offset_in_block;
            K_row_base = K_base + nb11 * token_pos;
        } else {
            K_row_base = K_base + nb11 * kv_pos;
        }

        if constexpr (kv_is_fp8) {
            // FP8 E4M3: dequantize on-the-fly
            const uint8_t * K_row_fp8 = reinterpret_cast<const uint8_t*>(K_row_base);
            tile_KT[0][d * KT_STRIDE + k] = fp8_e4m3_to_half(K_row_fp8[d]);
        } else {
            // FP16: direct load
            const sycl::half * K_row = reinterpret_cast<const sycl::half*>(K_row_base);
            tile_KT[0][d * KT_STRIDE + k] = K_row[d];
        }
    }
    // Zero-pad if first batch is partial
    if (kv_count_prefetch < XMX_BATCH_KV) {
        for (int idx = tid; idx < D * (XMX_BATCH_KV - kv_count_prefetch); idx += XMX_NTHREADS) {
            const int d = idx / (XMX_BATCH_KV - kv_count_prefetch);
            const int k_off = idx % (XMX_BATCH_KV - kv_count_prefetch);
            if (d < D) {
                tile_KT[0][d * KT_STRIDE + kv_count_prefetch + k_off] = sycl::half(0.0f);
            }
        }
    }

    sycl::group_barrier(item.get_group());

    // =========================================================================
    // Main loop over K/V sequence - WITH DOUBLE BUFFERING AND CAUSAL SKIP
    // =========================================================================
    int buf_compute = 0;  // Buffer index for computation (starts with prefetched data)

    // Causal skip optimization: During prefill (ne01 > ncols), we can skip KV batches
    // that are entirely beyond the causal boundary of all queries in this work-group.
    //
    // For causal attention during prefill:
    // - Query at position P can attend to KV positions 0..P
    // - The last query in this work-group is at absolute position (ic0 + ncols - 1)
    // - So we only need to process KV batches where kv_start <= last_q_pos
    //
    // Important: This optimization only works during prefill when queries are processed
    // at their final sequence positions. During generation with KV cache, the query index
    // (ic0) doesn't correspond to its absolute position, so we can't use this optimization.
    //
    // We detect prefill by checking if ne01 > ncols (multiple work-groups of queries).
    //
    // Multi-sequence batching: When enabled, we also bound the loop by seq_kv_end to only
    // process KV positions belonging to the sequences in this work-group.
    const bool is_prefill = (ne01 > ncols);
    int kv_loop_end_bound = seq_kv_end;  // Default bound from multi-sequence info
    // NOTE: mask can be arbitrary (not necessarily causal). Do not apply causal skip
    // based on mask presence, or we can incorrectly skip valid KV positions.

    for (int kv_start = kv_loop_start; kv_start < kv_loop_end_bound; kv_start += XMX_BATCH_KV) {
        const int kv_end = sycl::min(kv_start + XMX_BATCH_KV, seq_kv_end);
        const int kv_count = kv_end - kv_start;

        // Current K^T buffer to use for computation
        sycl::half * tile_KT_cur = tile_KT[buf_compute];

        // Next K^T buffer to load into (for prefetching)
        const int buf_load = 1 - buf_compute;
        sycl::half * tile_KT_next = tile_KT[buf_load];

        // Compute next batch bounds for prefetching
        const int next_kv_start = kv_start + XMX_BATCH_KV;
        const int next_kv_end = sycl::min(next_kv_start + XMX_BATCH_KV, seq_kv_end);
        const int next_kv_count = next_kv_end - next_kv_start;
        // Only prefetch if next batch is within loop boundary
        const bool has_next = (next_kv_start < kv_loop_end_bound);

        // ---------------------------------------------------------------------
        // PHASE 1: Compute Q @ K^T using XMX (using tile_KT_cur)
        // ---------------------------------------------------------------------
#if FATTN_XMX_DEBUG
        // Debug: Check K data in tile_KT_cur for first few elements
        if (head == 0 && tid == 0 && (ic0 + ncols - 1) == (ne01 - 1)) {
            sycl::ext::oneapi::experimental::printf(
                "[K_CHECK] kv_batch=%d kv_start=%d ne01=%d ne11=%d kv_loop_end=%d seq_kv_end=%d\n",
                kv_start / XMX_BATCH_KV, kv_start, ne01, ne11, kv_loop_end_bound, seq_kv_end);
        }
#endif
        // Phase 2b DISABLED - no need to initialize QK_acc since all tiles computed
        {

            // Compute QK = Q @ K^T
            for (int q_tile = sg_id * XMX_TM; q_tile < ncols_padded; q_tile += XMX_N_SG * XMX_TM) {
                if (q_tile >= ncols_padded) continue;

                for (int k_tile = 0; k_tile < XMX_BATCH_KV; k_tile += XMX_TN) {
                    // Phase 2b: Check if this XMX tile (q_tile:q_tile+8 x k_tile:k_tile+16)
                    // has any valid query-KV pairs. If not, skip the XMX computation.
                    // KV positions in this tile: [kv_start + k_tile, kv_start + k_tile + XMX_TN)
                    const int tile_kv_start = kv_start + k_tile;
                    const int tile_kv_end = tile_kv_start + XMX_TN;

                    // Phase 2b DISABLED for debugging - always compute all tiles
                    // The cross-sequence masking phase will still filter out invalid pairs
                    bool tile_has_valid = true;
                    (void)tile_kv_start;
                    (void)tile_kv_end;

                    sycl_xmx::joint_matrix<sycl::sub_group, sycl::half, sycl_xmx::use::a, XMX_TM, XMX_TK, sycl_xmx::layout::row_major> mat_Q;
                    sycl_xmx::joint_matrix<sycl::sub_group, sycl::half, sycl_xmx::use::b, XMX_TK, XMX_TN, sycl_xmx::layout::row_major> mat_KT;
                    sycl_xmx::joint_matrix<sycl::sub_group, float, sycl_xmx::use::accumulator, XMX_TM, XMX_TN> mat_QK;

                    sycl_xmx::joint_matrix_fill(sg, mat_QK, 0.0f);

                    #pragma unroll
                    for (int d_tile = 0; d_tile < D; d_tile += XMX_TK) {
                        sycl_xmx::joint_matrix_load(sg, mat_Q,
                            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                                &tile_Q[q_tile * D + d_tile]),
                            D);

                        sycl_xmx::joint_matrix_load(sg, mat_KT,
                            sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                                &tile_KT_cur[d_tile * KT_STRIDE + k_tile]),
                            KT_STRIDE);

                        sycl_xmx::joint_matrix_mad(sg, mat_QK, mat_Q, mat_KT, mat_QK);
                    }

                    sycl_xmx::joint_matrix_store(sg, mat_QK,
                        sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                            &QK_acc[q_tile * XMX_BATCH_KV + k_tile]),
                        XMX_BATCH_KV, sycl_xmx::layout::row_major);
                }
            }
        }
        sycl::group_barrier(item.get_group());

        // ---------------------------------------------------------------------
        // PHASE 2: Apply mask/softcap to QK, load V, AND prefetch next K
        // ---------------------------------------------------------------------

        // Apply mask, logit softcap, and cross-sequence masking to QK_acc (vectorized)
        // Process 4 elements at a time using float4
        //
        // Cross-sequence optimization: When q_seq_ids and kv_seq_ids are provided,
        // we mask out (set to -INF) any attention between queries and KV positions
        // that belong to different sequences. This is critical for efficient
        // multi-sequence batching in continuous batching scenarios.

        for (int j = tid; j < ncols; j += XMX_NTHREADS) {
            if (ic0 + j >= ne01) continue;

            float * qk_row = &QK_acc[j * XMX_BATCH_KV];
            // Use stride_mask (nb31/sizeof(half)) for mask indexing, matching CUDA implementation
            const sycl::half * mask_row = maskh ? &maskh[j * stride_mask + kv_start] : nullptr;

            // Get query's sequence ID for cross-sequence masking
            const int32_t q_seq = q_seq_ids ? q_seq_ids[ic0 + j] : -1;

            int k = 0;
            // Vectorized processing with float4
            for (; k + 3 < kv_count; k += 4) {
                sycl::float4 qk = *reinterpret_cast<sycl::float4*>(&qk_row[k]);

                if (use_logit_softcap) {
                    qk = logit_softcap * sycl::float4(
                        sycl::tanh(qk.x()), sycl::tanh(qk.y()),
                        sycl::tanh(qk.z()), sycl::tanh(qk.w()));
                }

                if (mask_row) {
                    // Avoid unaligned half4 loads when stride_mask is not a multiple of 4.
                    const sycl::half mh0 = mask_row[k + 0];
                    const sycl::half mh1 = mask_row[k + 1];
                    const sycl::half mh2 = mask_row[k + 2];
                    const sycl::half mh3 = mask_row[k + 3];
                    const sycl::float4 mask_val(static_cast<float>(mh0), static_cast<float>(mh1),
                                                static_cast<float>(mh2), static_cast<float>(mh3));

                    qk += slope * mask_val;
                }

                // Cross-sequence masking: set to -INF if KV belongs to different sequence
                // Vectorized: load 4 seq_ids at once using int4
                if (q_seq >= 0 && kv_seq_ids) {
                    const int kv_idx_base = kv_start + k;
                    // Vectorized load of 4 int32 sequence IDs
                    sycl::int4 kv_seqs = *reinterpret_cast<const sycl::int4*>(&kv_seq_ids[kv_idx_base]);
                    // Mask if: KV has valid seq (>= 0) AND it doesn't match query's seq
                    if (kv_seqs.x() >= 0 && kv_seqs.x() != q_seq) qk.x() = -FLT_MAX;
                    if (kv_seqs.y() >= 0 && kv_seqs.y() != q_seq) qk.y() = -FLT_MAX;
                    if (kv_seqs.z() >= 0 && kv_seqs.z() != q_seq) qk.z() = -FLT_MAX;
                    if (kv_seqs.w() >= 0 && kv_seqs.w() != q_seq) qk.w() = -FLT_MAX;
                }

                // Multi-token decode: per-query position-based causal masking
                // Each query can only attend to KV positions <= its own position
                if (multi_token_decode && q_positions) {
                    const int32_t q_pos = q_positions[ic0 + j];
                    const int kv_idx_base = kv_start + k;
                    // KV position = kv_base_pos + kv_idx
                    if (kv_base_pos + kv_idx_base + 0 > q_pos) qk.x() = -FLT_MAX;
                    if (kv_base_pos + kv_idx_base + 1 > q_pos) qk.y() = -FLT_MAX;
                    if (kv_base_pos + kv_idx_base + 2 > q_pos) qk.z() = -FLT_MAX;
                    if (kv_base_pos + kv_idx_base + 3 > q_pos) qk.w() = -FLT_MAX;
                }

                *reinterpret_cast<sycl::float4*>(&qk_row[k]) = qk;
            }
            // Handle remainder
            for (; k < kv_count; ++k) {
                float qk_val = qk_row[k];
                if (use_logit_softcap) {
                    qk_val = logit_softcap * sycl::tanh(qk_val);
                }
                if (mask_row) {
                    qk_val += slope * static_cast<float>(mask_row[k]);
                }
                // Cross-sequence masking for remainder elements
                if (q_seq >= 0 && kv_seq_ids) {
                    const int32_t kv_seq = kv_seq_ids[kv_start + k];
                    if (kv_seq >= 0 && kv_seq != q_seq) {
                        qk_val = -FLT_MAX;
                    }
                }
                // Multi-token decode: per-query position-based causal masking
                if (multi_token_decode && q_positions) {
                    const int32_t q_pos = q_positions[ic0 + j];
                    if (kv_base_pos + kv_start + k > q_pos) {
                        qk_val = -FLT_MAX;
                    }
                }
                qk_row[k] = qk_val;
            }
        }

        // Load V tile for current batch (with stride padding for XMX)
        if constexpr (kv_is_fp8) {
            // FP8 E4M3: element-by-element dequantization (can't vectorize)
            for (int idx = tid; idx < kv_count * D; idx += XMX_NTHREADS) {
                const int k = idx / D;
                const int d = idx % D;
                const int kv_pos = kv_start + k;

                const char * V_row_base;
                if (use_paged_attn && block_table != nullptr) {
                    const int logical_block = kv_pos / block_size;
                    const int offset_in_block = kv_pos % block_size;
                    const int physical_block = block_table[sequence * max_blocks_per_seq + logical_block];
                    const int token_pos = physical_block * block_size + offset_in_block;
                    V_row_base = V_base + nb21 * token_pos;
                } else {
                    V_row_base = V_base + nb21 * kv_pos;
                }
                const uint8_t * V_row_fp8 = reinterpret_cast<const uint8_t*>(V_row_base);
                tile_V[k * V_STRIDE + d] = fp8_e4m3_to_half(V_row_fp8[d]);
            }
        } else {
            // FP16: Vectorized V loading with half4 for better memory bandwidth
            constexpr int V_VEC_SIZE = 4;
            const int v_total_vecs = (kv_count * D) / V_VEC_SIZE;

            for (int vec_idx = tid; vec_idx < v_total_vecs; vec_idx += XMX_NTHREADS) {
                const int elem_idx = vec_idx * V_VEC_SIZE;
                const int k = elem_idx / D;
                const int d = elem_idx % D;
                const int kv_pos = kv_start + k;

                const sycl::half * V_row;
                if (use_paged_attn && block_table != nullptr) {
                    const int logical_block = kv_pos / block_size;
                    const int offset_in_block = kv_pos % block_size;
                    const int physical_block = block_table[sequence * max_blocks_per_seq + logical_block];
                    const int token_pos = physical_block * block_size + offset_in_block;
                    V_row = reinterpret_cast<const sycl::half*>(V_base + nb21 * token_pos);
                } else {
                    V_row = reinterpret_cast<const sycl::half*>(V_base + nb21 * kv_pos);
                }
                sycl::half4 v_vec = *reinterpret_cast<const sycl::half4*>(&V_row[d]);
                *reinterpret_cast<sycl::half4*>(&tile_V[k * V_STRIDE + d]) = v_vec;
            }
            // Handle remainder (if kv_count * D not divisible by 4)
            const int v_remainder_start = v_total_vecs * V_VEC_SIZE;
            for (int idx = tid; idx < (kv_count * D) - v_remainder_start; idx += XMX_NTHREADS) {
                const int elem_idx = v_remainder_start + idx;
                const int k = elem_idx / D;
                const int d = elem_idx % D;
                const int kv_pos = kv_start + k;

                const sycl::half * V_row;
                if (use_paged_attn && block_table != nullptr) {
                    const int logical_block = kv_pos / block_size;
                    const int offset_in_block = kv_pos % block_size;
                    const int physical_block = block_table[sequence * max_blocks_per_seq + logical_block];
                    const int token_pos = physical_block * block_size + offset_in_block;
                    V_row = reinterpret_cast<const sycl::half*>(V_base + nb21 * token_pos);
                } else {
                    V_row = reinterpret_cast<const sycl::half*>(V_base + nb21 * kv_pos);
                }
                tile_V[k * V_STRIDE + d] = V_row[d];
            }
        }
        // Zero-pad V stride padding (XMX_PAD is 0 now, but keep for safety)
        if (XMX_PAD > 0) {
            for (int idx = tid; idx < kv_count * XMX_PAD; idx += XMX_NTHREADS) {
                const int k_idx = XMX_PAD > 0 ? idx / XMX_PAD : 0;
                const int p_idx = XMX_PAD > 0 ? idx % XMX_PAD : 0;
                if (k_idx < kv_count && D + p_idx < V_STRIDE) { // Added bounds check for safety
                    tile_V[k_idx * V_STRIDE + D + p_idx] = sycl::half(0.0f);
                }
            }
        }

#if FATTN_XMX_DEBUG
        // Debug: Log prefetch info
        if (head == 0 && tid == 0 && (ic0 + ncols - 1) == (ne01 - 1)) {
            sycl::ext::oneapi::experimental::printf(
                "[PREFETCH] kv_batch=%d has_next=%d next_kv_start=%d next_kv_count=%d buf_load=%d\n",
                kv_start / XMX_BATCH_KV, has_next ? 1 : 0, next_kv_start, next_kv_count, buf_load);
        }
#endif
        // Prefetch next K batch into tile_KT_next
        if (has_next) {
#if FATTN_XMX_DEBUG
            // Debug: Check we are actually loading K
            int load_count = 0;
#endif
            for (int idx = tid; idx < next_kv_count * D; idx += XMX_NTHREADS) {
                const int k = idx / D;
                const int d = idx % D;
                const int kv_pos = next_kv_start + k;
#if FATTN_XMX_DEBUG
                load_count++;
#endif

                const char * K_row_base;
                if (use_paged_attn && block_table != nullptr) {
                    const int logical_block = kv_pos / block_size;
                    const int offset_in_block = kv_pos % block_size;
                    const int physical_block = block_table[sequence * max_blocks_per_seq + logical_block];
                    const int token_pos = physical_block * block_size + offset_in_block;
                    K_row_base = K_base + nb11 * token_pos;
                } else {
                    K_row_base = K_base + nb11 * kv_pos;
                }

                sycl::half val;
                if constexpr (kv_is_fp8) {
                    const uint8_t * K_row_fp8 = reinterpret_cast<const uint8_t*>(K_row_base);
                    val = fp8_e4m3_to_half(K_row_fp8[d]);
                } else {
                    const sycl::half * K_row = reinterpret_cast<const sycl::half*>(K_row_base);
                    val = K_row[d];
                }
                tile_KT_next[d * KT_STRIDE + k] = val;
#if FATTN_XMX_DEBUG
                // Debug: Log what we're loading
                if (head == 0 && (ic0 + ncols - 1) == (ne01 - 1) && idx == 0) {
                    sycl::ext::oneapi::experimental::printf(
                        "[PREFETCH_ELEM] tid=%d kv_pos=%d k=%d d=%d pos=%d val=%.4f K_base_offset=%ld\n",
                        tid, kv_pos, k, d, d * KT_STRIDE + k, static_cast<float>(val),
                        (long)(K_row_base - K_base));
                }
#endif
            }
#if FATTN_XMX_DEBUG
            // Debug: Log prefetch result - check at position [0*KT_STRIDE + 0] which should have d=0,k=0
            if (head == 0 && tid == 0 && (ic0 + ncols - 1) == (ne01 - 1)) {
                sycl::ext::oneapi::experimental::printf(
                    "[PREFETCH_DONE] loaded %d elems into buf[%d], tile_KT_next[0,KT_STRIDE,2*KT_STRIDE]=%.4f %.4f %.4f\n",
                    load_count, buf_load,
                    static_cast<float>(tile_KT_next[0]),
                    static_cast<float>(tile_KT_next[KT_STRIDE]),
                    static_cast<float>(tile_KT_next[2*KT_STRIDE]));
            }
#endif
            // Zero-pad if next batch is partial
            if (next_kv_count < XMX_BATCH_KV) {
                for (int idx = tid; idx < D * (XMX_BATCH_KV - next_kv_count); idx += XMX_NTHREADS) {
                    const int d = idx / (XMX_BATCH_KV - next_kv_count);
                    const int k_off = idx % (XMX_BATCH_KV - next_kv_count);
                    if (d < D) {
                        tile_KT_next[d * KT_STRIDE + next_kv_count + k_off] = sycl::half(0.0f);
                    }
                }
            }
        }

        sycl::group_barrier(item.get_group());

        // ---------------------------------------------------------------------
        // PHASE 3: Softmax computation and store to tile_S
        // ---------------------------------------------------------------------
        // Use a portion of SV_acc to temporarily store per-query max values
        // SV_acc has ncols_padded * D floats, we only need ncols floats for max
        float * batch_max_shared = SV_acc;  // Reuse SV_acc temporarily

        // First pass: compute max per query row and store to shared memory
        // Use vectorized reduction for better performance
        // NOTE: FATTN_KQ_MAX_OFFSET removed - was causing bug by being inconsistent
        // between batch_max computation and softmax weight computation
#if FATTN_XMX_DEBUG
        // Debug: Log before batch_max computation for last query, head 0
        const bool dbg_log = (head == 0 && tid == 0 && (ic0 + ncols - 1) == (ne01 - 1));
        if (dbg_log) {
            sycl::ext::oneapi::experimental::printf(
                "\n=== Query ic0=%d (last=%d), head=%d, kv_start=%d, kv_count=%d ===\n",
                ic0, ne01-1, head, kv_start, kv_count);
        }
#endif
        for (int j = tid; j < ncols; j += XMX_NTHREADS) {
            float batch_max = -FLT_MAX;
            if (ic0 + j < ne01) {
                const float * row = &QK_acc[j * XMX_BATCH_KV];
#if FATTN_XMX_DEBUG
                // Debug: Check QK_acc values for tid=0 and 1
                if (head == 0 && (ic0 + ncols - 1) == (ne01 - 1) && j == 0) {
                    sycl::ext::oneapi::experimental::printf(
                        "[QK_ACC] kv_batch=%d j=%d QK_acc[0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                        kv_start / XMX_BATCH_KV, j, row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]);
                }
#endif
                int k = 0;
                // Vectorized max using float4 (process 4 elements at a time)
                sycl::float4 vmax = sycl::float4(-FLT_MAX);
                for (; k + 3 < kv_count; k += 4) {
                    sycl::float4 v = *reinterpret_cast<const sycl::float4*>(&row[k]);
                    vmax = sycl::fmax(vmax, v);
                }
                // Reduce vector to scalar
                batch_max = sycl::fmax(sycl::fmax(vmax.x(), vmax.y()), sycl::fmax(vmax.z(), vmax.w()));
                // Handle remainder
                for (; k < kv_count; ++k) {
                    batch_max = sycl::fmax(batch_max, row[k]);
                }
            }
            batch_max_shared[j] = batch_max;
        }
        sycl::group_barrier(item.get_group());

        // All threads now have access to all batch_max values via shared memory
        // Each thread updates its own KQ_max and VKQ accumulators
        #pragma unroll
        for (int j = 0; j < ncols; ++j) {
            if (ic0 + j >= ne01) continue;

            const float batch_max = batch_max_shared[j];
            const float new_max = sycl::fmax(KQ_max[j], batch_max);
            // CRITICAL: Do NOT apply FTZ threshold to the rescaling factor!
            // The rescaling factor exp(old_max - new_max) preserves the relative
            // contribution of previous batches. Zeroing it would lose all previous
            // accumulation, which is incorrect.
            // FTZ should ONLY be applied to softmax weights (exp(kq - max)), not rescaling.
            const float diff_val = KQ_max[j] - new_max;
            const float scale_old = sycl::exp(diff_val);  // Never explicitly zero
            KQ_max[j] = new_max;

            // Rescale previous VKQ accumulator
            #pragma unroll
            for (int i = 0; i < D_per_thread; ++i) {
                VKQ[j][i] *= scale_old;
            }
            KQ_sum[j] *= scale_old;

#if FATTN_XMX_DEBUG
            // Debug: Log the update for query j (only for last query in work-group)
            if (dbg_log && j == (ncols - 1)) {
                sycl::ext::oneapi::experimental::printf(
                    "  kv_batch=%d: batch_max=%.6f, KQ_max_old->new=%.6f->%.6f, scale_old=%.6f, KQ_sum=%.6f\n",
                    kv_start / XMX_BATCH_KV, batch_max, batch_max_shared[j] - (new_max - KQ_max[j]), KQ_max[j], scale_old, KQ_sum[j]);
            }
#endif
        }

        // Second pass: compute softmax weights and store to tile_S
        // Now all threads have the correct KQ_max values
        for (int idx = tid; idx < ncols * kv_count; idx += XMX_NTHREADS) {
            const int j = idx / kv_count;
            const int k = idx % kv_count;

            if (ic0 + j >= ne01) {
                tile_S[j * S_STRIDE + k] = sycl::half(0.0f);
                continue;
            }

            const float kq_val = QK_acc[j * XMX_BATCH_KV + k];
            // Use FTZ threshold to avoid numerical issues with very small softmax weights
            const float diff = kq_val - KQ_max[j];
            const float w = diff >= SOFTMAX_FTZ_THRESHOLD ? sycl::exp(diff) : 0.0f;
            tile_S[j * S_STRIDE + k] = sycl::half(w);
        }
        // Zero-pad S for positions beyond kv_count in each query row
        // This is critical for partial batches (kv_count < XMX_BATCH_KV)
        // Without this, XMX S@V reads uninitialized shared memory
        if (kv_count < XMX_BATCH_KV) {
            const int pad_start = kv_count;
            const int pad_count = XMX_BATCH_KV - kv_count;
            for (int idx = tid; idx < ncols * pad_count; idx += XMX_NTHREADS) {
                const int j = idx / pad_count;
                const int k = pad_start + (idx % pad_count);
                tile_S[j * S_STRIDE + k] = sycl::half(0.0f);
            }
        }
        // Zero-pad S stride padding
        for (int idx = tid; idx < ncols_padded * XMX_PAD; idx += XMX_NTHREADS) {
            const int j_idx = XMX_PAD > 0 ? idx / XMX_PAD : 0;
            const int p_idx = XMX_PAD > 0 ? idx % XMX_PAD : 0;
            if (j_idx < ncols_padded && XMX_BATCH_KV + p_idx < S_STRIDE) {
                tile_S[j_idx * S_STRIDE + XMX_BATCH_KV + p_idx] = sycl::half(0.0f);
            }
        }
        // Zero-pad S for padding rows (if ncols < ncols_padded)
        if (ncols < ncols_padded) {
            for (int idx = tid; idx < (ncols_padded - ncols) * S_STRIDE; idx += XMX_NTHREADS) {
                tile_S[ncols * S_STRIDE + idx] = sycl::half(0.0f);
            }
        }

        // CRITICAL BARRIER: Ensure all tile_S writes complete before any thread reads tile_S
        // Without this barrier, threads reading tile_S for KQ_sum accumulation may see
        // partially written data from other threads, causing non-deterministic results.
        // This was the root cause of GPT-OSS 20B (D=64) non-determinism.
        sycl::group_barrier(item.get_group());

        // Accumulate softmax sum (needed for final normalization)
        // First compute sum and store to shared memory
        // Use vectorized sum for better performance
        for (int j = tid; j < ncols; j += XMX_NTHREADS) {
            float sum = 0.0f;
            if (ic0 + j < ne01) {
                const sycl::half * row = &tile_S[j * S_STRIDE];
                int k = 0;
                // Vectorized sum using half8 converted to float (process 8 elements at a time)
                sycl::float4 vsum = sycl::float4(0.0f);
                for (; k + 7 < kv_count; k += 8) {
                    // Load 8 halfs, convert to 2 float4s, and accumulate
                    sycl::half4 h0 = *reinterpret_cast<const sycl::half4*>(&row[k]);
                    sycl::half4 h1 = *reinterpret_cast<const sycl::half4*>(&row[k + 4]);
                    vsum += sycl::float4(static_cast<float>(h0.x()), static_cast<float>(h0.y()),
                                         static_cast<float>(h0.z()), static_cast<float>(h0.w()));
                    vsum += sycl::float4(static_cast<float>(h1.x()), static_cast<float>(h1.y()),
                                         static_cast<float>(h1.z()), static_cast<float>(h1.w()));
                }
                // Reduce vector to scalar
                sum = vsum.x() + vsum.y() + vsum.z() + vsum.w();
                // Handle remainder
                for (; k < kv_count; ++k) {
                    sum += static_cast<float>(row[k]);
                }
            }
            batch_max_shared[j] = sum;  // Reuse for sum
        }
        sycl::group_barrier(item.get_group());

        // All threads update their KQ_sum
        #pragma unroll
        for (int j = 0; j < ncols; ++j) {
            if (ic0 + j >= ne01) continue;
            KQ_sum[j] += batch_max_shared[j];
        }

        // NOTE: Barrier removed - KQ_sum is per-thread private state, no sync needed

        // ---------------------------------------------------------------------
        // PHASE 4: XMX-based S @ V computation
        // ---------------------------------------------------------------------
#if FATTN_XMX_DEBUG
        if (head == 0 && tid == 0 && kv_start == 0) {
            sycl::ext::oneapi::experimental::printf("[PHASE4] Start, ic0=%d\n", ic0);
        }
#endif
        // S: [ncols_padded, BATCH_KV] @ V: [BATCH_KV, D] -> SV: [ncols_padded, D]
        // Zero the SV_acc buffer first
        for (int idx = tid; idx < ncols_padded * D; idx += XMX_NTHREADS) {
            SV_acc[idx] = 0.0f;
        }
        sycl::group_barrier(item.get_group());

        // Each sub-group handles different D tiles
        // D=64: D_TILES=4, D=128: D_TILES=8, D=256: D_TILES=16
        constexpr int D_TILES = D / XMX_TN;  // Number of output tiles in D dimension
        constexpr int K_TILES = XMX_BATCH_KV / XMX_TK;  // Reduction tiles (32/16 = 2)

        // D=64 PARALLELISM FIX:
        // For D=64, D_TILES=4 but we have 32 sub-groups → only 4 active (87.5% idle!)
        // Solution: Distribute work as (d_tile, k_tile) pairs across sub-groups.
        //
        // Work distribution:
        // - Total work items = D_TILES * K_TILES = 4 * 2 = 8 for D=64
        // - Each sub-group handles one (d_tile, k_tile) pair
        // - Multiple sub-groups may compute the same d_tile with different k_tiles
        // - Results are atomically accumulated (or we use deterministic assignment)
        //
        // For D=64:  8 work items, each sub-group 0-7 gets one, sub-groups 8-31 idle
        // For D=128: 16 work items, sub-groups 0-15 each get one, 16-31 idle
        // For D=256: 32 work items, all sub-groups active
        //
        // This doubles active sub-groups for D=64 (4→8) and D=128 (8→16)
        constexpr int TOTAL_WORK_ITEMS = D_TILES * K_TILES;
        constexpr bool NEED_K_REDUCTION = (K_TILES > 1) && (TOTAL_WORK_ITEMS <= XMX_N_SG);

        // Assign work: sg_id -> (d_tile, k_tile)
        const int work_id = sg_id;
        const int d_tile = work_id % D_TILES;
        const int k_tile = work_id / D_TILES;

        // Only process if this sub-group has assigned work
        if (work_id < TOTAL_WORK_ITEMS) {
            const int d_start = d_tile * XMX_TN;
            const int k_start = k_tile * XMX_TK;

            // XMX matrices for this tile
            sycl_xmx::joint_matrix<sycl::sub_group, sycl::half, sycl_xmx::use::a, XMX_TM, XMX_TK, sycl_xmx::layout::row_major> mat_S;
            sycl_xmx::joint_matrix<sycl::sub_group, sycl::half, sycl_xmx::use::b, XMX_TK, XMX_TN, sycl_xmx::layout::row_major> mat_V;
            sycl_xmx::joint_matrix<sycl::sub_group, float, sycl_xmx::use::accumulator, XMX_TM, XMX_TN> mat_SV;

            sycl_xmx::joint_matrix_fill(sg, mat_SV, 0.0f);

            // Load S tile: [ncols_padded, TK] from position [0, k_start]
            sycl_xmx::joint_matrix_load(sg, mat_S,
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                    &tile_S[k_start]),
                S_STRIDE);

            // Load V tile: [TK, TN] from position [k_start, d_start]
            sycl_xmx::joint_matrix_load(sg, mat_V,
                sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                    &tile_V[k_start * V_STRIDE + d_start]),
                V_STRIDE);

            // Compute: SV = S @ V
            sycl_xmx::joint_matrix_mad(sg, mat_SV, mat_S, mat_V, mat_SV);

            if constexpr (NEED_K_REDUCTION) {
                // Store partial result to temporary buffer for reduction
                // Layout: [K_TILES][ncols_padded][D], indexed by [k_tile][row][d_start]
                const int partial_offset = k_tile * ncols_padded * D;
                sycl_xmx::joint_matrix_store(sg, mat_SV,
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        &SV_acc[partial_offset + d_start]),
                    D, sycl_xmx::layout::row_major);
            } else {
                // No reduction needed - each sub-group handles full K reduction
                // This path is taken when D_TILES >= XMX_N_SG (D >= 512, unlikely)
                // or when there's only one K_TILE
                sycl_xmx::joint_matrix_store(sg, mat_SV,
                    sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(
                        &SV_acc[d_start]),
                    D, sycl_xmx::layout::row_major);
            }
        }

#if FATTN_XMX_DEBUG
        if (head == 0 && tid == 0 && kv_start == 0) {
            sycl::ext::oneapi::experimental::printf("[PHASE4] XMX done\n");
        }
#endif
        sycl::group_barrier(item.get_group());

        // ---------------------------------------------------------------------
        // PHASE 4.5: Reduce partial K results (when NEED_K_REDUCTION)
        // ---------------------------------------------------------------------
        if constexpr (NEED_K_REDUCTION) {
            // Each thread reduces its assigned elements across K_TILES
            // SV_acc layout: [K_TILES][ncols_padded][D]
            // We sum partial[k][j][d] for k=0..K_TILES-1 into partial[0][j][d]
            for (int idx = tid; idx < ncols_padded * D; idx += XMX_NTHREADS) {
                float sum = SV_acc[idx];  // k_tile=0
                #pragma unroll
                for (int k = 1; k < K_TILES; ++k) {
                    sum += SV_acc[k * ncols_padded * D + idx];
                }
                SV_acc[idx] = sum;
            }
            sycl::group_barrier(item.get_group());
        }

        // ---------------------------------------------------------------------
        // PHASE 5: Accumulate SV_acc into per-thread VKQ
        // ---------------------------------------------------------------------
#if FATTN_XMX_DEBUG
        if (head == 0 && tid == 0 && kv_start == 0) {
            sycl::ext::oneapi::experimental::printf(
                "[SV] ic0=%d SV_acc[0][0..3]=%.4f %.4f %.4f %.4f\n",
                ic0, SV_acc[0], SV_acc[1], SV_acc[2], SV_acc[3]);
        }
#endif
        #pragma unroll
        for (int j = 0; j < ncols; ++j) {
            if (ic0 + j >= ne01) continue;
            #pragma unroll
            for (int i = 0; i < D_per_thread; ++i) {
                const int d_idx = tid + i * XMX_NTHREADS;
                if (d_idx < D) {
                    VKQ[j][i] += SV_acc[j * D + d_idx];
                }
            }
        }

        // Swap buffers for next iteration
        buf_compute = buf_load;

        // NOTE: End-of-loop barrier removed - next iteration's Q@K^T barrier provides sync
    }

#if FATTN_XMX_DEBUG
    // Debug: Log final state for last query, head 0
    if (head == 0 && tid == 0 && (ic0 + ncols - 1) == (ne01 - 1)) {
        const int j = ncols - 1;  // Last query in work-group
        sycl::ext::oneapi::experimental::printf(
            "\n  FINAL: KQ_max=%.6f, KQ_sum=%.6f\n", KQ_max[j], KQ_sum[j]);
    }
#endif

    // =========================================================================
    // Apply attention sinks if present
    // =========================================================================
    if (sinks) {
        const float * sinks_f = reinterpret_cast<const float *>(sinks);
        const float sink = sinks_f[head];

        #pragma unroll
        for (int j = 0; j < ncols; ++j) {
            if (ic0 + j >= ne01) continue;

            const float KQ_max_new = sycl::fmax(sink, KQ_max[j]);
            const float KQ_max_scale = sycl::native::exp(KQ_max[j] - KQ_max_new);
            KQ_max[j] = KQ_max_new;

            const float sink_softmax = sycl::native::exp(sink - KQ_max[j]);
            KQ_sum[j] = KQ_sum[j] * KQ_max_scale + sink_softmax;

            #pragma unroll
            for (int i = 0; i < D_per_thread; ++i) {
                VKQ[j][i] *= KQ_max_scale;
            }
        }
    }

    // =========================================================================
    // Write output
    // =========================================================================
    #pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (ic0 + j >= ne01) continue;

        const float inv_sum = (KQ_sum[j] > 0.0f) ? (1.0f / KQ_sum[j]) : 0.0f;
        float * dst_row = dst + D * (head + ne02 * ((ic0 + j) + ne01 * sequence));

        #pragma unroll
        for (int i = 0; i < D_per_thread; ++i) {
            const int d_idx = tid + i * XMX_NTHREADS;
            if (d_idx < D) {
                float val = VKQ[j][i] * inv_sum;
                dst_row[d_idx] = sycl::isfinite(val) ? val : 0.0f;
            }
        }
    }
}

#endif // SYCL_XMX_AVAILABLE

// =============================================================================
// Launch function for XMX-based flash attention
// =============================================================================

// Kernel name class for VTune/profiler visibility
template <int D, int ncols, bool use_logit_softcap, typename Q_type>
class fattn_xmx_f16_kernel_name;

template <int D, int ncols, bool use_logit_softcap, typename Q_type>
void launch_fattn_xmx_f16(
    const fattn_params & params,
    dpct::queue_ptr stream) {

#if SYCL_XMX_AVAILABLE

    // Shared memory size calculation - WITH DOUBLE BUFFERING for K^T AND XMX S@V
    // XMX requires at least XMX_TM rows for Q tile, even if ncols < XMX_TM
    constexpr int ncols_padded = (ncols < XMX_TM) ? XMX_TM : ncols;

    // tile_Q:      ncols_padded * D * sizeof(half)
    // tile_KT[2]:  D * (XMX_BATCH_KV + PAD) * sizeof(half) * 2  <-- DOUBLE BUFFERED
    // tile_V:      XMX_BATCH_KV * (D + PAD) * sizeof(half)  <-- with stride padding
    // tile_S:      ncols_padded * (XMX_BATCH_KV + PAD) * sizeof(half)  <-- softmax weights
    // QK_acc:      ncols_padded * XMX_BATCH_KV * sizeof(float)
    // SV_acc:      K_TILES * ncols_padded * D * sizeof(float)  <-- S@V partial results for K reduction
    constexpr int KT_STRIDE = XMX_BATCH_KV + XMX_PAD;
    constexpr int KT_SIZE = D * KT_STRIDE;
    constexpr int V_STRIDE = D + XMX_PAD;
    constexpr int S_STRIDE = XMX_BATCH_KV + XMX_PAD;
    constexpr int D_TILES = D / XMX_TN;
    constexpr int K_TILES = XMX_BATCH_KV / XMX_TK;
    constexpr int TOTAL_WORK_ITEMS = D_TILES * K_TILES;
    // Need K_TILES copies of SV_acc only when doing K reduction (small D)
    constexpr int SV_ACC_COPIES = ((K_TILES > 1) && (TOTAL_WORK_ITEMS <= XMX_N_SG)) ? K_TILES : 1;
    constexpr size_t shared_half = ncols_padded * D +           // tile_Q
                                   KT_SIZE * 2 +                 // tile_KT[2]
                                   XMX_BATCH_KV * V_STRIDE +     // tile_V
                                   ncols_padded * S_STRIDE;      // tile_S
    constexpr size_t shared_float = ncols_padded * XMX_BATCH_KV +        // QK_acc
                                    SV_ACC_COPIES * ncols_padded * D;    // SV_acc (with K reduction copies)
    constexpr size_t shared_mem_size = shared_half + shared_float * 2;

    const int n_query_blocks = (params.ne01 + ncols - 1) / ncols;
    sycl::range<3> grid(params.ne02 * params.ne03, 1, n_query_blocks);
    sycl::range<3> block(1, 1, XMX_NTHREADS);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<sycl::half, 1> shared_acc(sycl::range<1>(shared_mem_size), cgh);

        const char * Q_ptr = params.Q;
        const char * K_ptr = params.K;
        const char * V_ptr = params.V;
        const char * mask_ptr = params.mask;
        const char * sinks_ptr = params.sinks;
        float * dst_ptr = params.dst;
        const float scale_val = params.scale;
        const float max_bias_val = params.max_bias;
        const float m0_val = params.m0;
        const float m1_val = params.m1;
        const uint32_t n_head_log2_val = params.n_head_log2;
        const float logit_softcap_val = params.logit_softcap;
        const int ne00 = params.ne00, ne01 = params.ne01, ne02 = params.ne02, ne03 = params.ne03;
        const int nb01 = params.nb01, nb02 = params.nb02, nb03 = params.nb03;
        const int ne10 = params.ne10, ne11 = params.ne11, ne12 = params.ne12, ne13 = params.ne13;
        const int nb11 = params.nb11, nb12 = params.nb12;
        const int64_t nb13 = params.nb13;
        const int nb21 = params.nb21, nb22 = params.nb22;
        const int64_t nb23 = params.nb23;
        const int ne30 = params.ne30, ne31 = params.ne31, ne32 = params.ne32, ne33 = params.ne33;
        const int nb31 = params.nb31, nb32 = params.nb32;
        const int64_t nb33 = params.nb33;

        // Multi-sequence batching parameters
        const int n_seqs = params.n_seqs;
        const int32_t * seq_q_offsets = params.seq_q_offsets;
        const int32_t * seq_kv_offsets = params.seq_kv_offsets;
        const int32_t * q_seq_ids = params.q_seq_ids;
        const int32_t * kv_seq_ids = params.kv_seq_ids;

        // PagedAttention parameters
        const bool use_paged_attn = params.use_paged_attn;
        const int32_t pa_block_size = params.block_size;
        const int32_t pa_max_blocks = params.max_blocks_per_seq;
        const int32_t * pa_block_table = params.block_table;
        const int32_t * pa_seq_lens = params.seq_lens;

        // Multi-token decode parameters
        const bool multi_token_decode = params.multi_token_decode;
        const int32_t * q_positions = params.q_positions;
        const int32_t kv_base_pos = params.kv_base_pos;

        // Capture kv_is_fp8 flag for runtime dispatch
        const bool kv_fp8 = params.kv_is_fp8;

        cgh.parallel_for<fattn_xmx_f16_kernel_name<D, ncols, use_logit_softcap, Q_type>>(
            sycl::nd_range<3>(grid * block, block),
            [=](sycl::nd_item<3> item) [[sycl::reqd_sub_group_size(XMX_SG)]] {
                sycl::half * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();
                // Runtime dispatch: select kernel based on KV cache type
                // Note: both kernels are instantiated, but only one path is taken at runtime
                if (kv_fp8) {
                    flash_attn_xmx_f16_kernel<D, ncols, use_logit_softcap, Q_type, true>(
                        Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, dst_ptr,
                        scale_val, max_bias_val, m0_val, m1_val, n_head_log2_val, logit_softcap_val,
                        ne00, ne01, ne02, ne03,
                        nb01, nb02, nb03,
                        ne10, ne11, ne12, ne13,
                        nb11, nb12, nb13,
                        nb21, nb22, nb23,
                        ne30, ne31, ne32, ne33,
                        nb31, nb32, nb33,
                        n_seqs, seq_q_offsets, seq_kv_offsets,
                        q_seq_ids, kv_seq_ids,
                        use_paged_attn, pa_block_size, pa_max_blocks, pa_block_table, pa_seq_lens,
                        multi_token_decode, q_positions, kv_base_pos,
                        item, shared);
                } else {
                    flash_attn_xmx_f16_kernel<D, ncols, use_logit_softcap, Q_type, false>(
                        Q_ptr, K_ptr, V_ptr, mask_ptr, sinks_ptr, dst_ptr,
                        scale_val, max_bias_val, m0_val, m1_val, n_head_log2_val, logit_softcap_val,
                        ne00, ne01, ne02, ne03,
                        nb01, nb02, nb03,
                        ne10, ne11, ne12, ne13,
                        nb11, nb12, nb13,
                        nb21, nb22, nb23,
                        ne30, ne31, ne32, ne33,
                        nb31, nb32, nb33,
                        n_seqs, seq_q_offsets, seq_kv_offsets,
                        q_seq_ids, kv_seq_ids,
                        use_paged_attn, pa_block_size, pa_max_blocks, pa_block_table, pa_seq_lens,
                        multi_token_decode, q_positions, kv_base_pos,
                        item, shared);
                }
            });
    });
#else
    GGML_UNUSED(params);
    GGML_UNUSED(stream);
    GGML_ASSERT(false && "SYCL XMX (joint_matrix) not available");
#endif
}

// Check if XMX F16 kernel is available
inline bool fattn_xmx_f16_available() {
#if SYCL_XMX_AVAILABLE
    return true;
#else
    return false;
#endif
}

#endif // GGML_SYCL_FATTN_XMX_F16_HPP
