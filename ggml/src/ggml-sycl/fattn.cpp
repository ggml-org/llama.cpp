//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "fattn.hpp"
#include "fattn-vec.hpp"
#include "fattn-mma.hpp"
#include "fattn-mma-f16.hpp"
#include "fattn-xmx-f16.hpp"
#include "fattn-debug.hpp"
#include "fattn-esimd-f16.hpp"
#include "fattn-v2-partition.hpp"
#include "kv-cache-quant.hpp"

#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>

// =============================================================================
// Paged Attention V2 Configuration
// =============================================================================
// V2 uses multi-partition algorithm for long sequences (>512 tokens)
// This enables O(n) memory complexity for O(n²) attention
//
// Enable with environment variable: GGML_SYCL_PAGED_V2=1
// Requires paged KV cache layout: K/V stored as [blocks, heads, block_size, D]

static bool g_sycl_paged_v2_enabled = false;
static bool g_sycl_paged_v2_initialized = false;

static void init_paged_v2_config() {
    if (g_sycl_paged_v2_initialized) return;
    g_sycl_paged_v2_initialized = true;

    const char* env = std::getenv("GGML_SYCL_PAGED_V2");
    if (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) {
        g_sycl_paged_v2_enabled = true;
        fprintf(stderr, "[SYCL] Paged Attention V2 enabled for long sequences\n");
    }
}

// =============================================================================
// FP8 KV Cache Configuration
// =============================================================================
// FP8 quantization provides 2x memory reduction for KV cache
//
// Enable with environment variable: GGML_SYCL_KV_FP8=1
// Requires FP8-compatible KV cache allocation

static bool g_sycl_kv_fp8_enabled = false;
static bool g_sycl_kv_fp8_initialized = false;

static void init_kv_fp8_config() {
    if (g_sycl_kv_fp8_initialized) return;
    g_sycl_kv_fp8_initialized = true;

    const char* env = std::getenv("GGML_SYCL_KV_FP8");
    if (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) {
        g_sycl_kv_fp8_enabled = true;
        fprintf(stderr, "[SYCL] FP8 KV cache quantization enabled (2x memory savings)\n");
    }
}

// =============================================================================
// ESIMD Flash Attention Configuration
// =============================================================================
// ESIMD uses explicit SIMD operations for potentially better decode performance
// compared to XMX joint_matrix operations.
//
// Enabled by default. Disable with: GGML_SYCL_FA_ESIMD=0

static bool g_sycl_fa_esimd_enabled = false;
static bool g_sycl_fa_esimd_initialized = false;

static void init_fa_esimd_config() {
    if (g_sycl_fa_esimd_initialized) return;
    g_sycl_fa_esimd_initialized = true;

    // ESIMD FA is enabled by default when available (7-8% decode speedup)
    // Disable with GGML_SYCL_FA_ESIMD=0
    const char* env = std::getenv("GGML_SYCL_FA_ESIMD");
    bool enable_esimd = true;  // Default to enabled
    if (env && (strcmp(env, "0") == 0 || strcmp(env, "false") == 0)) {
        enable_esimd = false;
    }

    if (enable_esimd && fattn_esimd_f16_available()) {
        g_sycl_fa_esimd_enabled = true;
        GGML_SYCL_DEBUG("[SYCL] ESIMD Flash Attention enabled for decode\n");
    }
}

// ============================================================================
// Multi-sequence boundary detection from mask
// ============================================================================
// In continuous batching with unified KV cache, each query token belongs to a
// sequence, and can only attend to KV positions from its own sequence.
// The mask encodes this: 0 = allow attention, -INF = block attention.
//
// This function scans the F32 mask (before F16 cast) to detect sequence
// boundaries, which allows the kernel to skip cross-sequence computation.
//
// Returns true if sequences were detected, with seq_q_offsets and seq_kv_offsets
// populated. Returns false if single-sequence or detection failed.
//
static bool detect_sequence_boundaries_from_mask(
    const ggml_tensor * mask_f16,  // The F16 mask tensor (may be a cast of F32)
    int n_queries,                  // Number of query tokens
    int n_kv,                       // Number of KV positions
    std::vector<int32_t> & seq_q_offsets,   // Output: [n_seqs + 1]
    std::vector<int32_t> & seq_kv_offsets)  // Output: [n_seqs + 1]
{
    if (!mask_f16) {
        return false;
    }

    // Try to get the original F32 mask from the cast/copy operation's source
    // ggml_cast uses GGML_OP_CPY with src[0] = source tensor, src[1] = result
    const ggml_tensor * mask_f32 = nullptr;

    if (mask_f16->op == GGML_OP_CPY && mask_f16->src[0] &&
        mask_f16->src[0]->type == GGML_TYPE_F32) {
        mask_f32 = mask_f16->src[0];
    }

    // Need host-accessible F32 mask data
    // NOTE: In SYCL backend, even input tensors are typically in GPU buffers
    // by the time we reach here. Future work: scan mask in kernel using
    // a dedicated work-group, or pass sequence info through ggml graph.
    if (!mask_f32 || !mask_f32->data) {
        return false;
    }
    if (!ggml_backend_buffer_is_host(mask_f32->buffer)) {
        return false;
    }

    const float * mask_data = (const float *) mask_f32->data;
    const int64_t mask_stride = mask_f32->nb[1] / sizeof(float);  // Stride between query rows

    // Scan the mask to find sequence boundaries
    // For each query, find the first and last valid (non-INF) KV position
    // Sequence boundaries occur where there's a discontinuity in valid KV ranges

    seq_q_offsets.clear();
    seq_kv_offsets.clear();
    seq_q_offsets.push_back(0);

    int prev_kv_start = -1;
    int prev_kv_end = -1;

    for (int q = 0; q < n_queries; ++q) {
        const float * row = mask_data + q * mask_stride;

        // Find first valid KV position (value > -1e30)
        int kv_start = -1;
        for (int k = 0; k < n_kv; ++k) {
            if (row[k] > -1e30f) {
                kv_start = k;
                break;
            }
        }

        if (kv_start < 0) {
            // This query has no valid KV positions - shouldn't happen
            // Treat as continuing previous sequence
            continue;
        }

        // Find last valid KV position
        int kv_end = kv_start;
        for (int k = n_kv - 1; k >= kv_start; --k) {
            if (row[k] > -1e30f) {
                kv_end = k + 1;  // End is exclusive
                break;
            }
        }

        // Check if this is a new sequence
        // New sequence if: KV range doesn't overlap with previous, OR
        // KV start is before previous KV start (new sequence started earlier)
        bool new_seq = false;
        if (prev_kv_start < 0) {
            // First query - start first sequence
            new_seq = true;
        } else if (kv_start < prev_kv_start) {
            // KV range starts earlier than previous - definitely new sequence
            new_seq = true;
        } else if (kv_end <= prev_kv_start || kv_start >= prev_kv_end) {
            // Non-overlapping ranges - new sequence
            new_seq = true;
        }

        if (new_seq && q > 0) {
            // End previous sequence, start new one
            seq_q_offsets.push_back(q);
            seq_kv_offsets.push_back(prev_kv_end);
        }

        if (new_seq) {
            // Record this sequence's KV start
            if (seq_kv_offsets.empty()) {
                seq_kv_offsets.push_back(kv_start);
            }
        }

        prev_kv_start = (prev_kv_start < 0) ? kv_start : std::min(prev_kv_start, kv_start);
        prev_kv_end = std::max(prev_kv_end, kv_end);
    }

    // Close the last sequence
    seq_q_offsets.push_back(n_queries);
    if (prev_kv_end > 0) {
        seq_kv_offsets.push_back(prev_kv_end);
    } else {
        seq_kv_offsets.push_back(n_kv);
    }

    // Only enable multi-sequence optimization if we detected multiple sequences
    int n_seqs = (int)seq_q_offsets.size() - 1;
    if (n_seqs <= 1) {
        seq_q_offsets.clear();
        seq_kv_offsets.clear();
        return false;
    }

    return true;
}

// ============================================================================
// Multi-sequence boundary detection from sequence ID tensors
// ============================================================================
// Computes sequence boundaries from q_seq_ids and kv_seq_ids arrays.
// This enables the kernel to skip cross-sequence KV computation entirely
// (not just mask it), achieving true parallel speedup.
//
// Returns the number of sequences detected (0 if single sequence or error)
//
static int compute_sequence_boundaries_from_ids(
    const int32_t * q_seq_ids,     // [n_queries] Sequence ID for each query
    int n_queries,
    const int32_t * kv_seq_ids,    // [n_kv] Sequence ID for each KV position
    int n_kv,
    std::vector<int32_t> & seq_q_offsets,   // Output: [n_seqs + 1]
    std::vector<int32_t> & seq_kv_offsets)  // Output: [n_seqs + 1]
{
    if (!q_seq_ids || !kv_seq_ids || n_queries <= 0 || n_kv <= 0) {
        return 0;
    }

    seq_q_offsets.clear();
    seq_kv_offsets.clear();

    // Build a map of sequence_id -> {q_start, q_end, kv_start, kv_end}
    // Assumption: tokens are grouped by sequence (all tokens of seq 0, then seq 1, etc.)
    struct SeqRange {
        int q_start = -1, q_end = -1;
        int kv_start = -1, kv_end = -1;
    };
    std::vector<SeqRange> seq_ranges;

    // Scan query sequence IDs to find sequence boundaries
    int prev_seq = -1;
    for (int q = 0; q < n_queries; ++q) {
        int seq = q_seq_ids[q];
        if (seq != prev_seq) {
            // New sequence starts
            if (seq >= (int)seq_ranges.size()) {
                seq_ranges.resize(seq + 1);
            }
            if (seq_ranges[seq].q_start < 0) {
                seq_ranges[seq].q_start = q;
            }
            prev_seq = seq;
        }
        if (seq >= 0 && seq < (int)seq_ranges.size()) {
            seq_ranges[seq].q_end = q + 1;
        }
    }

    // Scan KV sequence IDs to find KV boundaries
    prev_seq = -1;
    for (int k = 0; k < n_kv; ++k) {
        int seq = kv_seq_ids[k];
        if (seq >= 0 && seq < (int)seq_ranges.size()) {
            if (seq_ranges[seq].kv_start < 0) {
                seq_ranges[seq].kv_start = k;
            }
            seq_ranges[seq].kv_end = k + 1;
        }
    }

    // Build ordered offset arrays for sequences that have both Q and KV
    int n_valid_seqs = 0;
    for (size_t s = 0; s < seq_ranges.size(); ++s) {
        if (seq_ranges[s].q_start >= 0 && seq_ranges[s].kv_start >= 0) {
            if (n_valid_seqs == 0) {
                // First sequence
                seq_q_offsets.push_back(seq_ranges[s].q_start);
                seq_kv_offsets.push_back(seq_ranges[s].kv_start);
            }
            n_valid_seqs++;
            seq_q_offsets.push_back(seq_ranges[s].q_end);
            seq_kv_offsets.push_back(seq_ranges[s].kv_end);
        }
    }

    // Only enable multi-sequence optimization if we detected multiple sequences
    if (n_valid_seqs <= 1) {
        seq_q_offsets.clear();
        seq_kv_offsets.clear();
        return 0;
    }

    return n_valid_seqs;
}

// Kernel selection is now done at runtime based on GPU capabilities.
// XMX kernel (3) is used on Intel GPUs with matrix extension support (Arc, etc.)
// MMA F16 kernel (2) is used on other SYCL devices.
// Kernel IDs:
// 0 = VEC kernel (simpler, one K/V position at a time)
// 1 = MMA kernel (tiled scalar, processes BATCH_KV positions at a time)
// 2 = MMA F16 kernel (scalar with SG_SIZE=16, named MMA but not using joint_matrix)
// 3 = XMX F16 kernel (using joint_matrix for Q@K^T acceleration)

// Check if flash attention is supported for the given operation
bool ggml_sycl_flash_attn_ext_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    // Check Q type - must be F32 or F16
    if (Q->type != GGML_TYPE_F32 && Q->type != GGML_TYPE_F16) {
        return false;
    }

    // Check K/V types - F16 or FP8 E4M3 (for memory savings)
    const bool kv_is_fp8 = (K->type == GGML_TYPE_F8_E4M3 && V->type == GGML_TYPE_F8_E4M3);
    const bool kv_is_f16 = (K->type == GGML_TYPE_F16 && V->type == GGML_TYPE_F16);
    if (!kv_is_f16 && !kv_is_fp8) {
        return false;
    }

    // Check destination type
    if (dst->type != GGML_TYPE_F32) {
        return false;
    }

    // Check mask type if present
    if (mask && mask->type != GGML_TYPE_F16) {
        return false;
    }

    // Check head dimension - must be a supported size
    const int D = Q->ne[0];
    if (!fattn_vec_supports_head_dim(D)) {
        return false;
    }

    // MMA kernel handles high head counts (>32), vec kernel handles <=32
    // Both are now supported

    // Check that tensors are contiguous
    if (Q->nb[0] != ggml_type_size(Q->type)) {
        return false;
    }
    // K/V stride depends on type: 2 bytes for F16, 1 byte for FP8
    const size_t kv_elem_size = kv_is_fp8 ? sizeof(uint8_t) : sizeof(sycl::half);
    if (K->nb[0] != kv_elem_size) {
        return false;
    }
    if (V->nb[0] != kv_elem_size) {
        return false;
    }
    if (mask && mask->nb[0] != sizeof(sycl::half)) {
        return false;
    }

    return true;
}

// Dispatcher that selects appropriate kernel based on head dimension and GPU capabilities
template <int D, typename Q_type>
static void ggml_sycl_flash_attn_ext_dispatch_ncols(
    ggml_backend_sycl_context & ctx,
    const fattn_params & params) {

    dpct::queue_ptr stream = ctx.stream();

    // Select ncols based on batch size (ne01 = number of queries)
    const int ne01 = params.ne01;
    float logit_softcap = params.logit_softcap;

    // Runtime kernel selection based on GPU capabilities
    // Check if the device has XMX (Intel matrix extension) support
    sycl::device dev = stream->get_device();
    const bool use_xmx = gpu_has_xmx(dev);

    // ESIMD kernel for decode (single query per head)
    // ESIMD uses explicit SIMD operations which can be faster than XMX for decode
    if (g_sycl_fa_esimd_enabled && ne01 <= 1 && fattn_esimd_f16_available()) {
        fattn_esimd_f16<D, Q_type>(params, *stream);
        return;
    }

    // Helper macro to dispatch based on softcap
    #define DISPATCH_NCOLS(NCOLS, LAUNCHER) \
        if (logit_softcap == 0.0f) { \
            LAUNCHER<D, NCOLS, false, Q_type>(params, stream); \
        } else { \
            LAUNCHER<D, NCOLS, true, Q_type>(params, stream); \
        }

    // Dispatch to appropriate kernel based on GPU capabilities
    if (use_xmx) {
        // XMX kernel - uses Intel joint_matrix for Q@K^T and S@V acceleration
        // Note: Even for single query, XMX kernel is faster than vector kernel
        // because XMX hardware throughput exceeds scalar operations despite
        // 7/8 of tiles being "wasted" in the 8x8 joint_matrix operations
        if (ne01 <= 1) {
            DISPATCH_NCOLS(1, launch_fattn_xmx_f16);
        } else if (ne01 <= 2) {
            DISPATCH_NCOLS(2, launch_fattn_xmx_f16);
        } else if (ne01 <= 4) {
            DISPATCH_NCOLS(4, launch_fattn_xmx_f16);
        } else {
            DISPATCH_NCOLS(8, launch_fattn_xmx_f16);
        }
    } else {
        // MMA F16 kernel - scalar fallback for non-XMX GPUs
        if (ne01 <= 1) {
            DISPATCH_NCOLS(1, launch_fattn_mma_f16);
        } else if (ne01 <= 2) {
            DISPATCH_NCOLS(2, launch_fattn_mma_f16);
        } else if (ne01 <= 4) {
            DISPATCH_NCOLS(4, launch_fattn_mma_f16);
        } else {
            DISPATCH_NCOLS(8, launch_fattn_mma_f16);
        }
    }

    #undef DISPATCH_NCOLS
}

// Main flash attention entry point
void ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    // Initialize configuration on first call
    init_paged_v2_config();
    init_kv_fp8_config();
    init_fa_esimd_config();

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];  // Attention sinks tensor (may be null)
    const ggml_tensor * q_seq_ids  = dst->src[5];  // Sequence IDs for query tokens (may be null)
    const ggml_tensor * kv_seq_ids = dst->src[6];  // Sequence IDs for KV positions (may be null)

    GGML_ASSERT(Q->type == GGML_TYPE_F32 || Q->type == GGML_TYPE_F16);
    GGML_ASSERT(K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_F8_E4M3);  // FP16 or FP8 KV cache
    GGML_ASSERT(V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_F8_E4M3);  // FP16 or FP8 KV cache
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // Extract scale, max_bias, and logit_softcap from op_params
    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    // Read use_paged_layout from op_params[4] (set by ggml_flash_attn_ext_set_paged_layout)
    // op_params layout: [0-2]=float scale/max_bias/logit_softcap, [3]=prec, [4]=use_paged_layout
    const int32_t use_paged_layout_i32 = ((const int32_t *) dst->op_params)[4];
    const bool use_paged_layout = (use_paged_layout_i32 != 0);

    // If using logit_softcap, adjust scale
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    // Calculate ALiBi parameters
    const uint32_t n_head = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // Build the params structure
    fattn_params params;

    // Use device-specific pointers for TP mode (KV cache is allocated per-device)
    const int device = ctx.device;
    params.Q = (const char *) ggml_sycl_get_data_ptr(Q, device);
    params.K = (const char *) ggml_sycl_get_data_ptr(K, device);
    params.V = (const char *) ggml_sycl_get_data_ptr(V, device);
    params.mask = mask ? (const char *) ggml_sycl_get_data_ptr(mask, device) : nullptr;
    params.sinks = sinks ? (const char *) ggml_sycl_get_data_ptr(sinks, device) : nullptr;
    params.dst = (float *) ggml_sycl_get_data_ptr(dst, device);

    params.scale = scale;
    params.max_bias = max_bias;
    params.m0 = m0;
    params.m1 = m1;
    params.n_head_log2 = n_head_log2;
    params.logit_softcap = logit_softcap;

    // Q dimensions: [batch, n_heads, n_queries, head_dim]
    params.ne00 = Q->ne[0];  // head_dim
    params.ne01 = Q->ne[1];  // n_queries
    params.ne02 = Q->ne[2];  // n_heads
    params.ne03 = Q->ne[3];  // batch

    params.nb01 = Q->nb[1];
    params.nb02 = Q->nb[2];
    params.nb03 = Q->nb[3];

    // K dimensions: [batch, n_kv_heads, n_kv, head_dim]
    params.ne10 = K->ne[0];  // head_dim
    params.ne11 = K->ne[1];  // n_kv (sequence length)
    params.ne12 = K->ne[2];  // n_kv_heads
    params.ne13 = K->ne[3];  // batch

    params.nb11 = K->nb[1];
    params.nb12 = K->nb[2];
    params.nb13 = K->nb[3];

    // V strides
    params.nb21 = V->nb[1];
    params.nb22 = V->nb[2];
    params.nb23 = V->nb[3];

    // Mask dimensions and strides (if present)
    // mask layout: [ne3, ne2, ne1, ne0] = [batch, heads, n_tokens_padded, n_kv]
    if (mask) {
        params.ne30 = mask->ne[0];  // n_kv
        params.ne31 = mask->ne[1];  // n_tokens_padded
        params.ne32 = mask->ne[2];  // heads
        params.ne33 = mask->ne[3];  // batch
        params.nb31 = mask->nb[1];
        params.nb32 = mask->nb[2];
        params.nb33 = mask->nb[3];

        // Assertion: ne11 (K seq len) should equal ne30 (mask's first dim)
        GGML_ASSERT(params.ne11 == params.ne30 && "K sequence length must match mask dimension");
    } else {
        params.ne30 = 0;
        params.ne31 = 0;
        params.ne32 = 0;
        params.ne33 = 0;
        params.nb31 = 0;
        params.nb32 = 0;
        params.nb33 = 0;
    }

    // Multi-sequence batching support
    // Use sequence ID tensors from src[5] and src[6] to enable cross-sequence skipping
    // This optimization reduces wasted computation when queries only need their own sequence's KV
    dpct::queue_ptr stream = ctx.stream();

    // Initialize legacy offset-based multi-seq to disabled
    params.n_seqs = 0;
    params.seq_q_offsets = nullptr;
    params.seq_kv_offsets = nullptr;

    // Thread-local storage for sequence boundary offsets
    // These persist across calls within the same thread
    thread_local std::vector<int32_t> tl_seq_q_offsets;
    thread_local std::vector<int32_t> tl_seq_kv_offsets;

    // Thread-local device buffers for seq_ids (reused across calls)
    // We need to manage these manually since host tensor->data can't be accessed from GPU
    thread_local int32_t * tl_q_seq_ids_dev = nullptr;
    thread_local int32_t * tl_kv_seq_ids_dev = nullptr;
    thread_local size_t tl_q_seq_ids_size = 0;
    thread_local size_t tl_kv_seq_ids_size = 0;
    thread_local sycl::queue * tl_alloc_queue = nullptr;

    // Thread-local device buffers for sequence boundary offsets
    thread_local int32_t * tl_seq_q_offsets_dev = nullptr;
    thread_local int32_t * tl_seq_kv_offsets_dev = nullptr;
    thread_local size_t tl_seq_offsets_capacity = 0;

    // Use sequence ID tensors if provided
    if (q_seq_ids && kv_seq_ids &&
        q_seq_ids->type == GGML_TYPE_I32 && kv_seq_ids->type == GGML_TYPE_I32 &&
        q_seq_ids->data && kv_seq_ids->data) {

        const size_t q_size = q_seq_ids->ne[0] * sizeof(int32_t);
        const size_t kv_size = kv_seq_ids->ne[0] * sizeof(int32_t);

        // Check if tensors are on host buffers (need to copy to device)
        bool q_on_host = q_seq_ids->buffer ? ggml_backend_buffer_is_host(q_seq_ids->buffer) : true;
        bool kv_on_host = kv_seq_ids->buffer ? ggml_backend_buffer_is_host(kv_seq_ids->buffer) : true;

        // Get the host pointers from thread-local cache (set by llama-kv-cache.cpp)
        // These are USM host pointers that are accessible from both CPU and GPU
        size_t cached_q_count = 0, cached_kv_count = 0;
        const int32_t * cached_q_seq_ids = ggml_sycl_get_seq_ids_host_q(&cached_q_count);
        const int32_t * cached_kv_seq_ids = ggml_sycl_get_seq_ids_host_kv(&cached_kv_count);

        // If tensors are on device (not host), use the cached host pointers instead
        // The scheduler creates device tensors but doesn't copy INPUT data, so tensor->data is invalid
        // However, the llama layer has set the USM host pointers via ggml_backend_sycl_set_seq_ids_host()
        const int32_t * host_q_ptr = q_on_host ? (const int32_t *)q_seq_ids->data : cached_q_seq_ids;
        const int32_t * host_kv_ptr = kv_on_host ? (const int32_t *)kv_seq_ids->data : cached_kv_seq_ids;

        if (!host_q_ptr || !host_kv_ptr) {
            // No valid host pointers available - skip optimization
            static bool warned_no_cache = false;
            if (!warned_no_cache) {
                fprintf(stderr, "[SEQ_IDS] WARNING: Device tensors detected but no cached host pointers\n");
                fprintf(stderr, "[SEQ_IDS]   Falling back to mask-based sequence detection\n");
                warned_no_cache = true;
            }
            // params.q_seq_ids and params.kv_seq_ids remain nullptr (default)
        } else {
            // We have valid host pointers (either from tensor or from cache)
            // Need to copy from host to device for kernel access
            // Reallocate device buffers if size changed or first time
            if (tl_alloc_queue != stream || tl_q_seq_ids_size < q_size) {
                if (tl_q_seq_ids_dev && tl_alloc_queue) {
                    sycl::free(tl_q_seq_ids_dev, *tl_alloc_queue);
                }
                tl_q_seq_ids_dev = sycl::malloc_device<int32_t>(q_seq_ids->ne[0], *stream);
                tl_q_seq_ids_size = q_size;
            }
            if (tl_alloc_queue != stream || tl_kv_seq_ids_size < kv_size) {
                if (tl_kv_seq_ids_dev && tl_alloc_queue) {
                    sycl::free(tl_kv_seq_ids_dev, *tl_alloc_queue);
                }
                tl_kv_seq_ids_dev = sycl::malloc_device<int32_t>(kv_seq_ids->ne[0], *stream);
                tl_kv_seq_ids_size = kv_size;
            }
            tl_alloc_queue = stream;

            // Copy from host (USM) to device using the correct host pointers
            stream->memcpy(tl_q_seq_ids_dev, host_q_ptr, q_size);
            stream->memcpy(tl_kv_seq_ids_dev, host_kv_ptr, kv_size);
            stream->wait(); // Ensure copy completes before kernel launch

            params.q_seq_ids  = tl_q_seq_ids_dev;
            params.kv_seq_ids = tl_kv_seq_ids_dev;

            // Compute sequence boundaries from the seq_ids arrays
            // This enables the kernel to skip cross-sequence KV computation entirely
            // Note: We use the HOST pointers for boundary computation (which happens on CPU)
            int n_queries = static_cast<int>(q_seq_ids->ne[0]);
            int n_kv = static_cast<int>(kv_seq_ids->ne[0]);

            int n_seqs = compute_sequence_boundaries_from_ids(
                host_q_ptr, n_queries,
                host_kv_ptr, n_kv,
                tl_seq_q_offsets, tl_seq_kv_offsets);

            // Copy sequence boundary offsets to device memory for kernel access
            // This enables the kernel to skip entire KV blocks for non-matching sequences
            if (n_seqs > 1) {
                const size_t offsets_size = (n_seqs + 1) * sizeof(int32_t);

                // Reallocate device buffers if capacity is insufficient
                if (tl_seq_offsets_capacity < offsets_size) {
                    if (tl_seq_q_offsets_dev && tl_alloc_queue) {
                        sycl::free(tl_seq_q_offsets_dev, *tl_alloc_queue);
                    }
                    if (tl_seq_kv_offsets_dev && tl_alloc_queue) {
                        sycl::free(tl_seq_kv_offsets_dev, *tl_alloc_queue);
                    }
                    tl_seq_q_offsets_dev = sycl::malloc_device<int32_t>(n_seqs + 1, *stream);
                    tl_seq_kv_offsets_dev = sycl::malloc_device<int32_t>(n_seqs + 1, *stream);
                    tl_seq_offsets_capacity = offsets_size;
                }

                // Copy offsets to device
                stream->memcpy(tl_seq_q_offsets_dev, tl_seq_q_offsets.data(), offsets_size);
                stream->memcpy(tl_seq_kv_offsets_dev, tl_seq_kv_offsets.data(), offsets_size);
                stream->wait();  // Ensure copy completes before kernel launch

                // Set params for kernel to use sequence boundary optimization
                params.n_seqs = n_seqs;
                params.seq_q_offsets = tl_seq_q_offsets_dev;
                params.seq_kv_offsets = tl_seq_kv_offsets_dev;
            }
        }
    } else {
        params.q_seq_ids = nullptr;
        params.kv_seq_ids = nullptr;
    }

    // PagedAttention support
    // Block table (src[7]) and seq_lens (src[8]) are set via ggml_flash_attn_ext_set_paged()
    const ggml_tensor * block_table = dst->src[7];
    const ggml_tensor * seq_lens_tensor = dst->src[8];

    if (block_table && seq_lens_tensor) {
        // Enable PagedAttention: K/V are stored in blocks, accessed via block_table
        params.use_paged_attn = true;
        params.block_size = 16;  // Fixed block size (matches vLLM and XMX tile size)
        // block_table shape: [max_blocks, n_seqs] where ne[0]=max_blocks (columns), ne[1]=n_seqs (rows)
        // Kernel access: block_table[seq * max_blocks + block] requires max_blocks = ne[0]
        params.max_blocks_per_seq = static_cast<int32_t>(block_table->ne[0]);
        params.block_table = (const int32_t *) ggml_sycl_get_data_ptr(block_table, device);
        params.seq_lens = (const int32_t *) ggml_sycl_get_data_ptr(seq_lens_tensor, device);

        #if 0  // Debug output disabled
        // Print only once to avoid flooding output
        static bool paged_attn_info_shown = false;
        if (!paged_attn_info_shown) {
            fprintf(stderr, "[SYCL] PagedAttention enabled: block_size=%d, max_blocks=%d\n",
                    params.block_size, params.max_blocks_per_seq);
            paged_attn_info_shown = true;
        }
        #endif
    } else {
        // Standard contiguous K/V mode
        params.use_paged_attn = false;
        params.block_size = 16;
        params.max_blocks_per_seq = 0;
        params.block_table = nullptr;
        params.seq_lens = nullptr;
    }

    // Set paged layout flag (read from op_params[4], set via ggml_flash_attn_ext_set_paged_layout)
    params.use_paged_layout = use_paged_layout;

    // Set FP8 KV cache flag - enables on-the-fly dequantization in flash attention kernel
    params.kv_is_fp8 = (K->type == GGML_TYPE_F8_E4M3 && V->type == GGML_TYPE_F8_E4M3);

    // ==========================================================================
    // Paged Attention V2 Dispatch (for long sequences with paged KV layout)
    // ==========================================================================
    // V2 uses multi-partition algorithm when:
    // 1. Paged attention is enabled with block tables
    // 2. GGML_SYCL_PAGED_V2=1 environment variable is set
    // 3. Max sequence length exceeds V2_PARTITION_SIZE (512)
    //
    // NOTE: V2 requires K/V to be stored in paged format:
    //   K/V: [num_blocks, num_kv_heads, block_size, head_dim]
    // This is not yet supported in the current llama.cpp KV cache layout.
    // V2 dispatch is prepared for future paged KV cache implementation.

    const int D = Q->ne[0];
    const int max_context_len = params.ne11;  // n_kv = sequence length
    bool use_v2_dispatch = false;

    // Check if V2 dispatch should be used:
    // V2 requires K/V in vLLM paged format [num_blocks, num_kv_heads, block_size, D]
    // Currently, use_paged_layout flag doesn't change the actual tensor layout (still 3D)
    // So we only enable V2 when explicitly requested via GGML_SYCL_PAGED_V2=1 env var
    //
    // NOTE: True 4D paged layout for V2 is not yet implemented.
    // The use_paged_layout flag currently only affects block table creation,
    // not the actual KV tensor memory layout.
    if (g_sycl_paged_v2_enabled && params.use_paged_attn &&
        should_use_paged_attention_v2(max_context_len)) {
        // V2 explicitly requested for long sequences
        use_v2_dispatch = true;
    } else if (g_sycl_paged_v2_enabled && !params.use_paged_attn) {
        // V2 requested but paged attention not active - show warning once
        static bool v2_warning_shown = false;
        if (!v2_warning_shown) {
            fprintf(stderr, "[SYCL] Paged Attention V2 requested but paged_attn not enabled\n");
            fprintf(stderr, "[SYCL]   Use --paged-attn or --paged-layout flag\n");
            fprintf(stderr, "[SYCL]   Using standard flash attention kernel instead\n");
            v2_warning_shown = true;
        }
    }

    // V2 dispatch: partition-based attention for long sequences
    // NOTE: V2 is designed for vLLM-style continuous batching where each "seq" in the batch
    // is a separate request with its own KV cache. In llama.cpp's single-sequence mode,
    // all query tokens share the same sequence's KV cache, so we need to check compatibility.

    // Get actual number of sequences from seq_lens tensor (already declared above)
    const int actual_n_seqs = seq_lens_tensor ? (int)seq_lens_tensor->ne[0] : 1;
    const int num_query_tokens = params.ne01;  // Number of query tokens in batch

    // V2 only works correctly when:
    // 1. We have continuous batching with multiple sequences (actual_n_seqs == num_query_tokens), OR
    // 2. Single sequence mode with one query token per call (num_query_tokens == 1)
    // For prefill (single sequence with many query tokens), fall back to standard kernel
    const bool v2_compatible = (actual_n_seqs == num_query_tokens) || (num_query_tokens == 1);

    if (use_v2_dispatch && !v2_compatible) {
        // Prefill mode: single sequence with multiple query tokens - V2 not designed for this
        // Fall back to standard kernel
        static bool v2_fallback_warned = false;
        if (!v2_fallback_warned) {
            fprintf(stderr, "[SYCL] V2 skipped: prefill mode (n_seqs=%d, n_query=%d) not supported\n",
                    actual_n_seqs, num_query_tokens);
            fprintf(stderr, "[SYCL]   V2 requires continuous batching (n_seqs==n_query) or single-token decode\n");
            v2_fallback_warned = true;
        }
        use_v2_dispatch = false;
    }

    if (use_v2_dispatch) {
        const int num_seqs = actual_n_seqs;  // Use actual sequence count, not query token count
        const int num_heads = Q->ne[2];    // Number of query heads
        const int num_kv_heads = K->ne[2]; // Number of KV heads
        const int block_size = params.block_size;

        // Get temporary buffer sizes
        const size_t temp_size = paged_attention_v2_temp_size(
            num_seqs, num_heads, max_context_len, D);

        // Allocate temporary buffers from device memory pool
        // These are: exp_sums, max_logits, tmp_out
        void * temp_buf = sycl::malloc_device(temp_size, *ctx.stream());
        GGML_ASSERT(temp_buf != nullptr);

        const int max_num_partitions = (max_context_len + V2_PARTITION_SIZE - 1) / V2_PARTITION_SIZE;
        const size_t exp_sums_size = num_seqs * num_heads * max_num_partitions * sizeof(float);

        float * exp_sums = (float *)temp_buf;
        float * max_logits = exp_sums + num_seqs * num_heads * max_num_partitions;
        float * tmp_out = max_logits + num_seqs * num_heads * max_num_partitions;

        // Output buffer (float) - dst is the output tensor
        float * out = (float *)dst->data;

        GGML_SYCL_DEBUG("[SYCL] V2 dispatch: num_seqs=%d num_heads=%d num_kv_heads=%d D=%d "
                        "max_context=%d partitions=%d block_size=%d\n",
                        num_seqs, num_heads, num_kv_heads, D,
                        max_context_len, max_num_partitions, block_size);

        // Dispatch based on head dimension
        switch (D) {
            case 64:
                launch_paged_attention_v2<64>(
                    out, exp_sums, max_logits, tmp_out,
                    (const sycl::half *)Q->data,
                    (const sycl::half *)K->data,
                    (const sycl::half *)V->data,
                    params.scale,
                    params.block_table, params.seq_lens,
                    num_seqs, num_heads, num_kv_heads,
                    params.max_blocks_per_seq, max_context_len,
                    block_size, ctx.stream());
                break;
            case 128:
                launch_paged_attention_v2<128>(
                    out, exp_sums, max_logits, tmp_out,
                    (const sycl::half *)Q->data,
                    (const sycl::half *)K->data,
                    (const sycl::half *)V->data,
                    params.scale,
                    params.block_table, params.seq_lens,
                    num_seqs, num_heads, num_kv_heads,
                    params.max_blocks_per_seq, max_context_len,
                    block_size, ctx.stream());
                break;
            case 256:
                launch_paged_attention_v2<256>(
                    out, exp_sums, max_logits, tmp_out,
                    (const sycl::half *)Q->data,
                    (const sycl::half *)K->data,
                    (const sycl::half *)V->data,
                    params.scale,
                    params.block_table, params.seq_lens,
                    num_seqs, num_heads, num_kv_heads,
                    params.max_blocks_per_seq, max_context_len,
                    block_size, ctx.stream());
                break;
            default:
                GGML_ABORT("Unsupported head dimension for V2 attention: %d", D);
        }

        // Wait for completion and free temp buffer
        ctx.stream()->wait();
        sycl::free(temp_buf, *ctx.stream());

        return;  // V2 dispatch complete
    }

    // ==========================================================================
    // Standard Dispatch (XMX or MMA-F16 kernels)
    // ==========================================================================

    if (Q->type == GGML_TYPE_F32) {
        switch (D) {
            case 64:
                ggml_sycl_flash_attn_ext_dispatch_ncols<64, float>(ctx, params);
                break;
            case 128:
                ggml_sycl_flash_attn_ext_dispatch_ncols<128, float>(ctx, params);
                break;
            case 256:
                ggml_sycl_flash_attn_ext_dispatch_ncols<256, float>(ctx, params);
                break;
            default:
                GGML_ABORT("Unsupported head dimension for SYCL flash attention: %d", D);
        }
    } else if (Q->type == GGML_TYPE_F16) {
        switch (D) {
            case 64:
                ggml_sycl_flash_attn_ext_dispatch_ncols<64, sycl::half>(ctx, params);
                break;
            case 128:
                ggml_sycl_flash_attn_ext_dispatch_ncols<128, sycl::half>(ctx, params);
                break;
            case 256:
                ggml_sycl_flash_attn_ext_dispatch_ncols<256, sycl::half>(ctx, params);
                break;
            default:
                GGML_ABORT("Unsupported head dimension for SYCL flash attention: %d", D);
        }
    } else {
         GGML_ABORT("Unsupported Q type for SYCL flash attention");
    }

    // Debug dumping controlled by GGML_SYCL_FA_DEBUG environment variable
    // Level 1: Basic inputs/outputs for first few heads
    // Level 2: Verbose mode with all heads and intermediate values
    if (fattn_debug_level() > 0) {
        // Wait for kernel to complete
        stream->wait();

        // Track call count
        static int fa_call_count = 0;
        fa_call_count++;

        // Only dump first N calls
        const int max_dumps = 20;
        if (fa_call_count <= max_dumps) {
            auto& dbg = get_fattn_debug_ctx();
            dbg.call_id = fa_call_count;
            dbg.n_queries = params.ne01;
            dbg.n_heads = params.ne02;
            dbg.n_kv_heads = params.ne12;
            dbg.n_kv = params.ne11;
            dbg.D = D;
            dbg.scale = params.scale;
            dbg.is_fa_on = true;

            dbg.open_file("on");

            int Q_type_size = (Q->type == GGML_TYPE_F32) ? sizeof(float) : sizeof(sycl::half);
            fattn_debug_dump_Q(stream, params.Q, Q_type_size, D, params.ne01, params.ne02,
                               params.nb01, params.nb02, params.scale);
            fattn_debug_dump_K(stream, params.K, D, params.ne11, params.ne12,
                               params.nb11, params.nb12);
            fattn_debug_dump_V(stream, params.V, D, params.ne11, params.ne12,
                               params.nb21, params.nb22);
            fattn_debug_dump_mask(stream, params.mask, params.ne30, params.ne01,
                                  params.nb31, params.ne30);
            fattn_debug_dump_output(stream, params.dst, D, params.ne01, params.ne02);

            dbg.close_file();

            fprintf(stderr, "[FA-DEBUG] FA_ON call %d: ne01=%d, ne02=%d, ne12=%d, D=%d, n_kv=%d, scale=%.4f\n",
                    fa_call_count, params.ne01, params.ne02, params.ne12, D, params.ne11, params.scale);
        }
    }
}
