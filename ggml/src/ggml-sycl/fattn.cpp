//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "fattn.hpp"
#include "common.hpp"

#include "fattn-debug.hpp"
#include "fattn-esimd-f16.hpp"
#include "fattn-mma-f16.hpp"
#include "fattn-mma.hpp"
#include "fattn-v2-esimd.hpp"
#include "fattn-v2-partition.hpp"
#include "fattn-vec.hpp"
#include "fattn-xmx-f16.hpp"
#include "kv-cache-quant.hpp"
#include "sycl-profiling.hpp"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

// Kernel names for VTune profiling
class fattn_v2_fill_block_table_kernel;
class fattn_v2_fill_block_table_runtime_kernel;
class fattn_v2_fill_seq_lens_kernel;

// =============================================================================
// V2 Partitioned Attention Compile-Time Switch
// =============================================================================
// When enabled (1), V2 multi-partition algorithm is available for sequences > 512
// When disabled (0), always uses standard XMX flash attention
// Can also be set via cmake: -DGGML_SYCL_FA_V2_ENABLED=OFF
//
// NOTE: V2 kernel now uses stride-based addressing (nb11, nb12, nb21, nb22)
// matching the XMX kernel pattern for compatibility with llama.cpp's KV cache layout.
#ifndef GGML_SYCL_FA_V2_ENABLED
#    define GGML_SYCL_FA_V2_ENABLED 1  // Enabled: stride-based addressing implemented
#endif

// =============================================================================
// Paged Attention V2 Configuration
// =============================================================================
// V2 uses multi-partition algorithm for long sequences (>512 tokens)
// This enables O(n) memory complexity for O(n²) attention
//
// Enable with environment variable: GGML_SYCL_PAGED_V2=1
//
// V2 modes:
// 1. With paged KV layout: Uses actual block tables for logical->physical mapping
// 2. With contiguous KV layout (auto-V2): Generates identity block table with block_size=1
//
// Auto-V2 allows V2 partitioning benefits without requiring paged KV cache changes

#if GGML_SYCL_FA_V2_ENABLED

static bool g_sycl_paged_v2_enabled     = false;
static bool g_sycl_paged_v2_initialized = false;

static void init_paged_v2_config() {
    if (g_sycl_paged_v2_initialized) {
        return;
    }
    g_sycl_paged_v2_initialized = true;

    const char * env = std::getenv("GGML_SYCL_PAGED_V2");
    if (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) {
        g_sycl_paged_v2_enabled = true;
        fprintf(stderr, "[SYCL] Paged Attention V2 enabled for long sequences (>512 tokens)\n");
        fprintf(stderr, "[SYCL]   Auto-V2 mode: identity block table with block_size=1\n");
    }
}

// =============================================================================
// Auto-V2: Thread-local buffers for identity block table (contiguous KV mode)
// =============================================================================
// When V2 is enabled but paged KV is not active, we generate an identity block
// table where block_table[i] = i and block_size = 1. This allows V2 partitioning
// to work with the existing contiguous 3D KV cache layout.

struct v2_auto_buffers {
    int32_t * block_table          = nullptr;  // Identity block table [num_seqs, max_blocks]
    int32_t * seq_lens             = nullptr;  // Sequence lengths [num_seqs]
    size_t    block_table_capacity = 0;        // Current capacity in elements
    size_t    seq_lens_capacity    = 0;        // Current capacity in elements

    // Persistent temp buffers for V2 kernel (avoids malloc/free during graph recording)
    // Layout: [exp_sums | max_logits | tmp_out]
    void * temp_buf          = nullptr;
    size_t temp_buf_capacity = 0;  // Current capacity in bytes

    sycl::queue * alloc_queue = nullptr;

    ~v2_auto_buffers() {
        if (block_table && alloc_queue) {
            ggml_sycl::unified_cache_sub_runtime_bytes(ggml_sycl_get_device_id_from_queue(*alloc_queue),
                                                       block_table_capacity * sizeof(int32_t));
            sycl::free(block_table, *alloc_queue);
        }
        if (seq_lens && alloc_queue) {
            ggml_sycl::unified_cache_sub_runtime_bytes(ggml_sycl_get_device_id_from_queue(*alloc_queue),
                                                       seq_lens_capacity * sizeof(int32_t));
            sycl::free(seq_lens, *alloc_queue);
        }
        if (temp_buf && alloc_queue) {
            ggml_sycl::unified_cache_sub_runtime_bytes(ggml_sycl_get_device_id_from_queue(*alloc_queue),
                                                       temp_buf_capacity);
            sycl::free(temp_buf, *alloc_queue);
        }
    }
};

static thread_local v2_auto_buffers g_v2_auto;

// Thread-local buffers for multi-sequence flash attention
// Freed on thread exit to avoid leaking device memory.
static std::atomic<bool> g_fattn_shutting_down{ false };
static std::atomic<bool> g_fattn_atexit_registered{ false };
static std::atomic<int>  g_seq_id_buffer_instances{ 0 };
static std::atomic<int>  g_seq_id_buffer_allocs{ 0 };

// atexit handler to prevent SYCL cleanup during static destruction
static void fattn_atexit_handler() {
    g_fattn_shutting_down.store(true);
}

static void register_fattn_atexit() {
    bool expected = false;
    if (g_fattn_atexit_registered.compare_exchange_strong(expected, true)) {
        std::atexit(fattn_atexit_handler);
    }
}

struct tl_seq_id_buffers {
    int32_t *     q_seq_ids_dev        = nullptr;
    int32_t *     kv_seq_ids_dev       = nullptr;
    size_t        q_seq_ids_size       = 0;
    size_t        kv_seq_ids_size      = 0;
    int32_t *     seq_q_offsets_dev    = nullptr;
    int32_t *     seq_kv_offsets_dev   = nullptr;
    size_t        seq_offsets_capacity = 0;
    sycl::queue * alloc_queue          = nullptr;

    tl_seq_id_buffers() {
        register_fattn_atexit();
        g_seq_id_buffer_instances.fetch_add(1);
    }

    ~tl_seq_id_buffers() {
        free_all();
        g_seq_id_buffer_instances.fetch_sub(1);
    }

    void free_ptr(int32_t *& ptr, size_t bytes) {
        if (ptr && alloc_queue) {
            sycl::free(ptr, *alloc_queue);
            ggml_sycl::unified_cache_sub_runtime_bytes(ggml_sycl_get_device_id_from_queue(*alloc_queue), bytes);
            ptr = nullptr;
            g_seq_id_buffer_allocs.fetch_sub(1);
        }
    }

    int32_t * alloc_ptr(size_t count, sycl::queue * stream) {
        const size_t bytes = sizeof(int32_t) * count;
        ggml_sycl::unified_cache_add_runtime_bytes(ggml_sycl_get_device_id_from_queue(*stream), bytes);
        int32_t * ptr = ggml_sycl_malloc_device_t<int32_t>(count, *stream, "fattn_block_table");
        if (ptr) {
            g_seq_id_buffer_allocs.fetch_add(1);
        } else {
            ggml_sycl::unified_cache_sub_runtime_bytes(ggml_sycl_get_device_id_from_queue(*stream), bytes);
        }
        return ptr;
    }

    void free_all() {
        if (g_fattn_shutting_down.load(std::memory_order_acquire)) {
            return;
        }
        free_ptr(q_seq_ids_dev, q_seq_ids_size);
        free_ptr(kv_seq_ids_dev, kv_seq_ids_size);
        free_ptr(seq_q_offsets_dev, seq_offsets_capacity);
        free_ptr(seq_kv_offsets_dev, seq_offsets_capacity);
        q_seq_ids_size       = 0;
        kv_seq_ids_size      = 0;
        seq_offsets_capacity = 0;
    }
};

static thread_local tl_seq_id_buffers g_tl_seq_buffers;

// Test hooks for verifying thread-local cleanup.
extern "C" int ggml_sycl_test_seq_id_buffer_instances() {
    return g_seq_id_buffer_instances.load();
}

extern "C" int ggml_sycl_test_seq_id_buffer_allocs() {
    return g_seq_id_buffer_allocs.load();
}

extern "C" void ggml_sycl_test_fattn_set_shutdown(int value) {
    g_fattn_shutting_down.store(value != 0, std::memory_order_release);
}

extern "C" void ggml_sycl_test_seq_id_buffers_free_all() {
    g_tl_seq_buffers.free_all();
}

extern "C" bool ggml_sycl_test_seq_id_buffers_touch(sycl::queue * stream) {
    if (!stream) {
        return false;
    }
    if (g_tl_seq_buffers.alloc_queue && g_tl_seq_buffers.alloc_queue != stream) {
        g_tl_seq_buffers.free_all();
    }
    g_tl_seq_buffers.alloc_queue = stream;

    if (!g_tl_seq_buffers.q_seq_ids_dev) {
        g_tl_seq_buffers.q_seq_ids_dev  = g_tl_seq_buffers.alloc_ptr(1, stream);
        g_tl_seq_buffers.q_seq_ids_size = sizeof(int32_t);
    }
    if (!g_tl_seq_buffers.kv_seq_ids_dev) {
        g_tl_seq_buffers.kv_seq_ids_dev  = g_tl_seq_buffers.alloc_ptr(1, stream);
        g_tl_seq_buffers.kv_seq_ids_size = sizeof(int32_t);
    }
    if (!g_tl_seq_buffers.seq_q_offsets_dev) {
        g_tl_seq_buffers.seq_q_offsets_dev = g_tl_seq_buffers.alloc_ptr(1, stream);
    }
    if (!g_tl_seq_buffers.seq_kv_offsets_dev) {
        g_tl_seq_buffers.seq_kv_offsets_dev = g_tl_seq_buffers.alloc_ptr(1, stream);
    }

    return g_tl_seq_buffers.q_seq_ids_dev && g_tl_seq_buffers.kv_seq_ids_dev && g_tl_seq_buffers.seq_q_offsets_dev &&
           g_tl_seq_buffers.seq_kv_offsets_dev;
}

// Pre-allocate V2 buffers before SYCL graph recording starts.
// This ensures V2 dispatch works during graph recording (malloc/free forbidden during recording).
// Called from ggml-sycl.cpp before graph recording.
//
// Parameters are estimated from cgraph FLASH_ATTN_EXT ops - we allocate for the maximum
// expected context length found in the graph.
void ggml_sycl_v2_pre_allocate_buffers(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph) {
    // Initialize V2 config if not already done
    init_paged_v2_config();

    // Skip if V2 is not enabled
    if (!g_sycl_paged_v2_enabled) {
        return;
    }

    // Find maximum context length from FLASH_ATTN_EXT ops in graph
    int max_context_len  = 0;
    int max_num_heads    = 0;
    int max_num_kv_heads = 0;
    int max_head_dim     = 0;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op != GGML_OP_FLASH_ATTN_EXT) {
            continue;
        }
        // K tensor has shape [D, n_kv, n_kv_heads] where n_kv is context length
        const ggml_tensor * K = node->src[1];
        if (!K) {
            continue;
        }

        const int ctx_len      = (int) K->ne[1];  // n_kv (current KV cache length)
        const int num_kv_heads = (int) K->ne[2];
        const int head_dim     = (int) K->ne[0];

        // Q tensor has shape [D, n_q, n_heads]
        const ggml_tensor * Q         = node->src[0];
        const int           num_heads = Q ? (int) Q->ne[2] : num_kv_heads;

        if (ctx_len > max_context_len) {
            max_context_len = ctx_len;
        }
        if (num_heads > max_num_heads) {
            max_num_heads = num_heads;
        }
        if (num_kv_heads > max_num_kv_heads) {
            max_num_kv_heads = num_kv_heads;
        }
        if (head_dim > max_head_dim) {
            max_head_dim = head_dim;
        }
    }

    // No FLASH_ATTN_EXT ops found, nothing to pre-allocate
    if (max_context_len == 0) {
        return;
    }

    // Only pre-allocate if context is long enough to use V2
    // V2 threshold is 512 tokens (from should_use_paged_attention_v2)
    if (max_context_len <= 512) {
        return;
    }

    // Calculate buffer sizes for auto-V2 mode
    // For auto-V2: block_size = 1, so max_blocks_per_seq = max_context_len
    const int num_seqs         = 1;  // Decode mode typically has 1 query per sequence
    const int block_table_size = num_seqs * max_context_len;
    const int seq_lens_size    = num_seqs;

    // Calculate temp buffer size
    const size_t temp_size = paged_attention_v2_temp_size(num_seqs, max_num_heads, max_context_len, max_head_dim);

    // Pre-allocate block_table if needed
    if (!g_v2_auto.block_table || g_v2_auto.alloc_queue != ctx.stream() ||
        g_v2_auto.block_table_capacity < (size_t) block_table_size) {
        if (g_v2_auto.block_table && g_v2_auto.alloc_queue) {
            ggml_sycl::unified_cache_sub_runtime_bytes(ggml_sycl_get_device_id_from_queue(*g_v2_auto.alloc_queue),
                                                       g_v2_auto.block_table_capacity * sizeof(int32_t));
            sycl::free(g_v2_auto.block_table, *g_v2_auto.alloc_queue);
        }
        ggml_sycl::unified_cache_add_runtime_bytes(ggml_sycl_get_device_id_from_queue(*ctx.stream()),
                                                   block_table_size * sizeof(int32_t));
        g_v2_auto.block_table          = ggml_sycl_malloc_device_t<int32_t>(block_table_size, *ctx.stream(), "fattn_block_table");
        if (!g_v2_auto.block_table) {
            ggml_sycl::unified_cache_sub_runtime_bytes(ggml_sycl_get_device_id_from_queue(*ctx.stream()),
                                                       block_table_size * sizeof(int32_t));
        }
        g_v2_auto.block_table_capacity = block_table_size;
        GGML_SYCL_DEBUG("[V2-PREALLOC] block_table allocated: %d elements\n", block_table_size);
    }

    // Pre-allocate seq_lens if needed
    if (!g_v2_auto.seq_lens || g_v2_auto.alloc_queue != ctx.stream() ||
        g_v2_auto.seq_lens_capacity < (size_t) seq_lens_size) {
        if (g_v2_auto.seq_lens && g_v2_auto.alloc_queue) {
            ggml_sycl::unified_cache_sub_runtime_bytes(ggml_sycl_get_device_id_from_queue(*g_v2_auto.alloc_queue),
                                                       g_v2_auto.seq_lens_capacity * sizeof(int32_t));
            sycl::free(g_v2_auto.seq_lens, *g_v2_auto.alloc_queue);
        }
        ggml_sycl::unified_cache_add_runtime_bytes(ggml_sycl_get_device_id_from_queue(*ctx.stream()),
                                                   seq_lens_size * sizeof(int32_t));
        g_v2_auto.seq_lens          = ggml_sycl_malloc_device_t<int32_t>(seq_lens_size, *ctx.stream(), "fattn_seq_lens");
        if (!g_v2_auto.seq_lens) {
            ggml_sycl::unified_cache_sub_runtime_bytes(ggml_sycl_get_device_id_from_queue(*ctx.stream()),
                                                       seq_lens_size * sizeof(int32_t));
        }
        g_v2_auto.seq_lens_capacity = seq_lens_size;
        GGML_SYCL_DEBUG("[V2-PREALLOC] seq_lens allocated: %d elements\n", seq_lens_size);
    }

    // Pre-allocate temp buffer if needed
    if (!g_v2_auto.temp_buf || g_v2_auto.alloc_queue != ctx.stream() || g_v2_auto.temp_buf_capacity < temp_size) {
        if (g_v2_auto.temp_buf && g_v2_auto.alloc_queue) {
            ggml_sycl::unified_cache_sub_runtime_bytes(ggml_sycl_get_device_id_from_queue(*g_v2_auto.alloc_queue),
                                                       g_v2_auto.temp_buf_capacity);
            sycl::free(g_v2_auto.temp_buf, *g_v2_auto.alloc_queue);
        }
        ggml_sycl::unified_cache_add_runtime_bytes(ggml_sycl_get_device_id_from_queue(*ctx.stream()), temp_size);
        g_v2_auto.temp_buf          = ggml_sycl_malloc_device_t<uint8_t>(temp_size, *ctx.stream(), "fattn_temp");
        if (!g_v2_auto.temp_buf) {
            ggml_sycl::unified_cache_sub_runtime_bytes(ggml_sycl_get_device_id_from_queue(*ctx.stream()), temp_size);
        }
        g_v2_auto.temp_buf_capacity = temp_size;
        GGML_SYCL_DEBUG("[V2-PREALLOC] temp_buf allocated: %zu bytes\n", temp_size);
    }

    // Update alloc_queue
    g_v2_auto.alloc_queue = ctx.stream();

    // Pre-fill identity block table (needed for graph recording)
    // This is a kernel launch, which is fine during graph recording prep
    ctx.stream()->parallel_for<class fattn_v2_fill_block_table_kernel>(
        sycl::range<1>(block_table_size), [block_table_ptr = g_v2_auto.block_table](sycl::id<1> idx) {
            block_table_ptr[idx] = static_cast<int32_t>(idx[0]);
        });

    // Wait for pre-allocation to complete before graph recording starts
    ctx.stream()->wait();

    fprintf(stderr, "[SYCL] V2 buffers pre-allocated for graph recording (ctx=%d, heads=%d, D=%d)\n", max_context_len,
            max_num_heads, max_head_dim);
}

#else   // GGML_SYCL_FA_V2_ENABLED == 0

// Stub when V2 is disabled
void ggml_sycl_v2_pre_allocate_buffers(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph) {
    (void) ctx;
    (void) cgraph;
    // V2 disabled at compile time, nothing to pre-allocate
}

#endif  // GGML_SYCL_FA_V2_ENABLED

// =============================================================================
// FP8 KV Cache Configuration
// =============================================================================
// FP8 quantization provides 2x memory reduction for KV cache
//
// Enable with environment variable: GGML_SYCL_KV_FP8=1
// Requires FP8-compatible KV cache allocation

static bool g_sycl_kv_fp8_enabled     = false;
static bool g_sycl_kv_fp8_initialized = false;

static void init_kv_fp8_config() {
    if (g_sycl_kv_fp8_initialized) {
        return;
    }
    g_sycl_kv_fp8_initialized = true;

    const char * env = std::getenv("GGML_SYCL_KV_FP8");
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

static bool g_sycl_fa_esimd_enabled     = false;
static bool g_sycl_fa_esimd_initialized = false;
static bool g_sycl_fa_safe_decode_initialized = false;
static bool g_sycl_fa_safe_decode_enabled     = true;

static void init_fa_esimd_config() {
    if (g_sycl_fa_esimd_initialized) {
        return;
    }
    g_sycl_fa_esimd_initialized = true;

    // ESIMD FA is enabled by default when available (7-8% decode speedup)
    // Disable with GGML_SYCL_FA_ESIMD=0
    const char * env          = std::getenv("GGML_SYCL_FA_ESIMD");
    bool         enable_esimd = true;  // Default to enabled
    if (env && (strcmp(env, "0") == 0 || strcmp(env, "false") == 0)) {
        enable_esimd = false;
    }

    if (enable_esimd && fattn_esimd_f16_available()) {
        g_sycl_fa_esimd_enabled = true;
        GGML_SYCL_DEBUG("[SYCL] ESIMD Flash Attention enabled for decode\n");
    }
}

static void init_fa_safe_decode_config() {
    if (g_sycl_fa_safe_decode_initialized) {
        return;
    }
    g_sycl_fa_safe_decode_initialized = true;
    const char * env = std::getenv("GGML_SYCL_FA_SAFE_DECODE");
    if (env && (strcmp(env, "0") == 0 || strcmp(env, "false") == 0)) {
        g_sycl_fa_safe_decode_enabled = false;
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
    const ggml_tensor *    mask_f16,        // The F16 mask tensor (may be a cast of F32)
    int                    n_queries,       // Number of query tokens
    int                    n_kv,            // Number of KV positions
    std::vector<int32_t> & seq_q_offsets,   // Output: [n_seqs + 1]
    std::vector<int32_t> & seq_kv_offsets)  // Output: [n_seqs + 1]
{
    if (!mask_f16) {
        return false;
    }

    // Try to get the original F32 mask from the cast/copy operation's source
    // ggml_cast uses GGML_OP_CPY with src[0] = source tensor, src[1] = result
    const ggml_tensor * mask_f32 = nullptr;

    if (mask_f16->op == GGML_OP_CPY && mask_f16->src[0] && mask_f16->src[0]->type == GGML_TYPE_F32) {
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

    const float * mask_data   = (const float *) mask_f32->data;
    const int64_t mask_stride = mask_f32->nb[1] / sizeof(float);  // Stride between query rows

    // Scan the mask to find sequence boundaries
    // For each query, find the first and last valid (non-INF) KV position
    // Sequence boundaries occur where there's a discontinuity in valid KV ranges

    seq_q_offsets.clear();
    seq_kv_offsets.clear();
    seq_q_offsets.push_back(0);

    int prev_kv_start = -1;
    int prev_kv_end   = -1;

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
        prev_kv_end   = std::max(prev_kv_end, kv_end);
    }

    // Close the last sequence
    seq_q_offsets.push_back(n_queries);
    if (prev_kv_end > 0) {
        seq_kv_offsets.push_back(prev_kv_end);
    } else {
        seq_kv_offsets.push_back(n_kv);
    }

    // Only enable multi-sequence optimization if we detected multiple sequences
    int n_seqs = (int) seq_q_offsets.size() - 1;
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
static int compute_sequence_boundaries_from_ids(const int32_t * q_seq_ids,   // [n_queries] Sequence ID for each query
                                                int             n_queries,
                                                const int32_t * kv_seq_ids,  // [n_kv] Sequence ID for each KV position
                                                int             n_kv,
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
            if (seq >= (int) seq_ranges.size()) {
                seq_ranges.resize(seq + 1);
            }
            if (seq_ranges[seq].q_start < 0) {
                seq_ranges[seq].q_start = q;
            }
            prev_seq = seq;
        }
        if (seq >= 0 && seq < (int) seq_ranges.size()) {
            seq_ranges[seq].q_end = q + 1;
        }
    }

    // Scan KV sequence IDs to find KV boundaries
    prev_seq = -1;
    for (int k = 0; k < n_kv; ++k) {
        int seq = kv_seq_ids[k];
        if (seq >= 0 && seq < (int) seq_ranges.size()) {
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
    // Enabled by default; allow explicit disable for debugging/regressions.
    static const bool fa_ext_enabled = []() {
        const char * env = std::getenv("GGML_SYCL_FLASH_ATTN_EXT");
        if (!env) {
            return true;
        }
        return !(strcmp(env, "0") == 0 || strcmp(env, "false") == 0);
    }();
    if (!fa_ext_enabled) {
        return false;
    }

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
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

    // Masked flash attention is supported; keep the gate in place via GGML_SYCL_FLASH_ATTN_EXT if needed.

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
static void ggml_sycl_flash_attn_ext_dispatch_ncols(ggml_backend_sycl_context & ctx, const fattn_params & params) {
    dpct::queue_ptr stream = ctx.stream();

    // Select ncols based on batch size (ne01 = number of queries)
    const int ne01          = params.ne01;
    float     logit_softcap = params.logit_softcap;

    // Runtime kernel selection based on GPU capabilities
    // Check if the device has XMX (Intel matrix extension) support
    sycl::device dev     = stream->get_device();
    bool         use_xmx = gpu_has_xmx(dev);
    const bool   safe_decode = g_sycl_fa_safe_decode_enabled && ne01 <= 1;
    if (safe_decode) {
        use_xmx = false;
    }

    // ESIMD kernel dispatch (uses explicit SIMD operations)
    // - Single query (ne01 <= 1): +7% speedup for decode on Mistral 7B
    // - Batched queries (ne01 <= 8): Only for D=64 due to SLM constraints
    if (!safe_decode && g_sycl_fa_esimd_enabled && fattn_esimd_f16_available()) {
        if (ne01 <= 1) {
            // Partitioned kernel: each thread processes disjoint KV ranges
            GGML_SYCL_KTRACE("fattn_esimd_f16", " D=%d ne01=%d", D, ne01);
            fattn_esimd_f16<D, Q_type>(params, *stream);
            return;
        } else if (false && D == 64 && ne01 <= 8 && fattn_esimd_batched_fits_slm<D>()) {
            // DISABLED: Batched ESIMD is 20% slower than XMX for pp
            // Keep single-query ESIMD for tg (ne01 <= 1) which has +7% speedup
            GGML_SYCL_KTRACE("fattn_esimd_f16_batched", " D=%d ne01=%d", D, ne01);
            fattn_esimd_f16_batched<D, Q_type>(params, *stream);
            return;
        }
    }

// Helper macro to dispatch based on softcap
#define DISPATCH_NCOLS(NCOLS, LAUNCHER)                    \
    if (logit_softcap == 0.0f) {                           \
        LAUNCHER<D, NCOLS, false, Q_type>(params, stream); \
    } else {                                               \
        LAUNCHER<D, NCOLS, true, Q_type>(params, stream);  \
    }

    // Dispatch to appropriate kernel based on GPU capabilities
    if (use_xmx) {
        // XMX kernel - uses Intel joint_matrix for Q@K^T and S@V acceleration
        // Note: Even for single query, XMX kernel is faster than vector kernel
        // because XMX hardware throughput exceeds scalar operations despite
        // 7/8 of tiles being "wasted" in the 8x8 joint_matrix operations
        if (ne01 <= 1) {
            GGML_SYCL_KTRACE("fattn_xmx_f16", " D=%d ncols=1 ne01=%d", D, ne01);
            DISPATCH_NCOLS(1, launch_fattn_xmx_f16);
        } else if (ne01 <= 2) {
            GGML_SYCL_KTRACE("fattn_xmx_f16", " D=%d ncols=2 ne01=%d", D, ne01);
            DISPATCH_NCOLS(2, launch_fattn_xmx_f16);
        } else if (ne01 <= 4) {
            GGML_SYCL_KTRACE("fattn_xmx_f16", " D=%d ncols=4 ne01=%d", D, ne01);
            DISPATCH_NCOLS(4, launch_fattn_xmx_f16);
        } else {
            GGML_SYCL_KTRACE("fattn_xmx_f16", " D=%d ncols=8 ne01=%d", D, ne01);
            DISPATCH_NCOLS(8, launch_fattn_xmx_f16);
        }
    } else {
        // MMA F16 kernel - scalar fallback for non-XMX GPUs
        if (ne01 <= 1) {
            GGML_SYCL_KTRACE("fattn_mma_f16", " D=%d ncols=1 ne01=%d", D, ne01);
            DISPATCH_NCOLS(1, launch_fattn_mma_f16);
        } else if (ne01 <= 2) {
            GGML_SYCL_KTRACE("fattn_mma_f16", " D=%d ncols=2 ne01=%d", D, ne01);
            DISPATCH_NCOLS(2, launch_fattn_mma_f16);
        } else if (ne01 <= 4) {
            GGML_SYCL_KTRACE("fattn_mma_f16", " D=%d ncols=4 ne01=%d", D, ne01);
            DISPATCH_NCOLS(4, launch_fattn_mma_f16);
        } else {
            GGML_SYCL_KTRACE("fattn_mma_f16", " D=%d ncols=8 ne01=%d", D, ne01);
            DISPATCH_NCOLS(8, launch_fattn_mma_f16);
        }
    }

#undef DISPATCH_NCOLS
}

// Main flash attention entry point
void ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_SYCL_PROFILE_SCOPE_FA("flash_attn");
    // Initialize configuration on first call
#if GGML_SYCL_FA_V2_ENABLED
    init_paged_v2_config();
#endif
    init_kv_fp8_config();
    init_fa_esimd_config();
    init_fa_safe_decode_config();

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    // Flash attention uses activation tensors (Q/K/V + KV cache). No weight tensors
    // are expected here, so unified-cache streaming is intentionally skipped.
    GGML_ASSERT(!ggml_sycl_tensor_is_weight(Q));
    GGML_ASSERT(!ggml_sycl_tensor_is_weight(K));
    GGML_ASSERT(!ggml_sycl_tensor_is_weight(V));

    // Check tiered dispatch for K/V cache tensors
    if (g_tiered_enabled.load(std::memory_order_acquire)) {
        // Check K tensor (name[0] checks if name is non-empty)
        if (K && K->name[0]) {
            ggml_sycl::memory_tier tier;
            bool                   found = false;
            void *                 ptr   = get_cached_tensor_ptr(K->name, &tier, &found);
            if (ptr) {
                GGML_SYCL_DEBUG("[SYCL] fattn K tiered hit: %s (tier=%d)\n", K->name, static_cast<int>(tier));
            } else if (found) {
                GGML_SYCL_DEBUG("[SYCL] fattn K tiered pending: %s\n", K->name);
            }
        }

        // Check V tensor (name[0] checks if name is non-empty)
        if (V && V->name[0]) {
            ggml_sycl::memory_tier tier;
            bool                   found = false;
            void *                 ptr   = get_cached_tensor_ptr(V->name, &tier, &found);
            if (ptr) {
                GGML_SYCL_DEBUG("[SYCL] fattn V tiered hit: %s (tier=%d)\n", V->name, static_cast<int>(tier));
            } else if (found) {
                GGML_SYCL_DEBUG("[SYCL] fattn V tiered pending: %s\n", V->name);
            }
        }
    }

    // Event-based synchronization moved to ubatch level (llama-context.cpp)
    // This flash_attn no longer waits on individual events - the wait happens
    // at the START of each ubatch before any kernels are submitted.
    // This ensures all previous ubatch's KV writes complete before this ubatch starts.
    const ggml_tensor * mask       = dst->src[3];
    const ggml_tensor * sinks      = dst->src[4];  // Attention sinks tensor (may be null)
    const ggml_tensor * q_seq_ids  = dst->src[5];  // Sequence IDs for query tokens (may be null)
    const ggml_tensor * kv_seq_ids = dst->src[6];  // Sequence IDs for KV positions (may be null)

    GGML_ASSERT(Q->type == GGML_TYPE_F32 || Q->type == GGML_TYPE_F16);
    GGML_ASSERT(K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_F8_E4M3);  // FP16 or FP8 KV cache
    GGML_ASSERT(V->type == GGML_TYPE_F16 || V->type == GGML_TYPE_F8_E4M3);  // FP16 or FP8 KV cache
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // Extract scale, max_bias, and logit_softcap from op_params
    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    // Read use_paged_layout from op_params[4] (set by ggml_flash_attn_ext_set_paged_layout)
    // op_params layout: [0-2]=float scale/max_bias/logit_softcap, [3]=prec, [4]=use_paged_layout
    const int32_t use_paged_layout_i32 = ((const int32_t *) dst->op_params)[4];
    const bool    use_paged_layout     = (use_paged_layout_i32 != 0);

    // If using logit_softcap, adjust scale
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    // Calculate ALiBi parameters
    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // Build the params structure
    fattn_params params;

    // Use device-specific pointers for TP mode (KV cache is allocated per-device)
    const int device = ctx.device;
    params.Q         = (const char *) ggml_sycl_get_data_ptr(Q, device);
    params.K         = (const char *) ggml_sycl_get_data_ptr(K, device);
    params.V         = (const char *) ggml_sycl_get_data_ptr(V, device);
    params.mask      = mask ? (const char *) ggml_sycl_get_data_ptr(mask, device) : nullptr;
    params.sinks     = sinks ? (const char *) ggml_sycl_get_data_ptr(sinks, device) : nullptr;
    params.dst       = (float *) ggml_sycl_get_data_ptr(dst, device);


    params.scale         = scale;
    params.max_bias      = max_bias;
    params.m0            = m0;
    params.m1            = m1;
    params.n_head_log2   = n_head_log2;
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
    params.n_seqs         = 0;
    params.seq_q_offsets  = nullptr;
    params.seq_kv_offsets = nullptr;

    // Thread-local storage for sequence boundary offsets
    // These persist across calls within the same thread
    thread_local std::vector<int32_t> tl_seq_q_offsets;
    thread_local std::vector<int32_t> tl_seq_kv_offsets;

    // Thread-local device buffers for seq_ids (reused across calls)
    // We need to manage these manually since host tensor->data can't be accessed from GPU
    tl_seq_id_buffers & tl_buffers = g_tl_seq_buffers;

    // Use sequence ID tensors if provided
    // Skip during SYCL graph recording - this optimization requires wait() calls
    // which are not allowed during graph recording
    if (!g_ggml_sycl_graph_recording && q_seq_ids && kv_seq_ids && q_seq_ids->type == GGML_TYPE_I32 &&
        kv_seq_ids->type == GGML_TYPE_I32 && q_seq_ids->data && kv_seq_ids->data) {
        const size_t q_size  = q_seq_ids->ne[0] * sizeof(int32_t);
        const size_t kv_size = kv_seq_ids->ne[0] * sizeof(int32_t);

        // Check if tensors are on host buffers (need to copy to device)
        bool q_on_host  = q_seq_ids->buffer ? ggml_backend_buffer_is_host(q_seq_ids->buffer) : true;
        bool kv_on_host = kv_seq_ids->buffer ? ggml_backend_buffer_is_host(kv_seq_ids->buffer) : true;

        // Get the host pointers from thread-local cache (set by llama-kv-cache.cpp)
        // These are USM host pointers that are accessible from both CPU and GPU
        size_t          cached_q_count = 0, cached_kv_count = 0;
        const int32_t * cached_q_seq_ids  = ggml_sycl_get_seq_ids_host_q(&cached_q_count);
        const int32_t * cached_kv_seq_ids = ggml_sycl_get_seq_ids_host_kv(&cached_kv_count);

        // If tensors are on device (not host), use the cached host pointers instead
        // The scheduler creates device tensors but doesn't copy INPUT data, so tensor->data is invalid
        // However, the llama layer has set the USM host pointers via ggml_backend_sycl_set_seq_ids_host()
        const int32_t * host_q_ptr  = q_on_host ? (const int32_t *) q_seq_ids->data : cached_q_seq_ids;
        const int32_t * host_kv_ptr = kv_on_host ? (const int32_t *) kv_seq_ids->data : cached_kv_seq_ids;

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
            if (tl_buffers.alloc_queue != stream || tl_buffers.q_seq_ids_size < q_size) {
                tl_buffers.free_ptr(tl_buffers.q_seq_ids_dev, tl_buffers.q_seq_ids_size);
                tl_buffers.q_seq_ids_dev  = tl_buffers.alloc_ptr(q_seq_ids->ne[0], stream);
                tl_buffers.q_seq_ids_size = q_size;
            }
            if (tl_buffers.alloc_queue != stream || tl_buffers.kv_seq_ids_size < kv_size) {
                tl_buffers.free_ptr(tl_buffers.kv_seq_ids_dev, tl_buffers.kv_seq_ids_size);
                tl_buffers.kv_seq_ids_dev  = tl_buffers.alloc_ptr(kv_seq_ids->ne[0], stream);
                tl_buffers.kv_seq_ids_size = kv_size;
            }
            tl_buffers.alloc_queue = stream;

            // Copy from host (USM) to device using the correct host pointers
            // Note: In-order queue ensures memcpy completes before subsequent kernel launch
            stream->memcpy(tl_buffers.q_seq_ids_dev, host_q_ptr, q_size);
            stream->memcpy(tl_buffers.kv_seq_ids_dev, host_kv_ptr, kv_size);

            params.q_seq_ids  = tl_buffers.q_seq_ids_dev;
            params.kv_seq_ids = tl_buffers.kv_seq_ids_dev;

            // Compute sequence boundaries from the seq_ids arrays
            // This enables the kernel to skip cross-sequence KV computation entirely
            // Note: We use the HOST pointers for boundary computation (which happens on CPU)
            int n_queries = static_cast<int>(q_seq_ids->ne[0]);
            int n_kv      = static_cast<int>(kv_seq_ids->ne[0]);

            int n_seqs = compute_sequence_boundaries_from_ids(host_q_ptr, n_queries, host_kv_ptr, n_kv,
                                                              tl_seq_q_offsets, tl_seq_kv_offsets);

            // Copy sequence boundary offsets to device memory for kernel access
            // This enables the kernel to skip entire KV blocks for non-matching sequences
            if (n_seqs > 1) {
                const size_t offsets_size = (n_seqs + 1) * sizeof(int32_t);

                // Reallocate device buffers if capacity is insufficient
                if (tl_buffers.seq_offsets_capacity < offsets_size) {
                    tl_buffers.free_ptr(tl_buffers.seq_q_offsets_dev, tl_buffers.seq_offsets_capacity);
                    tl_buffers.free_ptr(tl_buffers.seq_kv_offsets_dev, tl_buffers.seq_offsets_capacity);
                    tl_buffers.seq_q_offsets_dev    = tl_buffers.alloc_ptr(n_seqs + 1, stream);
                    tl_buffers.seq_kv_offsets_dev   = tl_buffers.alloc_ptr(n_seqs + 1, stream);
                    tl_buffers.seq_offsets_capacity = offsets_size;
                }

                // Copy offsets to device
                // Note: In-order queue ensures memcpy completes before subsequent kernel launch
                stream->memcpy(tl_buffers.seq_q_offsets_dev, tl_seq_q_offsets.data(), offsets_size);
                stream->memcpy(tl_buffers.seq_kv_offsets_dev, tl_seq_kv_offsets.data(), offsets_size);

                // Set params for kernel to use sequence boundary optimization
                params.n_seqs         = n_seqs;
                params.seq_q_offsets  = tl_buffers.seq_q_offsets_dev;
                params.seq_kv_offsets = tl_buffers.seq_kv_offsets_dev;
            }
        }
    } else {
        params.q_seq_ids  = nullptr;
        params.kv_seq_ids = nullptr;
    }

    // PagedAttention support
    // Block table (src[7]) and seq_lens (src[8]) are set via ggml_flash_attn_ext_set_paged()
    const ggml_tensor * block_table     = dst->src[7];
    const ggml_tensor * seq_lens_tensor = dst->src[8];

    if (block_table && seq_lens_tensor) {
        // Enable PagedAttention: K/V are stored in blocks, accessed via block_table
        params.use_paged_attn     = true;
        params.block_size         = 16;  // Fixed block size (matches vLLM and XMX tile size)
        // block_table shape: [max_blocks, n_seqs] where ne[0]=max_blocks (columns), ne[1]=n_seqs (rows)
        // Kernel access: block_table[seq * max_blocks + block] requires max_blocks = ne[0]
        params.max_blocks_per_seq = static_cast<int32_t>(block_table->ne[0]);
        params.block_table        = (const int32_t *) ggml_sycl_get_data_ptr(block_table, device);
        params.seq_lens           = (const int32_t *) ggml_sycl_get_data_ptr(seq_lens_tensor, device);

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
        params.use_paged_attn     = false;
        params.block_size         = 16;
        params.max_blocks_per_seq = 0;
        params.block_table        = nullptr;
        params.seq_lens           = nullptr;
    }

    // Set paged layout flag (read from op_params[4], set via ggml_flash_attn_ext_set_paged_layout)
    params.use_paged_layout = use_paged_layout;

    if (g_ggml_sycl_graph_recording && ctx.fa_graph_ptrs_recording) {
        ggml_backend_sycl_context::fa_graph_ptr_snapshot snap;
        snap.q           = params.Q;
        snap.k           = params.K;
        snap.v           = params.V;
        snap.dst         = params.dst;
        snap.mask        = params.mask;
        snap.sinks       = params.sinks;
        snap.block_table = params.block_table;
        snap.seq_lens    = params.seq_lens;
        ctx.fa_graph_ptrs.push_back(snap);
    }

    // Set FP8 KV cache flag - enables on-the-fly dequantization in flash attention kernel
    params.kv_is_fp8 = (K->type == GGML_TYPE_F8_E4M3 && V->type == GGML_TYPE_F8_E4M3);

    // Multi-token decode support - disabled by default
    // Will be enabled when q_positions array is provided via thread-local storage
    params.multi_token_decode = false;
    params.q_positions        = nullptr;
    params.kv_base_pos        = 0;

    const int D = Q->ne[0];

#if GGML_SYCL_FA_V2_ENABLED
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

    const int max_context_len = params.ne11;  // n_kv = sequence length
    bool      use_v2_dispatch = false;
    bool      use_auto_v2     = false;        // Auto-V2: identity block table for contiguous KV

    // Check if V2 dispatch should be used:
    // V2 has two modes:
    // 1. Paged V2: Uses actual block tables from paged KV cache
    // 2. Auto-V2: Generates identity block table for contiguous 3D KV cache
    //
    // Auto-V2 allows using V2's partitioned algorithm benefits (O(n) memory for softmax)
    // without requiring paged KV cache infrastructure changes.
    //
    // NOTE: V2 dispatch is NOW compatible with SYCL command graphs because:
    // 1. Temp buffers are persistent (g_v2_auto.temp_buf) - no malloc/free during dispatch
    // 2. No wait() calls after kernel launch
    // 3. All allocations happen on first use/resize only
    if (g_sycl_paged_v2_enabled && should_use_paged_attention_v2(max_context_len)) {
        if (params.use_paged_attn && params.block_table) {
            // Paged V2: Use actual block tables
            use_v2_dispatch = true;
            use_auto_v2     = false;
        } else {
            // Auto-V2: Generate identity block table for contiguous KV
            use_v2_dispatch                = true;
            use_auto_v2                    = true;
            static bool auto_v2_info_shown = false;
            if (!auto_v2_info_shown) {
                fprintf(stderr, "[SYCL] V2 Auto-Mode: Using identity block table for long sequences\n");
                auto_v2_info_shown = true;
            }
        }
    }

    // V2 dispatch: partition-based attention for long sequences
    // NOTE: V2 is designed for vLLM-style continuous batching where each "seq" in the batch
    // is a separate request with its own KV cache. In llama.cpp's single-sequence mode,
    // all query tokens share the same sequence's KV cache, so we need to check compatibility.

    // Get actual number of sequences from seq_lens tensor (already declared above)
    const int actual_n_seqs    = seq_lens_tensor ? (int) seq_lens_tensor->ne[0] : 1;
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
            fprintf(stderr, "[SYCL] V2 skipped: prefill mode (n_seqs=%d, n_query=%d) not supported\n", actual_n_seqs,
                    num_query_tokens);
            fprintf(stderr, "[SYCL]   V2 requires continuous batching (n_seqs==n_query) or single-token decode\n");
            v2_fallback_warned = true;
        }
        use_v2_dispatch = false;
    }

    if (use_v2_dispatch) {
        const int num_seqs     = actual_n_seqs;  // Use actual sequence count, not query token count
        const int num_heads    = Q->ne[2];       // Number of query heads
        const int num_kv_heads = K->ne[2];       // Number of KV heads

        // For auto-V2: block_size = 1 (each "block" is one token)
        // For paged V2: use actual block_size from params
        const int block_size         = use_auto_v2 ? 1 : params.block_size;
        const int max_blocks_per_seq = use_auto_v2 ? max_context_len : params.max_blocks_per_seq;

        // Get pointers to block_table and seq_lens
        const int32_t * v2_block_table = nullptr;
        const int32_t * v2_seq_lens    = nullptr;

        if (use_auto_v2) {
            // Auto-V2: Generate identity block table and seq_lens on device
            // block_table[seq][block] = block (identity mapping)
            // seq_lens[seq] = max_context_len (all seqs have same context)
            const size_t block_table_size = num_seqs * max_blocks_per_seq;
            const size_t seq_lens_size    = num_seqs;

            // Reallocate device buffers if needed
            // NOTE: Skip reallocation during SYCL graph recording (malloc/free forbidden)
            // This assumes buffers were allocated in prior decode steps before graph recording
            bool needs_realloc_block =
                !g_ggml_sycl_graph_recording &&
                (g_v2_auto.alloc_queue != ctx.stream() || g_v2_auto.block_table_capacity < block_table_size);
            bool needs_realloc_seq = !g_ggml_sycl_graph_recording && (g_v2_auto.alloc_queue != ctx.stream() ||
                                                                      g_v2_auto.seq_lens_capacity < seq_lens_size);

            if (needs_realloc_block) {
                if (g_v2_auto.block_table && g_v2_auto.alloc_queue) {
                    ggml_sycl::unified_cache_sub_runtime_bytes(
                        ggml_sycl_get_device_id_from_queue(*g_v2_auto.alloc_queue),
                        g_v2_auto.block_table_capacity * sizeof(int32_t));
                    sycl::free(g_v2_auto.block_table, *g_v2_auto.alloc_queue);
                }
                ggml_sycl::unified_cache_add_runtime_bytes(ggml_sycl_get_device_id_from_queue(*ctx.stream()),
                                                           block_table_size * sizeof(int32_t));
                g_v2_auto.block_table          = ggml_sycl_malloc_device_t<int32_t>(block_table_size, *ctx.stream(), "fattn_block_table");
                if (!g_v2_auto.block_table) {
                    ggml_sycl::unified_cache_sub_runtime_bytes(ggml_sycl_get_device_id_from_queue(*ctx.stream()),
                                                               block_table_size * sizeof(int32_t));
                }
                g_v2_auto.block_table_capacity = block_table_size;
            }
            if (needs_realloc_seq) {
                if (g_v2_auto.seq_lens && g_v2_auto.alloc_queue) {
                    ggml_sycl::unified_cache_sub_runtime_bytes(
                        ggml_sycl_get_device_id_from_queue(*g_v2_auto.alloc_queue),
                        g_v2_auto.seq_lens_capacity * sizeof(int32_t));
                    sycl::free(g_v2_auto.seq_lens, *g_v2_auto.alloc_queue);
                }
                ggml_sycl::unified_cache_add_runtime_bytes(ggml_sycl_get_device_id_from_queue(*ctx.stream()),
                                                           seq_lens_size * sizeof(int32_t));
                g_v2_auto.seq_lens          = ggml_sycl_malloc_device_t<int32_t>(seq_lens_size, *ctx.stream(), "fattn_seq_lens");
                if (!g_v2_auto.seq_lens) {
                    ggml_sycl::unified_cache_sub_runtime_bytes(ggml_sycl_get_device_id_from_queue(*ctx.stream()),
                                                               seq_lens_size * sizeof(int32_t));
                }
                g_v2_auto.seq_lens_capacity = seq_lens_size;
            }
            if (needs_realloc_block || needs_realloc_seq) {
                g_v2_auto.alloc_queue = ctx.stream();
            }

            // Fill identity block table: block_table[i] = i
            // Use a kernel for efficiency (parallel fill)
            ctx.stream()->parallel_for<class fattn_v2_fill_block_table_runtime_kernel>(
                sycl::range<1>(block_table_size), [block_table_ptr = g_v2_auto.block_table](sycl::id<1> idx) {
                    block_table_ptr[idx] = static_cast<int32_t>(idx[0]);
                });

            // Fill seq_lens: all sequences have the same context length
            ctx.stream()->parallel_for<class fattn_v2_fill_seq_lens_kernel>(
                sycl::range<1>(seq_lens_size), [seq_lens_ptr = g_v2_auto.seq_lens, ctx_len = max_context_len](
                                                   sycl::id<1> idx) { seq_lens_ptr[idx] = ctx_len; });

            // Note: In-order queue ensures parallel_for fills complete before V2 kernel

            v2_block_table = g_v2_auto.block_table;
            v2_seq_lens    = g_v2_auto.seq_lens;
        } else {
            // Paged V2: Use provided block_table and seq_lens
            v2_block_table = params.block_table;
            v2_seq_lens    = params.seq_lens;
        }

        // Get temporary buffer sizes
        const size_t temp_size = paged_attention_v2_temp_size(num_seqs, num_heads, max_context_len, D);

        // Use persistent temp buffer (avoids malloc/free during SYCL graph recording)
        // Reallocate if needed (buffer grows but never shrinks)
        // NOTE: Skip reallocation during SYCL graph recording (malloc/free forbidden)
        bool needs_realloc_temp = !g_ggml_sycl_graph_recording &&
                                  (g_v2_auto.alloc_queue != ctx.stream() || g_v2_auto.temp_buf_capacity < temp_size);
        if (needs_realloc_temp) {
            if (g_v2_auto.temp_buf && g_v2_auto.alloc_queue) {
                ggml_sycl::unified_cache_sub_runtime_bytes(
                    ggml_sycl_get_device_id_from_queue(*g_v2_auto.alloc_queue),
                    g_v2_auto.temp_buf_capacity);
                sycl::free(g_v2_auto.temp_buf, *g_v2_auto.alloc_queue);
            }
            ggml_sycl::unified_cache_add_runtime_bytes(ggml_sycl_get_device_id_from_queue(*ctx.stream()), temp_size);
            g_v2_auto.temp_buf          = ggml_sycl_malloc_device_t<uint8_t>(temp_size, *ctx.stream(), "fattn_temp");
            if (!g_v2_auto.temp_buf) {
                ggml_sycl::unified_cache_sub_runtime_bytes(ggml_sycl_get_device_id_from_queue(*ctx.stream()),
                                                           temp_size);
            }
            g_v2_auto.temp_buf_capacity = temp_size;
            g_v2_auto.alloc_queue       = ctx.stream();
        }

        // If we're graph recording and buffers don't exist, we can't proceed
        // This shouldn't happen if the first decode step ran before graph recording
        if (!g_v2_auto.temp_buf) {
            // Fall back to standard dispatch path
            static bool v2_graph_warning = false;
            if (!v2_graph_warning) {
                fprintf(stderr,
                        "[SYCL] WARNING: V2 temp buffer not allocated during graph recording, "
                        "falling back to standard FA\n");
                v2_graph_warning = true;
            }
            use_v2_dispatch = false;
        }

        // If V2 dispatch was disabled (e.g., no temp buffer during graph recording),
        // we'll fall through to standard dispatch after the #endif

        if (use_v2_dispatch) {
            // Use device pointers from params (obtained via ggml_sycl_get_data_ptr)
            float *            out   = params.dst;
            const sycl::half * Q_dev = (const sycl::half *) params.Q;
            const char *       K_dev = params.K;
            const char *       V_dev = params.V;

#    if SYCL_V2_ESIMD_AVAILABLE
            // ESIMD V2 kernel: optimized single-kernel version with SIMD vectorization
            // Preferred over scalar V2 when ESIMD is available (25-45% faster)
            // Uses element-index addressing (same as working ESIMD kernel)
            if (g_sycl_fa_esimd_enabled && v2_esimd_available()) {
                GGML_SYCL_DEBUG(
                    "[SYCL] V2 ESIMD dispatch: num_seqs=%d num_heads=%d num_kv_heads=%d D=%d "
                    "max_context=%d block_size=%d auto_v2=%d\n",
                    num_seqs, num_heads, num_kv_heads, D, max_context_len, block_size, use_auto_v2 ? 1 : 0);

                static bool v2_esimd_info_shown = false;
                if (!v2_esimd_info_shown) {
                    fprintf(stderr, "[SYCL] Using ESIMD-optimized V2 attention kernel\n");
                    v2_esimd_info_shown = true;
                }

                // Dispatch ESIMD V2 kernel based on head dimension and Q type
                // Q can be F32 (decode mode) or F16 (some models) - need correct template parameter
                // New signature uses params struct for proper element-index addressing
                if (Q->type == GGML_TYPE_F32) {
                    switch (D) {
                        case 64:
                            launch_v2_esimd_attention<64, float>(params, v2_block_table, v2_seq_lens, num_seqs,
                                                                 max_blocks_per_seq, max_context_len, block_size,
                                                                 ctx.stream());
                            break;
                        case 128:
                            launch_v2_esimd_attention<128, float>(params, v2_block_table, v2_seq_lens, num_seqs,
                                                                  max_blocks_per_seq, max_context_len, block_size,
                                                                  ctx.stream());
                            break;
                        default:
                            // D=256 not yet implemented in ESIMD V2, fall through to scalar V2
                            goto scalar_v2_dispatch;
                    }
                } else {
                    // Q is F16
                    switch (D) {
                        case 64:
                            launch_v2_esimd_attention<64, sycl::half>(params, v2_block_table, v2_seq_lens, num_seqs,
                                                                      max_blocks_per_seq, max_context_len, block_size,
                                                                      ctx.stream());
                            break;
                        case 128:
                            launch_v2_esimd_attention<128, sycl::half>(params, v2_block_table, v2_seq_lens, num_seqs,
                                                                       max_blocks_per_seq, max_context_len, block_size,
                                                                       ctx.stream());
                            break;
                        default:
                            // D=256 not yet implemented in ESIMD V2, fall through to scalar V2
                            goto scalar_v2_dispatch;
                    }
                }

                return;  // ESIMD V2 dispatch complete
            }
scalar_v2_dispatch:
#    endif  // SYCL_V2_ESIMD_AVAILABLE

            // Scalar V2 kernel: fallback when ESIMD is not available
            void * temp_buf = g_v2_auto.temp_buf;

            const int    max_num_partitions = (max_context_len + V2_PARTITION_SIZE - 1) / V2_PARTITION_SIZE;
            const size_t exp_sums_size      = num_seqs * num_heads * max_num_partitions * sizeof(float);

            float * exp_sums   = (float *) temp_buf;
            float * max_logits = exp_sums + num_seqs * num_heads * max_num_partitions;
            float * tmp_out    = max_logits + num_seqs * num_heads * max_num_partitions;

            GGML_SYCL_DEBUG(
                "[SYCL] V2 scalar dispatch: num_seqs=%d num_heads=%d num_kv_heads=%d D=%d "
                "max_context=%d partitions=%d block_size=%d auto_v2=%d\n",
                num_seqs, num_heads, num_kv_heads, D, max_context_len, max_num_partitions, block_size,
                use_auto_v2 ? 1 : 0);

            // Dispatch scalar V2 kernel based on head dimension
            switch (D) {
                case 64:
                    launch_paged_attention_v2<64>(out, exp_sums, max_logits, tmp_out, Q_dev, K_dev, V_dev, params.scale,
                                                  v2_block_table, v2_seq_lens, num_seqs, num_heads, num_kv_heads,
                                                  max_blocks_per_seq, max_context_len, block_size, params.nb11,
                                                  params.nb12, params.nb21, params.nb22, ctx.stream());
                    break;
                case 128:
                    launch_paged_attention_v2<128>(out, exp_sums, max_logits, tmp_out, Q_dev, K_dev, V_dev,
                                                   params.scale, v2_block_table, v2_seq_lens, num_seqs, num_heads,
                                                   num_kv_heads, max_blocks_per_seq, max_context_len, block_size,
                                                   params.nb11, params.nb12, params.nb21, params.nb22, ctx.stream());
                    break;
                case 256:
                    launch_paged_attention_v2<256>(out, exp_sums, max_logits, tmp_out, Q_dev, K_dev, V_dev,
                                                   params.scale, v2_block_table, v2_seq_lens, num_seqs, num_heads,
                                                   num_kv_heads, max_blocks_per_seq, max_context_len, block_size,
                                                   params.nb11, params.nb12, params.nb21, params.nb22, ctx.stream());
                    break;
                default:
                    GGML_ABORT("Unsupported head dimension for V2 attention: %d", D);
            }

            // Note: No wait() or free() here - temp buffer is persistent (g_v2_auto.temp_buf)
            // This allows V2 dispatch to work with SYCL command graphs

            return;  // V2 dispatch complete
        }  // if (use_v2_dispatch) - inner check after potential fallback
    }  // if (use_v2_dispatch) - main V2 execution block (line 893)
#endif  // GGML_SYCL_FA_V2_ENABLED

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

    // Buffer aliasing debug: trace flash attention output pointer and values
    // This helps identify if FA writes to expected buffer vs what MUL_MAT reads
    // NOTE: Only enabled when graphs are disabled (wait() is incompatible with graph recording)
    {
        static bool fa_buf_debug    = getenv("GGML_SYCL_BUFFER_ALIAS_DEBUG") != nullptr;
        static bool graphs_disabled = getenv("GGML_SYCL_DISABLE_GRAPH") != nullptr;
        if (fa_buf_debug && graphs_disabled) {
            static int fa_trace_count = 0;
            fa_trace_count++;
            if (fa_trace_count <= 10) {
                auto * stream = ctx.stream();
                stream->wait();  // Ensure kernel completes

                // Read first 4 floats from dst
                float  host_vals[4]   = { 0 };
                size_t total_elements = params.ne01 * params.ne02 * D;  // n_queries * n_heads * head_dim
                if (total_elements >= 4) {
                    stream->memcpy(host_vals, params.dst, 4 * sizeof(float)).wait();
                }

                bool is_zeros =
                    (host_vals[0] == 0.0f && host_vals[1] == 0.0f && host_vals[2] == 0.0f && host_vals[3] == 0.0f);

                fprintf(stderr,
                        "[FA_OUTPUT] %s dst=%p first4=[%.4f, %.4f, %.4f, %.4f] is_zeros=%d "
                        "ne01=%lld ne02=%lld D=%d\n",
                        dst->name, (void *) params.dst, host_vals[0], host_vals[1], host_vals[2], host_vals[3],
                        is_zeros ? 1 : 0, (long long) params.ne01, (long long) params.ne02, D);
            }
        }
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
            auto & dbg     = get_fattn_debug_ctx();
            dbg.call_id    = fa_call_count;
            dbg.n_queries  = params.ne01;
            dbg.n_heads    = params.ne02;
            dbg.n_kv_heads = params.ne12;
            dbg.n_kv       = params.ne11;
            dbg.D          = D;
            dbg.scale      = params.scale;
            dbg.is_fa_on   = true;

            dbg.open_file("on");

            int Q_type_size = (Q->type == GGML_TYPE_F32) ? sizeof(float) : sizeof(sycl::half);
            fattn_debug_dump_Q(stream, params.Q, Q_type_size, D, params.ne01, params.ne02, params.nb01, params.nb02,
                               params.scale);
            fattn_debug_dump_K(stream, params.K, D, params.ne11, params.ne12, params.nb11, params.nb12);
            fattn_debug_dump_V(stream, params.V, D, params.ne11, params.ne12, params.nb21, params.nb22);
            fattn_debug_dump_mask(stream, params.mask, params.ne30, params.ne01, params.nb31, params.ne30);
            fattn_debug_dump_output(stream, params.dst, D, params.ne01, params.ne02);

            dbg.close_file();

            fprintf(stderr, "[FA-DEBUG] FA_ON call %d: ne01=%d, ne02=%d, ne12=%d, D=%d, n_kv=%d, scale=%.4f\n",
                    fa_call_count, params.ne01, params.ne02, params.ne12, D, params.ne11, params.scale);
        }
    }
}
