//
// MIT license
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Persistent Token Generation Kernel
//
// This header defines the argument structures and function declarations for
// the persistent TG kernel. The persistent kernel aims to reduce kernel launch
// overhead by keeping a single kernel running through the entire forward pass
// with work-stealing synchronization.
//
// Design principles:
// - Single kernel launch per forward pass (reduces dispatch overhead)
// - Work-stealing for dynamic load balancing across compute units
// - Optimized for small batch sizes (M=1) typical in token generation
// - Pre-fetched weights and KV cache for memory-bound optimization
//

#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

// Forward declarations to avoid circular includes
// Include "unified-kernel.hpp" for full XMXConfig definition
struct XMXConfig;

namespace ggml_sycl {

// =============================================================================
// Layer Weight Pointers
// =============================================================================

/**
 * Pointers to all weights for a single transformer layer.
 *
 * Each pointer points to device memory containing the quantized weights.
 * The layout and quantization format are specified in PersistentTGArgs.
 */
struct LayerWeights {
    // Attention normalization
    const void * attn_norm;  // RMS norm weights [hidden_dim]

    // Attention projections (Q, K, V, O)
    const void * q_proj;     // Query projection [n_heads * head_dim, hidden_dim]
    const void * k_proj;     // Key projection [n_heads * head_dim, hidden_dim]
    const void * v_proj;     // Value projection [n_heads * head_dim, hidden_dim]
    const void * o_proj;     // Output projection [hidden_dim, n_heads * head_dim]

    // FFN normalization
    const void * ffn_norm;   // RMS norm weights [hidden_dim]

    // FFN projections (gate, up, down)
    const void * gate_proj;  // Gate projection [intermediate_dim, hidden_dim]
    const void * up_proj;    // Up projection [intermediate_dim, hidden_dim]
    const void * down_proj;  // Down projection [hidden_dim, intermediate_dim]
};

// =============================================================================
// KV Cache Structure
// =============================================================================

/**
 * KV cache pointers and metadata for a single layer.
 *
 * The cache stores past key/value states for autoregressive generation.
 * Layout: [n_heads, seq_len, head_dim] for both K and V.
 */
struct KVCache {
    void * k_cache;   // Key cache (device pointer, FP16)
    void * v_cache;   // Value cache (device pointer, FP16)
    int    seq_len;   // Current sequence length (number of cached tokens)
};

// =============================================================================
// Persistent TG Kernel Arguments
// =============================================================================

/**
 * Arguments for the persistent token generation kernel.
 *
 * Contains all data needed to execute a complete forward pass:
 * - Model architecture dimensions
 * - Per-layer weight pointers
 * - Per-layer KV cache pointers
 * - Input/output buffers
 * - Work distribution state
 */
struct PersistentTGArgs {
    // =========================================================================
    // Model Architecture Dimensions
    // =========================================================================

    int n_layers;          // Number of transformer layers
    int hidden_dim;        // Model hidden dimension (e.g., 4096)
    int n_heads;           // Number of attention heads
    int head_dim;          // Per-head dimension (hidden_dim / n_heads)
    int intermediate_dim;  // FFN intermediate dimension (e.g., 11008 for LLaMA)
    int vocab_size;        // Vocabulary size for output logits

    // =========================================================================
    // Quantization Configuration
    // =========================================================================

    int quant_type;        // GGML_TYPE_* enum value (e.g., GGML_TYPE_Q4_0)
    int layout_mode;       // Memory layout: 0=AOS, 1=SOA, 2=COALESCED

    // =========================================================================
    // Per-Layer Data (arrays of size n_layers)
    // =========================================================================

    const LayerWeights * layer_weights;  // Array of layer weight pointers [n_layers]
    KVCache *            kv_caches;      // Array of KV cache pointers [n_layers]

    // =========================================================================
    // Input/Output Buffers
    // =========================================================================

    const void * input_embedding;   // Input token embedding (device pointer, FP16)
    float *      output_logits;     // Output logits (device pointer, F32) [vocab_size]

    // =========================================================================
    // Work Distribution State
    // =========================================================================

    void * intermediate_buffer;  // Scratch buffer for intermediate activations
    int *  work_counter;         // Atomic counter for work-stealing (device pointer)
    int    total_tiles;          // Total number of work tiles to process
};

// =============================================================================
// Persistent TG Kernel Configuration
// =============================================================================

/**
 * Configuration for persistent kernel execution.
 *
 * Specifies tile dimensions, work distribution, and synchronization options.
 */
struct PersistentTGConfig {
    // =========================================================================
    // Tile Dimensions
    // =========================================================================

    int tile_m;  // M dimension tile size (typically 1 for TG)
    int tile_n;  // N dimension tile size (output columns per tile)
    int tile_k;  // K dimension tile size (reduction per iteration)

    // =========================================================================
    // Work Distribution
    // =========================================================================

    int n_workgroups;    // Number of persistent work-groups to launch
    int workgroup_size;  // Work-items per work-group (typically 256 or 512)

    // =========================================================================
    // Synchronization Options
    // =========================================================================

    bool use_split_barriers;  // Use split barriers for cooperative synchronization
};

// =============================================================================
// Function Declarations
// =============================================================================

/**
 * Launch the persistent token generation kernel.
 *
 * Executes a complete forward pass for a single token using a persistent
 * kernel that stays active across all layers. Work-stealing is used for
 * dynamic load balancing.
 *
 * @param q      SYCL queue for submission
 * @param args   Kernel arguments (model config, weights, KV cache, buffers)
 * @param config Kernel configuration (tiles, workgroups, sync options)
 * @return SYCL event for synchronization
 *
 * Prerequisites:
 * - All weight pointers must be valid device memory
 * - KV cache must be allocated with sufficient capacity
 * - work_counter must be initialized to 0
 * - intermediate_buffer must have sufficient size
 *
 * Thread safety: Not thread-safe. Caller must ensure exclusive queue access.
 */
sycl::event launch_persistent_tg_kernel(sycl::queue &             q,
                                        const PersistentTGArgs &  args,
                                        const PersistentTGConfig & config);

/**
 * Check if persistent TG kernel can be used for the given model configuration.
 *
 * The persistent kernel has hardware and configuration requirements:
 * - XMX support (for efficient small-batch matmul)
 * - Supported quantization types (Q4_0, Q4_K, Q6_K)
 * - Minimum hidden dimension for efficient tiling
 *
 * @param n_layers    Number of transformer layers
 * @param hidden_dim  Model hidden dimension
 * @param quant_type  GGML_TYPE_* quantization format
 * @param xmx_config  Hardware XMX configuration
 * @return true if persistent kernel can be used
 *
 * Performance: O(1) time, no memory allocation.
 */
bool can_use_persistent_tg(int                                    n_layers,
                           int                                    hidden_dim,
                           int                                    quant_type,
                           const XMXConfig & xmx_config);

}  // namespace ggml_sycl
