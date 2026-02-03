//
// MIT license
// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Persistent Token Generation Kernel Implementation
//
// This file implements the persistent TG kernel with work-stealing for
// reduced kernel launch overhead during autoregressive token generation.
//
// Design:
// - Single kernel processes all transformer operations
// - Work-stealing loop with atomic counter for load balancing
// - Operations decoded from linear work index to (layer, op, tile)
//
// Current Status: Skeleton implementation with stub operation dispatch.
// Future work will integrate actual RMS norm, matmul, attention kernels.
//

#include "persistent-tg-kernel.hpp"
#include "ggml.h"  // For GGML_TYPE_* constants

#include <algorithm>

namespace ggml_sycl {

// =============================================================================
// Operation Constants
// =============================================================================
// Each transformer layer has 8 operations in the following order:
//
// Attention block:
//   0: ATTN_NORM    - RMS normalization before attention
//   1: QKV_PROJ     - Fused Q/K/V projection (3 matmuls)
//   2: ATTN         - Scaled dot-product attention
//   3: OUT_PROJ     - Output projection
//
// FFN block:
//   4: FFN_NORM     - RMS normalization before FFN
//   5: GATE_UP      - Fused gate/up projection (2 matmuls)
//   6: SILU         - SiLU activation (element-wise)
//   7: DOWN         - Down projection

constexpr int OP_ATTN_NORM  = 0;
constexpr int OP_QKV_PROJ   = 1;
constexpr int OP_ATTN       = 2;
constexpr int OP_OUT_PROJ   = 3;
constexpr int OP_FFN_NORM   = 4;
constexpr int OP_GATE_UP    = 5;
constexpr int OP_SILU       = 6;
constexpr int OP_DOWN       = 7;
constexpr int OPS_PER_LAYER = 8;

// =============================================================================
// Work Item Encoding
// =============================================================================
// Work items are encoded as a single integer: work_idx = layer * OPS_PER_LAYER + op
// Tile subdivision happens within each operation based on output dimensions.

struct WorkItem {
    int layer;  // Transformer layer index [0, n_layers)
    int op;     // Operation index [0, OPS_PER_LAYER)
    int tile;   // Tile index within operation
};

// Decode work index to layer, operation, tile
// For now, work_idx directly maps to (layer, op), with tile=0 (single tile per op)
inline WorkItem decode_work_item(int work_idx, int n_layers) {
    WorkItem item;
    // Simple encoding: sequential ops across layers
    // work_idx = layer * OPS_PER_LAYER + op
    item.layer = work_idx / OPS_PER_LAYER;
    item.op    = work_idx % OPS_PER_LAYER;
    item.tile  = 0;  // Tile subdivision not yet implemented

    // Clamp to valid range
    if (item.layer >= n_layers) {
        item.layer = n_layers - 1;
        item.op    = OPS_PER_LAYER - 1;
    }
    return item;
}

// =============================================================================
// PersistentDMMVKernel Class Template
// =============================================================================
// The kernel class that runs on each work-item. Uses work-stealing to
// dynamically claim work tiles from a global atomic counter.

template <int WORKGROUP_SIZE> class PersistentDMMVKernel {
  public:
    // Constructor
    // @param args_      Kernel arguments (weights, KV cache, dimensions)
    // @param config_    Kernel configuration (tile sizes, sync options)
    // @param slm_       Local memory accessor for shared storage
    // @param item_      SYCL nd_item for thread identification
    PersistentDMMVKernel(const PersistentTGArgs args_, const PersistentTGConfig config_,
                         sycl::local_accessor<float, 1> slm_, sycl::nd_item<1> item_) :
        args(args_),
        config(config_),
        slm(slm_),
        item(item_) {}

    // Main work-stealing loop
    void run() {
        // Get thread identifiers
        const int local_id = static_cast<int>(item.get_local_id(0));
        // Note: group_id will be used when we implement tile-aware work distribution
        // const int group_id  = static_cast<int>(item.get_group(0));

        // Work-stealing loop: keep processing until all work is done
        while (true) {
            int work_idx = -1;

            // Thread 0 atomically fetches next work index
            if (local_id == 0) {
                // Create atomic reference to work counter
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>
                    counter_ref(*args.work_counter);

                // Atomically increment and get previous value
                work_idx = counter_ref.fetch_add(1);
            }

            // Broadcast work index to all threads in workgroup
            work_idx = sycl::group_broadcast(item.get_group(), work_idx, 0);

            // Exit if all work is done
            if (work_idx >= args.total_tiles) {
                break;
            }

            // Decode work item
            WorkItem wi = decode_work_item(work_idx, args.n_layers);

            // Dispatch to appropriate operation
            dispatch_operation(wi);

            // Barrier to ensure all threads complete before next iteration
            // Split barriers allow non-blocking arrival + deferred wait for better
            // latency hiding when computation can overlap with synchronization.
            if (config.use_split_barriers) {
                split_barrier_arrive();
                // Future optimization: do useful work here while waiting
                split_barrier_wait();
            } else {
                sycl::group_barrier(item.get_group());
            }
        }
    }

  private:
    const PersistentTGArgs         args;
    const PersistentTGConfig       config;
    sycl::local_accessor<float, 1> slm;
    sycl::nd_item<1>               item;

    // Dispatch to the appropriate operation based on work item
    void dispatch_operation(const WorkItem & wi) {
        // Get layer data pointers
        // const LayerWeights & weights = args.layer_weights[wi.layer];
        // KVCache &            kv      = args.kv_caches[wi.layer];

        switch (wi.op) {
            case OP_ATTN_NORM:
                dispatch_rms_norm(wi);
                break;
            case OP_QKV_PROJ:
                dispatch_qkv_projection(wi);
                break;
            case OP_ATTN:
                dispatch_attention(wi);
                break;
            case OP_OUT_PROJ:
                dispatch_output_projection(wi);
                break;
            case OP_FFN_NORM:
                dispatch_ffn_norm(wi);
                break;
            case OP_GATE_UP:
                dispatch_gate_up(wi);
                break;
            case OP_SILU:
                dispatch_silu(wi);
                break;
            case OP_DOWN:
                dispatch_down_projection(wi);
                break;
            default:
                // Unknown operation - should never happen
                break;
        }
    }

    // ==========================================================================
    // Stub Operation Implementations
    // ==========================================================================
    // These are placeholders that will be filled in with actual kernel logic.

    void dispatch_rms_norm(const WorkItem & wi) {
        (void) wi;
        // TODO: Implement RMS normalization
        // - Read hidden_dim values from intermediate buffer
        // - Compute RMS: sqrt(mean(x^2) + eps)
        // - Normalize: x / rms * weight
        // For now, no-op (placeholder)
    }

    void dispatch_qkv_projection(const WorkItem & wi) {
        (void) wi;
        // TODO: Implement fused Q/K/V projection
        // - Load normalized hidden state
        // - Multiply by Q, K, V weight matrices
        // - Store to intermediate buffer
        // For now, no-op (placeholder)
    }

    void dispatch_attention(const WorkItem & wi) {
        (void) wi;
        // TODO: Implement scaled dot-product attention
        // - Compute attention scores: Q @ K^T / sqrt(head_dim)
        // - Apply softmax
        // - Multiply by V
        // - Update KV cache
        // For now, no-op (placeholder)
    }

    void dispatch_output_projection(const WorkItem & wi) {
        (void) wi;
        // TODO: Implement output projection
        // - Load attention output
        // - Multiply by output projection weights
        // - Add residual connection
        // For now, no-op (placeholder)
    }

    void dispatch_ffn_norm(const WorkItem & wi) {
        (void) wi;
        // TODO: Implement FFN RMS normalization
        // Similar to dispatch_rms_norm but for FFN block
        // For now, no-op (placeholder)
    }

    void dispatch_gate_up(const WorkItem & wi) {
        (void) wi;
        // TODO: Implement fused gate/up projection
        // - Load normalized hidden state
        // - Compute gate = W_gate @ x
        // - Compute up = W_up @ x
        // For now, no-op (placeholder)
    }

    void dispatch_silu(const WorkItem & wi) {
        (void) wi;
        // TODO: Implement SiLU activation
        // - silu(x) = x * sigmoid(x)
        // - output = silu(gate) * up
        // For now, no-op (placeholder)
    }

    void dispatch_down_projection(const WorkItem & wi) {
        (void) wi;
        // TODO: Implement down projection
        // - Load activated FFN intermediate
        // - Multiply by down projection weights
        // - Add residual connection
        // For now, no-op (placeholder)
    }
};

// =============================================================================
// launch_persistent_tg_kernel Implementation
// =============================================================================

sycl::event launch_persistent_tg_kernel(sycl::queue &              q,
                                        const PersistentTGArgs &   args,
                                        const PersistentTGConfig & config) {
    // Calculate SLM (Shared Local Memory) size based on config
    // Reserve space for intermediate results during tile computation
    const size_t slm_floats = config.tile_n * config.tile_k;

    // Create nd_range based on config
    sycl::range<1>    global_range(config.n_workgroups * config.workgroup_size);
    sycl::range<1>    local_range(config.workgroup_size);
    sycl::nd_range<1> nd_range(global_range, local_range);

    // Submit the persistent kernel
    sycl::event e = q.submit([&](sycl::handler & cgh) {
        // Allocate local memory for the workgroup
        sycl::local_accessor<float, 1> slm(slm_floats, cgh);

        // Lambda captures args and config by value to ensure kernel safety
        auto args_copy   = args;
        auto config_copy = config;

        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            // Instantiate kernel based on workgroup size
            // Using fixed sizes for common configurations
            if (config_copy.workgroup_size == 256) {
                PersistentDMMVKernel<256> kernel(args_copy, config_copy, slm, item);
                kernel.run();
            } else if (config_copy.workgroup_size == 512) {
                PersistentDMMVKernel<512> kernel(args_copy, config_copy, slm, item);
                kernel.run();
            } else if (config_copy.workgroup_size == 128) {
                PersistentDMMVKernel<128> kernel(args_copy, config_copy, slm, item);
                kernel.run();
            } else {
                // Fallback to 256 for unsupported sizes
                PersistentDMMVKernel<256> kernel(args_copy, config_copy, slm, item);
                kernel.run();
            }
        });
    });

    return e;
}

// =============================================================================
// can_use_persistent_tg Implementation
// =============================================================================

bool can_use_persistent_tg(int n_layers, int hidden_dim, int quant_type,
                           const ggml_sycl_unified::XMXConfig & xmx_config) {
    const auto & cfg = xmx_config;

    // Check basic hardware support
    if (!cfg.supported) {
        return false;
    }

    // Check layer count is reasonable (0 < n_layers <= 128)
    if (n_layers <= 0 || n_layers > 128) {
        return false;
    }

    // Check hidden dimension is within supported range
    // Minimum: need enough parallelism for efficient GPU utilization
    // Maximum: limited by SLM and register pressure
    constexpr int MIN_HIDDEN_DIM = 512;
    constexpr int MAX_HIDDEN_DIM = 16384;
    if (hidden_dim < MIN_HIDDEN_DIM || hidden_dim > MAX_HIDDEN_DIM) {
        return false;
    }

    // Check quantization type is supported
    // Currently support Q4_0, Q4_K, Q6_K as per spec
    switch (quant_type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
            break;
        default:
            return false;
    }

    // Check hardware has sufficient SLM
    // Need at least 32KB for efficient tiling
    constexpr size_t MIN_SLM_SIZE = 32 * 1024;
    if (cfg.slm_size < MIN_SLM_SIZE) {
        return false;
    }

    // Check hardware has XMX FP16 support (required for efficient matmul)
    if (!cfg.supports_fp16) {
        return false;
    }

    return true;
}

}  // namespace ggml_sycl
