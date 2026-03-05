#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// UMA (Unified Memory Architecture) bandwidth-aware profiler.
//
// Inspired by APEX (arXiv:2506.03296), this profiler measures per-op execution
// times during inference to classify operations as bandwidth-bound vs compute-bound.
// On UMA systems (iGPU + shared memory), this data drives smarter layer splitting:
//   - Bandwidth-bound ops (FFN matmuls) benefit from GPU's ~2x higher bandwidth
//   - Compute-bound ops (attention) can tolerate CPU execution with minimal penalty
//
// Usage:
//   1. Create uma_profiler_data instance
//   2. Set params.cb_eval = uma_profiler_cb_eval, params.cb_eval_user_data = &data
//   3. Run a few inference iterations
//   4. Call uma_profiler_report() to get analysis and recommendations

struct uma_op_stats {
    int64_t  total_us    = 0;   // total time in microseconds
    int64_t  total_bytes = 0;   // total bytes read (weights + inputs)
    int64_t  total_flops = 0;   // total floating point operations
    uint32_t count       = 0;   // number of invocations
    int      layer       = -1;  // layer index (-1 for non-layer ops)

    // derived metrics (computed in report)
    double   avg_us          = 0.0;
    double   bandwidth_gbps  = 0.0; // effective bandwidth in GB/s
    double   compute_gflops  = 0.0; // effective compute in GFLOPS
    double   arithmetic_intensity = 0.0; // FLOPS/byte (roofline model)
};

enum uma_op_class {
    UMA_OP_CLASS_UNKNOWN   = 0,
    UMA_OP_CLASS_BANDWIDTH = 1, // bandwidth-bound: low arithmetic intensity
    UMA_OP_CLASS_COMPUTE   = 2, // compute-bound: high arithmetic intensity
};

struct uma_layer_analysis {
    int      layer = -1;
    int64_t  attn_us = 0;       // total attention time (us)
    int64_t  ffn_us  = 0;       // total FFN time (us)
    int64_t  attn_bytes = 0;    // attention weight bytes
    int64_t  ffn_bytes  = 0;    // FFN weight bytes
    double   attn_ai = 0.0;     // attention arithmetic intensity
    double   ffn_ai  = 0.0;     // FFN arithmetic intensity
    uma_op_class attn_class = UMA_OP_CLASS_UNKNOWN;
    uma_op_class ffn_class  = UMA_OP_CLASS_UNKNOWN;
};

struct uma_profiler_data {
    // per-op timing keyed by "op_name:layer_index"
    std::unordered_map<std::string, uma_op_stats> op_stats;

    // profiling state
    int64_t  last_op_start_us = 0;
    bool     profiling_active = true;
    uint32_t n_iterations     = 0;    // number of full forward passes profiled
    uint32_t max_iterations   = 5;    // stop profiling after this many iterations

    // roofline model parameters (auto-detected or user-supplied)
    double   gpu_bandwidth_gbps = 0.0;  // GPU effective memory bandwidth (GB/s)
    double   cpu_bandwidth_gbps = 0.0;  // CPU effective memory bandwidth (GB/s)
    double   gpu_compute_gflops = 0.0;  // GPU peak compute (GFLOPS)
    double   cpu_compute_gflops = 0.0;  // CPU peak compute (GFLOPS)
};

// Eval callback for the backend scheduler. Measures per-op timing.
bool uma_profiler_cb_eval(struct ggml_tensor * t, bool ask, void * user_data);

// Generate profiling report with per-layer analysis and overflow recommendations.
// Returns a human-readable report string.
std::string uma_profiler_report(uma_profiler_data & data);

// Get per-layer analysis sorted by layer index.
std::vector<uma_layer_analysis> uma_profiler_analyze_layers(const uma_profiler_data & data);

// Signal that a full forward pass has completed (call after each inference iteration).
void uma_profiler_iteration_done(uma_profiler_data & data);
