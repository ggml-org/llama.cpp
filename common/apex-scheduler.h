#pragma once

#include <cstdint>
#include <string>

// APEX-inspired scheduling decisions for hybrid CPU-GPU inference.
// Reference: APEX (arXiv:2506.03296) — Asynchronous Parallel CPU-GPU Execution
//
// On UMA systems (unified memory), decides whether offloading attention ops
// to CPU improves throughput, based on profiled execution times.

enum apex_strategy_t {
    APEX_STRATEGY_GPU_ONLY          = 0, // all ops on GPU (default)
    APEX_STRATEGY_ASYNC_OVERLAP     = 1, // CPU attention overlaps GPU FFN
    APEX_STRATEGY_ASYMMETRIC_PIPE   = 2, // asymmetric pipelining (batch splitting)
};

struct apex_decision {
    bool              cpu_offload_profitable = false;
    apex_strategy_t   strategy = APEX_STRATEGY_GPU_ONLY;

    double gpu_throughput_est;     // estimated tokens/sec GPU-only
    double hybrid_throughput_est;  // estimated tokens/sec with CPU offload

    // inequality parameters
    double ratio;                  // N_G / N_C (GPU vs CPU speed ratio)
    double threshold;              // 2*(T_glinear/T_gatt) + 3 + (T_gatt/T_glinear)

    // measured timings (microseconds, per-layer averages)
    double T_glinear_us;           // GPU linear (FFN) time
    double T_gatt_us;              // GPU attention time
    double T_catt_us;              // CPU attention time (estimated or measured)
};

struct apex_offload_policy {
    bool   active = false;
    int    offload_start_layer = 0;  // first layer to offload attention to CPU
    int    offload_end_layer = -1;   // last layer (-1 = all layers)

    // Returns true if the given op name at the given layer should run on CPU
    bool should_offload_to_cpu(const char * name, int layer) const;
};

// Evaluate the APEX critical inequality for decode-only workloads.
// T_glinear_us: average GPU linear layer time (microseconds)
// T_gatt_us: average GPU attention time (microseconds)
// T_catt_us: average CPU attention time (microseconds, 0 = estimate from GPU time)
// Returns decision with strategy recommendation.
apex_decision apex_evaluate_decode(
    double T_glinear_us,
    double T_gatt_us,
    double T_catt_us
);

// Evaluate the APEX inequality for mixed prefill+decode workloads.
// T_glinear_pref_us: GPU linear time during prefill
// T_gatt_pref_us: GPU attention time during prefill
// T_glinear_us: GPU linear time during decode
// T_gatt_us: GPU attention time during decode
// T_catt_us: CPU attention time during decode
apex_decision apex_evaluate_mixed(
    double T_glinear_pref_us,
    double T_gatt_pref_us,
    double T_glinear_us,
    double T_gatt_us,
    double T_catt_us
);

// Create offload policy from decision and model parameters.
// n_layers: total number of model layers
apex_offload_policy apex_create_policy(
    const apex_decision & decision,
    int n_layers
);

// Format decision as a log-friendly string.
std::string apex_decision_to_string(const apex_decision & decision);
