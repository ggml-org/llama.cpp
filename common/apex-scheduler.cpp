#include "apex-scheduler.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>

// Attention op names used by llama.cpp graph construction.
static bool is_attention_op(const char * name) {
    if (!name) {
        return false;
    }

    // strip layer suffix (e.g., "Qcur-0" -> "Qcur")
    const char * dash = strrchr(name, '-');
    size_t len = dash ? (size_t)(dash - name) : strlen(name);

    // check against known attention op names
    static const char * attn_ops[] = {
        "Qcur", "Kcur", "Vcur",
        "attn_out", "attn_norm",
        "kqv", "kq", "kq_soft_max", "kqv_out", "kqv_mla",
        "Qcur_normed", "Kcur_normed",
    };

    for (const char * op : attn_ops) {
        if (strlen(op) == len && strncmp(name, op, len) == 0) {
            return true;
        }
    }

    return false;
}

bool apex_offload_policy::should_offload_to_cpu(const char * name, int layer) const {
    if (!active) {
        return false;
    }

    if (layer < offload_start_layer) {
        return false;
    }

    if (offload_end_layer >= 0 && layer > offload_end_layer) {
        return false;
    }

    return is_attention_op(name);
}

apex_decision apex_evaluate_decode(
    double T_glinear_us,
    double T_gatt_us,
    double T_catt_us
) {
    apex_decision d = {};

    d.T_glinear_us = T_glinear_us;
    d.T_gatt_us    = T_gatt_us;

    // estimate CPU attention time if not provided
    if (T_catt_us <= 0.0) {
        T_catt_us = T_gatt_us * 10.0;
    }
    d.T_catt_us = T_catt_us;

    // guard against degenerate inputs
    if (T_glinear_us <= 0.0 || T_gatt_us <= 0.0) {
        d.gpu_throughput_est    = 0.0;
        d.hybrid_throughput_est = 0.0;
        d.ratio     = 0.0;
        d.threshold = 0.0;
        return d;
    }

    // APEX critical inequality:
    //   ratio = T_gatt / T_catt  (approximates N_G/N_C for attention throughput)
    //   threshold = 2*(T_glinear/T_gatt) + 3 + (T_gatt/T_glinear)
    //   offload profitable when ratio < threshold
    d.ratio     = T_gatt_us / T_catt_us;
    d.threshold = 2.0 * (T_glinear_us / T_gatt_us) + 3.0 + (T_gatt_us / T_glinear_us);

    if (d.ratio < d.threshold) {
        d.cpu_offload_profitable = true;
        d.strategy = APEX_STRATEGY_ASYNC_OVERLAP;
    }

    // throughput estimates (tokens per microsecond, per layer)
    double gpu_only_time = T_glinear_us + T_gatt_us;
    double hybrid_time   = std::max(T_glinear_us, T_catt_us);

    d.gpu_throughput_est    = 1.0 / gpu_only_time;
    d.hybrid_throughput_est = 1.0 / hybrid_time;

    // confirm strategy: only use async overlap if hybrid is actually faster
    if (!(d.hybrid_throughput_est > d.gpu_throughput_est && d.cpu_offload_profitable)) {
        d.cpu_offload_profitable = false;
        d.strategy = APEX_STRATEGY_GPU_ONLY;
    }

    return d;
}

apex_decision apex_evaluate_mixed(
    double T_glinear_pref_us,
    double T_gatt_pref_us,
    double T_glinear_us,
    double T_gatt_us,
    double T_catt_us
) {
    apex_decision d = {};

    d.T_glinear_us = T_glinear_us;
    d.T_gatt_us    = T_gatt_us;

    // estimate CPU attention time if not provided
    if (T_catt_us <= 0.0) {
        T_catt_us = T_gatt_us * 10.0;
    }
    d.T_catt_us = T_catt_us;

    // guard against degenerate inputs
    if (T_glinear_us <= 0.0 || T_gatt_us <= 0.0 ||
        T_glinear_pref_us <= 0.0 || T_gatt_pref_us <= 0.0) {
        d.gpu_throughput_est    = 0.0;
        d.hybrid_throughput_est = 0.0;
        d.ratio     = 0.0;
        d.threshold = 0.0;
        return d;
    }

    // In mixed prefill+decode workloads, the full cycle time includes both phases.
    // The CPU has the entire cycle to complete attention, making offload more favorable.
    double T_overlap = T_glinear_pref_us + T_gatt_pref_us + T_glinear_us + T_gatt_us;

    // ratio: GPU attention speed vs CPU attention speed
    d.ratio = T_gatt_us / T_catt_us;

    // threshold is more favorable in mixed mode because the CPU has more time
    // (the full overlap window) to complete its attention work
    d.threshold = 2.0 * (T_glinear_us / T_gatt_us) + 3.0 + (T_gatt_us / T_glinear_us);

    if (d.ratio < d.threshold) {
        d.cpu_offload_profitable = true;
        d.strategy = APEX_STRATEGY_ASYNC_OVERLAP;
    }

    // throughput estimates
    // GPU-only: full cycle processes one prefill + one decode token
    d.gpu_throughput_est = 1.0 / T_overlap;

    // hybrid: CPU attention runs during GPU prefill + GPU FFN decode
    // effective time = max(GPU_work_without_decode_attn, T_catt)
    double gpu_work = T_glinear_pref_us + T_gatt_pref_us + T_glinear_us;
    double hybrid_time = std::max(gpu_work, T_catt_us);
    d.hybrid_throughput_est = 1.0 / hybrid_time;

    // confirm strategy
    if (!(d.hybrid_throughput_est > d.gpu_throughput_est && d.cpu_offload_profitable)) {
        d.cpu_offload_profitable = false;
        d.strategy = APEX_STRATEGY_GPU_ONLY;
    }

    return d;
}

apex_offload_policy apex_create_policy(
    const apex_decision & decision,
    int n_layers
) {
    apex_offload_policy policy = {};

    if (!decision.cpu_offload_profitable) {
        return policy;
    }

    policy.active              = true;
    policy.offload_start_layer = 0;
    policy.offload_end_layer   = n_layers - 1;

    return policy;
}

std::string apex_decision_to_string(const apex_decision & decision) {
    const char * offload_str = decision.cpu_offload_profitable ? "yes" : "no";

    const char * strategy_str = "GPU_ONLY";
    switch (decision.strategy) {
        case APEX_STRATEGY_GPU_ONLY:        strategy_str = "GPU_ONLY";        break;
        case APEX_STRATEGY_ASYNC_OVERLAP:   strategy_str = "ASYNC_OVERLAP";   break;
        case APEX_STRATEGY_ASYMMETRIC_PIPE: strategy_str = "ASYMMETRIC_PIPE"; break;
    }

    // convert per-layer tokens/us to tokens/sec (multiply by 1e6)
    double gpu_tps    = decision.gpu_throughput_est    * 1e6;
    double hybrid_tps = decision.hybrid_throughput_est * 1e6;

    char buf[256];
    snprintf(buf, sizeof(buf),
        "APEX gate: offload=%s strategy=%s ratio=%.2f threshold=%.2f "
        "gpu_est=%.1f hybrid_est=%.1f tok/s/layer",
        offload_str, strategy_str, decision.ratio, decision.threshold,
        gpu_tps, hybrid_tps);

    return std::string(buf);
}
