#include "uma-profiler.h"
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>

// Estimate bytes read by an operation (weight tensor sizes + input tensor sizes).
static int64_t estimate_op_bytes(const struct ggml_tensor * t) {
    int64_t bytes = 0;

    // output tensor bytes
    bytes += ggml_nbytes(t);

    // source tensor bytes
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (t->src[i]) {
            bytes += ggml_nbytes(t->src[i]);
        }
    }

    return bytes;
}

// Estimate FLOPs for an operation.
static int64_t estimate_op_flops(const struct ggml_tensor * t) {
    if (t->op == GGML_OP_MUL_MAT) {
        // MUL_MAT: 2 * M * N * K (multiply-accumulate = 2 ops per element)
        const int64_t M = t->ne[1];
        const int64_t N = t->ne[0];
        const int64_t K = t->src[0] ? t->src[0]->ne[0] : 0;
        return 2 * M * N * K;
    }
    if (t->op == GGML_OP_MUL_MAT_ID) {
        // sparse MoE: FLOPs depend on active experts
        const int64_t M = t->ne[1];
        const int64_t N = t->ne[0];
        const int64_t K = t->src[0] ? t->src[0]->ne[0] : 0;
        return 2 * M * N * K;
    }
    if (t->op == GGML_OP_FLASH_ATTN_EXT) {
        // flash attention: approximate as 2*N*D*S (Q*K^T) + 2*N*D*S (attn*V)
        // where N = seq_len, D = head_dim, S = kv_len
        const int64_t D = t->src[0] ? t->src[0]->ne[0] : 0;
        const int64_t N = t->src[0] ? t->src[0]->ne[1] : 0;
        const int64_t S = t->src[1] ? t->src[1]->ne[1] : 0;
        return 4 * N * D * S;
    }

    // for other ops, rough estimate based on element count
    return ggml_nelements(t);
}

// Parse tensor name into (op_type, layer_index).
// Tensor names follow the pattern "op_name-layer_index" (e.g., "Qcur-0", "ffn_out-3").
static void parse_tensor_name(const char * name, std::string & op_type, int & layer) {
    op_type = name ? name : "unknown";
    layer = -1;

    if (!name) {
        return;
    }

    // find the last '-' to split name and layer index
    const char * dash = strrchr(name, '-');
    if (dash && dash[1] >= '0' && dash[1] <= '9') {
        op_type = std::string(name, dash - name);
        layer = atoi(dash + 1);
    }
}

// Classify an op name as attention or FFN.
// Returns: 'A' for attention, 'F' for FFN, '?' for unknown.
static char classify_op(const std::string & op_type) {
    // attention ops
    if (op_type == "Qcur" || op_type == "Kcur" || op_type == "Vcur" ||
        op_type == "attn_norm" || op_type == "attn_out" ||
        op_type == "Qcur_normed" || op_type == "Kcur_normed" ||
        op_type == "kqv_out" || op_type == "kq" || op_type == "kq_soft_max") {
        return 'A';
    }
    // FFN ops
    if (op_type == "ffn_norm" || op_type == "ffn_out" ||
        op_type == "ffn_up" || op_type == "ffn_gate" || op_type == "ffn_down" ||
        op_type == "ffn_moe_out" || op_type == "ffn_inp") {
        return 'F';
    }
    return '?';
}

bool uma_profiler_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * data = static_cast<uma_profiler_data *>(user_data);

    if (!data || !data->profiling_active) {
        return true; // pass-through
    }

    if (ask) {
        // "ask" phase: record start time for next compute
        data->last_op_start_us = ggml_time_us();
        return true; // always compute
    }

    // "done" phase: record timing
    const int64_t now_us = ggml_time_us();
    const int64_t elapsed_us = now_us - data->last_op_start_us;

    if (t->name[0] == '\0') {
        return true;
    }

    std::string op_type;
    int layer;
    parse_tensor_name(t->name, op_type, layer);

    std::string key = op_type + ":" + std::to_string(layer);
    auto & stats = data->op_stats[key];
    stats.total_us += elapsed_us;
    stats.total_bytes += estimate_op_bytes(t);
    stats.total_flops += estimate_op_flops(t);
    stats.count++;
    stats.layer = layer;

    return true;
}

void uma_profiler_iteration_done(uma_profiler_data & data) {
    data.n_iterations++;
    if (data.n_iterations >= data.max_iterations) {
        data.profiling_active = false;
    }
}

std::vector<uma_layer_analysis> uma_profiler_analyze_layers(const uma_profiler_data & data) {
    std::map<int, uma_layer_analysis> layers;

    for (const auto & [key, stats] : data.op_stats) {
        if (stats.layer < 0 || stats.count == 0) {
            continue;
        }

        auto & la = layers[stats.layer];
        la.layer = stats.layer;

        // parse op type from key (key = "op_type:layer")
        std::string op_type = key.substr(0, key.find(':'));
        char cls = classify_op(op_type);

        if (cls == 'A') {
            la.attn_us += stats.total_us;
            la.attn_bytes += stats.total_bytes;
            if (stats.total_bytes > 0) {
                la.attn_ai = static_cast<double>(stats.total_flops) / stats.total_bytes;
            }
        } else if (cls == 'F') {
            la.ffn_us += stats.total_us;
            la.ffn_bytes += stats.total_bytes;
            if (stats.total_bytes > 0) {
                la.ffn_ai = static_cast<double>(stats.total_flops) / stats.total_bytes;
            }
        }
    }

    // classify ops and collect results
    // arithmetic intensity threshold: below this = bandwidth-bound, above = compute-bound
    // typical threshold for modern GPUs is ~10 FLOPS/byte
    constexpr double ai_threshold = 5.0;

    std::vector<uma_layer_analysis> result;
    result.reserve(layers.size());
    for (auto & [layer_id, la] : layers) {
        la.attn_class = la.attn_ai > ai_threshold ? UMA_OP_CLASS_COMPUTE : UMA_OP_CLASS_BANDWIDTH;
        la.ffn_class  = la.ffn_ai  > ai_threshold ? UMA_OP_CLASS_COMPUTE : UMA_OP_CLASS_BANDWIDTH;
        result.push_back(la);
    }

    return result;
}

std::string uma_profiler_report(uma_profiler_data & data) {
    std::ostringstream report;

    if (data.n_iterations == 0) {
        report << "UMA Profiler: no iterations profiled yet\n";
        return report.str();
    }

    // compute derived metrics
    for (auto & [key, stats] : data.op_stats) {
        if (stats.count > 0) {
            stats.avg_us = static_cast<double>(stats.total_us) / stats.count;
            if (stats.total_us > 0) {
                stats.bandwidth_gbps = static_cast<double>(stats.total_bytes) / (stats.total_us * 1e3); // bytes/us = MB/s, /1e3 = GB/s
                stats.compute_gflops = static_cast<double>(stats.total_flops) / (stats.total_us * 1e3); // FLOPS/us = MFLOPS, /1e3 = GFLOPS
            }
            if (stats.total_bytes > 0) {
                stats.arithmetic_intensity = static_cast<double>(stats.total_flops) / stats.total_bytes;
            }
        }
    }

    auto layers = uma_profiler_analyze_layers(data);

    report << "\n=== UMA Bandwidth-Aware Profiler Report ===\n";
    report << "Iterations profiled: " << data.n_iterations << "\n\n";

    // summary: total time in attention vs FFN
    int64_t total_attn_us = 0, total_ffn_us = 0;
    int64_t total_attn_bytes = 0, total_ffn_bytes = 0;
    for (const auto & la : layers) {
        total_attn_us += la.attn_us;
        total_ffn_us += la.ffn_us;
        total_attn_bytes += la.attn_bytes;
        total_ffn_bytes += la.ffn_bytes;
    }

    const double total_ms = (total_attn_us + total_ffn_us) / 1e3;
    report << "Time distribution (across " << layers.size() << " layers):\n";
    report << "  Attention: " << total_attn_us / 1e3 << " ms ("
           << (total_ms > 0 ? 100.0 * total_attn_us / (total_attn_us + total_ffn_us) : 0) << "%)\n";
    report << "  FFN:       " << total_ffn_us / 1e3 << " ms ("
           << (total_ms > 0 ? 100.0 * total_ffn_us / (total_attn_us + total_ffn_us) : 0) << "%)\n";
    report << "\n";

    report << "Memory traffic:\n";
    report << "  Attention: " << total_attn_bytes / (1024.0 * 1024.0) << " MiB\n";
    report << "  FFN:       " << total_ffn_bytes / (1024.0 * 1024.0) << " MiB\n";
    report << "\n";

    // per-layer breakdown
    report << "Per-layer analysis (layer | attn_ms | ffn_ms | attn_AI | ffn_AI | attn_class | ffn_class):\n";
    for (const auto & la : layers) {
        const char * attn_cls = la.attn_class == UMA_OP_CLASS_COMPUTE ? "compute" :
                                la.attn_class == UMA_OP_CLASS_BANDWIDTH ? "bandwidth" : "unknown";
        const char * ffn_cls  = la.ffn_class  == UMA_OP_CLASS_COMPUTE ? "compute" :
                                la.ffn_class  == UMA_OP_CLASS_BANDWIDTH ? "bandwidth" : "unknown";

        char buf[256];
        snprintf(buf, sizeof(buf),
            "  layer %3d: attn=%6.2f ms  ffn=%6.2f ms  attn_AI=%.1f  ffn_AI=%.1f  [%s / %s]\n",
            la.layer, la.attn_us / 1e3, la.ffn_us / 1e3,
            la.attn_ai, la.ffn_ai, attn_cls, ffn_cls);
        report << buf;
    }
    report << "\n";

    // APEX-inspired recommendations for UMA
    report << "=== UMA Overflow Recommendations (APEX-inspired) ===\n";
    bool ffn_is_bandwidth_bound = false;
    bool attn_is_compute_bound = false;

    int n_bw_ffn = 0, n_compute_attn = 0;
    for (const auto & la : layers) {
        if (la.ffn_class == UMA_OP_CLASS_BANDWIDTH) n_bw_ffn++;
        if (la.attn_class == UMA_OP_CLASS_COMPUTE) n_compute_attn++;
    }

    if (!layers.empty()) {
        ffn_is_bandwidth_bound = n_bw_ffn > (int)layers.size() / 2;
        attn_is_compute_bound = n_compute_attn > (int)layers.size() / 2;
    }

    if (ffn_is_bandwidth_bound && attn_is_compute_bound) {
        report << "RECOMMENDATION: FFN is bandwidth-bound, attention is compute-bound.\n";
        report << "  -> On UMA, prefer LAYER_FRACTION_FFN overflow (keep FFN on GPU, attention on CPU)\n";
        report << "  -> GPU's ~2x higher effective bandwidth benefits FFN more than attention\n";
        report << "  -> This aligns with APEX's asymmetric pipelining strategy\n";
    } else if (ffn_is_bandwidth_bound) {
        report << "RECOMMENDATION: FFN is bandwidth-bound but attention is also bandwidth-bound.\n";
        report << "  -> On UMA, still prefer keeping FFN on GPU (larger weight matrices)\n";
        report << "  -> Use LAYER_FRACTION_FFN overflow where possible\n";
    } else {
        report << "OBSERVATION: Both attention and FFN appear compute-bound at current batch size.\n";
        report << "  -> Standard layer splitting may be sufficient\n";
        report << "  -> Consider increasing batch size for better GPU utilization\n";
    }

    report << "\n";

    // bandwidth utilization estimate
    if (total_ffn_us > 0 && total_ffn_bytes > 0) {
        const double ffn_bw = static_cast<double>(total_ffn_bytes) / (total_ffn_us * 1e3);
        report << "Effective FFN bandwidth: " << ffn_bw << " GB/s\n";
        if (data.gpu_bandwidth_gbps > 0) {
            report << "  GPU bandwidth utilization: "
                   << 100.0 * ffn_bw / data.gpu_bandwidth_gbps << "%\n";
        }
    }
    if (total_attn_us > 0 && total_attn_bytes > 0) {
        const double attn_bw = static_cast<double>(total_attn_bytes) / (total_attn_us * 1e3);
        report << "Effective attention bandwidth: " << attn_bw << " GB/s\n";
    }

    return report.str();
}
