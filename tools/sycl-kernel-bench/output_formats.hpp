#pragma once

#include <cmath>
#include <cstdio>
#include <string>

#include "benchmark_harness.hpp"

namespace sycl_bench {

enum class OutputFormat {
    CSV,
    JSON,
    JSONL,
};

inline const char * output_format_name(OutputFormat fmt) {
    switch (fmt) {
        case OutputFormat::CSV: return "csv";
        case OutputFormat::JSON: return "json";
        case OutputFormat::JSONL: return "jsonl";
        default: return "csv";
    }
}

inline std::string json_escape(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (char c : value) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '\"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    return out;
}

inline void print_json_number(FILE * out, double value) {
    if (std::isfinite(value)) {
        std::fprintf(out, "%.6f", value);
    } else {
        std::fputs("null", out);
    }
}

inline void print_csv_header(FILE * out, const BenchmarkConfig & cfg) {
    if (cfg.kernel_kind == KernelKind::DPAS_EXPLORATION) {
        std::fprintf(out,
                     "config,type_a,type_b,type_acc,dim_m,dim_n,dim_k,memory_pattern,grf_mode,repeat,ntiles,"
                     "prefetch_dist,throughput_tops,latency_ns,xmx_util_pct,bandwidth_gbps\n");
        return;
    }
    std::fprintf(out,
                 "kernel,quant,batch,dim_m,dim_n,dim_k,layout,memory,throughput_tps,latency_us,latency_std_us");
    if (cfg.include_percentiles) {
        std::fprintf(out, ",latency_p50_us,latency_p90_us,latency_p99_us");
    }
    std::fprintf(out, ",bandwidth_gbps,xmx_util_pct,variance_pct,max_abs_error,mean_abs_error");
    if (cfg.include_ref_metrics) {
        std::fprintf(out,
                     ",ref_total_us,ref_dequant_us,ref_gemm_us,ref_scale_us,ref_tflops,ref_tops,ref_bandwidth_gbps,"
                     "ref_arith_intensity");
    }
    std::fprintf(out, "\n");
}

inline void print_csv_row(FILE * out, const BenchmarkOutput & result) {
    const auto & cfg = result.config;
    const auto & res = result.result;
    if (cfg.kernel_kind == KernelKind::DPAS_EXPLORATION) {
        const std::string config_name = cfg.dpas_config_name.empty() ? cfg.kernel_name : cfg.dpas_config_name;
        std::fprintf(out,
                     "%s,%s,%s,%s,%lld,%lld,%lld,%s,%s,%d,%d,%d,%.6f,%.6f,%.2f,%.6f\n",
                     config_name.c_str(),
                     dpas_type_name(cfg.dpas_type_a),
                     dpas_type_name(cfg.dpas_type_b),
                     dpas_acc_name(cfg.dpas_type_acc),
                     static_cast<long long>(cfg.dim_m),
                     static_cast<long long>(cfg.dim_n),
                     static_cast<long long>(cfg.dim_k),
                     dpas_memory_pattern_name(cfg.dpas_memory_pattern),
                     dpas_grf_mode_name(cfg.dpas_grf_mode),
                     cfg.dpas_repeat,
                     cfg.dpas_n_tile_repeats,
                     dpas_prefetch_dist_from_pattern(cfg.dpas_memory_pattern),
                     res.throughput_tops,
                     res.latency_ns,
                     res.xmx_util_pct,
                     res.bandwidth_gbps);
        return;
    }
    std::fprintf(out,
                 "%s,%s,%lld,%lld,%lld,%lld,%s,%s,%.6f,%.6f,%.6f",
                 cfg.kernel_name.c_str(),
                 ggml_type_name(cfg.quant_type),
                 (long long) cfg.batch_size,
                 (long long) cfg.dim_m,
                 (long long) cfg.dim_n,
                 (long long) cfg.dim_k,
                 layout_name(cfg.layout),
                 memory_mode_name(cfg.memory_mode),
                 res.throughput_tps,
                 res.latency_us,
                 res.latency_std);
    if (cfg.include_percentiles) {
        std::fprintf(out, ",%.6f,%.6f,%.6f", res.latency_p50_us, res.latency_p90_us, res.latency_p99_us);
    }
    std::fprintf(out,
                 ",%.6f,%.2f,%.2f,%.6f,%.6f",
                 res.bandwidth_gbps,
                 res.xmx_util_pct,
                 res.variance_pct,
                 result.max_abs_error,
                 result.mean_abs_error);
    if (cfg.include_ref_metrics) {
        std::fprintf(out,
                     ",%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                     res.ref_total_us,
                     res.ref_dequant_us,
                     res.ref_gemm_us,
                     res.ref_scale_us,
                     res.ref_tflops,
                     res.ref_tops,
                     res.ref_bandwidth_gbps,
                     res.ref_arith_intensity);
    }
    std::fprintf(out, "\n");
}

inline void print_json(FILE * out, const BenchmarkOutput & result) {
    const auto & cfg = result.config;
    const auto & res = result.result;
    if (cfg.kernel_kind == KernelKind::DPAS_EXPLORATION) {
        const std::string config_name = cfg.dpas_config_name.empty() ? cfg.kernel_name : cfg.dpas_config_name;
        std::fprintf(out,
                     "{\n"
                     "  \"config\": \"%s\",\n"
                     "  \"type_a\": \"%s\",\n"
                     "  \"type_b\": \"%s\",\n"
                     "  \"type_acc\": \"%s\",\n"
                     "  \"dim_m\": %lld,\n"
                     "  \"dim_n\": %lld,\n"
                     "  \"dim_k\": %lld,\n"
                     "  \"memory_pattern\": \"%s\",\n"
                     "  \"grf_mode\": \"%s\",\n"
                     "  \"repeat\": %d,\n"
                     "  \"ntiles\": %d,\n"
                     "  \"prefetch_dist\": %d,\n"
                     "  \"throughput_tops\": %.6f,\n"
                     "  \"latency_ns\": %.6f,\n"
                     "  \"xmx_util_pct\": %.2f,\n"
                     "  \"bandwidth_gbps\": %.6f\n"
                     "}\n",
                     json_escape(config_name).c_str(),
                     dpas_type_name(cfg.dpas_type_a),
                     dpas_type_name(cfg.dpas_type_b),
                     dpas_acc_name(cfg.dpas_type_acc),
                     static_cast<long long>(cfg.dim_m),
                     static_cast<long long>(cfg.dim_n),
                     static_cast<long long>(cfg.dim_k),
                     dpas_memory_pattern_name(cfg.dpas_memory_pattern),
                     dpas_grf_mode_name(cfg.dpas_grf_mode),
                     cfg.dpas_repeat,
                     cfg.dpas_n_tile_repeats,
                     dpas_prefetch_dist_from_pattern(cfg.dpas_memory_pattern),
                     res.throughput_tops,
                     res.latency_ns,
                     res.xmx_util_pct,
                     res.bandwidth_gbps);
        return;
    }
    std::fprintf(out,
                 "{\n"
                 "  \"kernel\": \"%s\",\n"
                 "  \"quant\": \"%s\",\n"
                 "  \"batch\": %lld,\n"
                 "  \"dim_m\": %lld,\n"
                 "  \"dim_n\": %lld,\n"
                 "  \"dim_k\": %lld,\n"
                 "  \"layout\": \"%s\",\n"
                 "  \"memory\": \"%s\",\n"
                 "  \"throughput_tps\": %.6f,\n"
                 "  \"latency_us\": %.6f,\n"
                 "  \"latency_std_us\": %.6f",
                 json_escape(cfg.kernel_name).c_str(),
                 ggml_type_name(cfg.quant_type),
                 (long long) cfg.batch_size,
                 (long long) cfg.dim_m,
                 (long long) cfg.dim_n,
                 (long long) cfg.dim_k,
                 layout_name(cfg.layout),
                 memory_mode_name(cfg.memory_mode),
                 res.throughput_tps,
                 res.latency_us,
                 res.latency_std);
    if (cfg.include_percentiles) {
        std::fprintf(out,
                     ",\n"
                     "  \"latency_p50_us\": %.6f,\n"
                     "  \"latency_p90_us\": %.6f,\n"
                     "  \"latency_p99_us\": %.6f",
                     res.latency_p50_us,
                     res.latency_p90_us,
                     res.latency_p99_us);
    }
    std::fprintf(out,
                 ",\n"
                 "  \"bandwidth_gbps\": %.6f,\n"
                 "  \"xmx_util_pct\": %.2f,\n"
                 "  \"variance_pct\": %.2f,\n"
                 "  \"max_abs_error\": ",
                 res.bandwidth_gbps,
                 res.xmx_util_pct,
                 res.variance_pct);
    print_json_number(out, result.max_abs_error);
    std::fprintf(out, ",\n  \"mean_abs_error\": ");
    print_json_number(out, result.mean_abs_error);
    if (cfg.include_ref_metrics) {
        std::fprintf(out,
                     ",\n"
                     "  \"ref_total_us\": %.6f,\n"
                     "  \"ref_dequant_us\": %.6f,\n"
                     "  \"ref_gemm_us\": %.6f,\n"
                     "  \"ref_scale_us\": %.6f,\n"
                     "  \"ref_tflops\": %.6f,\n"
                     "  \"ref_tops\": %.6f,\n"
                     "  \"ref_bandwidth_gbps\": %.6f,\n"
                     "  \"ref_arith_intensity\": %.6f",
                     res.ref_total_us,
                     res.ref_dequant_us,
                     res.ref_gemm_us,
                     res.ref_scale_us,
                     res.ref_tflops,
                     res.ref_tops,
                     res.ref_bandwidth_gbps,
                     res.ref_arith_intensity);
    }
    std::fprintf(out, "\n}\n");
}

inline void print_jsonl(FILE * out, const BenchmarkOutput & result) {
    const auto & cfg = result.config;
    const auto & res = result.result;
    if (cfg.kernel_kind == KernelKind::DPAS_EXPLORATION) {
        const std::string config_name = cfg.dpas_config_name.empty() ? cfg.kernel_name : cfg.dpas_config_name;
        std::fprintf(out,
                     "{\"config\":\"%s\",\"type_a\":\"%s\",\"type_b\":\"%s\",\"type_acc\":\"%s\","
                     "\"dim_m\":%lld,\"dim_n\":%lld,\"dim_k\":%lld,"
                     "\"memory_pattern\":\"%s\",\"grf_mode\":\"%s\",\"repeat\":%d,\"ntiles\":%d,\"prefetch_dist\":%d,"
                     "\"throughput_tops\":%.6f,\"latency_ns\":%.6f,\"xmx_util_pct\":%.2f,\"bandwidth_gbps\":%.6f}\n",
                     json_escape(config_name).c_str(),
                     dpas_type_name(cfg.dpas_type_a),
                     dpas_type_name(cfg.dpas_type_b),
                     dpas_acc_name(cfg.dpas_type_acc),
                     static_cast<long long>(cfg.dim_m),
                     static_cast<long long>(cfg.dim_n),
                     static_cast<long long>(cfg.dim_k),
                     dpas_memory_pattern_name(cfg.dpas_memory_pattern),
                     dpas_grf_mode_name(cfg.dpas_grf_mode),
                     cfg.dpas_repeat,
                     cfg.dpas_n_tile_repeats,
                     dpas_prefetch_dist_from_pattern(cfg.dpas_memory_pattern),
                     res.throughput_tops,
                     res.latency_ns,
                     res.xmx_util_pct,
                     res.bandwidth_gbps);
        return;
    }
    std::fprintf(out,
                 "{\"kernel\":\"%s\",\"quant\":\"%s\",\"batch\":%lld,\"dim_m\":%lld,\"dim_n\":%lld,\"dim_k\":%lld,"
                 "\"layout\":\"%s\",\"memory\":\"%s\",\"throughput_tps\":%.6f,\"latency_us\":%.6f,"
                 "\"latency_std_us\":%.6f",
                 json_escape(cfg.kernel_name).c_str(),
                 ggml_type_name(cfg.quant_type),
                 (long long) cfg.batch_size,
                 (long long) cfg.dim_m,
                 (long long) cfg.dim_n,
                 (long long) cfg.dim_k,
                 layout_name(cfg.layout),
                 memory_mode_name(cfg.memory_mode),
                 res.throughput_tps,
                 res.latency_us,
                 res.latency_std);
    if (cfg.include_percentiles) {
        std::fprintf(out,
                     ",\"latency_p50_us\":%.6f,\"latency_p90_us\":%.6f,\"latency_p99_us\":%.6f",
                     res.latency_p50_us,
                     res.latency_p90_us,
                     res.latency_p99_us);
    }
    std::fprintf(out,
                 ",\"bandwidth_gbps\":%.6f,\"xmx_util_pct\":%.2f,\"variance_pct\":%.2f,\"max_abs_error\":",
                 res.bandwidth_gbps,
                 res.xmx_util_pct,
                 res.variance_pct);
    print_json_number(out, result.max_abs_error);
    std::fprintf(out, ",\"mean_abs_error\":");
    print_json_number(out, result.mean_abs_error);
    if (cfg.include_ref_metrics) {
        std::fprintf(out,
                     ",\"ref_total_us\":%.6f,\"ref_dequant_us\":%.6f,\"ref_gemm_us\":%.6f,\"ref_scale_us\":%.6f,"
                     "\"ref_tflops\":%.6f,\"ref_tops\":%.6f,\"ref_bandwidth_gbps\":%.6f,\"ref_arith_intensity\":%.6f",
                     res.ref_total_us,
                     res.ref_dequant_us,
                     res.ref_gemm_us,
                     res.ref_scale_us,
                     res.ref_tflops,
                     res.ref_tops,
                     res.ref_bandwidth_gbps,
                     res.ref_arith_intensity);
    }
    std::fprintf(out, "}\n");
}

}  // namespace sycl_bench
