#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "benchmark_harness.hpp"
#include "dpas_config.hpp"
#include "kernel_registry.hpp"
#include "model_shapes.hpp"
#include "output_formats.hpp"

#include "ggml-sycl.h"
#include "ggml.h"

using namespace sycl_bench;

enum class SampleStrategy {
    CUSTOM,
    DECODE,
    PROMPT,
    BOTH,
};

struct CmdParams {
    std::string kernel_name = "mmvq_aos";
    ggml_type   quant_type = GGML_TYPE_Q4_0;
    std::vector<int64_t> batch_sizes = { 1 };
    int64_t dim_m = 4096;
    int64_t dim_n = 4096;
    int64_t dim_k = 4096;
    int warmup = 10;
    int iterations = 100;
    MemoryMode memory_mode = MemoryMode::USM_DEVICE;
    OutputFormat output_format = OutputFormat::CSV;
    bool validate = false;
    double abs_tol = 1e-2;
    double rel_tol = 1e-2;
    bool list_devices = false;
    bool include_percentiles = false;
    bool include_ref_metrics = false;
    bool show_help = false;
    int device_id = -1;
    size_t transfer_bytes = 0;
    int64_t roofline_elements = 0;
    int roofline_ops = 0;
    double xmx_peak_tops = 0.0;
    double expect_tps = -1.0;
    double expect_tops = -1.0;
    double expect_bandwidth_gbps = -1.0;
    double expect_xmx_util_pct = -1.0;
    std::string dpas_config_name;
    DpasType dpas_type_a = DpasType::INT8;
    DpasType dpas_type_b = DpasType::INT8;
    DpasAccType dpas_type_acc = DpasAccType::INT32;
    DpasMemoryPattern dpas_memory_pattern = DpasMemoryPattern::DIRECT_GLOBAL;
    DpasGrfMode dpas_grf_mode = DpasGrfMode::GRF_128;
    int dpas_repeat = 8;
    int dpas_n_tile_repeats = 1;
    bool dpas_misaligned = false;
    bool dpas_device_opt = false;
    bool dpas_autotune = false;
    bool dpas_autotune_force = false;
    DpasTuneMetric dpas_autotune_metric = DpasTuneMetric::THROUGHPUT;
    std::string dpas_autotune_cache = "benchmark_results/dpas_tuning_cache.jsonl";
    int dpas_autotune_override_ntiles = 0;
    int dpas_autotune_override_prefetch = 0;
    bool dpas_memory_explicit = false;
    bool dpas_ntiles_explicit = false;
    bool dpas_grf_explicit = false;
    bool dpas_acc_explicit = false;
    std::string model_path;
    std::string model_filter;
    int model_max_shapes = 0;
    bool model_dedup = false;
    bool model_include_fp = false;
    bool quant_type_overridden = false;
    std::string emit_json_path;
    SampleStrategy sample_strategy = SampleStrategy::CUSTOM;
};

struct KernelRunSummary {
    std::string kernel;
    bool ok = false;
    std::string error;
    double latency_us = 0.0;
    double throughput_tps = 0.0;
    double throughput_tops = 0.0;
    double bandwidth_gbps = 0.0;
    double xmx_util_pct = 0.0;
};

struct ShapeSummary {
    std::string tensor;
    ggml_type type = GGML_TYPE_F32;
    int64_t dim_m = 0;
    int64_t dim_n = 0;
    int64_t dim_k = 0;
    int64_t instances = 1;
    std::vector<KernelRunSummary> runs;
    std::string winner;
};

static void print_usage(const char * argv0) {
    std::fprintf(stderr,
                 "Usage: %s [options]\n"
                 "  --kernel=mmvq_aos|mmvq_aos_baseline|mmvq_soa|mmvq_soa_baseline|mmvq_coalesced|mmvq_slm_cached|mmvq_prefetch|mmvq_wide_load|mmvq_esimd_block_load|mmvq_esimd_slm|"
                 "mmvq_xmx_tile_8x8|mmvq_xmx_tile_16x16|mmvq_xmx_aos_direct|mmvq_xmx_soa_direct|mmvq_xmx_double_buffer|"
                 "mmvq_esimd_dpas_1x16x32|mmvq_esimd_dpas_8x16x32|mmvq_esimd_dpas_chained|"
                 "mmvq_xmx_tile_64x64|mmvq_xmx_register_accum|mmvq_xmx_multi_wg|mmvq_xmx_persistent|"
                 "mmvq_esimd_large_tile|mmvq_esimd_persistent|mmvq_esimd_lsc_prefetch|"
                 "mmvq_hybrid_adaptive|mmvq_xmx_fused|mmvq_coalesced_xmx_aligned|"
                 "mmvq_esimd_hybrid|mmvq_esimd_cooperative|mmvq_q4_0_specialized|"
                 "mmvq_q6_k_specialized|mmvq_mxfp4_native|"
                 "mmq_aos|mmq_soa|mmq_coalesced|mmq|"
                 "onednn_fp16_gemm|onednn_int8_gemm|onednn_woq_gemm|unified_matmul|memory_bandwidth|roofline_compute|"
                 "dpas_baseline|dpas_sweep|dpas_memory_patterns (comma-separated to compare)\n"
                 "  --quant=Q4_0|Q8_0|Q6_K|Q4_K|Q5_K|Q2_K|Q3_K|Q4_1|Q5_0|Q5_1|MXFP4\n"
                 "  --batch=1,4,8,16,32,64\n"
                 "  --dim=4096 (sets dim_m, dim_n, dim_k)\n"
                 "  --dim_m=4096 --dim_n=4096 --dim_k=4096\n"
                 "  --model=<path> (derive shapes from GGUF model)\n"
                 "  --model-filter=<substring> (case-insensitive filter on tensor name)\n"
                 "  --model-max-shapes=<N> (limit shapes after sorting by size)\n"
                 "  --limit-shapes=<N> (alias for --model-max-shapes)\n"
                 "  --model-dedup (deduplicate shapes by type/M/K)\n"
                 "  --model-include-fp (include non-quantized tensors)\n"
                 "  --sample-strategy=custom|decode|prompt|both (model batch selection)\n"
                 "  --emit-json=<path> (write summary JSON with per-shape winner)\n"
                 "  --iterations=100 --warmup=10\n"
                 "  --memory=usm_device|usm_shared|buffer\n"
                 "  --output=csv|json|jsonl\n"
                 "  --bytes=<size> (memory_bandwidth only; defaults to batch GiB)\n"
                 "  --elements=<count> (roofline_compute only)\n"
                 "  --ops=<count> (roofline_compute only)\n"
                 "  --xmx-peak-tops=<tops> (derive xmx_util_pct from throughput_tops)\n"
                 "  --expect-tps=<min> --expect-tops=<min>\n"
                 "  --expect-bandwidth=<min_gbps> --expect-xmx-util=<min_pct>\n"
                 "  --dpas-config=<name> (dpas kernels only)\n"
                 "  --dpas-type-a=int8|fp16|bf16\n"
                 "  --dpas-type-b=int8|fp16|bf16\n"
                 "  --dpas-acc=int32|float\n"
                "  --dpas-memory=direct_global|slm_buffer|reg_prefetch|double_buffer|lsc_streaming|lsc_prefetch|lsc_prefetch2|lsc_prefetch3|lsc_prefetch4|lsc_prefetch5|lsc_prefetch6|lsc_prefetch8|lsc_prefetch10\n"
                 "  --dpas-grf=128|256\n"
                 "  --dpas-repeat=1|2|4|8\n"
                "  --dpas-ntiles=1|2|4|8 (N tiles per work-item)\n"
                 "  --dpas-misaligned\n"
                 "  --dpas-device-opt (heuristic tuning for dpas_memory_patterns)\n"
                 "  --dpas-autotune (autotune ntiles/prefetch for dpas_memory_patterns)\n"
                 "  --dpas-autotune-force (ignore cache)\n"
                 "  --dpas-autotune-metric=throughput|bandwidth\n"
                 "  --dpas-autotune-cache=<path>\n"
                 "  --dpas-autotune-override-ntiles=<count>\n"
                 "  --dpas-autotune-override-prefetch=<dist>\n"
                 "  --abs-tol=1e-2 --rel-tol=1e-2\n"
                 "  --device=<id>\n"
                 "  --validate\n"
                 "  --include-percentiles\n"
                 "  --include-ref-metrics\n"
                 "  --list-devices\n"
                 "  -h, --help\n",
                 argv0);
}

static bool parse_int_list(const std::string & input, std::vector<int64_t> & out) {
    out.clear();
    size_t start = 0;
    while (start < input.size()) {
        size_t end = input.find(',', start);
        if (end == std::string::npos) {
            end = input.size();
        }
        std::string token = input.substr(start, end - start);
        if (token.empty()) {
            return false;
        }
        size_t dash = token.find('-');
        if (dash != std::string::npos) {
            int64_t lo = std::strtoll(token.substr(0, dash).c_str(), nullptr, 10);
            int64_t hi = std::strtoll(token.substr(dash + 1).c_str(), nullptr, 10);
            if (lo <= 0 || hi < lo) {
                return false;
            }
            for (int64_t v = lo; v <= hi; ++v) {
                out.push_back(v);
            }
        } else {
            int64_t val = std::strtoll(token.c_str(), nullptr, 10);
            if (val <= 0) {
                return false;
            }
            out.push_back(val);
        }
        start = end + 1;
    }
    return !out.empty();
}

static ggml_type parse_quant_type(const std::string & input) {
    const std::string q = to_lower(input);
    if (q == "q4_0") return GGML_TYPE_Q4_0;
    if (q == "q4_1") return GGML_TYPE_Q4_1;
    if (q == "q5_0") return GGML_TYPE_Q5_0;
    if (q == "q5_1") return GGML_TYPE_Q5_1;
    if (q == "q8_0") return GGML_TYPE_Q8_0;
    if (q == "q2_k") return GGML_TYPE_Q2_K;
    if (q == "q3_k") return GGML_TYPE_Q3_K;
    if (q == "q4_k") return GGML_TYPE_Q4_K;
    if (q == "q5_k") return GGML_TYPE_Q5_K;
    if (q == "q6_k") return GGML_TYPE_Q6_K;
    if (q == "mxfp4") return GGML_TYPE_MXFP4;
    return GGML_TYPE_COUNT;
}

static bool parse_memory_mode(const std::string & input, MemoryMode & mode) {
    const std::string m = to_lower(input);
    if (m == "usm_device" || m == "device") {
        mode = MemoryMode::USM_DEVICE;
        return true;
    }
    if (m == "usm_shared" || m == "shared") {
        mode = MemoryMode::USM_SHARED;
        return true;
    }
    if (m == "buffer") {
        mode = MemoryMode::BUFFER;
        return true;
    }
    return false;
}

static bool parse_output_format(const std::string & input, OutputFormat & fmt) {
    const std::string v = to_lower(input);
    if (v == "csv") { fmt = OutputFormat::CSV; return true; }
    if (v == "json") { fmt = OutputFormat::JSON; return true; }
    if (v == "jsonl") { fmt = OutputFormat::JSONL; return true; }
    return false;
}

static bool parse_kernel_list(const std::string & input, std::vector<std::string> & out) {
    out.clear();
    size_t start = 0;
    while (start < input.size()) {
        size_t end = input.find(',', start);
        if (end == std::string::npos) {
            end = input.size();
        }
        std::string token = input.substr(start, end - start);
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);
        if (token.empty()) {
            return false;
        }
        out.push_back(std::move(token));
        start = end + 1;
    }
    return !out.empty();
}

static bool parse_sample_strategy(const std::string & input, SampleStrategy & out) {
    const std::string v = to_lower(input);
    if (v == "custom" || v == "manual") {
        out = SampleStrategy::CUSTOM;
        return true;
    }
    if (v == "decode") {
        out = SampleStrategy::DECODE;
        return true;
    }
    if (v == "prompt") {
        out = SampleStrategy::PROMPT;
        return true;
    }
    if (v == "both" || v == "prompt+decode") {
        out = SampleStrategy::BOTH;
        return true;
    }
    return false;
}

static void apply_sample_strategy(SampleStrategy strategy, std::vector<int64_t> & batch_sizes) {
    if (strategy == SampleStrategy::CUSTOM) {
        return;
    }
    std::vector<int64_t> next;
    if (strategy == SampleStrategy::DECODE) {
        next.push_back(1);
    } else if (strategy == SampleStrategy::PROMPT) {
        for (int64_t v : batch_sizes) {
            if (v > 1) {
                next.push_back(v);
            }
        }
    } else if (strategy == SampleStrategy::BOTH) {
        next = batch_sizes;
        next.push_back(1);
    }
    if (next.empty()) {
        next.push_back(1);
    }
    std::sort(next.begin(), next.end());
    next.erase(std::unique(next.begin(), next.end()), next.end());
    batch_sizes.swap(next);
}

static bool is_skippable_error(const std::string & error) {
    if (error.rfind("SKIP:", 0) == 0) {
        return true;
    }
    static const char * tokens[] = {
        "not enabled",
        "requires",
        "supports",
        "not supported",
        "Kernel layout not supported",
    };
    for (const char * token : tokens) {
        if (error.find(token) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static std::string pick_winner(const ShapeSummary & summary) {
    bool use_tops = false;
    for (const auto & run : summary.runs) {
        if (run.ok && run.throughput_tops > 0.0) {
            use_tops = true;
            break;
        }
    }
    double best_score = use_tops ? -1.0 : std::numeric_limits<double>::infinity();
    std::string winner;
    for (const auto & run : summary.runs) {
        if (!run.ok) {
            continue;
        }
        const double score = use_tops ? run.throughput_tops : run.latency_us;
        if (use_tops) {
            if (score > best_score) {
                best_score = score;
                winner = run.kernel;
            }
        } else {
            if (score < best_score) {
                best_score = score;
                winner = run.kernel;
            }
        }
    }
    return winner;
}

static void write_summary_json(const std::string & path, const std::vector<ShapeSummary> & summaries) {
    if (path.empty()) {
        return;
    }
    FILE * out = std::fopen(path.c_str(), "w");
    if (!out) {
        std::fprintf(stderr, "Failed to open %s for writing JSON summary.\n", path.c_str());
        return;
    }
    std::fputs("{\n  \"results\": [\n", out);
    for (size_t i = 0; i < summaries.size(); ++i) {
        const auto & entry = summaries[i];
        std::fprintf(out,
                     "    {\n"
                     "      \"tensor\": \"%s\",\n"
                     "      \"quant\": \"%s\",\n"
                     "      \"dim_m\": %lld,\n"
                     "      \"dim_n\": %lld,\n"
                     "      \"dim_k\": %lld,\n"
                     "      \"tensor_instances\": %lld,\n"
                     "      \"winner\": ",
                     json_escape(entry.tensor).c_str(),
                     ggml_type_name(entry.type),
                     static_cast<long long>(entry.dim_m),
                     static_cast<long long>(entry.dim_n),
                     static_cast<long long>(entry.dim_k),
                     static_cast<long long>(entry.instances));
        if (entry.winner.empty()) {
            std::fputs("null,\n", out);
        } else {
            std::fprintf(out, "\"%s\",\n", json_escape(entry.winner).c_str());
        }
        std::fputs("      \"runs\": [\n", out);
        for (size_t r = 0; r < entry.runs.size(); ++r) {
            const auto & run = entry.runs[r];
            std::fprintf(out,
                         "        {\n"
                         "          \"kernel\": \"%s\",\n"
                         "          \"ok\": %s,\n"
                         "          \"error\": \"%s\",\n"
                         "          \"latency_us\": ",
                         json_escape(run.kernel).c_str(),
                         run.ok ? "true" : "false",
                         json_escape(run.error).c_str());
            print_json_number(out, run.latency_us);
            std::fprintf(out, ",\n          \"throughput_tps\": ");
            print_json_number(out, run.throughput_tps);
            std::fprintf(out, ",\n          \"throughput_tops\": ");
            print_json_number(out, run.throughput_tops);
            std::fprintf(out, ",\n          \"bandwidth_gbps\": ");
            print_json_number(out, run.bandwidth_gbps);
            std::fprintf(out, ",\n          \"xmx_util_pct\": ");
            print_json_number(out, run.xmx_util_pct);
            std::fputs("\n        }", out);
            if (r + 1 < entry.runs.size()) {
                std::fputs(",", out);
            }
            std::fputs("\n", out);
        }
        std::fputs("      ]\n    }", out);
        if (i + 1 < summaries.size()) {
            std::fputs(",", out);
        }
        std::fputs("\n", out);
    }
    std::fputs("  ]\n}\n", out);
    std::fclose(out);
}

static bool parse_args(int argc, char ** argv, CmdParams & params) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            params.show_help = true;
            return true;
        }
        if (arg == "--validate") {
            params.validate = true;
            continue;
        }
        if (arg == "--include-percentiles") {
            params.include_percentiles = true;
            continue;
        }
        if (arg == "--include-ref-metrics") {
            params.include_ref_metrics = true;
            continue;
        }
        if (arg == "--model-dedup") {
            params.model_dedup = true;
            continue;
        }
        if (arg == "--model-include-fp") {
            params.model_include_fp = true;
            continue;
        }
        if (arg == "--list-devices") {
            params.list_devices = true;
            continue;
        }
        if (arg == "--dpas-misaligned") {
            params.dpas_misaligned = true;
            continue;
        }
        if (arg == "--dpas-device-opt") {
            params.dpas_device_opt = true;
            continue;
        }
        if (arg == "--dpas-autotune") {
            params.dpas_autotune = true;
            continue;
        }
        if (arg == "--dpas-autotune-force") {
            params.dpas_autotune_force = true;
            continue;
        }
        auto require_value = [&](const char * name) -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\\n", name);
                params.show_help = true;
                return nullptr;
            }
            return argv[++i];
        };

        auto parse_kv = [&](const std::string & key, const std::string & value) -> bool {
            if (key == "--kernel") {
                params.kernel_name = value;
                return true;
            }
            if (key == "--quant") {
                params.quant_type = parse_quant_type(value);
                params.quant_type_overridden = true;
                return params.quant_type != GGML_TYPE_COUNT;
            }
            if (key == "--batch") {
                return parse_int_list(value, params.batch_sizes);
            }
            if (key == "--dim") {
                const int64_t dim = std::strtoll(value.c_str(), nullptr, 10);
                if (dim <= 0) return false;
                params.dim_m = dim;
                params.dim_n = dim;
                params.dim_k = dim;
                return true;
            }
            if (key == "--dim_m") {
                params.dim_m = std::strtoll(value.c_str(), nullptr, 10);
                return params.dim_m > 0;
            }
            if (key == "--dim_n") {
                params.dim_n = std::strtoll(value.c_str(), nullptr, 10);
                return params.dim_n > 0;
            }
            if (key == "--dim_k") {
                params.dim_k = std::strtoll(value.c_str(), nullptr, 10);
                return params.dim_k > 0;
            }
            if (key == "--model") {
                params.model_path = value;
                return !params.model_path.empty();
            }
            if (key == "--model-filter") {
                params.model_filter = value;
                return true;
            }
            if (key == "--model-max-shapes") {
                params.model_max_shapes = std::atoi(value.c_str());
                return params.model_max_shapes >= 0;
            }
            if (key == "--limit-shapes") {
                params.model_max_shapes = std::atoi(value.c_str());
                return params.model_max_shapes >= 0;
            }
            if (key == "--emit-json") {
                params.emit_json_path = value;
                return !params.emit_json_path.empty();
            }
            if (key == "--sample-strategy") {
                return parse_sample_strategy(value, params.sample_strategy);
            }
            if (key == "--iterations") {
                params.iterations = std::atoi(value.c_str());
                return params.iterations > 0;
            }
            if (key == "--warmup") {
                params.warmup = std::atoi(value.c_str());
                return params.warmup >= 0;
            }
            if (key == "--memory") {
                return parse_memory_mode(value, params.memory_mode);
            }
            if (key == "--output") {
                return parse_output_format(value, params.output_format);
            }
            if (key == "--device") {
                params.device_id = std::atoi(value.c_str());
                return true;
            }
            if (key == "--bytes") {
                params.transfer_bytes = static_cast<size_t>(std::strtoull(value.c_str(), nullptr, 10));
                return params.transfer_bytes > 0;
            }
            if (key == "--elements") {
                params.roofline_elements = std::strtoll(value.c_str(), nullptr, 10);
                return params.roofline_elements > 0;
            }
            if (key == "--ops") {
                params.roofline_ops = std::atoi(value.c_str());
                return params.roofline_ops > 0;
            }
            if (key == "--xmx-peak-tops") {
                params.xmx_peak_tops = std::strtod(value.c_str(), nullptr);
                return params.xmx_peak_tops > 0.0;
            }
            if (key == "--expect-tps") {
                params.expect_tps = std::strtod(value.c_str(), nullptr);
                return params.expect_tps >= 0.0;
            }
            if (key == "--expect-tops") {
                params.expect_tops = std::strtod(value.c_str(), nullptr);
                return params.expect_tops >= 0.0;
            }
            if (key == "--expect-bandwidth") {
                params.expect_bandwidth_gbps = std::strtod(value.c_str(), nullptr);
                return params.expect_bandwidth_gbps >= 0.0;
            }
            if (key == "--expect-xmx-util") {
                params.expect_xmx_util_pct = std::strtod(value.c_str(), nullptr);
                return params.expect_xmx_util_pct >= 0.0;
            }
            if (key == "--dpas-config") {
                params.dpas_config_name = value;
                return true;
            }
            if (key == "--dpas-type-a") {
                return parse_dpas_type(to_lower(value), params.dpas_type_a);
            }
            if (key == "--dpas-type-b") {
                return parse_dpas_type(to_lower(value), params.dpas_type_b);
            }
            if (key == "--dpas-acc") {
                const bool ok = parse_dpas_acc_type(to_lower(value), params.dpas_type_acc);
                if (ok) {
                    params.dpas_acc_explicit = true;
                }
                return ok;
            }
            if (key == "--dpas-memory") {
                const bool ok = parse_dpas_memory_pattern(to_lower(value), params.dpas_memory_pattern);
                if (ok) {
                    params.dpas_memory_explicit = true;
                }
                return ok;
            }
            if (key == "--dpas-grf") {
                const bool ok = parse_dpas_grf_mode(to_lower(value), params.dpas_grf_mode);
                if (ok) {
                    params.dpas_grf_explicit = true;
                }
                return ok;
            }
            if (key == "--dpas-repeat") {
                params.dpas_repeat = std::atoi(value.c_str());
                return params.dpas_repeat > 0;
            }
            if (key == "--dpas-ntiles") {
                params.dpas_n_tile_repeats = std::atoi(value.c_str());
                if (params.dpas_n_tile_repeats > 0) {
                    params.dpas_ntiles_explicit = true;
                    return true;
                }
                return false;
            }
            if (key == "--dpas-autotune-cache") {
                params.dpas_autotune_cache = value;
                return !params.dpas_autotune_cache.empty();
            }
            if (key == "--dpas-autotune-metric") {
                return parse_dpas_tune_metric(to_lower(value), params.dpas_autotune_metric);
            }
            if (key == "--dpas-autotune-override-ntiles") {
                params.dpas_autotune_override_ntiles = std::atoi(value.c_str());
                if (params.dpas_autotune_override_ntiles == 0) {
                    return true;
                }
                return params.dpas_autotune_override_ntiles == 1 ||
                       params.dpas_autotune_override_ntiles == 2 ||
                       params.dpas_autotune_override_ntiles == 4 ||
                       params.dpas_autotune_override_ntiles == 8;
            }
            if (key == "--dpas-autotune-override-prefetch") {
                params.dpas_autotune_override_prefetch = std::atoi(value.c_str());
                return params.dpas_autotune_override_prefetch >= 0 &&
                       (params.dpas_autotune_override_prefetch <= 6 ||
                        params.dpas_autotune_override_prefetch == 8 ||
                        params.dpas_autotune_override_prefetch == 10);
            }
            if (key == "--abs-tol") {
                params.abs_tol = std::strtod(value.c_str(), nullptr);
                return params.abs_tol >= 0.0;
            }
            if (key == "--rel-tol") {
                params.rel_tol = std::strtod(value.c_str(), nullptr);
                return params.rel_tol >= 0.0;
            }
            std::fprintf(stderr, "Unknown option: %s\\n", key.c_str());
            params.show_help = true;
            return false;
        };

        size_t eq = arg.find('=');
        if (eq != std::string::npos) {
            const std::string key = arg.substr(0, eq);
            const std::string value = arg.substr(eq + 1);
            if (!parse_kv(key, value)) {
                params.show_help = true;
                return false;
            }
            continue;
        }

        if (arg == "--kernel" || arg == "--quant" || arg == "--batch" || arg == "--dim" || arg == "--dim_m" ||
            arg == "--dim_n" || arg == "--dim_k" || arg == "--model" || arg == "--model-filter" ||
            arg == "--model-max-shapes" || arg == "--limit-shapes" || arg == "--emit-json" ||
            arg == "--sample-strategy" || arg == "--iterations" || arg == "--warmup" ||
            arg == "--memory" || arg == "--output" || arg == "--device" || arg == "--bytes" ||
            arg == "--elements" || arg == "--ops" || arg == "--xmx-peak-tops" || arg == "--expect-tps" ||
            arg == "--expect-tops" ||
            arg == "--expect-bandwidth" || arg == "--expect-xmx-util" || arg == "--abs-tol" || arg == "--rel-tol" ||
            arg == "--dpas-config" || arg == "--dpas-type-a" || arg == "--dpas-type-b" || arg == "--dpas-acc" ||
            arg == "--dpas-memory" || arg == "--dpas-grf" || arg == "--dpas-repeat" || arg == "--dpas-ntiles" ||
            arg == "--dpas-autotune-cache" || arg == "--dpas-autotune-metric" ||
            arg == "--dpas-autotune-override-ntiles" ||
            arg == "--dpas-autotune-override-prefetch") {
            const char * value = require_value(arg.c_str());
            if (!value) {
                return false;
            }
            if (!parse_kv(arg, value)) {
                params.show_help = true;
                return false;
            }
            continue;
        }

        std::fprintf(stderr, "Unknown argument: %s\\n", arg.c_str());
        params.show_help = true;
        return false;
    }
    return true;
}

static bool check_expectations(const CmdParams & params, const BenchmarkOutput & output) {
    bool ok = true;
    if (params.expect_tps >= 0.0 && output.result.throughput_tps < params.expect_tps) {
        std::fprintf(stderr, "Throughput TPS %.3f below expectation %.3f\n",
                     output.result.throughput_tps, params.expect_tps);
        ok = false;
    }
    if (params.expect_tops >= 0.0 && output.result.throughput_tops < params.expect_tops) {
        std::fprintf(stderr, "Throughput TOPS %.3f below expectation %.3f\n",
                     output.result.throughput_tops, params.expect_tops);
        ok = false;
    }
    if (params.expect_bandwidth_gbps >= 0.0 && output.result.bandwidth_gbps < params.expect_bandwidth_gbps) {
        std::fprintf(stderr, "Bandwidth GB/s %.3f below expectation %.3f\n",
                     output.result.bandwidth_gbps, params.expect_bandwidth_gbps);
        ok = false;
    }
    if (params.expect_xmx_util_pct >= 0.0 && output.result.xmx_util_pct < params.expect_xmx_util_pct) {
        std::fprintf(stderr, "XMX util %.2f%% below expectation %.2f%%\n",
                     output.result.xmx_util_pct, params.expect_xmx_util_pct);
        ok = false;
    }
    return ok;
}

int main(int argc, char ** argv) {
    CmdParams params;
    if (!parse_args(argc, argv, params) || params.show_help) {
        print_usage(argv[0]);
        return params.show_help ? 0 : 1;
    }

    if (params.list_devices) {
        ggml_backend_sycl_print_sycl_devices();
        return 0;
    }

    std::vector<std::string> kernel_names;
    if (!parse_kernel_list(params.kernel_name, kernel_names)) {
        std::fprintf(stderr, "Invalid kernel list: %s\\n", params.kernel_name.c_str());
        return 1;
    }
    std::vector<const KernelInfo *> kernels;
    kernels.reserve(kernel_names.size());
    for (const auto & name : kernel_names) {
        const KernelInfo * kernel = find_kernel(name);
        if (!kernel) {
            std::fprintf(stderr, "Unknown kernel: %s\\n", name.c_str());
            return 1;
        }
        kernels.push_back(kernel);
    }
    const bool compare_mode = kernels.size() > 1 || !params.emit_json_path.empty();

    if (params.quant_type == GGML_TYPE_COUNT) {
        std::fprintf(stderr, "Unknown quant type. Use --quant=Q4_0|Q8_0|Q6_K|Q4_K|Q5_K|Q2_K|Q3_K|Q4_1|Q5_0|Q5_1|MXFP4\\n");
        return 1;
    }

    if (!params.model_path.empty()) {
        for (const auto * kernel : kernels) {
            if (kernel->kind == KernelKind::MEMORY_BANDWIDTH ||
                kernel->kind == KernelKind::ROOFLINE_COMPUTE ||
                kernel->kind == KernelKind::DPAS_EXPLORATION) {
                std::fprintf(stderr, "--model is not supported for the selected kernel kind.\\n");
                return 1;
            }
        }
        apply_sample_strategy(params.sample_strategy, params.batch_sizes);
    }

    BenchmarkHarness harness;
    bool printed_header = false;

    std::vector<ModelMatmulShape> shapes;
    if (!params.model_path.empty()) {
        std::string error;
        if (!load_model_matmul_shapes(params.model_path, shapes, error)) {
            std::fprintf(stderr, "Failed to load model shapes: %s\\n", error.c_str());
            return 1;
        }

        if (!params.model_include_fp) {
            shapes.erase(std::remove_if(shapes.begin(), shapes.end(),
                                        [](const ModelMatmulShape & s) {
                                            return !ggml_is_quantized(s.type);
                                        }),
                         shapes.end());
        }

        if (!params.model_filter.empty()) {
            const std::string needle = to_lower(params.model_filter);
            shapes.erase(std::remove_if(shapes.begin(), shapes.end(),
                                        [&](const ModelMatmulShape & s) {
                                            return to_lower(s.name).find(needle) == std::string::npos;
                                        }),
                         shapes.end());
        }

        if (params.model_dedup) {
            std::vector<ModelMatmulShape> deduped;
            deduped.reserve(shapes.size());
            for (const auto & shape : shapes) {
                bool seen = false;
                for (const auto & existing : deduped) {
                    if (existing.type == shape.type &&
                        existing.dim_m == shape.dim_m &&
                        existing.dim_k == shape.dim_k) {
                        seen = true;
                        break;
                    }
                }
                if (!seen) {
                    deduped.push_back(shape);
                }
            }
            shapes.swap(deduped);
        }

        if (kernels.size() == 1) {
            const KernelInfo * kernel = kernels.front();
            const int64_t q4_block = ggml_blck_size(GGML_TYPE_Q4_0);
            shapes.erase(std::remove_if(shapes.begin(), shapes.end(),
                                        [&](const ModelMatmulShape & s) {
                                            switch (kernel->kind) {
                                                case KernelKind::ONEDNN_WOQ_GEMM:
                                                    return s.type != GGML_TYPE_Q4_0;
                                                case KernelKind::UNIFIED_MATMUL:
                                                    return s.type != GGML_TYPE_Q4_0 ||
                                                           (q4_block > 0 && (s.dim_k % q4_block) != 0);
                                                case KernelKind::ONEDNN_FP16_GEMM:
                                                case KernelKind::ONEDNN_INT8_GEMM:
                                                case KernelKind::MMVQ:
                                                case KernelKind::MMQ:
                                                    return !ggml_is_quantized(s.type);
                                                default:
                                                    return false;
                                            }
                                        }),
                         shapes.end());
        }

        std::sort(shapes.begin(), shapes.end(),
                  [](const ModelMatmulShape & a, const ModelMatmulShape & b) {
                      const long double size_a =
                          static_cast<long double>(a.dim_m) * static_cast<long double>(a.dim_k) *
                          static_cast<long double>(a.instances);
                      const long double size_b =
                          static_cast<long double>(b.dim_m) * static_cast<long double>(b.dim_k) *
                          static_cast<long double>(b.instances);
                      if (size_a != size_b) {
                          return size_a > size_b;
                      }
                      return a.name < b.name;
                  });

        if (params.model_max_shapes > 0 && static_cast<int>(shapes.size()) > params.model_max_shapes) {
            shapes.resize(static_cast<size_t>(params.model_max_shapes));
        }

        if (shapes.empty()) {
            std::fprintf(stderr, "No tensors matched --model filters.\\n");
            return 1;
        }
        if (params.quant_type_overridden) {
            std::fprintf(stderr, "Warning: --quant is ignored when --model is provided.\\n");
        }
    } else {
        ModelMatmulShape shape{};
        shape.name = "";
        shape.type = params.quant_type;
        shape.dim_m = params.dim_m;
        shape.dim_n = params.dim_n;
        shape.dim_k = params.dim_k;
        shape.instances = 1;
        shape.n_dims = 2;
        shape.dims[0] = params.dim_k;
        shape.dims[1] = params.dim_m;
        shapes.push_back(shape);
    }

    std::vector<ShapeSummary> summaries;

    for (const auto & shape : shapes) {
        for (int64_t batch : params.batch_sizes) {
            ShapeSummary summary{};
            if (!params.emit_json_path.empty()) {
                summary.tensor = shape.name;
                summary.type = params.model_path.empty() ? params.quant_type : shape.type;
                summary.dim_m = shape.dim_m;
                summary.dim_k = shape.dim_k;
                summary.dim_n = params.model_path.empty() ? params.dim_n : batch;
                summary.instances = shape.instances;
                summary.runs.reserve(kernels.size());
            }

            for (const auto * kernel : kernels) {
                BenchmarkConfig config;
                config.kernel_name = kernel->name;
                config.quant_type = params.model_path.empty() ? params.quant_type : shape.type;
                config.layout = kernel->layout;
                config.kernel_kind = kernel->kind;
                config.tensor_name = shape.name;
                config.tensor_instances = shape.instances;
                config.batch_size = batch;
                config.dim_m = shape.dim_m;
                config.dim_n = params.model_path.empty() ? params.dim_n : batch;
                config.dim_k = shape.dim_k;
                config.warmup_iterations = params.warmup;
                config.measure_iterations = params.iterations;
                config.memory_mode = params.memory_mode;
                config.validate = params.validate;
                config.include_percentiles = params.include_percentiles;
                config.include_ref_metrics =
                    params.include_ref_metrics || (kernel->kind != KernelKind::MMVQ && kernel->kind != KernelKind::MMQ);
                config.transfer_bytes = params.transfer_bytes;
                config.roofline_elements = params.roofline_elements;
                config.roofline_ops = params.roofline_ops;
                config.abs_tol = params.abs_tol;
                config.rel_tol = params.rel_tol;
                config.device_id = params.device_id;
                config.dpas_config_name = params.dpas_config_name;
                config.dpas_type_a = params.dpas_type_a;
                config.dpas_type_b = params.dpas_type_b;
                config.dpas_type_acc = params.dpas_type_acc;
                config.dpas_memory_pattern = params.dpas_memory_pattern;
                config.dpas_grf_mode = params.dpas_grf_mode;
                config.dpas_repeat = params.dpas_repeat;
                config.dpas_n_tile_repeats = params.dpas_n_tile_repeats;
                config.dpas_misaligned = params.dpas_misaligned;
                config.dpas_device_opt = params.dpas_device_opt;
                config.dpas_autotune = params.dpas_autotune;
                config.dpas_autotune_force = params.dpas_autotune_force;
                config.dpas_autotune_metric = params.dpas_autotune_metric;
                config.dpas_autotune_cache = params.dpas_autotune_cache;
                config.dpas_autotune_override_ntiles = params.dpas_autotune_override_ntiles;
                config.dpas_autotune_override_prefetch = params.dpas_autotune_override_prefetch;
                config.dpas_memory_explicit = params.dpas_memory_explicit;
                config.dpas_ntiles_explicit = params.dpas_ntiles_explicit;
                config.dpas_grf_explicit = params.dpas_grf_explicit;
                config.dpas_acc_explicit = params.dpas_acc_explicit;

                BenchmarkOutput output;
                if (!harness.run(config, output)) {
                    const bool skippable = compare_mode
                        ? is_skippable_error(output.error)
                        : (output.error.rfind("SKIP:", 0) == 0);
                    if (skippable) {
                        std::fprintf(stderr, "Benchmark skipped: %s\\n", output.error.c_str());
                        if (!params.emit_json_path.empty()) {
                            KernelRunSummary run{};
                            run.kernel = kernel->name;
                            run.ok = false;
                            run.error = output.error;
                            summary.runs.push_back(std::move(run));
                        }
                        continue;
                    }
                    if (compare_mode) {
                        std::fprintf(stderr, "Benchmark failed: %s\\n", output.error.c_str());
                        if (!params.emit_json_path.empty()) {
                            KernelRunSummary run{};
                            run.kernel = kernel->name;
                            run.ok = false;
                            run.error = output.error;
                            summary.runs.push_back(std::move(run));
                        }
                        continue;
                    }
                    std::fprintf(stderr, "Benchmark failed: %s\\n", output.error.c_str());
                    return 1;
                }
                if (params.xmx_peak_tops > 0.0 && output.result.throughput_tops > 0.0) {
                    output.result.xmx_util_pct = (output.result.throughput_tops / params.xmx_peak_tops) * 100.0;
                }
                if (!check_expectations(params, output)) {
                    return 2;
                }

                if (!params.emit_json_path.empty()) {
                    KernelRunSummary run{};
                    run.kernel = kernel->name;
                    run.ok = true;
                    run.error.clear();
                    run.latency_us = output.result.latency_us;
                    run.throughput_tps = output.result.throughput_tps;
                    run.throughput_tops = output.result.throughput_tops;
                    run.bandwidth_gbps = output.result.bandwidth_gbps;
                    run.xmx_util_pct = output.result.xmx_util_pct;
                    summary.runs.push_back(std::move(run));
                }

                switch (params.output_format) {
                    case OutputFormat::CSV:
                        if (!printed_header) {
                            print_csv_header(stdout, config);
                            printed_header = true;
                        }
                        print_csv_row(stdout, output);
                        break;
                    case OutputFormat::JSON:
                        print_json(stdout, output);
                        break;
                    case OutputFormat::JSONL:
                        print_jsonl(stdout, output);
                        break;
                }
            }

            if (!params.emit_json_path.empty()) {
                summary.winner = pick_winner(summary);
                summaries.push_back(std::move(summary));
            }
        }
    }

    write_summary_json(params.emit_json_path, summaries);
    return 0;
}
