//
// tessera-higgs-proxy-main.cpp
//
// CLI entry point for the HIGGS structural proxy estimator. Reads
// a GGUF, computes per-tensor alpha_l, writes the
// ane.alpha-coefficients.v1 sidecar JSON, and emits a human-
// readable markdown report.
//
// The CLI surface is the same as the Python
// tools/tessera/estimate_higgs_alpha.py wrapper. The Python module
// is a thin subprocess wrapper around this binary when present on
// PATH; it falls back to the in-process NumPy path when this
// binary is not built (the dev / test path).
//
// Exit codes: 0 on success, non-zero on GGUF read failure / IO
// error / argument error.
//

#include "tessera-higgs-proxy.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void print_usage(const char * prog) {
    std::printf("usage: %s --gguf <path> --output <path> [options]\n", prog);
    std::printf("\n");
    std::printf("HIGGS per-layer alpha_l estimator (C++ structural proxy).\n");
    std::printf("Produces the ane.alpha-coefficients.v1 sidecar consumed by\n");
    std::printf("the iOS dispatch's L1 path. The CLI surface matches the\n");
    std::printf("Python wrapper at tools/tessera/estimate_higgs_alpha.py.\n");
    std::printf("\n");
    std::printf("Options:\n");
    std::printf("  --gguf <path>                 source GGUF (required)\n");
    std::printf("  --output <path>               sidecar JSON output (required)\n");
    std::printf("  --report <path>               markdown report output (default: <sidecar-stem>.report.md)\n");
    std::printf("  --bundle-name <name>          bundle name in the sidecar (default: <gguf-stem>)\n");
    std::printf("  --min-params-for-estimate <N> parameter-count threshold for the uniform-fallback gate (default: 1000000000)\n");
    std::printf("  --alpha-floor-fraction <F>    positive floor as a fraction of post-normalization mean (default: 0.001)\n");
    std::printf("  --alpha-floor <F>             absolute floor (default: 1e-6)\n");
    std::printf("  --verbose                     per-tensor summary to stderr\n");
    std::printf("\n");
    std::printf("The default t_l^2 measurement is the L1-on-ANE kernel dequant\n");
    std::printf("path (t_squared_source=l1_kernel_dequant): the per-element L1\n");
    std::printf("distance between the fp32 weight and the ANE-dequantized fp16\n");
    std::printf("weight via the TILE640 matmul dispatch. Set the environment\n");
    std::printf("variable TS_HIGGS_PROXY_LEGACY_OFFLINE=1 to restore the legacy\n");
    std::printf("offline ternary MSE proxy. See docs/tessera-higgs-estimator.md.\n");
}

int main(int argc, char ** argv) {
    std::string gguf_path;
    std::string output_path;
    std::string report_path;
    std::string bundle_name;

    ts_higgs_proxy_params p;
    p.J = 1;
    p.t_min = 0.01f;
    p.t_max = 0.10f;
    p.alpha_floor = 1e-6f;
    p.seed = 42u;
    p.verbose = false;
    p.min_params_for_estimate = 1000000000LL;
    p.alpha_floor_fraction = 1e-3f;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (a == "--gguf" && i + 1 < argc) {
            gguf_path = argv[++i];
        } else if (a == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (a == "--report" && i + 1 < argc) {
            report_path = argv[++i];
        } else if (a == "--bundle-name" && i + 1 < argc) {
            bundle_name = argv[++i];
        } else if (a == "--min-params-for-estimate" && i + 1 < argc) {
            p.min_params_for_estimate = std::strtoll(argv[++i], nullptr, 10);
        } else if (a == "--alpha-floor-fraction" && i + 1 < argc) {
            p.alpha_floor_fraction = std::strtof(argv[++i], nullptr);
        } else if (a == "--alpha-floor" && i + 1 < argc) {
            p.alpha_floor = std::strtof(argv[++i], nullptr);
        } else if (a == "--verbose") {
            p.verbose = true;
        } else {
            std::fprintf(stderr, "tessera-higgs-proxy: unknown argument: %s\n",
                         a.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }

    if (gguf_path.empty()) {
        std::fprintf(stderr, "tessera-higgs-proxy: --gguf is required\n");
        print_usage(argv[0]);
        return 2;
    }
    if (output_path.empty()) {
        std::fprintf(stderr, "tessera-higgs-proxy: --output is required\n");
        print_usage(argv[0]);
        return 2;
    }
    if (p.min_params_for_estimate < 0) {
        std::fprintf(stderr,
            "tessera-higgs-proxy: --min-params-for-estimate must be >= 0\n");
        return 2;
    }
    if (!(p.alpha_floor_fraction > 0.0f) || p.alpha_floor_fraction >= 1.0f) {
        std::fprintf(stderr,
            "tessera-higgs-proxy: --alpha-floor-fraction must be in (0, 1)\n");
        return 2;
    }

    ts_higgs_proxy_result result;
    int rc = ts_higgs_proxy_estimate(gguf_path.c_str(), &p, nullptr, nullptr,
                                     &result);
    if (rc != 0) {
        std::fprintf(stderr, "tessera-higgs-proxy: estimate failed (rc=%d)\n", rc);
        return 1;
    }

    std::string json = ts_higgs_proxy_to_json(
        &result, gguf_path.c_str(),
        bundle_name.empty() ? nullptr : bundle_name.c_str());
    rc = ts_higgs_proxy_write_json_atomic(output_path.c_str(), json);
    if (rc != 0) {
        std::fprintf(stderr, "tessera-higgs-proxy: sidecar write failed (rc=%d)\n",
                     rc);
        return 1;
    }

    // Markdown report. We construct the report in-memory here
    // (the report format mirrors the Python write_report).
    std::string report;
    report += "# HIGGS per-layer alpha report: ";
    if (!bundle_name.empty()) {
        report += bundle_name;
    } else {
        size_t slash = gguf_path.find_last_of("/\\");
        std::string stem = (slash == std::string::npos) ? gguf_path
                                                        : gguf_path.substr(slash + 1);
        size_t dot = stem.find_last_of('.');
        report += (dot == std::string::npos) ? stem : stem.substr(0, dot);
    }
    report += "\n\n";
    char buf[1024];
    std::snprintf(buf, sizeof(buf), "- **Model hash**: `%s`\n",
                  result.model_hash.c_str());
    report += buf;
    std::snprintf(buf, sizeof(buf), "- **Source GGUF**: `%s`\n",
                  gguf_path.c_str());
    report += buf;
    report += "- **Schema**: `ane.alpha-coefficients.v1` v1\n";
    report += "- **Fitness form**: `Sum_l alpha_l * t_l^2`\n";
    std::snprintf(buf, sizeof(buf), "- **Measurement source**: `%s`\n",
                  result.t_squared_source.c_str());
    report += buf;
    report += "- **Regime gate**: min_operating_bits=3.0, qep_off_switch=true\n";
    int64_t total_params = 0;
    for (const auto & lr : result.layers) total_params += lr.n_elem;
    bool fallback_global = (result.t_squared_source == "uniform_fallback");
    if (fallback_global) {
        std::snprintf(buf, sizeof(buf),
            "- **Global uniform fallback**: yes (model parameter count below threshold)\n");
        report += buf;
    } else {
        report += "- **Global uniform fallback**: no\n";
    }
    std::snprintf(buf, sizeof(buf),
        "- **Total parameters (counted tensors)**: %ld\n",
        (long)total_params);
    report += buf;
    std::snprintf(buf, sizeof(buf), "- **Layer count**: %zu\n",
                  result.layers.size());
    report += buf;
    report += "\n## Per-tensor results\n\n";
    report += "| Tensor | Family | Shape | t^2 | alpha | floor | fallback |\n";
    report += "|---|---|---|---:|---:|---|---|\n";
    for (const auto & lr : result.layers) {
        std::string shape = "[";
        for (size_t d = 0; d < lr.shape.size(); d++) {
            if (d > 0) shape += ", ";
            shape += std::to_string(lr.shape[d]);
        }
        shape += "]";
        std::snprintf(buf, sizeof(buf),
            "| `%s` | %s | %s | %.6e | %.6e | %s | %s |\n",
            lr.name.c_str(), lr.family.c_str(), shape.c_str(),
            (double)lr.t_squared, (double)lr.alpha_l,
            lr.alpha_floor_applied ? "yes" : "no", lr.fallback.c_str());
        report += buf;
    }

    if (report_path.empty()) {
        size_t dot = output_path.find_last_of('.');
        report_path = (dot == std::string::npos) ? (output_path + ".report.md")
                                                 : output_path.substr(0, dot) + ".report.md";
    }
    {
        std::ofstream f(report_path, std::ios::out | std::ios::trunc);
        if (f) {
            f.write(report.data(), (std::streamsize)report.size());
            f.flush();
        } else {
            std::fprintf(stderr, "tessera-higgs-proxy: report open %s failed\n",
                         report_path.c_str());
            return 1;
        }
    }

    // Terse run summary to stderr. The sidecar is the durable
    // artifact; this is just the run summary.
    std::fprintf(stderr, "wrote %s\n", output_path.c_str());
    std::fprintf(stderr, "  model_hash:      %s\n", result.model_hash.c_str());
    std::fprintf(stderr, "  layer_count:     %zu\n", result.layers.size());
    std::fprintf(stderr, "  total_params:    %ld\n", (long)total_params);
    std::fprintf(stderr, "  measurement:     %s\n", result.t_squared_source.c_str());
    std::fprintf(stderr, "  fallback_global: %s\n", fallback_global ? "true" : "false");
    std::fprintf(stderr, "  mean_alpha:      %.6e\n", (double)result.mean_alpha);
    std::fprintf(stderr, "wrote %s\n", report_path.c_str());

    return 0;
}
