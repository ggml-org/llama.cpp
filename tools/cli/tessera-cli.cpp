// tessera-cli.cpp - lightweight CLI shim for the Tessera Studio Swift app.
//
// A small command-line frontend over the same C++ engine modules the FFI
// surface (TesseraStudio/ffi/tessera_ffi.cpp) wraps. TesseraEngineBridge.swift
// shells out to this binary per generation call when the FFI path is
// unavailable for a given subcommand (model load, perplexity forward pass,
// coreml conversion - those need a real model context that the FFI does
// not keep alive across calls).
//
// All subcommands print JSON to stdout and human-readable errors to stderr.
// Exit codes: 0 success, 1 "valid request but engine path is not runnable
// via the CLI shell-out; try the FFI or the canonical llama-tessera entry
// point", 2 bad arguments, -1 hard error.

#include "tessera-dispatch.h"
#include "tessera-imatrix.h"
#include "tessera-corpus.h"
#include "tessera-awq.h"
#include "tessera-coreml-builder.h"
#include "tessera-sidecar-v3.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>

using json = nlohmann::json;

namespace {

void print_usage(const char * prog) {
    fprintf(stderr,
        "usage: %s <subcommand> [args]\n"
        "\n"
        "Subcommands:\n"
        "  version\n"
        "      Print the engine version string and exit.\n"
        "\n"
        "  list-models <dir>\n"
        "      Scan <dir> for .gguf files; print a JSON array of {name, path, size_bytes}.\n"
        "\n"
        "  inspect-sidecar <path>\n"
        "      Parse a v3 sidecar (TDQT magic) at <path>; print its header + per-row counts.\n"
        "\n"
        "  quantize <model> <output> [--config <json>]\n"
        "      Run the Tessera quantize pipeline end-to-end. Wraps ts_dispatch_run with\n"
        "      evolve_only=false, calibrate_only=false. The optional --config JSON object\n"
        "      fills ts_dispatch_params (imatrix_path, policy_path, evolve_seed, ...).\n"
        "\n"
        "  calibrate <corpus>\n"
        "      Load a calibration source. <corpus> ending in .npz is loaded via\n"
        "      ts_imatrix_load_npz; a directory is loaded via ts_corpus_load_directory.\n"
        "      Prints a one-line summary on stdout.\n"
        "\n"
        "  evolve <model> [--config <json>]\n"
        "      Run the AWQ-evolve GA. Currently parses and validates the config and\n"
        "      exits 1 - the per-layer GA worker needs the full dispatch pipeline\n"
        "      (llama-quantize --tessera-evolve) for the tensor family descriptor\n"
        "      machinery. FFI path is the canonical entry point for this once\n"
        "      tessera_ffi_evolve lands; the shell-out here is a placeholder.\n"
        "\n"
        "  evaluate <model> [--config <json>]\n"
        "      Run a perplexity forward pass. Same placeholder caveat as evolve;\n"
        "      the canonical entry is `llama-quantize --tessera-ppl <model>`.\n"
        "\n"
        "  convert <model> <output> --format <fmt>\n"
        "      Convert a Tessera GGUF to <fmt> (only \"coreml\" is recognized).\n"
        "      Placeholder; the CoreML builder needs dequantized weight tensors\n"
        "      which require a model context.\n"
        "\n"
        "Exit codes: 0 success, 1 fallback (try FFI / canonical binary), 2 bad args, -1 hard error.\n",
        prog);
}

// ----- shared helpers -----

int fail_args(const char * msg) {
    fprintf(stderr, "tessera-cli: %s\n", msg);
    return 2;
}

json error_json(const std::string & op, const std::string & msg) {
    json j;
    j["ok"]        = false;
    j["operation"] = op;
    j["error"]     = msg;
    return j;
}

json ok_json(const std::string & op) {
    json j;
    j["ok"]        = true;
    j["operation"] = op;
    return j;
}

// pull a flag value out of argv, return true on success. *out is the
// remaining arg; flag must appear in argv[pos] and argv[pos+1] must exist.
bool take_arg(int argc, char ** argv, int & pos, const char * flag, std::string & out) {
    if (std::strcmp(argv[pos], flag) != 0) {
        return false;
    }
    if (pos + 1 >= argc) {
        fprintf(stderr, "tessera-cli: %s requires a value\n", flag);
        std::exit(2);
    }
    out = argv[++pos];
    return true;
}

bool is_flag(const char * a) {
    return a && a[0] == '-' && a[1] != '\0';
}

bool ends_with(const std::string & s, const std::string & suf) {
    return s.size() >= suf.size() &&
           s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// ----- subcommand implementations -----

int cmd_version() {
    json j;
    j["ok"]      = true;
    j["version"] = "tessera-1.0.0-cpp";
    printf("%s\n", j.dump().c_str());
    return 0;
}

int cmd_list_models(const std::string & dir) {
    json j = ok_json("list-models");
    json arr = json::array();

    DIR * d = opendir(dir.c_str());
    if (d) {
        struct dirent * ent;
        while ((ent = readdir(d)) != nullptr) {
            std::string name(ent->d_name);
            if (!ends_with(name, ".gguf")) continue;
            std::string full = dir + "/" + name;
            struct stat st;
            json entry;
            entry["name"] = name;
            entry["path"] = full;
            if (stat(full.c_str(), &st) == 0) {
                entry["size_bytes"] = static_cast<int64_t>(st.st_size);
            } else {
                entry["size_bytes"] = nullptr;
            }
            arr.push_back(std::move(entry));
        }
        closedir(d);
    }
    j["models"] = std::move(arr);
    j["dir"]    = dir;
    printf("%s\n", j.dump().c_str());
    return 0;
}

int cmd_inspect_sidecar(const std::string & path) {
    ts_sidecar_v3 sc;
    std::string err;
    if (ts_sidecar_v3_read(path.c_str(), &sc, &err) != 0) {
        json j = error_json("inspect-sidecar",
                            err.empty() ? "read failed" : err);
        printf("%s\n", j.dump().c_str());
        return -1;
    }
    json j = ok_json("inspect-sidecar");
    j["path"]                = path;
    j["magic"]               = sc.header.magic;
    j["version"]             = sc.header.version;
    j["rows"]                = sc.header.rows;
    j["cols"]                = sc.header.cols;
    j["dtype"]               = sc.header.dtype;
    j["outlier_threshold"]   = sc.header.outlier_threshold;
    j["outlier_count_total"] = sc.header.outlier_count_total;
    j["n_row_meta"]          = static_cast<int64_t>(sc.row_meta.size());
    j["data_elements"]       = static_cast<int64_t>(sc.data.size());

    json counts = json::array();
    const int64_t n = std::min(static_cast<int64_t>(sc.row_outlier_counts.size()),
                               static_cast<int64_t>(256));
    for (int64_t i = 0; i < n; i++) {
        counts.push_back(sc.row_outlier_counts[i]);
    }
    j["row_outlier_counts"] = std::move(counts);
    printf("%s\n", j.dump().c_str());
    return 0;
}

// Fill a ts_dispatch_params from a config_json string. Mirrors the parsing
// in tessera_ffi.cpp:quantize() so the FFI and CLI share a config schema.
// Returns 0 on success, -1 on malformed JSON.
int parse_dispatch_config(const std::string & cfg_json, ts_dispatch_params & p) {
    try {
        json cfg = json::parse(cfg_json);
        auto str = [&](const char * k, std::string & dst) {
            if (cfg.contains(k) && cfg[k].is_string()) dst = cfg[k].get<std::string>();
        };
        auto i64 = [&](const char * k, uint64_t & dst) {
            if (cfg.contains(k) && cfg[k].is_number()) dst = cfg[k].get<uint64_t>();
        };
        auto i32 = [&](const char * k, int & dst) {
            if (cfg.contains(k) && cfg[k].is_number()) dst = cfg[k].get<int>();
        };
        auto f32 = [&](const char * k, float & dst) {
            if (cfg.contains(k) && cfg[k].is_number()) dst = cfg[k].get<float>();
        };
        auto boo = [&](const char * k, bool & dst) {
            if (cfg.contains(k) && cfg[k].is_boolean()) dst = cfg[k].get<bool>();
        };
        str("imatrix_path",     p.imatrix_path);
        str("policy_path",      p.policy_path);
        str("policy_out_path",  p.policy_out_path);
        str("calib_corpus",     p.calib_corpus);
        str("higgs_alpha_mode", p.higgs_alpha_mode);
        str("higgs_cache_dir",  p.higgs_cache_dir);
        str("awq_alpha",        p.awq_alpha);
        i64("evolve_seed",       p.evolve_seed);
        i32("evolve_iters",      p.evolve_iters);
        i32("evolve_islands",    p.evolve_islands);
        i32("evolve_population", p.evolve_population);
        boo("evolve_only",       p.evolve_only);
        boo("calibrate_only",    p.calibrate_only);
        f32("outlier_frac",      p.outlier_frac);
        f32("awq_clip",          p.awq_clip);
        i32("nthreads",          p.nthreads);
        boo("verbose",           p.verbose);
    } catch (const json::exception &) {
        return -1;
    }
    return 0;
}

int cmd_quantize(int argc, char ** argv, int pos) {
    // quantize <model> <output> [--config <json>]
    if (pos + 1 >= argc) {
        return fail_args("quantize requires <model> and <output>");
    }
    const std::string model  = argv[pos];
    const std::string output = argv[pos + 1];
    pos += 2;

    std::string cfg_json;
    while (pos < argc) {
        if (!take_arg(argc, argv, pos, "--config", cfg_json)) {
            return fail_args("unknown quantize flag (only --config is supported)");
        }
        pos++;
    }

    ts_dispatch_params p;
    p.input_path        = model;
    p.output_path       = output;
    p.higgs_alpha_mode  = "uniform";
    p.evolve_seed       = 42;
    p.evolve_iters      = 100;
    p.evolve_islands    = 4;
    p.evolve_population = 32;
    p.evolve_only       = false;
    p.calibrate_only    = false;
    p.outlier_frac      = 0.01f;
    p.awq_alpha         = "auto";
    p.awq_clip          = 0.0f;
    p.nthreads          = 4;
    p.verbose           = false;

    if (!cfg_json.empty() && parse_dispatch_config(cfg_json, p) != 0) {
        json j = error_json("quantize", "malformed --config JSON");
        printf("%s\n", j.dump().c_str());
        return -1;
    }

    ts_dispatch_result res;
    std::string err;
    const int rc = ts_dispatch_run(&p, &res, &err);
    json j;
    j["ok"]             = (rc == 0);
    j["operation"]      = "quantize";
    j["model"]          = model;
    j["output"]         = output;
    j["n_tensors_quant"] = static_cast<int64_t>(res.n_tensors_quantized);
    j["n_tensors_skip"] = static_cast<int64_t>(res.n_tensors_skipped);
    j["total_mse"]      = res.total_mse;
    j["policy_sha256"]  = res.policy_sha256;
    if (rc != 0) {
        j["error"] = err.empty() ? "ts_dispatch_run failed" : err;
    }
    printf("%s\n", j.dump().c_str());
    return rc == 0 ? 0 : -1;
}

int cmd_calibrate(int argc, char ** argv, int pos) {
    // calibrate <corpus>
    if (pos >= argc) {
        return fail_args("calibrate requires a path");
    }
    const std::string path = argv[pos];

    std::string err;
    if (ends_with(path, ".npz")) {
        ts_imatrix imx;
        const int rc = ts_imatrix_load_npz(path.c_str(), &imx, &err);
        json j;
        j["ok"]         = (rc == 0);
        j["operation"]  = "calibrate";
        j["kind"]       = "imatrix";
        j["path"]       = path;
        j["n_tensors"]  = static_cast<int64_t>(imx.data.size());
        j["source"]     = imx.source_path;
        if (rc != 0) {
            j["error"] = err.empty() ? "ts_imatrix_load_npz failed" : err;
            printf("%s\n", j.dump().c_str());
            return -1;
        }
        printf("%s\n", j.dump().c_str());
        return 0;
    }

    int64_t n_tokens = 0;
    int64_t in_dim   = 0;
    std::vector<float> data = ts_corpus_load_directory(path.c_str(), &n_tokens, &in_dim, &err);
    json j;
    j["ok"]        = !data.empty() || n_tokens > 0;
    j["operation"] = "calibrate";
    j["kind"]      = "corpus";
    j["path"]      = path;
    j["n_tokens"]  = n_tokens;
    j["in_dim"]    = in_dim;
    if (!j["ok"].get<bool>()) {
        j["error"] = err.empty() ? "ts_corpus_load_directory returned no data" : err;
        printf("%s\n", j.dump().c_str());
        return -1;
    }
    printf("%s\n", j.dump().c_str());
    return 0;
}

int cmd_evolve(int argc, char ** argv, int pos) {
    // evolve <model> [--config <json>]
    if (pos >= argc) {
        return fail_args("evolve requires <model>");
    }
    const std::string model = argv[pos];
    pos++;

    std::string cfg_json;
    while (pos < argc) {
        if (!take_arg(argc, argv, pos, "--config", cfg_json)) {
            return fail_args("unknown evolve flag (only --config is supported)");
        }
        pos++;
    }

    // Parse + validate the config the same way tessera_ffi.cpp:evolve() does.
    ts_awq_evolve_params p;
    p.population         = 32;
    p.generations        = 100;
    p.islands            = 4;
    p.migration_interval = 10;
    p.mutation_sigma     = 0.1f;
    p.crossover_rate     = 0.7f;
    p.heldout_weight     = 2.0f;
    p.seed               = 42;
    p.verbose            = false;

    if (!cfg_json.empty()) {
        try {
            json cfg = json::parse(cfg_json);
            if (cfg.contains("population")         && cfg["population"].is_number())         p.population         = cfg["population"].get<int64_t>();
            if (cfg.contains("generations")        && cfg["generations"].is_number())        p.generations        = cfg["generations"].get<int64_t>();
            if (cfg.contains("islands")            && cfg["islands"].is_number())            p.islands            = cfg["islands"].get<int64_t>();
            if (cfg.contains("migration_interval") && cfg["migration_interval"].is_number()) p.migration_interval = cfg["migration_interval"].get<int64_t>();
            if (cfg.contains("mutation_sigma")     && cfg["mutation_sigma"].is_number())     p.mutation_sigma     = cfg["mutation_sigma"].get<float>();
            if (cfg.contains("crossover_rate")     && cfg["crossover_rate"].is_number())     p.crossover_rate     = cfg["crossover_rate"].get<float>();
            if (cfg.contains("heldout_weight")     && cfg["heldout_weight"].is_number())     p.heldout_weight     = cfg["heldout_weight"].get<float>();
            if (cfg.contains("seed")               && cfg["seed"].is_number())               p.seed               = cfg["seed"].get<uint32_t>();
            if (cfg.contains("verbose")            && cfg["verbose"].is_boolean())           p.verbose            = cfg["verbose"].get<bool>();
        } catch (const json::exception &) {
            json j = error_json("evolve", "malformed --config JSON");
            printf("%s\n", j.dump().c_str());
            return -1;
        }
    }

    // The per-layer GA needs the full dispatch's tensor-descriptor walk
    // (regime descriptors, act_scales, family warm-start, streaming weight
    // load). That is the canonical path through llama-tessera's quantize
    // flow; the FFI does not yet expose tessera_ffi_evolve either, and
    // TesseraEngineBridge.swift falls back to this binary for evolve calls.
    // Until the FFI gets a real evolve path, return 1 + a structured error
    // JSON so the Swift layer can present a clean message to the user.
    json j = error_json("evolve",
        "evolve is not runnable from the tessera-cli shell-out yet; "
        "use: llama-tessera <model> TESSERA_T640 --tessera-evolve "
        "or the Studio 'Quantize' action");
    j["model"] = model;
    j["population"]         = p.population;
    j["generations"]        = p.generations;
    j["islands"]            = p.islands;
    j["migration_interval"] = p.migration_interval;
    printf("%s\n", j.dump().c_str());
    return 1;
}

int cmd_evaluate(int argc, char ** argv, int pos) {
    // evaluate <model> [--config <json>]
    if (pos >= argc) {
        return fail_args("evaluate requires <model>");
    }
    const std::string model = argv[pos];
    pos++;

    std::string cfg_json;
    while (pos < argc) {
        if (!take_arg(argc, argv, pos, "--config", cfg_json)) {
            return fail_args("unknown evaluate flag (only --config is supported)");
        }
        pos++;
    }

    // Same caveat as evolve: ts_ppl_probe needs a real model forward pass,
    // which is the dispatch / llama-quantize --tessera-ppl path. The FFI
    // returns a structured error JSON too; we mirror that contract here.
    json j = error_json("evaluate",
        "perplexity evaluation requires a loaded model forward pass; "
        "use: llama-quantize --tessera-ppl <model>");
    j["model"] = model;
    printf("%s\n", j.dump().c_str());
    return 0;
}

int cmd_convert(int argc, char ** argv, int pos) {
    // convert <model> <output> --format <fmt>
    if (pos + 1 >= argc) {
        return fail_args("convert requires <model> and <output>");
    }
    const std::string model  = argv[pos];
    const std::string output = argv[pos + 1];
    pos += 2;

    std::string fmt;
    while (pos < argc) {
        if (!take_arg(argc, argv, pos, "--format", fmt)) {
            return fail_args("unknown convert flag (only --format is supported)");
        }
        pos++;
    }
    if (fmt.empty()) {
        return fail_args("convert requires --format");
    }
    if (fmt != "coreml") {
        json j = error_json("convert", "unsupported --format (only \"coreml\" is recognized)");
        j["format"] = fmt;
        printf("%s\n", j.dump().c_str());
        return 2;
    }

    // CoreML conversion needs dequantized weight tensors via
    // ts_coreml_builder_tensor, which require a model context. The FFI
    // returns 1 here; mirror that contract.
    json j = error_json("convert",
        "CoreML conversion is not runnable from the tessera-cli shell-out yet; "
        "use the Studio 'Convert' action or the FFI once tessera_ffi_convert lands");
    j["model"]  = model;
    j["output"] = output;
    j["format"] = fmt;
    printf("%s\n", j.dump().c_str());
    return 1;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 2;
    }
    const std::string sub = argv[1];

    if (sub == "-h" || sub == "--help" || sub == "help") {
        print_usage(argv[0]);
        return 0;
    }
    if (sub == "version") {
        return cmd_version();
    }
    if (sub == "list-models") {
        if (argc < 3) return fail_args("list-models requires <dir>");
        return cmd_list_models(argv[2]);
    }
    if (sub == "inspect-sidecar") {
        if (argc < 3) return fail_args("inspect-sidecar requires <path>");
        return cmd_inspect_sidecar(argv[2]);
    }
    if (sub == "quantize") {
        return cmd_quantize(argc, argv, 2);
    }
    if (sub == "calibrate") {
        return cmd_calibrate(argc, argv, 2);
    }
    if (sub == "evolve") {
        return cmd_evolve(argc, argv, 2);
    }
    if (sub == "evaluate") {
        return cmd_evaluate(argc, argv, 2);
    }
    if (sub == "convert") {
        return cmd_convert(argc, argv, 2);
    }

    fprintf(stderr, "tessera-cli: unknown subcommand: %s\n", sub.c_str());
    print_usage(argv[0]);
    return 2;
}
