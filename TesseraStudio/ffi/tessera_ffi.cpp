// tessera_ffi.cpp - real Tessera engine FFI implementation.
//
// Implements the C API declared in include/tessera_ffi.h by calling the
// C++ engine modules compiled into llama-quantize-impl.
//
// Two paths share this file:
//
//   1. No-handle path (tessera_quantize, tessera_calibrate,
//      tessera_evolve, tessera_evaluate, tessera_convert). The first two
//      are fully implemented; the rest parse and validate the input and
//      return a structured "use the CLI" marker so the Swift layer can
//      fall back when no model context is available.
//
//   2. Model-context path (tessera_load_model, tessera_free_model,
//      tessera_evolve_model, tessera_evaluate_model,
//      tessera_convert_model). The first two are fully implemented - they
//      wrap llama.cpp's llama_model_load_from_file / llama_model_free.
//      The *_model() variants validate the handle, parse the JSON config,
//      and either run the engine or return a structured
//      "TODO: requires engine impl ..." JSON so the caller can fall back.
//      The TODO markers live in the *_model() bodies; grep for
//      "TODO: requires engine impl" to find the wiring work.

#include "include/tessera_ffi.h"

#include "tessera-dispatch.h"
#include "tessera-imatrix.h"
#include "tessera-corpus.h"
#include "tessera-awq.h"
#include "tessera-coreml-builder.h"
#include "tessera-sidecar-v3.h"
#include "tessera-ppl.h"

#include "llama.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <new>

#include <dirent.h>
#include <sys/stat.h>
#include <cstdio>

using json = nlohmann::json;

// duplicate a std::string into a malloc'd C string (caller frees via tessera_free_string)
static char * ts_ffi_strdup(const std::string & s) {
    char * p = static_cast<char *>(malloc(s.size() + 1));
    if (p) {
        memcpy(p, s.c_str(), s.size() + 1);
    }
    return p;
}

static std::string ts_ffi_error(const char * op, const std::string & msg) {
    json j;
    j["ok"]        = false;
    j["operation"] = op;
    j["error"]     = msg;
    return j.dump();
}

extern "C" {

int tessera_ffi_is_available(void) {
    return 1;
}

const char * tessera_ffi_version(void) {
    return "tessera-1.0.0-cpp";
}

// quantize: ts_dispatch_run loads the GGUF itself, so this is fully callable.
int tessera_quantize(const char * model_path,
                     const char * output_path,
                     const char * config_json) {
    if (!model_path || !output_path) {
        return -1;
    }

    ts_dispatch_params params;
    params.input_path        = model_path;
    params.output_path       = output_path;
    params.higgs_alpha_mode  = "uniform";
    params.evolve_seed       = 42;
    params.evolve_iters      = 100;
    params.evolve_islands    = 4;
    params.evolve_population = 32;
    params.evolve_only       = false;
    params.calibrate_only    = false;
    params.outlier_frac      = 0.01f;
    params.awq_alpha         = "auto";
    params.awq_clip          = 0.0f;
    params.nthreads          = 4;
    params.verbose           = false;

    if (config_json && config_json[0]) {
        try {
            json cfg = json::parse(config_json);
            auto str = [&](const char * key, std::string & dst) {
                if (cfg.contains(key)) dst = cfg[key].get<std::string>();
            };
            auto i64 = [&](const char * key, uint64_t & dst) {
                if (cfg.contains(key)) dst = cfg[key].get<uint64_t>();
            };
            auto i32 = [&](const char * key, int & dst) {
                if (cfg.contains(key)) dst = cfg[key].get<int>();
            };
            auto f32 = [&](const char * key, float & dst) {
                if (cfg.contains(key)) dst = cfg[key].get<float>();
            };
            auto boo = [&](const char * key, bool & dst) {
                if (cfg.contains(key)) dst = cfg[key].get<bool>();
            };
            str("imatrix_path",     params.imatrix_path);
            str("policy_path",      params.policy_path);
            str("policy_out_path",  params.policy_out_path);
            str("calib_corpus",     params.calib_corpus);
            str("higgs_alpha_mode", params.higgs_alpha_mode);
            str("higgs_cache_dir",  params.higgs_cache_dir);
            str("awq_alpha",        params.awq_alpha);
            i64("evolve_seed",       params.evolve_seed);
            i32("evolve_iters",      params.evolve_iters);
            i32("evolve_islands",    params.evolve_islands);
            i32("evolve_population", params.evolve_population);
            boo("evolve_only",       params.evolve_only);
            boo("calibrate_only",    params.calibrate_only);
            f32("outlier_frac",      params.outlier_frac);
            f32("awq_clip",          params.awq_clip);
            i32("nthreads",          params.nthreads);
            boo("verbose",           params.verbose);
        } catch (const json::exception &) {
            return -2; // malformed config_json
        }
    }

    ts_dispatch_result result;
    std::string err;
    return ts_dispatch_run(&params, &result, &err);
}

// calibrate: load an imatrix .npz or a corpus directory. Both are pure I/O.
int tessera_calibrate(const char * model_path,
                      const char * corpus_path,
                      const char * config_json) {
    (void)model_path;
    (void)config_json;
    if (!corpus_path) {
        return -1;
    }

    std::string err;
    std::string path(corpus_path);

    if (path.size() >= 4 && path.compare(path.size() - 4, 4, ".npz") == 0) {
        ts_imatrix imx;
        return ts_imatrix_load_npz(corpus_path, &imx, &err);
    }

    int64_t n_tokens = 0;
    int64_t in_dim   = 0;
    std::vector<float> data = ts_corpus_load_directory(corpus_path, &n_tokens, &in_dim, &err);
    if (data.empty() && n_tokens == 0) {
        return -1;
    }
    return 0;
}

// evolve: the GA needs ts_awq_layer weight pointers which require a loaded
// model context. Parse and validate the config, then signal "not available
// via FFI" with a non-zero return so the Swift layer can use the CLI.
int tessera_evolve(const char * model_path, const char * config_json) {
    (void)model_path;

    ts_awq_evolve_params params;
    params.population         = 32;
    params.generations        = 100;
    params.islands            = 4;
    params.migration_interval = 10;
    params.mutation_sigma     = 0.1f;
    params.crossover_rate     = 0.7f;
    params.heldout_weight     = 2.0f;
    params.seed               = 42;
    params.verbose            = false;

    if (config_json && config_json[0]) {
        try {
            json cfg = json::parse(config_json);
            if (cfg.contains("population"))         params.population         = cfg["population"].get<int64_t>();
            if (cfg.contains("generations"))        params.generations        = cfg["generations"].get<int64_t>();
            if (cfg.contains("islands"))            params.islands            = cfg["islands"].get<int64_t>();
            if (cfg.contains("migration_interval")) params.migration_interval = cfg["migration_interval"].get<int64_t>();
            if (cfg.contains("mutation_sigma"))     params.mutation_sigma     = cfg["mutation_sigma"].get<float>();
            if (cfg.contains("crossover_rate"))     params.crossover_rate     = cfg["crossover_rate"].get<float>();
            if (cfg.contains("heldout_weight"))     params.heldout_weight     = cfg["heldout_weight"].get<float>();
            if (cfg.contains("seed"))               params.seed               = cfg["seed"].get<uint32_t>();
            if (cfg.contains("verbose"))            params.verbose            = cfg["verbose"].get<bool>();
        } catch (const json::exception &) {
            return -2; // malformed config_json
        }
    }

    // config valid, but no weight tensors available without a model context
    return 1;
}

// evaluate: requires a model forward pass (ts_ppl_forward_fn). Return
// structured error JSON so the caller knows to use the CLI.
char * tessera_evaluate(const char * model_path, const char * config_json) {
    (void)model_path;
    (void)config_json;
    return ts_ffi_strdup(ts_ffi_error("evaluate",
        "perplexity evaluation requires a loaded model forward pass; "
        "use: llama-quantize --tessera-ppl <model>"));
}

// convert: coreml conversion needs dequantized weight tensors
// (ts_coreml_builder_tensor) which require a loaded model context.
int tessera_convert(const char * model_path,
                    const char * output_path,
                    const char * format) {
    if (!model_path || !output_path || !format) {
        return -1;
    }
    if (strcmp(format, "coreml") != 0) {
        return -2; // unsupported format
    }
    // valid request, but weight tensors are not available via FFI
    return 1;
}

// inspect_sidecar: fully callable - pure file I/O + serialization.
char * tessera_inspect_sidecar(const char * sidecar_path) {
    if (!sidecar_path) {
        return NULL;
    }

    ts_sidecar_v3 sc;
    std::string err;
    if (ts_sidecar_v3_read(sidecar_path, &sc, &err) != 0) {
        return ts_ffi_strdup(ts_ffi_error("inspect_sidecar",
                                          err.empty() ? "read failed" : err));
    }

    json j;
    j["ok"]                  = true;
    j["magic"]               = sc.header.magic;
    j["version"]             = sc.header.version;
    j["rows"]                = sc.header.rows;
    j["cols"]                = sc.header.cols;
    j["dtype"]               = sc.header.dtype;
    j["outlier_threshold"]   = sc.header.outlier_threshold;
    j["outlier_count_total"] = sc.header.outlier_count_total;
    j["n_row_meta"]          = static_cast<int64_t>(sc.row_meta.size());
    j["data_elements"]       = static_cast<int64_t>(sc.data.size());

    // cap the per-row counts to keep the response bounded
    json counts = json::array();
    int64_t n = std::min(static_cast<int64_t>(sc.row_outlier_counts.size()),
                         static_cast<int64_t>(256));
    for (int64_t i = 0; i < n; i++) {
        counts.push_back(sc.row_outlier_counts[i]);
    }
    j["row_outlier_counts"] = std::move(counts);

    return ts_ffi_strdup(j.dump());
}

// list_models: scan a directory for .gguf files. Always returns a valid
// JSON array (empty if the directory is missing or has no models).
char * tessera_list_models(const char * dir) {
    if (!dir) {
        return NULL;
    }

    json arr = json::array();
    DIR * d = opendir(dir);
    if (!d) {
        return ts_ffi_strdup(arr.dump());
    }

    struct dirent * ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string name(ent->d_name);
        if (name.size() < 5 || name.compare(name.size() - 5, 5, ".gguf") != 0) {
            continue;
        }
        std::string full = std::string(dir) + "/" + name;
        struct stat st;
        json entry;
        entry["name"] = name;
        entry["path"] = full;
        if (stat(full.c_str(), &st) == 0) {
            entry["size_bytes"] = static_cast<int64_t>(st.st_size);
        }
        arr.push_back(std::move(entry));
    }
    closedir(d);

    return ts_ffi_strdup(arr.dump());
}

// --- model-context path (header added 2026-08) ---
//
// The native impl currently supports load/free plus argument validation on
// the *_model() variants. The actual engine wiring (walk llama_model's
// ggml tensors to feed ts_awq_layer / ts_coreml_builder_tensor, build a
// ts_ppl_forward_fn over a llama_context, ...) is staged work - the *_model
// variants return a structured "TODO: requires engine impl ..." JSON so the
// Swift layer can keep its fallback path live while that wiring lands.
// Each TODO is a single line in the response so the next worker can grep
// for "TODO: requires engine impl" to find what is missing.

// Opaque model wrapper. Holds the llama_model* and the path it was loaded
// from (used for the TODO note so users know which model triggered the
// fallback). Allocated by tessera_load_model, freed by tessera_free_model.
struct tessera_model {
    llama_model * m;
    std::string   path;
};

// Process-wide flag set the first time the FFI is exercised. Cached so the
// env-var read happens once. When the user sets TESSERA_FFI_LOG=1 every
// FFI entry point prints a one-line trace to stderr (useful for Worker's 3
// e2e smoke test, which currently shells out to tessera-cli - this lets it
// see the in-process calls too).
static bool ts_ffi_log_enabled() {
    static const int kState = []() {
        const char * v = std::getenv("TESSERA_FFI_LOG");
        return (v && v[0] && v[0] != '0') ? 1 : 0;
    }();
    return kState == 1;
}

static void ts_ffi_log(const char * op, const char * detail = nullptr) {
    if (!ts_ffi_log_enabled()) {
        return;
    }
    if (detail && detail[0]) {
        std::fprintf(stderr, "[tessera-ffi] %s: %s\n", op, detail);
    } else {
        std::fprintf(stderr, "[tessera-ffi] %s\n", op);
    }
}

// Build a structured "not yet implemented" JSON for a *_model() variant.
// The Swift layer turns this into a .fallbackToCLI; the note is the single
// source of truth for what is missing so the next worker can grep the
// stderr trace to triage.
static std::string ts_ffi_model_todo(const char * op, const std::string & note) {
    json j;
    j["ok"]        = false;
    j["operation"] = std::string(op) + "_model";
    j["note"]      = note;
    return j.dump();
}

// Parse the GPU layers override from the (currently unused) extra JSON
// fields. We accept the parameter for forward compatibility - Worker 1 may
// add a "n_gpu_layers" key to the config, but keep the API stable.
static int32_t ts_ffi_resolve_gpu_layers(const int32_t * n_gpu_layers_in) {
    if (n_gpu_layers_in) {
        return *n_gpu_layers_in;
    }
    return 0;
}

tessera_model_handle_t tessera_load_model(const char * model_path,
                                          const int32_t * n_gpu_layers) {
    if (!model_path) {
        ts_ffi_log("load_model", "NULL model_path");
        return nullptr;
    }

    // Use llama.cpp defaults; Worker 1 may want to add GPU layers later
    // but the FFI surface is intentionally simple for the first cut.
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = ts_ffi_resolve_gpu_layers(n_gpu_layers);

    ts_ffi_log("load_model", model_path);
    llama_model * m = llama_model_load_from_file(model_path, params);
    if (!m) {
        ts_ffi_log("load_model", "llama_model_load_from_file returned NULL");
        return nullptr;
    }

    auto * wrapper = new (std::nothrow) tessera_model{m, model_path};
    if (!wrapper) {
        llama_model_free(m);
        ts_ffi_log("load_model", "wrapper alloc failed");
        return nullptr;
    }
    ts_ffi_log("load_model", "ok");
    return reinterpret_cast<tessera_model_handle_t>(wrapper);
}

void tessera_free_model(tessera_model_handle_t handle) {
    if (!handle) {
        return;
    }
    auto * wrapper = reinterpret_cast<tessera_model *>(handle);
    if (wrapper->m) {
        llama_model_free(wrapper->m);
    }
    delete wrapper;
}

// parse a ts_awq_evolve_params from config_json. Mirrors the no-handle
// tessera_evolve() parser so the two paths accept the same keys. Returns
// false on malformed JSON; on true the caller can use params as-is.
static bool ts_ffi_parse_awq_params(const char * config_json,
                                    ts_awq_evolve_params * out) {
    out->population         = 32;
    out->generations        = 100;
    out->islands            = 4;
    out->migration_interval = 10;
    out->mutation_sigma     = 0.1f;
    out->crossover_rate     = 0.7f;
    out->heldout_weight     = 2.0f;
    out->seed               = 42;
    out->verbose            = false;

    if (!config_json || !config_json[0]) {
        return true;
    }
    try {
        json cfg = json::parse(config_json);
        if (cfg.contains("population"))         out->population         = cfg["population"].get<int64_t>();
        if (cfg.contains("generations"))        out->generations        = cfg["generations"].get<int64_t>();
        if (cfg.contains("islands"))            out->islands            = cfg["islands"].get<int64_t>();
        if (cfg.contains("migration_interval")) out->migration_interval = cfg["migration_interval"].get<int64_t>();
        if (cfg.contains("mutation_sigma"))     out->mutation_sigma     = cfg["mutation_sigma"].get<float>();
        if (cfg.contains("crossover_rate"))     out->crossover_rate     = cfg["crossover_rate"].get<float>();
        if (cfg.contains("heldout_weight"))     out->heldout_weight     = cfg["heldout_weight"].get<float>();
        if (cfg.contains("seed"))               out->seed               = cfg["seed"].get<uint32_t>();
        if (cfg.contains("verbose"))            out->verbose            = cfg["verbose"].get<bool>();
    } catch (const json::exception &) {
        return false;
    }
    return true;
}

int tessera_evolve_model(tessera_model_handle_t handle, const char * config_json) {
    if (!handle) {
        return -1;
    }
    ts_awq_evolve_params params;
    if (!ts_ffi_parse_awq_params(config_json, &params)) {
        return -2;
    }
    auto * wrapper = reinterpret_cast<tessera_model *>(handle);
    ts_ffi_log("evolve_model", wrapper->path.c_str());

    // TODO: requires engine impl: walk llama_model's ggml tensors and feed
    // them as ts_awq_layer (one per quantizable weight) into
    // ts_awq_evolve_all() with a ts_awq_default_eval evaluator. The
    // current path returns the fallback marker so the Swift layer can
    // route to tessera-cli's evolve subcommand.
    return 1;
}

char * tessera_evaluate_model(tessera_model_handle_t handle, const char * config_json) {
    if (!handle) {
        return nullptr;
    }
    (void)config_json;
    auto * wrapper = reinterpret_cast<tessera_model *>(handle);
    ts_ffi_log("evaluate_model", wrapper->path.c_str());

    // TODO: requires engine impl: build a ts_ppl_forward_fn over a
    // llama_context (decode probe tokens, copy logits row-major into the
    // output buffer) and call ts_ppl_probe() on the F32 reference path.
    return ts_ffi_strdup(ts_ffi_model_todo("evaluate",
        "build ts_ppl_forward_fn over llama_context decode -> "
        "ts_ppl_probe on F32 reference path"));
}

int tessera_convert_model(tessera_model_handle_t handle,
                          const char * output_path,
                          const char * format) {
    if (!handle) {
        return -1;
    }
    if (!output_path || !format) {
        return -1;
    }
    if (std::strcmp(format, "coreml") != 0) {
        return -2; // unsupported format
    }
    auto * wrapper = reinterpret_cast<tessera_model *>(handle);
    ts_ffi_log("convert_model", wrapper->path.c_str());

    // TODO: requires engine impl: dequantize each llama_model ggml tensor
    // to fp16 (one ts_coreml_builder_tensor per quantizable weight) and
    // hand the array to ts_coreml_convert() with output_path as
    // mlpackage_path. Until this lands the FFI returns the fallback
    // marker so the Swift layer routes to tessera-cli's convert
    // subcommand.
    return 1;
}

void tessera_free_string(char * s) {
    free(s);
}

} // extern "C"
