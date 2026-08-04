#pragma once

#include <cstdint>
#include <string>

struct common_tessera_params {
    std::string mode = "default";
    std::string imatrix;
    std::string policy;
    std::string policy_out;
    std::string ga_checkpoint;
    std::string calib_corpus;
    std::string calib_corpus_out;
    uint64_t    evolve_seed = 0;
    int         evolve_iters = 8;
    int         evolve_islands = 4;
    int         evolve_population = 16;
    bool        evolve_only = false;
    bool        calibrate_only = false;
    float       outlier_frac = 0.005f;
    std::string awq_alpha = "auto";
    float       awq_clip = 1.0f;
    std::string ternary_threshold = "auto";
    std::string range_selection = "legacy";
    bool        champq = false;
    int         nthreads = 0;
    bool        kernel_fitness = false;
    std::string kernel_fitness_dir;
    float       kernel_fitness_blend = 1.0f;
    bool        w4a4 = false;
    float       w4a4_outlier_thresh = 6.0f;
    bool        acceptance = true;
    std::string acceptance_out;
    // L5 adaptive requantization loop (L2 -> L5 -> re-quantize, generational).
    // On by default: runtime-aware fixup of tensors whose L2 divergence
    // overshoots their type baseline is part of the core pipeline, not an
    // opt-in. Use --no-tessera-adaptive-requantize to disable for fast
    // iteration runs.
    bool        adaptive_requantize = true;
    int         l5_max_generations  = 3;
    float       l5_flag_multiplier  = 1.5f;
    float       l5_alpha_min        = 0.1f;
    float       l5_clip_min         = 0.1f;
    float       l5_outlier_overshoot_scale = 0.5f;
    float       l5_outlier_frac_cap = 0.25f;
    std::string l5_out;
    // self-improving capability loop: output-targeting ops that run then exit
    // without quantizing (precedent: --tessera-evolve-only / --tessera-calibrate-only)
    std::string capability_eval;
    std::string capability_out;
    std::string adapt_eval;
    std::string adapt_out;
    bool        adapt_dry_run = false;
    double      adapt_epsilon = 0.02;
    // tier-2 anonymizer: scrub a text payload then exit without quantizing
    std::string anonymize_in;
    std::string anonymize_out;
    std::string anonymize_level = "balanced";
    std::string anonymize_map;
    // north-star throughput harness: run workloads then exit without quantizing
    std::string throughput_workload;
    std::string throughput_out;
    // drafter training dataset preparation: convert llama.tessera.spec.v1 JSONL then exit
    std::string dataset_in;
    std::string dataset_out;
    std::string dataset_mode = "text";
    // D-PACE: compute adaptive position weights from DFlash telemetry then exit
    std::string dpace_in;
    std::string dpace_out;
    float       dpace_alpha = 0.1f;
    float       dpace_gamma = 3.0f;
    // Structured progress reporting for the quantize pipeline. When
    // progress_file is non-empty, the dispatch writes one NDJSON event per
    // tick to that path for the Studio UI to tail.
    std::string progress_file;
    bool        progress_force_terminal = false;
    // DuckDB-backed persistent store. When quantize_db is non-empty, the
    // dispatch opens (or creates) a DuckDB file at that path and records one
    // row per run/tensor/GA-result, plus bulk-logs every GA candidate eval
    // via the Appender API. The store also drives warm-start (GA seeds from
    // prior runs of the same family) and crash-resumability (tensors that
    // already converged are skipped). Empty = ephemeral, no DB.
    std::string quantize_db;
    // When set with --quantize-db, ignore existing converged tensors for
    // this run's model_hash and re-run the GA for every tensor.
    bool        force_requantize = false;
    // L2 forward-pass differential (Layer 2 of the runtime-aware pipeline).
    // When runtime_probe is non-empty, the dispatch records that the
    // caller is the tools/tessera/runtime_probe.py orchestrator and
    // passes the bf16 / quantized model paths and l2_out JSONL target
    // to the dispatch so they can be embedded in the L2 report's
    // provenance block. The actual matmul-output sidecar capture
    // happens in llama-cli / llama-imatrix via the
    // --tessera-matmul-output-dir CLI flag (common/arg.cpp). The
    // dispatch itself does not run the forwards; the orchestrator
    // shells out to llama-cli twice and reads the sidecars.
    std::string runtime_probe;
    std::string runtime_probe_bf16;
    std::string runtime_probe_l2_out;
    // --tessera-config FILE: path to an INI file that supplies default
    // values for any --tessera-* option. Populated by the --tessera-config
    // add_opt handler in common/arg.cpp. The file is loaded and applied at
    // the top of common_params_parse_ex, before env-var and CLI handling,
    // so env vars and explicit CLI flags naturally take precedence.
    std::string tessera_config_path;
};

const common_tessera_params & common_get_tessera_params();
