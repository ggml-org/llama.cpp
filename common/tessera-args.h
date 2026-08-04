#pragma once

#include "common.h"

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
    // Phase 16 unified GGUF writer (unified-writer subcommand).
    // 4-5 per-component GGUFs are merged into a single
    // gemma4-assistant GGUF using the per-tensor calibration
    // policy (sidecar JSON via --policy, OR the dispatch's
    // tessera_db via --tessera-db). At least one --{component}
    // flag is required. --out is the destination path.
    // The hparams come from a sibling JSON via --hparams, OR from
    // the first source GGUF's metadata (when --hparams is empty).
    std::string unified_out;
    std::string unified_policy;
    std::string unified_hparams;
    std::string unified_trunk;
    std::string unified_dflash;
    std::string unified_dspark;
    std::string unified_mtp;
    std::string unified_shared_embd;
    std::string unified_arch = "gemma4-assistant";   // currently only gemma4-assistant is supported
    // Phase M0a: multimodal-projector components. Each is a separate
    // per-component source GGUF that the unified writer absorbs into
    // the destination gemma4-assistant GGUF. Source tensors already
    // carry the v.* / a.* / mm.* prefix (tools/mtmd/clip.cpp:1831,
    // 2594+); the writer does not add a second prefix.
    std::string unified_vision_tower;
    std::string unified_audio_tower;
    std::string unified_mm_projector;
    // Optional mmproj-side hparams. Empty = the writer uses zero
    // defaults and the destination's loader treats the absence of
    // gemma4-assistant.vision.* / .audio.* / .mm.* KV keys as
    // "no mmproj in this GGUF" (the pre-M0a contract).
    std::string unified_mmproj_hparams;
    // Structured progress reporting for the quantize pipeline. When
    // progress_file is non-empty, the dispatch writes one NDJSON event per
    // tick to that path for the Studio UI to tail.
    std::string progress_file;
    bool        progress_force_terminal = false;
    // Unified tessera.duckdb store. When tessera_db is non-empty, the
    // dispatch opens (or creates) a DuckDB file at that path and records one
    // row per run / tensor / GA-result / L4 plan outcome, plus bulk-logs
    // every GA candidate eval via the per-table write buffer. The store
    // drives the GA warm-start (family seeds from prior runs of the same
    // model), crash-resumability (converged tensors are skipped), the
    // cross-pipeline tensor_stats feature table (C++ writes
    // kurtosis / eff_rank, Python writes rms / mean_abs / tail_ratio),
    // and the L5 feedback loop (l4_plan_outcome + l5_outcome). Empty
    // = ephemeral, no DB.
    std::string tessera_db;
    // When set with --tessera-db, ignore existing converged tensors for
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
    // --tessera-ane-profile-out PATH: NDJSON file for per-phase ANE
    // dispatch profiling. Empty = no profile. The host sets this
    // before the first dispatch (typically from the --tessera-ane-profile-out
    // add_opt handler in common/arg.cpp). The C++ side reads it via
    // common_ane_phase_profile_set_output in common/ane-mtp.mm. Marked
    // experimental until the consumer-side reader lands; see
    // docs/tessera-ane-pump.md for the schema and the
    // tests/test-ane-phase-profile-emit.cpp smoke test.
    std::string ane_profile_out;
};

const common_tessera_params & common_get_tessera_params();

// Tessera fork: subcommand-aware entry point. Inspects argv[1] for a known
// subcommand name and dispatches the rest of the arguments to the matching
// per-subcommand flag set; without a subcommand, falls through to the main
// quantize path (tessera_sc = TESSERA_SC_NONE). On `--help`, prints the
// subcommand list (or the active subcommand's flag set) and exits 0. The
// caller is responsible for invoking the subcommand's CLI handler after
// this returns (see llama_tessera_main in tools/quantize/quantize.cpp).
// Returns true on a clean parse, false on a parse error.
//
// HARD BREAK (Tier 2): the legacy --tessera-* / --calib-* flag surface is
// not parsed here. Old flags produce "unrecognized argument" via
// common_params_parse. Use the subcommand syntax: `llama-tessera
// <subcommand> [flags]` or omit the subcommand for the main quantize path.
bool common_tessera_params_parse(int argc, char ** argv, common_params & params, void(*print_usage)(int, char **) = nullptr);

// Tessera fork: the subcommand selected by the most recent
// common_tessera_params_parse call. TESSERA_SC_NONE = no subcommand.
enum tessera_subcommand common_tessera_active_subcommand();
