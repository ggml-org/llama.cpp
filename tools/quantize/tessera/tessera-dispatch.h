#pragma once

//
// tessera-dispatch.h
//
// Top-level Tessera pipeline orchestrator. Called from quantize.cpp as
// llama_model_quantize_tessera(). Decides which steps to run based on
// flag presence, runs calibration / GA / per-tensor quantize, and
// returns results ready for the GGUF writer.
//

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>

#include "tessera-acceptance.h"

// Forward declaration; full definition in tessera-dispatch-internal.h.
struct ts_dispatch_refine_entry;

struct ts_dispatch_params {
    std::string input_path;
    std::string output_path;
    std::string imatrix_path;     // empty = run calibration
    std::string policy_path;      // empty = run GA
    std::string policy_out_path;
    std::string calib_corpus;     // empty = use built-in mini-corpus
    std::string higgs_alpha_mode; // "auto", "uniform", "cache-only" (default: "uniform")
    std::string higgs_cache_dir;  // empty = default (~/.cache/tessera/higgs_alpha/)
    uint64_t    evolve_seed;
    int         evolve_iters;
    int         evolve_islands;
    int         evolve_population;
    bool        evolve_only;
    bool        calibrate_only;
    float       outlier_frac;
    std::string awq_alpha;        // "auto" or a float string
    float       awq_clip;
    int         nthreads;
    bool        verbose;
    // S5 kernel-direct fitness (L1 sidecar). When kernel_fitness is set the
    // GA evaluator scores candidates against the kernel's real dequant output
    // (from kernel_fitness_dir) blended with the offline proxy.
    bool        kernel_fitness;
    std::string kernel_fitness_dir;     // empty = $LLAMA_TILE640_DEBUG_DEQUANT_DIR
    float       kernel_fitness_blend;   // 0.0 = offline, 1.0 = kernel-direct
    // S9 W4A4 activation quantization. When w4a4 is set the pipeline computes
    // per-token activation scales and the LLM.int8 outlier decomposition from
    // the calibration activations and records them as sidecar metadata; the
    // weight-only contract is unchanged when w4a4 is false.
    bool        w4a4;
    float       w4a4_outlier_thresh;    // LLM.int8 |X| threshold (default 6.0)
    // S7 G6 acceptance gate. When set, re-quantizes each tensor under every
    // single-proxy expert and runs the composite-beats-single + ranking
    // disagreement test after quantization.
    bool        run_acceptance;
    ts_acceptance_config acceptance_config;
    // L5 adaptive requantization. When set, runs the L2 -> L5 -> re-quantize
    // loop after step 7 for up to l5_max_generations, tightening alpha/clip
    // (Stage A) or outlier_fraction (Stage B, A/B per tensor family) on
    // tensors whose L2 divergence overshoots their type baseline. See
    // docs/runtime-aware-pipeline.md Layer 5.
    bool        adaptive_requantize = false;
    int         l5_max_generations  = 3;     // generational cap
    float       l5_flag_multiplier  = 1.5f;  // L2 flag threshold = multiplier * type baseline
    float       l5_alpha_min        = 0.1f;  // floor for the Stage A alpha multiplier
    float       l5_clip_min         = 0.1f;  // floor for the Stage A clip multiplier
    float       l5_outlier_overshoot_scale = 0.5f;  // Stage B outlier_frac bump per unit overshoot
    float       l5_outlier_frac_cap = 0.25f; // Stage B outlier_fraction ceiling
    std::string l5_out_path;                 // empty = beside policy_out_path as <stem>.l5-loop.json
};

// Result of the Tessera pipeline for one tensor.
struct ts_dispatch_tensor_result {
    std::string name;
    std::string family;
    int64_t     out_dim;
    int64_t     in_dim;
    // quantized data (the 6 components) - opaque blob for the writer
    std::vector<uint8_t> packed;
    std::vector<uint8_t> page_scales;
    std::vector<uint8_t> lane_scales;
    std::vector<uint8_t> outlier_row_offsets;
    std::vector<uint8_t> outlier_cols;
    std::vector<uint8_t> outlier_vals;
    std::vector<uint8_t> act_scale;   // empty if alpha == 0
    float       mse;
    float       alpha_used;
    // routed expert + the profile actually applied to this tensor
    int         expert_id;
    std::string expert_name;
    float       profile_alpha;
    float       profile_clip;
    int         profile_awq_grid;
    int         profile_max_outliers;
    float       profile_outlier_thresh;
    bool        profile_use_septq;
    // modality axis: operative modality (0=text, 1=image, 2=audio) and the
    // per-modality AWQ alpha from the multimodal imatrix (0 when absent)
    int         modality_id;
    float       modality_alpha[3];
    // S9 W4A4 sidecar metadata (populated when params.w4a4 is set)
    bool        w4a4_enabled = false;
    int         w4a4_activation_bits = 0;
    std::string w4a4_scale_mode;        // "per_token" / "per_tensor"
    float       w4a4_outlier_frac = 0.0f;
    float       w4a4_act_scale_static = 0.0f;
    std::vector<uint32_t> w4a4_outlier_channels;
};

struct ts_dispatch_result {
    std::vector<ts_dispatch_tensor_result> tensors;
    std::string policy_json;      // serialized policy for writing
    std::string archive_json;     // serialized MAP-Elites archive (empty if no GA)
    std::string policy_sha256;
    int64_t     n_tensors_quantized;
    int64_t     n_tensors_skipped;
    float       total_mse;
    // S7 G6 acceptance gate result (populated when run_acceptance is set)
    bool                  acceptance_ran;
    ts_acceptance_result  acceptance;
    // L5 adaptive requantization result (populated when adaptive_requantize is set)
    bool        l5_ran = false;
    std::string l5_report_json;   // schema llama.tessera.l5-loop.v1
};

// Run the full Tessera quantization pipeline.
// Returns 0 on success, non-zero on error.
int ts_dispatch_run(const ts_dispatch_params * params,
                    ts_dispatch_result * result,
                    std::string * err_msg);

// L5 adaptive requantize refine loop. Normally called by ts_dispatch_run when
// params->adaptive_requantize is set; exposed so the integration test can
// drive it directly. refine_map is keyed by tensor name; in_ctx/ggml_ctx are
// the input GGUF (for re-reading source weights) and out_ggml_ctx is the
// output descriptor context (refreshed via ts_gguf_repoint_tensor_cluster).
struct ts_dispatch_refine_entry;
int ts_dispatch_run_l5_loop(
    const ts_dispatch_params * params,
    ts_dispatch_result * result,
    struct gguf_context * in_ctx,
    struct ggml_context * ggml_ctx,
    struct ggml_context * out_ggml_ctx,
    std::unordered_map<std::string, ts_dispatch_refine_entry> & refine_map);
