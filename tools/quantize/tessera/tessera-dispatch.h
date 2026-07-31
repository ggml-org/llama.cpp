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
};

struct ts_dispatch_result {
    std::vector<ts_dispatch_tensor_result> tensors;
    std::string policy_json;      // serialized policy for writing
    std::string archive_json;     // serialized MAP-Elites archive (empty if no GA)
    std::string policy_sha256;
    int64_t     n_tensors_quantized;
    int64_t     n_tensors_skipped;
    float       total_mse;
};

// Run the full Tessera quantization pipeline.
// Returns 0 on success, non-zero on error.
int ts_dispatch_run(const ts_dispatch_params * params,
                    ts_dispatch_result * result,
                    std::string * err_msg);
