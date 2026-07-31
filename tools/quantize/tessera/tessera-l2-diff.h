#pragma once

//
// tessera-l2-diff.h
//
// L2 BF16-vs-quantized differential (Layer 2 of the runtime-aware
// pipeline, see docs/runtime-aware-pipeline.md). The spec's L2 runs two
// forward passes and captures matmul-output divergence; the quantize tool
// cannot run full forwards, so this is the offline weight-level
// equivalent: for each quantized tensor, compare the BF16 source weights
// against the dequantized Tessera weights and report per-tensor
// divergence. Tensors whose relative Frobenius exceeds 1.5x their type's
// expected baseline are flagged for requantization (this feeds L5).
//
// The per-tensor metric block matches the runtime-probe schema; the
// forward-only fields (top1/top5 mismatch) are omitted at weight level.
//

#include <cstdint>
#include <string>
#include <vector>

// Per-tensor divergence between BF16 source and dequantized weights.
struct ts_l2_divergence {
    float max_abs;              // max_i |bf16_i - quant_i|
    float mean_abs;             // mean_i |bf16_i - quant_i|
    float relative_frobenius;   // ||bf16 - quant||_F^2 / ||bf16||_F^2
    float per_layer_norm;       // per-element RMS of the difference
};

// Result for one tensor, including the type-aware flag decision.
struct ts_l2_tensor_result {
    std::string tensor_name;
    std::string qtype;
    int64_t     rows;
    int64_t     cols;
    ts_l2_divergence divergence;
    float       expected_frob;      // type baseline (ts_l2_expected_frob)
    float       flag_threshold;     // flag_multiplier * expected_frob
    bool        flagged;            // relative_frobenius > flag_threshold
};

// Aggregate report over all tensors.
struct ts_l2_report {
    std::vector<ts_l2_tensor_result> tensors;
    int64_t n_flagged;
};

// One input tensor: paired BF16 / dequantized weight buffers.
struct ts_l2_tensor_input {
    const char  * name;
    const char  * qtype;        // lowercase ladder name, e.g. "tessera_t640"
    int64_t       rows;
    int64_t       cols;
    const float * bf16;         // source weights, (rows x cols) row-major
    const float * quant;        // dequantized Tessera weights, same layout
};

// Paths + flagging config. Model/corpus paths are recorded in the report
// for provenance; weight loading is the caller's responsibility (the
// quantize tool extracts the buffers and passes them via ts_l2_tensor_input).
struct ts_l2_config {
    char  bf16_model_path[1024];
    char  quant_model_path[1024];
    char  corpus_path[1024];
    char  output_json_path[1024];   // when non-empty, ts_l2_run writes here
    float flag_multiplier;          // default 1.5
};

void ts_l2_default_config(ts_l2_config * cfg);

// Expected relative Frobenius baseline for a quant type (spec 2.3 table).
// Unknown types fall back to a conservative 5e-2.
float ts_l2_expected_frob(const char * qtype);

// Core metric: divergence between two weight buffers of n elements.
ts_l2_divergence ts_l2_tensor_divergence(const float * bf16,
                                         const float * quant,
                                         int64_t n);

// Compute per-tensor divergence + flagging for all inputs, filling report.
// When cfg->output_json_path is non-empty, also writes the JSON report.
// Returns the number of flagged tensors (>= 0), or -1 on invalid args.
int ts_l2_run(const ts_l2_config * cfg,
              const ts_l2_tensor_input * inputs,
              int64_t n_tensors,
              ts_l2_report * report);

// Write the JSON report (schema llama.tessera.runtime-probe.v1).
// Returns 0 on success, -1 on error.
int ts_l2_write_report(const char * path,
                       const ts_l2_config * cfg,
                       const ts_l2_report * report);

// Read a JSON report back (consumed by L5 adaptive requantization).
// Returns 0 on success, -1 on error.
int ts_l2_load_report(const char * path, ts_l2_report * report);
