#pragma once

//
// tessera-w4a4.h
//
// W4A4 activation quantization (4-bit weights + 4-bit activations) with
// LLM.int8-style per-channel outlier handling. Implements the calibration
// side of docs/w4a4-calibration-design.md: per-token dynamic / per-tensor
// static activation scales, the LLM.int8 outlier decomposition, and an
// activation-aware wrapper around ts_quantize_2d. The runtime dequant
// kernel is out of scope (a ggml-metal / ggml-cpu change for a later wave).
//

#include <cstdint>
#include <string>
#include <vector>

// forward
struct ts_quant_params_2d;
struct ts_quant_result_2d;

// Activation scale policy. Per-token dynamic is the ship-first default
// (design doc section 2); per-tensor static is the A/B alternative.
enum ts_w4a4_scale_mode {
    TS_W4A4_PER_TOKEN  = 0,
    TS_W4A4_PER_TENSOR = 1,
};

struct ts_w4a4_config {
    bool                 enable;
    int                  activation_bits;   // default 4 (W4A4); 8 -> W4A8
    enum ts_w4a4_scale_mode scale_mode;     // default TS_W4A4_PER_TOKEN
    float                outlier_thresh;    // LLM.int8 |X| threshold (default 6.0)
    float                outlier_frac;      // max outlier fraction cap (default 0.001)
    float                static_percentile; // per-tensor static percentile (default 0.999)
};

struct ts_w4a4_config ts_w4a4_default_config(void);

// Signed INT range for N bits: 2^(N-1) - 1 (7 for N=4, 127 for N=8).
int ts_w4a4_qmax(int activation_bits);

// Per-token dynamic scales: scale[t] = max_c |X[t,c]| / qmax.
// Per-tensor static: a single scale from the static_percentile of |X| / qmax.
struct ts_w4a4_act_scales {
    enum ts_w4a4_scale_mode mode;
    int                     qmax;
    std::vector<float>      per_token;  // size n_tokens (per-token mode)
    float                   per_tensor; // scalar (per-tensor mode)
};

void ts_w4a4_compute_act_scales(const float * calib_X,
                                int64_t n_tokens, int64_t in_dim,
                                const ts_w4a4_config * cfg,
                                ts_w4a4_act_scales * out);

// LLM.int8 outlier detection: channel c is an outlier iff max_t |X[t,c]|
// exceeds outlier_thresh. The count is capped at outlier_frac * in_dim,
// keeping the highest-magnitude channels. channels is sorted ascending.
struct ts_w4a4_outliers {
    std::vector<uint32_t> channels;  // sorted ascending channel indices
    std::vector<uint8_t>  mask;      // per-channel boolean, size in_dim
    float                 frac;      // channels.size() / in_dim
};

void ts_w4a4_detect_outliers(const float * calib_X,
                             int64_t n_tokens, int64_t in_dim,
                             const ts_w4a4_config * cfg,
                             ts_w4a4_outliers * out);

// Mixed-precision decomposition: non-outlier channels quantize to INT4
// (per-token or per-tensor scale), outlier channels stay FP16.
struct ts_w4a4_decomp {
    ts_w4a4_act_scales    scales;
    std::vector<uint32_t> outlier_channels;              // sorted ascending
    std::vector<int8_t>   quant;                          // (n_tokens x in_dim), [-8,7]
    std::vector<uint16_t> outlier_vals;                   // (n_tokens x n_outliers) f16
};

void ts_w4a4_decompose(const float * calib_X,
                       int64_t n_tokens, int64_t in_dim,
                       const ts_w4a4_config * cfg,
                       const ts_w4a4_outliers * outliers,
                       ts_w4a4_decomp * out);

// Dequantize a decomposition back to F32 (for round-trip verification).
void ts_w4a4_recompose(const ts_w4a4_decomp * decomp,
                       int64_t n_tokens, int64_t in_dim,
                       float * out);

// Activation-aware weight quantization: derives per-channel activation
// magnitudes from the calibration activations and runs ts_quantize_2d with
// them, so the weight quantization accounts for the activation distribution
// (not just the weight distribution). Also reports the effective bits/weight
// (weight components + amortized activation-scale overhead; target ~4.5).
struct ts_w4a4_weight_result {
    ts_quant_result_2d *  base;            // owned by caller; filled in place
    float                 effective_bits;
    ts_w4a4_outliers      outliers;
    ts_w4a4_act_scales    scales;
};

int ts_w4a4_quantize_weights(const float * weights,
                             const float * calib_X,
                             int64_t out_dim, int64_t in_dim, int64_t n_tokens,
                             const ts_quant_params_2d * qparams,
                             const ts_w4a4_config * cfg,
                             ts_quant_result_2d * base_out,
                             ts_w4a4_weight_result * out);

// Sidecar metadata (design doc section 2 schema). Serialized as the "w4a4"
// object appended to the per-tensor policy / receipt JSON.
struct ts_w4a4_sidecar {
    bool                    enabled;
    int                     activation_bits;
    enum ts_w4a4_scale_mode scale_mode;
    float                   outlier_frac;
    float                   act_scale_static;   // per-tensor mode only
    std::vector<uint32_t>   outlier_channels;
};

std::string ts_w4a4_scale_mode_str(enum ts_w4a4_scale_mode mode);
std::string ts_w4a4_sidecar_json(const ts_w4a4_sidecar * sc);
