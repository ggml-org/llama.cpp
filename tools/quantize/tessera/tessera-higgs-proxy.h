// tessera-higgs-proxy.h
//
// HIGGS Linearity Theorem per-layer alpha_l estimator - structural proxy.
//
// C++ first-class port of the NumPy sidecar at
// tools/ane-mtp/estimate_higgs_alpha.py (and the Python module
// re-export at tools/tessera/estimate_higgs_alpha.py). Produces the
// same ane.alpha-coefficients.v1 sidecar JSON, so a sidecar produced
// by the C++ binary is byte-for-byte interchangeable with one
// produced by the Python path (modulo a documented float tolerance
// for the JSON float repr).
//
// The proxy is L1-agnostic by design: the t_l^2 measurement is
// parameterized into ts_higgs_proxy_measurement_fn. The default
// is the L1-on-ANE path (ts_higgs_proxy_measure_l1): ternary
// TILE640 quantization error + the F16 pre-dequant precision
// loss, modeled with the same dequant dispatch the
// GGML_OP_TILE640_MATMUL inference path uses. The offline
// ternary MSE stays available as a documented opt-in
// (TS_HIGGS_PROXY_LEGACY_OFFLINE=1, or pass
// ts_higgs_proxy_measure_offline as the measurement_fn).
//
// See docs/tessera-higgs-estimator.md for the math and the
// ane.alpha-coefficients.v1 schema. See docs/tessera-ane-ios-demo-design.md
// Phase 3.5 for the design of this C++ port.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Per-tensor family prior (the structural ranking).
//
// Mirrors the FAMILY_PRIOR dict in tools/ane-mtp/estimate_higgs_alpha.py
// EXACTLY (same suffix map, same float values, same "other" fallback).
// The values are the structural Hessian-trace proxy coefficients the
// iOS dispatch consumes as the per-tensor alpha. See the research doc
// docs/research-higgs-2026-07-30.md Section 1 for the SLQ / BAQ /
// HAWQ-V2 ranking derivation.
// ---------------------------------------------------------------------------

struct ts_higgs_proxy_family_entry {
    const char * suffix;
    const char * family;
    float        prior;
};

// The full family prior table, in the same order as the Python
// FAMILY_SUFFIXES tuple. The first matching suffix wins; the
// "other" entry at the end is the fallback for unknown names.
extern const ts_higgs_proxy_family_entry TS_HIGGS_PROXY_FAMILY_SUFFIXES[];

// "other" prior used when classify_family returns "other".
#define TS_HIGGS_PROXY_FAMILY_OTHER_PRIOR 0.50f

// Lookup a tensor name -> family key (the key into FAMILY_PRIOR).
// Returns "other" if no suffix matches. Strips a trailing ".weight"
// or ".bias" before the suffix match (the same convention as the
// NumPy classify_family).
std::string ts_higgs_proxy_classify_family(const std::string & name);

// Lookup a family key -> prior value. Returns the "other" prior
// for an unknown key.
float ts_higgs_proxy_family_prior(const std::string & family);

// ---------------------------------------------------------------------------
// Estimator parameters
// ---------------------------------------------------------------------------

struct ts_higgs_proxy_params {
    int64_t J;                       // number of measurement points (default 1, proxy is single-sample)
    float   t_min;                   // unused by the proxy, kept for parity with ts_higgs_params
    float   t_max;                   // unused by the proxy, kept for parity with ts_higgs_params
    float   alpha_floor;             // absolute floor (default 1e-6, never goes below this)
    uint32_t seed;                   // unused by the proxy, kept for parity with ts_higgs_params
    bool    verbose;                 // per-tensor print to stderr
    int64_t min_params_for_estimate; // uniform-fallback threshold (default 1B)
    float   alpha_floor_fraction;    // floor as a fraction of post-normalization mean (default 1e-3)
};

// ---------------------------------------------------------------------------
// Per-tensor result
// ---------------------------------------------------------------------------

struct ts_higgs_proxy_layer_result {
    std::string name;        // GGUF tensor name (e.g. "blk.16.attn_v.weight")
    std::string family;      // family key (e.g. "attn_v")
    std::vector<int64_t> shape; // original tensor shape, preserved for diagnostics
    int64_t  n_elem;         // total element count
    float    frobenius_norm_sq; // F32 Frobenius norm squared
    float    t_squared;      // relative Frobenius error (measurement function output)
    std::string t_squared_source; // "offline_ternary_mse" | "l1_kernel_dequant" | "uniform_fallback" | custom
    std::string dtype_source;    // GGUF dtype name as string (e.g. "F16", "Q4_0")
    float    alpha_l;        // post-normalization, post-floor
    bool     alpha_floor_applied;
    std::string fallback;   // "none" | "per_layer_uniform" | "global_uniform"
};

// ---------------------------------------------------------------------------
// Aggregated result
// ---------------------------------------------------------------------------

struct ts_higgs_proxy_result {
    std::vector<ts_higgs_proxy_layer_result> layers;
    int64_t  n_valid;
    int64_t  n_fallback_uniform;
    float    mean_alpha;
    std::string model_hash;       // SHA-256 first-16-hex of the GGUF (matches Python model_hash)
    std::string t_squared_source; // top-level: which measurement path was used
};

// ---------------------------------------------------------------------------
// t_l^2 measurement function (the L1-agnostic swap point)
//
// Given a dequantized F32 reference of n_elem elements for layer_idx,
// return t_l^2 (a non-negative relative Frobenius error). The ctx
// pointer is whatever the caller wants threaded through; the default
// C++ path passes nullptr.
//
// The function is allowed to return 0.0 for a degenerate reference
// (zero-norm, all-zero, NaN-bearing). The caller stamps the source
// label into the sidecar's per-tensor "t_squared_source" field.
// ---------------------------------------------------------------------------

typedef float (*ts_higgs_proxy_measurement_fn)(const float * W_flat,
                                               int64_t n_elem,
                                               int64_t layer_idx,
                                               void * ctx);

// The default measurement function: offline ternary MSE proxy. The
// NumPy `measure_t_squared_offline` algorithm in
// tools/ane-mtp/estimate_higgs_alpha.py, ported to scalar C++.
// t_l^2 = ||ternary_dequantize(ternary_round(W)) - W||_F^2 / ||W||_F^2
// where the per-tensor ternary scale is the mean absolute value.
// Returns 0.0 for a zero-norm or empty reference (no NaN, no Inf).
float ts_higgs_proxy_measure_offline(const float * W_flat, int64_t n_elem,
                                     int64_t layer_idx, void * ctx);

// ---------------------------------------------------------------------------
// L1-on-ANE t_l^2 measurement (the L1-agnostic path, Phase 0 source)
//
// Per-layer L1 distance between the original fp32 weight and the
// ANE-dequantized fp16 weight, modeled with the same dequant
// dispatch the GGML_OP_TILE640_MATMUL inference path uses:
//   - packed_W is the layer's TILE640 packing, concatenated
//     per-row flat rows [packed words | fp16 page_scales |
//     int8 lane_scales] (the quantize_row_tessera_t640_ref row
//     layout), out_dim rows of in_dim elements each;
//   - each row is dequantized with dequantize_row_tessera_t640_v2
//     (pre-decoded meta) when the dispatch cost model picks v2
//     (in_dim >= GGML_TESSERA_T640_V2_MIN_K and v2 enabled), the
//     C reference dequantize_row_tessera_t640 below the cutoff;
//   - the dequantized row is round-tripped through fp16 (the
//     bundle's pinned slot dtype - the F16 pre-dequant) before
//     the abs-diff, so the measurement captures both the ternary
//     quantization error AND the ANE fp16 precision loss.
//
// t_l^2 = (L1 / (out_dim * in_dim)) / max(|W_orig|): the mean
// absolute reconstruction error normalized by the largest
// original weight magnitude. Non-negative; 0.0 for a zero-norm /
// empty input (no NaN, no Inf).
// ---------------------------------------------------------------------------

typedef float (*ts_higgs_proxy_l1_measurement_fn)(const float * W_orig_flat,
                                                  const void * packed_W,
                                                  int64_t out_dim, int64_t in_dim,
                                                  int64_t layer_idx, void * ctx);

// The L1-on-ANE measurement. See the typedef comment for the
// packed_W layout contract and the normalization.
float ts_higgs_proxy_measure_l1(const float * W_orig_flat,
                                const void * packed_W,
                                int64_t out_dim, int64_t in_dim,
                                int64_t layer_idx, void * ctx);

// Produce the packed_W buffer ts_higgs_proxy_measure_l1 consumes:
// TILE640-pack each of the out_dim rows of in_dim elements with the
// C reference ternary quantizer (the deterministic creation path)
// and concatenate the per-row flat rows [packed words | fp16
// page_scales | int8 lane_scales]. packed is cleared first.
void ts_higgs_proxy_pack_tile640(const float * W_flat,
                                 int64_t out_dim, int64_t in_dim,
                                 std::vector<uint8_t> & packed);

// ---------------------------------------------------------------------------
// Estimator
//
// Read the GGUF, dequant every tensor to F32, classify family,
// compute t_l^2 via measurement_fn, and estimate alpha_l from the
// structural Hessian-trace proxy (FAMILY_PRIOR * n_elem / total_n_elem
// normalized so the positive mean is 1.0). Clamp to alpha_floor.
// Apply uniform fallback if total_params < min_params_for_estimate.
//
// measurement_fn may be NULL; the default is then the L1-on-ANE
// path (ts_higgs_proxy_measure_l1, t_squared_source
// "l1_kernel_dequant"). Setting the env var
// TS_HIGGS_PROXY_LEGACY_OFFLINE=1 restores the legacy offline
// ternary MSE default. An explicit measurement_fn keeps the
// legacy signature and its bit-identical output.
// measurement_ctx is passed through to the measurement function.
//
// gguf_path must be a readable GGUF file. Returns 0 on success, non-zero
// on read failure.
// ---------------------------------------------------------------------------

int ts_higgs_proxy_estimate(const char * gguf_path,
                            const ts_higgs_proxy_params * params,
                            ts_higgs_proxy_measurement_fn measurement_fn,
                            void * measurement_ctx,
                            ts_higgs_proxy_result * result);

// ---------------------------------------------------------------------------
// JSON I/O for the ane.alpha-coefficients.v1 sidecar.
//
// ts_higgs_proxy_to_json produces the exact same shape the NumPy
// build_sidecar writes (same keys, same key order, same value types).
// Floats are formatted with std::to_string's default precision (6
// significant digits), which is what the parity tests normalize
// against; numerical values agree to ~1e-6.
//
// ts_higgs_proxy_from_json loads the per-tensor alpha_l values
// (and minimal metadata) back into a result. It is forgiving of
// unknown top-level fields and matches the parser approach used by
// the iOS dispatch's existing ane_state_layout.v1 reader.
//
// ts_higgs_proxy_extract_alphas is the convenience for downstream
// consumers that only need the alpha vector.
// ---------------------------------------------------------------------------

std::string ts_higgs_proxy_to_json(const ts_higgs_proxy_result * result,
                                   const char * gguf_path,
                                   const char * bundle_name);

// Parse an ane.alpha-coefficients.v1 sidecar. Returns 0 on success,
// non-zero on parse failure. Fills the result; layers not present in
// the JSON are absent from the result.layers vector.
int ts_higgs_proxy_from_json(const char * json_str,
                             ts_higgs_proxy_result * result);

// Extract a flat alpha_l vector. Matches the shape of
// ts_higgs_extract_alphas in tessera-higgs.h.
std::vector<float> ts_higgs_proxy_extract_alphas(
    const ts_higgs_proxy_result * result);

// ---------------------------------------------------------------------------
// Model hash (content-addressed cache key)
//
// SHA-256 of the GGUF header (first 64KB) + last 64KB, hex truncated
// to 16 chars. Matches the Python model_hash() byte-for-byte.
// ---------------------------------------------------------------------------

std::string ts_higgs_proxy_model_hash(const char * gguf_path);

// ---------------------------------------------------------------------------
// JSON file I/O (crash-safe)
//
// ts_higgs_proxy_write_json_atomic writes to "<path>.tmp" first and
// renames into place on success. Returns 0 on success, non-zero on
// failure. The file is never partially written on disk.
// ---------------------------------------------------------------------------

int ts_higgs_proxy_write_json_atomic(const char * path,
                                     const std::string & json);
