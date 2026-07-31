//
// tessera-dispatch.cpp
//
// Top-level Tessera pipeline orchestrator. Loads the input GGUF, walks
// every tensor, routes quantizable weights through the regime classifier,
// quantizes via ts_quantize_2d / ts_quantize_3d, and writes the output
// GGUF with tessera metadata and policy JSON.
//

#include "tessera-dispatch.h"
#include "tessera-quant.h"
#include "tessera-awq.h"
#include "tessera-regime.h"
#include "tessera-gguf-writer.h"
#include "tessera-higgs.h"
#include "tessera-higgs-cache.h"
#include "tessera-search.h"
#include "tessera-imatrix.h"
#include "tessera-corpus.h"
#include "tessera-l1-fitness.h"
#include "tessera-ab-harness.h"
#include "tessera-mm-imatrix.h"
#include "tessera-mm-fitness.h"
#include "tessera-mm-awq.h"
#include "tessera-w4a4.h"
#include "tessera-acceptance.h"

#include "gguf.h"
#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>
#include <fstream>

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static std::vector<uint8_t> ts_to_bytes_u32(const std::vector<uint32_t> & v) {
    std::vector<uint8_t> out(v.size() * sizeof(uint32_t));
    std::memcpy(out.data(), v.data(), out.size());
    return out;
}

static std::vector<uint8_t> ts_to_bytes_u16(const std::vector<uint16_t> & v) {
    std::vector<uint8_t> out(v.size() * sizeof(uint16_t));
    std::memcpy(out.data(), v.data(), out.size());
    return out;
}

static std::vector<uint8_t> ts_to_bytes_i8(const std::vector<int8_t> & v) {
    std::vector<uint8_t> out(v.size() * sizeof(int8_t));
    std::memcpy(out.data(), v.data(), out.size());
    return out;
}

static std::vector<uint8_t> ts_to_bytes_i32(const std::vector<int32_t> & v) {
    std::vector<uint8_t> out(v.size() * sizeof(int32_t));
    std::memcpy(out.data(), v.data(), out.size());
    return out;
}

// A tensor is quantizable if it is a 2D or 3D F32/F16 weight matrix
// whose name maps to a known tensor family (attn, ffn, etc).
static bool ts_is_quantizable(const char * name, enum ggml_type type, int n_dims) {
    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16) {
        return false;
    }
    if (n_dims != 2 && n_dims != 3) {
        return false;
    }
    std::string family = ts_regime_infer_family(name);
    if (family.empty() || family == "other") {
        return false;
    }
    return true;
}

// Convert a ggml tensor's data to a flat F32 buffer.
// Handles F32 (copy) and F16 (convert). Returns empty on unsupported type.
static std::vector<float> ts_tensor_to_f32(const struct ggml_tensor * t) {
    const int64_t n = ggml_nelements(t);
    std::vector<float> out((size_t)n);

    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, (size_t)n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *)t->data, out.data(), n);
    } else {
        out.clear();
    }
    return out;
}

// Resolve per-channel AWQ activation scales for one tensor.
// Priority: (1) imatrix lookup, (2) mean |activation| derived from the
// calibration corpus when its width matches the tensor in_dim. Returns
// nullptr when neither source is available (AWQ scaling disabled). When the
// corpus path is used the result points into *scratch, which the caller must
// keep alive for as long as the returned pointer is needed.
static const float * ts_dispatch_act_scales(
        const ts_imatrix * imatrix, const char * name, int64_t in_dim,
        const float * calib_X, int64_t calib_in_dim, int64_t calib_n_tokens,
        std::vector<float> * scratch) {
    if (imatrix != nullptr) {
        int64_t dim = 0;
        const float * a = ts_imatrix_lookup(imatrix, name, &dim);
        if (a != nullptr && dim == in_dim) {
            return a;
        }
    }
    if (calib_X != nullptr && calib_in_dim == in_dim && calib_n_tokens > 0) {
        scratch->assign((size_t)in_dim, 0.0f);
        for (int64_t t = 0; t < calib_n_tokens; t++) {
            const float * row = calib_X + (size_t)t * in_dim;
            for (int64_t c = 0; c < in_dim; c++) {
                (*scratch)[(size_t)c] += std::fabs(row[c]);
            }
        }
        for (int64_t c = 0; c < in_dim; c++) {
            (*scratch)[(size_t)c] /= (float)calib_n_tokens;
        }
        return scratch->data();
    }
    return nullptr;
}

// Resolve a tensor's operative modality against the multimodal imatrix.
// Prefers the name-inferred modality when the imatrix has data for it, else
// the first present modality, else text. Returns the MM entry (nullptr when
// the tensor is absent from the imatrix) and writes the chosen modality.
static const ts_mm_imatrix_entry * ts_dispatch_mm_resolve(
        const ts_mm_imatrix * mm, const char * name, int inferred, int * modality) {
    const ts_mm_imatrix_entry * en = ts_mm_imatrix_entry_get(mm, name);
    if (en == nullptr) {
        *modality = inferred;
        return nullptr;
    }
    if (inferred >= 0 && inferred < TS_MODALITY_COUNT && en->has_modality[inferred]) {
        *modality = inferred;
        return en;
    }
    for (int m = 0; m < TS_MODALITY_COUNT; m++) {
        if (en->has_modality[m]) {
            *modality = m;
            return en;
        }
    }
    *modality = 0;
    return en;
}

// Run the per-modality AWQ alpha search for one tensor against the multimodal
// imatrix. Only modalities whose per-channel array length matches in_dim are
// usable as AWQ scales. Returns 0 on success and fills *result.
static int ts_dispatch_mm_awq(const ts_mm_imatrix * mm, const char * name,
                              const float * weights, int64_t out_dim, int64_t in_dim,
                              ts_mm_awq_result * result) {
    const float * act_mm[3] = { nullptr, nullptr, nullptr };
    for (int m = 0; m < TS_MODALITY_COUNT; m++) {
        int64_t d = 0;
        const float * a = ts_mm_imatrix_act_scales(mm, name, (ts_modality)m, &d);
        if (a != nullptr && d == in_dim) {
            act_mm[m] = a;
        }
    }
    ts_mm_awq_params mp = ts_mm_awq_default_params();
    mp.error_on_missing = false;   // partial modalities -> text fallback (M8)
    std::string merr;
    return ts_mm_awq_compute(weights, act_mm, nullptr, nullptr, nullptr,
                             out_dim, in_dim, &mp, result, &merr);
}

// Fixed quantization knobs shared across all GA candidate evaluations, plus
// the S5 kernel-direct fitness state. sidecar_cache holds the kernel dequant
// per tensor (loaded once; an empty vector means no sidecar is present) so the
// evaluator does not re-read disk per candidate. best_t2 / best_pair track,
// per tensor, the best-scoring candidate's blended t_l^2 and its
// (offline, kernel-direct) components, to feed the A/B harness after evolution.
struct ts_dispatch_eval_ctx {
    float    outlier_thresh;
    uint32_t seed;
    ts_l1_fitness_config l1;
    bool     verbose;
    int      mode_prints;   // tensors whose fitness mode has been logged
    std::unordered_map<std::string, std::vector<float>>      sidecar_cache;
    std::unordered_map<std::string, float>                   best_t2;
    std::unordered_map<std::string, std::pair<float, float>> best_pair;
    std::unordered_map<std::string, int>                     modality;  // per-tensor operative modality
};

// GA evaluator: quantize the layer with a candidate (alpha, clip) and score
// it. The GA maximizes `composite`, so report the negative t_l^2 (lower error
// -> higher fitness). By default t_l^2 is the offline relative Frobenius proxy
// ||W_hat - W||_F^2 / ||W||_F^2; with S5 kernel-direct fitness enabled it is
// blended with ||W_hat - dequant_kernel||_F^2 / ||W||_F^2, where
// dequant_kernel is the tensor's L1 sidecar (the kernel's real output).
static ts_awq_score ts_dispatch_awq_eval(const ts_awq_candidate * cand,
                                         const ts_awq_layer * layer,
                                         void * ctx) {
    ts_dispatch_eval_ctx * ec = (ts_dispatch_eval_ctx *)ctx;

    ts_awq_score score;
    score.mse           = std::numeric_limits<float>::infinity();
    score.relative_frob = std::numeric_limits<float>::infinity();
    score.heldout_mse   = std::numeric_limits<float>::infinity();
    score.composite     = -std::numeric_limits<float>::infinity();

    // route the layer to its expert and apply that expert's profile so the
    // GA scores candidates under the same knobs the final quantize uses
    int mod = 0;
    auto mit = ec->modality.find(layer->name);
    if (mit != ec->modality.end()) {
        mod = mit->second;
    }

    ts_regime_descriptor rd = {};
    rd.tensor_name = layer->name;
    rd.family      = layer->family;
    rd.kurtosis    = layer->kurtosis;
    rd.eff_rank    = layer->eff_rank;
    rd.modality    = mod;
    ts_regime_routing  rr   = ts_regime_classify(&rd);
    ts_expert_profile  prof = ts_expert_default_profile(rr.expert, mod);

    ts_quant_params_2d qp;
    qp.alpha          = cand->alpha * prof.alpha_scale;
    qp.clip           = cand->clip * prof.clip_scale;
    qp.max_outliers   = prof.max_outliers;
    qp.outlier_thresh = ec->outlier_thresh * prof.outlier_thresh;
    qp.use_imatrix    = layer->imatrix != nullptr;
    qp.use_septq      = prof.use_septq;
    qp.awq_grid       = prof.awq_grid;
    qp.seed           = ec->seed;

    ts_quant_result_2d qr;
    int rc = ts_quantize_2d(layer->weights, layer->act_scales,
                            layer->calib_X, layer->ref_output, layer->imatrix,
                            layer->out_dim, layer->in_dim, layer->n_tokens,
                            &qp, &qr);
    if (rc != 0) {
        return score;   // worst possible fitness
    }

    // qr.mse is the mean squared reconstruction error, so
    // ||W_hat - W||_F^2 = mse * n.
    const int64_t n = layer->out_dim * layer->in_dim;
    double frob2 = 0.0;
    for (int64_t i = 0; i < n; i++) {
        frob2 += (double)layer->weights[i] * (double)layer->weights[i];
    }
    float rel_frob = (frob2 > 0.0) ? (float)((double)qr.mse * (double)n / frob2) : qr.mse;

    // S5: kernel-direct t_l^2 from the L1 sidecar, blended with the proxy.
    float t2    = rel_frob;
    float kd_t2 = rel_frob;   // falls back to the proxy when no sidecar exists
    if (ec->l1.use_kernel_direct) {
        auto it = ec->sidecar_cache.find(layer->name);
        if (it == ec->sidecar_cache.end()) {
            std::vector<float> kdeq;
            int64_t sr = 0;
            int64_t sc = 0;
            if (ec->l1.sidecar_dir[0] != '\0' &&
                ts_l1_load_sidecar(ec->l1.sidecar_dir, layer->name.c_str(),
                                   &kdeq, &sr, &sc) == 0 && sr * sc == n) {
                ec->sidecar_cache[layer->name] = std::move(kdeq);
            } else {
                ec->sidecar_cache[layer->name] = std::vector<float>();
            }
            it = ec->sidecar_cache.find(layer->name);
            if (ec->verbose && ec->mode_prints < 3) {
                printf("tessera-dispatch: kernel-fitness: %s -> %s\n",
                       layer->name.c_str(),
                       it->second.empty() ? "offline proxy (no sidecar)"
                                          : "kernel-direct (L1 sidecar)");
                ec->mode_prints++;
            }
        }
        if (!it->second.empty() && (int64_t)it->second.size() == n &&
            (int64_t)qr.recon.size() == n) {
            kd_t2 = ts_l1_kernel_direct_t2(qr.recon.data(), layer->weights,
                                           it->second.data(), n);
            t2    = ts_l1_blended_t2(rel_frob, kd_t2, ec->l1.blend_factor);
        }
    }

    // record the best candidate's (offline, kernel) pair for the A/B harness
    if (ec->l1.use_kernel_direct) {
        auto bit = ec->best_t2.find(layer->name);
        if (bit == ec->best_t2.end() || t2 < bit->second) {
            ec->best_t2[layer->name]   = t2;
            ec->best_pair[layer->name] = std::make_pair(rel_frob, kd_t2);
        }
    }

    score.mse           = qr.mse;
    score.relative_frob = t2;       // t_l^2 used for fitness (blended when kernel-direct)
    score.heldout_mse   = qr.mse;   // no held-out split in standalone dispatch
    score.composite     = -t2;
    return score;
}

// Quantize a tensor with a forced expert profile and return relative
// Frobenius t_l^2 = ||W_hat - W||_F^2 / ||W||_F^2.
static float ts_dispatch_forced_t2(const float * weights, const float * act_scales,
                                   int64_t out_dim, int64_t in_dim,
                                   ts_expert_id expert, float base_alpha,
                                   float base_clip, float outlier_thresh,
                                   uint32_t seed) {
    ts_expert_profile prof = ts_expert_default_profile(expert);

    ts_quant_params_2d qp;
    qp.alpha          = base_alpha * prof.alpha_scale;
    qp.clip           = base_clip * prof.clip_scale;
    qp.max_outliers   = prof.max_outliers;
    qp.outlier_thresh = outlier_thresh * prof.outlier_thresh;
    qp.use_imatrix    = act_scales != nullptr;
    qp.use_septq      = prof.use_septq;
    qp.awq_grid       = prof.awq_grid;
    qp.seed           = seed;

    ts_quant_result_2d qr;
    int rc = ts_quantize_2d(weights, act_scales, nullptr, nullptr, act_scales,
                            out_dim, in_dim, 0, &qp, &qr);
    if (rc != 0) {
        return 1.0f;  // worst case
    }

    const int64_t n = out_dim * in_dim;
    double frob2 = 0.0;
    for (int64_t i = 0; i < n; i++) {
        frob2 += (double)weights[i] * (double)weights[i];
    }
    return (frob2 > 0.0) ? (float)((double)qr.mse * (double)n / frob2) : qr.mse;
}

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------

int ts_dispatch_run(const ts_dispatch_params * params,
                    ts_dispatch_result * result,
                    std::string * err_msg) {
    if (params == nullptr || result == nullptr) {
        if (err_msg) {
            *err_msg = "null params or result";
        }
        return 1;
    }

    *result = {};

    const bool verbose = params->verbose;

    // --- step 1: determine which steps to run ---
    const bool need_calibration = params->imatrix_path.empty() && params->policy_path.empty();
    const bool need_ga          = params->policy_path.empty();

    // --- step 2: calibration ---
    // Load precomputed per-channel activation statistics (the AWQ calibration
    // artifact) when an imatrix is provided.
    ts_imatrix imatrix;
    bool have_imatrix = false;
    ts_mm_imatrix mm_imatrix;
    bool have_mm_imatrix = false;
    if (!params->imatrix_path.empty()) {
        // multimodal imatrix (v3, modality_breakdown). Optional: a text-only
        // v2 file simply fails this load and falls through to the text path.
        std::string mmsg;
        if (ts_mm_imatrix_load(params->imatrix_path.c_str(), &mm_imatrix, &mmsg) == 0) {
            have_mm_imatrix = true;
            if (verbose) {
                printf("tessera-dispatch: calibration: loaded multimodal imatrix '%s' (%zu tensors)\n",
                       params->imatrix_path.c_str(), mm_imatrix.data.size());
            }
        }

        std::string imsg;
        if (ts_imatrix_load_npz(params->imatrix_path.c_str(), &imatrix, &imsg) == 0) {
            have_imatrix = true;
            if (verbose) {
                printf("tessera-dispatch: calibration: loaded imatrix '%s' (%zu tensors)\n",
                       params->imatrix_path.c_str(), imatrix.data.size());
            }
        } else if (!have_mm_imatrix) {
            if (err_msg) {
                *err_msg = "failed to load imatrix '" + params->imatrix_path + "': " + imsg;
            }
            return 1;
        } else if (verbose) {
            printf("tessera-dispatch: calibration: no text rollup in '%s' (multimodal only)\n",
                   params->imatrix_path.c_str());
        }
    }

    // Calibration activations. With no imatrix and no policy we use the
    // built-in mini-corpus (deterministic synthetic activations) or a
    // caller-supplied corpus directory; these feed the AWQ scale fit for any
    // tensor whose in_dim matches the corpus width.
    std::vector<float> calib_X;
    int64_t calib_n_tokens = 0;
    int64_t calib_in_dim   = 0;
    if (need_calibration) {
        if (params->calib_corpus.empty()) {
            ts_corpus_params cparams = ts_corpus_default_params();
            calib_X        = ts_corpus_generate(&cparams);
            calib_n_tokens = cparams.n_tokens;
            calib_in_dim   = cparams.in_dim;
            if (verbose) {
                printf("tessera-dispatch: calibration: built-in mini-corpus (%lld x %lld)\n",
                       (long long)calib_n_tokens, (long long)calib_in_dim);
            }
        } else {
            std::string cmsg;
            calib_X = ts_corpus_load_directory(params->calib_corpus.c_str(),
                                               &calib_n_tokens, &calib_in_dim, &cmsg);
            if (calib_X.empty()) {
                if (err_msg) {
                    *err_msg = "failed to load calib corpus '" + params->calib_corpus + "': " + cmsg;
                }
                return 1;
            }
            if (verbose) {
                printf("tessera-dispatch: calibration: loaded corpus '%s' (%lld x %lld)\n",
                       params->calib_corpus.c_str(), (long long)calib_n_tokens, (long long)calib_in_dim);
            }
        }
        // TODO(hardening): real per-layer calibration activations require a
        // model forward pass over the corpus (per-layer calib_X / ref_output).
        // ts_dispatch_params carries no tokenizer or forward callback, so the
        // corpus above is a data-free proxy; per-channel act_scales come from
        // the imatrix when one is provided.
    }

    if (params->calibrate_only) {
        if (verbose) {
            printf("tessera-dispatch: calibrate_only, returning early\n");
        }
        result->policy_json = "{}";
        return 0;
    }

    // --- step 3: GA configuration ---
    // The evolutionary search runs per-tensor once the weights are loaded
    // (step 5c); here the dispatch knobs are translated into GA params.
    float default_alpha = 0.5f;

    ts_awq_evolve_params evolve_params;
    evolve_params.population         = params->evolve_population > 0 ? params->evolve_population : 32;
    evolve_params.generations        = params->evolve_iters > 0 ? params->evolve_iters : 100;
    evolve_params.islands            = params->evolve_islands > 0 ? params->evolve_islands : 4;
    evolve_params.migration_interval = 10;
    evolve_params.mutation_sigma     = 0.1f;
    evolve_params.crossover_rate     = 0.7f;
    evolve_params.heldout_weight     = 2.0f;
    evolve_params.seed               = (uint32_t)params->evolve_seed;
    evolve_params.verbose            = verbose;

    if (need_ga) {
        if (verbose) {
            printf("tessera-dispatch: GA configured (seed=%llu iters=%d islands=%d pop=%d)\n",
                   (unsigned long long)params->evolve_seed,
                   params->evolve_iters,
                   params->evolve_islands,
                   params->evolve_population);
        }
    }

    if (params->evolve_only) {
        if (verbose) {
            printf("tessera-dispatch: evolve_only, returning early\n");
        }
        result->policy_json = "{}";
        return 0;
    }

    // --- step 4: resolve alpha ---
    if (params->awq_alpha != "auto" && !params->awq_alpha.empty()) {
        default_alpha = std::stof(params->awq_alpha);
    }

    // --- step 5: load input GGUF ---
    struct ggml_context * ggml_ctx = nullptr;
    struct gguf_init_params gparams = {
        /*no_alloc =*/ false,
        /*ctx      =*/ &ggml_ctx,
    };

    struct gguf_context * in_ctx = gguf_init_from_file(params->input_path.c_str(), gparams);
    if (in_ctx == nullptr) {
        if (err_msg) {
            *err_msg = "failed to open input GGUF: " + params->input_path;
        }
        return 1;
    }

    const int64_t n_tensors = gguf_get_n_tensors(in_ctx);
    if (verbose) {
        printf("tessera-dispatch: loaded '%s' (%lld tensors)\n",
               params->input_path.c_str(), (long long)n_tensors);
    }

    // --- step 5b: HIGGS alpha_l estimation / cache lookup ---
    std::string higgs_mode = params->higgs_alpha_mode.empty() ? "uniform" : params->higgs_alpha_mode;
    std::vector<float> higgs_alphas;   // empty = uniform
    bool higgs_active = false;

    if (higgs_mode != "uniform") {
        // collect quantizable tensor weights for cache key
        std::vector<const float *> higgs_wptrs;
        std::vector<int64_t> higgs_outs, higgs_ins;
        std::vector<std::vector<float>> higgs_wbufs;

        for (int64_t i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(in_ctx, i);
            const enum ggml_type type = gguf_get_tensor_type(in_ctx, i);
            const int64_t * ne = gguf_get_tensor_ne(in_ctx, i);
            int nd = GGML_MAX_DIMS;
            while (nd > 1 && ne[nd - 1] == 1) nd--;

            if (!ts_is_quantizable(name, type, nd)) continue;

            struct ggml_tensor * t = ggml_get_tensor(ggml_ctx, name);
            if (!t) continue;

            higgs_wbufs.push_back(ts_tensor_to_f32(t));
            if (higgs_wbufs.back().empty()) {
                higgs_wbufs.pop_back();
                continue;
            }
            higgs_wptrs.push_back(higgs_wbufs.back().data());
            higgs_outs.push_back(ne[1]);
            higgs_ins.push_back(ne[0]);
        }

        if (!higgs_wptrs.empty()) {
            ts_higgs_cache_key ckey = ts_higgs_cache_compute_key(
                higgs_wptrs.data(), higgs_outs.data(), higgs_ins.data(),
                (int64_t)higgs_wptrs.size());

            const std::string * cdir = params->higgs_cache_dir.empty()
                ? nullptr : &params->higgs_cache_dir;

            auto cached = ts_higgs_cache_load(&ckey, cdir);
            if (cached.has_value()) {
                higgs_alphas = std::move(cached.value());
                higgs_active = true;
                if (verbose) {
                    printf("tessera-dispatch: HIGGS cache hit (%lld layers, hash=%s...)\n",
                           (long long)higgs_alphas.size(), ckey.hex.substr(0, 12).c_str());
                }
            } else if (higgs_mode == "cache-only") {
                if (err_msg) {
                    *err_msg = "HIGGS cache miss and mode is cache-only (hash=" + ckey.hex + ")";
                }
                gguf_free(in_ctx);
                ggml_free(ggml_ctx);
                return 4;
            } else {
                // mode == "auto": estimation requires a model forward-pass
                // callback (metric_fn) not available in the standalone
                // dispatch. The offline harness (alpha_calibrate.py) produces
                // the cache artifact; log and fall back to uniform.
                if (verbose) {
                    printf("tessera-dispatch: HIGGS cache miss, falling back to uniform "
                           "(run alpha_calibrate.py to populate cache, hash=%s...)\n",
                           ckey.hex.substr(0, 12).c_str());
                }
            }
        }
    }

    // build search config for the GA
    ts_search_config search_cfg;
    search_cfg.layer_alpha = higgs_active ? higgs_alphas.data() : nullptr;
    search_cfg.n_layers    = higgs_active ? (int64_t)higgs_alphas.size() : 0;

    // --- step 5c: evolutionary per-tensor alpha search (GA) ---
    // Runs the real AWQ/GA search over the 2D quantizable tensors to produce a
    // per-tensor alpha. The evaluator quantizes each candidate via
    // ts_quantize_2d; the HIGGS alpha_l weights (search_cfg) score the
    // cross-layer composite via ts_search_fitness when the layer counts match.
    std::unordered_map<std::string, float> ga_alpha;

    // per-tensor multimodal state (populated when an MM imatrix is present):
    // the per-modality AWQ result (alpha + mse) and the operative modality.
    // mm_awq is shared with the quantize loop so the alpha search runs once.
    std::unordered_map<std::string, ts_mm_awq_result> mm_awq;
    std::unordered_map<std::string, int>              mm_modality;

    // MAP-Elites archive: best policy per regime cell, populated from the GA
    // results below and persisted to a sidecar JSON alongside the policy.
    ts_map_elites_archive archive;
    bool have_archive = false;

    if (need_ga) {
        std::vector<std::string>           ga_names;
        std::vector<std::vector<float>>    ga_wbufs;     // weight storage (owns data)
        std::vector<std::vector<float>>    ga_actbufs;   // corpus-derived act_scales storage
        std::vector<ts_awq_layer>          ga_layers;
        std::vector<ts_regime_descriptor>  ga_descs;     // regime descriptor per layer (archive axes)

        for (int64_t i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(in_ctx, i);
            const enum ggml_type type = gguf_get_tensor_type(in_ctx, i);
            const int64_t * ne = gguf_get_tensor_ne(in_ctx, i);
            int nd = GGML_MAX_DIMS;
            while (nd > 1 && ne[nd - 1] == 1) nd--;

            // the GA evolves 2D weight matrices; 3D MoE tensors fall back to
            // default_alpha in the quantize loop below.
            if (nd != 2 || !ts_is_quantizable(name, type, nd)) continue;

            struct ggml_tensor * t = ggml_get_tensor(ggml_ctx, name);
            if (!t) continue;

            std::vector<float> w = ts_tensor_to_f32(t);
            if (w.empty()) continue;

            const int64_t in_dim  = ne[0];
            const int64_t out_dim = ne[1];

            // resolve per-channel activation scales (imatrix, else corpus);
            // a corpus-derived buffer is moved into ga_actbufs to keep the
            // pointer valid for the whole evolution.
            std::vector<float> act_scratch;
            const float * act = ts_dispatch_act_scales(
                have_imatrix ? &imatrix : nullptr, name, in_dim,
                calib_X.empty() ? nullptr : calib_X.data(), calib_in_dim, calib_n_tokens,
                &act_scratch);
            if (act != nullptr && act == act_scratch.data()) {
                ga_actbufs.push_back(std::move(act_scratch));
                act = ga_actbufs.back().data();
            }

            // regime descriptors (kurtosis / eff_rank) feed the GA archive cell
            const float * imdata = nullptr;
            int64_t       imdim  = 0;
            if (have_imatrix) {
                imdata = ts_imatrix_lookup(&imatrix, name, &imdim);
            }
            ts_regime_descriptor desc = ts_regime_compute_descriptor(
                name, w.data(), out_dim, in_dim, imdata, imdata ? imdim : 0);

            // multimodal: resolve the operative modality (drives routing + the
            // archive axis) and run the per-modality AWQ alpha search (drives
            // the modality-weighted fitness below).
            if (have_mm_imatrix) {
                int mod = desc.modality;
                const ts_mm_imatrix_entry * en =
                    ts_dispatch_mm_resolve(&mm_imatrix, name, desc.modality, &mod);
                desc.modality = mod;
                if (en != nullptr) {
                    mm_modality[name] = mod;
                    ts_mm_awq_result mres;
                    if (ts_dispatch_mm_awq(&mm_imatrix, name, w.data(), out_dim, in_dim, &mres) == 0) {
                        mm_awq[name] = std::move(mres);
                    }
                }
            }

            ga_names.push_back(name);
            ga_wbufs.push_back(std::move(w));

            ts_awq_layer layer;
            layer.name        = ga_names.back();
            layer.family      = desc.family;
            layer.weights     = ga_wbufs.back().data();
            layer.act_scales  = act;
            layer.calib_X     = nullptr;
            layer.ref_output  = nullptr;
            layer.imatrix     = act;
            layer.out_dim     = out_dim;
            layer.in_dim      = in_dim;
            layer.n_tokens    = 0;
            layer.kurtosis    = desc.kurtosis;
            layer.eff_rank    = desc.eff_rank;
            ga_layers.push_back(layer);
            ga_descs.push_back(desc);
        }

        if (!ga_layers.empty()) {
            ts_dispatch_eval_ctx eval_ctx;
            eval_ctx.outlier_thresh = params->outlier_frac;
            eval_ctx.seed           = (uint32_t)params->evolve_seed;
            eval_ctx.verbose        = verbose;
            eval_ctx.mode_prints    = 0;
            eval_ctx.modality       = mm_modality;

            // S5 kernel-direct fitness config. The sidecar directory defaults
            // to the runtime hook's dump dir ($LLAMA_TILE640_DEBUG_DEQUANT_DIR).
            ts_l1_fitness_default_config(&eval_ctx.l1);
            eval_ctx.l1.use_kernel_direct = params->kernel_fitness;
            eval_ctx.l1.blend_factor      = params->kernel_fitness_blend;
            std::string kf_dir = params->kernel_fitness_dir;
            if (kf_dir.empty()) {
                const char * env = std::getenv("LLAMA_TILE640_DEBUG_DEQUANT_DIR");
                if (env != nullptr) {
                    kf_dir = env;
                }
            }
            if (!kf_dir.empty()) {
                snprintf(eval_ctx.l1.sidecar_dir, sizeof(eval_ctx.l1.sidecar_dir),
                         "%s", kf_dir.c_str());
            }
            if (params->kernel_fitness && verbose) {
                printf("tessera-dispatch: kernel-fitness: enabled (blend=%.2f dir='%s')\n",
                       (double)eval_ctx.l1.blend_factor, eval_ctx.l1.sidecar_dir);
            }

            std::vector<ts_awq_evolve_result> ga_results;
            int rc = ts_awq_evolve_all(ga_layers.data(), (int64_t)ga_layers.size(),
                                       ts_dispatch_awq_eval, &eval_ctx,
                                       &evolve_params, &ga_results);
            if (rc != 0) {
                if (err_msg) {
                    *err_msg = "ts_awq_evolve_all failed";
                }
                gguf_free(in_ctx);
                ggml_free(ggml_ctx);
                return 5;
            }

            // per-tensor alpha + per-layer relative Frobenius error
            std::vector<float> t2(ga_results.size());
            for (size_t l = 0; l < ga_results.size(); l++) {
                ga_alpha[ga_names[l]] = ga_results[l].best.alpha;
                t2[l]                 = ga_results[l].best_score.relative_frob;
            }

            // HIGGS-weighted pipeline composite (uniform when layer counts differ)
            ts_search_config fit_cfg;
            fit_cfg.layer_alpha = (search_cfg.n_layers == (int64_t)t2.size())
                                      ? search_cfg.layer_alpha : nullptr;
            fit_cfg.n_layers    = (int64_t)t2.size();

            // modality-weighted composite (M1) when multimodal data exists:
            // per-modality per-layer t_l^2 from the per-modality AWQ reconstruction
            // error, combined with the 0.5/0.3/0.2 weights via ts_mm_fitness_compute.
            // Missing modalities carry the text fallback (M8), so all three slots
            // are populated for every layer that has an MM AWQ result.
            float composite;
            const bool mm_composite = have_mm_imatrix && !mm_awq.empty();
            if (mm_composite) {
                const int64_t n_layers = (int64_t)t2.size();
                std::vector<float> t2_text(n_layers, 0.0f);
                std::vector<float> t2_image(n_layers, 0.0f);
                std::vector<float> t2_audio(n_layers, 0.0f);
                for (size_t l = 0; l < ga_results.size(); l++) {
                    auto it = mm_awq.find(ga_names[l]);
                    if (it != mm_awq.end()) {
                        t2_text[l]  = it->second.mse_per_modality[0];
                        t2_image[l] = it->second.mse_per_modality[1];
                        t2_audio[l] = it->second.mse_per_modality[2];
                    } else {
                        t2_text[l]  = t2[l];
                        t2_image[l] = t2[l];
                        t2_audio[l] = t2[l];
                    }
                }
                const float * t2_mm[3] = { t2_text.data(), t2_image.data(), t2_audio.data() };
                const bool present[3]  = { true, true, true };
                ts_mm_fitness_params fp = ts_mm_fitness_default_params();
                ts_mm_fitness_score fs = ts_mm_fitness_compute(
                    t2_mm, fit_cfg.layer_alpha, present, n_layers, &fp);
                composite = fit_cfg.layer_alpha ? fs.alpha_weighted : fs.composite;
            } else {
                composite = ts_search_fitness(t2.data(), &fit_cfg);
            }

            if (verbose) {
                printf("tessera-dispatch: GA done (%lld layers, composite=%.6f, higgs_weighted=%d, mm_weighted=%d)\n",
                       (long long)ga_layers.size(), composite, fit_cfg.layer_alpha != nullptr, (int)mm_composite);
            }

            // feed the GA outcomes into the MAP-Elites archive: one elite per
            // regime cell, keyed by each layer's descriptor. Fitness is the
            // HIGGS-weighted per-layer t_l^2 (lower is better).
            ts_archive_init(&archive, 5, 5, 8, 3);
            for (size_t l = 0; l < ga_results.size(); l++) {
                const float w_l     = fit_cfg.layer_alpha ? fit_cfg.layer_alpha[l] : 1.0f;
                const float fitness = w_l * t2[l];
                ts_archive_insert(&archive, &ga_descs[l], fitness,
                                  ga_results[l].best.alpha, ga_results[l].best.clip,
                                  ga_names[l].c_str());
            }
            have_archive = true;

            ts_archive_summary as = ts_archive_summarize(&archive);
            result->archive_json = ts_archive_to_json(&archive);
            if (verbose) {
                printf("tessera-dispatch: MAP-Elites archive (%d/%d cells occupied, "
                       "mean_fitness=%.6f best=%.6f)\n",
                       as.occupied_cells, as.total_cells, as.mean_fitness, as.best_fitness);
            }

            // S5: A/B comparison of the offline proxy vs kernel-direct t_l^2,
            // reported side by side (per-tensor scores + alpha-weighted
            // composites + ranking agreement).
            if (params->kernel_fitness) {
                std::vector<ts_ab_tensor_scores> ab_scores;
                ab_scores.reserve(ga_names.size());
                for (size_t l = 0; l < ga_names.size(); l++) {
                    auto pit = eval_ctx.best_pair.find(ga_names[l]);
                    if (pit == eval_ctx.best_pair.end()) {
                        continue;
                    }
                    ts_ab_tensor_scores s;
                    s.name              = ga_names[l];
                    s.offline_proxy_mse = pit->second.first;
                    s.kernel_direct_t2  = pit->second.second;
                    s.alpha_l           = (fit_cfg.layer_alpha != nullptr)
                                              ? fit_cfg.layer_alpha[l] : 1.0f;
                    ab_scores.push_back(std::move(s));
                }
                if (!ab_scores.empty()) {
                    ts_ab_harness_params ab_params;
                    ab_params.n_heldout       = 0;   // score all tensors
                    ab_params.measure_ranking = true;
                    ab_params.verbose         = false;
                    ts_ab_harness_result ab_result;
                    if (ts_ab_run(&ab_scores, &ab_params, &ab_result) == 0 && verbose) {
                        printf("tessera-dispatch: A/B harness: %s\n", ab_result.report.c_str());
                    }
                }
            }
        }
    }

    // --- step 6: prepare output GGUF ---
    struct gguf_context * out_ctx = gguf_init_empty();
    gguf_set_kv(out_ctx, in_ctx);

    // The cluster descriptors are allocated from a caller-owned context so the
    // writer stays allocation-free. Size it from the input metadata: each
    // quantizable tensor emits one cluster (ne[2] clusters for a 3D MoE
    // weight), up to 7 descriptors each. Over-counting only wastes a little
    // RAM; under-counting aborts ggml_init's fixed pool, so budget generously.
    int64_t n_cluster_tensors = 0;
    for (int64_t i = 0; i < n_tensors; i++) {
        const enum ggml_type type_i = gguf_get_tensor_type(in_ctx, i);
        const int64_t * ne_i = gguf_get_tensor_ne(in_ctx, i);
        int nd_i = GGML_MAX_DIMS;
        while (nd_i > 1 && ne_i[nd_i - 1] == 1) nd_i--;
        if (!ts_is_quantizable(gguf_get_tensor_name(in_ctx, i), type_i, nd_i)) continue;
        n_cluster_tensors += ((nd_i == 3) ? ne_i[2] : 1) * 7;
    }
    struct ggml_init_params out_init = {
        /*mem_size   =*/ (size_t)n_cluster_tensors * 512 + 64 * 1024,
        /*mem_buffer =*/ nullptr,
        /*no_alloc   =*/ true,
    };
    struct ggml_context * out_ggml_ctx = ggml_init(out_init);
    if (!out_ggml_ctx) {
        if (err_msg) {
            *err_msg = "ggml_init failed for output tensor context";
        }
        gguf_free(out_ctx);
        gguf_free(in_ctx);
        ggml_free(ggml_ctx);
        return 1;
    }

    // Quant-result buffers are referenced by the GGUF tensor descriptors by
    // data pointer, and gguf_write_to_file reads through those pointers after
    // the walk below completes. The results must therefore outlive the write,
    // so they are kept in function-scope deques (stable element addresses)
    // rather than as per-iteration locals.
    std::deque<ts_quant_result_2d>              cluster_results; // 2D weights
    std::deque<std::vector<ts_quant_result_2d>> moe_results;     // 3D MoE weights

    // --- step 7: walk tensors, quantize or copy through ---
    ts_quant_params_2d qparams;
    qparams.alpha          = default_alpha;
    qparams.clip           = params->awq_clip;
    qparams.max_outliers   = 0;
    qparams.outlier_thresh = params->outlier_frac;
    qparams.use_imatrix    = false;
    qparams.use_septq      = false;
    qparams.awq_grid       = 20;
    qparams.seed           = (uint32_t)params->evolve_seed;

    // S9 W4A4 activation quantization config. The weight-only contract is
    // unchanged when w4a4 is false; when true the per-tensor activation scales
    // and LLM.int8 outlier decomposition are computed from the calibration
    // activations and recorded as sidecar metadata.
    ts_w4a4_config wcfg = ts_w4a4_default_config();
    wcfg.enable         = params->w4a4;
    wcfg.outlier_thresh = params->w4a4_outlier_thresh > 0.0f
                              ? params->w4a4_outlier_thresh : wcfg.outlier_thresh;
    if (params->w4a4 && verbose) {
        printf("tessera-dispatch: W4A4 enabled (bits=%d scale_mode=%s outlier_thresh=%.2f)\n",
               wcfg.activation_bits, ts_w4a4_scale_mode_str(wcfg.scale_mode).c_str(),
               wcfg.outlier_thresh);
    }

    float total_mse = 0.0f;
    int64_t n_quantized = 0;
    int64_t n_skipped   = 0;

    // policy JSON accumulator
    std::string policy_json = "{\n  \"tensors\": [\n";
    bool first_policy_entry = true;

    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(in_ctx, i);
        const enum ggml_type type = gguf_get_tensor_type(in_ctx, i);
        const int64_t * ne = gguf_get_tensor_ne(in_ctx, i);

        // count dimensions (ne[d] == 1 for d >= n_dims)
        int n_dims = GGML_MAX_DIMS;
        while (n_dims > 1 && ne[n_dims - 1] == 1) {
            n_dims--;
        }

        struct ggml_tensor * t = ggml_get_tensor(ggml_ctx, name);
        if (t == nullptr) {
            fprintf(stderr, "tessera-dispatch: warning: tensor '%s' not found in ggml context, skipping\n", name);
            n_skipped++;
            continue;
        }

        if (!ts_is_quantizable(name, type, n_dims)) {
            // copy through unchanged
            gguf_add_tensor(out_ctx, t);
            n_skipped++;
            if (verbose) {
                printf("tessera-dispatch: copy-through %s (%s)\n", name, ggml_type_name(type));
            }
            continue;
        }

        // quantizable weight matrix
        const std::string family = ts_regime_infer_family(name);
        const int64_t in_dim  = ne[0];
        const int64_t out_dim = ne[1];

        // convert to F32
        std::vector<float> weights = ts_tensor_to_f32(t);
        if (weights.empty()) {
            fprintf(stderr, "tessera-dispatch: warning: unsupported type for '%s', copying through\n", name);
            gguf_add_tensor(out_ctx, t);
            n_skipped++;
            continue;
        }

        // resolve per-channel AWQ activation scales (imatrix, else corpus)
        std::vector<float> act_scratch;
        const float * act_scales = ts_dispatch_act_scales(
            have_imatrix ? &imatrix : nullptr, name, in_dim,
            calib_X.empty() ? nullptr : calib_X.data(), calib_in_dim, calib_n_tokens,
            &act_scratch);

        // imatrix regime stats for the descriptor (nullptr when unavailable)
        const float * imdata = nullptr;
        int64_t       imdim  = 0;
        if (have_imatrix) {
            imdata = ts_imatrix_lookup(&imatrix, name, &imdim);
        }

        // compute regime descriptor and route
        ts_regime_descriptor desc = ts_regime_compute_descriptor(
            name, weights.data(), out_dim, in_dim,
            imdata, imdata ? imdim : 0);

        // multimodal: resolve the operative modality, the per-modality AWQ
        // alpha, and the per-modality activation scales for this tensor.
        int   mm_mod        = desc.modality;
        float mm_alpha[3]   = { 0.0f, 0.0f, 0.0f };
        bool  have_mm       = false;
        bool  have_mm_alpha = false;
        if (have_mm_imatrix) {
            const ts_mm_imatrix_entry * en =
                ts_dispatch_mm_resolve(&mm_imatrix, name, desc.modality, &mm_mod);
            desc.modality = mm_mod;
            if (en != nullptr) {
                have_mm = true;
                auto it = mm_awq.find(name);
                if (it == mm_awq.end()) {
                    ts_mm_awq_result mres;
                    if (ts_dispatch_mm_awq(&mm_imatrix, name, weights.data(),
                                           out_dim, in_dim, &mres) == 0) {
                        it = mm_awq.emplace(name, std::move(mres)).first;
                    }
                }
                if (it != mm_awq.end()) {
                    for (int m = 0; m < 3; m++) {
                        mm_alpha[m] = it->second.best_alpha[m];
                    }
                    have_mm_alpha = true;
                }
                // per-modality act_scales for the operative modality override the
                // text rollup so the quantizer's act_scale field is modality-specific
                int64_t ad = 0;
                const float * ma = ts_mm_imatrix_act_scales(
                    &mm_imatrix, name, (ts_modality)mm_mod, &ad);
                if (ma != nullptr && ad == in_dim) {
                    act_scales = ma;
                }
            }
        }

        ts_regime_routing routing = ts_regime_classify(&desc);

        // per-tensor alpha: per-modality MM alpha > GA result > default
        float tensor_alpha;
        if (have_mm && have_mm_alpha) {
            tensor_alpha = mm_alpha[mm_mod];
        } else if (need_ga && ga_alpha.count(name)) {
            tensor_alpha = ga_alpha[name];
        } else {
            tensor_alpha = default_alpha;
        }

        // apply the routed expert's profile to a per-tensor copy of the base
        // params (qparams is shared across the loop, so never mutate it here)
        ts_expert_profile  profile = ts_expert_default_profile(routing.expert, desc.modality);
        ts_quant_params_2d tqp     = qparams;
        tqp.alpha           = tensor_alpha * profile.alpha_scale;
        tqp.clip           *= profile.clip_scale;
        tqp.use_septq       = profile.use_septq;
        tqp.awq_grid        = profile.awq_grid;
        tqp.max_outliers    = profile.max_outliers;
        tqp.outlier_thresh *= profile.outlier_thresh;

        if (verbose) {
            printf("tessera-dispatch: %s family=%s modality=%d expert=%s reason='%s' "
                   "alpha=%.3f clip=%.3f grid=%d outliers=%d septq=%d\n",
                   name, family.c_str(), (int)desc.modality, ts_expert_name(routing.expert),
                   routing.reason.c_str(), tqp.alpha, tqp.clip,
                   (int)tqp.awq_grid, (int)tqp.max_outliers, (int)tqp.use_septq);
        }

        // stamp the routed expert + applied profile onto the per-tensor result
        auto fill_expert_meta = [&](ts_dispatch_tensor_result & out) {
            out.expert_id              = (int)routing.expert;
            out.expert_name            = ts_expert_name(routing.expert);
            out.profile_alpha          = tqp.alpha;
            out.profile_clip           = tqp.clip;
            out.profile_awq_grid       = (int)tqp.awq_grid;
            out.profile_max_outliers   = (int)tqp.max_outliers;
            out.profile_outlier_thresh = tqp.outlier_thresh;
            out.profile_use_septq      = tqp.use_septq;
            out.modality_id            = (int)desc.modality;
            for (int m = 0; m < 3; m++) {
                out.modality_alpha[m] = mm_alpha[m];
            }
        };

        // S9 W4A4 sidecar for this tensor (populated in the 2D branch when
        // params->w4a4 is set; MoE 3D W4A4 is deferred). w4a4_policy_json is
        // appended to the per-tensor policy / receipt entry below.
        ts_w4a4_sidecar w4a4_sc = {};
        std::string     w4a4_policy_json;

        if (n_dims == 3) {
            // MoE expert tensor: (n_experts x out_dim x in_dim)
            const int64_t n_experts = ne[2];

            std::vector<ts_quant_result_2d> qresults;
            int rc = ts_quantize_3d(weights.data(),
                                    act_scales, nullptr, nullptr, act_scales,
                                    n_experts, out_dim, in_dim, 0,
                                    &tqp, &qresults);
            if (rc != 0) {
                if (err_msg) {
                    *err_msg = "ts_quantize_3d failed for " + std::string(name);
                }
                ggml_free(out_ggml_ctx);
                gguf_free(out_ctx);
                gguf_free(in_ctx);
                ggml_free(ggml_ctx);
                return 2;
            }

            // keep the expert results alive until after gguf_write_to_file
            moe_results.push_back(std::move(qresults));
            const std::vector<ts_quant_result_2d> & qr_keep = moe_results.back();

            // write per-expert clusters
            for (int64_t e = 0; e < n_experts; e++) {
                char exp_name[GGML_MAX_NAME];
                snprintf(exp_name, sizeof(exp_name), "%s.%lld", name, (long long)e);
                ts_gguf_write_tensor_cluster(out_ctx, out_ggml_ctx, exp_name, &qr_keep[(size_t)e], out_dim, in_dim);
                total_mse += qr_keep[(size_t)e].mse;
            }

            ts_dispatch_tensor_result tr;
            tr.name    = name;
            tr.family  = family;
            tr.out_dim = out_dim;
            tr.in_dim  = in_dim;
            fill_expert_meta(tr);
            // aggregate first expert's blobs for the result struct
            if (!qr_keep.empty()) {
                tr.packed              = ts_to_bytes_u32(qr_keep[0].packed);
                tr.page_scales         = ts_to_bytes_u16(qr_keep[0].page_scales);
                tr.lane_scales         = ts_to_bytes_i8(qr_keep[0].lane_scales);
                tr.outlier_row_offsets = ts_to_bytes_i32(qr_keep[0].outlier_row_offsets);
                tr.outlier_cols        = ts_to_bytes_i32(qr_keep[0].outlier_cols);
                tr.outlier_vals        = ts_to_bytes_u16(qr_keep[0].outlier_vals);
                tr.act_scale           = ts_to_bytes_u16(qr_keep[0].act_scale);
                tr.mse                 = qr_keep[0].mse;
                tr.alpha_used          = qr_keep[0].best_alpha;
            }
            result->tensors.push_back(std::move(tr));
            n_quantized++;

            if (verbose) {
                printf("tessera-dispatch: quantized %s (3D, %lld experts)\n",
                       name, (long long)n_experts);
            }
        } else {
            // standard 2D weight. The result lives in cluster_results (function
            // scope) so the buffers referenced by the GGUF descriptors stay
            // valid until gguf_write_to_file runs after the walk completes.
            ts_quant_result_2d &  qr = cluster_results.emplace_back();
            ts_w4a4_weight_result wres;
            wres.base = &qr;

            // W4A4 routes through the activation-aware wrapper when the
            // calibration width matches this tensor; otherwise the existing
            // weight-only path runs and (when w4a4 is on) the activation
            // metadata is still recorded from any width-matching calibration.
            const bool calib_match = params->w4a4 && !calib_X.empty() &&
                                     calib_in_dim == in_dim && calib_n_tokens > 0;
            int rc;
            if (calib_match) {
                rc = ts_w4a4_quantize_weights(weights.data(), calib_X.data(),
                                              out_dim, in_dim, calib_n_tokens,
                                              &tqp, &wcfg, &qr, &wres);
            } else {
                rc = ts_quantize_2d(weights.data(),
                                    act_scales,   // act_scales
                                    nullptr,      // calib_X
                                    nullptr,      // ref_output
                                    act_scales,   // imatrix
                                    out_dim, in_dim, 0,
                                    &tqp, &qr);
                if (rc == 0 && params->w4a4) {
                    const float * cx = (!calib_X.empty() && calib_in_dim == in_dim)
                                           ? calib_X.data() : nullptr;
                    const int64_t ct = (cx != nullptr) ? calib_n_tokens : 0;
                    ts_w4a4_detect_outliers(cx, ct, in_dim, &wcfg, &wres.outliers);
                    ts_w4a4_compute_act_scales(cx, ct, in_dim, &wcfg, &wres.scales);
                }
            }
            if (rc != 0) {
                if (err_msg) {
                    *err_msg = "ts_quantize_2d failed for " + std::string(name);
                }
                ggml_free(out_ggml_ctx);
                gguf_free(out_ctx);
                gguf_free(in_ctx);
                ggml_free(ggml_ctx);
                return 2;
            }

            ts_gguf_write_tensor_cluster(out_ctx, out_ggml_ctx, name, &qr, out_dim, in_dim);

            ts_dispatch_tensor_result tr;
            tr.name                = name;
            tr.family              = family;
            tr.out_dim             = out_dim;
            tr.in_dim              = in_dim;
            fill_expert_meta(tr);
            tr.packed              = ts_to_bytes_u32(qr.packed);
            tr.page_scales         = ts_to_bytes_u16(qr.page_scales);
            tr.lane_scales         = ts_to_bytes_i8(qr.lane_scales);
            tr.outlier_row_offsets = ts_to_bytes_i32(qr.outlier_row_offsets);
            tr.outlier_cols        = ts_to_bytes_i32(qr.outlier_cols);
            tr.outlier_vals        = ts_to_bytes_u16(qr.outlier_vals);
            tr.act_scale           = ts_to_bytes_u16(qr.act_scale);
            tr.mse                 = qr.mse;
            tr.alpha_used          = qr.best_alpha;

            // S9 W4A4 sidecar metadata + per-tensor receipt entry
            if (params->w4a4) {
                tr.w4a4_enabled          = true;
                tr.w4a4_activation_bits  = wcfg.activation_bits;
                tr.w4a4_scale_mode       = ts_w4a4_scale_mode_str(wcfg.scale_mode);
                tr.w4a4_outlier_frac     = wres.outliers.frac;
                tr.w4a4_act_scale_static = wres.scales.per_tensor;
                tr.w4a4_outlier_channels = wres.outliers.channels;

                w4a4_sc.enabled          = true;
                w4a4_sc.activation_bits  = wcfg.activation_bits;
                w4a4_sc.scale_mode       = wcfg.scale_mode;
                w4a4_sc.outlier_frac     = wres.outliers.frac;
                w4a4_sc.act_scale_static = wres.scales.per_tensor;
                w4a4_sc.outlier_channels = wres.outliers.channels;
                w4a4_policy_json         = ", " + ts_w4a4_sidecar_json(&w4a4_sc);

                if (verbose) {
                    printf("tessera-dispatch: %s w4a4 outliers=%zu frac=%.5f eff_bits=%.3f\n",
                           name, wres.outliers.channels.size(), wres.outliers.frac,
                           wres.effective_bits);
                }
            }

            total_mse += qr.mse;
            result->tensors.push_back(std::move(tr));
            n_quantized++;

            if (verbose) {
                printf("tessera-dispatch: quantized %s (mse=%.6f alpha=%.3f)\n",
                       name, qr.mse, qr.best_alpha);
            }
        }

        // accumulate policy entry
        if (!first_policy_entry) {
            policy_json += ",\n";
        }
        first_policy_entry = false;
        policy_json += "    {\"name\": \"" + std::string(name) + "\", "
                     + "\"family\": \"" + family + "\", "
                     + "\"modality\": " + std::to_string((int)desc.modality) + ", "
                     + "\"modality_alpha\": [" + std::to_string(mm_alpha[0]) + ", "
                     + std::to_string(mm_alpha[1]) + ", "
                     + std::to_string(mm_alpha[2]) + "], "
                     + "\"expert\": " + std::to_string((int)routing.expert) + ", "
                     + "\"expert_name\": \"" + ts_expert_name(routing.expert) + "\", "
                     + "\"alpha\": " + std::to_string(tensor_alpha) + ", "
                     + "\"profile\": {"
                     + "\"alpha\": " + std::to_string(tqp.alpha) + ", "
                     + "\"clip\": " + std::to_string(tqp.clip) + ", "
                     + "\"awq_grid\": " + std::to_string(tqp.awq_grid) + ", "
                     + "\"max_outliers\": " + std::to_string(tqp.max_outliers) + ", "
                     + "\"outlier_thresh\": " + std::to_string(tqp.outlier_thresh) + ", "
                     + "\"use_septq\": " + (tqp.use_septq ? "true" : "false")
                     + "}" + w4a4_policy_json + "}";
    }

    policy_json += "\n  ]\n}";

    // --- step 7b: G6 acceptance gate ---
    result->acceptance_ran = false;
    if (params->run_acceptance) {
        std::vector<ts_acceptance_tensor> acc_tensors;

        for (int64_t i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(in_ctx, i);
            const enum ggml_type type = gguf_get_tensor_type(in_ctx, i);
            const int64_t * ne = gguf_get_tensor_ne(in_ctx, i);
            int nd = GGML_MAX_DIMS;
            while (nd > 1 && ne[nd - 1] == 1) nd--;

            if (nd != 2 || !ts_is_quantizable(name, type, nd)) continue;

            struct ggml_tensor * t = ggml_get_tensor(ggml_ctx, name);
            if (!t) continue;

            std::vector<float> w = ts_tensor_to_f32(t);
            if (w.empty()) continue;

            const int64_t in_dim  = ne[0];
            const int64_t out_dim = ne[1];

            std::vector<float> act_scratch;
            const float * act = ts_dispatch_act_scales(
                have_imatrix ? &imatrix : nullptr, name, in_dim,
                calib_X.empty() ? nullptr : calib_X.data(), calib_in_dim, calib_n_tokens,
                &act_scratch);

            const float alpha = default_alpha;
            const float clip  = params->awq_clip;
            const float othresh = params->outlier_frac;
            const uint32_t seed = (uint32_t)params->evolve_seed;

            // composite: regime-routed
            ts_regime_descriptor desc = ts_regime_compute_descriptor(
                name, w.data(), out_dim, in_dim, nullptr, 0);
            ts_regime_routing routing = ts_regime_classify(&desc);
            float comp_t2 = ts_dispatch_forced_t2(
                w.data(), act, out_dim, in_dim, routing.expert,
                alpha, clip, othresh, seed);

            ts_acceptance_tensor at;
            memset(&at, 0, sizeof(at));
            snprintf(at.name, sizeof(at.name), "%s", name);
            at.composite_t2      = comp_t2;
            at.awq_t2            = ts_dispatch_forced_t2(w.data(), act, out_dim, in_dim, TS_EXPERT_AWQ,       alpha, clip, othresh, seed);
            at.rotation_t2       = ts_dispatch_forced_t2(w.data(), act, out_dim, in_dim, TS_EXPERT_DARTQUANT, alpha, clip, othresh, seed);
            at.lowrank_t2        = ts_dispatch_forced_t2(w.data(), act, out_dim, in_dim, TS_EXPERT_FLRQ,      alpha, clip, othresh, seed);
            at.hessian_t2        = ts_dispatch_forced_t2(w.data(), act, out_dim, in_dim, TS_EXPERT_SEPTQ,     alpha, clip, othresh, seed);
            at.offline_proxy_mse = comp_t2;
            at.kernel_direct_t2  = comp_t2;  // no sidecar in standalone dispatch
            at.held_out          = false;    // fraction-based fallback in ts_acceptance_run
            acc_tensors.push_back(at);
        }

        if (!acc_tensors.empty()) {
            ts_acceptance_run(&params->acceptance_config,
                              acc_tensors.data(), (int64_t)acc_tensors.size(),
                              &result->acceptance);
            result->acceptance_ran = true;
            if (verbose) {
                printf("tessera-dispatch: acceptance: %s\n", result->acceptance.verdict);
            }
        }
    }

    // --- step 8: write tessera metadata ---
    ts_gguf_writer_params wparams;
    wparams.seed           = (uint32_t)params->evolve_seed;
    wparams.alpha          = default_alpha;
    wparams.clip           = params->awq_clip;
    wparams.outlier_frac   = params->outlier_frac;
    wparams.policy_summary = policy_json;
    wparams.policy_sha256  = "";
    wparams.build_info     = "";
    wparams.main_tip       = "";
    wparams.w4a4_enabled         = params->w4a4;
    wparams.w4a4_activation_bits = wcfg.activation_bits;
    wparams.w4a4_scale_mode      = ts_w4a4_scale_mode_str(wcfg.scale_mode);
    wparams.w4a4_outlier_thresh  = wcfg.outlier_thresh;
    ts_gguf_write_metadata(out_ctx, &wparams);

    // --- step 9: write output GGUF ---
    if (!params->output_path.empty()) {
        if (!gguf_write_to_file(out_ctx, params->output_path.c_str(), false)) {
            if (err_msg) {
                *err_msg = "failed to write output GGUF: " + params->output_path;
            }
            ggml_free(out_ggml_ctx);
            gguf_free(out_ctx);
            gguf_free(in_ctx);
            ggml_free(ggml_ctx);
            return 3;
        }
        if (verbose) {
            printf("tessera-dispatch: wrote '%s'\n", params->output_path.c_str());
        }
    }

    // --- step 10: write policy JSON alongside ---
    if (!params->policy_out_path.empty()) {
        std::ofstream pf(params->policy_out_path);
        if (pf.is_open()) {
            pf << policy_json << "\n";
            if (verbose) {
                printf("tessera-dispatch: wrote policy '%s'\n", params->policy_out_path.c_str());
            }
        } else {
            fprintf(stderr, "tessera-dispatch: warning: could not write policy to '%s'\n",
                    params->policy_out_path.c_str());
        }

        // write the MAP-Elites archive sidecar alongside the policy
        if (have_archive) {
            const std::string archive_path = params->policy_out_path + ".archive.json";
            std::ofstream af(archive_path);
            if (af.is_open()) {
                af << result->archive_json << "\n";
                if (verbose) {
                    printf("tessera-dispatch: wrote archive '%s'\n", archive_path.c_str());
                }
            } else {
                fprintf(stderr, "tessera-dispatch: warning: could not write archive to '%s'\n",
                        archive_path.c_str());
            }
        }
    }

    // --- step 11: populate summary ---
    result->n_tensors_quantized = n_quantized;
    result->n_tensors_skipped   = n_skipped;
    result->total_mse           = total_mse;
    result->policy_json         = policy_json;
    result->policy_sha256       = "";

    // --- cleanup ---
    ggml_free(out_ggml_ctx);
    gguf_free(out_ctx);
    gguf_free(in_ctx);
    ggml_free(ggml_ctx);

    return 0;
}
