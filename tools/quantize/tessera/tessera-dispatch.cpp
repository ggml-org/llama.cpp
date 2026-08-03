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
#include "tessera-awq-fitness.h"
#include "tessera-regime.h"
#include "tessera-gguf-writer.h"
#include "tessera-higgs.h"
#include "tessera-higgs-cache.h"
#include "tessera-search.h"
#include "tessera-imatrix.h"
#include "tessera-corpus.h"
#include "tessera-l1-fitness.h"
#include "tessera-l2-diff.h"
#include "tessera-l5.h"
#include "tessera-ab-harness.h"
#include "tessera-mm-imatrix.h"
#include "tessera-mm-fitness.h"
#include "tessera-mm-awq.h"
#include "tessera-w4a4.h"
#include "tessera-acceptance.h"
#include "tessera-policy.h"
#include "tessera-progress.h"
#include "tessera-sharded-map.h"
#include "tessera-quantize-db.h"
#include "tessera-vec.h"

#include "gguf.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <atomic>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <sstream>

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
    // Accept any type that has a dequant path. This lets the pipeline consume
    // any source GGUF format (f32, f16, q8_0, q4_0, etc) since ts_tensor_to_f32
    // uses the generic ggml type-traits dequant. Skip the destination type
    // (TESSERA_T640) to avoid re-quantizing an already-quantized model.
    if (type == GGML_TYPE_TESSERA_T640) {
        return false;
    }
    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16) {
        const struct ggml_type_traits * traits = ggml_get_type_traits(type);
        if (!traits || !traits->to_float) {
            return false;
        }
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

    if (t->data == nullptr) {
        out.clear();
        return out;
    }
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, (size_t)n * sizeof(float));
    } else {
        // Generic dequant path via the type traits table. Handles F16, Q8_0,
        // and any other registered ggml type. This lets the pipeline consume
        // any source GGUF format, not just f16/f32.
        const struct ggml_type_traits * traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) {
            traits->to_float(t->data, out.data(), n);
        } else {
            out.clear();
        }
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
//
// The caches are sharded concurrent maps (ts_sharded_map). The GA fans
// per-tensor work across threads; each tensor is written by exactly one
// worker, but a plain unordered_map would race on rehash across different
// keys. Sharding by key hash gives lock-free operation for the common case
// (two threads touching different keys land in different shards) and a
// fine-grained lock for the rare same-shard collision. The `mode_prints`
// counter is the only remaining shared mutable state; it uses a relaxed
// atomic since the exact count does not matter (just a log-rate cap).
struct ts_dispatch_eval_ctx {
    float    outlier_thresh;
    uint32_t seed;
    ts_l1_fitness_config l1;
    bool     verbose;
    std::atomic<int> mode_prints{0};   // tensors whose fitness mode has been logged
    ts_sharded_map<std::string, std::vector<float>>      sidecar_cache;
    ts_sharded_map<std::string, float>                   best_t2;
    ts_sharded_map<std::string, std::pair<float, float>> best_pair;
    const std::unordered_map<std::string, int> * modality = nullptr;  // per-tensor operative modality (read-only)
};

// GA evaluator: quantize the layer with a candidate (alpha, clip) and score
// it. The GA maximizes `composite`, so report the negative t_l^2 (lower error
// -> higher fitness). By default t_l^2 is the offline relative Frobenius proxy
// ||W_hat - W||_F^2 / ||W||_F^2; with S5 kernel-direct fitness enabled it is
// blended with ||W_hat - dequant_kernel||_F^2 / ||W||_F^2, where
// dequant_kernel is the tensor's L1 sidecar (the kernel's real output).
//
// B2: when the layer carries per-channel second_moment telemetry (the imatrix
// E[x^2]), the evaluator additionally computes the Python parity fitness
// (ts_awq_evaluate_layer, a faithful port of awq-evolve.py:_evaluate_layer)
// and uses its train_error as the composite driver. This makes the GA
// optimize the same reconstruction-error objective Python's _evaluate_uncached
// optimizes, instead of the offline Frobenius proxy. The S5 kernel-direct
// path still runs (and still feeds the A/B harness) when use_kernel_direct is
// set, but the composite reported back to the GA is the Python fitness so
// the two stay aligned.
static ts_awq_score ts_dispatch_awq_eval(const ts_awq_candidate * cand,
                                         const ts_awq_layer * layer,
                                         void * ctx) {
    ts_dispatch_eval_ctx * ec = (ts_dispatch_eval_ctx *)ctx;

    ts_awq_score score;
    score.mse           = std::numeric_limits<float>::infinity();
    score.relative_frob = std::numeric_limits<float>::infinity();
    score.heldout_mse   = std::numeric_limits<float>::infinity();
    score.composite     = -std::numeric_limits<float>::infinity();

    // B2 Python parity path: when second_moment is present, compute the
    // awq-evolve.py fitness directly. This is the canonical "match Python"
    // objective. We still run the S5 path below to record best_t2/best_pair
    // for the A/B harness when use_kernel_direct is on.
    ts_awq_score py_score;
    bool have_py = false;
    if (layer->second_moment != nullptr) {
        if (ts_awq_evaluate_layer(*cand, *layer, &py_score) == 0) {
            have_py = true;
        }
    }

    // route the layer to its expert and apply that expert's profile so the
    // GA scores candidates under the same knobs the final quantize uses
    int mod = 0;
    if (ec->modality) {
        auto mit = ec->modality->find(layer->name);
        if (mit != ec->modality->end()) {
            mod = mit->second;
        }
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
    qp.alpha          = cand->genes.alpha * prof.alpha_scale;
    qp.clip           = cand->genes.clip * prof.clip_scale;
    qp.max_outliers   = prof.max_outliers;
    qp.outlier_thresh = ec->outlier_thresh * prof.outlier_thresh;
    qp.use_imatrix    = layer->imatrix != nullptr;
    qp.use_septq      = prof.use_septq;
    qp.awq_grid       = prof.awq_grid;
    qp.seed           = ec->seed;

    // Use the streaming MSE evaluator when kernel-direct fitness is OFF (the
    // default). This avoids the 700 MB per-candidate allocation of the full
    // ts_quantize_2d (ws + core + recon + packed + scales) and instead uses
    // ~132 KB of per-row scratch. The MSE is bit-identical. When kernel-direct
    // is ON, fall back to the full path because it needs qr.recon.
    const int64_t n = layer->out_dim * layer->in_dim;
    float mse_val;
    ts_quant_result_2d qr;  // only populated when kernel-direct is on
    if (!ec->l1.use_kernel_direct) {
        // Streaming path: O(in_dim) scratch per candidate.
        mse_val = ts_quantize_mse_streaming(
            layer->weights, layer->act_scales,
            qp.alpha, qp.clip,
            layer->out_dim, layer->in_dim);
        if (mse_val < 0.0f) {
            return score;  // worst possible fitness
        }
    } else {
        // Full path: needs qr.recon for kernel-direct t2.
        int rc = ts_quantize_2d(layer->weights, layer->act_scales,
                                layer->calib_X, layer->ref_output, layer->imatrix,
                                layer->out_dim, layer->in_dim, layer->n_tokens,
                                &qp, &qr);
        if (rc != 0) {
            return score;   // worst possible fitness
        }
        mse_val = qr.mse;
    }

    // mse is the mean squared reconstruction error, so
    // ||W_hat - W||_F^2 = mse * n.
    float frob2 = ts_vec_dotpr(layer->weights, layer->weights, n);
    float rel_frob = (frob2 > 0.0f) ? (mse_val * (float)n / frob2) : mse_val;

    // S5: kernel-direct t_l^2 from the L1 sidecar, blended with the proxy.
    float t2    = rel_frob;
    float kd_t2 = rel_frob;   // falls back to the proxy when no sidecar exists
    if (ec->l1.use_kernel_direct) {
        // Lazy sidecar load: at most one disk read per tensor (the with_lock
        // lambda runs atomically under the shard lock, so the first caller
        // for a given tensor name populates the cache and subsequent callers
        // for the SAME tensor see the entry). Different tensors hash to
        // different shards and load concurrently with no contention.
        const std::vector<float> * kdeq_ptr = ec->sidecar_cache.with_lock(
            layer->name,
            [&](std::unordered_map<std::string, std::vector<float>> & m,
               const std::string & key) -> const std::vector<float> *
        {
            auto it = m.find(key);
            if (it != m.end()) {
                return &it->second;
            }
            std::vector<float> kdeq;
            int64_t sr = 0;
            int64_t sc = 0;
            if (ec->l1.sidecar_dir[0] != '\0' &&
                ts_l1_load_sidecar(ec->l1.sidecar_dir, key.c_str(),
                                   &kdeq, &sr, &sc) == 0 && sr * sc == n) {
                m.emplace(key, std::move(kdeq));
            } else {
                m.emplace(key, std::vector<float>());
            }
            it = m.find(key);
            if (ec->verbose && ec->mode_prints.load(std::memory_order_relaxed) < 3) {
                printf("tessera-dispatch: kernel-fitness: %s -> %s\n",
                       key.c_str(),
                       it->second.empty() ? "offline proxy (no sidecar)"
                                          : "kernel-direct (L1 sidecar)");
                ec->mode_prints.fetch_add(1, std::memory_order_relaxed);
            }
            return &it->second;
        });
        if (!kdeq_ptr->empty() && (int64_t)kdeq_ptr->size() == n &&
            (int64_t)qr.recon.size() == n) {
            kd_t2 = ts_l1_kernel_direct_t2(qr.recon.data(), layer->weights,
                                           kdeq_ptr->data(), n);
            t2    = ts_l1_blended_t2(rel_frob, kd_t2, ec->l1.blend_factor);
        }
    }

    // record the best candidate's (offline, kernel) pair for the A/B harness.
    // Two threads in the same layer can race here (same GA layer evaluated by
    // different generations within ts_awq_evolve, but ts_awq_evolve evaluates
    // candidates serially per layer), so the shard lock is sufficient.
    if (ec->l1.use_kernel_direct) {
        ec->best_t2.with_lock(layer->name,
            [&](std::unordered_map<std::string, float> & m,
               const std::string & key)
        {
            auto bit = m.find(key);
            if (bit == m.end() || t2 < bit->second) {
                m[key] = t2;
                ec->best_pair.assign(key, std::make_pair(rel_frob, kd_t2));
            }
        });
    }

    score.mse           = qr.mse;
    score.relative_frob = t2;       // t_l^2 used for fitness (blended when kernel-direct)
    score.heldout_mse   = qr.mse;   // no held-out split in standalone dispatch
    score.composite     = -t2;

    // B2: when the Python parity fitness was computed, drive the GA from it.
    // ts_awq_evaluate_layer leaves composite at 0, so derive the GA composite
    // from train_error (lower-is-better -> negate). The Python fitness is the
    // same reconstruction-error objective awq-evolve.py optimizes; using it
    // here keeps the dispatch GA aligned with the reference implementation.
    if (have_py) {
        score.mse         = py_score.mse;
        score.heldout_mse = py_score.heldout_mse;
        // relative_frob carries the kernel-direct t_l^2 above when S5 is on;
        // otherwise use the Python tail_error so the reported score still
        // reflects a meaningful reconstruction metric.
        if (!ec->l1.use_kernel_direct) {
            score.relative_frob = py_score.relative_frob;
        }
        score.composite   = -py_score.mse;
    }
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

    float resolved_alpha = base_alpha * prof.alpha_scale;
    float resolved_clip  = base_clip * prof.clip_scale;

    // Streaming MSE path: ~132 KB scratch instead of 700 MB.
    float mse = ts_quantize_mse_streaming(
        weights, act_scales, resolved_alpha, resolved_clip,
        out_dim, in_dim);
    if (mse < 0.0f) {
        return 1.0f;  // worst case
    }

    const int64_t n = out_dim * in_dim;
    float frob2 = ts_vec_dotpr(weights, weights, n);
    return (frob2 > 0.0f) ? (mse * (float)n / frob2) : mse;
}

// ---------------------------------------------------------------------------
// L5 adaptive requantize refine loop (dispatch L2 -> L5 -> re-quantize)
// ---------------------------------------------------------------------------

#include "tessera-dispatch-internal.h"

// Re-read one tensor's source weights from the input GGUF as F32. Returns
// an empty vector on failure (matching ts_tensor_to_f32's contract).
static std::vector<float> ts_refine_reread_source(struct gguf_context * in_ctx,
                                                  struct ggml_context * ggml_ctx,
                                                  const ts_dispatch_refine_entry & e) {
    const char * name = gguf_get_tensor_name(in_ctx, e.gguf_idx);
    if (name == nullptr) {
        return {};
    }
    struct ggml_tensor * t = ggml_get_tensor(ggml_ctx, name);
    if (t == nullptr) {
        return {};
    }
    return ts_tensor_to_f32(t);
}

// Relative Frobenius between source weights and a quant result's recon.
// Matches the L2 weight-level metric (||src - recon||_F^2 / ||src||_F^2).
static float ts_refine_rel_frob(const float * src, const ts_quant_result_2d * qr,
                                int64_t n) {
    if (src == nullptr || qr == nullptr || n <= 0) {
        return 0.0f;
    }
    double num = 0.0;
    double den = 0.0;
    const float * recon = qr->recon.data();
    for (int64_t i = 0; i < n; i++) {
        const double d = (double)src[i] - (double)recon[i];
        num += d * d;
        den += (double)src[i] * (double)src[i];
    }
    if (den == 0.0) {
        return 0.0f;
    }
    return (float)(num / den);
}

// Join a vector of strings with a separator (used for the per-family JSON
// array in the report).
static std::string ts_join(const std::vector<std::string> & parts,
                           const std::string & sep) {
    std::string out;
    for (size_t i = 0; i < parts.size(); i++) {
        if (i) out += sep;
        out += parts[i];
    }
    return out;
}

// ---------------------------------------------------------------------------
// DuckDB store plumbing
//
// ts_dispatch_db is the per-run state threaded through the GA hooks. It owns
// the open ts_quantize_db plus the per-tensor appenders used to bulk-log GA
// candidate evaluations. The appenders live in a sharded map keyed by tensor
// name so concurrent GA workers (one per layer) each get their own appender
// without contending; DuckDB Appenders are not thread-safe across rows of the
// same table, but distinct Appenders on the same Connection are.
//
// All DB calls are best-effort: a failure logs a one-line warning and the
// pipeline continues. The DB is a recording/warm-start aid, not a correctness
// requirement, so a corrupt file or full disk must never block quantization.
struct ts_dispatch_db {
    ts_quantize_db * db = nullptr;     // owned; null when --quantize-db is unset
    std::string      run_id;           // empty until begin_run succeeds
    std::string      model_hash;       // empty when hashing failed
    // Resume set: tensor names with a converged result for this run_id.
    // Populated at startup; the layer_skip_lookup callback checks membership.
    std::unordered_set<std::string> converged;
    // Open appenders keyed by tensor name. Each GA worker opens one on first
    // eval_recorder call for its layer and flushes+closes it when evolve_all
    // returns. The map's shard lock serializes open/close but per-row appends
    // are uncontended once a worker holds its appender pointer (only that
    // worker writes to it for the layer's GA duration).
    ts_sharded_map<std::string, ts_quantize_db_appender *> appenders;
    // Counter for eval rows lost to appender failures (best-effort logging;
    // reported once at run end so a silent DB problem is visible).
    std::atomic<int64_t> eval_rows_dropped{0};
};

// Open the store and begin a run. Returns a heap-allocated ts_dispatch_db
// (owned by the caller via unique_ptr) or nullptr on failure / when the path
// is empty. model_path is hashed to fingerprint this run for warm-start.
static ts_dispatch_db * ts_dispatch_db_open(
    const ts_dispatch_params * params, bool verbose)
{
    if (params->quantize_db_path.empty()) {
        return nullptr;
    }
    ts_dispatch_db * wrap = new (std::nothrow) ts_dispatch_db();
    if (wrap == nullptr) return nullptr;
    std::string err;
    wrap->db = ts_quantize_db_open(params->quantize_db_path, &err);
    if (wrap->db == nullptr) {
        fprintf(stderr, "tessera-dispatch: warning: --quantize-db open '%s' "
                        "failed: %s (continuing without DB)\n",
                params->quantize_db_path.c_str(), err.c_str());
        return nullptr;
    }
    wrap->model_hash = ts_quantize_db_hash_gguf(params->input_path);
    // config_json: a compact summary of the knobs that affect GA output. This
    // is informational; warm-start keying uses model_hash only (config drift
    // across runs is acceptable for the family-seed query, which ranks by
    // best_composite rather than filtering by config).
    std::string config = "{\"evolve_iters\":" + std::to_string(params->evolve_iters)
        + ",\"evolve_islands\":" + std::to_string(params->evolve_islands)
        + ",\"evolve_population\":" + std::to_string(params->evolve_population)
        + ",\"outlier_frac\":" + std::to_string(params->outlier_frac)
        + ",\"awq_clip\":" + std::to_string(params->awq_clip)
        + ",\"awq_alpha\":\"" + params->awq_alpha + "\"}";
    wrap->run_id = ts_quantize_db_begin_run(wrap->db, params->input_path,
                                            wrap->model_hash,
                                            "tessera-dev", config, &err);
    if (wrap->run_id.empty()) {
        fprintf(stderr, "tessera-dispatch: warning: begin_run failed: %s "
                        "(continuing without DB)\n", err.c_str());
        return nullptr;
    }
    // Resume: pull the list of tensors that already converged for this
    // model_hash in prior runs (or this run, if it crashed mid-way). We look
    // across all runs of the same model so a re-launch after a crash
    // continues from the last checkpoint. force_requantize skips this.
    if (!params->force_requantize && !wrap->model_hash.empty()) {
        // Find any run_id for this model_hash with completed tensors. Goes
        // through the public list_converged_for_model helper so the dispatch
        // does not reach into the duckdb::Connection (which would require
        // pulling duckdb.hpp into every translation unit that includes the
        // dispatch).
        std::vector<std::string> done;
        if (ts_quantize_db_list_converged_for_model(wrap->db,
                                                    wrap->model_hash,
                                                    &done) == 0) {
            wrap->converged.insert(done.begin(), done.end());
        }
    }
    if (verbose) {
        printf("tessera-dispatch: --quantize-db opened '%s' (run_id=%s, "
               "model_hash=%s, %zu converged tensors)\n",
               params->quantize_db_path.c_str(), wrap->run_id.c_str(),
               wrap->model_hash.empty() ? "(hash failed)" : wrap->model_hash.c_str(),
               wrap->converged.size());
    }
    return wrap;
}

// Finalize: mark the run complete (or failed) and close any appenders left
// open by an early-return path. Called from the unique_ptr deleter so every
// exit from ts_dispatch_run cleans up.
static void ts_dispatch_db_close(ts_dispatch_db * wrap, const char * status) {
    if (wrap == nullptr || wrap->db == nullptr) return;
    if (!wrap->run_id.empty()) {
        std::string err;
        if (ts_quantize_db_complete_run(wrap->db, wrap->run_id, status, &err) != 0
            && !err.empty()) {
            fprintf(stderr, "tessera-dispatch: warning: complete_run(%s) "
                            "failed: %s\n", status, err.c_str());
        }
    }
    // Close any appenders that were never explicitly flushed (e.g. an early
    // return between evolve_all and the per-tensor close loop).
    wrap->appenders.with_lock("__close_all__",
        [&](std::unordered_map<std::string, ts_quantize_db_appender *> & m,
           const std::string &)
    {
        for (auto & kv : m) {
            ts_quantize_db_appender_flush(kv.second);
            ts_quantize_db_appender_close(kv.second);
        }
        m.clear();
    });
    if (wrap->eval_rows_dropped.load(std::memory_order_relaxed) > 0) {
        fprintf(stderr, "tessera-dispatch: warning: %lld GA eval rows dropped "
                        "(DB appender failures)\n",
                (long long)wrap->eval_rows_dropped.load(std::memory_order_relaxed));
    }
    delete wrap->db;
    wrap->db = nullptr;
}

// Per-evaluation callback: opens (lazily) and writes to the layer's appender.
// Thread-safety: the GA evaluates a layer's population across multiple
// candidate-eval threads (n_eval_threads), so this callback fires
// concurrently for the SAME layer->name. DuckDB Appenders are not thread-safe
// across rows of one appender, so the entire open+append sequence runs under
// the shard lock for that tensor name. Different tensors hash to different
// shards and append concurrently without contention.
static void ts_dispatch_eval_recorder(const ts_awq_layer * layer,
                                      int32_t generation, int32_t island,
                                      int32_t candidate_idx,
                                      const ts_awq_candidate * cand,
                                      const ts_awq_score    * score,
                                      void * user) {
    auto * wrap = static_cast<ts_dispatch_db *>(user);
    if (wrap == nullptr || wrap->db == nullptr || layer == nullptr) return;
    const std::string key = layer->name;
    ts_quantize_db_eval_row row;
    row.run_id        = wrap->run_id;
    row.tensor_name   = key;
    row.generation    = generation;
    row.island        = island;
    row.candidate_idx = candidate_idx;
    row.alpha         = cand ? cand->genes.alpha : 0.0f;
    row.clip          = cand ? cand->genes.clip  : 0.0f;
    row.composite     = score ? score->composite     : 0.0f;
    row.mse           = score ? score->mse           : 0.0f;
    row.relative_frob = score ? score->relative_frob : 0.0f;

    bool dropped = wrap->appenders.with_lock(key,
        [&](std::unordered_map<std::string, ts_quantize_db_appender *> & m,
           const std::string & k) -> bool
    {
        auto it = m.find(k);
        ts_quantize_db_appender * ap = (it != m.end()) ? it->second : nullptr;
        if (ap == nullptr) {
            std::string err;
            ap = ts_quantize_db_appender_open(wrap->db, wrap->run_id, k, &err);
            if (ap == nullptr) {
                fprintf(stderr, "tessera-dispatch: warning: appender open for "
                                "'%s' failed: %s\n", k.c_str(), err.c_str());
                return true;  // dropped
            }
            m.emplace(k, ap);
        }
        return ts_quantize_db_appender_row(ap, row) != 0;
    });
    if (dropped) {
        wrap->eval_rows_dropped.fetch_add(1, std::memory_order_relaxed);
    }
}

// Look up a family warm-start seed in the persistent store. Used as the
// family_seed_lookup hook so the first layer of each family can warm-start
// from prior runs when the in-memory cache is empty.
static bool ts_dispatch_family_seed_lookup(const char * family,
                                           ts_awq_candidate * out,
                                           float * out_composite,
                                           void * user) {
    auto * wrap = static_cast<ts_dispatch_db *>(user);
    if (wrap == nullptr || wrap->db == nullptr || family == nullptr ||
        out == nullptr) {
        return false;
    }
    ts_quantize_db_family_seed seed;
    if (!ts_quantize_db_lookup_family_seed(wrap->db, family, wrap->run_id,
                                           &seed)) {
        return false;
    }
    out->genes.alpha = seed.best_alpha;
    out->genes.clip  = seed.best_clip;
    out->expert_hint = -1;
    if (out_composite) *out_composite = seed.best_composite;
    return true;
}

// Resume hook: short-circuit the GA for tensors that already converged for
// this model in a prior run. Reconstructs a minimal result from the stored
// ga_results row so the downstream per-tensor quantize step still has alpha.
static bool ts_dispatch_layer_skip(const ts_awq_layer * layer,
                                   ts_awq_evolve_result * out_result,
                                   void * user) {
    auto * wrap = static_cast<ts_dispatch_db *>(user);
    if (wrap == nullptr || wrap->db == nullptr || layer == nullptr ||
        out_result == nullptr) {
        return false;
    }
    if (wrap->converged.find(layer->name) == wrap->converged.end()) {
        return false;
    }
    // Look up by model_hash across all runs of this model (the resume set
    // was built cross-run, so the matching ga_result row usually lives
    // under a different run_id than this one).
    ts_quantize_db_ga_result gr;
    if (!ts_quantize_db_load_ga_result_for_model(wrap->db, wrap->model_hash,
                                                 layer->name, &gr)) {
        return false;
    }
    out_result->best.genes.alpha = gr.best_alpha;
    out_result->best.genes.clip  = gr.best_clip;
    out_result->best.expert_hint = -1;
    out_result->best_score.composite     = gr.best_composite;
    out_result->best_score.mse           = gr.best_mse;
    out_result->best_score.relative_frob = gr.best_mse;
    out_result->best_score.heldout_mse   = gr.best_mse;
    out_result->generations_run = 0;
    out_result->evaluations     = 0;
    out_result->converged       = true;
    out_result->warm_started    = true;   // resume == strongest warm-start
    return true;
}

// Run the L5 adaptive requantize loop over the captured 2D tensors. Mutates
// the deque entries in place and repoints the GGUF descriptors. Emits one
// JSON object per generation to l5_report_json on the result struct.
//
// Returns 0 on success (including the case where nothing was flagged and the
// loop terminates immediately), non-zero only on a hard setup error.
//
// Declared non-static so the integration test in test_l5_dispatch.cpp can
// drive it directly with a constructed refine_map (the test fixture builds
// the in-memory state without going through the full GGUF walk).
int ts_dispatch_run_l5_loop(
    const ts_dispatch_params * params,
    ts_dispatch_result * result,
    struct gguf_context * in_ctx,
    struct ggml_context * ggml_ctx,
    struct ggml_context * out_ggml_ctx,
    std::unordered_map<std::string, ts_dispatch_refine_entry> & refine_map) {

    if (refine_map.empty()) {
        return 0;
    }

    result->l5_ran = true;

    // L5 params (alpha/clip are floors for the multipliers; both clamped below).
    ts_l5_adaptive_params l5p;
    ts_l5_adaptive_default_params(&l5p);
    l5p.alpha_scale = 0.5f;  // base multiplier: tighten by 2x at overshoot 1
    l5p.clip_scale  = 0.5f;
    l5p.min_alpha   = params->l5_alpha_min;
    l5p.min_clip    = params->l5_clip_min;

    // Build the per-tensor L2 input view once; the loop refreshes the `quant`
    // pointer after each re-quantize from the deque's recon.
    const float flag_multiplier = params->l5_flag_multiplier;
    const int   max_generations = params->l5_max_generations;
    const float outlier_overshoot_scale = params->l5_outlier_overshoot_scale;
    const float outlier_frac_cap = params->l5_outlier_frac_cap;

    std::string report_json = "{\"schema\":\"llama.tessera.l5-loop.v1\",\"generations\":[";
    bool first_gen = true;

    // Snapshot the tensor list so we iterate in a stable order.
    std::vector<std::string> names;
    names.reserve(refine_map.size());
    for (auto & kv : refine_map) {
        names.push_back(kv.first);
    }
    std::sort(names.begin(), names.end());

    for (int gen = 0; gen < max_generations; gen++) {
        // (a) L2 measure: pair each tensor's re-read source with its recon.
        std::vector<ts_l2_tensor_input> inputs;
        inputs.reserve(names.size());
        std::vector<std::vector<float>> src_keepalive(names.size());
        for (size_t i = 0; i < names.size(); i++) {
            auto it = refine_map.find(names[i]);
            if (it == refine_map.end()) continue;
            ts_dispatch_refine_entry & e = it->second;
            src_keepalive[i] = ts_refine_reread_source(in_ctx, ggml_ctx, e);
            if (src_keepalive[i].empty()) continue;
            ts_l2_tensor_input tin = {};
            tin.name  = e.name.c_str();
            tin.qtype = "tessera_t640";
            tin.rows  = e.out_dim;
            tin.cols  = e.in_dim;
            tin.bf16  = src_keepalive[i].data();
            tin.quant = e.qr->recon.data();
            inputs.push_back(tin);
        }
        if (inputs.empty()) break;

        ts_l2_report l2rep = {};
        ts_l2_config l2cfg;
        ts_l2_default_config(&l2cfg);
        snprintf(l2cfg.bf16_model_path, sizeof(l2cfg.bf16_model_path), "%s", params->input_path.c_str());
        snprintf(l2cfg.corpus_path,     sizeof(l2cfg.corpus_path),     "%s", params->calib_corpus.c_str());
        l2cfg.output_json_path[0] = '\0';  // in-memory only; we emit our own report
        l2cfg.flag_multiplier = flag_multiplier;
        if (ts_l2_run(&l2cfg, inputs.data(), (int64_t)inputs.size(), &l2rep) < 0) {
            break;
        }

        // (b) L5 plan
        ts_l5_adaptive_plan plan;
        if (ts_l5_adaptive_requant(&l2rep, &l5p, gen, &plan) < 0) {
            break;
        }
        if (plan.n_requant <= 0 || plan.specs.empty()) {
            // nothing flagged this generation; loop has converged
            if (!first_gen) report_json += ",";
            first_gen = false;
            report_json += "{\"generation\":" + std::to_string(gen) +
                           ",\"n_flagged\":0,\"n_requant\":0,\"converged\":true}";
            break;
        }

        // (c) A/B per family + re-quantize flagged tensors.
        // Group specs by tensor family, then for each family with >=1 flagged
        // tensor run Stage A (alpha/clip multiplier) and Stage B (outlier_fraction
        // bump) on one representative tensor, pick the winner by rel_frob, and
        // apply that strategy to every flagged tensor in the family.
        std::map<std::string, std::vector<const ts_l5_requant_spec *>> by_family;
        for (const auto & spec : plan.specs) {
            const std::string fam = ts_regime_infer_family(spec.tensor_name.c_str());
            by_family[fam].push_back(&spec);
        }

        // Track per-tensor before/after for the report.
        std::vector<std::tuple<std::string, std::string, float, float>> deltas;
        std::vector<std::string> family_winners;

        for (const auto & fam_kv : by_family) {
            const std::string & fam = fam_kv.first;
            const auto & specs = fam_kv.second;
            if (specs.empty()) continue;

            // Representative: the first spec's tensor. Re-read its source once.
            auto rep_it = refine_map.find(specs[0]->tensor_name);
            if (rep_it == refine_map.end()) continue;
            ts_dispatch_refine_entry & rep = rep_it->second;
            std::vector<float> rep_src = ts_refine_reread_source(in_ctx, ggml_ctx, rep);
            if (rep_src.empty()) continue;
            const int64_t rep_n = rep.out_dim * rep.in_dim;

            // --- Stage A: tighten alpha/clip as multipliers on current values.
            //     alpha is the AWQ exponent in [0,1]; clip is the outlier clip
            //     in [0.7,1.0]. new_alpha/new_clip from L5 are multipliers in
            //     [min_alpha, 0.5]. If alpha_current is 0 (AWQ off), Stage A
            //     cannot tighten alpha, so it tightens clip only.
            ts_quant_params_2d tqp_A = rep.tqp;
            const float a_mult = std::clamp(specs[0]->new_alpha, 0.0f, 1.0f);
            const float c_mult = std::clamp(specs[0]->new_clip,  0.0f, 1.0f);
            if (tqp_A.alpha > 0.0f) {
                tqp_A.alpha = std::clamp(tqp_A.alpha * a_mult, 0.0f, 1.0f);
            }
            tqp_A.clip = std::clamp(tqp_A.clip * c_mult, 0.0f, 1.0f);

            // --- Stage B: bump outlier_thresh by overshoot instead.
            //     outlier_thresh on tqp is the selection threshold; raising it
            //     selects more rows for the exact residual, lowering recon error.
            ts_quant_params_2d tqp_B = rep.tqp;
            const float bump = 1.0f + outlier_overshoot_scale * specs[0]->overshoot;
            tqp_B.outlier_thresh = std::clamp(tqp_B.outlier_thresh * bump,
                                              0.0f, outlier_frac_cap);

            // Evaluate both on the representative. We re-quantize into scratch
            // results (NOT the deque) so the comparison is non-destructive; the
            // winning strategy is then applied to each flagged tensor's deque
            // element in place.
            const float * act = rep.act_scales_copy.empty()
                                    ? nullptr : rep.act_scales_copy.data();
            // Streaming MSE for the A/B comparison (only needs the scalar,
            // not the full packed output). ~132 KB scratch per call.
            float mse_A = ts_quantize_mse_streaming(
                rep_src.data(), act, tqp_A.alpha, tqp_A.clip,
                rep.out_dim, rep.in_dim);
            float mse_B = ts_quantize_mse_streaming(
                rep_src.data(), act, tqp_B.alpha, tqp_B.clip,
                rep.out_dim, rep.in_dim);
            float rep_frob2 = ts_vec_dotpr(rep_src.data(), rep_src.data(), rep_n);
            float frob_A = (mse_A >= 0.0f && rep_frob2 > 0.0f)
                               ? (mse_A * (float)rep_n / rep_frob2) : 1e30f;
            float frob_B = (mse_B >= 0.0f && rep_frob2 > 0.0f)
                               ? (mse_B * (float)rep_n / rep_frob2) : 1e30f;

            bool stage_b_wins = (frob_B < frob_A);
            family_winners.push_back("{\"family\":\"" + fam + "\",\"stage\":\"" +
                                     (stage_b_wins ? "B" : "A") +
                                     "\",\"frob_A\":" + std::to_string(frob_A) +
                                     ",\"frob_B\":" + std::to_string(frob_B) + "}");

            // Apply the winning strategy to every flagged tensor in this family.
            for (const auto * spec : specs) {
                auto it = refine_map.find(spec->tensor_name);
                if (it == refine_map.end()) continue;
                ts_dispatch_refine_entry & e = it->second;

                std::vector<float> src = ts_refine_reread_source(in_ctx, ggml_ctx, e);
                if (src.empty()) continue;
                const int64_t n = e.out_dim * e.in_dim;

                ts_quant_params_2d tightened = e.tqp;
                if (stage_b_wins) {
                    tightened.outlier_thresh = std::clamp(
                        tightened.outlier_thresh * (1.0f + outlier_overshoot_scale * spec->overshoot),
                        0.0f, outlier_frac_cap);
                } else {
                    const float am = std::clamp(spec->new_alpha, 0.0f, 1.0f);
                    const float cm = std::clamp(spec->new_clip,  0.0f, 1.0f);
                    if (tightened.alpha > 0.0f) {
                        tightened.alpha = std::clamp(tightened.alpha * am, 0.0f, 1.0f);
                    }
                    tightened.clip = std::clamp(tightened.clip * cm, 0.0f, 1.0f);
                }

                const float before = ts_refine_rel_frob(src.data(), e.qr, n);
                const float * e_act = e.act_scales_copy.empty()
                                          ? nullptr : e.act_scales_copy.data();
                int rc = ts_quantize_2d(src.data(), e_act, nullptr, nullptr, e_act,
                                        e.out_dim, e.in_dim, 0, &tightened, e.qr);
                if (rc != 0) {
                    continue;
                }
                // Refresh the GGUF descriptors: in-place re-quant may have
                // reallocated the deque element's buffers.
                ts_gguf_repoint_tensor_cluster(out_ggml_ctx, e.name.c_str(), e.qr);
                const float after = ts_refine_rel_frob(src.data(), e.qr, n);

                // Refresh result->tensors copies so downstream consumers see
                // the refined bytes and applied profile.
                for (auto & tr : result->tensors) {
                    if (tr.name == e.name) {
                        tr.packed              = ts_to_bytes_u32(e.qr->packed);
                        tr.page_scales         = ts_to_bytes_u16(e.qr->page_scales);
                        tr.lane_scales         = ts_to_bytes_i8(e.qr->lane_scales);
                        tr.outlier_row_offsets = ts_to_bytes_i32(e.qr->outlier_row_offsets);
                        tr.outlier_cols        = ts_to_bytes_i32(e.qr->outlier_cols);
                        tr.outlier_vals        = ts_to_bytes_u16(e.qr->outlier_vals);
                        tr.act_scale           = ts_to_bytes_u16(e.qr->act_scale);
                        tr.mse                 = e.qr->mse;
                        tr.alpha_used          = e.qr->best_alpha;
                        tr.profile_alpha       = tightened.alpha;
                        tr.profile_clip        = tightened.clip;
                        tr.profile_outlier_thresh = tightened.outlier_thresh;
                        break;
                    }
                }

                deltas.push_back(std::make_tuple(e.name, fam, before, after));
            }
        }

        // Generation report entry
        if (!first_gen) report_json += ",";
        first_gen = false;
        report_json += "{\"generation\":" + std::to_string(gen) +
                       ",\"n_flagged\":" + std::to_string((int64_t)l2rep.n_flagged) +
                       ",\"n_requant\":" + std::to_string(plan.n_requant) +
                       ",\"families\":[" + ts_join(family_winners, ",") + "]" +
                       ",\"tensors\":[";
        bool first_d = true;
        for (const auto & d : deltas) {
            if (!first_d) report_json += ",";
            first_d = false;
            report_json += "{\"name\":\"" + std::get<0>(d) +
                           "\",\"family\":\"" + std::get<1>(d) +
                           "\",\"before\":" + std::to_string(std::get<2>(d)) +
                           ",\"after\":" + std::to_string(std::get<3>(d)) + "}";
        }
        report_json += "]}";
    }

    report_json += "]}";
    result->l5_report_json = std::move(report_json);

    // Write the report beside policy_out_path unless the caller overrode it.
    if (!params->l5_out_path.empty()) {
        std::ofstream out(params->l5_out_path);
        if (out) {
            out << result->l5_report_json;
        }
    } else if (!params->policy_out_path.empty()) {
        std::string p = params->policy_out_path + ".l5-loop.json";
        std::ofstream out(p);
        if (out) {
            out << result->l5_report_json;
        }
    }

    if (params->verbose) {
        printf("tessera-dispatch: l5 loop wrote %zu bytes of report\n",
               result->l5_report_json.size());
    }

    return 0;
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

    // Persistent store. Opened once; closed via the unique_ptr deleter so
    // every return path (early or normal) marks the run done. nullptr when
    // --quantize-db is not set or open failed (the pipeline runs ephemeral).
    // db_run_status defaults to "failed" and is only flipped to "completed"
    // at the successful return, so any early-return path leaves the row
    // marked failed without needing to touch every return site.
    struct ts_dispatch_db_deleter {
        std::string * status;
        void operator()(ts_dispatch_db * w) const {
            ts_dispatch_db_close(w, status ? status->c_str() : "completed");
        }
    };
    std::string db_run_status = "failed";
    std::unique_ptr<ts_dispatch_db, ts_dispatch_db_deleter>
        db_guard(ts_dispatch_db_open(params, verbose),
                 ts_dispatch_db_deleter{&db_run_status});
    ts_dispatch_db * db_wrap = db_guard.get();

    // Structured progress reporter. Lives for the whole dispatch; each phase
    // resets the counters via ts_progress_set_phase. Terminal output auto-
    // enables on TTY stderr; NDJSON writes to progress_file when set. The
    // unique_ptr calls finish + destroy on scope exit so every early-return
    // path cleans up without per-site teardown. The TESSERA_PROGRESS_FILE
    // env var is a fallback so a shell can enable UI streaming without editing
    // launch commands.
    std::string progress_file = params->progress_file;
    if (progress_file.empty()) {
        const char * env = std::getenv("TESSERA_PROGRESS_FILE");
        if (env && env[0] != '\0') {
            progress_file = env;
        }
    }
    struct ts_progress_deleter {
        void operator()(ts_progress * p) const {
            ts_progress_finish(p);
            ts_progress_destroy(p);
        }
    };
    std::unique_ptr<ts_progress, ts_progress_deleter> prog_guard(
        ts_progress_create(
            ts_progress_phase::SETUP, 0,
            progress_file.empty() ? nullptr : progress_file.c_str(),
            params->progress_force_terminal || verbose));
    ts_progress * prog = prog_guard.get();

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
        } else if (ts_imatrix_load_gguf(params->imatrix_path.c_str(), &imatrix, &imsg) == 0) {
            // GGUF imatrix (emitted by llama-imatrix) - fall back when NPZ fails.
            have_imatrix = true;
            if (verbose) {
                printf("tessera-dispatch: calibration: loaded GGUF imatrix '%s' (%zu tensors)\n",
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
        db_run_status = "completed";
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
    // Early termination: stop a tensor's GA after 10 consecutive generations
    // with < 1e-5 improvement in the composite score. Most tensors converge
    // by gen 15-30; without this the GA wastes 70+% of its evaluations on
    // stagnant populations. Combined with family warm-start (the seed_candidate
    // field, populated by evolve_all from same-family converged tensors),
    // later tensors in a family converge in 1-5 generations.
    evolve_params.stagnation_limit   = 10;
    evolve_params.stagnation_epsilon = 1e-5f;
    // One-shot family hypothesis test: accept a family seed if this tensor's
    // eval scores within 95% of the seed's original composite. This skips the
    // GA entirely for tensors that share the family optimum (most of them in a
    // transformer), costing 1 eval instead of 640+.
    evolve_params.seed_accept_ratio = 0.95f;
    evolve_params.seed_composite    = 0.0f;  // populated per-layer by evolve_all
    // TESSERA_STAGNATION_LIMIT overrides the default early-termination window
    // (generations without improvement). Useful for tests that need to force
    // convergence on small fixtures, and for users who want to tune the GA's
    // exploration/exploitation trade-off.
    {
        const char * env = std::getenv("TESSERA_STAGNATION_LIMIT");
        if (env != nullptr && env[0] != '\0') {
            int parsed = std::atoi(env);
            if (parsed > 0) evolve_params.stagnation_limit = parsed;
        }
    }
    evolve_params.seed_candidate     = nullptr;  // set per-layer by evolve_all
    // Threading model: "serial layers, parallel candidates". One layer is
    // loaded at a time (180 MB weight buffer shared read-only); all threads
    // evaluate different candidates from the GA population in parallel. This
    // keeps peak memory at 1 weight buffer + n_threads scratch buffers
    // instead of n_threads x (weight + scratch), which is critical on memory-
    // constrained systems (16 GB M1 with an 8 GB model).
    //
    // TESSERA_QUANTIZE_THREADS controls both the layer-parallel thread count
    // (n_threads, used by evolve_all's work queue) and the candidate-parallel
    // thread count (n_eval_threads, used inside each ts_awq_evolve call).
    // On memory-constrained systems, set n_threads=1 so only one layer is
    // loaded at a time, and let n_eval_threads parallelize the candidates.
    {
        int32_t n_threads = 0;
        const char * env = std::getenv("TESSERA_QUANTIZE_THREADS");
        if (env != nullptr && env[0] != '\0') {
            int parsed = std::atoi(env);
            if (parsed > 0) {
                n_threads = parsed;
            } else if (parsed == 0) {
                n_threads = 1;  // explicit "0" disables threading
            }
        }
        if (n_threads == 0) {
            unsigned int hw = std::thread::hardware_concurrency();
            n_threads = hw > 0 ? (int32_t)hw : 1;
            if (n_threads > 8) {
                n_threads = 8;
            }
        }
        // On systems with limited memory, prefer serial layers + parallel
        // candidates. The TESSERA_QUANTIZE_LAYERS env var overrides: set to
        // >1 to enable layer-level parallelism (uses more memory).
        int32_t n_layer_threads = 1;  // serial by default (one layer at a time)
        const char * lenv = std::getenv("TESSERA_QUANTIZE_LAYERS");
        if (lenv != nullptr && lenv[0] != '\0') {
            int parsed = std::atoi(lenv);
            if (parsed > 0) n_layer_threads = parsed;
        }
        evolve_params.n_threads      = n_layer_threads;
        evolve_params.n_eval_threads = n_threads;
    }

    if (need_ga) {
        if (verbose) {
            printf("tessera-dispatch: GA configured (seed=%llu iters=%d islands=%d pop=%d layer_threads=%d eval_threads=%d)\n",
                   (unsigned long long)params->evolve_seed,
                   params->evolve_iters,
                   params->evolve_islands,
                   params->evolve_population,
                   evolve_params.n_threads,
                   evolve_params.n_eval_threads);
        }
    }

    if (params->evolve_only) {
        if (verbose) {
            printf("tessera-dispatch: evolve_only, returning early\n");
        }
        result->policy_json = "{}";
        db_run_status = "completed";
        return 0;
    }

    // --- step 4: resolve alpha ---
    if (params->awq_alpha != "auto" && !params->awq_alpha.empty()) {
        default_alpha = std::stof(params->awq_alpha);
    }

    // --- step 4b: read calibration policy (if any) ---
    // ts_policy_read consumes the canonical tensor_families shape emitted by
    // tools/tessera/awq-evolve.py and the legacy top-level tensors shape. When
    // present, the per-tensor loop below resolves each tensor name to a family
    // and overrides the AWQ alpha/clip and sparse-residual threshold from the
    // policy, eliminating the Python pre-step.
    ts_policy policy;
    bool have_policy = false;
    if (!params->policy_path.empty()) {
        std::string pmsg;
        if (ts_policy_read(params->policy_path.c_str(), &policy, &pmsg) != 0) {
            if (err_msg) {
                *err_msg = "failed to load policy '" + params->policy_path + "': " + pmsg;
            }
            return 1;
        }
        have_policy = true;
        if (verbose) {
            printf("tessera-dispatch: loaded policy '%s' (%zu families, search_schema='%s')\n",
                   params->policy_path.c_str(), policy.tensors.size(),
                   policy.search_schema.c_str());
        }
    }

    // --- step 5: load input GGUF ---
    // Use no_alloc=true so ggml creates tensor metadata without allocating
    // 8+ GB of resident memory for the weights. Then mmap the GGUF file and
    // patch each tensor's data pointer into the mmap'd region. macOS lazily
    // pages in only the tensor regions actually read; clean pages are evicted
    // under memory pressure. This drops the model's RSS contribution from the
    // full file size (~8 GB for q8_0) to ~one active tensor (~180 MB).
    struct ggml_context * ggml_ctx = nullptr;
    struct gguf_init_params gparams = {
        /*no_alloc =*/ true,
        /*ctx      =*/ &ggml_ctx,
    };

    struct gguf_context * in_ctx = gguf_init_from_file(params->input_path.c_str(), gparams);
    if (in_ctx == nullptr) {
        if (err_msg) {
            *err_msg = "failed to open input GGUF: " + params->input_path;
        }
        return 1;
    }

    // Mmap the GGUF file read-only and patch tensor data pointers.
    // The gguf data section starts at gguf_get_data_offset(in_ctx); each
    // tensor's data is at data_offset + gguf_get_tensor_offset(in_ctx, i).
    {
        int fd = open(params->input_path.c_str(), O_RDONLY);
        if (fd < 0) {
            if (err_msg) *err_msg = "open failed for mmap: " + params->input_path;
            gguf_free(in_ctx);
            ggml_free(ggml_ctx);
            return 1;
        }
        struct stat st;
        if (fstat(fd, &st) != 0) {
            close(fd);
            if (err_msg) *err_msg = "fstat failed for mmap";
            gguf_free(in_ctx);
            ggml_free(ggml_ctx);
            return 1;
        }
        size_t file_size = (size_t)st.st_size;
        void * mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) {
            close(fd);
            if (err_msg) *err_msg = "mmap failed for input GGUF";
            gguf_free(in_ctx);
            ggml_free(ggml_ctx);
            return 1;
        }
        // The fd can be closed after mmap; the mapping stays valid.
        close(fd);
        // Patch each tensor's data pointer into the mmap'd region.
        const size_t data_off = gguf_get_data_offset(in_ctx);
        const int64_t n_t = gguf_get_n_tensors(in_ctx);
        for (int64_t i = 0; i < n_t; i++) {
            const char * tname = gguf_get_tensor_name(in_ctx, i);
            size_t toff = gguf_get_tensor_offset(in_ctx, i);
            struct ggml_tensor * t = ggml_get_tensor(ggml_ctx, tname);
            if (t) {
                t->data = (char *)mapped + data_off + toff;
            }
        }
        // Hold the mapping for the lifetime of in_ctx/ggml_ctx via a
        // cleanup lambda stored in the result (or just leak it - the OS
        // reclaims on process exit, and the quantize tool is short-lived).
        // For correctness, store in a static so ts_refine_reread_source
        // and other readers that re-access tensors still work.
    }

    const int64_t n_tensors = gguf_get_n_tensors(in_ctx);
    if (verbose) {
        printf("tessera-dispatch: loaded '%s' (%lld tensors)\n",
               params->input_path.c_str(), (long long)n_tensors);
    }
    // Seed the GA phase total with n_tensors (upper bound; refined at the
    // evolve_all call site once ga_layers.size() is known).
    ts_progress_set_phase(prog, ts_progress_phase::GA_EVOLVE, n_tensors,
                          "per-tensor alpha search");

    // --- step 5b: HIGGS alpha_l estimation / cache lookup ---
    std::string higgs_mode = params->higgs_alpha_mode.empty() ? "uniform" : params->higgs_alpha_mode;
    std::vector<float> higgs_alphas;   // empty = uniform
    bool higgs_active = false;

    if (higgs_mode != "uniform") {
        // collect quantizable tensor weights for cache key
        std::vector<const float *> higgs_wptrs;
        std::vector<int64_t> higgs_outs, higgs_ins;
        std::vector<std::vector<float>> higgs_wbufs;

        ts_progress_set_phase(prog, ts_progress_phase::HIGGS, n_tensors,
                              "alpha_l estimation");
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
            ts_progress_inc(prog, 1, name);
        }
        ts_progress_inc(prog, n_tensors - (int64_t)higgs_wptrs.size(), nullptr);

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
        std::vector<std::vector<float>>    ga_wbufs;     // legacy: stays empty with streaming load
        std::vector<std::vector<float>>    ga_actbufs;   // corpus-derived act_scales storage
        // Per-layer weight loader state: holds the ggml tensor pointer (from
        // the mmap'd context) and the per-call f32 buffer. The GA worker
        // populates buf on load, clears it on release, so only ~8 layers'
        // f32 data is alive at once (one per worker thread).
        struct ts_ga_weight_loader {
            struct ggml_tensor * tensor;
            std::vector<float>   buf;
        };
        std::vector<ts_ga_weight_loader>   ga_weight_loaders;
        std::vector<ts_awq_layer>          ga_layers;
        std::vector<ts_regime_descriptor>  ga_descs;     // regime descriptor per layer (archive axes)

        ts_progress_set_phase(prog, ts_progress_phase::GA_PREP, n_tensors,
                              "collect GA layers");
        ga_weight_loaders.reserve((size_t)n_tensors);  // prevent realloc (raw ptrs stored)
        ga_layers.reserve((size_t)n_tensors);
        ga_names.reserve((size_t)n_tensors);
        ga_descs.reserve((size_t)n_tensors);
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
            const float * imdata_max_abs = nullptr;
            int64_t       imdim_max_abs  = 0;
            if (have_imatrix) {
                imdata = ts_imatrix_lookup(&imatrix, name, &imdim);
                // Per-channel max |activation| (.in_maxabs) - the localized
                // outlier signal that kurtosis (a global scalar) cannot see.
                imdata_max_abs = ts_imatrix_lookup_max_abs(&imatrix, name, &imdim_max_abs);
            }
            ts_regime_descriptor desc = ts_regime_compute_descriptor(
                name, w.data(), out_dim, in_dim, imdata, imdata ? imdim : 0,
                imdata_max_abs, imdata_max_abs ? imdim_max_abs : 0);

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
            // NOTE: weights are NOT stored in ga_wbufs. The f32 conversion is
            // needed only for the regime descriptor above; the GA loads weights
            // on demand via weights_load_fn to avoid holding all 254 layers'
            // f32 data in memory (~32 GB for an 8B model - fatal on 16 GB).
            // ga_wbufs is kept for API compatibility but stays empty here.

            ts_awq_layer layer;
            layer.name        = ga_names.back();
            layer.family      = desc.family;
            layer.weights     = nullptr;  // loaded on demand by the GA worker
            layer.act_scales  = act;
            layer.calib_X     = nullptr;
            layer.ref_output  = nullptr;
            layer.imatrix     = act;
            // B2 fitness: when the imatrix provides per-channel E[x^2] for
            // this tensor, expose it as second_moment so the GA evaluator can
            // run the Python parity fitness (ts_awq_evaluate_layer). The
            // fourth_moment / max_abs fields stay null and the evaluator
            // falls back to second^2 / sqrt(second), matching Python's
            // Layer.from_npz fallback when in_sum4 / in_maxabs are absent.
            // train/heldout activations are not yet wired here (no calib
            // split), so the evaluator uses the diagonal second-moment loss
            // for train_error and treats heldout == train, exactly as Python
            // does when train_activations is None.
            layer.second_moment       = (imdata && imdim == in_dim) ? imdata : nullptr;
            layer.fourth_moment       = nullptr;
            layer.max_abs             = nullptr;
            layer.train_activations   = nullptr;
            layer.heldout_activations = nullptr;
            layer.ref_train_output    = nullptr;
            layer.ref_heldout_output  = nullptr;
            layer.out_dim     = out_dim;
            layer.in_dim      = in_dim;
            layer.n_tokens    = 0;
            layer.n_tokens_h  = 0;
            layer.kurtosis    = desc.kurtosis;
            layer.eff_rank    = desc.eff_rank;
            // Streaming weight loader: each GA worker calls this to fetch the
            // layer's f32 weights on demand from the mmap'd ggml context.
            // The buffer is owned per-call and freed by weights_release_fn
            // after the layer's GA completes. This keeps peak memory at
            // ~8 concurrent layers (one per worker) instead of all 254.
            layer.weights_load_fn = [](void * ud) -> const float * {
                auto * ml = static_cast<struct ts_ga_weight_loader *>(ud);
                ml->buf = ts_tensor_to_f32(ml->tensor);
                return ml->buf.empty() ? nullptr : ml->buf.data();
            };
            layer.weights_release_fn = [](void * ud, const float *) {
                auto * ml = static_cast<struct ts_ga_weight_loader *>(ud);
                ml->buf.clear();
                ml->buf.shrink_to_fit();
            };
            // NOTE: ga_weight_loaders MUST be reserved before the loop so the
            // vector never reallocates. layer.weights_user_data stores a raw
            // pointer to ga_weight_loaders.back(); a reallocation would dangle
            // every prior pointer and crash the GA workers (SIGSEGV in
            // ggml_nelements via the loader callback).
            ga_weight_loaders.push_back({ggml_get_tensor(ggml_ctx, name), {}});
            layer.weights_user_data = &ga_weight_loaders.back();
            layer.layer_depth = ts_quantize_db_layer_depth(name);
            ga_layers.push_back(layer);
            ga_descs.push_back(desc);
            // Record the tensor in the persistent registry. One row per
            // quantizable weight, with the regime descriptors (kurtosis /
            // eff_rank) computed above. Best-effort: a failure here only
            // loses metadata, never blocks quantization.
            if (db_wrap != nullptr) {
                ts_quantize_db_tensor trec;
                trec.run_id      = db_wrap->run_id;
                trec.name        = name;
                trec.family      = desc.family;
                trec.layer_depth = layer.layer_depth;
                trec.out_dim     = out_dim;
                trec.in_dim      = in_dim;
                trec.n_elements  = out_dim * in_dim;
                trec.kurtosis    = desc.kurtosis;
                trec.eff_rank    = desc.eff_rank;
                trec.source_type = ggml_type_name(type);
                std::string terr;
                if (ts_quantize_db_insert_tensor(db_wrap->db, trec, &terr) != 0
                    && !terr.empty()) {
                    fprintf(stderr, "tessera-dispatch: warning: "
                                    "insert_tensor('%s') failed: %s\n",
                            name, terr.c_str());
                }
            }
            ts_progress_inc(prog, 1, name);
        }
        // Charge any filtered iterations so the bar completes.
        ts_progress_inc(prog, n_tensors - (int64_t)ga_layers.size(), nullptr);

        if (!ga_layers.empty()) {
            ts_dispatch_eval_ctx eval_ctx;
            eval_ctx.outlier_thresh = params->outlier_frac;
            eval_ctx.seed           = (uint32_t)params->evolve_seed;
            eval_ctx.verbose        = verbose;
            eval_ctx.mode_prints.store(0, std::memory_order_relaxed);
            eval_ctx.modality       = &mm_modality;

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
            // Refine the GA progress total now that the quantizable 2D layer
            // count is known (smaller than n_tensors: excludes norms, embeds,
            // 1D/3D tensors). evolve_all will call on_phase_change when it
            // enters screening vs main evolution so the UI can distinguish
            // them; on_layer_done fires per completed layer in each phase.
            evolve_params.on_layer_done = [](int64_t, int64_t,
                                             const char * name, void * user) {
                ts_progress * p = static_cast<ts_progress *>(user);
                ts_progress_inc(p, 1, name);
            };
            evolve_params.on_layer_done_user = prog;
            evolve_params.on_phase_change = [](const char * phase,
                                               int64_t n_layers,
                                               void * user) {
                ts_progress * p = static_cast<ts_progress *>(user);
                ts_progress_set_phase(p, ts_progress_phase::GA_SCREEN,
                                      n_layers, phase);
                // The main evolution is reported as GA_EVOLVE so the UI shows
                // the right label; screen stays GA_SCREEN.
                if (std::string(phase) == "evolve") {
                    ts_progress_set_phase(p, ts_progress_phase::GA_EVOLVE,
                                          n_layers, "per-tensor alpha search");
                }
            };
            evolve_params.on_phase_change_user = prog;
            // DB hooks: per-evaluation appender, family warm-start lookup,
            // and resume skip. All three no-op when db_wrap is null.
            if (db_wrap != nullptr) {
                evolve_params.eval_recorder           = ts_dispatch_eval_recorder;
                evolve_params.eval_recorder_user      = db_wrap;
                evolve_params.family_seed_lookup      = ts_dispatch_family_seed_lookup;
                evolve_params.family_seed_lookup_user = db_wrap;
                evolve_params.layer_skip_lookup       = ts_dispatch_layer_skip;
                evolve_params.layer_skip_lookup_user  = db_wrap;
            }
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
                ga_alpha[ga_names[l]] = ga_results[l].best.genes.alpha;
                t2[l]                 = ga_results[l].best_score.relative_frob;
            }

            // Persist GA results + flush per-tensor eval appenders. Each
            // layer's appender was opened lazily inside eval_recorder; close
            // it now so the ga_evaluations rows for this tensor are visible
            // to subsequent queries (warm-start, resume, the duckdb CLI).
            if (db_wrap != nullptr) {
                for (size_t l = 0; l < ga_results.size(); l++) {
                    // Flush + close this layer's appender (no-op if none).
                    db_wrap->appenders.with_lock(ga_names[l],
                        [&](std::unordered_map<std::string,
                                                ts_quantize_db_appender *> & m,
                           const std::string & key)
                    {
                        auto it = m.find(key);
                        if (it != m.end()) {
                            ts_quantize_db_appender_flush(it->second);
                            ts_quantize_db_appender_close(it->second);
                            m.erase(it);
                        }
                    });
                    ts_quantize_db_ga_result gr;
                    gr.run_id          = db_wrap->run_id;
                    gr.tensor_name     = ga_names[l];
                    gr.family          = ga_layers[l].family;
                    gr.best_alpha      = ga_results[l].best.genes.alpha;
                    gr.best_clip       = ga_results[l].best.genes.clip;
                    gr.best_composite  = ga_results[l].best_score.composite;
                    gr.best_mse        = ga_results[l].best_score.mse;
                    gr.generations_run = (int32_t)ga_results[l].generations_run;
                    gr.n_evaluations   = ga_results[l].evaluations;
                    gr.converged       = ga_results[l].converged;
                    gr.warm_started    = ga_results[l].warm_started;
                    std::string gerr;
                    if (ts_quantize_db_insert_ga_result(db_wrap->db, gr, &gerr) != 0
                        && !gerr.empty()) {
                        fprintf(stderr, "tessera-dispatch: warning: "
                                        "insert_ga_result('%s') failed: %s\n",
                                ga_names[l].c_str(), gerr.c_str());
                    }
                }
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
                                  ga_results[l].best.genes.alpha, ga_results[l].best.genes.clip,
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
                    std::pair<float, float> pair_val;
                    if (!eval_ctx.best_pair.find_copy(ga_names[l], pair_val)) {
                        continue;
                    }
                    ts_ab_tensor_scores s;
                    s.name              = ga_names[l];
                    s.offline_proxy_mse = pair_val.first;
                    s.kernel_direct_t2  = pair_val.second;
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

    // L5 adaptive requantize: per-tensor metadata captured during step 7 so the
    // refine loop can target 2D tensors by name without re-walking the GGUF.
    // Only 2D tensors are eligible; 3D MoE re-quantize is deferred.
    std::unordered_map<std::string, ts_dispatch_refine_entry> refine_map;

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

    // The quantize-write loop is the second long phase. Bump progress per
    // tensor written (quantized or copied through).
    ts_progress_set_phase(prog, ts_progress_phase::QUANTIZE, n_tensors,
                          "write quantized tensors");

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
            ts_progress_inc(prog, 1, name);
            continue;
        }

        if (!ts_is_quantizable(name, type, n_dims)) {
            // copy through unchanged
            gguf_add_tensor(out_ctx, t);
            n_skipped++;
            if (verbose) {
                printf("tessera-dispatch: copy-through %s (%s)\n", name, ggml_type_name(type));
            }
            ts_progress_inc(prog, 1, name);
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
            ts_progress_inc(prog, 1, name);
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

        // Apply the calibration policy when one was loaded. The matched
        // family's AWQ + Tessera genes override the routed profile so the
        // runtime consumes Python's output natively (no Python pre-step).
        // moment_mix / tail_guard / ternary_threshold are surfaced via the
        // result for the candidate evaluator; the fields that map onto
        // ts_quant_params_2d (alpha/clip/outlier_thresh) are applied here.
        const ts_policy_tensor * pfam = have_policy
            ? ts_policy_select(policy.tensors, name) : nullptr;
        if (pfam != nullptr) {
            tqp.alpha         = pfam->genes.alpha;
            tqp.clip          = pfam->genes.clip;
            // dispatch already treats outlier_frac as the selection threshold
            // (params->outlier_frac -> qparams.outlier_thresh above), so the
            // policy's outlier_fraction gene maps onto the same knob here.
            tqp.outlier_thresh = pfam->genes.outlier_fraction;
            if (verbose) {
                printf("tessera-dispatch: %s matched policy family '%s' "
                       "(exact=%d alpha=%.3f clip=%.3f frac=%.4f thresh=%.3f mix=%.3f tail=%.3f)\n",
                       name, pfam->family.c_str(), (int)pfam->exact,
                       pfam->genes.alpha, pfam->genes.clip, pfam->genes.outlier_fraction,
                       pfam->genes.ternary_threshold, pfam->genes.moment_mix,
                       pfam->genes.tail_guard);
            }
        }

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
            ts_progress_inc(prog, 1, name);

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
            ts_progress_inc(prog, 1, name);

            // Capture for the L5 refine loop (gated by params->adaptive_requantize;
            // cheap when the loop is off, since it only fires then).
            if (params->adaptive_requantize) {
                ts_dispatch_refine_entry entry;
                entry.name     = name;
                entry.family   = family;
                entry.gguf_idx = i;
                entry.out_dim  = out_dim;
                entry.in_dim   = in_dim;
                entry.qr       = &qr;  // stable: cluster_results deque element
                entry.tqp      = tqp;
                if (act_scales != nullptr) {
                    entry.act_scales_copy.assign(act_scales, act_scales + in_dim);
                }
                refine_map.emplace(name, std::move(entry));
            }

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

    // --- step 7a: L5 adaptive requantize loop ---
    // Runs when --tessera-adaptive-requantize is set. Re-quantizes flagged
    // tensors in place (refreshing the GGUF descriptors via
    // ts_gguf_repoint_tensor_cluster) and emits an l5-loop.json report.
    if (params->adaptive_requantize) {
        ts_dispatch_run_l5_loop(params, result, in_ctx, ggml_ctx, out_ggml_ctx, refine_map);
    }

    // --- step 7b: G6 acceptance gate ---
    result->acceptance_ran = false;
    if (params->run_acceptance) {
        // Two phases:
        //   (a) serial: load each 2D quantizable tensor's weights + act scales
        //       (touches the ggml ctx, must be single-threaded).
        //   (b) parallel: the 6 ts_dispatch_forced_t2 calls per tensor are
        //       CPU-bound and independent across tensors. Fan out across the
        //       same thread budget as the GA.
        // Acceptance gate work items: store only metadata + the ggml tensor
        // pointer (not the f32 weights). Each worker loads the tensor on
        // demand from the mmap'd GGUF, evaluates 6 experts, then frees.
        // Peak: n_threads x 180 MB instead of 254 x 180 MB = 45 GB.
        struct acc_work_item {
            std::string name;
            struct ggml_tensor * tensor;  // from mmap'd ggml_ctx; data valid
            std::vector<float> act_scratch;
            const float * act;
            int64_t out_dim, in_dim;
            ts_expert_id routed_expert;
        };
        std::vector<acc_work_item> work;
        work.reserve((size_t)n_tensors);

        const float alpha = default_alpha;
        const float clip  = params->awq_clip;
        const float othresh = params->outlier_frac;
        const uint32_t seed = (uint32_t)params->evolve_seed;

        ts_progress_set_phase(prog, "accept-prep", n_tensors,
                              "collect tensors for acceptance gate");
        for (int64_t i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(in_ctx, i);
            const enum ggml_type type = gguf_get_tensor_type(in_ctx, i);
            const int64_t * ne = gguf_get_tensor_ne(in_ctx, i);
            int nd = GGML_MAX_DIMS;
            while (nd > 1 && ne[nd - 1] == 1) nd--;

            if (nd != 2 || !ts_is_quantizable(name, type, nd)) {
                ts_progress_inc(prog, 1, name);
                continue;
            }

            struct ggml_tensor * t = ggml_get_tensor(ggml_ctx, name);
            if (!t) {
                ts_progress_inc(prog, 1, name);
                continue;
            }

            acc_work_item item;
            item.name     = name;
            item.tensor   = t;
            item.out_dim  = ne[1];
            item.in_dim   = ne[0];
            item.act = ts_dispatch_act_scales(
                have_imatrix ? &imatrix : nullptr, name, item.in_dim,
                calib_X.empty() ? nullptr : calib_X.data(), calib_in_dim, calib_n_tokens,
                &item.act_scratch);

            // Regime descriptor needs the weights for kurtosis/eff_rank.
            // Load temporarily, compute, free.
            {
                std::vector<float> w = ts_tensor_to_f32(t);
                if (w.empty()) {
                    ts_progress_inc(prog, 1, name);
                    continue;
                }
                ts_regime_descriptor desc = ts_regime_compute_descriptor(
                    name, w.data(), item.out_dim, item.in_dim, nullptr, 0);
                ts_regime_routing routing = ts_regime_classify(&desc);
                item.routed_expert = routing.expert;
            }

            work.push_back(std::move(item));
            ts_progress_inc(prog, 1, name);
        }

        std::vector<ts_acceptance_tensor> acc_tensors(work.size());

        // Parallel 6x re-quantize per tensor. Each worker loads the tensor
        // from the mmap'd GGUF on demand (streaming), evaluates 6 experts
        // via ts_quantize_mse_streaming (132 KB scratch each), then the
        // loaded weights go out of scope. Peak: 4 x 180 MB + 4 x 132 KB.
        ts_progress_set_phase(prog, "accept-run", (int64_t)work.size(),
                              "acceptance gate (6 experts per tensor, streaming)");

        const int32_t acc_threads = std::max(1, std::min(
            (int32_t) std::thread::hardware_concurrency(),
            std::min((int32_t)8, (int32_t)work.size())));

        if (acc_threads <= 1 || work.size() < 2) {
            for (size_t idx = 0; idx < work.size(); idx++) {
                const auto & item = work[idx];
                std::vector<float> w = ts_tensor_to_f32(item.tensor);
                if (w.empty()) continue;
                ts_acceptance_tensor at;
                memset(&at, 0, sizeof(at));
                snprintf(at.name, sizeof(at.name), "%s", item.name.c_str());
                at.composite_t2 = ts_dispatch_forced_t2(w.data(), item.act, item.out_dim, item.in_dim, item.routed_expert, alpha, clip, othresh, seed);
                at.awq_t2       = ts_dispatch_forced_t2(w.data(), item.act, item.out_dim, item.in_dim, TS_EXPERT_AWQ,       alpha, clip, othresh, seed);
                at.rotation_t2  = ts_dispatch_forced_t2(w.data(), item.act, item.out_dim, item.in_dim, TS_EXPERT_DARTQUANT, alpha, clip, othresh, seed);
                at.lowrank_t2   = ts_dispatch_forced_t2(w.data(), item.act, item.out_dim, item.in_dim, TS_EXPERT_FLRQ,      alpha, clip, othresh, seed);
                at.hessian_t2   = ts_dispatch_forced_t2(w.data(), item.act, item.out_dim, item.in_dim, TS_EXPERT_SEPTQ,     alpha, clip, othresh, seed);
                at.offline_proxy_mse = at.composite_t2;
                at.kernel_direct_t2  = at.composite_t2;
                at.held_out          = false;
                acc_tensors[idx] = at;
                ts_progress_inc(prog, 1, item.name.c_str());
            }
        } else {
            std::atomic<size_t> next_idx(0);
            auto acc_worker = [&]() {
                for (;;) {
                    size_t idx = next_idx.fetch_add(1, std::memory_order_relaxed);
                    if (idx >= work.size()) return;
                    const auto & item = work[idx];
                    // Load weights from mmap'd GGUF on demand.
                    std::vector<float> w = ts_tensor_to_f32(item.tensor);
                    if (w.empty()) continue;
                    ts_acceptance_tensor at;
                    memset(&at, 0, sizeof(at));
                    snprintf(at.name, sizeof(at.name), "%s", item.name.c_str());
                    at.composite_t2 = ts_dispatch_forced_t2(w.data(), item.act, item.out_dim, item.in_dim, item.routed_expert, alpha, clip, othresh, seed);
                    at.awq_t2       = ts_dispatch_forced_t2(w.data(), item.act, item.out_dim, item.in_dim, TS_EXPERT_AWQ,       alpha, clip, othresh, seed);
                    at.rotation_t2  = ts_dispatch_forced_t2(w.data(), item.act, item.out_dim, item.in_dim, TS_EXPERT_DARTQUANT, alpha, clip, othresh, seed);
                    at.lowrank_t2   = ts_dispatch_forced_t2(w.data(), item.act, item.out_dim, item.in_dim, TS_EXPERT_FLRQ,      alpha, clip, othresh, seed);
                    at.hessian_t2   = ts_dispatch_forced_t2(w.data(), item.act, item.out_dim, item.in_dim, TS_EXPERT_SEPTQ,     alpha, clip, othresh, seed);
                    at.offline_proxy_mse = at.composite_t2;
                    at.kernel_direct_t2  = at.composite_t2;
                    at.held_out          = false;
                    acc_tensors[idx] = at;
                    ts_progress_inc(prog, 1, item.name.c_str());
                }
            };
            std::vector<std::thread> pool;
            pool.reserve((size_t)acc_threads);
            for (int32_t t = 0; t < acc_threads; t++) pool.emplace_back(acc_worker);
            for (auto & th : pool) th.join();
        }
        // work is destroyed here; acc_tensors owns the results.

        if (!acc_tensors.empty()) {
            ts_acceptance_run(&params->acceptance_config,
                              acc_tensors.data(), (int64_t)acc_tensors.size(),
                              &result->acceptance);
            result->acceptance_ran = true;
            if (verbose) {
                printf("tessera-dispatch: acceptance: %s\n", result->acceptance.verdict);
            }
            // Persist per-tensor acceptance rows. The verdict is the same
            // string the gate emits at the run level; per-tensor verdict
            // would need an extra ts_acceptance API, so we record the t2
            // breakdown (the part useful for offline analysis).
            if (db_wrap != nullptr) {
                for (const auto & at : acc_tensors) {
                    ts_quantize_db_acceptance arec;
                    arec.run_id       = db_wrap->run_id;
                    arec.tensor_name  = at.name;
                    arec.family       = ts_regime_infer_family(at.name);
                    arec.composite_t2 = at.composite_t2;
                    arec.awq_t2       = at.awq_t2;
                    arec.rotation_t2  = at.rotation_t2;
                    arec.lowrank_t2   = at.lowrank_t2;
                    arec.hessian_t2   = at.hessian_t2;
                    arec.verdict      = result->acceptance.composite_wins
                                            ? "pass" : "fail";
                    std::string aerr;
                    if (ts_quantize_db_insert_acceptance(db_wrap->db, arec, &aerr) != 0
                        && !aerr.empty()) {
                        fprintf(stderr, "tessera-dispatch: warning: "
                                        "insert_acceptance('%s') failed: %s\n",
                                at.name, aerr.c_str());
                    }
                }
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

    // --- finalize progress reporting ---
    ts_progress_set_phase(prog, ts_progress_phase::FINALIZE, 0, "write GGUF");

    // --- cleanup ---
    ggml_free(out_ggml_ctx);
    gguf_free(out_ctx);
    gguf_free(in_ctx);
    ggml_free(ggml_ctx);

    // Successful return: the db_guard deleter marks this run "completed".
    // Any early-return path above leaves db_run_status at "failed".
    db_run_status = "completed";

    // prog_guard's deleter runs here: ts_progress_finish + ts_progress_destroy.
    return 0;
}
