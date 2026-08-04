#include "tessera-regime.h"

#include <cmath>
#include <cstring>
#include <algorithm>

// --- family inference ---

struct ts_family_pattern {
    const char * fragment;
    const char * family;
};

// ordered by specificity: longer fragments first to avoid prefix collisions
static const ts_family_pattern ts_family_patterns[] = {
    { "attn_output", "attn_out"  },
    { "attn_out",    "attn_out"  },
    { "attn_q",      "attn_q"    },
    { "attn_k",      "attn_k"    },
    { "attn_v",      "attn_v"    },
    { "ffn_gate",    "ffn_gate"  },
    { "ffn_up",      "ffn_up"    },
    { "ffn_down",    "ffn_down"  },
};

std::string ts_regime_infer_family(const char * tensor_name) {
    if (!tensor_name) {
        return "unknown";
    }
    for (const auto & p : ts_family_patterns) {
        if (strstr(tensor_name, p.fragment)) {
            return p.family;
        }
    }
    return "unknown";
}

// --- modality inference ---

int ts_regime_infer_modality(const char * tensor_name) {
    if (!tensor_name) {
        return 0;
    }
    // Empty string has no role prefix and matches no fragment -> text.
    if (tensor_name[0] == '\0') {
        return 0;
    }

    // ---- First pass: explicit role prefixes (M0b) ----
    // Real mmproj GGUFs (clip.cpp:1831) use "v." for vision tower, "a." for
    // audio tower, and "mm." for the text-side projector. The old fragment
    // matcher below misses most of these (e.g. "v.blk.0.attn_q.weight" has
    // no vision/image/vit/patch/pixel/img substring), so the FLRQ/LRQ
    // modality branches in ts_regime_classify were inert for real mmproj
    // tensors. The role prefix is authoritative for v./a.; mm.* tensors
    // still fall through to the fragment check below because real projector
    // names ("mm.up.weight", "mm.input_projection.weight", ...) never
    // contain vision/audio substrings, and hand-written test fixtures
    // (e.g. "mm.vision_embed.weight" in test_modality_routing.cpp) rely on
    // the fragment path.
    if (tensor_name[0] == 'v' && tensor_name[1] == '.') {
        return 1;
    }
    if (tensor_name[0] == 'a' && tensor_name[1] == '.') {
        return 2;
    }

    // ---- Second pass: legacy fragment-based detection ----
    // For older mmproj GGUFs and hand-written test fixtures that don't use
    // the v./a. role prefix. Precedence: role prefix first, fragment second.
    // image / vision embedder tensors
    static const char * image_fragments[] = {
        "vision", "image", "vit", "patch", "pixel", "img",
    };
    for (const char * f : image_fragments) {
        if (strstr(tensor_name, f)) {
            return 1;
        }
    }
    // audio / acoustic embedder tensors
    static const char * audio_fragments[] = {
        "audio", "acoustic", "speech", "wav",
    };
    for (const char * f : audio_fragments) {
        if (strstr(tensor_name, f)) {
            return 2;
        }
    }
    return 0;
}

// --- regime classification ---

static bool ts_family_contains(const std::string & family, const char * sub) {
    return family.find(sub) != std::string::npos;
}

ts_regime_routing ts_regime_classify(const ts_regime_descriptor * desc) {
    ts_regime_routing r;
    r.tensor_name = desc->tensor_name;

    const float kurt = desc->kurtosis;
    const float er   = desc->eff_rank;
    const std::string & fam = desc->family;
    const int modality = desc->modality;

    // modality-specific regimes. Text (modality 0) falls through to the
    // generic cascade below unchanged.
    if (modality == 2 && kurt > 5.0f) {
        // audio activations are heavy-tailed; a factored low-rank residual
        // handles the long tails better than rotation/permutation experts
        r.expert     = TS_EXPERT_FLRQ;
        r.reason     = "audio + kurtosis > 5: heavy-tailed acoustic activations, factored low-rank";
        r.confidence = 0.85f;
        return r;
    }
    if (modality == 1 && er < 0.3f) {
        // vision activations are spatially low-rank; an explicit low-rank
        // residual captures the structure
        r.expert     = TS_EXPERT_LRQ;
        r.reason     = "image + eff_rank < 0.3: spatially low-rank vision activations";
        r.confidence = 0.82f;
        return r;
    }

    // massive outliers in down_proj (DuQuant observation)
    if (kurt > 10.0f && ts_family_contains(fam, "down")) {
        r.expert     = TS_EXPERT_DARTQUANT;
        r.reason     = "kurtosis > 10 in down_proj: rotation handles massive outliers";
        r.confidence = 0.95f;
        return r;
    }

    // heavy tails: rotation or permutation
    if (kurt > 10.0f) {
        r.expert     = TS_EXPERT_DARTQUANT;
        r.reason     = "kurtosis > 10: distribution-aware rotation";
        r.confidence = 0.85f;
        return r;
    }
    if (kurt > 5.0f) {
        r.expert     = TS_EXPERT_CHAMPQ;
        r.reason     = "kurtosis > 5: channel permutation smooths heavy tails";
        r.confidence = 0.75f;
        return r;
    }

    // spectrally compact: low-rank residual helps
    if (er < 0.15f) {
        r.expert     = TS_EXPERT_FLRQ;
        r.reason     = "eff_rank < 0.15: highly compact spectrum, factored low-rank";
        r.confidence = 0.85f;
        return r;
    }
    if (er < 0.3f) {
        r.expert     = TS_EXPERT_LRQ;
        r.reason     = "eff_rank < 0.3: low-rank residual captures structure";
        r.confidence = 0.80f;
        return r;
    }

    // attention K/V projections are typically well-behaved
    if (fam == "attn_k" || fam == "attn_v") {
        r.expert     = TS_EXPERT_AWQ;
        r.reason     = "attention K/V: well-behaved, diagonal scaling sufficient";
        r.confidence = 0.85f;
        return r;
    }

    // well-conditioned, light tails
    if (er > 0.7f && kurt < 3.0f) {
        r.expert     = TS_EXPERT_AWQ;
        r.reason     = "well-conditioned (eff_rank > 0.7, kurtosis < 3): plain AWQ";
        r.confidence = 0.90f;
        return r;
    }

    r.expert     = TS_EXPERT_AWQ;
    r.reason     = "default regime: AWQ diagonal scaling";
    r.confidence = 0.50f;
    return r;
}

std::vector<ts_regime_routing> ts_regime_route_all(
    const ts_regime_descriptor * descs, int64_t n_tensors) {
    std::vector<ts_regime_routing> routings;
    routings.reserve(n_tensors);
    for (int64_t i = 0; i < n_tensors; i++) {
        routings.push_back(ts_regime_classify(&descs[i]));
    }
    return routings;
}

// --- expert profiles ---

const char * ts_expert_name(ts_expert_id expert) {
    switch (expert) {
        case TS_EXPERT_AWQ:       return "AWQ";
        case TS_EXPERT_LRQ:       return "LRQ";
        case TS_EXPERT_DARTQUANT: return "DartQuant";
        case TS_EXPERT_FLRQ:      return "FLRQ";
        case TS_EXPERT_CHAMPQ:    return "CHAMP-Q";
        case TS_EXPERT_SEPTQ:     return "SEPTQ";
        default:                  return "unknown";
    }
}

ts_expert_profile ts_expert_default_profile(ts_expert_id expert, int modality_id) {
    // baseline: identity multipliers, no SEPTQ, default grid, no forced outliers
    ts_expert_profile p;
    p.alpha_scale    = 1.0f;
    p.clip_scale     = 1.0f;
    p.use_septq      = false;
    p.awq_grid       = 20;
    p.max_outliers   = 0;
    p.outlier_thresh = 1.0f;

    switch (expert) {
        case TS_EXPERT_AWQ:
            // baseline diagonal scaling, no adjustment
            break;
        case TS_EXPERT_DARTQUANT:
            // rotation expert for massive outliers: tighter selection threshold
            // and a larger repair budget (+50% over the nominal base budget of 8)
            p.outlier_thresh = 0.8f;
            p.max_outliers   = 12;
            break;
        case TS_EXPERT_CHAMPQ:
            // permutation expert: finer alpha search, slightly stronger scaling
            p.awq_grid    = 40;
            p.alpha_scale = 1.1f;
            break;
        case TS_EXPERT_FLRQ:
            // factored low-rank: Hessian-compensation proxy, gentler clip
            p.use_septq  = true;
            p.clip_scale = 0.9f;
            break;
        case TS_EXPERT_LRQ:
            // aggressive low-rank: Hessian proxy, reduced alpha and clip
            p.use_septq   = true;
            p.alpha_scale = 0.9f;
            p.clip_scale  = 0.85f;
            break;
        case TS_EXPERT_SEPTQ:
            // Hessian compensation expert: SEPTQ on, finer grid
            p.use_septq = true;
            p.awq_grid  = 30;
            break;
        default:
            break;
    }

    // per-modality adjustments on top of the expert baseline
    if (modality_id == 2) {
        // audio: more sensitive to clipping -> tighter clip
        p.clip_scale *= 0.8f;
    } else if (modality_id == 1) {
        // image: more spatial outliers -> wider outlier budget
        p.max_outliers += 4;
    }

    return p;
}

// --- descriptor computation ---

static float ts_regime_kurtosis(const float * x, int64_t n) {
    if (n < 4) {
        return 3.0f;
    }
    float mean = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        mean += x[i];
    }
    mean /= (float)n;

    float m2 = 0.0f, m4 = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float d = x[i] - mean;
        float d2 = d * d;
        m2 += d2;
        m4 += d2 * d2;
    }
    m2 /= (float)n;
    m4 /= (float)n;

    if (m2 < 1e-24f) {
        return 3.0f;
    }
    // excess kurtosis
    return m4 / (m2 * m2) - 3.0f;
}

static float ts_regime_eff_rank(const float * x, int64_t n) {
    if (n < 1) {
        return 0.0f;
    }
    float sum = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        sum += fabsf(x[i]);
    }
    if (sum < 1e-24f) {
        return 0.0f;
    }
    // spectral entropy: H = -sum(p * log(p)), eff_rank = exp(H) / n
    float H = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float p = fabsf(x[i]) / sum;
        if (p > 1e-12f) {
            H -= p * logf(p);
        }
    }
    return expf(H) / (float)n;
}

static float ts_regime_percentile(const float * x, int64_t n, float pct) {
    if (n < 1) {
        return 0.0f;
    }
    std::vector<float> sorted(x, x + n);
    std::sort(sorted.begin(), sorted.end());
    float idx = pct * (float)(n - 1);
    int64_t lo = (int64_t)idx;
    int64_t hi = std::min(lo + 1, n - 1);
    float frac = idx - (float)lo;
    return sorted[lo] * (1.0f - frac) + sorted[hi] * frac;
}

ts_regime_descriptor ts_regime_compute_descriptor(
    const char * tensor_name,
    const float * weights, int64_t out_dim, int64_t in_dim,
    const float * imatrix_data, int64_t imatrix_dim,
    const float * imatrix_max_abs, int64_t imatrix_max_abs_dim) {

    ts_regime_descriptor desc;
    desc.tensor_name       = tensor_name ? tensor_name : "";
    desc.family            = ts_regime_infer_family(tensor_name);
    desc.out_dim           = out_dim;
    desc.in_dim            = in_dim;
    desc.modality          = ts_regime_infer_modality(tensor_name);
    desc.kurtosis          = 3.0f;
    desc.eff_rank          = 0.5f;
    desc.mean_magnitude    = 0.0f;
    desc.p99               = 0.0f;
    desc.max_outlier_ratio = 0.0f;

    if (imatrix_data && imatrix_dim > 0) {
        desc.kurtosis       = ts_regime_kurtosis(imatrix_data, imatrix_dim);
        desc.eff_rank       = ts_regime_eff_rank(imatrix_data, imatrix_dim);
        desc.p99            = ts_regime_percentile(imatrix_data, imatrix_dim, 0.99f);

        float sum = 0.0f;
        for (int64_t i = 0; i < imatrix_dim; i++) {
            sum += fabsf(imatrix_data[i]);
        }
        desc.mean_magnitude = sum / (float)imatrix_dim;
    } else if (weights && out_dim > 0 && in_dim > 0) {
        // fallback: derive stats from weight magnitudes per input channel
        int64_t n = in_dim;
        std::vector<float> col_mag(n, 0.0f);
        for (int64_t j = 0; j < in_dim; j++) {
            float s = 0.0f;
            for (int64_t i = 0; i < out_dim; i++) {
                s += fabsf(weights[i * in_dim + j]);
            }
            col_mag[j] = s / (float)out_dim;
        }
        desc.kurtosis       = ts_regime_kurtosis(col_mag.data(), n);
        desc.eff_rank       = ts_regime_eff_rank(col_mag.data(), n);
        desc.p99            = ts_regime_percentile(col_mag.data(), n, 0.99f);

        float sum = 0.0f;
        for (int64_t j = 0; j < n; j++) {
            sum += col_mag[j];
        }
        desc.mean_magnitude = sum / (float)n;
    }

    // Per-channel max |activation| -> localized outlier concentration.
    // The routing thresholds (kurtosis, eff_rank) are global scalars derived
    // from the per-channel mean-squared-act vector, so they cannot tell the
    // rotation/permutation experts WHICH channels carry the heavy tail. The
    // ratio of the largest per-channel max to the median max is a cheap,
    // scale-free proxy for "how concentrated are the outliers". 1.0 means
    // uniform (no localized outlier); >=~5 means a small set of channels
    // dominates. The experts key off this to grow the per-row repair budget.
    if (imatrix_max_abs && imatrix_max_abs_dim > 1) {
        std::vector<float> mags(imatrix_max_abs_dim);
        float max_abs = 0.0f;
        for (int64_t i = 0; i < imatrix_max_abs_dim; i++) {
            float v = fabsf(imatrix_max_abs[i]);
            mags[i] = v;
            if (v > max_abs) {
                max_abs = v;
            }
        }
        if (max_abs > 1e-30f) {
            float med = ts_regime_percentile(mags.data(), imatrix_max_abs_dim, 0.5f);
            if (med > 1e-30f) {
                desc.max_outlier_ratio = max_abs / med;
            }
        }
    }

    return desc;
}

ts_regime_descriptor ts_regime_compute_descriptor(
    const char * tensor_name,
    const float * weights, int64_t out_dim, int64_t in_dim,
    const float * imatrix_data, int64_t imatrix_dim) {
    // Forward to the max_abs-aware overload with no per-channel max. Keeps
    // max_outlier_ratio = 0 so callers that have no max data behave exactly
    // as before (the experts keep their default outlier budget).
    return ts_regime_compute_descriptor(
        tensor_name, weights, out_dim, in_dim,
        imatrix_data, imatrix_dim,
        nullptr, 0);
}

// --- summary ---

ts_regime_summary ts_regime_summarize(const std::vector<ts_regime_routing> * routings,
                                      const ts_regime_descriptor * descs,
                                      int64_t n_tensors) {
    ts_regime_summary s;
    memset(s.count_per_expert, 0, sizeof(s.count_per_expert));
    s.mean_kurtosis = 0.0f;
    s.mean_eff_rank = 0.0f;

    if (routings) {
        for (const auto & r : *routings) {
            if (r.expert >= 0 && r.expert < TS_EXPERT_COUNT) {
                s.count_per_expert[r.expert]++;
            }
        }
    }

    if (descs && n_tensors > 0) {
        for (int64_t i = 0; i < n_tensors; i++) {
            s.mean_kurtosis += descs[i].kurtosis;
            s.mean_eff_rank += descs[i].eff_rank;
        }
        s.mean_kurtosis /= (float)n_tensors;
        s.mean_eff_rank /= (float)n_tensors;
    }

    return s;
}
