//
// tessera-mm-fitness.cpp
//
// Modality-aware GA fitness weighting. Composite objective is the
// alpha-weighted Linearity-Theorem form Sum_l alpha_l * t_l^2 extended
// across modality: per-modality losses are combined with GA-evolved
// weights (default text/image/audio = 0.5/0.3/0.2, M1).
//

#include "tessera-mm-fitness.h"

#include <cstring>

#define TS_MM_N_MODALITY 3

ts_mm_fitness_params ts_mm_fitness_default_params() {
    ts_mm_fitness_params p;
    p.modality_weights[0] = 0.5f;   // text
    p.modality_weights[1] = 0.3f;   // image
    p.modality_weights[2] = 0.2f;   // audio
    p.per_family_breakdown = false;
    return p;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// a modality contributes only if present and backed by data
static bool ts_mm_active(const float * t2, const bool * present, int m) {
    return present && present[m] && t2 != nullptr;
}

// renormalize weights over active modalities (M4: single-modality filter
// collapses to that modality; missing modalities drop out of the sum)
static void ts_mm_norm_weights(const float * w, const bool * active, float w_norm[TS_MM_N_MODALITY]) {
    float wsum = 0.0f;
    for (int m = 0; m < TS_MM_N_MODALITY; m++) {
        w_norm[m] = 0.0f;
        if (active[m]) wsum += w[m];
    }
    if (wsum <= 0.0f) return;
    for (int m = 0; m < TS_MM_N_MODALITY; m++) {
        if (active[m]) w_norm[m] = w[m] / wsum;
    }
}

static float ts_mm_mean(const float * x, int64_t n) {
    if (!x || n <= 0) return 0.0f;
    float s = 0.0f;
    for (int64_t i = 0; i < n; i++) s += x[i];
    return s / (float)n;
}

// ---------------------------------------------------------------------------
// Composite fitness
// ---------------------------------------------------------------------------

ts_mm_fitness_score ts_mm_fitness_compute(
    const float * t2_per_modality[3],
    const float * alpha_l,
    const bool present[3],
    int64_t n_layers,
    const ts_mm_fitness_params * params) {

    ts_mm_fitness_score score;
    score.composite = 0.0f;
    score.alpha_weighted = 0.0f;
    for (int m = 0; m < TS_MM_N_MODALITY; m++) score.per_modality[m] = 0.0f;

    if (!t2_per_modality || n_layers <= 0) return score;

    ts_mm_fitness_params p = params ? *params : ts_mm_fitness_default_params();

    bool active[TS_MM_N_MODALITY];
    for (int m = 0; m < TS_MM_N_MODALITY; m++) {
        active[m] = ts_mm_active(t2_per_modality[m], present, m);
    }

    float w_norm[TS_MM_N_MODALITY];
    ts_mm_norm_weights(p.modality_weights, active, w_norm);

    // per-modality loss = mean_l(t_l^2); composite = weighted sum
    for (int m = 0; m < TS_MM_N_MODALITY; m++) {
        if (!active[m]) continue;
        score.per_modality[m] = ts_mm_mean(t2_per_modality[m], n_layers);
        score.composite += w_norm[m] * score.per_modality[m];
    }

    // alpha-weighted: Sum_l alpha_l * (modality-weighted t_l^2 for layer l)
    for (int64_t l = 0; l < n_layers; l++) {
        float wt2 = 0.0f;
        for (int m = 0; m < TS_MM_N_MODALITY; m++) {
            if (active[m]) wt2 += w_norm[m] * t2_per_modality[m][l];
        }
        float alpha = alpha_l ? alpha_l[l] : 1.0f;
        score.alpha_weighted += alpha * wt2;
    }

    return score;
}

// ---------------------------------------------------------------------------
// Per-family breakdown
// ---------------------------------------------------------------------------

std::vector<ts_mm_family_score> ts_mm_fitness_family_breakdown(
    const char ** tensor_names,
    const char ** tensor_families,
    const float * t2_per_modality[3],
    const bool present[3],
    int64_t n_tensors,
    const ts_mm_fitness_params * params) {

    (void)tensor_names;

    std::vector<ts_mm_family_score> out;
    if (!tensor_families || !t2_per_modality || n_tensors <= 0) return out;

    ts_mm_fitness_params p = params ? *params : ts_mm_fitness_default_params();

    bool active[TS_MM_N_MODALITY];
    for (int m = 0; m < TS_MM_N_MODALITY; m++) {
        active[m] = ts_mm_active(t2_per_modality[m], present, m);
    }

    float w_norm[TS_MM_N_MODALITY];
    ts_mm_norm_weights(p.modality_weights, active, w_norm);

    // per-family accumulators, first-seen order
    std::vector<float> sum_mm[TS_MM_N_MODALITY];

    for (int64_t t = 0; t < n_tensors; t++) {
        const char * fam = tensor_families[t] ? tensor_families[t] : "";

        size_t idx = out.size();
        for (size_t i = 0; i < out.size(); i++) {
            if (out[i].family == fam) { idx = i; break; }
        }
        if (idx == out.size()) {
            ts_mm_family_score fs;
            fs.family = fam;
            fs.composite = 0.0f;
            fs.n_tensors = 0;
            for (int m = 0; m < TS_MM_N_MODALITY; m++) fs.loss_per_modality[m] = 0.0f;
            out.push_back(fs);
            for (int m = 0; m < TS_MM_N_MODALITY; m++) sum_mm[m].push_back(0.0f);
        }

        out[idx].n_tensors++;
        for (int m = 0; m < TS_MM_N_MODALITY; m++) {
            if (active[m]) sum_mm[m][idx] += t2_per_modality[m][t];
        }
    }

    for (size_t i = 0; i < out.size(); i++) {
        float loss[TS_MM_N_MODALITY];
        for (int m = 0; m < TS_MM_N_MODALITY; m++) {
            loss[m] = (active[m] && out[i].n_tensors > 0)
                ? sum_mm[m][i] / (float)out[i].n_tensors
                : 0.0f;
            out[i].loss_per_modality[m] = loss[m];
        }
        float comp = 0.0f;
        for (int m = 0; m < TS_MM_N_MODALITY; m++) comp += w_norm[m] * loss[m];
        out[i].composite = comp;
    }

    return out;
}
