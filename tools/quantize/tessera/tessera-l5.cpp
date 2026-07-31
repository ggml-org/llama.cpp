#include "tessera-l5.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

// ladder ordered low -> high precision
static const char * TS_L5_LADDER[] = {
    "q2_k", "q3_k", "q4_k", "q5_k", "q6_k", "tessera_t640", "q8_0",
};
static const int TS_L5_LADDER_LEN = 7;

// --- Sensitivity metrics ---

ts_score_map ts_l5_imatrix_magnitude(const float * imatrix_vals,
                                     const char ** tensor_names,
                                     const int64_t * tensor_dims,
                                     int64_t n_tensors) {
    ts_score_map out;
    if (n_tensors <= 0) {
        return out;
    }

    float peak = 0.0f;
    for (int64_t i = 0; i < n_tensors; i++) {
        int64_t dim = tensor_dims[i];
        float sum = 0.0f;
        const float * base = imatrix_vals;
        int64_t offset = 0;
        for (int64_t j = 0; j < i; j++) {
            offset += tensor_dims[j];
        }
        for (int64_t k = 0; k < dim; k++) {
            sum += fabsf(base[offset + k]);
        }
        float mean = (dim > 0) ? sum / (float)dim : 0.0f;
        mean = std::max(0.0f, mean);
        out[tensor_names[i]] = mean;
        if (mean > peak) {
            peak = mean;
        }
    }

    if (peak > 0.0f) {
        for (auto & kv : out) {
            kv.second /= peak;
        }
    }
    return out;
}

ts_score_map ts_l5_gradient_proxy(const float * output_sensitivity,
                                  const char ** tensor_names,
                                  int64_t n_tensors) {
    ts_score_map out;
    if (n_tensors <= 0) {
        return out;
    }

    float peak = 0.0f;
    for (int64_t i = 0; i < n_tensors; i++) {
        float v = std::max(0.0f, output_sensitivity[i]);
        out[tensor_names[i]] = v;
        if (v > peak) {
            peak = v;
        }
    }

    if (peak > 0.0f) {
        for (auto & kv : out) {
            kv.second /= peak;
        }
    }
    return out;
}

ts_score_map ts_l5_layer_position_prior(const char ** tensor_names,
                                        int64_t n_tensors,
                                        int64_t n_layers_total) {
    ts_score_map out;
    if (n_tensors <= 0) {
        return out;
    }

    if (n_layers_total < 1) {
        for (int64_t i = 0; i < n_tensors; i++) {
            out[tensor_names[i]] = 0.5f;
        }
        return out;
    }

    for (int64_t i = 0; i < n_tensors; i++) {
        std::string name(tensor_names[i]);
        int idx = -1;

        // parse "blk.<i>." pattern
        if (name.size() > 4 && name.compare(0, 4, "blk.") == 0) {
            size_t dot = name.find('.', 4);
            if (dot != std::string::npos) {
                std::string num = name.substr(4, dot - 4);
                bool valid = !num.empty();
                for (char c : num) {
                    if (c < '0' || c > '9') { valid = false; break; }
                }
                if (valid) {
                    idx = std::stoi(num);
                }
            }
        }

        if (idx < 0 || idx >= n_layers_total) {
            out[name] = 0.5f;
        } else if (n_layers_total == 1) {
            out[name] = 1.0f;
        } else {
            float frac = (float)idx / (float)(n_layers_total - 1);
            out[name] = frac;
        }
    }
    return out;
}

ts_score_map ts_l5_combine(const ts_score_map ** scorers,
                           const float * weights,
                           int64_t n_scorers) {
    ts_score_map out;
    if (n_scorers <= 0) {
        return out;
    }

    // collect union of keys
    for (int64_t s = 0; s < n_scorers; s++) {
        for (const auto & kv : *scorers[s]) {
            if (out.find(kv.first) == out.end()) {
                out[kv.first] = 0.0f;
            }
        }
    }

    for (auto & kv : out) {
        float sum = 0.0f;
        for (int64_t s = 0; s < n_scorers; s++) {
            auto it = scorers[s]->find(kv.first);
            float val = (it != scorers[s]->end()) ? it->second : 0.0f;
            sum += weights[s] * val;
        }
        kv.second = sum;
    }
    return out;
}

// --- EMA ---

void ts_l5_ema_init(ts_l5_ema * ema, float beta) {
    ema->state.clear();
    ema->beta = beta;
}

void ts_l5_ema_update(ts_l5_ema * ema, const ts_score_map * new_scores) {
    for (const auto & kv : *new_scores) {
        auto it = ema->state.find(kv.first);
        if (it == ema->state.end()) {
            ema->state[kv.first] = kv.second;
        } else {
            it->second = ema->beta * it->second + (1.0f - ema->beta) * kv.second;
        }
    }
}

// --- Percentile ---

ts_score_map ts_l5_percentile_rank(const ts_score_map * scores) {
    ts_score_map out;
    int64_t n = (int64_t)scores->size();
    if (n == 0) {
        return out;
    }
    if (n == 1) {
        out[scores->begin()->first] = 0.5f;
        return out;
    }

    // sort by value ascending
    std::vector<std::pair<std::string, float>> items(scores->begin(), scores->end());
    std::sort(items.begin(), items.end(),
              [](const auto & a, const auto & b) { return a.second < b.second; });

    int64_t i = 0;
    while (i < n) {
        int64_t j = i;
        while (j + 1 < n && items[j + 1].second == items[i].second) {
            j++;
        }
        float avg = 0.5f * (float)(i + j) / (float)(n - 1);
        for (int64_t k = i; k <= j; k++) {
            out[items[k].first] = avg;
        }
        i = j + 1;
    }
    return out;
}

// --- Pick top ---

std::vector<std::string> ts_l5_pick_top(const ts_score_map * scores,
                                        float fraction) {
    std::vector<std::string> out;
    if (fraction <= 0.0f || scores->empty()) {
        return out;
    }
    if (fraction >= 1.0f) {
        for (const auto & kv : *scores) {
            out.push_back(kv.first);
        }
        return out;
    }

    // sort descending by score
    std::vector<std::pair<std::string, float>> items(scores->begin(), scores->end());
    std::sort(items.begin(), items.end(),
              [](const auto & a, const auto & b) { return a.second > b.second; });

    int64_t count = std::max((int64_t)1, (int64_t)std::round(fraction * (float)items.size()));
    count = std::min(count, (int64_t)items.size());
    for (int64_t i = 0; i < count; i++) {
        out.push_back(items[i].first);
    }
    return out;
}

// --- Expected MSE delta ---

float ts_l5_expected_mse_delta(const char * tensor_name,
                               float current_score, float target_score) {
    (void)tensor_name;
    float delta = current_score - target_score;
    return std::max(0.0f, delta);
}

// --- Ladder ---

int ts_l5_ladder_index(const char * qtype) {
    for (int i = 0; i < TS_L5_LADDER_LEN; i++) {
        if (strcmp(qtype, TS_L5_LADDER[i]) == 0) {
            return i;
        }
    }
    return -1;
}

const char * ts_l5_step_up(const char * qtype) {
    int idx = ts_l5_ladder_index(qtype);
    if (idx < 0 || idx + 1 >= TS_L5_LADDER_LEN) {
        return nullptr;
    }
    return TS_L5_LADDER[idx + 1];
}

const char * ts_l5_step_down(const char * qtype) {
    int idx = ts_l5_ladder_index(qtype);
    if (idx <= 0) {
        return nullptr;
    }
    return TS_L5_LADDER[idx - 1];
}

// --- Orchestrator ---

int ts_l5_orchestrate_step(const ts_score_map * sensitivity,
                           const char ** current_qtypes,
                           int64_t n_tensors,
                           int64_t generation,
                           const ts_orchestrator_params * params,
                           ts_requant_plan * plan) {
    plan->actions.clear();
    plan->total_expected_delta = 0.0f;
    plan->generation = generation;

    if (n_tensors <= 0 || sensitivity->empty()) {
        return 0;
    }

    // build name list aligned with current_qtypes
    std::vector<std::string> names;
    names.reserve(n_tensors);
    for (const auto & kv : *sensitivity) {
        names.push_back(kv.first);
    }

    // pick top fraction
    std::vector<std::string> top = ts_l5_pick_top(sensitivity, params->top_fraction);

    // for each top tensor, compute expected delta and maybe generate action
    for (const auto & name : top) {
        auto it = sensitivity->find(name);
        if (it == sensitivity->end()) {
            continue;
        }
        float score = it->second;

        // find this tensor's index to get its qtype
        int64_t tidx = -1;
        for (int64_t i = 0; i < n_tensors; i++) {
            if (names[i] == name) {
                tidx = i;
                break;
            }
        }
        if (tidx < 0) {
            continue;
        }

        const char * cur_qtype = current_qtypes[tidx];
        const char * next = ts_l5_step_up(cur_qtype);
        if (next == nullptr) {
            continue;
        }

        // target score: stepping up reduces sensitivity impact
        float target_score = score * 0.5f;
        float delta = ts_l5_expected_mse_delta(name.c_str(), score, target_score);

        if (delta > params->delta_threshold) {
            ts_requant_action action;
            action.tensor_name = name;
            action.type = TS_REQUANT_STEP_UP;
            action.from_qtype = cur_qtype;
            action.to_qtype = next;
            action.expected_delta = delta;
            plan->actions.push_back(action);
            plan->total_expected_delta += delta;
        }
    }

    return (int)plan->actions.size();
}

// --- Adaptive requantization ---

void ts_l5_adaptive_default_params(ts_l5_adaptive_params * p) {
    if (p == nullptr) {
        return;
    }
    p->alpha_scale = 0.5f;
    p->clip_scale  = 0.5f;
    p->min_alpha   = 0.1f;
    p->min_clip    = 0.1f;
}

int ts_l5_adaptive_requant(const ts_l2_report * report,
                           const ts_l5_adaptive_params * params,
                           int64_t generation,
                           ts_l5_adaptive_plan * plan) {
    if (plan == nullptr) {
        return -1;
    }
    plan->specs.clear();
    plan->n_requant  = 0;
    plan->generation = generation;
    if (report == nullptr) {
        return 0;
    }

    ts_l5_adaptive_params p;
    if (params != nullptr) {
        p = *params;
    } else {
        ts_l5_adaptive_default_params(&p);
    }

    for (const auto & t : report->tensors) {
        if (!t.flagged) {
            continue;
        }

        // how far the observed divergence overshoots the type baseline
        float overshoot = (t.expected_frob > 0.0f)
                              ? t.divergence.relative_frobenius / t.expected_frob
                              : 1.0f;
        if (overshoot < 1.0f) {
            overshoot = 1.0f;
        }

        // tighten proportionally: worse tensors get smaller alpha/clip
        float new_alpha = p.alpha_scale / overshoot;
        float new_clip  = p.clip_scale  / overshoot;
        if (new_alpha < p.min_alpha) {
            new_alpha = p.min_alpha;
        }
        if (new_clip < p.min_clip) {
            new_clip = p.min_clip;
        }

        ts_l5_requant_spec s;
        s.tensor_name = t.tensor_name;
        s.qtype       = t.qtype;
        s.divergence  = t.divergence.relative_frobenius;
        s.expected    = t.expected_frob;
        s.overshoot   = overshoot;
        s.new_alpha   = new_alpha;
        s.new_clip    = new_clip;
        plan->specs.push_back(s);
        plan->n_requant++;
    }

    return (int)plan->n_requant;
}
