//
// tessera-acceptance.cpp
//
// G6 acceptance gate: composite-beats-single (Test 1) and ranking
// disagreement (Test 2). See tessera-acceptance.h and
// docs/research-alignment-2026-07-30.md Section 5.
//

#include "tessera-acceptance.h"
#include "tessera-ab-harness.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static const char * TS_ACCEPTANCE_SCHEMA = "llama.tessera.acceptance.v1";

void ts_acceptance_default_config(ts_acceptance_config * cfg) {
    if (!cfg) return;
    cfg->n_heldout_tensors = 0;
    cfg->heldout_fraction  = 0.2f;
    cfg->verbose           = false;
    cfg->output_path[0]    = '\0';
}

// Compute the mean of a field across held-out tensors.
static float ts_acceptance_mean_field(const ts_acceptance_tensor * tensors,
                                      int64_t n, int field_offset) {
    float sum = 0.0f;
    int64_t count = 0;
    for (int64_t i = 0; i < n; i++) {
        if (!tensors[i].held_out) continue;
        sum += *(const float *)((const char *)&tensors[i] + field_offset);
        count++;
    }
    return count > 0 ? sum / (float)count : 0.0f;
}

int ts_acceptance_run(const ts_acceptance_config * config,
                      const ts_acceptance_tensor * tensors,
                      int64_t n_tensors,
                      ts_acceptance_result * result) {
    if (!config || !tensors || !result || n_tensors < 0) {
        return -1;
    }

    memset(result, 0, sizeof(*result));
    result->n_tensors_total = n_tensors;

    if (n_tensors == 0) {
        snprintf(result->verdict, sizeof(result->verdict),
                 "ACCEPTANCE SKIP: no tensors");
        return 0;
    }

    // --- held-out selection ---
    // If no tensors are explicitly marked held_out, use the heldout_fraction
    // to mark the trailing fraction as held-out (deterministic: last N).
    int64_t n_heldout = 0;
    for (int64_t i = 0; i < n_tensors; i++) {
        if (tensors[i].held_out) n_heldout++;
    }

    // work on a mutable copy so we can apply the fraction-based fallback
    std::vector<ts_acceptance_tensor> work(tensors, tensors + n_tensors);

    if (n_heldout == 0) {
        int64_t n_hold = config->n_heldout_tensors > 0
            ? config->n_heldout_tensors
            : (int64_t)(config->heldout_fraction * (float)n_tensors);
        if (n_hold < 1 && n_tensors > 1) n_hold = 1;
        if (n_hold > n_tensors) n_hold = n_tensors;
        for (int64_t i = n_tensors - n_hold; i < n_tensors; i++) {
            work[i].held_out = true;
        }
        n_heldout = n_hold;
    }
    result->n_tensors_heldout = n_heldout;

    // --- Test 1: composite beats single-proxy ---
    // Mean t_l^2 over held-out tensors for each approach.
    const int off_comp = (int)((const char *)&work[0].composite_t2 - (const char *)&work[0]);
    const int off_awq  = (int)((const char *)&work[0].awq_t2       - (const char *)&work[0]);
    const int off_rot  = (int)((const char *)&work[0].rotation_t2  - (const char *)&work[0]);
    const int off_lr   = (int)((const char *)&work[0].lowrank_t2   - (const char *)&work[0]);
    const int off_hess = (int)((const char *)&work[0].hessian_t2   - (const char *)&work[0]);

    result->composite_t2 = ts_acceptance_mean_field(work.data(), n_tensors, off_comp);
    result->awq_t2       = ts_acceptance_mean_field(work.data(), n_tensors, off_awq);
    result->rotation_t2  = ts_acceptance_mean_field(work.data(), n_tensors, off_rot);
    result->lowrank_t2   = ts_acceptance_mean_field(work.data(), n_tensors, off_lr);
    result->hessian_t2   = ts_acceptance_mean_field(work.data(), n_tensors, off_hess);

    result->best_single_t2 = result->awq_t2;
    if (result->rotation_t2 < result->best_single_t2) result->best_single_t2 = result->rotation_t2;
    if (result->lowrank_t2  < result->best_single_t2) result->best_single_t2 = result->lowrank_t2;
    if (result->hessian_t2  < result->best_single_t2) result->best_single_t2 = result->hessian_t2;

    result->composite_wins = result->composite_t2 < result->best_single_t2;
    result->improvement_pct = (result->best_single_t2 > 1e-12f)
        ? (result->best_single_t2 - result->composite_t2) / result->best_single_t2 * 100.0f
        : 0.0f;

    // --- Test 2: ranking disagreement ---
    // Kendall tau between offline proxy and kernel-direct rankings.
    std::vector<float> offline_scores;
    std::vector<float> kernel_scores;
    offline_scores.reserve(n_tensors);
    kernel_scores.reserve(n_tensors);
    for (int64_t i = 0; i < n_tensors; i++) {
        offline_scores.push_back(work[i].offline_proxy_mse);
        kernel_scores.push_back(work[i].kernel_direct_t2);
    }

    result->kendall_tau = ts_ab_kendall_tau(offline_scores.data(),
                                            kernel_scores.data(),
                                            n_tensors);
    result->ranking_disagreement = 1.0f - fabsf(result->kendall_tau);
    result->novelty_survives = result->ranking_disagreement > 0.05f;

    // --- verdict ---
    result->acceptance_passed = result->composite_wins && result->novelty_survives;

    snprintf(result->verdict, sizeof(result->verdict),
             "G6 %s | composite_t2=%.6g best_single=%.6g (%.1f%% %s) | "
             "tau=%.4f disagreement=%.4f novelty=%s | "
             "proxies: awq=%.6g rot=%.6g lr=%.6g hess=%.6g | "
             "held_out=%lld/%lld",
             result->acceptance_passed ? "PASS" : "FAIL",
             (double)result->composite_t2,
             (double)result->best_single_t2,
             (double)result->improvement_pct,
             result->composite_wins ? "better" : "worse",
             (double)result->kendall_tau,
             (double)result->ranking_disagreement,
             result->novelty_survives ? "survives" : "null",
             (double)result->awq_t2,
             (double)result->rotation_t2,
             (double)result->lowrank_t2,
             (double)result->hessian_t2,
             (long long)result->n_tensors_heldout,
             (long long)result->n_tensors_total);

    if (config->verbose) {
        printf("tessera-acceptance: %s\n", result->verdict);
    }

    // --- optional JSON report ---
    if (config->output_path[0] != '\0') {
        ts_acceptance_write_json(config->output_path, config, result,
                                 work.data(), n_tensors);
    }

    return 0;
}

// --- JSON serialization ---

static void ts_acc_json_escape(std::string * out, const char * in) {
    for (const char * p = in; *p; p++) {
        switch (*p) {
            case '"':  *out += "\\\""; break;
            case '\\': *out += "\\\\"; break;
            case '\n': *out += "\\n";  break;
            case '\t': *out += "\\t";  break;
            default:   *out += *p;     break;
        }
    }
}

static void ts_acc_json_float(std::string * out, float v) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.9g", (double)v);
    *out += buf;
}

int ts_acceptance_write_json(const char * path,
                             const ts_acceptance_config * config,
                             const ts_acceptance_result * result,
                             const ts_acceptance_tensor * tensors,
                             int64_t n_tensors) {
    if (!path || !result) return -1;
    (void)config;  // reserved for provenance fields

    std::string j;
    j += "{";

    j += "\"schema\":\"";
    j += TS_ACCEPTANCE_SCHEMA;
    j += "\"";

    j += ",\"acceptance_passed\":";
    j += result->acceptance_passed ? "true" : "false";

    j += ",\"composite_t2\":";
    ts_acc_json_float(&j, result->composite_t2);

    j += ",\"best_single_t2\":";
    ts_acc_json_float(&j, result->best_single_t2);

    j += ",\"improvement_pct\":";
    ts_acc_json_float(&j, result->improvement_pct);

    j += ",\"composite_wins\":";
    j += result->composite_wins ? "true" : "false";

    j += ",\"kendall_tau\":";
    ts_acc_json_float(&j, result->kendall_tau);

    j += ",\"ranking_disagreement\":";
    ts_acc_json_float(&j, result->ranking_disagreement);

    j += ",\"novelty_survives\":";
    j += result->novelty_survives ? "true" : "false";

    j += ",\"per_proxy\":{\"awq\":";
    ts_acc_json_float(&j, result->awq_t2);
    j += ",\"rotation\":";
    ts_acc_json_float(&j, result->rotation_t2);
    j += ",\"lowrank\":";
    ts_acc_json_float(&j, result->lowrank_t2);
    j += ",\"hessian\":";
    ts_acc_json_float(&j, result->hessian_t2);
    j += "}";

    j += ",\"n_tensors_total\":";
    j += std::to_string(result->n_tensors_total);

    j += ",\"n_tensors_heldout\":";
    j += std::to_string(result->n_tensors_heldout);

    j += ",\"verdict\":\"";
    ts_acc_json_escape(&j, result->verdict);
    j += "\"";

    // per-tensor breakdown
    j += ",\"tensors\":[";
    for (int64_t i = 0; i < n_tensors; i++) {
        if (i > 0) j += ",";
        j += "{\"name\":\"";
        ts_acc_json_escape(&j, tensors[i].name);
        j += "\",\"composite_t2\":";
        ts_acc_json_float(&j, tensors[i].composite_t2);
        j += ",\"awq_t2\":";
        ts_acc_json_float(&j, tensors[i].awq_t2);
        j += ",\"rotation_t2\":";
        ts_acc_json_float(&j, tensors[i].rotation_t2);
        j += ",\"lowrank_t2\":";
        ts_acc_json_float(&j, tensors[i].lowrank_t2);
        j += ",\"hessian_t2\":";
        ts_acc_json_float(&j, tensors[i].hessian_t2);
        j += ",\"offline_proxy_mse\":";
        ts_acc_json_float(&j, tensors[i].offline_proxy_mse);
        j += ",\"kernel_direct_t2\":";
        ts_acc_json_float(&j, tensors[i].kernel_direct_t2);
        j += ",\"held_out\":";
        j += tensors[i].held_out ? "true" : "false";
        j += "}";
    }
    j += "]";

    j += "}";

    FILE * f = fopen(path, "w");
    if (!f) return -1;
    fwrite(j.c_str(), 1, j.size(), f);
    fputc('\n', f);
    fclose(f);
    return 0;
}
