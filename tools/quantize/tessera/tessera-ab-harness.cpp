//
// tessera-ab-harness.cpp
//
// G6 acceptance harness: alpha-weighted composites, Kendall tau ranking
// agreement, and receipt serialization.
//

#include "tessera-ab-harness.h"

#include <cstdio>
#include <cstring>

// Count concordant and discordant pairs between two score arrays.
// A pair (i, j) is concordant when both arrays order it the same way,
// discordant when they order it oppositely; ties count as neither.
static void ts_ab_pair_counts(const float * a, const float * b, int64_t n,
                              int64_t * concordant, int64_t * discordant) {
    int64_t c = 0;
    int64_t d = 0;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i + 1; j < n; ++j) {
            const float p = (a[i] - a[j]) * (b[i] - b[j]);
            if (p > 0.0f) {
                c++;
            } else if (p < 0.0f) {
                d++;
            }
        }
    }
    *concordant = c;
    *discordant = d;
}

float ts_ab_kendall_tau(const float * a, const float * b, int64_t n) {
    if (a == nullptr || b == nullptr || n < 2) {
        return 0.0f;
    }

    int64_t concordant = 0;
    int64_t discordant = 0;
    ts_ab_pair_counts(a, b, n, &concordant, &discordant);

    const double pairs = (double)n * (double)(n - 1) / 2.0;
    return (float)((double)(concordant - discordant) / pairs);
}

int ts_ab_run(const std::vector<ts_ab_tensor_scores> * scores_in,
              const ts_ab_harness_params * params,
              ts_ab_harness_result * result) {
    if (scores_in == nullptr || params == nullptr || result == nullptr) {
        return -1;
    }

    int64_t n = (int64_t)scores_in->size();
    if (params->n_heldout > 0 && params->n_heldout < n) {
        n = params->n_heldout;
    }

    result->scores.assign(scores_in->begin(), scores_in->begin() + n);
    result->composite_offline      = 0.0f;
    result->composite_kernel       = 0.0f;
    result->kendall_tau            = 0.0f;
    result->ranking_disagreement   = 0.0f;
    result->composite_beats_single = false;
    result->report.clear();

    if (n == 0) {
        result->report = "ts_ab_run: no tensors";
        return 0;
    }

    float min_kernel = result->scores[0].kernel_direct_t2;
    for (int64_t i = 0; i < n; ++i) {
        const ts_ab_tensor_scores & s = result->scores[i];
        result->composite_offline += s.alpha_l * s.offline_proxy_mse;
        result->composite_kernel  += s.alpha_l * s.kernel_direct_t2;
        if (s.kernel_direct_t2 < min_kernel) {
            min_kernel = s.kernel_direct_t2;
        }
    }

    // lower t_l^2 is better; the composite beats the best single proxy
    // only if the aggregate kernel error falls below the best single tensor
    result->composite_beats_single = result->composite_kernel < min_kernel;

    if (params->measure_ranking && n >= 2) {
        std::vector<float> offline(n);
        std::vector<float> kernel(n);
        for (int64_t i = 0; i < n; ++i) {
            offline[i] = result->scores[i].offline_proxy_mse;
            kernel[i]  = result->scores[i].kernel_direct_t2;
        }

        int64_t concordant = 0;
        int64_t discordant = 0;
        ts_ab_pair_counts(offline.data(), kernel.data(), n, &concordant, &discordant);

        const double pairs = (double)n * (double)(n - 1) / 2.0;
        result->kendall_tau          = (float)((double)(concordant - discordant) / pairs);
        result->ranking_disagreement = (float)((double)discordant / pairs);
    }

    char buf[512];
    snprintf(buf, sizeof(buf),
             "ts_ab_run: n=%lld composite_offline=%.6g composite_kernel=%.6g "
             "kendall_tau=%.4f ranking_disagreement=%.4f composite_beats_single=%s",
             (long long)n,
             (double)result->composite_offline,
             (double)result->composite_kernel,
             (double)result->kendall_tau,
             (double)result->ranking_disagreement,
             result->composite_beats_single ? "true" : "false");
    result->report = buf;

    if (params->verbose) {
        std::printf("%s\n", result->report.c_str());
    }

    return 0;
}

static void ts_ab_json_escape(std::string * out, const std::string & in) {
    for (char c : in) {
        switch (c) {
            case '"':  *out += "\\\""; break;
            case '\\': *out += "\\\\"; break;
            case '\n': *out += "\\n";  break;
            case '\t': *out += "\\t";  break;
            default:   *out += c;      break;
        }
    }
}

std::string ts_ab_receipt_json(const ts_ab_harness_result * result) {
    if (result == nullptr) {
        return "{}";
    }

    char num[64];
    std::string json;
    json += "{";

    json += "\"n_tensors\":";
    snprintf(num, sizeof(num), "%lld", (long long)result->scores.size());
    json += num;

    json += ",\"composite_offline\":";
    snprintf(num, sizeof(num), "%.9g", (double)result->composite_offline);
    json += num;

    json += ",\"composite_kernel\":";
    snprintf(num, sizeof(num), "%.9g", (double)result->composite_kernel);
    json += num;

    json += ",\"kendall_tau\":";
    snprintf(num, sizeof(num), "%.9g", (double)result->kendall_tau);
    json += num;

    json += ",\"ranking_disagreement\":";
    snprintf(num, sizeof(num), "%.9g", (double)result->ranking_disagreement);
    json += num;

    json += ",\"composite_beats_single\":";
    json += result->composite_beats_single ? "true" : "false";

    json += ",\"scores\":[";
    for (size_t i = 0; i < result->scores.size(); ++i) {
        const ts_ab_tensor_scores & s = result->scores[i];
        if (i > 0) {
            json += ",";
        }
        json += "{\"name\":\"";
        ts_ab_json_escape(&json, s.name);
        json += "\",\"offline_proxy_mse\":";
        snprintf(num, sizeof(num), "%.9g", (double)s.offline_proxy_mse);
        json += num;
        json += ",\"kernel_direct_t2\":";
        snprintf(num, sizeof(num), "%.9g", (double)s.kernel_direct_t2);
        json += num;
        json += ",\"alpha_l\":";
        snprintf(num, sizeof(num), "%.9g", (double)s.alpha_l);
        json += num;
        json += "}";
    }
    json += "]";

    json += "}";
    return json;
}
