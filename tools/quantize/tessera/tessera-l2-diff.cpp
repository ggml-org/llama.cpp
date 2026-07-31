//
// tessera-l2-diff.cpp
//
// L2 BF16-vs-quantized weight-level differential. See tessera-l2-diff.h.
//

#include "tessera-l2-diff.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

using json = nlohmann::json;

static const char * TS_L2_SCHEMA = "llama.tessera.runtime-probe.v1";

// finite stand-in for an infinite relative error (zero-norm reference)
static const float TS_L2_INF = 1e30f;

void ts_l2_default_config(ts_l2_config * cfg) {
    if (cfg == nullptr) {
        return;
    }
    cfg->bf16_model_path[0]   = '\0';
    cfg->quant_model_path[0]  = '\0';
    cfg->corpus_path[0]       = '\0';
    cfg->output_json_path[0]  = '\0';
    cfg->flag_multiplier      = 1.5f;
}

float ts_l2_expected_frob(const char * qtype) {
    if (qtype == nullptr) {
        return 5e-2f;
    }
    if (strcmp(qtype, "f16") == 0 || strcmp(qtype, "f32") == 0) {
        return 1e-5f;
    }
    if (strcmp(qtype, "q8_0") == 0) {
        return 1e-3f;
    }
    if (strcmp(qtype, "q4_k") == 0 || strcmp(qtype, "q4_0") == 0) {
        return 5e-2f;
    }
    if (strcmp(qtype, "tessera_t640") == 0 || strcmp(qtype, "t640") == 0) {
        return 2e-2f;
    }
    return 5e-2f;
}

ts_l2_divergence ts_l2_tensor_divergence(const float * bf16,
                                         const float * quant,
                                         int64_t n) {
    ts_l2_divergence d = { 0.0f, 0.0f, 0.0f, 0.0f };
    if (bf16 == nullptr || quant == nullptr || n <= 0) {
        return d;
    }

    double max_abs = 0.0;
    double sum_abs = 0.0;
    double num     = 0.0;   // ||bf16 - quant||_F^2
    double den     = 0.0;   // ||bf16||_F^2
    for (int64_t i = 0; i < n; i++) {
        const double diff = (double)bf16[i] - (double)quant[i];
        const double a    = fabs(diff);
        if (a > max_abs) {
            max_abs = a;
        }
        sum_abs += a;
        num += diff * diff;
        den += (double)bf16[i] * (double)bf16[i];
    }

    d.max_abs  = (float)max_abs;
    d.mean_abs = (float)(sum_abs / (double)n);
    d.relative_frobenius = (den > 0.0) ? (float)(num / den)
                                       : (num > 0.0 ? TS_L2_INF : 0.0f);
    d.per_layer_norm = (float)sqrt(num / (double)n);
    return d;
}

int ts_l2_run(const ts_l2_config * cfg,
              const ts_l2_tensor_input * inputs,
              int64_t n_tensors,
              ts_l2_report * report) {
    if (cfg == nullptr || inputs == nullptr || report == nullptr || n_tensors < 0) {
        return -1;
    }

    const float mult = cfg->flag_multiplier > 0.0f ? cfg->flag_multiplier : 1.5f;

    report->tensors.clear();
    report->tensors.reserve((size_t)n_tensors);
    report->n_flagged = 0;

    for (int64_t i = 0; i < n_tensors; i++) {
        const ts_l2_tensor_input & in = inputs[i];
        const int64_t n = in.rows * in.cols;

        ts_l2_tensor_result r;
        r.tensor_name = in.name != nullptr ? in.name : "";
        r.qtype       = in.qtype != nullptr ? in.qtype : "";
        r.rows        = in.rows;
        r.cols        = in.cols;
        r.divergence  = ts_l2_tensor_divergence(in.bf16, in.quant, n);
        r.expected_frob  = ts_l2_expected_frob(in.qtype);
        r.flag_threshold = mult * r.expected_frob;
        r.flagged        = r.divergence.relative_frobenius > r.flag_threshold;

        if (r.flagged) {
            report->n_flagged++;
        }
        report->tensors.push_back(std::move(r));
    }

    if (cfg->output_json_path[0] != '\0') {
        if (ts_l2_write_report(cfg->output_json_path, cfg, report) != 0) {
            return -1;
        }
    }
    return (int)report->n_flagged;
}

int ts_l2_write_report(const char * path,
                       const ts_l2_config * cfg,
                       const ts_l2_report * report) {
    if (path == nullptr || report == nullptr) {
        return -1;
    }

    json j;
    j["schema"]          = TS_L2_SCHEMA;
    j["layer"]           = "L2";
    j["bf16_model"]      = cfg != nullptr ? cfg->bf16_model_path : "";
    j["quant_model"]     = cfg != nullptr ? cfg->quant_model_path : "";
    j["corpus"]          = cfg != nullptr ? cfg->corpus_path : "";
    j["flag_multiplier"] = cfg != nullptr ? cfg->flag_multiplier : 1.5f;
    j["n_tensors"]       = (int64_t)report->tensors.size();
    j["n_flagged"]       = report->n_flagged;

    json tensors = json::array();
    for (const auto & r : report->tensors) {
        json t;
        t["tensor"] = r.tensor_name;
        t["qtype"]  = r.qtype;
        t["shape"]  = json::array({ r.rows, r.cols });

        json div;
        div["max_abs"]            = r.divergence.max_abs;
        div["mean_abs"]           = r.divergence.mean_abs;
        div["relative_frobenius"] = r.divergence.relative_frobenius;
        div["per_layer_norm"]     = r.divergence.per_layer_norm;
        t["divergence"] = div;

        t["expected_frob"]  = r.expected_frob;
        t["flag_threshold"] = r.flag_threshold;
        t["flagged"]        = r.flagged;
        tensors.push_back(t);
    }
    j["tensors"] = tensors;

    std::ofstream out(path);
    if (!out) {
        return -1;
    }
    out << j.dump(2) << "\n";
    return out.good() ? 0 : -1;
}

int ts_l2_load_report(const char * path, ts_l2_report * report) {
    if (path == nullptr || report == nullptr) {
        return -1;
    }

    std::ifstream in(path);
    if (!in) {
        return -1;
    }
    std::stringstream ss;
    ss << in.rdbuf();

    json j;
    try {
        j = json::parse(ss.str());
    } catch (const std::exception &) {
        return -1;
    }

    if (j.value("schema", std::string()) != TS_L2_SCHEMA) {
        return -1;
    }

    report->tensors.clear();
    report->n_flagged = 0;

    if (!j.contains("tensors") || !j["tensors"].is_array()) {
        return -1;
    }
    for (const auto & t : j["tensors"]) {
        ts_l2_tensor_result r;
        r.tensor_name = t.value("tensor", std::string());
        r.qtype       = t.value("qtype", std::string());

        r.rows = 0;
        r.cols = 0;
        if (t.contains("shape") && t["shape"].is_array() && t["shape"].size() == 2) {
            r.rows = t["shape"][0].get<int64_t>();
            r.cols = t["shape"][1].get<int64_t>();
        }

        const json & div = t["divergence"];
        r.divergence.max_abs            = div.value("max_abs", 0.0f);
        r.divergence.mean_abs           = div.value("mean_abs", 0.0f);
        r.divergence.relative_frobenius = div.value("relative_frobenius", 0.0f);
        r.divergence.per_layer_norm     = div.value("per_layer_norm", 0.0f);

        r.expected_frob  = t.value("expected_frob", 0.0f);
        r.flag_threshold = t.value("flag_threshold", 0.0f);
        r.flagged        = t.value("flagged", false);

        if (r.flagged) {
            report->n_flagged++;
        }
        report->tensors.push_back(std::move(r));
    }
    return 0;
}
