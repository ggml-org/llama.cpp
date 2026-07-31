//
// tessera-adapt.cpp
//
// Adaptation engine v1: score the candidate eval, run the collapse
// guard, write a schema-versioned receipt. See tessera-adapt.h and
// docs/self-improving-loop-design.md sections 4.5 / 4.7.
//

#include "tessera-adapt.h"
#include "tessera-capability-eval.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <ctime>
#include <fstream>
#include <string>

using json = nlohmann::json;

static const char * TS_ADAPT_RECEIPT_SCHEMA = "llama.tessera.adapt.v1";

void ts_adapt_default_params(ts_adapt_params * params) {
    if (!params) return;
    params->dry_run               = false;
    params->input_eval_path[0]    = '\0';
    params->output_receipt_path[0] = '\0';
    params->guard_epsilon         = 0.02;  // learning.guardEpsilon default (design section 6)
}

static void ts_adapt_timestamp(char * buf, size_t n) {
    time_t t = time(NULL);
    struct tm tm_utc;
#if defined(_WIN32)
    gmtime_s(&tm_utc, &t);
#else
    gmtime_r(&t, &tm_utc);
#endif
    strftime(buf, n, "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
}

static json ts_adapt_score_json(const ts_capability_score * s) {
    json j;
    j["mechanical"]         = s->mechanical;
    j["api_currency"]       = s->api_currency;
    j["hard_tail"]          = s->hard_tail;
    j["personal_style"]     = s->personal_style;
    j["general_competence"] = s->general_competence;
    return j;
}

static int ts_adapt_write_receipt(const ts_adapt_params * params,
                                  const ts_capability_score * score,
                                  const ts_capability_score * baseline,
                                  bool has_baseline,
                                  bool guard_passed) {
    json j;
    j["schema"] = TS_ADAPT_RECEIPT_SCHEMA;

    char ts[64];
    ts_adapt_timestamp(ts, sizeof(ts));
    j["timestamp"] = ts;

    j["dry_run"]         = params->dry_run;
    j["guard_epsilon"]   = params->guard_epsilon;
    j["input_eval_path"] = params->input_eval_path;
    j["guard_passed"]    = guard_passed;
    j["adapted"]         = false;  // v1 writes the receipt only; no adapter is produced yet

    j["score"] = ts_adapt_score_json(score);

    j["has_baseline"] = has_baseline;
    if (has_baseline && baseline) {
        j["baseline"] = ts_adapt_score_json(baseline);
    } else {
        j["baseline"] = nullptr;
    }

    std::ofstream f(params->output_receipt_path, std::ios::binary);
    if (!f) {
        return -1;
    }
    f << j.dump(2) << "\n";
    return f.good() ? 0 : -1;
}

int ts_adapt_run(const ts_adapt_params * params) {
    if (!params) {
        return -1;
    }
    if (params->input_eval_path[0] == '\0' || params->output_receipt_path[0] == '\0') {
        return -1;
    }

    ts_capability_score score;
    ts_capability_score baseline;
    bool has_baseline = false;
    std::string err;
    if (ts_capability_score_load(params->input_eval_path, &score, &baseline, &has_baseline, &err) != 0) {
        fprintf(stderr, "tessera-adapt: %s\n", err.c_str());
        return -1;
    }

    const bool guard_passed = ts_capability_score_passes_guard(&score, has_baseline ? &baseline : NULL, params->guard_epsilon);

    if (ts_adapt_write_receipt(params, &score, has_baseline ? &baseline : NULL, has_baseline, guard_passed) != 0) {
        fprintf(stderr, "tessera-adapt: cannot write receipt: %s\n", params->output_receipt_path);
        return -1;
    }

    if (!guard_passed) {
        // collapse guard tripped: never adapt into a regression
        return 1;
    }

    // Priority-3 rejection-sampling LoRA harness (drafter co-adaptation)
    // plugs in here; v1 records intent and produces no adapter.
    return 0;
}
