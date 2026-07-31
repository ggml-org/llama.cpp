//
// tessera-capability-eval.cpp
//
// Multi-axis capability eval: per-axis pass-fraction reduction plus the
// weighted-sum, Pareto, and guard lenses. See tessera-capability-eval.h
// and docs/self-improving-loop-design.md section 4.7.
//

#include "tessera-capability-eval.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>

using json = nlohmann::json;

static const int64_t TS_CAPABILITY_EVAL_SCHEMA_VERSION = 1;

double ts_capability_score_weighted_sum(const ts_capability_score * s, const double weights[5]) {
    if (!s || !weights) {
        return 0.0;
    }
    // weights[4] (general_competence) is the guard axis and is not summed.
    return s->mechanical     * weights[0]
         + s->api_currency   * weights[1]
         + s->hard_tail      * weights[2]
         + s->personal_style * weights[3];
}

// Track >= and set *strict on a strict >; bail on any <.
static bool ts_cap_ge_gt(double a, double b, bool * strict) {
    if (a < b) {
        return false;
    }
    if (a > b) {
        *strict = true;
    }
    return true;
}

bool ts_capability_score_dominates(const ts_capability_score * a, const ts_capability_score * b) {
    if (!a || !b) {
        return false;
    }
    bool strict = false;
    if (!ts_cap_ge_gt(a->mechanical,         b->mechanical,         &strict)) return false;
    if (!ts_cap_ge_gt(a->api_currency,       b->api_currency,       &strict)) return false;
    if (!ts_cap_ge_gt(a->hard_tail,          b->hard_tail,          &strict)) return false;
    if (!ts_cap_ge_gt(a->personal_style,     b->personal_style,     &strict)) return false;
    if (!ts_cap_ge_gt(a->general_competence, b->general_competence, &strict)) return false;
    return strict;
}

bool ts_capability_score_passes_guard(const ts_capability_score * s, const ts_capability_score * baseline, double epsilon) {
    if (!s) {
        return false;
    }
    if (!baseline) {
        return true;
    }
    return s->general_competence >= baseline->general_competence - epsilon;
}

// Reduce one axis object {"pass":N,"fail":M} to a pass fraction. An axis
// with zero instances scores 0.0.
static int ts_cap_axis_fraction(const json & axes, const char * name, double * out, std::string * err) {
    if (!axes.contains(name) || !axes.at(name).is_object()) {
        if (err) *err = std::string("missing axis: ") + name;
        return -1;
    }
    const json & a = axes.at(name);
    const int64_t pass = a.value("pass", int64_t(-1));
    const int64_t fail = a.value("fail", int64_t(-1));
    if (pass < 0 || fail < 0) {
        if (err) *err = std::string("axis '") + name + "' needs non-negative pass/fail counts";
        return -1;
    }
    const int64_t total = pass + fail;
    *out = total > 0 ? (double)pass / (double)total : 0.0;
    return 0;
}

// Read a baseline score vector (five fractions); fail loudly on a partial
// object rather than silently defaulting an axis.
static int ts_cap_baseline_from_json(const json & b, ts_capability_score * out, std::string * err) {
    const char * keys[5] = { "mechanical", "api_currency", "hard_tail", "personal_style", "general_competence" };
    double * dst[5] = {
        &out->mechanical, &out->api_currency, &out->hard_tail, &out->personal_style, &out->general_competence,
    };
    for (int i = 0; i < 5; i++) {
        if (!b.contains(keys[i]) || !b.at(keys[i]).is_number()) {
            if (err) *err = std::string("baseline missing axis: ") + keys[i];
            return -1;
        }
        *dst[i] = b.at(keys[i]).get<double>();
    }
    return 0;
}

int ts_capability_score_load(const char * path,
                             ts_capability_score * out,
                             ts_capability_score * baseline,
                             bool * has_baseline,
                             std::string * err_msg) {
    auto fail = [&](const std::string & msg) -> int {
        if (err_msg) *err_msg = msg;
        return -1;
    };

    if (!path || !out) {
        return fail("null path or out");
    }
    if (has_baseline) *has_baseline = false;

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return fail(std::string("failed to open: ") + path);
    }
    std::stringstream ss;
    ss << f.rdbuf();

    json j;
    try {
        j = json::parse(ss.str());
    } catch (const std::exception & e) {
        return fail(std::string("json parse error: ") + e.what());
    }

    const int64_t version = j.value("schema_version", int64_t(-1));
    if (version != TS_CAPABILITY_EVAL_SCHEMA_VERSION) {
        return fail("unsupported schema_version: " + std::to_string(version));
    }

    if (!j.contains("axes") || !j.at("axes").is_object()) {
        return fail("missing 'axes' object");
    }
    const json & axes = j.at("axes");

    try {
        if (ts_cap_axis_fraction(axes, "mechanical",         &out->mechanical,         err_msg) != 0) return -1;
        if (ts_cap_axis_fraction(axes, "api_currency",       &out->api_currency,       err_msg) != 0) return -1;
        if (ts_cap_axis_fraction(axes, "hard_tail",          &out->hard_tail,          err_msg) != 0) return -1;
        if (ts_cap_axis_fraction(axes, "personal_style",     &out->personal_style,     err_msg) != 0) return -1;
        if (ts_cap_axis_fraction(axes, "general_competence", &out->general_competence, err_msg) != 0) return -1;
    } catch (const std::exception & e) {
        return fail(std::string("malformed axes: ") + e.what());
    }

    if (baseline && j.contains("baseline") && j.at("baseline").is_object()) {
        if (ts_cap_baseline_from_json(j.at("baseline"), baseline, err_msg) != 0) {
            return -1;
        }
        if (has_baseline) *has_baseline = true;
    }

    return 0;
}
