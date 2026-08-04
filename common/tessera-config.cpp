//
// tessera-config.cpp: INI-style parser for the --tessera-config file.
// See common/tessera-config.h for the public API and precedence rules.
//

#include "tessera-config.h"

#include "tessera-debug/tessera-debug.h"
#include "tessera-debug/tessera-matmul-output.h"

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// Strip ASCII whitespace from both ends of s.
static std::string strip(const std::string & s) {
    size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r')) b++;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r')) e--;
    return s.substr(b, e - b);
}

// Resolve a quoted value: the surrounding ' or " are stripped; embedded
// \" or \' become a literal quote. On success the result is appended to
// `out` and the function returns the index just past the closing quote.
// On a missing close quote, returns std::string::npos.
static size_t unquote_into(const std::string & raw, size_t i, std::string & out) {
    const char q = raw[i];
    i++;  // consume opening quote
    while (i < raw.size() && raw[i] != q) {
        if (raw[i] == '\\' && i + 1 < raw.size() &&
            (raw[i + 1] == q || raw[i + 1] == '\\')) {
            out.push_back(raw[i + 1]);
            i += 2;
        } else {
            out.push_back(raw[i]);
            i++;
        }
    }
    if (i >= raw.size()) {
        return std::string::npos;
    }
    return i + 1;  // consume closing quote
}

// Parse a single key = value (or key: value) line. Returns true on
// success and fills out_key + out_value. Tolerates surrounding whitespace;
// recognises bare, single-quoted, and double-quoted values.
static bool parse_kv_line(const std::string & line, std::string & out_key, std::string & out_value) {
    size_t i = 0;
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) i++;

    // key
    const size_t key_begin = i;
    while (i < line.size() && line[i] != '=' && line[i] != ':' &&
           line[i] != ' ' && line[i] != '\t') {
        if (line[i] == '#' || line[i] == ';') {
            return false;
        }
        i++;
    }
    out_key = strip(line.substr(key_begin, i - key_begin));
    if (out_key.empty()) return false;

    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) i++;
    if (i >= line.size() || (line[i] != '=' && line[i] != ':')) {
        return false;
    }
    i++;

    while (i < line.size() && (line[i] == ' ' || line[i] == '\t')) i++;
    if (i >= line.size()) {
        out_value.clear();
        return true;
    }

    if (line[i] == '"' || line[i] == '\'') {
        const size_t end = unquote_into(line, i, out_value);
        if (end == std::string::npos) return false;
        size_t j = end;
        while (j < line.size() && (line[j] == ' ' || line[j] == '\t')) j++;
        if (j != line.size()) return false;
        return true;
    }

    const size_t val_begin = i;
    while (i < line.size() && line[i] != '#' && line[i] != ';') i++;
    out_value = strip(line.substr(val_begin, i - val_begin));
    return true;
}

}  // namespace

bool tessera_config_parse(const std::string & text,
                          const std::string & source_label,
                          tessera_config & out,
                          std::string & err) {
    std::istringstream iss(text);
    std::string raw_line;

    std::string current_section;
    bool section_specified = false;
    int line_no = 0;

    while (std::getline(iss, raw_line)) {
        line_no++;
        if (!raw_line.empty() && raw_line.back() == '\r') raw_line.pop_back();

        const std::string trimmed = strip(raw_line);
        if (trimmed.empty()) continue;
        if (trimmed[0] == '#' || trimmed[0] == ';') continue;

        if (trimmed[0] == '[') {
            if (trimmed.back() != ']') {
                err = "tessera-config: " + source_label + ":" +
                      std::to_string(line_no) +
                      ": unclosed section header (expected ']' at end of line)\n";
                return false;
            }
            const std::string body = strip(trimmed.substr(1, trimmed.size() - 2));
            if (body.empty()) {
                err = "tessera-config: " + source_label + ":" +
                      std::to_string(line_no) +
                      ": empty section header\n";
                return false;
            }
            for (char c : body) {
                const bool ok = (c >= 'a' && c <= 'z') ||
                                (c >= 'A' && c <= 'Z') ||
                                (c >= '0' && c <= '9') ||
                                c == '-' || c == '_';
                if (!ok) {
                    err = "tessera-config: " + source_label + ":" +
                          std::to_string(line_no) +
                          ": invalid character in section name '" + body + "'\n";
                    return false;
                }
            }
            current_section = body;
            section_specified = true;
            continue;
        }

        std::string key, value;
        if (!parse_kv_line(trimmed, key, value)) {
            err = "tessera-config: " + source_label + ":" +
                  std::to_string(line_no) +
                  ": malformed line (expected 'key = value')\n";
            return false;
        }

        std::map<std::string, std::string> & bucket =
            section_specified ? out.sections[current_section] : out.global;

        if (bucket.find(key) != bucket.end()) {
            err = "tessera-config: " + source_label + ":" +
                  std::to_string(line_no) +
                  ": duplicate key '" + key + "'" +
                  (section_specified ? " in section [" + current_section + "]\n" : "\n");
            return false;
        }
        bucket[key] = value;
    }
    return true;
}

bool tessera_config_load(const std::string & path,
                         tessera_config & out,
                         std::string & err) {
    std::ifstream f(path);
    if (!f.is_open()) {
        err = "tessera-config: failed to open '" + path + "'\n";
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return tessera_config_parse(ss.str(), path, out, err);
}

namespace {

// All setters share the same signature: take (params, value, err) and
// return true on success. They mirror the validators used by the
// --tessera-* add_opt lambdas in common/arg.cpp so the config-file path
// rejects the same bad inputs the CLI would.
using cfg_setter = std::function<bool(common_tessera_params &, const std::string &, std::string &)>;

static bool set_str(std::string common_tessera_params::* field,
                    const std::string & v, std::string &,
                    const char *,
                    common_tessera_params & p) {
    (p.*field) = v;
    return true;
}

static bool set_int_min(int common_tessera_params::* field, int lo,
                        const std::string & v, std::string & err,
                        const char * key,
                        common_tessera_params & p) {
    try {
        size_t pos = 0;
        long n = std::stol(v, &pos);
        if (pos != v.size()) throw std::invalid_argument("trailing chars");
        if (n < lo) {
            err = std::string(key) + " must be >= " + std::to_string(lo) + ", got " + v;
            return false;
        }
        (p.*field) = (int) n;
        return true;
    } catch (...) {
        err = "invalid integer value for " + std::string(key) + ": '" + v + "'";
        return false;
    }
}

static bool set_uint64(uint64_t common_tessera_params::* field,
                       const std::string & v, std::string & err,
                       const char * key,
                       common_tessera_params & p) {
    try {
        size_t pos = 0;
        unsigned long long n = std::stoull(v, &pos);
        if (pos != v.size()) throw std::invalid_argument("trailing chars");
        (p.*field) = (uint64_t) n;
        return true;
    } catch (...) {
        err = "invalid unsigned integer value for " + std::string(key) + ": '" + v + "'";
        return false;
    }
}

static bool set_float_range(float common_tessera_params::* field,
                            float lo, float hi, bool hi_inclusive,
                            const std::string & v, std::string & err,
                            const char * key,
                            common_tessera_params & p) {
    try {
        size_t pos = 0;
        float f = std::stof(v, &pos);
        if (pos != v.size()) throw std::invalid_argument("trailing chars");
        if (f < lo || (hi_inclusive ? f > hi : f >= hi)) {
            err = std::string(key) + " must be in [" + std::to_string(lo) + ", " +
                  std::to_string(hi) + "], got " + v;
            return false;
        }
        (p.*field) = f;
        return true;
    } catch (...) {
        err = "invalid float value for " + std::string(key) + ": '" + v + "'";
        return false;
    }
}

static bool set_float_positive(float common_tessera_params::* field,
                               const std::string & v, std::string & err,
                               const char * key,
                               common_tessera_params & p) {
    try {
        size_t pos = 0;
        float f = std::stof(v, &pos);
        if (pos != v.size()) throw std::invalid_argument("trailing chars");
        if (f <= 0.0f) {
            err = std::string(key) + " must be > 0, got " + v;
            return false;
        }
        (p.*field) = f;
        return true;
    } catch (...) {
        err = "invalid float value for " + std::string(key) + ": '" + v + "'";
        return false;
    }
}

static bool set_float_nonneg(float common_tessera_params::* field,
                             const std::string & v, std::string & err,
                             const char * key,
                             common_tessera_params & p) {
    try {
        size_t pos = 0;
        float f = std::stof(v, &pos);
        if (pos != v.size()) throw std::invalid_argument("trailing chars");
        if (f < 0.0f) {
            err = std::string(key) + " must be >= 0, got " + v;
            return false;
        }
        (p.*field) = f;
        return true;
    } catch (...) {
        err = "invalid float value for " + std::string(key) + ": '" + v + "'";
        return false;
    }
}

static bool set_double_nonneg(double common_tessera_params::* field,
                              const std::string & v, std::string & err,
                              const char * key,
                              common_tessera_params & p) {
    try {
        size_t pos = 0;
        double d = std::stod(v, &pos);
        if (pos != v.size()) throw std::invalid_argument("trailing chars");
        if (d < 0.0) {
            err = std::string(key) + " must be >= 0, got " + v;
            return false;
        }
        (p.*field) = d;
        return true;
    } catch (...) {
        err = "invalid number value for " + std::string(key) + ": '" + v + "'";
        return false;
    }
}

static bool set_bool(bool common_tessera_params::* field,
                     const std::string & v, std::string & err,
                     const char * key,
                     common_tessera_params & p) {
    if (v == "true" || v == "1" || v == "on" || v == "yes" || v == "enabled") {
        (p.*field) = true;
        return true;
    }
    if (v == "false" || v == "0" || v == "off" || v == "no" || v == "disabled") {
        (p.*field) = false;
        return true;
    }
    err = "invalid boolean value for " + std::string(key) + ": '" + v +
          "' (expected true/false/1/0/on/off/yes/no/enabled/disabled)";
    return false;
}

// Wrapper macros reduce boilerplate and keep the per-key handlers
// visually aligned in the table below. Each macro takes a typed setter
// and binds it to a (params, value, err) signature.
#define SET_STR(FIELD)        ([](common_tessera_params & p, const std::string & v, std::string & err) -> bool { return set_str(&common_tessera_params::FIELD, v, err, #FIELD, p); })
#define SET_INT_MIN(FIELD, LO) ([](common_tessera_params & p, const std::string & v, std::string & err) -> bool { return set_int_min(&common_tessera_params::FIELD, (LO), v, err, #FIELD, p); })
#define SET_U64(FIELD)        ([](common_tessera_params & p, const std::string & v, std::string & err) -> bool { return set_uint64(&common_tessera_params::FIELD, v, err, #FIELD, p); })
#define SET_F_RANGE(FIELD, LO, HI, HI_INC) ([](common_tessera_params & p, const std::string & v, std::string & err) -> bool { return set_float_range(&common_tessera_params::FIELD, (LO), (HI), (HI_INC), v, err, #FIELD, p); })
#define SET_F_POS(FIELD)      ([](common_tessera_params & p, const std::string & v, std::string & err) -> bool { return set_float_positive(&common_tessera_params::FIELD, v, err, #FIELD, p); })
#define SET_F_NONNEG(FIELD)   ([](common_tessera_params & p, const std::string & v, std::string & err) -> bool { return set_float_nonneg(&common_tessera_params::FIELD, v, err, #FIELD, p); })
#define SET_D_NONNEG(FIELD)   ([](common_tessera_params & p, const std::string & v, std::string & err) -> bool { return set_double_nonneg(&common_tessera_params::FIELD, v, err, #FIELD, p); })
#define SET_BOOL(FIELD)       ([](common_tessera_params & p, const std::string & v, std::string & err) -> bool { return set_bool(&common_tessera_params::FIELD, v, err, #FIELD, p); })

struct key_handler {
    const char *  key;
    cfg_setter    fn;
};

static bool dispatch(const std::map<std::string, std::string> & bucket,
                     const std::string & section_name,
                     const std::vector<key_handler> & handlers,
                     common_tessera_params & p,
                     std::string & err) {
    for (const auto & kv : bucket) {
        const std::string & k = kv.first;
        const std::string & v = kv.second;
        bool found = false;
        for (const auto & h : handlers) {
            if (k == h.key) {
                found = true;
                if (!h.fn(p, v, err)) {
                    std::string prefix = section_name.empty()
                        ? "tessera-config: "
                        : "tessera-config: in [" + section_name + "]: ";
                    err = prefix + err + "\n";
                    return false;
                }
                break;
            }
        }
        if (!found) {
            err = "tessera-config: ";
            if (!section_name.empty()) err += "in [" + section_name + "]: ";
            err += "unknown key '" + k + "'\n";
            return false;
        }
    }
    return true;
}

// Per-section key tables. The CLI key name is the same as the config key
// (with --tessera- stripped); see examples/tessera-config.ini for the
// full mapping documented for users.
static const std::vector<key_handler> k_evolve = {
    {"evolve-iters",      SET_INT_MIN(evolve_iters, 1)},
    {"evolve-islands",    SET_INT_MIN(evolve_islands, 1)},
    {"evolve-population", SET_INT_MIN(evolve_population, 1)},
    {"evolve-seed",       SET_U64(evolve_seed)},
    {"evolve-only",       SET_BOOL(evolve_only)},
};
static const std::vector<key_handler> k_dpace = {
    {"dpace",       SET_STR(dpace_in)},
    {"dpace-out",   SET_STR(dpace_out)},
    {"dpace-alpha", SET_F_NONNEG(dpace_alpha)},
    {"dpace-gamma", SET_F_POS(dpace_gamma)},
};
static const std::vector<key_handler> k_l5 = {
    {"l5-generations",            SET_INT_MIN(l5_max_generations, 1)},
    {"l5-flag-multiplier",        SET_F_POS(l5_flag_multiplier)},
    {"l5-alpha-min",              SET_F_RANGE(l5_alpha_min, 0.0f, 1.0f, true)},
    {"l5-clip-min",               SET_F_RANGE(l5_clip_min, 0.0f, 1.0f, true)},
    {"l5-outlier-overshoot-scale",SET_F_NONNEG(l5_outlier_overshoot_scale)},
    {"l5-outlier-frac-cap",       SET_F_RANGE(l5_outlier_frac_cap, 0.0f, 1.0f, true)},
    {"l5-out",                    SET_STR(l5_out)},
};
static const std::vector<key_handler> k_awq = {
    {"awq-alpha", SET_STR(awq_alpha)},
    {"awq-clip",  SET_F_POS(awq_clip)},
};
// [runtime] covers --tessera-runtime-probe*, --tessera-l2-out, and the
// dequant/matmul-output sidecar paths. The sidecar keys write to the
// tessera_debug / tessera_matmul_output module globals (matching the
// behaviour of the corresponding add_opt handlers in arg.cpp). Built
// inline (instead of via the SET_* macros) so the sidecar keys can
// call the namespace functions directly.
static const std::vector<key_handler> k_runtime = {
    {"runtime-probe",         [](common_tessera_params & p, const std::string & v, std::string &) -> bool { p.runtime_probe = v; return true; }},
    {"runtime-probe-bf16",    [](common_tessera_params & p, const std::string & v, std::string &) -> bool { p.runtime_probe_bf16 = v; return true; }},
    {"l2-out",                [](common_tessera_params & p, const std::string & v, std::string &) -> bool { p.runtime_probe_l2_out = v; return true; }},
    {"dequant-dir",           [](common_tessera_params &, const std::string & v, std::string &) -> bool { tessera_debug::set_dequant_dir(v); return true; }},
    {"dequant-stride",        [](common_tessera_params &, const std::string & v, std::string & err) -> bool {
        try {
            size_t pos = 0;
            long n = std::stol(v, &pos);
            if (pos != v.size() || n < 1) throw std::invalid_argument("range");
            tessera_debug::set_dequant_stride((int64_t) n);
            return true;
        } catch (...) {
            err = "dequant-stride must be an integer >= 1, got '" + v + "'";
            return false;
        }
    }},
    {"matmul-output-dir",     [](common_tessera_params &, const std::string & v, std::string &) -> bool { tessera_matmul_output::set_matmul_output_dir(v); return true; }},
    {"matmul-output-stride",  [](common_tessera_params &, const std::string & v, std::string & err) -> bool {
        try {
            size_t pos = 0;
            long n = std::stol(v, &pos);
            if (pos != v.size() || n < 1) throw std::invalid_argument("range");
            tessera_matmul_output::set_matmul_output_stride((int64_t) n);
            return true;
        } catch (...) {
            err = "matmul-output-stride must be an integer >= 1, got '" + v + "'";
            return false;
        }
    }},
};
static const std::vector<key_handler> k_anonymize = {
    {"anonymize",      SET_STR(anonymize_in)},
    {"anonymize-out",  SET_STR(anonymize_out)},
    {"anonymize-level",[](common_tessera_params & p, const std::string & v, std::string & err) -> bool {
        if (v != "light" && v != "balanced" && v != "aggressive") {
            err = "anonymize-level must be light/balanced/aggressive, got '" + v + "'";
            return false;
        }
        p.anonymize_level = v;
        return true;
    }},
    {"anonymize-map",  SET_STR(anonymize_map)},
};
static const std::vector<key_handler> k_throughput = {
    {"throughput",     SET_STR(throughput_workload)},
    {"throughput-out", SET_STR(throughput_out)},
};
static const std::vector<key_handler> k_adapt = {
    {"adapt",         SET_STR(adapt_eval)},
    {"adapt-out",     SET_STR(adapt_out)},
    {"adapt-dry-run", SET_BOOL(adapt_dry_run)},
    {"adapt-epsilon", SET_D_NONNEG(adapt_epsilon)},
};
static const std::vector<key_handler> k_dataset = {
    {"dataset",      SET_STR(dataset_in)},
    {"dataset-out",  SET_STR(dataset_out)},
    {"dataset-mode", [](common_tessera_params & p, const std::string & v, std::string & err) -> bool {
        if (v != "text" && v != "pairs" && v != "lk" && v != "dflash") {
            err = "dataset-mode must be text/pairs/lk/dflash, got '" + v + "'";
            return false;
        }
        p.dataset_mode = v;
        return true;
    }},
};
static const std::vector<key_handler> k_policy = {
    {"policy",            SET_STR(policy)},
    {"policy-out",        SET_STR(policy_out)},
    {"outlier-frac",      SET_F_RANGE(outlier_frac, 0.0f, 1.0f, true)},
    {"range-selection", [](common_tessera_params & p, const std::string & v, std::string & err) -> bool {
        if (v != "legacy" && v != "imatrix-mse" && v != "septq") {
            err = "range-selection must be legacy/imatrix-mse/septq, got '" + v + "'";
            return false;
        }
        p.range_selection = v;
        return true;
    }},
    {"ternary-threshold", SET_STR(ternary_threshold)},
};
static const std::vector<key_handler> k_kernel_fitness = {
    {"kernel-fitness",       SET_BOOL(kernel_fitness)},
    {"kernel-fitness-blend", SET_F_RANGE(kernel_fitness_blend, 0.0f, 1.0f, true)},
    {"kernel-fitness-dir",   SET_STR(kernel_fitness_dir)},
};
static const std::vector<key_handler> k_capability = {
    {"capability-eval", SET_STR(capability_eval)},
    {"capability-out",  SET_STR(capability_out)},
};
static const std::vector<key_handler> k_acceptance = {
    {"acceptance",     SET_BOOL(acceptance)},
    {"acceptance-out", SET_STR(acceptance_out)},
};
static const std::vector<key_handler> k_w4a4 = {
    {"w4a4",                SET_BOOL(w4a4)},
    {"w4a4-outlier-thresh", SET_F_POS(w4a4_outlier_thresh)},
};
// [l15] routes to the L1.5 reference dtype module global.
static const std::vector<key_handler> k_l15 = {
    {"l15-dtype", [](common_tessera_params &, const std::string & v, std::string & err) -> bool {
        if (v != "f16" && v != "f32") {
            err = "l15-dtype must be f16/f32, got '" + v + "'";
            return false;
        }
        tessera_debug::set_l15_dtype(v);
        return true;
    }},
};
static const std::vector<key_handler> k_ga = {
    {"ga-checkpoint", SET_STR(ga_checkpoint)},
};
// [general] holds the umbrella / cross-cutting flags. The same key
// table also applies to entries in the global bucket (i.e. keys with
// no [section] header).
static const std::vector<key_handler> k_general = {
    {"mode", [](common_tessera_params & p, const std::string & v, std::string & err) -> bool {
        if (v != "off" && v != "default" && v != "calibrate-only" && v != "evolve-only") {
            err = "mode must be off/default/calibrate-only/evolve-only, got '" + v + "'";
            return false;
        }
        p.mode = v;
        return true;
    }},
    {"nthreads",            SET_INT_MIN(nthreads, 0)},
    {"imatrix",             SET_STR(imatrix)},
    {"champq",              SET_BOOL(champq)},
    {"calibrate-only",      SET_BOOL(calibrate_only)},
    {"adaptive-requantize", SET_BOOL(adaptive_requantize)},
    {"force-requantize",    SET_BOOL(force_requantize)},
};

struct section_handlers {
    const char * name;
    const std::vector<key_handler> * keys;
};

static const std::vector<section_handlers> section_table = {
    {"evolve",        &k_evolve},
    {"dpace",         &k_dpace},
    {"l5",            &k_l5},
    {"awq",           &k_awq},
    {"runtime",       &k_runtime},
    {"anonymize",     &k_anonymize},
    {"throughput",    &k_throughput},
    {"adapt",         &k_adapt},
    {"dataset",       &k_dataset},
    {"policy",        &k_policy},
    {"kernel-fitness",&k_kernel_fitness},
    {"capability",    &k_capability},
    {"acceptance",    &k_acceptance},
    {"w4a4",          &k_w4a4},
    {"l15",           &k_l15},
    {"ga",            &k_ga},
    {"general",       &k_general},
};

}  // namespace

bool tessera_config_apply(const tessera_config & cfg,
                          common_tessera_params & tessera_params,
                          std::string & err) {
    // The global bucket and [general] are equivalent: the same key
    // dispatch table applies. We walk both, so a user can place keys in
    // either place without surprise. If the same key is defined in both,
    // that's a hard error.
    std::map<std::string, std::string> merged = cfg.global;
    auto git = cfg.sections.find("general");
    if (git != cfg.sections.end()) {
        for (const auto & kv : git->second) {
            if (!merged.insert(kv).second) {
                err = "tessera-config: key '" + kv.first + "' is defined both globally and in [general]\n";
                return false;
            }
        }
    }
    if (!dispatch(merged, "", k_general, tessera_params, err)) return false;

    for (const auto & sec : cfg.sections) {
        if (sec.first == "general") continue;  // already merged above
        const section_handlers * sh = nullptr;
        for (const auto & t : section_table) {
            if (t.name == sec.first) { sh = &t; break; }
        }
        if (sh == nullptr) {
            err = "tessera-config: unknown section [" + sec.first + "]\n";
            return false;
        }
        if (!dispatch(sec.second, sec.first, *sh->keys, tessera_params, err)) return false;
    }
    return true;
}
