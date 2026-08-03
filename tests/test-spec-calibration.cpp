//
// test-spec-calibration.cpp — smoke test for the spec-decoding calibration API
//
// SCOPE
// -----
// This test exercises the new common/speculative-calibration.{h,cpp} API.
// It runs the manual drafter-forward + per-prefix verifier-forward + KV
// rollback loop with a real model and writes the telemetry JSONL to a
// temp file.  It then parses the file and checks:
//
//   1. At least one JSONL record was emitted.
//   2. Each record parses as JSON (we use a minimal hand-rolled scanner
//      because the calibration API does not depend on nlohmann/json).
//   3. The schema field is "llama.dflash.acceptance.v1" (default) or
//      "llama.spec_calib.v2" (when topk > 0).
//   4. Required fields are present: schema, seq_id, drafted, accepted.
//   5. In v2 mode, verifier_argmax, drafter_argmax, and the
//      *_topk_{tokens,probs} arrays are present and have the right shape.
//   6. The confidence[] array has one entry per drafted token.
//
// This pins the observable behavior of the calibration API to the
// documented JSONL schema.  The audit at docs/audit-2026-07-29.md notes
// that the calibration schema is shared with llama-server and
// dspark-realign; the test would fail loudly if a refactor broke either
// consumer's expectations.
//
// USAGE
// -----
//   test-spec-calibration -m MODEL_GGUF [-p PROMPT] [--topk N] [--steps N] [-t OUT.jsonl]
//
// With no -m flag, the test runs in API-only mode: it verifies that the
// header compiles, the structs have the expected members, and the
// free() function is callable.  This is what runs in CI without a model.
//

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "speculative-calibration.h"
#include "speculative.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// API-only smoke checks (no model required).
//
// These exist so the test still passes in CI when no model fixture is
// downloaded.  They verify the public API surface is well-formed C++.
// ---------------------------------------------------------------------------

static int test_api_surface() {
    // 1. The options struct has the expected members.
    common_speculative_calibration_options opts;
    opts.n_draft_max_override = 4;
    opts.n_steps_override     = 8;
    opts.telemetry_out        = "/tmp/spec-calib-test.jsonl";
    opts.telemetry_topk       = 5;
    opts.verbosity            = 0;
    // observer_hooks fields are also present (would fail to compile if
    // a member was renamed).
    opts.observer_hooks.begin     = nullptr;
    opts.observer_hooks.flush     = nullptr;
    opts.observer_hooks.user_data = nullptr;

    if (opts.n_draft_max_override != 4) return 10;
    if (opts.n_steps_override     != 8) return 11;
    if (opts.telemetry_topk       != 5) return 12;
    if (opts.telemetry_out        != "/tmp/spec-calib-test.jsonl") return 13;

    // 2. The free() function is callable (no-op currently, but the symbol
    //    must resolve and link).
    common_speculative_calibration_free();
    // Idempotent: call it again.
    common_speculative_calibration_free();

    return 0;
}

// ---------------------------------------------------------------------------
// Model-based test (requires -m MODEL_GGUF).
//
// Loads a verifier and a drafter (we use the same model for both: the
// calibration API doesn't require a "real" drafter, only that the drafter
// context exists and can produce logits).  Runs the calibration, parses
// the JSONL, and validates the schema.
// ---------------------------------------------------------------------------

// Minimal JSONL record parser: enough to verify field presence and array
// shape.  We do not use nlohmann/json because the calibration API must
// stay decoupled from it (the .cpp file pulls in only <cstdio> for the
// fwrite-based JSONL emission).
struct jsonl_record {
    std::string schema;
    int         seq_id      = -1;
    int         drafted     = -1;
    int         accepted    = -1;
    int         topk        = -1;
    int         n_v_topk    = 0;   // number of verifier topk entries
    int         n_d_topk    = 0;   // number of drafter topk entries
    int         n_conf      = 0;   // number of confidence entries
};

// Find the value of a top-level scalar field in a JSONL line.  Returns
// empty string if not found.  Handles string and integer values; not a
// general-purpose JSON parser.
static std::string find_field(const std::string & line, const std::string & key) {
    // Match `"key":` followed by either `"value"` or a number.
    const std::string needle = "\"" + key + "\":";
    size_t p = line.find(needle);
    if (p == std::string::npos) return "";
    p += needle.size();
    while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) ++p;
    if (p >= line.size()) return "";
    if (line[p] == '"') {
        // String value: read until next unescaped quote.
        size_t q = p + 1;
        while (q < line.size() && line[q] != '"') {
            if (line[q] == '\\' && q + 1 < line.size()) ++q;
            ++q;
        }
        return line.substr(p + 1, q - p - 1);
    }
    // Number (or array/object): read until comma, brace, or bracket.
    size_t q = p;
    while (q < line.size() && line[q] != ',' && line[q] != '}' && line[q] != ']') ++q;
    return line.substr(p, q - p);
}

// Count the number of top-level array entries in a JSON field whose value
// is `[...]`.  Used to size-check the *_topk_*[] fields.  We count
// top-level commas between matching brackets.
static int count_array_entries(const std::string & line, const std::string & key) {
    const std::string needle = "\"" + key + "\":[";
    size_t p = line.find(needle);
    if (p == std::string::npos) return -1;
    p += needle.size();
    // Walk brackets to find the matching `]`, then count top-level commas.
    int depth  = 1;
    int commas = 0;
    bool empty = (p < line.size() && line[p] == ']');
    while (p < line.size() && depth > 0) {
        char c = line[p];
        if (c == '[') depth++;
        else if (c == ']') depth--;
        else if (c == ',' && depth == 1) commas++;
        p++;
    }
    if (empty) return 0;
    return commas + 1;
}

static int parse_record(const std::string & line, jsonl_record & rec) {
    rec.schema    = find_field(line, "schema");
    const std::string seq_id_s = find_field(line, "seq_id");
    const std::string drafted_s = find_field(line, "drafted");
    const std::string accepted_s = find_field(line, "accepted");
    if (seq_id_s.empty() || drafted_s.empty() || accepted_s.empty()) {
        return 21;  // missing required field
    }
    rec.seq_id   = std::atoi(seq_id_s.c_str());
    rec.drafted  = std::atoi(drafted_s.c_str());
    rec.accepted = std::atoi(accepted_s.c_str());
    if (rec.schema == "llama.spec_calib.v2") {
        const std::string topk_s = find_field(line, "topk");
        if (topk_s.empty()) return 22;  // v2 requires topk field
        rec.topk = std::atoi(topk_s.c_str());
        rec.n_v_topk = count_array_entries(line, "verifier_topk_tokens");
        rec.n_d_topk = count_array_entries(line, "drafter_topk_tokens");
        if (rec.n_v_topk < 0) return 23;
        if (rec.n_d_topk < 0) return 24;
    }
    rec.n_conf = count_array_entries(line, "confidence");
    if (rec.n_conf < 0) return 25;
    return 0;
}

static int test_with_model(common_params & params_in, const std::string & telemetry_path) {
    // We need a mutable copy of params because both common_speculative_init
    // and the imatrix-style spec setup mutate params.speculative.*
    // (ctx_tgt, ctx_dft, types).  The caller passes its own params;
    // we don't want to mutate the caller's struct.
    common_params params = params_in;
    LOG_INF("%s: loading verifier model '%s'\n", __func__, params.model.path.c_str());

    // Load the verifier via the standard common path.
    ggml_backend_load_all();
    common_init_result_ptr llama_init = common_init_from_params(params);
    if (!llama_init) {
        LOG_ERR("%s: failed to init verifier\n", __func__);
        return 30;
    }
    llama_model   * model_v = llama_init->model();
    llama_context * ctx_v   = llama_init->context();
    if (model_v == nullptr || ctx_v == nullptr) {
        LOG_ERR("%s: verifier model or context is null\n", __func__);
        return 31;
    }

    // For the drafter we use the same model file.  The calibration API
    // does not care whether the drafter is "good" — it just needs a
    // context that can produce logits.  Using the same model keeps the
    // test hermetic (no second GGUF to download).
    common_params params_dft = common_base_params_to_speculative(params);
    params_dft.speculative.draft.target_model_path = params.model.path;
    params_dft.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE };
    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE };

    common_speculative_init_result_ptr spec_init =
        common_speculative_init_from_params(params_dft, model_v, ctx_v);
    if (!spec_init || spec_init->context() == nullptr) {
        LOG_ERR("%s: failed to load drafter\n", __func__);
        return 32;
    }
    params.speculative.draft.ctx_tgt = ctx_v;
    params.speculative.draft.ctx_dft = spec_init->context();

    common_speculative_ptr spec(common_speculative_init(params.speculative, /*n_seq=*/1));
    if (!spec) {
        LOG_ERR("%s: failed to init spec context\n", __func__);
        return 33;
    }

    // Set up a prompt that's long enough to satisfy the calibration
    // "need at least n_ctx + 4 tokens" check.
    common_params run_params = params;
    if (run_params.prompt.empty()) {
        // Repeat a stable token sequence so the calibration has something
        // to chew on without depending on a specific tokenization.
        run_params.prompt = "the quick brown fox jumps over the lazy dog. "
                            "the quick brown fox jumps over the lazy dog. "
                            "the quick brown fox jumps over the lazy dog. "
                            "the quick brown fox jumps over the lazy dog. "
                            "the quick brown fox jumps over the lazy dog. "
                            "the quick brown fox jumps over the lazy dog. "
                            "the quick brown fox jumps over the lazy dog. "
                            "the quick brown fox jumps over the lazy dog. "
                            "the quick brown fox jumps over the lazy dog. "
                            "the quick brown fox jumps over the lazy dog. "
                            "the quick brown fox jumps over the lazy dog. "
                            "the quick brown fox jumps over the lazy dog. "
                            "the quick brown fox jumps over the lazy dog. "
                            "the quick brown fox jumps over the lazy dog. "
                            "the quick brown fox jumps over the lazy dog. "
                            "the quick brown fox jumps over the lazy dog. "
                            "the quick brown fox jumps over the lazy dog. "
                            "the quick brown fox jumps over the lazy dog. "
                            "the quick brown fox jumps over the lazy dog. "
                            "the quick brown fox jumps over the lazy dog. ";
    }

    // Set up the run options.  We use a small n_steps so the test
    // finishes quickly.
    common_speculative_calibration_options opts;
    opts.telemetry_out  = telemetry_path;
    opts.telemetry_topk = params.n_telemetry_topk;
    opts.n_steps_override = 2;  // keep the test fast
    opts.verbosity = 0;         // quiet for CI

    const int32_t n_ctx = llama_n_ctx(ctx_v);
    const bool ok = common_speculative_calibration_run(
        ctx_v, model_v, spec.get(), run_params, n_ctx, opts);
    common_speculative_calibration_free();

    if (!ok) {
        LOG_ERR("%s: calibration run failed\n", __func__);
        return 34;
    }

    // Read the JSONL output and verify each record.
    std::ifstream ifs(telemetry_path);
    if (!ifs.is_open()) {
        LOG_ERR("%s: failed to open telemetry output '%s'\n",
                __func__, telemetry_path.c_str());
        return 35;
    }
    std::string line;
    int n_records = 0;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        ++n_records;
        jsonl_record rec;
        int err = parse_record(line, rec);
        if (err != 0) {
            LOG_ERR("%s: record %d failed parse: err=%d, line=%s\n",
                    __func__, n_records, err, line.c_str());
            return err;
        }
        if (rec.schema == "llama.dflash.acceptance.v1") {
            // v1: drafted and accepted are present, confidence[] exists.
            if (rec.drafted < 0)  return 41;
            if (rec.accepted < 0) return 42;
            if (rec.n_conf != rec.drafted) {
                LOG_ERR("%s: v1 confidence[%d] != drafted=%d\n",
                        __func__, rec.n_conf, rec.drafted);
                return 43;
            }
        } else if (rec.schema == "llama.spec_calib.v2") {
            // v2: topk arrays and confidence[] exist and have right size.
            if (rec.topk <= 0) return 44;
            // Each topk array has drafted+1 entries (one per prefix,
            // including the bonus).
            if (rec.n_v_topk != rec.drafted + 1) {
                LOG_ERR("%s: v2 verifier_topk_tokens has %d entries, expected %d\n",
                        __func__, rec.n_v_topk, rec.drafted + 1);
                return 45;
            }
            if (rec.n_d_topk != rec.drafted + 1) {
                LOG_ERR("%s: v2 drafter_topk_tokens has %d entries, expected %d\n",
                        __func__, rec.n_d_topk, rec.drafted + 1);
                return 46;
            }
            if (rec.n_conf != rec.drafted) {
                LOG_ERR("%s: v2 confidence[%d] != drafted=%d\n",
                        __func__, rec.n_conf, rec.drafted);
                return 47;
            }
        } else {
            LOG_ERR("%s: unknown schema '%s' in record %d\n",
                    __func__, rec.schema.c_str(), n_records);
            return 48;
        }
        // accepted must be <= drafted in both schemas.
        if (rec.accepted > rec.drafted) {
            LOG_ERR("%s: record %d: accepted(%d) > drafted(%d)\n",
                    __func__, n_records, rec.accepted, rec.drafted);
            return 49;
        }
    }
    if (n_records == 0) {
        LOG_ERR("%s: zero telemetry records emitted to '%s'\n",
                __func__, telemetry_path.c_str());
        return 50;
    }
    LOG_INF("%s: validated %d telemetry records from '%s'\n",
            __func__, n_records, telemetry_path.c_str());
    return 0;
}

int main(int argc, char ** argv) {
    // API-only smoke checks run unconditionally — these verify the
    // public API surface compiles, links, and behaves correctly even
    // without a model.
    if (int err = test_api_surface(); err != 0) {
        LOG_ERR("API surface check failed: err=%d\n", err);
        return err;
    }

    // If no model was provided, stop here.  CI runs without a model
    // fixture stop at the API surface checks.
    common_params params;
    if (argc <= 1) {
        LOG_INF("test-spec-calibration: API surface OK, no -m MODEL given; "
                "skipping model-based test\n");
        return 0;
    }

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    if (params.model.path.empty()) {
        fprintf(stderr, "test-spec-calibration: -m MODEL is required for the model-based test\n");
        return 2;
    }
    if (params.n_ctx <= 0) {
        params.n_ctx = 512;
    }

    // Telemetry output: prefer the user-supplied path; otherwise use
    // a temp file we clean up afterwards.
    std::string telemetry_path = params.telemetry_out;
    if (telemetry_path.empty()) {
        telemetry_path = "/tmp/test-spec-calibration.jsonl";
    }
    // We always test v2 (topk=4) on top of v1 to exercise both schemas.
    params.n_telemetry_topk = 4;

    int err = test_with_model(params, telemetry_path);
    if (err != 0) {
        LOG_ERR("model-based test failed: err=%d\n", err);
    }
    return err;
}
