//
// test-tessera-runtime.cpp - smoke test for the runtime spec engine
// (common/tessera-runtime.{h,cpp})
//
// SCOPE
// -----
// Exercises the extern-C tessera_rt_* API with a real model:
//
//   1. tessera_rt_load() loads trunk + drafter (same fixture GGUF for
//      both, as in test-spec-calibration).
//   2. tessera_rt_generate() with telemetry_topk > 0 emits one
//      llama.tessera.spec.v1 record per spec step through on_trace.
//      Records must parse, carry "provenance":"runtime" and a shared
//      non-empty "sid", keep accepted <= drafted, and have the spec.v1
//      per-position array shapes.
//   3. tessera_rt_generate() with telemetry_topk == 0 emits no trace
//      callbacks at all (the cheap path).
//   4. Invalid arguments fail cleanly with a non-empty error string.
//
// USAGE
// -----
//   test-tessera-runtime -m MODEL_GGUF
//
// With no -m flag, only the API-surface checks (no model) run.
//

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "tessera-runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// API-only smoke checks (no model required).
// ---------------------------------------------------------------------------

static int test_api_surface() {
    // last_error is always callable and starts empty
    (void) tessera_rt_last_error();

    // free(NULL) is safe
    tessera_rt_free(nullptr);

    // invalid load arguments fail cleanly and set an error
    if (tessera_rt_load("", "", 512, 1, 0, 3) != nullptr) return 10;
    if (std::string(tessera_rt_last_error()).empty())     return 11;

    if (tessera_rt_load("/nonexistent-trunk.gguf", nullptr, 512, 1, 0, 3) != nullptr) return 12;
    if (tessera_rt_load("/nonexistent-trunk.gguf", "/nonexistent-draft.gguf", 0, 1, 0, 3) != nullptr) return 13;
    if (tessera_rt_load("/nonexistent-trunk.gguf", "/nonexistent-draft.gguf", 512, 1, 0, 0) != nullptr) return 14;

    // generate on a null handle fails cleanly
    if (tessera_rt_generate(nullptr, "hello", 8, 0, nullptr, nullptr, nullptr) != -1) return 15;
    if (std::string(tessera_rt_last_error()).empty()) return 16;

    return 0;
}

// ---------------------------------------------------------------------------
// Minimal JSONL field scanner (same approach as test-spec-calibration:
// the runtime API has no nlohmann/json dependency, so neither does the
// test).
// ---------------------------------------------------------------------------

static std::string find_field(const std::string & line, const std::string & key) {
    const std::string needle = "\"" + key + "\":";
    size_t p = line.find(needle);
    if (p == std::string::npos) return "";
    p += needle.size();
    while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) ++p;
    if (p >= line.size()) return "";
    if (line[p] == '"') {
        size_t q = p + 1;
        while (q < line.size() && line[q] != '"') {
            if (line[q] == '\\' && q + 1 < line.size()) ++q;
            ++q;
        }
        return line.substr(p + 1, q - p - 1);
    }
    size_t q = p;
    while (q < line.size() && line[q] != ',' && line[q] != '}' && line[q] != ']') ++q;
    return line.substr(p, q - p);
}

static int count_array_entries(const std::string & line, const std::string & key) {
    const std::string needle = "\"" + key + "\":[";
    size_t p = line.find(needle);
    if (p == std::string::npos) return -1;
    p += needle.size();
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

// ---------------------------------------------------------------------------
// Model-based test.
// ---------------------------------------------------------------------------

struct cb_state {
    int n_tokens = 0;
    int n_traces = 0;
    std::vector<std::string> lines;
};

static void on_token(const char * /*piece*/, int32_t /*token_id*/, void * ud) {
    static_cast<cb_state *>(ud)->n_tokens++;
}

static void on_trace(const char * jsonl_line, void * ud) {
    cb_state * st = static_cast<cb_state *>(ud);
    st->n_traces++;
    st->lines.push_back(jsonl_line);
}

static int test_with_model(const common_params & params) {
    const int32_t n_ctx       = params.n_ctx > 0 ? params.n_ctx : 512;
    const int32_t n_threads   = params.cpuparams.n_threads;
    const int32_t n_gpu       = params.n_gpu_layers;
    const int32_t draft_max   = 2;
    const int32_t topk        = 4;
    const int32_t max_tokens  = 40;

    LOG_INF("%s: loading trunk+drafter '%s' (n_ctx=%d, draft_max=%d)\n",
            __func__, params.model.path.c_str(), n_ctx, draft_max);

    tessera_rt * rt = tessera_rt_load(
            params.model.path.c_str(), params.model.path.c_str(),
            (uint32_t) n_ctx, n_threads, n_gpu, draft_max);
    if (rt == nullptr) {
        LOG_ERR("%s: tessera_rt_load failed: %s\n", __func__, tessera_rt_last_error());
        return 30;
    }

    const std::string prompt =
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog.";

    int err = 0;

    // run 1: capture on
    {
        cb_state st;
        const int32_t n_gen = tessera_rt_generate(
                rt, prompt.c_str(), max_tokens, topk, on_token, on_trace, &st);
        if (n_gen < 0) {
            LOG_ERR("%s: tessera_rt_generate failed: %s\n", __func__, tessera_rt_last_error());
            tessera_rt_free(rt);
            return 31;
        }
        if (n_gen == 0) {
            LOG_ERR("%s: zero tokens generated\n", __func__);
            tessera_rt_free(rt);
            return 32;
        }
        if (st.n_tokens != n_gen) {
            LOG_ERR("%s: on_token called %d times, generate returned %d\n",
                    __func__, st.n_tokens, n_gen);
            tessera_rt_free(rt);
            return 33;
        }
        if (st.lines.empty()) {
            LOG_ERR("%s: topk=%d but no trace records emitted\n", __func__, topk);
            tessera_rt_free(rt);
            return 34;
        }

        std::string sid_first;
        for (size_t r = 0; r < st.lines.size(); ++r) {
            const std::string & line = st.lines[r];

            if (find_field(line, "schema") != "llama.tessera.spec.v1") {
                LOG_ERR("%s: record %zu has wrong schema\n", __func__, r);
                err = 35; break;
            }

            const std::string prov = find_field(line, "provenance");
            if (prov != "runtime") {
                LOG_ERR("%s: record %zu provenance='%s', expected 'runtime'\n",
                        __func__, r, prov.c_str());
                err = 36; break;
            }

            const std::string sid = find_field(line, "sid");
            if (sid.empty()) {
                LOG_ERR("%s: record %zu has no sid\n", __func__, r);
                err = 37; break;
            }
            if (sid_first.empty()) {
                sid_first = sid;
            } else if (sid != sid_first) {
                LOG_ERR("%s: record %zu sid differs from first record\n", __func__, r);
                err = 38; break;
            }

            const int drafted  = std::atoi(find_field(line, "drafted").c_str());
            const int accepted = std::atoi(find_field(line, "accepted").c_str());
            if (drafted < 0 || accepted < 0) {
                LOG_ERR("%s: record %zu missing drafted/accepted\n", __func__, r);
                err = 39; break;
            }
            if (accepted > drafted) {
                LOG_ERR("%s: record %zu accepted(%d) > drafted(%d)\n",
                        __func__, r, accepted, drafted);
                err = 40; break;
            }

            // accepted_tokens = accepted drafts + bonus
            const int n_acc_tok = count_array_entries(line, "accepted_tokens");
            if (n_acc_tok != accepted + 1) {
                LOG_ERR("%s: record %zu accepted_tokens has %d entries, expected %d\n",
                        __func__, r, n_acc_tok, accepted + 1);
                err = 41; break;
            }

            const int n_conf = count_array_entries(line, "confidence");
            if (n_conf != drafted) {
                LOG_ERR("%s: record %zu confidence has %d entries, expected %d\n",
                        __func__, r, n_conf, drafted);
                err = 42; break;
            }

            // per-position top-k arrays: one entry per prefix incl. bonus
            const int n_v = count_array_entries(line, "verifier_topk_tokens");
            const int n_d = count_array_entries(line, "drafter_topk_tokens");
            if (n_v != drafted + 1) {
                LOG_ERR("%s: record %zu verifier_topk_tokens has %d entries, expected %d\n",
                        __func__, r, n_v, drafted + 1);
                err = 43; break;
            }
            if (n_d != drafted + 1) {
                LOG_ERR("%s: record %zu drafter_topk_tokens has %d entries, expected %d\n",
                        __func__, r, n_d, drafted + 1);
                err = 44; break;
            }
        }

        if (err == 0) {
            LOG_INF("%s: validated %zu runtime records (sid=%s, tokens=%d)\n",
                    __func__, st.lines.size(), sid_first.c_str(), n_gen);
        }
    }

    // run 2: capture off - zero trace callbacks, generation still works
    if (err == 0) {
        cb_state st;
        const int32_t n_gen = tessera_rt_generate(
                rt, prompt.c_str(), max_tokens, /*topk=*/0, on_token, on_trace, &st);
        if (n_gen < 0) {
            LOG_ERR("%s: topk=0 generate failed: %s\n", __func__, tessera_rt_last_error());
            err = 45;
        } else if (st.n_traces != 0) {
            LOG_ERR("%s: topk=0 emitted %d trace records, expected 0\n",
                    __func__, st.n_traces);
            err = 46;
        } else if (st.n_tokens != n_gen) {
            LOG_ERR("%s: topk=0 on_token count %d != returned %d\n",
                    __func__, st.n_tokens, n_gen);
            err = 47;
        } else {
            LOG_INF("%s: topk=0 cheap path OK (tokens=%d, traces=0)\n", __func__, n_gen);
        }
    }

    tessera_rt_free(rt);
    return err;
}

int main(int argc, char ** argv) {
    if (int err = test_api_surface(); err != 0) {
        LOG_ERR("API surface check failed: err=%d\n", err);
        return err;
    }

    if (argc <= 1) {
        LOG_INF("test-tessera-runtime: API surface OK, no -m MODEL given; "
                "skipping model-based test\n");
        return 0;
    }

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    if (params.model.path.empty()) {
        fprintf(stderr, "test-tessera-runtime: -m MODEL is required for the model-based test\n");
        return 2;
    }

    llama_backend_init();

    const int err = test_with_model(params);
    if (err != 0) {
        LOG_ERR("model-based test failed: err=%d\n", err);
    }

    llama_backend_free();
    return err;
}
