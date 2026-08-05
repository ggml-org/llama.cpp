//
// test-tessera-replay-parity.cpp - replay/topk-deepening parity for the
// runtime-traces pipeline (design doc section 14)
//
// SCOPE
// -----
// The curation stage replays promoted runtime sessions by pointing the
// EXISTING calibration loop (llama-imatrix's engine) at the decoded session
// corpus and requesting deepened topk. This test pins the invariants that
// replay relies on, end to end with a real model:
//
//   1. RUNTIME DETERMINISM: two tessera_rt_generate() runs over the same
//      prompt (CPU, single thread, greedy both models) emit byte-identical
//      trace records, so captured sessions are stable replay input.
//   2. TOPK DEEPENING: calibration over the decoded session corpus at
//      topk 64 reproduces a direct topk-64 calibration run on the same
//      corpus byte-for-byte (verifier side exact, per the spec; drafter
//      side included here because the test pins greedy draft selection,
//      making the draft trajectory reproducible).
//      Runtime records carry topk 16; replay records carry topk 64.
//   3. SCHEMA PARITY: every top-level field of a calibration record is
//      present in a runtime record. Runtime records are schema-identical
//      plus the additive "provenance" and "sid" fields.
//
// Acceptance COUNTS are deliberately not compared across the two engines.
// The accept SEMANTICS now match: both engines judge draft[i] with the
// verifier row conditioned on prefix + draft[0..i-1] (the calibration
// off-by-one that compared v_argmax[i] to draft[i-1] was fixed on
// fix/calibration-offbyone). The DRAFT policies still differ, though:
// calibration always forwards n_draft_max greedy drafts, while the
// runtime's draft-simple head applies its own sampler plus the p_min /
// n_min early-stop gates - so the two engines are not guaranteed to
// propose identical draft tokens, and count parity is only expected
// token-for-token when the draft trajectories coincide. Both sides are
// checked for internal consistency (accepted <= drafted) here.
//
// USAGE
// -----
//   test-tessera-replay-parity -m MODEL_GGUF
//
// With no -m flag the test is a no-op success, like the other model-gated
// tessera tests. Determinism requires CPU single-threaded runs, so the
// test forces n_gpu_layers=0 / n_threads=1 itself instead of trusting
// -ngl / -t (ctest passes only -m).
//

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "speculative-calibration.h"
#include "speculative.h"
#include "tessera-runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal JSONL scanners (same approach as test-tessera-runtime: no
// nlohmann/json dependency).
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

static std::vector<int> parse_int_array(const std::string & line, const std::string & key) {
    std::vector<int> out;
    const std::string needle = "\"" + key + "\":[";
    size_t p = line.find(needle);
    if (p == std::string::npos) return out;
    p += needle.size();
    while (p < line.size() && line[p] != ']') {
        while (p < line.size() && (line[p] == ' ' || line[p] == ',')) ++p;
        if (p >= line.size() || line[p] == ']') break;
        char * end = nullptr;
        const long v = std::strtol(line.c_str() + p, &end, 10);
        if (end == line.c_str() + p) break;
        out.push_back((int) v);
        p = (size_t) (end - line.c_str());
    }
    return out;
}

// Top-level JSON object keys: quoted strings at brace depth 1 (outside any
// array) that are immediately followed by ':'.
static std::set<std::string> top_level_keys(const std::string & line) {
    std::set<std::string> keys;
    int brace = 0;
    int bracket = 0;
    size_t p = 0;
    while (p < line.size()) {
        const char c = line[p];
        if (c == '[') { bracket++; p++; continue; }
        if (c == ']') { bracket--; p++; continue; }
        if (c == '"') {
            size_t q = p + 1;
            while (q < line.size() && line[q] != '"') {
                if (line[q] == '\\' && q + 1 < line.size()) ++q;
                ++q;
            }
            const std::string s = line.substr(p + 1, q - p - 1);
            size_t r = q + 1;
            while (r < line.size() && (line[r] == ' ' || line[r] == '\t')) ++r;
            if (brace == 1 && bracket == 0 && r < line.size() && line[r] == ':') {
                keys.insert(s);
            }
            p = q + 1;
            continue;
        }
        if (c == '{') brace++;
        else if (c == '}') brace--;
        p++;
    }
    return keys;
}

// ---------------------------------------------------------------------------
// Calibration driver: one fresh model + drafter + calibration run over a
// corpus, returning the emitted telemetry lines. Mirrors the setup in
// test-spec-calibration.cpp (verifier GGUF doubles as the drafter).
//
// The run is forced to CPU single-thread (n_gpu_layers=0, n_threads=1)
// regardless of CLI args: parity here means byte-for-byte, and Metal /
// multithreaded forwards are not byte-stable on this fixture.
// ---------------------------------------------------------------------------

static int run_calibration(const common_params & params_base,
                           int32_t n_ctx,
                           const std::string & corpus,
                           int topk,
                           const std::string & telemetry_path,
                           std::vector<std::string> & out_lines) {
    common_params params = params_base;
    params.n_ctx = n_ctx;
    params.prompt = corpus;
    params.n_gpu_layers = 0;
    params.cpuparams.n_threads       = 1;
    params.cpuparams_batch.n_threads = 1;
    params.speculative.draft.n_gpu_layers = 0; // keep the drafter on CPU too
    // Calibration picks drafts through common_sampler; the default chain
    // ends in dist sampling with an unpinned seed, which would make the
    // draft trajectory - and every captured distribution - differ run to
    // run. Pin greedy (top_k=1), the same policy the runtime engine uses.
    params.sampling.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
    params.sampling.top_k    = 1;
    params.sampling.seed     = 42;

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

    common_speculative_calibration_options opts;
    opts.telemetry_out  = telemetry_path;
    opts.telemetry_topk = topk;
    opts.verbosity      = 0;

    const bool ok = common_speculative_calibration_run(
        ctx_v, model_v, spec.get(), params, llama_n_ctx(ctx_v), opts);
    common_speculative_calibration_free();
    if (!ok) {
        LOG_ERR("%s: calibration run failed (topk=%d)\n", __func__, topk);
        return 34;
    }

    std::ifstream ifs(telemetry_path);
    if (!ifs.is_open()) {
        LOG_ERR("%s: failed to open telemetry output '%s'\n",
                __func__, telemetry_path.c_str());
        return 35;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty()) out_lines.push_back(line);
    }
    return 0;
}

// Replace the per-call session uuid so two runs can be compared without
// tripping over the one field that legitimately differs between calls.
static std::string normalize_sid(const std::string & line) {
    const std::string key = "\"sid\":\"";
    size_t p = line.find(key);
    if (p == std::string::npos) return line;
    size_t q = line.find('"', p + key.size());
    if (q == std::string::npos) return line;
    return line.substr(0, p + key.size()) + "<sid>" + line.substr(q);
}

// ---------------------------------------------------------------------------
// Model-based test.
// ---------------------------------------------------------------------------

struct cb_state {
    std::vector<std::string> lines;
};

static void on_token_noop(const char * /*piece*/, int32_t /*token_id*/, void * /*ud*/) {}

static void on_trace_collect(const char * jsonl_line, void * ud) {
    static_cast<cb_state *>(ud)->lines.push_back(jsonl_line);
}

static int test_with_model(const common_params & params) {
    const std::string & model_path = params.model.path;
    // Determinism gate: byte-for-byte parity requires CPU single-thread.
    // Force it here instead of trusting -t/-ngl so the test is hermetic
    // under ctest (which passes only -m).
    const int32_t n_threads = 1;
    const int32_t n_gpu     = 0;
    const int32_t draft_max = 2;
    const int32_t rt_topk   = 16;   // runtime capture depth (spec section 15)
    const int32_t rp_topk   = 64;   // replay deepening depth (spec section 12.2)
    const int32_t max_tokens = 200;

    const std::string prompt =
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog.";

    int err = 0;

    // -- Runtime capture ----------------------------------------------------

    tessera_rt * rt = tessera_rt_load(
            model_path.c_str(), model_path.c_str(),
            /*n_ctx=*/512, n_threads, n_gpu, draft_max);
    if (rt == nullptr) {
        LOG_ERR("%s: tessera_rt_load failed: %s\n", __func__, tessera_rt_last_error());
        return 40;
    }

    cb_state run_a;
    cb_state run_b;
    if (tessera_rt_generate(rt, prompt.c_str(), max_tokens, rt_topk,
                            on_token_noop, on_trace_collect, &run_a) < 0) {
        LOG_ERR("%s: runtime generate failed: %s\n", __func__, tessera_rt_last_error());
        tessera_rt_free(rt);
        return 41;
    }
    if (tessera_rt_generate(rt, prompt.c_str(), max_tokens, rt_topk,
                            on_token_noop, on_trace_collect, &run_b) < 0) {
        LOG_ERR("%s: runtime generate (2nd) failed: %s\n", __func__, tessera_rt_last_error());
        tessera_rt_free(rt);
        return 42;
    }
    tessera_rt_free(rt);

    if (run_a.lines.empty()) {
        LOG_ERR("%s: runtime capture emitted no records\n", __func__);
        return 43;
    }
    if (run_a.lines.size() != run_b.lines.size()) {
        LOG_ERR("%s: runtime capture is not deterministic: %zu vs %zu records\n",
                __func__, run_a.lines.size(), run_b.lines.size());
        return 44;
    }
    for (size_t i = 0; i < run_a.lines.size(); ++i) {
        // sid is a fresh uuid per generate() call; everything else must
        // be byte-identical across the two runs.
        if (normalize_sid(run_a.lines[i]) != normalize_sid(run_b.lines[i])) {
            LOG_ERR("%s: runtime capture diverged at record %zu\n", __func__, i);
            return 44;
        }
    }
    LOG_INF("%s: runtime capture deterministic across two runs (%zu records)\n",
            __func__, run_a.lines.size());

    // Runtime records: schema + provenance + per-record consistency.
    std::vector<llama_token> session_tokens;
    for (size_t r = 0; r < run_a.lines.size(); ++r) {
        const std::string & line = run_a.lines[r];
        if (find_field(line, "schema") != "llama.tessera.spec.v1") {
            LOG_ERR("%s: runtime record %zu has wrong schema\n", __func__, r);
            return 45;
        }
        if (find_field(line, "provenance") != "runtime") {
            LOG_ERR("%s: runtime record %zu missing provenance:runtime\n", __func__, r);
            return 46;
        }
        if (std::atoi(find_field(line, "topk").c_str()) != rt_topk) {
            LOG_ERR("%s: runtime record %zu topk != %d\n", __func__, r, rt_topk);
            return 47;
        }
        const int drafted  = std::atoi(find_field(line, "drafted").c_str());
        const int accepted = std::atoi(find_field(line, "accepted").c_str());
        if (accepted < 0 || accepted > drafted) {
            LOG_ERR("%s: runtime record %zu inconsistent accepted/drafted\n", __func__, r);
            return 48;
        }
        const std::vector<int> toks = parse_int_array(line, "accepted_tokens");
        if ((int) toks.size() != accepted + 1) {
            LOG_ERR("%s: runtime record %zu accepted_tokens shape\n", __func__, r);
            return 49;
        }
        for (int t : toks) session_tokens.push_back((llama_token) t);
    }

    // -- Decode the session to a corpus (the curation stage's decode step) --

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  // vocab-only use
    llama_model * vocab_model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (vocab_model == nullptr) {
        LOG_ERR("%s: failed to load model for vocab\n", __func__);
        return 50;
    }
    const llama_vocab * vocab = llama_model_get_vocab(vocab_model);

    std::string corpus;
    {
        std::vector<char> buf(session_tokens.size() * 8 + 16);
        int32_t n = llama_detokenize(vocab, session_tokens.data(),
                                     (int32_t) session_tokens.size(),
                                     buf.data(), (int32_t) buf.size(),
                                     /*remove_special=*/false, /*unparse_special=*/false);
        if (n < 0) {
            buf.resize((size_t) -n + 1);
            n = llama_detokenize(vocab, session_tokens.data(),
                                 (int32_t) session_tokens.size(),
                                 buf.data(), (int32_t) buf.size(),
                                 false, false);
        }
        if (n < 0) {
            LOG_ERR("%s: failed to detokenize the session\n", __func__);
            llama_model_free(vocab_model);
            return 51;
        }
        corpus.assign(buf.data(), (size_t) n);
    }

    // The calibration context is padded UP to a multiple of 256
    // (GGML_PAD in llama-context.cpp) and the loop needs at least
    // n_ctx + 4 tokens, so pin the replay context at the smallest pad
    // unit and grow the decoded corpus until it clears the floor.
    const int32_t n_ctx_calib = 256;

    // Count the corpus the SAME way the calibration loop tokenizes it
    // (add_special=true, parse_special=params.parse_special - default
    // false); the counts can differ by tens of tokens otherwise.
    auto count_corpus_tokens = [&]() -> int32_t {
        std::vector<llama_token> toks(corpus.size() + 16, 0);
        int32_t n = llama_tokenize(
            vocab, corpus.c_str(), (int32_t) corpus.size(),
            toks.data(), (int32_t) toks.size(),
            true, params.parse_special);
        if (n < 0) {
            toks.resize((size_t) -n);
            n = llama_tokenize(
                vocab, corpus.c_str(), (int32_t) corpus.size(),
                toks.data(), (int32_t) toks.size(),
                true, params.parse_special);
        }
        return n;
    };

    int32_t n_corpus = count_corpus_tokens();
    int n_repeats = 1;
    if (n_corpus >= 40) {
        // Repeat the session text until it clears the n_ctx + 4 floor.
        // The replay run and the direct run use the same final corpus,
        // so parity is preserved.
        const std::string corpus_seed = corpus;
        while (n_corpus < n_ctx_calib + 4 && n_repeats < 64) {
            corpus += "\n" + corpus_seed;
            n_corpus = count_corpus_tokens();
            ++n_repeats;
        }
    }
    llama_model_free(vocab_model);
    if (n_corpus <= 0) {
        LOG_ERR("%s: failed to tokenize the decoded corpus\n", __func__);
        return 52;
    }
    if (n_corpus < n_ctx_calib + 4) {
        LOG_ERR("%s: decoded corpus too small to replay: %d tokens, need %d\n",
                __func__, n_corpus, n_ctx_calib + 4);
        return 53;
    }
    LOG_INF("%s: replay corpus: %d tokens (%d repeat%s), calibration n_ctx=%d\n",
            __func__, n_corpus, n_repeats, n_repeats == 1 ? "" : "s", n_ctx_calib);

    // -- Topk deepening parity ----------------------------------------------

    std::vector<std::string> replay_lines;
    std::vector<std::string> direct_lines;
    if (int e = run_calibration(params, n_ctx_calib, corpus, rp_topk,
                                "/tmp/tessera-replay-parity-replay.jsonl",
                                replay_lines)) {
        return e;
    }
    if (int e = run_calibration(params, n_ctx_calib, corpus, rp_topk,
                                "/tmp/tessera-replay-parity-direct.jsonl",
                                direct_lines)) {
        return e;
    }
    if (replay_lines.empty()) {
        LOG_ERR("%s: replay calibration emitted no records\n", __func__);
        return 54;
    }
    if (replay_lines != direct_lines) {
        LOG_ERR("%s: replay at topk %d differs from a direct topk-%d run "
                "(%zu vs %zu records)\n",
                __func__, rp_topk, rp_topk,
                replay_lines.size(), direct_lines.size());
        for (size_t i = 0; i < replay_lines.size() && i < direct_lines.size(); ++i) {
            if (replay_lines[i] != direct_lines[i]) {
                LOG_ERR("%s: first divergence at record %zu\n", __func__, i);
                break;
            }
        }
        return 55;
    }
    for (size_t r = 0; r < replay_lines.size(); ++r) {
        if (std::atoi(find_field(replay_lines[r], "topk").c_str()) != rp_topk) {
            LOG_ERR("%s: replay record %zu topk != %d\n", __func__, r, rp_topk);
            return 56;
        }
        const int drafted  = std::atoi(find_field(replay_lines[r], "drafted").c_str());
        const int accepted = std::atoi(find_field(replay_lines[r], "accepted").c_str());
        if (accepted < 0 || accepted > drafted) {
            LOG_ERR("%s: replay record %zu inconsistent accepted/drafted\n", __func__, r);
            return 57;
        }
    }
    LOG_INF("%s: topk-%d replay reproduces the direct calibration run "
            "byte-for-byte (%zu records)\n", __func__, rp_topk, replay_lines.size());

    // -- Schema parity: runtime = calibration + provenance + sid ------------

    const std::set<std::string> keys_calib  = top_level_keys(replay_lines.front());
    const std::set<std::string> keys_runtime = top_level_keys(run_a.lines.front());
    if (keys_calib.empty() || keys_runtime.empty()) {
        LOG_ERR("%s: failed to scan top-level keys\n", __func__);
        return 58;
    }
    for (const std::string & k : keys_calib) {
        if (!keys_runtime.count(k)) {
            LOG_ERR("%s: runtime record is missing calibration field '%s'\n",
                    __func__, k.c_str());
            return 59;
        }
    }
    if (!keys_runtime.count("provenance") || !keys_runtime.count("sid")) {
        LOG_ERR("%s: runtime record lacks the additive provenance/sid fields\n", __func__);
        return 60;
    }
    if (keys_calib.count("provenance") || keys_calib.count("sid")) {
        LOG_ERR("%s: calibration record unexpectedly carries provenance/sid\n", __func__);
        return 61;
    }
    LOG_INF("%s: schema parity OK (%zu calibration fields, runtime adds provenance+sid)\n",
            __func__, keys_calib.size());

    return err;
}

int main(int argc, char ** argv) {
    if (argc <= 1) {
        LOG_INF("test-tessera-replay-parity: no -m MODEL given; skipping\n");
        return 0;
    }

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    if (params.model.path.empty()) {
        fprintf(stderr, "test-tessera-replay-parity: -m MODEL is required\n");
        return 2;
    }

    llama_backend_init();
    ggml_backend_load_all();

    const int err = test_with_model(params);
    if (err != 0) {
        LOG_ERR("model-based test failed: err=%d\n", err);
    }

    llama_backend_free();
    return err;
}
