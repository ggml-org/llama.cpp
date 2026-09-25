// llama-system-one-batch: answer many System One requests from one loaded model.
//
// A benchmark should measure the model and the readout, not an HTTP server's scheduler. This
// loads the weights once, then walks a JSONL of requests through exactly the product path --
// template -> segments -> ids -> decode -> answer -- and writes a JSONL of answers with the time
// each one took. Nothing here listens on a port, so nothing here can be measured against a stale
// server, contend for one, or be blocked by a route's own guards.
//
// Every readout is handled, chosen from the plan rather than from a flag:
//   one sequence, answers at slots          letter_slot, masked_slot
//   one sequence per question, scored       scored_slot   (embeddings, pooling none)
//   one sequence per option, pooled         rank_head     (pooling rank)
//
// usage: llama-system-one-batch <model.gguf> <requests.jsonl>
//        [--out FILE] [--template-file F] [--threads N] [--ngl N] [--labels A,B,C]
//
// A request line is the body /v1/systemone takes, plus an optional "id":
//   {"id": "s0.q3", "state": "...", "questions": {"k": {"type":"noul","instructions":"..."}}}

#include "system-one.h"

#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ordered_json, not json: nlohmann's object is a std::map, so plain `json` would sort the
// request's keys alphabetically -- silently reordering both the questions and a choice's
// criteria. The order of options is part of the question.
using json = nlohmann::ordered_json;
using clk  = std::chrono::steady_clock;

static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// Each request starts its positions at 0, so whatever the last one left in the cache has to go.
// Memory-less models (every encoder here) have none, which is why this is guarded rather than
// assumed.
static void reset_memory(llama_context * ctx) {
    if (auto * mem = llama_get_memory(ctx)) {
        llama_memory_clear(mem, true);
    }
}

static system_one_kind kind_of(const std::string & s) {
    if (s == "choice") return SYSTEM_ONE_KIND_CHOICE;
    if (s == "score")  return SYSTEM_ONE_KIND_SCORE;
    if (s == "noul")   return SYSTEM_ONE_KIND_NOUL;
    throw std::invalid_argument("unknown question type \"" + s + "\"");
}

// The wire format's `criteria`: an object for a choice (key -> description), an array for a
// score (the levels in order). noul's two options are the template's, not the caller's.
static void parse_question(const json & q, system_one_question & out) {
    out.kind = kind_of(q.at("type").get<std::string>());
    out.text = q.value("instructions", std::string());
    if (out.kind == SYSTEM_ONE_KIND_NOUL) {
        out.options = {"no", "yes"};
        return;
    }
    if (!q.contains("criteria")) {
        throw std::invalid_argument("a choice or score question needs \"criteria\"");
    }
    const json & c = q.at("criteria");
    if (c.is_object()) {
        for (const auto & it : c.items()) {
            out.options.push_back(it.key());
            out.descs.push_back(it.value().is_string() ? it.value().get<std::string>() : "");
        }
    } else {
        for (const auto & v : c) {
            out.options.push_back(v.get<std::string>());
            out.descs.push_back("");
        }
    }
}

// ---------------------------------------------------------------- decode paths

static std::vector<system_one_answer> decode_slots(llama_context * ctx, const system_one_plan & p) {
    const auto & t = p.sequences.front().tok;
    llama_batch b = llama_batch_init((int32_t) t.ids.size(), 0, 1);
    b.n_tokens = (int32_t) t.ids.size();
    for (size_t i = 0; i < t.ids.size(); i++) {
        b.token[i] = t.ids[i]; b.pos[i] = (llama_pos) i;
        b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 0;
    }
    for (int s : t.slots) { b.logits[s] = 1; }
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    if (rc != 0) { throw std::runtime_error("decode failed (" + std::to_string(rc) + ")"); }

    std::vector<const float *> rows;
    for (int s : t.slots) {
        const float * r = llama_get_logits_ith(ctx, s);
        if (!r) { throw std::runtime_error("no logits at an answer slot"); }
        rows.push_back(r);
    }
    return system_one_answers_from_logits(p, rows);
}

// scored_slot: one sequence per question, every option marked in it, and the model's own head
// turns each marked position into a single number. The scores go back question-major, which is
// the layout answers_from_scores already takes.
static std::vector<system_one_answer> decode_scored(llama_context * ctx, const system_one_plan & p) {
    std::vector<float> scores;
    for (const auto & seq : p.sequences) {
        const auto & t = seq.tok;
        reset_memory(ctx);
        llama_batch b = llama_batch_init((int32_t) t.ids.size(), 0, 1);
        b.n_tokens = (int32_t) t.ids.size();
        for (size_t i = 0; i < t.ids.size(); i++) {
            b.token[i] = t.ids[i]; b.pos[i] = (llama_pos) i;
            b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1;
        }
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc != 0) { throw std::runtime_error("decode failed (" + std::to_string(rc) + ")"); }
        for (int s : t.slots) {
            const float * e = llama_get_embeddings_ith(ctx, s);
            if (!e) { throw std::runtime_error("no score at a marked position"); }
            scores.push_back(e[0]);
        }
    }
    return system_one_answers_from_scores(p, scores);
}

// rank_head: one sequence per option, each pooled to one number by the classification head.
static std::vector<system_one_answer> decode_ranked(llama_context * ctx, const system_one_plan & p) {
    const uint32_t n_ubatch  = llama_n_ubatch(ctx);
    const uint32_t n_seq_max = llama_n_seq_max(ctx);
    std::vector<float> scores(p.sequences.size(), 0.0f);

    for (size_t i = 0; i < p.sequences.size(); ) {
        size_t n_seq = 0, n_tok = 0;
        while (i + n_seq < p.sequences.size() && n_seq < n_seq_max) {
            const size_t len = p.sequences[i + n_seq].tok.ids.size();
            if (len > n_ubatch) {
                throw std::runtime_error("one option sequence is longer than the micro-batch -- raise -ub");
            }
            if (n_tok + len > n_ubatch) break;
            n_tok += len; n_seq++;
        }
        reset_memory(ctx);

        llama_batch b = llama_batch_init((int32_t) n_tok, 0, (int32_t) n_seq);
        b.n_tokens = 0;
        for (size_t k = 0; k < n_seq; k++) {
            const auto & ids = p.sequences[i + k].tok.ids;
            for (size_t j = 0; j < ids.size(); j++) {
                const int32_t at = b.n_tokens++;
                b.token[at] = ids[j]; b.pos[at] = (llama_pos) j;
                b.n_seq_id[at] = 1; b.seq_id[at][0] = (llama_seq_id) k; b.logits[at] = 1;
            }
        }
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc != 0) { throw std::runtime_error("decode failed (" + std::to_string(rc) + ")"); }
        for (size_t k = 0; k < n_seq; k++) {
            const float * pooled = llama_get_embeddings_seq(ctx, (llama_seq_id) k);
            if (!pooled) { throw std::runtime_error("the model returned no pooled score"); }
            scores[i + k] = pooled[0];
        }
        i += n_seq;
    }
    return system_one_answers_from_scores(p, scores);
}

// ---------------------------------------------------------------- main

static int run(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <requests.jsonl> [--out FILE] [--template-file F] "
                        "[--threads N] [--ngl N] [--labels A,B,C]\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1], req_path = argv[2];
    std::string out_path, tmpl_path, labels_csv;
    int nthreads = std::max(1u, std::thread::hardware_concurrency() / 2), ngl = 0;
    for (int i = 3; i + 1 < argc; i++) {
        if      (strcmp(argv[i], "--out")           == 0) out_path   = argv[++i];
        else if (strcmp(argv[i], "--template-file") == 0) tmpl_path  = argv[++i];
        else if (strcmp(argv[i], "--threads")       == 0) nthreads   = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ngl")           == 0) ngl        = atoi(argv[++i]);
        else if (strcmp(argv[i], "--labels")        == 0) labels_csv = argv[++i];
    }

    std::vector<json> reqs;
    {
        std::ifstream f(req_path);
        if (!f) { fprintf(stderr, "cannot open %s\n", req_path.c_str()); return 1; }
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty()) { reqs.push_back(json::parse(line)); }
        }
    }
    if (reqs.empty()) { fprintf(stderr, "%s holds no requests\n", req_path.c_str()); return 1; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", model_path.c_str()); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    system_one_params cfg = system_one_params_from_model(model);
    if (!tmpl_path.empty()) {
        std::ifstream f(tmpl_path);
        if (!f) { fprintf(stderr, "cannot open %s\n", tmpl_path.c_str()); return 1; }
        std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        while (!s.empty() && s.back() == '\n') { s.pop_back(); }
        cfg.template_src = s;
    }
    if (!labels_csv.empty()) {
        cfg.labels.clear();
        std::stringstream ss(labels_csv);
        for (std::string t; std::getline(ss, t, ','); ) { cfg.labels.push_back(t); }
        cfg.labels_are_default = false;
    }

    // Every plan is built before the context, because the context has to be sized for the
    // largest of them -- a bidirectional sequence cannot be split across a micro-batch.
    // the id is echoed back exactly as it came in -- a number stays a number, so a caller
    // can index by it rather than by string
    struct item { json id; system_one_plan plan; std::vector<std::string> keys;
                  std::vector<system_one_question> qs; };
    std::vector<item> items;
    items.reserve(reqs.size());
    size_t max_tok = 0, max_seq = 1;
    for (size_t i = 0; i < reqs.size(); i++) {
        item it;
        it.id = reqs[i].contains("id") ? reqs[i]["id"] : json((int64_t) i);
        const json & st = reqs[i].at("state");
        const std::string state = st.is_string() ? st.get<std::string>() : st.dump();
        for (const auto & e : reqs[i].at("questions").items()) {
            system_one_question q;
            parse_question(e.value(), q);
            it.keys.push_back(e.key());
            it.qs.push_back(std::move(q));
        }
        it.plan = system_one_build_plan(cfg, vocab, state, it.qs);
        system_one_tokenize_plan(vocab, cfg, it.plan);
        size_t tok = 0;
        for (const auto & s : it.plan.sequences) { tok = std::max(tok, s.tok.ids.size()); }
        max_tok = std::max(max_tok, tok);
        max_seq = std::max(max_seq, it.plan.sequences.size());
        items.push_back(std::move(it));
    }

    const system_one_context_needs need = system_one_required_context(cfg);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t) std::max<size_t>(max_tok + 64, 512);
    cp.n_batch         = cp.n_ctx;
    cp.n_ubatch        = cp.n_ctx;
    cp.n_seq_max       = (uint32_t) std::min<size_t>(max_seq, 32);
    cp.n_threads       = nthreads;
    cp.n_threads_batch = nthreads;
    cp.embeddings      = need.embeddings;
    cp.pooling_type    = need.pooling_type;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create a context\n"); return 1; }

    std::ofstream out;
    if (!out_path.empty()) { out.open(out_path); }
    std::ostream & os = out_path.empty() ? std::cout : out;

    fprintf(stderr, "%zu requests, ctx %u, n_seq_max %u, readout %s, threads %d, ngl %d\n",
            items.size(), cp.n_ctx, cp.n_seq_max, system_one_readout_name(cfg.readout), nthreads, ngl);

    for (auto & it : items) {
        reset_memory(ctx);
        const auto t0 = clk::now();
        std::vector<system_one_answer> answers;
        try {
            answers = it.plan.rank_pooling ? decode_ranked(ctx, it.plan)
                    : it.plan.scored_slots ? decode_scored(ctx, it.plan)
                                           : decode_slots(ctx, it.plan);
        } catch (const std::exception & e) {
            os << json{{"id", it.id}, {"error", e.what()}}.dump() << "\n";
            continue;
        }
        const double ms = ms_since(t0);

        json ja;
        int32_t n_tokens = 0;
        for (const auto & s : it.plan.sequences) { n_tokens += (int32_t) s.tok.ids.size(); }
        for (size_t qi = 0; qi < it.qs.size(); qi++) {
            const auto & a = answers[qi];
            json one;
            switch (it.qs[qi].kind) {
                case SYSTEM_ONE_KIND_NOUL:
                    one["noul"] = a.probs.size() > 1 ? a.probs[1] : 0.0f;
                    break;
                case SYSTEM_ONE_KIND_CHOICE: {
                    one["choice"] = it.qs[qi].options[a.choice];
                    json pr;
                    for (size_t o = 0; o < it.qs[qi].options.size(); o++) {
                        pr[it.qs[qi].options[o]] = a.probs[o];
                    }
                    one["probabilities"] = pr;
                    break;
                }
                case SYSTEM_ONE_KIND_SCORE:
                    one["score"] = system_one_score_expectation(a);
                    one["probabilities"] = a.probs;
                    break;
            }
            one["confidence"] = a.confidence;
            one["logits"]     = a.logits;
            ja[it.keys[qi]]   = one;
        }
        os << json{{"id", it.id}, {"answers", ja}, {"ms", ms},
                   {"n_tokens", n_tokens}, {"n_sequences", it.plan.sequences.size()}}.dump() << "\n";
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
