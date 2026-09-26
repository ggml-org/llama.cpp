// llama-system-one -- ask a System One model typed questions and print the answers.
//
// A System One model does not write text. Give it a state and a set of typed questions and
// it returns, in one forward pass, a distribution over each question's declared options:
//
//   llama-system-one -m model.gguf --state "I was charged twice. Please refund the duplicate."
//       --noul   "refund:The customer is asking for a refund."
//       --choice "queue:Which team should handle this?:billing=refunds,technical=a fault"
//       --score  "urgency:How urgent is this?:routine,soon,urgent,critical"
//
// --system-one-request takes the same request as JSON, which is how a prompt is kept identical
// between this and the server.

#include "system-one.h"

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

// How many option sequences to put in one decode.
static constexpr size_t MAX_BATCHED_SEQS = 32;

static void print_usage(int, char ** argv) {
    LOG("\nexample:\n");
    LOG("  %s -m model.gguf --state \"...\" --noul \"ok:Is the order complete?\"\n", argv[0]);
    LOG("  %s -m model.gguf --system-one-request request.json --json\n", argv[0]);
    LOG("\n");
}

static bool read_file(const std::string & path, std::string & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

// "key:instructions:rest" -- the instructions may not contain a colon, which is why the
// JSON body exists for anything more elaborate than a one-off from the shell.
static bool split_spec(const std::string & spec, std::string & key, std::string & instr, std::string & rest) {
    const auto a = spec.find(':');
    if (a == std::string::npos) return false;
    key = spec.substr(0, a);
    const auto b = spec.find(':', a + 1);
    if (b == std::string::npos) {
        instr = spec.substr(a + 1);
        rest.clear();
    } else {
        instr = spec.substr(a + 1, b - a - 1);
        rest  = spec.substr(b + 1);
    }
    return !key.empty();
}

static void parse_options(const std::string & rest, bool with_descs,
                          std::vector<std::string> & options, std::vector<std::string> & descs) {
    for (auto & piece : string_split(rest, ",")) {
        if (piece.empty()) continue;
        const auto eq = with_descs ? piece.find('=') : std::string::npos;
        if (eq == std::string::npos) {
            options.push_back(piece);
            descs.push_back("");
        } else {
            options.push_back(piece.substr(0, eq));
            descs.push_back(piece.substr(eq + 1));
        }
    }
}

// The flags and the JSON body describe the same thing; both land here.
static bool collect_questions(const common_params & params,
                              std::vector<std::string> & keys,
                              std::vector<system_one_question> & qs,
                              std::string & state,
                              std::vector<std::string> & cfg_labels,
                              float & temperature,
                              std::string & err) {
    if (!params.so_request_file.empty()) {
        std::string body;
        if (!read_file(params.so_request_file, body)) { err = "cannot read " + params.so_request_file; return false; }
        json req;
        try {
            req = json::parse(body);
        } catch (const std::exception & e) {
            err = std::string("invalid request JSON: ") + e.what();
            return false;
        }
        temperature = req.value("temperature", 1.0f);
        if (req.contains("labels")) {
            try {
                cfg_labels = req.at("labels").get<std::vector<std::string>>();
            } catch (const std::exception &) {
                err = "\"labels\" must be an array of strings";
                return false;
            }
        }
        if (req.contains("state")) {
            const json & st = req.at("state");
            if (st.is_array()) {
                // reserved for content parts carrying media
                err = "\"state\" as a list of content parts is not supported yet -- pass a string or an object";
                return false;
            }
            state = st.is_string() ? st.get<std::string>() : st.dump();
        }
        if (!req.contains("questions") || !req.at("questions").is_object()) {
            err = "request has no \"questions\" object";
            return false;
        }
        for (const auto & item : req.at("questions").items()) {
            const auto & q = item.value();
            if (q.contains("temperature")) {
                err = "question \"" + item.key() + "\": \"temperature\" is one value for the whole request, not per question -- move it to the top level";
                return false;
            }

            const std::string type = q.value("type", "noul");
            system_one_question out;
            out.text = q.value("instructions", "");
            if (type == "noul") {
                out.kind = SYSTEM_ONE_KIND_NOUL;
            } else if (type == "choice") {
                out.kind = SYSTEM_ONE_KIND_CHOICE;
                if (!q.contains("criteria") || !q.at("criteria").is_object()) {
                    err = "choice question \"" + item.key() + "\" needs a \"criteria\" object";
                    return false;
                }
                for (const auto & c : q.at("criteria").items()) {
                    out.options.push_back(c.key());
                    out.descs.push_back(c.value().is_string() ? c.value().get<std::string>() : std::string());
                }
            } else if (type == "score") {
                out.kind = SYSTEM_ONE_KIND_SCORE;
                if (!q.contains("criteria") || !q.at("criteria").is_array()) {
                    err = "score question \"" + item.key() + "\" needs a \"criteria\" array of levels";
                    return false;
                }
                out.options = q.at("criteria").get<std::vector<std::string>>();
            } else {
                err = "unknown question type \"" + type + "\" for \"" + item.key() + "\"";
                return false;
            }
            keys.push_back(item.key());
            qs.push_back(std::move(out));
        }
        return true;
    }

    for (const auto & [kind, spec] : params.so_questions) {
        const std::string form =
            kind == "noul"   ? "--noul wants KEY:INSTRUCTIONS"
          : kind == "choice" ? "--choice wants KEY:INSTRUCTIONS:opt[=desc][,opt[=desc]...]"
                             : "--score wants KEY:INSTRUCTIONS:level[,level...]";

        std::string key, instr, rest;
        if (!split_spec(spec, key, instr, rest) || (rest.empty() && kind != "noul")) {
            err = form;
            return false;
        }

        system_one_question q;
        q.text = instr;
        if (kind == "noul") {
            q.kind = SYSTEM_ONE_KIND_NOUL;
        } else if (kind == "choice") {
            q.kind = SYSTEM_ONE_KIND_CHOICE;
            parse_options(rest, true, q.options, q.descs);
        } else {
            q.kind = SYSTEM_ONE_KIND_SCORE;
            parse_options(rest, false, q.options, q.descs);
        }

        keys.push_back(key);
        qs.push_back(std::move(q));
    }
    return true;
}

// The slot readouts: one decode over the whole sequence, logits asked for at the answer slots.
static std::vector<system_one_answer> decode_slots(llama_context * ctx, const system_one_plan & p) {
    const auto & t = p.sequences.front().tok;

    llama_batch batch = llama_batch_init((int32_t) t.ids.size(), 0, 1);
    batch.n_tokens = (int32_t) t.ids.size();
    for (size_t i = 0; i < t.ids.size(); i++) {
        batch.token[i]     = t.ids[i];
        batch.pos[i]       = (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 0;
    }
    for (int slot : t.slots) {
        if (slot < 0 || slot >= (int) t.ids.size()) {
            llama_batch_free(batch);
            throw std::runtime_error("answer slot outside the sequence");
        }
        batch.logits[slot] = 1;
    }

    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        throw std::runtime_error("llama_decode failed with " + std::to_string(rc));
    }

    std::vector<const float *> rows(t.slots.size(), nullptr);
    for (size_t qi = 0; qi < t.slots.size(); qi++) {
        rows[qi] = llama_get_logits_ith(ctx, t.slots[qi]);
        if (rows[qi] == nullptr) {
            throw std::runtime_error("no logits at answer slot " + std::to_string(qi));
        }
    }
    return system_one_answers_from_logits(p, rows);
}

// rank_head: one sequence per option, scored by the model's head. They are independent, so a
// chunk of them shares a decode -- bounded by seq_id count, and by the ubatch, since a
// bidirectional sequence cannot be split across one.
//
// Where the plan says the sequences share a prefix, it is decoded once and its KV copied, so
// the state costs one pass instead of K.
static std::vector<system_one_answer> decode_ranked(llama_context * ctx, const system_one_plan & p) {
    const uint32_t n_ubatch  = llama_n_ubatch(ctx);
    const uint32_t n_seq_max = llama_n_seq_max(ctx);

    std::vector<float> scores(p.sequences.size(), 0.0f);

    for (size_t i = 0; i < p.sequences.size(); ) {
        size_t n_seq = 0;
        size_t n_tok = 0;
        while (i + n_seq < p.sequences.size() && n_seq < n_seq_max) {
            const size_t len = p.sequences[i + n_seq].tok.ids.size();
            if (len > n_ubatch) {
                throw std::runtime_error("one option sequence is " + std::to_string(len) +
                      " tokens, more than the micro-batch (" + std::to_string(n_ubatch) + ") -- raise -ub");
            }
            if (n_tok + len > n_ubatch) break;
            n_tok += len;
            n_seq++;
        }

        llama_memory_clear(llama_get_memory(ctx), true);

        // the prefix every sequence in this chunk agrees on, if sharing is allowed at all
        size_t n_prefix = 0;
        if (n_seq > 1) {
            n_prefix = p.sequences[i + 1].n_shared;
            for (size_t k = 2; k < n_seq; k++) {
                n_prefix = std::min(n_prefix, p.sequences[i + k].n_shared);
            }
            // only worth a decode of its own if it is most of the work
            if (n_prefix * 2 < p.sequences[i].tok.ids.size()) {
                n_prefix = 0;
            }
        }

        if (n_prefix > 0) {
            const auto & ids = p.sequences[i].tok.ids;
            llama_batch pre = llama_batch_init((int32_t) n_prefix, 0, 1);
            pre.n_tokens = (int32_t) n_prefix;
            for (size_t j = 0; j < n_prefix; j++) {
                pre.token[j]     = ids[j];
                pre.pos[j]       = (llama_pos) j;
                pre.n_seq_id[j]  = 1;
                pre.seq_id[j][0] = 0;
                pre.logits[j]    = 0;
            }
            const int rc = llama_decode(ctx, pre);
            llama_batch_free(pre);
            if (rc != 0) {
                throw std::runtime_error("llama_decode failed on the shared prefix");
            }
            for (size_t k = 1; k < n_seq; k++) {
                llama_memory_seq_cp(llama_get_memory(ctx), 0, (llama_seq_id) k, 0, (llama_pos) n_prefix);
            }
        }

        llama_batch batch = llama_batch_init((int32_t) n_tok, 0, (int32_t) n_seq);
        batch.n_tokens = 0;
        for (size_t k = 0; k < n_seq; k++) {
            const auto & ids = p.sequences[i + k].tok.ids;
            for (size_t j = n_prefix; j < ids.size(); j++) {
                const int32_t at = batch.n_tokens++;
                batch.token[at]     = ids[j];
                batch.pos[at]       = (llama_pos) j;
                batch.n_seq_id[at]  = 1;
                batch.seq_id[at][0] = (llama_seq_id) k;
                batch.logits[at]    = 1;
            }
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            throw std::runtime_error("llama_decode failed on the batch starting at sequence " + std::to_string(i));
        }

        for (size_t k = 0; k < n_seq; k++) {
            const float * pooled = llama_get_embeddings_seq(ctx, (llama_seq_id) k);
            if (pooled == nullptr) {
                throw std::runtime_error("the model returned no pooled score -- is it a classification head?");
            }
            scores[i + k] = pooled[0];
        }

        i += n_seq;
    }

    return system_one_answers_from_scores(p, scores);
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SYSTEM_ONE, print_usage)) {
        return 1;
    }
    common_init();

    // The readout decides how the context has to be built, so the model is loaded first.
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model_params mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == nullptr) {
        LOG_ERR("%s: failed to load model '%s'\n", __func__, params.model.path.c_str());
        return 1;
    }

    system_one_params cfg;
    std::string err;
    try {
        cfg = system_one_params_from_model(model);
    } catch (const std::exception & e) {
        LOG_ERR("%s: %s\n", __func__, e.what());
        llama_model_free(model);
        return 1;
    }
    if (!params.so_template_file.empty()) {
        if (!read_file(params.so_template_file, cfg.template_src)) {
            LOG_ERR("%s: cannot read template %s\n", __func__, params.so_template_file.c_str());
            llama_model_free(model);
            return 1;
        }
    }

    std::string state = params.so_state;
    if (!params.so_state_file.empty() && !read_file(params.so_state_file, state)) {
        LOG_ERR("%s: cannot read state file %s\n", __func__, params.so_state_file.c_str());
        llama_model_free(model);
        return 1;
    }

    std::vector<std::string>           keys;
    std::vector<system_one_question>  qs;
    std::vector<std::string> req_labels;
    float                    req_temperature = 1.0f;
    if (!collect_questions(params, keys, qs, state, req_labels, req_temperature, err)) {
        LOG_ERR("%s: %s\n", __func__, err.c_str());
        llama_model_free(model);
        return 1;
    }
    if (qs.empty()) {
        LOG_ERR("%s: no questions -- pass --noul / --choice / --score, or --system-one-request FILE\n", __func__);
        llama_model_free(model);
        return 1;
    }

    system_one_plan plan;
    if (!params.so_labels.empty()) {
        req_labels = string_split<std::string>(params.so_labels, ',');
    }
    if (!req_labels.empty()) {
        // the labels say where an answer is read, so a format written by an overridden
        // template has to be able to name them too
        cfg.labels             = req_labels;
        cfg.labels_are_default = false;
    }

    try {
        plan = system_one_build_plan(cfg, llama_model_get_vocab(model), state, qs);
        system_one_tokenize_plan(llama_model_get_vocab(model), cfg, plan);
    } catch (const std::exception & e) {
        LOG_ERR("%s: %s\n", __func__, e.what());
        llama_model_free(model);
        return 1;
    }

    // What the checkpoint's readout needs of the context, applied here rather than by the
    // library, so every value the context is built with is visible at this one place.
    const system_one_context_needs need = system_one_required_context(cfg);

    if (need.embeddings) {
        params.embedding = true;
    }
    if (need.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED) {
        params.pooling_type = need.pooling_type;
    }

    // Scoring one sequence per option wants them in one decode, capped so a 255-option
    // question does not demand a 255-sequence micro-batch.
    const size_t n_seq = std::min<size_t>(plan.sequences.size(), MAX_BATCHED_SEQS);
    if (n_seq > 1) {
        size_t n_tok = 0;
        bool   share = false;
        for (size_t i = 0; i < n_seq; i++) {
            n_tok += plan.sequences[i].tok.ids.size();
            share  = share || plan.sequences[i].n_shared > 0;
        }
        if (share) {
            // sharing copies a range of one sequence's KV to the others, and a partial
            // seq_cp() only works inside one stream -- which is what a unified cache is
            params.kv_unified = true;
        }
        params.n_parallel = std::max<int32_t>(params.n_parallel, (int32_t) n_seq);
        params.n_ubatch   = std::max<int32_t>(params.n_ubatch,   (int32_t) n_tok);
        params.n_batch    = std::max<int32_t>(params.n_batch,    params.n_ubatch);
        params.n_ctx      = std::max<int32_t>(params.n_ctx,      params.n_ubatch);
    }

    llama_context_params cparams = common_context_params_to_llama(params);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        LOG_ERR("%s: failed to create the context\n", __func__);
        llama_model_free(model);
        return 1;
    }

    std::vector<system_one_answer> answers;
    try {
        answers = plan.rank_pooling ? decode_ranked(ctx, plan) : decode_slots(ctx, plan);
    } catch (const std::exception & e) {
        LOG_ERR("%s: %s\n", __func__, e.what());
        llama_free(ctx); llama_model_free(model);
        return 1;
    }

    // A caller recalibrating on its own distribution can scale the answers; T = 1, the common
    // case, costs nothing. The model's own T is already folded into its weights.
    const float t = params.so_temperature > 0.0f ? params.so_temperature : req_temperature;
    for (auto & a : answers) {
        system_one_apply_temperature(a, t);
    }

    // One rendering loop for every readout: the answers arrived in request order either way.
    json out_answers;
    int32_t n_tokens = 0;
    for (const auto & s : plan.sequences) n_tokens += (int32_t) s.tok.ids.size();

    for (size_t qi = 0; qi < qs.size(); qi++) {
        const auto & a = answers[qi];
        const auto & q = qs[qi];
        json ja;
        switch (q.kind) {
            case SYSTEM_ONE_KIND_NOUL:
                ja["noul"] = a.probs.size() > 1 ? a.probs[1] : 0.0f;
                break;
            case SYSTEM_ONE_KIND_CHOICE: {
                ja["choice"] = q.options[a.choice];
                json probs;
                for (size_t i = 0; i < q.options.size() && i < a.probs.size(); i++) {
                    probs[q.options[i]] = a.probs[i];
                }
                ja["probabilities"] = probs;
            } break;
            case SYSTEM_ONE_KIND_SCORE:
                ja["score"]         = system_one_score_expectation(a);
                ja["legend"]        = q.options;
                ja["probabilities"] = a.probs;
                break;
        }
        ja["confidence"] = a.confidence;
        ja["logits"]     = a.logits;
        out_answers[keys[qi]] = ja;
    }

    if (params.so_json) {
        json res;
        res["model"]   = params.model.path;
        res["answers"] = out_answers;
        res["usage"]   = json{
            {"input_tokens",  n_tokens},
            {"output_tokens", 0},
            {"sequences",     (int) plan.sequences.size()},
        };
        LOG("%s\n", res.dump(2).c_str());
    } else {
        for (size_t qi = 0; qi < qs.size(); qi++) {
            const auto & a = answers[qi];
            const auto & q = qs[qi];
            LOG("\n%s  (%s, confidence %.3f)\n", keys[qi].c_str(),
                q.kind == SYSTEM_ONE_KIND_NOUL   ? "noul"   :
                q.kind == SYSTEM_ONE_KIND_CHOICE ? "choice" : "score", a.confidence);
            switch (q.kind) {
                case SYSTEM_ONE_KIND_NOUL:
                    LOG("  P(true) = %.4f\n", a.probs.size() > 1 ? a.probs[1] : 0.0f);
                    break;
                case SYSTEM_ONE_KIND_CHOICE:
                    LOG("  -> %s\n", q.options[a.choice].c_str());
                    for (size_t i = 0; i < q.options.size() && i < a.probs.size(); i++) {
                        LOG("     %-24s %.4f\n", q.options[i].c_str(), a.probs[i]);
                    }
                    break;
                case SYSTEM_ONE_KIND_SCORE:
                    // the axis, not a fraction: the index of a level, 0 .. n-1
                    LOG("  score = %.4f on 0..%zu  (0 %s .. %zu %s)\n",
                        system_one_score_expectation(a), q.options.size() - 1,
                        q.options.front().c_str(), q.options.size() - 1, q.options.back().c_str());
                    for (size_t i = 0; i < q.options.size() && i < a.probs.size(); i++) {
                        LOG("     %-24s %.4f\n", q.options[i].c_str(), a.probs[i]);
                    }
                    break;
            }
        }
        LOG("\n%d prompt tokens, %zu sequence(s), 0 tokens generated\n",
            n_tokens, plan.sequences.size());
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
