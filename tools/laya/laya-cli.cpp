// llama-laya-cli: run the self-contained laya decision model.
//
// Loads a laya GGUF twice: once through the tools/laya library (self-contained
// ggml graph) and once through libllama purely to borrow the vocabulary.
// Reads a JSON file containing `state` + `questions`, ports the PyTorch
// `build_sequence` tokenizer pipeline, runs one forward pass through the
// encoder + decision head, and prints per-question answers
// (choice/score/noul + probabilities + confidence + action) plus the raw
// per-question outputs used to compare against tests/laya/golden.
//
// Usage:
//   llama-laya-cli -m laya-f16.gguf -f input.json
//
// input.json format (same as the golden fixtures under tests/laya/golden):
//   {
//     "state": <string | dict | list>,
//     "questions": {
//       "q1": {"type": "choice", "instructions": "...", "criteria": {...}},
//       "q2": {"type": "score",  "instructions": "...", "criteria": [...]},
//       "q3": {"type": "noul",   "instructions": "..."}
//     }
//   }

#include "common.h"
#include "json.h"
#include "llama.h"
#include "laya.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const char * QTYPE_NAMES[] = { "choice", "score", "noul" };

// ---- JSON helpers (ports of laya/common.py) ----
// Python's json.dumps uses ", " / ": " separators and ensure_ascii=False;
// nlohmann's compact dump() removes the separator spaces, so a small
// serializer is needed to reproduce the reference's serialize_state /
// render_criterion byte-for-byte.

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char) c;
                }
        }
    }
    return out;
}

static std::string python_dump(const common_json & v) {
    if (v.is_string()) {
        return "\"" + json_escape((std::string) v) + "\"";
    }
    if (v.is_array()) {
        std::string out = "[";
        bool first = true;
        for (auto it = v.begin(); it != v.end(); ++it) {
            if (!first) out += ", ";
            first = false;
            out += python_dump(*it);
        }
        out += "]";
        return out;
    }
    if (v.is_object()) {
        std::string out = "{";
        bool first = true;
        for (auto it = v.begin(); it != v.end(); ++it) {
            if (!first) out += ", ";
            first = false;
            out += "\"" + json_escape(it.key()) + "\": " + python_dump(it.value());
        }
        out += "}";
        return out;
    }
    if (v.is_boolean()) {
        return v.get<bool>() ? "true" : "false";
    }
    if (v.is_null()) {
        return "null";
    }
    if (v.is_number_integer()) {
        return std::to_string(v.get<long long>());
    }
    if (v.is_number_float()) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%.17g", v.get<double>());
        return buf;
    }
    return "null";
}

static std::string serialize_state(const common_json & state) {
    if (state.is_string()) {
        return (std::string) state;
    }
    return python_dump(state);
}

static std::string render_criterion(const common_json & value) {
    if (value.is_string()) {
        return (std::string) value;
    }
    return python_dump(value);
}

static std::vector<std::string> render_options(const common_json & q) {
    std::vector<std::string> opts;
    const std::string t = q["type"];
    const common_json crit = q.contains("criteria") ? q["criteria"] : common_json::object();
    if (t == "choice") {
        for (auto it = crit.begin(); it != crit.end(); ++it) {
            const std::string k = it.key();
            const std::string v = render_criterion(it.value());
            opts.push_back(v.empty() ? k : k + ": " + v);
        }
    } else if (t == "score") {
        int32_t i = 0;
        for (auto it = crit.begin(); it != crit.end(); ++it, ++i) {
            opts.push_back("level " + std::to_string(i) + ": " + render_criterion(it.value()));
        }
    } else { // noul
        const std::string fc = crit.contains("false") ? render_criterion(crit["false"]) : "";
        const std::string tc = crit.contains("true")  ? render_criterion(crit["true"])  : "";
        opts.push_back("false: " + (fc.empty() ? "no, the statement does not hold" : fc));
        opts.push_back("true: "  + (tc.empty() ? "yes, the statement holds"          : tc));
    }
    return opts;
}

static std::string replace_all(std::string s, const std::string & from, const std::string & to) {
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.length(), to);
        pos += to.length();
    }
    return s;
}

// ---- build_sequence port (laya/common.py) ----
// Format: [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
struct Seq {
    std::vector<llama_token> ids;
    std::vector<int32_t>     markers;
    int32_t                  qtype = 0;
};

static Seq build_sequence(
        const laya_model * model,
        llama_token cls_token_id,
        llama_token sep_token_id,
        llama_token mask_token_id,
        const common_json & state,
        const common_json & q,
        int32_t max_len,
        int32_t head_max_len) {
    const std::string t    = q["type"];
    const std::string ins  = replace_all((std::string) q["instructions"], "[MASK]", " ");

    std::vector<std::string> opts = render_options(q);

    const auto tokenize = [&](const std::string & text) {
        return laya_tokenize(model, text);
    };

    // head_ids = tokenizer("%s question: %s" % (t, ins), add_special_tokens=False)
    std::vector<llama_token> head_ids = tokenize(t + " question: " + ins);

    // opt_ids = [ [mask] + tokenizer(" " + opt)[:48] ... ]
    std::vector<std::vector<llama_token>> opt_ids;
    for (const auto & opt : opts) {
        std::vector<llama_token> o = { mask_token_id };
        std::vector<llama_token> tok = tokenize(" " + replace_all(opt, "[MASK]", " "));
        if (tok.size() > 48) {
            tok.resize(48);
        }
        o.insert(o.end(), tok.begin(), tok.end());
        opt_ids.push_back(o);
    }

    int32_t opt_budget = head_max_len;
    for (const auto & o : opt_ids) {
        opt_budget -= (int32_t) o.size();
    }
    if (opt_budget < 16) {
        const int32_t per = std::max(4, (head_max_len - 16) / std::max(1, (int32_t) opt_ids.size()));
        for (auto & o : opt_ids) {
            if ((int32_t) o.size() > per) o.resize(per);
        }
        opt_budget = head_max_len;
        for (const auto & o : opt_ids) {
            opt_budget -= (int32_t) o.size();
        }
    }
    if ((int32_t) head_ids.size() > std::max(8, opt_budget)) {
        head_ids.resize(std::max(8, opt_budget));
    }

    std::vector<llama_token> ids = { cls_token_id };
    ids.insert(ids.end(), head_ids.begin(), head_ids.end());
    ids.push_back(sep_token_id);

    std::vector<int32_t> markers;
    for (const auto & o : opt_ids) {
        markers.push_back((int32_t) ids.size());
        ids.insert(ids.end(), o.begin(), o.end());
    }
    ids.push_back(sep_token_id);

    const int32_t room = std::max(0, max_len - (int32_t) ids.size() - 1);
    std::vector<llama_token> st = tokenize(replace_all(serialize_state(state), "[MASK]", " "));
    if ((int32_t) st.size() > room) {
        st.resize(room);
    }
    ids.insert(ids.end(), st.begin(), st.end());
    ids.push_back(sep_token_id);

    if ((int32_t) ids.size() > max_len) {
        ids.resize(max_len);
    }

    Seq seq;
    seq.ids = ids;
    for (int32_t m : markers) {
        if (m < max_len) {
            seq.markers.push_back(m);
        }
    }
    return seq;
}

// ---- temperature / confidence (ports of laya/common.py + agent.py) ----

static float clamp_temperature(float t) {
    if (std::isnan(t) || std::isinf(t)) {
        return 1.0f;
    }
    return std::min(5.0f, std::max(0.5f, t));
}

// temperature_by_options is not persisted in the GGUF, so only the base
// per-qtype temperature applies (like the reference when the map is empty).
static float temperature_for_qtype(const std::vector<float> & temperature, int32_t qtype) {
    if (temperature.empty()) {
        return 1.0f;
    }
    return clamp_temperature(temperature[qtype % (int32_t) temperature.size()]);
}

static float confidence_from_probs(const std::vector<float> & p) {
    const int k = (int) p.size();
    if (k < 2) {
        return 1.0f;
    }
    float ent = 0.0f;
    for (int i = 0; i < k; ++i) {
        if (p[i] > 0.0f) {
            ent -= p[i] * std::log(p[i]);
        }
    }
    return std::min(1.0f, std::max(0.0f, 1.0f - ent / std::log((float) k)));
}

int main(int argc, char ** argv) {
    setlocale(LC_ALL, "");

    std::string model_path;
    std::string input_path;
    int n_threads = 1;
    int n_bench   = 0; // >0: repeat the forward pass in-process for timing/stability

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) model_path = argv[++i];
        } else if (arg == "-f" || arg == "--file") {
            if (i + 1 < argc) input_path = argv[++i];
        } else if (arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) n_threads = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "-b" || arg == "--bench") {
            if (i + 1 < argc) n_bench = std::max(0, std::atoi(argv[++i]));
        } else if (arg == "-h" || arg == "--help") {
            printf("Usage: %s -m <laya-f16.gguf> -f <input.json> [-t threads] [-b bench_runs]\n", argv[0]);
            return 0;
        } else {
            fprintf(stderr, "laya: unknown argument: %s\n", arg.c_str());
            return 1;
        }
    }

    if (model_path.empty() || input_path.empty()) {
        fprintf(stderr, "laya: missing -m <model> or -f <input>\n");
        return 1;
    }

    // load the self-contained model
    laya_model * model = nullptr;
    try {
        model = laya_model_load_from_file(model_path.c_str());
    } catch (const std::exception & e) {
        fprintf(stderr, "laya: failed to load model: %s\n", e.what());
        return 1;
    }
    const laya_hparams & hp = laya_model_hparams(model);

    // borrow the vocabulary from libllama (no inference / graph / batch API)
    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true; // only the tokenizer is needed, skip the weights
    llama_model * vmodel = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!vmodel) {
        fprintf(stderr, "laya: failed to load vocab from '%s'\n", model_path.c_str());
        laya_model_free(model);
        return 1;
    }
    const struct llama_vocab * vocab = llama_model_get_vocab(vmodel);

    // read input JSON
    FILE * fin = fopen(input_path.c_str(), "rb");
    if (!fin) {
        fprintf(stderr, "laya: failed to open input file '%s'\n", input_path.c_str());
        llama_model_free(vmodel);
        laya_model_free(model);
        return 1;
    }
    fseek(fin, 0, SEEK_END);
    const long fsize = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    std::string text((size_t) fsize, '\0');
    const size_t nread = fread(&text[0], 1, (size_t) fsize, fin);
    fclose(fin);
    text.resize(nread);

    common_json input;
    try {
        input = common_json::parse(text);
    } catch (const std::exception & e) {
        fprintf(stderr, "laya: failed to parse JSON: %s\n", e.what());
        llama_model_free(vmodel);
        laya_model_free(model);
        return 1;
    }

    const common_json & state = input["state"];
    const common_json & questions = input["questions"];

    // one sequence per question
    const int32_t max_len      = hp.max_len > 0 ? hp.max_len : 1024;
    const int32_t head_max_len = hp.head_max_len > 0 ? hp.head_max_len : 256;

    std::vector<Seq> seqs;
    for (auto it = questions.begin(); it != questions.end(); ++it) {
        const common_json & q = it.value();
        const std::string t = q["type"];
        int32_t qtype = 0;
        for (int i = 0; i < 3; ++i) {
            if (t == QTYPE_NAMES[i]) {
                qtype = i;
            }
        }
        Seq seq = build_sequence(model, llama_vocab_bos(vocab), llama_vocab_sep(vocab),
                                 llama_vocab_mask(vocab), state, q, max_len, head_max_len);
        seq.qtype = qtype;
        seqs.push_back(std::move(seq));
    }

    const int32_t n_seqs = (int32_t) seqs.size();
    if (n_seqs == 0) {
        fprintf(stderr, "laya: no questions in input\n");
        llama_model_free(vmodel);
        laya_model_free(model);
        return 1;
    }

    for (int32_t s = 0; s < n_seqs; ++s) {
        if ((int32_t) seqs[s].markers.size() > LAYA_MAX_MARKERS) {
            fprintf(stderr, "laya: question %d has %d options > LAYA_MAX_MARKERS=%d\n",
                    s, (int) seqs[s].markers.size(), LAYA_MAX_MARKERS);
            llama_model_free(vmodel);
            laya_model_free(model);
            return 1;
        }
    }

    // pack all sequences into one batch
    int32_t n_tokens = 0;
    for (const auto & s : seqs) {
        n_tokens += (int32_t) s.ids.size();
    }

    std::vector<int32_t> tokens(n_tokens);
    std::vector<int32_t> positions(n_tokens);
    std::vector<int32_t> seq_id(n_tokens);
    std::vector<int32_t> qtype(n_tokens);
    std::vector<int32_t> marker_pos(LAYA_MAX_MARKERS * n_seqs, 0);
    std::vector<int32_t> marker_mask(LAYA_MAX_MARKERS * n_seqs, 0);
    std::vector<int32_t> seq_start(n_seqs);

    int32_t t = 0;
    for (int32_t s = 0; s < n_seqs; ++s) {
        const Seq & seq = seqs[s];
        seq_start[s] = t;
        for (size_t i = 0; i < seq.ids.size(); ++i) {
            tokens[t]   = seq.ids[i];
            positions[t] = (int32_t) i;
            seq_id[t]   = s;
            qtype[t]    = seq.qtype;
            ++t;
        }
        const int32_t k = (int32_t) seq.markers.size();
        for (int32_t j = 0; j < k && j < LAYA_MAX_MARKERS; ++j) {
            // row-major [n_markers_max, n_seqs]: index = m + n_markers_max * s
            // marker positions are sequence-local; the flattened encoder output
            // concatenates the sequences, so offset by this sequence's start.
            marker_pos[j + LAYA_MAX_MARKERS * s]  = seq_start[s] + seq.markers[j];
            marker_mask[j + LAYA_MAX_MARKERS * s] = 1;
        }
    }

    laya_batch batch;
    batch.n_tokens   = n_tokens;
    batch.n_seqs     = n_seqs;
    batch.tokens     = tokens.data();
    batch.positions  = positions.data();
    batch.seq_id     = seq_id.data();
    batch.qtype      = qtype.data();
    batch.marker_pos = marker_pos.data();
    batch.marker_mask = marker_mask.data();
    batch.seq_start  = seq_start.data();

    laya_context * ctx = nullptr;
    try {
        ctx = laya_init(model, n_threads);
    } catch (const std::exception & e) {
        fprintf(stderr, "laya: failed to init context: %s\n", e.what());
        llama_model_free(vmodel);
        laya_model_free(model);
        return 1;
    }

    laya_result result;
    const int n_runs = n_bench > 0 ? n_bench : 1;
    std::vector<double> run_ms;
    bool deterministic = true;
    std::vector<float> first_logits, first_act;
    for (int run = 0; run < n_runs; ++run) {
        const auto t0 = std::chrono::steady_clock::now();
        if (laya_encode(ctx, batch, result) != 0) {
            fprintf(stderr, "laya: forward pass failed\n");
            laya_free(ctx);
            llama_model_free(vmodel);
            laya_model_free(model);
            return 1;
        }
        const auto t1 = std::chrono::steady_clock::now();
        run_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        if (run == 0) {
            first_logits = result.logits;
            first_act    = result.act_logits;
        } else if (result.logits != first_logits || result.act_logits != first_act) {
            deterministic = false;
        }
    }

    // bench statistics
    double sum_ms = 0.0, min_ms = run_ms[0], max_ms = run_ms[0];
    for (double v : run_ms) { sum_ms += v; min_ms = std::min(min_ms, v); max_ms = std::max(max_ms, v); }
    const double mean_ms = sum_ms / run_ms.size();
    double var_ms = 0.0;
    for (double v : run_ms) { var_ms += (v - mean_ms) * (v - mean_ms); }
    const double std_ms = std::sqrt(var_ms / run_ms.size());
    std::vector<double> sorted_ms = run_ms;
    std::sort(sorted_ms.begin(), sorted_ms.end());
    const double median_ms = sorted_ms[sorted_ms.size() / 2];

    if (n_bench > 0) {
        fprintf(stderr,
                "laya bench: tokens=%d seqs=%d runs=%d threads=%d  "
                "mean=%.2fms median=%.2fms min=%.2fms max=%.2fms std=%.2fms  deterministic=%s\n",
                n_tokens, n_seqs, n_runs, n_threads,
                mean_ms, median_ms, min_ms, max_ms, std_ms, deterministic ? "yes" : "no");
    }

    const int32_t n_markers_max = result.n_markers_max;
    const int32_t n_act         = result.n_act;
    const std::vector<float> & temperature = hp.temperature;

    // per-question answers
    common_json answers = common_json::object();
    common_json per_question = common_json::object();

    int32_t qid = 0;
    for (auto it = questions.begin(); it != questions.end(); ++it, ++qid) {
        const std::string qid_name = it.key();
        const common_json & q = it.value();
        const std::string t = q["type"];
        const Seq & seq = seqs[qid];
        const int32_t k = (int32_t) seq.markers.size();
        const int32_t qt = seq.qtype;

        const float t_scale = temperature_for_qtype(temperature, qt);
        std::vector<float> logits_k(k);
        for (int32_t j = 0; j < k; ++j) {
            logits_k[j] = result.logits[qid*n_markers_max + j] / t_scale;
        }

        // softmax
        std::vector<float> p(k);
        float max_l = logits_k[0];
        for (int32_t j = 1; j < k; ++j) {
            max_l = std::max(max_l, logits_k[j]);
        }
        float sum = 0.0f;
        for (int32_t j = 0; j < k; ++j) {
            p[j] = std::exp(logits_k[j] - max_l);
            sum += p[j];
        }
        for (int32_t j = 0; j < k; ++j) {
            p[j] /= sum;
        }

        const float conf = confidence_from_probs(p);

        // action head softmax
        std::vector<float> act(n_act);
        float amax = result.act_logits[qid*n_act];
        for (int32_t j = 0; j < n_act; ++j) {
            act[j] = result.act_logits[qid*n_act + j];
            amax = std::max(amax, act[j]);
        }
        float asum = 0.0f;
        for (int32_t j = 0; j < n_act; ++j) {
            act[j] = std::exp(act[j] - amax);
            asum += act[j];
        }
        for (int32_t j = 0; j < n_act; ++j) {
            act[j] /= asum;
        }
        const float act_probability = act[0];

        common_json ans = common_json::object();
        ans["type"] = t;
        ans["confidence"] = (double) std::round(conf*10000.0)/10000.0;
        ans["action"] = common_json::object();
        ans["action"]["act_probability"] = (double) std::round(act_probability*10000.0)/10000.0;

        if (t == "choice") {
            const common_json & crit = q["criteria"];
            std::vector<std::string> keys;
            for (auto itc = crit.begin(); itc != crit.end(); ++itc) {
                keys.push_back(itc.key());
            }
            int argmax = 0;
            for (int32_t j = 1; j < k; ++j) {
                if (p[j] > p[argmax]) {
                    argmax = j;
                }
            }
            ans["choice"] = keys[argmax];
            common_json probs = common_json::object();
            for (int32_t j = 0; j < k; ++j) {
                probs[keys[j]] = (double) std::round(p[j]*10000.0)/10000.0;
            }
            ans["probabilities"] = probs;
        } else if (t == "score") {
            float exp_score = 0.0f;
            for (int32_t j = 0; j < k; ++j) {
                exp_score += j * p[j];
            }
            ans["score"] = (double) std::round(exp_score*10000.0)/10000.0;
            common_json legend = common_json::object();
            common_json probs = common_json::object();
            for (int32_t j = 0; j < k; ++j) {
                legend[std::to_string(j)] = render_options(q)[j];
                probs[std::to_string(j)]  = (double) std::round(p[j]*10000.0)/10000.0;
            }
            ans["legend"] = legend;
            ans["probabilities"] = probs;
        } else { // noul
            ans["noul"] = (double) std::round(p[1]*10000.0)/10000.0;
        }

        answers[qid_name] = ans;

        // per-question raw outputs (for comparison with golden fixtures)
        common_json pq = common_json::object();
        pq["qtype"] = t;
        common_json opt_arr = common_json::array();
        for (const auto & o : render_options(q)) {
            opt_arr.push_back(o);
        }
        pq["options"] = opt_arr;
        common_json ids_arr = common_json::array();
        for (auto tok : seq.ids) {
            ids_arr.push_back((int64_t) tok);
        }
        pq["input_ids"] = ids_arr;
        common_json mk_arr = common_json::array();
        for (auto m : seq.markers) {
            mk_arr.push_back((int64_t) m);
        }
        pq["marker_pos"] = mk_arr;
        common_json rl_arr = common_json::array();
        for (int32_t j = 0; j < k; ++j) {
            rl_arr.push_back((double) result.logits[qid*n_markers_max + j]);
        }
        pq["raw_logits"] = rl_arr;
        common_json al_arr = common_json::array();
        for (int32_t j = 0; j < n_act; ++j) {
            al_arr.push_back((double) act[j]);
        }
        pq["act_logits"] = al_arr;
        pq["act_probability"] = (double) act_probability;
        per_question[qid_name] = pq;
    }

    common_json out = common_json::object();
    out["model"] = model_path;
    out["state"] = state;
    out["questions"] = questions;
    out["per_question"] = per_question;
    out["answers"] = answers;

    if (n_bench > 0) {
        common_json b = common_json::object();
        b["runs"]   = n_runs;
        b["tokens"] = n_tokens;
        b["seqs"]   = n_seqs;
        b["threads"] = n_threads;
        b["mean_ms"] = mean_ms;
        b["median_ms"] = median_ms;
        b["min_ms"] = min_ms;
        b["max_ms"] = max_ms;
        b["std_ms"] = std_ms;
        b["deterministic"] = deterministic;
        out["bench"] = b;
    }

    printf("%s\n", out.dump(2).c_str());

    laya_free(ctx);
    llama_model_free(vmodel);
    laya_model_free(model);
    return 0;
}
