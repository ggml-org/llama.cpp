#include "system-one.h"

#include "chat.h"                 // pulls in the jinja engine the chat templates use
#include "common.h"               // common_tokenize, string_split
#include "json.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

using json = common_json;


system_one_answer system_one_answer_from_logits(const float * row,
                                                const std::vector<llama_token> & letter_ids,
                                                int n_options);

// ---------------------------------------------------------------- model metadata

static bool meta_str(const llama_model * model, const char * key, std::string & out) {
    const int32_t n = llama_model_meta_val_str(model, key, nullptr, 0);
    if (n < 0) return false;
    std::vector<char> buf(n + 1);
    if (llama_model_meta_val_str(model, key, buf.data(), buf.size()) < 0) return false;
    out.assign(buf.data());
    return true;
}

// -1 when the key is absent; the buffer is optional, so this asks without reading.
static bool meta_has(const llama_model * model, const char * key) {
    return llama_model_meta_val_str(model, key, nullptr, 0) >= 0;
}

static std::vector<std::string> split_array(const std::string & s) {
    // llama.cpp stringifies a kv array as [a, b, c]
    std::vector<std::string> out;
    for (auto & piece : string_split(s, ",")) {
        std::string cur;
        for (char c : piece) {
            if (c != '[' && c != ']' && c != '"' && c != ' ') cur += c;
        }
        if (!cur.empty()) out.push_back(cur);
    }
    return out;
}

const char * system_one_readout_name(enum system_one_readout readout) {
    switch (readout) {
        case SYSTEM_ONE_READOUT_MASKED_SLOT: return "masked_slot";
        case SYSTEM_ONE_READOUT_RANK_HEAD:   return "rank_head";
        default:                             return "letter_slot";
    }
}

system_one_params system_one_params_from_model(const llama_model * model) {
    system_one_params out;
    const char * tmpl = llama_model_chat_template(model, "system_one");
    if (tmpl == nullptr || tmpl[0] == '\0') {
        throw std::runtime_error("model has no System One template (tokenizer.chat_template.system_one); "
              "convert it with its template, or pass one in the request");
    }
    out.template_src = tmpl;

    std::string s;
    // The readout follows from what the checkpoint *is*, so it does not have to be declared:
    //   a classification head  -> the answer is that head's score, one sequence per option
    //   bidirectional          -> the answer is read at a mask token inside the question
    //   otherwise              -> the answer is the next token after the question
    {
        std::string arch;
        bool causal = true;
        if (meta_str(model, "general.architecture", arch)) {
            std::string c;
            if (meta_str(model, (arch + ".attention.causal").c_str(), c) && !c.empty()) {
                causal = (c == "true" || c == "1");
            }
        }
        const bool has_cls_head = llama_model_cls_label(model, 0) != nullptr;
        out.causal = causal;

        out.readout = has_cls_head ? SYSTEM_ONE_READOUT_RANK_HEAD
                    : (causal      ? SYSTEM_ONE_READOUT_LETTER_SLOT
                                   : SYSTEM_ONE_READOUT_MASKED_SLOT);
    }

    if (meta_str(model, "system_one.segment_separator", s) && !s.empty()) out.segment_separator = s;
    if (meta_str(model, "system_one.labels", s) && !s.empty()) out.labels = split_array(s);

    if (out.labels.empty()) {
        // the usual alphabet, so a plain letter format needs no key at all
        out.labels_are_default = true;
        for (char c = 'A'; c <= 'Z'; c++) out.labels.push_back(std::string(1, c));
        for (char c = 'a'; c <= 'z'; c++) out.labels.push_back(std::string(1, c));
    }

    // the rest is ordinary model metadata
    const llama_vocab * vocab = llama_model_get_vocab(model);

    auto piece = [&](llama_token id) {
        const char * t = id >= 0 ? llama_vocab_get_text(vocab, id) : nullptr;
        return t ? std::string(t) : std::string();
    };
    // Only what the checkpoint declares. llama_vocab_bos() falls back to a per-family default
    // when the GGUF carries no id -- 11, a comma, on the BPE path -- and a template asking for
    // this model's BOS would then write that comma into the prompt.
    if (meta_has(model, "tokenizer.ggml.bos_token_id")) {
        out.bos_text = piece(llama_vocab_bos(vocab));
    }
    if (meta_has(model, "tokenizer.ggml.eos_token_id")) {
        out.eos_text = piece(llama_vocab_eos(vocab));
    }

    if (out.readout == SYSTEM_ONE_READOUT_MASKED_SLOT) {
        out.mask_token = llama_vocab_mask(vocab);
        if (out.mask_token < 0) {
            throw std::runtime_error("masked_slot needs the model's tokenizer.ggml.mask_token_id");
        }
        const char * piece = llama_vocab_get_text(vocab, out.mask_token);
        out.mask_text = piece ? piece : "";
    }

    return out;
}

// ---------------------------------------------------------------- rendering

static const char * kind_name(enum system_one_kind k) {
    switch (k) {
        case SYSTEM_ONE_KIND_CHOICE: return "choice";
        case SYSTEM_ONE_KIND_SCORE:  return "score";
        default:           return "noul";
    }
}

static void render_and_split(const system_one_params & cfg, const json & inp,
                             std::vector<std::string> & segments) {
    std::string rendered;
    try {
        jinja::lexer lexer;
        auto lexer_res = lexer.tokenize(cfg.template_src);
        jinja::program prog = jinja::parse_from_tokens(lexer_res);

        jinja::context ctx(lexer_res.source);
        jinja::global_from_json(ctx, inp, false);

        jinja::runtime runtime(ctx);
        const jinja::value results = runtime.execute(prog);
        auto parts = jinja::runtime::gather_string_parts(results);
        rendered = parts->as_string().str();
    } catch (const std::exception & e) {
        throw std::invalid_argument(std::string("system_one template failed to render: ") + e.what());
    }

    segments.clear();
    size_t pos = 0;
    while (true) {
        const size_t next = rendered.find(cfg.segment_separator, pos);
        if (next == std::string::npos) {
            segments.push_back(rendered.substr(pos));
            break;
        }
        segments.push_back(rendered.substr(pos, next - pos));
        pos = next + cfg.segment_separator.size();
    }
    segments.erase(std::remove_if(segments.begin(), segments.end(),
                                  [](const std::string & s) { return s.empty(); }),
                   segments.end());
    return;
}

std::vector<std::string> system_one_render_segments(const system_one_params & cfg,
                                                    const std::string & state,
                                                    const std::vector<system_one_question> & qs) {
    std::vector<std::string> segments;
    if (cfg.segment_separator.empty()) {
        throw std::runtime_error("system_one.template.segment_separator is empty");
    }

    json questions = json::array();
    for (size_t i = 0; i < qs.size(); i++) {
        const system_one_question & q = qs[i];
        json options = json::array();
        for (size_t j = 0; j < q.options.size(); j++) {
            const bool has_desc = j < q.descs.size() && !q.descs[j].empty();
            options.push_back(json{
                {"label",  j < cfg.labels.size() ? cfg.labels[j] : std::string("?")},
                {"option", q.options[j]},
                {"desc",   has_desc ? json(q.descs[j]) : json(nullptr)},
            });
        }
        questions.push_back(json{
            {"k",       (int) i + 1},          // 1-based, as templates count questions
            {"kind",    kind_name(q.kind)},
            {"text",    q.text},
            {"options", options},
        });
    }

    const json inp = json{
        {"state",     state},
        {"questions", questions},
        {"sep",       cfg.segment_separator},
        {"mask",      cfg.mask_text},      // empty unless the format places a mask token
        {"bos_token", cfg.bos_text},       // as a chat template gets them: the prompt says
        {"eos_token", cfg.eos_text},       // where they go, no flag decides it
    };

    render_and_split(cfg, inp, segments);

    if (segments.size() < qs.size()) {
        throw std::invalid_argument("template produced " + std::to_string(segments.size()) + " segments for " +
              std::to_string(qs.size()) + " questions: the separator must precede every question block");
    }
    return segments;
}

static void render_pair_segments(const system_one_params & cfg,
                          const std::string & state,
                          const system_one_question & q,
                          size_t option_index,
                          std::vector<std::string> & segments) {
    if (option_index >= q.options.size()) {
        throw std::runtime_error("option index out of range");
    }
    const bool has_desc = option_index < q.descs.size() && !q.descs[option_index].empty();

    const json inp = json{
        {"state", state},
        {"question", json{
            {"kind", kind_name(q.kind)},
            {"text", q.text},
        }},
        {"option", json{
            {"label",  option_index < cfg.labels.size() ? cfg.labels[option_index] : std::string("?")},
            {"option", q.options[option_index]},
            {"desc",   has_desc ? json(q.descs[option_index]) : json(nullptr)},
            {"index",  (int) option_index},
        }},
        {"sep",       cfg.segment_separator},
        {"mask",      cfg.mask_text},
        {"bos_token", cfg.bos_text},
        {"eos_token", cfg.eos_text},
    };

    render_and_split(cfg, inp, segments);
    if (segments.empty()) {
        throw std::invalid_argument("template rendered nothing for this question/option pair");
    }
    return;
}

// ---------------------------------------------------------------- tokenization

system_one_tokenized system_one_resolve_question_slots(const llama_vocab * vocab,
                                                       const system_one_params & cfg,
                                                       const std::vector<std::string> & question_segments,
                                                       size_t prefix_positions) {
    system_one_tokenized out;
    std::vector<llama_token> & ids_out   = out.ids;
    std::vector<int>         & slots_out = out.slots;
    ids_out.clear();
    slots_out.clear();

    const bool masked = cfg.readout == SYSTEM_ONE_READOUT_MASKED_SLOT;

    for (size_t i = 0; i < question_segments.size(); i++) {
        // BOS and the mask are written into the template as text and have to parse back to
        // their ids, which is also what HF's tokenizers do regardless of add_special_tokens.
        const auto ids = common_tokenize(vocab, question_segments[i], false, true);
        if (ids.empty()) {
            throw std::invalid_argument("question segment " + std::to_string(i) + " tokenized to nothing");
        }
        const size_t base = prefix_positions + ids_out.size();
        ids_out.insert(ids_out.end(), ids.begin(), ids.end());

        if (!masked) {
            slots_out.push_back((int) (base + ids.size() - 1));
            continue;
        }

        // the answer is read at the mask token itself; take the last one in the segment
        int slot = -1;
        for (size_t j = ids.size(); j-- > 0; ) {
            if (ids[j] == cfg.mask_token) { slot = (int) (base + j); break; }
        }
        if (slot < 0) {
            throw std::invalid_argument("question segment " + std::to_string(i) +
                  " has no mask token (" + std::to_string(cfg.mask_token) + ") after tokenization");
        }
        slots_out.push_back(slot);
    }
    return out;
}

system_one_tokenized system_one_tokenize_segments(const llama_vocab * vocab,
                                                  const system_one_params & cfg,
                                                  const std::vector<std::string> & segments,
                                                  size_t n_questions) {
    system_one_tokenized out;
    if (segments.size() < n_questions) {
        throw std::invalid_argument("fewer segments than questions");
    }

    out.ids.clear();
    out.slots.clear();

    // the question blocks are the trailing segments; everything before them is the state
    const size_t first_question = segments.size() - n_questions;
    for (size_t i = 0; i < first_question; i++) {
        const auto ids = common_tokenize(vocab, segments[i], false, true);
        if (ids.empty()) {
            throw std::invalid_argument("segment " + std::to_string(i) + " tokenized to nothing");
        }
        out.ids.insert(out.ids.end(), ids.begin(), ids.end());
    }

    std::vector<llama_token> q_ids;
    const std::vector<std::string> q_segments(segments.begin() + first_question, segments.end());
    const system_one_tokenized q = system_one_resolve_question_slots(vocab, cfg, q_segments, out.ids.size());
    q_ids     = q.ids;
    out.slots = q.slots;
    out.ids.insert(out.ids.end(), q_ids.begin(), q_ids.end());
    return out;
}

std::vector<llama_token> system_one_label_tokens(const llama_vocab * vocab,
                                                 const std::vector<std::string> & labels,
                                                 bool labels_are_default) {
    // Tokenize each label rather than looking it up in the vocab: a format whose answers are
    // " A", " B" is read at the tokens a BPE vocab spells "ĠA", "ĠB".
    std::vector<llama_token> out(labels.size(), -1);
    for (size_t i = 0; i < labels.size(); i++) {
        const auto ids = common_tokenize(vocab, labels[i], false, false);
        if (ids.size() == 1) {
            out[i] = ids[0];
            continue;
        }
        // naming the label is not enough: the caller may never have chosen it
        throw std::invalid_argument("the answer label \"" + labels[i] + "\" is not a single token for this "
              "tokenizer; " + (labels_are_default
                 ? "it comes from the default A-Za-z alphabet, not from anything this request "
                   "or checkpoint asked for -- set system_one.labels, or send \"labels\", to "
                   "name the labels this format actually writes"
                 : "it was named by system_one.labels or by the request"));
    }
    return out;
}

// ---------------------------------------------------------------- planning

system_one_plan system_one_build_plan(const system_one_params & cfg,
                                      const llama_vocab * vocab,
                                      const std::string & state,
                                      const std::vector<system_one_question> & qs_in) {
    system_one_plan out;
    out = system_one_plan();
    if (qs_in.empty()) {
        throw std::invalid_argument("a request needs at least one question");
    }

    // noul's two sides belong to the format; a caller may still word them itself
    std::vector<system_one_question> qs;
    qs.reserve(qs_in.size());
    for (auto q : qs_in) {
        if (q.kind == SYSTEM_ONE_KIND_NOUL && q.options.empty()) {
            q.options = {"no", "yes"};
        }
        qs.push_back(std::move(q));
    }

    for (const auto & q : qs) {
        if (q.options.size() < 2) {
            throw std::invalid_argument("every question needs at least two options");
        }
        out.n_options.push_back((int) q.options.size());
    }

    if (cfg.readout == SYSTEM_ONE_READOUT_RANK_HEAD) {
        // the head scores a whole sequence, so each option gets one of its own
        out.rank_pooling = true;

        for (size_t qi = 0; qi < qs.size(); qi++) {
            for (size_t oi = 0; oi < qs[qi].options.size(); oi++) {
                system_one_sequence seq;
                render_pair_segments(cfg, state, qs[qi], oi, seq.segments);
                seq.n_question_segments = 0;   // the head scores the sequence, nothing is read at a slot
                seq.question = qi;
                seq.option   = oi;
                out.sequences.push_back(std::move(seq));
            }
        }
        return out;
    }

    // a slot readout answers every question from one sequence
    out.labels = system_one_label_tokens(vocab, cfg.labels, cfg.labels_are_default);
    for (const auto & q : qs) {
        if (q.options.size() > cfg.labels.size()) {
            throw std::invalid_argument("a question has " + std::to_string(q.options.size()) + " options but this "
                "readout has only " + std::to_string(cfg.labels.size()) + " labels; a model "
                "with a scoring head (rank_head) has no such limit");
        }
    }

    system_one_sequence seq;
    seq.segments = system_one_render_segments(cfg, state, qs);
    seq.n_question_segments = qs.size();
    out.sequences.push_back(std::move(seq));

    return out;
}

void system_one_tokenize_plan(const llama_vocab * vocab, const system_one_params & cfg,
                              system_one_plan & p) {
    for (auto & seq : p.sequences) {
        seq.tok = system_one_tokenize_segments(vocab, cfg, seq.segments, seq.n_question_segments);
    }

    // Reuse is legal only where the model is causal -- a bidirectional prefix depends on what
    // follows it -- so both numbers stay at zero otherwise. And it must stop short of any
    // answer slot, or that slot is never decoded and has no logits.
    for (auto & seq : p.sequences) {
        if (!cfg.causal) continue;
        seq.n_reusable = seq.tok.slots.empty()
            ? seq.tok.ids.size()
            : (size_t) *std::min_element(seq.tok.slots.begin(), seq.tok.slots.end());
    }

    // The option sequences of one question share a head -- the state and the question. Measure
    // it rather than assume where it ends: the template decides that.
    for (size_t i = 0; i < p.sequences.size(); i++) {
        auto & seq = p.sequences[i];
        seq.shares_with = i;
        seq.n_shared    = 0;
        if (!cfg.causal) continue;

        for (size_t j = 0; j < i; j++) {
            if (p.sequences[j].question != seq.question) continue;

            const auto & a = p.sequences[j].tok.ids;
            const auto & b = seq.tok.ids;
            size_t n = 0;
            while (n < a.size() && n < b.size() && a[n] == b[n]) n++;

            // the sequence still has to contribute something of its own
            if (n > seq.n_shared && n < b.size()) {
                seq.shares_with = j;
                seq.n_shared    = n;
            }
        }
    }
    return;
}

std::vector<system_one_answer> system_one_answers_from_logits(const system_one_plan & p,
                                                              const std::vector<const float *> & rows) {
    std::vector<system_one_answer> out;
    if (p.rank_pooling) {
        throw std::invalid_argument("this readout is scored by the model's head, not read at a slot -- "
              "use system_one_answers_from_scores()");
    }
    if (rows.size() != p.n_options.size()) {
        throw std::invalid_argument("got " + std::to_string(rows.size()) + " logit rows for " +
              std::to_string(p.n_options.size()) + " questions");
    }

    out.clear();
    out.reserve(rows.size());
    for (size_t qi = 0; qi < rows.size(); qi++) {
        if (rows[qi] == nullptr) {
            throw std::invalid_argument("no logits for question " + std::to_string(qi));
        }
        out.push_back(system_one_answer_from_logits(rows[qi], p.labels, p.n_options[qi]));
    }
    return out;
}

std::vector<system_one_answer> system_one_answers_from_scores(const system_one_plan & p,
                                                              const std::vector<float> & scores) {
    std::vector<system_one_answer> out;
    if (scores.size() != p.sequences.size()) {
        throw std::runtime_error("expected one score per planned sequence");
    }

    out.assign(p.n_options.size(), system_one_answer());
    std::vector<std::vector<float>> per_question(p.n_options.size());
    for (size_t i = 0; i < p.sequences.size(); i++) {
        const auto & seq = p.sequences[i];
        if (seq.question >= per_question.size()) {
            throw std::runtime_error("a planned sequence refers to a question that is not in the request");
        }
        per_question[seq.question].push_back(scores[i]);
    }

    for (size_t qi = 0; qi < per_question.size(); qi++) {
        if ((int) per_question[qi].size() != p.n_options[qi]) {
            throw std::runtime_error("a question did not get one score per option");
        }
        out[qi] = system_one_answer_from_scores(per_question[qi]);
    }
    return out;
}

// ---------------------------------------------------------------- readout

static std::vector<float> softmax(const std::vector<float> & x) {
    const float mx = *std::max_element(x.begin(), x.end());
    std::vector<float> p(x.size());
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); i++) { p[i] = std::exp(x[i] - mx); sum += p[i]; }
    for (auto & v : p) v /= sum;
    return p;
}

system_one_answer system_one_answer_from_scores(const std::vector<float> & scores) {
    system_one_answer a;
    a.logits = scores;                 // the head's raw output, as the caller gets it
    a.probs  = softmax(scores);
    a.choice = (int) (std::max_element(a.probs.begin(), a.probs.end()) - a.probs.begin());

    // confidence = 1 - H(p)/ln K
    if (scores.size() > 1) {
        double h = 0.0;
        for (float p : a.probs) if (p > 0.0f) h -= (double) p * std::log((double) p);
        a.confidence = (float) (1.0 - h / std::log((double) scores.size()));
    } else {
        a.confidence = 1.0f;
    }
    return a;
}

system_one_answer system_one_answer_from_logits(const float * row, const std::vector<llama_token> & letter_ids, int n_options) {
    // the only difference between the two readouts is where the numbers come from
    std::vector<float> scores(n_options);
    for (int j = 0; j < n_options; j++) scores[j] = row[letter_ids[j]];
    return system_one_answer_from_scores(scores);
}

void system_one_apply_temperature(system_one_answer & a, float t) {
    if (t <= 0.0f || std::fabs(t - 1.0f) <= 1e-6f || a.logits.empty()) {
        return;
    }

    std::vector<float> scaled(a.logits.size());
    for (size_t i = 0; i < scaled.size(); i++) scaled[i] = a.logits[i] / t;

    a.probs  = softmax(scaled);
    a.choice = (int) (std::max_element(a.probs.begin(), a.probs.end()) - a.probs.begin());

    if (a.probs.size() > 1) {
        double h = 0.0;
        for (float p : a.probs) if (p > 0.0f) h -= (double) p * std::log((double) p);
        a.confidence = (float) (1.0 - h / std::log((double) a.probs.size()));
    } else {
        a.confidence = 1.0f;
    }
}

float system_one_score_expectation(const system_one_answer & a) {
    double e = 0.0;
    for (size_t i = 0; i < a.probs.size(); i++) e += (double) i * (double) a.probs[i];
    return (float) e;
}

system_one_context_needs system_one_required_context(const system_one_params & cfg) {
    system_one_context_needs need;

    if (cfg.readout == SYSTEM_ONE_READOUT_RANK_HEAD) {
        need.embeddings   = true;
        need.pooling_type = LLAMA_POOLING_TYPE_RANK;
    }

    return need;
}

