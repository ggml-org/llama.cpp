// llama-trace-moe — GLM-5.2 / DeepSeek-MoE expert-routing tracer.
//
// Planned artifact referenced by GLM52_TRACE_PLAN.md (Phase 1, Story 1):
//   /Users/spotted/projects/llama.cpp/examples/trace-moe/
//
// Captures real MoE routing tensors (ffn_moe_topk, ffn_moe_weights) per
// (token, layer) via the ggml backend eval callback, pairs them, and writes a
// compact, asynchronous JSONL trace that gguf2mlx.tracing.analyze consumes.
//
// Heavy fields never run on the eval hot path: the callback only builds compact
// records and pushes them to a bounded queue; a dedicated writer thread drains
// the queue and writes JSONL + a <trace>.meta.json sidecar at the end.
//
// The --trace-* flags are pre-scanned out of argv so common_params_parse only
// sees standard llama-cli flags (-m, -p, -f, -ngl, -c, -n, --temp, --jinja,
// -cnv, -st, --chat-template-kwargs, ...).
//
// Build:
//   cmake --build build-metal --target llama-trace-moe
//
// Run:
//   build-metal/bin/llama-trace-moe \
//       -m /path/to/GLM-5.2-mixed-00001-of-00009.gguf -ngl 999 -c 32768 \
//       -f prompt.txt -n 256 --temp 0 --jinja -cnv -st \
//       --trace-out traces/run.jsonl --trace-task-label coding \
//       --trace-language en --trace-script Latin --trace-prompt-family coding \
//       --trace-test-id coding_01 --trace-phase both --trace-backpressure sample

#include "arg.h"
#include "common.h"
#include "log.h"
#include <filesystem>

#include "llama.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

constexpr int TRACE_SCHEMA_VERSION = 1;

enum class Backpressure { Block, Drop, Sample };

struct TraceConfig {
    std::string trace_out;
    std::string task_label     = "adhoc";
    std::string language       = "en";
    std::string script         = "Latin";
    std::string prompt_family  = "misc";
    std::string test_id        = "adhoc";
    std::string phase          = "both";  // prefill|generation|both
    std::string backpressure   = "sample";
    std::string trace_layers;             // "0,1,2,6" or "0..78" or "" = all
    int         trace_max_tokens = 0;     // 0 = unlimited
    // chat-template flags that this example (LLAMA_EXAMPLE_COMMON) cannot parse and
    // that the tracer would ignore anyway (it tokenizes params.prompt verbatim).
    // Collected here so main() can log one honest notice.
    std::vector<std::string> stripped_chat_flags;
};

// ---------------------------------------------------------------------------
// JSON helpers (no external json dependency; records are small and flat).
// ---------------------------------------------------------------------------
std::string join(const std::vector<std::string> & v, const std::string & sep) {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) out += sep;
        out += v[i];
    }
    return out;
}

void json_escape_append(const std::string & s, std::string & out) {
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
}

std::string render_record(
        const std::string & run_id, const std::string & model, const char * phase,
        int token_index, int layer,
        const std::vector<int32_t> & experts, const std::vector<float> & weights,
        float entropy, int n_expert_total,
        const TraceConfig & cfg) {
    std::string s;
    s.reserve(256 + experts.size() * 24);
    s += "{\"schema_version\":";
    s += std::to_string(TRACE_SCHEMA_VERSION);
    s += ",\"event\":\"moe_topk\"";
    s += ",\"run_id\":\"";     json_escape_append(run_id, s); s += "\"";
    s += ",\"model\":\"";      json_escape_append(model, s);  s += "\"";
    s += ",\"phase\":\"";      s += phase;                     s += "\"";
    s += ",\"token_index\":";  s += std::to_string(token_index);
    s += ",\"layer\":";        s += std::to_string(layer);
    s += ",\"task_label\":\""; json_escape_append(cfg.task_label, s);  s += "\"";
    s += ",\"language\":\"";   json_escape_append(cfg.language, s);    s += "\"";
    s += ",\"script\":\"";     json_escape_append(cfg.script, s);      s += "\"";
    s += ",\"prompt_family\":\""; json_escape_append(cfg.prompt_family, s); s += "\"";
    s += ",\"test_id\":\"";    json_escape_append(cfg.test_id, s);     s += "\"";
    s += ",\"n_expert_used\":"; s += std::to_string((int) experts.size());
    if (n_expert_total > 0) { s += ",\"n_expert\":"; s += std::to_string(n_expert_total); }
    // experts
    s += ",\"experts\":[";
    for (size_t i = 0; i < experts.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(experts[i]);
    }
    s += "],\"weights\":[";
    for (size_t i = 0; i < weights.size(); ++i) {
        if (i) s += ",";
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.6g", (double) weights[i]);
        s += buf;
    }
    s += "]";
    s += ",\"router_entropy\":"; { char buf[32]; std::snprintf(buf, sizeof(buf), "%.6g", (double) entropy); s += buf; }
    s += "}\n";
    return s;
}

// ---------------------------------------------------------------------------
// Async, bounded writer.
// ---------------------------------------------------------------------------
struct TraceWriter {
    std::FILE * fh = nullptr;
    std::string out_path;
    std::mutex  mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    size_t      max_queue = 8192;
    Backpressure bp = Backpressure::Sample;
    std::atomic<bool> stop{false};
    std::thread writer;
    std::atomic<uint64_t> written{0};
    std::atomic<uint64_t> dropped{0};
    std::atomic<uint64_t> sampled{0};

    bool open(const std::string & path, Backpressure b, size_t qsize) {
        out_path = path;
        bp = b;
        max_queue = qsize > 0 ? qsize : 8192;
        fh = std::fopen(path.c_str(), "w");
        if (!fh) return false;
        writer = std::thread([this]{ run(); });
        return true;
    }

    void run() {
        while (true) {
            std::string line;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [this]{ return stop.load() || !q.empty(); });
                if (q.empty() && stop.load()) break;
                line = std::move(q.front());
                q.pop_front();
            }
            std::fputs(line.c_str(), fh);
            ++written;
            if (written.load() % 64 == 0) std::fflush(fh);
        }
        if (std::fflush(fh) != 0) {
            LOG_ERR("%s: flush failed\n", __func__);
        }
    }

    void push(std::string && line) {
        bool accepted = false;
        if (bp == Backpressure::Sample) {
            // Adaptive sampling: when near-full, probabilistically shed load.
            double fill = 0.0;
            {
                std::lock_guard<std::mutex> lk(mu);
                fill = (double) q.size() / (double) max_queue;
            }
            if (fill > 0.9) {
                // keep roughly 1 of every 2 under pressure
                if ((written.load() ^ dropped.load()) & 1u) {
                    ++sampled;
                    return;
                }
            }
            std::lock_guard<std::mutex> lk(mu);
            if (q.size() < max_queue) { q.emplace_back(std::move(line)); accepted = true; }
        } else if (bp == Backpressure::Drop) {
            std::lock_guard<std::mutex> lk(mu);
            if (q.size() < max_queue) { q.emplace_back(std::move(line)); accepted = true; }
        } else { // Block
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [this]{ return q.size() < max_queue || stop.load(); });
            if (!stop.load()) { q.emplace_back(std::move(line)); accepted = true; }
        }
        if (accepted) cv.notify_one();
        else ++dropped;
    }

    void close() {
        stop.store(true);
        cv.notify_all();
        if (writer.joinable()) writer.join();
        if (fh) { std::fclose(fh); fh = nullptr; }
    }
};

// ---------------------------------------------------------------------------
// Trace state shared with the eval callback.
// ---------------------------------------------------------------------------
struct TraceState {
    std::string run_id;
    std::string model_name;
    TraceConfig cfg;
    TraceWriter writer;

    // decode loop signals
    const char * current_phase = "prefill";  // "prefill" | "generation"
    int token_base = 0;        // token index of the first token of the current batch
    int batch_n_tokens = 0;    // n_tokens of the batch currently being decoded
    int phase_tokens_emitted[2] = {0, 0}; // [prefill, generation]
    int max_new_tokens = 0;    // mirror of params.n_predict
    int n_expert_total = 0;

    std::vector<int> selected_layers; // empty = all
    bool trace_prefill = true;
    bool trace_generation = true;

    // per-batch pending topk buffer keyed by layer
    std::unordered_map<int, std::vector<int32_t>> pending_topk;
    std::unordered_map<int, int> pending_topk_n_used;   // n_used of the pending topk
    std::unordered_map<int, int> pending_topk_n_tokens; // n_tokens of the pending topk

    // (readback now uses a per-call fresh buffer; no shared scratch field)
};

int phase_index(const char * phase) {
    return std::strcmp(phase, "generation") == 0 ? 1 : 0;
}

int extract_layer(const char * name) {
    // tensors in the eval callback are named like "ffn_moe_topk-42" or "ffn_moe_weights-7"
    std::string s(name);
    auto pos = s.find_last_of('-');
    if (pos == std::string::npos) return -1;
    try { return std::stoi(s.substr(pos + 1)); }
    catch (...) { return -1; }
}

bool is_topk_tensor(const char * name) {
    std::string s(name);
    return s.rfind("ffn_moe_topk", 0) == 0;
}
bool is_weights_tensor(const char * name) {
    // avoid the _norm/_softmax/_sum variants: exact base name + layer suffix only
    std::string s(name);
    if (s.rfind("ffn_moe_weights", 0) != 0) return false;
    // accept "ffn_moe_weights" or "ffn_moe_weights-N"
    if (s.size() == std::string("ffn_moe_weights").size()) return true;
    return s[std::string("ffn_moe_weights").size()] == '-';
}

bool layer_selected(const TraceState & st, int layer) {
    if (st.selected_layers.empty()) return true;
    for (int l : st.selected_layers) if (l == layer) return true;
    return false;
}

float compute_entropy(const std::vector<float> & w) {
    double total = 0.0;
    for (float x : w) total += x;
    if (total <= 0) return 0.0f;
    double h = 0.0;
    for (float x : w) {
        if (x <= 0) continue;
        double p = x / total;
        h -= p * std::log2(p);
    }
    return (float) h;
}

// Emit ONE record for a single (token-in-batch, layer) routing event.
// topk_ptr/weights_ptr each have n_used contiguous elements for this token.
void emit_one(TraceState & st, int layer, int n_used, int tok_in_batch,
             const int32_t * topk_ptr, const float * weights_ptr) {
    int pi = phase_index(st.current_phase);
    bool phase_on = (pi == 0) ? st.trace_prefill : st.trace_generation;
    if (!phase_on) return;
    if (st.cfg.trace_max_tokens > 0 && st.phase_tokens_emitted[pi] >= st.cfg.trace_max_tokens) {
        return;
    }
    std::vector<int32_t> experts(n_used);
    std::vector<float>    weights(n_used);
    for (int k = 0; k < n_used; ++k) {
        experts[k] = topk_ptr[k];
        weights[k] = weights_ptr[k];
    }
    float ent = compute_entropy(weights);
    std::string rec = render_record(
        st.run_id, st.model_name, st.current_phase,
        st.token_base + tok_in_batch, layer, experts, weights, ent, st.n_expert_total, st.cfg);
    st.writer.push(std::move(rec));
    st.phase_tokens_emitted[pi]++;
}

bool trace_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) return true;  // we are interested when data is available
    auto * st = static_cast<TraceState *>(user_data);
    const char * name = t->name;
    if (!name || !name[0]) return true;
    if (!is_topk_tensor(name) && !is_weights_tensor(name)) return true;

    int layer = extract_layer(name);
    if (layer < 0) return true;

    if (!layer_selected(*st, layer)) return true;

    // ALWAYS copy to a fresh host buffer via ggml_backend_tensor_get. We never
    // read t->data directly even when ggml_backend_buffer_is_host(t->buffer) is
    // true, because on the Metal backend intermediate MoE tensors can live in
    // shared host memory whose GPU write has not landed by the time the eval
    // callback fires (this produced garbage expert IDs during fast single-token
    // generation decodes). ggml_backend_tensor_get performs the backend's sync.
    // The MoE topk/weights tensors are byte-contiguous in the [n_used, n_tokens]
    // frame (verified for GLM-5.2: topk ne=[N,T] nb=[4,N*4]; weights ne=[1,N,T]
    // nb=[4,4,N*4] — the degenerate dim0 makes dim0+dim1 a flat N-element run),
    // so a flat element read by `idx = k + tok*n_used` is correct after the copy.
    size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> raw(nbytes);
    ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
    const uint8_t * data = raw.data();

    if (is_topk_tensor(name)) {
        // I32 [n_used, n_tokens] (contiguous)
        if (t->type != GGML_TYPE_I32) return true;
        int n_used   = (int) t->ne[0];
        int n_tokens = (int) t->ne[1];
        size_t n = (size_t) n_used * (size_t) n_tokens;
        std::vector<int32_t> topk(n);
        const int32_t * src = reinterpret_cast<const int32_t *>(data);
        for (size_t i = 0; i < n; ++i) topk[i] = src[i];
        st->pending_topk[layer] = std::move(topk);
        st->pending_topk_n_used[layer] = n_used;
        st->pending_topk_n_tokens[layer] = n_tokens;
        return true;
    }

    // weights tensor (F32). Its ne may be [1, n_used, n_tokens] (degenerate
    // dim0) but the copied bytes are contiguous in [n_used, n_tokens], so we
    // read flat using the pending TOPK's n_used/n_tokens (the source of truth
    // for which experts were selected and how many tokens are in this batch).
    if (t->type != GGML_TYPE_F32) return true;
    auto it = st->pending_topk.find(layer);
    if (it == st->pending_topk.end()) {
        return true;  // weights without pending topk — drop
    }
    std::vector<int32_t> topk = std::move(it->second);
    st->pending_topk.erase(it);
    int n_used   = st->pending_topk_n_used[layer];
    int n_tokens = st->pending_topk_n_tokens[layer];
    const float * wsrc = reinterpret_cast<const float *>(data);
    size_t total_elems = nbytes / sizeof(float);
    for (int tok = 0; tok < n_tokens; ++tok) {
        std::vector<float> w(n_used, 0.0f);
        for (int k = 0; k < n_used; ++k) {
            size_t idx = (size_t) k + (size_t) tok * (size_t) n_used;
            if (idx < total_elems) w[k] = wsrc[idx];
        }
        const int32_t * tk = topk.data() + (size_t) tok * (size_t) n_used;
        emit_one(*st, layer, n_used, tok, tk, w.data());
    }
    return true;
}

// ---------------------------------------------------------------------------
// argv pre-scan for --trace-* flags.
// ---------------------------------------------------------------------------
TraceConfig config_from_trace_flags(
        std::vector<std::string> & trace_args,
        int & argc, char ** & argv_out) {
    TraceConfig cfg;
    // rebuild argv without --trace-* args
    static thread_local std::vector<char*> kept;
    kept.clear();
    kept.push_back(argv_out[0]);
    for (size_t i = 1; i < trace_args.size(); ++i) {
        const std::string & a = trace_args[i];
        auto take_next = [&](std::string & dst) {
            if (i + 1 < trace_args.size()) { dst = trace_args[++i]; }
        };
        if (a == "--trace-out")              take_next(cfg.trace_out);
        else if (a == "--trace-task-label")  take_next(cfg.task_label);
        else if (a == "--trace-language")    take_next(cfg.language);
        else if (a == "--trace-script")      take_next(cfg.script);
        else if (a == "--trace-prompt-family") take_next(cfg.prompt_family);
        else if (a == "--trace-test-id")     take_next(cfg.test_id);
        else if (a == "--trace-phase")       take_next(cfg.phase);
        else if (a == "--trace-backpressure") take_next(cfg.backpressure);
        else if (a == "--trace-layers")      take_next(cfg.trace_layers);
        else if (a == "--trace-max-tokens")  {
            std::string tmp; take_next(tmp); cfg.trace_max_tokens = std::atoi(tmp.c_str());
        }
        // These flags are set_examples()-restricted to CLI/SERVER/MTMD and the
        // tracer would ignore them regardless (it never applies a chat template).
        // Strip them so llama-cli-style invocations don't fail with
        // "invalid argument"; main() warns the user they had no effect.
        else if (a == "--jinja" || a == "--no-jinja" ||
                 a == "-cnv" || a == "--conversation" ||
                 a == "-no-cnv" || a == "--no-conversation" ||
                 a == "-st" || a == "--single-turn") {
            cfg.stripped_chat_flags.push_back(a);
        }
        else if (a == "--chat-template-kwargs") {
            cfg.stripped_chat_flags.push_back(a);
            std::string tmp; take_next(tmp);  // swallow the JSON value
        }
        else {
            // keep this argument for common_params_parse
            kept.push_back(const_cast<char*>(a.c_str()));
        }
    }
    argv_out = kept.data();
    argc = (int) kept.size();
    return cfg;
}

std::vector<int> parse_layers(const std::string & spec) {
    std::vector<int> out;
    if (spec.empty()) return out;
    std::string s = spec;
    // handle "0..78"
    auto dd = s.find("..");
    if (dd != std::string::npos) {
        int lo = std::atoi(s.c_str());
        int hi = std::atoi(s.c_str() + dd + 2);
        for (int i = lo; i <= hi; ++i) out.push_back(i);
        return out;
    }
    std::string cur;
    for (char c : s + ",") {
        if (c == ',') { if (!cur.empty()) { out.push_back(std::atoi(cur.c_str())); cur.clear(); } }
        else cur += c;
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // Pre-scan to peel --trace-* off argv before common_params_parse sees them.
    std::vector<std::string> raw_args(argv, argv + argc);
    char ** argv_kept = argv;
    int argc_kept = argc;
    TraceConfig cfg = config_from_trace_flags(raw_args, argc_kept, argv_kept);

    if (cfg.trace_out.empty()) {
        LOG_ERR("ERROR: --trace-out <path.jsonl> is required\n");
        return 2;
    }

    common_params params;
    common_init();
    if (!common_params_parse(argc_kept, argv_kept, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    if (!cfg.stripped_chat_flags.empty()) {
        LOG_WRN("%s: ignoring chat-template flags %s (the tracer tokenizes "
                "params.prompt verbatim; it does not apply a chat template). "
                "For templated prompts, pre-template and pass via -p. "
                "See traces/README.md.",
                __func__, join(cfg.stripped_chat_flags, ", ").c_str());
    }
    if (params.prompt.empty() && params.in_files.empty()) {
        // require a prompt
        LOG_ERR("ERROR: provide -p <prompt> or -f <file>\n");
        return 2;
    }

    // backpressure
    Backpressure bp = Backpressure::Sample;
    if (cfg.backpressure == "block") bp = Backpressure::Block;
    else if (cfg.backpressure == "drop") bp = Backpressure::Drop;
    else bp = Backpressure::Sample;

    TraceState st;
    st.cfg = cfg;
    st.run_id = cfg.test_id + "-" + cfg.language + "-" + std::to_string((long long) std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    const std::string model_name = std::filesystem::path(params.model.path).parent_path().filename().string();
    st.model_name = model_name.empty() ? params.model.path : model_name;
    st.max_new_tokens = params.n_predict;
    st.selected_layers = parse_layers(cfg.trace_layers);
    st.trace_prefill = (cfg.phase == "prefill" || cfg.phase == "both");
    st.trace_generation = (cfg.phase == "generation" || cfg.phase == "both");

    if (!st.writer.open(cfg.trace_out, bp, /*qsize*/ 8192)) {
        LOG_ERR("ERROR: could not open trace output: %s\n", cfg.trace_out.c_str());
        return 2;
    }

    params.cb_eval = trace_cb_eval;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    auto t_start_mono = std::chrono::steady_clock::now();

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("ERROR: failed to init llama\n");
        st.writer.close();
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    // Note: there is no stable public llama.cpp API to read n_expert_total directly;
    // leave 0 here — the analyzer tolerates an absent n_expert field.
    (void) model;

    // ---- tokenize prompt ----
    // common_params_parse already loads -p/--prompt and -f/--file into params.prompt,
    // so we tokenize that verbatim (this mirrors examples/eval-callback, which also
    // does NOT apply a chat template). For chat-template parity with llama-cli, use
    // scripts/tracing/run_glm52_moe_trace.sh, which relies on --jinja -cnv being
    // honored by common_init_from_params for generation.
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    if (tokens.empty()) {
        LOG_ERR("ERROR: no input tokens — provide -p <prompt> or -f <file>\n");
        st.writer.close();
        return 1;
    }
    LOG_INF("%s: prompt tokens = %zu\n", __func__, tokens.size());

    // ---- sampler for generation ----
    struct llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(params.sampling.temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    uint64_t prompt_n_tokens = tokens.size();

    // ---------------- prefill ----------------
    if (st.trace_prefill && !tokens.empty()) {
        st.current_phase = "prefill";
        st.token_base = 0;
        st.batch_n_tokens = (int) tokens.size();
        llama_batch batch = llama_batch_get_one(tokens.data(), (int) tokens.size());
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("ERROR: prefill decode failed\n");
        }
    } else if (!tokens.empty()) {
        // still need a prefill decode to seed KV cache for generation
        llama_batch batch = llama_batch_get_one(tokens.data(), (int) tokens.size());
        llama_decode(ctx, batch);
    }

    // ---------------- generation ----------------
    uint64_t gen_n_tokens = 0;
    if (st.trace_generation && params.n_predict > 0) {
        st.current_phase = "generation";
        // seed generation from the last logits of prefill (greedy)
        llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
        for (int step = 0; step < params.n_predict; ++step) {
            if (llama_vocab_is_eog(vocab, new_token_id)) break;
            if (st.cfg.trace_max_tokens > 0 && st.phase_tokens_emitted[1] >= st.cfg.trace_max_tokens) {
                // keep generating for KV but stop tracing? For simplicity stop generation too.
                break;
            }
            st.token_base = (int) prompt_n_tokens + step;
            st.batch_n_tokens = 1;
            llama_batch batch = llama_batch_get_one(&new_token_id, 1);
            int ret = llama_decode(ctx, batch);
            if (ret != 0) {
                LOG_WRN("%s: generation decode failed at step %d (ret=%d)\n", __func__, step, ret);
                break;
            }
            gen_n_tokens++;
            new_token_id = llama_sampler_sample(smpl, ctx, -1);
        }
    }

    auto t_end_mono = std::chrono::steady_clock::now();
    double wall = std::chrono::duration<double>(t_end_mono - t_start_mono).count();

    llama_sampler_free(smpl);

    st.writer.close();

    // ---------------- metadata sidecar ----------------
    std::string meta = cfg.trace_out + ".meta.json";
    std::FILE * mf = std::fopen(meta.c_str(), "w");
    if (mf) {
        std::fprintf(mf, "{\n");
        std::fprintf(mf, "  \"schema_version\": %d,\n", TRACE_SCHEMA_VERSION);
        std::fprintf(mf, "  \"run_id\": \"%s\",\n", st.run_id.c_str());
        std::fprintf(mf, "  \"model\": \"%s\",\n", st.model_name.c_str());
        std::fprintf(mf, "  \"model_path\": \"%s\",\n", params.model.path.c_str());
        std::fprintf(mf, "  \"command_line\": \"llama-trace-moe ...\",\n");
        std::fprintf(mf, "  \"prompt_sha256\": \"(see run log)\",\n");
        std::fprintf(mf, "  \"task_label\": \"%s\",\n", cfg.task_label.c_str());
        std::fprintf(mf, "  \"language\": \"%s\",\n", cfg.language.c_str());
        std::fprintf(mf, "  \"script\": \"%s\",\n", cfg.script.c_str());
        std::fprintf(mf, "  \"prompt_family\": \"%s\",\n", cfg.prompt_family.c_str());
        std::fprintf(mf, "  \"test_id\": \"%s\",\n", cfg.test_id.c_str());
        // Derive thinking mode best-effort from the default chat-template kwargs map.
        std::string thinking_mode = "unknown";
        std::string reasoning_effort = "unknown";
        {
            auto it = params.default_template_kwargs.find("enable_thinking");
            if (it != params.default_template_kwargs.end()) {
                thinking_mode = (it->second == "false" || it->second == "0") ? "disabled" : "enabled";
            }
            auto ir = params.default_template_kwargs.find("reasoning_effort");
            if (ir != params.default_template_kwargs.end()) {
                reasoning_effort = ir->second.empty() ? "none" : ir->second;
            }
        }
        std::fprintf(mf, "  \"thinking_mode\": \"%s\",\n", thinking_mode.c_str());
        std::fprintf(mf, "  \"reasoning_effort\": \"%s\",\n", reasoning_effort.c_str());
        std::fprintf(mf, "  \"max_new_tokens\": %d,\n", params.n_predict);
        std::fprintf(mf, "  \"phase_modes\": {\"prefill\": %s, \"generation\": %s},\n",
            st.trace_prefill ? "true" : "false",
            st.trace_generation ? "true" : "false");
        std::fprintf(mf, "  \"trace_layers\": \"%s\",\n",
            cfg.trace_layers.empty() ? "all" : cfg.trace_layers.c_str());
        std::fprintf(mf, "  \"trace_max_tokens\": %d,\n", cfg.trace_max_tokens);
        std::fprintf(mf, "  \"backpressure\": \"%s\",\n", cfg.backpressure.c_str());
        std::fprintf(mf, "  \"queue_size\": %zu,\n", st.writer.max_queue);
        std::fprintf(mf, "  \"records_written\": %llu,\n", (unsigned long long) st.writer.written.load());
        std::fprintf(mf, "  \"records_dropped\": %llu,\n", (unsigned long long) st.writer.dropped.load());
        std::fprintf(mf, "  \"records_sampled\": %llu,\n", (unsigned long long) st.writer.sampled.load());
        std::fprintf(mf, "  \"prompt_token_count\": %llu,\n", (unsigned long long) prompt_n_tokens);
        std::fprintf(mf, "  \"gen_token_count\": %llu,\n", (unsigned long long) gen_n_tokens);
        std::fprintf(mf, "  \"wall_seconds\": %.3f,\n", wall);
        std::fprintf(mf, "  \"cli_path\": \"llama-trace-moe\",\n");
        std::fprintf(mf, "  \"cli_build\": \"patched llama.cpp build-metal\"\n");
        std::fprintf(mf, "}\n");
        std::fclose(mf);
    }

    LOG("%s: trace written to %s (%llu records, %llu dropped, %llu sampled, %.2fs)\n",
        __func__, cfg.trace_out.c_str(),
        (unsigned long long) st.writer.written.load(),
        (unsigned long long) st.writer.dropped.load(),
        (unsigned long long) st.writer.sampled.load(),
        wall);

    llama_backend_free();
    return 0;
}
