#include "llama.h"
#include "llama-cpp.h"
#include "ggml-cpp.h"
#include "sampler-bench-common.h"

#include "../../src/llama-ext.h"
#include "../../src/llama-grammar.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using bench_clock = std::chrono::steady_clock;

enum comp : int {
    C_EVAL, C_D2H, C_INIT, C_RETAIN, C_PREP, C_MASK, C_UPLOAD, C_GRAPH, C_READBACK, C_CHAIN, C_VALIDATE,
    C_RETRY_STATE, C_RETRY_INIT, C_RETRY_PREP, C_RETRY_MASK, C_RETRY_COPY, C_RETRY_UPLOAD, C_RETRY_GRAPH,
    C_RETRY_READBACK, C_RETRY_CHAIN, C_ACCEPT, C_RESET, C_COUNT,
};

static const char * comp_names[C_COUNT] = {
    "eval", "logits_d2h", "cand_init", "retain_copy", "mask_prep", "mask_apply", "upload", "graph", "readback",
    "chain", "validate", "retry_state", "retry_cand_init", "retry_mask_prep", "retry_mask_apply", "retry_logits_copy",
    "retry_upload", "retry_graph", "retry_readback", "retry_chain", "accept", "reset",
};

enum policy { POLICY_STC, POLICY_GF };
static const char * policy_names[] = {"sample-then-check", "grammar-first"};

enum path_kind { PATH_CPU_SCAN, PATH_CPU_TRIE, PATH_BACKEND };
static const char * path_names[] = {"cpu-scan", "cpu-trie", "backend"};

struct step_timer {
    std::array<double, C_COUNT> us = {};
    bench_clock::time_point t;

    void start() { t = bench_clock::now(); }
    void lap(comp c) {
        const auto now = bench_clock::now();
        us[c] += std::chrono::duration<double, std::micro>(now - t).count();
        t = now;
    }
};

struct step_result {
    llama_token proposal = LLAMA_TOKEN_NULL; // unconstrained pick (sample-then-check only)
    llama_token selected = LLAMA_TOKEN_NULL;
    bool retried = false;
};

struct trace_step {
    llama_token expected;
    llama_token forbidden;
};

struct options {
    std::string model_path;
    std::string tools_path = "tools/test-grammar-tools.json";
    std::string device_name = "gpu";
    std::string jsonl_path;
    std::string replay_path;
    std::string dump_path;
    std::vector<double> rates = {0.0, 0.01, 0.1, 0.5, 1.0};
    std::vector<policy> policies = {POLICY_STC, POLICY_GF};
    std::vector<path_kind> paths = {PATH_CPU_SCAN, PATH_CPU_TRIE, PATH_BACKEND};
    int steps = 1024;
    int repetitions = 1;
    int top_k = 40;
    float top_p = 0.95f;
    float min_p = 0.05f;
    double eval_us = 0.0;
    bool logits_copy = true;
    bool device_mask = false;
    bool jsonl_steps = true;
    int n_gpu_layers = 99;
    uint64_t seed = 1234;
};

static llama_sampler_ptr make_chain(const options & opt) {
    auto params = llama_sampler_chain_default_params();
    params.no_perf = true;
    llama_sampler_ptr chain(llama_sampler_chain_init(params));
    if (opt.top_k > 0) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_top_k(opt.top_k));
    }
    if (opt.top_p < 1.0f) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_top_p(opt.top_p, 1));
    }
    if (opt.min_p > 0.0f) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_min_p(opt.min_p, 1));
    }
    llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
    return chain;
}

static std::string chain_description(const options & opt) {
    std::string s;
    if (opt.top_k > 0)     { s += "top-k=" + std::to_string(opt.top_k) + " "; }
    if (opt.top_p < 1.0f)  { s += "top-p=" + std::to_string(opt.top_p).substr(0, 4) + " "; }
    if (opt.min_p > 0.0f)  { s += "min-p=" + std::to_string(opt.min_p).substr(0, 4) + " "; }
    return s + "greedy";
}

// Grammar replay: expected tokens plus one grammar-rejected token per position. Also checks that the trie mask used by the backend path equals the candidate scan used by the host sampler at every replayed state.
static std::vector<trace_step> make_trace(const llama_vocab * vocab, const std::string & grammar, const std::string & text) {
    int32_t count = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, false, false);
    require(count < 0, "fixture tokenization failed");
    std::vector<llama_token> tokens(-count);
    require(llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), false, false) == -count,
            "fixture tokenization length changed");
    require(llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL, "model has no EOS token");
    tokens.push_back(llama_vocab_eos(vocab));

    llama_sampler_ptr oracle(llama_sampler_init_grammar(vocab, grammar.c_str(), "root"));
    require(bool(oracle), "cannot initialize grammar");
    std::unique_ptr<llama_grammar, decltype(&llama_grammar_free_impl)> trie(
            llama_grammar_init_impl(vocab, grammar.c_str(), "root", false, nullptr, 0, nullptr, 0), llama_grammar_free_impl);
    require(bool(trie), "cannot initialize grammar");
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token_data> candidates(n_vocab);
    std::vector<trace_step> trace;
    for (const auto token : tokens) {
        for (int32_t id = 0; id < n_vocab; ++id) {
            candidates[id] = {id, 0.0f, 0.0f};
        }
        llama_token_data_array cur = {candidates.data(), candidates.size(), -1, false};
        llama_sampler_apply(oracle.get(), &cur);
        const auto words = llama_grammar_get_allowed_tokens(*trie);
        for (int32_t id = 0; id < n_vocab; ++id) {
            const bool scan = std::isfinite(candidates[id].logit);
            const bool bit = (words[id/32] >> (id%32)) & 1;
            require(scan == bit, "trie mask differs from candidate scan at trace token " + std::to_string(trace.size()) +
                    ", vocab id " + std::to_string(id));
        }
        require(std::isfinite(candidates[token].logit), "fixture rejected at token " + std::to_string(trace.size()));
        llama_token forbidden = LLAMA_TOKEN_NULL;
        for (const auto & candidate : candidates) {
            if (candidate.logit == -INFINITY) {
                forbidden = candidate.id;
                break;
            }
        }
        require(forbidden != LLAMA_TOKEN_NULL, "fixture state has no rejected token");
        trace.push_back({token, forbidden});
        llama_sampler_accept(oracle.get(), token);
        llama_grammar_accept_impl(*trie, token);
    }
    return trace;
}

// Model logits replay file: magic, version, n_vocab, n_steps, tokens[n_steps], f32 logits[n_steps][n_vocab].
static constexpr uint32_t REPLAY_MAGIC = 0x524c474c; // "LGLR"

struct replay_data {
    uint32_t n_vocab = 0;
    std::vector<llama_token> tokens;
    std::vector<float> logits;
};

static replay_data load_replay(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    require(in.good(), "cannot open replay file " + path);
    uint32_t header[4];
    in.read((char *) header, sizeof(header));
    require(in.good() && header[0] == REPLAY_MAGIC && header[1] == 1, "not a replay logits file: " + path);
    replay_data data;
    data.n_vocab = header[2];
    data.tokens.resize(header[3]);
    data.logits.resize(size_t(header[2])*header[3]);
    in.read((char *) data.tokens.data(), data.tokens.size()*sizeof(llama_token));
    in.read((char *) data.logits.data(), data.logits.size()*sizeof(float));
    require(in.good(), "truncated replay file " + path);
    return data;
}

// Teacher-forced model run: logits that predict each trace token after a chat prompt listing the tools.
static void dump_logits(const options & opt, const common_json & fixture, const std::vector<trace_step> & trace) {
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = opt.n_gpu_layers;
    mparams.load_mode = LLAMA_LOAD_MODE_DIRECT_IO;
    llama_model_ptr model(llama_model_load_from_file(opt.model_path.c_str(), mparams));
    require(bool(model), "cannot load model");
    const auto * vocab = llama_model_get_vocab(model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    const std::string system = "You are a tool-calling assistant. Reply only with a JSON array of tool calls.\nTools:\n" +
        fixture.at("tools").dump();
    const std::string user = "Perform every tool call needed for the task, in order.";
    llama_chat_message msgs[] = {{"system", system.c_str()}, {"user", user.c_str()}};
    std::vector<char> buf(2*(system.size() + user.size()) + 256);
    int32_t len = llama_chat_apply_template("chatml", msgs, 2, true, buf.data(), buf.size());
    require(len > 0 && size_t(len) <= buf.size(), "cannot apply chat template");
    const std::string prompt = std::string(buf.data(), len) + "<think>\n\n</think>\n\n";
    int32_t count = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokens(count);
    require(llama_tokenize(vocab, prompt.data(), prompt.size(), tokens.data(), count, true, true) == count, "prompt tokenization failed");
    const size_t n_prompt = tokens.size();
    for (size_t i = 0; i + 1 < trace.size(); ++i) {
        tokens.push_back(trace[i].expected);
    }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = tokens.size() + 64;
    cparams.n_batch = 512;
    cparams.n_ubatch = 512;
    cparams.no_perf = true;
    llama_context_ptr ctx(llama_init_from_model(model.get(), cparams));
    require(bool(ctx), "cannot create context");

    std::ofstream out(opt.dump_path, std::ios::binary);
    require(out.good(), "cannot write " + opt.dump_path);
    const uint32_t header[4] = {REPLAY_MAGIC, 1, uint32_t(n_vocab), uint32_t(trace.size())};
    out.write((const char *) header, sizeof(header));
    for (const auto & step : trace) {
        out.write((const char *) &step.expected, sizeof(llama_token));
    }
    llama_batch batch = llama_batch_init(512, 0, 1);
    size_t written = 0;
    for (size_t start = 0; start < tokens.size(); start += 512) {
        batch.n_tokens = 0;
        for (size_t i = start; i < std::min(tokens.size(), start + 512); ++i) {
            const int32_t j = batch.n_tokens++;
            batch.token[j] = tokens[i];
            batch.pos[j] = i;
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = i + 1 >= n_prompt;
        }
        require(llama_decode(ctx.get(), batch) == 0, "decode failed");
        for (int32_t j = 0; j < batch.n_tokens; ++j) {
            if (batch.logits[j]) {
                const float * row = llama_get_logits_ith(ctx.get(), j);
                require(row != nullptr, "missing logits");
                out.write((const char *) row, n_vocab*sizeof(float));
                ++written;
            }
        }
    }
    llama_batch_free(batch);
    require(written == trace.size() && out.good(), "logit dump incomplete");
    printf("dumped %zu teacher-forced logit rows (prompt %zu tokens) to %s\n", written, n_prompt, opt.dump_path.c_str());
}

struct sampler_graph {
    std::vector<llama_sampler *> samplers;
    ggml_backend_t backend;
    ggml_context_ptr ctx;
    ggml_backend_ptr cpu;
    ggml_backend_sched_ptr sched;
    ggml_backend_buffer_ptr buffer;
    ggml_cgraph * gf = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * sampled = nullptr;

    sampler_graph(std::vector<llama_sampler *> samplers_, ggml_backend_t backend, ggml_tensor * logits, int32_t n_vocab)
            : samplers(std::move(samplers_)), backend(backend) {
        auto * buft = ggml_backend_get_default_buffer_type(backend);
        for (auto * smpl : samplers) {
            require(smpl->iface->backend_init(smpl, buft, 1),
                    std::string("backend does not support sampler ") + llama_sampler_name(smpl));
        }
        const size_t max_nodes = GGML_DEFAULT_GRAPH_SIZE;
        ggml_init_params params = {max_nodes*ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false), nullptr, true};
        ctx.reset(ggml_init(params));
        require(bool(ctx), "cannot create graph context");
        gf = ggml_new_graph_custom(ctx.get(), max_nodes, false);
        input = logits;
        if (!input) {
            input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_vocab);
            ggml_set_input(input);
        }
        llama_sampler_data data = {input, nullptr, nullptr, nullptr};
        for (auto * smpl : samplers) {
            smpl->iface->backend_apply(smpl, ctx.get(), gf, &data);
            require(!data.skip, "backend sampling row unusable");
        }
        require(data.sampled != nullptr, "backend chain did not select a token");
        sampled = data.sampled;
        ggml_set_output(sampled);
        ggml_build_forward_expand(gf, sampled);
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            auto * node = ggml_graph_node(gf, i);
            require(ggml_backend_supports_op(backend, node), "unsupported backend op " + std::string(ggml_op_name(node->op)));
        }
        if (logits) {
            if (ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                cpu.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
                require(bool(cpu), "cannot initialize CPU backend");
            }
            ggml_backend_t backends[] = {backend, cpu.get()};
            sched.reset(ggml_backend_sched_new(backends, nullptr, cpu ? 2 : 1, GGML_DEFAULT_GRAPH_SIZE, false, true));
            require(bool(sched), "cannot create scheduler");
            for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
                ggml_backend_sched_set_tensor_backend(sched.get(), ggml_graph_node(gf, i), backend);
            }
            require(ggml_backend_sched_alloc_graph(sched.get(), gf), "cannot allocate sampler graph");
        } else {
            buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
            require(bool(buffer), "cannot allocate sampler graph");
        }
    }

    ~sampler_graph() {
        for (auto * smpl : samplers) {
            smpl->iface->backend_reset(smpl);
        }
    }

    void set_input() {
        for (auto * smpl : samplers) {
            smpl->iface->backend_set_input(smpl, sched.get());
        }
    }

    void compute() {
        const auto status = sched ? ggml_backend_sched_graph_compute(sched.get(), gf) : ggml_backend_graph_compute(backend, gf);
        require(status == GGML_STATUS_SUCCESS, "sampler graph failed");
    }

    llama_token read() const {
        llama_token token;
        ggml_backend_tensor_get(sampled, &token, 0, sizeof(token));
        return token;
    }
};

struct logits_env {
    int32_t n_vocab;
    bool copy_to_host;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * model = nullptr;    // model output logits
    ggml_tensor * retained = nullptr; // llama_context::retain_logits target
    std::vector<float> host;          // ctx->logits
    const float * master = nullptr;   // reference values (synthetic or replay row)

    logits_env(ggml_backend_t backend, int32_t n_vocab, bool copy_to_host) : n_vocab(n_vocab), copy_to_host(copy_to_host), host(n_vocab) {
        ggml_init_params params = {4*ggml_tensor_overhead(), nullptr, true};
        ctx.reset(ggml_init(params));
        model = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_vocab);
        retained = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_vocab);
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        require(bool(buffer), "cannot allocate logits buffer");
    }

    const float * for_host(step_timer & t) {
        if (!copy_to_host) {
            return master;
        }
        ggml_backend_tensor_get(model, host.data(), 0, size_t(n_vocab)*sizeof(float));
        t.lap(C_D2H);
        return host.data();
    }
};

struct runner {
    path_kind kind;
    policy pol;
    virtual ~runner() = default;
    virtual void reset() = 0;
    virtual step_result step(step_timer & t, logits_env & env, llama_token accept) = 0;
    size_t cache_hits = 0, cache_misses = 0;
};

static void init_candidates(std::vector<llama_token_data> & cand, const float * logits, int32_t n_vocab) {
    for (int32_t id = 0; id < n_vocab; ++id) {
        cand[id] = {id, logits[id], 0.0f};
    }
}

struct cpu_runner : runner {
    const llama_vocab * vocab;
    std::string grammar_str;
    llama_sampler_ptr grammar;
    std::unique_ptr<llama_grammar, decltype(&llama_grammar_free_impl)> raw {nullptr, llama_grammar_free_impl};
    llama_sampler_ptr chain;
    std::vector<llama_token_data> cand;

    cpu_runner(path_kind kind_, policy pol_, const llama_vocab * vocab, const std::string & grammar_str, const options & opt)
            : vocab(vocab), grammar_str(grammar_str), chain(make_chain(opt)), cand(llama_vocab_n_tokens(vocab)) {
        kind = kind_;
        pol = pol_;
        if (kind == PATH_CPU_SCAN) {
            grammar.reset(llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root"));
            require(bool(grammar), "cannot initialize grammar");
        }
        reset();
    }

    void reset() override {
        if (kind == PATH_CPU_SCAN) {
            llama_sampler_reset(grammar.get());
        } else if (raw) {
            llama_grammar_reset_impl(*raw);
        } else {
            raw.reset(llama_grammar_init_impl(vocab, grammar_str.c_str(), "root", false, nullptr, 0, nullptr, 0));
            require(bool(raw), "cannot initialize grammar");
        }
        llama_sampler_reset(chain.get());
    }

    void flush_stats() {
        if (raw) {
            cache_hits += raw->mask_cache->hits;
            cache_misses += raw->mask_cache->misses;
            raw->mask_cache->hits = raw->mask_cache->misses = 0;
        }
    }

    bool valid(llama_token token) {
        llama_token_data candidate = {token, 0.0f, 0.0f};
        llama_token_data_array single = {&candidate, 1, -1, false};
        if (kind == PATH_CPU_SCAN) {
            llama_sampler_apply(grammar.get(), &single);
        } else {
            llama_grammar_apply_impl(*raw, &single);
        }
        return candidate.logit != -INFINITY;
    }

    void mask(llama_token_data_array & cur, step_timer & t, comp prep, comp apply) {
        if (kind == PATH_CPU_SCAN) {
            llama_sampler_apply(grammar.get(), &cur);
            t.lap(prep);
            return;
        }
        const auto words = llama_grammar_get_allowed_tokens(*raw);
        t.lap(prep);
        for (size_t i = 0; i < cur.size; ++i) {
            const llama_token id = cur.data[i].id;
            if (!((words[id/32] >> (id%32)) & 1)) {
                cur.data[i].logit = -INFINITY;
            }
        }
        t.lap(apply);
    }

    llama_token run_chain(llama_token_data_array & cur) {
        llama_sampler_apply(chain.get(), &cur);
        require(cur.selected >= 0 && size_t(cur.selected) < cur.size, "host chain selected no token");
        return cur.data[cur.selected].id;
    }

    step_result step(step_timer & t, logits_env & env, llama_token accept) override {
        step_result r;
        const int32_t n_vocab = cand.size();
        const float * logits = env.for_host(t);
        init_candidates(cand, logits, n_vocab);
        llama_token_data_array cur = {cand.data(), cand.size(), -1, false};
        t.lap(C_INIT);
        if (pol == POLICY_GF) {
            mask(cur, t, C_PREP, C_MASK);
            r.selected = run_chain(cur);
            t.lap(C_CHAIN);
        } else {
            r.proposal = r.selected = run_chain(cur);
            t.lap(C_CHAIN);
            const bool ok = valid(r.proposal);
            t.lap(C_VALIDATE);
            if (!ok) {
                r.retried = true;
                init_candidates(cand, logits, n_vocab);
                cur = {cand.data(), cand.size(), -1, false};
                t.lap(C_RETRY_INIT);
                mask(cur, t, C_RETRY_PREP, C_RETRY_MASK);
                r.selected = run_chain(cur);
                t.lap(C_RETRY_CHAIN);
            }
        }
        if (kind == PATH_CPU_SCAN) {
            llama_sampler_accept(grammar.get(), accept);
        } else {
            llama_grammar_accept_impl(*raw, accept);
        }
        llama_sampler_accept(chain.get(), accept);
        t.lap(C_ACCEPT);
        return r;
    }
};

struct backend_runner : runner {
    llama_sampler_ptr grammar;
    llama_sampler_ptr chain;
    llama_sampler_ptr retry_grammar;
    llama_sampler_ptr retry_chain;
    std::unique_ptr<sampler_graph> main_graph;
    std::unique_ptr<sampler_graph> retry_graph;
    const llama_sampler_row row = {0, LLAMA_SAMPLER_PREFIX_COMMITTED, nullptr, 0};

    backend_runner(policy pol_, const llama_vocab * vocab, const std::string & grammar_str, const options & opt,
            ggml_backend_t backend, logits_env & env) : chain(make_chain(opt)) {
        kind = PATH_BACKEND;
        pol = pol_;
        grammar.reset(llama_sampler_init_grammar(vocab, grammar_str.c_str(), "root"));
        require(bool(grammar), "cannot initialize grammar");
        const int32_t n_vocab = llama_vocab_n_tokens(vocab);
        if (pol == POLICY_GF) {
            main_graph = std::make_unique<sampler_graph>(std::vector<llama_sampler *>{grammar.get(), chain.get()}, backend, env.model, n_vocab);
        } else {
            main_graph = std::make_unique<sampler_graph>(std::vector<llama_sampler *>{chain.get()}, backend, env.model, n_vocab);
            retry_grammar.reset(llama_sampler_clone(grammar.get()));
            retry_chain = make_chain(opt);
            retry_graph = std::make_unique<sampler_graph>(std::vector<llama_sampler *>{retry_grammar.get(), retry_chain.get()}, backend, nullptr, n_vocab);
        }
        reset();
    }

    ~backend_runner() override {
        retry_graph.reset();
        main_graph.reset();
    }

    void reset() override {
        llama_sampler_reset(grammar.get());
        llama_sampler_reset(chain.get());
    }

    step_result step(step_timer & t, logits_env & env, llama_token accept) override {
        step_result r;
        if (pol == POLICY_GF) {
            require(llama_sampler_grammar_prepare_rows(grammar.get(), &row, 1, true), "grammar mask preparation failed");
            require(grammar->iface->backend_row_usable(grammar.get(), 0), "grammar has no valid token");
            t.lap(C_PREP);
            main_graph->set_input();
            t.lap(C_UPLOAD);
            main_graph->compute();
            t.lap(C_GRAPH);
            r.selected = main_graph->read();
            t.lap(C_READBACK);
        } else {
            ggml_backend_tensor_copy(env.model, env.retained);
            t.lap(C_RETAIN);
            main_graph->set_input();
            t.lap(C_UPLOAD);
            main_graph->compute();
            t.lap(C_GRAPH);
            r.proposal = r.selected = main_graph->read();
            t.lap(C_READBACK);
            llama_token_data candidate = {r.proposal, 0.0f, 0.0f};
            llama_token_data_array single = {&candidate, 1, -1, false};
            llama_sampler_apply(grammar.get(), &single);
            t.lap(C_VALIDATE);
            if (candidate.logit == -INFINITY) {
                r.retried = true;
                llama_sampler_copy(grammar.get(), retry_grammar.get());
                t.lap(C_RETRY_STATE);
                require(llama_sampler_grammar_prepare_rows(retry_grammar.get(), &row, 1, true), "grammar mask preparation failed");
                require(retry_grammar->iface->backend_row_usable(retry_grammar.get(), 0), "grammar has no valid token");
                t.lap(C_RETRY_PREP);
                ggml_backend_tensor_copy(env.retained, retry_graph->input);
                t.lap(C_RETRY_COPY);
                retry_graph->set_input();
                t.lap(C_RETRY_UPLOAD);
                retry_graph->compute();
                t.lap(C_RETRY_GRAPH);
                r.selected = retry_graph->read();
                t.lap(C_RETRY_READBACK);
            }
        }
        llama_sampler_accept(grammar.get(), accept);
        llama_sampler_accept(chain.get(), accept);
        t.lap(C_ACCEPT);
        return r;
    }
};

struct step_record {
    std::array<double, C_COUNT> us;
    double total;
    step_result result;
    bool forced;
};

static uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30))*0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27))*0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

struct workload {
    const std::vector<trace_step> & trace;
    std::vector<float> & base;      // synthetic noise, modified in place
    const replay_data * replay;
    double rate;
    uint64_t seed;

    bool forced(int i) const {
        if (replay || rate <= 0.0) {
            return false;
        }
        return rate >= 1.0 || double(mix64(seed ^ uint64_t(i)) >> 11)*(1.0/9007199254740992.0) < rate;
    }

    void set(logits_env & env, int i, bool apply) const {
        const auto & step = trace[i % trace.size()];
        if (replay) {
            if (apply) {
                env.master = replay->logits.data() + size_t(i % trace.size())*env.n_vocab;
                ggml_backend_tensor_set(env.model, env.master, 0, size_t(env.n_vocab)*sizeof(float));
            }
            return;
        }
        const bool f = forced(i);
        const float e = apply ? 15.0f : base_value(step.expected);
        const float r = apply && f ? 20.0f : base_value(step.forbidden);
        base[step.expected] = e;
        base[step.forbidden] = r;
        ggml_backend_tensor_set(env.model, &e, size_t(step.expected)*sizeof(float), sizeof(float));
        ggml_backend_tensor_set(env.model, &r, size_t(step.forbidden)*sizeof(float), sizeof(float));
        env.master = base.data();
    }

    static float noise(uint64_t seed, llama_token id) {
        return float(double(mix64(seed*0x100000001b3ull + uint64_t(id)) >> 11)*(1.0/9007199254740992.0)*6.0 - 3.0);
    }

    float base_value(llama_token id) const { return noise(seed, id); }
};

static void spin_us(double us) {
    if (us <= 0.0) {
        return;
    }
    const auto end = bench_clock::now() + std::chrono::duration_cast<bench_clock::duration>(std::chrono::duration<double, std::micro>(us));
    while (bench_clock::now() < end) {
    }
}

static std::vector<step_record> measure(runner & run, logits_env & env, const workload & work, int steps, double eval_us) {
    std::vector<step_record> records(steps);
    run.reset();
    const size_t n_trace = work.trace.size();
    for (int i = 0; i < steps; ++i) {
        const auto & step = work.trace[i % n_trace];
        work.set(env, i, true);
        step_timer t;
        t.start();
        spin_us(eval_us);
        t.lap(C_EVAL);
        if (i > 0 && i % n_trace == 0) {
            run.reset();
            t.lap(C_RESET);
        }
        const auto result = run.step(t, env, step.expected);
        work.set(env, i, false);
        auto & rec = records[i];
        rec.us = t.us;
        rec.total = std::accumulate(t.us.begin(), t.us.end(), 0.0);
        rec.result = result;
        rec.forced = work.forced(i);
        if (!work.replay) {
            require(result.selected == step.expected, std::string(path_names[run.kind]) + " " + policy_names[run.pol] +
                    ": token mismatch at step " + std::to_string(i) + ": expected " + std::to_string(step.expected) +
                    ", got " + std::to_string(result.selected));
            require(run.pol == POLICY_GF || result.retried == rec.forced,
                    "rejection schedule not reproduced at step " + std::to_string(i));
        }
    }
    if (auto * cpu = dynamic_cast<cpu_runner *>(&run)) {
        cpu->flush_stats();
    }
    return records;
}

struct cell_stats {
    std::string label;
    double rate;
    policy pol;
    path_kind kind;
    size_t steps = 0;
    size_t retries = 0;
    size_t model_disagree = 0;
    size_t mismatch_vs_scan = 0;
    size_t cache_hits = 0, cache_misses = 0;
    std::array<double, C_COUNT> comp_mean = {};
    double mean = 0, p50 = 0, p99 = 0, rep_sd = 0;
    std::vector<double> rep_means;
};

static double percentile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, size_t(q*(v.size() - 1) + 0.5))];
}

static std::string fmt_rate(double rate) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%g%%", rate*100.0);
    return buf;
}

static void usage(const char * exe) {
    printf("Usage: %s -m MODEL [options]\n"
            "  -n N                 measured steps per cell and repetition (default 1024)\n"
            "  --repetitions N      repetitions; path order rotates (default 1)\n"
            "  --device NAME        gpu (first GPU), cpu, or a ggml device name (default gpu)\n"
            "  --rates LIST         forced rejection rates, comma-separated (default 0,0.01,0.1,0.5,1)\n"
            "  --policies LIST      stc,gf (default both)\n"
            "  --paths LIST         cpu-scan,cpu-trie,backend (default all)\n"
            "  --top-k N --top-p P --min-p P   chain before greedy (default 40, 0.95, 0.05; 0/1/0 disables)\n"
            "  --eval-us US         simulated model evaluation per step, busy wait (default 0)\n"
            "  --logits-copy on|off host paths copy logits out of the backend buffer (default on)\n"
            "  --device-mask on|off allow the backend to build grammar masks on the device (default off)\n"
            "  --replay FILE        replay saved model logits instead of synthetic logits\n"
            "  --dump-logits FILE   run the model teacher-forced over the trace and save its logits\n"
            "  -ngl N               GPU layers for --dump-logits (default 99)\n"
            "  --jsonl FILE         write per-step and summary records\n"
            "  --jsonl-steps on|off include per-step records (default on)\n"
            "  --tools JSON         fixture (default tools/test-grammar-tools.json)\n"
            "  --seed N             synthetic noise and rejection schedule seed (default 1234)\n", exe);
}

static std::vector<std::string> split(const std::string & s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

static bool on_off(const std::string & value) {
    require(value == "on" || value == "off", "expected on or off: " + value);
    return value == "on";
}

static options parse(int argc, char ** argv, bool & help) {
    options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            help = true;
            return opt;
        }
        require(i + 1 < argc, "missing value for " + arg);
        const std::string value = argv[++i];
        if      (arg == "-m")             { opt.model_path = value; }
        else if (arg == "-n")             { opt.steps = positive_int(value); }
        else if (arg == "--repetitions")  { opt.repetitions = positive_int(value); }
        else if (arg == "--device")       { opt.device_name = value; }
        else if (arg == "--tools")        { opt.tools_path = value; }
        else if (arg == "--jsonl")        { opt.jsonl_path = value; }
        else if (arg == "--jsonl-steps")  { opt.jsonl_steps = on_off(value); }
        else if (arg == "--replay")       { opt.replay_path = value; }
        else if (arg == "--dump-logits")  { opt.dump_path = value; }
        else if (arg == "-ngl")           { opt.n_gpu_layers = std::stoi(value); }
        else if (arg == "--top-k")        { opt.top_k = std::stoi(value); }
        else if (arg == "--top-p")        { opt.top_p = std::stof(value); }
        else if (arg == "--min-p")        { opt.min_p = std::stof(value); }
        else if (arg == "--eval-us")      { opt.eval_us = std::stod(value); require(opt.eval_us >= 0, "--eval-us must be >= 0"); }
        else if (arg == "--logits-copy")  { opt.logits_copy = on_off(value); }
        else if (arg == "--device-mask")  { opt.device_mask = on_off(value); }
        else if (arg == "--seed")         { opt.seed = std::stoull(value); }
        else if (arg == "--rates") {
            opt.rates.clear();
            for (const auto & r : split(value)) {
                opt.rates.push_back(std::stod(r));
                require(opt.rates.back() >= 0.0 && opt.rates.back() <= 1.0, "rates must be in [0, 1]");
            }
        } else if (arg == "--policies") {
            opt.policies.clear();
            for (const auto & p : split(value)) {
                require(p == "stc" || p == "gf", "unknown policy " + p);
                opt.policies.push_back(p == "stc" ? POLICY_STC : POLICY_GF);
            }
        } else if (arg == "--paths") {
            opt.paths.clear();
            for (const auto & p : split(value)) {
                const auto it = std::find_if(std::begin(path_names), std::end(path_names), [&](const char * n) { return p == n; });
                require(it != std::end(path_names), "unknown path " + p);
                opt.paths.push_back(path_kind(it - std::begin(path_names)));
            }
        } else {
            throw std::runtime_error("unknown argument " + arg);
        }
    }
    require(!opt.policies.empty() && !opt.paths.empty() && !opt.rates.empty(), "empty policy, path or rate list");
    return opt;
}

static int run(int argc, char ** argv) {
    bool help = false;
    const options opt = parse(argc, argv, help);
    if (help) {
        usage(argv[0]);
        return 0;
    }
    require(!opt.model_path.empty(), "provide -m MODEL");
    setenv("LLAMA_GRAMMAR_GPU_MASK", opt.device_mask ? "1" : "0", 1);

    const common_json fixture = load_fixture(opt.tools_path);
    const std::string grammar = json_schema_to_grammar(make_schema(fixture.at("tools")), true);

    llama_backend_init();
    ggml_backend_load_all();
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    mparams.n_gpu_layers = 0;
    llama_model_ptr model(llama_model_load_from_file(opt.model_path.c_str(), mparams));
    require(bool(model), "cannot load model vocabulary");
    const auto * vocab = llama_model_get_vocab(model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    printf("Preparing replay trace and checking trie masks against the candidate scan (%d tokens)...\n", n_vocab);
    const auto trace = make_trace(vocab, grammar, fixture.at("examples").dump());
    printf("tools=%zu grammar_bytes=%zu vocab=%d trace_tokens=%zu; trie mask == scan mask at every trace state\n",
            fixture.at("tools").size(), grammar.size(), n_vocab, trace.size());

    if (!opt.dump_path.empty()) {
        model.reset();
        dump_logits(opt, fixture, trace);
        return 0;
    }

    replay_data replay;
    if (!opt.replay_path.empty()) {
        replay = load_replay(opt.replay_path);
        require(replay.n_vocab == uint32_t(n_vocab), "replay vocabulary size differs from model");
        require(replay.tokens.size() == trace.size(), "replay length differs from trace");
        for (size_t i = 0; i < trace.size(); ++i) {
            require(replay.tokens[i] == trace[i].expected, "replay token differs from trace at " + std::to_string(i));
        }
    }
    const bool replaying = !opt.replay_path.empty();

    ggml_backend_dev_t device = nullptr;
    if (opt.device_name == "gpu") {
        device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!device) {
            device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        }
    } else if (opt.device_name == "cpu") {
        device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    } else {
        device = ggml_backend_dev_by_name(opt.device_name.c_str());
    }
    require(device != nullptr, "requested backend device unavailable: " + opt.device_name);
    ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    require(bool(backend), "cannot initialize backend");
    auto * reg = ggml_backend_dev_backend_reg(device);
    if (auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"))) {
        set_threads(backend.get(), 1);
    }
    const bool device_mask = opt.device_mask && ggml_backend_reg_get_proc_address(reg, "ggml_backend_grammar_compute");

    logits_env env(backend.get(), n_vocab, opt.logits_copy);
    std::vector<float> base(n_vocab);
    for (int32_t id = 0; id < n_vocab; ++id) {
        base[id] = workload::noise(opt.seed, id);
    }
    ggml_backend_tensor_set(env.model, base.data(), 0, base.size()*sizeof(float));

    struct cell { policy pol; path_kind kind; std::unique_ptr<runner> run; };
    std::vector<cell> cells;
    for (const auto pol : opt.policies) {
        for (const auto kind : opt.paths) {
            std::unique_ptr<runner> r;
            if (kind == PATH_BACKEND) {
                r = std::make_unique<backend_runner>(pol, vocab, grammar, opt, backend.get(), env);
            } else {
                r = std::make_unique<cpu_runner>(kind, pol, vocab, grammar, opt);
            }
            cells.push_back({pol, kind, std::move(r)});
        }
    }

    printf("backend=%s chain=\"%s\" mask_source=%s logits_copy=%s eval_us=%g steps=%d repetitions=%d mode=%s\n",
            ggml_backend_dev_name(device), chain_description(opt).c_str(),
            device_mask ? "device (GPU grammar prototype)" : "host (trie/cache/fallback)",
            opt.logits_copy ? "on" : "off", opt.eval_us, opt.steps, opt.repetitions,
            replaying ? ("replay " + opt.replay_path).c_str() : "synthetic");

    {
        workload warm = {trace, base, replaying ? &replay : nullptr, 0.5, opt.seed};
        for (auto & c : cells) {
            measure(*c.run, env, warm, trace.size(), 0.0);
            c.run->cache_hits = c.run->cache_misses = 0;
        }
        printf("Warmup complete\n");
    }

    FILE * jsonl = nullptr;
    if (!opt.jsonl_path.empty()) {
        jsonl = fopen(opt.jsonl_path.c_str(), "w");
        require(jsonl != nullptr, "cannot write " + opt.jsonl_path);
    }

    const std::vector<double> rates = replaying ? std::vector<double>{-1.0} : opt.rates;
    std::vector<cell_stats> all;
    for (const double rate : rates) {
        const std::string label = rate < 0 ? "replay" : fmt_rate(rate);
        workload work = {trace, base, replaying ? &replay : nullptr, std::max(rate, 0.0), opt.seed};
        std::vector<std::vector<step_record>> per_cell(cells.size());
        std::vector<std::vector<double>> rep_means(cells.size());
        std::vector<std::vector<llama_token>> selected(cells.size());
        for (int rep = 0; rep < opt.repetitions; ++rep) {
            for (size_t k = 0; k < cells.size(); ++k) {
                const size_t ci = (k + rep) % cells.size();
                auto & c = cells[ci];
                auto records = measure(*c.run, env, work, opt.steps, opt.eval_us);
                double sum = 0;
                for (int i = 0; i < opt.steps; ++i) {
                    const auto & rec = records[i];
                    sum += rec.total;
                    if (rep == 0) {
                        selected[ci].push_back(rec.result.selected);
                    }
                    if (jsonl && opt.jsonl_steps) {
                        fprintf(jsonl, "{\"type\":\"step\",\"mode\":\"%s\",\"rate\":%g,\"policy\":\"%s\",\"path\":\"%s\",\"rep\":%d,"
                                "\"step\":%d,\"trace_pos\":%zu,\"expected\":%d,\"proposal\":%d,\"selected\":%d,\"forced\":%s,"
                                "\"retried\":%s,\"total_us\":%.3f,\"us\":{",
                                replaying ? "replay" : "synthetic", std::max(rate, 0.0), policy_names[c.pol], path_names[c.kind],
                                rep, i, i % trace.size(), trace[i % trace.size()].expected, rec.result.proposal,
                                rec.result.selected, rec.forced ? "true" : "false", rec.result.retried ? "true" : "false", rec.total);
                        bool first = true;
                        for (int k2 = 0; k2 < C_COUNT; ++k2) {
                            if (rec.us[k2] > 0) {
                                fprintf(jsonl, "%s\"%s\":%.3f", first ? "" : ",", comp_names[k2], rec.us[k2]);
                                first = false;
                            }
                        }
                        fprintf(jsonl, "}}\n");
                    }
                }
                rep_means[ci].push_back(sum/opt.steps);
                per_cell[ci].insert(per_cell[ci].end(), records.begin(), records.end());
            }
        }

        std::map<policy, size_t> reference;
        for (size_t ci = 0; ci < cells.size(); ++ci) {
            if (!reference.count(cells[ci].pol) || cells[ci].kind == PATH_CPU_SCAN) {
                reference[cells[ci].pol] = ci;
            }
        }
        std::vector<cell_stats> stats(cells.size());
        for (size_t ci = 0; ci < cells.size(); ++ci) {
            auto & s = stats[ci];
            const auto & recs = per_cell[ci];
            s.label = label;
            s.rate = rate;
            s.pol = cells[ci].pol;
            s.kind = cells[ci].kind;
            s.steps = recs.size();
            std::vector<double> totals;
            for (const auto & rec : recs) {
                totals.push_back(rec.total);
                s.retries += rec.result.retried;
                for (int k = 0; k < C_COUNT; ++k) {
                    s.comp_mean[k] += rec.us[k]/recs.size();
                }
            }
            s.mean = std::accumulate(totals.begin(), totals.end(), 0.0)/totals.size();
            s.p50 = percentile(totals, 0.5);
            s.p99 = percentile(totals, 0.99);
            s.rep_means = std::move(rep_means[ci]);
            double var = 0;
            for (const double m : s.rep_means) {
                const double avg = std::accumulate(s.rep_means.begin(), s.rep_means.end(), 0.0)/s.rep_means.size();
                var += (m - avg)*(m - avg);
            }
            s.rep_sd = s.rep_means.size() > 1 ? std::sqrt(var/(s.rep_means.size() - 1)) : 0.0;
            for (int i = 0; i < opt.steps; ++i) {
                s.model_disagree += selected[ci][i] != trace[i % trace.size()].expected;
                s.mismatch_vs_scan += selected[ci][i] != selected[reference[s.pol]][i];
            }
            if (auto * cpu = dynamic_cast<cpu_runner *>(cells[ci].run.get())) {
                s.cache_hits = cpu->cache_hits;
                s.cache_misses = cpu->cache_misses;
                cpu->cache_hits = cpu->cache_misses = 0;
            }
            all.push_back(s);
        }

        for (const auto pol : opt.policies) {
            printf("\n=== %s | policy: %s (all paths below use this policy) | eval_us=%g ===\n",
                    rate < 0 ? "real-logit replay" : ("synthetic, forced rejection " + label).c_str(), policy_names[pol], opt.eval_us);
            printf("%-9s %10s %9s %9s %9s %9s %9s %8s %9s %s\n", "path", "us/token", "rep_sd", "p50", "p99",
                    "tok/s", "retries", "retry%", "mismatch", "vs backend");
            const cell_stats * be = nullptr;
            for (const auto & s : stats) {
                if (s.pol == pol && s.kind == PATH_BACKEND) {
                    be = &s;
                }
            }
            std::vector<const cell_stats *> rows;
            for (const auto & s : stats) {
                if (s.pol == pol) {
                    rows.push_back(&s);
                }
            }
            for (const auto * s : rows) {
                std::string versus = "-";
                if (be && s != be) {
                    char buf[64];
                    snprintf(buf, sizeof(buf), "backend %.2fx %s", s->mean > be->mean ? s->mean/be->mean : be->mean/s->mean,
                            s->mean > be->mean ? "faster" : "slower");
                    versus = buf;
                }
                printf("%-9s %10.2f %9.2f %9.2f %9.2f %9.1f %9zu %7.2f%% %9zu %s\n", path_names[s->kind], s->mean, s->rep_sd,
                        s->p50, s->p99, 1e6/s->mean, s->retries, 100.0*s->retries/s->steps, s->mismatch_vs_scan, versus.c_str());
            }
            printf("components, mean us/token over all steps (retry_* averaged over all steps, not only retries):\n");
            printf("  %-18s", "component");
            for (const auto * s : rows) {
                printf(" %10s", path_names[s->kind]);
            }
            printf("\n");
            for (int k = 0; k < C_COUNT; ++k) {
                bool any = false;
                for (const auto * s : rows) {
                    any = any || s->comp_mean[k] > 0;
                }
                if (!any) {
                    continue;
                }
                printf("  %-18s", comp_names[k]);
                for (const auto * s : rows) {
                    printf(" %10.2f", s->comp_mean[k]);
                }
                printf("\n");
            }
            if (opt.eval_us > 0) {
                printf("  %-18s", "sampling share");
                for (const auto * s : rows) {
                    printf(" %9.1f%%", 100.0*(s->mean - s->comp_mean[C_EVAL])/s->mean);
                }
                printf("\n");
            }
            for (const auto * s : rows) {
                if (s->kind == PATH_CPU_TRIE) {
                    printf("  cpu-trie mask cache: %zu hits, %zu misses\n", s->cache_hits, s->cache_misses);
                }
            }
            if (replaying) {
                for (const auto * s : rows) {
                    printf("  %s: selected != trace token at %zu/%zu steps (model disagrees with the fixture)\n",
                            path_names[s->kind], s->model_disagree, s->steps/opt.repetitions);
                }
            }
        }
        if (reference.count(POLICY_STC) && reference.count(POLICY_GF)) {
            size_t n = 0;
            for (int i = 0; i < opt.steps; ++i) {
                n += selected[reference[POLICY_STC]][i] != selected[reference[POLICY_GF]][i];
            }
            printf("cross-policy token mismatches (cpu-scan stc vs gf): %zu\n", n);
        }
    }

    if (jsonl) {
        for (const auto & s : all) {
            common_json comps = common_json::object();
            for (int k = 0; k < C_COUNT; ++k) {
                if (s.comp_mean[k] > 0) {
                    comps[comp_names[k]] = s.comp_mean[k];
                }
            }
            common_json line = {
                {"type", "summary"}, {"mode", replaying ? "replay" : "synthetic"}, {"rate", std::max(s.rate, 0.0)},
                {"policy", policy_names[s.pol]}, {"path", path_names[s.kind]}, {"device", ggml_backend_dev_name(device)},
                {"chain", chain_description(opt)}, {"mask_source", device_mask ? "device" : "host"},
                {"logits_copy", opt.logits_copy}, {"eval_us", opt.eval_us}, {"steps", s.steps},
                {"repetitions", opt.repetitions}, {"mean_us", s.mean}, {"rep_sd_us", s.rep_sd}, {"rep_means_us", s.rep_means},
                {"p50_us", s.p50}, {"p99_us", s.p99}, {"tok_s", 1e6/s.mean}, {"retries", s.retries},
                {"retry_rate", double(s.retries)/s.steps}, {"mismatch_vs_cpu_scan", s.mismatch_vs_scan},
                {"model_disagree", s.model_disagree}, {"components_us", comps},
            };
            if (s.kind == PATH_CPU_TRIE) {
                line["trie_cache_hits"] = s.cache_hits;
                line["trie_cache_misses"] = s.cache_misses;
            }
            fprintf(jsonl, "%s\n", line.dump().c_str());
        }
        fclose(jsonl);
        printf("\nwrote %s\n", opt.jsonl_path.c_str());
    }
    return 0;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    int result;
    try {
        result = run(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        result = 1;
    }
    llama_backend_free();
    return result;
}
