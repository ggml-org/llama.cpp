#include "llama.h"
#include "llama-cpp.h"
#include "ggml-cpp.h"
#include "sampler-bench-common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using bench_clock = std::chrono::steady_clock;

struct trace_step {
    llama_token expected;
    llama_token rejected;
};

static std::vector<trace_step> make_trace(const llama_vocab * vocab, const std::string & grammar,
        const std::string & text) {
    int32_t count = llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, false, false);
    require(count < 0, "fixture tokenization failed");
    std::vector<llama_token> tokens(-count);
    require(llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), false, false) == -count,
            "fixture tokenization length changed");
    require(llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL, "model has no EOS token");
    tokens.push_back(llama_vocab_eos(vocab));

    llama_sampler_ptr oracle(llama_sampler_init_grammar(vocab, grammar.c_str(), "root"));
    require(bool(oracle), "cannot initialize grammar");
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token_data> candidates(n_vocab);
    std::vector<trace_step> trace;
    for (const auto token : tokens) {
        for (int32_t id = 0; id < n_vocab; ++id) {
            candidates[id] = {id, 0.0f, 0.0f};
        }
        llama_token_data_array cur = {candidates.data(), candidates.size(), -1, false};
        llama_sampler_apply(oracle.get(), &cur);
        require(std::isfinite(candidates[token].logit), "fixture rejected at token " + std::to_string(trace.size()));
        llama_token rejected = LLAMA_TOKEN_NULL;
        for (const auto & candidate : candidates) {
            if (candidate.logit == -INFINITY) {
                rejected = candidate.id;
                break;
            }
        }
        require(rejected != LLAMA_TOKEN_NULL, "fixture state has no rejected token");
        trace.push_back({token, rejected});
        llama_sampler_accept(oracle.get(), token);
    }
    return trace;
}

static llama_sampler_ptr make_sampler(const llama_vocab * vocab, const std::string & grammar) {
    auto params = llama_sampler_chain_default_params();
    params.no_perf = true;
    llama_sampler_ptr chain(llama_sampler_chain_init(params));
    llama_sampler * grmr = llama_sampler_init_grammar(vocab, grammar.c_str(), "root");
    require(grmr != nullptr, "cannot initialize grammar");
    llama_sampler_chain_add(chain.get(), grmr);
    llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
    return chain;
}

struct backend_graph {
    llama_sampler * sampler;
    ggml_context_ptr ctx;
    ggml_backend_ptr cpu;
    ggml_backend_sched_ptr sched;
    ggml_cgraph * graph;
    ggml_tensor * logits;
    llama_sampler_data data = {};

    backend_graph(llama_sampler * sampler, ggml_backend_t backend, int32_t n_vocab) : sampler(sampler) {
        require(sampler->iface->backend_init(sampler, ggml_backend_get_default_buffer_type(backend), 1),
                "backend does not support the complete grammar + greedy chain");
        sampler->iface->backend_reset(sampler);
        ggml_init_params params = {64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true};
        ctx.reset(ggml_init(params));
        require(bool(ctx), "cannot create graph context");
        graph = ggml_new_graph(ctx.get());
        logits = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n_vocab);
        ggml_set_input(logits);
        data.logits = logits;
        sampler->iface->backend_apply(sampler, ctx.get(), graph, &data);
        require(data.sampled != nullptr && !data.skip, "backend did not build token selection");
        ggml_set_output(data.sampled);
        ggml_build_forward_expand(graph, data.sampled);
        if (ggml_backend_dev_type(ggml_backend_get_device(backend)) != GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
            require(bool(cpu), "cannot initialize CPU backend");
        }
        ggml_backend_t backends[] = {backend, cpu.get()};
        sched.reset(ggml_backend_sched_new(backends, nullptr, cpu ? 2 : 1, GGML_DEFAULT_GRAPH_SIZE, false, true));
        require(bool(sched), "cannot create scheduler");
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            require(ggml_backend_supports_op(backend, node), "unsupported backend op " + std::string(ggml_op_name(node->op)));
            ggml_backend_sched_set_tensor_backend(sched.get(), node, backend);
        }
        ggml_backend_sched_set_tensor_backend(sched.get(), logits, backend);
        require(ggml_backend_sched_alloc_graph(sched.get(), graph), "cannot allocate sampler graph");
        std::vector<float> zeros(n_vocab, 0.0f);
        ggml_backend_tensor_set(logits, zeros.data(), 0, zeros.size()*sizeof(float));
    }

    ~backend_graph() {
        sampler->iface->backend_reset(sampler);
    }

    void set_logit(llama_token token, float value) {
        ggml_backend_tensor_set(logits, &value, size_t(token)*sizeof(float), sizeof(float));
    }

    llama_token sample() {
        sampler->iface->backend_set_input(sampler, sched.get());
        require(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS, "sampler graph failed");
        llama_token token;
        ggml_backend_tensor_get(data.sampled, &token, 0, sizeof(token));
        llama_sampler_accept(sampler, token);
        return token;
    }
};

static double measure(llama_sampler * sampler, backend_graph * backend, int32_t n_vocab,
        const std::vector<trace_step> & trace, int steps) {
    std::vector<llama_token_data> candidates(backend ? 0 : n_vocab);
    llama_sampler_reset(sampler);
    double elapsed = 0.0;
    for (int i = 0; i < steps; ++i) {
        const auto & step = trace[i % trace.size()];
        if (backend) {
            backend->set_logit(step.expected, 10.0f);
            backend->set_logit(step.rejected, 20.0f);
        }
        const auto start = bench_clock::now();
        if (i > 0 && i % trace.size() == 0) {
            llama_sampler_reset(sampler);
        }
        llama_token token;
        if (backend) {
            token = backend->sample();
        } else {
            for (int32_t id = 0; id < n_vocab; ++id) {
                candidates[id] = {id, 0.0f, 0.0f};
            }
            candidates[step.expected].logit = 10.0f;
            candidates[step.rejected].logit = 20.0f;
            llama_token_data_array cur = {candidates.data(), candidates.size(), -1, false};
            llama_sampler_apply(sampler, &cur);
            require(cur.selected >= 0 && size_t(cur.selected) < cur.size, "CPU selected no token");
            token = cur.data[cur.selected].id;
            llama_sampler_accept(sampler, token);
        }
        elapsed += std::chrono::duration<double>(bench_clock::now() - start).count();
        require(token == step.expected, "token mismatch at step " + std::to_string(i) +
                ": expected " + std::to_string(step.expected) + ", got " + std::to_string(token));
        if (backend) {
            backend->set_logit(step.expected, 0.0f);
            backend->set_logit(step.rejected, 0.0f);
        }
    }
    return elapsed;
}

static void usage(const char * exe) {
    printf("Usage: %s -m MODEL [-n STEPS] [--repetitions N] [--device NAME|cpu|gpu]\n"
            "       [--tools JSON] [--write-schema JSON] [--write-grammar GBNF] [--generate-only]\n"
            "Defaults: 512 steps, 1 repetition, first GPU, tools/test-grammar-tools.json\n"
            "Loads vocabulary only; replays synthetic logits, no model decode.\n", exe);
}

static int run(int argc, char ** argv) {
    std::string model_path, schema_path, grammar_path;
    std::string tools_path = "tools/test-grammar-tools.json";
    std::string device_name = "gpu";
    int steps = 512;
    int repetitions = 1;
    bool generate_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        }
        if (arg == "--generate-only") {
            generate_only = true;
            continue;
        }
        require(i + 1 < argc, "missing value for " + arg);
        const std::string value = argv[++i];
        if      (arg == "-m")              { model_path = value; }
        else if (arg == "-n")              { steps = positive_int(value); }
        else if (arg == "--repetitions")   { repetitions = positive_int(value); }
        else if (arg == "--device")        { device_name = value; }
        else if (arg == "--tools")         { tools_path = value; }
        else if (arg == "--write-schema")  { schema_path = value; }
        else if (arg == "--write-grammar") { grammar_path = value; }
        else { throw std::runtime_error("unknown argument " + arg); }
    }
    const common_json fixture = load_fixture(tools_path);
    const common_json schema = make_schema(fixture.at("tools"));
    const std::string grammar = json_schema_to_grammar(schema, true);
    write_file(schema_path, schema.dump(4) + "\n");
    write_file(grammar_path, grammar);
    printf("tools=%zu grammar_bytes=%zu\n", fixture.at("tools").size(), grammar.size());
    if (generate_only) {
        require(!schema_path.empty() || !grammar_path.empty(), "--generate-only needs an output path");
        return 0;
    }
    require(!model_path.empty(), "provide -m MODEL or --generate-only");
    llama_backend_init();
    ggml_backend_load_all();
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true;
    mparams.n_gpu_layers = 0;
    llama_model_ptr model(llama_model_load_from_file(model_path.c_str(), mparams));
    require(bool(model), "cannot load model vocabulary");
    const auto * vocab = llama_model_get_vocab(model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    printf("Preparing and validating replay against vocabulary (%d tokens)...\n", n_vocab);
    const auto trace = make_trace(vocab, grammar, fixture.at("examples").dump());
    ggml_backend_dev_t device = nullptr;
    if (device_name == "gpu") {
        device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!device) {
            device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
        }
    } else if (device_name == "cpu") {
        device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    } else {
        device = ggml_backend_dev_by_name(device_name.c_str());
    }
    require(device != nullptr, "requested backend device unavailable: " + device_name);
    ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
    require(bool(backend), "cannot initialize backend");
    if (auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device), "ggml_backend_set_n_threads"))) {
        set_threads(backend.get(), 1);
    }
    auto cpu_sampler = make_sampler(vocab, grammar);
    auto backend_sampler = make_sampler(vocab, grammar);
    backend_graph graph(backend_sampler.get(), backend.get(), n_vocab);
    printf("sampler-only: vocab=%d trace_tokens=%zu steps=%d repetitions=%d backend=%s threads=1\n",
            n_vocab, trace.size(), steps, repetitions, ggml_backend_dev_name(device));
    printf("grammar + greedy; synthetic logits; warmup=one full trace per path; token checks enabled\n");
    measure(cpu_sampler.get(), nullptr, n_vocab, trace, trace.size());
    measure(backend_sampler.get(), &graph, n_vocab, trace, trace.size());
    printf("Warmup complete\n");
    std::vector<double> rates[2];
    for (int rep = 0; rep < repetitions; ++rep) {
        for (int turn = 0; turn < 2; ++turn) {
            const int path = (rep + turn) % 2;
            const double seconds = measure(path ? backend_sampler.get() : cpu_sampler.get(), path ? &graph : nullptr,
                    n_vocab, trace, steps);
            rates[path].push_back(steps/seconds);
            printf("%s rep=%d steps=%d seconds=%.6f tok/s=%.2f us/token=%.2f\n",
                    path ? "backend" : "cpu", rep + 1, steps, seconds, steps/seconds, 1e6*seconds/steps);
        }
    }
    double means[2];
    for (int path = 0; path < 2; ++path) {
        const auto & values = rates[path];
        means[path] = std::accumulate(values.begin(), values.end(), 0.0)/values.size();
        double variance = 0.0;
        for (const double value : values) {
            variance += (value - means[path])*(value - means[path]);
        }
        printf("%s mean_tok/s=%.2f sample_sd=%.2f min=%.2f max=%.2f\n", path ? "backend" : "cpu",
                means[path], values.size() > 1 ? std::sqrt(variance/(values.size() - 1)) : 0.0,
                *std::min_element(values.begin(), values.end()), *std::max_element(values.begin(), values.end()));
    }
    printf("backend/cpu=%.3fx; all sampled tokens matched (including higher-logit rejection probes)\n", means[1]/means[0]);
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
