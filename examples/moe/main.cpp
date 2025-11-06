#include "llama.h"
#ifdef LLAMA_MOE_ENABLE
#include "llama-moe.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef LLAMA_MOE_ENABLE

struct options {
    std::string model_path;
    std::string prompt = "Hello world";
    int steps = -1;
    std::string json_path;
};

static bool parse_args(int argc, char ** argv, options & out) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--prompt" && i + 1 < argc) {
            out.prompt = argv[++i];
        } else if (arg == "--steps" && i + 1 < argc) {
            out.steps = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--json" && i + 1 < argc) {
            out.json_path = argv[++i];
        } else if (arg.rfind("--", 0) == 0) {
            std::cerr << "Unknown option: " << arg << "\n";
            return false;
        } else if (out.model_path.empty()) {
            out.model_path = arg;
        } else {
            std::cerr << "Unexpected argument: " << arg << "\n";
            return false;
        }
    }
    if (out.model_path.empty()) {
        return false;
    }
    return true;
}

static std::vector<llama_token> tokenize_prompt(const llama_vocab * vocab, const std::string & prompt) {
    const int32_t max_tokens = prompt.size() + 16;
    std::vector<llama_token> tokens(max_tokens);
    const int32_t n = llama_tokenize(vocab, prompt.c_str(), prompt.length(), tokens.data(), max_tokens, true, true);
    if (n < 0) {
        return {};
    }
    tokens.resize(n);
    return tokens;
}

struct run_result {
    std::vector<float> logits;
    llama_perf_context_data perf{};
    llama_moe_cache_stats stats{};
};

static run_result run_trace(const llama_model * model, llama_context * ctx, const std::vector<llama_token> & prompt_tokens, int steps, bool collect_stats) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    run_result res;
    res.logits.resize(static_cast<size_t>(steps) * n_vocab);

    llama_batch batch = llama_batch_init(1, 0, 1);

    for (int step = 0; step < steps; ++step) {
        llama_token token = prompt_tokens[std::min(step, static_cast<int>(prompt_tokens.size() - 1))];
        batch.n_tokens = 1;
        batch.token[0] = token;
        batch.pos[0] = step;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;

        const int32_t rc = llama_decode(ctx, batch);
        if (rc != 0) {
            std::cerr << "llama_decode failed with code " << rc << " at step " << step << "\n";
            break;
        }

        const float * logits = llama_get_logits(ctx);
        std::copy(logits, logits + n_vocab, res.logits.begin() + static_cast<size_t>(step) * n_vocab);
    }

    llama_batch_free(batch);

    res.perf = llama_perf_context(ctx);
#ifdef LLAMA_MOE_ENABLE
    if (collect_stats) {
        llama_moe_cache_get_stats(ctx, &res.stats);
    }
#else
    (void) collect_stats;
#endif

    return res;
}

static void emit_json(const std::string & path,
                      int steps,
                      int vocab,
                      double max_diff,
                      double mean_diff,
                      const llama_perf_context_data & perf_moe,
                      const llama_perf_context_data & perf_dense,
                      const llama_moe_cache_stats & stats) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open JSON output file: " << path << "\n";
        return;
    }
    ofs << "{\n";
    ofs << "  \"steps\": " << steps << ",\n";
    ofs << "  \"vocab\": " << vocab << ",\n";
    ofs << "  \"max_abs_diff\": " << max_diff << ",\n";
    ofs << "  \"mean_abs_diff\": " << mean_diff << ",\n";
    ofs << "  \"moe_perf\": {\n";
    ofs << "    \"tokens\": " << perf_moe.n_eval << ",\n";
    ofs << "    \"time_ms\": " << perf_moe.t_eval_ms << ",\n";
    ofs << "    \"tok_per_s\": " << (perf_moe.t_eval_ms > 0 ? (perf_moe.n_eval / (perf_moe.t_eval_ms / 1000.0)) : 0.0) << "\n";
    ofs << "  },\n";
    ofs << "  \"dense_perf\": {\n";
    ofs << "    \"tokens\": " << perf_dense.n_eval << ",\n";
    ofs << "    \"time_ms\": " << perf_dense.t_eval_ms << ",\n";
    ofs << "    \"tok_per_s\": " << (perf_dense.t_eval_ms > 0 ? (perf_dense.n_eval / (perf_dense.t_eval_ms / 1000.0)) : 0.0) << "\n";
    ofs << "  },\n";
    ofs << "  \"cache_stats\": {\n";
    ofs << "    \"resident\": " << stats.resident << ",\n";
    ofs << "    \"capacity_bytes\": " << stats.capacity_bytes << ",\n";
    ofs << "    \"loads\": " << stats.loads << ",\n";
    ofs << "    \"hits\": " << stats.hits << ",\n";
    ofs << "    \"evictions\": " << stats.evictions << ",\n";
    ofs << "    \"prefetch_requests\": " << stats.prefetch_requests << "\n";
    ofs << "  }\n";
    ofs << "}\n";
}
#endif // LLAMA_MOE_ENABLE

int main(int argc, char ** argv) {
#ifdef LLAMA_MOE_ENABLE
    options opts;
    if (!parse_args(argc, argv, opts)) {
        std::cerr << "Usage: moe-validate <model.gguf> [--prompt \"text\"] [--steps N] [--json path]\n";
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(opts.model_path.c_str(), mparams);
    if (model == nullptr) {
        std::cerr << "Failed to load model: " << opts.model_path << "\n";
        llama_backend_free();
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.moe_enable = true;
    cparams.moe_prefetch = true;
    llama_context * moe_ctx = llama_init_from_model(model, cparams);
    if (moe_ctx == nullptr) {
        std::cerr << "Failed to create MoE context\n";
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    llama_context_params dense_params = cparams;
    dense_params.moe_enable = false;
    llama_context * dense_ctx = llama_init_from_model(model, dense_params);
    if (dense_ctx == nullptr) {
        std::cerr << "Failed to create dense fallback context\n";
        llama_free(moe_ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    auto tokens = tokenize_prompt(vocab, opts.prompt);
    if (tokens.empty()) {
        std::cerr << "Failed to tokenize prompt\n";
        llama_free(dense_ctx);
        llama_free(moe_ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const int steps = (opts.steps > 0) ? std::min(opts.steps, static_cast<int>(tokens.size())) : static_cast<int>(tokens.size());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    auto moe_run = run_trace(model, moe_ctx, tokens, steps, true);
    auto dense_run = run_trace(model, dense_ctx, tokens, steps, false);

    double max_diff = 0.0;
    double sum_diff = 0.0;
    size_t count = static_cast<size_t>(steps) * n_vocab;
    for (size_t i = 0; i < count; ++i) {
        double diff = std::abs(moe_run.logits[i] - dense_run.logits[i]);
        sum_diff += diff;
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    double mean_diff = count > 0 ? sum_diff / static_cast<double>(count) : 0.0;

    auto tok_per_s = [](const llama_perf_context_data & perf) -> double {
        return perf.t_eval_ms > 0 ? (perf.n_eval / (perf.t_eval_ms / 1000.0)) : 0.0;
    };

    std::cout << "Prompt tokens: " << tokens.size() << " | Steps evaluated: " << steps << "\n";
    std::cout << "Max abs diff: " << max_diff << " | Mean abs diff: " << mean_diff << "\n";
    std::cout << "Dense  : tokens=" << dense_run.perf.n_eval << " time_ms=" << dense_run.perf.t_eval_ms
              << " tok/s=" << tok_per_s(dense_run.perf) << "\n";
    std::cout << "MoE    : tokens=" << moe_run.perf.n_eval << " time_ms=" << moe_run.perf.t_eval_ms
              << " tok/s=" << tok_per_s(moe_run.perf) << "\n";
    std::cout << "MoE cache: resident=" << moe_run.stats.resident
              << " loads=" << moe_run.stats.loads
              << " hits=" << moe_run.stats.hits
              << " evictions=" << moe_run.stats.evictions
              << " prefetch_requests=" << moe_run.stats.prefetch_requests << "\n";

    if (!opts.json_path.empty()) {
        emit_json(opts.json_path, steps, n_vocab, max_diff, mean_diff, moe_run.perf, dense_run.perf, moe_run.stats);
        std::cout << "Wrote JSON summary to " << opts.json_path << "\n";
    }

    llama_free(dense_ctx);
    llama_free(moe_ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
#else
    std::cerr << "This build of llama.cpp was compiled without LLAMA_MOE_ENABLE.\n";
    (void) argc;
    (void) argv;
    return 1;
#endif
}
