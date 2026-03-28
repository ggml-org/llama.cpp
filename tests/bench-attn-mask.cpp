// Benchmark for custom attention mask overhead
// Usage: bench-attn-mask <model.gguf> [n_prompt] [n_iter]
//
// Measures decode time for 4 configurations:
//   A) No custom mask (baseline)
//   B) Custom mask = nullptr (check overhead)
//   C) Dense causal mask (full n_pos*n_pos, all 0.0 = same as causal)
//   D) Sparse window mask (window=2, restrictive)
//
// Reports: avg ms/decode, overhead vs baseline

#include "llama.h"
#include "common.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <algorithm>

struct bench_result {
    const char * name;
    std::vector<double> times_ms;

    double avg() const {
        if (times_ms.empty()) return 0.0;
        return std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();
    }

    double median() const {
        if (times_ms.empty()) return 0.0;
        auto sorted = times_ms;
        std::sort(sorted.begin(), sorted.end());
        size_t n = sorted.size();
        return (n % 2 == 0) ? (sorted[n/2 - 1] + sorted[n/2]) / 2.0 : sorted[n/2];
    }

    double stdev() const {
        if (times_ms.size() < 2) return 0.0;
        double m = avg();
        double sum = 0.0;
        for (double t : times_ms) sum += (t - m) * (t - m);
        return std::sqrt(sum / (times_ms.size() - 1));
    }
};

static std::vector<llama_pos> make_positions(int n) {
    std::vector<llama_pos> pos(n);
    for (int i = 0; i < n; ++i) pos[i] = i;
    return pos;
}

static std::vector<float> make_zeros_mask(int n) {
    return std::vector<float>(n * n, 0.0f);
}

static std::vector<float> make_window_mask(int n, int window) {
    std::vector<float> mask(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (j <= i && (i - j) <= window) {
                mask[i * n + j] = 0.0f;
            } else {
                mask[i * n + j] = -INFINITY;
            }
        }
    }
    return mask;
}

static double run_decode(llama_context * ctx, const llama_token * tokens, int n_tokens) {
    llama_memory_clear(llama_get_memory(ctx), true);

    llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(tokens), n_tokens);

    auto t0 = std::chrono::high_resolution_clock::now();
    int rc = llama_decode(ctx, batch);
    auto t1 = std::chrono::high_resolution_clock::now();

    if (rc != 0) return -1.0;

    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf> [n_prompt=64] [n_iter=20]\n", argv[0]);
        return 1;
    }

    const char * model_path = argv[1];
    const int n_prompt = argc > 2 ? atoi(argv[2]) : 64;
    const int n_iter   = argc > 3 ? atoi(argv[3]) : 20;
    const int n_ctx    = n_prompt * 2; // ensure headroom for KV cache

    // Build a prompt that tokenizes to roughly n_prompt tokens
    std::string prompt_base = "The quick brown fox jumps over the lazy dog. ";
    std::string prompt;
    while ((int)prompt.size() < n_prompt * 6) { // ~1.5 chars/token for English
        prompt += prompt_base;
    }

    // Load model
    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = n_ctx;
    cparams.n_batch = n_prompt;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // Tokenize once to get actual n_tokens (cap to n_prompt to leave KV headroom)
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tok_buf(n_ctx);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), tok_buf.data(), n_prompt, true, false);
    if (n_tokens < 0) n_tokens = n_prompt;
    if (n_tokens > n_prompt) n_tokens = n_prompt;

    tok_buf.resize(n_tokens);

    printf("Model:    %s\n", model_path);
    printf("n_ctx:    %d\n", n_ctx);
    printf("n_tokens: %d\n", n_tokens);
    printf("n_iter:   %d\n", n_iter);
    printf("\n");

    auto positions = make_positions(n_tokens);
    auto mask_zeros  = make_zeros_mask(n_tokens);
    auto mask_window = make_window_mask(n_tokens, 2);

    struct config {
        const char * name;
        const float * mask;
        const llama_pos * pos;
        int32_t n_pos;
    };

    config configs[] = {
        { "A) No mask (baseline)",  nullptr,            nullptr,          0        },
        { "B) nullptr mask",        nullptr,            nullptr,          0        },
        { "C) Dense zeros mask",    mask_zeros.data(),  positions.data(), n_tokens },
        { "D) Window=2 mask",       mask_window.data(), positions.data(), n_tokens },
    };

    const int n_configs = sizeof(configs) / sizeof(configs[0]);
    bench_result results[4];

    // Warmup: 2 decodes to compile Metal/CUDA kernels
    printf("Warming up...\n");
    fflush(stdout);
    for (int w = 0; w < 2; ++w) {
        llama_set_attn_mask(ctx, nullptr, nullptr, 0);
        run_decode(ctx, tok_buf.data(), n_tokens);
    }

    for (int c = 0; c < n_configs; ++c) {
        results[c].name = configs[c].name;

        for (int i = 0; i < n_iter; ++i) {
            llama_set_attn_mask(ctx, configs[c].mask, configs[c].pos, configs[c].n_pos);
            double t = run_decode(ctx, tok_buf.data(), n_tokens);
            if (t < 0.0) {
                fprintf(stderr, "Decode failed for config %s iter %d\n", configs[c].name, i);
                break;
            }
            results[c].times_ms.push_back(t);
        }
    }

    // Report
    printf("\n%-25s  %8s  %8s  %8s  %8s\n", "Config", "Avg(ms)", "Med(ms)", "Std(ms)", "Overhead");
    printf("%-25s  %8s  %8s  %8s  %8s\n", "------", "-------", "-------", "-------", "--------");

    double baseline_med = results[0].median();

    for (int c = 0; c < n_configs; ++c) {
        double med = results[c].median();
        double overhead_pct = baseline_med > 0 ? ((med - baseline_med) / baseline_med) * 100.0 : 0.0;

        printf("%-25s  %8.2f  %8.2f  %8.2f  %+7.2f%%\n",
               results[c].name,
               results[c].avg(),
               med,
               results[c].stdev(),
               overhead_pct);
    }

    printf("\n");

    llama_set_attn_mask(ctx, nullptr, nullptr, 0);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
