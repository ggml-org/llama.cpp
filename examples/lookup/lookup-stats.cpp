#include "ggml.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>

int main(int argc, char ** argv){
    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }

    const int n_draft = params.n_draft;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    // load the model
    llama_init_result llama_init = llama_init_from_gpt_params(params);

    llama_model * model = llama_init.model;
    llama_context * ctx = llama_init.context;

    // tokenize the prompt
    std::vector<llama_token> inp;
    inp = ::llama_tokenize(ctx, params.prompt, true, true);

    llama_ngram_cache ngram_cache_context;
    llama_ngram_cache ngram_cache_dynamic;
    llama_ngram_cache ngram_cache_static;
    int64_t t_draft_flat_us = 0;
    int64_t t_draft_us = 0;

    {
        const int64_t t_start_draft_us = ggml_time_us();

        if (!params.lookup_cache_static.empty()) {
            try {
                ngram_cache_static = llama_ngram_cache_load(params.lookup_cache_static);
            } catch (std::ifstream::failure const &) {
                fprintf(stderr, "error: failed to open static lookup cache: %s", params.lookup_cache_static.c_str());
                exit(1);
            }
        }

        if (!params.lookup_cache_dynamic.empty()) {
            try {
                ngram_cache_dynamic = llama_ngram_cache_load(params.lookup_cache_dynamic);
            } catch (std::ifstream::failure const &) {} // if the file does not exist it will simply be created at the end of the program
        }

        t_draft_flat_us += ggml_time_us() - t_start_draft_us;
    }

    const int n_input = inp.size();
    const int n_ctx = llama_n_ctx(ctx);

    int n_drafted = 0;
    int n_accept  = 0;

    const int64_t t_start_ms = ggml_time_ms();

    // Iterate over input tokens in chunks of size n_ctx.
    // Each chunk is treated as if a sequential generation but with pre-determined tokens to ensure reproducibility.
    for (int i_start = 0; i_start + n_ctx < n_input; i_start += n_ctx) {
        const std::vector<llama_token> inp_slice(inp.begin() + i_start, inp.begin() + i_start + n_ctx);
        std::vector<llama_token> pseudo_output;
        pseudo_output.push_back(inp_slice[0]);

        while ((int) pseudo_output.size() < n_ctx) {
            // Simulate drafting and decoding from draft:
            std::vector<llama_token> draft;
            draft.push_back(pseudo_output.back());

            {
                const int64_t t_start_draft_us = ggml_time_us();
                llama_ngram_cache_draft(pseudo_output, draft, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, ngram_cache_context, ngram_cache_dynamic, ngram_cache_static);
                t_draft_us += ggml_time_us() - t_start_draft_us;
            }

            n_drafted += draft.size() - 1;

            for (size_t j = 1; j < draft.size() && (int) pseudo_output.size() < n_ctx; ++j) {
                const llama_token ground_truth = inp_slice[pseudo_output.size()];
                const llama_token drafted = draft[j];

                if (ground_truth != drafted) {
                    break;
                }

                ++n_accept;
                pseudo_output.push_back(ground_truth);

                {
                    const int64_t t_start_draft_us = ggml_time_us();
                    llama_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, pseudo_output, 1, false);
                    t_draft_us += ggml_time_us() - t_start_draft_us;
                }
            }

            // After each simulated batch decoding simulate the sampling of a single token:
            if ((int) pseudo_output.size() < n_ctx) {
                pseudo_output.push_back(inp_slice[pseudo_output.size()]);
                {
                    const int64_t t_start_draft_us = ggml_time_us();
                    llama_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, pseudo_output, 1, false);
                    t_draft_us += ggml_time_us() - t_start_draft_us;
                }
            }

            draft.erase(draft.begin());

        }
        if (i_start > 0 && i_start / 100000 != (i_start - n_ctx) / 100000) {
            const int64_t t_now_ms = ggml_time_ms();
            const int64_t eta_ms   = (n_input - i_start) * (t_now_ms - t_start_ms) / i_start;
            const int64_t eta_min  = eta_ms / (60*1000);
            const int64_t eta_s    = (eta_ms - 60*1000*eta_min) / 1000;

            LOG_TEE("lookup-stats: %d/%d done, ETA: %02" PRId64 ":%02" PRId64 "\n", i_start, n_input, eta_min, eta_s);
        }

        // After each chunk, update the dynamic ngram cache with the context ngram cache:
        llama_ngram_cache_merge(ngram_cache_dynamic, ngram_cache_context);
        ngram_cache_context.clear();
    }

    LOG_TEE("\n");

    LOG_TEE("\n");
    LOG_TEE("n_draft      = %d\n", n_draft);
    LOG_TEE("n_predict    = %d\n", n_input - n_input % n_ctx);
    LOG_TEE("n_drafted    = %d\n", n_drafted);
    LOG_TEE("t_draft_flat = %.2f ms\n", t_draft_flat_us*1e-3);
    LOG_TEE("t_draft      = %.2f ms, %.2f us per token, %.2f tokens per second\n",
            t_draft_us*1e-3, 1.0f*t_draft_us/n_drafted, n_drafted/(1e-6*t_draft_us));
    LOG_TEE("n_accept     = %d\n", n_accept);
    LOG_TEE("accept       = %.3f%%\n", 100.0f * n_accept / n_drafted);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    fprintf(stderr, "\n\n");

    return 0;
}
