#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"

#include <algorithm>
#include <cfloat>
#include <clocale>
#include <string>
#include <unordered_map>
#include <vector>

static std::vector<std::string> k_prompts = {
    "What is the tallest mountain in the world?",
    "Who was the first person to win two Nobel Prizes?",
    "Which country invented paper?",
    "What organ is primarily responsible for pumping blood throughout the body?",
    "Which planet is known for its prominent ring system?",
    "Who directed the movie 'Inception'?",
    "What is the freezing point of water in Fahrenheit?",
    "Which animal is known to have the longest lifespan?",
    "What language has the most native speakers worldwide?",
    "What is the capital city of Canada?",
    "Who is credited with inventing the World Wide Web?",
    "Which metal is liquid at room temperature?",
    "What is the term for an animal that eats both plants and meat?",
    "Who painted 'The Starry Night'?",
    "What gas do humans exhale that plants use for photosynthesis?",
    "What year did World War II end?",
    "Which continent has the most countries?",
    "Who wrote the novel 'Frankenstein'?",
    "What does DNA stand for?",
    "What is the main ingredient in traditional Japanese miso soup?"
};
static const size_t k_n_prompts = 20;

struct request_result {
    int32_t     request_id = -1;
    int32_t     n_prompt   = 0;
    int32_t     n_decoded  = 0;
    float       ttft_ms    = 0.f;  // time to first token
    float       tpot_ms    = 0.f;  // avg time per output token (excluding the first token)
    float       tps        = 0.f;  // generation tokens/s
    float       e2e_ms     = 0.f;  // time from arrival to last token
    std::string response   = "";
};

static void add_request_from_pool(struct llama_paged_scheduler * scheduler,
                                  llama_context *                ctx,
                                  size_t                         pool_index,
                                  int                            seq_id) {
    const std::string        input_prompt = k_prompts[pool_index % k_n_prompts];
    std::vector<llama_token> tokens       = common_tokenize(ctx, input_prompt, true);

    bool success = llama_paged_scheduler_add_request(scheduler, tokens.data(), tokens.size(), seq_id);
    if (!success) {
        LOG_ERR("Failed to add request %ld from pool\n", pool_index);
    }
    LOG_INF("%s: Successfully added request %ld: %s\n", __func__, pool_index, input_prompt.c_str());
}

struct callback_context {
    llama_context * ctx;
};

static void log_output(int32_t req_id, const llama_token * /*tokens*/, int32_t n_tokens, void * user_data) {
    if (!user_data) {
        printf("\n--- [Request %d n_tokens] ---\n%d\n---------------------------\n", req_id, n_tokens);
        return;
    }
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    srand(1234);

    common_params params;
    params.warmup = false;  // we do not support warmup for paged KV cache yet

    // Example parameters that work
    // params.n_sequences = 100;
    // params.n_parallel  = params.n_sequences;
    // params.n_predict   = 150;
    // params.n_batch     = 1024;
    // params.n_ubatch    = params.n_batch;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PAGED)) {
        return 1;
    }

    if (params.n_sequences != params.n_parallel) {
        LOG_INF("%s: n_sequences (%d) needs to be equal to n_parallel (%d)\n", __func__, params.n_sequences,
                params.n_parallel);
        return 1;
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto   llama_init = common_init_from_params(params);
    auto * model      = llama_init->model();
    auto * ctx        = llama_init->context();

    if (!model || !ctx) {
        LOG_ERR("%s: failed to load model or create context\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    LOG_INF("%s: Loaded model and created context\n", __func__);

    struct llama_paged_scheduler * scheduler = llama_paged_scheduler_init(ctx);
    if (!scheduler) {
        LOG_ERR("%s: Failed to initialize scheduler.\n", __func__);
        return 1;
    }

    callback_context cb_ctx = { ctx };
    llama_paged_scheduler_set_on_finish(scheduler, log_output, &cb_ctx);

    std::unordered_map<int32_t, common_sampler *> samplers;
    for (int i = 0; i < params.n_sequences; ++i) {
        add_request_from_pool(scheduler, ctx, (size_t) i, i);
        samplers[i] = common_sampler_init(model, params.sampling);
    }

    std::vector<request_result>              results;
    std::unordered_map<int32_t, std::string> accumulated_responses;

    llama_batch batch = {};
    LOG_INF(
        "%s: Start continuous batching loop. n_seq=%d, n_predict=%d, "
        "n_gpu_blocks=%d, n_cpu_blocks=%d\n",
        __func__, params.n_sequences, params.n_predict, params.n_gpu_blocks, params.n_cpu_blocks);

    const int64_t t_start_us = ggml_time_us();

    while (true) {
        const bool success = llama_paged_scheduler_prepare_batch(scheduler, &batch);
        if (!success || batch.n_tokens == 0) {
            break;
        }

        LOG_DBG("prepared batch: batch.n_tokens=%d, n_batch=%d, n_sequences=%d, n_parallel=%d\n", batch.n_tokens,
                params.n_batch, params.n_sequences, params.n_parallel);
        GGML_ASSERT(batch.n_tokens <= params.n_batch && "batch exceeds n_batch");

        if (llama_decode(ctx, batch) != 0) {
            LOG_INF("%s: llama_decode failed\n", __func__);
            break;
        }
        llama_synchronize(ctx);

        std::vector<llama_token> sampled_tokens;
        std::vector<int8_t>      stop_flags;

        const llama_paged_batch_info * info = llama_paged_scheduler_get_batch_info(scheduler);
        GGML_ASSERT(info != nullptr && "llama_paged_batch_info is nullptr.");
        for (int i = 0; i < info->n_seq; ++i) {
            int32_t request_id = batch.seq_id[info->batch_offsets[i]][0];

            auto it = samplers.find(request_id);
            if (it == samplers.end()) {
                samplers[request_id] = common_sampler_init(model, params.sampling);
                it                   = samplers.find(request_id);
            }
            common_sampler * sampler = it->second;

            int32_t     last_token_in_batch = info->batch_offsets[i] + info->batch_lens[i] - 1;
            llama_token next_token          = common_sampler_sample(sampler, ctx, last_token_in_batch);
            common_sampler_accept(sampler, next_token, /*accept_grammar=*/true);
            sampled_tokens.push_back(next_token);
            accumulated_responses[request_id] += common_token_to_piece(ctx, next_token);

            llama_paged_seq_state state = {};
            llama_paged_scheduler_get_seq_state(scheduler, request_id, &state);

            LOG_DBG("[Request %d] token=%s, decoded=%d, ttft=%.1fms\n", request_id,
                    common_token_to_piece(ctx, next_token).c_str(), state.n_decoded,
                    state.n_decoded == 1 ? (state.t_first_token_us - state.t_arrival_us) / 1000.0f : 0.0f);

            bool stop = llama_vocab_is_eog(vocab, next_token) || state.n_decoded >= params.n_predict;
            stop_flags.push_back(stop ? 1 : 0);

            if (stop) {
                const int64_t t_finished = ggml_time_us();  // microsecs

                request_result r;
                r.request_id = request_id;
                r.n_prompt   = state.n_prompt;
                r.n_decoded  = state.n_decoded;
                r.ttft_ms    = (state.t_first_token_us - state.t_arrival_us) / 1000.0f;
                r.e2e_ms     = (t_finished - state.t_arrival_us) / 1000.0f;
                r.tps        = state.n_decoded > 0 && t_finished > state.t_first_token_us ?
                                   state.n_decoded / ((t_finished - state.t_first_token_us) / 1e6f) :
                                   0.0f;
                r.tpot_ms    = state.n_decoded > 1 && t_finished > state.t_first_token_us ?
                                   (t_finished - state.t_first_token_us) / 1000.0f / (state.n_decoded - 1) :
                                   0.0f;
                r.response   = accumulated_responses.count(request_id) ? accumulated_responses[request_id] : "";
                results.push_back(r);

                LOG_DBG("[Request %d] finished. decoded=%d tokens, tps=%.1f\n", request_id, state.n_decoded,
                        state.n_decoded / (r.e2e_ms / 1000.0f));

                accumulated_responses.erase(request_id);
                common_sampler_free(samplers[request_id]);
                samplers.erase(request_id);
            }
        }

        llama_paged_scheduler_update(scheduler, &batch, sampled_tokens.data(), stop_flags.data());
    }

    LOG_INF("%s: Finished paged example.\n", __func__);

    // For easy visualization of output
    std::sort(results.begin(), results.end(),
              [](const request_result & a, const request_result & b) { return a.request_id < b.request_id; });

    LOG_INF("%s: Paged KV cache outputs:\n", __func__);
    for (const auto & res : results) {
        LOG_INF("Request Id: %d:<s>%s<\\s>\n", res.request_id, res.response.c_str());
    }

    // Summary
    const float elapsed_s = (ggml_time_us() - t_start_us) / 1e6f;

    int32_t total_prompt_tokens  = 0;
    int32_t total_decoded_tokens = 0;
    float   sum_ttft_ms          = 0.f;
    float   sum_tpot_ms          = 0.f;
    float   sum_e2e_ms           = 0.f;
    float   min_ttft_ms          = FLT_MAX;
    float   max_ttft_ms          = 0.f;
    float   min_tps              = FLT_MAX;
    float   max_tps              = 0.f;

    for (const auto & r : results) {
        total_prompt_tokens += r.n_prompt;
        total_decoded_tokens += r.n_decoded;
        sum_ttft_ms += r.ttft_ms;
        sum_tpot_ms += r.tpot_ms;
        sum_e2e_ms += r.e2e_ms;
        min_ttft_ms = std::min(min_ttft_ms, r.ttft_ms);
        max_ttft_ms = std::max(max_ttft_ms, r.ttft_ms);
        min_tps     = std::min(min_tps, r.tps);
        max_tps     = std::max(max_tps, r.tps);
    }

    const int32_t n_results   = (int32_t) results.size();
    const float   avg_ttft_ms = n_results > 0 ? sum_ttft_ms / n_results : 0.f;
    const float   avg_tpot_ms = n_results > 0 ? sum_tpot_ms / n_results : 0.f;
    const float   avg_e2e_ms  = n_results > 0 ? sum_e2e_ms / n_results : 0.f;
    const float   agg_tps     = elapsed_s > 0 ? total_decoded_tokens / elapsed_s : 0.f;

    LOG_INF("\n");
    LOG_INF("=== Paged KV Cache Summary ===\n");
    LOG_INF("  n_sequences          : %d\n", params.n_sequences);
    LOG_INF("  n_predict            : %d\n", params.n_predict);
    LOG_INF("  n_batch              : %d\n", params.n_batch);
    LOG_INF("  n_gpu_blocks         : %d\n", params.n_gpu_blocks);
    LOG_INF("  n_cpu_blocks         : %d\n", params.n_cpu_blocks);
    LOG_INF("  total elapsed        : %.2f s\n", elapsed_s);
    LOG_INF("  total prompt tokens  : %d\n", total_prompt_tokens);
    LOG_INF("  total decoded tokens : %d\n", total_decoded_tokens);
    LOG_INF("  aggregate tps        : %.2f tokens/s\n", agg_tps);
    LOG_INF("  --- per-request latency ---\n");
    LOG_INF("  ttft  avg / min / max : %.1f / %.1f / %.1f ms\n", avg_ttft_ms, min_ttft_ms, max_ttft_ms);
    LOG_INF("  tpot  avg             : %.1f ms/token\n", avg_tpot_ms);
    LOG_INF("  e2e   avg             : %.1f ms\n", avg_e2e_ms);
    LOG_INF("  tps   min / max       : %.1f / %.1f tokens/s\n", min_tps, max_tps);
    LOG_INF("==============================\n");

    // Clean-up
    for (auto & [rid, s] : samplers) {
        common_sampler_free(s);
    }
    llama_paged_scheduler_free(scheduler);
    llama_backend_free();

    return 0;
}
