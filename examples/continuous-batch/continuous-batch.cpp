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
#include <unordered_set>
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

enum class seq_status { PREFILL, DECODE, DONE };

struct sequence_state {
    int32_t     seq_id           = -1;
    std::string prompt           = "";
    int32_t     n_prompt         = 0;
    int32_t     n_past           = 0;
    int32_t     n_decoded        = 0;
    seq_status  status           = seq_status::PREFILL;
    int64_t     t_arrival_us     = 0;
    int64_t     t_first_token_us = 0;
    int64_t     t_finished_us    = 0;

    std::vector<llama_token> prompt_tokens;
    std::vector<llama_token> output_tokens;

    common_sampler * sampler = nullptr;
};

struct request_result {
    int32_t     request_id = -1;
    int32_t     n_prompt   = 0;
    int32_t     n_decoded  = 0;
    float       ttft_ms    = 0.f;  // time to first token
    float       tpot_ms    = 0.f;  // avg time per output token (excl. first)
    float       tps        = 0.f;  // generation tokens/s
    float       e2e_ms     = 0.f;  // arrival to last token
    std::string response   = "";
};

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    srand(1234);

    common_params params;

    // Mirror paged main.cpp defaults exactly for a fair comparison
    params.kv_unified = true;
    params.kv_paged   = false;
    params.warmup     = false;

    // Example set of parameters that work
    // params.n_sequences = 100;
    // params.n_parallel  = params.n_sequences;
    // params.n_predict   = 150;
    // params.n_batch     = 1024;
    // params.n_ubatch    = params.n_batch;

    // Using PAGED examples args because I don't want to create a new one for unified
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

    const llama_vocab * vocab       = llama_model_get_vocab(model);
    const int32_t       n_sequences = params.n_sequences;
    const int32_t       n_predict   = params.n_predict;
    const int32_t       n_ctx_seq   = llama_n_ctx_seq(ctx);

    LOG_INF("%s: Start continuous batching loop. n_seq=%d, n_predict=%d, n_batch=%d\n", __func__, n_sequences,
            n_predict, params.n_batch);

    // Initialize sequences
    std::vector<sequence_state> seqs(n_sequences);
    for (int i = 0; i < n_sequences; ++i) {
        sequence_state & s = seqs[i];
        s.seq_id           = i;
        s.prompt           = k_prompts[i % k_n_prompts];
        s.t_arrival_us     = ggml_time_us();
        s.status           = seq_status::PREFILL;
        s.sampler          = common_sampler_init(model, params.sampling);

        s.prompt_tokens = common_tokenize(ctx, s.prompt, /*add_bos=*/true);
        s.n_prompt      = (int32_t) s.prompt_tokens.size();

        if (s.n_prompt > n_ctx_seq) {
            LOG_WRN("%s: prompt for seq %d too long (%d > %d), truncating\n", __func__, i, s.n_prompt, n_ctx_seq);
            s.prompt_tokens.resize(n_ctx_seq);
            s.n_prompt = n_ctx_seq;
        }

        LOG_INF("%s: seq %d: \"%s\" (%d tokens)\n", __func__, i, s.prompt.c_str(), s.n_prompt);
    }

    // Continuous batching inference loop
    const int64_t                            t_start_us = ggml_time_us();
    int32_t                                  n_finished = 0;
    std::vector<request_result>              results;
    std::unordered_map<int32_t, std::string> accumulated_responses;

    while (n_finished < n_sequences) {
        llama_batch batch = llama_batch_init(params.n_batch, 0, 1);
        batch.n_tokens    = 0;

        std::unordered_map<int32_t, int32_t> logit_pos;            // seq_id mapped to batch index
        std::unordered_set<int32_t>          decode_submitted;     // sequences with decode token in pass 1
        int32_t                              tokens_in_batch = 0;  // used for token budget

        // Pass 1 (decode has higher priority): one decode token per decode sequence
        for (int i = 0; i < n_sequences; ++i) {
            sequence_state & s = seqs[i];
            if (s.status != seq_status::DECODE) {
                continue;
            }
            if (tokens_in_batch >= params.n_batch) {
                break;
            }

            const llama_token last_tok = s.output_tokens.empty() ? s.prompt_tokens.back() : s.output_tokens.back();

            batch.token[batch.n_tokens]     = last_tok;
            batch.pos[batch.n_tokens]       = s.n_past;
            batch.n_seq_id[batch.n_tokens]  = 1;
            batch.seq_id[batch.n_tokens][0] = s.seq_id;
            batch.logits[batch.n_tokens]    = 1;

            logit_pos[s.seq_id] = batch.n_tokens;
            decode_submitted.insert(s.seq_id);
            batch.n_tokens++;
            tokens_in_batch++;
        }

        // Pass 2: prefill tokens within remaining budget
        for (int i = 0; i < n_sequences; ++i) {
            sequence_state & s = seqs[i];
            if (s.status != seq_status::PREFILL) {
                continue;
            }

            const int32_t submitted = s.n_past;
            const int32_t remaining = s.n_prompt - submitted;
            if (remaining <= 0) {
                s.status = seq_status::DECODE;
                continue;
            }

            // Budget system to avoid running out of VRAM for decodes
            const int32_t budget = params.n_batch - tokens_in_batch;
            const int32_t to_add = std::min(remaining, budget);
            if (to_add <= 0) {
                break;
            }

            for (int32_t tok_pos = 0; tok_pos < to_add; ++tok_pos) {
                const bool is_last = (tok_pos == (to_add - 1)) && (submitted + to_add == s.n_prompt);

                batch.token[batch.n_tokens]     = s.prompt_tokens[submitted + tok_pos];
                batch.pos[batch.n_tokens]       = submitted + tok_pos;
                batch.n_seq_id[batch.n_tokens]  = 1;
                batch.seq_id[batch.n_tokens][0] = s.seq_id;
                batch.logits[batch.n_tokens]    = is_last ? 1 : 0;

                if (is_last) {
                    logit_pos[s.seq_id] = batch.n_tokens;
                }

                batch.n_tokens++;
                tokens_in_batch++;
            }

            s.n_past += to_add;
            if (s.n_past >= s.n_prompt) {
                s.status = seq_status::DECODE;
            }
        }

        if (batch.n_tokens == 0) {
            llama_batch_free(batch);
            break;
        }

        LOG_INF("batch.n_tokens=%d, n_batch=%d, n_sequences=%d, n_parallel=%d\n", batch.n_tokens, params.n_batch,
                params.n_sequences, params.n_parallel);
        GGML_ASSERT(batch.n_tokens <= params.n_batch && "batch exceeds n_batch");

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: llama_decode failed\n", __func__);
            llama_batch_free(batch);
            break;
        }
        llama_synchronize(ctx);

        // Sampling the decoded tokens
        for (auto & [seq_id, logit_idx] : logit_pos) {
            sequence_state & s = seqs[seq_id];
            if (s.status == seq_status::DONE) {
                continue;
            }

            llama_token next_tok = common_sampler_sample(s.sampler, ctx, logit_idx);
            common_sampler_accept(s.sampler, next_tok, /*accept_grammar=*/true);
            accumulated_responses[seq_id] += common_token_to_piece(ctx, next_tok);

            if (s.n_decoded == 0) {
                s.t_first_token_us = ggml_time_us();
            }

            s.output_tokens.push_back(next_tok);
            s.n_decoded++;
            if (decode_submitted.count(seq_id)) {
                s.n_past++;
            }

            // TTFT will only be relevant for first generated token the rest will be 0.0
            LOG_INF("[Request %d] token=%s, decoded=%d, ttft=%.1fms\n", seq_id,
                    common_token_to_piece(ctx, next_tok).c_str(), s.n_decoded,
                    s.n_decoded == 1 ? (s.t_first_token_us - s.t_arrival_us) / 1000.0f : 0.0f);

            const bool is_eog    = llama_vocab_is_eog(vocab, next_tok);
            const bool hit_limit = s.n_decoded >= n_predict;
            const bool ctx_full  = s.n_past >= n_ctx_seq;

            if (is_eog || hit_limit || ctx_full) {
                s.t_finished_us = ggml_time_us();
                s.status        = seq_status::DONE;
                n_finished++;

                llama_memory_seq_rm(llama_get_memory(ctx), s.seq_id, -1, -1);

                request_result r;
                r.request_id = s.seq_id;
                r.n_prompt   = s.n_prompt;
                r.n_decoded  = s.n_decoded;
                r.ttft_ms    = (s.t_first_token_us - s.t_arrival_us) / 1000.0f;
                r.e2e_ms     = (s.t_finished_us - s.t_arrival_us) / 1000.0f;
                r.tps        = s.n_decoded > 0 && s.t_finished_us > s.t_first_token_us ?
                                   s.n_decoded / ((s.t_finished_us - s.t_first_token_us) / 1e6f) :
                                   0.0f;
                r.tpot_ms    = s.n_decoded > 1 && s.t_finished_us > s.t_first_token_us ?
                                   (s.t_finished_us - s.t_first_token_us) / 1000.0f / (s.n_decoded - 1) :
                                   0.0f;
                r.response   = accumulated_responses.count(s.seq_id) ? accumulated_responses[s.seq_id] : "";
                results.push_back(r);

                LOG_INF("[Request %d] finished. decoded=%d tokens, tps=%.1f\n", s.seq_id, s.n_decoded,
                        s.n_decoded / ((s.t_finished_us - s.t_arrival_us) / 1e6f));

                if (ctx_full && !is_eog && !hit_limit) {
                    LOG_WRN("%s: seq %d hit context limit\n", __func__, seq_id);
                }
                accumulated_responses.erase(s.seq_id);
            }
        }

        llama_batch_free(batch);
    }

    // Per-request output
    std::sort(results.begin(), results.end(),
              [](const request_result & a, const request_result & b) { return a.request_id < b.request_id; });

    LOG_INF("%s: OUTPUTS:\n", __func__);
    for (const auto & r : results) {
        LOG_INF("Request Id: %d:<s>%s<\\s>\n", r.request_id, r.response.c_str());
    }

    // Summary metrics
    const float elapsed_s = (ggml_time_us() - t_start_us) / 1e6f;

    int32_t total_prompt_tokens  = 0;
    int32_t total_decoded_tokens = 0;
    float   sum_ttft_ms          = 0.f;
    float   sum_tpot_ms          = 0.f;  // avg time per output token
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
    LOG_INF("=== Unified KV Cache Summary ===\n");
    LOG_INF("  n_sequences          : %d\n", n_sequences);
    LOG_INF("  n_predict            : %d\n", n_predict);
    LOG_INF("  n_batch              : %d\n", params.n_batch);
    LOG_INF("  total elapsed        : %.2f s\n", elapsed_s);
    LOG_INF("  total prompt tokens  : %d\n", total_prompt_tokens);
    LOG_INF("  total decoded tokens : %d\n", total_decoded_tokens);
    LOG_INF("  aggregate tps        : %.2f tokens/s\n", agg_tps);
    LOG_INF("  --- per-request latency ---\n");
    LOG_INF("  ttft  avg / min / max : %.1f / %.1f / %.1f ms\n", avg_ttft_ms, min_ttft_ms, max_ttft_ms);
    LOG_INF("  tpot  avg             : %.1f ms/token\n", avg_tpot_ms);
    LOG_INF("  e2e   avg             : %.1f ms\n", avg_e2e_ms);
    LOG_INF("  tps   min / max       : %.1f / %.1f tokens/s\n", min_tps, max_tps);
    LOG_INF("================================\n");

    // Clean-up
    for (auto & s : seqs) {
        if (s.sampler) {
            common_sampler_free(s.sampler);
        }
    }
    llama_backend_free();
    return 0;
}
