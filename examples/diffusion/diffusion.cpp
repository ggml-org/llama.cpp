#include "diffusion.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

struct diffusion_params diffusion_default_params(void) {
    struct diffusion_params params = {};
    params.steps                   = 64;
    params.eps                     = 1e-3f;
    params.temperature             = 0.2f;
    params.top_p                   = 0.95f;
    params.top_k                   = 0;
    params.mask_token_id           = LLAMA_TOKEN_NULL;
    params.algorithm               = DIFFUSION_ALG_ORIGIN;
    params.alg_temp                = 0.0f;
    params.step_callback           = nullptr;
    params.step_callback_user_data = nullptr;
    params.seed                    = 0;
    return params;
}

llama_token * diffusion_generate(llama_context * ctx, const llama_token * input_tokens, int32_t n_input,
                                 int32_t max_length, struct diffusion_params params, int32_t * n_generated) {
    if (!ctx || !input_tokens || n_input <= 0 || max_length <= n_input) {
        if (n_generated) {
            *n_generated = 0;
        }
        return nullptr;
    }

    const llama_model * model = llama_get_model(ctx);
    if (!model) {
        if (n_generated) {
            *n_generated = 0;
        }
        return nullptr;
    }

    llama_token * output_tokens = new llama_token[max_length];
    if (!output_tokens) {
        if (n_generated) {
            *n_generated = 0;
        }
        return nullptr;
    }

    // Initialize with input and pad with mask tokens
    std::copy(input_tokens, input_tokens + n_input, output_tokens);
    std::fill(output_tokens + n_input, output_tokens + max_length, params.mask_token_id);

    std::mt19937 rng(params.seed);

    std::vector<float> timesteps(params.steps + 1);
    for (int32_t i = 0; i <= params.steps; i++) {
        timesteps[i] = 1.0f - (float) i / params.steps * (1.0f - params.eps);
    }

    llama_set_causal_attn(ctx, false);

    int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);
    std::vector<llama_token_data> conf_candidates;
    conf_candidates.reserve(max_length);
    std::vector<int32_t> mask_positions;
    mask_positions.reserve(max_length);

    struct llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (params.top_k > 0) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(params.top_k));
    }
    if (params.top_p < 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(params.top_p, 1));
    }
    if (params.temperature > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(params.temperature));
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(params.seed));

    struct llama_sampler * dist_sampler = llama_sampler_init_dist(params.seed);

    for (int32_t step = 0; step < params.steps; step++) {
        if (params.step_callback) {
            if (!params.step_callback(step, params.steps, output_tokens, max_length, params.step_callback_user_data)) {
                break;
            }
        }

        llama_batch batch = llama_batch_init(max_length, 0, 1);
        batch.n_tokens    = max_length;
        for (int32_t i = 0; i < max_length; i++) {
            batch.token[i]     = output_tokens[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;
        }

        int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            LOG_ERR("%s: failed to decode at step %d, ret = %d\n", __func__, step, ret);
            llama_batch_free(batch);
            delete[] output_tokens;
            if (n_generated) {
                *n_generated = 0;
            }
            return nullptr;
        }

        float * raw_logits = llama_get_logits(ctx);
        if (!raw_logits) {
            LOG_ERR("%s: failed to get logits at step %d\n", __func__, step);
            llama_batch_free(batch);
            delete[] output_tokens;
            if (n_generated) {
                *n_generated = 0;
            }
            return nullptr;
        }

        std::vector<float> shifted_logits(max_length * n_vocab);

        //Move logits to left, because decode path will shift logits to right
        // Position 0 keeps its own logits: shifted_logits[0] = raw_logits[0]
        std::copy(raw_logits, raw_logits + n_vocab, shifted_logits.data());

        // Positions 1+ get logits from previous position: shifted_logits[i] = raw_logits[i-1]
        for (int32_t i = 1; i < max_length; i++) {
            std::copy(raw_logits + (i - 1) * n_vocab, raw_logits + i * n_vocab, shifted_logits.data() + i * n_vocab);
        }

        float * logits = shifted_logits.data();

        mask_positions.clear();
        for (int32_t i = 0; i < max_length; i++) {
            if (output_tokens[i] == params.mask_token_id) {
                mask_positions.push_back(i);
            }
        }

        if (mask_positions.empty()) {
            break;
        }

        float t = timesteps[step];
        float s = timesteps[step + 1];

        if (params.algorithm == DIFFUSION_ALG_ORIGIN) {
            float p_transfer = (step < params.steps - 1) ? (1.0f - s / t) : 1.0f;

            for (int32_t pos : mask_positions) {
                if (std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < p_transfer) {
                    candidates.clear();
                    for (int32_t token_id = 0; token_id < n_vocab; token_id++) {
                        candidates.emplace_back(llama_token_data{ token_id, logits[pos * n_vocab + token_id], 0.0f });
                    }

                    llama_token_data_array cur_p = {
                        /* .data       = */ candidates.data(),
                        /* .size       = */ candidates.size(),
                        /* .selected   = */ -1,
                        /* .sorted     = */ false,
                    };

                    llama_sampler_apply(sampler, &cur_p);
                    output_tokens[pos] = cur_p.data[cur_p.selected].id;
                }
            }
        } else {
            candidates.clear();
            candidates.shrink_to_fit();

            std::vector<std::pair<float, int32_t>> confidences;
            std::vector<llama_token>               sampled_tokens(mask_positions.size());

            for (size_t i = 0; i < mask_positions.size(); i++) {
                int32_t pos        = mask_positions[i];
                float * pos_logits = logits + pos * n_vocab;

                candidates.clear();
                for (int32_t token_id = 0; token_id < n_vocab; token_id++) {
                    candidates.emplace_back(llama_token_data{ token_id, pos_logits[token_id], 0.0f });
                }

                llama_token_data_array cur_p = {
                    /* .data       = */ candidates.data(),
                    /* .size       = */ candidates.size(),
                    /* .selected   = */ -1,
                    /* .sorted     = */ false,
                };

                llama_sampler_apply(sampler, &cur_p);

                llama_token sampled_token = cur_p.data[cur_p.selected].id;

                float confidence = 0.0f;
                if (params.algorithm == DIFFUSION_ALG_ENTROPY) {
                    const float epsilon = 1e-10f;
                    for (size_t j = 0; j < cur_p.size; j++) {
                        float prob = cur_p.data[j].p;
                        confidence += prob * logf(prob + epsilon);
                    }
                } else if (params.algorithm == DIFFUSION_ALG_TOPK_MARGIN) {
                    std::partial_sort(cur_p.data, cur_p.data + 2, cur_p.data + cur_p.size,
                                      [](const llama_token_data & a, const llama_token_data & b) { return a.p > b.p; });
                    confidence = cur_p.data[0].p - cur_p.data[1].p;
                } else {
                    confidence = cur_p.data[cur_p.selected].p;
                }

                sampled_tokens[i] = sampled_token;
                confidences.emplace_back(confidence, i);
            }

            int32_t num_transfer =
                (step < params.steps - 1) ? (int32_t) (mask_positions.size() * (1.0f - s / t)) : mask_positions.size();

            if (num_transfer > 0) {
                if (params.alg_temp == 0.0f) {
                    std::partial_sort(confidences.begin(), confidences.begin() + num_transfer, confidences.end(),
                                      [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                                          if (a.first != b.first) {
                                              return a.first > b.first;
                                          }
                                          return a.second < b.second;
                                      });
                } else {
                    conf_candidates.clear();

                    for (int32_t pos = 0; pos < max_length; pos++) {
                        float conf_logit = -std::numeric_limits<float>::infinity();

                        auto it = std::find(mask_positions.begin(), mask_positions.end(), pos);
                        if (it != mask_positions.end()) {
                            size_t mask_idx = std::distance(mask_positions.begin(), it);
                            conf_logit = confidences[mask_idx].first / params.alg_temp;  // Apply temperature scaling
                        }

                        conf_candidates.emplace_back(llama_token_data{ pos, conf_logit, 0.0f });
                    }

                    llama_token_data_array conf_array = {
                        /* .data       = */ conf_candidates.data(),
                        /* .size       = */ conf_candidates.size(),
                        /* .selected   = */ -1,
                        /* .sorted     = */ false,
                    };

                    for (int32_t i = 0; i < num_transfer; i++) {
                        // Apply distribution sampler to get selected index
                        llama_sampler_apply(dist_sampler, &conf_array);
                        int selected_idx      = conf_array.selected;
                        confidences[i].second = conf_candidates[selected_idx].id;

                        conf_candidates[selected_idx].p = 0.0f;
                        conf_array.selected             = -1;
                    }
                }

                if (params.alg_temp == 0.0f) {
                    // Deterministic - use confidence order
                    for (int32_t i = 0; i < num_transfer; i++) {
                        int32_t     mask_idx = confidences[i].second;
                        int32_t     pos      = mask_positions[mask_idx];
                        llama_token token    = sampled_tokens[mask_idx];
                        output_tokens[pos]   = token;
                    }
                } else {
                    for (int32_t i = 0; i < num_transfer; i++) {
                        int32_t pos = confidences[i].second;
                        auto    it  = std::find(mask_positions.begin(), mask_positions.end(), pos);
                        if (it != mask_positions.end()) {
                            int32_t mask_idx   = std::distance(mask_positions.begin(), it);
                            output_tokens[pos] = sampled_tokens[mask_idx];
                        }
                    }
                }
            }
        }

        llama_batch_free(batch);
    }

    llama_sampler_free(sampler);
    llama_sampler_free(dist_sampler);

    if (n_generated) {
        *n_generated = max_length;
    }

    return output_tokens;
}
