#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <limits.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

typedef bool (*diffusion_step_callback_t)(int32_t step,
                                          int32_t total_steps,
                                          const llama_token * tokens,
                                          int32_t n_tokens,
                                          void * user_data);

enum diffusion_alg {
    DIFFUSION_ALG_ORIGIN       = 0,
    DIFFUSION_ALG_MASKGIT_PLUS = 1,
    DIFFUSION_ALG_TOPK_MARGIN  = 2,
    DIFFUSION_ALG_ENTROPY      = 3,
};

struct diffusion_params {
    int32_t                   steps;
    float                     eps;
    float                     temperature;
    float                     top_p;
    int32_t                   top_k;
    llama_token               mask_token_id;
    enum diffusion_alg        algorithm;
    float                     alg_temp;
    diffusion_step_callback_t step_callback;
    void *                    step_callback_user_data;
    int32_t                   seed;
};


static diffusion_params diffusion_default_params() {
    diffusion_params params        = {};
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

static void diffusion_generate(llama_context * ctx,
                        const llama_token * input_tokens,
                        llama_token * output_tokens,
                        int32_t n_input,
                        int32_t max_length,
                        struct diffusion_params params,
                        int32_t & n_generated) {

    n_generated = 0;
    if (!ctx || !input_tokens || !output_tokens || n_input <= 0 || max_length <= n_input) {
        return;
    }

    const llama_model * model = llama_get_model(ctx);

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

    std::vector<llama_token_data> candidates(n_vocab);

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

    llama_batch batch = llama_batch_init(max_length, 0, 1);
    batch.n_tokens    = max_length;

    int64_t total_sampling_time = 0;
    int64_t total_time = 0;

    int64_t time_start = ggml_time_us();
    for (int32_t step = 0; step < params.steps; step++) {
        if (params.step_callback) {
            if (!params.step_callback(step, params.steps, output_tokens, max_length, params.step_callback_user_data)) {
                break;
            }
        }

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
            break;
        }

        float * raw_logits = llama_get_logits(ctx);
        if (!raw_logits) {
            LOG_ERR("%s: failed to get logits at step %d\n", __func__, step);
            break;
        }

        auto get_logits_for_pos = [&](int32_t pos) -> const float * {
            return pos == 0 ? raw_logits : raw_logits + (pos - 1) * n_vocab;
        };

        int64_t time_start_sampling = ggml_time_us();

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
                    const float * pos_logits = get_logits_for_pos(pos);
                    for (int32_t token_id = 0; token_id < n_vocab; token_id++) {
                        candidates[token_id].id    = token_id;
                        candidates[token_id].logit = pos_logits[token_id];
                        candidates[token_id].p     = 0.0f;
                    }

                    llama_token_data_array cur_p = {
                        /* .data       = */ candidates.data(),
                        /* .size       = */ (size_t) n_vocab,  // Reset size to full vocab
                        /* .selected   = */ -1,
                        /* .sorted     = */ false,
                    };

                    llama_sampler_apply(sampler, &cur_p);
                    output_tokens[pos] = cur_p.data[cur_p.selected].id;
                }
            }
        } else {
            std::vector<std::pair<float, int32_t>> confidences;
            std::vector<llama_token>               sampled_tokens(mask_positions.size());

            for (size_t i = 0; i < mask_positions.size(); i++) {
                int32_t       pos        = mask_positions[i];
                const float * pos_logits = get_logits_for_pos(pos);

                for (int32_t token_id = 0; token_id < n_vocab; token_id++) {
                    candidates[token_id].logit = pos_logits[token_id];
                    candidates[token_id].p     = 0.0f;
                    candidates[token_id].id    = token_id;
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
        int64_t time_end_sampling = ggml_time_us();
        total_sampling_time += time_end_sampling - time_start_sampling;
    }
    int64_t time_end = ggml_time_us();
    total_time += time_end - time_start;

    LOG_INF("\ntotal time: %0.2fms, time per step: %0.2fms, sampling time per step: %0.2fms\n",
            total_time / 1000.0, total_time / 1000.0 / params.steps, total_sampling_time / 1000.0 / params.steps);


    llama_batch_free(batch);
    llama_sampler_free(sampler);
    llama_sampler_free(dist_sampler);

    n_generated = max_length;
}




static std::string format_input_text(const std::string & prompt, bool use_chat_template, llama_model * model) {
    if (!use_chat_template) {
        return prompt;
    }

    auto chat_templates = common_chat_templates_init(model, "");

    common_chat_templates_inputs inputs;
    common_chat_msg              user_msg;
    user_msg.role                = "user";
    user_msg.content             = prompt;
    inputs.add_generation_prompt = true;
    inputs.messages.push_back(user_msg);

    auto result = common_chat_templates_apply(chat_templates.get(), inputs);

    return result.prompt;
}

struct callback_data {
    const common_params_diffusion * diff_params;
    const llama_vocab *             vocab;
    int32_t                         n_input;
};

static bool diffusion_step_callback(int32_t step,
                                    int32_t total_steps,
                                    const llama_token * tokens,
                                    int32_t n_tokens,
                                    void * user_data) {
    (void)user_data;

    callback_data * data = static_cast<callback_data *>(user_data);

    auto print_progress_bar = [](int32_t step, int32_t total_steps) {
        int progress_percent = (step * 100) / total_steps;
        int progress_bars    = (step * 50) / total_steps;
        LOG_INF("\rdiffusion step: %d/%d [%s%s] %d%%",
            step,
            total_steps,
            std::string(progress_bars, '=').c_str(),
            std::string(50 - progress_bars, ' ').c_str(),
            progress_percent);
    };

    if (data->diff_params->visual_mode) {
        // Visual mode: clear
        LOG_INF("\033[2J\033[H");  // Clear screen and move cursor to top-left

        print_progress_bar(step, total_steps);

        LOG_INF("\n");

        std::string current_text = " ";

        for (int32_t i = data->n_input; i < n_tokens; i++) {
            std::string token_str;
            if (tokens[i] != llama_vocab_mask(data->vocab)) {
                char piece[256];
                int  n_chars = llama_token_to_piece(data->vocab, tokens[i], piece, sizeof(piece), 0, false);
                if (n_chars > 0) {
                    piece[n_chars] = '\0';
                    token_str      = piece;
                }
            } else {
                token_str = " ";
            }

            current_text += token_str;
        }

        LOG_INF("%s\n", current_text.c_str());
    } else {
        print_progress_bar(step, total_steps);
    }

    return true;
}

int main(int argc, char ** argv) {
    ggml_time_init();

    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DIFFUSION)) {
        return 1;
    }

    const char * alg_names[] = { "ORIGIN", "MASKGIT_PLUS", "TOPK_MARGIN", "ENTROPY" };
    const char * alg_name    = (params.diffusion.algorithm >= 0 && params.diffusion.algorithm <= 3) ?
                                   alg_names[params.diffusion.algorithm] :
                                   "UNKNOWN";

    common_init();
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = params.n_gpu_layers;
    model_params.devices            = params.devices.data();
    model_params.use_mmap           = params.use_mmap;
    model_params.use_mlock          = params.use_mlock;
    model_params.check_tensors      = params.check_tensors;

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (!model) {
        LOG_ERR("error: failed to load model '%s'\n", params.model.path.c_str());
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx                = params.n_ctx;
    ctx_params.n_batch              = params.n_batch;
    ctx_params.n_ubatch             = params.n_ubatch;
    ctx_params.flash_attn           = params.flash_attn;
    ctx_params.no_perf              = params.no_perf;
    ctx_params.type_k               = params.cache_type_k;
    ctx_params.type_v               = params.cache_type_v;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOG_ERR("error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    llama_set_n_threads(ctx, params.cpuparams.n_threads, params.cpuparams_batch.n_threads);

    const llama_vocab * vocab            = llama_model_get_vocab(model);
    std::string         formatted_prompt = format_input_text(params.prompt, params.enable_chat_template, model);

    std::vector<llama_token> input_tokens = common_tokenize(vocab, formatted_prompt,
                                                            /*add special tokens*/ true,
                                                            /*parse special*/ true);
    int                      n_input      = input_tokens.size();

    if (n_input >= params.n_ctx) {
        LOG_ERR("error: input too long (%d tokens), max context is %d\n", n_input, params.n_ctx);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    struct diffusion_params ldiff_params = diffusion_default_params();
    ldiff_params.steps                   = params.diffusion.steps;
    ldiff_params.eps                     = params.diffusion.eps;
    ldiff_params.temperature             = params.sampling.temp;
    ldiff_params.top_p                   = params.sampling.top_p;
    ldiff_params.top_k                   = params.sampling.top_k;
    ldiff_params.algorithm               = static_cast<enum diffusion_alg>(params.diffusion.algorithm);
    ldiff_params.alg_temp                = params.diffusion.alg_temp;
    ldiff_params.seed                    = params.sampling.seed;

    llama_token mask_token_id = llama_vocab_mask(vocab);
    GGML_ASSERT(mask_token_id != LLAMA_TOKEN_NULL);

    LOG_INF("diffusion_params: - %-25s llama_token      = %d\n", "mask_token_id", mask_token_id);
    LOG_INF("diffusion_params: - %-25s u32              = %d\n", "steps", params.diffusion.steps);
    LOG_INF("diffusion_params: - %-25s f32              = %.6f\n", "eps", params.diffusion.eps);
    LOG_INF("diffusion_params: - %-25s u32              = %d (%s)\n", "algorithm", params.diffusion.algorithm,
            alg_name);
    LOG_INF("diffusion_params: - %-25s f32              = %.3f\n", "alg_temp", params.diffusion.alg_temp);

    ldiff_params.mask_token_id = mask_token_id;

    callback_data cb_data = { &params.diffusion, vocab, n_input };

    ldiff_params.step_callback           = diffusion_step_callback;
    ldiff_params.step_callback_user_data = &cb_data;

    int32_t n_generated = 0;

    std::vector<llama_token> output_tokens(params.n_ubatch);
    diffusion_generate(ctx, input_tokens.data(), output_tokens.data(), n_input, params.n_ubatch,
                       ldiff_params, n_generated);

    if (n_generated > 0) {
        if (params.diffusion.visual_mode) {
            //clear screen and move cursor to top-left
            LOG_INF("\033[2J\033[H");
        }
        output_tokens.erase(output_tokens.begin(), output_tokens.begin() + n_input);
        std::string output_data = common_detokenize(vocab, output_tokens, false);
        LOG_INF("\n%s\n", output_data.c_str());
    } else {
        LOG_INF("Error: diffusion generation failed\n");
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
