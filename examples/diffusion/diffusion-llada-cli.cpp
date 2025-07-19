#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"


#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

enum remasking_type {
    REMASKING_LOW_CONFIDENCE = 0,
    REMASKING_RANDOM         = 1,
};

struct diffusion_params_llada {
    int32_t                   steps;
    int32_t                   max_length;
    int32_t                   block_length;
    float                     temperature;
    float                     cfg_scale;
    llama_token               mask_token_id;
    enum remasking_type       remasking;
    bool (*step_callback)(int32_t step, int32_t total_steps, const llama_token * tokens, int32_t n_tokens, void * user_data);
    void *                    step_callback_user_data;
    int32_t                   seed;
};

static diffusion_params_llada diffusion_default_params_llada() {
    diffusion_params_llada params  = {};
    params.steps                   = 128;
    params.max_length              = 256;
    params.block_length            = 32;
    params.temperature             = 0.0f;
    params.cfg_scale               = 0.2f;
    params.mask_token_id           = LLAMA_TOKEN_NULL;
    params.remasking               = REMASKING_LOW_CONFIDENCE;
    params.step_callback           = nullptr;
    params.step_callback_user_data = nullptr;
    params.seed                    = 0;
    return params;
}

static void add_gumbel_noise(float * logits, int32_t n_vocab, float temperature, std::mt19937 & rng) {
    if (temperature == 0.0f) {
        return;
    }

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (int32_t i = 0; i < n_vocab; i++) {
        double noise        = uniform(rng);
        // Prevent log(0)
        noise               = std::max(noise, 1e-20);
        double gumbel_noise = std::pow(-std::log(noise), temperature);
        logits[i]           = std::exp(logits[i]) / gumbel_noise;
    }
}

static std::vector<int32_t> get_num_transfer_tokens(int32_t mask_count, int32_t steps) {
    std::vector<int32_t> num_transfer_tokens(steps);

    int32_t base      = mask_count / steps;
    int32_t remainder = mask_count % steps;

    for (int32_t i = 0; i < steps; i++) {
        num_transfer_tokens[i] = base + (i < remainder ? 1 : 0);
    }

    return num_transfer_tokens;
}

static void diffusion_generate_llada(llama_context * ctx,
                                     const llama_token * input_tokens,
                                     llama_token * output_tokens,
                                     int32_t n_input,
                                     struct diffusion_params_llada params,
                                     int32_t & n_generated) {
    n_generated = 0;
    if (!ctx || !input_tokens || !output_tokens || n_input <= 0) {
        return;
    }

    const llama_model * model = llama_get_model(ctx);

    std::vector<llama_token> in(params.max_length, params.mask_token_id);
    std::copy(input_tokens, input_tokens + n_input, in.begin());

    GGML_ASSERT(params.max_length % params.block_length == 0);
    int num_blocks = params.max_length / params.block_length;

    GGML_ASSERT(params.steps % num_blocks == 0);

    int steps = params.steps / num_blocks;

    int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_set_causal_attn(ctx, false);

    // Pre-allocate buffers for Classifier-Free Guidance
    int32_t                  logits_size = n_vocab * params.max_length;
    std::vector<float>       cond_logits_buffer;
    std::vector<llama_token> un_x_buffer;
    if (params.cfg_scale > 0.0f) {
        cond_logits_buffer.resize(logits_size);
        un_x_buffer.resize(params.max_length);
    }

    llama_batch batch = llama_batch_init(params.max_length, 0, 1);
    batch.n_tokens    = params.max_length;

    std::vector<llama_token> argmax;
    std::mt19937             rng(params.seed);

    int64_t total_sampling_time = 0;
    int64_t total_time          = 0;

    std::vector<float> confidence(params.max_length);

    int64_t time_start = ggml_time_us();
    for (int block_num = 0; block_num < num_blocks; block_num++) {
        // Get number of tokens to transfer for this step
        int32_t block_start = n_input + block_num * params.block_length;
        int32_t block_end   = std::min(n_input + (block_num + 1) * params.block_length, params.max_length);

        // Count masked tokens in current block
        int32_t block_mask_count = 0;
        for (int i = block_start; i < block_end; i++) {
            if (in[i] == params.mask_token_id) {
                block_mask_count++;
            }
        }
        auto num_transfer_tokens = get_num_transfer_tokens(block_mask_count, steps);

        for (int step = 0; step < steps; step++) {
            if (params.step_callback) {
                if (!params.step_callback(step + block_num * steps,
                                          params.steps, in.data(),
                                          params.max_length,
                                          params.step_callback_user_data)) {
                    break;
                }
            }

            float * logits = nullptr;

            if (params.cfg_scale > 0.0f) {
                for (int32_t i = 0; i < batch.n_tokens; i++) {
                    batch.token[i]     = in[i];
                    batch.pos[i]       = i;
                    batch.n_seq_id[i]  = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i]    = 1;
                }

                int ret = llama_decode(ctx, batch);
                if (ret != 0) {
                    LOG_ERR("Failed to generate conditional");
                }
                float * cond_logits_ptr = llama_get_logits(ctx);
                std::memcpy(cond_logits_buffer.data(), cond_logits_ptr, logits_size * sizeof(float));

                std::copy(in.begin(), in.end(), un_x_buffer.begin());
                for (int32_t i = 0; i < n_input; i++) {
                    un_x_buffer[i] = params.mask_token_id;
                }

                for (int32_t i = 0; i < batch.n_tokens; i++) {
                    batch.token[i]     = un_x_buffer[i];
                    batch.pos[i]       = i;
                    batch.n_seq_id[i]  = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i]    = 1;
                }
                ret = llama_decode(ctx, batch);
                GGML_ASSERT(ret == 0);
                float * uncond_logits = llama_get_logits(ctx);
                for (int32_t i = 0; i < logits_size; i++) {
                    cond_logits_buffer[i] =
                        uncond_logits[i] + (params.cfg_scale + 1.0f) * (cond_logits_buffer[i] - uncond_logits[i]);
                }

                logits = cond_logits_buffer.data();
            } else {
                // Standard generation without CFG
                for (int32_t i = 0; i < batch.n_tokens; i++) {
                    batch.token[i]     = in[i];
                    batch.pos[i]       = i;
                    batch.n_seq_id[i]  = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i]    = 1;
                }

                int ret = llama_decode(ctx, batch);
                if (ret != 0) {
                    LOG_ERR("Failed to generate");
                }
                logits = llama_get_logits(ctx);
            }

            int64_t time_start_sampling = ggml_time_us();

            if (params.temperature > 0.0f) {
                add_gumbel_noise(logits, n_vocab, params.temperature, rng);
            }

            argmax.clear();

            for (int i = 0; i < params.max_length; ++i) {
                float       max_value = std::numeric_limits<float>::min();
                llama_token tok       = LLAMA_TOKEN_NULL;
                for (int vob = 0; vob < n_vocab; vob++) {
                    if (logits[n_vocab * i + vob] > max_value) {
                        max_value = logits[n_vocab * i + vob];
                        tok       = vob;
                    }
                }
                argmax.push_back(tok);
            }

            // Create mask index to track which positions are masked
            std::vector<bool> mask_index(params.max_length);
            for (int i = 0; i < params.max_length; i++) {
                mask_index[i] = (in[i] == params.mask_token_id);
            }

            if (params.remasking == REMASKING_LOW_CONFIDENCE) {
                // inplace softmax + argmax calculation. TODO: check why llama_sampler is so slow here
                for (int i = block_start; i < block_end; i++) {
                    if (mask_index[i]) {
                        float * pos_logits = logits + i * n_vocab;

                        llama_token best_token = 0;
                        float       max_logit  = pos_logits[0];
                        for (int32_t j = 1; j < n_vocab; j++) {
                            if (pos_logits[j] > max_logit) {
                                max_logit  = pos_logits[j];
                                best_token = j;
                            }
                        }

                        float sum_exp = 0.0f;
                        for (int32_t j = 0; j < n_vocab; j++) {
                            sum_exp += std::exp(pos_logits[j] - max_logit);
                        }

                        float prob    = std::exp(pos_logits[best_token] - max_logit) / sum_exp;
                        confidence[i] = prob;

                        argmax[i] = best_token;
                    } else {
                        confidence[i] = -std::numeric_limits<float>::infinity();  // Non-masked positions
                    }
                }
            } else if (params.remasking == REMASKING_RANDOM) {
                // Random remasking: assign random values for masked positions
                std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
                for (int i = 0; i < params.max_length; i++) {
                    if (mask_index[i]) {
                        confidence[i] = uniform(rng);
                    } else {
                        confidence[i] = -std::numeric_limits<float>::infinity();  // Non-masked positions
                    }
                }
            }

            for (int i = n_input + (block_num + 1) * params.block_length; i < params.max_length; i++) {
                confidence[i] = -std::numeric_limits<float>::infinity();
            }

            int32_t transfer_count = num_transfer_tokens[step];

            std::vector<std::pair<float, int32_t>> conf_pairs;
            for (int i = n_input; i < params.max_length; i++) {
                if (mask_index[i] && confidence[i] > -std::numeric_limits<float>::infinity()) {
                    conf_pairs.push_back({ confidence[i], i });
                }
            }

            std::partial_sort(
                conf_pairs.begin(), conf_pairs.begin() + std::min(transfer_count, (int32_t) conf_pairs.size()),
                conf_pairs.end(), [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
                    return a.first > b.first;
                });

            for (int i = 0; i < std::min(transfer_count, (int32_t) conf_pairs.size()); i++) {
                int32_t pos = conf_pairs[i].second;
                in[pos]      = argmax[pos];
            }

            int64_t time_end_sampling = ggml_time_us();
            total_sampling_time += time_end_sampling - time_start_sampling;
        }
    }
    int64_t time_end = ggml_time_us();
    total_time += time_end - time_start;

    LOG_INF("\ntotal time: %0.2fms, time per step: %0.2fms, sampling time per step: %0.2fms\n", total_time / 1000.0,
            total_time / 1000.0 / params.steps, total_sampling_time / 1000.0 / params.steps);

    llama_batch_free(batch);

    memcpy(output_tokens, in.data(), in.size() * sizeof(llama_token));

    n_generated = params.max_length;
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
    const common_params_diffusion_llada * diff_params;
    const llama_vocab *                   vocab;
    int32_t                               n_input;
};

static bool diffusion_step_callback(int32_t step,
                                    int32_t total_steps,
                                    const llama_token * tokens,
                                    int32_t n_tokens,
                                    void * user_data) {
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

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DIFFUSION_LLADA)) {
        return 1;
    }

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

    char arch_str[128];
    GGML_ASSERT(llama_model_meta_val_str(model, "general.architecture", arch_str, 128) >= 0 &&
                std::string(arch_str) == "llada");

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

    // For LLaDA models, forcefully add BOS token at the beginning. TODO: check why this is needed vs HF
    llama_token bos_token = llama_vocab_bos(vocab);
    if (bos_token != LLAMA_TOKEN_NULL && (input_tokens.empty() || input_tokens[0] != bos_token)) {
        input_tokens.insert(input_tokens.begin(), bos_token);
    }

    int n_input = input_tokens.size();

    if (n_input >= params.n_ctx) {
        LOG_ERR("error: input too long (%d tokens), max context is %d\n", n_input, params.n_ctx);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_token mask_token_id = llama_vocab_mask(vocab);
    GGML_ASSERT(mask_token_id != LLAMA_TOKEN_NULL);

    diffusion_params_llada llada_params = diffusion_default_params_llada();
    llada_params.steps                  = params.diffusion_llada.steps;
    llada_params.block_length           = params.diffusion_llada.block_length;
    llada_params.temperature            = params.sampling.temp;
    llada_params.cfg_scale              = params.diffusion_llada.cfg_scale;
    llada_params.remasking              = static_cast<enum remasking_type>(params.diffusion_llada.remasking);
    llada_params.mask_token_id          = mask_token_id;
    llada_params.seed                   = params.sampling.seed;
    llada_params.max_length             = params.n_ubatch;

    callback_data cb_data = { &params.diffusion_llada, vocab, n_input };
    llada_params.step_callback           = diffusion_step_callback;
    llada_params.step_callback_user_data = &cb_data;

    LOG_INF("Using LLaDA diffusion generation\n");
    LOG_INF("llada_diffusion_params: - %-25s llama_token      = %d\n", "mask_token_id", mask_token_id);
    LOG_INF("llada_diffusion_params: - %-25s u32              = %d\n", "steps", llada_params.steps);
    LOG_INF("llada_diffusion_params: - %-25s u32              = %d\n", "max_length", llada_params.max_length);
    LOG_INF("llada_diffusion_params: - %-25s u32              = %d\n", "block_length", llada_params.block_length);
    LOG_INF("llada_diffusion_params: - %-25s f32              = %.3f\n", "temperature", llada_params.temperature);
    LOG_INF("llada_diffusion_params: - %-25s f32              = %.3f\n", "cfg_scale", llada_params.cfg_scale);

    int32_t n_generated = 0;
    std::vector<llama_token> output_tokens(params.n_ubatch);

    diffusion_generate_llada(ctx, input_tokens.data(), output_tokens.data(), n_input, llada_params, n_generated);

    if (n_generated > 0) {
        if (params.diffusion_llada.visual_mode) {
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
