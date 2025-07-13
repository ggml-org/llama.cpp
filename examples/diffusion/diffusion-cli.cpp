#include <limits.h>

#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "diffusion.h"

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
    llama_token                     mask_token_id;  // Store mask token separately since it's not in diffusion params
};

static bool diffusion_step_callback(int32_t step, int32_t total_steps, const llama_token * tokens, int32_t n_tokens,
                                    void * user_data) {
    callback_data * data = static_cast<callback_data *>(user_data);

    if (data->diff_params->visual_mode) {
        // Visual mode: clear
        std::cerr << "\033[2J\033[H";  // Clear screen and move cursor to top-left

        int progress_percent = (step * 100) / total_steps;
        int progress_bars    = (step * 50) / total_steps;
        std::cerr << "Diffusion Step " << step << "/" << total_steps << " [" << std::string(progress_bars, '=')
                  << std::string(50 - progress_bars, ' ') << "] " << progress_percent << "%\n";

        std::string current_text = " ";

        for (int32_t i = data->n_input; i < n_tokens; i++) {
            std::string token_str;
            if (tokens[i] != data->mask_token_id) {
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

        std::cerr << current_text << "\n";
        std::cerr << std::flush;
    } else {
        int progress_percent = (step * 100) / total_steps;
        int progress_bars    = (step * 50) / total_steps;

        std::cerr << "\rDiffusion Step " << step << "/" << total_steps << " [" << std::string(progress_bars, '=')
                  << std::string(50 - progress_bars, ' ') << "] " << progress_percent << "%" << std::flush;
    }

    return true;  // Continue generation
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
                                                            true,  // add_special tokens
                                                            true   // parse_special
    );
    int                      n_input      = input_tokens.size();

    if (n_input >= params.n_ctx) {
        LOG_ERR("error: input too long (%d tokens), max context is %d\n", n_input, params.n_ctx);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    struct diffusion_params ldiff_params = diffusion_default_params();
    ldiff_params.steps                  = params.diffusion.steps;
    ldiff_params.eps                    = params.diffusion.eps;
    ldiff_params.temperature            = params.sampling.temp;
    ldiff_params.top_p                  = params.sampling.top_p;
    ldiff_params.top_k                  = params.sampling.top_k;
    ldiff_params.algorithm              = static_cast<enum diffusion_algorithm>(params.diffusion.algorithm);
    ldiff_params.alg_temp               = params.diffusion.alg_temp;
    ldiff_params.seed                   = params.sampling.seed;

    llama_token mask_token_id = llama_vocab_mask(vocab);
    GGML_ASSERT(mask_token_id != LLAMA_TOKEN_NULL);

    LOG_INF("diffusion_params: - %-25s llama_token      = %d\n", "mask_token_id", mask_token_id);
    LOG_INF("diffusion_params: - %-25s u32              = %d\n", "steps", params.diffusion.steps);
    LOG_INF("diffusion_params: - %-25s f32              = %.6f\n", "eps", params.diffusion.eps);
    LOG_INF("diffusion_params: - %-25s u32              = %d (%s)\n", "algorithm", params.diffusion.algorithm,
            alg_name);
    LOG_INF("diffusion_params: - %-25s f32              = %.3f\n", "alg_temp", params.diffusion.alg_temp);

    ldiff_params.mask_token_id = mask_token_id;

    callback_data cb_data = { &params.diffusion, vocab, n_input, mask_token_id };

    ldiff_params.step_callback           = diffusion_step_callback;
    ldiff_params.step_callback_user_data = &cb_data;

    int32_t       n_generated = 0;
    llama_token * generated   = diffusion_generate(ctx, input_tokens.data(), n_input, params.diffusion.max_length,
                                                   ldiff_params, &n_generated);

    if (params.diffusion.visual_mode) {
        std::cerr << "\033[2J\033[H";  // Clear screen and move cursor to top-left
    } else {
        std::cerr << "\r" << std::string(80, ' ') << "\r" << std::flush;
    }

    if (generated && n_generated > 0) {
        std::vector<llama_token> output_tokens(generated + n_input, generated + n_generated);

        std::string output_data = common_detokenize(vocab, output_tokens, false);
        std::cout << output_data << std::endl;

        delete[] generated;
    } else {
        std::cerr << "Error: diffusion generation failed" << std::endl;
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
