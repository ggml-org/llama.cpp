#include "arg.h"
#include "chat.h"
#include "common.h"
#include "diffusion.h"
#include "llama.h"
#include "log.h"

#include <limits.h>

#include <clocale>
#include <cstring>
#include <string>
#include <vector>

struct callback_data {
    diffusion_params *  diff_params;
    const llama_vocab * vocab;
    int32_t             n_input;
};

static bool diffusion_step_callback(int32_t             step,
                                    int32_t             total_steps,
                                    const llama_token * tokens,
                                    int32_t             n_tokens,
                                    void *              user_data) {
    (void) user_data;

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

static std::string format_input_text(const std::string & prompt, const std::string & system_prompt, bool use_chat_template, llama_model * model) {
    if (!use_chat_template) {
        return prompt;
    }

    auto chat_templates = common_chat_templates_init(model, "");
    common_chat_templates_inputs inputs;
    common_chat_msg system_msg;

    if (!system_prompt.empty()) {
        system_msg.role = "system";
        system_msg.content = system_prompt;
        inputs.messages.push_back(system_msg);
    }

    common_chat_msg user_msg;
    user_msg.role = "user";
    user_msg.content = prompt;

    inputs.messages.push_back(user_msg);
    inputs.add_generation_prompt = true;

    auto result = common_chat_templates_apply(chat_templates.get(), inputs);

    return result.prompt;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    ggml_time_init();

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DIFFUSION)) {
        return 1;
    }

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = params.n_gpu_layers;
    model_params.devices            = params.devices.data();
    model_params.use_mmap           = params.use_mmap;
    model_params.use_direct_io      = params.use_direct_io;
    model_params.use_mlock          = params.use_mlock;
    model_params.check_tensors      = params.check_tensors;

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (!model) {
        LOG_ERR("error: failed to load model '%s'\n", params.model.path.c_str());
        return 1;
    }

    if (!llama_model_is_diffusion(model)) {
        LOG_ERR("error: unsupported model for diffusion");
        llama_model_free(model);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx                = params.n_ctx;
    ctx_params.n_batch              = params.n_batch;
    ctx_params.n_ubatch             = params.n_ubatch;
    ctx_params.flash_attn_type      = params.flash_attn_type;
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

    std::string         formatted_prompt = format_input_text(params.prompt, params.system_prompt, params.enable_chat_template, model);

    std::vector<llama_token> input_tokens = common_tokenize(vocab,
                                                            formatted_prompt,
                                                            /*add special tokens*/ true,
                                                            /*parse special*/ true);

    int n_input = input_tokens.size();

    if (static_cast<uint32_t>(n_input) >= llama_n_ctx(ctx)) {
        LOG_ERR("error: input too long (%d tokens), max context is %d\n", n_input, llama_n_ctx(ctx));
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_token mask_token_id = llama_vocab_mask(vocab);

    GGML_ASSERT(mask_token_id != LLAMA_TOKEN_NULL);

    bool visual_mode = params.diffusion.visual_mode;

    int32_t                  n_generated = 0;
    std::vector<llama_token> output_tokens(params.n_ubatch);

    struct diffusion_params diff_params;

    char shift_logits_str[8];
    if (llama_model_meta_val_str(model, "diffusion.shift_logits", shift_logits_str, sizeof(shift_logits_str)) >= 0) {
        diff_params.shift_logits = (strcmp(shift_logits_str, "true") == 0);
    } else {
        diff_params.shift_logits = true;
    }

    //Use either eps or block length, but not both
    GGML_ASSERT((params.diffusion.eps == 0) ^ (params.diffusion.block_length == 0));

    if (params.diffusion.eps) {
        diff_params.schedule = DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED;
        diff_params.eps      = params.diffusion.eps;
    } else if (params.diffusion.block_length) {
        diff_params.schedule     = DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED;
        diff_params.block_length = params.diffusion.block_length;
    }

    diff_params.mask_token_id    = mask_token_id;
    diff_params.seed             = params.sampling.seed;
    diff_params.temperature      = params.sampling.temp;
    diff_params.steps            = params.diffusion.steps;
    diff_params.algorithm        = static_cast<diffusion_algorithm>(params.diffusion.algorithm);
    diff_params.max_length       = params.n_ubatch;
    diff_params.top_p            = params.sampling.top_p;
    diff_params.top_k            = params.sampling.top_k;
    diff_params.visual_mode      = params.diffusion.visual_mode;
    diff_params.add_gumbel_noise = params.diffusion.add_gumbel_noise;

    diff_params.step_callback           = diffusion_step_callback;
    callback_data cb_data               = { &diff_params, vocab, n_input };
    diff_params.step_callback_user_data = &cb_data;

    const char * alg_names[]   = {
        "DIFFUSION_ALGORITHM_ORIGIN",
        "DIFFUSION_ALGORITHM_ENTROPY_BASED",
        "DIFFUSION_ALGORITHM_MARGIN_BASED",
        "DIFFUSION_ALGORITHM_RANDOM",
        "DIFFUSION_ALGORITHM_CONFIDENCE_BASED",
    };
    const char * sched_names[] = {
        "DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED",
        "DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED",
    };
    const char * alg_name =
        (diff_params.algorithm >= 0 && diff_params.algorithm <= 4) ? alg_names[diff_params.algorithm] : "UNKNOWN";
    const char * sched_name =
        (diff_params.schedule >= 0 && diff_params.schedule <= 1) ? sched_names[diff_params.schedule] : "UNKNOWN";

    LOG_INF("diffusion_params: - %-25s llama_token      = %d\n", "mask_token_id", mask_token_id);
    LOG_INF("diffusion_params: - %-25s u32              = %d\n", "steps", diff_params.steps);
    LOG_INF("diffusion_params: - %-25s u32              = %d\n", "max_length", diff_params.max_length);
    LOG_INF("diffusion_params: - %-25s enum             = %d (%s)\n", "algorithm", diff_params.algorithm, alg_name);
    LOG_INF("diffusion_params: - %-25s enum             = %d (%s)\n", "schedule", diff_params.schedule, sched_name);
    LOG_INF("diffusion_params: - %-25s f32              = %.3f\n", "temperature", diff_params.temperature);
    if (diff_params.schedule == DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED) {
        LOG_INF("diffusion_params: - %-25s f32              = %.6f\n", "eps", diff_params.eps);
        LOG_INF("diffusion_params: - %-25s f32              = %.3f\n", "alg_temp", diff_params.alg_temp);
    }
    if (diff_params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) {
        LOG_INF("diffusion_params: - %-25s u32              = %d\n", "block_length", diff_params.block_length);
        LOG_INF("diffusion_params: - %-25s f32              = %.3f\n", "cfg_scale", diff_params.cfg_scale);
    }

    diffusion_generate(ctx, input_tokens.data(), output_tokens.data(), n_input, diff_params, n_generated);

    if (n_generated > 0) {
        if (visual_mode) {
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
