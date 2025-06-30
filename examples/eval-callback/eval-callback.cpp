#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"
#include "sampling.h"

#include <cstdio>
#include <string>
#include <vector>
#include <iostream>

struct callback_data {
    std::vector<uint8_t> data;
    std::string parse_layer_name;
    int current_token_index = -1;
};

static std::string ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

static void ggml_print_tensor_block(const std::string& tensor_name, uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t token_idx) {
    const int64_t dim = ne[0];
    std::cout << "=== TOKEN " << token_idx << " ===\n";
    std::cout << "--- TENSOR: " << tensor_name << " ---\n";
    std::cout << "SHAPE: [" << dim << "]\n";
    std::cout << "DATA:\n";

    for (int64_t i = 0; i < dim; ++i) {
        size_t offset = i * nb[0];
        float v;

        switch (type) {
            case GGML_TYPE_F16: v = ggml_fp16_to_fp32(*(ggml_fp16_t *) &data[offset]); break;
            case GGML_TYPE_F32: v = *(float *) &data[offset]; break;
            default: GGML_ABORT("Unsupported tensor type");
        }

        std::cout << v;
        if (i < dim - 1) std::cout << ", ";
    }

    std::cout << "\n\n";
}

static bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (callback_data *) user_data;

    if (ask) {
        return std::string(t->name) == cb_data->parse_layer_name;
    }

    if (std::string(t->name) != cb_data->parse_layer_name) {
        return false;
    }

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    if (!ggml_is_quantized(t->type)) {
        uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();
        ggml_print_tensor_block(t->name, data, t->type, t->ne, t->nb, cb_data->current_token_index);
    }

    return true;
}

static bool run(llama_context * ctx, const common_params & params, callback_data & cb_data) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    llama_sampler * sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    cb_data.current_token_index = -1;
    if (llama_decode(ctx, batch)) {
        LOG_ERR("Failed to evaluate prompt\n");
        llama_sampler_free(sampler);
        return false;
    }

    std::string result;
    llama_token token;

    for (int i = 0; i < params.n_predict; ++i) {
        token = llama_sampler_sample(sampler, ctx, -1);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        char buf[128];
        int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        if (n < 0) {
            LOG_ERR("Failed to convert token to string\n");
            llama_sampler_free(sampler);
            return false;
        }
        result += std::string(buf, n);  // <-- store instead of printing

        llama_batch new_batch = llama_batch_get_one(&token, 1);
        cb_data.current_token_index = i;
        if (llama_decode(ctx, new_batch)) {
            LOG_ERR("Failed to decode sampled token\n");
            llama_sampler_free(sampler);
            return false;
        }
    }

    llama_sampler_free(sampler);

    // Output final result
    std::cout << "\n\nFull output:\n" << result << "\n";

    return true;
}


int main(int argc, char **argv) {
    callback_data cb_data;
    common_params params;
    std::string parse_layer_value;
    std::vector<char*> filtered_argv;
    std::vector<std::string> prompts;

    filtered_argv.push_back(argv[0]);

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.compare(0, 2, "--") == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }

        if (arg == "--parse-layer") {
            if (i + 1 < argc) {
                parse_layer_value = argv[++i];
            } else {
                fprintf(stderr, "error: --parse-layer requires an argument\n");
                return 1;
            }
            continue;
        } else if (arg == "--prompt") {
            if (i + 1 < argc) {
                prompts.emplace_back(argv[++i]);
            } else {
                fprintf(stderr, "error: --prompt requires an argument\n");
                return 1;
            }
            continue;
        }

        filtered_argv.push_back(argv[i]);
    }

    if (!common_params_parse((int)filtered_argv.size(), filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (!parse_layer_value.empty()) {
        LOG_INF("Parse layer argument value: %s\n", parse_layer_value.c_str());
    }
    cb_data.parse_layer_name = parse_layer_value;

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval = ggml_debug;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    common_init_result llama_init = common_init_from_params(params);
    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("\n");

    if (prompts.empty()) {
        prompts.emplace_back("What is the capital of France?");  // Fallback default
    }


    for (const auto& prompt : prompts) {
        LOG_INF("Running prompt: %s\n", prompt.c_str());
        params.prompt = prompt;
        if (!run(ctx, params, cb_data)) {
            LOG_ERR("Failed on prompt: %s\n", prompt.c_str());
            return 1;
        }
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
