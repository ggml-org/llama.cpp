#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <regex>

static void print_usage(int, char ** argv) {
    const std::string usage_template = R"(
        example usage:

          Print tensors:

          {prog} -m model.gguf -p "Hello my name is" --verbose

          The tensors to be printed can be filtered with --tensor-filter option.

          Save logits/embeddings:

          {prog} -m model.gguf -p "Hello my name is" --save-logits

          Add --embedding to save embeddings)" "\n";

    // Fix the source code indentation above that is introduced by the raw string literal.
    std::string usage = std::regex_replace(usage_template, std::regex("\\n {8}"), "\n");
    usage = std::regex_replace(usage, std::regex("\\{prog\\}"), argv[0]);
    LOG("%s\n", usage.c_str());
}

static bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data);

struct callback_data {
    std::vector<uint8_t>    data;
    std::vector<std::regex> tensor_filters;

    callback_data() = default;

    callback_data(common_params & params, const std::vector<std::string> & filter_patterns) {
        for (const auto & pattern : filter_patterns) {
            try {
                std::string anchored_pattern = "^" + pattern;
                tensor_filters.emplace_back(anchored_pattern, std::regex::optimize);
            } catch (const std::regex_error & e) {
                throw std::runtime_error("Invalid regex pattern '" + pattern + "': " + e.what());
            }
        }
        params.cb_eval           = ggml_debug;
        params.cb_eval_user_data = this;
    }
};

struct output_data {
    float *                  data_ptr    = nullptr;
    int                      data_size   = 0;
    std::string              type_suffix;
    std::vector<float>       storage;
    std::string              prompt;
    std::vector<llama_token> tokens;

    output_data(llama_context * ctx, const llama_model * model, const common_params & params) {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const bool add_bos = llama_vocab_get_add_bos(vocab);

        tokens = common_tokenize(ctx, params.prompt, add_bos);
        prompt = params.prompt;

        if (params.embedding) {
            const int  n_embd          = llama_model_n_embd_out(model);
            const bool pooling_enabled = llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_NONE;
            const int  n_embd_count    = pooling_enabled ? 1 : tokens.size();
            const int  n_embeddings    = n_embd * n_embd_count;

            float * embeddings;
            if (pooling_enabled) {
                embeddings = llama_get_embeddings_seq(ctx, 0);
                storage.resize(n_embeddings);
                common_embd_normalize(embeddings, storage.data(), n_embeddings, params.embd_normalize);
                embeddings = storage.data();
            } else {
                embeddings = llama_get_embeddings(ctx);
            }

            data_ptr = embeddings;
            data_size = n_embeddings;
            type_suffix = "-embeddings";
        } else {
            const float * logits = llama_get_logits_ith(ctx, tokens.size() - 1);
            const int n_logits = llama_vocab_n_tokens(vocab);

            data_ptr = const_cast<float*>(logits);
            data_size = n_logits;
            type_suffix = "";
        }
    }
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

static inline float ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    union {
        float f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.bits << 16;
    return u.f;
}

static float ggml_get_float_value(const uint8_t * data, ggml_type type,
        const size_t * nb, size_t i0, size_t i1, size_t i2, size_t i3) {
    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
    switch (type) {
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*(const ggml_fp16_t *) &data[i]);
        case GGML_TYPE_F32:
            return *(const float *) &data[i];
        case GGML_TYPE_I64:
            return (float) *(const int64_t *) &data[i];
        case GGML_TYPE_I32:
            return (float) *(const int32_t *) &data[i];
        case GGML_TYPE_I16:
            return (float) *(const int16_t *) &data[i];
        case GGML_TYPE_I8:
            return (float) *(const int8_t *) &data[i];
        case GGML_TYPE_BF16:
            return ggml_compute_bf16_to_fp32(*(const ggml_bf16_t *) &data[i]);
        default:
            GGML_ABORT("fatal error");
    }
}

static void ggml_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
    GGML_ASSERT(n > 0);
    float sum    = 0;
    float sum_sq = 0.0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    const float v = ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    sum    += v;
                    sum_sq += v * v;
                }
            }
        }
    }
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        LOG_DBG("                                     [\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2*n) {
                LOG_DBG("                                      ..., \n");
                i2 = ne[2] - n;
            }
            LOG_DBG("                                      [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2*n) {
                    LOG_DBG("                                       ..., \n");
                    i1 = ne[1] - n;
                }
                LOG_DBG("                                       [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2*n) {
                        LOG_DBG("..., ");
                        i0 = ne[0] - n;
                    }
                    const float v = ggml_get_float_value(data, type, nb, i0, i1, i2, i3);
                    LOG_DBG("%12.4f", v);
                    if (i0 < ne[0] - 1) {
                        LOG_DBG(", ");
                    }
                }
                LOG_DBG("],\n");
            }
            LOG_DBG("                                      ],\n");
        }
        LOG_DBG("                                     ]\n");
        LOG_DBG("                                     sum    = %f\n", sum);
        LOG_DBG("                                     sum_sq = %f\n", sum_sq);
    }

    if (std::isnan(sum)) {
        LOG_ERR("encountered NaN - aborting\n");
        exit(0);
    }
}

/**
 * GGML operations callback during the graph execution.
 *
 * @param t current tensor
 * @param ask when ask is true, the scheduler wants to know if we are interested in data from this tensor
 *            if we return true, a follow-up call will be made with ask=false in which we can do the actual collection.
 *            see ggml_backend_sched_eval_callback
 * @param user_data user data to pass at each call back
 * @return true to receive data or continue the graph, false otherwise
 */
static bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb_data = (callback_data *) user_data;

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    if (ask) {
        return true; // Always retrieve data
    }

    bool matches_filter = cb_data->tensor_filters.empty();

    if (!matches_filter) {
        for (const auto & filter : cb_data->tensor_filters) {
            if (std::regex_search(t->name, filter)) {
                matches_filter = true;
                break;
            }
        }
    }

    char src1_str[128] = {0};
    if (src1) {
        snprintf(src1_str, sizeof(src1_str), "%s{%s}", src1->name, ggml_ne_string(src1).c_str());
    }

    if (matches_filter) {
        LOG_DBG("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__,
             t->name,
             ggml_type_name(t->type),
             ggml_op_desc(t),
             src0->name,
             ggml_ne_string(src0).c_str(),
             src1 ? src1_str : "",
             ggml_ne_string(t).c_str());
    }

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    if (!ggml_is_quantized(t->type) && matches_filter) {
        uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();
        ggml_print_tensor(data, t->type, t->ne, t->nb, 3);
    }

    return true;
}


static void save_output_data(const output_data & output, const std::string & model_name, const std::string & output_dir) {
    std::filesystem::create_directory(output_dir);
    auto base_path = std::filesystem::path{output_dir} / ("llamacpp-" + model_name + output.type_suffix);

    // Save logits/embeddings to binary file.
    {
        std::filesystem::path filepath{base_path.string() + ".bin"};
        std::ofstream file{filepath, std::ios::binary};
        if (!file) {
            throw std::runtime_error("failed to open binary output file: " + filepath.string());
        }
        file.write(reinterpret_cast<const char*>(output.data_ptr), output.data_size * sizeof(float));
        LOG("Data saved to %s\n", filepath.c_str());
    }

    // Save logits/embeddings to text file.
    {
        std::filesystem::path filepath{base_path.string() + ".txt"};
        std::ofstream file{filepath};
        if (!file) {
            throw std::runtime_error("failed to open text output file: " + filepath.string());
        }
        for (int i = 0; i < output.data_size; i++) {
            file << i << ": " << output.data_ptr[i] << '\n';
        }
        LOG("Data saved to %s\n", filepath.c_str());
    }

    // Save prompt and tokens to text file.
    {
        std::filesystem::path filepath{base_path.string() + "-prompt.txt"};
        std::ofstream file{filepath};
        if (!file) {
            throw std::runtime_error("failed to open prompt output file: " + filepath.string());
        }

        file << "prompt: " << output.prompt << '\n';
        file << "n_tokens: " << output.tokens.size() << '\n';

        file << "token ids: ";
        for (size_t i = 0; i < output.tokens.size(); i++) {
            file << output.tokens[i];
            if (i + 1 < output.tokens.size()) {
                file << ", ";
            }
        }
        file << '\n';
        LOG("Prompt saved to %s\n", filepath.c_str());
    }

    // Save token ids to binary file.
    {
        std::filesystem::path filepath{base_path.string() + "-tokens.bin"};
        std::ofstream file{filepath, std::ios::binary};
        if (!file) {
            throw std::runtime_error("failed to open tokens binary file: " + filepath.string());
        }
        file.write(reinterpret_cast<const char*>(output.tokens.data()), output.tokens.size() * sizeof(llama_token));
        LOG("Tokens saved to %s\n", filepath.c_str());
    }

}

static void print_tokenized_prompt(llama_context * ctx, const std::vector<llama_token> & tokens, const std::string & prompt) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    LOG("Model add_bos: %s\n", llama_vocab_get_add_bos(vocab) ? "true" : "false");
    LOG("Input prompt: \"%s\"\n", prompt.c_str());
    LOG("Token ids (%zu):\n", tokens.size());

    for (auto id : tokens) {
        std::string piece(128, '\0');
        int n = llama_token_to_piece(vocab, id, piece.data(), piece.size(), 0, true);
        if (n < 0) {
            LOG_ERR("failed to convert token %d to piece\n", id);
            continue;
        }
        piece.resize(n);
        LOG("%s(%d) ", piece.c_str(), id);
    }
    LOG("\n");
}

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos);

    if (tokens.empty()) {
        LOG_ERR("%s : there are not input tokens to process - (try to provide a prompt with '-p')\n", __func__);
        return false;
    }

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return false;
    }

    print_tokenized_prompt(ctx, tokens, params.prompt);

    if (params.save_logits) {
        output_data output {ctx, model, params};
        std::filesystem::path model_path{params.model.path};
        std::string model_name{model_path.stem().string()};
        save_output_data(output, model_name, params.logits_output_dir);
    }

    return true;
}

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DEBUG, print_usage)) {
        return 1;
    }

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    callback_data cb_data(params, params.tensor_filter);

    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    if (!run(ctx, params)) {
        return 1;
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
