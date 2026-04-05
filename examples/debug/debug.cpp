#include "debug.h"
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <regex>

// ---------------------------------------------------------------------------
// Tensor-saving callback
// Saves each tensor matching the filter to <save_tensors_dir>/<name>.bin
// (flat float32, row-major matching PyTorch layout) and a .shape sidecar.
// ---------------------------------------------------------------------------

static float tensor_elem_to_f32(const uint8_t * data, ggml_type type,
                                  const size_t * nb,
                                  int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
    size_t offset = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
    switch (type) {
        case GGML_TYPE_F32:  return *(const float *)    (data + offset);
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(*(const ggml_fp16_t *)(data + offset));
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*(const ggml_bf16_t *)(data + offset));
        default: return 0.0f; // skip quantized
    }
}

struct save_tensor_callback_data : base_callback_data {
    std::string save_dir;

    save_tensor_callback_data() = default;

    save_tensor_callback_data(common_params & params,
                               const std::vector<std::string> & filter_patterns,
                               std::string dir)
        : base_callback_data(params, filter_patterns), save_dir(std::move(dir)) {
        // Override the callback set by base_callback_data
        params.cb_eval           = save_cb;
        params.cb_eval_user_data = this;
    }

    static bool save_cb(struct ggml_tensor * t, bool ask, void * user_data) {
        auto * cb = static_cast<save_tensor_callback_data *>(user_data);

        if (ask) return true;

        // Check filter
        bool match = cb->tensor_filters.empty();
        for (const auto & f : cb->tensor_filters) {
            if (std::regex_search(t->name, f)) { match = true; break; }
        }
        if (!match || ggml_is_quantized(t->type)) return true;

        // Fetch data (may be on device)
        const bool is_host = ggml_backend_buffer_is_host(t->buffer);
        const size_t n_bytes = ggml_nbytes(t);
        if (!is_host) {
            cb->data.resize(n_bytes);
            ggml_backend_tensor_get(t, cb->data.data(), 0, n_bytes);
        }
        const uint8_t * raw = is_host ? (const uint8_t *) t->data : cb->data.data();

        // Flatten to float32 in GGML dimension order (ne[0] fastest = innermost)
        const int64_t ne0 = t->ne[0], ne1 = t->ne[1], ne2 = t->ne[2], ne3 = t->ne[3];
        const int64_t n_elem = ne0 * ne1 * ne2 * ne3;
        std::vector<float> floats;
        floats.reserve(n_elem);
        for (int64_t i3 = 0; i3 < ne3; i3++)
        for (int64_t i2 = 0; i2 < ne2; i2++)
        for (int64_t i1 = 0; i1 < ne1; i1++)
        for (int64_t i0 = 0; i0 < ne0; i0++)
            floats.push_back(tensor_elem_to_f32(raw, t->type, t->nb, i0, i1, i2, i3));

        // Create output directory
        std::filesystem::create_directories(cb->save_dir);

        // Sanitize tensor name for filename (replace '/' with '_')
        std::string safe_name = t->name;
        for (char & c : safe_name) if (c == '/') c = '_';

        // Write binary
        {
            std::string path = cb->save_dir + "/" + safe_name + ".bin";
            std::ofstream f(path, std::ios::binary);
            f.write((const char *) floats.data(), n_elem * sizeof(float));
        }
        // Write shape sidecar
        {
            std::string path = cb->save_dir + "/" + safe_name + ".shape";
            std::ofstream f(path);
            f << ne0 << " " << ne1 << " " << ne2 << " " << ne3 << "\n";
        }

        LOG("Saved tensor '%s' [%lld %lld %lld %lld] (%lld elems) to %s/\n",
            t->name, (long long)ne0, (long long)ne1, (long long)ne2, (long long)ne3,
            (long long)n_elem, cb->save_dir.c_str());

        return true;
    }
};

static void print_usage(int /*argc*/, char ** argv) {
    const std::string usage_template = R"(
        example usage:

          Print tensors:

          {prog} -m model.gguf -p "Hello my name is" --verbose

          The tensors to be printed can be filtered with --tensor-filter option.

          Save logits/embeddings:

          {prog} -m model.gguf -p "Hello my name is" --save-logits

          Add --embedding to save embeddings

          Save intermediate tensors for comparison:

          {prog} -m model.gguf -p "Hello my name is" --tensor-filter "inp_scaled|attn_norm-0|attn_out-0|result_norm" --save-tensors data/tensors-llm

          Each matching tensor is written as a flat float32 .bin + .shape sidecar.)" "\n";

    // Fix the source code indentation above that is introduced by the raw string literal.
    std::string usage = std::regex_replace(usage_template, std::regex("\\n {8}"), "\n");
    usage = std::regex_replace(usage, std::regex("\\{prog\\}"), argv[0]);
    LOG("%s\n", usage.c_str());
}

static bool has_pooling(llama_context * ctx) {
    switch (llama_pooling_type(ctx)) {
        case LLAMA_POOLING_TYPE_NONE:
        case LLAMA_POOLING_TYPE_UNSPECIFIED:
            return false;
        default:
            return true;
    }
}

struct output_data {
    float *                  data_ptr    = nullptr;
    int                      data_size   = 0;
    std::string              type_suffix;
    std::vector<float>       embd_norm;
    std::string              prompt;
    std::vector<llama_token> tokens;

    output_data(llama_context * ctx, const llama_model * model, const common_params & params) {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const bool add_bos = llama_vocab_get_add_bos(vocab);

        tokens = common_tokenize(ctx, params.prompt, add_bos);
        prompt = params.prompt;

        if (params.embedding) {
            const int n_embd       = llama_model_n_embd_out(model);
            const bool pooling     = has_pooling(ctx);
            const int n_embd_count = pooling ? 1 : tokens.size();
            const int n_floats     = n_embd * n_embd_count;

            float * embd_raw = pooling ? llama_get_embeddings_seq(ctx, 0) : llama_get_embeddings(ctx);
            if (embd_raw == nullptr) {
                throw std::runtime_error("failed to get embeddings from the model");
            }

            LOG_DBG("pooling_enabled: %s\n", pooling ? "true" : "false");
            LOG_DBG("n_embd: %d\n", n_embd);
            LOG_DBG("n_floats: %d\n", n_floats);
            LOG_DBG("n_embd_count: %d\n", n_embd_count);

            data_ptr    = embd_raw;
            data_size   = n_floats;
            type_suffix = "-embeddings";

            if (params.embd_normalize >= 0) {
                embd_norm.resize(n_floats);
                for (int i = 0; i < n_embd_count; i++) {
                    common_embd_normalize(embd_raw+i*n_embd, embd_norm.data()+i*n_embd, n_embd, params.embd_normalize);
                }
                data_ptr = embd_norm.data();
            }
        } else {
            const float * logits = llama_get_logits_ith(ctx, tokens.size() - 1);
            const int n_logits = llama_vocab_n_tokens(vocab);

            data_ptr = const_cast<float*>(logits);
            data_size = n_logits;
            type_suffix = "";
        }
    }
};

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

    // Pre-parse --save-tensors <dir> before handing off to common_params_parse.
    // The option is consumed here and removed from argv so common_params_parse
    // does not see an unknown flag.
    std::string save_tensors_dir;
    {
        std::vector<char *> new_argv;
        new_argv.push_back(argv[0]);
        for (int i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "--save-tensors" && i + 1 < argc) {
                save_tensors_dir = argv[++i];
            } else {
                new_argv.push_back(argv[i]);
            }
        }
        argc = static_cast<int>(new_argv.size());
        for (int i = 0; i < argc; i++) argv[i] = new_argv[i];
    }

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DEBUG, print_usage)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // Use save_tensor_callback_data when --save-tensors is given, otherwise
    // the standard verbose callback.
    // Use optional+emplace to construct in-place so that the constructor's
    // `this` pointer (stored in params.cb_eval_user_data) is stable.
    std::optional<save_tensor_callback_data> save_cb;
    std::optional<base_callback_data>        base_cb;
    if (!save_tensors_dir.empty()) {
        save_cb.emplace(params, params.tensor_filter, save_tensors_dir);
    } else {
        base_cb.emplace(params, params.tensor_filter);
    }

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
