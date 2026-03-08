#include "ggml-cpp.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "common.h"
#include "arg.h"
#include "log.h"

#include <cstdint>
#include <string>
#include <vector>

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        common_batch_add(batch, tokens[pos], pos, {0}, true);
    }
    batch.n_tokens = n_tokens;
    if (llama_decode(lctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    llama_batch_free(batch);
    return ret;
}

int main(int argc, char ** argv) {
    common_params params;
    params.escape = false;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_RESULTS)) {
        return 1;
    }
    if (params.out_file.empty()) {
        LOG_ERR("%s: an output file must be specified", __func__);
        return 1;
    }
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);
    common_init_result_ptr llama_init = common_init_from_params(params);
    struct llama_model   * model = llama_init->model();
    struct llama_context * lctx  = llama_init->context();
    if (model == nullptr) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    const std::vector<llama_token> tokens_calc = common_tokenize(lctx, params.prompt, true);
    const std::vector<float> logits_calc = get_logits(model, lctx, tokens_calc);
    GGML_ASSERT(logits_calc.size() == tokens_calc.size()*n_vocab);

    struct gguf_init_params gguf_params = {
        /*.no_alloc   =*/ true,
        /*.ctx        =*/ nullptr,
    };
    gguf_context_ptr gguf_ctx_model(gguf_init_from_file(params.model.path.c_str(), gguf_params));

    if (params.check) {
        LOG_INF("%s: loading results from %s...\n", __func__, params.out_file.c_str());
        gguf_context_ptr gguf_ctx;
        {
            struct gguf_init_params gguf_params = {
                /*no_alloc =*/ true,
                /*ctx      =*/ nullptr,
            };
            gguf_ctx.reset(gguf_init_from_file(params.out_file.c_str(), gguf_params));
        }
        const std::string path_model_disk = gguf_get_val_str(gguf_ctx.get(), gguf_find_key(gguf_ctx.get(), "path_model"));
        GGML_ASSERT(path_model_disk == params.model.path); // TODO better checks

        auto load_tensor_data = [&](const std::string & name, void * dst, const size_t size){
            const int64_t tid    = gguf_find_tensor(gguf_ctx.get(), name.c_str());
            const size_t  offset = gguf_get_data_offset(gguf_ctx.get()) + gguf_get_tensor_offset(gguf_ctx.get(), tid);
            GGML_ASSERT(size == gguf_get_tensor_size(gguf_ctx.get(), tid));

            FILE * file = ggml_fopen(params.out_file.c_str(), "rb");
            if (file == nullptr) {
                throw std::runtime_error("failed to open results file");
            }
            if (fseek(file, offset, SEEK_SET) != 0) {
                throw std::runtime_error("fseek failed");
            }
            const size_t nbytes_read = fread(dst, 1, size, file);
            if (nbytes_read != size) {
                throw std::runtime_error("fread failed");
            }
        };

        std::vector<llama_token> tokens_disk(tokens_calc.size());
        load_tensor_data("tokens", tokens_disk.data(), tokens_disk.size()*sizeof(llama_token));
        GGML_ASSERT(tokens_disk.size() == tokens_calc.size());
        for (size_t i = 0; i < tokens_calc.size(); i++) {
            GGML_ASSERT(tokens_disk[i] == tokens_calc[i]);
        }

        std::vector<float> logits_disk(logits_calc.size());
        load_tensor_data("logits", logits_disk.data(), logits_disk.size()*sizeof(float));
        const double nmse_val = nmse(logits_disk, logits_calc);
        LOG_INF("%s: NMSE=%.3e\n", __func__, nmse_val);

        if (nmse_val > 1e-6) {
            printf("\033[1;31mFAIL\033[0m\n");
            return 1;
        }

        printf("\033[1;32mOK\033[0m\n");
        return 0;
    }

    ggml_context_ptr ggml_ctx_calc;
    {
        const size_t size_tokens = tokens_calc.size()*sizeof(llama_token) + ggml_tensor_overhead();
        const size_t size_logits = logits_calc.size()*sizeof(float)  + ggml_tensor_overhead();
        struct ggml_init_params params = {
            /*.mem_size   =*/ size_tokens + size_logits,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };
        ggml_ctx_calc.reset(ggml_init(params));
    }

    gguf_context_ptr gguf_ctx(gguf_init_empty());
    gguf_set_val_str(gguf_ctx.get(), "path_model", params.model.path.c_str());
    {
        ggml_tensor * t_tokens = ggml_new_tensor_1d(ggml_ctx_calc.get(), GGML_TYPE_I32, tokens_calc.size());
        ggml_set_name(t_tokens, "tokens");
        int32_t * tokens_data = (int32_t *) t_tokens->data;
        for (uint32_t i = 0; i < tokens_calc.size(); i++) {
            tokens_data[i] = tokens_calc[i];
        }
        gguf_add_tensor(gguf_ctx.get(), t_tokens);
    }
    {
        ggml_tensor * t_logits = ggml_new_tensor_2d(ggml_ctx_calc.get(), GGML_TYPE_F32, tokens_calc.size(), n_vocab);
        ggml_set_name(t_logits, "logits");
        float * logits_data = ggml_get_data_f32(t_logits);
        for (uint32_t i = 0; i < tokens_calc.size(); i++) {
            const float * logits_ith = llama_get_logits_ith(lctx, i);
            for (uint32_t j = 0; j < n_vocab; j++) {
                logits_data[i*n_vocab + j] = logits_ith[j];
            }
        }
        gguf_add_tensor(gguf_ctx.get(), t_logits);
    }
    LOG_INF("%s: writing results to %s...\n", __func__, params.out_file.c_str());
    gguf_write_to_file(gguf_ctx.get(), params.out_file.c_str(), /*only_meta =*/ false);
    return 0;
}

