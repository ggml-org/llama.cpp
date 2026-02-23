#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

#include <cstring>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const float * a, const float * b, size_t n) {
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < n; i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    std::mt19937 * gen = (std::mt19937 *) userdata;
    std::normal_distribution<float> dis(0.0f, 1e-2f);

    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    const int64_t ne = ggml_nelements(tensor);
    std::vector<float> tmp(ne);
    for (int64_t i = 0; i < ne; i++) {
        tmp[i] = dis(*gen);
    }
    ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
}

enum test_mode {
    TEST_MODE_GEN_MODEL,
    TEST_MODE_GEN_RESULTS,
    TEST_MODE_TEST_VS_DISK,
    TEST_MODE_TEST_BACKENDS,
};

static const char * test_mode_name(test_mode mode) {
    switch (mode) {
        case TEST_MODE_GEN_MODEL:
            return "gen-model";
        case TEST_MODE_GEN_RESULTS:
            return "gen-results";
        case TEST_MODE_TEST_VS_DISK:
            return "test-vs-disk";
        case TEST_MODE_TEST_BACKENDS:
            return "test-backends";
    }
    GGML_ABORT("fatal error");
}

static void usage(char ** argv) {
    printf("Usage: %s [mode] [-m/--model <path_model>] [-r/--results <path_results>]\n", argv[0]);
    printf("    valid modes:\n");
    printf("      - %s (generate a dummy model and save it to disk)\n", test_mode_name(TEST_MODE_GEN_MODEL));
    printf("      - %s (load a model from disk, calculate results, and save the results to disk)\n", test_mode_name(TEST_MODE_GEN_RESULTS));
    printf("      - %s (default, load model and results from disk, calculate results from model, compare vs. disk)\n", test_mode_name(TEST_MODE_TEST_VS_DISK));
    printf("      - %s (test wheter model results are consistent for 2+ ggml backends, model can be loaded from disk, dummy otherwise)\n", test_mode_name(TEST_MODE_TEST_BACKENDS));
}

static int gen_model(const char * path_model) {
    const uint32_t n_ctx   = 32;
    const uint32_t n_embd  = 32;
    const uint32_t n_head  = 1;
    const uint32_t n_ff    = 32;
    const uint32_t n_vocab = 32;
    const uint32_t n_layer = 1;

    gguf_context_ptr metadata(gguf_init_empty());
    gguf_set_val_str(metadata.get(), "general.architecture",                   "llama");
    gguf_set_val_u32(metadata.get(), "llama.context_length",                   n_ctx);
    gguf_set_val_u32(metadata.get(), "llama.embedding_length",                 n_embd);
    gguf_set_val_u32(metadata.get(), "llama.attention.head_count",             n_head);
    gguf_set_val_u32(metadata.get(), "llama.feed_forward_length",              n_ff);
    gguf_set_val_u32(metadata.get(), "llama.block_count",                      n_layer);
    gguf_set_val_f32(metadata.get(), "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(metadata.get(), "llama.vocab_size",                       n_vocab);

    gguf_set_val_str(metadata.get(), "tokenizer.ggml.model",  "no_vocab");

    auto add_tensor = [&](const std::string & name, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1, int64_t ne3 = 1) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F32;
        t.ne[0] = ne0;
        t.ne[1] = ne1;
        t.ne[2] = ne2;
        t.ne[3] = ne3;
        t.nb[0] = 4;
        for (int dim = 1; dim < GGML_MAX_DIMS; dim++) {
            t.nb[dim] = t.nb[dim - 1] * t.ne[dim - 1];
        }
        ggml_set_name(&t, name.c_str());
        gguf_add_tensor(metadata.get(), &t);
    };

    add_tensor("token_embd.weight", n_embd, n_vocab);

    std::random_device rd;
    std::mt19937 gen(rd());

    llama_model_params params = llama_model_default_params();
    llama_model_ptr model(llama_model_init(metadata.get(), set_tensor_data, &gen, params));
    llama_model_save_to_file(model.get(), path_model);

    return 0;
}

static int gen_results(const char * path_model, const char * path_results) {
    llama_model_params   model_params = llama_model_default_params();
    llama_context_params ctx_params   = llama_context_default_params();
    llama_model_ptr model(llama_model_load_from_file(path_model, model_params));
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));

    const uint32_t n_vocab  = 32;
    const uint32_t n_tokens = 32;
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        common_batch_add(batch, pos, pos, {0}, true);
    }
    batch.n_tokens = n_tokens;
    if (llama_decode(lctx.get(), batch)) {
        throw std::runtime_error("failed to decode batch");
    }

    ggml_context_ptr ggml_ctx_logits;
    {
        const size_t size_logits = n_tokens*ggml_row_size(GGML_TYPE_F32, n_vocab) + ggml_tensor_overhead();
        struct ggml_init_params params = {
            /*.mem_size   =*/ size_logits,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };
        ggml_ctx_logits.reset(ggml_init(params));
    }
    ggml_tensor * logits = ggml_new_tensor_2d(ggml_ctx_logits.get(), GGML_TYPE_F32, n_vocab, n_tokens);
    ggml_set_name(logits, "logits");
    float * logits_data = ggml_get_data_f32(logits);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx.get(), i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            logits_data[i*n_vocab + j] = logits_ith[j];
        }
    }

    gguf_context_ptr gguf_ctx_logits(gguf_init_empty());
    gguf_add_tensor(gguf_ctx_logits.get(), logits);
    gguf_write_to_file(gguf_ctx_logits.get(), path_results, /*only_meta =*/ false);

    return 0;
}

static int test_vs_disk(const char * path_model, const char * path_results) {
    llama_model_params   model_params = llama_model_default_params();
    llama_context_params ctx_params   = llama_context_default_params();
    llama_model_ptr model(llama_model_load_from_file(path_model, model_params));
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));

    const uint32_t n_vocab  = 32;
    const uint32_t n_tokens = 32;
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        common_batch_add(batch, pos, pos, {0}, true);
    }
    if (llama_decode(lctx.get(), batch)) {
        throw std::runtime_error("failed to decode batch");
    }
    std::vector<float> logits_calc(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx.get(), i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            logits_calc[i*n_vocab + j] = logits_ith[j];
        }
    }

    ggml_context_ptr ggml_ctx_logits;
    gguf_context_ptr gguf_ctx_logits;
    {
        struct ggml_context * tmp;
        struct gguf_init_params gguf_params = {
            /*no_alloc =*/ false,
            /*ctx      =*/ &tmp,
        };
        gguf_ctx_logits.reset(gguf_init_from_file(path_results, gguf_params));
        ggml_ctx_logits.reset(tmp);
    }
    const int64_t tid    = gguf_find_tensor(gguf_ctx_logits.get(), "logits");
    const size_t  offset = gguf_get_data_offset(gguf_ctx_logits.get()) + gguf_get_tensor_offset(gguf_ctx_logits.get(), tid);
    const size_t  size   = gguf_get_tensor_size(gguf_ctx_logits.get(), tid);
    GGML_ASSERT(size == n_tokens*ggml_row_size(GGML_TYPE_F32, n_vocab));

    FILE * file = ggml_fopen(path_results, "rb");
    if (file == nullptr) {
        throw std::runtime_error("failed to open results file");
    }
    if (fseek(file, offset, SEEK_SET) != 0) {
        throw std::runtime_error("fseek failed");
    }
    std::vector<float> logits_disk(n_tokens*n_vocab);
    {
        const size_t ne_read = fread(logits_disk.data(), sizeof(float), logits_disk.size(), file);
        if (ne_read != logits_disk.size()) {
            throw std::runtime_error("fread failed");
        }
    }

    const double nmse_val = nmse(logits_calc.data(), logits_disk.data(), n_tokens*n_vocab);
    if (nmse_val > 1e-6) {
        printf("\033[1;31mFAIL\033[0m\n");
        return 1;
    }

    printf("\033[1;32mOK\033[0m\n");
    return 0;
}

int main(int argc, char ** argv) {
    test_mode mode = TEST_MODE_TEST_VS_DISK;
    const char * path_model = nullptr;
    const char * path_results = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (i + 1 >= argc) {
                usage(argv);
                return 1;
            }
            path_model = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--results") == 0) {
            if (i + 1 >= argc) {
                usage(argv);
                return 1;
            }
            path_results = argv[++i];
            continue;
        }
        for (test_mode m : {TEST_MODE_GEN_MODEL, TEST_MODE_GEN_RESULTS, TEST_MODE_TEST_VS_DISK, TEST_MODE_TEST_BACKENDS}) {
            if (strcmp(test_mode_name(m), argv[i]) == 0) {
                mode = m;
                break;
            }
        }
    }

    if (path_model == nullptr) {
        printf("ERROR: a model path is required\n\n");
        usage(argv);
        return 1;
    }

    if (path_results == nullptr && mode != TEST_MODE_GEN_MODEL) {
        printf("ERROR: a results path is required for mode %s\n\n", test_mode_name(mode));
        usage(argv);
        return 1;
    }

    try {
        switch (mode) {
            case TEST_MODE_GEN_MODEL:
                return gen_model(path_model);
            case TEST_MODE_GEN_RESULTS:
                return gen_results(path_model, path_results);
            case TEST_MODE_TEST_VS_DISK:
                return test_vs_disk(path_model, path_results);
            case TEST_MODE_TEST_BACKENDS:
                printf("not implemented");
                return -1;
        }
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
