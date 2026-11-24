// ref: https://github.com/ggml-org/llama.cpp/issues/32662
// a model with no tokenizer gets its vocab size from metadata only, so the loader must bound it

#include "llama.h"
#include "gguf.h"

#include <cstdint>
#include <filesystem>
#include <string>

// llama model with no tensors and a "none" tokenizer, loading stops after the vocab
static llama_model * load_vocab_only(uint32_t n_vocab) {
    // relative path, ctest runs each test in its build dir
    const std::string path = "test-model-load-vocab.gguf";

    gguf_context * ctx = gguf_init_empty();
    gguf_set_val_str(ctx, "general.architecture",                   "llama");
    gguf_set_val_u32(ctx, "llama.block_count",                      1);
    gguf_set_val_u32(ctx, "llama.context_length",                   64);
    gguf_set_val_u32(ctx, "llama.embedding_length",                 64);
    gguf_set_val_u32(ctx, "llama.feed_forward_length",              128);
    gguf_set_val_u32(ctx, "llama.attention.head_count",             2);
    gguf_set_val_f32(ctx, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(ctx, "llama.vocab_size",                       n_vocab);
    gguf_set_val_str(ctx, "tokenizer.ggml.model",                   "none");
    const bool ok = gguf_write_to_file(ctx, path.c_str(), false);
    gguf_free(ctx);
    GGML_ASSERT(ok);

    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    llama_model * model = llama_model_load_from_file(path.c_str(), params);
    std::filesystem::remove(path);
    return model;
}

int main() {
    llama_backend_init();

    llama_model * model = load_vocab_only(128);
    GGML_ASSERT(model != nullptr);
    GGML_ASSERT(llama_vocab_n_tokens(llama_model_get_vocab(model)) == 128);
    llama_model_free(model);

    // one above the cap in llama-vocab.cpp
    model = load_vocab_only((1u << 24) + 1);
    GGML_ASSERT(model == nullptr);

    llama_backend_free();

    return 0;
}
