#include "ggml.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static bool silent_progress_callback(float /* progress */, void * /* user_data */) {
    return true;
}

static void set_base_vocab_metadata(
        struct gguf_context * ctx,
        std::vector<const char *> tokens,
        const char * tokenizer_model = "t5") {
    gguf_set_val_str(ctx, "general.architecture", "mpt");
    gguf_set_val_str(ctx, "tokenizer.ggml.model", tokenizer_model);
    gguf_set_arr_str(ctx, "tokenizer.ggml.tokens", tokens.data(), tokens.size());
}

static llama_model_ptr load_vocab_from_gguf(const struct gguf_context * ctx) {
    FILE * file = tmpfile();
    GGML_ASSERT(file);

    if (!gguf_write_to_file_ptr(ctx, file, true)) {
        fclose(file);
        throw std::runtime_error("failed to write GGUF metadata");
    }
    rewind(file);

    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    params.use_mmap   = false;
    params.no_alloc   = true;
    params.progress_callback = silent_progress_callback;

    llama_model_ptr model(llama_model_load_from_file_ptr(file, params));
    fclose(file);
    return model;
}

static bool load_vocab_succeeds(const struct gguf_context * ctx) {
    return load_vocab_from_gguf(ctx) != nullptr;
}

static bool tokenize_succeeds(llama_model * model, const std::string & text) {
    const llama_vocab * vocab = llama_model_get_vocab(model);

    int32_t n_tokens = llama_tokenize(vocab, text.data(), (int32_t) text.size(), nullptr, 0, true, true);
    if (n_tokens < 0) {
        n_tokens = -n_tokens;
    }

    std::vector<llama_token> tokens(n_tokens);
    return llama_tokenize(vocab, text.data(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(), true, true) >= 0;
}

static void test_duplicate_token_string_fails_load() {
    gguf_context_ptr ctx(gguf_init_empty());
    set_base_vocab_metadata(ctx.get(), { "x", "x" });

    GGML_ASSERT(!load_vocab_succeeds(ctx.get()));
}

static void test_add_bos_without_bos_id_fails_load() {
    gguf_context_ptr ctx(gguf_init_empty());
    set_base_vocab_metadata(ctx.get(), { "<pad>", "</s>", "<unk>", "a" });
    gguf_set_val_bool(ctx.get(), "tokenizer.ggml.add_bos_token", true);

    GGML_ASSERT(!load_vocab_succeeds(ctx.get()));
}

static void test_add_eos_without_eos_id_fails_load() {
    gguf_context_ptr ctx(gguf_init_empty());
    set_base_vocab_metadata(ctx.get(), { "[PAD]", "[CLS]", "[SEP]", "[UNK]", "a" }, "bert");
    gguf_set_val_u32(ctx.get(), "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(ctx.get(), "tokenizer.ggml.sep_token_id", 2);
    gguf_set_val_u32(ctx.get(), "tokenizer.ggml.unk_token_id", 3);
    gguf_set_val_bool(ctx.get(), "tokenizer.ggml.add_eos_token", true);

    GGML_ASSERT(!load_vocab_succeeds(ctx.get()));
}

static void test_add_sep_without_sep_id_fails_load() {
    gguf_context_ptr ctx(gguf_init_empty());
    set_base_vocab_metadata(ctx.get(), { "<pad>", "</s>", "<unk>", "a" });
    gguf_set_val_bool(ctx.get(), "tokenizer.ggml.add_sep_token", true);

    GGML_ASSERT(!load_vocab_succeeds(ctx.get()));
}

static void test_add_bos_with_bos_id_loads_and_tokenizes() {
    gguf_context_ptr ctx(gguf_init_empty());
    set_base_vocab_metadata(ctx.get(), { "<pad>", "</s>", "<unk>", "<s>", "a" });
    gguf_set_val_bool(ctx.get(), "tokenizer.ggml.add_bos_token", true);
    gguf_set_val_u32(ctx.get(), "tokenizer.ggml.bos_token_id", 3);

    llama_model_ptr model = load_vocab_from_gguf(ctx.get());
    GGML_ASSERT(model);
    GGML_ASSERT(tokenize_succeeds(model.get(), "a"));
}

int main() {
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);

    FILE * file = tmpfile();
#ifdef _WIN32
    if (!file) {
        fprintf(stderr, "failed to create tmpfile(), needs elevated privileges on Windows");
        fprintf(stderr, "skipping tests");
        return 0;
    }
#else
    GGML_ASSERT(file);
#endif
    fclose(file);

    test_duplicate_token_string_fails_load();
    test_add_bos_without_bos_id_fails_load();
    test_add_eos_without_eos_id_fails_load();
    test_add_sep_without_sep_id_fails_load();
    test_add_bos_with_bos_id_loads_and_tokenizes();

    return 0;
}
