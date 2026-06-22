#include "gguf.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <vector>

static void add_base_vocab_metadata(struct gguf_context * ctx) {
    gguf_set_val_str(ctx, "general.architecture", "mpt");
    gguf_set_val_str(ctx, "tokenizer.ggml.model", "t5");

    const char * tokens[] = {
        "<pad>", "</s>", "<unk>", "a", "b", "c", "d", "e",
    };
    gguf_set_arr_str(ctx, "tokenizer.ggml.tokens", tokens, sizeof(tokens) / sizeof(tokens[0]));
}

static bool load_vocab_from_ctx(const struct gguf_context * ctx) {
    FILE * file = tmpfile();
    if (!file) {
        fprintf(stderr, "failed to create temporary file\n");
        return false;
    }

    if (!gguf_write_to_file_ptr(ctx, file, false)) {
        fprintf(stderr, "failed to write GGUF test file\n");
        fclose(file);
        return false;
    }

    fflush(file);
    rewind(file);

    llama_model_params params = llama_model_default_params();
    params.vocab_only = true;
    params.use_mmap  = false;
    params.no_alloc  = true;

    llama_model * model = llama_model_load_from_file_ptr(file, params);
    fclose(file);

    if (model) {
        llama_model_free(model);
        return true;
    }

    return false;
}

static bool expect_load_result(const char * name, const struct gguf_context * ctx, const bool expected) {
    const bool actual = load_vocab_from_ctx(ctx);
    if (actual != expected) {
        fprintf(stderr, "%s: expected load %s, got %s\n",
                name, expected ? "success" : "failure", actual ? "success" : "failure");
        return false;
    }
    return true;
}

static gguf_context * make_base_ctx() {
    gguf_context * ctx = gguf_init_empty();
    add_base_vocab_metadata(ctx);
    return ctx;
}

int main() {
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();

    std::vector<gguf_context *> ctxs;
    bool ok = true;

    {
        gguf_context * ctx = make_base_ctx();
        ctxs.push_back(ctx);
        ok = expect_load_result("valid minimal vocab", ctx, true) && ok;
    }

    {
        gguf_context * ctx = make_base_ctx();
        ctxs.push_back(ctx);
        const uint8_t scores[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        gguf_set_arr_data(ctx, "tokenizer.ggml.scores", GGUF_TYPE_UINT8, scores, sizeof(scores));
        ok = expect_load_result("uint8 scores", ctx, false) && ok;
    }

    {
        gguf_context * ctx = make_base_ctx();
        ctxs.push_back(ctx);
        const float scores[] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        gguf_set_arr_data(
                ctx, "tokenizer.ggml.scores",
                GGUF_TYPE_FLOAT32, scores, sizeof(scores) / sizeof(scores[0]));
        ok = expect_load_result("float32 scores", ctx, true) && ok;
    }

    {
        gguf_context * ctx = make_base_ctx();
        ctxs.push_back(ctx);
        const uint8_t token_types[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        gguf_set_arr_data(
                ctx, "tokenizer.ggml.token_type",
                GGUF_TYPE_UINT8, token_types, sizeof(token_types));
        ok = expect_load_result("uint8 token types", ctx, false) && ok;
    }

    {
        gguf_context * ctx = make_base_ctx();
        ctxs.push_back(ctx);
        const int32_t token_types[] = { 1, 1, 1, 1, 1, 1, 1, 1 };
        gguf_set_arr_data(
                ctx, "tokenizer.ggml.token_type",
                GGUF_TYPE_INT32, token_types, sizeof(token_types) / sizeof(token_types[0]));
        ok = expect_load_result("int32 token types", ctx, true) && ok;
    }

    {
        gguf_context * ctx = make_base_ctx();
        ctxs.push_back(ctx);
        const uint8_t suppress_tokens[] = { 0 };
        gguf_set_arr_data(
                ctx, "tokenizer.ggml.suppress_tokens",
                GGUF_TYPE_UINT8, suppress_tokens, sizeof(suppress_tokens));
        ok = expect_load_result("uint8 suppress tokens", ctx, false) && ok;
    }

    {
        gguf_context * ctx = make_base_ctx();
        ctxs.push_back(ctx);
        const int32_t suppress_tokens[] = { 0 };
        gguf_set_arr_data(
                ctx, "tokenizer.ggml.suppress_tokens",
                GGUF_TYPE_INT32, suppress_tokens, sizeof(suppress_tokens) / sizeof(suppress_tokens[0]));
        ok = expect_load_result("int32 suppress tokens", ctx, true) && ok;
    }

    {
        gguf_context * ctx = make_base_ctx();
        ctxs.push_back(ctx);
        gguf_set_arr_data(
                ctx, "tokenizer.ggml.precompiled_charsmap",
                GGUF_TYPE_UINT8, nullptr, 0);
        ok = expect_load_result("empty precompiled charsmap", ctx, true) && ok;
    }

    {
        gguf_context * ctx = make_base_ctx();
        ctxs.push_back(ctx);
        const uint8_t precompiled_charsmap[] = { 0 };
        gguf_set_arr_data(
                ctx, "tokenizer.ggml.precompiled_charsmap",
                GGUF_TYPE_UINT8, precompiled_charsmap, sizeof(precompiled_charsmap));
        ok = expect_load_result("short precompiled charsmap", ctx, false) && ok;
    }

    for (gguf_context * ctx : ctxs) {
        gguf_free(ctx);
    }

    llama_backend_free();
    return ok ? 0 : 1;
}
