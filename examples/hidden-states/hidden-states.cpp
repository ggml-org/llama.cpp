#include "llama.h"

#include <cstdio>
#include <cstring>
#include <vector>

// -- RAII Wrappers for Automatic Resource Cleanup -----------------------

struct LlamaModel {
    llama_model * model;
    LlamaModel() : model(nullptr) {}
    LlamaModel(llama_model * m) : model(m) {}
    ~LlamaModel() { if (model) llama_model_free(model); }
    LlamaModel(const LlamaModel &) = delete;
    LlamaModel & operator=(const LlamaModel &) = delete;
    operator llama_model *() const { return model; }
    explicit operator bool() const { return model != nullptr; }
};

struct LlamaContext {
    llama_context * ctx;
    LlamaContext() : ctx(nullptr) {}
    LlamaContext(llama_context * c) : ctx(c) {}
    ~LlamaContext() { if (ctx) llama_free(ctx); }
    LlamaContext(const LlamaContext &) = delete;
    LlamaContext & operator=(const LlamaContext &) = delete;
    operator llama_context *() const { return ctx; }
    explicit operator bool() const { return ctx != nullptr; }
};

struct LlamaBackend {
    LlamaBackend() { llama_backend_init(); }
    ~LlamaBackend() { llama_backend_free(); }
    LlamaBackend(const LlamaBackend &) = delete;
    LlamaBackend & operator=(const LlamaBackend &) = delete;
};

struct LlamaBatch {
    llama_batch batch;
    bool initialized;
    LlamaBatch() : batch{}, initialized(false) {}
    void init(int32_t n_tokens, int32_t embd, int32_t n_seq_max) {
        batch = llama_batch_init(n_tokens, embd, n_seq_max);
        initialized = true;
    }
    ~LlamaBatch() { if (initialized) llama_batch_free(batch); }
    LlamaBatch(const LlamaBatch &) = delete;
    LlamaBatch & operator=(const LlamaBatch &) = delete;
    operator llama_batch *() { return &batch; }
};

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    LlamaBackend backend;

    llama_model_params model_params = llama_model_default_params();
    LlamaModel model(llama_model_load_from_file(argv[1], model_params));
    if (!model) {
        fprintf(stderr, "%s: failed to load model '%s'\n", __func__, argv[1]);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.extract_hidden_states = true;

    LlamaContext ctx(llama_init_from_model(model, ctx_params));
    if (!ctx) {
        fprintf(stderr, "%s: failed to create context\n", __func__);
        return 1;
    }

    const char * prompt = "Hello, world!";
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(512);
    int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt), tokens.data(), tokens.size(), true, true);
    if (n_tokens < 0) {
        fprintf(stderr, "%s: failed to tokenize\n", __func__);
        return 1;
    }
    tokens.resize(n_tokens);

    printf("Prompt: %s\n", prompt);
    printf("Tokens: %d\n", n_tokens);

    LlamaBatch batch_wrapper;
    batch_wrapper.init(tokens.size(), 0, 1);
    llama_batch & batch = batch_wrapper.batch;
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = (llama_pos) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == tokens.size() - 1) ? 1 : 0;
    }
    batch.n_tokens = tokens.size();

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "%s: failed to decode\n", __func__);
        return 1;
    }

    // synchronize before reading hidden states (CUDA async write race)
    llama_synchronize(ctx);

    int n_layers = llama_model_n_layer(model);
    int32_t n_hidden_tokens = llama_get_hidden_state_n_tokens(ctx);

    if (n_hidden_tokens == 0) {
        fprintf(stderr, "Error: no hidden states extracted (count=0)\n");
        return 1;
    }

    printf("Hidden states: %d tokens, %d layers\n", n_hidden_tokens, n_layers);

    for (int il = 0; il < n_layers && il < 5; il++) {
        float * hidden = llama_get_hidden_state(ctx, il);
        if (hidden) {
            printf("Layer %d: [", il);
            for (int i = 0; i < 5; i++) {
                printf("%.4f%s", hidden[i], i < 4 ? ", " : "");
            }
            printf("...]\n");
        } else {
            printf("Layer %d: NULL\n", il);
        }
    }

    return 0;
}
