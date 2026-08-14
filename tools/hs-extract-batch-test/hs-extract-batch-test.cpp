// Test if warmup + dynamic extract_hidden_states toggle causes value differences
// Reproduces the server's exact initialization path

#include "common.h"
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model> <prompt> [mode]\n", argv[0]);
        fprintf(stderr, "  mode=0: No warmup, extract_hidden_states=true from start (CLI style)\n");
        fprintf(stderr, "  mode=1: Warmup first, then toggle extract_hidden_states (server style)\n");
        return 1;
    }

    const char * model_path = argv[1];
    const char * prompt = argv[2];
    int mode = argc > 3 ? atoi(argv[3]) : 0;

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 100;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "Failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens;
    tokens.resize(strlen(prompt) + 16);
    int n_tokens = llama_tokenize(vocab, prompt, strlen(prompt), tokens.data(), tokens.size(), true, true);
    tokens.resize(n_tokens);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 2048;
    cparams.n_ubatch = 512;

    if (mode == 0) {
        // CLI style: extract_hidden_states=true from creation
        cparams.extract_hidden_states = true;
        cparams.no_perf = true;
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) { fprintf(stderr, "Failed to init context\n"); return 1; }

        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "Decode failed\n"); return 1; }
        llama_synchronize(ctx);

        float * hs = llama_get_hidden_state(ctx, 10);
        printf("Mode 0 (CLI) - Layer 10, first 5 values:\n");
        for (int i = 0; i < 5; i++) printf("  [%d]: %.8f\n", i, hs[i]);

        llama_free(ctx);
    } else {
        // Server style: extract_hidden_states=false at creation, warmup, then toggle
        cparams.extract_hidden_states = false;
        cparams.no_perf = true;
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) { fprintf(stderr, "Failed to init context\n"); return 1; }

        // Warmup (server does this in common_init_from_params)
        llama_token bos = llama_vocab_bos(vocab);
        llama_token eos = llama_vocab_eos(vocab);
        std::vector<llama_token> warmup_tokens;
        if (bos != LLAMA_TOKEN_NULL) warmup_tokens.push_back(bos);
        if (eos != LLAMA_TOKEN_NULL) warmup_tokens.push_back(eos);
        if (!warmup_tokens.empty()) {
            llama_batch wb = llama_batch_get_one(warmup_tokens.data(), warmup_tokens.size());
            llama_decode(ctx, wb);
            llama_synchronize(ctx);
            fprintf(stderr, "Warmup done with %zu tokens\n", warmup_tokens.size());
        }

        // Toggle extract_hidden_states ON (server does this per-request)
        llama_set_extract_hidden_states(ctx, true);

        // Clear memory (our fix)
        llama_memory_clear(llama_get_memory(ctx), true);

        // Decode the actual prompt
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "Decode failed\n"); return 1; }
        llama_synchronize(ctx);

        float * hs = llama_get_hidden_state(ctx, 10);
        if (!hs) {
            fprintf(stderr, "No hidden states available!\n");
            // Try to diagnose
            int32_t n_ht = llama_get_hidden_state_n_tokens(ctx);
            fprintf(stderr, "n_hidden_tokens = %d\n", n_ht);
        } else {
            printf("Mode 1 (Server-style) - Layer 10, first 5 values:\n");
            for (int i = 0; i < 5; i++) printf("  [%d]: %.8f\n", i, hs[i]);
        }

        llama_free(ctx);
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
