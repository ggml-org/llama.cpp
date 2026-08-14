#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// RAII wrappers (LlamaBackend, LlamaModel, LlamaContext, LlamaBatch) are in
// common/llama-raii.h (shared with hs-extract tools and examples).
#include "llama-raii.h"

// Backward compat alias: this test uses LlamaBatchRAII instead of LlamaBatch.
using LlamaBatchRAII = LlamaBatch;

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    LlamaBackend backend;

    // Load model
    llama_model_params mparams = llama_model_default_params();
    LlamaModel model(llama_model_load_from_file(argv[1], mparams));
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", argv[1]);
        return 1;
    }

    // Create context with hidden state extraction enabled
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 512;
    cparams.n_ubatch = 512;
    cparams.extract_hidden_states = true;

    LlamaContext ctx(llama_init_from_model(model, cparams));
    if (!ctx) {
        fprintf(stderr, "Failed to create context\n");
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_embd = llama_model_n_embd_out(model);
    const int n_layer = llama_model_n_layer(model);

    printf("Model loaded: n_vocab=%d, n_embd=%d, n_layer=%d\n", n_vocab, n_embd, n_layer);

    // Tokenize a simple prompt
    const char * prompt = "Hello world";
    int n_tokens = llama_tokenize(llama_model_get_vocab(model), prompt, strlen(prompt), NULL, 0, true, true);
    if (n_tokens < 0) {
        n_tokens = -n_tokens; // negative means buffer too small, get actual count
    }
    if (n_tokens <= 0) {
        fprintf(stderr, "Tokenization error: %d\n", n_tokens);
        return 1;
    }

    std::vector<llama_token> tokens(n_tokens);
    n_tokens = llama_tokenize(llama_model_get_vocab(model), prompt, strlen(prompt), tokens.data(), n_tokens, true, true);

    printf("Tokenized '%s' -> %d tokens\n", prompt, n_tokens);

    // Decode using llama_batch_init (modern API)
    LlamaBatchRAII batch_wrapper;
    batch_wrapper.init(n_tokens, 0, 1);
    llama_batch & batch = batch_wrapper.batch;
    for (int i = 0; i < n_tokens; i++) {
        batch.token[i]    = tokens[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = (i == n_tokens - 1) ? 1 : 0;
    }
    batch.n_tokens = n_tokens;

    int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        fprintf(stderr, "llama_decode failed with %d\n", ret);
        return 1;
    }

    printf("Decode succeeded.\n");

    llama_synchronize(ctx);

    // Test hidden state extraction for layer 0
    int32_t n_hidden_tokens = llama_get_hidden_state_n_tokens(ctx);
    printf("n_hidden_tokens = %d\n", (int)n_hidden_tokens);

    if (n_hidden_tokens <= 0) {
        fprintf(stderr, "FAIL: n_hidden_tokens is %d (expected > 0)\n", (int)n_hidden_tokens);
        return 1;
    }

    int non_zero = 0;
    int total = 0;

    // Check first 3 layers
    int check_layers = n_layer < 3 ? n_layer : 3;
    for (int il = 0; il < check_layers; il++) {
        float * hs = llama_get_hidden_state(ctx, il);
        if (!hs) {
            fprintf(stderr, "FAIL: llama_get_hidden_state(ctx, %d) returned NULL\n", il);
            return 1;
        }

        // Check for non-zero values
        for (int i = 0; i < n_hidden_tokens * n_embd; i++) {
            total++;
            if (fabsf(hs[i]) > 1e-6f) {
                non_zero++;
            }
        }
    }

    // Show some values from layer 0
    printf("Layer 0 first 10 values: ");
    float * hs0 = llama_get_hidden_state(ctx, 0);
    for (int i = 0; i < 10 && i < n_embd; i++) {
        printf("%.6f ", hs0[i]);
    }
    printf("\n");

    if (total > 0) {
        printf("Non-zero / total (first %d layers): %d / %d (%.1f%%)\n", check_layers, non_zero, total, 100.0f * non_zero / total);
    } else {
        printf("Non-zero / total (first %d layers): no data (total=0)\n", check_layers);
    }

    // RAII destructors handle all cleanup automatically

    if (non_zero > 0) {
        printf("PASS: Hidden states contain non-zero values.\n");
        return 0;
    }
    fprintf(stderr, "FAIL: All hidden state values are zero.\n");
    return 1;
}
