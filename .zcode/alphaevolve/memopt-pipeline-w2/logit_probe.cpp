// Standalone greedy argmax token-id probe for memopt-pipeline-w1.
// Emits one token id per line on stdout for n_predict decode steps.
// Based on examples/simple/simple.cpp. NOT part of the S1 diff.
//
// Build:
//   c++ -std=c++17 -I<WT>/include -I<WT>/ggml/include -I<WT>/common \
//       <WT>/logit_probe.cpp \
//       -L<WT>/build-g1/src -lllama -L<WT>/build-g1/common -lllama-common \
//       -L<WT>/build-g1/ggml/src -lggml -lggml-base \
//       -Wl,-rpath,<WT>/build-g1/src -Wl,-rpath,<WT>/build-g1/common \
//       -Wl,-rpath,<WT>/build-g1/ggml/src \
//       -o <WT>/build-g1/bin/logit_probe
//
// Usage: logit_probe -m <model> [-ctk TYPE] [-ctv TYPE] [-kvu] [-fa on|off] \
//                    [-p PROMPT] [-n N]
#include "llama.h"
#include "common.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static ggml_type parse_type(const char * s) {
    if (!s) return GGML_TYPE_F16;
    if (strcmp(s, "f32") == 0) return GGML_TYPE_F32;
    if (strcmp(s, "f16") == 0) return GGML_TYPE_F16;
    if (strcmp(s, "q8_0") == 0) return GGML_TYPE_Q8_0;
    if (strcmp(s, "q4_0") == 0) return GGML_TYPE_Q4_0;
    return GGML_TYPE_F16;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    std::string prompt = "The quick brown fox jumps over the lazy dog. Explain why";
    int ngl = 99;
    int n_predict = 40;
    ggml_type type_k = GGML_TYPE_F16;
    ggml_type type_v = GGML_TYPE_F16;
    bool kv_unified = false;
    const char * fa = "auto"; // flash attn: on/off/auto

    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-m") == 0 && i+1 < argc) {
                model_path = argv[++i];
            } else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) {
                n_predict = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-ngl") == 0 && i+1 < argc) {
                ngl = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-p") == 0 && i+1 < argc) {
                prompt = argv[++i];
            } else if (strcmp(argv[i], "-ctk") == 0 && i+1 < argc) {
                type_k = parse_type(argv[++i]);
            } else if (strcmp(argv[i], "-ctv") == 0 && i+1 < argc) {
                type_v = parse_type(argv[++i]);
            } else if (strcmp(argv[i], "-fa") == 0 && i+1 < argc) {
                fa = argv[++i];
            } else if (strcmp(argv[i], "-kvu") == 0) {
                kv_unified = true;
            }
        }
        if (model_path.empty()) { fprintf(stderr, "no -m\n"); return 1; }
    }

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) { fprintf(stderr, "load failed\n"); return 2; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "tokenize failed\n"); return 3;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt + n_predict + 16;
    ctx_params.n_batch = ctx_params.n_ctx;
    ctx_params.n_ubatch = ctx_params.n_ctx;
    ctx_params.type_k = type_k;
    ctx_params.type_v = type_v;
    ctx_params.kv_unified = kv_unified;
    if (strcmp(fa, "on") == 0)  ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (strcmp(fa, "off") == 0) ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (strcmp(fa, "auto") == 0) ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    ctx_params.no_perf = true;
    ctx_params.embeddings = false;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 4; }

    const int n_vocab = llama_vocab_n_tokens(vocab);

    // decode the prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int) prompt_tokens.size());
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "prompt decode failed\n"); return 5; }

    llama_token last = prompt_tokens.back();

    // greedy decode n_predict tokens
    for (int i = 0; i < n_predict; i++) {
        float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        llama_token best = 0;
        float bestv = logits[0];
        for (int v = 1; v < n_vocab; v++) {
            if (logits[v] > bestv) { bestv = logits[v]; best = (llama_token) v; }
        }
        printf("%d\n", (int) best);
        last = best;
        // next decode step with the single new token
        batch = llama_batch_get_one(&last, 1);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "step %d decode failed\n", i); return 6; }
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
