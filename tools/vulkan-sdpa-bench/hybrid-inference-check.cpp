#include "llama.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    const char * model_path = argc > 1 ? argv[1] : "C:\\AI\\models\\Ornith-1.5-9B-AD-Q4_K-IQ4_XS.gguf";

    llama_backend_init();
    ggml_backend_load_all();

    ggml_backend_dev_t device = ggml_backend_dev_by_name("Vulkan1");
    if (!device) {
        std::fprintf(stderr, "Vulkan1 device not found\n");
        llama_backend_free();
        return 2;
    }

    ggml_backend_dev_t devices[] = { device, nullptr };
    llama_model_params model_params = llama_model_default_params();
    model_params.devices = devices;
    model_params.n_gpu_layers = 99;
    model_params.split_mode = LLAMA_SPLIT_MODE_NONE;

    llama_model * model = llama_model_load_from_file(model_path, model_params);
    if (!model) {
        std::fprintf(stderr, "model load failed\n");
        llama_backend_free();
        return 3;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::string prompt;
    for (int i = 0; i < 70; ++i) prompt += "The sky is blue. ";
    prompt += "Answer briefly: what color is the sky?";

    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0 || n_prompt < 256) {
        std::fprintf(stderr, "prompt token count is %d; expected at least 256\n", n_prompt);
        llama_model_free(model);
        llama_backend_free();
        return 4;
    }
    std::printf("prompt_tokens=%d\n", n_prompt);

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 1024;
    context_params.n_batch = 512;
    context_params.n_ubatch = 512;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    context_params.type_k = GGML_TYPE_Q8_0;
    context_params.type_v = GGML_TYPE_Q8_0;
    llama_context * context = llama_init_from_model(model, context_params);
    if (!context) {
        std::fprintf(stderr, "context init failed\n");
        llama_model_free(model);
        llama_backend_free();
        return 5;
    }

    llama_batch batch = llama_batch_init(static_cast<int32_t>(prompt_tokens.size()), 0, 1);
    batch.n_tokens = static_cast<int32_t>(prompt_tokens.size());
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = prompt_tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i == batch.n_tokens - 1;
    }

    bool finite = llama_decode(context, batch) == 0;
    const int n_vocab = llama_vocab_n_tokens(vocab);
    for (int i = 0; finite && i < n_vocab; ++i) finite = std::isfinite(llama_get_logits_ith(context, -1)[i]);

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    llama_token token = finite ? llama_sampler_sample(sampler, context, -1) : LLAMA_TOKEN_NULL;
    llama_batch_free(batch);

    for (int step = 0; finite && step < 8; ++step) {
        char piece[128];
        const int piece_size = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
        std::printf("%d: %d %.*s\n", step + 1, token, piece_size > 0 ? piece_size : 0, piece_size > 0 ? piece : "");
        batch = llama_batch_get_one(&token, 1);
        finite = llama_decode(context, batch) == 0;
        const float * logits = finite ? llama_get_logits_ith(context, -1) : nullptr;
        for (int i = 0; finite && i < n_vocab; ++i) finite = std::isfinite(logits[i]);
        if (finite) token = llama_sampler_sample(sampler, context, -1);
    }

    std::printf("%s finite_logits=%s logits_checked=%lld\n",
            finite ? "PASS" : "FAIL", finite ? "yes" : "no",
            static_cast<long long>(n_vocab) * 9);
    llama_sampler_free(sampler);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return finite ? 0 : 6;
}
