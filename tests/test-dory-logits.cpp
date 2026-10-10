#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 5 && argc != 6) {
        fprintf(stderr, "usage: logits model.gguf K chunk-size output.bin [token-ids.txt]\n");
        return 2;
    }
    const int loops = std::atoi(argv[2]);
    const int chunk = std::atoi(argv[3]);
    if (chunk < 1 || chunk > 4096) { return 2; }
    std::vector<llama_token> tokens = {3, 4, 5, 6, 7, 8, 9, 10};
    if (argc == 6) {
        std::ifstream stream(argv[5]);
        if (!stream) { return 2; }
        tokens.clear();
        int64_t token;
        while (stream >> token) {
            if (token < 0 || token > INT32_MAX || tokens.size() >= 16384) { return 2; }
            tokens.push_back(llama_token(token));
        }
        if (!stream.eof() || tokens.empty()) { return 2; }
    }
    const int n_tokens = int(tokens.size());
    llama_backend_init();
    llama_model_kv_override overrides[3] = {};
    overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
    std::strcpy(overrides[0].key, "dory.recurrent_loop_count");
    overrides[0].val_i64 = loops;
    const char * cache_mode = std::getenv("DORY_TEST_KV_MODE");
    if (cache_mode) {
        if (std::strlen(cache_mode) >= sizeof(overrides[1].val_str)) { return 2; }
        overrides[1].tag = LLAMA_KV_OVERRIDE_TYPE_STR;
        std::strcpy(overrides[1].key, "dory.recurrent_kv_cache_mode");
        std::strcpy(overrides[1].val_str, cache_mode);
    }
    auto mp = llama_model_default_params();
    const char * gpu_env = std::getenv("DORY_TEST_GPU_LAYERS");
    mp.n_gpu_layers = gpu_env ? std::atoi(gpu_env) : 0;
    mp.kv_overrides = overrides;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { return 3; }
    auto cp = llama_context_default_params();
    cp.n_ctx = n_tokens + 32 > 64 ? n_tokens + 32 : 64;
    cp.n_batch = chunk;
    cp.n_ubatch = chunk;
    const char * threads_env = std::getenv("DORY_TEST_THREADS");
    const int threads = threads_env ? std::atoi(threads_env) : 1;
    if (threads < 1 || threads > 256) { return 2; }
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); return 4; }
    auto * batch = llama_batch_ext_init(ctx);
    auto * out = std::fopen(argv[4], "wb");
    if (!out) { return 5; }
    const auto nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    for (auto token : tokens) { if (token >= nv) { return 2; } }
    for (int start = 0; start < n_tokens; start += chunk) {
        llama_batch_ext_clear(batch);
        const int end = start + chunk < n_tokens ? start + chunk : n_tokens;
        for (int i = start; i < end; ++i) {
            const int idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
            const llama_pos pos = i;
            llama_batch_ext_set_pos(batch, idx, &pos);
            llama_batch_ext_set_output_logits(batch, idx, true);
        }
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) { return 6; }
        for (int i = 0; i < end - start; ++i) {
            const auto * logits = llama_get_logits_ith(ctx, i);
            if (!logits || std::fwrite(logits, sizeof(float), nv, out) != (size_t) nv) { return 7; }
        }
    }
    std::fclose(out);
    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
