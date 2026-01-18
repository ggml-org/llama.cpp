// Minimal SYCL model load + decode repro harness for host_cache crashes.
//
// Usage:
//   LLAMA_SYCL_TEST_MODEL=/path/to/model.gguf ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-sycl-model-repro

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "llama.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"

#if !defined(GGML_USE_SYCL)
int main() {
    std::fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

static const char * pick_model_path(int argc, char ** argv) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--model") == 0) {
            return argv[i + 1];
        }
    }
    const char * env = std::getenv("LLAMA_SYCL_TEST_MODEL");
    if (env && *env) {
        return env;
    }
    const char * fallback = "/Storage/GenAI/models/mistral-7b-v0.1.Q4_0.gguf";
    if (std::filesystem::exists(fallback)) {
        return fallback;
    }
    return nullptr;
}

int main(int argc, char ** argv) {
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);
    }
    setenv("GGML_SYCL_HOST_CACHE_GUARD", "1", 1);
    setenv("GGML_SYCL_WEIGHTS_EVICTABLE", "1", 1);

    if (const char * env = std::getenv("LLAMA_SYCL_TEST_UNIFIED_CACHE_PCT")) {
        const int pct = std::atoi(env);
        if (pct >= 1 && pct <= 100) {
            ggml_backend_sycl_set_unified_cache_budget_pct(pct);
        }
    }
    if (const char * env = std::getenv("LLAMA_SYCL_TEST_UNIFIED_CACHE_HOST_PCT")) {
        const int pct = std::atoi(env);
        if (pct >= 1 && pct <= 100) {
            ggml_backend_sycl_set_unified_cache_host_budget_pct(pct);
        }
    }

    const char * model_path = pick_model_path(argc, argv);
    if (!model_path || !std::filesystem::exists(model_path)) {
        std::fprintf(stderr, "SKIP: model not found (set LLAMA_SYCL_TEST_MODEL or pass --model).\n");
        return 0;
    }

    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    llama_model_params mparams = llama_model_default_params();
    mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
    mparams.main_gpu = 0;
    mparams.n_gpu_layers = 999;
    mparams.use_mmap = true;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        std::fprintf(stderr, "FAIL: failed to load model %s\n", model_path);
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 16;
    cparams.n_ubatch = 16;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "FAIL: failed to init context\n");
        llama_model_free(model);
        return 1;
    }

    const char * prompt = "1, 2, 3, 4, 5,";
    const auto * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens(256);
    int n = llama_tokenize(vocab, prompt, static_cast<int>(std::strlen(prompt)), tokens.data(),
                           static_cast<int>(tokens.size()), true, false);
    if (n < 0) {
        std::fprintf(stderr, "FAIL: tokenization failed (need %d tokens)\n", -n);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    tokens.resize(static_cast<size_t>(n));

    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        std::fprintf(stderr, "FAIL: llama_decode returned %d\n", ret);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_free(ctx);
    llama_model_free(model);
    std::printf("SYCL model repro test: PASS\n");
    return 0;
}

#endif
