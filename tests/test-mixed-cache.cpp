/*------------------------------------------------------------------------------
 * Unit tests for llama-kv-cache-mixed.h and mixed KV cache implementation.
 * Comprehensive tests for mixed KV cache functionality.
 *
 * USAGE: ./bin/test-mixed-cache <test_name1> <test_name2>
 *
 * When adding a new test, do the following:
 *
 *   1. Add the new test_mixed_cache_<description> function
 *   2. Add `RUN_TEST(test_mixed_cache_<description>);` to main
 *----------------------------------------------------------------------------*/

#include "../src/llama-arch.h"
#include "../src/llama-batch.h"
#include "../src/llama-hparams.h"
#include "../src/llama-impl.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-kv-cache-mixed.h"
#include "../src/llama-model.h"

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <memory>

/*- Helpers ------------------------------------------------------------------*/

static std::shared_ptr<llama_model> _make_model(
    llm_arch arch = LLM_ARCH_LLAMA,
    uint32_t n_layer = 4,
    uint32_t n_embd_head_k = 4,
    uint32_t n_embd_head_v = 4,
    uint32_t n_head = 8,
    uint32_t n_head_kv = 2) {

    llama_model_params params;
    params.tensor_buft_overrides = nullptr;
    std::shared_ptr<llama_model> model(new llama_model(params));
    model->hparams = llama_hparams();
    model->arch = arch;

    model->hparams.n_layer = n_layer;
    model->hparams.n_embd_head_k = n_embd_head_k;
    model->hparams.n_embd_head_v = n_embd_head_v;

    // If set to 0, assume the test will fill out the array elementwise (hybrid)
    if (n_head > 0) {
        auto& n_head_arr = model->hparams.n_head_arr;
        std::fill(n_head_arr.begin(), n_head_arr.end(), n_head);
    }
    if (n_head_kv > 0) {
        auto& n_head_kv_arr = model->hparams.n_head_kv_arr;
        std::fill(n_head_kv_arr.begin(), n_head_kv_arr.end(), n_head_kv);
    }

    return model;
}

struct log_scope {
    const char * name;
    explicit log_scope(const char * name) : name(name) {
        LLAMA_LOG_INFO("--------\n");
        LLAMA_LOG_INFO("START: %s\n", name);
    }
    ~log_scope() {
        LLAMA_LOG_INFO("END: %s\n", name);
        LLAMA_LOG_INFO("--------\n");
    }
};

#define RUN_TEST(test_name)                                                \
    do {                                                                   \
        bool run_test = argc < 2;                                          \
        std::vector<std::string> args(argv + 1, argv + argc);              \
        if (std::find(args.begin(), args.end(), #test_name) != args.end()) \
            run_test = true;                                               \
        if (run_test) {                                                    \
            log_scope __log_scope(#test_name);                             \
            test_name();                                                   \
        }                                                                  \
    } while (0)

/*- Mixed Cache Tests --------------------------------------------------------*/

/* Test that the mixed cache can be constructed and destructed safely */
static void test_mixed_cache_constructor() {
    auto model = _make_model();
    
    // Create mixed cache configuration
    llama_kv_cache_mixed_config config;
    config.enable_quantization = true;
    config.quantization_threshold = 32;
    config.group_size = 16;
    config.hot_type_k = GGML_TYPE_F16;
    config.hot_type_v = GGML_TYPE_F16;
    config.cold_type_k = GGML_TYPE_Q4_0;
    config.cold_type_v = GGML_TYPE_Q4_0;
    
    llama_kv_cache_mixed cache(
        /* model    */ *model,
        /* filter   */ nullptr,
        /* v_trans  */ false,
        /* offload  */ false,
        /* kv_size  */ 10,
        /* n_seq_max */ 10,
        /* n_pad    */ 10,
        /* config   */ config
    );
}

/* Test mixed cache configuration options */
static void test_mixed_cache_config() {
    auto model = _make_model();
    
    // Test with quantization disabled
    llama_kv_cache_mixed_config config1;
    config1.enable_quantization = false;
    config1.hot_type_k = GGML_TYPE_F32;
    config1.hot_type_v = GGML_TYPE_F32;
    
    llama_kv_cache_mixed cache1(
        /* model    */ *model,
        /* filter   */ nullptr,
        /* v_trans  */ false,
        /* offload  */ false,
        /* kv_size  */ 5,
        /* n_seq_max */ 5,
        /* n_pad    */ 5,
        /* config   */ config1
    );
    
    // Test with quantization enabled
    llama_kv_cache_mixed_config config2;
    config2.enable_quantization = true;
    config2.quantization_threshold = 16;
    config2.group_size = 8;
    config2.hot_type_k = GGML_TYPE_F16;
    config2.hot_type_v = GGML_TYPE_F16;
    config2.cold_type_k = GGML_TYPE_Q4_0;
    config2.cold_type_v = GGML_TYPE_Q4_0;
    
    llama_kv_cache_mixed cache2(
        /* model    */ *model,
        /* filter   */ nullptr,
        /* v_trans  */ false,
        /* offload  */ false,
        /* kv_size  */ 20,
        /* n_seq_max */ 10,
        /* n_pad    */ 10,
        /* config   */ config2
    );
}

/* Test mixed cache quantization behavior */
static void test_mixed_cache_quantization() {
    auto model = _make_model();
    
    llama_kv_cache_mixed_config config;
    config.enable_quantization = true;
    config.quantization_threshold = 4; // Small threshold for testing
    config.fp16_window_size = 2;       // Keep only 2 tokens in FP16
    config.group_size = 2;             // Quantize in groups of 2
    config.hot_type_k = GGML_TYPE_F16;
    config.hot_type_v = GGML_TYPE_F16;
    config.cold_type_k = GGML_TYPE_Q4_0;
    config.cold_type_v = GGML_TYPE_Q4_0;
    
    llama_kv_cache_mixed cache(
        /* model    */ *model,
        /* filter   */ nullptr,
        /* v_trans  */ false,
        /* offload  */ false,
        /* kv_size  */ 20,
        /* n_seq_max */ 10,
        /* n_pad    */ 10,
        /* config   */ config
    );

    // Test quantization threshold behavior
    // Test with layer 0
    int32_t layer_id = 0;
    
    // Initially, should not quantize (no tokens yet)
    // GGML_ASSERT(!cache.do_quant(layer_id));
    
    // Get initial debug info
    printf("Initial state - Head: %u, Used: %u\n", cache.get_head(), cache.get_used());
    
    // Test basic quantization state
    // printf("Layer %d quantization needed: %s\n", layer_id, cache.do_quant(layer_id) ? "true" : "false");
}

/* Test memory usage information */
static void test_mixed_cache_memory_info() {
    auto model = _make_model();
    
    llama_kv_cache_mixed_config config;
    config.enable_quantization = true;
    config.quantization_threshold = 16;
    config.hot_type_k = GGML_TYPE_F16;
    config.hot_type_v = GGML_TYPE_F16;
    config.cold_type_k = GGML_TYPE_Q4_0;
    config.cold_type_v = GGML_TYPE_Q4_0;
    
    llama_kv_cache_mixed cache(
        /* model    */ *model,
        /* filter   */ nullptr,
        /* v_trans  */ false,
        /* offload  */ false,
        /* kv_size  */ 50,
        /* n_seq_max */ 10,
        /* n_pad    */ 10,
        /* config   */ config
    );

    // Test basic cache properties
    printf("Cache size: %u, Cache head: %u, Cache used: %u\n",
           cache.get_size(), cache.get_head(), cache.get_used());
}

/*- Main ---------------------------------------------------------------------*/

int main(int argc, char* argv[]) {
    // Mixed Cache Tests
    RUN_TEST(test_mixed_cache_constructor);
    RUN_TEST(test_mixed_cache_config);
    RUN_TEST(test_mixed_cache_quantization);
    RUN_TEST(test_mixed_cache_memory_info);
    return 0;
} 