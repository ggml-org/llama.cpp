#include "../src/llama-arch.h"
#include "../src/llama-hparams.h"
#include "../src/llama-impl.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-model.h"

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <memory>

/*- Helpers ------------------------------------------------------------------*/

static std::shared_ptr<llama_model> _make_model() {
    llama_model_params params;
    params.tensor_buft_overrides = nullptr;
    std::shared_ptr<llama_model> model(new llama_model(params));
    model->hparams = llama_hparams();
    model->arch = LLM_ARCH_LLAMA;
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

#define LOG_SCOPE() log_scope __log_scope(__func__)

/*- Unified Cache ------------------------------------------------------------*/

/* Test that the unified cache can be constructed and destructed safely */
static void test_llama_kv_cache_unified_constructor() {
    LOG_SCOPE();
    auto model = _make_model();
    llama_kv_cache_unified cache(
        /* model   */ *model,
        /* type_k  */ GGML_TYPE_F32,
        /* type_v  */ GGML_TYPE_F16,
        /* v_trans */ false,
        /* offload */ false,
        /* kv_size */ 10,
        /* padding */ 10
    );
}

/*- Recurrent Cache ----------------------------------------------------------*/

/* Test that the recurrent cache can be constructed and destructed safely */
static void test_llama_kv_cache_recurrent_constructor() {
    LOG_SCOPE();
    auto model = _make_model();
    llama_kv_cache_recurrent cache(
        /* model   */ *model,
        /* type_k  */ GGML_TYPE_F32,
        /* type_v  */ GGML_TYPE_F16,
        /* offload */ false,
        /* kv_size */ 10
    );
}

/*- Hybrid Cache -------------------------------------------------------------*/

/* Test that the hybrid cache can be constructed and destructed safely */
static void test_llama_kv_cache_hybrid_constructor() {
    LOG_SCOPE();
    auto model = _make_model();
    model->hparams.n_layer = 4;
    model->hparams.n_embd_head_k = 4;
    model->hparams.n_embd_head_v = 4;
    auto& recurrent_layer_arr = model->hparams.recurrent_layer_arr;
    recurrent_layer_arr[0] = 1;
    recurrent_layer_arr[1] = 0;
    recurrent_layer_arr[2] = 1;
    recurrent_layer_arr[3] = 0;
    auto& n_head_kv_arr = model->hparams.n_head_kv_arr;
    n_head_kv_arr[0] = 16;
    n_head_kv_arr[1] = 8;
    n_head_kv_arr[2] = 16;
    n_head_kv_arr[3] = 8;

    std::unique_ptr<llama_kv_cache_unified> u_cache(
        new llama_kv_cache_unified(
            /* model   */ *model,
            /* type_k  */ GGML_TYPE_F32,
            /* type_v  */ GGML_TYPE_F16,
            /* v_trans */ false,
            /* offload */ false,
            /* kv_size */ 20,
            /* padding */ 2
        )
    );
    auto * u_cache_ptr = u_cache.get();
    std::unique_ptr<llama_kv_cache_recurrent> r_cache (
        new llama_kv_cache_recurrent(
            /* model   */ *model,
            /* type_k  */ GGML_TYPE_F32,
            /* type_v  */ GGML_TYPE_F16,
            /* offload */ false,
            /* kv_size */ 10
        )
    );
    auto * r_cache_ptr = r_cache.get();

    std::vector<llama_kv_cache_hybrid::child_cache> children;
    children.emplace_back(std::move(u_cache), std::vector<size_t>{1, 3});
    children.emplace_back(std::move(r_cache), std::vector<size_t>{0, 2});

    llama_kv_cache_hybrid cache(model->hparams, std::move(children));

    GGML_ASSERT(cache.get_child_cache<llama_kv_cache_unified>() == u_cache_ptr);
    GGML_ASSERT(cache.get_child_cache<llama_kv_cache_recurrent>() == r_cache_ptr);
}

/*- Main ---------------------------------------------------------------------*/

int main() {
    // Unified Cache Tests
    test_llama_kv_cache_unified_constructor();
    // Recurrent Cache Tests
    test_llama_kv_cache_recurrent_constructor();
    // Hybrid Cache Tests
    test_llama_kv_cache_hybrid_constructor();
    return 0;
}
