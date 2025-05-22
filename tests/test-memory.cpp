/*------------------------------------------------------------------------------
 * Unit tests for llama-memory.h and derived memory implementations. It contains
 * a number of tests which can be run all together or separately.
 *
 * USAGE: ./bin/test-memory <test_name1> <test_name2>
 *
 * When adding a new test, do the following:
 *
 *   1. Add the new test_<memory_type>_description function under the
 *      appropriate memory type section
 *
 *   2. Add `RUN_TEST(test_<memory_type>_description);` to main
 *----------------------------------------------------------------------------*/

#include "../src/llama-arch.h"
#include "../src/llama-batch.h"
#include "../src/llama-hparams.h"
#include "../src/llama-impl.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-model.h"

#include "common.h"
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

/*- Unified Cache ------------------------------------------------------------*/

/* Test that the unified cache can be constructed and destructed safely */
static void test_llama_kv_cache_unified_constructor() {
    auto model = _make_model();
    llama_kv_cache_unified cache(
        /* model    */ *model,
        /* filter   */ nullptr,
        /* type_k   */ GGML_TYPE_F32,
        /* type_v   */ GGML_TYPE_F16,
        /* v_trans  */ false,
        /* offload  */ false,
        /* kv_size  */ 10,
        /* n_seq_max */ 10,
        /* padding  */ 10,
        /* n_swa    */ 0,
        /* swa_type */ LLAMA_SWA_TYPE_NONE
    );
}

/* Test that the unified cache can operate with a single seq */
static void test_llama_kv_cache_unified_single_seq() {
    auto model = _make_model();
    llama_kv_cache_unified cache(
        /* model    */ *model,
        /* filter   */ nullptr,
        /* type_k   */ GGML_TYPE_F32,
        /* type_v   */ GGML_TYPE_F16,
        /* v_trans  */ false,
        /* offload  */ false,
        /* kv_size  */ 10,
        /* n_seq_max */ 10,
        /* padding  */ 10,
        /* n_swa    */ 0,
        /* swa_type */ LLAMA_SWA_TYPE_NONE
    );
    // GGML_ASSERT(cache.get_used_cells() == 0);

    // Create the micro batch with a single 3-token sequence
    //
    // NOTE: A bunch of these asserts were just me figuring out how the batches
    //  relate to each other, but they're left for future readers to help in the
    //  same understanding process.
    llama_seq_id seq_id = 42;
    llama_batch batch = llama_batch_init(3, 0, 1);
    common_batch_add(batch, 101, 0, {seq_id}, false);
    common_batch_add(batch, 1,   1, {seq_id}, false);
    common_batch_add(batch, 102, 2, {seq_id}, false);
    llama_sbatch sbatch(batch, 0, true, false);
    GGML_ASSERT(batch.n_tokens == 3);
    GGML_ASSERT(sbatch.n_tokens == 3);
    GGML_ASSERT(!sbatch.seq.empty());
    llama_ubatch ubatch = sbatch.split_simple(4);
    printf("ubatch.n_seqs=%d\n", ubatch.n_seqs);
    GGML_ASSERT(ubatch.n_seqs == 3);
    GGML_ASSERT(ubatch.n_seq_tokens == 1);
    GGML_ASSERT(ubatch.n_tokens == 3);
    GGML_ASSERT(ubatch.seq_id[0][0] == seq_id);
    GGML_ASSERT(ubatch.seq_id[1][0] == seq_id);
    GGML_ASSERT(ubatch.seq_id[2][0] == seq_id);

    // Find a slot for a new sequence
    GGML_ASSERT(cache.find_slot(ubatch));

    // Clean up
    llama_batch_free(batch);
}

/*- Recurrent Cache ----------------------------------------------------------*/

/* Test that the recurrent cache can be constructed and destructed safely */
static void test_llama_kv_cache_recurrent_constructor() {
    auto model = _make_model(LLM_ARCH_MAMBA);
    llama_kv_cache_recurrent cache(
        /* model   */ *model,
        /* type_k  */ GGML_TYPE_F32,
        /* type_v  */ GGML_TYPE_F16,
        /* offload */ false,
        /* kv_size */ 10,
        /* n_seq_max */ 10
    );
}

/*- Main ---------------------------------------------------------------------*/

int main(int argc, char* argv[]) {
    // Unified Cache Tests
    RUN_TEST(test_llama_kv_cache_unified_constructor);
    RUN_TEST(test_llama_kv_cache_unified_single_seq);
    // Recurrent Cache Tests
    RUN_TEST(test_llama_kv_cache_recurrent_constructor);
    return 0;
}
