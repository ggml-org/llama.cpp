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
#include "ggml.h"
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

static llama_batch _make_batch(
    std::vector<std::vector<llama_token>>  token_seqs,
    std::vector<std::vector<llama_seq_id>> seq_ids) {
    GGML_ASSERT(token_seqs.size() == seq_ids.size());

    size_t total_tokens = 0;
    for (const auto & token_seq : token_seqs) {
        total_tokens += token_seq.size();
    }
    size_t max_seq_ids = 0;
    for (const auto & seq_ids_i : seq_ids) {
        max_seq_ids = std::max(max_seq_ids, seq_ids_i.size());
    }
    llama_batch batch = llama_batch_init(total_tokens, 0, max_seq_ids);

    for (size_t i = 0; i < token_seqs.size(); ++i) {
        const auto& token_seq = token_seqs[i];
        const auto& seq_ids_i = seq_ids[i];
        for (int pos = 0; pos < (int)token_seq.size(); ++pos) {
            common_batch_add(batch, token_seq[pos], pos, seq_ids_i, false);
        }
    }
    return batch;
}

static bool is_source_tensor(ggml_tensor * child, ggml_tensor * parent) {
    if (!child || !parent) return false;
    for (size_t i = 0; i < GGML_MAX_SRC; ++i) {
        if (child->src[i] == parent) {
            return true;
        } else if (child->src[i] != nullptr && is_source_tensor(child->src[i], parent)) {
            return true;
        }
    }
    return false;
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
        /* model     */ *model,
        /* filter    */ nullptr,
        /* type_k    */ GGML_TYPE_F32,
        /* type_v    */ GGML_TYPE_F16,
        /* v_trans   */ false,
        /* offload   */ false,
        /* kv_size   */ 10,
        /* n_seq_max */ 1,
        /* padding   */ 10,
        /* n_swa     */ 0,
        /* swa_type  */ LLAMA_SWA_TYPE_NONE
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
        /* n_seq_max */ 1,
        /* padding  */ 10,
        /* n_swa    */ 0,
        /* swa_type */ LLAMA_SWA_TYPE_NONE
    );

    // // Create the micro batch with a single 3-token sequence
    // llama_batch batch1 = _make_batch({{101, 1, 102}}, {{42}});
    // llama_sbatch sbatch1 = cache.sbatch_init(batch1, false);
    // llama_ubatch ubatch1 = cache.ubatch_next(sbatch1, 4, false);

    // // Find a slot for a new sequence
    // GGML_ASSERT(cache.find_slot(ubatch1));

    // // Cache the k/v for a single layer in this slot
    // ggml_context * ctx = ggml_init({10240, NULL, false});
    // ggml_tensor * k1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, model->hparams.n_embd_k_gqa(0));
    // ggml_tensor * v1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, model->hparams.n_embd_v_gqa(0));
    // ggml_tensor * k1_view = cache.cpy_k(ctx, k1, 0);
    // ggml_tensor * v1_view = cache.cpy_v(ctx, v1, 0);
    // GGML_ASSERT(is_source_tensor(k1_view, k1));
    // GGML_ASSERT(is_source_tensor(v1_view, v1));

    // // Create a second batch with different tokens and find a slot for it
    // llama_batch batch2 = _make_batch({{1, 2, 3, 4}}, {{5}});
    // llama_sbatch sbatch2 = cache.sbatch_init(batch2, false);
    // llama_ubatch ubatch2 = cache.ubatch_next(sbatch2, 4, false);
    // GGML_ASSERT(cache.find_slot(ubatch2));

    // // Add some different tensors
    // ggml_tensor * k2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, model->hparams.n_embd_k_gqa(0));
    // ggml_tensor * v2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, model->hparams.n_embd_v_gqa(0));
    // ggml_tensor * k2_view = cache.cpy_k(ctx, k2, 0);
    // ggml_tensor * v2_view = cache.cpy_v(ctx, v2, 0);
    // GGML_ASSERT(is_source_tensor(k2_view, k2));
    // GGML_ASSERT(is_source_tensor(v2_view, v2));

    // // Make sure first batch's k/v aren't cache hit
    // GGML_ASSERT(!is_source_tensor(k2_view, k1));
    // GGML_ASSERT(!is_source_tensor(v2_view, v1));

    // // Re-find the slot for the first batch and make sure they cache hit
    // GGML_ASSERT(cache.find_slot(ubatch1));

    // // Clean up
    // llama_batch_free(batch1);
    // llama_batch_free(batch2);
    // ggml_free(ctx);
}

/*- Recurrent Cache ----------------------------------------------------------*/

/* Test that the recurrent cache can be constructed and destructed safely */
static void test_llama_kv_cache_recurrent_constructor() {
    auto model = _make_model(LLM_ARCH_MAMBA);
    llama_kv_cache_recurrent cache(
        /* model     */ *model,
        /* filter    */ nullptr,
        /* type_k    */ GGML_TYPE_F32,
        /* type_v    */ GGML_TYPE_F16,
        /* offload   */ false,
        /* kv_size   */ 10,
        /* n_seq_max */ 1
    );
}

/*- Hybrid Cache -------------------------------------------------------------*/

/* Test that the hybrid cache can be constructed and destructed safely */
static void test_llama_kv_cache_hybrid_constructor() {
    auto model = _make_model(
        /* arch          =*/ LLM_ARCH_LLAMA,
        /* n_layer       =*/ 4,
        /* n_embd_head_k =*/ 4,
        /* n_embd_head_v =*/ 4,
        /* n_head        =*/ 0,
        /* n_head_kv     =*/ 0
    );
    auto recurrent_filter = [](int32_t il) {
        return il == 0 || il == 2;
    };
    auto unified_filter = [&recurrent_filter](int32_t il) {
        return !recurrent_filter(il);
    };
    auto& n_head_arr = model->hparams.n_head_arr;
    n_head_arr[0] = 16;
    n_head_arr[1] = 32;
    n_head_arr[2] = 16;
    n_head_arr[3] = 32;
    auto& n_head_kv_arr = model->hparams.n_head_kv_arr;
    n_head_kv_arr[0] = 16;
    n_head_kv_arr[1] = 8;
    n_head_kv_arr[2] = 16;
    n_head_kv_arr[3] = 8;

    std::unique_ptr<llama_kv_cache_unified> u_cache(
        new llama_kv_cache_unified(
            /* model     */ *model,
            /* filter    */ unified_filter,
            /* type_k    */ GGML_TYPE_F32,
            /* type_v    */ GGML_TYPE_F16,
            /* v_trans   */ false,
            /* offload   */ false,
            /* kv_size   */ 10,
            /* n_seq_max */ 1,
            /* padding   */ 10,
            /* n_swa     */ 0,
            /* swa_type  */ LLAMA_SWA_TYPE_NONE
        )
    );
    auto * u_cache_ptr = u_cache.get();
    std::unique_ptr<llama_kv_cache_recurrent> r_cache (
        new llama_kv_cache_recurrent(
            /* model     */ *model,
            /* filter    */ recurrent_filter,
            /* type_k    */ GGML_TYPE_F32,
            /* type_v    */ GGML_TYPE_F16,
            /* offload   */ false,
            /* kv_size   */ 10,
            /* n_seq_max */ 1
        )
    );
    auto * r_cache_ptr = r_cache.get();

    std::vector<llama_kv_cache_hybrid::child_cache> children;
    children.emplace_back(std::move(u_cache), std::vector<size_t>{1, 3});
    children.emplace_back(std::move(r_cache), std::vector<size_t>{0, 2});

    llama_kv_cache_hybrid cache(std::move(children));

    GGML_ASSERT(cache.get_child_cache<llama_kv_cache_unified>() == u_cache_ptr);
    GGML_ASSERT(cache.get_child_cache<llama_kv_cache_recurrent>() == r_cache_ptr);
}

/*- Main ---------------------------------------------------------------------*/

int main(int argc, char* argv[]) {
    // Unified Cache Tests
    RUN_TEST(test_llama_kv_cache_unified_constructor);
    RUN_TEST(test_llama_kv_cache_unified_single_seq);
    // Recurrent Cache Tests
    RUN_TEST(test_llama_kv_cache_recurrent_constructor);
    // Hybrid Cache Tests
    RUN_TEST(test_llama_kv_cache_hybrid_constructor);
    return 0;
}
