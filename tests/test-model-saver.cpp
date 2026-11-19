#include "llama-model.h"
#include "llama-model-saver.h"

#include <cmath>
#include <cstdio>
#include <memory>

static void check_u32(const llama_model_saver & saver, llm_kv key, const uint32_t * values, size_t count) {
    const int64_t id = gguf_find_key(saver.gguf_ctx, saver.llm_kv(key).c_str());
    GGML_ASSERT(id >= 0);
    if (count == 1) {
        GGML_ASSERT(gguf_get_kv_type(saver.gguf_ctx, id) == GGUF_TYPE_UINT32);
        GGML_ASSERT(gguf_get_val_u32(saver.gguf_ctx, id) == values[0]);
    } else {
        GGML_ASSERT(gguf_get_kv_type(saver.gguf_ctx, id) == GGUF_TYPE_ARRAY);
        GGML_ASSERT(gguf_get_arr_type(saver.gguf_ctx, id) == GGUF_TYPE_UINT32);
        GGML_ASSERT(gguf_get_arr_n(saver.gguf_ctx, id) == count);
        const auto * data = static_cast<const uint32_t *>(gguf_get_arr_data(saver.gguf_ctx, id));
        for (size_t i = 0; i < count; ++i) {
            GGML_ASSERT(data[i] == values[i]);
        }
    }
}

static void test_metadata(llm_arch arch, bool uniform, uint32_t n_nextn, bool yarn_rescaled) {
    std::unique_ptr<llama_model> model(llama_model_create(arch, llama_model_default_params()));
    auto & hp = model->hparams;
    hp.n_layer_all = 3 + n_nextn;
    hp.n_layer_nextn = n_nextn;
    hp.rope_yarn_log_mul = 1.5f;
    hp.rope_freq_scale_train = 0.25f;

    for (uint32_t il = 0; il < hp.n_layer_all; ++il) {
        const bool last = il == hp.n_layer_all - 1;
        hp.n_ff_arr[il] = uniform || !last ? 32 : 64;
        hp.n_ff_exp_arr[il] = uniform || !last ? 16 : 32;
        hp.n_expert_used_arr[il] = uniform || !last ? 2 : 1;
        hp.n_head_arr[il] = uniform || !last ? 4 : 2;
        hp.n_head_kv_arr[il] = uniform || !last ? 2 : 1;
        hp.is_recr_impl[il] = uniform || !last;
        hp.dsv4_compress_ratios[il] = uniform || !last ? 4 : 1;
        hp.rope_pattern[il] = uniform ? il < 3 : il % 2;
        hp.is_indexer_full_impl[il] = uniform ? il < 3 : il % 2;
        hp.kv_source_impl[il] = uniform ? 0 : il / 2;
    }

    llama_model_saver saver(model.get());
    saver.add_kv_from_model();
    const size_t n_all = uniform ? 1 : hp.n_layer_all;
    const size_t n_trunk = uniform ? 1 : 3;
    check_u32(saver, LLM_KV_FEED_FORWARD_LENGTH, hp.n_ff_arr.data(), n_all);
    check_u32(saver, LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hp.n_ff_exp_arr.data(), n_all);
    check_u32(saver, LLM_KV_EXPERT_USED_COUNT, hp.n_expert_used_arr.data(), n_all);
    check_u32(saver, LLM_KV_ATTENTION_HEAD_COUNT, hp.n_head_arr.data(), n_all);
    check_u32(saver, LLM_KV_ATTENTION_HEAD_COUNT_KV, hp.n_head_kv_arr.data(), n_all);
    check_u32(saver, LLM_KV_ATTENTION_RECURRENT_LAYERS, hp.is_recr_impl.data(), n_all);
    check_u32(saver, LLM_KV_ATTENTION_COMPRESS_RATIOS, hp.dsv4_compress_ratios.data(), n_all);
    check_u32(saver, LLM_KV_ATTENTION_ROPE_PATTERN, hp.rope_pattern.data(), n_trunk);
    check_u32(saver, LLM_KV_ATTENTION_INDEXER_TYPES, hp.is_indexer_full_impl.data(), n_trunk);
    check_u32(saver, LLM_KV_ATTENTION_KV_SOURCE_MAP, hp.kv_source_impl.data(), n_trunk);

    const int64_t id = gguf_find_key(saver.gguf_ctx, saver.llm_kv(LLM_KV_ROPE_SCALING_YARN_LOG_MUL).c_str());
    GGML_ASSERT(id >= 0);
    GGML_ASSERT(gguf_get_kv_type(saver.gguf_ctx, id) == GGUF_TYPE_FLOAT32);
    const float expected = yarn_rescaled ? 0.15f : 1.5f;
    GGML_ASSERT(fabsf(gguf_get_val_f32(saver.gguf_ctx, id) - expected) < 1e-6f);
}

int main() {
    const struct {
        llm_arch arch;
        bool yarn_rescaled;
    } cases[] = {
        { LLM_ARCH_GIGACHAT35, true },
        { LLM_ARCH_DEEPSEEK2,  true },
        { LLM_ARCH_DEEPSEEK32, true },
        { LLM_ARCH_MISTRAL4,   true },
        { LLM_ARCH_XING4_0,    true },
        { LLM_ARCH_QWEN3NEXT,  false },
        { LLM_ARCH_QWEN35,     false },
        { LLM_ARCH_QWEN35MOE,  false },
        { LLM_ARCH_GLM_DSA,    false },
        { LLM_ARCH_MISTRAL3,   false },
        { LLM_ARCH_MISTRAL_LARGE4, false },
    };
    for (const auto & c : cases) {
        for (bool uniform : { false, true }) {
            for (uint32_t n_nextn : { 0, 1, 2 }) {
                test_metadata(c.arch, uniform, n_nextn, c.yarn_rescaled);
            }
        }
        printf("%s: OK\n", llm_arch_name(c.arch));
    }
    return 0;
}
