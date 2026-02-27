#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const float * a, const float * b, size_t n) {
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < n; i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    std::hash<std::string> hasher;
    std::mt19937 gen(hasher(tensor->name) + *(const size_t *) userdata);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    GGML_ASSERT(tensor->type == GGML_TYPE_F32);
    const int64_t ne = ggml_nelements(tensor);
    std::vector<float> tmp(ne);
    for (int64_t i = 0; i < ne; i++) {
        tmp[i] = dis(gen);
    }
    ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
}

enum test_mode {
    TEST_MODE_GEN_MODEL,
    TEST_MODE_GEN_RESULTS,
    TEST_MODE_TEST_VS_DISK,
    TEST_MODE_TEST_BACKENDS,
};

static const char * test_mode_name(test_mode mode) {
    switch (mode) {
        case TEST_MODE_GEN_MODEL:
            return "gen-model";
        case TEST_MODE_GEN_RESULTS:
            return "gen-results";
        case TEST_MODE_TEST_VS_DISK:
            return "test-vs-disk";
        case TEST_MODE_TEST_BACKENDS:
            return "test-backends";
    }
    GGML_ABORT("fatal error");
}

static void usage(char ** argv) {
    printf("Usage: %s [mode] [-m/--model <path_model>] [-r/--results <path_results>]\n", argv[0]);
    printf("    valid modes:\n");
    printf("      - %s (generate a dummy model and save it to disk)\n", test_mode_name(TEST_MODE_GEN_MODEL));
    printf("      - %s (load a model from disk, calculate results, and save the results to disk)\n", test_mode_name(TEST_MODE_GEN_RESULTS));
    printf("      - %s (default, load model and results from disk, calculate results from model, compare vs. disk)\n", test_mode_name(TEST_MODE_TEST_VS_DISK));
    printf("      - %s (test wheter model results are consistent for 2+ ggml backends, model can be loaded from disk, dummy otherwise)\n", test_mode_name(TEST_MODE_TEST_BACKENDS));
}

static int gen_model(const char * path_model) {
    const uint32_t n_ctx   = 32;
    const uint32_t n_embd  = 32;
    const uint32_t n_head  = 1;
    const uint32_t n_ff    = 32;
    const uint32_t n_vocab = 32;
    const uint32_t n_layer = 1;

    gguf_context_ptr metadata(gguf_init_empty());
    gguf_set_val_str(metadata.get(), "general.architecture",                   "llama");
    gguf_set_val_u32(metadata.get(), "llama.context_length",                   n_ctx);
    gguf_set_val_u32(metadata.get(), "llama.embedding_length",                 n_embd);
    gguf_set_val_u32(metadata.get(), "llama.attention.head_count",             n_head);
    gguf_set_val_u32(metadata.get(), "llama.feed_forward_length",              n_ff);
    gguf_set_val_u32(metadata.get(), "llama.block_count",                      n_layer);
    gguf_set_val_f32(metadata.get(), "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(metadata.get(), "llama.vocab_size",                       n_vocab);

    gguf_set_val_str(metadata.get(), "tokenizer.ggml.model",  "no_vocab");

    auto add_tensor = [&](const std::string & name, int64_t ne0, int64_t ne1 = 1, int64_t ne2 = 1, int64_t ne3 = 1) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F32;
        t.ne[0] = ne0;
        t.ne[1] = ne1;
        t.ne[2] = ne2;
        t.ne[3] = ne3;
        t.nb[0] = 4;
        for (int dim = 1; dim < GGML_MAX_DIMS; dim++) {
            t.nb[dim] = t.nb[dim - 1] * t.ne[dim - 1];
        }
        ggml_set_name(&t, name.c_str());
        gguf_add_tensor(metadata.get(), &t);
    };

    add_tensor("token_embd.weight", n_embd, n_vocab);

    std::random_device rd;
    std::mt19937 gen(rd());

    size_t seed = 1234;
    llama_model_params params = llama_model_default_params();
    llama_model_ptr model(llama_model_init_from_user(metadata.get(), set_tensor_data, &seed, params));
    llama_model_save_to_file(model.get(), path_model);

    return 0;
}

static int gen_results(const char * path_model, const char * path_results) {
    llama_model_params   model_params = llama_model_default_params();
    llama_context_params ctx_params   = llama_context_default_params();
    llama_model_ptr model(llama_model_load_from_file(path_model, model_params));
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));

    const uint32_t n_vocab  = 32;
    const uint32_t n_tokens = 32;
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        common_batch_add(batch, pos, pos, {0}, true);
    }
    batch.n_tokens = n_tokens;
    if (llama_decode(lctx.get(), batch)) {
        throw std::runtime_error("failed to decode batch");
    }

    ggml_context_ptr ggml_ctx_logits;
    {
        const size_t size_logits = n_tokens*ggml_row_size(GGML_TYPE_F32, n_vocab) + ggml_tensor_overhead();
        struct ggml_init_params params = {
            /*.mem_size   =*/ size_logits,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };
        ggml_ctx_logits.reset(ggml_init(params));
    }
    ggml_tensor * logits = ggml_new_tensor_2d(ggml_ctx_logits.get(), GGML_TYPE_F32, n_vocab, n_tokens);
    ggml_set_name(logits, "logits");
    float * logits_data = ggml_get_data_f32(logits);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx.get(), i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            logits_data[i*n_vocab + j] = logits_ith[j];
        }
    }

    gguf_context_ptr gguf_ctx_logits(gguf_init_empty());
    gguf_add_tensor(gguf_ctx_logits.get(), logits);
    gguf_write_to_file(gguf_ctx_logits.get(), path_results, /*only_meta =*/ false);

    return 0;
}

static int test_vs_disk(const char * path_model, const char * path_results) {
    llama_model_params   model_params = llama_model_default_params();
    llama_context_params ctx_params   = llama_context_default_params();
    llama_model_ptr model(llama_model_load_from_file(path_model, model_params));
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));

    const uint32_t n_vocab  = 32;
    const uint32_t n_tokens = 32;
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        common_batch_add(batch, pos, pos, {0}, true);
    }
    if (llama_decode(lctx.get(), batch)) {
        throw std::runtime_error("failed to decode batch");
    }
    std::vector<float> logits_calc(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx.get(), i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            logits_calc[i*n_vocab + j] = logits_ith[j];
        }
    }

    ggml_context_ptr ggml_ctx_logits;
    gguf_context_ptr gguf_ctx_logits;
    {
        struct ggml_context * tmp;
        struct gguf_init_params gguf_params = {
            /*no_alloc =*/ false,
            /*ctx      =*/ &tmp,
        };
        gguf_ctx_logits.reset(gguf_init_from_file(path_results, gguf_params));
        ggml_ctx_logits.reset(tmp);
    }
    const int64_t tid    = gguf_find_tensor(gguf_ctx_logits.get(), "logits");
    const size_t  offset = gguf_get_data_offset(gguf_ctx_logits.get()) + gguf_get_tensor_offset(gguf_ctx_logits.get(), tid);
    const size_t  size   = gguf_get_tensor_size(gguf_ctx_logits.get(), tid);
    GGML_ASSERT(size == n_tokens*ggml_row_size(GGML_TYPE_F32, n_vocab));

    FILE * file = ggml_fopen(path_results, "rb");
    if (file == nullptr) {
        throw std::runtime_error("failed to open results file");
    }
    if (fseek(file, offset, SEEK_SET) != 0) {
        throw std::runtime_error("fseek failed");
    }
    std::vector<float> logits_disk(n_tokens*n_vocab);
    {
        const size_t ne_read = fread(logits_disk.data(), sizeof(float), logits_disk.size(), file);
        if (ne_read != logits_disk.size()) {
            throw std::runtime_error("fread failed");
        }
    }

    const double nmse_val = nmse(logits_calc.data(), logits_disk.data(), n_tokens*n_vocab);
    if (nmse_val > 1e-6) {
        printf("\033[1;31mFAIL\033[0m\n");
        return 1;
    }

    printf("\033[1;32mOK\033[0m\n");
    return 0;
}

static std::vector<float> get_logits(
        const llm_arch arch, const bool moe, const size_t seed, const std::vector<llama_token> & tokens,
        const std::vector<ggml_backend_dev_t> & devs) {
    const uint32_t n_ctx   = 128;
    const uint32_t n_embd  = arch == LLM_ARCH_GEMMA3N ? 64 : 256;
    const uint32_t n_head  = arch == LLM_ARCH_GEMMA3N ? 1 : 2;
    const uint32_t n_ff    = arch == LLM_ARCH_GEMMA3N ? 96 : 384;
    const uint32_t n_vocab = 128;
    uint32_t       n_layer = 2;
    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    }

    const uint32_t n_embd_head = n_embd / n_head;

    llama_model_saver ms(arch);
    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE, llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,           n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,       n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,     n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,          n_layer);

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false); // TODO
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f); // TODO
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head);
    }

    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS,          std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,                  "no_vocab");
    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,              n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,               n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));
        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         uint32_t(2)); // sigmoid
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
    }

    ms.add_kv(LLM_KV_SSM_INNER_SIZE,     arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE ? 64 : 2*n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,    uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,     uint32_t(32));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK, n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,    arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,  uint32_t(3));

    std::mt19937 gen(seed);

    llama_model_params model_params = llama_model_default_params();
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;

    size_t tmp = seed;
    llama_model_ptr model(llama_model_init_from_user(ms.gguf_ctx, set_tensor_data, &tmp, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }

    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    GGML_ASSERT(tokens.size() == n_ctx);
    for (uint32_t pos = 0; pos < n_ctx; pos++) {
        common_batch_add(batch, tokens[pos], pos, {0}, true);
    }
    batch.n_tokens = n_ctx;
    if (llama_decode(lctx.get(), batch)) {
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_ctx*n_vocab);
    for (uint32_t i = 0; i < n_ctx; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx.get(), i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    return ret;
}

static bool moe_mandatory(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_GROK:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_RND1:
        case LLM_ARCH_PADDLEOCR:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_STEP35:
            return true;
        default:
            return false;
    }
}

static bool moe_implemented(const llm_arch arch) {
    if (moe_mandatory(arch)) {
        return true;
    }
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_LLAMA_EMBED:
            return true;
        default:
            return false;
    }
}

static int test_backends(const size_t seed, const ggml_log_level log_level) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    const uint32_t n_ctx   = 128;
    const uint32_t n_vocab = 128;
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> tokens;
    tokens.reserve(n_ctx);
    for (uint32_t i = 0; i < n_ctx; i++) {
        tokens.push_back(dis(gen));
    }

    bool all_ok = true;
    common_log_flush(common_log_main());
    printf("|%15s|%30s|%6s|%8s|%6s|\n", "Model arch.", "Device", "Config", "NMSE", "Status");
    printf("|---------------|------------------------------|------|--------|------|\n");
    for (const llm_arch & arch : llm_arch_all()) {
        if (arch == LLM_ARCH_CLIP || arch == LLM_ARCH_GPTJ || arch == LLM_ARCH_UNKNOWN) {
            continue; // These models don't have usable implementations.
        }
        if (arch == LLM_ARCH_MPT) {
            continue; // TODO check whether mpt.cpp is correct
        }
        if (arch == LLM_ARCH_BERT || arch == LLM_ARCH_MODERN_BERT || arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE ||
                arch == LLM_ARCH_NEO_BERT || arch == LLM_ARCH_JINA_BERT_V2 || arch == LLM_ARCH_JINA_BERT_V3) {
            continue; // TODO vocab
        }
        if (arch == LLM_ARCH_MINICPM3 || arch == LLM_ARCH_DEEPSEEK2 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_PLM) {
            continue; // TODO LoRA rank
        }
        if (arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_COHERE2 || arch == LLM_ARCH_EXAONE_MOE ||
                arch == LLM_ARCH_OPENAI_MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35) {
            continue; // TODO sliding window
        }
        if (arch == LLM_ARCH_T5 || arch == LLM_ARCH_T5ENCODER) {
            continue; // TODO attention buckets
        }
        if (arch == LLM_ARCH_RWKV6 || arch == LLM_ARCH_RWKV6QWEN2 || arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7) {
            continue; // TODO RWKV
        }
        if (arch == LLM_ARCH_CHAMELEON) {
            continue; // TODO tensor shapes
        }
        if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
            continue; // TODO needs special hparams
        }
        if (arch == LLM_ARCH_APERTUS) {
            continue; // TODO xielu
        }
        if (arch == LLM_ARCH_KIMI_LINEAR) {
            continue; // TODO MLA
        }
        if (arch == LLM_ARCH_LLAMA4) {
            continue; // TODO attn_scale problems
        }
        if (arch == LLM_ARCH_AFMOE) {
            continue; // TODO segfault
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const std::vector<float> logits_cpu = get_logits(arch, moe, seed, tokens, {});
            for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    continue;
                }
                const std::vector<float> logits_dev = get_logits(arch, moe, seed, tokens, {dev});
                const double nmse_val = nmse(logits_cpu.data(), logits_dev.data(), logits_cpu.size());
                const bool ok = nmse_val <= 1e-6;
                all_ok = all_ok && ok;
                char nmse_str[10];
                snprintf(nmse_str, sizeof(nmse_str), "%.2e", nmse_val);
                printf("|%15s|%30s|%6s|%8s|%17s|\n", llm_arch_name(arch), ggml_backend_dev_description(dev),
                    moe ? "MoE" : "Dense", nmse_str, ok ? "\033[1;32mOK\033[0m" : "\033[1;31mFAIL\033[0m");
            }
        }
    }
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
    return all_ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    common_init();
    test_mode mode = TEST_MODE_TEST_BACKENDS;
    const char * path_model = nullptr;
    const char * path_results = nullptr;
    ggml_log_level log_level = GGML_LOG_LEVEL_ERROR;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (i + 1 >= argc) {
                usage(argv);
                return 1;
            }
            path_model = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--results") == 0) {
            if (i + 1 >= argc) {
                usage(argv);
                return 1;
            }
            path_results = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            log_level = GGML_LOG_LEVEL_INFO;
            continue;
        }
        for (test_mode m : {TEST_MODE_GEN_MODEL, TEST_MODE_GEN_RESULTS, TEST_MODE_TEST_VS_DISK, TEST_MODE_TEST_BACKENDS}) {
            if (strcmp(test_mode_name(m), argv[i]) == 0) {
                mode = m;
                break;
            }
        }
    }

    if (path_model == nullptr && mode != TEST_MODE_TEST_BACKENDS) {
        printf("ERROR: a model path is required\n\n");
        usage(argv);
        return 1;
    }

    if (path_results == nullptr && mode != TEST_MODE_TEST_BACKENDS && mode != TEST_MODE_GEN_MODEL) {
        printf("ERROR: a results path is required for mode %s\n\n", test_mode_name(mode));
        usage(argv);
        return 1;
    }

    try {
        switch (mode) {
            case TEST_MODE_GEN_MODEL:
                return gen_model(path_model);
            case TEST_MODE_GEN_RESULTS:
                return gen_results(path_model, path_results);
            case TEST_MODE_TEST_VS_DISK:
                return test_vs_disk(path_model, path_results);
            case TEST_MODE_TEST_BACKENDS:
                return test_backends(1234, log_level); // TODO
        }
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
