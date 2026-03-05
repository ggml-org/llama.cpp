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

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Taken from test-llama-arch.cpp
static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    std::hash<std::string> hasher;
    std::mt19937 gen(hasher(tensor->name) + *(const size_t *) userdata);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

// Taken from test-llama-arch.cpp
static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

// Taken from test-llama-arch.cpp
static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, const size_t seed) {
    llama_model_params model_params = llama_model_default_params();
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0; // will be set from model
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;

    size_t tmp = seed;
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx, set_tensor_data, &tmp, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static bool compare_logits(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    const float threshold = 2e-5f; // Relaxed threshold for state save/load numerical precision
    for (size_t i = 0; i < a.size(); i++) {
        if (std::abs(a[i] - b[i]) > threshold) {
            return false;
        }
    }
    return true;
}

static std::vector<float> get_logits_from_context(llama_context * lctx) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(lctx)));
    std::vector<float> logits;
    logits.reserve(n_vocab);
    const float * logits_ith = llama_get_logits_ith(lctx, -1);
    for (int j = 0; j < n_vocab; j++) {
        logits.push_back(logits_ith[j]);
    }
    return logits;
}

static bool test_save_and_load_state(const gguf_context_ptr & gguf_ctx, int seed) {
    const int key_idx = gguf_find_key(gguf_ctx.get(), "general.architecture");
    const char * arch_name = (key_idx == -1) ? "unknown" : gguf_get_val_str(gguf_ctx.get(), key_idx);

    const int vocab_key_idx = gguf_find_key(gguf_ctx.get(), "llama.vocab_size");
    const uint32_t n_vocab = (vocab_key_idx == -1) ? 128 : gguf_get_val_u32(gguf_ctx.get(), vocab_key_idx);

    const std::vector<llama_token> session_tokens = get_tokens(16, n_vocab, seed);
    const char * session_file = "test_session.tmp";
    int n_ctx = 0;
    bool ok = true;
    try {
        auto model_and_ctx = get_model_and_ctx(gguf_ctx.get(), seed);
        llama_model * model = model_and_ctx.first.get();

        std::vector<float> logits1;
        std::vector<float> logits2;
        // Decode a few tokens and save the session state.
        {
            llama_context * ctx = model_and_ctx.second.get();
            llama_batch batch = llama_batch_init(session_tokens.size(), 0, 1);
            for (size_t i = 0; i < session_tokens.size(); ++i) {
                common_batch_add(batch, session_tokens[i], i, {0}, i == (session_tokens.size() -1));
            }

            if (llama_decode(ctx, batch) != 0) {
                throw std::runtime_error("llama_decode failed");
            }
            llama_batch_free(batch);

            logits1 = get_logits_from_context(ctx);
            n_ctx = llama_n_ctx(ctx);

            llama_state_save_file(ctx, session_file, session_tokens.data(), session_tokens.size());
        }
        // Create a new llama_context and load and restore the session state
        {
            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx = n_ctx;
            ctx_params.n_threads = 4;
            ctx_params.n_threads_batch = 4;
            llama_context * ctx = llama_init_from_model(model, ctx_params);

            std::vector<llama_token> loaded_tokens(session_tokens.size());
            size_t n_loaded_tokens = 0;
            if (llama_state_load_file(ctx, session_file, loaded_tokens.data(), loaded_tokens.size(), &n_loaded_tokens) != 1) {
                throw std::runtime_error("llama_state_load_file failed");
            }

            if (n_loaded_tokens != session_tokens.size()) {
                throw std::runtime_error("loaded incorrect number of tokens");
            }

            loaded_tokens.resize(n_loaded_tokens);
            if (loaded_tokens != session_tokens) {
                throw std::runtime_error("loaded session tokens do not match");
            }

            llama_memory_t mem = llama_get_memory(ctx);
            fprintf(stderr, "Before replay: KV cache seq 0 max pos = %d\n", llama_memory_seq_pos_max(mem, 0));

            if (!common_replay_last_token(ctx, loaded_tokens.back(), n_loaded_tokens)) {
                throw std::runtime_error("failed to replay last token");
            }

            fprintf(stderr, "After replay:  KV cache seq 0 max pos = %d\n", llama_memory_seq_pos_max(mem, 0));

            logits2 = get_logits_from_context(ctx);

            // Verify we can continue decoding after load
            llama_token next_token = get_tokens(1, n_vocab, seed + 100)[0];
            llama_batch batch = llama_batch_init(1, 0, 1);
            common_batch_add(batch, next_token, n_loaded_tokens + 1, {0}, true);
            if (llama_decode(ctx, batch) != 0) {
                llama_batch_free(batch);
                throw std::runtime_error("failed to decode next token after load");
            }
            llama_batch_free(batch);

            llama_free(ctx);
        }
        // Verify the logits from the original and the restore session state.
        if (!compare_logits(logits1, logits2)) {
            ok = false;
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "Exception during test for %s: %s\n", arch_name, e.what());
        ok = false;
    }
    std::remove(session_file);
    fprintf(stderr, "Test save_load_state for arch '%s': %s\n", arch_name, ok ? "PASSED" : "FAILED");
    return ok;
}

static gguf_context_ptr transformer_model() {
    const llm_arch arch = LLM_ARCH_LLAMA;

    gguf_context_ptr ret{gguf_init_empty()};
    llama_model_saver ms{arch, ret.get()};

    const uint32_t n_ctx       = 128;
    const uint32_t n_vocab     = 50;
    const uint32_t n_embd      = 256;
    const uint32_t n_head      = 2;
    const uint32_t n_ff        = 384;
    const uint32_t n_layer     = 2;
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,        llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                  n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,              n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,            n_embd);
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,         n_ff);
    ms.add_kv(LLM_KV_BLOCK_COUNT,                 n_layer);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,        n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,     n_head);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,        n_embd_head);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,             "no_vocab");

    return ret;
}

static gguf_context_ptr recurrent_model() {
    const llm_arch arch = LLM_ARCH_MAMBA;

    gguf_context_ptr ret{gguf_init_empty()};
    llama_model_saver ms{arch, ret.get()};

    const uint32_t n_ctx   = 128;
    const uint32_t n_vocab = 128;
    const uint32_t n_embd  = 256;
    const uint32_t n_layer = 2;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,        llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                  n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,              n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,            n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,                 n_layer);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,             uint32_t(4));
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,              2 * n_embd);
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,              uint32_t(16));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,          n_embd / 16);
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,             "no_vocab");

    return ret;
}

static gguf_context_ptr hybrid_model() {
    const llm_arch arch = LLM_ARCH_JAMBA;

    gguf_context_ptr ret{gguf_init_empty()};
    llama_model_saver ms{arch, ret.get()};

    const uint32_t n_ctx   = 128;
    const uint32_t n_vocab = 128;
    const uint32_t n_embd  = 256;
    const uint32_t n_head  = 2;
    const uint32_t n_ff    = 384;
    const uint32_t n_layer = 4;
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,        llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                  n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,              n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,            n_embd);
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,         n_ff);
    ms.add_kv(LLM_KV_BLOCK_COUNT,                 n_layer);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,        n_embd_head);
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,             "no_vocab");

    std::vector<uint32_t> n_head_per_layer;
    n_head_per_layer.reserve(n_layer);
    for (uint32_t il = 0; il < n_layer; il++) {
        n_head_per_layer.push_back(il == 1 ? 0 : n_head);
    }
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,    n_head_per_layer);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);

    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,      uint32_t(4));
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,       2 * n_embd);
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,       uint32_t(16));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,   n_embd / 16);

    return ret;
}

static int test_save_load_models(const size_t seed) {
    std::vector<gguf_context_ptr> models_to_test;
    // Add more models to test here.
    models_to_test.push_back(transformer_model());
    models_to_test.push_back(recurrent_model());
    models_to_test.push_back(hybrid_model());

    bool all_ok = true;
    for (const gguf_context_ptr & gguf_ctx : models_to_test) {
        all_ok = all_ok && test_save_and_load_state(gguf_ctx, seed);
    }
    return all_ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    common_init();
    std::random_device rd;

    size_t seed = rd();
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                return 1;
            }
        }
    }

    try {
        return test_save_load_models(seed);
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
