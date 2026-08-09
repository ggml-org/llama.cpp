// Regression test for the DSV4 (DeepSeek-V4) compound KV cache state restore path.
//
// llama_kv_cache_dsv4::state_read() used to clear the compressed K caches of EVERY
// sequence before refilling only the sequence being restored, silently zeroing the
// compressed context of all other sequences. See:
//   https://github.com/ggml-org/llama.cpp/issues/26777
//
// This test builds a tiny synthetic deepseek4 model, decodes into two sequences,
// then restores sequence 0 from its own state blob (the minimal trigger) and asserts
// that sequence 1's serialized state is bit-identical across that restore.

#include "common.h"
#include "log.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-model-saver.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static const uint32_t N_VOCAB  = 128;
static const uint32_t N_EMBD   = 128;
static const uint32_t N_HEAD   = 2;
static const uint32_t N_LAYER  = 2;
static const uint32_t N_CTX    = 1024;
static const uint32_t N_SEQ    = 2;
static const uint32_t N_TOKENS = 256;

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
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
        GGML_ABORT("unexpected tensor type");
    }
}

// Minimal deepseek4 metadata: every key llama_model_deepseek4::load_arch_hparams reads.
static gguf_context_ptr dsv4_gguf_ctx() {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(LLM_ARCH_DEEPSEEK4, ret.get());

    const uint32_t n_embd_head = N_EMBD/N_HEAD;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(LLM_ARCH_DEEPSEEK4));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                N_VOCAB);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            N_CTX);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          N_EMBD);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           N_EMBD);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               N_LAYER);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(0));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,       uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,      N_HEAD);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,   uint32_t(1)); // DSV4 stores a single shared KV head

    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,       uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,    uint32_t(128));

    // MoE (deepseek4 is MoE-only and requires sqrt-softplus gating)
    ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));
    ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, uint32_t(64));
    ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,       1.0f);
    ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,        true);
    ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         uint32_t(4)); // sqrt-softplus
    ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,           7.0f);

    // sparse attention indexer (built for the ratio-4 layers)
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,      uint32_t(8));

    // DSV4 specifics
    ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,        uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,          uint32_t(32));
    ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,   10000.0f);
    ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,              uint32_t(4)); // build_hc_pre asserts hc == 4
    ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(3));
    ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,            1e-6f);
    ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                    uint32_t(0));

    // layer 0 feeds the CSA (+ indexer) caches, layer 1 feeds the HCA cache
    ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>({ 4, 128 }));

    ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT, n_embd_head);
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,      "no_vocab");

    return ret;
}

static bool decode_seq(llama_context * lctx, llama_seq_id seq_id, uint32_t n_tokens, size_t seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, N_VOCAB - 1);

    const uint32_t n_chunk = 64;
    for (uint32_t off = 0; off < n_tokens; off += n_chunk) {
        const uint32_t n = std::min(n_chunk, n_tokens - off);
        llama_batch batch = llama_batch_init(n, 0, 1);
        for (uint32_t i = 0; i < n; i++) {
            common_batch_add(batch, dis(gen), off + i, { seq_id }, off + i + 1 == n_tokens);
        }
        const bool ok = llama_decode(lctx, batch) == 0;
        llama_batch_free(batch);
        if (!ok) {
            return false;
        }
    }
    return true;
}

static std::vector<uint8_t> get_seq_state(llama_context * lctx, llama_seq_id seq_id) {
    const size_t size = llama_state_seq_get_size_ext(lctx, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
    std::vector<uint8_t> buf(size);
    const size_t n = llama_state_seq_get_data_ext(lctx, buf.data(), buf.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (n == 0) {
        throw std::runtime_error("llama_state_seq_get_data_ext failed");
    }
    buf.resize(n);
    return buf;
}

// Reports how the two blobs differ; returns the number of differing bytes.
static size_t report_diff(const char * what, const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    if (a.size() != b.size()) {
        printf("  %s: SIZE MISMATCH %zu -> %zu\n", what, a.size(), b.size());
        return a.size() + b.size();
    }
    size_t n_diff = 0;
    size_t n_zeroed = 0;
    size_t first = a.size();
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) {
            if (n_diff == 0) {
                first = i;
            }
            n_diff++;
            if (b[i] == 0) {
                n_zeroed++;
            }
        }
    }
    if (n_diff == 0) {
        printf("  %s: identical (%zu bytes)\n", what, a.size());
    } else {
        printf("  %s: %zu/%zu bytes differ (first at offset %zu; %zu of them became zero)\n",
                what, n_diff, a.size(), first, n_zeroed);
    }
    return n_diff;
}

int main(int argc, char ** argv) {
    GGML_UNUSED(argc);
    GGML_UNUSED(argv);

    common_init();
    ggml_backend_load_all();

    gguf_context_ptr gguf_ctx = dsv4_gguf_ctx();

    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = [](float, void *) { return true; };

    size_t seed = 1234;
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &seed, model_params));
    if (!model) {
        fprintf(stderr, "%s: failed to build synthetic deepseek4 model\n", __func__);
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = N_CTX;
    ctx_params.n_seq_max       = N_SEQ;
    ctx_params.n_batch         = 64;
    ctx_params.n_ubatch        = 64;
    ctx_params.n_threads       = 4;
    ctx_params.n_threads_batch = 4;

    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        fprintf(stderr, "%s: failed to create context\n", __func__);
        return 1;
    }

    // populate both sequences
    for (llama_seq_id s = 0; s < (llama_seq_id) N_SEQ; s++) {
        if (!decode_seq(lctx.get(), s, N_TOKENS, 1000 + s)) {
            fprintf(stderr, "%s: decode failed for seq %d\n", __func__, s);
            return 1;
        }
    }

    const std::vector<uint8_t> seq1_before = get_seq_state(lctx.get(), 1);
    const std::vector<uint8_t> seq0_saved  = get_seq_state(lctx.get(), 0);

    printf("%s: seq0 state %zu bytes, seq1 state %zu bytes\n", __func__, seq0_saved.size(), seq1_before.size());

    // The trigger: restore seq 0 from its own blob. Sequence 1 is not mentioned and
    // must not be touched.
    const size_t n_set = llama_state_seq_set_data_ext(
            lctx.get(), seq0_saved.data(), seq0_saved.size(), 0, LLAMA_STATE_SEQ_FLAGS_NONE);
    if (n_set == 0) {
        fprintf(stderr, "%s: llama_state_seq_set_data_ext failed\n", __func__);
        return 1;
    }

    const std::vector<uint8_t> seq1_after = get_seq_state(lctx.get(), 1);
    const std::vector<uint8_t> seq0_after = get_seq_state(lctx.get(), 0);

    printf("%s: after restoring seq 0 from its own state:\n", __func__);
    const size_t n_diff_seq0 = report_diff("seq0 (the restored sequence)", seq0_saved, seq0_after);
    const size_t n_diff_seq1 = report_diff("seq1 (the untouched neighbor)", seq1_before, seq1_after);

    if (n_diff_seq1 != 0) {
        fprintf(stderr, "%s: FAIL - restoring seq 0 corrupted the state of seq 1\n", __func__);
        return 1;
    }
    if (n_diff_seq0 != 0) {
        fprintf(stderr, "%s: FAIL - seq 0 did not round-trip through its own state blob\n", __func__);
        return 1;
    }

    printf("%s: OK\n", __func__);
    return 0;
}
