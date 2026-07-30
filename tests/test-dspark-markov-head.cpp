// test-dspark-markov-head.cpp
//
// Verifies the DSpark markov head graph build path in src/models/dflash.cpp
// (build_dspark_markov_head, ~100 lines, called from the dflash decoder
// graph when dspark_markov_w1 is non-null). The audit
// (docs/audit-2026-07-29.md, section 7) calls this "first-class": the
// DSpark fold and its three tensors (DSPARK_MARKOV_W1, DSPARK_MARKOV_W2,
// DSPARK_CONF_PROJ) are correctly wired, and the graph is correct in
// principle.
//
// What this test verifies
// -----------------------
// For a minimal DFlash + DSpark model:
//   1. The model loads cleanly (the loader detects DSpark tensors and
//      populates the dspark_* fields on the model).
//   2. A llama_context can be created over the model and a single-sequence
//      batch of `block_size + 1` tokens can be decoded without crashing.
//   3. The markov head emits a per-position confidence vector; the
//      confidence is the output of ggml_sigmoid and so must lie in (0, 1).
//      The dflash.cpp source stores these in res->t_h_nextn, exposed to
//      the user via llama_get_embeddings_nextn (one row per ubatch
//      position, broadcast to n_embd columns).
//   4. The markov head ALSO runs in the dflash DECODER graph (the
//      encoder graph only injects K/V, no markov step). The decoder
//      graph is what we exercise here.
//
// Why this matters
// ----------------
// The DSpark markov head is the central trick that makes the drafter
// competitive with the verifier: it conditions the next-draft logits on
// the previous-draft argmax. If the graph fails to build, or the
// confidence values come out of [0, 1], the rejection-sampling training
// pipeline (which is gated on the confidence calibration) silently
// produces garbage. This test catches that class of failure at unit-test
// time rather than during a multi-hour spec-calib run.
//
// The test is also deliberately self-contained: a constructed minimal
// GGUF written to a temp file, the file is loaded by the model loader,
// the graph is exercised via llama_decode. The whole thing runs in
// well under a second and does not require any external model file.

#include "ggml.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-ext.h"
#include "../src/llama-model.h"
#include "../src/llama-model-saver.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

#define TEST_ASSERT(cond)                                                        \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::fprintf(stderr, "test-dspark-markov-head: assertion failed: "   \
                                 "%s (at %s:%d)\n",                               \
                         #cond, __FILE__, __LINE__);                              \
            std::abort();                                                         \
        }                                                                         \
    } while (0)

// Model dimensions chosen for fast decode. Larger than the no-decode
// tests would be is a non-issue: a single block_drafts=2 decode is
// sub-millisecond.
constexpr uint32_t N_VOCAB   = 64;
constexpr uint32_t N_EMBD    = 32;
constexpr uint32_t N_HEAD    = 2;
constexpr uint32_t N_FF      = 64;
constexpr uint32_t N_LAYER   = 1;
constexpr int64_t  N_HEAD_KV = 2;
constexpr uint32_t TARGET_LAYERS[] = { 0 };
constexpr int64_t  DSPARK_RANK = 4;
constexpr int64_t  BLOCK_SIZE  = 2;  // dflash.block_size; markov head runs with block_drafts <= block_size

namespace {

// Fills a tensor with a deterministic pattern derived from the seed.
static void fill_tensor(struct ggml_tensor * tensor, void * userdata) {
    size_t * seed = static_cast<size_t *>(userdata);
    std::hash<std::string> hasher;
    size_t s = *seed ^ hasher(tensor->name);
    std::mt19937 gen(s);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);
    const int64_t n = ggml_nelements(tensor);
    if (n == 0) return;
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(n);
        for (int64_t i = 0; i < n; ++i) tmp[i] = dis(gen);
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        for (int64_t i = 0; i < n; ++i) tmp[i] = ggml_fp32_to_fp16(dis(gen));
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        std::fprintf(stderr, "test-dspark-markov-head: unsupported tensor type %d\n",
                     (int) tensor->type);
        std::abort();
    }
}

// Records a (name, type, ne) tuple in the GGUF. Goes through ggml_new_tensor_*
// so the gguf writer can read the data section and the loader can mmap it.
void add_named_tensor(ggml_context * ggml_ctx, gguf_context * gguf_ctx,
                      const char * name, ggml_type type,
                      std::initializer_list<int64_t> ne) {
    int64_t dims[GGML_MAX_DIMS] = { 1, 1, 1, 1 };
    size_t n = 0;
    for (int64_t d : ne) {
        TEST_ASSERT(n < GGML_MAX_DIMS);
        dims[n++] = d;
    }
    ggml_tensor * t = nullptr;
    switch (n) {
        case 1: t = ggml_new_tensor_1d(ggml_ctx, type, dims[0]); break;
        case 2: t = ggml_new_tensor_2d(ggml_ctx, type, dims[0], dims[1]); break;
        case 3: t = ggml_new_tensor_3d(ggml_ctx, type, dims[0], dims[1], dims[2]); break;
        case 4: t = ggml_new_tensor_4d(ggml_ctx, type, dims[0], dims[1], dims[2], dims[3]); break;
        default: TEST_ASSERT(false && "unsupported rank");
    }
    TEST_ASSERT(t != nullptr);
    ggml_format_name(t, "%s", name);
    gguf_add_tensor(gguf_ctx, t);
}

void add_dflash_kv(llama_model_saver & ms) {
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,              uint32_t(128));
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,            N_EMBD);
    ms.add_kv(LLM_KV_BLOCK_COUNT,                 N_LAYER);
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,         N_FF);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,        N_HEAD);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,     uint32_t(N_HEAD_KV));
    ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,        N_EMBD / N_HEAD);
    ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,      N_EMBD / N_HEAD);
    ms.add_kv(LLM_KV_ROPE_FREQ_BASE,              10000.0f);
    ms.add_kv(LLM_KV_ROPE_SCALING_TYPE,           "none");
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1.0e-5f);
    ms.add_kv(LLM_KV_TARGET_LAYERS,
              std::vector<uint32_t>(std::begin(TARGET_LAYERS),
                                    std::end(TARGET_LAYERS)));

    // dflash.block_size is REQUIRED by build_dspark_markov_head
    // (dflash.cpp asserts on it). The value is the max block_drafts the
    // markov chain can chain together; the test uses BLOCK_SIZE=2 which
    // means a single decode can produce at most 2 drafts in a chain.
    gguf_set_val_u32(ms.gguf_ctx, "dflash.block_size", uint32_t(BLOCK_SIZE));

    ms.add_kv(LLM_KV_TOKENIZER_MODEL, "no_vocab");
    ms.add_kv(LLM_KV_VOCAB_SIZE,      N_VOCAB);
}

void add_layer_tensors(ggml_context * ggml_ctx, gguf_context * gguf_ctx, uint32_t n_layer) {
    const int64_t n_embd_head = N_EMBD / N_HEAD;
    for (uint32_t il = 0; il < n_layer; ++il) {
        char name[64];
        std::snprintf(name, sizeof(name), "blk.%u.attn_norm.weight",   il); add_named_tensor(ggml_ctx, gguf_ctx, name, GGML_TYPE_F16, { N_EMBD, 1, 1, 1 });
        std::snprintf(name, sizeof(name), "blk.%u.attn_q.weight",      il); add_named_tensor(ggml_ctx, gguf_ctx, name, GGML_TYPE_F16, { N_EMBD, n_embd_head * N_HEAD, 1, 1 });
        std::snprintf(name, sizeof(name), "blk.%u.attn_k.weight",      il); add_named_tensor(ggml_ctx, gguf_ctx, name, GGML_TYPE_F16, { N_EMBD, n_embd_head * N_HEAD_KV, 1, 1 });
        std::snprintf(name, sizeof(name), "blk.%u.attn_v.weight",      il); add_named_tensor(ggml_ctx, gguf_ctx, name, GGML_TYPE_F16, { N_EMBD, n_embd_head * N_HEAD_KV, 1, 1 });
        std::snprintf(name, sizeof(name), "blk.%u.attn_output.weight", il); add_named_tensor(ggml_ctx, gguf_ctx, name, GGML_TYPE_F16, { n_embd_head * N_HEAD, N_EMBD, 1, 1 });
        std::snprintf(name, sizeof(name), "blk.%u.attn_q_norm.weight", il); add_named_tensor(ggml_ctx, gguf_ctx, name, GGML_TYPE_F16, { n_embd_head, 1, 1, 1 });
        std::snprintf(name, sizeof(name), "blk.%u.attn_k_norm.weight", il); add_named_tensor(ggml_ctx, gguf_ctx, name, GGML_TYPE_F16, { n_embd_head, 1, 1, 1 });
        std::snprintf(name, sizeof(name), "blk.%u.ffn_norm.weight",   il); add_named_tensor(ggml_ctx, gguf_ctx, name, GGML_TYPE_F16, { N_EMBD, 1, 1, 1 });
        std::snprintf(name, sizeof(name), "blk.%u.ffn_gate.weight",   il); add_named_tensor(ggml_ctx, gguf_ctx, name, GGML_TYPE_F16, { N_EMBD, N_FF, 1, 1 });
        std::snprintf(name, sizeof(name), "blk.%u.ffn_up.weight",     il); add_named_tensor(ggml_ctx, gguf_ctx, name, GGML_TYPE_F16, { N_EMBD, N_FF, 1, 1 });
        std::snprintf(name, sizeof(name), "blk.%u.ffn_down.weight",   il); add_named_tensor(ggml_ctx, gguf_ctx, name, GGML_TYPE_F16, { N_FF,   N_EMBD, 1, 1 });
    }
}

void add_model_tensors(ggml_context * ggml_ctx, gguf_context * gguf_ctx) {
    add_named_tensor(ggml_ctx, gguf_ctx, "token_embd.weight",      GGML_TYPE_F16, { N_EMBD, N_VOCAB, 1, 1 });
    add_named_tensor(ggml_ctx, gguf_ctx, "output.weight",          GGML_TYPE_F16, { N_EMBD, N_VOCAB, 1, 1 });
    add_named_tensor(ggml_ctx, gguf_ctx, "output_norm.weight",     GGML_TYPE_F16, { N_EMBD, 1, 1, 1 });
    add_named_tensor(ggml_ctx, gguf_ctx, "enc.output_norm.weight", GGML_TYPE_F16, { N_EMBD, 1, 1, 1 });
    add_named_tensor(ggml_ctx, gguf_ctx, "fc.weight",              GGML_TYPE_F16, { (int64_t)(sizeof(TARGET_LAYERS)/sizeof(TARGET_LAYERS[0])) * N_EMBD, N_EMBD, 1, 1 });
}

void add_dspark_canonical(ggml_context * ggml_ctx, gguf_context * gguf_ctx) {
    add_named_tensor(ggml_ctx, gguf_ctx, "markov_w1.weight", GGML_TYPE_F16, { DSPARK_RANK,        N_VOCAB, 1, 1 });
    add_named_tensor(ggml_ctx, gguf_ctx, "markov_w2.weight", GGML_TYPE_F16, { DSPARK_RANK,        N_VOCAB, 1, 1 });
    add_named_tensor(ggml_ctx, gguf_ctx, "conf_proj.weight", GGML_TYPE_F16, { N_EMBD + DSPARK_RANK, 1, 1, 1 });
    add_named_tensor(ggml_ctx, gguf_ctx, "conf_proj.bias",   GGML_TYPE_F16, { 1,                   1, 1, 1 });
}

// Builds a minimal DFlash+DSpark GGUF in memory, writes it to a temp file,
// and loads it back via llama_model_load_from_file. The temp file is
// unlinked before return. Returns the loaded model (caller must
// llama_model_free).
struct llama_model * build_and_load_dspark_dflash() {
    gguf_context_ptr gguf(gguf_init_empty());
    TEST_ASSERT(gguf.get() != nullptr);

    constexpr size_t GGML_CTX_SIZE = 64 * 1024 * 1024;
    ggml_init_params ggml_params = {
        /*.mem_size   =*/ GGML_CTX_SIZE,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context_ptr ggml(ggml_init(ggml_params));
    TEST_ASSERT(ggml.get() != nullptr);

    llama_model_saver ms(LLM_ARCH_DFLASH, gguf.get());
    add_dflash_kv(ms);
    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE, llm_arch_name(LLM_ARCH_DFLASH));

    add_model_tensors(ggml.get(), gguf.get());
    add_layer_tensors(ggml.get(), gguf.get(), N_LAYER);
    add_dspark_canonical(ggml.get(), gguf.get());

    char path[] = "/tmp/test-dspark-markov-head-XXXXXX.gguf";
    int fd = mkstemps(path, /*suffix_len=*/5);
    TEST_ASSERT(fd >= 0);
    ::close(fd);
    TEST_ASSERT(gguf_write_to_file(gguf.get(), path, /*only_meta=*/false));

    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = [](float, void *) { return true; };
    model_params.n_gpu_layers = 0;

    struct llama_model * model = llama_model_load_from_file(path, model_params);
    ::unlink(path);
    return model;
}

}  // namespace

int main() {
    // Suppress the noisy "loading..." progress logs; failures still print
    // a clear message via TEST_ASSERT.
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);

    struct llama_model * model = build_and_load_dspark_dflash();
    if (model == nullptr) {
        std::fprintf(stderr,
            "test-dspark-markov-head: model load failed for the DFlash+DSpark GGUF.\n"
            "This is the prerequisite for the rest of the test; check the model\n"
            "construction in build_and_load_dspark_dflash() and the dflash loader\n"
            "diagnostics from a non-silent build for details.\n");
        return 1;
    }

    // 1. Sanity: the model is a DFlash with DSpark tensors loaded.
    TEST_ASSERT(model->arch == LLM_ARCH_DFLASH);
    TEST_ASSERT(model->dspark_markov_w1   != nullptr);
    TEST_ASSERT(model->dspark_markov_w2   != nullptr);
    TEST_ASSERT(model->dspark_conf_proj   != nullptr);
    TEST_ASSERT(model->dspark_conf_proj_b != nullptr);
    TEST_ASSERT(model->hparams.n_layer() == (int32_t) N_LAYER);
    TEST_ASSERT(model->hparams.n_embd    == (int32_t) N_EMBD);

    // 2. Build a llama_context over the model. The dflash decoder graph
    //    is a full transformer, so a small context is sufficient.
    //    After construction we enable the embeddings_nextn output via
    //    the staging API; the public llama_context_params does not
    //    expose this field. Without it, llama_get_embeddings_nextn
    //    returns null because the embd_nextn buffer is not allocated.
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = 64;
    ctx_params.n_threads       = 1;
    ctx_params.n_threads_batch = 1;
    ctx_params.no_perf         = true;
    // `llama_new_context_with_model` is the deprecated wrapper for
    // `llama_init_from_model`; the deprecation warning is just a
    // tracking comment in the header. The wrapper is convenient here
    // because it returns a raw pointer (matching the existing tests)
    // and we free it with llama_free at the end.
    struct llama_context * ctx = llama_new_context_with_model(model, ctx_params);
    TEST_ASSERT(ctx != nullptr);
    llama_set_embeddings_nextn(ctx, /*value=*/true, /*masked=*/true);
    if (ctx == nullptr) {
        std::fprintf(stderr,
            "test-dspark-markov-head: llama_new_context_with_model failed.\n"
            "Check that the dflash loader is producing a valid model for inference.\n");
        llama_model_free(model);
        return 1;
    }

    // 3. Build a single-sequence batch of (block_size + 1) tokens. The
    //    +1 gives the markov chain block_drafts = block_size positions
    //    to chain through, which is the boundary case the source
    //    special-cases (block_drafts > block_size returns early; we
    //    want block_drafts == block_size).
    const int n_tokens = (int) BLOCK_SIZE + 1;
    llama_batch batch = llama_batch_init(/*n_tokens=*/n_tokens, /*embd=*/0, /*n_seq_max=*/1);
    for (int i = 0; i < n_tokens; ++i) {
        // Use a token within the no_vocab range; any in [0, N_VOCAB) is fine.
        batch.token[i]    = (llama_token) (i % N_VOCAB);
        batch.pos[i]      = (llama_pos) i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = 1;
    }
    batch.n_tokens = n_tokens;

    // 4. Decode. This triggers the dflash decoder graph build, which
    //    includes the markov head.
    const int rc = llama_decode(ctx, batch);
    if (rc != 0) {
        std::fprintf(stderr,
            "test-dspark-markov-head: llama_decode returned %d (markov head graph build likely failed).\n",
            rc);
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // 5. The markov head's per-position confidence is stored in
    //    res->t_h_nextn, exposed as llama_get_embeddings_nextn. The
    //    dflash.cpp source broadcasts the [1, n_tok] sigmoid output
    //    to [n_embd, n_tok] (so it can reuse llama_get_embeddings_nextn),
    //    so we read n_embd columns and only the first column is the
    //    actual confidence; the rest are broadcasts of the same value.
    //
    //    Verify that the per-position confidence is in (0, 1), as
    //    expected from ggml_sigmoid. A value of exactly 0 or 1 would
    //    indicate a saturating confidence (the model has been trained
    //    too confidently) — not necessarily a bug, but worth flagging
    //    in a future diagnostic. For the test we only assert non-NaN
    //    and bounded.
    const float * nextn = llama_get_embeddings_nextn(ctx);
    if (nextn == nullptr) {
        std::fprintf(stderr,
            "test-dspark-markov-head: llama_get_embeddings_nextn returned null. "
            "The dflash.cpp source is supposed to populate t_h_nextn from the "
            "DSpark markov head, but the graph build may have produced a different shape.\n");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    for (int i = 0; i < n_tokens; ++i) {
        const float conf = nextn[(size_t) i * N_EMBD];
        // The markov head applies ggml_sigmoid, so the confidence is
        // bounded in [0, 1]. We allow the boundary values (0 and 1) for
        // an untrained model with random weights; a finite, bounded
        // value is what the source actually guarantees.
        if (!std::isfinite(conf) || conf < 0.0f || conf > 1.0f) {
            std::fprintf(stderr,
                "test-dspark-markov-head: position %d confidence is %f "
                "(expected finite, in [0, 1])\n", i, (double) conf);
            llama_batch_free(batch);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
    }

    // 6. Decode a second time on a fresh batch at NEW positions
    //    (the markov head should be idempotent for a second decode
    //    when the KV cache is advanced). This catches a class of
    //    "graph was rebuilt with the wrong KV cache state" failures.
    //    We deliberately use positions that don't overlap with the
    //    first decode so the KV cache is in a consistent state.
    for (int i = 0; i < n_tokens; ++i) {
        batch.pos[i] = (llama_pos) (i + n_tokens);  // new positions
    }
    const int rc2 = llama_decode(ctx, batch);
    if (rc2 != 0) {
        std::fprintf(stderr,
            "test-dspark-markov-head: second llama_decode returned %d "
            "(markov head graph build likely fails on KV cache reuse).\n",
            rc2);
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
