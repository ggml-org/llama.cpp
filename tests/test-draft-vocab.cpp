// tests the reduced draft LM head (llama_context_params::n_vocab_dynamic) against the full head on a small random model

#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static const int32_t n_vocab = 40960;
static const int32_t n_base  = 32768;        // the lowest ids the reduced head always computes
static const int32_t eos     = n_vocab - 1;  // above the base, so it has to be picked on its own

static void set_tensor_data(ggml_tensor * tensor, void * /*userdata*/) {
    std::mt19937 gen(std::hash<std::string>{}(tensor->name));
    std::normal_distribution<float> dis(0.0f, 0.1f);

    std::vector<float> data(ggml_nelements(tensor));
    for (float & v : data) {
        v = dis(gen);
    }
    ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
}

static gguf_context * make_metadata() {
    gguf_context * meta = gguf_init_empty();

    gguf_set_val_str(meta, "general.architecture",                   "llama");
    gguf_set_val_u32(meta, "llama.vocab_size",                       n_vocab);
    gguf_set_val_u32(meta, "llama.context_length",                   256);
    gguf_set_val_u32(meta, "llama.embedding_length",                 64);
    gguf_set_val_u32(meta, "llama.feed_forward_length",              128);
    gguf_set_val_u32(meta, "llama.block_count",                      1);
    gguf_set_val_u32(meta, "llama.attention.head_count",             2);
    gguf_set_val_u32(meta, "llama.attention.head_count_kv",          2);
    gguf_set_val_f32(meta, "llama.attention.layer_norm_rms_epsilon", 1e-5f);

    std::vector<std::string>  tokens(n_vocab);
    std::vector<const char *> token_ptrs(n_vocab);
    std::vector<float>        scores(n_vocab, 0.0f);
    for (int32_t i = 0; i < n_vocab; ++i) {
        tokens[i]     = "tok_" + std::to_string(i);
        token_ptrs[i] = tokens[i].c_str();
    }
    gguf_set_val_str (meta, "tokenizer.ggml.model", "test");
    gguf_set_arr_str (meta, "tokenizer.ggml.tokens", token_ptrs.data(), n_vocab);
    gguf_set_arr_data(meta, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), n_vocab);
    gguf_set_val_u32 (meta, "tokenizer.ggml.eos_token_id", eos);

    return meta;
}

// decodes the same tokens as the full head and returns the number of mismatches
static int run(llama_model * model, uint32_t n_dynamic, int32_t n_expect) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 256;
    cparams.n_batch         = 64;
    cparams.n_ubatch        = 64;
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;

    llama_context * ctx_full = llama_init_from_model(model, cparams);
    cparams.n_vocab_dynamic = n_dynamic;
    llama_context * ctx_dyn  = llama_init_from_model(model, cparams);
    if (ctx_full == nullptr || ctx_dyn == nullptr) {
        fprintf(stderr, "%s: failed to create the contexts\n", __func__);
        return 1;
    }

    int n_fail = 0;

    // a batch with no outputs, a batch with several outputs and a batch with one output
    const std::vector<std::vector<bool>> steps = {
        std::vector<bool>(8, false),
        std::vector<bool>(16, true),
        { true },
    };

    llama_pos pos = 0;
    for (const auto & outputs : steps) {
        llama_batch batch = llama_batch_init((int32_t) outputs.size(), 0, 1);
        for (size_t i = 0; i < outputs.size(); ++i) {
            batch.token   [i]    = (llama_token) ((7919*pos + 13) % n_vocab);
            batch.pos     [i]    = pos++;
            batch.n_seq_id[i]    = 1;
            batch.seq_id  [i][0] = 0;
            batch.logits  [i]    = outputs[i];
        }
        batch.n_tokens = (int32_t) outputs.size();

        if (llama_decode(ctx_full, batch) != 0 || llama_decode(ctx_dyn, batch) != 0) {
            fprintf(stderr, "%s: decode failed\n", __func__);
            llama_batch_free(batch);
            return n_fail + 1;
        }

        for (size_t i = 0; i < outputs.size(); ++i) {
            if (!outputs[i]) {
                continue;
            }
            const float * full = llama_get_logits_ith(ctx_full, (int32_t) i);
            const float * dyn  = llama_get_logits_ith(ctx_dyn,  (int32_t) i);

            int32_t n_finite = 0;
            int32_t best_full = 0;
            int32_t best_dyn  = 0;
            for (int32_t t = 0; t < n_vocab; ++t) {
                if (std::isfinite(dyn[t])) {
                    n_finite++;
                    if (std::fabs(dyn[t] - full[t]) > 1e-3f*(1.0f + std::fabs(full[t]))) {
                        fprintf(stderr, "%s: K = %u, token %d: logit %f, full head %f\n", __func__, n_dynamic, t, dyn[t], full[t]);
                        n_fail++;
                    }
                } else if (t < n_base || t == eos) {
                    fprintf(stderr, "%s: K = %u, token %d is not computed\n", __func__, n_dynamic, t);
                    n_fail++;
                }
                best_full = full[t] > full[best_full] ? t : best_full;
                best_dyn  = dyn[t]  > dyn[best_dyn]   ? t : best_dyn;
            }

            if (n_finite != n_expect) {
                fprintf(stderr, "%s: K = %u: %d logits computed, expected %d\n", __func__, n_dynamic, n_finite, n_expect);
                n_fail++;
            }
            // the scorer has the full rank of this small head, so the best tokens of each output are always picked
            if (best_dyn != best_full) {
                fprintf(stderr, "%s: K = %u: best token %d, full head %d\n", __func__, n_dynamic, best_dyn, best_full);
                n_fail++;
            }
            std::vector<int32_t> top(n_vocab - n_base);
            std::iota(top.begin(), top.end(), n_base);
            const int32_t n_top = std::min<int32_t>(n_dynamic/2, (int32_t) top.size());
            std::partial_sort(top.begin(), top.begin() + n_top, top.end(), [&](int32_t a, int32_t b) { return full[a] > full[b]; });
            for (int32_t k = 0; k < n_top; ++k) {
                if (!std::isfinite(dyn[top[k]])) {
                    fprintf(stderr, "%s: K = %u: token %d, rank %d among the dynamic ids, is not picked\n", __func__, n_dynamic, top[k], k);
                    n_fail++;
                    break;
                }
            }
        }

        llama_batch_free(batch);
    }

    llama_free(ctx_dyn);
    llama_free(ctx_full);

    return n_fail;
}

int main(void) {
    llama_backend_init();

    gguf_context * meta = make_metadata();
    llama_model  * model = llama_model_init_from_user(meta, set_tensor_data, nullptr, llama_model_default_params());
    if (model == nullptr) {
        fprintf(stderr, "%s: failed to create the model\n", __func__);
        return 1;
    }

    int n_fail = 0;
    n_fail += run(model, 1024,                n_base + 1024);
    n_fail += run(model, n_vocab - n_base,    n_vocab);
    n_fail += run(model, n_vocab - n_base + 1, n_vocab); // too many dynamic rows, falls back to the full head

    llama_model_free(model);
    gguf_free(meta);
    llama_backend_free();

    printf("%s: %s\n", __func__, n_fail == 0 ? "OK" : "FAILED");

    return n_fail == 0 ? 0 : 1;
}
