// tests the reduced draft LM head (llama_context_params::vocab_subset) against the full head on a small random model

#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <set>
#include <string>
#include <vector>

static const int32_t n_vocab = 40960;
static const int32_t n_embd  = 64;
static const int32_t eos     = n_vocab - 1; // not in the lists, so it has to be added on its own

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
    gguf_set_val_u32(meta, "llama.embedding_length",                 n_embd);
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

// returns the number of logits that differ from the full head; computed = ids expected to get logits, empty = all (full head fallback)
static int run(llama_model * model, const char * name, const std::vector<llama_token> & list, const std::set<int32_t> & computed) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 256;
    cparams.n_batch         = 64;
    cparams.n_ubatch        = 64;
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;

    llama_context * ctx_full = llama_init_from_model(model, cparams);
    cparams.vocab_subset   = list.data();
    cparams.n_vocab_subset = list.size();
    llama_context * ctx_list = llama_init_from_model(model, cparams);
    if (ctx_full == nullptr || ctx_list == nullptr) {
        fprintf(stderr, "%s: %s: failed to create the contexts\n", __func__, name);
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

        if (llama_decode(ctx_full, batch) != 0 || llama_decode(ctx_list, batch) != 0) {
            fprintf(stderr, "%s: %s: decode failed\n", __func__, name);
            llama_batch_free(batch);
            return n_fail + 1;
        }

        for (size_t i = 0; i < outputs.size(); ++i) {
            if (!outputs[i]) {
                continue;
            }
            const float * full = llama_get_logits_ith(ctx_full, (int32_t) i);
            const float * part = llama_get_logits_ith(ctx_list, (int32_t) i);

            for (int32_t t = 0; t < n_vocab; ++t) {
                // listed ids match the full head, the others are exactly -inf
                const bool want = computed.empty() || computed.count(t);
                const bool ok   = want ? std::fabs(part[t] - full[t]) <= 1e-3f*(1.0f + std::fabs(full[t])) : part[t] == -INFINITY;
                if (!ok) {
                    fprintf(stderr, "%s: %s: token %d (%s): logit %f, full head %f\n", __func__, name, t, want ? "listed" : "not listed", part[t], full[t]);
                    n_fail++;
                }
            }
        }

        llama_batch_free(batch);
    }

    llama_free(ctx_list);
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

    // scattered ids and a run of consecutive ids, given unsorted and with duplicates
    std::vector<llama_token> list;
    for (int32_t id = 0; id < 20000; id += 5) {
        list.push_back(id);
    }
    for (int32_t id = 30999; id >= 30000; --id) {
        list.push_back(id);
    }
    list.push_back(100);

    std::set<int32_t> computed(list.begin(), list.end());
    computed.insert(eos);

    std::vector<llama_token> bad = list;
    bad.push_back(n_vocab);

    std::vector<llama_token> all(n_vocab);
    for (int32_t id = 0; id < n_vocab; ++id) {
        all[id] = id;
    }

    int n_fail = 0;
    n_fail += run(model, "list",                list, computed);
    n_fail += run(model, "id outside the vocab", bad,  {}); // falls back to the full head
    n_fail += run(model, "every id",            all,  {}); // falls back to the full head

    llama_model_free(model);
    gguf_free(meta);
    llama_backend_free();

    printf("%s: %s\n", __func__, n_fail == 0 ? "OK" : "FAILED");

    return n_fail == 0 ? 0 : 1;
}
