#include "get-model.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct batch_owner {
    llama_batch batch;

    batch_owner(int32_t n_tokens, int32_t n_seq_max) : batch(llama_batch_init(n_tokens, 0, n_seq_max)) {}

    ~batch_owner() { llama_batch_free(batch); }

    batch_owner(const batch_owner &)             = delete;
    batch_owner & operator=(const batch_owner &) = delete;
};

bool expect(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", message);
    }
    return condition;
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const char * text) {
    const int32_t required = -llama_tokenize(vocab, text, std::strlen(text), nullptr, 0, true, true);
    if (required <= 0) {
        return {};
    }

    std::vector<llama_token> tokens(required);
    const int32_t written = llama_tokenize(vocab, text, std::strlen(text), tokens.data(), tokens.size(), true, true);
    if (written != required) {
        return {};
    }
    return tokens;
}

bool decode_and_check(llama_context *                  ctx,
                      const std::vector<llama_token> & tokens,
                      size_t                           begin,
                      size_t                           end,
                      std::vector<float> &             manual_sum) {
    batch_owner owner(static_cast<int32_t>(end - begin), 1);
    owner.batch.n_tokens = static_cast<int32_t>(end - begin);
    for (size_t i = begin; i < end; ++i) {
        const size_t row           = i - begin;
        owner.batch.token[row]     = tokens[i];
        owner.batch.pos[row]       = static_cast<llama_pos>(i);
        owner.batch.n_seq_id[row]  = 1;
        owner.batch.seq_id[row][0] = 0;
        owner.batch.logits[row]    = 1;
    }

    if (llama_decode(ctx, owner.batch) != 0) {
        return expect(false, "llama_decode failed");
    }

    const size_t dimension = manual_sum.size();
    for (size_t row = 0; row < end - begin; ++row) {
        const float * embedding = llama_get_embeddings_ith(ctx, static_cast<int32_t>(row));
        if (!expect(embedding != nullptr, "missing token embedding")) {
            return false;
        }
        for (size_t component = 0; component < dimension; ++component) {
            manual_sum[component] += embedding[component];
        }
    }

    if (!expect(llama_pooling_seq_get_count(ctx, 0) == end, "cumulative token count drifted")) {
        return false;
    }

    const float * mean = llama_get_embeddings_seq(ctx, 0);
    if (!expect(mean != nullptr, "missing cumulative sequence embedding")) {
        return false;
    }
    for (size_t component = 0; component < dimension; ++component) {
        const float expected = manual_sum[component] / static_cast<float>(end);
        if (!expect(mean[component] == expected, "cumulative mean differs from exact FP32 sum/count")) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers       = 0;
    llama_model * model             = llama_model_load_from_file(get_model_or_exit(argc, argv), model_params);
    if (!expect(model != nullptr, "failed to load model")) {
        return EXIT_FAILURE;
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx                = 128;
    context_params.n_batch              = 64;
    context_params.n_ubatch             = 64;
    context_params.n_seq_max            = 2;
    context_params.embeddings           = true;
    context_params.pooling_type         = LLAMA_POOLING_TYPE_MEAN_CUMULATIVE;
    context_params.attention_type       = LLAMA_ATTENTION_TYPE_CAUSAL;
    llama_context * ctx                 = llama_init_from_model(model, context_params);
    if (!expect(ctx != nullptr, "failed to create cumulative pooling context")) {
        llama_model_free(model);
        llama_backend_free();
        return EXIT_FAILURE;
    }

    const std::vector<llama_token> tokens =
        tokenize(llama_model_get_vocab(model), "Cumulative masked mean pooling must remain exact across decode calls.");
    if (!expect(tokens.size() >= 4, "test prompt tokenized too short")) {
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return EXIT_FAILURE;
    }

    bool               ok    = true;
    const size_t       split = tokens.size() / 2;
    std::vector<float> manual_sum(llama_model_n_embd_out(model), 0.0f);

    ok                                      = ok && decode_and_check(ctx, tokens, 0, split, manual_sum);
    const std::vector<float> checkpoint_sum = manual_sum;

    std::vector<uint8_t> checkpoint(llama_pooling_seq_get_size(ctx, 0));
    if (!ok || !expect(!checkpoint.empty(), "pooling checkpoint is empty") ||
        !expect(llama_pooling_seq_get_data(ctx, checkpoint.data(), checkpoint.size(), 0) == checkpoint.size(),
                "failed to snapshot cumulative pooling state")) {
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return EXIT_FAILURE;
    }

    ok = ok && decode_and_check(ctx, tokens, split, tokens.size(), manual_sum);

    ok = ok && expect(llama_pooling_seq_set_data(ctx, checkpoint.data(), checkpoint.size(), 1) == checkpoint.size(),
                      "failed to restore cumulative pooling state");
    ok = ok && expect(llama_pooling_seq_get_count(ctx, 1) == split, "restored cumulative count drifted");

    const float * restored = llama_get_embeddings_seq(ctx, 1);
    ok                     = ok && expect(restored != nullptr, "restored mean is missing");
    if (restored != nullptr) {
        for (size_t component = 0; component < manual_sum.size(); ++component) {
            const float expected = checkpoint_sum[component] / static_cast<float>(split);
            ok                   = ok && expect(restored[component] == expected, "restored cumulative mean drifted");
        }
    }

    const uint64_t restored_count = llama_pooling_seq_get_count(ctx, 1);
    checkpoint[0] ^= 0xff;
    ok = ok && expect(llama_pooling_seq_set_data(ctx, checkpoint.data(), checkpoint.size(), 1) == 0,
                      "corrupted cumulative pooling state was accepted");
    ok = ok && expect(llama_pooling_seq_get_count(ctx, 1) == restored_count,
                      "failed restore mutated existing cumulative state");

    llama_pooling_seq_rm(ctx, 0);
    ok = ok && expect(llama_pooling_seq_get_count(ctx, 0) == 0, "pooling state removal did not reset the count");
    ok = ok && expect(llama_get_embeddings_seq(ctx, 0) == nullptr, "pooling state removal left a stale mean");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
