#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static bool decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens,
                           llama_seq_id seq_id, llama_pos start_pos) {
    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], start_pos + (llama_pos) i, {seq_id}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;
    bool ok = (llama_decode(ctx, batch) == 0);
    llama_batch_free(batch);
    return ok;
}

// Relocate [src, src+m) of seq 1 to dst two ways and compare the next-token logits:
//   reference = in-place llama_memory_seq_add (llama.cpp's trusted position shift)
//   primitive = get_range save + set_range restore at dst
// Both go through the same K re-anchoring, so the logits must match bit-for-bit. A from-scratch
// decode at dst is NOT a valid oracle: the K-shift is not bit-exact versus a fresh rope, so on
// models with near-tied logits the greedy token can differ even though the relocation is correct.
static bool relocation_matches_seq_add(llama_context * ctx, const llama_model * model,
                                       const std::vector<llama_token> & ctx_tokens,
                                       const std::vector<llama_token> & pred_tok,
                                       llama_pos src, llama_pos dst, int m) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto * mem = llama_get_memory(ctx);

    if (!decode_tokens(ctx, ctx_tokens, 1, src)) {
        return false;
    }
    llama_memory_seq_add(mem, 1, src, src + m, dst - src);
    if (!decode_tokens(ctx, pred_tok, 1, dst + m)) {
        return false;
    }
    std::vector<float> ref(llama_get_logits_ith(ctx, -1), llama_get_logits_ith(ctx, -1) + n_vocab);
    llama_memory_seq_rm(mem, 1, -1, -1);

    if (!decode_tokens(ctx, ctx_tokens, 1, src)) {
        return false;
    }
    size_t bs = llama_state_seq_get_range(ctx, 1, src, src + m, nullptr, 0);
    if (bs == 0) {
        return false;
    }
    std::vector<uint8_t> buf(bs);
    if (llama_state_seq_get_range(ctx, 1, src, src + m, buf.data(), buf.size()) != bs) {
        return false;
    }
    llama_memory_seq_rm(mem, 1, -1, -1);
    if (!llama_state_seq_set_range(ctx, 1, buf.data(), buf.size(), dst)) {
        return false;
    }
    if (!decode_tokens(ctx, pred_tok, 1, dst + m)) {
        return false;
    }
    std::vector<float> got(llama_get_logits_ith(ctx, -1), llama_get_logits_ith(ctx, -1) + n_vocab);
    llama_memory_seq_rm(mem, 1, -1, -1);

    float max_diff = 0.0f;
    for (int i = 0; i < n_vocab; ++i) {
        max_diff = std::max(max_diff, std::fabs(ref[i] - got[i]));
    }
    return max_diff == 0.0f;
}

int main(int argc, char ** argv) {
    common_params params;
    params.sampling.seed = 42;
    params.n_ctx         = 512;
    params.n_parallel    = 2;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr init = common_init_from_params(params);
    llama_model   * model = init->model();
    llama_context * ctx   = init->context();
    if (!model || !ctx) {
        fprintf(stderr, "%s: failed to init\n", __func__);
        return 1;
    }

    const char * prompt = "The quick brown fox jumps over the lazy dog.";
    std::vector<llama_token> tokens = common_tokenize(ctx, prompt, true, false);
    if (tokens.size() < 8) {
        fprintf(stderr, "%s: prompt tokenized to fewer than 8 tokens\n", __func__);
        return 1;
    }

    const int      n    = (int) tokens.size();
    const int      m    = n - 1;
    const int      HALF = n / 2;
    const llama_pos SHIFT = 128;

    if (SHIFT + n >= params.n_ctx) {
        fprintf(stderr, "%s: SHIFT + n exceeds n_ctx\n", __func__);
        return 1;
    }

    std::vector<llama_token> ctx_tokens(tokens.begin(), tokens.begin() + m);

    // Test 1: same-position round-trip, get_range then set_range at the original position.
    // After restore, a decode of tokens[m] must yield the same greedy token as seq 0.
    {
        const llama_seq_id seq_ref  = 0;
        const llama_seq_id seq_test = 1;

        if (!decode_tokens(ctx, ctx_tokens, seq_ref, 0) ||
            !decode_tokens(ctx, ctx_tokens, seq_test, 0)) {
            fprintf(stderr, "%s: test 1: context decode failed\n", __func__);
            return 1;
        }

        // Decode the prediction token into seq_ref to establish the oracle.
        std::vector<llama_token> pred_tok = { tokens[m] };
        if (!decode_tokens(ctx, pred_tok, seq_ref, m)) {
            fprintf(stderr, "%s: test 1: ref decode failed\n", __func__);
            return 1;
        }

        auto sparams = llama_sampler_chain_default_params();
        llama_sampler * smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        llama_token ref_tok = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_free(smpl);

        size_t buf_size = llama_state_seq_get_range(ctx, seq_test, 0, m, nullptr, 0);
        if (buf_size == 0) {
            fprintf(stderr, "%s: test 1: get_range size query returned 0\n", __func__);
            return 1;
        }

        std::vector<uint8_t> buf(buf_size);
        if (llama_state_seq_get_range(ctx, seq_test, 0, m, buf.data(), buf.size()) != buf_size) {
            fprintf(stderr, "%s: test 1: get_range wrote wrong byte count\n", __func__);
            return 1;
        }

        llama_memory_seq_rm(llama_get_memory(ctx), seq_test, -1, -1);

        if (!llama_state_seq_set_range(ctx, seq_test, buf.data(), buf.size(), 0)) {
            fprintf(stderr, "%s: test 1: set_range failed\n", __func__);
            return 1;
        }

        if (!decode_tokens(ctx, pred_tok, seq_test, m)) {
            fprintf(stderr, "%s: test 1: test decode failed\n", __func__);
            return 1;
        }

        sparams = llama_sampler_chain_default_params();
        smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        llama_token test_tok = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_free(smpl);

        if (test_tok != ref_tok) {
            fprintf(stderr, "%s: FAILED test 1: token mismatch (%d != %d)\n", __func__, test_tok, ref_tok);
            return 1;
        }
        fprintf(stderr, "%s: PASSED test 1\n", __func__);

        llama_memory_seq_rm(llama_get_memory(ctx), seq_ref,  -1, -1);
        llama_memory_seq_rm(llama_get_memory(ctx), seq_test, -1, -1);
    }

    // Test 2: positive relocation. Save [0, m), restore at SHIFT, and require the result to
    // match the in-place seq_add shift to SHIFT bit-for-bit (see relocation_matches_seq_add).
    {
        std::vector<llama_token> pred_tok = { tokens[m] };
        if (!relocation_matches_seq_add(ctx, model, ctx_tokens, pred_tok, 0, SHIFT, m)) {
            fprintf(stderr, "%s: FAILED test 2: relocation does not match seq_add shift\n", __func__);
            return 1;
        }
        fprintf(stderr, "%s: PASSED test 2\n", __func__);
    }

    // Test 3: partial-range restore, save and restore only the tail while keeping the head.
    // Confirms that set_range clears only [new_p0, new_p0+n_cells) and leaves [0, HALF) intact.
    {
        const llama_seq_id seq_ref  = 0;
        const llama_seq_id seq_test = 1;

        std::vector<llama_token> pred_tok = { tokens[m] };

        if (!decode_tokens(ctx, ctx_tokens, seq_ref, 0) ||
            !decode_tokens(ctx, ctx_tokens, seq_test, 0)) {
            fprintf(stderr, "%s: test 3: context decode failed\n", __func__);
            return 1;
        }

        if (!decode_tokens(ctx, pred_tok, seq_ref, m)) {
            fprintf(stderr, "%s: test 3: ref decode failed\n", __func__);
            return 1;
        }

        auto sparams = llama_sampler_chain_default_params();
        llama_sampler * smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        llama_token ref_tok = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_free(smpl);

        // Save tail [HALF, m) from seq_test; remove only the tail; restore at HALF (delta=0).
        size_t buf_size = llama_state_seq_get_range(ctx, seq_test, HALF, m, nullptr, 0);
        if (buf_size == 0) {
            fprintf(stderr, "%s: test 3: get_range returned 0\n", __func__);
            return 1;
        }

        std::vector<uint8_t> buf(buf_size);
        if (llama_state_seq_get_range(ctx, seq_test, HALF, m, buf.data(), buf.size()) != buf_size) {
            fprintf(stderr, "%s: test 3: get_range wrote wrong byte count\n", __func__);
            return 1;
        }

        // Remove only the tail; head [0, HALF) stays.
        llama_memory_seq_rm(llama_get_memory(ctx), seq_test, HALF, m);

        if (!llama_state_seq_set_range(ctx, seq_test, buf.data(), buf.size(), HALF)) {
            fprintf(stderr, "%s: test 3: set_range failed\n", __func__);
            return 1;
        }

        if (!decode_tokens(ctx, pred_tok, seq_test, m)) {
            fprintf(stderr, "%s: test 3: test decode failed\n", __func__);
            return 1;
        }

        sparams = llama_sampler_chain_default_params();
        smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        llama_token test_tok = llama_sampler_sample(smpl, ctx, -1);
        llama_sampler_free(smpl);

        if (test_tok != ref_tok) {
            fprintf(stderr, "%s: FAILED test 3: token mismatch after partial restore (%d != %d)\n", __func__, test_tok, ref_tok);
            return 1;
        }
        fprintf(stderr, "%s: PASSED test 3\n", __func__);

        llama_memory_seq_rm(llama_get_memory(ctx), seq_ref,  -1, -1);
        llama_memory_seq_rm(llama_get_memory(ctx), seq_test, -1, -1);
    }

    // Test 4: negative relocation delta, restore at an earlier position than the source.
    {
        std::vector<llama_token> pred_tok = { tokens[m] };
        if (!relocation_matches_seq_add(ctx, model, ctx_tokens, pred_tok, SHIFT, 0, m)) {
            fprintf(stderr, "%s: FAILED test 4: relocation does not match seq_add shift\n", __func__);
            return 1;
        }
        fprintf(stderr, "%s: PASSED test 4\n", __func__);
    }

    // Test 5: offset-source relocation, neither source nor destination at position 0.
    {
        std::vector<llama_token> pred_tok = { tokens[m] };
        if (!relocation_matches_seq_add(ctx, model, ctx_tokens, pred_tok, SHIFT, 2 * SHIFT, m)) {
            fprintf(stderr, "%s: FAILED test 5: relocation does not match seq_add shift\n", __func__);
            return 1;
        }
        fprintf(stderr, "%s: PASSED test 5\n", __func__);
    }

    fprintf(stderr, "%s: all tests PASSED\n", __func__);
    return 0;
}
