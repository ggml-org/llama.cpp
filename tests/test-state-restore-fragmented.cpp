// Test for state restore with fragmented KV cache
// This tests the fix for: https://github.com/ggml-org/llama.cpp/pull/XXXX
// The issue was that state restore required contiguous KV cache slots,
// which fails when the cache is fragmented.
//
// The fix changes find_slot(ubatch, true) to find_slot(ubatch, false)
// in state_read_meta(), allowing non-contiguous slot allocation.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <vector>
#include <cstdio>
#include <cstring>

int main(int argc, char ** argv) {
    common_params params;

    params.prompt = "The quick brown fox";
    params.sampling.seed = 1234;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // Need multiple sequences to create fragmentation
    if (params.n_parallel < 3) {
        params.n_parallel = 3;
    }

    common_init();

    if (params.n_predict < 0) {
        params.n_predict = 8;
    }

    // init
    common_init_result llama_init = common_init_from_params(params);

    llama_model * model = llama_init.model.get();
    llama_context * ctx = llama_init.context.get();

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    GGML_UNUSED(model);

    // tokenize prompt
    auto tokens = common_tokenize(ctx, params.prompt, true);

    // Step 1: Process tokens on seq 0
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        common_batch_add(batch, tokens[i], i, {0}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode seq 0\n", __func__);
        return 1;
    }

    // Step 2: Process tokens on seq 1 (to create fragmentation later)
    common_batch_clear(batch);
    for (size_t i = 0; i < tokens.size(); i++) {
        common_batch_add(batch, tokens[i], i, {1}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode seq 1\n", __func__);
        return 1;
    }

    // Step 3: Process tokens on seq 2 (to create more fragmentation)
    common_batch_clear(batch);
    for (size_t i = 0; i < tokens.size(); i++) {
        common_batch_add(batch, tokens[i], i, {2}, false);
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode seq 2\n", __func__);
        return 1;
    }

    fprintf(stderr, "%s : processed prompt on seq 0, 1, 2 (%zu tokens each)\n", __func__, tokens.size());

    // Step 4: Save state of seq 0
    std::vector<uint8_t> seq_state(llama_state_seq_get_size(ctx, 0));
    const size_t ncopy = llama_state_seq_get_data(ctx, seq_state.data(), seq_state.size(), 0);
    if (ncopy != seq_state.size()) {
        fprintf(stderr, "%s : failed to save seq 0 state\n", __func__);
        return 1;
    }
    fprintf(stderr, "%s : saved seq 0 state, %zu bytes\n", __func__, ncopy);

    // Step 5: Clear seq 1 to create a "hole" in the KV cache (fragmentation)
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, 1, -1, -1);
    fprintf(stderr, "%s : cleared seq 1 to create fragmentation\n", __func__);

    // Step 6: Clear seq 0 as well
    llama_memory_seq_rm(mem, 0, -1, -1);
    fprintf(stderr, "%s : cleared seq 0\n", __func__);

    // Now the cache has:
    // - A hole where seq 0 was (at the beginning)
    // - A hole where seq 1 was (in the middle)
    // - seq 2 data (at the end)
    // This creates fragmentation - there's no contiguous block large enough
    // for the seq 0 state if we only look for contiguous slots

    // Step 7: Restore seq 0 state into seq 1 (should work with non-contiguous allocation)
    // We use seq 1 since it's a valid sequence ID (0 to n_parallel-1)
    // Before the fix, this would fail with "failed to find available cells in kv cache"
    const size_t nset = llama_state_seq_set_data(ctx, seq_state.data(), seq_state.size(), 1);
    if (nset != seq_state.size()) {
        fprintf(stderr, "%s : FAILED to restore seq state into fragmented cache (got %zu, expected %zu)\n",
                __func__, nset, seq_state.size());
        fprintf(stderr, "%s : This is the bug - state restore fails with fragmented KV cache\n", __func__);
        llama_batch_free(batch);
        return 1;
    }
    fprintf(stderr, "%s : restored state into seq 1, %zu bytes\n", __func__, nset);

    // Step 8: Verify we can decode with the restored state
    // Generate one token to verify the restored state is usable
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(params.sampling.seed));

    auto next_token = llama_sampler_sample(smpl, ctx, -1);
    auto next_token_str = common_token_to_piece(ctx, next_token);

    common_batch_clear(batch);
    common_batch_add(batch, next_token, (int)tokens.size(), {1}, true);

    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode with restored state\n", __func__);
        llama_sampler_free(smpl);
        llama_batch_free(batch);
        return 1;
    }

    fprintf(stderr, "%s : successfully decoded with restored state, generated: '%s'\n", __func__, next_token_str.c_str());
    fprintf(stderr, "%s : SUCCESS - state restore works with fragmented KV cache\n", __func__);

    llama_sampler_free(smpl);
    llama_batch_free(batch);

    return 0;
}