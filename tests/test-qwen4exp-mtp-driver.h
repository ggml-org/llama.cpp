#pragma once

#include "speculative.h"

// Exercise the public driver against the synthetic models from test-qwen4exp-mtp.cpp.
static std::vector<llama_tokens> driver_drafts(
        llama_model * target_model, llama_model * draft_model,
        bool chained, int n_seq, int draft_ubatch, bool adaptive, int prompt_length = 1, bool at_limit = false) {
    auto cp = llama_context_default_params();
    cp.n_ctx = 128 * n_seq;
    cp.n_batch = cp.n_ubatch = 32;
    cp.n_seq_max = n_seq;
    cp.n_outputs_max = 32;
    cp.n_threads = cp.n_threads_batch = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    llama_context_ptr target(llama_init_from_model(target_model, cp));
    require(bool(target), "driver target context");
    cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cp.n_ubatch = draft_ubatch;
    llama_context_ptr draft(llama_init_from_model(draft_model, cp));
    require(bool(draft), "driver draft context");
    if (at_limit) {
        prompt_length = int(llama_n_ctx(draft.get())) - 2;
    }

    common_params_speculative params;
    params.types = {adaptive ? COMMON_SPECULATIVE_TYPE_DRAFT_MTP_ADAPTIVE : COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
    params.draft.ctx_tgt = target.get();
    params.draft.ctx_dft = draft.get();
    params.draft.n_max = 4;
    params.draft.n_min_adaptive = 1;
    params.draft.backend_sampling = false;
    params.draft.chain = chained;
    common_speculative_ptr spec(common_speculative_init(params, n_seq));
    require(bool(spec), "initialize public MTP driver");

    std::vector<llama_tokens> all_drafts;
    const std::vector<int> expected_lengths = adaptive ? std::vector<int>{1, 1, 2, 2, 2, 2, 3, 1}
                                                     : std::vector<int>{at_limit ? 2 : 4};
    for (size_t round = 0; round < expected_lengths.size(); ++round) {
        // Rewind to an identical prompt while keeping acceptance feedback in the driver.
        // This isolates depth changes from target-model acceptance variability.
        llama_memory_clear(llama_get_memory(target.get()), true);
        llama_memory_clear(llama_get_memory(draft.get()), true);
        llama_batch batch = llama_batch_init(32, 0, 1);
        std::vector<llama_tokens> prompts(n_seq), results(n_seq);
        for (int seq = 0; seq < n_seq; ++seq) {
            prompts[seq].assign(prompt_length, llama_token(3 + seq));
        }
        for (int pos = 0; pos < prompt_length; ) {
            common_batch_clear(batch);
            const int end = std::min(prompt_length, pos + 32 / n_seq);
            for (int seq = 0; seq < n_seq; ++seq) {
                for (int i = pos; i < end; ++i) {
                    common_batch_add(batch, prompts[seq][i], i, {seq}, true);
                }
            }
            require(llama_decode(target.get(), batch) == 0, "driver target prefill");
            require(common_speculative_process(spec.get(), batch), "driver processes target hidden states");
            pos = end;
        }
        llama_batch_free(batch);

        for (int seq = 0; seq < n_seq; ++seq) {
            common_speculative_begin(spec.get(), seq, prompts[seq]);
            auto & dp = common_speculative_get_draft_params(spec.get(), seq);
            dp.drafting = true;
            dp.pos0 = prompt_length;
            dp.id_last = 5 + seq;
            dp.prompt = &prompts[seq];
            dp.result = &results[seq];
            dp.n_max = at_limit ? 2 : adaptive && round + 1 == expected_lengths.size() ? 1 : -1;
        }
        common_speculative_draft(spec.get());
        for (int seq = 0; seq < n_seq; ++seq) {
            require(results[seq].size() == size_t(expected_lengths[round]),
                    "driver respects configured/adaptive depth and remaining-context cap");
            all_drafts.push_back(results[seq]);
            common_speculative_accept(spec.get(), seq, results[seq].size());
        }
    }
    return all_drafts;
}

static void test_driver(llama_model * target, llama_model * head) {
    for (const auto & config : std::vector<std::pair<int, int>>{{1, 2}, {2, 8}}) {
        const auto sequential = driver_drafts(target, head, false, config.first, config.second, false);
        const auto chained = driver_drafts(target, head, true, config.first, config.second, false);
        require(chained == sequential, "driver fallback preserves sequential greedy drafts");
        fprintf(stderr, "PASS driver fallback sequences=%d ubatch=%d\n", config.first, config.second);
    }
    for (int ubatch : {2, 8}) {
        const auto sequential = driver_drafts(target, head, false, 1, ubatch, true);
        const auto chained = driver_drafts(target, head, true, 1, ubatch, true);
        require(chained == sequential, "adaptive driver preserves greedy drafts through depth and cap changes");
        fprintf(stderr, "PASS adaptive driver depth changes and context cap ubatch=%d\n", ubatch);
    }
    // With two catch-up rows and depth four, ubatches 4/5 must flush the prefix;
    // ubatch 6 can merge it into one chain exactly at capacity.
    for (int ubatch : {4, 5, 6}) {
        const auto sequential = driver_drafts(target, head, false, 1, ubatch, false, 2);
        const auto chained = driver_drafts(target, head, true, 1, ubatch, false, 2);
        require(chained == sequential, "catch-up admission preserves full-capacity drafts");
        fprintf(stderr, "PASS driver catch-up capacity ubatch=%d\n", ubatch);
    }
    const auto sequential = driver_drafts(target, head, false, 1, 8, false, 1, true);
    const auto chained = driver_drafts(target, head, true, 1, 8, false, 1, true);
    require(chained == sequential, "driver honors supplied cap with an occupied context near its limit");
    fprintf(stderr, "PASS driver supplied cap near occupied context limit\n");
}
