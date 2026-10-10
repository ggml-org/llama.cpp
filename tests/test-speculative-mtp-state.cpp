#include "arg.h"
#include "common.h"
#include "log.h"
#include "speculative.h"

#include <cstdint>
#include <cstring>
#include <vector>

static bool test_state(common_params params) {
    params.speculative.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    params.speculative.draft.n_max = 3;
    params.speculative.draft.n_min = 0;
    params.speculative.draft.p_min = 0.0f;
    params.speculative.draft.probabilistic = false;
    params.n_parallel = 2;
    params.kv_unified = true;
    params.prompt = "The capital of France is";

    auto target = common_init_from_params(params);
    auto * model = target->model();
    auto * ctx_tgt = target->context();
    if (!model || !ctx_tgt) {
        return false;
    }
    auto draft = common_speculative_init_from_params(params, model, ctx_tgt);
    auto * ctx_dft = draft->context();
    if (!ctx_dft) {
        return false;
    }
    params.speculative.draft.ctx_tgt = ctx_tgt;
    params.speculative.draft.ctx_dft = ctx_dft;
    common_speculative_ptr spec(common_speculative_init(params.speculative, 2));
    if (!spec) {
        return false;
    }

    const auto prompt = common_tokenize(ctx_tgt, params.prompt, true, true);
    common_batch batch(ctx_tgt);
    for (llama_seq_id seq = 0; seq < 2; ++seq) {
        for (size_t i = 0; i < prompt.size(); ++i) {
            batch.add(prompt[i], i, seq, true);
        }
    }
    if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0 || !common_speculative_process(spec.get(), batch)) {
        return false;
    }

    std::vector<uint8_t> state, other;
    if (!common_speculative_get_state(spec.get(), 0, state) || state.empty() || !common_speculative_get_state(spec.get(), 1, other)) {
        LOG_ERR("MTP hidden state was not saved\n");
        return false;
    }
    auto roundtrip = [&](const std::vector<uint8_t> & expected) {
        std::vector<uint8_t> actual;
        return common_speculative_get_state(spec.get(), 0, actual) && actual == expected;
    };

    std::vector<uint8_t> invalid = state;
    invalid.pop_back();
    common_speculative_set_state(spec.get(), 0, invalid);
    if (!roundtrip(state)) {
        return false;
    }
    invalid = state;
    const int32_t bad_rows = -1;
    std::memcpy(invalid.data() + sizeof(uint32_t) + sizeof(int32_t), &bad_rows, sizeof(bad_rows));
    common_speculative_set_state(spec.get(), 0, invalid);
    if (!roundtrip(state)) {
        return false;
    }

    const auto save_kv = [&](llama_context * ctx) {
        std::vector<uint8_t> data(llama_state_seq_get_size(ctx, 0));
        if (llama_state_seq_get_data(ctx, data.data(), data.size(), 0) != data.size()) {
            data.clear();
        }
        return data;
    };
    const auto target_kv = save_kv(ctx_tgt);
    const auto draft_kv = save_kv(ctx_dft);
    if (target_kv.empty() || draft_kv.empty()) {
        return false;
    }
    const auto restore_kv = [&]() {
        return llama_state_seq_set_data(ctx_tgt, target_kv.data(), target_kv.size(), 0) == target_kv.size() &&
               llama_state_seq_set_data(ctx_dft, draft_kv.data(), draft_kv.size(), 0) == draft_kv.size();
    };
    const auto get_draft = [&]() {
        llama_tokens result;
        auto & dp = common_speculative_get_draft_params(spec.get(), 0);
        dp.drafting = true;
        dp.pos0 = prompt.size();
        dp.id_last = prompt.back();
        dp.prompt = &prompt;
        dp.result = &result;
        common_speculative_draft(spec.get());
        dp = {};
        return result;
    };
    const auto expected = get_draft();
    common_speculative_accept(spec.get(), 0, 0);
    if (expected.empty() || roundtrip(state)) {
        LOG_ERR("accept did not select a different hidden row\n");
        return false;
    }
    std::vector<uint8_t> accepted;
    if (!common_speculative_get_state(spec.get(), 0, accepted)) {
        return false;
    }
    if (!restore_kv()) {
        return false;
    }
    batch.clear();
    batch.add(prompt.back(), prompt.size(), 0, true);
    if (llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0 || !common_speculative_process(spec.get(), batch) || roundtrip(state)) {
        LOG_ERR("MTP verification state was not replaced\n");
        return false;
    }
    if (!restore_kv()) {
        return false;
    }
    common_speculative_set_state(spec.get(), 0, state);
    if (!roundtrip(state) || get_draft() != expected) {
        LOG_ERR("draft changed after checkpoint restore\n");
        return false;
    }

    common_speculative_set_state(spec.get(), 0, state);
    common_speculative_accept(spec.get(), 0, 0);
    if (!roundtrip(accepted) || !common_speculative_get_state(spec.get(), 1, invalid) || invalid != other) {
        LOG_ERR("verification rows or the other sequence changed\n");
        return false;
    }
    LOG_INF("MTP hidden-state checkpoint: PASS\n");
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_ctx = 256;
    params.n_batch = params.n_ubatch = 64;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }
    common_init();
    llama_backend_init();
    const bool ok = test_state(params);
    llama_backend_free();
    return ok ? 0 : 1;
}
