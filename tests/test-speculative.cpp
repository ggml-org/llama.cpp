#include "speculative.h"
#include "testing.h"

#include <iostream>

static common_params_speculative ngram_params() {
    common_params_speculative params;
    params.types = {COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V};
    params.ngram_map_k4v.size_n = 2;
    params.ngram_map_k4v.size_m = 3;
    params.ngram_map_k4v.min_hits = 1;
    return params;
}

static void draft(common_speculative * spec, const llama_tokens & history, llama_tokens & result) {
    auto & request = common_speculative_get_draft_params(spec, 0);
    request.drafting = true;
    request.n_max = -1;
    request.pos0 = llama_pos(history.size());
    request.id_last = 20;
    request.prompt = &history;
    request.result = &result;
    common_speculative_draft(spec);
}

int main(int argc, char ** argv) {
    testing t;
    if (argc == 2) t.set_filter(argv[1]);

    t.test("empty_statistics", [&](testing & t) {
        t.assert_true(common_speculative_get_stats(nullptr).empty());
        t.assert_equal(COMMON_SPECULATIVE_TYPE_NONE, common_speculative_get_last_type(nullptr, 0));
        common_params_speculative params;
        common_speculative_ptr spec(common_speculative_init(params, 1));
        t.assert_true(!spec);
    });

    t.test("ngram_source_and_acceptance_statistics", [&](testing & t) {
        auto params = ngram_params();
        common_speculative_ptr spec(common_speculative_init(params, 1));
        if (!t.assert_true(bool(spec))) return;
        const auto before = common_speculative_get_stats(spec.get());
        if (!t.assert_equal(size_t(1), before.size())) return;
        t.assert_equal(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, before[0].type);
        t.assert_equal(size_t(0), before[0].n_call_draft);
        t.assert_equal(COMMON_SPECULATIVE_TYPE_NONE, common_speculative_get_last_type(spec.get(), 0));
        t.assert_equal(COMMON_SPECULATIVE_TYPE_NONE, common_speculative_get_last_type(spec.get(), -1));
        t.assert_equal(COMMON_SPECULATIVE_TYPE_NONE, common_speculative_get_last_type(spec.get(), 1));

        const llama_tokens history{99, 10, 20, 30, 40, 50, 98, 10};
        common_speculative_begin(spec.get(), 0, history);
        llama_tokens result;
        draft(spec.get(), history, result);
        t.assert_true(result == llama_tokens({30, 40, 50}));
        t.assert_equal(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, common_speculative_get_last_type(spec.get(), 0));
        auto stats = common_speculative_get_stats(spec.get());
        t.assert_equal(size_t(1), stats[0].n_call_begin);
        t.assert_equal(size_t(1), stats[0].n_call_draft);
        t.assert_equal(size_t(1), stats[0].n_gen_drafts);
        t.assert_equal(size_t(3), stats[0].n_gen_tokens);
        t.assert_equal(size_t(0), stats[0].n_predict);
        t.assert_equal(size_t(0), before[0].n_call_draft);

        common_speculative_accept(spec.get(), 0, 2);
        stats = common_speculative_get_stats(spec.get());
        t.assert_equal(size_t(1), stats[0].n_call_accept);
        t.assert_equal(size_t(1), stats[0].n_acc_drafts);
        t.assert_equal(size_t(2), stats[0].n_acc_tokens);
        t.assert_equal(size_t(0), stats[0].n_predict);
        common_speculative_draft(spec.get());
        t.assert_equal(size_t(1), common_speculative_get_stats(spec.get())[0].n_call_draft);
    });

    t.test("empty_lookup_is_not_a_generated_draft", [&](testing & t) {
        auto params = ngram_params();
        common_speculative_ptr spec(common_speculative_init(params, 1));
        const llama_tokens history{10};
        common_speculative_begin(spec.get(), 0, history);
        llama_tokens result;
        draft(spec.get(), history, result);
        t.assert_true(result.empty());
        const auto stats = common_speculative_get_stats(spec.get());
        t.assert_equal(size_t(1), stats[0].n_call_draft);
        t.assert_equal(size_t(0), stats[0].n_gen_drafts);
        t.assert_equal(size_t(0), stats[0].n_gen_tokens);
        t.assert_equal(COMMON_SPECULATIVE_TYPE_NONE, common_speculative_get_last_type(spec.get(), 0));
    });

    t.test("earlier_empty_implementation_allows_ngram", [&](testing & t) {
        auto params = ngram_params();
        params.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        params.ngram_simple.size_n = 32;
        params.ngram_simple.size_m = 3;
        common_speculative_ptr spec(common_speculative_init(params, 1));
        const llama_tokens history{99, 10, 20, 30, 40, 50, 98, 10};
        common_speculative_begin(spec.get(), 0, history);
        llama_tokens result;
        draft(spec.get(), history, result);
        const auto stats = common_speculative_get_stats(spec.get());
        if (!t.assert_equal(size_t(2), stats.size())) return;
        t.assert_equal(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, stats[0].type);
        t.assert_equal(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, stats[1].type);
        t.assert_equal(size_t(1), stats[0].n_call_draft);
        t.assert_equal(size_t(0), stats[0].n_gen_drafts);
        t.assert_equal(size_t(1), stats[1].n_call_draft);
        t.assert_equal(size_t(1), stats[1].n_gen_drafts);
        t.assert_equal(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, common_speculative_get_last_type(spec.get(), 0));
    });

    t.test("selected_implementation_stops_lower_priority_lookup", [&](testing & t) {
        auto params = ngram_params();
        params.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        params.ngram_map_k = params.ngram_map_k4v;
        common_speculative_ptr spec(common_speculative_init(params, 1));
        const llama_tokens history{99, 10, 20, 30, 40, 50, 98, 10};
        common_speculative_begin(spec.get(), 0, history);
        llama_tokens result;
        draft(spec.get(), history, result);
        const auto stats = common_speculative_get_stats(spec.get());
        if (!t.assert_equal(size_t(2), stats.size())) return;
        t.assert_equal(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, stats[0].type);
        t.assert_equal(size_t(1), stats[0].n_gen_drafts);
        t.assert_equal(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, stats[1].type);
        t.assert_equal(size_t(0), stats[1].n_call_draft);
        t.assert_equal(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, common_speculative_get_last_type(spec.get(), 0));
    });

    return t.summary();
}
