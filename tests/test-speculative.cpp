#include "speculative.h"
#include "ngram-map.h"
#include "testing.h"

#include <algorithm>
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

static llama_tokens continuation(int value, size_t length = 3) {
    llama_tokens result;
    for (size_t i = 0; i < length; ++i) result.push_back(llama_token(30 + 10*i + value));
    return result;
}

static llama_tokens history_for(const std::vector<int> & values, size_t length = 3) {
    llama_tokens result{99};
    for (auto value : values) {
        result.insert(result.end(), {10, 20});
        const auto next = continuation(value, length);
        result.insert(result.end(), next.begin(), next.end());
        result.push_back(99);
    }
    result.insert(result.end(), {98, 10});
    return result;
}

static llama_tokens lookup(common_ngram_map & map, const llama_tokens & history) {
    llama_tokens result;
    common_ngram_map_draft(map, history, 20, result);
    return result;
}

static common_params streamed_dflash_params(int predictions, uint16_t ngram) {
    common_params params;
    params.n_batch = params.n_ubatch = 256;
    params.n_parallel = 1;
    params.fit_params = false;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.shared_device_memory_bytes = size_t(1024)*1024*1024;
    params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V};
    params.speculative.draft.mparams.path = "matching-dflash.gguf";
    params.speculative.draft.n_max = predictions;
    params.speculative.ngram_map_k4v.size_m = ngram;
    return params;
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

    t.test("k4v_emits_selected_continuation_and_attributes_acceptance", [&](testing & t) {
        for (int selected = 0; selected < COMMON_NGRAM_MAX_VALUES; ++selected) {
            std::vector<int> values{0, 1, 2, 3};
            values.insert(values.end(), 12, selected);
            values.push_back((selected + 1) % COMMON_NGRAM_MAX_VALUES);
            const auto history = history_for(values);
            common_ngram_map map(2, 3, false, 1);
            common_ngram_map_begin(map, history);
            t.assert_true("selected continuation must not come from the latest different occurrence",
                    lookup(map, history) == continuation(selected));
            if (!t.assert_equal(size_t(1), map.keys.size())) return;
            t.assert_equal(uint16_t(selected), map.last_draft_value_idx);
            common_ngram_map_accept(map, 2);
            for (int slot = 0; slot < COMMON_NGRAM_MAX_VALUES; ++slot) {
                t.assert_equal(int16_t(slot == selected ? 2 : 3), map.keys[0].values[slot].n_accepted);
            }
            auto expected = continuation(selected);
            expected.resize(2);
            t.assert_true(lookup(map, history) == expected);
        }
    });

    t.test("k4v_counts_each_history_occurrence_once", [&](testing & t) {
        const auto history = history_for({0, 0});
        common_ngram_map map(2, 3, false, 1);
        common_ngram_map_begin(map, history);
        for (int repeat = 0; repeat < 4; ++repeat) {
            t.assert_true(lookup(map, history) == continuation(0));
            if (!t.assert_equal(size_t(1), map.keys.size())) return;
            t.assert_equal(uint16_t(2), map.keys[0].values[0].value_num);
        }
    });

    t.test("k4v_updates_new_occurrences_beyond_cached_match", [&](testing & t) {
        auto history = history_for({0, 0});
        common_ngram_map map(2, 3, false, 1);
        common_ngram_map_begin(map, history);
        t.assert_true(lookup(map, history) == continuation(0));
        history.insert(history.end(), {20, 31, 41, 51, 99, 10});
        t.assert_true(lookup(map, history) == continuation(0));
        if (!t.assert_equal(size_t(1), map.keys.size())) return;
        t.assert_equal(uint16_t(2), map.keys[0].values[0].value_num);
        t.assert_equal(uint16_t(1), map.keys[0].values[1].value_num);
        t.assert_true(lookup(map, history) == continuation(0));
        t.assert_equal(uint16_t(2), map.keys[0].values[0].value_num);
        t.assert_equal(uint16_t(1), map.keys[0].values[1].value_num);
    });

    t.test("k4v_new_dominant_value_keeps_its_own_acceptance_limit", [&](testing & t) {
        auto history = history_for({0, 0});
        common_ngram_map map(2, 3, false, 1);
        common_ngram_map_begin(map, history);
        t.assert_true(lookup(map, history) == continuation(0));
        common_ngram_map_accept(map, 1);
        for (int occurrence = 0; occurrence < 5; ++occurrence) {
            history.insert(history.end(), {20, 31, 41, 51, 99, 10});
            lookup(map, history);
        }
        t.assert_true(lookup(map, history) == continuation(1));
        if (!t.assert_equal(size_t(1), map.keys.size())) return;
        t.assert_equal(int16_t(1), map.keys[0].values[0].n_accepted);
        t.assert_equal(int16_t(3), map.keys[0].values[1].n_accepted);
        t.assert_equal(uint16_t(2), map.keys[0].values[0].value_num);
        t.assert_equal(uint16_t(5), map.keys[0].values[1].value_num);
    });

    t.test("k4v_frequency_boundary_and_ambiguous_abstention", [&](testing & t) {
        for (const auto & values : {std::vector<int>{0, 0, 1}, std::vector<int>{0, 1}}) {
            const auto history = history_for(values);
            common_ngram_map map(2, 3, false, 1);
            common_ngram_map_begin(map, history);
            const auto result = lookup(map, history);
            if (values.size() == 3) t.assert_true(result == continuation(0));
            else {
                t.assert_true(result.empty());
                t.assert_true(!map.last_draft_created);
            }
        }
    });

    t.test("k4v_minimum_hits_and_zero_acceptance", [&](testing & t) {
        const auto history = history_for({0, 0});
        common_ngram_map map(2, 3, false, 2);
        common_ngram_map_begin(map, history);
        t.assert_true(lookup(map, history).empty());
        t.assert_true(lookup(map, history) == continuation(0));
        if (!t.assert_equal(size_t(1), map.keys.size())) return;
        t.assert_equal(uint16_t(2), map.keys[0].key_num);
        t.assert_equal(uint16_t(2), map.keys[0].values[0].value_num);
        common_ngram_map_accept(map, 0);
        t.assert_true(lookup(map, history).empty());
        t.assert_true("an empty result is not a created draft", !map.last_draft_created);
        common_ngram_map_begin(map, history);
        t.assert_true(lookup(map, history).empty());
        t.assert_true(lookup(map, history) == continuation(0));
    });

    t.test("k4v_begin_invalidates_replaced_or_shortened_history", [&](testing & t) {
        for (bool shorter : {false, true}) {
            const auto history = history_for({0, 0});
            common_ngram_map map(2, 3, false, 1);
            common_ngram_map_begin(map, history);
            t.assert_true(lookup(map, history) == continuation(0));
            common_ngram_map_accept(map, 1);
            const auto replacement = shorter ? llama_tokens{10} : history_for({1, 1});
            common_ngram_map_begin(map, replacement);
            t.assert_true(map.keys.empty());
            t.assert_true(!map.last_draft_created);
            t.assert_equal(uint32_t(0), map.key_map_last_idx);
            t.assert_true(std::all_of(map.key_map.begin(), map.key_map.end(), [](uint32_t index) { return index == 0; }));
            t.assert_true(lookup(map, replacement) == (shorter ? llama_tokens{} : continuation(1)));
        }
    });

    t.test("ngram_collision_and_optional_hash_index", [&](testing & t) {
        const llama_tokens history{99, 70, 80, 71, 81, 91, 99, 10, 20, 30, 40, 50, 98, 10};
        for (bool indexed : {false, true}) {
            common_ngram_map map(2, 3, false, 1);
            common_ngram_map_begin(map, history);
            if (indexed) map.key_map.assign(1, 1);
            else map.key_map.clear();
            t.assert_true(lookup(map, history) == continuation(0));
        }
    });

    t.test("ngram_short_history_and_native_continuation_lengths", [&](testing & t) {
        for (uint16_t length : {1, 3, 7, 48}) {
            for (bool key_only : {false, true}) {
                common_ngram_map map(2, length, key_only, 1);
                const llama_tokens short_history(size_t(4 + length - 1), 10);
                common_ngram_map_begin(map, short_history);
                t.assert_true(lookup(map, short_history).empty());
                const auto history = history_for({0}, length);
                common_ngram_map_begin(map, history);
                t.assert_true(lookup(map, history) == continuation(0, length));
            }
        }
    });

    t.test("k4v_hit_counter_saturates_without_recounting_history", [&](testing & t) {
        const auto history = history_for({0});
        common_ngram_map map(2, 3, false, 1);
        common_ngram_map_begin(map, history);
        for (size_t i = 0; i < 16400; ++i) lookup(map, history);
        if (!t.assert_equal(size_t(1), map.keys.size())) return;
        t.assert_equal(uint16_t(16380), map.keys[0].key_num);
        t.assert_equal(uint16_t(1), map.keys[0].values[0].value_num);
    });

    t.test("k4v_value_count_saturates_and_four_value_limit_is_preserved", [&](testing & t) {
        auto values = std::vector<int>(16400, 0);
        values.insert(values.end(), {1, 2, 3, 4});
        const auto history = history_for(values);
        common_ngram_map map(2, 3, false, 1);
        common_ngram_map_begin(map, history);
        t.assert_true(lookup(map, history) == continuation(0));
        if (!t.assert_equal(size_t(1), map.keys.size())) return;
        t.assert_equal(uint16_t(16380), map.keys[0].values[0].value_num);
        for (int slot = 1; slot < COMMON_NGRAM_MAX_VALUES; ++slot) {
            t.assert_equal(uint16_t(1), map.keys[0].values[slot].value_num);
        }
        t.assert_true(lookup(map, history) == continuation(0));
        t.assert_equal(uint16_t(1), map.keys[0].values[3].value_num);
    });

    t.test("key_only_zero_acceptance_does_not_publish_an_empty_draft", [&](testing & t) {
        const auto history = history_for({0});
        common_ngram_map map(2, 3, true, 1);
        common_ngram_map_begin(map, history);
        t.assert_true(lookup(map, history) == continuation(0));
        common_ngram_map_accept(map, 0);
        t.assert_true(lookup(map, history).empty());
        t.assert_true(!map.last_draft_created);
    });

    t.test("ngram_configuration_rejects_unrepresentable_lengths", [&](testing & t) {
        for (const auto & dimensions : {std::vector<uint16_t>{0, 3, 1}, {2, 0, 1}, {2, 32768, 1}, {2, 3, 0}}) {
            bool rejected = false;
            try {
                common_ngram_map map(dimensions[0], dimensions[1], false, dimensions[2]);
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            t.assert_true(rejected);
        }
    });

    t.test("streamed_dflash_rollback_covers_both_independent_proposal_limits", [&](testing & t) {
        for (int predictions = 1; predictions <= 7; ++predictions) {
            for (uint16_t ngram = 1; ngram <= 7; ++ngram) {
                const auto params = streamed_dflash_params(predictions, ngram);
                const auto depth = uint32_t(std::max(predictions, int(ngram)));
                const auto target = common_context_params_to_llama(params);
                t.assert_equal(depth, target.n_rs_seq);
                t.assert_equal(predictions, params.speculative.draft.n_max);
                const auto limits = common_speculative_get_output_limits(params.n_batch, 1,
                        common_speculative_n_max(&params.speculative));
                t.assert_equal(int32_t(depth + 1), limits.total);
                t.assert_equal(int32_t(depth + 1), limits.per_seq);
                t.assert_true("mixed public admission must remain closed until the driver is qualified",
                        common_streamed_dflash_config_error(params) != nullptr);
            }
        }
    });

    t.test("streamed_dflash_workload_rejects_invalid_limits_before_context_creation", [&](testing & t) {
        const char * causes[] = {"1-7", "1-7", "size-m", "size-m", "size-m", "lookup", "lookup",
                "microbatch", "microbatch", "positive batch", "positive batch", "proposal sources"};
        for (int change = 0; change < 12; ++change) {
            auto params = streamed_dflash_params(3, 7);
            switch (change) {
                case 0: params.speculative.draft.n_max = 0; break;
                case 1: params.speculative.draft.n_max = 8; break;
                case 2: params.speculative.ngram_map_k4v.size_m = 0; break;
                case 3: params.speculative.ngram_map_k4v.size_m = 8; break;
                case 4: params.speculative.ngram_map_k4v.size_m = 48; break;
                case 5: params.speculative.ngram_map_k4v.size_n = 0; break;
                case 6: params.speculative.ngram_map_k4v.min_hits = 0; break;
                case 7: params.n_ubatch = 7; break;
                case 8: params.n_batch = 7; break;
                case 9: params.n_ubatch = -1; break;
                case 10: params.n_batch = 0; break;
                case 11: params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP); break;
            }
            bool rejected = false;
            try {
                common_context_params_to_llama(params);
            } catch (const std::invalid_argument & error) {
                rejected = std::string(error.what()).find(causes[change]) != std::string::npos;
            }
            t.assert_true(rejected);
        }
    });

    t.test("streamed_workload_resolution_preserves_native_and_single_drafter_limits", [&](testing & t) {
        auto params = streamed_dflash_params(3, 7);
        params.shared_device_memory_bytes = 0;
        t.assert_equal(uint32_t(3), common_context_params_to_llama(params).n_rs_seq);
        params.shared_device_memory_bytes = size_t(1024)*1024*1024;
        params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
        t.assert_equal(uint32_t(3), common_context_params_to_llama(params).n_rs_seq);
        params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
        t.assert_equal(uint32_t(3), common_context_params_to_llama(params).n_rs_seq);
        params.speculative.types = {COMMON_SPECULATIVE_TYPE_NONE};
        t.assert_equal(uint32_t(0), common_context_params_to_llama(params).n_rs_seq);
        params = streamed_dflash_params(3, 7);
        params.n_batch = 8;
        params.n_ubatch = 0;
        t.assert_equal(uint32_t(7), common_context_params_to_llama(params).n_rs_seq);
        params.speculative.types = {COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
                COMMON_SPECULATIVE_TYPE_NONE, COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH,
                COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V};
        t.assert_equal(uint32_t(7), common_context_params_to_llama(params).n_rs_seq);
    });

    t.test("streamed_dflash_model_presence_is_deferred_until_artifact_resolution", [&](testing & t) {
        auto params = streamed_dflash_params(3, 7);
        params.speculative.types = {COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
        params.speculative.draft.mparams = {};
        // The server checks the resolved sidecar before loading target weights.
        t.assert_true(common_streamed_dflash_config_error(params) == nullptr);
    });

    t.test("shared_draft_verification_validation_survives_removed_target_arena_flags", [&](testing & t) {
        const auto target = streamed_dflash_params(3, 7);
        auto params = common_base_params_to_speculative(target);
        t.assert_equal(size_t(0), params.shared_device_memory_bytes);
        t.assert_equal(size_t(0), params.kv_stream_pool_bytes);
        uint32_t depth = 0;
        t.assert_true(common_streamed_dflash_verification_error(params, depth) == nullptr);
        t.assert_equal(uint32_t(7), depth);
        t.assert_equal(3, params.speculative.draft.n_max);
        params.speculative.ngram_map_k4v.size_m = 48;
        t.assert_true(common_streamed_dflash_verification_error(params, depth) != nullptr);
    });

    return t.summary();
}
