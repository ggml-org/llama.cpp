#include "../src/llama-memory-phase.h"
#include "../src/llama-cparams.h"
#include "testing.h"
#include <limits>

using phase_status = llama_memory_text_phase_status;
using text_phase   = llama_memory_text_phase;


static llama_memory_text_phase_signal signal(text_phase phase, uint32_t tokens = 1) {
    return { phase, tokens, true, true, false };
}

int main() {
    testing t;
    t.test("shared_assistant_bindings_are_read_only_and_versioned", [](testing & t) {
        using role = llama_memory_cache_role;
        llama_memory_cache_requirement local{{101,2},{},role::bounded_resident,50,262144,1024,256,256,6656,
            0,{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,256,256,16,false}};
        auto full=local;
        full.identity={102,3}; full.role=role::full_history; full.layers=10;
        full.window_rows=full.headroom_rows=0; full.row_bytes=3328;
        full.format={GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,512,512,4,false};
        for (uint32_t block=0;block<4;++block) {
            const auto & source=block == 3 ? full : local;
            auto alias=source;
            alias.identity={201+block,1}; alias.source=source.identity;
            alias.role=role::read_only_source; alias.layers=1;
            alias.source_first_layer=source.layers-1;
            llama_memory_cache_read_binding binding;
            const uint32_t first=block == 3 ? 0 : 8192-1024;
            if (!t.assert_true(llama_memory_cache_read_binding_make(alias,source,first,8192,7,binding))) return;
            t.assert_true(llama_memory_cache_read_binding_valid(binding,source,first,8192,7));
            llama_memory_cache_bound bound;
            if (!t.assert_true(llama_memory_cache_bound_make(binding.cache,&source,bound))) return;
            t.assert_equal(size_t(0),bound.storage_bytes);
            t.assert_true(!bound.writable);
            llama_memory_workload_requirement work{llama_memory_work_phase::mtp_draft,
                binding.cache.identity,1,1,1,0,256,true,false,true};
            t.assert_true(llama_memory_workload_requirement_valid(work,binding.cache));
            work.writes_cache=true;
            t.assert_true(!llama_memory_workload_requirement_valid(work,binding.cache));
            auto changed=source;
            ++changed.identity.generation;
            t.assert_true(!llama_memory_cache_read_binding_valid(binding,changed,first,8192,7));
            changed=source; changed.format.value_type=GGML_TYPE_Q8_0;
            t.assert_true(!llama_memory_cache_read_binding_valid(binding,changed,first,8192,7));
            t.assert_true(!llama_memory_cache_read_binding_valid(binding,source,first,8192,8));
            t.assert_true(!llama_memory_cache_read_binding_valid(binding,source,first,8193,7));
            const auto saved=binding;
            for (int failure=0;failure<5;++failure) {
                auto invalid=alias;
                if (failure == 0) invalid.source_first_layer=source.layers;
                if (failure == 1) invalid.role=role::full_history;
                if (failure == 2) invalid.source.generation++;
                if (failure == 3) invalid.format.key_type=GGML_TYPE_Q4_0;
                if (failure == 4) invalid.identity.id=source.identity.id;
                t.assert_true(!llama_memory_cache_read_binding_make(invalid,source,first,8192,7,binding));
                t.assert_equal(saved.lease_revision,binding.lease_revision);
                t.assert_equal(saved.cache.identity.id,binding.cache.identity.id);
            }
            t.assert_true(!llama_memory_cache_read_binding_make(alias,source,first,8192,0,binding));
            t.assert_true(!llama_memory_cache_read_binding_make(alias,source,8193,8192,7,binding));
            t.assert_true(!llama_memory_cache_read_binding_make(alias,source,first,262145,7,binding));
            if (block != 3) t.assert_true(!llama_memory_cache_read_binding_make(alias,source,0,8192,7,binding));
        }
    });
    t.test("serial_draft_rebuild_uses_its_measured_block_width", [](testing & t) {
        llama_cparams params{};
        params.n_ctx = 8192;
        params.n_ubatch = 256;
        for (uint32_t rows : {2u, 4u, 8u}) {
            params.serial_block_rows = rows;
            t.assert_equal(rows, params.serial_graph_reserve_rows(false));
            t.assert_equal(rows, params.serial_graph_reserve_rows(true));
        }
        params.serial_block_rows = 0;
        t.assert_equal(uint32_t(256), params.serial_graph_reserve_rows(false));
        t.assert_equal(uint32_t(4), params.serial_graph_reserve_rows(true));
        params.n_ctx = 2;
        t.assert_equal(uint32_t(2), params.serial_graph_reserve_rows(false));
        t.assert_equal(uint32_t(2), params.serial_graph_reserve_rows(true));
        params.n_ctx = 8192; params.n_ubatch = 4; params.serial_block_rows = 2;
        t.assert_equal(uint32_t(2), params.serial_graph_reserve_rows(true));
    });

    t.test("explicit_intent_distinguishes_one_token_prompt_and_tg1", [](testing & t) {
        llama_memory_text_phase_tracker tracker;
        auto                            result = tracker.notify(signal(text_phase::prefill, 1));
        t.assert_true(result.status == phase_status::changed);
        t.assert_true(result.before == text_phase::unspecified);
        t.assert_true(result.after == text_phase::prefill);
        t.assert_equal(uint64_t(1), result.revision);

        result = tracker.notify(signal(text_phase::decode, 1));
        t.assert_true(result.status == phase_status::changed);
        t.assert_true(result.before == text_phase::prefill);
        t.assert_true(result.after == text_phase::decode);
        t.assert_equal(uint64_t(2), result.revision);
    });

    t.test("large_and_final_short_prompt_batches_keep_prefill", [](testing & t) {
        llama_memory_text_phase_tracker tracker;
        t.assert_true(tracker.notify(signal(text_phase::prefill, 512)).status == phase_status::changed);
        const auto revision = tracker.snapshot().revision;
        t.assert_true(tracker.notify(signal(text_phase::prefill, 1)).status == phase_status::unchanged);
        const auto snapshot = tracker.snapshot();
        t.assert_true(snapshot.phase == text_phase::prefill);
        t.assert_equal(revision, snapshot.revision);
        t.assert_equal(uint64_t(2), snapshot.notifications);
    });

    t.test("repeated_decode_notifications_do_not_create_transitions", [](testing & t) {
        llama_memory_text_phase_tracker tracker;
        t.assert_true(tracker.notify(signal(text_phase::decode)).status == phase_status::changed);
        for (int i = 0; i < 32; ++i) {
            t.assert_true(tracker.notify(signal(text_phase::decode)).status == phase_status::unchanged);
        }
        const auto snapshot = tracker.snapshot();
        t.assert_true(snapshot.phase == text_phase::decode);
        t.assert_equal(uint64_t(1), snapshot.revision);
        t.assert_equal(uint64_t(33), snapshot.notifications);
    });

    t.test("serial_request_can_return_from_decode_to_prefill", [](testing & t) {
        llama_memory_text_phase_tracker tracker;
        t.assert_true(tracker.notify(signal(text_phase::decode)).status == phase_status::changed);
        const auto result = tracker.notify(signal(text_phase::prefill, 64));
        t.assert_true(result.status == phase_status::changed);
        t.assert_true(result.before == text_phase::decode);
        t.assert_true(result.after == text_phase::prefill);
        t.assert_equal(uint64_t(2), result.revision);
    });

    t.test("invalid_parallel_speculative_and_non_text_signals_are_closed", [](testing & t) {
        llama_memory_text_phase_tracker      tracker;
        const llama_memory_text_phase_signal invalid[] = {
            { text_phase::unspecified, 1, true,  true,  false },
            { text_phase::prefill,     0, true,  true,  false },
            { text_phase::prefill,     1, false, true,  false },
            { text_phase::prefill,     1, true,  false, false },
            { text_phase::decode,      1, true,  true,  true  },
        };
        for (const auto & candidate : invalid) {
            const auto before = tracker.snapshot();
            const auto result = tracker.notify(candidate);
            t.assert_true(result.status == (candidate.phase == text_phase::unspecified || candidate.tokens == 0 ?
                                                phase_status::invalid_signal :
                                                phase_status::unsupported_execution));
            const auto after = tracker.snapshot();
            t.assert_true(after.phase == before.phase);
            t.assert_equal(before.revision, after.revision);
            t.assert_equal(before.notifications, after.notifications);
        }
    });

    t.test("speculative_phases_are_explicit_and_repeated_work_does_not_replan", [](testing & t) {
        llama_memory_work_phase_tracker tracker;
        using phase=llama_memory_work_phase;
        using status=llama_memory_work_phase_status;
        for (phase item : {phase::target_prefill,phase::mtp_draft,phase::target_verify,
                phase::recurrent_spill,phase::mtp_catchup,phase::recurrent_restore,phase::mtp_draft}) {
            const auto before=tracker.snapshot();
            const auto result=tracker.notify({item,4,true});
            t.assert_true(result.status == status::changed);
            t.assert_equal(before.revision+1,result.revision);
            t.assert_true(result.after == item);
            t.assert_true(tracker.notify({item,1,true}).status == status::unchanged);
            t.assert_equal(result.revision,tracker.snapshot().revision);
        }
        t.assert_equal(uint64_t(14),tracker.snapshot().notifications);
    });

    t.test("invalid_speculative_signals_leave_phase_identity_unchanged", [](testing & t) {
        llama_memory_work_phase_tracker tracker;
        using phase=llama_memory_work_phase;
        using status=llama_memory_work_phase_status;
        t.assert_true(tracker.notify({phase::target_verify,1,true}).status == status::changed);
        const auto before=tracker.snapshot();
        t.assert_true(tracker.notify({phase::recurrent_spill,0,true}).status == status::invalid_signal);
        t.assert_true(tracker.notify({phase::mtp_draft,1,false}).status == status::unsupported_execution);
        t.assert_true(tracker.snapshot().phase == before.phase);
        t.assert_equal(before.notifications,tracker.snapshot().notifications);
    });

    t.test("independent_bounded_groups_and_read_only_aliases", [](testing & t) {
        using role = llama_memory_cache_role;
        llama_memory_cache_requirement target{{101, 2}, {}, role::full_history, 16, 262144, 0, 0, 256, 1664};
        llama_memory_cache_requirement qwen{{102, 7}, {}, role::bounded_resident, 5, 262144, 2048, 256, 256, 1664};
        llama_memory_cache_requirement gemma{{103, 3}, {}, role::bounded_resident, 50, 131072, 1024, 512, 256, 1152};
        target.format = {GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,256,256,4,false};
        qwen.format = {GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,128,128,8,false};
        gemma.format = {GGML_TYPE_Q4_0,GGML_TYPE_Q4_0,128,128,8,false};
        llama_memory_cache_bound result;
        t.assert_true(llama_memory_cache_bound_make(target, nullptr, result));
        t.assert_equal(size_t(6979321856), result.storage_bytes);
        t.assert_equal(uint32_t(262144), result.capacity_rows);
        t.assert_true(result.writable);
        t.assert_true(llama_memory_cache_bound_make(qwen, nullptr, result));
        t.assert_equal(uint32_t(2304), result.capacity_rows);
        t.assert_equal(size_t(19169280), result.storage_bytes);
        qwen.history_rows = 8192;
        t.assert_true(llama_memory_cache_bound_make(qwen, nullptr, result));
        t.assert_equal(uint32_t(2304), result.capacity_rows);
        qwen.history_rows = 1000;
        t.assert_true(llama_memory_cache_bound_make(qwen, nullptr, result));
        t.assert_equal(uint32_t(1024), result.capacity_rows);
        t.assert_true(llama_memory_cache_bound_make(gemma, nullptr, result));
        t.assert_equal(uint32_t(1536), result.capacity_rows);
        auto alias = target;
        alias.identity = {104, 1};
        alias.source = target.identity;
        alias.role = role::read_only_source;
        alias.layers = 1;
        alias.source_first_layer = 15;
        t.assert_true(llama_memory_cache_bound_make(alias, &target, result));
        t.assert_equal(size_t(0), result.storage_bytes);
        t.assert_equal(uint32_t(262144), result.capacity_rows);
        t.assert_true(!result.writable);
        const auto valid_alias = alias;
        alias.format.key_head_dim = 128;
        alias.format.value_head_dim = 128;
        alias.format.kv_heads = 8;
        t.assert_true(!llama_memory_cache_bound_make(alias, &target, result));
        alias = valid_alias;
        for (int fault = 0; fault < 8; ++fault) {
            auto bad = valid_alias;
            switch (fault) {
                case 0: bad.source_first_layer = 16; break;
                case 1: bad.layers = 2; break;
                case 2: bad.row_bytes++; break;
                case 3: bad.identity.id = target.identity.id; break;
                case 4: bad.row_padding *= 2; break;
                case 5: bad.history_rows--; break;
                case 6: std::swap(bad.format.key_type, bad.format.value_type); break;
                case 7: bad.format.value_transposed = !bad.format.value_transposed; break;
            }
            const auto before = result;
            t.assert_true(!llama_memory_cache_bound_make(bad, &target, result));
            t.assert_equal(before.capacity_rows, result.capacity_rows);
            t.assert_equal(before.storage_bytes, result.storage_bytes);
        }
        alias.source.generation++;
        const auto before = result;
        t.assert_true(!llama_memory_cache_bound_make(alias, &target, result));
        t.assert_equal(before.storage_bytes, result.storage_bytes);
        t.assert_equal(before.capacity_rows, result.capacity_rows);
        t.assert_equal(before.writable, result.writable);
    });

    t.test("cache_bounds_fail_transactionally_before_overflow", [](testing & t) {
        llama_memory_cache_requirement base{{101, 1}, {}, llama_memory_cache_role::bounded_resident,
                                          5, 262144, 2048, 256, 256, 1664};
        base.format = {GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,128,128,8,false};
        for (int fault = 0; fault < 16; ++fault) {
            auto r = base;
            switch (fault) {
                case 0: r.identity.id = 0; break;
                case 1: r.identity.generation = 0; break;
                case 2: r.layers = 0; break;
                case 3: r.history_rows = 0; break;
                case 4: r.row_padding = 3; break;
                case 5: r.window_rows = UINT32_MAX; break;
                case 6: r.headroom_rows = UINT32_MAX; break;
                case 7: r.row_bytes = SIZE_MAX; break;
                case 8: r.source = {102, 1}; break;
                case 9: r.source_first_layer = 1; break;
                case 10: r.row_bytes = 0; break;
                case 11: r.row_padding = 0; break;
                case 12: r.role = static_cast<llama_memory_cache_role>(-1); break;
                case 13: r.role = llama_memory_cache_role::full_history; r.history_rows = UINT32_MAX; r.window_rows = r.headroom_rows = 0; break;
                case 14: r.format = {}; break;
                case 15: r.format.value_type = static_cast<ggml_type>(-1); break;
            }
            llama_memory_cache_bound out{31, 17, false};
            t.assert_true(!llama_memory_cache_bound_make(r, nullptr, out));
            t.assert_equal(uint32_t(31), out.capacity_rows);
            t.assert_equal(size_t(17), out.storage_bytes);
            t.assert_true(!out.writable);
        }
    });

    t.test("cache_alias_rejects_invalid_source_type_storage", [](testing & t) {
        llama_memory_cache_requirement source{{101, 2}, {}, llama_memory_cache_role::full_history,
                                             2, 4096, 0, 0, 256, 1664};
        source.format = {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 128, 128, 8, false};
        auto alias = source;
        alias.identity = {102, 1};
        alias.source = source.identity;
        alias.role = llama_memory_cache_role::read_only_source;
        alias.layers = 1;
        source.format.key_type = static_cast<ggml_type>(-1);
        t.assert_true(!llama_memory_cache_format_equal(alias.format, source.format));
        llama_memory_cache_bound output{31, 17, true};
        t.assert_true(!llama_memory_cache_bound_make(alias, &source, output));
        t.assert_equal(uint32_t(31), output.capacity_rows);
        t.assert_equal(size_t(17), output.storage_bytes);
        t.assert_true(output.writable);
    });

    t.test("injection_prediction_and_verification_are_distinct_workloads", [](testing & t) {
        using phase = llama_memory_work_phase;
        llama_memory_cache_requirement cache{{101, 1}, {}, llama_memory_cache_role::bounded_resident,
                                           5, 262144, 2048, 256, 256, 1664};
        cache.format = {GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,128,128,8,false};
        llama_memory_workload_requirement inject{phase::draft_feature_injection, cache.identity, 256, 0, 0, 0, 256, true, true, false};
        t.assert_true(llama_memory_workload_requirement_valid(inject, cache));
        auto predict = inject;
        predict.phase = phase::draft_block_prediction;
        predict.input_rows = predict.query_rows = predict.output_rows = 8;
        t.assert_true(llama_memory_workload_requirement_valid(predict, cache));
        predict.input_rows = predict.query_rows = predict.output_rows = 16;
        t.assert_true(llama_memory_workload_requirement_valid(predict, cache));
        auto verify = predict;
        verify.phase = phase::target_verify;
        verify.rollback_depth = 7;
        t.assert_true(llama_memory_workload_requirement_valid(verify, cache));
        verify.query_rows = 0;
        t.assert_true(!llama_memory_workload_requirement_valid(verify, cache));
        predict.microbatch_rows = 8;
        t.assert_true(!llama_memory_workload_requirement_valid(predict, cache));
        inject.query_rows = 1;
        t.assert_true(!llama_memory_workload_requirement_valid(inject, cache));
        inject.query_rows = 0;
        inject.cache.generation++;
        t.assert_true(!llama_memory_workload_requirement_valid(inject, cache));
        inject.cache = cache.identity;
        inject.serial = false;
        t.assert_true(!llama_memory_workload_requirement_valid(inject, cache));
        inject.serial = true;
        cache.role = llama_memory_cache_role::read_only_source;
        t.assert_true(!llama_memory_workload_requirement_valid(inject, cache));
        verify = {phase::target_verify, cache.identity, 8, 8, 8, 7, 256, true, false, true};
        t.assert_true(llama_memory_workload_requirement_valid(verify, cache));
        llama_memory_work_phase_tracker tracker;
        t.assert_true(tracker.notify({phase::draft_feature_injection,256,true}).status == llama_memory_work_phase_status::changed);
        t.assert_true(tracker.notify({phase::draft_block_prediction,8,true}).status == llama_memory_work_phase_status::changed);
        t.assert_true(tracker.snapshot().phase == phase::draft_block_prediction);
    });

    return t.summary();
}
