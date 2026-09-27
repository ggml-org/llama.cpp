#include "../src/llama-kv-stream-policy.h"
#include "../src/llama-kv-stream-feedback.h"
#include "testing.h"

#include <cmath>
#include <limits>

using status = llama_kv_stream_policy_status;

static llama_kv_stream_policy_config config(uint32_t pages, uint32_t layers = 16, bool conversion = false) {
    llama_kv_stream_policy_config c;
    c.shape = {GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 256, 256, 4, 256, 128};
    c.capabilities = {{GGML_TYPE_Q8_0, true, true, true, true}, {GGML_TYPE_Q4_0, true, true, true, true}, !conversion, true};
    c.pool_bytes = size_t(pages)*425984 + (conversion ? 1048576 : 0);
    c.layers = layers;
    return c;
}

static bool start(testing & t, const llama_kv_stream_policy_config & c, llama_kv_stream_policy_state & state) {
    return t.assert_true(llama_kv_stream_policy_initialize(c, state).status == status::success);
}

static llama_kv_stream_policy_observation observe(uint32_t pages, uint32_t queries = 1) {
    llama_kv_stream_policy_observation o;
    o.active_tokens = size_t(pages)*256;
    o.query_tokens = queries;
    return o;
}

// Start at an existing decode boundary so feedback, rather than decode entry, controls adaptation.
static void seed(llama_kv_stream_policy_state & state, uint32_t active) {
    state.decode_active_pages = active;
    state.feedback_initialized = true;
    state.feedback_epoch = 7;
}
static void feedback(llama_kv_stream_policy_observation & o, const llama_kv_stream_policy_state & s,
        uint64_t misses, double busy, uint32_t peak = 0) {
    o.feedback = {true, 7, s.samples + 100, s.misses + misses, busy, peak};
}

// Independent descending oracle retained from the production decision rule, used only on small budgets.
static uint32_t reference_target(uint32_t pool, uint32_t layers, uint32_t active, uint32_t minimum, double ratio) {
    const uint32_t maximum = std::min(active, (pool - minimum)/layers);
    for (uint32_t r = maximum;; --r) {
        if (active == r || double(pool - r*layers) >= ratio*double(active - r)) return r;
        if (r == 0) return 0;
    }
}

int main() {
    testing t;
    t.test("target_and_mtp_share_seventeen_layer_budget_without_sharing_identity", [](testing & t) {
        auto c = config(358, 17);
        c.caches = {{101, 16}, {202, 1}};
        llama_kv_stream_policy_state state;
        if (!start(t, c, state)) return;
        t.assert_equal(uint32_t(17), state.budget.layers);
        t.assert_equal(uint32_t(20), state.resident_pages_per_layer);
        t.assert_equal(uint32_t(18), state.ring_slots);
        t.assert_equal(size_t(358)*425984, state.budget.conversion_offset);
        llama_kv_stream_policy_layout layout;
        if (!t.assert_true(llama_kv_stream_policy_layout_make(c, state, 256, layout).status == status::success)) return;
        t.assert_equal(size_t(17), layout.layers.size());
        t.assert_equal(uint64_t(101), layout.layers[0].cache_id);
        t.assert_equal(uint32_t(0), layout.layers[0].cache_layer);
        t.assert_equal(uint64_t(101), layout.layers[15].cache_id);
        t.assert_equal(uint32_t(15), layout.layers[15].cache_layer);
        t.assert_equal(uint64_t(202), layout.layers[16].cache_id);
        t.assert_equal(uint32_t(0), layout.layers[16].cache_layer);
        t.assert_equal(layout.ring.bytes + size_t(17)*20*425984, layout.conversion_offset);

        llama_kv_stream_policy_decision streamed;
        if (!t.assert_true(llama_kv_stream_policy_step(c, state, observe(24), streamed).status == status::success)) return;
        if (!t.assert_true(llama_kv_stream_policy_layout_make(c, streamed.next, 24*256, layout).status == status::success)) return;
        size_t accounted = layout.ring.bytes;
        for (const auto & layer : layout.layers) accounted += layer.planes.bytes;
        t.assert_equal(layout.conversion_offset, accounted);
        t.assert_equal(uint64_t(101), layout.layers[15].cache_id);
        t.assert_equal(uint32_t(15), layout.layers[15].cache_layer);
        t.assert_equal(uint64_t(202), layout.layers[16].cache_id);
        t.assert_equal(uint32_t(0), layout.layers[16].cache_layer);

        auto changed = c;
        changed.caches[1].id = 303;
        llama_kv_stream_policy_decision decision;
        t.assert_true(llama_kv_stream_policy_step(changed, state, observe(1), decision).status == status::invalid_state);
        changed = c;
        changed.caches[1].id = changed.caches[0].id;
        t.assert_true(llama_kv_stream_policy_initialize(changed, state).status == status::invalid_config);
        changed = c;
        changed.caches[1].layers = 2;
        t.assert_true(llama_kv_stream_policy_initialize(changed, state).status == status::invalid_config);
        t.assert_equal(uint32_t(20), state.resident_pages_per_layer);

        for (const auto & types : {
                std::pair{GGML_TYPE_F16, GGML_TYPE_F16},
                std::pair{GGML_TYPE_Q5_0, GGML_TYPE_Q4_0}}) {
            auto alternate = c;
            alternate.shape.type_k = types.first;
            alternate.shape.type_v = types.second;
            alternate.capabilities.k.type = types.first;
            alternate.capabilities.v.type = types.second;
            size_t minimum = 0;
            if (!t.assert_true(llama_kv_stream_policy_minimum_pool_bytes(alternate, minimum).status == status::success)) return;
            ggml_kv_stream_execution page;
            if (!t.assert_true(ggml_kv_stream_resolve(alternate.shape, alternate.capabilities, 256, page).status == ggml_kv_stream_status::success)) return;
            t.assert_equal(size_t(18)*page.storage.bytes + page.conversion.bytes, minimum);
            alternate.pool_bytes = minimum - 1;
            t.assert_true(llama_kv_stream_policy_initialize(alternate, state).status == status::invalid_budget);
            alternate.pool_bytes = minimum;
            if (!start(t, alternate, state)) return;
            t.assert_equal(uint32_t(17), state.budget.layers);
            t.assert_equal(uint32_t(1), state.ring_slots);
            t.assert_equal(uint32_t(1), state.resident_pages_per_layer);
        }
    });
    t.test("minimum_pool_is_exact_and_transactional", [](testing & t) {
        for (const auto & types : {
                std::pair{GGML_TYPE_F16,GGML_TYPE_F16},
                std::pair{GGML_TYPE_Q8_0,GGML_TYPE_Q4_0},
                std::pair{GGML_TYPE_Q5_0,GGML_TYPE_Q5_0}}) {
            auto c = config(5,4);
            c.shape.type_k = types.first;
            c.shape.type_v = types.second;
            c.capabilities.k.type = types.first;
            c.capabilities.v.type = types.second;
            c.pool_bytes = 0;
            size_t minimum = 77;
            if (!t.assert_true(llama_kv_stream_policy_minimum_pool_bytes(
                    c,minimum).status == status::success)) return;
            c.pool_bytes = minimum;
            llama_kv_stream_policy_state state;
            if (!t.assert_true(llama_kv_stream_policy_initialize(
                    c,state).status == status::success)) return;
            t.assert_equal(uint32_t(5),state.budget.pages);
            t.assert_equal(uint32_t(1),state.ring_slots);
            t.assert_equal(uint32_t(1),state.resident_pages_per_layer);
        }
        auto invalid = config(5,4);
        invalid.layers = 0;
        size_t unchanged = 91;
        t.assert_true(llama_kv_stream_policy_minimum_pool_bytes(
            invalid,unchanged).status != status::success);
        t.assert_equal(size_t(91),unchanged);
    });

    t.test("runtime_feedback_units_continuity_and_policy_hysteresis", [](testing & t) {
        llama_kv_stream_feedback_window window;
        ggml_kv_stream_copy_feedback sample;
        sample.available = true; sample.samples = 100; sample.misses = 10;
        sample.bytes = 1000; sample.timed_bytes = 500; sample.copy_ms = 2; sample.elapsed_ms = 10; sample.peak_slots = 16;
        t.assert_true(window.add(sample));
        t.assert_true(std::abs(window.snapshot().copy_busy_ratio-.4) < 1e-12);
        auto c = config(160); llama_kv_stream_policy_state s; start(t,c,s); seed(s,12);
        s.feedback_epoch = window.snapshot().epoch;
        for (int i = 0; i < 3; ++i) {
            if (i) t.assert_true(window.add(sample));
            auto o = observe(12); o.feedback = window.snapshot();
            llama_kv_stream_policy_decision d;
            t.assert_true(llama_kv_stream_policy_step(c,s,o,d).status == status::success);
            t.assert_true(d.feedback_used);
            t.assert_true(d.partition_changed == (i == 2)); s = d.next;
        }
        t.assert_equal(uint32_t(8),s.resident_pages_per_layer);
        t.assert_equal(uint32_t(32),s.ring_slots);
        const auto before = window.snapshot();
        sample.timed_bytes = sample.bytes+1;
        t.assert_true(!window.add(sample)); t.assert_equal(before.samples,window.snapshot().samples);
        sample.timed_bytes = 500; sample.copy_ms = NAN; t.assert_true(!window.add(sample));
        sample.copy_ms = 2; sample.misses = 101; t.assert_true(!window.add(sample));
        sample.misses = 10; sample.timed_bytes = 0; t.assert_true(!window.add(sample));
        sample.timed_bytes = 500; sample.elapsed_ms = 0; t.assert_true(!window.add(sample));
        sample.elapsed_ms = 10; sample.copy_ms = 100;
        t.assert_true(window.add(sample)); t.assert_equal(1.0,window.snapshot().copy_busy_ratio);
        sample.samples = UINT64_MAX; t.assert_true(!window.add(sample));
        window.reset(); t.assert_true(!window.snapshot().available && before.epoch != window.snapshot().epoch);
    });
    t.test("feedback_sampling_backs_off_after_clean_runs_and_recovers_on_miss", [](testing & t) {
        llama_kv_stream_feedback_sampler sampler(8,2);
        const auto selected=[&](size_t layers,size_t uploads) {
            sampler.begin_run();
            size_t count=0;
            for (size_t layer=0;layer<layers;++layer) for (size_t upload=0;upload<uploads;++upload)
                count += sampler.select(upload==0,true);
            return count;
        };
        t.assert_equal(size_t(24),selected(3,8));
        t.assert_equal(size_t(1),sampler.stride());
        t.assert_true(sampler.observe(24,0));
        t.assert_equal(size_t(1),sampler.stride());
        t.assert_true(sampler.observe(24,0));
        t.assert_equal(size_t(2),sampler.stride());
        t.assert_equal(size_t(12),selected(3,8));
        t.assert_true(sampler.observe(12,0) && sampler.observe(12,0));
        t.assert_equal(size_t(4),sampler.stride());
        t.assert_equal(size_t(6),selected(3,8));
        t.assert_true(sampler.observe(6,0) && sampler.observe(6,0));
        t.assert_equal(size_t(8),sampler.stride());
        t.assert_equal(size_t(3),selected(3,8));
        sampler.begin_run();
        t.assert_true(!sampler.select(true,false));
        t.assert_true(sampler.select(true,true));
        t.assert_true(!sampler.observe(0,0));
        t.assert_true(!sampler.observe(3,4));
        t.assert_equal(size_t(8),sampler.stride());
        t.assert_true(sampler.observe(5,1));
        t.assert_equal(size_t(1),sampler.stride());
        sampler.reset();
        t.assert_equal(size_t(1),sampler.stride());
    });
    t.test("span_trials_require_matching_samples_and_a_measurable_gain", [](testing & t) {
        for (double bounded : {8.0,9.99,12.0}) {
            llama_kv_stream_span_tuner tuner(2);
            tuner.observe(NAN,false); tuner.observe(1,true);
            t.assert_true(!tuner.selected() && !tuner.bounded());
            for (int i = 0; i < 3; ++i) tuner.observe(10,false);
            t.assert_true(tuner.bounded() && !tuner.selected());
            for (int i = 0; i < 3; ++i) tuner.observe(bounded,true);
            t.assert_true(tuner.selected()); t.assert_true(tuner.bounded() == (bounded == 8));
            tuner.observe(0.01,!tuner.bounded()); t.assert_true(tuner.bounded() == (bounded == 8));
            tuner.reset(); t.assert_true(!tuner.selected() && !tuner.bounded());
        }
    });
    t.test("startup_uses_all_pages_after_conversion_reservation", [](testing & t) {
        auto c = config(160, 16, true); c.pool_bytes += 13;
        llama_kv_stream_policy_state s;
        if (!start(t, c, s)) return;
        t.assert_equal(uint32_t(160), s.budget.pages);
        t.assert_equal(uint32_t(9), s.resident_pages_per_layer);
        t.assert_equal(uint32_t(16), s.ring_slots);
        t.assert_equal(uint32_t(16), s.budget.minimum_ring_slots);
        t.assert_equal(size_t(1048576), s.budget.page.conversion.bytes);
        t.assert_equal(size_t(13), s.budget.unused_bytes);
        t.assert_equal(size_t(160)*425984, s.budget.conversion_offset);
    });

    t.test("exact_minimum_pool_and_invalid_inputs_are_transactional", [](testing & t) {
        auto c = config(17);
        llama_kv_stream_policy_state s;
        if (!start(t, c, s)) return;
        t.assert_equal(uint32_t(1), s.resident_pages_per_layer);
        t.assert_equal(uint32_t(1), s.ring_slots);
        for (int bad = 0; bad < 6; ++bad) {
            auto invalid = c;
            if (bad == 0) --invalid.pool_bytes;
            if (bad == 1) invalid.layers = 0;
            if (bad == 2) invalid.initial_ring_slots = 2;
            if (bad == 3) invalid.overlap_ratio = std::numeric_limits<double>::quiet_NaN();
            if (bad == 4) invalid.grow_evaluations = 0;
            if (bad == 5) invalid.cooldown_evaluations = 0;
            s.ring_slots = 77;
            t.assert_true(llama_kv_stream_policy_initialize(invalid, s).status != status::success);
            t.assert_equal(uint32_t(77), s.ring_slots);
        }
    });

    t.test("short_context_keeps_capacity_without_repartition_churn", [](testing & t) {
        auto c = config(6971);
        llama_kv_stream_policy_state s;
        if (!start(t, c, s)) return;
        s.starved = 10; s.overprovisioned = 10;
        llama_kv_stream_policy_decision d;
        t.assert_true(llama_kv_stream_policy_step(c, s, observe(1), d).status == status::success);
        t.assert_true(!d.partition_changed && !d.layout_changed);
        t.assert_equal(uint32_t(435), d.next.resident_pages_per_layer);
        t.assert_equal(uint32_t(0), d.next.starved);
        llama_kv_stream_policy_layout layout;
        t.assert_true(llama_kv_stream_policy_layout_make(c, d.next, 1, layout).status == status::success);
        if (!t.assert_true(layout.layers.size() == 16)) return;
        t.assert_equal(uint32_t(435), layout.layers[0].capacity_pages);
        t.assert_equal(uint32_t(1), layout.layers[0].resident_live_pages);
        t.assert_equal(uint32_t(0), layout.layers[0].streamed_pages);
    });

    t.test("decode_entry_selects_reference_overlap_before_feedback", [](testing & t) {
        auto c = config(6971);
        llama_kv_stream_policy_state s;
        if (!start(t, c, s)) return;
        s.evaluations_since_repartition = 0;
        llama_kv_stream_policy_decision d;
        t.assert_true(llama_kv_stream_policy_step(c, s, observe(673), d).status == status::success);
        t.assert_equal(uint32_t(418), d.next.resident_pages_per_layer);
        t.assert_equal(uint32_t(283), d.next.ring_slots);
        t.assert_true(d.partition_changed && d.layout_changed && !d.feedback_used);
        t.assert_equal(uint32_t(0), d.next.evaluations_since_repartition);
    });

    t.test("concentration_respects_layer_capacity_and_spreads_splits", [](testing & t) {
        auto c = config(160);
        llama_kv_stream_policy_state s;
        if (!start(t, c, s)) return;
        llama_kv_stream_policy_decision d;
        t.assert_true(llama_kv_stream_policy_step(c, s, observe(12), d).status == status::success);
        t.assert_true(!d.partition_changed && d.layout_changed);
        llama_kv_stream_policy_layout layout;
        if (!t.assert_true(llama_kv_stream_policy_layout_make(c, d.next, 12*256, layout).status == status::success)) return;
        for (size_t i = 0; i < layout.layers.size(); ++i) {
            t.assert_equal(i % 4 == 0 ? uint32_t(0) : uint32_t(12), layout.layers[i].capacity_pages);
        }
        auto prefill = observe(12, 64);
        t.assert_true(llama_kv_stream_policy_step(c, d.next, prefill, d).status == status::success);
        t.assert_equal(uint32_t(0), d.next.decode_active_pages);
    });

    t.test("fixed_tiny_ring_uses_multiple_waves", [](testing & t) {
        auto c = config(33); c.initial_ring_slots = 1; c.fixed_ring = true;
        llama_kv_stream_policy_state s;
        if (!start(t, c, s)) return;
        llama_kv_stream_policy_decision d;
        t.assert_true(llama_kv_stream_policy_step(c, s, observe(5), d).status == status::success);
        t.assert_true(!d.partition_changed);
        llama_kv_stream_policy_layout layout;
        if (!t.assert_true(llama_kv_stream_policy_layout_make(c, d.next, 5*256, layout).status == status::success)) return;
        for (const auto & layer : layout.layers) {
            t.assert_equal(uint32_t(2), layer.capacity_pages);
            t.assert_equal(uint32_t(3), layer.streamed_pages);
            t.assert_equal(uint32_t(3), layer.waves);
        }
    });

    t.test("unreachable_overlap_preserves_one_resident_page_per_layer", [](testing & t) {
        auto c = config(39);
        llama_kv_stream_policy_state s;
        if (!start(t, c, s)) return;
        llama_kv_stream_policy_decision d;
        t.assert_true(llama_kv_stream_policy_step(c, s, observe(65), d).status == status::success);
        t.assert_equal(uint32_t(1), d.next.resident_pages_per_layer);
        t.assert_equal(uint32_t(23), d.next.ring_slots);
        llama_kv_stream_policy_layout layout;
        if (!t.assert_true(llama_kv_stream_policy_layout_make(c, d.next, 65*256, layout).status == status::success)) return;
        t.assert_equal(uint32_t(3), layout.layers[0].waves);
    });

    t.test("feedback_growth_cooldown_and_promotion_match_reference", [](testing & t) {
        auto c = config(160);
        llama_kv_stream_policy_state s;
        if (!start(t, c, s)) return;
        seed(s, 12); s.starved = 2; s.evaluations_since_repartition = 2;
        auto o = observe(12); feedback(o, s, 12, .7);
        llama_kv_stream_policy_decision d;
        t.assert_true(llama_kv_stream_policy_step(c, s, o, d).status == status::success);
        t.assert_true(!d.partition_changed);
        t.assert_equal(uint32_t(3), d.next.starved);
        s = d.next; s.evaluations_since_repartition = 64; feedback(o, s, 12, .7);
        t.assert_true(llama_kv_stream_policy_step(c, s, o, d).status == status::success);
        t.assert_equal(uint32_t(8), d.next.resident_pages_per_layer);
        t.assert_equal(uint32_t(32), d.next.ring_slots);
        s = d.next; s.overprovisioned = 7; s.evaluations_since_repartition = 64;
        feedback(o, s, 0, .25, 0);
        t.assert_true(llama_kv_stream_policy_step(c, s, o, d).status == status::success);
        t.assert_equal(uint32_t(9), d.next.resident_pages_per_layer);
        t.assert_equal(uint32_t(16), d.next.ring_slots);
    });

    t.test("ninety_percent_saturation_blocks_demotion_and_clean_overgrowth_heals_gradually", [](testing & t) {
        auto c = config(7010);
        llama_kv_stream_policy_state s;
        if (!start(t, c, s)) return;
        seed(s, 716); s.resident_pages_per_layer = 416; s.ring_slots = 354; s.starved = 10;
        auto o = observe(716); feedback(o, s, 10, .9, s.ring_slots);
        llama_kv_stream_policy_decision d;
        t.assert_true(llama_kv_stream_policy_step(c, s, o, d).status == status::success);
        t.assert_true(!d.partition_changed);
        s.resident_pages_per_layer = 368; s.ring_slots = 1122;
        s.overprovisioned = 7; s.evaluations_since_repartition = 64;
        feedback(o, s, 0, .25, 0);
        t.assert_true(llama_kv_stream_policy_step(c, s, o, d).status == status::success);
        t.assert_equal(uint32_t(369), d.next.resident_pages_per_layer);
        t.assert_equal(uint32_t(1106), d.next.ring_slots);
    });

    t.test("deadline_misses_grow_beyond_geometry_until_ninety_percent_busy", [](testing & t) {
        auto c = config(6593); c.grow_evaluations = 1; c.cooldown_evaluations = 1;
        llama_kv_stream_policy_state s;
        if (!start(t,c,s)) return;
        seed(s,576); s.resident_pages_per_layer=399; s.ring_slots=209;
        llama_kv_stream_policy_decision d;
        auto step=[&](uint64_t misses,double busy,uint32_t peak=UINT32_MAX) {
            s.evaluations_since_repartition=1;
            auto o=observe(576); feedback(o,s,misses,busy,peak == UINT32_MAX ? s.ring_slots : peak);
            t.assert_true(llama_kv_stream_policy_step(c,s,o,d).status == status::success);
            s=d.next;
        };
        step(10,.52); t.assert_equal(uint32_t(398),s.resident_pages_per_layer); t.assert_equal(uint32_t(225),s.ring_slots);
        t.assert_true(s.spread_streaming);
        step(10,.70); t.assert_equal(uint32_t(397),s.resident_pages_per_layer); t.assert_equal(uint32_t(241),s.ring_slots);
        step(10,.899); t.assert_equal(uint32_t(396),s.resident_pages_per_layer); t.assert_equal(uint32_t(257),s.ring_slots);
        step(10,.90); t.assert_equal(uint32_t(396),s.resident_pages_per_layer); t.assert_equal(uint32_t(257),s.ring_slots);
        step(0,.70,0); t.assert_equal(uint32_t(396),s.resident_pages_per_layer); t.assert_equal(uint32_t(257),s.ring_slots);
        s.resident_pages_per_layer=1; s.ring_slots=6577;
        step(10,.50); t.assert_equal(uint32_t(1),s.resident_pages_per_layer); t.assert_equal(uint32_t(6577),s.ring_slots);
    });

    t.test("complete_layer_deadlines_not_upload_samples_control_growth", [](testing & t) {
        auto c=config(6593); c.grow_evaluations=1; c.cooldown_evaluations=64;
        llama_kv_stream_policy_state s;
        if (!start(t,c,s)) return;
        seed(s,640); s.resident_pages_per_layer=395; s.ring_slots=273;
        llama_kv_stream_policy_decision d;
        auto o=observe(640);
        feedback(o,s,20,.55,s.ring_slots);
        o.feedback.layer_samples=s.layer_samples+16;
        o.feedback.layer_misses=s.layer_misses;
        t.assert_true(llama_kv_stream_policy_step(c,s,o,d).status == status::success);
        t.assert_true(!d.partition_changed);
        s=d.next;
        feedback(o,s,0,.55,s.ring_slots);
        o.feedback.layer_samples=s.layer_samples+16;
        o.feedback.layer_misses=s.layer_misses+1;
        t.assert_true(llama_kv_stream_policy_step(c,s,o,d).status == status::success);
        t.assert_equal(uint32_t(394),d.next.resident_pages_per_layer);
        t.assert_equal(uint32_t(289),d.next.ring_slots);
        t.assert_true(d.next.spread_streaming);
    });



    t.test("full_ring_without_a_layer_miss_does_not_demote", [](testing & t) {
        auto c=config(6593); c.grow_evaluations=1; c.cooldown_evaluations=1;
        llama_kv_stream_policy_state s;
        if (!start(t,c,s)) return;
        seed(s,576); s.resident_pages_per_layer=399; s.ring_slots=209;
        s.evaluations_since_repartition=1;
        auto o=observe(576); feedback(o,s,0,.52,s.ring_slots);
        llama_kv_stream_policy_decision d;
        t.assert_true(llama_kv_stream_policy_step(c,s,o,d).status == status::success);
        t.assert_true(!d.partition_changed);
        t.assert_equal(uint32_t(399),d.next.resident_pages_per_layer);
        t.assert_equal(uint32_t(209),d.next.ring_slots);
    });

    t.test("feedback_epoch_reset_and_repeated_samples_do_not_train", [](testing & t) {
        auto c = config(160);
        llama_kv_stream_policy_state s;
        if (!start(t, c, s)) return;
        seed(s, 12); s.samples = 100; s.misses = 10; s.starved = 2;
        auto o = observe(12); o.feedback = {true, 8, 200, 20, .7, 16};
        llama_kv_stream_policy_decision d;
        t.assert_true(llama_kv_stream_policy_step(c, s, o, d).status == status::success);
        t.assert_true(d.feedback_reset && !d.feedback_used && !d.partition_changed);
        t.assert_equal(uint32_t(0), d.next.starved);
        s = d.next;
        o.feedback.samples += 100; o.feedback.misses += 10;
        t.assert_true(llama_kv_stream_policy_step(c, s, o, d).status == status::success);
        t.assert_true(d.feedback_used);
        t.assert_equal(uint32_t(1), d.next.starved);
        s = d.next;
        t.assert_true(llama_kv_stream_policy_step(c, s, o, d).status == status::success);
        t.assert_true(!d.feedback_used);
        t.assert_equal(uint32_t(1), d.next.starved);
        o.feedback.samples = 2; o.feedback.misses = 1;
        t.assert_true(llama_kv_stream_policy_step(c, d.next, o, d).status == status::success);
        t.assert_true(d.feedback_reset && !d.feedback_used);
    });

    t.test("bad_feedback_is_ignored_and_hysteresis_counters_saturate", [](testing & t) {
        auto c = config(160);
        llama_kv_stream_policy_state s;
        if (!start(t, c, s)) return;
        seed(s, 12); s.starved = UINT32_MAX; s.evaluations_since_repartition = 0;
        auto o = observe(12); feedback(o, s, 10, .7);
        llama_kv_stream_policy_decision d;
        t.assert_true(llama_kv_stream_policy_step(c, s, o, d).status == status::success);
        t.assert_equal(UINT32_MAX, d.next.starved);
        t.assert_true(!d.partition_changed);
        o.feedback.copy_busy_ratio = std::numeric_limits<double>::infinity();
        t.assert_true(llama_kv_stream_policy_step(c, s, o, d).status == status::success);
        t.assert_true(d.feedback_reset && !d.feedback_used);
        t.assert_equal(uint32_t(0), d.next.starved);
    });

    t.test("small_layouts_conserve_pages_and_bound_every_plane", [](testing & t) {
        for (uint32_t layers = 1; layers <= 6; ++layers) for (uint32_t extra = 1; extra <= 20; ++extra) {
            auto c = config(layers + extra, layers);
            llama_kv_stream_policy_state s;
            if (!start(t, c, s)) return;
            for (uint32_t active = 0; active <= 16; ++active) for (uint32_t q : {1, 64}) {
                llama_kv_stream_policy_decision d;
                if (!t.assert_true(llama_kv_stream_policy_step(c, s, observe(active, q), d).status == status::success)) return;
                llama_kv_stream_policy_layout layout;
                if (!t.assert_true(llama_kv_stream_policy_layout_make(c, d.next, size_t(active)*256, layout).status == status::success)) return;
                // Independent forward assignment from the production formula; do not reuse the inverse lookup.
                std::vector<uint32_t> expected(layers, d.next.resident_pages_per_layer);
                if (d.next.decode_active_pages) {
                    const uint64_t deficit = uint64_t(active - d.next.resident_pages_per_layer)*layers;
                    const uint64_t by_ring = (deficit + d.next.ring_slots - 1)/d.next.ring_slots;
                    const uint64_t by_active = (deficit + active - 1)/active;
                    const uint32_t splits = uint32_t(std::max(std::min(uint64_t(layers), by_ring), by_active));
                    expected.assign(layers, active);
                    for (uint32_t split = 0; split < splits; ++split) {
                        expected[size_t(split)*layers/splits] -= uint32_t(deficit/splits + (split < deficit % splits));
                    }
                }
                bool addresses_changed = d.next.ring_slots != s.ring_slots;
                size_t ordinal = 0;
                uint64_t capacity = d.next.ring_slots, live = 0;
                size_t offset = layout.ring.bytes;
                for (const auto & layer : layout.layers) {
                    t.assert_equal(expected[ordinal++], layer.capacity_pages);
                    addresses_changed = addresses_changed || layer.capacity_pages != s.resident_pages_per_layer;
                    t.assert_equal(offset, layer.offset);
                    t.assert_true(layer.planes.v_offset >= layer.planes.k_bytes);
                    t.assert_true(layer.offset + layer.planes.bytes <= layout.conversion_offset);
                    offset += layer.planes.bytes;
                    capacity += layer.capacity_pages;
                    live += layer.resident_live_pages + layer.streamed_pages;
                    t.assert_equal(active, layer.resident_live_pages + layer.streamed_pages);
                    if (layer.streamed_pages) {
                        t.assert_true(uint64_t(layer.waves)*d.next.ring_slots >= layer.streamed_pages);
                        t.assert_true(uint64_t(layer.waves - 1)*d.next.ring_slots < layer.streamed_pages);
                    } else t.assert_equal(uint32_t(0), layer.waves);
                }
                t.assert_true(d.layout_changed == addresses_changed);
                t.assert_equal(uint64_t(layers + extra), capacity);
                t.assert_equal(uint64_t(layers)*active, live);
                t.assert_equal(layout.conversion_offset, offset);
                t.assert_equal(c.pool_bytes, offset + layout.conversion_bytes + layout.unused_bytes);
            }
        }
    });

    t.test("overlap_target_matches_descending_oracle", [](testing & t) {
        for (uint32_t layers = 1; layers <= 8; ++layers) for (uint32_t extra = 1; extra <= 12; ++extra) {
            auto c = config(layers + extra, layers);
            llama_kv_stream_policy_state s;
            if (!start(t, c, s)) return;
            for (uint32_t gap = 1; gap <= 12; ++gap) for (double ratio : {.5, 1.0, 1.1, 2.0, 8.0, 1e300}) {
                c.overlap_ratio = ratio;
                const uint32_t active = s.resident_pages_per_layer + gap;
                llama_kv_stream_policy_decision d;
                if (!t.assert_true(llama_kv_stream_policy_step(c, s, observe(active), d).status == status::success)) return;
                t.assert_equal(std::max(1u,reference_target(
                    s.budget.pages,layers,active,s.budget.minimum_ring_slots,ratio)),
                    d.next.resident_pages_per_layer);
            }
        }
    });

    t.test("very_large_target_does_not_scan_billions_of_pages", [](testing & t) {
        if (sizeof(size_t) < 8) return;
        auto c = config(1, 1);
        c.shape = {GGML_TYPE_F16, GGML_TYPE_F16, 64, 64, 1, 1, 128};
        c.capabilities = {{GGML_TYPE_F16, true, true, true, true}, {GGML_TYPE_F16, true, true, true, true}, true, true};
        c.pool_bytes = size_t(UINT32_MAX)*256;
        llama_kv_stream_policy_state s;
        if (!start(t, c, s)) return;
        llama_kv_stream_policy_observation o; o.active_tokens = UINT32_MAX;
        llama_kv_stream_policy_decision d;
        t.assert_true(llama_kv_stream_policy_step(c, s, o, d).status == status::success);
        t.assert_equal(uint32_t(1), d.next.resident_pages_per_layer);
        t.assert_equal(UINT32_MAX-1, d.next.ring_slots);
    });

    t.test("corrupt_state_and_changed_budget_preserve_output", [](testing & t) {
        auto c = config(160);
        llama_kv_stream_policy_state s;
        if (!start(t, c, s)) return;
        llama_kv_stream_policy_decision d; d.next.ring_slots = 77;
        ++s.ring_slots;
        t.assert_true(llama_kv_stream_policy_step(c, s, observe(12), d).status == status::invalid_state);
        t.assert_equal(uint32_t(77), d.next.ring_slots);
        --s.ring_slots; c.pool_bytes += 128;
        t.assert_true(llama_kv_stream_policy_step(c, s, observe(12), d).status == status::invalid_state);
        t.assert_equal(uint32_t(77), d.next.ring_slots);
    });

    t.test("quant_pairs_change_bytes_not_page_policy", [](testing & t) {
        for (auto k : {GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_F32})
            for (auto v : {GGML_TYPE_Q4_0, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_F32}) {
                auto c = config(160);
                c.shape.type_k = k; c.shape.type_v = v;
                c.capabilities = {{k, true, true, k != GGML_TYPE_F32, true},
                                  {v, true, true, v != GGML_TYPE_F32, true}, true, true};
                ggml_kv_stream_execution page;
                if (!t.assert_true(ggml_kv_stream_resolve(c.shape, c.capabilities, 256, page).status == ggml_kv_stream_status::success)) return;
                c.pool_bytes = 160*page.storage.bytes + page.conversion.bytes;
                llama_kv_stream_policy_state s;
                if (!start(t, c, s)) return;
                t.assert_equal(uint32_t(9), s.resident_pages_per_layer);
                t.assert_equal(uint32_t(16), s.ring_slots);
                llama_kv_stream_policy_layout layout;
                t.assert_true(llama_kv_stream_policy_layout_make(c, s, 256, layout).status == status::success);
                t.assert_equal(c.pool_bytes, layout.conversion_offset + layout.conversion_bytes);
            }
    });

    t.test("phase_cutoff_and_active_page_rounding_are_explicit", [](testing & t) {
        auto c = config(160);
        llama_kv_stream_policy_state s;
        if (!start(t, c, s)) return;
        llama_kv_stream_policy_decision d;
        auto o = observe(12, 32);
        t.assert_true(llama_kv_stream_policy_step(c, s, o, d).status == status::success);
        t.assert_equal(uint32_t(12), d.next.decode_active_pages);
        o.query_tokens = 33;
        t.assert_true(llama_kv_stream_policy_step(c, s, o, d).status == status::success);
        t.assert_equal(uint32_t(0), d.next.decode_active_pages);
        o = observe(12); ++o.active_tokens;
        t.assert_true(llama_kv_stream_policy_step(c, s, o, d).status == status::success);
        t.assert_equal(uint32_t(13), d.next.decode_active_pages);
        o.query_tokens = 0;
        t.assert_true(llama_kv_stream_policy_step(c, s, o, d).status == status::invalid_observation);
    });

    t.test("impossible_feedback_deltas_reset_without_repartition", [](testing & t) {
        auto c = config(160);
        llama_kv_stream_policy_state s;
        if (!start(t, c, s)) return;
        seed(s, 12); s.samples = 100; s.misses = 10; s.starved = 2;
        auto o = observe(12); o.feedback = {true, 7, 110, 50, .2, 0};
        llama_kv_stream_policy_decision d;
        t.assert_true(llama_kv_stream_policy_step(c, s, o, d).status == status::success);
        t.assert_true(d.feedback_reset && !d.feedback_used && !d.partition_changed);
        t.assert_equal(uint32_t(0), d.next.starved);
        o.feedback.samples = 10; o.feedback.misses = 11;
        t.assert_true(llama_kv_stream_policy_step(c, s, o, d).status == status::success);
        t.assert_true(!d.next.feedback_initialized && d.feedback_reset);
    });

    t.test("nonlinear_page_padding_and_stale_decode_extent_are_rejected", [](testing & t) {
        auto c = config(160);
        c.shape = {GGML_TYPE_F16, GGML_TYPE_F16, 1, 1, 1, 1, 128};
        c.capabilities = {{GGML_TYPE_F16, true, true, true, true}, {GGML_TYPE_F16, true, true, true, true}, true, true};
        llama_kv_stream_policy_state s;
        t.assert_true(llama_kv_stream_policy_initialize(c, s).status == status::geometry_error);
        c = config(160);
        if (!start(t, c, s)) return;
        seed(s, 12);
        llama_kv_stream_policy_layout layout; layout.unused_bytes = 77;
        t.assert_true(llama_kv_stream_policy_layout_make(c, s, 13*256, layout).status == status::invalid_observation);
        t.assert_equal(size_t(77), layout.unused_bytes);
    });


    t.test("geometric_overlap_deficit_grows_even_when_copy_is_saturated", [](testing & t) {
        auto c = config(630);
        llama_kv_stream_policy_state s;
        if (!start(t, c, s)) return;
        seed(s, 257); s.starved = 2;
        auto o = observe(257); feedback(o, s, 40, .99, s.ring_slots);
        llama_kv_stream_policy_decision d;
        t.assert_true(llama_kv_stream_policy_step(c, s, o, d).status == status::success);
        t.assert_equal(uint32_t(23), d.next.resident_pages_per_layer);
        t.assert_equal(uint32_t(262), d.next.ring_slots);
    });


    t.test("growth_replans_current_frontier_without_mutating_old_state", [](testing & t) {
        auto current = config(24, 4);
        llama_kv_stream_policy_state initial;
        if (!start(t, current, initial)) return;
        llama_kv_stream_policy_decision before;
        if (!t.assert_true(llama_kv_stream_policy_step(
                current, initial, observe(8), before).status == status::success)) return;
        llama_kv_stream_policy_layout old_layout;
        if (!t.assert_true(llama_kv_stream_policy_layout_make(
                current, before.next, 8*256, old_layout).status == status::success)) return;

        llama_kv_stream_policy_rebind growth;
        const auto larger = config(64, 4).pool_bytes;
        if (!t.assert_true(llama_kv_stream_policy_grow(
                current, 8*256, true, larger, growth).status == status::success)) return;
        t.assert_equal(larger, growth.config.pool_bytes);
        t.assert_equal(uint32_t(64), growth.state.budget.pages);
        t.assert_true(growth.state.resident_pages_per_layer >=
            before.next.resident_pages_per_layer);
        t.assert_true(growth.layout.conversion_offset > old_layout.conversion_offset);
        t.assert_equal(size_t(4), growth.layout.layers.size());
        bool moved_value_plane = false;
        for (size_t layer = 0; layer < growth.layout.layers.size(); ++layer) {
            moved_value_plane = moved_value_plane ||
                growth.layout.layers[layer].offset +
                    growth.layout.layers[layer].planes.v_offset !=
                old_layout.layers[layer].offset +
                    old_layout.layers[layer].planes.v_offset;
        }
        t.assert_true(moved_value_plane);

        const auto unchanged = growth;
        t.assert_true(llama_kv_stream_policy_grow(
            current, 8*256, true, current.pool_bytes, growth).status == status::invalid_budget);
        t.assert_equal(unchanged.config.pool_bytes, growth.config.pool_bytes);
        t.assert_equal(unchanged.state.budget.pages, growth.state.budget.pages);
        t.assert_equal(unchanged.layout.conversion_offset, growth.layout.conversion_offset);
    });

    t.test("shrink_replans_minimum_pool_with_streamed_layers", [](testing & t) {
        auto current = config(64,4);
        llama_kv_stream_policy_state initial;
        if (!start(t,current,initial)) return;
        llama_kv_stream_policy_decision before;
        if (!t.assert_true(llama_kv_stream_policy_step(
                current,initial,observe(8),before).status == status::success)) return;
        llama_kv_stream_policy_layout old_layout;
        if (!t.assert_true(llama_kv_stream_policy_layout_make(
                current,before.next,8*256,old_layout).status == status::success)) return;

        llama_kv_stream_policy_rebind shrink;
        const auto minimum=config(5,4).pool_bytes;
        if (!t.assert_true(llama_kv_stream_policy_shrink(
                current,8*256,true,minimum,shrink).status == status::success)) return;
        t.assert_equal(minimum,shrink.config.pool_bytes);
        t.assert_equal(uint32_t(5),shrink.state.budget.pages);
        t.assert_equal(uint32_t(1),shrink.state.resident_pages_per_layer);
        t.assert_equal(uint32_t(1),shrink.state.ring_slots);
        t.assert_equal(uint32_t(8),shrink.state.decode_active_pages);
        t.assert_true(shrink.layout.conversion_offset<old_layout.conversion_offset);
        for (const auto & layer:shrink.layout.layers) {
            t.assert_equal(uint32_t(1),layer.capacity_pages);
            t.assert_equal(uint32_t(7),layer.streamed_pages);
            t.assert_equal(uint32_t(7),layer.waves);
        }

        const auto unchanged=shrink;
        t.assert_true(llama_kv_stream_policy_shrink(
            current,8*256,true,current.pool_bytes,shrink).status == status::invalid_budget);
        t.assert_equal(unchanged.config.pool_bytes,shrink.config.pool_bytes);
        t.assert_equal(unchanged.state.ring_slots,shrink.state.ring_slots);
    });
    t.test("complete_layer_admission_grows_uniform_prefill_ring", [&](testing & t) {
        const auto c = config(4262, 17);
        llama_kv_stream_policy_state state;
        if (!start(t, c, state)) return;
        t.assert_equal(uint32_t(250), state.resident_pages_per_layer);
        t.assert_equal(uint32_t(12), state.ring_slots);
        llama_kv_stream_policy_decision decision;
        const size_t frontier = 98044, reserved = frontier + 4;
        t.assert_true(llama_kv_stream_policy_reserve_layer(
            c, state, frontier, reserved, 16, decision).status == status::success);
        t.assert_true(decision.layout_changed);
        t.assert_true(decision.next.ring_slots > state.ring_slots);
        t.assert_equal(c.pool_bytes, decision.next.budget.pool_bytes);
        llama_kv_stream_policy_layout layout;
        if (!t.assert_true(llama_kv_stream_policy_layout_make(
                c, decision.next, frontier, layout).status == status::success)) return;
        const uint32_t pages = uint32_t((reserved - 1)/256 + 1);
        t.assert_true(double(decision.next.ring_slots) >=
            c.overlap_ratio * double(pages - layout.layers[16].capacity_pages));
        const auto admitted = decision.next;
        t.assert_true(llama_kv_stream_policy_reserve_layer(
            c, admitted, frontier, reserved, 16, decision).status == status::success);
        t.assert_true(!decision.layout_changed);
        t.assert_equal(admitted.ring_slots, decision.next.ring_slots);

        auto fixed = c;
        fixed.fixed_ring = true;
        const auto unchanged = decision.next;
        t.assert_true(llama_kv_stream_policy_reserve_layer(
            fixed, state, frontier, reserved, 16, decision).status == status::invalid_budget);
        t.assert_equal(unchanged.ring_slots, decision.next.ring_slots);
        t.assert_true(llama_kv_stream_policy_reserve_layer(
            c, state, frontier, reserved, 17, decision).status == status::invalid_observation);
        t.assert_true(llama_kv_stream_policy_reserve_layer(
            c, state, frontier, frontier - 1, 16, decision).status == status::invalid_observation);
        t.assert_true(llama_kv_stream_policy_reserve_layer(
            c, state, frontier, size_t(4263)*256, 16, decision).status == status::invalid_budget);
        t.assert_equal(unchanged.ring_slots, decision.next.ring_slots);
    });
    t.test("complete_layer_admission_covers_tail_pages_and_concentrated_placement", [&](testing & t) {
        auto c = config(4262, 17);
        c.overlap_ratio = 1.0;
        llama_kv_stream_policy_state state;
        if (!start(t, c, state)) return;
        llama_kv_stream_policy_decision decision;
        const size_t frontier = size_t(262)*256;
        if (!t.assert_true(llama_kv_stream_policy_reserve_layer(
                c, state, frontier, frontier, 16, decision).status == status::success)) return;
        t.assert_true(!decision.layout_changed);
        t.assert_true(llama_kv_stream_policy_reserve_layer(
            c, state, frontier, frontier + 1, 16, decision).status == status::success);
        t.assert_true(decision.layout_changed);
        t.assert_true(decision.next.ring_slots >= 263 - decision.next.resident_pages_per_layer);

        if (!t.assert_true(llama_kv_stream_policy_step(c, state,
                observe(383), decision).status == status::success)) return;
        const auto concentrated = decision.next;
        t.assert_true(llama_kv_stream_policy_reserve_layer(
            c, concentrated, 383*256, 383*256 + 4, 16, decision).status == status::success);
        llama_kv_stream_policy_layout layout;
        if (!t.assert_true(llama_kv_stream_policy_layout_make(
                c, decision.next, 383*256, layout).status == status::success)) return;
        t.assert_true(decision.next.ring_slots >= 384 - layout.layers[16].capacity_pages);

        for (uint32_t pages = 17; pages <= 97; pages += 4) {
            const auto small = config(pages, 16);
            llama_kv_stream_policy_state initial;
            if (!start(t, small, initial)) return;
            for (uint32_t active = 1; active <= pages; ++active) {
                const size_t tokens = size_t(active)*256;
                if (!t.assert_true(llama_kv_stream_policy_reserve_layer(
                        small, initial, tokens - 1, tokens, 15, decision).status == status::success)) return;
                if (!t.assert_true(llama_kv_stream_policy_layout_make(
                        small, decision.next, tokens - 1, layout).status == status::success)) return;
                t.assert_equal(pages, decision.next.resident_pages_per_layer*16 + decision.next.ring_slots);
                t.assert_true(decision.next.ring_slots >= active - std::min(active, layout.layers[15].capacity_pages));
            }
        }
    });
    return t.summary();
}
