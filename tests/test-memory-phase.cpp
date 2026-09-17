#include "../src/llama-memory-phase.h"
#include "testing.h"

using phase_status = llama_memory_text_phase_status;
using text_phase   = llama_memory_text_phase;

static llama_memory_text_phase_signal signal(text_phase phase, uint32_t tokens = 1) {
    return { phase, tokens, true, true, false };
}

int main() {
    testing t;

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

    return t.summary();
}
