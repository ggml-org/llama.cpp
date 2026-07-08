#include "../src/llama-turbo-innerq-runtime.h"

#include <cmath>
#include <cstdio>

static int failures = 0;

static void check(bool cond, const char * msg) {
    if (!cond) {
        std::printf("FAIL: %s\n", msg);
        ++failures;
    }
}

int main() {
    llama_turbo_innerq_runtime_state state;

    {
        const auto snap = state.peek();
        check(!snap.dirty, "fresh state is clean");
        check(!snap.finalized, "fresh state is not finalized");
        check(snap.abort_reason == 0, "fresh abort_reason is zero");
        check(snap.retry_count == 0, "fresh retry_count is zero");
        check(!snap.freeze_last_good, "fresh freeze_last_good is false");
        for (float v : snap.scale_inv) {
            check(std::fabs(v - 1.0f) < 1e-7f, "fresh scale_inv is identity");
        }
    }

    {
        float probe[4] = {0.5f, 0.75f, 1.25f, 1.5f};
        state.publish_scale_inv(probe, 4, true);

        const auto snap = state.peek();
        check(snap.dirty, "publish_scale_inv marks dirty");
        check(snap.finalized, "publish_scale_inv stores finalized");
        check(std::fabs(snap.scale_inv[0] - 0.5f) < 1e-7f, "scale_inv[0] copied");
        check(std::fabs(snap.scale_inv[1] - 0.75f) < 1e-7f, "scale_inv[1] copied");
        check(std::fabs(snap.scale_inv[2] - 1.25f) < 1e-7f, "scale_inv[2] copied");
        check(std::fabs(snap.scale_inv[3] - 1.5f) < 1e-7f, "scale_inv[3] copied");
        check(std::fabs(snap.scale_inv[4] - 1.0f) < 1e-7f, "tail padded to identity");
    }

    {
        llama_turbo_innerq_runtime_snapshot snap;
        check(state.consume_if_dirty(snap), "consume_if_dirty sees pending scale update");
        check(!state.peek().dirty, "consume_if_dirty clears dirty");
        check(std::fabs(snap.scale_inv[0] - 0.5f) < 1e-7f, "consume snapshot preserves scale_inv");
        check(!state.consume_if_dirty(snap), "second consume sees clean state");
    }

    {
        state.publish_abort(2, 1, true);
        const auto snap = state.peek();
        check(snap.abort_reason == 2, "publish_abort stores abort_reason");
        check(snap.retry_count == 1, "publish_abort stores retry_count");
        check(snap.freeze_last_good, "publish_abort stores freeze flag");
        check(snap.dirty, "publish_abort marks runtime dirty");

        llama_turbo_innerq_runtime_snapshot consumed;
        check(state.consume_if_dirty(consumed), "abort-only update is observable via consume_if_dirty");
        check(consumed.abort_reason == 2, "consumed abort_reason matches published value");
        check(consumed.retry_count == 1, "consumed retry_count matches published value");
        check(consumed.freeze_last_good, "consumed freeze flag matches published value");
    }

    {
        float repaired[2] = {0.9f, 1.1f};
        state.publish_scale_inv(repaired, 2, false);

        llama_turbo_innerq_runtime_snapshot consumed;
        check(state.consume_if_dirty(consumed), "fresh scale publish is still consumable after abort");
        check(consumed.abort_reason == 0, "fresh scale publish clears stale abort_reason");
        check(consumed.retry_count == 0, "fresh scale publish clears stale retry_count");
        check(!consumed.freeze_last_good, "fresh scale publish clears stale freeze flag");
        check(std::fabs(consumed.scale_inv[0] - 0.9f) < 1e-7f, "fresh scale publish keeps new scale_inv[0]");
        check(std::fabs(consumed.scale_inv[1] - 1.1f) < 1e-7f, "fresh scale publish keeps new scale_inv[1]");
    }

    if (failures > 0) {
        std::printf("test-turbo-innerq-runtime: %d failures\n", failures);
        return 1;
    }

    std::printf("test-turbo-innerq-runtime: PASS\n");
    return 0;
}
