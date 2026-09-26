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

static void check_identity(const llama_turbo_innerq_runtime_snapshot & snap, const char * msg) {
    for (float v : snap.scale_inv) {
        check(std::fabs(v - 1.0f) < 1e-7f, msg);
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
        check_identity(snap, "fresh scale_inv is identity");
        check(!state.should_attach_scale_tensor(), "fresh state does not attach scale tensor");
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
        check(state.should_attach_scale_tensor(), "finalized scale publish activates scale tensor");

        llama_turbo_innerq_runtime_snapshot consumed;
        check(state.consume_if_dirty(consumed), "consume_if_dirty sees pending scale update");
        check(!state.peek().dirty, "consume_if_dirty clears dirty");
        check(std::fabs(consumed.scale_inv[0] - 0.5f) < 1e-7f, "consume snapshot preserves scale_inv");
        check(!state.consume_if_dirty(consumed), "second consume sees clean state");
    }

    {
        state.publish_abort(2, 1, false);
        const auto snap = state.peek();
        check(snap.abort_reason == 2, "publish_abort stores abort_reason");
        check(snap.retry_count == 1, "publish_abort stores retry_count");
        check(!snap.freeze_last_good, "publish_abort stores freeze=false");
        check(snap.dirty, "publish_abort marks runtime dirty");
        check(!state.should_attach_scale_tensor(), "abort without freeze disables scale tensor");

        llama_turbo_innerq_runtime_snapshot consumed;
        check(state.consume_if_dirty(consumed), "abort-only update is observable via consume_if_dirty");
        check(consumed.abort_reason == 2, "consumed abort_reason matches published value");
        check(consumed.retry_count == 1, "consumed retry_count matches published value");
        check(!consumed.freeze_last_good, "consumed freeze flag matches published value");
        check_identity(consumed, "abort without freeze resets scale_inv to identity");
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
        check(!state.should_attach_scale_tensor(), "unfinalized fresh scale publish keeps scale tensor inactive");
    }

    {
        float frozen[2] = {0.8f, 1.2f};
        state.publish_scale_inv(frozen, 2, true);
        llama_turbo_innerq_runtime_snapshot consumed;
        check(state.consume_if_dirty(consumed), "finalized scale publish before freeze is consumable");
        check(state.should_attach_scale_tensor(), "finalized scale publish re-enables scale tensor");

        state.publish_abort(3, 0, true);
        check(state.should_attach_scale_tensor(), "freeze-last-good keeps scale tensor active");
        check(state.consume_if_dirty(consumed), "freeze-last-good update is consumable");
        check(consumed.abort_reason == 3, "freeze-last-good stores abort_reason");
        check(consumed.freeze_last_good, "freeze-last-good stores freeze flag");
        check(std::fabs(consumed.scale_inv[0] - 0.8f) < 1e-7f, "freeze-last-good preserves scale_inv[0]");
        check(std::fabs(consumed.scale_inv[1] - 1.2f) < 1e-7f, "freeze-last-good preserves scale_inv[1]");
    }

    if (failures > 0) {
        std::printf("test-turbo-innerq-runtime: %d failures\n", failures);
        return 1;
    }

    std::printf("test-turbo-innerq-runtime: PASS\n");
    return 0;
}
