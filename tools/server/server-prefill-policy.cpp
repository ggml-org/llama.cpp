// server-prefill-policy.cpp: see server-prefill-policy.h.
//
// ASCII only. No em-dash, no unicode arrows.

#include "server-prefill-policy.h"

#include <algorithm>

namespace tessera_prefill {

void budget_calculator::reset(int32_t n_batch,
                              int32_t already_in_batch,
                              int32_t n_decodes_pending) {
    // Two limits apply simultaneously:
    //   1. the shared per-iteration prefill cap (cfg_.iteration_cap_tokens)
    //   2. the physical batch size n_batch
    // The latter is consumed by both decode tokens (already_in_batch +
    // n_decodes_pending) and prefill tokens. We must leave at least
    // decode_floor_tokens of headroom so the decodes already chosen for this
    // iteration still fit.
    int32_t batch_room = n_batch - already_in_batch;
    // Reserve space for the decodes we have not yet added. Clamp at 0: a
    // negative headroom means the caller already overfilled the batch.
    batch_room -= std::max<int32_t>(0, n_decodes_pending);
    batch_room  = std::max<int32_t>(0, batch_room - cfg_.decode_floor_tokens);
    batch_room_ = batch_room;

    if (cfg_.iteration_cap_tokens > 0) {
        cap_       = cfg_.iteration_cap_tokens;
        remaining_ = cap_;
    } else {
        // No shared cap: behave like the legacy code, only the batch size
        // limits prefill.
        cap_       = 0;
        remaining_ = std::numeric_limits<int32_t>::max();
    }
    capped_any_ = false;
}

int32_t budget_calculator::request_quantum(int32_t slot_remaining_tokens,
                                            int32_t forced_quantum) {
    if (slot_remaining_tokens <= 0) {
        return 0;
    }
    // The per-slot forced quantum already accounts for long-context adaptive
    // shrinking; we take the min of that and the slot's remaining work.
    int32_t q = std::min<int32_t>(forced_quantum, slot_remaining_tokens);
    if (q <= 0) {
        return 0;
    }
    // Then clamp to the shared budget and the batch headroom.
    int32_t q_before = q;
    q = std::min<int32_t>(q, remaining_);
    q = std::min<int32_t>(q, batch_room_);
    if (q <= 0) {
        return 0;
    }
    if (q < q_before) {
        capped_any_ = true;
    }
    // Reserve the granted tokens. A negative remaining_ here would indicate a
    // logic bug in the caller; clamp to keep the policy self-correcting.
    remaining_ -= q;
    batch_room_ -= q;
    return q;
}

} // namespace tessera_prefill
