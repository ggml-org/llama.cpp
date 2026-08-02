// server-admission.cpp: see server-admission.h.
//
// ASCII only. No em-dash, no unicode arrows.

#include "server-admission.h"

#include <algorithm>
#include <limits>

namespace tessera_admission {

std::optional<int32_t> choose_preempt_victim(
        const controller_snapshot & snap,
        int64_t candidate_score,
        const admission_config & cfg) {
    // No-op when the controller is disabled or below the soft cap.
    if (cfg.max_admitted <= 0) {
        return std::nullopt;
    }
    if (snap.n_processing < cfg.max_admitted) {
        return std::nullopt;
    }
    // Find the lowest-scoring preemptable slot. vLLM V1 preempts
    // max(running, key=(priority, arrival_time)) - i.e. the lowest priority.
    // Our scheduler score already encodes the same intent (prefix reuse,
    // age, acceptance, cost) so the lowest score is the natural victim.
    const slot_snapshot * victim = nullptr;
    int64_t victim_score = std::numeric_limits<int64_t>::max();
    for (const auto & s : snap.slots) {
        if (!s.processing || !s.preemptable) {
            continue;
        }
        if (s.score_milli < victim_score) {
            victim_score = s.score_milli;
            victim       = &s;
        }
    }
    if (victim == nullptr) {
        // Nothing to preempt (all slots non-preemptable). Defer instead.
        return std::nullopt;
    }
    // Only preempt if the candidate clearly outranks the victim. This avoids
    // thrash when scores are close.
    if (candidate_score - victim_score < cfg.min_score_advantage_milli) {
        return std::nullopt;
    }
    return victim->id;
}

} // namespace tessera_admission
