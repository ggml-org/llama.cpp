#pragma once

// server-admission: dynamic admission control + recompute preemption.
//
// Closes vLLM concurrency-study gap #3: slot count is fixed at startup and
// the server cannot over-subscribe or preempt under KV pressure. vLLM V1
// admits up to a running cap and preempts via recompute (it dropped swap as
// not worth the complexity).
//
// Design notes:
//
//   * The admission controller sits *above* get_available_slot(). When a
//     SERVER_TASK_TYPE_COMPLETION arrives and every slot is busy, the
//     controller decides whether to (a) defer the task (the existing path),
//     or (b) preempt the lowest-scoring active request to make room. The
//     existing scheduler score (1000*prefix + 8*age + 16*acceptance
//     - 4*suffix_cost) is the victim-selection signal.
//
//   * We do NOT raise the number of slots above n_parallel: the slot array
//     is preallocated and indexed by id, so the "soft admission cap" is
//     implemented as a configurable *target* on the number of in-flight
//     requests the controller will allow before it starts preempting. When
//     the cap is below n_parallel the controller is inert (the existing slot
//     admission already enforces a smaller bound).
//
//   * Preemption is recompute-only (vLLM V1 policy). We discard the victim's
//     KV and re-queue the task. The radix cache / block-radix prefix cache
//     will usually re-attach most of the prefix on readmission, so the
//     recompute cost is far below a full re-prefill.
//
//   * This module is policy-only. It returns a decision (preempt slot X, or
//     defer); server-context.cpp performs the actual KV release and task
//     re-queue. That keeps the heavy code paths (llama_memory_seq_rm, queue
//     manipulation) in one place.
//
// ASCII only. No em-dash, no unicode arrows.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace tessera_admission {

struct admission_config {
    // Soft cap on in-flight (processing) requests. When the number of
    // processing slots is >= this and a new task arrives, the controller may
    // preempt. Set to 0 to disable the controller (pure legacy behavior).
    // Default is high (256) so single-box servers see no behavior change
    // unless they explicitly enable tighter admission.
    int32_t max_admitted = 256;
    // If non-zero, do not preempt more than this many requests per iteration.
    // Prevents a thundering herd of preemptions when many large prompts
    // arrive together.
    int32_t max_preemptions_per_iter = 1;
    // Minimum scheduler score delta required before we will preempt an active
    // request to admit a queued one. A new arrival with the same score as the
    // lowest active request should not thrash; only a clearly higher-scoring
    // arrival wins. Default is conservative (0 = "strictly greater wins").
    int64_t min_score_advantage_milli = 0;
};

// Per-slot snapshot the controller uses to rank victims. Mirrors the fields
// the existing scheduler score consumes, so the controller's decision is
// consistent with prefill ordering.
struct slot_snapshot {
    int32_t     id                 = -1;
    bool        processing         = false;
    // The existing scheduler score, evaluated for this slot. Higher is better.
    int64_t     score_milli        = 0;
    // Number of prompt tokens still left to prefill. A slot mid-prefill is a
    // cheaper preempt target (less work to redo) than one deep into decode,
    // all else equal - but we let the caller's score decide, not this struct.
    int32_t     remaining_prefill  = 0;
    int32_t     decoded_so_far     = 0;
    // True if the slot is allowed to be preempted at all (e.g. a parent slot
    // driving parallel sampling children should not be preempted).
    bool        preemptable        = true;
};

struct controller_snapshot {
    std::vector<slot_snapshot> slots;
    int32_t                     n_processing = 0;
};

// Pure stateless decision. Returns the id of the slot to preempt, or
// std::nullopt if the new arrival should just be deferred (no advantage to
// preempting). candidate_score is the would-be score of the queued arrival
// being considered for admission.
//
// The caller is responsible for actually performing the preempt (KV release
// + task re-queue). This function only chooses the victim.
std::optional<int32_t> choose_preempt_victim(
        const controller_snapshot & snap,
        int64_t candidate_score,
        const admission_config & cfg);

// Helper: count processing slots.
inline int32_t count_processing(const controller_snapshot & snap) {
    return snap.n_processing;
}

} // namespace tessera_admission
