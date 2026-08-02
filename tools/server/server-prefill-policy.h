#pragma once

// server-prefill-policy: per-iteration shared prefill cap.
//
// This closes the vLLM concurrency-study gap #1: llama-server does NOT cap
// per-request prefill per step, so one long prompt stalls all decodes. vLLM V1
// defaults to 8192-token chunks (Sarathi-Serve showed up to 10x decode and
// 1.33x e2e gains). The existing prefill_quantum_for() caps a single slot's
// contribution when interactive work is pending, but it leaves no *shared*
// budget across slots: N simultaneous long prefills can still push every
// decode out of the iteration.
//
// This module owns the policy; the integration into pre_decode() lives in
// server-context.cpp. The policy is a host-side scheduling decision only -
// no new kernels, composes naturally with the paged KV (each chunk just
// appends blocks) and the overlap scheduler.
//
// Lifecycle: one budget_calculator per server_context_impl. reset() is called
// at the top of each pre_decode() pass, then each prefill slot calls
// request_quantum() and consumes the returned number of tokens.
//
// ASCII only. No em-dash, no unicode arrows.

#include <cstdint>

namespace tessera_prefill {

struct policy_config {
    // Shared per-iteration prefill cap, in tokens. Default matches vLLM V1's
    // long-prefill-token-threshold. 0 disables the cap (legacy behavior).
    int32_t iteration_cap_tokens = 8192;
    // Minimum reserved for decode when the batch is mixed. A prefill is
    // rejected by request_quantum() if granting even one more token would
    // leave fewer than this many tokens of headroom for the decodes already
    // queued in the batch. Set to 1 to interleave at the smallest possible
    // granularity (one decode token of headroom).
    int32_t decode_floor_tokens  = 1;
};

// Stateful helper that hands out prefill quanta from a shared per-iteration
// budget. One instance per server_context_impl; reused across iterations.
class budget_calculator {
  public:
    explicit budget_calculator(policy_config cfg) : cfg_(cfg) {}

    void configure(policy_config cfg) { cfg_ = cfg; }
    const policy_config & config() const { return cfg_; }

    // Call at the start of each pre_decode(), before any prefill slot is
    // visited. Records the batch size budget for this iteration.
    //   n_batch                  : llama_n_batch(ctx_tgt)
    //   already_in_batch         : tokens already added to the batch this
    //                              iteration (decodes from generating slots)
    //   n_decodes_pending        : number of decode tokens we expect to add
    //                              (slots in SLOT_STATE_GENERATING)
    void reset(int32_t n_batch, int32_t already_in_batch, int32_t n_decodes_pending);

    // Returns the maximum number of prefill tokens this slot may add to the
    // batch right now. reduced_from_forced is the value the existing
    // prefill_quantum_for() would grant for this slot in isolation (it
    // already folds in long-context adaptive shrinking); the shared budget may
    // reduce it further.
    //
    //   slot_remaining_tokens : how many prompt tokens this slot still needs
    //                            to ingest
    //   forced_quantum         : the per-slot quantum from prefill_quantum_for
    //
    // The result is >0 as long as both the shared cap and the batch headroom
    // allow at least one token; otherwise 0 and the slot defers to the next
    // iteration.
    int32_t request_quantum(int32_t slot_remaining_tokens, int32_t forced_quantum);

    // How many tokens of the shared cap remain (diagnostic for /metrics).
    int32_t remaining() const { return remaining_; }

    // True iff the cap reduced at least one slot's quantum since the last
    // reset(). Used by metrics to count "shared-cap yields" separately from
    // the existing per-slot adaptive yields.
    bool capped_any() const { return capped_any_; }

  private:
    policy_config cfg_;
    int32_t cap_             = 0;   // effective cap this iteration (0 = off)
    int32_t remaining_       = 0;   // shared budget still available
    int32_t batch_room_      = 0;   // headroom in the physical batch
    bool    capped_any_      = false;
};

} // namespace tessera_prefill
