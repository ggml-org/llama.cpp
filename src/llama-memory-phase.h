#pragma once

#include <cstdint>

enum class llama_memory_text_phase {
    unspecified,
    prefill,
    decode,
};

struct llama_memory_text_phase_signal {
    llama_memory_text_phase phase         = llama_memory_text_phase::unspecified;
    uint32_t                tokens        = 0;
    bool                    serial        = false;
    bool                    ordinary_text = false;
    bool                    speculative   = false;
};

enum class llama_memory_text_phase_status {
    changed,
    unchanged,
    invalid_signal,
    unsupported_execution,
    transition_failed,
    exhausted,
};

struct llama_memory_text_phase_snapshot {
    llama_memory_text_phase phase         = llama_memory_text_phase::unspecified;
    uint64_t                revision      = 0;
    uint64_t                notifications = 0;
};

struct llama_memory_text_phase_result {
    llama_memory_text_phase_status status   = llama_memory_text_phase_status::invalid_signal;
    llama_memory_text_phase        before   = llama_memory_text_phase::unspecified;
    llama_memory_text_phase        after    = llama_memory_text_phase::unspecified;
    uint64_t                       revision = 0;
};

// Track explicit logical-batch intent. Token count is validation metadata, not a phase heuristic.
class llama_memory_text_phase_tracker {
  public:
    llama_memory_text_phase_result   notify(const llama_memory_text_phase_signal & signal) noexcept;
    llama_memory_text_phase_snapshot snapshot() const noexcept;

  private:
    llama_memory_text_phase current    = llama_memory_text_phase::unspecified;
    uint64_t                generation = 0;
    uint64_t                count      = 0;
};
