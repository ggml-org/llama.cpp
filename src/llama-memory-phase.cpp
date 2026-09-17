#include "llama-memory-phase.h"

#include <limits>

llama_memory_text_phase_result llama_memory_text_phase_tracker::notify(
    const llama_memory_text_phase_signal & signal) noexcept {
    using status      = llama_memory_text_phase_status;
    const auto before = current;
    if (signal.phase == llama_memory_text_phase::unspecified || signal.tokens == 0) {
        return { status::invalid_signal, before, before, generation };
    }
    if (!signal.serial || !signal.ordinary_text || signal.speculative) {
        return { status::unsupported_execution, before, before, generation };
    }
    if (count == std::numeric_limits<uint64_t>::max() ||
        (signal.phase != current && generation == std::numeric_limits<uint64_t>::max())) {
        return { status::exhausted, before, before, generation };
    }
    ++count;
    if (signal.phase == current) {
        return { status::unchanged, before, current, generation };
    }
    current = signal.phase;
    ++generation;
    return { status::changed, before, current, generation };
}

llama_memory_text_phase_snapshot llama_memory_text_phase_tracker::snapshot() const noexcept {
    return { current, generation, count };
}
