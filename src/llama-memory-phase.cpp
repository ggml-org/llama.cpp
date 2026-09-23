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

static bool valid_work_phase(llama_memory_work_phase phase) {
    switch (phase) {
        case llama_memory_work_phase::target_prefill:
        case llama_memory_work_phase::target_verify:
        case llama_memory_work_phase::recurrent_spill:
        case llama_memory_work_phase::recurrent_restore:
        case llama_memory_work_phase::mtp_catchup:
        case llama_memory_work_phase::mtp_draft:
            return true;
        case llama_memory_work_phase::none:
            return false;
    }
    return false;
}

llama_memory_work_phase_result llama_memory_work_phase_tracker::notify(
        const llama_memory_work_phase_signal & signal) noexcept {
    using status=llama_memory_work_phase_status;
    const auto before=current;
    if (!valid_work_phase(signal.phase) || !signal.tokens) return {status::invalid_signal,before,before,revision};
    if (!signal.serial) return {status::unsupported_execution,before,before,revision};
    if (notifications == std::numeric_limits<uint64_t>::max() ||
            (signal.phase != current && revision == std::numeric_limits<uint64_t>::max())) {
        return {status::exhausted,before,before,revision};
    }
    ++notifications;
    if (signal.phase == current) return {status::unchanged,before,current,revision};
    current=signal.phase;
    return {status::changed,before,current,++revision};
}

llama_memory_work_phase_snapshot llama_memory_work_phase_tracker::snapshot() const noexcept {
    return {current,revision,notifications};
}
