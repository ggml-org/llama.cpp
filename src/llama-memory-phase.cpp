#include "llama-memory-phase.h"

#include <limits>
#include <algorithm>
#include <cstring>

// Read unchecked enum storage before validating its numeric range.
static int32_t cache_type_code(const ggml_type & type) noexcept {
    static_assert(sizeof(type) == sizeof(int32_t), "unexpected cache type size");
    int32_t code;
    std::memcpy(&code, &type, sizeof(code));
    return code;
}

bool llama_memory_cache_format_equal(const llama_memory_cache_format & a, const llama_memory_cache_format & b) noexcept {
    return cache_type_code(a.key_type) == cache_type_code(b.key_type) &&
        cache_type_code(a.value_type) == cache_type_code(b.value_type) && a.key_head_dim == b.key_head_dim &&
        a.value_head_dim == b.value_head_dim && a.kv_heads == b.kv_heads && a.value_transposed == b.value_transposed;
}

bool llama_memory_cache_bound_make(const llama_memory_cache_requirement & r,
        const llama_memory_cache_requirement * source, llama_memory_cache_bound & output) noexcept {
    if (!r.identity.id || !r.identity.generation || !r.layers || !r.history_rows || !r.row_bytes ||
            !r.row_padding || (r.row_padding & (r.row_padding-1)) ||
            cache_type_code(r.format.key_type) < GGML_TYPE_F32 || cache_type_code(r.format.key_type) >= GGML_TYPE_COUNT ||
            cache_type_code(r.format.value_type) < GGML_TYPE_F32 || cache_type_code(r.format.value_type) >= GGML_TYPE_COUNT ||
            !r.format.key_head_dim || !r.format.value_head_dim || !r.format.kv_heads) return false;
    if (r.role == llama_memory_cache_role::read_only_source) {
        if (!source || source->role == llama_memory_cache_role::read_only_source ||
                r.source.id != source->identity.id || r.source.generation != source->identity.generation ||
                r.identity.id == source->identity.id || r.source_first_layer > source->layers ||
                r.layers > source->layers-r.source_first_layer || r.row_bytes != source->row_bytes ||
                r.history_rows != source->history_rows || r.window_rows != source->window_rows ||
                r.headroom_rows != source->headroom_rows || r.row_padding != source->row_padding ||
                !llama_memory_cache_format_equal(r.format, source->format)) return false;
        llama_memory_cache_bound bound;
        if (!llama_memory_cache_bound_make(*source, nullptr, bound)) return false;
        output = {bound.capacity_rows, 0, false};
        return true;
    }
    if (source || r.source.id || r.source.generation || r.source_first_layer) return false;
    uint32_t rows = r.history_rows;
    switch (r.role) {
        case llama_memory_cache_role::full_history:
            if (r.window_rows || r.headroom_rows) return false;
            break;
        case llama_memory_cache_role::bounded_resident:
            if (!r.window_rows || !r.headroom_rows || r.headroom_rows > UINT32_MAX-r.window_rows) return false;
            rows = std::min(rows, r.window_rows+r.headroom_rows);
            break;
        default:
            return false;
    }
    if (rows > UINT32_MAX-(r.row_padding-1)) return false;
    rows = (rows+r.row_padding-1)&~(r.row_padding-1);
    if (r.row_bytes > SIZE_MAX/r.layers) return false;
    const size_t row_bytes = r.row_bytes*r.layers;
    if (rows > SIZE_MAX/row_bytes) return false;
    output = {rows, row_bytes*rows, true};
    return true;
}

bool llama_memory_cache_read_binding_make(const llama_memory_cache_requirement & alias,
        const llama_memory_cache_requirement & source, uint32_t first_position, uint32_t end_position,
        uint64_t lease_revision, llama_memory_cache_read_binding & output) noexcept {
    llama_memory_cache_bound bound;
    if (alias.role != llama_memory_cache_role::read_only_source || !lease_revision ||
            first_position > end_position || end_position > source.history_rows ||
            !llama_memory_cache_bound_make(alias,&source,bound) ||
            end_position-first_position > bound.capacity_rows) return false;
    output={alias,first_position,end_position,lease_revision};
    return true;
}

bool llama_memory_cache_read_binding_valid(const llama_memory_cache_read_binding & binding,
        const llama_memory_cache_requirement & source, uint32_t first_position, uint32_t end_position,
        uint64_t lease_revision) noexcept {
    if (binding.first_position != first_position || binding.end_position != end_position ||
            binding.lease_revision != lease_revision) return false;
    llama_memory_cache_read_binding checked;
    return llama_memory_cache_read_binding_make(binding.cache,source,first_position,end_position,lease_revision,checked);
}

bool llama_memory_workload_requirement_valid(const llama_memory_workload_requirement & r,
        const llama_memory_cache_requirement & cache) noexcept {
    if (!r.serial || !r.input_rows || r.input_rows > r.microbatch_rows || r.output_rows > r.input_rows ||
            !r.cache.id || !r.cache.generation || r.cache.id != cache.identity.id ||
            r.cache.generation != cache.identity.generation ||
            (r.writes_cache && cache.role == llama_memory_cache_role::read_only_source)) return false;
    switch (r.phase) {
        case llama_memory_work_phase::draft_feature_injection:
            return !r.query_rows && !r.output_rows && !r.rollback_depth && r.writes_cache;
        case llama_memory_work_phase::draft_block_prediction:
            return r.query_rows == r.input_rows && r.output_rows == r.input_rows && !r.rollback_depth;
        case llama_memory_work_phase::target_verify:
            return r.query_rows == r.input_rows && r.output_rows;
        case llama_memory_work_phase::target_prefill:
        case llama_memory_work_phase::mtp_catchup:
        case llama_memory_work_phase::mtp_draft:
            return r.query_rows == r.input_rows && !r.rollback_depth;
        case llama_memory_work_phase::recurrent_spill:
        case llama_memory_work_phase::recurrent_restore:
            return !r.query_rows && !r.output_rows && r.rollback_depth;
        default:
            return false;
    }
}

llama_memory_text_phase_result llama_memory_text_phase_tracker::notify(
    const llama_memory_text_phase_signal & signal) noexcept {
    using status      = llama_memory_text_phase_status;
    const auto before = current;
    if (signal.phase == llama_memory_text_phase::unspecified || signal.tokens == 0) {
        return { status::invalid_signal, before, before, generation };
    }
    if (!signal.serial || !signal.ordinary_target || signal.speculative) {
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
        case llama_memory_work_phase::draft_feature_injection:
        case llama_memory_work_phase::draft_block_prediction:
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
