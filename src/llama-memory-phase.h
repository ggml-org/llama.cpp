#pragma once

#include <cstdint>
#include <cstddef>
#include "ggml.h"

struct llama_memory_phase_cache_identity {
    uint64_t id = 0;
    uint64_t generation = 0;
};

enum class llama_memory_cache_role {
    full_history,
    bounded_resident,
    read_only_source,
};

struct llama_memory_cache_format {
    ggml_type key_type = GGML_TYPE_COUNT;
    ggml_type value_type = GGML_TYPE_COUNT;
    uint32_t key_head_dim = 0;
    uint32_t value_head_dim = 0;
    uint32_t kv_heads = 0;
    bool value_transposed = false;
};

bool llama_memory_cache_format_equal(const llama_memory_cache_format & a, const llama_memory_cache_format & b) noexcept;

// row_bytes is encoded K+V bytes per position per layer, not an element count.
struct llama_memory_cache_requirement {
    llama_memory_phase_cache_identity identity;
    llama_memory_phase_cache_identity source;
    llama_memory_cache_role role = llama_memory_cache_role::full_history;
    uint32_t layers = 0;
    uint32_t history_rows = 0;
    uint32_t window_rows = 0;
    uint32_t headroom_rows = 0;
    uint32_t row_padding = 1;
    size_t row_bytes = 0;
    uint32_t source_first_layer = 0;
    llama_memory_cache_format format;
};

struct llama_memory_cache_bound {
    uint32_t capacity_rows = 0;
    size_t storage_bytes = 0;
    bool writable = false;
};

// Describe storage only. This does not allocate, admit kernels or retain the source lease.
// A read-only alias requires its current owning source descriptor. Failure preserves output.
bool llama_memory_cache_bound_make(const llama_memory_cache_requirement & requirement,
        const llama_memory_cache_requirement * source, llama_memory_cache_bound & output) noexcept;

// Logical read validity only; the reader must also retain storage and its final GPU completion.
struct llama_memory_cache_read_binding {
    llama_memory_cache_requirement cache;
    uint32_t first_position = 0, end_position = 0;
    uint64_t lease_revision = 0;
};

// Bind one read-only source range without allocating storage. Failure preserves output.
bool llama_memory_cache_read_binding_make(const llama_memory_cache_requirement & alias,
        const llama_memory_cache_requirement & source, uint32_t first_position, uint32_t end_position,
        uint64_t lease_revision, llama_memory_cache_read_binding & output) noexcept;
// Reject a stale logical prefix or physical binding even if its raw address is unchanged.
bool llama_memory_cache_read_binding_valid(const llama_memory_cache_read_binding & binding,
        const llama_memory_cache_requirement & source, uint32_t first_position, uint32_t end_position,
        uint64_t lease_revision) noexcept;

enum class llama_memory_text_phase {
    unspecified,
    prefill,
    decode,
};

struct llama_memory_text_phase_signal {
    llama_memory_text_phase phase         = llama_memory_text_phase::unspecified;
    uint32_t                tokens        = 0;
    bool                    serial        = false;
    bool                    ordinary_target = false; // Primary target: text or admitted prefill embeddings.
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

enum class llama_memory_work_phase {
    none,
    target_prefill,
    target_verify,
    recurrent_spill,
    recurrent_restore,
    mtp_catchup,
    mtp_draft,
    draft_feature_injection,
    draft_block_prediction,
};

// Physical graph workload, distinct from a logical batch or a model's auxiliary-layer count.
struct llama_memory_workload_requirement {
    llama_memory_work_phase phase = llama_memory_work_phase::none;
    llama_memory_phase_cache_identity cache;
    uint32_t input_rows = 0;
    uint32_t query_rows = 0;
    uint32_t output_rows = 0;
    uint32_t rollback_depth = 0;
    uint32_t microbatch_rows = 0;
    bool serial = false;
    bool writes_cache = false;
    bool causal = true;
};

// Check metadata against a separately validated cache descriptor; backend admission is separate.
bool llama_memory_workload_requirement_valid(const llama_memory_workload_requirement & requirement,
        const llama_memory_cache_requirement & cache) noexcept;

struct llama_memory_work_phase_signal {
    llama_memory_work_phase phase = llama_memory_work_phase::none;
    uint32_t tokens = 0;
    bool serial = false;
};

enum class llama_memory_work_phase_status {
    changed,
    unchanged,
    invalid_signal,
    unsupported_execution,
    exhausted,
};

struct llama_memory_work_phase_snapshot {
    llama_memory_work_phase phase = llama_memory_work_phase::none;
    uint64_t revision = 0;
    uint64_t notifications = 0;
};

struct llama_memory_work_phase_result {
    llama_memory_work_phase_status status = llama_memory_work_phase_status::invalid_signal;
    llama_memory_work_phase before = llama_memory_work_phase::none;
    llama_memory_work_phase after = llama_memory_work_phase::none;
    uint64_t revision = 0;
};

class llama_memory_work_phase_tracker {
public:
    llama_memory_work_phase_result notify(const llama_memory_work_phase_signal & signal) noexcept;
    llama_memory_work_phase_snapshot snapshot() const noexcept;
private:
    llama_memory_work_phase current = llama_memory_work_phase::none;
    uint64_t revision = 0;
    uint64_t notifications = 0;
};
