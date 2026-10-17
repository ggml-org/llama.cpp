#pragma once

#include "llama-memory-phase-owner.h"
#include "../ggml/src/ggml-backend-memory.h"

#include <memory>

struct llama_memory_bounded_kv_layer {
    uint32_t id = 0;
    size_t k_offset = 0, v_offset = 0;
    size_t k_bytes = 0, v_bytes = 0;
};

struct llama_memory_bounded_kv_layout {
    llama_memory_cache_requirement cache;
    llama_memory_cache_allocation_measurement allocation;
    ggml_backend_buffer_type_t buft = nullptr;
    std::vector<llama_memory_bounded_kv_layer> layers;
};

// Quote serial stock K/V roots, including backend padding; this neither allocates storage nor admits attention.
bool llama_memory_bounded_kv_layout_make(const llama_memory_cache_requirement & cache,
        const std::vector<uint32_t> & layer_ids, ggml_backend_buffer_type_t buft,
        llama_memory_bounded_kv_layout & output) noexcept;

struct llama_memory_bounded_kv_binding;

// Retain a read-only physical source until all submitted readers finish. Stock cache metadata is not borrowed here.
class llama_memory_bounded_kv_read {
public:
    explicit operator bool() const noexcept;
    ggml_backend_buffer_t buffer() const noexcept;
    const llama_memory_bounded_kv_layout * layout() const noexcept;
    void reset() noexcept;
private:
    friend class llama_memory_bounded_kv;
    std::shared_ptr<const llama_memory_bounded_kv_binding> binding;
};

// Owner-thread-only storage lifetime. Stock cache owns cells, masks and logical writes; the caller owns completion.
// A live cache keeps its persistent address; moving it requires a separate drained content-migration operation.
// Complete outstanding writes before destroying the owner; a write ticket is not a backend completion event.
class llama_memory_bounded_kv : public llama_memory_consumer {
public:
    explicit llama_memory_bounded_kv(llama_memory_bounded_kv_layout layout, llama_memory_resource resource = {});
    llama_memory_bounded_kv(const llama_memory_bounded_kv &) = delete;
    llama_memory_bounded_kv & operator=(const llama_memory_bounded_kv &) = delete;
    // Register the persistent cache in every named serial phase without allocating backend memory.
    bool register_resources(llama_memory_execution_plan & plan, const std::vector<llama_memory_stage_id> & stages) const;
    // Keep an already-bound region fixed; phase changes may resize other regions, not this cache.
    bool prepare(const llama_memory_transition_target & target, const llama_memory_layout & layout,
            std::unique_ptr<llama_memory_preparation> & output) override;
    bool bind(ggml_backend_memory_lease_t lease) noexcept;
    llama_memory_bounded_kv_read read(llama_memory_phase_cache_identity expected) const noexcept;
    uint64_t begin_write(llama_memory_phase_cache_identity expected) noexcept;
    // Complete compute/copies before finishing. Failed writes close read admission until explicit reconstruction.
    bool finish_write(uint64_t ticket, bool success) noexcept;
    bool reconstruct(llama_memory_phase_cache_identity expected) noexcept;
    llama_memory_phase_cache_identity identity() const noexcept;
    const llama_memory_bounded_kv_layout & layout() const noexcept;
private:
    struct proposal;
    llama_memory_bounded_kv_layout planned;
    llama_memory_resource resource;
    std::shared_ptr<llama_memory_bounded_kv_binding> active;
    uint64_t next_ticket = 0, pending_ticket = 0;
    bool valid = false, preparing = false;
};
