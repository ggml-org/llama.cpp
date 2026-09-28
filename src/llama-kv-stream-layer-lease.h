#pragma once

#include "llama-kv-stream-policy.h"

class llama_kv_stream_logical_cache;
struct ggml_tensor;

#include <memory>
#include <vector>

struct llama_kv_stream_layer_lease_layout {
    llama_kv_stream_policy_config policy;
    llama_kv_stream_policy_state state;
    size_t active_tokens = 0;
    uint64_t layout_revision = 0;
    uint64_t content_generation = 0;
    // Zero uses active_tokens. Otherwise materialize the current physical
    // placement here while reserving future logical tokens up to active_tokens.
    size_t placement_tokens = 0;
};

struct llama_kv_stream_complete_layer_request {
    uint32_t layer = 0;
    uint32_t query_tokens = 0;
    uint64_t layout_revision = 0;
    uint64_t content_generation = 0;
    uint64_t cache_id = 0;
    // Zero uses the full physical reservation.
    size_t active_tokens = 0;
};

struct llama_kv_stream_population_stats {
    size_t bytes = 0;
    size_t calls = 0;
};

struct llama_kv_stream_complete_layer_lease;
using llama_kv_stream_complete_layer_lease_t = llama_kv_stream_complete_layer_lease *;

struct layer_lease_state;

class llama_kv_stream_ring_guard {
public:
    ~llama_kv_stream_ring_guard();
    const std::vector<uint8_t> & blocked_slots() const noexcept;
    const ggml_kv_stream_layout & ring_layout() const noexcept;
    ggml_backend_buffer_t pool_buffer() const noexcept;

private:
    friend class llama_kv_stream_layer_lease_owner;
    llama_kv_stream_ring_guard(std::shared_ptr<layer_lease_state> state, std::vector<uint8_t> blocked);
    std::shared_ptr<layer_lease_state> state;
    std::vector<uint8_t> blocked;
};


// Owns ring slots and optional one-time population for one immutable physical layout.
class llama_kv_stream_layer_lease_owner {
public:
    ~llama_kv_stream_layer_lease_owner();

    static std::unique_ptr<llama_kv_stream_layer_lease_owner> create(
        ggml_backend_memory_lease_t pool,
        const llama_kv_stream_layer_lease_layout & layout);

    // Same-layer requests can share one physical reservation with different query widths.
    llama_kv_stream_complete_layer_lease_t acquire(
        const llama_kv_stream_complete_layer_request & request);

    // Synchronize one upload before returning; query widths share the ready reservation.
    // Resident reuse requires the caller to reset the content mirror on every physical layout change.
    llama_kv_stream_complete_layer_lease_t acquire_populated(ggml_backend_t backend,
        const llama_kv_stream_complete_layer_request & request, const llama_kv_stream_logical_cache & cache,
        bool reuse_resident = false);
    // Rebase a populated prefix after logical suffix rejection without moving its
    // unchanged device bytes or releasing occupied ring slots.
    bool adopt_truncated_prefix(llama_kv_stream_complete_layer_lease_t previous,
        const llama_kv_stream_logical_cache & cache);
    // Extend an already populated reservation only over newly committed host rows.
    // Existing plans become stale on success; reacquire widths under the new generation.
    bool publish_tail(ggml_backend_t backend, llama_kv_stream_complete_layer_lease_t previous,
        const llama_kv_stream_logical_cache & cache, llama_kv_stream_population_stats & delta);
    // Queue encoded rows into reserved device spans without advancing the host frontier.
    // The caller retains source, reservation, and backend until the copy fence completes.
    bool stage_tail_async(ggml_backend_t backend, llama_kv_stream_complete_layer_lease_t previous,
        size_t first, bool value, const ggml_tensor * encoded, size_t row, size_t count,
        llama_kv_stream_population_stats & staged);
    // After the writer's host/device completion, claim already-staged bytes without H2D.
    bool adopt_staged_tail(llama_kv_stream_complete_layer_lease_t previous,
        const llama_kv_stream_logical_cache & cache, const llama_kv_stream_population_stats & staged);

    // Freeze occupied slots while a target prefetch sequence may submit copies.
    std::shared_ptr<llama_kv_stream_ring_guard> hold_ring();

    // Replace layout identity only after every retained reservation has retired.
    bool rebind(const llama_kv_stream_layer_lease_layout & layout);
    bool can_repartition() const;
    void close();
    bool closed() const;

    size_t active_reservations() const;
    size_t active_leases() const;
    size_t ring_slots_used() const;
    uint64_t layout_revision() const;
    uint64_t content_generation() const;

private:
    llama_kv_stream_layer_lease_owner();
    struct implementation;
    std::unique_ptr<implementation> impl;
};

// Retain the complete-layer guard for asynchronous consumers.
llama_kv_stream_complete_layer_lease_t llama_kv_stream_complete_layer_lease_retain(
    llama_kv_stream_complete_layer_lease_t lease);
void llama_kv_stream_complete_layer_lease_free(
    llama_kv_stream_complete_layer_lease_t lease);

// The plan is borrowed and remains valid until the final complete-layer lease reference is freed.
ggml_kv_stream_span_plan_t llama_kv_stream_complete_layer_lease_plan(
    llama_kv_stream_complete_layer_lease_t lease);
uint32_t llama_kv_stream_complete_layer_lease_layer(
    llama_kv_stream_complete_layer_lease_t lease);
size_t llama_kv_stream_complete_layer_lease_resident_tokens(
    llama_kv_stream_complete_layer_lease_t lease);
size_t llama_kv_stream_complete_layer_lease_ring_first(
    llama_kv_stream_complete_layer_lease_t lease);
size_t llama_kv_stream_complete_layer_lease_ring_slots(
    llama_kv_stream_complete_layer_lease_t lease);
uint64_t llama_kv_stream_complete_layer_lease_layout_revision(
    llama_kv_stream_complete_layer_lease_t lease);
uint64_t llama_kv_stream_complete_layer_lease_content_generation(
    llama_kv_stream_complete_layer_lease_t lease);
llama_kv_stream_population_stats llama_kv_stream_complete_layer_lease_population(
    llama_kv_stream_complete_layer_lease_t lease);
