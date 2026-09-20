#pragma once

#include "llama-kv-stream-policy.h"

#include <memory>

struct llama_kv_stream_layer_lease_layout {
    llama_kv_stream_policy_config policy;
    llama_kv_stream_policy_state state;
    size_t active_tokens = 0;
    uint64_t layout_revision = 0;
    uint64_t content_generation = 0;
};

struct llama_kv_stream_complete_layer_request {
    uint32_t layer = 0;
    uint32_t query_tokens = 0;
    uint64_t layout_revision = 0;
    uint64_t content_generation = 0;
};

struct llama_kv_stream_complete_layer_lease;
using llama_kv_stream_complete_layer_lease_t = llama_kv_stream_complete_layer_lease *;

// Owns ring-slot admission for one immutable physical layout. It does not copy or publish KV contents.
class llama_kv_stream_layer_lease_owner {
public:
    ~llama_kv_stream_layer_lease_owner();

    static std::unique_ptr<llama_kv_stream_layer_lease_owner> create(
        ggml_backend_memory_lease_t pool,
        const llama_kv_stream_layer_lease_layout & layout);

    // Same-layer requests can share one physical reservation with different query widths.
    llama_kv_stream_complete_layer_lease_t acquire(
        const llama_kv_stream_complete_layer_request & request);

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
