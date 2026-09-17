#pragma once
#include "llama-kv-stream-resident.h"
#include "llama-kv-stream-publication.h"
#include "llama-memory-transition.h"

struct llama_kv_stream_session_config {
    llama_kv_stream_policy_config policy;
    uint32_t max_batch_rows = 0, query_heads = 0;
    bool measure = false;
    bool native_graph_attention = false;
    bool resume_decode = false;
    llama_memory_resource_id pool_resource = 0;
    llama_memory_stage_id prefill_stage = 0;
    llama_memory_stage_id decode_stage = 0;
};

// Serial append-only device consumer. The backend outlives the session; recurrent state belongs to the text model.
class llama_kv_stream_session : public llama_memory_consumer {
public:
    ~llama_kv_stream_session();
    static std::unique_ptr<llama_kv_stream_session> create(ggml_backend_t backend,
            std::shared_ptr<llama_kv_stream_content> content, const llama_kv_stream_session_config & config,
            ggml_backend_memory_lease_t pool, ggml_backend_memory_lease_t writer, ggml_backend_memory_lease_t partial);
    // Begin a complete contiguous append. Phase intent is explicit, not inferred from a one-token batch.
    bool begin(size_t active_tokens, uint32_t query_tokens, bool decode);
    // Adopt an externally restored contiguous host-cache prefix while idle.
    bool restore(size_t tokens);
    bool produce(uint32_t layer, const ggml_tensor * k, const ggml_tensor * v);
    bool attention(uint32_t layer, ggml_tensor * q, ggml_tensor * mask, ggml_tensor * output, float scale);
    // Cancellation may follow model-state mutation; this session cannot be resumed without reconstruction.
    void abort();
    bool active() const noexcept;
    bool failed() const noexcept;
    size_t tokens() const noexcept;
    size_t granted_bytes() const noexcept;
    uint64_t layout_revision() const noexcept;

    // Resize into a disjoint caller-owned pool while idle. Host contents and token frontier remain authoritative.
    bool grow_pool(ggml_backend_memory_lease_t pool, size_t pool_bytes, bool decode);
    bool shrink_pool(ggml_backend_memory_lease_t pool, size_t pool_bytes, bool decode);
    // Snapshot only; callers must not use it as a lifetime guard.
    llama_kv_stream_binding_view binding_view() const noexcept;
    const llama_kv_stream_policy_state & policy() const noexcept;
    llama_kv_stream_publication_frontiers publication_frontiers() const noexcept;
    bool set_workspaces(const std::vector<ggml_backend_memory_lease_t> & leases);
    // Scratch is reconstructible; detach only while idle, and attach a disjoint grant before beginning work.
    bool set_attention_workspace(ggml_backend_memory_lease_t lease, bool decode);
    void release_graphs();
    size_t captured_layers() const;

    // Prepare a release-before-commit pool resize for the common memory-transition coordinator.
    bool prepare(const llama_memory_transition_target & target, const llama_memory_layout & layout,
            std::unique_ptr<llama_memory_preparation> & output) override;
private:
    bool rebind_pool(ggml_backend_memory_lease_t pool, size_t pool_bytes, bool decode, bool growing);
    llama_kv_stream_session();
    struct implementation;
    std::unique_ptr<implementation> impl;
};
