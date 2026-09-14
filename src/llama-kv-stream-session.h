#pragma once
#include "llama-kv-stream-resident.h"

struct llama_kv_stream_session_config {
    llama_kv_stream_policy_config policy;
    uint32_t max_batch_rows = 0, query_heads = 0;
    bool measure = false;
    bool native_graph_attention = false;
    bool resume_decode = false;
};

// Serial append-only device consumer. The backend outlives the session; recurrent state belongs to the text model.
class llama_kv_stream_session {
public:
    ~llama_kv_stream_session();
    static std::unique_ptr<llama_kv_stream_session> create(ggml_backend_t backend,
            std::shared_ptr<llama_kv_stream_content> content, const llama_kv_stream_session_config & config,
            ggml_backend_memory_lease_t pool, ggml_backend_memory_lease_t writer, ggml_backend_memory_lease_t partial);
    // Begin a complete contiguous append. Phase intent is explicit, not inferred from a one-token batch.
    bool begin(size_t active_tokens, uint32_t query_tokens, bool decode);
    bool produce(uint32_t layer, const ggml_tensor * k, const ggml_tensor * v);
    bool attention(uint32_t layer, ggml_tensor * q, ggml_tensor * mask, ggml_tensor * output, float scale);
    // Cancellation may follow model-state mutation; this session cannot be resumed without reconstruction.
    void abort();
    bool active() const noexcept;
    bool failed() const noexcept;
    size_t tokens() const noexcept;
    size_t granted_bytes() const noexcept;
    uint64_t layout_revision() const noexcept;
    const llama_kv_stream_policy_state & policy() const noexcept;
    bool set_workspaces(const std::vector<ggml_backend_memory_lease_t> & leases);
    // Scratch is reconstructible; detach only while idle, and attach a disjoint grant before beginning work.
    bool set_attention_workspace(ggml_backend_memory_lease_t lease, bool decode);
    void release_graphs();
    size_t captured_layers() const;
private:
    llama_kv_stream_session();
    struct implementation;
    std::unique_ptr<implementation> impl;
};
