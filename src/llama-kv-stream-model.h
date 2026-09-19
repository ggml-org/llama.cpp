#pragma once
#include "llama-kv-stream-session.h"

// Exact device-local grants contributed to the common text compute/KV parent.
struct llama_kv_stream_memory_requirements {
    ggml_backend_buffer_type_t buffer_type = nullptr;
    size_t pool_bytes = 0, writer_bytes = 0;
    size_t attention_prefill_bytes = 0, attention_decode_bytes = 0;
    size_t alignment = 1;
    size_t shared_device_memory_bytes = 0;
};

// Borrowed candidate leases; the model retains them only after complete session reconstruction.
struct llama_kv_stream_memory_binding {
    ggml_backend_buffer_t parent = nullptr;
    ggml_backend_memory_lease_t pool = nullptr, writer = nullptr, attention = nullptr;
    llama_memory_resource_id pool_resource = 0, writer_resource = 0, attention_resource = 0;
    llama_memory_stage_id prefill_stage = 0, decode_stage = 0;
};

struct llama_kv_stream_runtime_diagnostics {
    uint64_t layout_revision = 0;
    size_t pool_bytes = 0, writer_bytes = 0, attention_bytes = 0;
    uint32_t resident_pages_per_layer = 0, ring_slots = 0, active_pages = 0;
    size_t last_copy_bytes = 0, last_copy_calls = 0;
    double last_copy_ms = 0, last_elapsed_ms = 0;
    bool streaming_active = false;
};

struct llama_kv_stream_model_config {
    ggml_backend_t backend = nullptr;
    llama_kv_stream_host_config host;
    size_t pool_bytes = 0;
    uint32_t max_batch_rows = 0, query_heads = 0;
    bool measure = false;
    bool resume_decode = true;
    bool cross_token_prefetch = true;
    size_t shared_device_memory_bytes = 0;
};

// Own the host-KV execution buffer and a serial session; proxy-buffer references retain the runtime state.
class llama_kv_stream_model {
public:
    static std::unique_ptr<llama_kv_stream_model> create(const llama_kv_stream_model_config & config);
    ~llama_kv_stream_model();
    ggml_backend_buffer_t buffer() const noexcept;
    std::shared_ptr<llama_kv_stream_host> host() const noexcept;
    bool begin(size_t active_tokens, uint32_t query_tokens, bool decode);
    bool complete() const noexcept;
    void abort();
    bool reset(bool clear_bytes);
    bool restore(size_t tokens);
    bool truncate(size_t tokens);
    size_t tokens() const noexcept;
    size_t granted_bytes() const noexcept;
    bool set_workspaces(const std::vector<ggml_backend_memory_lease_t> & leases);
    void release_graphs();
    size_t captured_layers() const;
    bool memory_requirements(llama_kv_stream_memory_requirements & output) const noexcept;
    // Suspend private grants before parent allocation; attach validates all regions before publication.
    bool prepare_shared_memory();
    bool resume_private_memory();
    bool attach_shared_memory(const llama_kv_stream_memory_binding & binding);
    // The shared owner calls detach only after destroying its coordinator and workspace consumer.
    bool detach_shared_memory() noexcept;
    llama_memory_consumer * memory_consumer() noexcept;
    bool uses_shared_memory() const noexcept;
    ggml_backend_buffer_t shared_parent() const noexcept;
    size_t device_grant_bytes() const noexcept;
    size_t pool_grant_bytes() const noexcept;
    size_t writer_grant_bytes() const noexcept;
    size_t attention_grant_bytes() const noexcept;
    bool runtime_diagnostics(llama_kv_stream_runtime_diagnostics & output) const noexcept;
private:
    llama_kv_stream_model() = default;
    struct implementation;
    std::shared_ptr<implementation> impl;
    ggml_backend_buffer_t proxy = nullptr;
};
