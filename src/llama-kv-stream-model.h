#pragma once
#include "llama-kv-stream-session.h"

struct llama_kv_stream_model_config {
    ggml_backend_t backend = nullptr;
    llama_kv_stream_host_config host;
    size_t pool_bytes = 0;
    uint32_t max_batch_rows = 0, query_heads = 0;
    bool measure = false;
    bool resume_decode = true;
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
    size_t tokens() const noexcept;
    size_t granted_bytes() const noexcept;
    bool set_workspaces(const std::vector<ggml_backend_memory_lease_t> & leases);
    void release_graphs();
    size_t captured_layers() const;
private:
    llama_kv_stream_model() = default;
    struct implementation;
    std::shared_ptr<implementation> impl;
    ggml_backend_buffer_t proxy = nullptr;
};
