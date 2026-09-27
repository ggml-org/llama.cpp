#pragma once

#include "ggml-cpp.h"
#include <cstdint>
#include <memory>

class llama_kv_stream_logical_cache;
class llama_kv_stream_model;
struct ggml_tensor;

// Host-backed metadata routes MTP writes and attention through the target-owned cache.
class llama_kv_stream_mtp_proxy {
public:
    static std::unique_ptr<llama_kv_stream_mtp_proxy> create(
        ggml_backend_dev_t device, std::shared_ptr<llama_kv_stream_logical_cache> cache,
        llama_kv_stream_model * target);
    ~llama_kv_stream_mtp_proxy();

    ggml_tensor * key() const noexcept;
    ggml_tensor * value() const noexcept;
    // Complete any queued host write after the graph has submitted its consumers.
    bool complete_publication();
    bool arm(size_t first, uint32_t rows);
    // Retire a rejected suffix and its ring lease before stock KV cells are reused.
    bool remove_suffix(size_t first, size_t last);
    void disarm() noexcept;
    bool active() const noexcept;
    size_t attention_calls() const noexcept;

private:
    llama_kv_stream_mtp_proxy() = default;
    struct implementation;
    std::shared_ptr<implementation> impl;
    ggml_context_ptr context;
    ggml_backend_buffer_ptr proxy;
    ggml_tensor * key_root = nullptr;
    ggml_tensor * value_root = nullptr;
};
