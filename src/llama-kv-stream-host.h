#pragma once

#include "../ggml/src/ggml-kv-stream.h"
#include "ggml-backend.h"

#include <memory>

struct llama_kv_stream_host_config {
    uint64_t cache_id = 0;
    ggml_kv_stream_shape shape;
    ggml_kv_stream_capabilities capabilities;
    size_t context_tokens = 0;
    uint32_t layers = 0;
};

struct llama_kv_stream_host_layer {
    void * k = nullptr;
    void * v = nullptr;
};

// Optional CUDA adapters. Registration retains the caller's buffer until after unregistration.
// Neither adapter falls back to pageable storage or changes an existing registration's ownership.
// The owner must keep its bytes alive; do not externally unregister a successful wrapper.
ggml_backend_buffer_type_t llama_kv_stream_host_buffer_type(ggml_backend_dev_t device);
ggml_backend_buffer_t llama_kv_stream_host_register(ggml_backend_dev_t device, ggml_backend_buffer_t owner);

// Authoritative host bytes, independent of device leases. Queued users must retain this owner until completion.
// Raw access has no dirty tracking yet; callers must serialize writes against copies/device work.
class llama_kv_stream_host {
public:
    ~llama_kv_stream_host();
    llama_kv_stream_host(const llama_kv_stream_host &) = delete;
    llama_kv_stream_host & operator=(const llama_kv_stream_host &) = delete;

    // Allocate and zero the canonical planes. expected_type must promise the required host allocation class.
    static std::shared_ptr<llama_kv_stream_host> create(
            const llama_kv_stream_host_config & config, ggml_backend_buffer_type_t expected_type);
    // Retain existing storage without clearing its contents; reject a different buffer type or insufficient space.
    static std::shared_ptr<llama_kv_stream_host> from_buffer(
            const llama_kv_stream_host_config & config, ggml_backend_buffer_type_t expected_type, ggml_backend_buffer_t buffer);

    uint64_t cache_id() const noexcept;
    const llama_kv_stream_host_config & config() const noexcept;
    const ggml_kv_stream_layout & layout() const noexcept;
    size_t stride() const noexcept;
    size_t bytes() const noexcept;
    ggml_backend_buffer_t buffer() const noexcept;
    // Layer ordinals follow full-attention execution order. Failure leaves output unchanged.
    bool layer(uint32_t index, llama_kv_stream_host_layer & output) const noexcept;

private:
    llama_kv_stream_host() = default;
    llama_kv_stream_host_config cfg;
    ggml_kv_stream_layout planes;
    size_t layer_stride = 0, storage_bytes = 0;
    void * base = nullptr;
    ggml_backend_buffer_t storage = nullptr;
};
