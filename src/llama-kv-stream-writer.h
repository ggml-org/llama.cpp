#pragma once

#include "llama-kv-stream-resident.h"

// Internal bounded SET_ROWS executor. Its owner retains the scratch lease and drains before reconfiguration.
class llama_kv_stream_writer {
public:
    ~llama_kv_stream_writer();
    static std::unique_ptr<llama_kv_stream_writer> create(ggml_backend_t backend, ggml_backend_buffer_t buffer,
            size_t scratch_bytes, const ggml_kv_stream_shape & shape, size_t max_rows);
    bool accepts(const ggml_tensor * source, bool value) const;
    // Queue private-host download and publication on the backend stream; drain both before scratch reuse.
    bool generate(const ggml_tensor * source, bool value, void * host,
            const std::function<bool(const ggml_tensor *, size_t, size_t)> & publish);
    llama_kv_stream_write_stats stats() const;

private:
    llama_kv_stream_writer() = default;
    struct implementation;
    std::unique_ptr<implementation> impl;
};
