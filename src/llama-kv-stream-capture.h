#pragma once
#include "llama-kv-stream-resident.h"
#include "llama-memory-executor-cuda.h"

// KV-aware admission around the existing CUDA executor, not a second native graph cache.
// Owner-thread only. Backend and fixed graph/tensor metadata must outlive this wrapper.
// All mutable graph storage must be leased. Non-view, read-only WEIGHTS buffers are retained separately.
// All submissions for a graph key use this wrapper; external CUDA cache mutation is not allowed.
class llama_kv_stream_cuda_executor : public llama_memory_executor_backend {
public:
    explicit llama_kv_stream_cuda_executor(ggml_backend_t backend);
    ~llama_kv_stream_cuda_executor() override;
    bool supported() const noexcept;
    // Move the authentic KV binding pin only on success. Bindings must include the KV region and graph workspace.
    bool bind(llama_kv_stream_resident & resident, llama_memory_execution & owner, ggml_cgraph * graph,
            const std::vector<ggml_backend_memory_lease_t> & bindings, size_t active_tokens);
    // Values may change after synchronization; pointers, shapes, topology, and runtime identity must still match.
    ggml_status compute_async(size_t active_tokens);
    bool drain() override;
    void quiesce() noexcept;
    llama_memory_executor_result retire();
    llama_memory_executor_result retire_if_affected(const std::vector<llama_memory_resource_id> & resources);
    bool is_captured() const;
    bool ready(size_t active_tokens) const;
private:
    struct implementation;
    std::unique_ptr<implementation> impl;
};
