#pragma once

#include "llama-memory-executor.h"
#include "../ggml/src/ggml-cuda-graph.h"

// Owner-thread-only adapter for one fixed GGML graph and its CUDA cache entry.
// Backend, tensor metadata, and non-leased dependencies must outlive this adapter.
// All submissions for this graph key must use the adapter; no external cache mutation is allowed.
class llama_memory_cuda_executor : public llama_memory_executor_backend {
public:
    explicit llama_memory_cuda_executor(ggml_backend_t backend);
    ~llama_memory_cuda_executor() override;
    llama_memory_cuda_executor(const llama_memory_cuda_executor &) = delete;
    llama_memory_cuda_executor & operator=(const llama_memory_cuda_executor &) = delete;

    // Discover optional CUDA hooks without linking llama against the CUDA runtime.
    bool supported() const noexcept;

    // Attach an idle graph with fixed topology and the complete leased dependency set.
    // Rejection preserves the existing attachment. Successful attachment discards stale native cache state.
    bool bind(ggml_cgraph * graph, const std::vector<ggml_backend_memory_lease_t> & bindings, uint64_t revision);

    // Pin before submission. One retained pin covers all queued work until drain; it is not a launch counter.
    ggml_status compute_async(const std::vector<ggml_backend_memory_lease_t> & bindings, uint64_t revision);

    // Wait for this backend's compute and joined copies before returning its execution pin.
    bool drain() override;
    void quiesce() noexcept;
    llama_memory_executor_result retire();
    llama_memory_executor_result retire_if_affected(const std::vector<llama_memory_resource_id> & resources);

    // Native capture is lazy: an attached graph can execute without a CUDA graph instance.
    bool is_captured() const;
    bool ready() const noexcept;
    size_t outstanding() const noexcept;

private:
    ggml_backend_t backend;
    ggml_cgraph * graph = nullptr;
    const void * key = nullptr;
    ggml_backend_cuda_graph_release_t release_graph = nullptr;
    ggml_backend_cuda_graph_is_captured_t query_graph = nullptr;
    llama_memory_executor executor;
    llama_memory_execution pending;
};
