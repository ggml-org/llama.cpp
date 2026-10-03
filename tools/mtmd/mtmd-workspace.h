#pragma once

#include "mtmd.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "../../ggml/src/ggml-backend-memory.h"

#include <memory>
#include <vector>

class llama_context_memory;

// Internal serial scheduler adapter. The scheduler and backends outlive this owner.
class MTMD_API mtmd_compute_workspace {
public:
    mtmd_compute_workspace(ggml_backend_sched_t sched, std::vector<ggml_backend_t> backends, size_t graph_capacity);
    ~mtmd_compute_workspace();
    mtmd_compute_workspace(const mtmd_compute_workspace &) = delete;
    mtmd_compute_workspace & operator=(const mtmd_compute_workspace &) = delete;

    // Measure a detached graph without allocating its compute buffers.
    bool measure(ggml_cgraph * graph, std::vector<ggml_backend_memory_workspace_group> & output);
    // Borrow one committed lease per measured canonical buffer-type group.
    bool attach(const std::vector<ggml_backend_memory_lease_t> & leases);
    // Borrow measured scratch through the target's exclusive serial owner.
    bool borrow(llama_context_memory & parent);
    // Check the actual graph against its grants before allocating tensor addresses.
    bool alloc_graph(ggml_cgraph * graph);
    // Retain execution dependencies until drain or release completes scheduler work.
    ggml_status compute_async(ggml_cgraph * graph);
    bool drain();
    // Retire captures before the caller overwrites graph metadata; keep workspace leases attached.
    bool retire_graph();
    // Drain, retire native captures, and detach leases before storage can be reused.
    bool release();
    bool supported() const noexcept;
    bool ready() const noexcept;

private:
    struct implementation;
    std::unique_ptr<implementation> impl;
};

// Internal vision-only seam. The optional compute type selects the caller's shared arena type.
// Measurement retires previous workspace grants; image embeddings keep their host ownership.
MTMD_API bool mtmd_batch_measure_compute_workspace(mtmd_batch * batch,
    std::vector<ggml_backend_memory_workspace_group> & output,
    ggml_backend_buffer_type_t compute_type = nullptr);
MTMD_API bool mtmd_attach_compute_workspace(mtmd_context * ctx,
    const std::vector<ggml_backend_memory_lease_t> & leases);
MTMD_API bool mtmd_borrow_compute_workspace(mtmd_context * ctx, llama_context_memory & parent);
MTMD_API bool mtmd_release_compute_workspace(mtmd_context * ctx);
