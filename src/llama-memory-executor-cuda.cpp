#include "llama-memory-executor-cuda.h"

#include <new>
#include <utility>

struct llama_memory_cuda_capture : llama_memory_executable {
    ggml_backend_t backend;
    const void * key;
    ggml_backend_cuda_graph_release_t release_graph;
    bool owned = false;

    // Delay ownership until the common guard accepts all leased dependencies.
    llama_memory_cuda_capture(ggml_backend_t backend, const void * key, ggml_backend_cuda_graph_release_t release_graph) :
        backend(backend), key(key), release_graph(release_graph) {}

    // The adapter drains first; the common snapshot releases leased storage after this destructor.
    ~llama_memory_cuda_capture() override {
        if (owned) release_graph(backend, key);
    }
};

// Resolve backend-local hooks without requiring CUDA in a CPU-only build.
llama_memory_cuda_executor::llama_memory_cuda_executor(ggml_backend_t backend) : backend(backend) {
    if (!backend) return;
    auto * device = ggml_backend_get_device(backend);
    if (!device) return;
    auto * reg = ggml_backend_dev_backend_reg(device);
    if (!reg) return;
    release_graph = reinterpret_cast<ggml_backend_cuda_graph_release_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_graph_release"));
    query_graph = reinterpret_cast<ggml_backend_cuda_graph_is_captured_t>(
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_graph_is_captured"));
}

// No pin or native cache entry may outlive the borrowed backend through this adapter.
llama_memory_cuda_executor::~llama_memory_cuda_executor() {
    const auto result = retire();
    GGML_ASSERT(result.status == llama_memory_executor_status::retired ||
                result.status == llama_memory_executor_status::unchanged);
}

// Require both hooks; graph-disabled CUDA can still provide direct execution and no-op retirement.
bool llama_memory_cuda_executor::supported() const noexcept {
    return release_graph && query_graph;
}

// Validate and retain bindings before changing any native graph cache state.
bool llama_memory_cuda_executor::bind(
        ggml_cgraph * candidate, const std::vector<ggml_backend_memory_lease_t> & bindings, uint64_t revision) {
    if (!supported() || !candidate || ggml_graph_n_nodes(candidate) == 0) return false;
    const void * candidate_key = ggml_graph_node(candidate, 0);
    try {
        auto native = std::make_unique<llama_memory_cuda_capture>(backend, candidate_key, release_graph);
        auto * capture = native.get();
        std::unique_ptr<llama_memory_executable> executable = std::move(native);
        if (!executor.capture(executable, bindings, revision)) return false;
        capture->owned = true;
        graph = candidate;
        key = candidate_key;
        // A reused first-node address or graph UID must not resurrect a previous binding's capture.
        ggml_backend_synchronize(backend);
        release_graph(backend, key);
        return true;
    } catch (const std::bad_alloc &) {
        return false;
    }
}

// Retain one pin across the whole queue, including failure after partial submission.
ggml_status llama_memory_cuda_executor::compute_async(
        const std::vector<ggml_backend_memory_lease_t> & bindings, uint64_t revision) {
    if (!graph || ggml_graph_n_nodes(graph) == 0 || ggml_graph_node(graph, 0) != key) return GGML_STATUS_FAILED;
    auto pin = executor.acquire(bindings, revision);
    if (!pin) return GGML_STATUS_FAILED;
    if (!pending) pending = std::move(pin);
    try {
        const auto result = ggml_backend_graph_compute_async(backend, graph);
        if (result != GGML_STATUS_SUCCESS) executor.quiesce();
        return result;
    } catch (...) {
        executor.quiesce();
        throw;
    }
}

// CUDA's primary stream joins its internal compute streams and backend-mediated copies.
bool llama_memory_cuda_executor::drain() {
    if (!supported()) return false;

    ggml_backend_synchronize(backend);
    pending.reset();
    return true;
}
void llama_memory_cuda_executor::release_completed() noexcept {
    pending.reset();
}


// Block new launches before a multi-consumer coordinator begins draining.
void llama_memory_cuda_executor::quiesce() noexcept {
    executor.quiesce();
}

// Retire the backend cache entry before the common guard releases its leased dependencies.
llama_memory_executor_result llama_memory_cuda_executor::retire() {
    const auto result = executor.retire(*this);
    if (result.status == llama_memory_executor_status::retired) graph = nullptr;
    return result;
}

// Do not synchronize or disturb a capture for an unrelated resource change.
llama_memory_executor_result llama_memory_cuda_executor::retire_if_affected(
        const std::vector<llama_memory_resource_id> & resources) {
    const auto result = executor.retire_if_affected(*this, resources);
    if (result.status == llama_memory_executor_status::retired) graph = nullptr;
    return result;
}

// Query native cache state without creating an entry or exposing a CUDA handle.
bool llama_memory_cuda_executor::is_captured() const {
    return supported() && key && query_graph(backend, key);
}

// Report submission admission independently of whether lazy native capture has occurred.
bool llama_memory_cuda_executor::ready() const noexcept {
    return executor.ready();
}

// All pending launches share one lifetime pin, preventing per-launch queue metadata growth.
size_t llama_memory_cuda_executor::outstanding() const noexcept {
    return executor.outstanding();
}
