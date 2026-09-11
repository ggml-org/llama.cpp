#pragma once

#include "llama-memory-requirements.h"
#include "../ggml/src/ggml-backend-memory.h"

#include <exception>
#include <memory>

// Own native capture resources and any dependencies not represented by arena leases.
// Destruction invalidates the capture and must not throw or submit new work.
struct llama_memory_executable {
    virtual ~llama_memory_executable() = default;
};

struct llama_memory_execution_state;

// Move-only pin: the backend must retain it until all compute/copy work using this capture completes.
// Releasing it is a completion promise, not a device synchronization operation.
class llama_memory_execution {
public:
    llama_memory_execution() = default;
    llama_memory_execution(llama_memory_execution &&) noexcept = default;
    llama_memory_execution & operator=(llama_memory_execution &&) noexcept = default;
    llama_memory_execution(const llama_memory_execution &) = delete;
    llama_memory_execution & operator=(const llama_memory_execution &) = delete;

    // Access native resources only while this pin remains valid.
    explicit operator bool() const noexcept;
    llama_memory_executable * executable() const noexcept;

    // Release the pin after completion; retained leases may become reusable.
    void reset() noexcept;

private:
    friend class llama_memory_executor;
    explicit llama_memory_execution(std::shared_ptr<llama_memory_execution_state> state);
    std::shared_ptr<llama_memory_execution_state> state;
};

struct llama_memory_executor_backend {
    virtual ~llama_memory_executor_backend() = default;

    // Wait for affected compute and copies, then release their execution pins.
    // False or an exception leaves the executor closed and its capture retained.
    virtual bool drain() = 0;
};

enum class llama_memory_executor_status {
    retired,
    unchanged,
    busy,
    pending,
    drain_failed,
};

struct llama_memory_executor_result {
    llama_memory_executor_status status = llama_memory_executor_status::unchanged;
    std::exception_ptr exception;
};

// Owner-thread-only capture guard; backend queues must outlive their outstanding pins.
// Bindings are the complete leased dependency set, with session-unique region/resource IDs.
// Runtime revision covers captured consumer assumptions, not ordinary tensor writes or the arena epoch.
class llama_memory_executor {
public:
    llama_memory_executor() = default;
    ~llama_memory_executor();
    llama_memory_executor(const llama_memory_executor &) = delete;
    llama_memory_executor & operator=(const llama_memory_executor &) = delete;

    // Adopt an idle executable after retaining its leases; failure leaves the caller's executable unchanged.
    // Exact aliases are normalized; conflicting IDs, null leases, and replacement of a live capture are rejected.
    bool capture(
            std::unique_ptr<llama_memory_executable> & executable,
            const std::vector<ggml_backend_memory_lease_t> & bindings,
            uint64_t runtime_revision);

    // Compare native view identity, full region metadata, and runtime revision, ignoring lease/arena generations.
    bool matches(const std::vector<ggml_backend_memory_lease_t> & bindings, uint64_t runtime_revision) const;

    // Acquire before submission; an empty result rejects stale or closed execution.
    llama_memory_execution acquire(
            const std::vector<ggml_backend_memory_lease_t> & bindings, uint64_t runtime_revision) const;

    // Test whether a proposed resource change touches this capture.
    bool affected_by(const std::vector<llama_memory_resource_id> & resources) const noexcept;

    // Close admission, drain, and invalidate native resources before releasing leases.
    // Failure or unreturned pins remain fail-closed and can be retried.
    llama_memory_executor_result retire(llama_memory_executor_backend & backend);
    llama_memory_executor_result retire_if_affected(
            llama_memory_executor_backend & backend, const std::vector<llama_memory_resource_id> & resources);

    // Inspect launch admission and the number of outstanding pins, not GPU completion.
    bool ready() const noexcept;
    size_t outstanding() const noexcept;

private:
    std::shared_ptr<llama_memory_execution_state> captured;
    bool accepting = false;
    bool busy = false;
};
