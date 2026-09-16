#pragma once

#include "ggml-backend.h"

#include <memory>

// One-shot backend completion with an explicit synchronous fallback.
// The producer and any waiting backend must outlive this object.
class llama_memory_completion {
public:
    static std::unique_ptr<llama_memory_completion> create(ggml_backend_t producer);
    // Represent work that the caller already completed without another backend synchronization.
    static std::unique_ptr<llama_memory_completion> completed(ggml_backend_t producer);
    ~llama_memory_completion();
    llama_memory_completion(llama_memory_completion &&) noexcept;
    llama_memory_completion & operator=(llama_memory_completion &&) noexcept;
    llama_memory_completion(const llama_memory_completion &) = delete;
    llama_memory_completion & operator=(const llama_memory_completion &) = delete;

    bool valid() const noexcept;
    bool event_backed() const noexcept;
    bool recorded() const noexcept;
    bool host_ready() const noexcept;

    // Record after producer submission. Fallback mode synchronizes the producer here.
    bool record();
    // Use the same-device backend wait without declaring host visibility.
    // The backend can implement this as a queue dependency or a host wait.
    bool wait(ggml_backend_t consumer);
    // Promise that this consumer completed every operation that can reference the event.
    bool release_waiter(ggml_backend_t consumer) noexcept;
    // Make completion host-visible. Repeated calls do not repeat the wait.
    bool synchronize();

private:
    llama_memory_completion();
    struct implementation;
    std::unique_ptr<implementation> impl;
};
