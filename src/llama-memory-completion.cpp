#include "llama-memory-completion.h"

#include "../ggml/src/ggml-backend-impl.h"

#include <algorithm>
#include <new>
#include <utility>
#include <vector>

struct llama_memory_completion::implementation {
    ggml_backend_t producer = nullptr;
    ggml_backend_event_t event = nullptr;
    std::vector<ggml_backend_t> waiters;
    bool submitted = false;
    bool ready = false;

    ~implementation() {
        if (event) {
            if (submitted && !ready) ggml_backend_event_synchronize(event);
            for (auto * waiter : waiters) ggml_backend_synchronize(waiter);
            ggml_backend_event_free(event);
        }
    }
};

llama_memory_completion::llama_memory_completion() = default;
llama_memory_completion::~llama_memory_completion() = default;
llama_memory_completion::llama_memory_completion(llama_memory_completion &&) noexcept = default;
llama_memory_completion & llama_memory_completion::operator=(llama_memory_completion &&) noexcept = default;

std::unique_ptr<llama_memory_completion> llama_memory_completion::create(ggml_backend_t producer) {
    if (!producer || !producer->device) return {};
    try {
        auto result = std::unique_ptr<llama_memory_completion>(new llama_memory_completion);
        result->impl = std::make_unique<implementation>();
        result->impl->producer = producer;
        auto * device = producer->device;
        if (producer->iface.event_record && device->iface.event_new && device->iface.event_free && device->iface.event_synchronize) {
            result->impl->event = ggml_backend_event_new(device);
        }
        return result;
    } catch (const std::bad_alloc &) {
        return {};
    }
}
std::unique_ptr<llama_memory_completion> llama_memory_completion::completed(ggml_backend_t producer) {
    if (!producer || !producer->device) return {};
    try {
        auto result = std::unique_ptr<llama_memory_completion>(new llama_memory_completion);
        result->impl = std::make_unique<implementation>();
        result->impl->producer = producer;
        result->impl->submitted = true;
        result->impl->ready = true;
        return result;
    } catch (const std::bad_alloc &) {
        return {};
    }
}


bool llama_memory_completion::valid() const noexcept {
    return impl && impl->producer;
}

bool llama_memory_completion::event_backed() const noexcept {
    return valid() && impl->event;
}

bool llama_memory_completion::recorded() const noexcept {
    return valid() && impl->submitted;
}

bool llama_memory_completion::host_ready() const noexcept {
    return valid() && impl->ready;
}

bool llama_memory_completion::record() {
    if (!valid() || impl->submitted) return false;
    if (impl->event) {
        ggml_backend_event_record(impl->event,impl->producer);
    } else {
        ggml_backend_synchronize(impl->producer);
        impl->ready = true;
    }
    impl->submitted = true;
    return true;
}

bool llama_memory_completion::wait(ggml_backend_t consumer) {
    if (!valid() || !consumer || !impl->submitted) return false;
    if (!impl->event) return true;
    if (consumer->device != impl->producer->device || !consumer->iface.event_wait) return false;
    const auto found=std::find(impl->waiters.begin(),impl->waiters.end(),consumer);
    const bool insert=found == impl->waiters.end();
    if (insert) impl->waiters.push_back(consumer);
    ggml_backend_event_wait(consumer,impl->event);
    return true;
}

bool llama_memory_completion::release_waiter(ggml_backend_t consumer) noexcept {
    if (!valid() || !impl->event || !consumer) return false;
    const auto found=std::find(impl->waiters.begin(),impl->waiters.end(),consumer);
    if (found == impl->waiters.end()) return false;
    impl->waiters.erase(found);
    return true;
}

bool llama_memory_completion::synchronize() {
    if (!valid() || !impl->submitted) return false;
    if (!impl->ready) {
        ggml_backend_event_synchronize(impl->event);
        impl->ready = true;
    }
    return true;
}
