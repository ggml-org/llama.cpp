#include "../src/llama-memory-completion.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "testing.h"
#include "ggml-cpp.h"

#include <cstring>
#include <stdexcept>
#include <memory>

struct fake_event_state {
    bool recorded = false;
};

struct fake_backend {
    ggml_backend_device device{};
    ggml_backend backend{};
    size_t event_new = 0;
    size_t event_free = 0;
    size_t event_record = 0;
    size_t event_wait = 0;
    size_t event_sync = 0;
    bool throw_wait = false;
    size_t backend_sync = 0;
    bool fail_event_new = false;

    explicit fake_backend(bool events = true) {
        device.context = this;
        backend.context = this;
        backend.device = &device;
        backend.iface.synchronize = [](ggml_backend_t value) {
            ++static_cast<fake_backend *>(value->context)->backend_sync;
        };
        if (!events) return;
        device.iface.event_new = [](ggml_backend_dev_t value) -> ggml_backend_event_t {
            auto * state = static_cast<fake_backend *>(value->context);
            ++state->event_new;
            if (state->fail_event_new) return nullptr;
            return new ggml_backend_event{value,new fake_event_state};
        };
        device.iface.event_free = [](ggml_backend_dev_t value,ggml_backend_event_t event) {
            ++static_cast<fake_backend *>(value->context)->event_free;
            delete static_cast<fake_event_state *>(event->context);
            delete event;
        };
        device.iface.event_synchronize = [](ggml_backend_dev_t value,ggml_backend_event_t event) {
            auto * state = static_cast<fake_backend *>(value->context);
            GGML_ASSERT(static_cast<fake_event_state *>(event->context)->recorded);
            ++state->event_sync;
        };
        backend.iface.event_record = [](ggml_backend_t value,ggml_backend_event_t event) {
            auto * state = static_cast<fake_backend *>(value->context);
            static_cast<fake_event_state *>(event->context)->recorded = true;
            ++state->event_record;
        };
        backend.iface.event_wait = [](ggml_backend_t value,ggml_backend_event_t event) {
            auto * state = static_cast<fake_backend *>(value->context);
            GGML_ASSERT(static_cast<fake_event_state *>(event->context)->recorded);
            ++state->event_wait;
            if (state->throw_wait) throw std::runtime_error("wait");
        };
    }

    ggml_backend_t get() { return &backend; }
};

int main(int argc, char ** argv) {
    testing t;

    t.test("invalid_and_synchronous_fallback_are_explicit", [](testing & t) {
        t.assert_true(!llama_memory_completion::create(nullptr));
        fake_backend producer(false),consumer(false);
        auto value=llama_memory_completion::create(producer.get());
        if (!t.assert_true(value != nullptr)) return;
        t.assert_true(!value->event_backed());
        t.assert_true(!value->recorded() && !value->host_ready());
        t.assert_true(!value->wait(consumer.get()));
        t.assert_true(value->record());
        t.assert_equal(size_t(1),producer.backend_sync);
        t.assert_true(value->recorded() && value->host_ready());
        t.assert_true(value->wait(consumer.get()));
        t.assert_equal(size_t(0),consumer.backend_sync);
        t.assert_true(value->synchronize());
        t.assert_equal(size_t(1),producer.backend_sync);
        t.assert_true(!value->record());
    });

    t.test("backend_wait_does_not_change_logical_host_visibility", [](testing & t) {
        fake_backend backend;
        {
            auto value=llama_memory_completion::create(backend.get());
            if (!t.assert_true(value != nullptr)) return;
            t.assert_true(value->event_backed());
            t.assert_equal(size_t(1),backend.event_new);
            t.assert_true(!value->wait(backend.get()));
            t.assert_true(value->record());
            t.assert_true(value->recorded() && !value->host_ready());
            t.assert_equal(size_t(1),backend.event_record);
            t.assert_equal(size_t(0),backend.backend_sync);
            t.assert_true(value->wait(backend.get()));
            t.assert_equal(size_t(1),backend.event_wait);
            t.assert_equal(size_t(0),backend.event_sync);
            t.assert_true(!value->host_ready());
            t.assert_true(value->synchronize());
            t.assert_true(value->host_ready());
            t.assert_equal(size_t(1),backend.event_sync);
            t.assert_true(value->synchronize());
            t.assert_equal(size_t(1),backend.event_sync);
        }
        t.assert_equal(size_t(1),backend.event_free);
        t.assert_equal(size_t(1),backend.event_sync);
        t.assert_equal(size_t(1),backend.backend_sync);
    });

    t.test("wait_rejects_another_device_or_missing_consumer_capability", [](testing & t) {
        fake_backend producer,foreign,consumer;
        consumer.backend.device=&producer.device;
        auto value=llama_memory_completion::create(producer.get());
        if (!t.assert_true(value && value->record())) return;
        t.assert_true(!value->wait(foreign.get()));
        t.assert_equal(size_t(0),foreign.event_wait);
        t.assert_true(value->wait(consumer.get()));
        t.assert_equal(size_t(1),consumer.event_wait);
        consumer.backend.iface.event_wait=nullptr;
        t.assert_true(!value->wait(consumer.get()));
        t.assert_equal(size_t(1),consumer.event_wait);
        t.assert_true(!value->wait(nullptr));
    });

    t.test("event_allocation_failure_uses_synchronous_fallback", [](testing & t) {
        fake_backend backend; backend.fail_event_new=true;
        auto value=llama_memory_completion::create(backend.get());
        if (!t.assert_true(value != nullptr)) return;
        t.assert_equal(size_t(1),backend.event_new);
        t.assert_true(!value->event_backed());
        t.assert_true(value->record());
        t.assert_equal(size_t(1),backend.backend_sync);
        t.assert_equal(size_t(0),backend.event_record);
        t.assert_equal(size_t(0),backend.event_free);
    });

    t.test("incomplete_event_interface_does_not_allocate_an_event", [](testing & t) {
        fake_backend backend;
        backend.device.iface.event_synchronize=nullptr;
        auto value=llama_memory_completion::create(backend.get());
        if (!t.assert_true(value != nullptr)) return;
        t.assert_true(!value->event_backed());
        t.assert_equal(size_t(0),backend.event_new);
        t.assert_true(value->record());
        t.assert_equal(size_t(1),backend.backend_sync);
    });

    t.test("released_waiter_is_not_synchronized_during_teardown", [](testing & t) {
        fake_backend producer,consumer;
        consumer.backend.device=&producer.device;
        {
            auto value=llama_memory_completion::create(producer.get());
            if (!t.assert_true(value && value->record() && value->wait(consumer.get()))) return;
            t.assert_true(value->wait(consumer.get()));
            t.assert_equal(size_t(2),consumer.event_wait);
            t.assert_true(!value->release_waiter(producer.get()));
            ggml_backend_synchronize(consumer.get());
            t.assert_true(value->release_waiter(consumer.get()));
            t.assert_true(!value->release_waiter(consumer.get()));
        }
        t.assert_equal(size_t(1),consumer.backend_sync);
        t.assert_equal(size_t(1),producer.event_sync);
        t.assert_equal(size_t(1),producer.event_free);
    });

    t.test("throwing_backend_wait_remains_tracked_for_safe_teardown", [](testing & t) {
        fake_backend producer,consumer;
        consumer.backend.device=&producer.device;
        consumer.throw_wait=true;
        bool caught=false;
        {
            auto value=llama_memory_completion::create(producer.get());
            if (!t.assert_true(value && value->record())) return;
            try { value->wait(consumer.get()); }
            catch (const std::runtime_error &) { caught=true; }
            t.assert_true(caught);
            t.assert_equal(size_t(1),consumer.event_wait);
        }
        t.assert_equal(size_t(1),consumer.backend_sync);
        t.assert_equal(size_t(1),producer.event_sync);
        t.assert_equal(size_t(1),producer.event_free);
    });

    t.test("destruction_waits_for_a_recorded_event", [](testing & t) {
        fake_backend backend;
        {
            auto value=llama_memory_completion::create(backend.get());
            if (!t.assert_true(value && value->record())) return;
            t.assert_equal(size_t(0),backend.event_sync);
        }
        t.assert_equal(size_t(1),backend.event_sync);
        t.assert_equal(size_t(1),backend.event_free);
    });

    t.test("move_transfers_exactly_one_event_owner", [](testing & t) {
        fake_backend first,second;
        auto a=llama_memory_completion::create(first.get());
        auto b=llama_memory_completion::create(second.get());
        if (!t.assert_true(a && b && a->record() && b->record())) return;
        *b=std::move(*a);
        t.assert_true(!a->valid());
        t.assert_true(b->valid() && b->event_backed() && b->recorded());
        t.assert_equal(size_t(1),second.event_sync);
        t.assert_equal(size_t(1),second.event_free);
        b.reset(); a.reset();
        t.assert_equal(size_t(1),first.event_sync);
        t.assert_equal(size_t(1),first.event_free);
    });

    t.test("real_cpu_uses_the_synchronous_fallback", [](testing & t) {
        ggml_backend_ptr backend(ggml_backend_cpu_init());
        auto value=llama_memory_completion::create(backend.get());
        if (!t.assert_true(value != nullptr)) return;
        t.assert_true(!value->event_backed());
        t.assert_true(value->record());
        t.assert_true(value->recorded() && value->host_ready());
        t.assert_true(value->wait(backend.get()));
        t.assert_true(value->synchronize());
    });

    if (argc > 1 && std::strcmp(argv[1],"--cuda") == 0) {
        ggml_backend_load_all();
        auto * device=ggml_backend_dev_by_name("CUDA0");
        ggml_backend_ptr producer(device ? ggml_backend_dev_init(device,nullptr) : nullptr);
        ggml_backend_ptr consumer(device ? ggml_backend_dev_init(device,nullptr) : nullptr);
        t.test("real_cuda_records_waits_and_synchronizes_one_event", [&](testing & t) {
            if (!t.assert_true(producer && consumer)) return;
            auto value=llama_memory_completion::create(producer.get());
            if (!t.assert_true(value && value->event_backed())) return;
            t.assert_true(value->record());
            t.assert_true(value->wait(consumer.get()));
            t.assert_true(!value->host_ready());
            t.assert_true(value->synchronize() && value->host_ready());
        });
    }

    return t.summary();
}
