#include "../src/llama-kv-stream-binding.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "testing.h"

#include <cstring>
#include <stdexcept>

using arena_ptr = std::unique_ptr<ggml_backend_memory_arena, decltype(&ggml_backend_memory_arena_free)>;
using lease_ptr = std::unique_ptr<ggml_backend_memory_lease, decltype(&ggml_backend_memory_lease_free)>;
using result = llama_memory_executor_status;

// Small aligned F16 pages exercise the same policy contract without a model or large allocation.
static llama_kv_stream_policy_config config() {
    llama_kv_stream_policy_config c;
    c.shape = {GGML_TYPE_F16, GGML_TYPE_F16, 64, 64, 1, 1, 128};
    c.capabilities = {{GGML_TYPE_F16, true, true, true, true}, {GGML_TYPE_F16, true, true, true, true}, true, true};
    c.pool_bytes = 4096;
    c.layers = 2;
    return c;
}

struct fixture {
    ggml_backend_buffer_type_t type;
    size_t bias = 0;
    arena_ptr arena{nullptr, ggml_backend_memory_arena_free};
    lease_ptr lease{nullptr, ggml_backend_memory_lease_free};
    // Keep a nonzero parent offset to catch accidental double-offset arithmetic.
    explicit fixture(ggml_backend_buffer_type_t type = ggml_backend_cpu_buffer_type(), size_t offset = 128, size_t size = 4096) : type(type) {
        arena.reset(ggml_backend_memory_arena_new(type, 16384));
        GGML_ASSERT(arena);
        bias = (128 - reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(ggml_backend_memory_arena_parent(arena.get()))) % 128) % 128;
        GGML_ASSERT(place(offset, size));
        lease.reset(ggml_backend_memory_arena_acquire(arena.get(), 7));
        GGML_ASSERT(lease);
    }
    // Failed commits keep the old placement intact.
    bool place(size_t offset, size_t size) {
        GGML_ASSERT(ggml_backend_memory_arena_quiesce(arena.get()));
        GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
        GGML_ASSERT(ggml_backend_memory_arena_reserve_at(arena.get(), 7, offset + bias, size, ggml_backend_buft_get_alignment(type), 0, nullptr));
        const bool ok = ggml_backend_memory_arena_commit(arena.get());
        if (!ok) ggml_backend_memory_arena_rollback(arena.get());
        GGML_ASSERT(ggml_backend_memory_arena_resume(arena.get()));
        return ok;
    }
};

// Override only a test view's address lookup; restore it before the fixture releases the buffer.
struct base_probe {
    ggml_backend_buffer_t buffer;
    decltype(ggml_backend_buffer_i::get_base) original;
    void * address;
    size_t calls = 0;
    inline static base_probe * active = nullptr;
    explicit base_probe(ggml_backend_buffer_t buffer) :
        buffer(buffer), original(buffer->iface.get_base), address(ggml_backend_buffer_get_base(buffer)) {
        GGML_ASSERT(!active);
        active = this;
        buffer->iface.get_base = [](ggml_backend_buffer_t) { ++active->calls; return active->address; };
    }
    ~base_probe() { buffer->iface.get_base = original; active = nullptr; }
};

struct events {
    std::vector<int> order;
    int created = 0, destroyed = 0;
    std::function<void()> on_destroy;
};
struct native : llama_memory_executable {
    events & seen;
    llama_kv_stream_binding_view snapshot;
    std::shared_ptr<int> host;
    native(events & seen, const llama_kv_stream_binding_view & view, std::shared_ptr<int> host) : seen(seen), snapshot(view), host(std::move(host)) {
        ++seen.created;
    }
    // Native references disappear while the execution guard still retains the lease.
    ~native() override {
        if (seen.on_destroy) seen.on_destroy();
        seen.order.push_back(2);
        ++seen.destroyed;
    }
};
static llama_kv_stream_binding_factory factory(events & seen, std::shared_ptr<int> host = std::make_shared<int>(19)) {
    return [&seen, host](const llama_kv_stream_binding_view & view) { return std::make_unique<native>(seen, view, host); };
}
struct fake_backend : llama_memory_executor_backend {
    events & seen;
    llama_memory_execution pending;
    bool fail = false, throws = false, hold_pin = false;
    std::function<void()> callback;
    explicit fake_backend(events & seen) : seen(seen) {}
    // Completion releases the pin; a failed drain must keep it alive.
    bool drain() override {
        seen.order.push_back(1);
        if (callback) callback();
        if (throws) throw std::runtime_error("injected drain failure");
        if (fail) return false;
        if (!hold_pin) pending.reset();
        return true;
    }
};

int main(int argc, char ** argv) {
    const bool cuda = argc > 1 && std::strcmp(argv[1], "--cuda") == 0;
    testing t;
    t.test("resolver_has_no_cpu_fallback", [](testing & t) {
        ggml_backend_ptr cpu(ggml_backend_cpu_init());
        t.assert_true(llama_kv_stream_device_buffer_type(nullptr) == nullptr);
        t.assert_true(llama_kv_stream_device_buffer_type(ggml_backend_get_device(cpu.get())) == nullptr);
    });
    t.test("binding_caches_one_region_and_validated_policy", [](testing & t) {
        fixture f; events seen; fake_backend backend(seen);
        llama_kv_stream_binding b(31, f.type);
        const auto generation = ggml_backend_memory_arena_generation(f.arena.get());
        auto * base = ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(f.lease.get()));
        if (!t.assert_true(b.bind(f.lease.get(), config(), factory(seen)))) return;
        t.assert_equal(uint64_t(31), b.view()->cache_id);
        t.assert_equal(uint64_t(1), b.view()->revision);
        t.assert_true(b.view()->lease == f.lease.get());
        t.assert_true(b.view()->base == base);
        t.assert_equal(size_t(4096), b.view()->capacity);
        t.assert_equal(uint32_t(16), b.view()->initial_policy.budget.pages);
        f.lease.reset();
        for (int i = 0; i < 1000; ++i) {
            auto pin = b.acquire();
            t.assert_true(bool(pin));
            t.assert_equal(size_t(1), ggml_backend_memory_arena_lease_count(f.arena.get()));
        }
        t.assert_equal(generation, ggml_backend_memory_arena_generation(f.arena.get()));
        seen.on_destroy = [&] { t.assert_equal(size_t(1), ggml_backend_memory_arena_lease_count(f.arena.get())); };
        t.assert_true(b.detach(backend).status == result::retired);
        t.assert_equal(size_t(0), ggml_backend_memory_arena_lease_count(f.arena.get()));
        t.assert_true(seen.order == std::vector<int>({1, 2}));
        t.assert_true(!b.ready() && b.view() == nullptr && !b.acquire());
    });
    t.test("invalid_grants_do_not_construct_resources", [](testing & t) {
        fixture f; events seen;
        for (int bad = 0; bad < 6; ++bad) {
            llama_kv_stream_binding b(bad == 0 ? 0 : 31, bad == 1 ? nullptr : f.type);
            auto c = config();
            if (bad == 3) c.pool_bytes = 8192;
            if (bad == 4) c.layers = 0;
            if (bad == 5) c.shape.alignment = 3;
            t.assert_true(!b.bind(bad == 2 ? nullptr : f.lease.get(), c, factory(seen)));
            t.assert_true(!b.ready());
        }
        t.assert_equal(0, seen.created);
        t.assert_equal(size_t(1), ggml_backend_memory_arena_lease_count(f.arena.get()));
    });
    t.test("absolute_alignment_not_just_region_offset", [](testing & t) {
        // CPU guarantees 64-byte alignment; an odd 64-byte offset cannot satisfy 128-byte alignment in both views.
        fixture a(ggml_backend_cpu_buffer_type(), 64), b(ggml_backend_cpu_buffer_type(), 128);
        events seen;
        for (auto * f : {&a, &b}) {
            auto * base = ggml_backend_buffer_get_base(ggml_backend_memory_lease_buffer(f->lease.get()));
            llama_kv_stream_binding binding(31, f->type);
            const bool aligned = reinterpret_cast<uintptr_t>(base) % 128 == 0;
            t.assert_equal(aligned, binding.bind(f->lease.get(), config(), factory(seen)));
        }
    });
    t.test("factory_failure_and_exception_preserve_unbound_state", [](testing & t) {
        fixture f; events seen;
        llama_kv_stream_binding b(31, f.type);
        t.assert_true(!b.bind(f.lease.get(), config(), {}));
        t.assert_true(!b.bind(f.lease.get(), config(), [](const auto &) { return nullptr; }));
        bool caught = false;
        try { b.bind(f.lease.get(), config(), [](const auto &) -> std::unique_ptr<llama_memory_executable> { throw std::runtime_error("factory"); }); }
        catch (const std::runtime_error &) { caught = true; }
        t.assert_true(caught && !b.ready());
        t.assert_true(b.bind(f.lease.get(), config(), factory(seen)));
    });
    t.test("live_binding_cannot_be_replaced", [](testing & t) {
        fixture f; events seen; fake_backend backend(seen);
        llama_kv_stream_binding b(31, f.type);
        if (!t.assert_true(b.bind(f.lease.get(), config(), factory(seen)))) return;
        auto * view = b.view();
        t.assert_true(!b.bind(f.lease.get(), config(), factory(seen)));
        t.assert_true(b.view() == view && b.ready());
        t.assert_equal(1, seen.created);
    });
    t.test("detach_failures_retain_storage_and_close_admission", [](testing & t) {
        fixture f; events seen; fake_backend backend(seen);
        llama_kv_stream_binding b(31, f.type);
        if (!t.assert_true(b.bind(f.lease.get(), config(), factory(seen)))) return;
        backend.pending = b.acquire(); f.lease.reset();
        backend.fail = true;
        t.assert_true(b.detach(backend).status == result::drain_failed);
        t.assert_true(!b.ready() && !b.acquire() && b.view() == nullptr);
        t.assert_true(!f.place(256, 8192));
        backend.fail = false; backend.throws = true;
        auto failed = b.detach(backend);
        t.assert_true(failed.status == result::drain_failed && failed.exception != nullptr);
        backend.throws = false; backend.hold_pin = true;
        t.assert_true(b.detach(backend).status == result::pending);
        t.assert_equal(0, seen.destroyed);
        backend.hold_pin = false;
        t.assert_true(b.detach(backend).status == result::retired);
        t.assert_true(f.place(256, 8192));
    });
    t.test("rebind_changes_device_revision_not_host_identity", [](testing & t) {
        fixture f; events seen; fake_backend backend(seen);
        auto host = std::make_shared<int>(19);
        llama_kv_stream_binding b(31, f.type);
        if (!t.assert_true(b.bind(f.lease.get(), config(), factory(seen, host)))) return;
        const auto first = *b.view();
        f.lease.reset();
        t.assert_true(b.detach(backend).status == result::retired);
        t.assert_equal(uint64_t(31), b.cache_id());
        if (!t.assert_true(f.place(256, 8192))) return;
        f.lease.reset(ggml_backend_memory_arena_acquire(f.arena.get(), 7));
        auto c = config(); c.pool_bytes = 8192;
        if (!t.assert_true(b.bind(f.lease.get(), c, factory(seen, host)))) return;
        t.assert_equal(first.revision + 1, b.view()->revision);
        t.assert_equal(first.cache_id, b.view()->cache_id);
        t.assert_true(first.base != b.view()->base);
        t.assert_equal(size_t(8192), b.view()->capacity);
        auto pin = b.acquire();
        t.assert_true(static_cast<native *>(pin.executable())->host == host);
    });
    t.test("execution_pin_outlives_binding_and_retains_host", [](testing & t) {
        fixture f; events seen;
        std::weak_ptr<int> weak;
        llama_memory_execution pending;
        {
            auto host = std::make_shared<int>(19); weak = host;
            llama_kv_stream_binding b(31, f.type);
            if (!t.assert_true(b.bind(f.lease.get(), config(), factory(seen, host)))) return;
            pending = b.acquire(); f.lease.reset();
        }
        t.assert_true(!weak.expired());
        t.assert_true(!f.place(256, 8192));
        t.assert_equal(0, seen.destroyed);
        pending.reset();
        t.assert_true(weak.expired());
        t.assert_equal(1, seen.destroyed);
        t.assert_true(f.place(256, 8192));
    });
    t.test("callbacks_cannot_reenter_binding", [](testing & t) {
        fixture f; events seen; fake_backend backend(seen);
        llama_kv_stream_binding b(31, f.type);
        if (!t.assert_true(b.bind(f.lease.get(), config(), [&](const auto & view) {
            t.assert_true(!b.bind(f.lease.get(), config(), factory(seen)));
            t.assert_true(b.detach(backend).status == result::busy);
            t.assert_true(!b.acquire());
            return std::make_unique<native>(seen, view, std::make_shared<int>(19));
        }))) return;
        backend.callback = [&] { t.assert_true(b.detach(backend).status == result::busy); };
        t.assert_true(b.detach(backend).status == result::retired);
    });
    t.test("address_failures_are_atomic_and_lookup_is_cached", [](testing & t) {
        fixture f; events seen; fake_backend backend(seen);
        base_probe probe(ggml_backend_memory_lease_buffer(f.lease.get()));
        const auto actual = probe.address;
        llama_kv_stream_binding b(31, f.type);
        auto lookup = probe.buffer->iface.get_base;
        probe.buffer->iface.get_base = nullptr;
        t.assert_true(!b.bind(f.lease.get(), config(), factory(seen)));
        probe.buffer->iface.get_base = lookup;
        for (auto address : {uintptr_t(129), UINTPTR_MAX - 127}) {
            probe.address = reinterpret_cast<void *>(address);
            t.assert_true(!b.bind(f.lease.get(), config(), factory(seen)));
        }
        probe.address = actual;
        probe.calls = 0;
        if (!t.assert_true(b.bind(f.lease.get(), config(), factory(seen)))) return;
        for (int i = 0; i < 1000; ++i) {
            auto pin = b.acquire();
            t.assert_true(bool(pin) && b.view()->base == actual);
        }
        t.assert_equal(size_t(1), probe.calls);
        t.assert_true(b.detach(backend).status == result::retired);
    });
    t.test("wrong_type_and_inconsistent_view_size_are_rejected", [](testing & t) {
        fixture f; events seen;
        auto other = *f.type;
        llama_kv_stream_binding wrong(31, &other), correct(31, f.type);
        t.assert_true(!wrong.bind(f.lease.get(), config(), factory(seen)));
        auto * buffer = ggml_backend_memory_lease_buffer(f.lease.get());
        const auto original = buffer->size;
        buffer->size = original - 1;
        t.assert_true(!correct.bind(f.lease.get(), config(), factory(seen)));
        buffer->size = original;
        t.assert_equal(0, seen.created);
    });
    t.test("larger_grant_does_not_expand_the_requested_pool", [](testing & t) {
        fixture f(ggml_backend_cpu_buffer_type(), 128, 8192); events seen;
        llama_kv_stream_binding b(31, f.type);
        auto c = config(); c.capabilities.direct_pair = false;
        if (!t.assert_true(b.bind(f.lease.get(), c, factory(seen)))) return;
        t.assert_equal(size_t(4096), b.view()->capacity);
        t.assert_equal(size_t(256), b.view()->initial_policy.budget.page.conversion.bytes);
        t.assert_equal(uint32_t(15), b.view()->initial_policy.budget.pages);
    });
    t.test("construction_retains_the_lease_even_if_caller_releases_it", [](testing & t) {
        fixture f; events seen;
        llama_kv_stream_binding b(31, f.type);
        t.assert_true(!b.bind(f.lease.get(), config(), [&](const auto &) -> std::unique_ptr<llama_memory_executable> {
            f.lease.reset();
            t.assert_true(!f.place(256, 8192));
            throw std::bad_alloc();
        }));
        t.assert_true(!b.ready());
        t.assert_equal(size_t(0), ggml_backend_memory_arena_lease_count(f.arena.get()));
        t.assert_true(f.place(256, 8192));
        f.lease.reset(ggml_backend_memory_arena_acquire(f.arena.get(), 7));
        if (!t.assert_true(b.bind(f.lease.get(), config(), factory(seen)))) return;
        t.assert_equal(uint64_t(1), b.view()->revision);
    });
    t.test("quiesce_closes_acquisition_and_detach_is_idempotent", [](testing & t) {
        fixture f; events seen; fake_backend backend(seen);
        llama_kv_stream_binding b(31, f.type);
        t.assert_true(b.detach(backend).status == result::unchanged);
        if (!t.assert_true(b.bind(f.lease.get(), config(), factory(seen)))) return;
        b.quiesce();
        t.assert_true(!b.ready() && b.view() == nullptr && !b.acquire());
        t.assert_true(seen.order.empty());
        t.assert_true(b.detach(backend).status == result::retired);
        t.assert_true(b.detach(backend).status == result::unchanged);
        t.assert_true(seen.order == std::vector<int>({1, 2}));
    });
    if (cuda) {
        ggml_backend_load_all();
        auto * reg = ggml_backend_reg_by_name("CUDA");
        if (!t.assert_true(reg && ggml_backend_reg_dev_count(reg))) return t.summary();
        auto * device = ggml_backend_reg_dev_get(reg, 0);
        auto * type = llama_kv_stream_device_buffer_type(device);
        t.test("cuda_device_local_lease_and_default_type_rejection", [&](testing & t) {
            if (!t.assert_true(type != nullptr)) return;
            ggml_backend_ptr gpu(ggml_backend_dev_init(device, nullptr));
            if (!t.assert_true(gpu != nullptr)) return;
            fixture f(type); events seen; fake_backend backend(seen);
            std::vector<float> values(16, 3.5f), copied(16);
            llama_kv_stream_binding b(31, type);
            fixture ordinary(ggml_backend_dev_buffer_type(device));
            t.assert_true(!b.bind(ordinary.lease.get(), config(), factory(seen)));
            if (!t.assert_true(b.bind(f.lease.get(), config(), factory(seen)))) return;
            ggml_backend_buffer_clear(b.view()->buffer, 0x17);
            ggml_context_ptr ctx(ggml_init({4096, nullptr, true}));
            if (!t.assert_true(ctx != nullptr)) return;
            auto * tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 16);
            if (!t.assert_true(ggml_backend_tensor_alloc(b.view()->buffer, tensor, b.view()->base) == GGML_STATUS_SUCCESS)) return;
            backend.pending = b.acquire();
            ggml_backend_tensor_set_async(gpu.get(), tensor, values.data(), 0, values.size()*sizeof(float));
            ggml_backend_tensor_get_async(gpu.get(), tensor, copied.data(), 0, copied.size()*sizeof(float));
            backend.callback = [&] { ggml_backend_synchronize(gpu.get()); };
            seen.on_destroy = [&] {
                t.assert_true(copied == values);
                t.assert_equal(size_t(1), ggml_backend_memory_arena_lease_count(f.arena.get()));
            };
            f.lease.reset();
            t.assert_true(b.detach(backend).status == result::retired);
            t.assert_equal(size_t(0), ggml_backend_memory_arena_lease_count(f.arena.get()));
        });
    }
    if (cuda) {
        auto * reg = ggml_backend_reg_by_name("CUDA");
        if (reg && ggml_backend_reg_dev_count(reg) > 1) t.test("cuda_wrong_device_lease_is_rejected", [&](testing & t) {
            auto * first = llama_kv_stream_device_buffer_type(ggml_backend_reg_dev_get(reg, 0));
            auto * second = llama_kv_stream_device_buffer_type(ggml_backend_reg_dev_get(reg, 1));
            if (!t.assert_true(first && second && first != second)) return;
            fixture f(first); events seen;
            llama_kv_stream_binding b(31, second);
            t.assert_true(!b.bind(f.lease.get(), config(), factory(seen)));
            t.assert_equal(0, seen.created);
        });
    }
    return t.summary();
}
