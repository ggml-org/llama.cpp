#include "../src/llama-memory-executor-cuda.h"
#include "testing.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <stdexcept>

#include <cstring>

using arena_ptr = std::unique_ptr<ggml_backend_memory_arena, decltype(&ggml_backend_memory_arena_free)>;
using lease_ptr = std::unique_ptr<ggml_backend_memory_lease, decltype(&ggml_backend_memory_lease_free)>;
using retirement = llama_memory_executor_status;

struct cuda_fixture {
    ggml_backend_ptr backend;
    ggml_context_ptr context;
    arena_ptr arena{nullptr, ggml_backend_memory_arena_free};
    lease_ptr lease{nullptr, ggml_backend_memory_lease_free};
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;
    size_t alignment;
    size_t extent = 16384;
    uint32_t flags;
    static constexpr size_t count = 1024;

    // Keep the allocation tiny; this test does not need to stop a production model.
    cuda_fixture(ggml_backend_dev_t device, bool persistent = false) :
        backend(ggml_backend_dev_init(device, nullptr)),
        alignment(ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend.get()))),
        flags(persistent ? GGML_BACKEND_MEMORY_REGION_PERSISTENT : 0) {
        GGML_ASSERT(backend && extent % alignment == 0);
        arena.reset(ggml_backend_memory_arena_new(ggml_backend_get_default_buffer_type(backend.get()), 4*extent));
        GGML_ASSERT(arena && repartition(0, extent));
        rebuild();
    }

    // Repartition must fail while a nonpersistent region is still pinned.
    bool repartition(size_t offset, size_t size) {
        GGML_ASSERT(ggml_backend_memory_arena_quiesce(arena.get()));
        GGML_ASSERT(ggml_backend_memory_arena_begin(arena.get(), GGML_BACKEND_MEMORY_PLAN_NONE));
        GGML_ASSERT(ggml_backend_memory_arena_reserve_at(arena.get(), 11, offset, size, alignment, flags, nullptr));
        const bool result = ggml_backend_memory_arena_commit(arena.get());
        GGML_ASSERT(ggml_backend_memory_arena_resume(arena.get()));
        return result;
    }

    // Rebuild metadata only after the previous adapter has retired its capture.
    void rebuild() {
        lease.reset(ggml_backend_memory_arena_acquire(arena.get(), 11));
        GGML_ASSERT(lease);
        context.reset(ggml_init({1024*1024, nullptr, true}));
        GGML_ASSERT(context);
        input = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, count);
        output = ggml_scale(context.get(), input, 2.0f);
        graph = ggml_new_graph_custom(context.get(), 16, false);
        ggml_build_forward_expand(graph, output);
        auto * buffer = ggml_backend_memory_lease_buffer(lease.get());
        auto * base = static_cast<char *>(ggml_backend_buffer_get_base(buffer));
        GGML_ASSERT(ggml_backend_tensor_alloc(buffer, input, base) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(ggml_backend_tensor_alloc(buffer, output, base + count*sizeof(float)) == GGML_STATUS_SUCCESS);
    }

    // Ordinary data writes are not changes to the capture's runtime revision.
    void set(float value) {
        std::vector<float> data(count, value);
        ggml_backend_tensor_set(input, data.data(), 0, data.size()*sizeof(float));
    }

    // Read after drain to check numerical replay, not just successful submission.
    void check(testing & t, float value) {
        std::vector<float> data(count);
        ggml_backend_tensor_get(output, data.data(), 0, data.size()*sizeof(float));
        t.assert_true(std::all_of(data.begin(), data.end(), [&](float x) { return x == value; }));
    }

    // Two stable calls complete native warmup; the third exercises replay.
    bool warm(testing & t, llama_memory_cuda_executor & adapter, uint64_t revision = 1) {
        set(3.0f);
        for (int i = 0; i < 3; ++i) {
            if (!t.assert_true(adapter.compute_async({lease.get()}, revision) == GGML_STATUS_SUCCESS)) return false;
            if (!t.assert_true(adapter.drain())) return false;
            check(t, 6.0f);
        }
        return true;
    }
};

// Inject an error after real CUDA submission to exercise retained ownership on failure.
struct submission_fault {
    using compute_t = ggml_status (*)(ggml_backend_t, ggml_cgraph *);
    inline static submission_fault * current = nullptr;
    ggml_backend_t backend;
    compute_t original;
    bool throws;

    submission_fault(ggml_backend_t backend, bool throws) :
        backend(backend), original(backend->iface.graph_compute), throws(throws) {
        GGML_ASSERT(current == nullptr);
        current = this;
        backend->iface.graph_compute = compute;
    }
    ~submission_fault() {
        backend->iface.graph_compute = original;
        current = nullptr;
    }
    submission_fault(const submission_fault &) = delete;
    submission_fault & operator=(const submission_fault &) = delete;

    static ggml_status compute(ggml_backend_t backend, ggml_cgraph * graph) {
        GGML_ASSERT(current && current->backend == backend);
        const auto result = current->original(backend, graph);
        if (result != GGML_STATUS_SUCCESS) return result;
        if (current->throws) throw std::runtime_error("after CUDA submission");
        return GGML_STATUS_FAILED;
    }
};

int main(int argc, char ** argv) {
    const bool use_cuda = argc > 1 && std::strcmp(argv[1], "--cuda") == 0;
    const bool expect_capture = !(argc > 2 && std::strcmp(argv[2], "--no-graphs") == 0);
    const bool expect_unsupported = argc > 2 && std::strcmp(argv[2], "--unsupported") == 0;
    testing t;

    t.test("unsupported_backend_rejects_without_allocation", [](testing & t) {
        ggml_backend_ptr cpu(ggml_backend_cpu_init());
        for (auto * backend : {cpu.get(), static_cast<ggml_backend_t>(nullptr)}) {
            llama_memory_cuda_executor adapter(backend);
            t.assert_true(!adapter.supported());
            t.assert_true(!adapter.bind(nullptr, {}, 0));
            t.assert_true(adapter.compute_async({}, 0) == GGML_STATUS_FAILED);
            t.assert_true(!adapter.ready() && !adapter.is_captured());
            t.assert_equal(size_t(0), adapter.outstanding());
            t.assert_true(adapter.retire().status == retirement::unchanged);
        }
    });

    if (!use_cuda) return t.summary();
    ggml_backend_load_all();
    auto * reg = ggml_backend_reg_by_name("CUDA");
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
        t.assert_true("Explicit CUDA test requires an available CUDA device", false);
        return t.summary();
    }
    auto * device = ggml_backend_reg_dev_get(reg, 0);
    if (expect_unsupported) {
        t.test("experimental_stream_optimizer_is_rejected", [&](testing & t) {
            ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
            llama_memory_cuda_executor adapter(backend.get());
            t.assert_true(!adapter.supported());
            t.assert_true(!adapter.bind(nullptr, {}, 0));
            t.assert_true(adapter.compute_async({}, 0) == GGML_STATUS_FAILED);
        });
        return t.summary();
    }

    t.test("real_capture_replay_and_data_updates", [&](testing & t) {
        cuda_fixture f(device);
        llama_memory_cuda_executor adapter(f.backend.get());
        t.assert_true(adapter.supported());
        if (!t.assert_true(adapter.bind(f.graph, {f.lease.get()}, 1))) return;
        t.assert_true(!adapter.is_captured());
        if (!f.warm(t, adapter)) return;
        t.assert_true(adapter.is_captured() == expect_capture);
        f.set(7.0f);
        t.assert_true(adapter.compute_async({f.lease.get()}, 1) == GGML_STATUS_SUCCESS);
        t.assert_equal(size_t(1), adapter.outstanding());
        t.assert_true(adapter.drain());
        f.check(t, 14.0f);
        t.assert_equal(size_t(0), adapter.outstanding());
    });

    t.test("stale_inputs_and_failed_rebind_preserve_capture", [&](testing & t) {
        cuda_fixture f(device);
        llama_memory_cuda_executor adapter(f.backend.get());
        t.assert_true(!adapter.bind(nullptr, {f.lease.get()}, 1));
        t.assert_true(!adapter.bind(f.graph, {nullptr}, 1));
        if (!t.assert_true(adapter.bind(f.graph, {f.lease.get(), f.lease.get()}, 1))) return;
        if (!f.warm(t, adapter)) return;
        t.assert_true(!adapter.bind(f.graph, {f.lease.get()}, 2));
        t.assert_true(adapter.compute_async({f.lease.get()}, 2) == GGML_STATUS_FAILED);
        t.assert_true(adapter.compute_async({}, 1) == GGML_STATUS_FAILED);
        t.assert_true(adapter.compute_async({nullptr}, 1) == GGML_STATUS_FAILED);
        t.assert_true(adapter.is_captured() == expect_capture);
        t.assert_true(adapter.compute_async({f.lease.get()}, 1) == GGML_STATUS_SUCCESS);
        t.assert_true(adapter.drain());
        f.check(t, 6.0f);
    });

    t.test("unrelated_changes_do_not_drain_or_invalidate", [&](testing & t) {
        cuda_fixture f(device);
        llama_memory_cuda_executor adapter(f.backend.get());
        if (!t.assert_true(adapter.bind(f.graph, {f.lease.get()}, 1))) return;
        if (!f.warm(t, adapter)) return;
        t.assert_true(adapter.compute_async({f.lease.get()}, 1) == GGML_STATUS_SUCCESS);
        t.assert_true(adapter.retire_if_affected({99}).status == retirement::unchanged);
        t.assert_equal(size_t(1), adapter.outstanding());
        t.assert_true(adapter.ready() && adapter.is_captured() == expect_capture);
        t.assert_true(adapter.retire_if_affected({11}).status == retirement::retired);
        t.assert_true(!adapter.is_captured() && !adapter.ready());
        t.assert_equal(size_t(0), adapter.outstanding());
        f.check(t, 6.0f);
    });

    t.test("queued_replays_hold_storage_until_native_retirement", [&](testing & t) {
        cuda_fixture f(device);
        llama_memory_cuda_executor adapter(f.backend.get());
        if (!t.assert_true(adapter.bind(f.graph, {f.lease.get()}, 1))) return;
        if (!f.warm(t, adapter)) return;
        for (int i = 0; i < 32; ++i) {
            t.assert_true(adapter.compute_async({f.lease.get()}, 1) == GGML_STATUS_SUCCESS);
        }
        t.assert_equal(size_t(1), adapter.outstanding());
        f.lease.reset();
        t.assert_true(!f.repartition(f.extent, 2*f.extent));
        adapter.quiesce();
        t.assert_true(adapter.compute_async({}, 1) == GGML_STATUS_FAILED);
        t.assert_true(adapter.retire().status == retirement::retired);
        t.assert_true(!adapter.is_captured());
        t.assert_true(f.repartition(f.extent, 2*f.extent));
        f.rebuild();
        if (!t.assert_true(adapter.bind(f.graph, {f.lease.get()}, 2))) return;
        if (!f.warm(t, adapter, 2)) return;
        t.assert_true(adapter.is_captured() == expect_capture);
    });

    t.test("persistent_capture_survives_unrelated_arena_commit", [&](testing & t) {
        cuda_fixture f(device, true);
        llama_memory_cuda_executor adapter(f.backend.get());
        if (!t.assert_true(adapter.bind(f.graph, {f.lease.get()}, 1))) return;
        if (!f.warm(t, adapter)) return;
        const auto old_generation = ggml_backend_memory_lease_generation(f.lease.get());
        t.assert_true(f.repartition(0, f.extent));
        lease_ptr fresh(ggml_backend_memory_arena_acquire(f.arena.get(), 11), ggml_backend_memory_lease_free);
        t.assert_true(ggml_backend_memory_lease_generation(fresh.get()) > old_generation);
        t.assert_true(adapter.compute_async({fresh.get()}, 1) == GGML_STATUS_SUCCESS);
        t.assert_true(adapter.is_captured() == expect_capture);
        t.assert_true(adapter.drain());
        f.check(t, 6.0f);
    });

    t.test("destructor_drains_and_retires_before_last_lease_release", [&](testing & t) {
        cuda_fixture f(device);
        {
            llama_memory_cuda_executor adapter(f.backend.get());
            if (!t.assert_true(adapter.bind(f.graph, {f.lease.get()}, 1))) return;
            if (!f.warm(t, adapter)) return;
            t.assert_true(adapter.compute_async({f.lease.get()}, 1) == GGML_STATUS_SUCCESS);
            f.lease.reset();
            t.assert_true(!f.repartition(f.extent, 2*f.extent));
        }
        t.assert_equal(size_t(0), ggml_backend_memory_arena_lease_count(f.arena.get()));
        t.assert_true(f.repartition(f.extent, 2*f.extent));
    });

    t.test("retiring_one_graph_preserves_other_native_cache_entries", [&](testing & t) {
        cuda_fixture f(device), g(device);
        llama_memory_cuda_executor first(f.backend.get()), second(f.backend.get());
        if (!t.assert_true(first.bind(f.graph, {f.lease.get()}, 1))) return;
        if (!t.assert_true(second.bind(g.graph, {g.lease.get()}, 1))) return;
        if (!f.warm(t, first) || !g.warm(t, second)) return;
        t.assert_true(first.is_captured() == expect_capture);
        t.assert_true(second.is_captured() == expect_capture);
        t.assert_true(first.retire().status == retirement::retired);
        t.assert_true(!first.is_captured());
        t.assert_true(second.is_captured() == expect_capture);
        t.assert_true(second.compute_async({g.lease.get()}, 1) == GGML_STATUS_SUCCESS);
        t.assert_true(second.drain());
        g.check(t, 6.0f);
    });

    t.test("drain_completes_backend_ordered_output_copy", [&](testing & t) {
        cuda_fixture f(device);
        llama_memory_cuda_executor adapter(f.backend.get());
        if (!t.assert_true(adapter.bind(f.graph, {f.lease.get()}, 1))) return;
        f.set(9.0f);
        std::vector<float> output(cuda_fixture::count, 0);
        t.assert_true(adapter.compute_async({f.lease.get()}, 1) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get_async(f.backend.get(), f.output, output.data(), 0, output.size()*sizeof(float));
        t.assert_true(adapter.drain());
        t.assert_true(std::all_of(output.begin(), output.end(), [](float x) { return x == 18.0f; }));
    });

    t.test("partial_submission_failure_keeps_pins_and_closes_admission", [&](testing & t) {
        for (bool throws : {false, true}) {
            cuda_fixture f(device);
            llama_memory_cuda_executor adapter(f.backend.get());
            if (!t.assert_true(adapter.bind(f.graph, {f.lease.get()}, 1))) return;
            if (!f.warm(t, adapter)) return;
            {
                submission_fault fault(f.backend.get(), throws);
                bool caught = false;
                try {
                    t.assert_true(adapter.compute_async({f.lease.get()}, 1) == GGML_STATUS_FAILED);
                } catch (const std::runtime_error &) {
                    caught = true;
                }
                t.assert_true(caught == throws);
            }
            t.assert_true(!adapter.ready());
            t.assert_equal(size_t(1), adapter.outstanding());
            t.assert_true(adapter.compute_async({f.lease.get()}, 1) == GGML_STATUS_FAILED);
            f.lease.reset();
            t.assert_true(!f.repartition(f.extent, 2*f.extent));
            t.assert_true(adapter.retire().status == retirement::retired);
            t.assert_true(!adapter.is_captured());
            t.assert_true(f.repartition(f.extent, 2*f.extent));
        }
    });

    return t.summary();
}
