#include "../src/llama-context-memory.h"
#include "../ggml/src/ggml-cuda-graph.h"
#include "testing.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cstring>

struct fixture {
    ggml_backend_ptr device;
    ggml_backend_ptr cpu{ggml_backend_cpu_init()};
    std::vector<ggml_backend_t> backends;
    std::vector<ggml_backend_buffer_type_t> bufts;
    ggml_backend_sched_ptr sched;
    std::vector<ggml_backend_memory_workspace_group> groups;
    ggml_context_ptr ctx;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_cgraph * graph = nullptr;

    explicit fixture(ggml_backend_dev_t dev) : device(dev ? ggml_backend_dev_init(dev, nullptr) : ggml_backend_cpu_init()) {
        GGML_ASSERT(device && cpu);
        backends = {device.get(), cpu.get()};
        bufts = {ggml_backend_get_default_buffer_type(device.get()), ggml_backend_get_default_buffer_type(cpu.get())};
        sched.reset(ggml_backend_sched_new(backends.data(), bufts.data(), 2, 256, false, true));
        GGML_ASSERT(sched);
        size_t sizes[] = {8192, 8192, 4096, 4096}, count = 2;
        groups.resize(count);
        GGML_ASSERT(ggml_backend_memory_plan_workspace_groups(bufts.data(), sizes, 2, 2, groups.data(), &count));
        groups.resize(count);
    }

    // Exercise graph metadata reconstruction inside one unchanged scheduler-workspace lifetime.
    void rebuild() {
        ggml_backend_sched_synchronize(sched.get());
        ggml_backend_sched_reset(sched.get());
        ctx.reset(ggml_init({1024*1024, nullptr, true}));
        input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 16);
        output = ggml_scale(ctx.get(), input, 3.0f);
        ggml_set_input(input);
        ggml_set_output(input); // Preserve the input across repeated replay; otherwise GGML may reuse it in place.
        ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx.get(), 16, false);
        ggml_build_forward_expand(graph, output);
        ggml_backend_sched_set_tensor_backend(sched.get(), input, device.get());
        ggml_backend_sched_set_tensor_backend(sched.get(), output, device.get());
        GGML_ASSERT(ggml_backend_sched_reserve(sched.get(), graph));
        // Reservation resets scheduler assignments; restore the explicit device placement before allocation.
        ggml_backend_sched_set_tensor_backend(sched.get(), input, device.get());
        ggml_backend_sched_set_tensor_backend(sched.get(), output, device.get());
        GGML_ASSERT(ggml_backend_sched_alloc_graph(sched.get(), graph));
        GGML_ASSERT(input->data != output->data);
    }

    // Real scheduler dispatch with output-copy completion checked after the owner drains.
    void run(testing & t, llama_context_memory & owner, float value) {
        float data[16], result[16];
        for (float & x : data) x = value;
        ggml_backend_tensor_set(input, data, 0, sizeof(data));
        for (int i = 0; i < 3; ++i) t.assert_true(owner.compute_async(graph) == GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get_async(device.get(), output, result, 0, sizeof(result));
        owner.synchronize();
        for (float x : result) t.assert_equal(value*3, x);
    }
};

// Paired same-binary microbenchmark: dispatch cost only, not full-model tokens per second.
static void benchmark(ggml_backend_dev_t dev) {
    ggml_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    for (bool coordinated : {false, true, true, false}) {
        fixture f(dev);
        std::unique_ptr<llama_context_memory> owner;
        std::vector<llama_compute_arena_binding> legacy;
        if (coordinated) {
            owner = llama_context_memory::create(f.sched.get(), f.backends, f.groups);
            GGML_ASSERT(owner);
        } else {
            GGML_ASSERT(llama_prepare_compute_arena_bindings(f.sched.get(), f.backends, f.groups, legacy));
        }
        f.rebuild();
        float input[16];
        for (float & x : input) x = 1;
        ggml_backend_tensor_set(f.input, input, 0, sizeof(input));
        auto dispatch = [&] {
            const auto status = owner ? owner->compute_async(f.graph) : ggml_backend_sched_graph_compute_async(f.sched.get(), f.graph);
            GGML_ASSERT(status == GGML_STATUS_SUCCESS);
        };
        auto drain = [&] {
            if (owner) owner->synchronize();
            else ggml_backend_sched_synchronize(f.sched.get());
        };
        for (int i = 0; i < 64; ++i) dispatch();
        drain();
        double timings[2] = {};
        constexpr int iterations = 10000;
        for (int mode = 0; mode < 2; ++mode) {
            const auto begin = std::chrono::steady_clock::now();
            for (int i = 0; i < iterations; ++i) {
                dispatch();
                if (mode) drain();
            }
            drain();
            timings[mode] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count() / iterations;
        }
        std::printf("workspace-bench,%s,queued_us=%.3f,synchronized_us=%.3f\n",
            coordinated ? "coordinated" : "legacy", timings[0], timings[1]);
        if (!owner) {
            if (dev) {
                auto release = reinterpret_cast<ggml_backend_cuda_graph_release_all_t>(
                    ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_graph_release_all"));
                GGML_ASSERT(release);
                release(f.device.get());
            }
            for (const auto & binding : legacy) {
                GGML_ASSERT(ggml_backend_sched_detach_memory_lease(f.sched.get(), f.backends[binding.first_slot]));
            }
        }
    }
}

int main(int argc, char ** argv) {
    testing t;
    ggml_backend_dev_t dev = nullptr;
    if (argc > 1 && std::strcmp(argv[1], "--cuda") == 0) {
        ggml_backend_load_all();
        auto * reg = ggml_backend_reg_by_name("CUDA");
        if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
            t.assert_true("CUDA required", false);
            return t.summary();
        }
        dev = ggml_backend_reg_dev_get(reg, 0);
    }

    t.test("unsupported_and_invalid_inputs_reject_without_mutation", [&](testing & t) {
        fixture f(dev);
        t.assert_true(!llama_context_memory::supported({nullptr}));
        t.assert_true(!llama_context_memory::supported({}));
        if (dev) t.assert_true(!llama_context_memory::supported({f.device.get(), f.device.get(), f.cpu.get()}));
        t.assert_true(!llama_context_memory::create(nullptr, f.backends, f.groups));
        auto bad = f.groups;
        bad[0].first_slot = 99;
        t.assert_true(!llama_context_memory::create(f.sched.get(), f.backends, bad));
        t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(f.sched.get(), f.device.get()));
    });

    t.test("scheduler_owner_preserves_measured_maximum_and_computes", [&](testing & t) {
        fixture f(dev);
        t.assert_true(llama_context_memory::supported(f.backends));
        auto owner = llama_context_memory::create(f.sched.get(), f.backends, f.groups);
        if (!t.assert_true(bool(owner))) return;
        t.assert_true(owner->uses_arenas());
        t.assert_equal(size_t(8192), ggml_backend_sched_get_buffer_size(f.sched.get(), f.device.get()));
        f.rebuild();
        f.run(t, *owner, 4);
        t.assert_true(owner->compute_async(nullptr) == GGML_STATUS_FAILED);
        owner.reset();
        for (auto * backend : f.backends) t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(f.sched.get(), backend));
    });

    t.test("repeated_graph_rebuild_and_workspace_recreation", [&](testing & t) {
        fixture f(dev);
        for (int cycle = 0; cycle < 4; ++cycle) {
            auto owner = llama_context_memory::create(f.sched.get(), f.backends, f.groups);
            if (!t.assert_true(bool(owner))) return;
            for (int graph = 0; graph < 4; ++graph) {
                f.rebuild();
                f.run(t, *owner, float(cycle + graph));
            }
            t.assert_true(owner->compute_async(f.graph) == GGML_STATUS_SUCCESS);
            owner.reset();
            t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(f.sched.get(), f.device.get()));
        }
    });


    t.test("failed_creation_preserves_foreign_attachment_and_can_retry", [&](testing & t) {
        ggml_backend_ptr first(ggml_backend_cpu_init()), second(ggml_backend_cpu_init());
        ggml_backend_buffer_type other = *ggml_backend_cpu_buffer_type();
        std::vector<ggml_backend_t> backends{first.get(), second.get()};
        ggml_backend_buffer_type_t bufts[] = {ggml_backend_cpu_buffer_type(), &other};
        ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends.data(), bufts, 2, 256, false, true));
        const size_t a = ggml_backend_buft_get_alignment(bufts[0]);
        std::vector<ggml_backend_memory_workspace_group> groups{{bufts[0], 8192, a, 0}, {bufts[1], 8192, a, 1}};
        ggml_backend_buffer_ptr foreign(ggml_backend_buft_alloc_buffer(bufts[1], 8192));
        GGML_ASSERT(ggml_backend_sched_set_buffer_range(sched.get(), second.get(), foreign.get(), 0, 8192));
        t.assert_true(!llama_context_memory::create(sched.get(), backends, groups));
        t.assert_equal(size_t(0), ggml_backend_sched_get_buffer_size(sched.get(), first.get()));
        t.assert_equal(size_t(8192), ggml_backend_sched_get_buffer_size(sched.get(), second.get()));
        GGML_ASSERT(ggml_backend_sched_clear_buffer_range(sched.get(), second.get()));
        other.iface.alloc_buffer = [](ggml_backend_buffer_type_t, size_t) -> ggml_backend_buffer_t { return nullptr; };
        t.assert_true(!llama_context_memory::create(sched.get(), backends, groups));
        other.iface.alloc_buffer = ggml_backend_cpu_buffer_type()->iface.alloc_buffer;
        auto owner = llama_context_memory::create(sched.get(), backends, groups);
        t.assert_true(bool(owner));
    });

    if (dev) {
        t.test("retirement_clears_scheduler_created_cuda_captures", [&](testing & t) {
            fixture f(dev);
            auto owner = llama_context_memory::create(f.sched.get(), f.backends, f.groups);
            if (!t.assert_true(bool(owner))) return;
            f.rebuild();
            f.run(t, *owner, 2);
            auto * reg = ggml_backend_dev_backend_reg(dev);
            auto query = reinterpret_cast<ggml_backend_cuda_graph_is_captured_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_graph_is_captured"));
            if (!t.assert_true(query != nullptr)) return;
            const void * key = ggml_graph_node(f.graph, 0);
            t.assert_true(query(f.device.get(), key));
            owner.reset();
            t.assert_true(!query(f.device.get(), key));
        });
    }
    if (argc > 1 && (std::strcmp(argv[argc - 1], "--bench") == 0) && t.failures == 0) benchmark(dev);
    return t.summary();
}
