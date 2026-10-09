#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

// A CPU-backed device whose uploads and kernels execute only when explicitly waited on.
struct deferred_device {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_device device = *backend->device;
    ggml_backend_device host_device = *cpu->device;
    ggml_backend_reg reg = *device.reg;
    ggml_backend_buffer_type buft = *ggml_backend_cpu_buffer_type();
    std::vector<std::function<void()>> queue;
    std::vector<unsigned char> last_upload;
    size_t completed = 0;
    int uploads = 0;
    int upload_completions = 0;
    int computes = 0;
    int syncs = 0;
    int event_waits = 0;
    int live_events = 0;
    bool fail_compute = false;
    bool events;
    std::function<void(ggml_cgraph *)> before_cpu;
    decltype(ggml_backend_i::graph_compute) compute = backend->iface.graph_compute;

    static deferred_device & get(ggml_backend_t b) {
        return *static_cast<deferred_device *>(b->device->context);
    }
    void drain(size_t end) {
        while (completed < end) { queue[completed++](); }
    }
    explicit deferred_device(bool with_events) : events(with_events) {
        GGML_ASSERT(cpu && backend);
        ggml_backend_cpu_set_n_threads(cpu, 1);
        ggml_backend_cpu_set_n_threads(backend, 1);
        reg.iface.get_proc_address = [](ggml_backend_reg_t, const char * name) -> void * {
            static auto stream_ordered = +[](ggml_backend_dev_t) { return true; };
            return std::strcmp(name, "ggml_backend_async_is_stream_ordered") == 0 ?
                reinterpret_cast<void *>(stream_ordered) : nullptr;
        };
        host_device.context = this;
        cpu->device = &host_device;
        device.context = this;
        device.reg = &reg;
        buft.device = &device;
        buft.iface.is_host = [](ggml_backend_buffer_type_t) { return false; };
        buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t type, size_t size) {
            auto buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
            if (buffer) { buffer->buft = type; }
            return buffer;
        };
        device.iface.get_buffer_type = [](ggml_backend_dev_t dev) {
            return &static_cast<deferred_device *>(dev->context)->buft;
        };
        device.iface.supports_buft = [](ggml_backend_dev_t dev, ggml_backend_buffer_type_t type) {
            return type == &static_cast<deferred_device *>(dev->context)->buft;
        };
        device.iface.event_new = [](ggml_backend_dev_t dev) -> ggml_backend_event_t {
            auto & self = *static_cast<deferred_device *>(dev->context);
            if (!self.events) { return nullptr; }
            ++self.live_events;
            return new ggml_backend_event{dev, new size_t(0)};
        };
        device.iface.event_free = [](ggml_backend_dev_t dev, ggml_backend_event_t event) {
            --static_cast<deferred_device *>(dev->context)->live_events;
            delete static_cast<size_t *>(event->context);
            delete event;
        };
        device.iface.event_synchronize = [](ggml_backend_dev_t dev, ggml_backend_event_t event) {
            auto & self = *static_cast<deferred_device *>(dev->context);
            ++self.event_waits;
            self.drain(*static_cast<size_t *>(event->context));
        };
        backend->device = &device;
        backend->iface.cpy_tensor_async = nullptr;
        backend->iface.set_tensor_async = [](ggml_backend_t b, ggml_tensor * t,
                                               const void * data, size_t offset, size_t size) {
            auto & self = get(b);
            ++self.uploads;
            self.queue.push_back([=, &self] {
                ggml_backend_tensor_set(t, data, offset, size);
                self.last_upload.resize(size);
                ggml_backend_tensor_get(t, self.last_upload.data(), offset, size);
                ++self.upload_completions;
            });
        };
        backend->iface.synchronize = [](ggml_backend_t b) {
            auto & self = get(b);
            ++self.syncs;
            self.drain(self.queue.size());
        };
        backend->iface.event_record = [](ggml_backend_t b, ggml_backend_event_t event) {
            *static_cast<size_t *>(event->context) = get(b).queue.size();
        };
        backend->iface.graph_compute = [](ggml_backend_t b, ggml_cgraph * graph) {
            auto & self = get(b);
            if (self.fail_compute) { return GGML_STATUS_FAILED; }
            // Scheduler graph views are temporary; preserve their node list until execution.
            auto ctx = ggml_init({ggml_graph_overhead(), nullptr, true});
            auto copy = ggml_new_graph(ctx);
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                ggml_graph_add_node(copy, ggml_graph_node(graph, i));
            }
            self.queue.push_back([=, &self] {
                GGML_ASSERT(self.compute(b, copy) == GGML_STATUS_SUCCESS);
                ++self.computes;
                ggml_free(ctx);
            });
            return GGML_STATUS_SUCCESS;
        };
    }
    ~deferred_device() {
        GGML_ASSERT(completed == queue.size() && live_events == 0);
        ggml_backend_free(backend);
        ggml_backend_free(cpu);
    }
};

static bool run_case(const char * name, bool events, bool parallel, bool host_users, bool dependent, bool failure,
                     bool expect_stream_ordered = true) {
    deferred_device dev(events);
    ggml_backend_t backends[] = {dev.backend, dev.cpu};
    auto sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, parallel, false);
    auto ctx = ggml_init({ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true});
    GGML_ASSERT(sched && ctx);
    auto input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_input(input);
    auto intermediate = ggml_scale(ctx, input, 2.0f);
    // Retain this allocation so the deliberate overwrite cannot also clobber a later host input.
    ggml_set_output(intermediate);
    auto output = ggml_scale(ctx, intermediate, 3.0f);
    ggml_set_output(output);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_sched_set_tensor_backend(sched, input, dev.cpu);
    ggml_backend_sched_set_tensor_backend(sched, intermediate, dev.cpu);
    ggml_backend_sched_set_tensor_backend(sched, output, dev.backend);
    ggml_tensor * host_input = nullptr;
    ggml_tensor * host_output = nullptr;
    if (host_users || dependent) {
        host_input = dependent ? output : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
        if (host_users) {
            ggml_set_input(host_input);
            ggml_backend_sched_set_tensor_backend(sched, host_input, dev.backend);
        }
        host_output = ggml_scale(ctx, host_input, 5.0f);
        ggml_set_output(host_output);
        ggml_build_forward_expand(graph, host_output);
        ggml_backend_sched_set_tensor_backend(sched, host_output, dev.cpu);
    }
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
    GGML_ASSERT(ggml_backend_sched_get_n_splits(sched) == (host_output ? 3 : 2));
    dev.syncs = 0;
    bool ok = true;
    // Disabled copies must not allocate the experimental upload-completion event either.
    if (!expect_stream_ordered) { ok &= dev.live_events == 0; }
    bool host_split_checked = false;
    // Intercept only the independent host split: it must see the upload complete without
    // draining unrelated device compute, even though all its copied inputs are user inputs.
    dev.cpu->iface.graph_compute = [](ggml_backend_t b, ggml_cgraph * g) {
        auto & self = deferred_device::get(b);
        if (self.before_cpu) { self.before_cpu(g); }
        return self.compute(b, g);
    };
    if (host_users) {
        dev.before_cpu = [&](ggml_cgraph * g) {
            if (ggml_graph_node(g, ggml_graph_n_nodes(g) - 1) != host_output) { return; }
            host_split_checked = true;
            ok &= dev.upload_completions == 1 && dev.computes == 0 && dev.syncs == 0;
            const float overwritten[] = {-99, -99, -99, -99};
            ggml_backend_tensor_set(intermediate, overwritten, 0, sizeof(overwritten));
        };
    }
    const int runs = parallel ? 2 * ggml_backend_sched_get_n_copies(sched) + 1 : 1;
    for (int step = 0; step < runs; ++step) {
        const float values[] = {float(step + 1), 1, -2, 3};
        ggml_backend_tensor_set(input, values, 0, sizeof(values));
        if (host_users) { ggml_backend_tensor_set(host_input, values, 0, sizeof(values)); }
        dev.fail_compute = failure;
        auto status = ggml_backend_sched_graph_compute_async(sched, graph);
        ok &= status == (failure ? GGML_STATUS_FAILED : GGML_STATUS_SUCCESS);
        if (expect_stream_ordered && events && !parallel) {
            // Returning success or failure releases host source ownership, but not device work.
            ok &= dev.uploads == 1 && dev.upload_completions == 1 && dev.event_waits == 1;
            ok &= dev.computes == (dependent ? 1 : 0) && dev.syncs == (dependent ? 1 : 0);
        } else {
            ok &= dev.uploads == 0;
        }
        if (failure) {
            const float expected[] = {2 * values[0], 2 * values[1], 2 * values[2], 2 * values[3]};
            ok &= dev.last_upload.size() == sizeof(expected) &&
                  std::memcmp(dev.last_upload.data(), expected, sizeof(expected)) == 0;
        }
        const float overwritten[] = {99, 99, 99, 99};
        if (!host_output) { ggml_backend_tensor_set(intermediate, overwritten, 0, sizeof(overwritten)); }
        ggml_backend_sched_synchronize(sched);
        if (!failure) {
            float actual[4];
            ggml_backend_tensor_get(output, actual, 0, sizeof(actual));
            for (int i = 0; i < 4; ++i) { ok &= actual[i] == 6 * values[i]; }
            if (host_output) {
                ggml_backend_tensor_get(host_output, actual, 0, sizeof(actual));
                for (int i = 0; i < 4; ++i) { ok &= actual[i] == (dependent ? 30 : 5) * values[i]; }
            }
        }
    }
    ok &= !host_users || host_split_checked;
    std::printf("%s: uploads=%d, completed=%d, computes=%d, waits=%d, syncs=%d: %s\n",
                name, dev.uploads, dev.upload_completions, dev.computes, dev.event_waits, dev.syncs, ok ? "PASS" : "FAIL");
    ggml_backend_sched_free(sched);
    ggml_free(ctx);
    return ok;
}

static bool mapped_weights() {
    deferred_device dev(true);
    auto ctx = ggml_init({ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true});
    alignas(64) float values[] = {1, 2, 3, 4};
    auto buffer = ggml_backend_cpu_buffer_from_ptr(values, sizeof(values));
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 2);
    GGML_ASSERT(ggml_backend_tensor_alloc(buffer, weights, values) == GGML_STATUS_SUCCESS);
    auto input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    ggml_set_input(input);
    auto output = ggml_mul_mat(ctx, weights, input);
    ggml_set_output(output);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_t backends[] = {dev.backend, dev.cpu};
    auto sched = ggml_backend_sched_new(backends, nullptr, 2, GGML_DEFAULT_GRAPH_SIZE, false, false);
    ggml_backend_sched_set_tensor_backend(sched, weights, dev.cpu);
    ggml_backend_sched_set_tensor_backend(sched, input, dev.backend);
    ggml_backend_sched_set_tensor_backend(sched, output, dev.backend);
    GGML_ASSERT(ggml_backend_sched_alloc_graph(sched, graph));
    const float vector[] = {2, 3};
    ggml_backend_tensor_set(input, vector, 0, sizeof(vector));
    bool ok = ggml_backend_sched_graph_compute_async(sched, graph) == GGML_STATUS_SUCCESS;
    // A regular dense operation on a mapped WEIGHTS buffer must retain blocking-copy staging.
    ok &= dev.uploads == 0;
    values[0] = values[1] = values[2] = values[3] = -99;
    ggml_backend_sched_synchronize(sched);
    float actual[2];
    ggml_backend_tensor_get(output, actual, 0, sizeof(actual));
    ok &= actual[0] == 8 && actual[1] == 18;
    std::printf("mapped WEIGHTS MUL_MAT: uploads=%d: %s\n", dev.uploads, ok ? "PASS" : "FAIL");
    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--expect-sync") == 0) {
        // Run with an event-capable device so fallback cannot hide accidental opt-in.
        return run_case("synchronous selection", true, false, false, false, false, false) ? 0 : 1;
    }
    if (argc != 1) { return 2; }
    bool ok = true;
    ok &= run_case("async return", true, false, false, false, false);
    ok &= run_case("host user-input split", true, false, true, false, false);
    ok &= run_case("host dependency", true, false, false, true, false);
    ok &= run_case("compute failure", true, false, false, false, true);
    ok &= run_case("serial null events", false, false, false, false, false);
    ok &= run_case("parallel null events", false, true, false, false, false);
    ok &= mapped_weights();
    return ok ? 0 : 1;
}
