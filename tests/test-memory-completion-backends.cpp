#include "../src/llama-kv-stream-publication.h"
#include "../src/llama-memory-completion.h"
#include "ggml-cpp.h"
#include "testing.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using pair_dependencies = llama_kv_stream_publication_pair_dependencies;

static ggml_backend_meta_split_state mirrored(const ggml_tensor *, void *) {
    return { GGML_BACKEND_SPLIT_AXIS_MIRRORED, { 0 }, { 1 }, 1 };
}

static std::unique_ptr<llama_kv_stream_publications> publications(llama_kv_stream_publication_ticket & ticket,
                                                                  std::shared_ptr<void>                owner) {
    auto result = llama_kv_stream_publications::create({ 0, 1, 1, 1 });
    if (!result || !result->reserve(0, 1, 1, { std::move(owner) }, {}, ticket)) {
        return {};
    }
    return result;
}

static bool dependencies(ggml_backend_t backend, pair_dependencies & output, bool mixed) {
    for (size_t plane = 0; plane < 2; ++plane) {
        for (size_t domain = 0; domain < 2; ++domain) {
            auto & value = domain ? output.host[plane] : output.device[plane];
            value        = mixed && (plane + domain) % 2 == 0 ? llama_memory_completion::completed(backend) :
                                                                llama_memory_completion::create(backend);
            if (!value || (!value->recorded() && !value->record())) {
                return false;
            }
        }
    }
    return true;
}

struct graph_fixture {
    static constexpr size_t count = 64;
    ggml_context_ptr        context;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor *           input       = nullptr;
    ggml_tensor *           output      = nullptr;
    ggml_tensor *           input_view  = nullptr;
    ggml_tensor *           output_view = nullptr;
    ggml_cgraph *           graph       = nullptr;
    std::vector<float>      source;

    explicit graph_fixture(ggml_backend_t backend) {
        context.reset(ggml_init({ 65536, nullptr, true }));
        GGML_ASSERT(context);
        input  = ggml_new_tensor_1d(context.get(), GGML_TYPE_F32, count);
        output = ggml_scale(context.get(), input, 2.0f);
        graph  = ggml_new_graph_custom(context.get(), 16, false);
        ggml_build_forward_expand(graph, output);
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), backend));
        GGML_ASSERT(buffer);
        input_view  = ggml_view_1d(context.get(), input, count, 0);
        output_view = ggml_view_1d(context.get(), output, count, 0);
        GGML_ASSERT(ggml_backend_view_init(input_view) == GGML_STATUS_SUCCESS);
        GGML_ASSERT(ggml_backend_view_init(output_view) == GGML_STATUS_SUCCESS);
        source.resize(count);
        for (size_t i = 0; i < count; ++i) {
            source[i] = float(i) * 0.25f - 3.0f;
        }
        ggml_backend_tensor_set(input_view, source.data(), 0, source.size() * sizeof(float));
    }

    bool submit(ggml_backend_t backend) const {
        return ggml_backend_graph_compute_async(backend, graph) == GGML_STATUS_SUCCESS;
    }

    bool exact(const std::vector<float> & result) const {
        if (result.size() != count) {
            return false;
        }
        for (size_t i = 0; i < count; ++i) {
            if (result[i] != 2.0f * source[i]) {
                return false;
            }
        }
        return true;
    }

    bool exact() const {
        std::vector<float> result(count);
        ggml_backend_tensor_get(output_view, result.data(), 0, result.size() * sizeof(float));
        return exact(result);
    }
};

struct arguments {
    std::vector<std::string> devices;
    bool                     meta            = false;
    bool                     list            = false;
    bool                     expect_event    = false;
    bool                     expect_fallback = false;
};

static bool parse(int argc, char ** argv, arguments & output) {
    if (argc == 1) {
        output.devices         = { "CPU" };
        output.expect_fallback = true;
        return true;
    }
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--list")) {
            output.list = true;
        } else if (!std::strcmp(argv[i], "--backend") && i + 1 < argc) {
            output.devices = { argv[++i] };
        } else if (!std::strcmp(argv[i], "--meta") && i + 2 < argc) {
            output.meta    = true;
            output.devices = { argv[++i], argv[++i] };
        } else if (!std::strcmp(argv[i], "--expect-event")) {
            output.expect_event = true;
        } else if (!std::strcmp(argv[i], "--expect-fallback")) {
            output.expect_fallback = true;
        } else {
            return false;
        }
    }
    return output.list || (!output.devices.empty() && output.expect_event != output.expect_fallback);
}

int main(int argc, char ** argv) {
    arguments args;
    if (!parse(argc, argv, args)) {
        std::fprintf(stderr,
                     "usage: %s --list | (--backend NAME | --meta NAME NAME) (--expect-event | --expect-fallback)\n",
                     argv[0]);
        return 2;
    }
    ggml_backend_load_all();
    if (args.list) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto *                 device = ggml_backend_dev_get(i);
            ggml_backend_dev_props props{};
            ggml_backend_dev_get_props(device, &props);
            std::printf("DEVICE name=%s type=%d events=%d description=%s\n", props.name, int(props.type),
                        int(props.caps.events), props.description);
        }
        return 0;
    }
    std::vector<ggml_backend_dev_t> devices;
    for (const auto & name : args.devices) {
        auto * device = ggml_backend_dev_by_name(name.c_str());
        if (!device) {
            std::fprintf(stderr, "missing backend device: %s\n", name.c_str());
            return 3;
        }
        devices.push_back(device);
    }
    auto * target =
        args.meta ? ggml_backend_meta_device(devices.data(), devices.size(), mirrored, nullptr) : devices.front();
    ggml_backend_ptr producer(ggml_backend_dev_init(target, nullptr));
    ggml_backend_ptr consumer(ggml_backend_dev_init(target, nullptr));
    if (!producer || !consumer) {
        std::fprintf(stderr, "backend initialization failed\n");
        return 4;
    }
    testing t;
    bool    event_mode = false;
    t.test(args.meta ? "meta_alias_pair_uses_common_fallback" : "real_backend_pair_preserves_alias_output",
           [&](testing & t) {
               graph_fixture graph(producer.get());
               if (!t.assert_true(graph.submit(producer.get()))) {
                   return;
               }
               pair_dependencies deps;
               if (!t.assert_true(dependencies(producer.get(), deps, false))) {
                   return;
               }
               event_mode = deps.device[0]->event_backed();
               for (size_t plane = 0; plane < 2; ++plane) {
                   t.assert_equal(event_mode, deps.device[plane]->event_backed());
                   t.assert_equal(event_mode, deps.host[plane]->event_backed());
               }
               t.assert_equal(args.expect_event, event_mode);
               llama_kv_stream_publication_ticket ticket;
               auto                               owner = std::make_shared<int>(7);
               std::weak_ptr<int>                 weak  = owner;
               auto                               state = publications(ticket, owner);
               owner.reset();
               size_t callbacks = 0;
               auto   pair      = state ? llama_kv_stream_publication_pair::create(ticket, 0, std::move(deps),
                                                                                   [&] {
                                                                                ++callbacks;
                                                                                return true;
                                                                                   }) :
                                          nullptr;
               if (!t.assert_true(pair && !weak.expired() && pair->wait_device(consumer.get()))) {
                   return;
               }
               t.assert_true(pair->device_ready() && !pair->host_ready());
               t.assert_equal(size_t(1), state->frontiers().device);
               std::vector<float> queued(graph.count);
               ggml_backend_tensor_get_async(consumer.get(), graph.output_view, queued.data(), 0,
                                             queued.size() * sizeof(float));
               t.assert_equal(size_t(0), state->frontiers().host);
               if (!t.assert_true(pair->publish_host())) {
                   return;
               }
               t.assert_true(pair->host_ready() && ticket.committed());
               t.assert_equal(size_t(1), callbacks);
               ggml_backend_synchronize(consumer.get());
               t.assert_true(pair->release_device(consumer.get()));
               pair.reset();
               t.assert_true(graph.exact(queued));
               t.assert_true(graph.exact());
               t.assert_true(ticket.retire());
               t.assert_true(weak.expired());
           });
    t.test("mixed_dependencies_fail_without_host_visibility", [&](testing & t) {
        pair_dependencies deps;
        if (!t.assert_true(dependencies(producer.get(), deps, true))) {
            return;
        }
        llama_kv_stream_publication_ticket ticket;
        auto                               state = publications(ticket, std::make_shared<int>(9));
        auto pair = state ? llama_kv_stream_publication_pair::create(ticket, 0, std::move(deps), [] { return false; }) :
                            nullptr;
        if (!t.assert_true(pair && pair->wait_device(consumer.get()))) {
            return;
        }
        t.assert_true(!pair->publish_host() && ticket.failed());
        const auto frontiers = state->frontiers();
        t.assert_equal(size_t(1), frontiers.device);
        t.assert_equal(size_t(0), frontiers.host);
        t.assert_equal(size_t(0), frontiers.committed);
        pair.reset();
        t.assert_true(ticket.retire());
    });
    std::printf("CAPABILITY backend=%s completion=%s wait=%s\n", ggml_backend_name(producer.get()),
                event_mode ? "event-backed" : "synchronous-fallback",
                event_mode ? "backend-defined" : "already-complete");
    return t.summary();
}
