#include "kv-stream-block-test.h"
#include "../ggml/src/ggml-kv-stream-copy.h"
#include "../src/llama-kv-stream-model.h"
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <set>
#include <string>

namespace {
struct failure_probe {
    static failure_probe * active;
    size_t fail, calls = 0, invalid = 0, synchronizations = 0;
    bool poison;
    cudaError_t status;
    bool capture_llama;
    std::set<void *> streams, events;
    std::string log;
    ggml_log_callback previous;
    void * user;
    // Interpose only one owner's initialization; all other CUDA calls remain real.
    failure_probe(size_t fail, bool poison, cudaError_t status = cudaErrorMemoryAllocation, bool capture_llama = false) :
            fail(fail), poison(poison), status(status), capture_llama(capture_llama) {
        GGML_ASSERT(!active); active = this;
        ggml_log_get(&previous, &user);
        const ggml_log_callback capture = [](ggml_log_level, const char * text, void * p) {
            static_cast<failure_probe *>(p)->log += text;
        };
        ggml_log_set(capture, this);
        if (capture_llama) llama_log_set(capture, this);
    }
    ~failure_probe() { if (capture_llama) llama_log_set(nullptr, nullptr); ggml_log_set(previous, user); active = nullptr; }
    template<class T> bool inject(T * output) {
        if (++calls != fail) return false;
        *output = poison ? reinterpret_cast<T>(UINTPTR_MAX) : nullptr;
        return true;
    }
};
failure_probe * failure_probe::active = nullptr;

template<class T> T original(const char * name) {
    auto symbol = dlsym(RTLD_NEXT, name);
    GGML_ASSERT(symbol); return reinterpret_cast<T>(symbol);
}
bool invalid(void * handle) {
    if (handle != reinterpret_cast<void *>(UINTPTR_MAX)) return false;
    if (failure_probe::active) ++failure_probe::active->invalid;
    return true;
}
}

extern "C" cudaError_t CUDARTAPI cudaStreamCreateWithFlags(cudaStream_t * output, unsigned flags) {
    auto * p = failure_probe::active;
    if (p && p->inject(output)) return p->status;
    const auto result = original<decltype(&cudaStreamCreateWithFlags)>("cudaStreamCreateWithFlags")(output, flags);
    if (p && result == cudaSuccess) p->streams.insert(*output);
    return result;
}
extern "C" cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t * output, unsigned flags) {
    auto * p = failure_probe::active;
    if (p && p->inject(output)) return p->status;
    const auto result = original<decltype(&cudaEventCreateWithFlags)>("cudaEventCreateWithFlags")(output, flags);
    if (p && result == cudaSuccess) p->events.insert(*output);
    return result;
}
extern "C" cudaError_t CUDARTAPI cudaEventCreate(cudaEvent_t * output) {
    auto * p = failure_probe::active;
    if (p && p->inject(output)) return p->status;
    const auto result = original<decltype(&cudaEventCreate)>("cudaEventCreate")(output);
    if (p && result == cudaSuccess) p->events.insert(*output);
    return result;
}
extern "C" cudaError_t CUDARTAPI cudaStreamSynchronize(cudaStream_t stream) {
    if (failure_probe::active) ++failure_probe::active->synchronizations;
    if (invalid(stream)) return cudaErrorInvalidResourceHandle;
    return original<decltype(&cudaStreamSynchronize)>("cudaStreamSynchronize")(stream);
}
extern "C" cudaError_t CUDARTAPI cudaStreamDestroy(cudaStream_t stream) {
    if (invalid(stream)) return cudaErrorInvalidResourceHandle;
    const auto result = original<decltype(&cudaStreamDestroy)>("cudaStreamDestroy")(stream);
    if (failure_probe::active && result == cudaSuccess) failure_probe::active->streams.erase(stream);
    return result;
}
extern "C" cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
    if (invalid(event)) return cudaErrorInvalidResourceHandle;
    const auto result = original<decltype(&cudaEventDestroy)>("cudaEventDestroy")(event);
    if (failure_probe::active && result == cudaSuccess) failure_probe::active->events.erase(event);
    return result;
}

int main(int argc, char ** argv) {
    const bool null_only = argc == 2 && !std::strcmp(argv[1], "--null-only");
    ggml_backend_load_all();
    auto * dev = ggml_backend_dev_by_name("CUDA0");
    if (!dev) return 77;
    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    ggml_backend_synchronize(backend.get());
    auto get = reinterpret_cast<ggml_kv_stream_copy_ops_get>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(dev), "ggml_backend_kv_stream_copy_ops"));
    if (!get || !get()) return 77;
    const auto * ops = get();
    if (argc == 2 && !std::strcmp(argv[1], "--footprint")) {
        for (size_t slots : {size_t(2), size_t(2640), size_t(5514)}) {
            size_t before, after, total;
            GGML_ASSERT(cudaMemGetInfo(&before, &total) == cudaSuccess);
            std::unique_ptr<void, void(*)(void*)> bank(ops->prepare(backend.get(), slots, true), ops->free_prepared);
            GGML_ASSERT(bank && cudaMemGetInfo(&after, &total) == cudaSuccess);
            std::cout << "COPY_PREFLIGHT_FOOTPRINT,slots=" << slots << ",device_delta=" << (before >= after ? before-after : 0)
                      << ",feedback_bytes=" << (slots+5)*2*sizeof(uint64_t) << '\n';
        }
        return 0;
    }
    const ggml_kv_stream_shape shape{GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, 256, 256, 4, 256, 128};
    ggml_kv_stream_layout layout;
    GGML_ASSERT(ggml_kv_stream_layout_make(shape, 512, layout).status == ggml_kv_stream_status::success);
    ggml_backend_buffer_ptr device(ggml_backend_buft_alloc_buffer(llama_kv_stream_device_buffer_type(dev), layout.bytes));
    ggml_backend_buffer_ptr host(ggml_backend_buft_alloc_buffer(llama_kv_stream_host_buffer_type(dev), layout.bytes));
    GGML_ASSERT(device && host);
    testing t;
#ifdef KV_COPY_INIT_MODEL_TEST
    t.test("model_preflight_propagates_resource_oom_before_inference", [&](testing & t) {
        fixture f(backend.get(), true, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, 513, false, 1);
        failure_probe probe(1, true, cudaErrorMemoryAllocation, true);
        uint32_t unavailable = 99;
        auto model = llama_kv_stream_model::create({backend.get(), f.host->config(), f.policy.pool_bytes, 256, 4, true}, &unavailable);
        t.assert_true(!model);
        t.assert_equal(uint32_t(0), unavailable);
        t.assert_equal(size_t(1), probe.calls);
        t.assert_equal(size_t(0), probe.invalid);
        t.assert_true(probe.log.find("preflight failed") != std::string::npos);
        t.assert_true(probe.log.find("out of memory") != std::string::npos);
        t.assert_true(probe.streams.empty() && probe.events.empty());
    });
#endif
    t.test("prepared_copy_resources_reject_invalid_capacity_before_creation", [&](testing & t) {
        failure_probe probe(1, true);
        t.assert_true(!ops->prepare(backend.get(), 0, true));
        t.assert_true(!ops->prepare(backend.get(), SIZE_MAX, true));
        t.assert_true(!ops->prepare(nullptr, 2, true));
        t.assert_equal(size_t(0), probe.calls);
        t.assert_true(probe.streams.empty() && probe.events.empty());
    });
    t.test("prepared_copy_resources_fail_before_binding_and_keep_original_error", [&](testing & t) {
        if (!t.assert_true(ops->version >= 11)) return;
        for (size_t failure = 1; failure <= 11; ++failure) {
            failure_probe probe(failure, true);
            void * prepared = ops->prepare(backend.get(), 2, true);
            t.assert_true(!prepared);
            if (prepared) ops->free_prepared(prepared);
            t.assert_equal(failure, probe.calls);
            t.assert_equal(size_t(0), probe.invalid);
            t.assert_true(probe.streams.empty() && probe.events.empty());
            t.assert_true(probe.log.find("out of memory") != std::string::npos);
        }
    });
    t.test("prepared_copy_resources_rebind_without_creation_and_serialize_feedback", [&](testing & t) {
        if (!t.assert_true(ops->version >= 11)) return;
        std::unique_ptr<void, void(*)(void*)> prepared(ops->prepare(backend.get(), 2, true), ops->free_prepared);
        GGML_ASSERT(prepared);
        std::unique_ptr<void, void(*)(void*)> first(nullptr, ops->free), next(nullptr, ops->free);
        {
            failure_probe probe(1, true);
            first.reset(ops->create_prepared(backend.get(), prepared.get(), device.get(), host.get(), shape, 1));
            next.reset(ops->create_prepared(backend.get(), prepared.get(), device.get(), host.get(), shape, 2));
            t.assert_true(first && next);
            t.assert_true(ops->measure(first.get(), true) && ops->measure(next.get(), true));
            t.assert_equal(size_t(0), probe.calls);
            // No second owner may overwrite counters whose old readback is still retained.
            t.assert_true(ops->begin(first.get()));
            t.assert_true(!ops->begin(next.get()));
            ops->drain(first.get());
            t.assert_true(!ops->begin(next.get()));
            first.reset();
            t.assert_true(ops->begin(next.get()));
            ops->drain(next.get());
            t.assert_equal(size_t(0), probe.calls);
        }
        prepared.reset();
        t.assert_true(ops->begin(next.get()));
        ops->drain(next.get());
        next.reset();
    });
    t.test("prepared_copy_resources_reject_over_capacity_without_native_creation", [&](testing & t) {
        if (!t.assert_true(ops->version >= 11)) return;
        std::unique_ptr<void, void(*)(void*)> prepared(ops->prepare(backend.get(), 1, false), ops->free_prepared);
        GGML_ASSERT(prepared);
        failure_probe probe(1, true);
        t.assert_true(!ops->create_prepared(backend.get(), prepared.get(), device.get(), host.get(), shape, 2));
        std::unique_ptr<void, void(*)(void*)> queue(ops->create_prepared(backend.get(), prepared.get(), device.get(), host.get(), shape, 1), ops->free);
        t.assert_true(bool(queue));
        t.assert_true(!ops->measure(queue.get(), true));
        t.assert_equal(size_t(0), probe.calls);
    });
    t.test("copy_queue_creation_failures_preserve_error_and_release_owned_handles", [&](testing & t) {
        for (bool poison : {false, true}) {
            if (poison && null_only) continue;
            for (size_t failure = 1; failure <= 6; ++failure) {
                failure_probe probe(failure, poison);
                void * queue = ops->create(backend.get(), device.get(), host.get(), shape, 2);
                t.assert_true(!queue);
                if (queue) ops->free(queue);
                t.assert_equal(failure, probe.calls);
                t.assert_equal(size_t(0), probe.invalid);
                t.assert_equal(size_t(0), probe.synchronizations);
                t.assert_true(probe.streams.empty() && probe.events.empty());
                t.assert_true(probe.log.find("out of memory") != std::string::npos);
                t.assert_true(probe.log.find(failure == 1 ? "cudaStreamCreateWithFlags" : "cudaEventCreateWithFlags") != std::string::npos);
                t.assert_true(cudaGetLastError() == cudaSuccess);
            }
        }
    });
    t.test("feedback_creation_failures_preserve_error_and_allow_retry", [&](testing & t) {
        std::unique_ptr<void, void(*)(void*)> queue(ops->create(backend.get(), device.get(), host.get(), shape, 2), ops->free);
        GGML_ASSERT(queue && ops->measure(queue.get(), true) && ops->measure(queue.get(), false));
        for (bool poison : {false, true}) {
            if (poison && null_only) continue;
            for (size_t failure = 1; failure <= 5; ++failure) {
                {
                    failure_probe probe(failure, poison);
                    t.assert_true(!ops->measure(queue.get(), true));
                    t.assert_equal(failure, probe.calls);
                    t.assert_equal(size_t(0), probe.invalid);
                    t.assert_true(probe.streams.empty() && probe.events.empty());
                    t.assert_true(probe.log.find("out of memory") != std::string::npos);
                    t.assert_true(probe.log.find(failure == 1 ? "cudaStreamCreateWithFlags" : failure < 4 ? "cudaEventCreateWithFlags" : "cudaEventCreate") != std::string::npos);
                    t.assert_true(cudaGetLastError() == cudaSuccess);
                }
                t.assert_true(ops->measure(queue.get(), true) && ops->measure(queue.get(), false));
            }
        }
    });
    t.test("non_memory_creation_errors_are_not_mislabeled_as_oom", [&](testing & t) {
        failure_probe probe(1, !null_only, cudaErrorInvalidValue);
        void * queue = ops->create(backend.get(), device.get(), host.get(), shape, 2);
        t.assert_true(!queue);
        if (queue) ops->free(queue);
        t.assert_true(probe.log.find("invalid argument") != std::string::npos);
        t.assert_true(probe.log.find("out of memory") == std::string::npos);
        t.assert_equal(size_t(0), probe.invalid);
    });
    return t.summary();
}
