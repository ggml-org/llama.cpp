#include "../ggml/src/ggml-cuda/common.cuh"
#include "../ggml/src/ggml-backend-impl.h"
#include <cstdlib>

// A simulated host CC must match the code the driver actually loads, not just one target in a fat binary.
bool kv_stream_test_is_sm61_only() {
#ifdef __CUDA_ARCH_LIST__
    for (int arch : {__CUDA_ARCH_LIST__}) if (arch != 610) return false;
    return true;
#else
    return false;
#endif
}

// Never force a host architecture over a mixed-target binary with different device-side feature guards.
bool kv_stream_test_is_arch_only(int cc) {
#ifdef __CUDA_ARCH_LIST__
    for (int arch : {__CUDA_ARCH_LIST__}) if (arch != cc) return false;
    return true;
#else
    GGML_UNUSED(cc); return false;
#endif
}

// Constrain metadata only; this must not change any driver/device setting.
size_t kv_stream_test_shared_limit(ggml_backend_t backend, size_t limit) {
    ggml_backend_synchronize(backend);
    auto * ctx=static_cast<ggml_backend_cuda_context *>(backend->context);
    auto & info=const_cast<ggml_cuda_device_info &>(ggml_cuda_info());
    const size_t previous=info.devices[ctx->device].smpbo;
    info.devices[ctx->device].smpbo=limit;
    return previous;
}

// This single-process test restores the non-const device-info object before the borrowed backend is destroyed.
int kv_stream_test_override_cc(ggml_backend_t backend, int cc) {
    ggml_backend_synchronize(backend);
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    auto & info = const_cast<ggml_cuda_device_info &>(ggml_cuda_info());
    const int previous = info.devices[ctx->device].cc;
    info.devices[ctx->device].cc = cc;
    return previous;
}

// The expected saved-state width belongs to the compiled kernel, including forward-JIT builds.
int kv_stream_test_compiled_cc(ggml_backend_t backend) {
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    return ggml_cuda_highest_compiled_arch(ggml_cuda_info().devices[ctx->device].cc);
}

// Eager-only profiles must verify outputs and lifetime without expecting a native graph cache entry.
bool kv_stream_test_native_graphs_enabled() {
#ifdef GGML_CUDA_USE_GRAPHS
    return std::getenv("GGML_CUDA_DISABLE_GRAPHS") == nullptr;
#else
    return false;
#endif
}
