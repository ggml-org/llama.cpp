#include "../ggml/src/ggml-cuda/common.cuh"
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-cuda/kv-stream-dispatch.cuh"
#include "../ggml/src/ggml-cuda/kv-stream-runtime.cuh"
#include "../ggml/src/ggml-cuda/kv-stream-tile.cuh"
#include "../ggml/src/ggml-cuda/fattn.cuh"
#include "ggml-cpp.h"
#include <cstdlib>

// Exercise the real tile family on the current device, without pretending its architecture is older.
bool kv_stream_test_native_tile(ggml_backend_t backend, ggml_tensor * op) {
    auto & ctx=*static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx.device);
    const size_t bytes=ggml_cuda_flash_attn_ext_selected_alloc_size(ggml_cuda_kv_stream_kernel_family::tile,op);
    ggml_backend_buffer_ptr buffer(ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend),bytes));
    if (!buffer) return false;
    auto native=*op;
    native.data=ggml_backend_buffer_get_base(buffer.get()); native.buffer=buffer.get();
    ggml_cuda_flash_attn_ext_selected(ctx,&native,ggml_cuda_kv_stream_kernel_family::tile);
    CUDA_CHECK(cudaMemcpyAsync(op->data,native.data,ggml_nbytes(op),cudaMemcpyDeviceToDevice,ctx.stream()));
    ggml_backend_synchronize(backend);
    return true;
}

bool kv_stream_test_tile_plan(ggml_backend_t backend, const ggml_tensor * op, size_t tokens,
        ggml_kv_stream_resume_plan & plan, size_t & bound) {
    auto & ctx=*static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx.device);
    const auto * q=op->src[0], * k=op->src[1], * v=op->src[2];
    return ggml_cuda_kv_stream_tile_plan(ctx,k->type,v->type,uint32_t(q->ne[2]),uint32_t(k->ne[2]),
        uint32_t(q->ne[1]),tokens,plan,uint32_t(k->ne[0])) &&
        ggml_cuda_kv_stream_tile_prefill_bound(ctx,k->type,v->type,uint32_t(q->ne[2]),uint32_t(k->ne[2]),
            uint32_t(q->ne[1]),tokens,uint32_t(k->ne[0]),bound);
}

bool kv_stream_test_tile_run(ggml_backend_t backend, const ggml_tensor * op, const ggml_kv_stream_span_plan_view & view,
        ggml_backend_buffer_t workspace, const ggml_kv_stream_resume_plan & plan, size_t first, size_t end) {
    auto & ctx=*static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx.device);
    return ggml_cuda_kv_stream_tile_launch(ctx,op,view,workspace,plan,first,end);
}

// Resolve and load real device code without claiming that the whole streamed workload is admitted.
bool kv_stream_test_dimension_instance(ggml_backend_t backend, ggml_type key, ggml_type value,
        uint32_t width, uint32_t columns, uint32_t groups, bool mma, bool resume, bool dense) {
    auto * ctx=static_cast<ggml_backend_cuda_context *>(backend->context);
    ggml_cuda_set_device(ctx->device);
    const auto prepared=[](auto kernel) {
        if (!kernel) return false;
        cudaFuncAttributes attributes{};
        const auto status=cudaFuncGetAttributes(&attributes,kernel);
        if (status != cudaSuccess) { (void) cudaGetLastError(); return false; }
        return attributes.maxThreadsPerBlock > 0 && attributes.numRegs > 0;
    };
    if (mma && width == 512) return prepared(ggml_cuda_kv_stream_mma_runtime_kernel(key,value,columns,groups,resume,dense));
    if (mma) return prepared(ggml_cuda_kv_stream_mma_kernel(key,value,columns,groups,resume,dense,width,width));
    return prepared(ggml_cuda_kv_stream_tile_kernel(key,value,columns,groups,width,width));
}

// Encoded K/V pairs share one arithmetic body; input/resume policies remain distinct.
bool kv_stream_test_runtime_shared_instances() {
    const auto first=ggml_cuda_kv_stream_mma_runtime_kernel(GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,4,8,false,false);
    return first && first == ggml_cuda_kv_stream_mma_runtime_kernel(GGML_TYPE_Q8_0,GGML_TYPE_Q8_0,4,8,false,false) &&
        first == ggml_cuda_kv_stream_mma_runtime_kernel(GGML_TYPE_Q5_1,GGML_TYPE_Q5_1,4,8,false,false) &&
        first != ggml_cuda_kv_stream_mma_runtime_kernel(GGML_TYPE_F16,GGML_TYPE_F16,4,8,false,false) &&
        first != ggml_cuda_kv_stream_mma_runtime_kernel(GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,4,8,true,false) &&
        first != ggml_cuda_kv_stream_mma_runtime_kernel(GGML_TYPE_Q8_0,GGML_TYPE_Q4_0,8,8,false,false);
}

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
