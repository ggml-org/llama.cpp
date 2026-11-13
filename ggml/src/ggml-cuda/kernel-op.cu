#include "kernel-op.cuh"

#include "ggml-cuda.h"

#include <map>
#include <mutex>
#include <string>
#include <tuple>

// the modules are loaded with the driver API, as linked for the VMM pool
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && !defined(GGML_CUDA_NO_VMM)
#define GGML_CUDA_KERNEL_OP_AVAILABLE
#endif

// the CPU backend calls it if it gets the node
static void ggml_cuda_kernel_op_cpu(ggml_tensor * dst, int, int, void *) {
    GGML_ABORT("%s: a kernel op runs on " GGML_CUDA_NAME " only", dst->name);
}

ggml_custom_op_t ggml_cuda_kernel_op_fun() {
#ifdef GGML_CUDA_KERNEL_OP_AVAILABLE
    return ggml_cuda_kernel_op_cpu;
#else
    return nullptr;
#endif // GGML_CUDA_KERNEL_OP_AVAILABLE
}

static const ggml_cuda_kernel * ggml_cuda_kernel_of(const ggml_tensor * op) {
    if (op->op != GGML_OP_CUSTOM) {
        return nullptr;
    }
    ggml_custom_op_params params;
    memcpy(&params, op->op_params, sizeof(params));
    return params.fun == ggml_cuda_kernel_op_cpu ? (const ggml_cuda_kernel *) params.userdata : nullptr;
}

int ggml_cuda_kernel_op_stream(const ggml_tensor * op, bool * join) {
    const ggml_cuda_kernel * k = ggml_cuda_kernel_of(op);
    *join = k != nullptr && k->stream == 0 && k->join;
    return k != nullptr ? k->stream : 0;
}

#ifdef GGML_CUDA_KERNEL_OP_AVAILABLE
// the kernel on the device, nullptr if the module or the kernel cannot be loaded
static CUfunction ggml_cuda_kernel_function(int device, const ggml_cuda_kernel & k) {
    static std::mutex mutex;
    static std::map<std::pair<int, const void *>, CUmodule> modules;
    static std::map<std::tuple<int, const void *, std::string>, CUfunction> functions;

    std::lock_guard<std::mutex> lock(mutex);
    const auto key = std::make_tuple(device, (const void *) k.image, std::string(k.name));
    if (const auto it = functions.find(key); it != functions.end()) {
        return it->second;
    }

    const auto [it, is_new] = modules.try_emplace({ device, k.image }, nullptr);
    if (is_new) {
        ggml_cuda_set_device(device);
        // the driver API loads the module into the current context: make it the one of the runtime
        size_t mem_free  = 0;
        size_t mem_total = 0;
        CUDA_CHECK(cudaMemGetInfo(&mem_free, &mem_total));
        const CUresult err = cuModuleLoadData(&it->second, k.image);
        if (err != CUDA_SUCCESS) {
            GGML_LOG_WARN("%s: device %d cannot load the module of kernel %s: %s\n", __func__, device, k.name, cu_get_error_str(err));
            it->second = nullptr;
        }
    }

    CUfunction fun = nullptr;
    if (it->second != nullptr) {
        const CUresult err = cuModuleGetFunction(&fun, it->second, k.name);
        if (err != CUDA_SUCCESS) {
            GGML_LOG_WARN("%s: kernel %s is not in its module: %s\n", __func__, k.name, cu_get_error_str(err));
            fun = nullptr;
        }
    }
    functions[key] = fun;
    return fun;
}

static cudaEvent_t ggml_cuda_kernel_op_event(cudaEvent_t & event) {
    if (event == nullptr) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }
    return event;
}
#endif // GGML_CUDA_KERNEL_OP_AVAILABLE

bool ggml_cuda_kernel_op_supported(int device, const ggml_tensor * op) {
#ifdef GGML_CUDA_KERNEL_OP_AVAILABLE
    const ggml_cuda_kernel * k = ggml_cuda_kernel_of(op);
    if (k == nullptr || k->image == nullptr || k->name == nullptr || k->stream < 0 || k->stream > GGML_CUDA_KERNEL_OP_MAX_SIDE) {
        return false;
    }
    return ggml_cuda_kernel_function(device, *k) != nullptr;
#else
    GGML_UNUSED_VARS(device, op);
    return false;
#endif // GGML_CUDA_KERNEL_OP_AVAILABLE
}

void ggml_cuda_op_kernel(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#ifdef GGML_CUDA_KERNEL_OP_AVAILABLE
    const ggml_cuda_kernel & k = *ggml_cuda_kernel_of(dst);
    const CUfunction fun = ggml_cuda_kernel_function(ctx.device, k);
    GGML_ASSERT(fun != nullptr);

    cudaStream_t main   = ctx.stream();
    cudaStream_t stream = main;
    if (k.stream > 0) {
        // the side stream goes on after the nodes before this one
        stream = ctx.stream(ctx.device, GGML_CUDA_MAX_STREAMS - k.stream);
        const cudaEvent_t fork = ggml_cuda_kernel_op_event(ctx.kernel_op_events[0]);
        CUDA_CHECK(cudaEventRecord(fork, main));
        CUDA_CHECK(cudaStreamWaitEvent(stream, fork, 0));
        ctx.kernel_op_side_open |= 1u << k.stream;
    } else if (k.join) {
        for (int i = 1; i <= GGML_CUDA_KERNEL_OP_MAX_SIDE; ++i) {
            if ((ctx.kernel_op_side_open & (1u << i)) == 0) {
                continue;
            }
            const cudaEvent_t join = ggml_cuda_kernel_op_event(ctx.kernel_op_events[i]);
            CUDA_CHECK(cudaEventRecord(join, ctx.stream(ctx.device, GGML_CUDA_MAX_STREAMS - i)));
            CUDA_CHECK(cudaStreamWaitEvent(main, join, 0));
        }
        ctx.kernel_op_side_open = 0;
    }

    void * args[GGML_MAX_SRC + 2];
    int n_args = 0;
    args[n_args++] = &dst->data;
    for (int i = 0; i < GGML_MAX_SRC && dst->src[i] != nullptr; ++i) {
        args[n_args++] = &dst->src[i]->data;
    }
    if (k.params != nullptr) {
        args[n_args++] = const_cast<void *>(k.params);
    }

    CU_CHECK(cuLaunchKernel(fun, k.grid[0], k.grid[1], k.grid[2], k.block[0], k.block[1], k.block[2], k.smem, (CUstream) stream, args, nullptr));
#else
    GGML_UNUSED_VARS(ctx, dst); // not reached, see ggml_cuda_kernel_op_supported
#endif // GGML_CUDA_KERNEL_OP_AVAILABLE
}
