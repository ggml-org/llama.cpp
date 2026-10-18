#include "sleep.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// %globaltimer is a nanosecond wall clock, unlike clock64() it is unaffected by the SM clock and by frequency scaling
static __device__ __forceinline__ uint64_t globaltimer_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// a single thread is enough, the next kernel on the same stream cannot start before this one retires
static __global__ void sleep_ns(const uint64_t ns) {
    const uint64_t t0 = globaltimer_ns();

    while (globaltimer_ns() - t0 < ns) {}
}

void ggml_cuda_op_sleep(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    sleep_ns<<<1, 1, 0, ctx.stream()>>>(1000*(uint64_t) ggml_get_op_params_i32(dst, 0));
    CUDA_CHECK(cudaGetLastError());
}

#else

void ggml_cuda_op_sleep(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("GGML_OP_SLEEP requires the %%globaltimer register, which is only available on CUDA");
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
