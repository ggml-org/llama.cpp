#include "sleep.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// %globaltimer is a nanosecond wall clock, unlike clock64() it is unaffected by the SM clock and by frequency scaling
static __device__ __forceinline__ uint64_t globaltimer_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// a single thread is enough, the following memcpy on the same stream cannot start before this kernel retires
static __global__ void sleep_ns(const uint64_t ns) {
    const uint64_t t0 = globaltimer_ns();

    while (globaltimer_ns() - t0 < ns) {}
}

void ggml_cuda_op_sleep(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    cudaStream_t stream = ctx.stream();

    sleep_ns<<<1, 1, 0, stream>>>(1000*(uint64_t) ggml_get_op_params_i32(dst, 0));
    CUDA_CHECK(cudaMemcpyAsync(dst->data, src0->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, stream));
}

#else

void ggml_cuda_op_sleep(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("GGML_OP_SLEEP requires the %%globaltimer register, which is only available on CUDA");
}

#endif
