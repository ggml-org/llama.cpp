#pragma once

// Simplified API for asynchronous data loading.

#include "common.cuh"


static __device__ __forceinline__ unsigned int ggml_cuda_cvta_generic_to_shared(void * generic_ptr) {
#ifdef CP_ASYNC_AVAILABLE
    return __cvta_generic_to_shared(generic_ptr);
#else
    GGML_UNUSED(generic_ptr);
    NO_DEVICE_CODE;
    return 0;
#endif // CP_ASYNC_AVAILABLE
}

// Copies data from global to shared memory, cg == cache global.
// Both the src and dst pointers must be aligned to 16 bit.
// Shared memory uses 32 bit addressing, the pointer is passed as unsigned int.
// Generic pointers can be converted to 32 bit shared memory pointers using __cvta_generic_to_shared.
// Only the 16 bit copy is exposed because 4 and 8 bit copies did not yield performance improvements.
template <int preload>
static __device__ __forceinline__ void cp_async_cg_16(const unsigned int dst, const void * src) {
    static_assert(preload == 0 || preload == 64 || preload == 128 || preload == 256, "bad preload");
#ifdef CP_ASYNC_AVAILABLE
#if CUDART_VERSION >= 11040
    if (preload == 256) {
        asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    } else if (preload == 128) {
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    } else if (preload == 64) {
        asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    } else
#endif // CUDART_VERSION >= 11040
    {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
            : : "r"(dst), "l"(src));
    }
#else
    GGML_UNUSED(dst);
    GGML_UNUSED(src);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// 4 byte copy through L1 (ca == cache all) for sources that are only 4 byte aligned, e.g. the 36 byte NVFP4 blocks.
static __device__ __forceinline__ void cp_async_ca_4(const unsigned int dst, const void * src) {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" : : "r"(dst), "l"(src));
#else
    GGML_UNUSED(dst);
    GGML_UNUSED(src);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// Groups the copies issued so far by this thread, see cp_async_wait_group.
static __device__ __forceinline__ void cp_async_commit_group() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;");
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// Waits until at most N of this thread's copy groups are still pending. Like cp_async_wait_all this
// does not synchronize between threads.
template <int N>
static __device__ __forceinline__ void cp_async_wait_group() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;" : : "n"(N));
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}

// Bulk asynchronous copies (sm_90+): one instruction moves a contiguous, 16 byte aligned range whose size is a
// multiple of 16 bytes and signals completion on an mbarrier in shared memory, without any per thread copy
// instructions or a block wide barrier. The waiting side spins on the barrier phase.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#define CP_ASYNC_BULK_AVAILABLE
#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// initialize an mbarrier that completes a phase after `arrivals` arrivals (and all expected transaction bytes)
static __device__ __forceinline__ void mbarrier_init(uint64_t * bar, const uint32_t arrivals) {
#ifdef CP_ASYNC_BULK_AVAILABLE
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" : : "r"(ggml_cuda_cvta_generic_to_shared(bar)), "r"(arrivals) : "memory");
#else
    GGML_UNUSED(bar);
    GGML_UNUSED(arrivals);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_BULK_AVAILABLE
}

// makes the initialized barriers visible to the async proxy, call once after all inits and before use
static __device__ __forceinline__ void mbarrier_fence_init() {
#ifdef CP_ASYNC_BULK_AVAILABLE
    asm volatile("fence.mbarrier_init.release.cluster;" : : : "memory");
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_BULK_AVAILABLE
}

static __device__ __forceinline__ void mbarrier_arrive(uint64_t * bar) {
#ifdef CP_ASYNC_BULK_AVAILABLE
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" : : "r"(ggml_cuda_cvta_generic_to_shared(bar)) : "memory");
#else
    GGML_UNUSED(bar);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_BULK_AVAILABLE
}

// arrive and register `bytes` of bulk copy transactions that must complete before the phase flips
static __device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t * bar, const uint32_t bytes) {
#ifdef CP_ASYNC_BULK_AVAILABLE
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" : : "r"(ggml_cuda_cvta_generic_to_shared(bar)), "r"(bytes) : "memory");
#else
    GGML_UNUSED(bar);
    GGML_UNUSED(bytes);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_BULK_AVAILABLE
}

// wait until the phase with the given parity (0 or 1) has completed
static __device__ __forceinline__ void mbarrier_wait_parity(uint64_t * bar, const uint32_t parity) {
#ifdef CP_ASYNC_BULK_AVAILABLE
    asm volatile(
        "{\n"
        ".reg .pred done;\n"
        "wait_loop_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
        "@!done bra wait_loop_%=;\n"
        "}\n"
        : : "r"(ggml_cuda_cvta_generic_to_shared(bar)), "r"(parity) : "memory");
#else
    GGML_UNUSED(bar);
    GGML_UNUSED(parity);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_BULK_AVAILABLE
}

// copy `bytes` (multiple of 16) from 16 byte aligned global memory to 16 byte aligned shared memory,
// completion is counted on the mbarrier
static __device__ __forceinline__ void cp_async_bulk_g2s(void * dst, const void * src, const uint32_t bytes, uint64_t * bar) {
#ifdef CP_ASYNC_BULK_AVAILABLE
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
        : : "r"(ggml_cuda_cvta_generic_to_shared(dst)), "l"(src), "r"(bytes), "r"(ggml_cuda_cvta_generic_to_shared(bar)) : "memory");
#else
    GGML_UNUSED(dst);
    GGML_UNUSED(src);
    GGML_UNUSED(bytes);
    GGML_UNUSED(bar);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_BULK_AVAILABLE
}

// 2D tensor copy (TMA) of one box described by the tensor map at coordinates (c0 in elements of the inner
// dimension, c1 rows) into 128 byte aligned shared memory, completion counted on the mbarrier. tmap must be a
// __grid_constant__ kernel parameter or live in global/constant memory.
static __device__ __forceinline__ void cp_async_bulk_tensor_2d_g2s(void * dst, const void * tmap, const int c0, const int c1, uint64_t * bar) {
#ifdef CP_ASYNC_BULK_AVAILABLE
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
        : : "r"(ggml_cuda_cvta_generic_to_shared(dst)), "l"(tmap), "r"(c0), "r"(c1), "r"(ggml_cuda_cvta_generic_to_shared(bar)) : "memory");
#else
    GGML_UNUSED(dst);
    GGML_UNUSED(tmap);
    GGML_UNUSED(c0);
    GGML_UNUSED(c1);
    GGML_UNUSED(bar);
    NO_DEVICE_CODE;
#endif // CP_ASYNC_BULK_AVAILABLE
}

// Makes each thread wait until its asynchronous data copies are done.
// This does NOT provide any additional synchronization.
// In particular, when copying data with multiple warps a call to __syncthreads will be needed.
static __device__ __forceinline__ void cp_async_wait_all() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_all;");
#else
    NO_DEVICE_CODE;
#endif // CP_ASYNC_AVAILABLE
}
