#include "../ggml/src/ggml-cuda/common.cuh"
#include "../ggml/src/ggml-cuda/mma.cuh"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace {

using namespace ggml_cuda_mma;

__global__ void nvfp4_native_mma_scale_probe_kernel(
        const uint32_t a_regs,
        const uint32_t b_regs,
        const uint32_t a_scale,
        const uint32_t b_scale,
        float * out_acc) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    tile<16, 8, int>   A;
    tile<8, 8, int>    B;
    tile<16, 8, float> C;

#pragma unroll
    for (int i = 0; i < tile<16, 8, int>::ne; ++i) {
        A.x[i] = (int) a_regs;
    }

#pragma unroll
    for (int i = 0; i < tile<8, 8, int>::ne; ++i) {
        B.x[i] = (int) b_regs;
    }

#pragma unroll
    for (int i = 0; i < tile<16, 8, float>::ne; ++i) {
        C.x[i] = 0.0f;
    }

    mma_nvfp4_block_scaled(C, A, B, a_scale, b_scale);

#pragma unroll
    for (int i = 0; i < tile<16, 8, float>::ne; ++i) {
        out_acc[threadIdx.x * tile<16, 8, float>::ne + i] = C.x[i];
    }
#else
    GGML_UNUSED_VARS(a_regs, b_regs, a_scale, b_scale, out_acc);
#endif
}

static bool nvfp4_native_mma_probe_supported() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }

    cudaDeviceProp prop = {};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return false;
    }

    const int cc = 100 * prop.major + 10 * prop.minor;
    return blackwell_mma_available(cc) && ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_BLACKWELL;
}

} // namespace

extern "C" bool ggml_cuda_nvfp4_native_mma_scale_probe(
        const uint32_t a_regs,
        const uint32_t b_regs,
        const uint32_t a_scale,
        const uint32_t b_scale,
        float * out_acc) {
    if (!out_acc) {
        return false;
    }
    if (!nvfp4_native_mma_probe_supported()) {
        return false;
    }

    float * d_out = nullptr;
    if (cudaMalloc(&d_out, 32 * 4 * sizeof(float)) != cudaSuccess) {
        return false;
    }

    nvfp4_native_mma_scale_probe_kernel<<<1, 32>>>(a_regs, b_regs, a_scale, b_scale, d_out);

    const cudaError_t launch_err = cudaGetLastError();
    const cudaError_t sync_err   = cudaDeviceSynchronize();
    const cudaError_t copy_err   = (launch_err == cudaSuccess && sync_err == cudaSuccess)
        ? cudaMemcpy(out_acc, d_out, 32 * 4 * sizeof(float), cudaMemcpyDeviceToHost)
        : cudaErrorUnknown;

    cudaFree(d_out);

    return launch_err == cudaSuccess && sync_err == cudaSuccess && copy_err == cudaSuccess;
}
