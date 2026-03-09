#include "../ggml/src/ggml-common.h"
#include "../ggml/src/ggml-cuda/common.cuh"
#include "../ggml/src/ggml-cuda/vecdotq.cuh"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>

namespace {

struct block_nvfp4_mmvq_probe {
    uint32_t sc4_u32[4];
    uint32_t qs_u32[32];
};

static_assert(sizeof(block_nvfp4_mmvq_probe) == 144, "unexpected block_nvfp4_mmq size");

constexpr int k_probe_rows = 4;
constexpr int k_probe_chunks = 16;

__global__ void nvfp4_mmvq_path_probe_kernel(
        const block_nvfp4 * __restrict__ x,
        const block_nvfp4_mmvq_probe * __restrict__ y,
        float * __restrict__ partial) {
    const int chunk = threadIdx.x;
    const int row = threadIdx.y;

    if (chunk >= k_probe_chunks || row >= k_probe_rows) {
        return;
    }

    partial[row * k_probe_chunks + chunk] =
        vec_dot_nvfp4_nvfp4_mmvq_mma((const void *) x, (const void *) y, row, chunk * 2);
}

static bool probe_supported() {
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

extern "C" bool ggml_cuda_nvfp4_mmvq_path_probe(
        const void * x_host,
        const void * y_host,
        float * out_host) {
    if (!x_host || !y_host || !out_host) {
        return false;
    }
    if (!probe_supported()) {
        return false;
    }

    const block_nvfp4 * logical_x = reinterpret_cast<const block_nvfp4 *>(x_host);
    const block_nvfp4_mmvq_probe * logical_y = reinterpret_cast<const block_nvfp4_mmvq_probe *>(y_host);

    block_nvfp4 * d_x = nullptr;
    block_nvfp4_mmvq_probe * d_y = nullptr;
    float * d_partial = nullptr;

    if (cudaMalloc(&d_x, sizeof(block_nvfp4)) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&d_y, sizeof(block_nvfp4_mmvq_probe)) != cudaSuccess) {
        cudaFree(d_x);
        return false;
    }
    if (cudaMalloc(&d_partial, k_probe_rows * k_probe_chunks * sizeof(float)) != cudaSuccess) {
        cudaFree(d_y);
        cudaFree(d_x);
        return false;
    }

    bool ok = true;
    ok = ok && cudaMemcpy(d_x, logical_x, sizeof(block_nvfp4), cudaMemcpyHostToDevice) == cudaSuccess;
    ok = ok && cudaMemcpy(d_y, logical_y, sizeof(block_nvfp4_mmvq_probe), cudaMemcpyHostToDevice) == cudaSuccess;
    ok = ok && cudaMemset(d_partial, 0, k_probe_rows * k_probe_chunks * sizeof(float)) == cudaSuccess;

    float host_partial[k_probe_rows * k_probe_chunks] = {};
    if (ok) {
        nvfp4_mmvq_path_probe_kernel<<<dim3(1, 1, 1), dim3(k_probe_chunks, k_probe_rows, 1)>>>(d_x, d_y, d_partial);

        ok = ok && cudaGetLastError() == cudaSuccess;
        ok = ok && cudaDeviceSynchronize() == cudaSuccess;
        ok = ok && cudaMemcpy(host_partial, d_partial, sizeof(host_partial), cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    if (ok) {
        for (int row = 0; row < k_probe_rows; ++row) {
            float sum = 0.0f;
            for (int chunk = 0; chunk < k_probe_chunks; ++chunk) {
                sum += host_partial[row * k_probe_chunks + chunk];
            }
            out_host[row] = sum;
        }
    }

    cudaFree(d_partial);
    cudaFree(d_y);
    cudaFree(d_x);

    return ok;
}
