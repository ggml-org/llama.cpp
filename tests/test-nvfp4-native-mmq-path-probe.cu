#include "../ggml/src/ggml-common.h"
#include "../ggml/src/ggml-cuda/common.cuh"
#include "../ggml/src/ggml-cuda/mmq.cuh"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr int k_probe_rows = 128;
constexpr int k_probe_cols = 8;
constexpr int k_probe_warp = 32;
constexpr int k_probe_nwarps = 8;

__global__ void nvfp4_native_mmq_path_probe_kernel(
        const block_nvfp4 * __restrict__ x,
        const block_nvfp4_mmq * __restrict__ y,
        float * __restrict__ out) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    extern __shared__ int smem[];

    int * ids    = smem;
    int * tile_y = ids + k_probe_cols;
    int * tile_x = tile_y + GGML_PAD(k_probe_cols * MMQ_TILE_Y_K, k_probe_nwarps * k_probe_warp);

    const int tid = threadIdx.y * k_probe_warp + threadIdx.x;

    if (tid < k_probe_cols) {
        ids[tid] = tid;
    }

    for (int l = tid; l < k_probe_cols * MMQ_TILE_Y_K; l += k_probe_nwarps * k_probe_warp) {
        tile_y[l] = reinterpret_cast<const int *>(y)[l];
    }

    __syncthreads();

    load_tiles_nvfp4_fp4<k_probe_rows, false>((const char *) x, tile_x, 0, k_probe_rows - 1, 1);

    __syncthreads();

    float sum[k_probe_cols * k_probe_rows / (k_probe_nwarps * k_probe_warp)] = { 0.0f };
    vec_dot_nvfp4_nvfp4_mma<k_probe_cols, k_probe_rows>(tile_x, tile_y, sum, 0);

    __syncthreads();

    mmq_write_back_mma<GGML_TYPE_NVFP4, k_probe_cols, k_probe_rows, false>(
        sum, ids, out, k_probe_rows, k_probe_rows - 1, k_probe_cols - 1, 1.0f);
#else
    GGML_UNUSED_VARS(x, y, out);
#endif
}

static bool nvfp4_native_mmq_probe_supported() {
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

static std::vector<block_nvfp4> pack_probe_rows(const block_nvfp4 * logical_rows) {
    const int packed_count = (k_probe_rows + 3) / 4;
    std::vector<block_nvfp4> packed((size_t) packed_count);
    std::memset(packed.data(), 0, packed.size() * sizeof(block_nvfp4));

    for (int row = 0; row < k_probe_rows; ++row) {
        const int pack = row >> 2;
        const int lane = row & 3;

        std::memcpy(packed[pack].scales[lane], logical_rows[row].scales[0], sizeof(packed[pack].scales[lane]));
        std::memcpy(packed[pack].qs[lane],     logical_rows[row].qs[0],     sizeof(packed[pack].qs[lane]));
    }

    return packed;
}

} // namespace

extern "C" bool ggml_cuda_nvfp4_native_mmq_path_probe(
        const void * x_host,
        const void * y_host,
        float * out_host) {
    if (!x_host || !y_host || !out_host) {
        return false;
    }
    if (!nvfp4_native_mmq_probe_supported()) {
        return false;
    }

    const block_nvfp4 * logical_x = reinterpret_cast<const block_nvfp4 *>(x_host);
    const block_nvfp4_mmq * logical_y = reinterpret_cast<const block_nvfp4_mmq *>(y_host);

    std::vector<block_nvfp4> packed_x = pack_probe_rows(logical_x);
    std::vector<block_nvfp4_mmq> probe_y(k_probe_cols);
    std::memcpy(probe_y.data(), logical_y, probe_y.size() * sizeof(block_nvfp4_mmq));

    block_nvfp4 * d_x = nullptr;
    block_nvfp4_mmq * d_y = nullptr;
    float * d_out = nullptr;

    if (cudaMalloc(&d_x, packed_x.size() * sizeof(block_nvfp4)) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&d_y, probe_y.size() * sizeof(block_nvfp4_mmq)) != cudaSuccess) {
        cudaFree(d_x);
        return false;
    }
    if (cudaMalloc(&d_out, k_probe_rows * k_probe_cols * sizeof(float)) != cudaSuccess) {
        cudaFree(d_y);
        cudaFree(d_x);
        return false;
    }

    bool ok = true;
    ok = ok && cudaMemcpy(d_x, packed_x.data(), packed_x.size() * sizeof(block_nvfp4), cudaMemcpyHostToDevice) == cudaSuccess;
    ok = ok && cudaMemcpy(d_y, probe_y.data(), probe_y.size() * sizeof(block_nvfp4_mmq), cudaMemcpyHostToDevice) == cudaSuccess;
    ok = ok && cudaMemset(d_out, 0, k_probe_rows * k_probe_cols * sizeof(float)) == cudaSuccess;

    if (ok) {
        constexpr size_t smem_bytes =
            (k_probe_cols +
             GGML_PAD(k_probe_cols * MMQ_TILE_Y_K, k_probe_nwarps * k_probe_warp) +
             k_probe_rows * MMQ_MMA_TILE_X_K_NVFP4) * sizeof(int);

        nvfp4_native_mmq_path_probe_kernel<<<dim3(1, 1, 1), dim3(k_probe_warp, k_probe_nwarps, 1), smem_bytes>>>(
            d_x, d_y, d_out);

        ok = ok && cudaGetLastError() == cudaSuccess;
        ok = ok && cudaDeviceSynchronize() == cudaSuccess;
        ok = ok && cudaMemcpy(out_host, d_out, k_probe_rows * k_probe_cols * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    cudaFree(d_out);
    cudaFree(d_y);
    cudaFree(d_x);

    return ok;
}
