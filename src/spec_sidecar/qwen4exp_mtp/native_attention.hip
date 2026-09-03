// SPDX-License-Identifier: MIT
// Instantiate the exact ggml gfx1030 F16 FATTN template used by the target.
#include "native_attention.h"

#include "../../../ggml/src/ggml-cuda/common.cuh"
#include "../../../ggml/src/ggml-cuda/fattn-tile.cuh"

#include <algorithm>

namespace qwen4exp_sidecar {
namespace {

static int choose_parallel_blocks(int physical, int & parallel) {
    constexpr int head_dim = 256;
    constexpr int heads = 24;
    constexpr int grouped_heads = 4;
    constexpr int nbatch_fa = 32;
    if (physical < 256 || physical > 131072 || physical % 256 != 0) return -1;
    const auto kernel = flash_attn_tile<head_dim, head_dim, 1, grouped_heads,
            false, true, GGML_TYPE_F16, GGML_TYPE_F16>;
    int device = -1;
    hipDeviceProp_t properties{};
    int max_blocks_per_sm = 0;
    if (hipGetDevice(&device) != hipSuccess ||
            hipGetDeviceProperties(&properties, device) != hipSuccess ||
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm,
                kernel, 32 * grouped_heads, 0) != hipSuccess ||
            max_blocks_per_sm <= 0 || properties.multiProcessorCount <= 0) return -1;

    const int ntiles_kv = (physical + nbatch_fa - 1) / nbatch_fa;
    parallel = std::min(max_blocks_per_sm, ntiles_kv);
    const int ntiles_dst = heads / grouped_heads;
    const int blocks_per_wave = properties.multiProcessorCount * max_blocks_per_sm;
    int nwaves_best = 0;
    int efficiency_best = 0;
    for (int candidate = parallel; candidate <= ntiles_kv; ++candidate) {
        const int total = ntiles_dst * candidate;
        const int nwaves = (total + blocks_per_wave - 1) / blocks_per_wave;
        const int efficiency = 100 * total / (nwaves * blocks_per_wave);
        if (efficiency_best >= 95 && nwaves > nwaves_best) break;
        if (efficiency > efficiency_best) {
            nwaves_best = nwaves;
            efficiency_best = efficiency;
            parallel = candidate;
        }
    }
    return parallel > 0 ? 0 : -1;
}

} // namespace

__attribute__((visibility("hidden"))) int launch_native_sparse_attention(
        hipStream_t stream,
        const float * query,
        const __half * key_position_major,
        const __half * value_position_major,
        const __half * mask,
        int physical,
        int partial_capacity,
        float * partial_output,
        float2 * partial_meta,
        float * output,
        int * parallel_used) {
    constexpr int head_dim = 256;
    constexpr int heads = 24;
    constexpr int kv_heads = 2;
    int parallel = 0;
    if (stream == nullptr || query == nullptr || key_position_major == nullptr ||
            value_position_major == nullptr || mask == nullptr ||
            partial_output == nullptr || partial_meta == nullptr || output == nullptr ||
            choose_parallel_blocks(physical, parallel) != 0 ||
            partial_capacity < parallel) return -1;

    // Preserved native-oracle anchor: physical = 6912 and parallel = 18 on V620.
    const uint3 ne01 = init_fastdiv_values(1);
    const dim3 tile_grid(1, parallel, heads / 4);
    const dim3 tile_block(32, 4, 1);
    hipLaunchKernelGGL(
            (flash_attn_tile<head_dim, head_dim, 1, 4, false, true,
                    GGML_TYPE_F16, GGML_TYPE_F16>),
            tile_grid, tile_block, 0, stream,
            reinterpret_cast<const char *>(query),
            reinterpret_cast<const char *>(key_position_major),
            reinterpret_cast<const char *>(value_position_major),
            reinterpret_cast<const char *>(mask),
            static_cast<const char *>(nullptr),
            static_cast<const int *>(nullptr),
            partial_output, partial_meta,
            0.0625f, 0.0f, 1.0f, 1.0f, 16u, 0.0f,
            head_dim, ne01, heads, 1,
            head_dim * static_cast<int>(sizeof(float)),
            head_dim * static_cast<int>(sizeof(float)),
            heads * head_dim * static_cast<int>(sizeof(float)),
            head_dim, physical, kv_heads, 1,
            kv_heads * head_dim * static_cast<int>(sizeof(__half)),
            head_dim * static_cast<int>(sizeof(__half)),
            static_cast<int64_t>(physical) * kv_heads * head_dim * sizeof(__half),
            kv_heads * head_dim * static_cast<int>(sizeof(__half)),
            head_dim * static_cast<int>(sizeof(__half)),
            static_cast<int64_t>(physical) * kv_heads * head_dim * sizeof(__half),
            1, 1, 1,
            physical * static_cast<int>(sizeof(__half)),
            physical * static_cast<int>(sizeof(__half)),
            static_cast<int64_t>(physical) * sizeof(__half));
    if (hipGetLastError() != hipSuccess) return -1;

    hipLaunchKernelGGL((flash_attn_combine_results<head_dim>),
            dim3(1, heads, 1), dim3(head_dim, 1, 1),
            parallel * sizeof(float2), stream,
            partial_output, partial_meta, output, parallel);
    if (hipGetLastError() != hipSuccess) return -1;
    if (parallel_used != nullptr) *parallel_used = parallel;
    return 0;
}

} // namespace qwen4exp_sidecar
