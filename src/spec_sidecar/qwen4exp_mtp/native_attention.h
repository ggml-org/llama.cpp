// SPDX-License-Identifier: MIT
#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace qwen4exp_sidecar {

// Exact gfx1030 native-template path. The caller owns all storage and stream
// ordering. Physical cache widths must be 256-cell aligned; the host dispatch
// reproduces ggml's occupancy/efficiency choice of parallel partitions.
int launch_native_sparse_attention(
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
        int * parallel_used);

} // namespace qwen4exp_sidecar
