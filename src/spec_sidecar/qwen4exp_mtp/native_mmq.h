// SPDX-License-Identifier: MIT
#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstddef>

namespace qwen4exp_sidecar {

// Exact current-tree gfx1030 Q4_0 prompt MMQ for native-MMQ widths 9–16.
// Native widths 2–8 dispatch MMVQ and are deliberately rejected by this
// wrapper. The caller supplies one block_q8_1_mmq scratch block per padded
// 128-value activation block and row, plus 16 guard blocks.
size_t native_q4_prompt_mmq_scratch_bytes(int input_width, int rows);
int launch_native_q4_prompt_mmq(
        hipStream_t stream,
        const void * weight_q4_0,
        const float * input,
        int input_width,
        int output_width,
        int rows,
        void * q8_scratch,
        size_t q8_scratch_bytes,
        float * output);

// Exact ggml gfx1030 F16 fallback: F32 activations round to F16, hipBLAS uses
// F16 inputs/accumulation/output, and the materialized result expands to F32.
int launch_native_f16_prompt_batch(
        hipStream_t stream,
        const __half * weight,
        const float * input,
        int input_width,
        int output_width,
        int rows,
        __half * input_f16_scratch,
        size_t input_f16_count,
        __half * output_f16_scratch,
        size_t output_f16_count,
        float * output);

} // namespace qwen4exp_sidecar
