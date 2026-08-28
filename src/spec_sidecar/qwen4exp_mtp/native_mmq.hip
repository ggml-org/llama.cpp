// SPDX-License-Identifier: MIT
// Exact current-tree gfx1030 prompt-batch matrix contracts.
#include "native_mmq.h"

#include "../../../ggml/src/ggml-cuda/common.cuh"
#include "../../../ggml/src/ggml-cuda/mmq.cuh"

#include <hipblas/hipblas.h>
#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// The imported template headers retain defensive GGML_ABORT branches. All
// caller-controlled dimensions are checked before launch; keep those branches
// self-contained rather than adding a runtime ggml dependency.
extern "C" __attribute__((visibility("hidden"))) void ggml_abort(
        const char * file, int line, const char * format, ...) {
    std::fprintf(stderr, "qwen4exp native MMQ assertion at %s:%d: ", file, line);
    va_list arguments;
    va_start(arguments, format);
    std::vfprintf(stderr, format, arguments);
    va_end(arguments);
    std::abort();
}

namespace qwen4exp_sidecar {
namespace {

constexpr int PROMPT_J = 16;
constexpr int PROMPT_GUARD_BLOCKS = PROMPT_J;

template <mmq_q8_1_ds_layout layout>
__global__ void quantize_prompt_mmq_q8_1(const float * __restrict__ input,
        block_q8_1_mmq * __restrict__ output, int input_width,
        int padded_width, int rows) {
    constexpr int values_per_scale = layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int values_per_sum = layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;
    const int64_t i0 = (static_cast<int64_t>(blockDim.x) * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= padded_width) return;
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    const int64_t base = static_cast<int64_t>(row) * input_width;
    const float4 * input4 = reinterpret_cast<const float4 *>(input);
    const float4 values = i0 < input_width ? input4[(base + i0) / 4] :
            make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float maximum = fabsf(values.x);
    maximum = fmaxf(maximum, fabsf(values.y));
    maximum = fmaxf(maximum, fabsf(values.z));
    maximum = fmaxf(maximum, fabsf(values.w));
#pragma unroll
    for (int offset = values_per_scale / 8; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum,
                __shfl_xor_sync(0xffffffffu, maximum, offset, WARP_SIZE));
    }
    float sum = 0.0f;
    if constexpr (layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = values.x + values.y + values.z + values.w;
#pragma unroll
        for (int offset = values_per_sum / 8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffffu, sum, offset, WARP_SIZE);
        }
    }
    const float inverse = 127.0f / maximum;
    char4 quantized;
    quantized.x = roundf(values.x * inverse);
    quantized.y = roundf(values.y * inverse);
    quantized.z = roundf(values.z * inverse);
    quantized.w = roundf(values.w * inverse);
    const float scale = 1.0f / inverse;
    const int64_t k_block = i0 / QK8_1_MMQ;
    const int64_t quant_index = i0 % QK8_1_MMQ;
    const int64_t block = k_block * rows + row;
    reinterpret_cast<char4 *>(output[block].qs)[quant_index / 4] = quantized;
    if constexpr (layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        if (quant_index % 32 == 0) {
            output[block].ds4[quant_index / 32] = make_half2(scale, sum);
        }
    }
}

template <int prompt_j, bool fallback>
int launch_q4_prompt_mmq_quantized(hipStream_t stream, const char * weight,
        const int * activation, float * output, int input_width,
        int output_width, int rows) {
    constexpr int cc = GGML_CUDA_CC_RDNA2;
    const ggml_cuda_mmq_config config(
            GGML_TYPE_Q4_0, 256, 2, 128, prompt_j,
            GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K,
            false, false, fallback);
    const int shared = static_cast<int>(mmq_get_nbytes_shared(config, cc));
    const int nty = (output_width + config.I - 1) / config.I;
    const int ntx_value = (rows + prompt_j - 1) / prompt_j;
    const dim3 grid(nty, ntx_value, 1);
    const dim3 block(32, config.nthreads / 32, 1);
    const uint3 blocks_per_input = init_fastdiv_values(input_width / 32);
    const uint3 one = init_fastdiv_values(1);
    const uint3 ntx = init_fastdiv_values(ntx_value);
    hipLaunchKernelGGL((mul_mat_q<GGML_TYPE_Q4_0, prompt_j, fallback>),
            grid, block, shared, stream,
            weight, activation,
            static_cast<const int32_t *>(nullptr),
            static_cast<const int32_t *>(nullptr),
            static_cast<const int2 *>(nullptr),
            static_cast<const int32_t *>(nullptr),
            output, static_cast<float *>(nullptr), static_cast<const float *>(nullptr),
            blocks_per_input, output_width, rows, input_width / 32,
            rows, output_width,
            one, one, 0, 0, 0,
            one, one, 0, 0, 0, ntx);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

template <int prompt_j>
int launch_q4_prompt_mmq_for_j(hipStream_t stream, const void * weight_q4_0,
        const void * q8_scratch, float * output, int input_width,
        int output_width, int rows) {
    return output_width % 128 == 0 ?
            launch_q4_prompt_mmq_quantized<prompt_j, false>(stream,
                static_cast<const char *>(weight_q4_0),
                static_cast<const int *>(q8_scratch), output,
                input_width, output_width, rows) :
            launch_q4_prompt_mmq_quantized<prompt_j, true>(stream,
                static_cast<const char *>(weight_q4_0),
                static_cast<const int *>(q8_scratch), output,
                input_width, output_width, rows);
}

__global__ void prompt_f32_to_f16(const float * input, __half * output, size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __float2half(input[index]);
}

__global__ void prompt_f16_to_f32(const __half * input, float * output, size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = __half2float(input[index]);
}

} // namespace

__attribute__((visibility("hidden"))) size_t native_q4_prompt_mmq_scratch_bytes(
        int input_width, int rows) {
    if (input_width <= 0 || input_width % 32 != 0 ||
            rows < 9 || rows > PROMPT_J) return 0;
    const int padded = ((input_width + 511) / 512) * 512;
    const size_t blocks = static_cast<size_t>(rows) * padded / QK8_1_MMQ;
    return (blocks + PROMPT_GUARD_BLOCKS) * sizeof(block_q8_1_mmq);
}

__attribute__((visibility("hidden"))) int launch_native_q4_prompt_mmq(
        hipStream_t stream, const void * weight_q4_0, const float * input,
        int input_width, int output_width, int rows, void * q8_scratch,
        size_t q8_scratch_bytes, float * output) {
    const size_t needed = native_q4_prompt_mmq_scratch_bytes(input_width, rows);
    if (stream == nullptr || weight_q4_0 == nullptr || input == nullptr ||
            output == nullptr || q8_scratch == nullptr || needed == 0 ||
            q8_scratch_bytes < needed || output_width <= 0) return -1;
    const int padded = ((input_width + 511) / 512) * 512;
    if (hipMemsetAsync(q8_scratch, 0, needed, stream) != hipSuccess) return -1;
    const dim3 quant_grid(rows, (padded + 511) / 512, 1);
    quantize_prompt_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4>
            <<<quant_grid, 128, 0, stream>>>(input,
                static_cast<block_q8_1_mmq *>(q8_scratch), input_width, padded, rows);
    if (hipGetLastError() != hipSuccess) return -1;
    return launch_q4_prompt_mmq_for_j<PROMPT_J>(stream, weight_q4_0,
            q8_scratch, output, input_width, output_width, rows);
}

__attribute__((visibility("hidden"))) int launch_native_f16_prompt_batch(
        hipStream_t stream, const __half * weight, const float * input,
        int input_width, int output_width, int rows,
        __half * input_f16_scratch, size_t input_f16_count,
        __half * output_f16_scratch, size_t output_f16_count, float * output) {
    const size_t input_count = static_cast<size_t>(input_width) * rows;
    const size_t output_count = static_cast<size_t>(output_width) * rows;
    if (stream == nullptr || weight == nullptr || input == nullptr || output == nullptr ||
            input_f16_scratch == nullptr || output_f16_scratch == nullptr ||
            input_width <= 0 || output_width <= 0 || rows <= 1 ||
            input_f16_count < input_count || output_f16_count < output_count) return -1;
    prompt_f32_to_f16<<<(input_count + 255) / 256, 256, 0, stream>>>(
            input, input_f16_scratch, input_count);
    if (hipGetLastError() != hipSuccess) return -1;
    hipblasHandle_t handle = nullptr;
    if (hipblasCreate(&handle) != HIPBLAS_STATUS_SUCCESS) return -1;
    hipblasStatus_t status = hipblasSetStream(handle, stream);
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        status = hipblasGemmEx(handle, HIPBLAS_OP_T, HIPBLAS_OP_N,
                output_width, rows, input_width,
                &alpha,
                weight, HIPBLAS_R_16F, input_width,
                input_f16_scratch, HIPBLAS_R_16F, input_width,
                &beta,
                output_f16_scratch, HIPBLAS_R_16F, output_width,
                HIPBLAS_COMPUTE_16F, HIPBLAS_GEMM_DEFAULT);
    }
    const hipblasStatus_t destroy_status = hipblasDestroy(handle);
    if (status != HIPBLAS_STATUS_SUCCESS || destroy_status != HIPBLAS_STATUS_SUCCESS) return -1;
    prompt_f16_to_f32<<<(output_count + 255) / 256, 256, 0, stream>>>(
            output_f16_scratch, output, output_count);
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

} // namespace qwen4exp_sidecar
