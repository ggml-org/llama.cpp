// SPDX-License-Identifier: MIT
#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <rocprim/block/block_radix_sort.hpp>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace qwen4exp_sidecar {

static constexpr int N_EMBD = 2560;
static constexpr int N_HC = 4;
static constexpr int HC_WIDTH = N_EMBD * N_HC;
static constexpr int HC_LOW_RANK = 320;
static constexpr int N_HEAD = 24;
static constexpr int N_HEAD_KV = 2;
static constexpr int HEAD_DIM = 256;
static constexpr int N_ROT = 64;
static constexpr int N_EXPERT = 512;
static constexpr int N_EXPERT_USED = 10;
static constexpr int N_FF = 640;
static constexpr int N_VOCAB = 248320;
static constexpr int INDEX_HEADS = 4;
static constexpr int INDEX_DIM = 128;
static constexpr int INDEX_TOP_K = 2048;
static constexpr int QSA_RATIO = 4;
static constexpr float RMS_EPS = 1.0e-6f;
static constexpr float FREQ_BASE = 10000000.0f;

struct block_q4_0 {
    uint16_t d;
    uint8_t qs[16];
};

struct block_q8_1 {
    __half2 ds;
    int8_t qs[32];
};

// gfx1030's native Q4_0 MMVQ path retains the signed high-nibble sums from
// Q8_1 quantization so packed udot8/sdot8 reconstruct the exact integer dot.
struct block_q8_1_sum_hi {
    int8_t sum_hi[2];
};

static_assert(sizeof(block_q4_0) == 18, "Q4_0 block size");
static_assert(sizeof(block_q8_1) == 36, "Q8_1 block size");
static_assert(sizeof(block_q8_1_sum_hi) == 2, "Q8_1 sum-hi size");

__device__ __forceinline__ float half_bits_to_float(uint16_t bits) {
    __half h;
    std::memcpy(&h, &bits, sizeof(h));
    return __half2float(h);
}

__device__ __forceinline__ float q4_value(const block_q4_0 * block, int i) {
    const uint8_t packed = block->qs[i & 15];
    const int q = i < 16 ? packed & 15 : packed >> 4;
    return (q - 8) * half_bits_to_float(block->d);
}

__device__ __forceinline__ float warp_sum_32(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor(value, offset, 32);
    }
    return value;
}

__device__ __forceinline__ float fattn_wave_sum_32(float value) {
#if __has_builtin(__builtin_amdgcn_wave_reduce_fadd_f32)
    return __builtin_amdgcn_wave_reduce_fadd_f32(value, 0);
#else
    return warp_sum_32(value);
#endif
}

__device__ __forceinline__ float fattn_wave_max_32(float value) {
#if __has_builtin(__builtin_amdgcn_wave_reduce_fmax_f32)
    return __builtin_amdgcn_wave_reduce_fmax_f32(value, 0);
#else
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor(value, offset, 32));
    }
    return value;
#endif
}

__device__ __forceinline__ float block_sum(float value, float * shared) {
    value = warp_sum_32(value);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int wave = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) {
        shared[wave] = value;
    }
    __syncthreads();
    value = lane < static_cast<int>(blockDim.x) / 32 ? shared[lane] : 0.0f;
    return warp_sum_32(value);
}

__global__ void dequant_q4_row(const block_q4_0 * row, float * output, int width) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < width) {
        output[i] = q4_value(row + (i >> 5), i & 31);
    }
}

__global__ void dequant_q4_selected_row(const block_q4_0 * matrix,
        const int32_t * selected_row, float * output, int width, int rows) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < width) {
        const int row = *selected_row;
        output[i] = row >= 0 && row < rows ?
                q4_value(matrix + static_cast<size_t>(row) * (width / 32) + (i >> 5), i & 31) : NAN;
    }
}

__global__ void mark_nonfinite_f32(const float * input, size_t count, int32_t * nonfinite) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count && !isfinite(input[i])) {
        atomicExch(nonfinite, 1);
    }
}

// Reproduce ggml's Q8_1 activation bytes before each quantized matrix-vector
// product. Using the dequantized Q4 weights directly changes Flash Next's HC
// trajectory materially; Q8_1 parity keeps the native arithmetic within a few
// float ULPs on gfx1030.
__global__ void quantize_q8_1(const float * __restrict__ input,
        block_q8_1 * __restrict__ output, int width) {
    const int block = static_cast<int>(blockIdx.x);
    const int column = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    const float value = input[static_cast<size_t>(column) * width + block * 32 + lane];
    float maximum = fabsf(value);
    float sum = value;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor(maximum, offset, 32));
        sum += __shfl_xor(sum, offset, 32);
    }
    const float d = maximum / 127.0f;
    block_q8_1 & result = output[static_cast<size_t>(column) * (width / 32) + block];
    result.qs[lane] = d == 0.0f ? 0 : static_cast<int8_t>(roundf(value / d));
    if (lane == 0) {
        result.ds = __floats2half2_rn(d, sum);
    }
}

// Q8_1 plus the gfx1030 MMVQ high-nibble sidecar. This is deliberately a
// separate entry point: other architectures use the ordinary Q4_0 dot path.
__global__ void quantize_q8_1_sum_hi(const float * __restrict__ input,
        block_q8_1 * __restrict__ output,
        block_q8_1_sum_hi * __restrict__ sum_hi, int width) {
    const int block = static_cast<int>(blockIdx.x);
    const int column = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    const float value = input[static_cast<size_t>(column) * width + block * 32 + lane];
    float maximum = fabsf(value);
    float sum = value;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor(maximum, offset, 32));
        sum += __shfl_xor(sum, offset, 32);
    }
    const float d = maximum / 127.0f;
    const int quant = d == 0.0f ? 0 : static_cast<int>(roundf(value / d));
    const size_t at = static_cast<size_t>(column) * (width / 32) + block;
    output[at].qs[lane] = static_cast<int8_t>(quant);
    if (lane == 0) {
        output[at].ds = __floats2half2_rn(d, sum);
    }
    int low = 0;
    int high = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        low += __shfl(quant, i, 32) >> 4;
        high += __shfl(quant, 16 + i, 32) >> 4;
    }
    if (lane == 0) {
        sum_hi[at].sum_hi[0] = static_cast<int8_t>(low);
        sum_hi[at].sum_hi[1] = static_cast<int8_t>(high);
    }
}

__device__ __forceinline__ int packed_sdot8_i4(int a, int b, int c) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1030__)
    return __builtin_amdgcn_sdot8(a, b, c, false);
#else
    const uint32_t ua = static_cast<uint32_t>(a);
    const uint32_t ub = static_cast<uint32_t>(b);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int va = (ua >> (4 * i)) & 15;
        int vb = (ub >> (4 * i)) & 15;
        va = va & 8 ? va - 16 : va;
        vb = vb & 8 ? vb - 16 : vb;
        c += va * vb;
    }
    return c;
#endif
}

__device__ __forceinline__ int packed_udot8_u4(uint32_t a, uint32_t b, int c) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1030__)
    return static_cast<int>(__builtin_amdgcn_udot8(a, b, static_cast<uint32_t>(c), false));
#else
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        c += static_cast<int>((a >> (4 * i)) & 15) *
                static_cast<int>((b >> (4 * i)) & 15);
    }
    return c;
#endif
}

__global__ void gemv_q4_q8_batched(const block_q4_0 * __restrict__ weights,
        const block_q8_1 * __restrict__ input, float * __restrict__ output,
        int n_in, int n_out) {
    const int row = static_cast<int>(blockIdx.x);
    const int column = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= n_out) {
        return;
    }
    const int blocks = n_in / 32;
    const block_q4_0 * wr = weights + static_cast<size_t>(row) * blocks;
    const block_q8_1 * x = input + static_cast<size_t>(column) * blocks;
    float sum = 0.0f;
    const int half = lane & 1;
    for (int b = lane >> 1; b < blocks; b += 16) {
        const block_q4_0 * q4 = wr + b;
        const block_q8_1 * q8 = x + b;
        int integer_dot = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int at = half * 8 + j;
            const uint8_t packed = q4->qs[at];
            integer_dot += static_cast<int>(packed & 15) * static_cast<int>(q8->qs[at]);
            integer_dot += static_cast<int>(packed >> 4) * static_cast<int>(q8->qs[at + 16]);
        }
        const float2 ds8 = __half22float2(q8->ds);
        sum += half_bits_to_float(q4->d) * (integer_dot * ds8.x - 4.0f * ds8.y);
    }
    sum = warp_sum_32(sum);
    if (lane == 0) {
        output[static_cast<size_t>(column) * n_out + row] = sum;
    }
}

// Bit-exact clone of the target gfx1030 packed Q4_0 MMVQ arithmetic.
__global__ void gemv_q4_q8_dot8_batched(const block_q4_0 * __restrict__ weights,
        const block_q8_1 * __restrict__ input,
        const block_q8_1_sum_hi * __restrict__ sum_hi,
        float * __restrict__ output, int n_in, int n_out) {
    const int row = static_cast<int>(blockIdx.x);
    const int column = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= n_out) return;
    const int blocks = n_in / 32;
    const block_q4_0 * wr = weights + static_cast<size_t>(row) * blocks;
    const block_q8_1 * xr = input + static_cast<size_t>(column) * blocks;
    const block_q8_1_sum_hi * hr = sum_hi + static_cast<size_t>(column) * blocks;
    const int iqs = 2 * (lane & 1);
    float sum = 0.0f;
    for (int block = lane >> 1; block < blocks; block += 16) {
        const uint32_t * q4 = reinterpret_cast<const uint32_t *>(wr[block].qs);
        int acc_low = 0;
        int acc_high = 0;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint32_t qw_unsigned = q4[iqs + i];
            const uint32_t qw_signed = qw_unsigned ^ 0x88888888u;
            const uint32_t q8_low = reinterpret_cast<const uint32_t *>(xr[block].qs)[iqs + i];
            const uint32_t q8_high = reinterpret_cast<const uint32_t *>(xr[block].qs)[iqs + i + 4];
            const uint32_t activation_low = (q8_low & 0x0f0f0f0fu) |
                    ((q8_high & 0x0f0f0f0fu) << 4);
            const uint32_t activation_high = ((q8_low >> 4) & 0x0f0f0f0fu) |
                    (q8_high & 0xf0f0f0f0u);
            acc_low = packed_udot8_u4(qw_unsigned, activation_low, acc_low);
            acc_high = packed_sdot8_i4(static_cast<int>(qw_signed),
                    static_cast<int>(activation_high), acc_high);
        }
        const int integer_dot = acc_low + 16 * acc_high +
                128 * static_cast<int>(hr[block].sum_hi[iqs / 2]);
        const float2 ds8 = __half22float2(xr[block].ds);
        sum += half_bits_to_float(wr[block].d) *
                (integer_dot * ds8.x - 4.0f * ds8.y);
    }
    sum = warp_sum_32(sum);
    if (lane == 0) {
        output[static_cast<size_t>(column) * n_out + row] = sum;
    }
}

// Grid x selects independently normalized rows. The gamma vector is laid out
// as the same flattened rows, which supports both full-width MTP hnorm (one
// 10,240-wide row) and grouped HC norms (four 2,560-wide rows).
__global__ void rms_norm_rows(const float * __restrict__ input,
        const float * __restrict__ weight, float * __restrict__ output,
        int width) {
    __shared__ float partial[32];
    const int row = static_cast<int>(blockIdx.x);
    const float * x = input + static_cast<size_t>(row) * width;
    float * y = output + static_cast<size_t>(row) * width;
    float sum = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < width; i += static_cast<int>(blockDim.x)) {
        sum += x[i] * x[i];
    }
    sum = block_sum(sum, partial);
    const float scale = rsqrtf(sum / width + RMS_EPS);
    for (int i = static_cast<int>(threadIdx.x); i < width; i += static_cast<int>(blockDim.x)) {
        y[i] = scale * x[i] * weight[static_cast<size_t>(row) * width + i];
    }
}

__global__ void rms_norm_shared_rows(const float * __restrict__ input,
        const float * __restrict__ weight, float * __restrict__ output,
        int width) {
    __shared__ float partial[32];
    const int row = static_cast<int>(blockIdx.x);
    const float * x = input + static_cast<size_t>(row) * width;
    float * y = output + static_cast<size_t>(row) * width;
    float sum = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < width; i += static_cast<int>(blockDim.x)) {
        sum += x[i] * x[i];
    }
    sum = block_sum(sum, partial);
    const float scale = rsqrtf(sum / width + RMS_EPS);
    for (int i = static_cast<int>(threadIdx.x); i < width; i += static_cast<int>(blockDim.x)) {
        y[i] = scale * x[i] * weight[i];
    }
}

__global__ void rms_norm_heads(const float * __restrict__ input,
        const float * __restrict__ weight, float * __restrict__ output, int heads) {
    __shared__ float partial[32];
    const int head = static_cast<int>(blockIdx.x);
    if (head >= heads) {
        return;
    }
    const float * x = input + static_cast<size_t>(head) * HEAD_DIM;
    float * y = output + static_cast<size_t>(head) * HEAD_DIM;
    const int lane = static_cast<int>(threadIdx.x);
    const float value = lane < HEAD_DIM ? x[lane] : 0.0f;
    const float sum = block_sum(value * value, partial);
    if (lane < HEAD_DIM) {
        y[lane] = value * rsqrtf(sum / HEAD_DIM + RMS_EPS) * weight[lane];
    }
}

__global__ void repeat_embedding(const float * input, float * output) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < HC_WIDTH) {
        output[i] = input[i % N_EMBD];
    }
}

__global__ void pack_eh(const float * embedding, const float * hidden, float * output) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= N_HC * 2 * N_EMBD) {
        return;
    }
    const int stream = i / (2 * N_EMBD);
    const int offset = i % (2 * N_EMBD);
    output[i] = offset < N_EMBD ? embedding[stream * N_EMBD + offset]
                                : hidden[stream * N_EMBD + offset - N_EMBD];
}

__global__ void silu_scaled(float * values, int count, float scale) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) {
        float x = values[i] * scale;
        asm volatile("" : "+v"(x));
        values[i] = x / (1.0f + expf(-x));
    }
}

__global__ void sigmoid_values(float * values, int count) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) {
        values[i] = 1.0f / (1.0f + expf(-values[i]));
    }
}

__global__ void hc_mixed(const float * norm, const float * gate, float * output) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < N_EMBD) {
        float sum = 0.0f;
#pragma unroll
        for (int stream = 0; stream < N_HC; ++stream) {
            const int at = stream * N_EMBD + i;
            float product = norm[at] * gate[at];
            // Native graph has a materialized MUL followed by separate ADD
            // nodes. Prevent contraction/reassociation across that boundary.
            asm volatile("" : "+v"(product));
            sum += product;
            asm volatile("" : "+v"(sum));
        }
        output[i] = sum / static_cast<float>(N_HC);
    }
}

__global__ void float_to_half(const float * input, __half * output, int count) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) {
        output[i] = __float2half(input[i]);
    }
}

__global__ void store_cache_row_f16(const float * key, const float * value,
        const float * index_key, __half * key_cache, __half * value_cache,
        __half * index_cache, int position) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < N_HEAD_KV * HEAD_DIM) {
        key_cache[static_cast<size_t>(position) * N_HEAD_KV * HEAD_DIM + i] = __float2half(key[i]);
        value_cache[static_cast<size_t>(position) * N_HEAD_KV * HEAD_DIM + i] = __float2half(value[i]);
    }
    if (i < INDEX_DIM) {
        index_cache[static_cast<size_t>(position) * INDEX_DIM + i] = __float2half(index_key[i]);
    }
}

__global__ void copy_cache_rows_f16(__half * destination, const __half * source,
        int destination_row, int source_row, int row_count, int width) {
    const size_t count = static_cast<size_t>(row_count) * width;
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        destination[static_cast<size_t>(destination_row) * width + i] =
                source[static_cast<size_t>(source_row) * width + i];
    }
}

__global__ void shift_cache_rows_f16(__half * cache, int source_min,
        int source_max, int delta, int width) {
    const int step = delta > 0 ? -1 : 1;
    int position = delta > 0 ? source_max - 1 : source_min;
    for (int row = 0; row < source_max - source_min; ++row, position += step) {
        const size_t source = static_cast<size_t>(position) * width;
        const size_t destination = static_cast<size_t>(position + delta) * width;
        for (int i = static_cast<int>(threadIdx.x); i < width; i += static_cast<int>(blockDim.x)) {
            cache[destination + i] = cache[source + i];
        }
        __syncthreads();
    }
}

__global__ void clear_selection_mask(uint8_t * mask, int count) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) {
        mask[i] = 0;
    }
}

__global__ void mark_selection_mask(uint8_t * mask, const int32_t * selected,
        int selected_count, int limit) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < selected_count) {
        const int32_t position = selected[i];
        if (position >= 0 && position < limit) {
            mask[position] = 1;
        }
    }
}

__global__ void qsa_mask_to_f16(const uint8_t * __restrict__ selected,
        __half * __restrict__ mask, int count) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) mask[i] = selected[i] != 0 ? __float2half(0.0f) : __float2half(-INFINITY);
}

__global__ void attention_dense_f16_32(const float * __restrict__ query,
        const __half * __restrict__ key_cache, const __half * __restrict__ value_cache,
        float * __restrict__ output, int n_positions) {
    const int head = static_cast<int>(blockIdx.x);
    const int kv_head = head / (N_HEAD / N_HEAD_KV);
    const int lane = static_cast<int>(threadIdx.x);
    __shared__ __half probabilities[32];
    if (n_positions <= 0 || n_positions > 32 || blockDim.x != 32) {
        return;
    }
    const float * q = query + static_cast<size_t>(head) * HEAD_DIM;
    float score = -INFINITY;
    if (lane < n_positions) {
        const __half2 * key = reinterpret_cast<const __half2 *>(key_cache +
                (static_cast<size_t>(lane) * N_HEAD_KV + kv_head) * HEAD_DIM);
        score = 0.0f;
        for (int i = 0; i < HEAD_DIM / 2; ++i) {
            const __half2 q_pair = __floats2half2_rn(
                    q[2 * i] * rsqrtf(static_cast<float>(HEAD_DIM)),
                    q[2 * i + 1] * rsqrtf(static_cast<float>(HEAD_DIM)));
            score = __builtin_amdgcn_fdot2(q_pair, key[i], score, false);
        }
    }
    float max_score = fattn_wave_max_32(score);
    max_score += 3.0f * 0.6931f;
    const float probability = lane < n_positions ? expf(score - max_score) : 0.0f;
    probabilities[lane] = __float2half(probability);
    const float denominator = fattn_wave_sum_32(probability);
    __syncthreads();
    for (int dimension = lane; dimension < HEAD_DIM; dimension += 32) {
        __half sum = __float2half(0.0f);
        for (int position = 0; position < n_positions; ++position) {
            const __half * value = value_cache +
                    (static_cast<size_t>(position) * N_HEAD_KV + kv_head) * HEAD_DIM;
            sum = __hadd(sum, __hmul(probabilities[position], value[dimension]));
        }
        output[static_cast<size_t>(head) * HEAD_DIM + dimension] = __half2float(sum) / denominator;
    }
}

// Scan the physical cache in native 64-cell tiles while applying a QSA mask.
// One wave32 owns one query head, two cache cells per lane, and eight output
// dimensions per lane. This preserves tile-order F16 softmax/value arithmetic.
__global__ void attention_masked_f16(const float * __restrict__ query,
        const __half * __restrict__ key_cache, const __half * __restrict__ value_cache,
        const uint8_t * __restrict__ selected, float * __restrict__ output,
        int n_physical) {
    const int head = static_cast<int>(blockIdx.x);
    const int kv_head = head / (N_HEAD / N_HEAD_KV);
    const int lane = static_cast<int>(threadIdx.x);
    if (blockDim.x != 32 || n_physical <= 0 || n_physical % 64 != 0) {
        return;
    }
    const float * q = query + static_cast<size_t>(head) * HEAD_DIM;
    __shared__ __half probabilities[64];
    __half accumulators[HEAD_DIM / 32];
#pragma unroll
    for (int d = 0; d < HEAD_DIM / 32; ++d) {
        accumulators[d] = __float2half(0.0f);
    }
    float running_max = -FLT_MAX / 2.0f;
    float denominator_part = 0.0f;
    for (int base = 0; base < n_physical; base += 64) {
        float scores[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int slot = 0; slot < 2; ++slot) {
            const int position = base + slot * 32 + lane;
            if (selected[position] != 0) {
                const __half2 * key = reinterpret_cast<const __half2 *>(key_cache +
                        (static_cast<size_t>(position) * N_HEAD_KV + kv_head) * HEAD_DIM);
                float score = 0.0f;
                for (int i = 0; i < HEAD_DIM / 2; ++i) {
                    const __half2 q_pair = __floats2half2_rn(
                            q[2 * i] * rsqrtf(static_cast<float>(HEAD_DIM)),
                            q[2 * i + 1] * rsqrtf(static_cast<float>(HEAD_DIM)));
                    score = __builtin_amdgcn_fdot2(q_pair, key[i], score, false);
                }
                scores[slot] = score;
            }
        }
        const float tile_max = fattn_wave_max_32(fmaxf(scores[0], scores[1]));
        const float new_max = fmaxf(running_max, tile_max + 3.0f * 0.6931f);
        const float old_scale = expf(running_max - new_max);
        const __half old_scale_half = __float2half(old_scale);
#pragma unroll
        for (int d = 0; d < HEAD_DIM / 32; ++d) {
            accumulators[d] = __hmul(accumulators[d], old_scale_half);
        }
        float probability_sum = 0.0f;
#pragma unroll
        for (int slot = 0; slot < 2; ++slot) {
            const float probability = scores[slot] > -INFINITY
                    ? expf(scores[slot] - new_max) : 0.0f;
            probabilities[slot * 32 + lane] = __float2half(probability);
            probability_sum += probability;
        }
        denominator_part = denominator_part * old_scale + probability_sum;
        running_max = new_max;
        __syncthreads();
#pragma unroll
        for (int d = 0; d < HEAD_DIM / 32; ++d) {
            const int dimension = lane + d * 32;
#pragma unroll
            for (int offset = 0; offset < 64; ++offset) {
                const int position = base + offset;
                const __half * value = value_cache +
                        (static_cast<size_t>(position) * N_HEAD_KV + kv_head) * HEAD_DIM;
                accumulators[d] = __hadd(accumulators[d],
                        __hmul(probabilities[offset], value[dimension]));
            }
        }
        __syncthreads();
    }
    const float denominator = fattn_wave_sum_32(denominator_part);
#pragma unroll
    for (int d = 0; d < HEAD_DIM / 32; ++d) {
        const int dimension = lane + d * 32;
        output[static_cast<size_t>(head) * HEAD_DIM + dimension] =
                __half2float(accumulators[d]) / denominator;
    }
}

__global__ void gate_attention(float * attention, const float * gate, int count) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) {
        attention[i] *= 1.0f / (1.0f + expf(-gate[i]));
    }
}

__global__ void hc_combine(float * residual, const float * block_output,
        const float * inject) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < HC_WIDTH) {
        const int stream = i / N_EMBD;
        const float scatter = 2.0f / (1.0f + expf(-inject[stream] / static_cast<float>(N_HC)));
        float product = block_output[i % N_EMBD] * scatter;
        asm volatile("" : "+v"(product));
        residual[i] += product;
    }
}

__global__ void split_query_gate(const float * source, float * query, float * gate) {
    const int head = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    query[head * HEAD_DIM + lane] = source[head * 2 * HEAD_DIM + lane];
    gate[head * HEAD_DIM + lane] = source[head * 2 * HEAD_DIM + HEAD_DIM + lane];
}

__global__ void index_rope_neox(float * values, int rows, int position_base,
        int position_stride) {
    const int row = static_cast<int>(blockIdx.x);
    const int i = static_cast<int>(threadIdx.x);
    if (row >= rows || i >= N_ROT / 2) {
        return;
    }
    float * x = values + static_cast<size_t>(row) * INDEX_DIM;
    const int position = position_base + row * position_stride;
    const float theta_scale = powf(FREQ_BASE, -2.0f / N_ROT);
    const float theta = position * powf(theta_scale, static_cast<float>(i));
    const float c = cosf(theta);
    const float s = sinf(theta);
    const float x0 = x[i];
    const float x1 = x[i + N_ROT / 2];
    x[i] = x0 * c - x1 * s;
    x[i + N_ROT / 2] = x0 * s + x1 * c;
}

__global__ void qsa_pool_index_f16(const __half * __restrict__ index_cache,
        float * __restrict__ pooled, int complete_blocks) {
    const int block = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (block >= complete_blocks || lane >= INDEX_DIM) return;
    float sum = 0.0f;
#pragma unroll
    for (int member = 0; member < QSA_RATIO; ++member) {
        sum += __half2float(index_cache[
                static_cast<size_t>(block * QSA_RATIO + member) * INDEX_DIM + lane]);
    }
    pooled[static_cast<size_t>(block) * INDEX_DIM + lane] = sum * (1.0f / QSA_RATIO);
}

__device__ __forceinline__ uint64_t qsa_score_key(float value, int block) {
    if (block < 0 || value != value) return 0;
    if (value == 0.0f) value = 0.0f;
    const uint32_t bits = __float_as_uint(value);
    const uint32_t ordered = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
    return (static_cast<uint64_t>(ordered) << 32) |
            static_cast<uint32_t>(0xffffffffu - static_cast<uint32_t>(block));
}

// Hierarchical exact QSA selection. Each 2,048-item radix partition retains
// the best 513 blocks, which is the largest possible block count needed to
// represent INDEX_TOP_K + QSA_RATIO - 1 selected tokens. Repeating the same
// reduction over retained pairs is lossless for the global top 513 and scales
// to all 32,768 complete blocks in the declared 131,072-position state.
static constexpr int QSA_SORT_THREADS = 256;
static constexpr int QSA_SORT_ITEMS = 8;
static constexpr int QSA_SORT_CAPACITY = QSA_SORT_THREADS * QSA_SORT_ITEMS;
static constexpr int QSA_MAX_RANKED_BLOCKS = INDEX_TOP_K / QSA_RATIO + 1;

__global__ void qsa_select_candidates_scores(const float * __restrict__ scores,
        int score_count, uint64_t * __restrict__ output_keys,
        int32_t * __restrict__ output_blocks) {
    using sort_type = rocprim::block_radix_sort<uint64_t,
            QSA_SORT_THREADS, QSA_SORT_ITEMS, int32_t>;
    __shared__ typename sort_type::storage_type storage;
    uint64_t keys[QSA_SORT_ITEMS];
    int32_t blocks[QSA_SORT_ITEMS];
    const int base = static_cast<int>(blockIdx.x) * QSA_SORT_CAPACITY;
#pragma unroll
    for (int item = 0; item < QSA_SORT_ITEMS; ++item) {
        const int local = static_cast<int>(threadIdx.x) * QSA_SORT_ITEMS + item;
        const int block = base + local;
        blocks[item] = block < score_count ? block : -1;
        keys[item] = qsa_score_key(block < score_count ? scores[block] : -INFINITY,
                blocks[item]);
    }
    sort_type().sort_desc(keys, blocks, storage);
#pragma unroll
    for (int item = 0; item < QSA_SORT_ITEMS; ++item) {
        const int rank = static_cast<int>(threadIdx.x) * QSA_SORT_ITEMS + item;
        if (rank < QSA_MAX_RANKED_BLOCKS) {
            const int output = static_cast<int>(blockIdx.x) * QSA_MAX_RANKED_BLOCKS + rank;
            output_keys[output] = keys[item];
            output_blocks[output] = blocks[item];
        }
    }
}

__global__ void qsa_select_candidates_pairs(const uint64_t * __restrict__ input_keys,
        const int32_t * __restrict__ input_blocks, int input_count,
        uint64_t * __restrict__ output_keys, int32_t * __restrict__ output_blocks) {
    using sort_type = rocprim::block_radix_sort<uint64_t,
            QSA_SORT_THREADS, QSA_SORT_ITEMS, int32_t>;
    __shared__ typename sort_type::storage_type storage;
    uint64_t keys[QSA_SORT_ITEMS];
    int32_t blocks[QSA_SORT_ITEMS];
    const int base = static_cast<int>(blockIdx.x) * QSA_SORT_CAPACITY;
#pragma unroll
    for (int item = 0; item < QSA_SORT_ITEMS; ++item) {
        const int local = static_cast<int>(threadIdx.x) * QSA_SORT_ITEMS + item;
        const int input = base + local;
        keys[item] = input < input_count ? input_keys[input] : 0;
        blocks[item] = input < input_count ? input_blocks[input] : -1;
    }
    sort_type().sort_desc(keys, blocks, storage);
#pragma unroll
    for (int item = 0; item < QSA_SORT_ITEMS; ++item) {
        const int rank = static_cast<int>(threadIdx.x) * QSA_SORT_ITEMS + item;
        if (rank < QSA_MAX_RANKED_BLOCKS) {
            const int output = static_cast<int>(blockIdx.x) * QSA_MAX_RANKED_BLOCKS + rank;
            output_keys[output] = keys[item];
            output_blocks[output] = blocks[item];
        }
    }
}

__global__ void qsa_select_mask_final(const uint64_t * __restrict__ input_keys,
        const int32_t * __restrict__ input_blocks, int input_count,
        int complete_blocks, int query_position, uint8_t * __restrict__ selected,
        int32_t * __restrict__ selected_blocks) {
    using sort_type = rocprim::block_radix_sort<uint64_t,
            QSA_SORT_THREADS, QSA_SORT_ITEMS, int32_t>;
    __shared__ typename sort_type::storage_type storage;
    uint64_t keys[QSA_SORT_ITEMS];
    int32_t blocks[QSA_SORT_ITEMS];
#pragma unroll
    for (int item = 0; item < QSA_SORT_ITEMS; ++item) {
        const int input = static_cast<int>(threadIdx.x) * QSA_SORT_ITEMS + item;
        keys[item] = input < input_count ? input_keys[input] : 0;
        blocks[item] = input < input_count ? input_blocks[input] : -1;
    }
    sort_type().sort_desc(keys, blocks, storage);
    const int valid_tokens = query_position + 1;
    const int target_tokens = min(valid_tokens, INDEX_TOP_K + QSA_RATIO - 1);
    const int tail = valid_tokens % QSA_RATIO;
    const int ranked_tokens = target_tokens - tail;
    const int full_ranked_blocks = ranked_tokens / QSA_RATIO;
    const int partial_ranked_tokens = ranked_tokens % QSA_RATIO;
    const int ranked_blocks = full_ranked_blocks + (partial_ranked_tokens != 0);
#pragma unroll
    for (int item = 0; item < QSA_SORT_ITEMS; ++item) {
        const int rank = static_cast<int>(threadIdx.x) * QSA_SORT_ITEMS + item;
        if (rank < ranked_blocks) {
            const int block = blocks[item];
            if (block >= 0 && block < complete_blocks) {
                selected_blocks[rank] = block;
                const int members = rank < full_ranked_blocks ?
                        QSA_RATIO : partial_ranked_tokens;
                for (int member = 0; member < members; ++member) {
                    selected[block * QSA_RATIO + member] = 1;
                }
            }
        }
    }
    if (threadIdx.x == 0) {
        const int tail_start = complete_blocks * QSA_RATIO;
        for (int member = 0; member < tail; ++member) {
            selected[tail_start + member] = 1;
        }
    }
}

// Exact current-tree F32 MMVF contract used by Qwen4Exp index scores on
// gfx1030: four columns, 64 threads, float2 MADs, two wave32 reductions.
__global__ void index_head_dots_native(const float * __restrict__ keys,
        const float * __restrict__ query, float * __restrict__ head_dots,
        int blocks) {
    __shared__ float wave_sums[32];
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (row >= blocks || tid >= 64) return;
    if (tid < 32) wave_sums[tid] = 0.0f;
    __syncthreads();

    const float2 key = reinterpret_cast<const float2 *>(
            keys + static_cast<size_t>(row) * INDEX_DIM)[tid];
    float sums[INDEX_HEADS] = {0.0f};
#pragma unroll
    for (int head = 0; head < INDEX_HEADS; ++head) {
        const float2 value = reinterpret_cast<const float2 *>(
                query + static_cast<size_t>(head) * INDEX_DIM)[tid];
        sums[head] += key.x * value.x;
        sums[head] += key.y * value.y;
    }
#pragma unroll
    for (int head = 0; head < INDEX_HEADS; ++head) {
        sums[head] = warp_sum_32(sums[head]);
        wave_sums[tid / 32] = sums[head];
        __syncthreads();
        if (tid < 32) {
            sums[head] = wave_sums[tid];
            sums[head] = warp_sum_32(sums[head]);
        }
        __syncthreads();
    }
    if (tid < INDEX_HEADS) {
        head_dots[static_cast<size_t>(tid) * blocks + row] = sums[tid];
    }
}

__global__ void index_head_relu_transpose_native(
        const float * __restrict__ head_dots,
        float * __restrict__ block_heads, int blocks) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < blocks * INDEX_HEADS) {
        const int block = i / INDEX_HEADS;
        const int head = i % INDEX_HEADS;
        block_heads[i] = fmaxf(head_dots[static_cast<size_t>(head) * blocks + block], 0.0f);
    }
}

__global__ void index_sum_heads_native(const float * __restrict__ block_heads,
        float * __restrict__ scores, int blocks) {
    const int block = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (block >= blocks || lane >= 32) return;
    float value = lane < INDEX_HEADS ?
            block_heads[static_cast<size_t>(block) * INDEX_HEADS + lane] : 0.0f;
    value = warp_sum_32(value);
    if (lane == 0) scores[block] = value;
}

__global__ void index_scores(const float * __restrict__ keys,
        const float * __restrict__ query, float * __restrict__ scores,
        int blocks) {
    __shared__ float partial[INDEX_DIM];
    __shared__ float head_scores[INDEX_HEADS];
    const int block = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (block >= blocks || lane >= INDEX_DIM) {
        return;
    }
    const float key = keys[static_cast<size_t>(block) * INDEX_DIM + lane];
#pragma unroll
    for (int head = 0; head < INDEX_HEADS; ++head) {
        partial[lane] = key * query[head * INDEX_DIM + lane];
        __syncthreads();
        for (int offset = INDEX_DIM / 2; offset > 0; offset >>= 1) {
            if (lane < offset) {
                partial[lane] += partial[lane + offset];
            }
            __syncthreads();
        }
        if (lane == 0) {
            head_scores[head] = fmaxf(partial[0], 0.0f);
        }
        __syncthreads();
    }
    if (lane == 0) {
        float result = 0.0f;
#pragma unroll
        for (int head = 0; head < INDEX_HEADS; ++head) {
            result += head_scores[head];
        }
        scores[block] = result;
    }
}

__global__ void expand_qsa_scores(const float * __restrict__ block_scores,
        float * __restrict__ token_scores, int n_kv, int query_position) {
    const int cell = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (cell >= n_kv) {
        return;
    }
    if (cell > query_position) {
        token_scores[cell] = -INFINITY;
        return;
    }
    const int tail_start = ((query_position + 1) / QSA_RATIO) * QSA_RATIO;
    if (cell >= tail_start) {
        token_scores[cell] = block_scores[0] + 1.0e9f;
        return;
    }
    token_scores[cell] = block_scores[cell / QSA_RATIO];
}

// Text uses equal positions in all M-RoPE sections, reducing to NEOX rotation
// over the first 64 dimensions for this model.
__global__ void rope_neox(float * values, int heads, int position) {
    const int head = static_cast<int>(blockIdx.x);
    const int i = static_cast<int>(threadIdx.x);
    if (head >= heads || i >= N_ROT / 2) {
        return;
    }
    float * x = values + static_cast<size_t>(head) * HEAD_DIM;
    const float theta_scale = powf(FREQ_BASE, -2.0f / N_ROT);
    const float theta = position * powf(theta_scale, static_cast<float>(i));
    const float c = cosf(theta);
    const float s = sinf(theta);
    const float x0 = x[i];
    const float x1 = x[i + N_ROT / 2];
    x[i] = x0 * c - x1 * s;
    x[i + N_ROT / 2] = x0 * s + x1 * c;
}

// Match ggml's default F16-weight MMVF path: F32 activations are rounded to
// half2, products accumulate in half2 per thread, and wave results reduce in
// F32. The block size is selected from the input width by the caller.
template <int block_size>
__global__ void gemv_f16_f32_batched(const __half * __restrict__ weights,
        const float * __restrict__ input, float * __restrict__ output,
        int n_in, int n_out) {
    static_assert(block_size >= 32 && block_size <= 256 && block_size % 32 == 0,
            "F16 GEMV block must contain whole wave32 groups");
    const int row = static_cast<int>(blockIdx.x);
    const int column = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    __shared__ float wave_sums[32];
    if (tid < 32) {
        wave_sums[tid] = 0.0f;
    }
    __syncthreads();
    if (row >= n_out) {
        return;
    }
    const __half2 * matrix = reinterpret_cast<const __half2 *>(
            weights + static_cast<size_t>(row) * n_in);
    const float2 * vector = reinterpret_cast<const float2 *>(
            input + static_cast<size_t>(column) * n_in);
    __half2 sum_half = __floats2half2_rn(0.0f, 0.0f);
    for (int i = tid; i < n_in / 2; i += block_size) {
        const float2 x = vector[i];
        const __half2 x_half = __floats2half2_rn(x.x, x.y);
        sum_half = __hadd2(sum_half, __hmul2(matrix[i], x_half));
    }
    const float2 pair = __half22float2(sum_half);
    float sum = warp_sum_32(pair.x + pair.y);
    if constexpr (block_size > 32) {
        wave_sums[tid / 32] = sum;
        __syncthreads();
        if (tid < 32) {
            sum = warp_sum_32(wave_sums[tid]);
        }
    }
    if (tid == 0) {
        output[static_cast<size_t>(column) * n_out + row] = sum;
    }
}

} // namespace qwen4exp_sidecar
