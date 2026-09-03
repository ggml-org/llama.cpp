// SPDX-License-Identifier: MIT
// Batched Qwen35MoE MTP catch-up kernels. Rows are packed as [row][feature].
#pragma once

namespace qwen35moe_sidecar {

__global__ void deq_rows_q4_0_b8(const block_q4_0 * __restrict__ weights,
        int t0, int t1, int t2, int t3, int t4, int t5, int t6, int t7,
        float * __restrict__ output, int n_rows) {
    const int feature = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    if (feature >= N_EMBD || row >= n_rows) return;
    const int tokens[8] = { t0, t1, t2, t3, t4, t5, t6, t7 };
    const int blocks = N_EMBD / 32;
    const block_q4_0 * block = weights + static_cast<size_t>(tokens[row]) * blocks + (feature >> 5);
    output[static_cast<size_t>(row) * N_EMBD + feature] = q4_0_get(block, feature & 31);
}

__global__ void rmsnorm_b(const float * __restrict__ input,
        const float * __restrict__ weights, float * __restrict__ output, int width) {
    __shared__ float reduction[256];
    const int row = static_cast<int>(blockIdx.x);
    const float * source = input + static_cast<size_t>(row) * width;
    float * destination = output + static_cast<size_t>(row) * width;
    float sum = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < width; i += static_cast<int>(blockDim.x)) {
        sum += source[i] * source[i];
    }
    reduction[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = static_cast<int>(blockDim.x) / 2; offset > 0; offset >>= 1) {
        if (static_cast<int>(threadIdx.x) < offset) {
            reduction[threadIdx.x] += reduction[threadIdx.x + offset];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(reduction[0] / width + RMS_EPS);
    for (int i = static_cast<int>(threadIdx.x); i < width; i += static_cast<int>(blockDim.x)) {
        destination[i] = source[i] * scale * weights[i];
    }
}

__global__ void concat2_b(const float * __restrict__ first,
        const float * __restrict__ second, float * __restrict__ output) {
    const int feature = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    if (feature < N_EMBD) {
        output[static_cast<size_t>(row) * 2 * N_EMBD + feature] =
                first[static_cast<size_t>(row) * N_EMBD + feature];
    } else if (feature < 2 * N_EMBD) {
        output[static_cast<size_t>(row) * 2 * N_EMBD + feature] =
                second[static_cast<size_t>(row) * N_EMBD + feature - N_EMBD];
    }
}

__global__ void gemv_q4_0_b(const block_q4_0 * __restrict__ weights,
        const float * __restrict__ input, float * __restrict__ output,
        int n_in, int n_out) {
    const int output_row = static_cast<int>(blockIdx.x);
    const int batch_row = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (output_row >= n_out) return;
    const int blocks = n_in / 32;
    const block_q4_0 * row_weights = weights + static_cast<size_t>(output_row) * blocks;
    const float * row_input = input + static_cast<size_t>(batch_row) * n_in;
    float sum = 0.0f;
    for (int block = lane; block < blocks; block += 32) {
        const block_q4_0 * q4 = row_weights + block;
        const float scale = f16_to_f32(q4->d);
        const float * x = row_input + block * 32;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint8_t packed = q4->qs[i];
            dot += static_cast<float>((packed & 0x0f) - 8) * x[i];
            dot += static_cast<float>((packed >> 4) - 8) * x[i + 16];
        }
        sum += scale * dot;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down(sum, offset, 32);
    if (lane == 0) output[static_cast<size_t>(batch_row) * n_out + output_row] = sum;
}

__global__ void rmsnorm_heads_b(const float * __restrict__ input,
        const float * __restrict__ weights, float * __restrict__ output, int n_heads) {
    __shared__ float reduction[256];
    const int head = static_cast<int>(blockIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    const size_t offset = (static_cast<size_t>(row) * n_heads + head) * HEAD_DIM;
    const float * source = input + offset;
    float * destination = output + offset;
    float sum = 0.0f;
    for (int i = lane; i < HEAD_DIM; i += static_cast<int>(blockDim.x)) sum += source[i] * source[i];
    reduction[lane] = sum;
    __syncthreads();
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride >>= 1) {
        if (lane < stride) reduction[lane] += reduction[lane + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(reduction[0] / HEAD_DIM + RMS_EPS);
    for (int i = lane; i < HEAD_DIM; i += static_cast<int>(blockDim.x)) destination[i] = source[i] * scale * weights[i];
}

__global__ void rope_neox_b(float * __restrict__ values, int n_heads, int position_base) {
    const int head = static_cast<int>(blockIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    const int i = static_cast<int>(threadIdx.x);
    if (i >= N_ROT / 2) return;
    float * current = values + (static_cast<size_t>(row) * n_heads + head) * HEAD_DIM;
    const float theta = (position_base + row) * powf(FREQ_BASE, -2.0f * i / N_ROT);
    const float cosine = cosf(theta);
    const float sine = sinf(theta);
    const float first = current[i];
    const float second = current[i + N_ROT / 2];
    current[i] = first * cosine - second * sine;
    current[i + N_ROT / 2] = first * sine + second * cosine;
}

__global__ void kv_store_b(const float * __restrict__ key,
        const float * __restrict__ value, __half * __restrict__ key_cache,
        __half * __restrict__ value_cache, int position_base, int pending_base) {
    const int feature = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    const int width = N_HEAD_KV * HEAD_DIM;
    if (feature >= width) return;
    const size_t source = static_cast<size_t>(row) * width + feature;
    const size_t destination = static_cast<size_t>(position_base + row - pending_base) * width + feature;
    key_cache[destination] = __float2half(key[source]);
    value_cache[destination] = __float2half(value[source]);
}

} // namespace qwen35moe_sidecar
