#pragma once
#include "kernels.hip"

namespace qwen4exp_sidecar {

template <int block_size>
__global__ void gemv_f32_f32_batched(const float * __restrict__ weights,
        const float * __restrict__ input, float * __restrict__ output,
        int n_in, int n_out) {
    static_assert(block_size >= 32 && block_size <= 256 && block_size % 32 == 0,
            "F32 GEMV block must contain whole wave32 groups");
    const int row = static_cast<int>(blockIdx.x);
    const int column = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    __shared__ float wave_sums[32];
    if (tid < 32) wave_sums[tid] = 0.0f;
    __syncthreads();
    if (row >= n_out) return;
    const float2 * matrix = reinterpret_cast<const float2 *>(
            weights + static_cast<size_t>(row) * n_in);
    const float2 * vector = reinterpret_cast<const float2 *>(
            input + static_cast<size_t>(column) * n_in);
    float sum = 0.0f;
    for (int i = tid; i < n_in / 2; i += block_size) {
        const float2 w = matrix[i];
        const float2 x = vector[i];
        sum = fmaf(w.x, x.x, sum);
        sum = fmaf(w.y, x.y, sum);
    }
    sum = warp_sum_32(sum);
    if constexpr (block_size > 32) {
        wave_sums[tid / 32] = sum;
        __syncthreads();
        if (tid < 32) sum = warp_sum_32(wave_sums[tid]);
    }
    if (tid == 0) output[static_cast<size_t>(column) * n_out + row] = sum;
}

__global__ void router_top10(const float * __restrict__ logits,
        int32_t * __restrict__ selected, float * __restrict__ weights) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float maximum = -INFINITY;
    for (int i = 0; i < N_EXPERT; ++i) maximum = fmaxf(maximum, logits[i]);
    float probabilities[N_EXPERT];
    float total = 0.0f;
    for (int i = 0; i < N_EXPERT; ++i) {
        probabilities[i] = expf(logits[i] - maximum);
        total += probabilities[i];
    }
    for (int i = 0; i < N_EXPERT; ++i) probabilities[i] /= total;
    for (int slot = 0; slot < N_EXPERT_USED; ++slot) {
        int best = -1;
        for (int i = 0; i < N_EXPERT; ++i) {
            bool used = false;
            for (int prior = 0; prior < slot; ++prior) used = used || selected[prior] == i;
            if (!used && (best < 0 || probabilities[i] > probabilities[best] ||
                    (probabilities[i] == probabilities[best] && i < best))) best = i;
        }
        selected[slot] = best;
        weights[slot] = probabilities[best];
    }
    float selected_sum = 0.0f;
    for (int slot = 0; slot < N_EXPERT_USED; ++slot) selected_sum += weights[slot];
    selected_sum = fmaxf(selected_sum, 6.103515625e-5f);
    for (int slot = 0; slot < N_EXPERT_USED; ++slot) weights[slot] /= selected_sum;
}

// Reproduce the target graph's 512-thread softmax, argsort, 512-thread
// SUM_ROWS reduction, clamp, and division rather than serializing the router.
__global__ void router_top10_native(const float * __restrict__ logits,
        int32_t * __restrict__ selected, float * __restrict__ weights) {
    __shared__ float workspace[32];
    __shared__ float probabilities[N_EXPERT];
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int wave = tid >> 5;
    float maximum = logits[tid];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor(maximum, offset, 32));
    }
    if (lane == 0) workspace[wave] = maximum;
    __syncthreads();
    maximum = lane < N_EXPERT / 32 ? workspace[lane] : -INFINITY;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor(maximum, offset, 32));
    }
    const float probability = expf(logits[tid] - maximum);
    probabilities[tid] = probability;
    __syncthreads();
    float total = probability;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        total += __shfl_xor(total, offset, 32);
    }
    if (lane == 0) workspace[wave] = total;
    __syncthreads();
    total = lane < N_EXPERT / 32 ? workspace[lane] : 0.0f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        total += __shfl_xor(total, offset, 32);
    }
    probabilities[tid] *= 1.0f / total;
    __syncthreads();
    if (tid == 0) {
        for (int slot = 0; slot < N_EXPERT_USED; ++slot) {
            int best = -1;
            for (int i = 0; i < N_EXPERT; ++i) {
                bool used = false;
                for (int prior = 0; prior < slot; ++prior) used |= selected[prior] == i;
                if (!used && (best < 0 || probabilities[i] > probabilities[best] ||
                        (probabilities[i] == probabilities[best] && i < best))) best = i;
            }
            selected[slot] = best;
        }
    }
    __syncthreads();
    float selected_probability = tid < N_EXPERT_USED ? probabilities[selected[tid]] : 0.0f;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        selected_probability += __shfl_xor(selected_probability, offset, 32);
    }
    const float selected_sum = fmaxf(selected_probability, 6.103515625e-5f);
    if (tid < N_EXPERT_USED) {
        weights[tid] = probabilities[selected[tid]] / selected_sum;
    }
}

__global__ void gemv_q4_q8_experts(const block_q4_0 * __restrict__ weights,
        const block_q8_1 * __restrict__ input, const int32_t * __restrict__ experts,
        float * __restrict__ output, int n_in, int n_out, int experts_used,
        int input_stride_blocks) {
    const int row = static_cast<int>(blockIdx.x);
    const int slot = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= n_out || slot >= experts_used) return;
    const int blocks = n_in / 32;
    const size_t expert_stride = static_cast<size_t>(n_out) * blocks;
    const block_q4_0 * wr = weights + static_cast<size_t>(experts[slot]) * expert_stride +
            static_cast<size_t>(row) * blocks;
    float sum = 0.0f;
    const int half = lane & 1;
    for (int b = lane >> 1; b < blocks; b += 16) {
        const block_q4_0 * q4 = wr + b;
        const block_q8_1 * q8 = input + static_cast<size_t>(slot) * input_stride_blocks + b;
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
    if (lane == 0) output[static_cast<size_t>(slot) * n_out + row] = sum;
}

// Expert-indexed form of the bit-exact gfx1030 packed Q4_0 MMVQ path.
__global__ void gemv_q4_q8_dot8_experts(const block_q4_0 * __restrict__ weights,
        const block_q8_1 * __restrict__ input,
        const block_q8_1_sum_hi * __restrict__ sum_hi,
        const int32_t * __restrict__ experts, float * __restrict__ output,
        int n_in, int n_out, int experts_used, int input_stride_blocks) {
    const int row = static_cast<int>(blockIdx.x);
    const int slot = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= n_out || slot >= experts_used) return;
    const int blocks = n_in / 32;
    const size_t expert_stride = static_cast<size_t>(n_out) * blocks;
    const block_q4_0 * wr = weights + static_cast<size_t>(experts[slot]) * expert_stride +
            static_cast<size_t>(row) * blocks;
    const block_q8_1 * xr = input + static_cast<size_t>(slot) * input_stride_blocks;
    const block_q8_1_sum_hi * hr = sum_hi + static_cast<size_t>(slot) * input_stride_blocks;
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
    if (lane == 0) output[static_cast<size_t>(slot) * n_out + row] = sum;
}

__global__ void silu_multiply(float * gate, const float * up, int count) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) {
        const float value = gate[i];
        gate[i] = value / (1.0f + expf(-value)) * up[i];
    }
}

__global__ void weighted_expert_sum(const float * expert_outputs,
        const float * weights, float * output, int width, int experts_used) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < width) {
        float sum = expert_outputs[i] * weights[0];
        for (int expert = 1; expert < experts_used; ++expert) {
            sum += expert_outputs[static_cast<size_t>(expert) * width + i] * weights[expert];
        }
        output[i] = sum;
    }
}

// The target graph materializes MUL and then chains ten expert ADD nodes in
// expert order. Keep both boundaries so the compiler cannot contract/reorder.
__global__ void weighted_expert_sum_native(const float * expert_outputs,
        const float * weights, float * output, int width, int experts_used) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= width) return;
    float sum = expert_outputs[i] * weights[0];
    asm volatile("" : "+v"(sum));
    for (int expert = 1; expert < experts_used; ++expert) {
        float product = expert_outputs[static_cast<size_t>(expert) * width + i] * weights[expert];
        asm volatile("" : "+v"(product));
        sum += product;
        asm volatile("" : "+v"(sum));
    }
    output[i] = sum;
}

__global__ void sigmoid_scale(float * values, const float * scalar, int count) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) {
        values[i] *= 1.0f / (1.0f + expf(-scalar[0]));
    }
}

__global__ void add_vectors(float * destination, const float * source, int count) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) destination[i] += source[i];
}

} // namespace qwen4exp_sidecar
