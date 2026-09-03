// SPDX-License-Identifier: MIT
// HIP MTP sidecar kernels for Qwen3.6/Qwen3.5-MoE 35B-A3B.
//
// This provider is deliberately separate from the dense Qwen3.8-27B provider:
// the MoE MTP block has a 2048-wide hidden state, 16/2 attention heads, 256
// routed experts, and a full-vocabulary shared output head.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstring>

namespace qwen35moe_sidecar {

static constexpr int N_EMBD = 2048;
static constexpr int N_HEAD = 16;
static constexpr int N_HEAD_KV = 2;
static constexpr int HEAD_DIM = 256;
static constexpr int N_ROT = 64;
static constexpr int N_FF = 512;
static constexpr int N_EXPERT = 256;
static constexpr int N_EXPERT_USED = 8;
static constexpr int SC_N_VOCAB = 248320;
static constexpr int N_HEAD_ROWS = 40960;
static constexpr int MTP_DRAFT_TOP_K = 32;
static constexpr float RMS_EPS = 1e-6f;
static constexpr float FREQ_BASE = 10000000.0f;
static constexpr int ATTN_CHUNK = 2048;

struct block_q4_0 {
    uint16_t d;
    uint8_t qs[16];
};
static_assert(sizeof(block_q4_0) == 18, "q4_0 block size");

__device__ __forceinline__ float f16_to_f32(uint16_t bits) {
    __half value;
    memcpy(&value, &bits, sizeof(value));
    return __half2float(value);
}

__device__ __forceinline__ float q4_0_get(const block_q4_0 * block, int index) {
    const uint8_t packed = block->qs[index & 15];
    const int quant = index < 16 ? packed & 0x0f : packed >> 4;
    return (quant - 8) * f16_to_f32(block->d);
}

#if defined(__gfx1030__) || defined(SPEC_SIDECAR_GFX1030)
struct block_q8_1 {
    __half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(block_q8_1) == 36, "q8_1 block size");

struct block_q8_1_sum_hi {
    int8_t sum_hi[2];
};
static_assert(sizeof(block_q8_1_sum_hi) == 2, "q8_1 high-nibble sum size");

__device__ __forceinline__ uint32_t get_int4(const int8_t * values, int index) {
    return reinterpret_cast<const uint32_t *>(values)[index];
}

__device__ __forceinline__ uint32_t get_int2(const uint8_t * values, int index) {
    const uint16_t * values16 = reinterpret_cast<const uint16_t *>(values);
    return static_cast<uint32_t>(values16[2 * index]) |
            (static_cast<uint32_t>(values16[2 * index + 1]) << 16);
}

__device__ __forceinline__ int udot8_u4(uint32_t a, uint32_t b, int accumulator) {
    return static_cast<int>(__builtin_amdgcn_udot8(a, b, static_cast<uint32_t>(accumulator), false));
}

__device__ __forceinline__ int sdot8_i4(int a, int b, int accumulator) {
    return __builtin_amdgcn_sdot8(a, b, accumulator, false);
}

template <int vdr>
__device__ __forceinline__ float q4_q8_dot8(const block_q4_0 * q4,
        const block_q8_1 * q8, const block_q8_1_sum_hi * sum_hi, int offset) {
    int low = 0;
    int high = 0;
#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        const uint32_t q4_unsigned = get_int2(q4->qs, offset + i);
        const uint32_t q4_signed = q4_unsigned ^ 0x88888888u;
        const uint32_t q8_0 = get_int4(q8->qs, offset + i);
        const uint32_t q8_1 = get_int4(q8->qs, offset + i + 4);
        const uint32_t q8_low = (q8_0 & 0x0f0f0f0fu) | ((q8_1 & 0x0f0f0f0fu) << 4);
        const uint32_t q8_high = ((q8_0 >> 4) & 0x0f0f0f0fu) | (q8_1 & 0xf0f0f0f0u);
        low = udot8_u4(q4_unsigned, q8_low, low);
        high = sdot8_i4(static_cast<int>(q4_signed), static_cast<int>(q8_high), high);
    }
    const int integer_dot = low + 16 * high + 128 * static_cast<int>(sum_hi->sum_hi[offset / vdr]);
    const float2 scales = __half22float2(q8->ds);
    return f16_to_f32(q4->d) * (integer_dot * scales.x - (8 * vdr / 4) * scales.y);
}

__global__ void rmsnorm_q8_1(const float * __restrict__ input, const float * __restrict__ weights,
        float * __restrict__ output, block_q8_1 * __restrict__ quantized,
        block_q8_1_sum_hi * __restrict__ sum_hi, int n) {
    __shared__ float sums[256];
    float sum = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
        sum += input[i] * input[i];
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
        __syncthreads();
    }
    const float scale = rsqrtf(sums[0] / n + RMS_EPS);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int blocks = n / 32;
    for (int block = warp; block < blocks; block += blockDim.x / 32) {
        const int index = block * 32 + lane;
        const float value = input[index] * scale * weights[index];
        if (output != nullptr) output[index] = value;
        float amax = fabsf(value);
        float total = value;
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_down(amax, offset, 32));
            total += __shfl_down(total, offset, 32);
        }
        amax = __shfl(amax, 0, 32);
        total = __shfl(total, 0, 32);
        const float d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / d));
        quantized[block].qs[lane] = q;
        int high0 = 0;
        int high1 = 0;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            high0 += __shfl(static_cast<int>(q), i, 32) >> 4;
            high1 += __shfl(static_cast<int>(q), 16 + i, 32) >> 4;
        }
        if (lane == 0) {
            quantized[block].ds = make_half2(d, total);
            sum_hi[block].sum_hi[0] = static_cast<int8_t>(high0);
            sum_hi[block].sum_hi[1] = static_cast<int8_t>(high1);
        }
    }
}

__global__ void quantize_q8_1_batched(const float * __restrict__ input,
        block_q8_1 * __restrict__ quantized, block_q8_1_sum_hi * __restrict__ sum_hi,
        int n, int stride) {
    const int batch = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int blocks = n / 32;
    const float * source = input + static_cast<size_t>(batch) * stride;
    for (int block = warp; block < blocks; block += blockDim.x / 32) {
        const int index = block * 32 + lane;
        const float value = source[index];
        float amax = fabsf(value);
        float total = value;
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_down(amax, offset, 32));
            total += __shfl_down(total, offset, 32);
        }
        amax = __shfl(amax, 0, 32);
        total = __shfl(total, 0, 32);
        const float d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / d));
        quantized[static_cast<size_t>(batch) * blocks + block].qs[lane] = q;
        int high0 = 0;
        int high1 = 0;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            high0 += __shfl(static_cast<int>(q), i, 32) >> 4;
            high1 += __shfl(static_cast<int>(q), 16 + i, 32) >> 4;
        }
        if (lane == 0) {
            quantized[static_cast<size_t>(batch) * blocks + block].ds = make_half2(d, total);
            sum_hi[static_cast<size_t>(batch) * blocks + block].sum_hi[0] = static_cast<int8_t>(high0);
            sum_hi[static_cast<size_t>(batch) * blocks + block].sum_hi[1] = static_cast<int8_t>(high1);
        }
    }
}

__global__ void gemv_q4_q8_1_dot8(const block_q4_0 * __restrict__ weights,
        const block_q8_1 * __restrict__ input, const block_q8_1_sum_hi * __restrict__ sum_hi,
        float * __restrict__ output, int n_in, int n_out) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= n_out) return;
    const int blocks = n_in / 32;
    const block_q4_0 * row_weights = weights + static_cast<size_t>(row) * blocks;
    float sum = 0.0f;
    for (int block = lane / 2; block < blocks; block += 16) {
        sum += q4_q8_dot8<2>(row_weights + block, input + block, sum_hi + block,
                2 * (lane & 1));
    }
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down(sum, offset, 32);
    if (lane == 0) output[row] = sum;
}

__global__ void gemv_q4_q8_experts_dot8(const block_q4_0 * __restrict__ weights,
        const block_q8_1 * __restrict__ input, const block_q8_1_sum_hi * __restrict__ sum_hi,
        const int32_t * __restrict__ experts, float * __restrict__ output,
        int n_in, int n_out, int experts_used, int input_stride_blocks) {
    const int row = static_cast<int>(blockIdx.x);
    const int slot = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= n_out || slot >= experts_used) return;
    const int blocks = n_in / 32;
    const size_t expert_stride = static_cast<size_t>(n_out) * blocks;
    const block_q4_0 * row_weights = weights + static_cast<size_t>(experts[slot]) * expert_stride +
            static_cast<size_t>(row) * blocks;
    const block_q8_1 * input_row = input + static_cast<size_t>(slot) * input_stride_blocks;
    const block_q8_1_sum_hi * sum_row = sum_hi + static_cast<size_t>(slot) * input_stride_blocks;
    float sum = 0.0f;
    for (int block = lane / 2; block < blocks; block += 16) {
        sum += q4_q8_dot8<2>(row_weights + block, input_row + block, sum_row + block,
                2 * (lane & 1));
    }
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down(sum, offset, 32);
    if (lane == 0) output[static_cast<size_t>(slot) * n_out + row] = sum;
}
#endif

__global__ void gemv_q4_0(const block_q4_0 * __restrict__ weights,
        const float * __restrict__ input, float * __restrict__ output,
        int n_in, int n_out) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= n_out) return;
    const int blocks = n_in / 32;
    const block_q4_0 * row_weights = weights + static_cast<size_t>(row) * blocks;
    float sum = 0.0f;
    for (int block = lane; block < blocks; block += 32) {
        const block_q4_0 * q4 = row_weights + block;
        const float scale = f16_to_f32(q4->d);
        const float * x = input + block * 32;
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
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 32);
    }
    if (lane == 0) output[row] = sum;
}

__global__ void gemv_f32(const float * __restrict__ weights,
        const float * __restrict__ input, float * __restrict__ output,
        int n_in, int n_out) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= n_out) return;
    const float * row_weights = weights + static_cast<size_t>(row) * n_in;
    float sum = 0.0f;
    for (int i = lane; i < n_in; i += 32) {
        sum = fmaf(row_weights[i], input[i], sum);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 32);
    }
    if (lane == 0) output[row] = sum;
}

__global__ void deq_row_q4_0(const block_q4_0 * __restrict__ weights,
        const int * __restrict__ row_ptr, float * __restrict__ output,
        int n_in, int n_rows) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= n_in) return;
    const int row = *row_ptr;
    if (row < 0 || row >= n_rows) {
        output[index] = 0.0f;
        return;
    }
    const int blocks = n_in / 32;
    output[index] = q4_0_get(weights + static_cast<size_t>(row) * blocks + index / 32,
            index & 31);
}

__global__ void rmsnorm(const float * __restrict__ input, const float * __restrict__ weights,
        float * __restrict__ output, int n) {
    __shared__ float sums[256];
    float sum = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
        sum += input[i] * input[i];
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
        __syncthreads();
    }
    const float scale = rsqrtf(sums[0] / n + RMS_EPS);
    for (int i = static_cast<int>(threadIdx.x); i < n; i += static_cast<int>(blockDim.x)) {
        output[i] = input[i] * scale * weights[i];
    }
}

__global__ void rmsnorm_heads(const float * __restrict__ input, const float * __restrict__ weights,
        float * __restrict__ output, int n_heads) {
    __shared__ float sums[256];
    const int head = static_cast<int>(blockIdx.x);
    if (head >= n_heads) return;
    const float * input_head = input + head * HEAD_DIM;
    float sum = 0.0f;
    for (int i = static_cast<int>(threadIdx.x); i < HEAD_DIM; i += static_cast<int>(blockDim.x)) {
        sum += input_head[i] * input_head[i];
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x + offset];
        __syncthreads();
    }
    const float scale = rsqrtf(sums[0] / HEAD_DIM + RMS_EPS);
    for (int i = static_cast<int>(threadIdx.x); i < HEAD_DIM; i += static_cast<int>(blockDim.x)) {
        output[head * HEAD_DIM + i] = input_head[i] * scale * weights[i];
    }
}

__global__ void rope_neox(float * __restrict__ input, int n_heads, int position) {
    const int head = static_cast<int>(blockIdx.x);
    const int i = static_cast<int>(threadIdx.x);
    if (head >= n_heads || i >= N_ROT / 2) return;
    float * values = input + head * HEAD_DIM;
    const float theta = position * powf(FREQ_BASE, -2.0f * i / N_ROT);
    const float cosine = cosf(theta);
    const float sine = sinf(theta);
    const float x0 = values[i];
    const float x1 = values[i + N_ROT / 2];
    values[i] = x0 * cosine - x1 * sine;
    values[i + N_ROT / 2] = x0 * sine + x1 * cosine;
}

__global__ void concat2(const float * __restrict__ first, const float * __restrict__ second,
        float * __restrict__ output, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) output[i] = first[i];
    else if (i < 2 * n) output[i] = second[i - n];
}

__global__ void add_inplace(float * __restrict__ destination,
        const float * __restrict__ source, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) destination[i] += source[i];
}

__global__ void multiply_sigmoid(float * __restrict__ values,
        const float * __restrict__ gate, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) values[i] *= 1.0f / (1.0f + expf(-gate[i]));
}

__global__ void silu_mul(float * __restrict__ gate, const float * __restrict__ up, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        const float value = gate[i];
        gate[i] = value / (1.0f + expf(-value)) * up[i];
    }
}

__global__ void split_q_gate(const float * __restrict__ full,
        float * __restrict__ query, float * __restrict__ gate) {
    const int head = static_cast<int>(blockIdx.x);
    const int i = static_cast<int>(threadIdx.x);
    if (head >= N_HEAD || i >= HEAD_DIM) return;
    query[head * HEAD_DIM + i] = full[head * 2 * HEAD_DIM + i];
    gate[head * HEAD_DIM + i] = full[head * 2 * HEAD_DIM + HEAD_DIM + i];
}

__global__ void kv_store(const float * __restrict__ key, const float * __restrict__ value,
        __half * __restrict__ key_cache, __half * __restrict__ value_cache, int position) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int width = N_HEAD_KV * HEAD_DIM;
    if (i < width) {
        key_cache[static_cast<size_t>(position) * width + i] = __float2half(key[i]);
        value_cache[static_cast<size_t>(position) * width + i] = __float2half(value[i]);
    }
}

// One block per query head. The cache is chunked so the shared score array is
// bounded even when the model context is long.
__global__ void attn_1q(const float * __restrict__ query,
        const __half * __restrict__ key_cache, const __half * __restrict__ value_cache,
        float * __restrict__ output, int n_pos) {
    const int head = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    const int kv_head = head / (N_HEAD / N_HEAD_KV);
    __shared__ float scores[ATTN_CHUNK];
    __shared__ float reduction[256];
    const float * q = query + head * HEAD_DIM;
    const float scale = rsqrtf(static_cast<float>(HEAD_DIM));
    float accumulator = 0.0f;
    float max_seen = -1e30f;
    float sum_seen = 0.0f;

    for (int base = 0; base < n_pos; base += ATTN_CHUNK) {
        const int count = min(ATTN_CHUNK, n_pos - base);
        for (int p = lane; p < count; p += blockDim.x) {
            const __half * k = key_cache + (static_cast<size_t>(base + p) * N_HEAD_KV + kv_head) * HEAD_DIM;
            float dot = 0.0f;
#pragma unroll 8
            for (int i = 0; i < HEAD_DIM; ++i) dot += q[i] * __half2float(k[i]);
            scores[p] = dot * scale;
        }
        __syncthreads();
        float maximum = -1e30f;
        for (int p = lane; p < count; p += blockDim.x) maximum = fmaxf(maximum, scores[p]);
        reduction[lane] = maximum;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (lane < offset) reduction[lane] = fmaxf(reduction[lane], reduction[lane + offset]);
            __syncthreads();
        }
        const float chunk_max = reduction[0];
        __syncthreads();
        float chunk_sum_local = 0.0f;
        for (int p = lane; p < count; p += blockDim.x) {
            const float weight = expf(scores[p] - chunk_max);
            scores[p] = weight;
            chunk_sum_local += weight;
        }
        reduction[lane] = chunk_sum_local;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (lane < offset) reduction[lane] += reduction[lane + offset];
            __syncthreads();
        }
        const float chunk_sum = reduction[0];
        const float new_max = fmaxf(max_seen, chunk_max);
        const float old_scale = expf(max_seen - new_max);
        const float new_scale = expf(chunk_max - new_max);
        accumulator *= old_scale;
        if (lane < HEAD_DIM) {
            float value_sum = 0.0f;
            for (int p = 0; p < count; ++p) {
                value_sum += scores[p] * __half2float(value_cache[
                        (static_cast<size_t>(base + p) * N_HEAD_KV + kv_head) * HEAD_DIM + lane]);
            }
            accumulator += new_scale * value_sum;
        }
        sum_seen = sum_seen * old_scale + chunk_sum * new_scale;
        max_seen = new_max;
        __syncthreads();
    }
    if (lane < HEAD_DIM) output[head * HEAD_DIM + lane] = accumulator / sum_seen;
}

__global__ void router_top8(const float * __restrict__ logits,
        int32_t * __restrict__ selected, float * __restrict__ weights) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float maximum = -1e30f;
    for (int i = 0; i < N_EXPERT; ++i) maximum = fmaxf(maximum, logits[i]);
    float probabilities[N_EXPERT];
    float total = 0.0f;
    for (int i = 0; i < N_EXPERT; ++i) {
        probabilities[i] = expf(logits[i] - maximum);
        total += probabilities[i];
    }
    total = fmaxf(total, 1e-30f);
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

// Parallel form of router_top8 with the same arithmetic order for the softmax
// and selected-weight normalizations. Only the independent exponentials and
// top-8 searches are distributed across the 256 lanes.
__global__ void router_top8_parallel(const float * __restrict__ logits,
        int32_t * __restrict__ selected, float * __restrict__ weights) {
    __shared__ float probabilities[N_EXPERT];
    __shared__ float reduction_values[N_EXPERT];
    __shared__ int reduction_indices[N_EXPERT];
    __shared__ float maximum;
    __shared__ float total;
    const int tid = static_cast<int>(threadIdx.x);

    reduction_values[tid] = fmaxf(-1e30f, logits[tid]);
    __syncthreads();
    for (int offset = N_EXPERT / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            reduction_values[tid] = fmaxf(reduction_values[tid], reduction_values[tid + offset]);
        }
        __syncthreads();
    }
    if (tid == 0) maximum = reduction_values[0];
    __syncthreads();

    probabilities[tid] = expf(logits[tid] - maximum);
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < N_EXPERT; ++i) sum += probabilities[i];
        total = fmaxf(sum, 1e-30f);
    }
    __syncthreads();
    probabilities[tid] /= total;
    __syncthreads();

    for (int slot = 0; slot < N_EXPERT_USED; ++slot) {
        bool used = false;
        for (int prior = 0; prior < slot; ++prior) used = used || selected[prior] == tid;
        reduction_values[tid] = used ? -1e30f : probabilities[tid];
        reduction_indices[tid] = used ? -1 : tid;
        __syncthreads();
        for (int offset = N_EXPERT / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                const int rhs_index = reduction_indices[tid + offset];
                if (rhs_index >= 0 && (reduction_indices[tid] < 0 ||
                        reduction_values[tid + offset] > reduction_values[tid] ||
                        (reduction_values[tid + offset] == reduction_values[tid] &&
                         rhs_index < reduction_indices[tid]))) {
                    reduction_values[tid] = reduction_values[tid + offset];
                    reduction_indices[tid] = rhs_index;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            selected[slot] = reduction_indices[0];
            weights[slot] = reduction_values[0];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float selected_sum = 0.0f;
        for (int slot = 0; slot < N_EXPERT_USED; ++slot) selected_sum += weights[slot];
        selected_sum = fmaxf(selected_sum, 6.103515625e-5f);
        for (int slot = 0; slot < N_EXPERT_USED; ++slot) weights[slot] /= selected_sum;
    }
}

__global__ void gemv_q4_experts(const block_q4_0 * __restrict__ weights,
        const float * __restrict__ input, const int32_t * __restrict__ experts,
        float * __restrict__ output, int n_in, int n_out, int experts_used) {
    const int row = static_cast<int>(blockIdx.x);
    const int slot = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= n_out || slot >= experts_used) return;
    const int blocks = n_in / 32;
    const size_t expert_stride = static_cast<size_t>(n_out) * blocks;
    const block_q4_0 * row_weights = weights + static_cast<size_t>(experts[slot]) * expert_stride +
            static_cast<size_t>(row) * blocks;
    float sum = 0.0f;
    for (int block = lane; block < blocks; block += 32) {
        const block_q4_0 * q4 = row_weights + block;
        const float scale = f16_to_f32(q4->d);
        const float * x = input + block * 32;
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
    if (lane == 0) output[static_cast<size_t>(slot) * n_out + row] = sum;
}

__global__ void weighted_sum(const float * __restrict__ expert_output,
        const float * __restrict__ weights, float * __restrict__ output,
        int width, int experts_used) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= width) return;
    float sum = expert_output[i] * weights[0];
    for (int slot = 1; slot < experts_used; ++slot) {
        sum += expert_output[static_cast<size_t>(slot) * width + i] * weights[slot];
    }
    output[i] = sum;
}

__global__ void scale_sigmoid(float * __restrict__ values, const float * __restrict__ scalar, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) values[i] *= 1.0f / (1.0f + expf(-scalar[0]));
}

__global__ void argmax_partial(const float * __restrict__ input, int n,
        float * __restrict__ partial_max, int * __restrict__ partial_idx) {
    __shared__ float values[256];
    __shared__ int indices[256];
    const int tid = static_cast<int>(threadIdx.x);
    const int base = static_cast<int>(blockIdx.x) * 256 * 8;
    float maximum = -1e30f;
    int index = -1;
    for (int k = 0; k < 8; ++k) {
        const int i = base + k * 256 + tid;
        if (i < n && (index < 0 || input[i] > maximum)) {
            maximum = input[i];
            index = i;
        }
    }
    values[tid] = maximum;
    indices[tid] = index;
    __syncthreads();
    for (int offset = 128; offset > 0; offset >>= 1) {
        if (tid < offset && indices[tid + offset] >= 0 &&
                (indices[tid] < 0 || values[tid + offset] > values[tid])) {
            values[tid] = values[tid + offset];
            indices[tid] = indices[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_max[blockIdx.x] = values[0];
        partial_idx[blockIdx.x] = indices[0];
    }
}

__global__ void argmax_final(const float * __restrict__ partial_max,
        const int * __restrict__ partial_idx, int parts, int * __restrict__ output) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float maximum = -1e30f;
    int index = -1;
    for (int i = 0; i < parts; ++i) {
        if (partial_idx[i] >= 0 && (index < 0 || partial_max[i] > maximum)) {
            maximum = partial_max[i];
            index = partial_idx[i];
        }
    }
    output[0] = index;
}

__global__ void topk_partial(const float * __restrict__ input, float * __restrict__ partial_values,
        int * __restrict__ partial_indices, int parts) {
    __shared__ float values[256];
    __shared__ int indices[256];
    __shared__ int selected[MTP_DRAFT_TOP_K];
    const int tid = static_cast<int>(threadIdx.x);
    const int base = static_cast<int>(blockIdx.x) * 256 * 8;
    if (tid < MTP_DRAFT_TOP_K) selected[tid] = -1;
    __syncthreads();
    for (int slot = 0; slot < MTP_DRAFT_TOP_K; ++slot) {
        float best = -1e30f;
        int best_index = -1;
        for (int k = 0; k < 8; ++k) {
            const int i = base + k * 256 + tid;
            if (i >= N_HEAD_ROWS) continue;
            bool already = false;
            for (int prior = 0; prior < slot; ++prior) already = already || selected[prior] == i;
            if (!already && (best_index < 0 || input[i] > best ||
                    (input[i] == best && i < best_index))) {
                best = input[i];
                best_index = i;
            }
        }
        values[tid] = best;
        indices[tid] = best_index;
        __syncthreads();
        for (int offset = 128; offset > 0; offset >>= 1) {
            if (tid < offset && indices[tid + offset] >= 0 &&
                    (indices[tid] < 0 || values[tid + offset] > values[tid] ||
                     (values[tid + offset] == values[tid] && indices[tid + offset] < indices[tid]))) {
                values[tid] = values[tid + offset];
                indices[tid] = indices[tid + offset];
            }
            __syncthreads();
        }
        if (tid == 0) {
            selected[slot] = indices[0];
            partial_values[blockIdx.x * MTP_DRAFT_TOP_K + slot] = values[0];
            partial_indices[blockIdx.x * MTP_DRAFT_TOP_K + slot] = indices[0];
        }
        __syncthreads();
    }
    (void) parts;
}

__global__ void topk_final(const float * __restrict__ partial_values,
        const int * __restrict__ partial_indices, int parts,
        const int * __restrict__ remap, int * __restrict__ output_ids,
        float * __restrict__ output_values) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int selected[MTP_DRAFT_TOP_K];
    for (int slot = 0; slot < MTP_DRAFT_TOP_K; ++slot) {
        float best = -1e30f;
        int best_index = -1;
        for (int i = 0; i < parts * MTP_DRAFT_TOP_K; ++i) {
            const int candidate = partial_indices[i];
            const float value = partial_values[i];
            bool already = false;
            for (int prior = 0; prior < slot; ++prior) already = already || selected[prior] == candidate;
            if (!already && candidate >= 0 && value == value &&
                    (best_index < 0 || value > best || (value == best && candidate < best_index))) {
                best = value;
                best_index = candidate;
            }
        }
        selected[slot] = best_index;
        output_values[slot] = best;
        output_ids[slot] = best_index >= 0 ? remap[best_index] : -1;
    }
}

__device__ __forceinline__ bool topk_better(
        float lhs_value, int lhs_index, float rhs_value, int rhs_index) {
    return lhs_index >= 0 && (rhs_index < 0 || lhs_value > rhs_value ||
            (lhs_value == rhs_value && lhs_index < rhs_index));
}

// rocPRIM returns one duplicate-free top-k row, but does not promise result
// order. Sort its 32 entries cooperatively before remapping. This preserves the
// legacy value-descending/index-ascending proposal order without rerunning the
// O(k^3) scalar partition merger (about 0.5 ms per drafted token on gfx1030).
__global__ void sort_remap_topk(const float * __restrict__ values,
        const int * __restrict__ indices, const int * __restrict__ remap,
        int * __restrict__ output_ids, float * __restrict__ output_values) {
    __shared__ float sorted_values[MTP_DRAFT_TOP_K];
    __shared__ int sorted_indices[MTP_DRAFT_TOP_K];
    const int tid = static_cast<int>(threadIdx.x);
    if (blockIdx.x != 0 || tid >= MTP_DRAFT_TOP_K) return;
    sorted_values[tid] = values[tid];
    sorted_indices[tid] = indices[tid];
    __syncthreads();

    for (int width = 2; width <= MTP_DRAFT_TOP_K; width <<= 1) {
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
            const int peer = tid ^ stride;
            if (peer > tid) {
                const bool better_first = (tid & width) == 0;
                const bool peer_better = topk_better(sorted_values[peer], sorted_indices[peer],
                        sorted_values[tid], sorted_indices[tid]);
                const bool self_better = topk_better(sorted_values[tid], sorted_indices[tid],
                        sorted_values[peer], sorted_indices[peer]);
                if ((better_first && peer_better) || (!better_first && self_better)) {
                    const float value = sorted_values[tid];
                    const int index = sorted_indices[tid];
                    sorted_values[tid] = sorted_values[peer];
                    sorted_indices[tid] = sorted_indices[peer];
                    sorted_values[peer] = value;
                    sorted_indices[peer] = index;
                }
            }
            __syncthreads();
        }
    }

    const int index = sorted_indices[tid];
    output_values[tid] = sorted_values[tid];
    output_ids[tid] = index >= 0 ? remap[index] : -1;
}

__global__ void sample_topk(const int * __restrict__ ids, const float * __restrict__ logits,
        float temperature, float p_min, double uniform, int step,
        int * __restrict__ output_id, float * __restrict__ output_probs,
        int * __restrict__ output_count, int * __restrict__ active) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || *active == 0) return;
    float maximum = -1e30f;
    int valid = 0;
    for (int i = 0; i < MTP_DRAFT_TOP_K; ++i) {
        if (ids[i] >= 0 && logits[i] == logits[i]) {
            maximum = fmaxf(maximum, logits[i]);
            ++valid;
        }
    }
    if (valid == 0 || !(temperature > 0.0f) || !(maximum == maximum)) {
        *active = 0;
        return;
    }
    float probabilities[MTP_DRAFT_TOP_K];
    float total = 0.0f;
    for (int i = 0; i < MTP_DRAFT_TOP_K; ++i) {
        probabilities[i] = ids[i] >= 0 && logits[i] == logits[i]
                ? expf((logits[i] - maximum) / temperature) : 0.0f;
        total += probabilities[i];
    }
    if (!(total > 0.0f) || !(total == total)) {
        *active = 0;
        return;
    }
    for (int i = 0; i < MTP_DRAFT_TOP_K; ++i) probabilities[i] /= total;
    const double u = uniform < 1.0 ? (uniform > 0.0 ? uniform : 0.0) : 0.9999999999999999;
    int selected = MTP_DRAFT_TOP_K - 1;
    double cumulative = 0.0;
    for (int i = 0; i < MTP_DRAFT_TOP_K; ++i) {
        cumulative += probabilities[i];
        if (u < cumulative) {
            selected = i;
            break;
        }
    }
    if (ids[selected] < 0 || probabilities[selected] < p_min) {
        *active = 0;
        return;
    }
    output_id[step] = ids[selected];
    for (int i = 0; i < MTP_DRAFT_TOP_K; ++i) output_probs[step * MTP_DRAFT_TOP_K + i] = probabilities[i];
    *output_count = step + 1;
}

} // namespace qwen35moe_sidecar
