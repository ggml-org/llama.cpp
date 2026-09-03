#pragma once
#include "kernels.hip"
#include <rocprim/block/block_radix_sort.hpp>

namespace qwen4exp_sidecar {

struct block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
};
static_assert(sizeof(block_q6_K) == 210, "Q6_K block size");

// gfx1030's native single-column Q6_K MMVQ uses four Q8_1 blocks packed into
// one 128-value layout. Each lane quantizes four adjacent values, unlike the
// ordinary one-value-per-lane Q8_1 path.
struct block_q8_1_layout_128 {
    __half2 ds[4];
    uint32_t qs[32];
};
static_assert(sizeof(block_q8_1_layout_128) == 4 * sizeof(block_q8_1),
        "Q8_1 x4 layout size");

__global__ void quantize_q8_1_layout_128(const float * __restrict__ input,
        block_q8_1_layout_128 * __restrict__ output, int width) {
    const int layout = static_cast<int>(blockIdx.x);
    const int column = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int q8_block = tid / 8;
    const int word = tid & 7;
    const int base = layout * 128 + q8_block * 32 + word * 4;
    const float * row = input + static_cast<size_t>(column) * width;
    const float x0 = row[base + 0];
    const float x1 = row[base + 1];
    const float x2 = row[base + 2];
    const float x3 = row[base + 3];
    float maximum = fmaxf(fmaxf(fabsf(x0), fabsf(x1)),
            fmaxf(fabsf(x2), fabsf(x3)));
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_xor(maximum, offset, 8));
    }
    const float d = maximum / 127.0f;
    const float d_inverse = maximum == 0.0f ? 0.0f : 1.0f / d;
    float sum = x0 + x1 + x2 + x3;
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset, 8);
    }
    const int8_t q0 = maximum == 0.0f ? 0 : static_cast<int8_t>(roundf(x0 * d_inverse));
    const int8_t q1 = maximum == 0.0f ? 0 : static_cast<int8_t>(roundf(x1 * d_inverse));
    const int8_t q2 = maximum == 0.0f ? 0 : static_cast<int8_t>(roundf(x2 * d_inverse));
    const int8_t q3 = maximum == 0.0f ? 0 : static_cast<int8_t>(roundf(x3 * d_inverse));
    block_q8_1_layout_128 & result =
            output[static_cast<size_t>(column) * (width / 128) + layout];
    result.qs[q8_block * 8 + word] = static_cast<uint32_t>(static_cast<uint8_t>(q0)) |
            (static_cast<uint32_t>(static_cast<uint8_t>(q1)) << 8) |
            (static_cast<uint32_t>(static_cast<uint8_t>(q2)) << 16) |
            (static_cast<uint32_t>(static_cast<uint8_t>(q3)) << 24);
    if (word == 0) result.ds[q8_block] = __floats2half2_rn(d, sum);
}

__device__ __forceinline__ int packed_sdot4_i8(int a, int b, int c) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1030__)
    return __builtin_amdgcn_sdot4(a, b, c, false);
#else
    const uint32_t ua = static_cast<uint32_t>(a);
    const uint32_t ub = static_cast<uint32_t>(b);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int va = static_cast<int8_t>((ua >> (8 * i)) & 255);
        const int vb = static_cast<int8_t>((ub >> (8 * i)) & 255);
        c += va * vb;
    }
    return c;
#endif
}

__device__ __forceinline__ int packed_q6_sub32(uint32_t values) {
    uint32_t result = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int value = static_cast<int>((values >> (8 * i)) & 63) - 32;
        result |= static_cast<uint32_t>(static_cast<uint8_t>(value)) << (8 * i);
    }
    return static_cast<int>(result);
}

__device__ __forceinline__ float q6_k_layout_pair_dot(
        const block_q6_K & block, const block_q8_1_layout_128 * layouts,
        int q8_base, int subblock_pair) {
    const int p = subblock_pair;
    const int subblock0 = (p & 1) + (p >> 1) * 4;
    const int subblocks[2] = {subblock0, subblock0 + 2};
    const uint32_t * ql = reinterpret_cast<const uint32_t *>(block.ql) + p * 8;
    const uint4 ql_low = reinterpret_cast<const uint4 *>(ql)[0];
    const uint4 ql_high = reinterpret_cast<const uint4 *>(ql + 4)[0];
    const uint32_t * qh = reinterpret_cast<const uint32_t *>(block.qh) + (p >> 1) * 8;
    const uint4 qh_low = reinterpret_cast<const uint4 *>(qh)[0];
    const uint4 qh_high = reinterpret_cast<const uint4 *>(qh + 4)[0];
    const int qh_base_shift = (p & 1) * 2;
    const uint32_t ql_low_values[4] = {ql_low.x, ql_low.y, ql_low.z, ql_low.w};
    const uint32_t ql_high_values[4] = {ql_high.x, ql_high.y, ql_high.z, ql_high.w};
    const uint32_t qh_low_values[4] = {qh_low.x, qh_low.y, qh_low.z, qh_low.w};
    const uint32_t qh_high_values[4] = {qh_high.x, qh_high.y, qh_high.z, qh_high.w};
    float sum = 0.0f;
#pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
        const int subblock = subblocks[pair];
        const int q8_index = q8_base + subblock;
        const block_q8_1_layout_128 & layout = layouts[q8_index / 4];
        const int inner = q8_index & 3;
        const int32_t * activation = reinterpret_cast<const int32_t *>(layout.qs + inner * 8);
        const int ql_shift = 4 * pair;
        const int qh_shift = qh_base_shift + 4 * pair;
        int dot_low = 0;
        int dot_high = 0;
#pragma unroll
        for (int word = 0; word < 4; ++word) {
            const int low_nibbles = (ql_low_values[word] >> ql_shift) & 0x0f0f0f0f;
            const int high_bits = ((qh_low_values[word] >> qh_shift) << 4) & 0x30303030;
            const int weights4 = packed_q6_sub32(low_nibbles | high_bits);
            dot_low = packed_sdot4_i8(weights4, activation[word], dot_low);
        }
#pragma unroll
        for (int word = 0; word < 4; ++word) {
            const int low_nibbles = (ql_high_values[word] >> ql_shift) & 0x0f0f0f0f;
            const int high_bits = ((qh_high_values[word] >> qh_shift) << 4) & 0x30303030;
            const int weights4 = packed_q6_sub32(low_nibbles | high_bits);
            dot_high = packed_sdot4_i8(weights4, activation[4 + word], dot_high);
        }
        const float d8 = __half2float(layout.ds[inner].x);
        sum += d8 * (dot_low * static_cast<int>(block.scales[2 * subblock]) +
                dot_high * static_cast<int>(block.scales[2 * subblock + 1]));
    }
    return half_bits_to_float(block.d) * sum;
}

__global__ void gemv_q6_k_q8_1_layout_128(const block_q6_K * __restrict__ weights,
        const block_q8_1_layout_128 * __restrict__ input,
        float * __restrict__ output, int n_in, int n_out) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= n_out || blockDim.x != 32) return;
    const int blocks = n_in / 256;
    const block_q6_K * wr = weights + static_cast<size_t>(row) * blocks;
    float sum = 0.0f;
    for (int block = lane / 4; block < blocks; block += 8) {
        sum += q6_k_layout_pair_dot(wr[block], input, block * 8, lane & 3);
    }
    sum = warp_sum_32(sum);
    if (lane == 0) output[row] = sum;
}

// Native-layout Q6_K x Q8_1 matrix-vector product. Each wave lane owns the
// same eight-value sub-block used by ggml's Q6_K MMVQ implementation.
__global__ void gemv_q6_k_q8_1(const block_q6_K * __restrict__ weights,
        const block_q8_1 * __restrict__ input, float * __restrict__ output,
        int n_in, int n_out) {
    const int row = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if (row >= n_out || blockDim.x != 32) return;
    const int blocks = n_in / 256;
    const block_q6_K * wr = weights + static_cast<size_t>(row) * blocks;
    const int element0 = lane * 8;
    float sum = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        const block_q6_K * q6 = wr + block;
        const float d = half_bits_to_float(q6->d);
        const int half = element0 >> 7;
        const int within_half = element0 & 127;
        const int group = within_half >> 5;
        const int lane0 = within_half & 31;
        const float scale = d * q6->scales[half * 8 + (within_half >> 4)];
        const uint8_t * low = q6->ql + half * 64 + (group & 1) * 32 + lane0;
        const uint8_t * high = q6->qh + half * 32 + lane0;
        const int high_shift = group * 2;
        const bool high_nibble = group >= 2;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int quant = (high_nibble ? (low[e] >> 4) : (low[e] & 15)) |
                    (((high[e] >> high_shift) & 3) << 4);
            const int element = element0 + e;
            const block_q8_1 & q8 = input[block * 8 + element / 32];
            sum += scale * (quant - 32) * static_cast<int>(q8.qs[element & 31]) *
                    __half2float(q8.ds.x);
        }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 32);
    }
    if (lane == 0) output[row] = sum;
}

static constexpr int PROPOSAL_TOP_K = 32;

__device__ __forceinline__ uint64_t proposal_radix_key(float value, int index) {
    if (index < 0 || value != value) return 0;
    if (value == 0.0f) value = 0.0f;
    const uint32_t bits = __float_as_uint(value);
    const uint32_t ordered = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
    return (static_cast<uint64_t>(ordered) << 32) |
            static_cast<uint32_t>(0xffffffffu - static_cast<uint32_t>(index));
}

// First stage: each block sorts 2,048 full-vocabulary logits and emits 32.
__global__ void proposal_topk_vocab_partial(const float * __restrict__ logits,
        int count, float * __restrict__ values, int32_t * __restrict__ indices) {
    using sort_type = rocprim::block_radix_sort<uint64_t, 256, 8, int>;
    __shared__ typename sort_type::storage_type storage;
    uint64_t keys[8];
    int ids[8];
    const int base = static_cast<int>(blockIdx.x) * 2048;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int index = base + static_cast<int>(threadIdx.x) * 8 + k;
        const float value = index < count ? logits[index] : -INFINITY;
        ids[k] = index < count ? index : -1;
        keys[k] = proposal_radix_key(value, ids[k]);
    }
    sort_type().sort_desc(keys, ids, storage);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int rank = static_cast<int>(threadIdx.x) * 8 + k;
        if (rank < PROPOSAL_TOP_K) {
            const int index = ids[k];
            indices[static_cast<size_t>(blockIdx.x) * PROPOSAL_TOP_K + rank] = index;
            values[static_cast<size_t>(blockIdx.x) * PROPOSAL_TOP_K + rank] =
                    index >= 0 ? logits[index] : -INFINITY;
        }
    }
}

// Second stage: reduce up to 2,048 candidate pairs per block.
__global__ void proposal_topk_candidate_partial(const float * __restrict__ input_values,
        const int32_t * __restrict__ input_indices, int count,
        float * __restrict__ output_values, int32_t * __restrict__ output_indices) {
    using sort_type = rocprim::block_radix_sort<uint64_t, 256, 8, int>;
    __shared__ typename sort_type::storage_type storage;
    uint64_t keys[8];
    int slots[8];
    const int base = static_cast<int>(blockIdx.x) * 2048;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int slot = base + static_cast<int>(threadIdx.x) * 8 + k;
        const int index = slot < count ? input_indices[slot] : -1;
        const float value = slot < count ? input_values[slot] : -INFINITY;
        slots[k] = slot < count ? slot : -1;
        keys[k] = proposal_radix_key(value, index);
    }
    sort_type().sort_desc(keys, slots, storage);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int rank = static_cast<int>(threadIdx.x) * 8 + k;
        if (rank < PROPOSAL_TOP_K) {
            const int slot = slots[k];
            const int index = slot >= 0 ? input_indices[slot] : -1;
            output_indices[static_cast<size_t>(blockIdx.x) * PROPOSAL_TOP_K + rank] = index;
            output_values[static_cast<size_t>(blockIdx.x) * PROPOSAL_TOP_K + rank] =
                    slot >= 0 ? input_values[slot] : -INFINITY;
        }
    }
}

// Final stage receives at most 64 candidates for the 248,320-row head.
__global__ void proposal_topk_final(const float * __restrict__ input_values,
        const int32_t * __restrict__ input_indices, int count,
        float * __restrict__ output_values, int32_t * __restrict__ output_indices) {
    using sort_type = rocprim::block_radix_sort<uint64_t, 256, 1, int>;
    __shared__ typename sort_type::storage_type storage;
    const int slot = static_cast<int>(threadIdx.x);
    const int index = slot < count ? input_indices[slot] : -1;
    const float value = slot < count ? input_values[slot] : -INFINITY;
    uint64_t keys[1] = {proposal_radix_key(value, index)};
    int sources[1] = {slot < count ? slot : -1};
    sort_type().sort_desc(keys, sources, storage);
    if (slot < PROPOSAL_TOP_K) {
        const int selected_slot = sources[0];
        output_indices[slot] = selected_slot >= 0 ? input_indices[selected_slot] : -1;
        output_values[slot] = selected_slot >= 0 ? input_values[selected_slot] : -INFINITY;
    }
}

__global__ void proposal_normalize_top32(const float * __restrict__ logits,
        float temperature, float * __restrict__ probabilities) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float maximum = -INFINITY;
    for (int i = 0; i < PROPOSAL_TOP_K; ++i) maximum = fmaxf(maximum, logits[i]);
    float sum = 0.0f;
    for (int i = 0; i < PROPOSAL_TOP_K; ++i) {
        probabilities[i] = expf((logits[i] - maximum) / temperature);
        sum += probabilities[i];
    }
    for (int i = 0; i < PROPOSAL_TOP_K; ++i) probabilities[i] /= sum;
}

// Sample an already-normalized top-32 row with the same sticky p_min stop
// contract as the generic MTP sidecar. A rejected step still receives a safe
// top-1 token so later fixed-count device work cannot index an invalid row;
// output_count keeps that suffix invisible to the caller.
__global__ void proposal_sample_top32(const int32_t * __restrict__ ids,
        const float * __restrict__ probabilities, float p_min, double uniform,
        int step, int32_t * __restrict__ output_tokens,
        int32_t * __restrict__ output_ids, float * __restrict__ output_probs,
        int32_t * __restrict__ output_count, int32_t * __restrict__ active) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int32_t fallback = ids[0];
    output_tokens[step] = fallback;
    if (*active == 0) return;

    double cumulative = 0.0;
    const double bounded = uniform < 1.0 ? (uniform > 0.0 ? uniform : 0.0) :
            0.9999999999999999;
    int selected = PROPOSAL_TOP_K - 1;
    for (int i = 0; i < PROPOSAL_TOP_K; ++i) {
        cumulative += probabilities[i];
        if (bounded < cumulative) {
            selected = i;
            break;
        }
    }
    if (ids[selected] < 0 || probabilities[selected] < p_min) {
        *active = 0;
        return;
    }
    output_tokens[step] = ids[selected];
    for (int i = 0; i < PROPOSAL_TOP_K; ++i) {
        output_ids[static_cast<size_t>(step) * PROPOSAL_TOP_K + i] = ids[i];
        output_probs[static_cast<size_t>(step) * PROPOSAL_TOP_K + i] = probabilities[i];
    }
    *output_count = step + 1;
}

} // namespace qwen4exp_sidecar
