#include "moe-mmq-cutlass.cuh"

#ifdef GGML_CUDA_CUTLASS_MOE
#    include <cuda_bf16.h>
#    include <cuda_fp8.h>

#    include <climits>
#    include <type_traits>

static __device__ __forceinline__ uint8_t moe_cutlass_mxfp8_scale(float amax) {
    if (!(amax > 0.0f)) {
        return 0;
    }

    constexpr float e4m3_max = 448.0f;
    const int       exponent = __float2int_ru(log2f(amax / e4m3_max));
    return (uint8_t) max(0, min(254, exponent + 127));
}

static __device__ __forceinline__ float moe_cutlass_swiglu_oai(float x, float g) {
    constexpr float alpha = 1.702f;
    constexpr float limit = 7.0f;

    x = fminf(x, limit);
    g = fmaxf(fminf(g, limit), -limit);

    const float denominator = __fadd_rn(1.0f, __expf(__fmul_rn(-x, alpha)));
    const float swish       = __fdividef(x, denominator);
    return __fmaf_rn(swish, g, swish);
}

static __device__ __forceinline__ float moe_cutlass_half_warp_amax(float value0, float value1) {
    float amax = fmaxf(fabsf(value0), fabsf(value1));
#    pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 16));
    }
    return amax;
}

static __device__ __forceinline__ int64_t moe_cutlass_scale_offset(int row, int k_block, int padded_k_blocks) {
    const int inner_k       = k_block % 4;
    const int inner_m       = (row % 128) / 32;
    const int outer_m       = row % 32;
    const int k_tile        = k_block / 4;
    const int m_tile        = row / 128;
    const int k_tile_stride = 512;
    const int m_tile_stride = (padded_k_blocks / 4) * k_tile_stride;
    return (int64_t) m_tile * m_tile_stride + (int64_t) k_tile * k_tile_stride + outer_m * 16 + inner_m * 4 + inner_k;
}

static __device__ __forceinline__ uint8_t * moe_cutlass_scale_ptr(uint8_t *       scales,
                                                                  const int32_t * expert_bounds,
                                                                  int             expert,
                                                                  int             row,
                                                                  int             k_block,
                                                                  int             padded_k_blocks) {
    const int64_t start = ((int64_t) expert_bounds[expert] + (int64_t) expert * 127) / 128 * 128;
    return scales + start * padded_k_blocks +
           moe_cutlass_scale_offset(row - expert_bounds[expert], k_block, padded_k_blocks);
}

static __global__ void moe_cutlass_quantize_broadcast(const float * __restrict__ src,
                                                      const int32_t * __restrict__ ids,
                                                      const int32_t * __restrict__ ids_src1,
                                                      const int32_t * __restrict__ expert_bounds,
                                                      uint8_t * __restrict__ dst,
                                                      uint8_t * __restrict__ scales,
                                                      int64_t n_cols,
                                                      int64_t n_cols_padded,
                                                      int64_t stride_token,
                                                      int     n_expert_used,
                                                      int64_t ids_stride) {
    const int64_t token   = blockIdx.x;
    const int     k_block = blockIdx.y;
    const int     lane    = threadIdx.x;
    const int64_t k       = (int64_t) k_block * WARP_SIZE + lane;
    const float   value   = k < n_cols ? src[token * stride_token + k] : 0.0f;

    float amax = fabsf(value);
#    pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, WARP_SIZE));
    }

    const uint8_t       scale     = moe_cutlass_mxfp8_scale(amax);
    const float         inv_scale = amax == 0.0f ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(scale));
    const __nv_fp8_e4m3 quantized(value * inv_scale);
    const int           padded_k_blocks = n_cols_padded / WARP_SIZE;

    for (int slot = 0; slot < n_expert_used; ++slot) {
        const int64_t route                    = token * n_expert_used + slot;
        const int     expert                   = ids[token * ids_stride + slot];
        const int     row                      = ids_src1[route];
        dst[(int64_t) row * n_cols_padded + k] = quantized.__x;
        if (lane == 0) {
            *moe_cutlass_scale_ptr(scales, expert_bounds, expert, row, k_block, padded_k_blocks) = scale;
        }
    }
}

template <int n_expert_used>
static __global__ void moe_cutlass_quantize_broadcast_cta(const float * __restrict__ src,
                                                          const int32_t * __restrict__ ids,
                                                          const int32_t * __restrict__ ids_src1,
                                                          const int32_t * __restrict__ expert_bounds,
                                                          uint8_t * __restrict__ dst,
                                                          uint8_t * __restrict__ scales,
                                                          int64_t n_cols,
                                                          int64_t n_cols_padded,
                                                          int64_t stride_token,
                                                          int64_t ids_stride) {
    constexpr int warps = 8;
    __shared__ int route_rows[32];
    __shared__ int route_experts[32];

    const int64_t token = blockIdx.x;
    if (threadIdx.x < (unsigned) n_expert_used) {
        const int slot        = threadIdx.x;
        const int64_t route   = token * n_expert_used + slot;
        route_rows[slot]      = ids_src1[route];
        route_experts[slot]   = ids[token * ids_stride + slot];
    }
    __syncthreads();

    const int warp              = threadIdx.x / WARP_SIZE;
    const int lane              = threadIdx.x % WARP_SIZE;
    const int half              = lane / 16;
    const int pair_lane         = lane % 16;
    const int padded_k_blocks   = n_cols_padded / WARP_SIZE;
    const int paired_k_blocks   = padded_k_blocks / 2;

    for (int pair_block = warp; pair_block < paired_k_blocks; pair_block += warps) {
        const int     k_block = pair_block * 2 + half;
        const int64_t k       = (int64_t) k_block * WARP_SIZE + pair_lane * 2;
        float2        value    = { 0.0f, 0.0f };
        if (k + 1 < n_cols) {
            value = *reinterpret_cast<const float2 *>(src + token * stride_token + k);
        } else {
            if (k < n_cols) {
                value.x = src[token * stride_token + k];
            }
            if (k + 1 < n_cols) {
                value.y = src[token * stride_token + k + 1];
            }
        }

        const float   amax      = moe_cutlass_half_warp_amax(value.x, value.y);
        const uint8_t scale     = moe_cutlass_mxfp8_scale(amax);
        const float   inv_scale = amax == 0.0f ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(scale));
        const __nv_fp8_e4m3 q0(value.x * inv_scale);
        const __nv_fp8_e4m3 q1(value.y * inv_scale);
        const uint16_t       packed = (uint16_t) q0.__x | ((uint16_t) q1.__x << 8);

#    pragma unroll
        for (int slot = 0; slot < n_expert_used; ++slot) {
            const int row = route_rows[slot];
            *reinterpret_cast<uint16_t *>(dst + (int64_t) row * n_cols_padded + k) = packed;
            if (pair_lane == 0) {
                *moe_cutlass_scale_ptr(
                    scales, expert_bounds, route_experts[slot], row, k_block, padded_k_blocks) = scale;
            }
        }
    }
}

static __global__ void moe_cutlass_quantize_routes(const float * __restrict__ src,
                                                   const int32_t * __restrict__ ids,
                                                   const int32_t * __restrict__ ids_src1,
                                                   const int32_t * __restrict__ expert_bounds,
                                                   uint8_t * __restrict__ dst,
                                                   uint8_t * __restrict__ scales,
                                                   int64_t n_cols,
                                                   int64_t n_cols_padded,
                                                   int     n_expert_used,
                                                   int64_t ids_stride) {
    const int64_t token           = blockIdx.x;
    const int     k_block         = blockIdx.y;
    const int     slot            = blockIdx.z;
    const int     lane            = threadIdx.x;
    const int64_t k               = (int64_t) k_block * WARP_SIZE + lane;
    const int     padded_k_blocks = n_cols_padded / WARP_SIZE;

    const int64_t route  = token * n_expert_used + slot;
    const int     expert = ids[token * ids_stride + slot];
    const int     row    = ids_src1[route];
    const float   value  = k < n_cols ? src[route * n_cols + k] : 0.0f;

    float amax = fabsf(value);
#    pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, WARP_SIZE));
    }

    const uint8_t       scale     = moe_cutlass_mxfp8_scale(amax);
    const float         inv_scale = amax == 0.0f ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(scale));
    const __nv_fp8_e4m3 quantized(value * inv_scale);
    dst[(int64_t) row * n_cols_padded + k] = quantized.__x;
    if (lane == 0) {
        *moe_cutlass_scale_ptr(scales, expert_bounds, expert, row, k_block, padded_k_blocks) = scale;
    }
}

static __global__ void moe_cutlass_scatter(const __nv_bfloat16 * __restrict__ src,
                                           const int32_t * __restrict__ ids_dst,
                                           float * __restrict__ dst,
                                           int64_t n_cols,
                                           int64_t n_rows) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_rows * n_cols) {
        return;
    }
    const int64_t row                          = index / n_cols;
    const int64_t col                          = index % n_cols;
    dst[(int64_t) ids_dst[row] * n_cols + col] = __bfloat162float(src[index]);
}

static __global__ void moe_cutlass_w13_epilogue(const __nv_bfloat16 * __restrict__ gate_up,
                                                const float * __restrict__ bias,
                                                const int32_t * __restrict__ ids,
                                                const int32_t * __restrict__ ids_dst,
                                                const int32_t * __restrict__ expert_bounds,
                                                uint8_t * __restrict__ dst,
                                                uint8_t * __restrict__ scales,
                                                int64_t n_ff,
                                                int64_t n_ff_padded,
                                                int     n_expert_used,
                                                int64_t ids_stride) {
    const int64_t row     = blockIdx.x;
    const int     k_block = blockIdx.y;
    const int     lane    = threadIdx.x;
    const int64_t k       = (int64_t) k_block * WARP_SIZE + lane;
    const int64_t route   = ids_dst[row];
    const int64_t token   = route / n_expert_used;
    const int     slot    = route % n_expert_used;
    const int     expert  = ids[token * ids_stride + slot];

    float value = 0.0f;
    if (k < n_ff) {
        const int64_t stride = 2 * n_ff;
        const float gate = __fadd_rn(__bfloat162float(gate_up[row * stride + k]), bias[(int64_t) expert * stride + k]);
        const float up =
            __fadd_rn(__bfloat162float(gate_up[row * stride + n_ff + k]), bias[(int64_t) expert * stride + n_ff + k]);
        value = moe_cutlass_swiglu_oai(gate, up);
    }

    float amax = fabsf(value);
#    pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, WARP_SIZE));
    }

    const uint8_t       scale     = moe_cutlass_mxfp8_scale(amax);
    const float         inv_scale = amax == 0.0f ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(scale));
    const __nv_fp8_e4m3 quantized(value * inv_scale);
    dst[row * n_ff_padded + k] = quantized.__x;
    if (lane == 0) {
        const int padded_k_blocks                                                            = n_ff_padded / WARP_SIZE;
        *moe_cutlass_scale_ptr(scales, expert_bounds, expert, row, k_block, padded_k_blocks) = scale;
    }
}

template <int rows_per_cta>
static __global__ void moe_cutlass_w13_epilogue_cta(const __nv_bfloat16 * __restrict__ gate_up,
                                                    const float * __restrict__ bias,
                                                    const int32_t * __restrict__ ids,
                                                    const int32_t * __restrict__ ids_dst,
                                                    const int32_t * __restrict__ row_expert,
                                                    const int32_t * __restrict__ expert_bounds,
                                                    uint8_t * __restrict__ dst,
                                                    uint8_t * __restrict__ scales,
                                                    int64_t n_ff,
                                                    int64_t n_ff_padded,
                                                    int64_t n_rows,
                                                    int     n_expert_used,
                                                    int64_t ids_stride) {
    constexpr int warps = 8;
    static_assert(warps % rows_per_cta == 0, "rows_per_cta must divide the CTA warp count");
    constexpr int warps_per_row = warps / rows_per_cta;
    __shared__ int expert_shared[rows_per_cta];

    const int64_t first_row = (int64_t) blockIdx.x * rows_per_cta;
    if (threadIdx.x < rows_per_cta && first_row + threadIdx.x < n_rows) {
        const int64_t row = first_row + threadIdx.x;
        if (row_expert != nullptr) {
            expert_shared[threadIdx.x] = row_expert[row];
        } else {
            const int64_t route = ids_dst[row];
            const int64_t token = route / n_expert_used;
            const int     slot  = route - token * n_expert_used;
            expert_shared[threadIdx.x] = ids[token * ids_stride + slot];
        }
    }
    __syncthreads();

    const int     cta_warp         = threadIdx.x / WARP_SIZE;
    const int     row_in_cta       = cta_warp / warps_per_row;
    const int     row_warp         = cta_warp % warps_per_row;
    const int64_t row              = first_row + row_in_cta;
    if (row >= n_rows) {
        return;
    }

    const int     expert           = expert_shared[row_in_cta];
    const int     lane             = threadIdx.x % WARP_SIZE;
    const int     half             = lane / 16;
    const int     pair_lane        = lane % 16;
    const int64_t stride           = 2 * n_ff;
    const int     padded_k_blocks  = n_ff_padded / WARP_SIZE;
    const int     paired_k_blocks  = padded_k_blocks / 2;

    for (int pair_block = row_warp; pair_block < paired_k_blocks; pair_block += warps_per_row) {
        const int     k_block = pair_block * 2 + half;
        const int64_t k       = (int64_t) k_block * WARP_SIZE + pair_lane * 2;
        float2        value    = { 0.0f, 0.0f };
        if (k + 1 < n_ff) {
            const __nv_bfloat162 gate_pair =
                *reinterpret_cast<const __nv_bfloat162 *>(gate_up + row * stride + k);
            const __nv_bfloat162 up_pair =
                *reinterpret_cast<const __nv_bfloat162 *>(gate_up + row * stride + n_ff + k);
            const float2 gate_value = __bfloat1622float2(gate_pair);
            const float2 up_value   = __bfloat1622float2(up_pair);
            const float2 gate_bias  = *reinterpret_cast<const float2 *>(bias + (int64_t) expert * stride + k);
            const float2 up_bias    = *reinterpret_cast<const float2 *>(bias + (int64_t) expert * stride + n_ff + k);
            const float gate0       = __fadd_rn(gate_value.x, gate_bias.x);
            const float gate1       = __fadd_rn(gate_value.y, gate_bias.y);
            const float up0         = __fadd_rn(up_value.x, up_bias.x);
            const float up1         = __fadd_rn(up_value.y, up_bias.y);
            value.x                 = moe_cutlass_swiglu_oai(gate0, up0);
            value.y                 = moe_cutlass_swiglu_oai(gate1, up1);
        } else {
            if (k < n_ff) {
                const float gate =
                    __fadd_rn(__bfloat162float(gate_up[row * stride + k]), bias[(int64_t) expert * stride + k]);
                const float up = __fadd_rn(__bfloat162float(gate_up[row * stride + n_ff + k]),
                                           bias[(int64_t) expert * stride + n_ff + k]);
                value.x = moe_cutlass_swiglu_oai(gate, up);
            }
            if (k + 1 < n_ff) {
                const float gate = __fadd_rn(__bfloat162float(gate_up[row * stride + k + 1]),
                                             bias[(int64_t) expert * stride + k + 1]);
                const float up = __fadd_rn(__bfloat162float(gate_up[row * stride + n_ff + k + 1]),
                                           bias[(int64_t) expert * stride + n_ff + k + 1]);
                value.y = moe_cutlass_swiglu_oai(gate, up);
            }
        }

        const float   amax      = moe_cutlass_half_warp_amax(value.x, value.y);
        const uint8_t scale     = moe_cutlass_mxfp8_scale(amax);
        const float   inv_scale = amax == 0.0f ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(scale));
        const __nv_fp8_e4m3 q0(value.x * inv_scale);
        const __nv_fp8_e4m3 q1(value.y * inv_scale);
        const uint16_t       packed = (uint16_t) q0.__x | ((uint16_t) q1.__x << 8);
        *reinterpret_cast<uint16_t *>(dst + row * n_ff_padded + k) = packed;
        if (pair_lane == 0) {
            *moe_cutlass_scale_ptr(scales, expert_bounds, expert, row, k_block, padded_k_blocks) = scale;
        }
    }
}

static __global__ void moe_cutlass_w2_finalize(const __nv_bfloat16 * __restrict__ down,
                                               const float * __restrict__ bias,
                                               const float * __restrict__ weights,
                                               const int32_t * __restrict__ ids,
                                               const int32_t * __restrict__ ids_src1,
                                               float * __restrict__ dst,
                                               int64_t n_embd,
                                               int64_t n_tokens,
                                               int     n_expert_used,
                                               int64_t ids_stride) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_tokens * n_embd) {
        return;
    }
    const int64_t token = index / n_embd;
    const int64_t col   = index % n_embd;

    float sum = 0.0f;
    for (int slot = 0; slot < n_expert_used; ++slot) {
        const int64_t route  = token * n_expert_used + slot;
        const int     expert = ids[token * ids_stride + slot];
        const int     row    = ids_src1[route];
        const float   value =
            __fadd_rn(__bfloat162float(down[(int64_t) row * n_embd + col]), bias[(int64_t) expert * n_embd + col]);
        sum = __fadd_rn(sum, __fmul_rn(value, weights[route]));
    }
    dst[token * n_embd + col] = sum;
}

size_t ggml_cuda_moe_cutlass_activation_size(int64_t n_rows, int64_t n_cols) {
    return (size_t) n_rows * n_cols;
}

size_t ggml_cuda_moe_cutlass_scale_size(int64_t n_rows, int n_experts, int64_t n_cols) {
    const int64_t padded_rows     = GGML_PAD(n_rows + (int64_t) n_experts * 127, 128);
    const int64_t padded_k_blocks = GGML_PAD((n_cols + WARP_SIZE - 1) / WARP_SIZE, 4);
    return (size_t) padded_rows * padded_k_blocks;
}

void ggml_cuda_moe_cutlass_quantize_broadcast(const float *   src,
                                              const int32_t * ids,
                                              const int32_t * ids_src1,
                                              const int32_t * expert_bounds,
                                              uint8_t *       dst,
                                              uint8_t *       scales,
                                              int64_t         n_cols,
                                              int64_t         n_cols_padded,
                                              int64_t         stride_token,
                                              int64_t         n_tokens,
                                              int             n_experts,
                                              int             n_expert_used,
                                              int64_t         ids_stride,
                                              cudaStream_t    stream) {
    GGML_ASSERT(n_cols_padded >= n_cols && n_cols_padded % 128 == 0);
    const int64_t k_blocks = n_cols_padded / WARP_SIZE;
    CUDA_CHECK(cudaMemsetAsync(
        scales, 0, ggml_cuda_moe_cutlass_scale_size(n_tokens * n_expert_used, n_experts, n_cols_padded), stream));
    moe_cutlass_quantize_broadcast<<<dim3(n_tokens, k_blocks, 1), WARP_SIZE, 0, stream>>>(
        src, ids, ids_src1, expert_bounds, dst, scales, n_cols, n_cols_padded, stride_token, n_expert_used, ids_stride);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_moe_cutlass_quantize_broadcast_cta(const float *   src,
                                                  const int32_t * ids,
                                                  const int32_t * ids_src1,
                                                  const int32_t * expert_bounds,
                                                  uint8_t *       dst,
                                                  uint8_t *       scales,
                                                  int64_t         n_cols,
                                                  int64_t         n_cols_padded,
                                                  int64_t         stride_token,
                                                  int64_t         n_tokens,
                                                  int             n_experts,
                                                  int             n_expert_used,
                                                  int64_t         ids_stride,
                                                  cudaStream_t    stream) {
    if (n_cols <= 0 || n_cols % 2 != 0 || n_cols_padded < n_cols || n_cols_padded % 128 != 0 ||
        stride_token % 2 != 0 || n_tokens <= 0 || n_tokens > UINT_MAX || n_expert_used != 4) {
        return false;
    }

    constexpr int threads = 256;
    CUDA_CHECK(cudaMemsetAsync(
        scales, 0, ggml_cuda_moe_cutlass_scale_size(n_tokens * n_expert_used, n_experts, n_cols_padded), stream));
    moe_cutlass_quantize_broadcast_cta<4><<<(unsigned) n_tokens, threads, 0, stream>>>(
        src, ids, ids_src1, expert_bounds, dst, scales, n_cols, n_cols_padded, stride_token, ids_stride);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

void ggml_cuda_moe_cutlass_quantize_routes(const float *   src,
                                           const int32_t * ids,
                                           const int32_t * ids_src1,
                                           const int32_t * expert_bounds,
                                           uint8_t *       dst,
                                           uint8_t *       scales,
                                           int64_t         n_cols,
                                           int64_t         n_cols_padded,
                                           int64_t         n_tokens,
                                           int             n_experts,
                                           int             n_expert_used,
                                           int64_t         ids_stride,
                                           cudaStream_t    stream) {
    GGML_ASSERT(n_cols_padded >= n_cols && n_cols_padded % 128 == 0);
    const int64_t k_blocks = n_cols_padded / WARP_SIZE;
    CUDA_CHECK(cudaMemsetAsync(
        scales, 0, ggml_cuda_moe_cutlass_scale_size(n_tokens * n_expert_used, n_experts, n_cols_padded), stream));
    moe_cutlass_quantize_routes<<<dim3(n_tokens, k_blocks, n_expert_used), WARP_SIZE, 0, stream>>>(
        src, ids, ids_src1, expert_bounds, dst, scales, n_cols, n_cols_padded, n_expert_used, ids_stride);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_moe_cutlass_scatter(const void *    src,
                                   const int32_t * ids_dst,
                                   float *         dst,
                                   int64_t         n_cols,
                                   int64_t         n_rows,
                                   cudaStream_t    stream) {
    constexpr int threads = 256;
    const int64_t n       = n_rows * n_cols;
    moe_cutlass_scatter<<<(n + threads - 1) / threads, threads, 0, stream>>>((const __nv_bfloat16 *) src, ids_dst, dst,
                                                                             n_cols, n_rows);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_moe_cutlass_w13_epilogue(const void *    gate_up,
                                        const float *   bias,
                                        const int32_t * ids,
                                        const int32_t * ids_dst,
                                        const int32_t * expert_bounds,
                                        uint8_t *       dst,
                                        uint8_t *       scales,
                                        int64_t         n_ff,
                                        int64_t         n_ff_padded,
                                        int64_t         n_rows,
                                        int             n_experts,
                                        int             n_expert_used,
                                        int64_t         ids_stride,
                                        cudaStream_t    stream) {
    GGML_ASSERT(n_ff_padded >= n_ff && n_ff_padded % 128 == 0);
    const int64_t k_blocks = n_ff_padded / WARP_SIZE;
    CUDA_CHECK(cudaMemsetAsync(scales, 0, ggml_cuda_moe_cutlass_scale_size(n_rows, n_experts, n_ff_padded), stream));
    moe_cutlass_w13_epilogue<<<dim3(n_rows, k_blocks, 1), WARP_SIZE, 0, stream>>>(
        (const __nv_bfloat16 *) gate_up, bias, ids, ids_dst, expert_bounds, dst, scales, n_ff, n_ff_padded,
        n_expert_used, ids_stride);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_moe_cutlass_w13_epilogue_cta(const void *    gate_up,
                                            const float *   bias,
                                            const int32_t * ids,
                                            const int32_t * ids_dst,
                                            const int32_t * row_expert,
                                            const int32_t * expert_bounds,
                                            uint8_t *       dst,
                                            uint8_t *       scales,
                                            int64_t         n_ff,
                                            int64_t         n_ff_padded,
                                            int64_t         n_rows,
                                            int             n_experts,
                                            int             n_expert_used,
                                            int             rows_per_cta,
                                            int64_t         ids_stride,
                                            cudaStream_t    stream) {
    if (n_ff <= 0 || n_ff % 2 != 0 || n_ff_padded < n_ff || n_ff_padded % 128 != 0 || n_rows <= 0 ||
        n_rows > UINT_MAX || n_expert_used <= 0 || n_expert_used > 32 ||
        (rows_per_cta != 1 && rows_per_cta != 4 && rows_per_cta != 8)) {
        return false;
    }

    constexpr int threads = 256;
    CUDA_CHECK(cudaMemsetAsync(scales, 0, ggml_cuda_moe_cutlass_scale_size(n_rows, n_experts, n_ff_padded), stream));
    const unsigned blocks = (unsigned) ((n_rows + rows_per_cta - 1) / rows_per_cta);
    if (rows_per_cta == 1) {
        moe_cutlass_w13_epilogue_cta<1><<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16 *) gate_up, bias, ids, ids_dst, row_expert, expert_bounds, dst, scales, n_ff,
            n_ff_padded, n_rows, n_expert_used, ids_stride);
    } else if (rows_per_cta == 4) {
        moe_cutlass_w13_epilogue_cta<4><<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16 *) gate_up, bias, ids, ids_dst, row_expert, expert_bounds, dst, scales, n_ff,
            n_ff_padded, n_rows, n_expert_used, ids_stride);
    } else {
        moe_cutlass_w13_epilogue_cta<8><<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16 *) gate_up, bias, ids, ids_dst, row_expert, expert_bounds, dst, scales, n_ff,
            n_ff_padded, n_rows, n_expert_used, ids_stride);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

void ggml_cuda_moe_cutlass_w2_finalize(const void *    down,
                                       const float *   bias,
                                       const float *   weights,
                                       const int32_t * ids,
                                       const int32_t * ids_src1,
                                       float *         dst,
                                       int64_t         n_embd,
                                       int64_t         n_tokens,
                                       int             n_expert_used,
                                       int64_t         ids_stride,
                                       cudaStream_t    stream) {
    constexpr int threads = 256;
    const int64_t n       = n_tokens * n_embd;
    moe_cutlass_w2_finalize<<<(n + threads - 1) / threads, threads, 0, stream>>>(
        (const __nv_bfloat16 *) down, bias, weights, ids, ids_src1, dst, n_embd, n_tokens, n_expert_used, ids_stride);
    CUDA_CHECK(cudaGetLastError());
}

#    include "cute/tensor.hpp"
#    include "cutlass/cutlass.h"
#    include "cutlass/detail/sm100_blockscaled_layout.hpp"
#    include "cutlass/epilogue/collective/collective_builder.hpp"
#    include "cutlass/gemm/collective/collective_builder.hpp"
#    include "cutlass/gemm/device/gemm_universal_adapter.h"
#    include "cutlass/gemm/dispatch_policy.hpp"
#    include "cutlass/gemm/group_array_problem_shape.hpp"
#    include "cutlass/gemm/kernel/gemm_universal.hpp"
#    include "cutlass/util/packed_stride.hpp"

namespace ggml_moe_cutlass_sm120 {

using namespace cute;

using Activation = cutlass::float_e4m3_t;
using Weight     = cutlass::float_e2m1_t;
using Scale      = cutlass::float_ue8m0_t;
using Output     = cutlass::bfloat16_t;

template <int TileN, bool SwapAB> struct kernel_traits {
    static constexpr bool swap_ab = SwapAB;

    using ElementA         = std::conditional_t<SwapAB, Weight, Activation>;
    using ElementB         = std::conditional_t<SwapAB, Activation, Weight>;
    using ElementSFA       = Scale;
    using ElementSFB       = Scale;
    using ElementAMainloop = cute::tuple<ElementA, ElementSFA>;
    using ElementBMainloop = cute::tuple<ElementB, ElementSFB>;
    using LayoutA          = cutlass::layout::RowMajor;
    using LayoutB          = cutlass::layout::ColumnMajor;
    using LayoutD          = std::conditional_t<SwapAB, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
    using TileShape        = Shape<_128, Int<TileN>, _128>;
    using ClusterShape     = Shape<_1, _1, _1>;

    static constexpr int alignment_a = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int alignment_b = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int alignment_d = 128 / cutlass::sizeof_bits<Output>::value;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassBlockScaledTensorOp,
        TileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        float,
        float,
        void,
        LayoutD *,
        alignment_d,
        Output,
        LayoutD *,
        alignment_d,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
    using StageCount         = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassBlockScaledTensorOp,
        ElementAMainloop,
        LayoutA *,
        alignment_a,
        ElementBMainloop,
        LayoutB *,
        alignment_b,
        float,
        TileShape,
        ClusterShape,
        StageCount,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
    using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA    = typename GemmKernel::InternalStrideA;
    using StrideB    = typename GemmKernel::InternalStrideB;
    using StrideD    = typename GemmKernel::InternalStrideD;
    using LayoutSFA  = typename CollectiveMainloop::InternalLayoutSFA;
    using LayoutSFB  = typename CollectiveMainloop::InternalLayoutSFB;
    using BlockScaleConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;

    static constexpr int scale_granularity = CollectiveMainloop::TiledMma::SFVecSize;
    static_assert(scale_granularity == 32);
};

template <typename Traits> struct grouped_metadata {
    typename Traits::ProblemShape::UnderlyingProblemShape * shapes;
    const typename Traits::ElementA **                      a;
    const typename Traits::ElementB **                      b;
    Output **                                               d;
    typename Traits::StrideA *                              stride_a;
    typename Traits::StrideB *                              stride_b;
    typename Traits::StrideD *                              stride_d;
    const Scale **                                          scale_a;
    const Scale **                                          scale_b;
    typename Traits::LayoutSFA *                            layout_scale_a;
    typename Traits::LayoutSFB *                            layout_scale_b;
};

static size_t align_up(size_t value, size_t alignment = 128) {
    return (value + alignment - 1) & ~(alignment - 1);
}

template <typename T> static T * take(char * base, size_t & offset, int count) {
    offset     = align_up(offset, alignof(T) > 128 ? alignof(T) : 128);
    T * result = base == nullptr ? nullptr : reinterpret_cast<T *>(base + offset);
    offset += sizeof(T) * count;
    return result;
}

template <typename Traits> static grouped_metadata<Traits> make_metadata(char * base, int groups, size_t & size) {
    size = 0;
    grouped_metadata<Traits> result;
    result.shapes         = take<typename Traits::ProblemShape::UnderlyingProblemShape>(base, size, groups);
    result.a              = take<const typename Traits::ElementA *>(base, size, groups);
    result.b              = take<const typename Traits::ElementB *>(base, size, groups);
    result.d              = take<Output *>(base, size, groups);
    result.stride_a       = take<typename Traits::StrideA>(base, size, groups);
    result.stride_b       = take<typename Traits::StrideB>(base, size, groups);
    result.stride_d       = take<typename Traits::StrideD>(base, size, groups);
    result.scale_a        = take<const Scale *>(base, size, groups);
    result.scale_b        = take<const Scale *>(base, size, groups);
    result.layout_scale_a = take<typename Traits::LayoutSFA>(base, size, groups);
    result.layout_scale_b = take<typename Traits::LayoutSFB>(base, size, groups);
    size                  = align_up(size);
    return result;
}

template <typename Traits>
static __global__ void setup_metadata(grouped_metadata<Traits> metadata,
                                      const uint8_t *          activation,
                                      const uint8_t *          activation_scales,
                                      const char *             weights,
                                      const uint8_t *          weight_scales,
                                      const int32_t *          expert_bounds,
                                      void *                   dst,
                                      int                      n,
                                      int                      k,
                                      int                      weight_scale_stride,
                                      int                      groups,
                                      bool                     pdl) {
    using Problem          = typename Traits::ProblemShape::UnderlyingProblemShape;
    using BlockScaleConfig = typename Traits::BlockScaleConfig;

    const int expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= groups) {
        return;
    }

#    if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if (pdl) {
        asm volatile("griddepcontrol.wait;");
        asm volatile("griddepcontrol.launch_dependents;");
    }
#    endif

    const int     row_begin              = expert_bounds[expert];
    const int     m                      = expert_bounds[expert + 1] - row_begin;
    const int     padded_n               = GGML_PAD(n, 128);
    const int     padded_k               = GGML_PAD(k, 128);
    const int     padded_k_blocks        = padded_k / 32;
    const int64_t activation_scale_begin = ((int64_t) row_begin + (int64_t) expert * 127) / 128 * 128 * padded_k_blocks;

    if constexpr (Traits::swap_ab) {
        metadata.shapes[expert] = Problem(n, m, k);
        metadata.a[expert] =
            reinterpret_cast<const typename Traits::ElementA *>(weights + (int64_t) expert * n * k / 2);
        metadata.b[expert] = reinterpret_cast<const typename Traits::ElementB *>(activation + (int64_t) row_begin * k);
        metadata.stride_a[expert] = cutlass::make_cute_packed_stride(typename Traits::StrideA{}, make_shape(n, k, 1));
        metadata.stride_b[expert] = cutlass::make_cute_packed_stride(typename Traits::StrideB{}, make_shape(m, k, 1));
        metadata.stride_d[expert] = cutlass::make_cute_packed_stride(typename Traits::StrideD{}, make_shape(n, m, 1));
        metadata.scale_a[expert] =
            reinterpret_cast<const Scale *>(weight_scales + (int64_t) expert * weight_scale_stride);
        metadata.scale_b[expert]        = reinterpret_cast<const Scale *>(activation_scales + activation_scale_begin);
        const auto shape                = make_shape(padded_n, m, padded_k, 1);
        metadata.layout_scale_a[expert] = BlockScaleConfig::tile_atom_to_shape_SFA(shape);
        metadata.layout_scale_b[expert] = BlockScaleConfig::tile_atom_to_shape_SFB(shape);
    } else {
        metadata.shapes[expert] = Problem(m, n, k);
        metadata.a[expert] = reinterpret_cast<const typename Traits::ElementA *>(activation + (int64_t) row_begin * k);
        metadata.b[expert] =
            reinterpret_cast<const typename Traits::ElementB *>(weights + (int64_t) expert * n * k / 2);
        metadata.stride_a[expert] = cutlass::make_cute_packed_stride(typename Traits::StrideA{}, make_shape(m, k, 1));
        metadata.stride_b[expert] = cutlass::make_cute_packed_stride(typename Traits::StrideB{}, make_shape(n, k, 1));
        metadata.stride_d[expert] = cutlass::make_cute_packed_stride(typename Traits::StrideD{}, make_shape(m, n, 1));
        metadata.scale_a[expert]  = reinterpret_cast<const Scale *>(activation_scales + activation_scale_begin);
        metadata.scale_b[expert] =
            reinterpret_cast<const Scale *>(weight_scales + (int64_t) expert * weight_scale_stride);
        const auto shape                = make_shape(m, padded_n, padded_k, 1);
        metadata.layout_scale_a[expert] = BlockScaleConfig::tile_atom_to_shape_SFA(shape);
        metadata.layout_scale_b[expert] = BlockScaleConfig::tile_atom_to_shape_SFB(shape);
    }
    metadata.d[expert] = reinterpret_cast<Output *>(dst) + (int64_t) row_begin * n;
}

template <int TileN, bool SwapAB>
static bool run_grouped_gemm(ggml_backend_cuda_context &       ctx,
                             const ggml_cuda_moe_weight_view & weight,
                             const uint8_t *                   activation,
                             const uint8_t *                   activation_scales,
                             const int32_t *                   expert_bounds,
                             void *                            dst,
                             int                               n_experts,
                             int                               n,
                             int                               k,
                             int                               sm_count,
                             cudaStream_t                      stream,
                             bool                              pdl,
                             bool                              require) {
    using Traits = kernel_traits<TileN, SwapAB>;
    using Gemm   = typename Traits::Gemm;

    size_t metadata_size = 0;
    make_metadata<Traits>(nullptr, n_experts, metadata_size);
    ggml_cuda_pool_alloc<char> metadata_alloc(ctx.pool());
    char *                     metadata_data = metadata_alloc.alloc(metadata_size);
    grouped_metadata<Traits>   metadata      = make_metadata<Traits>(metadata_data, n_experts, metadata_size);

    constexpr int      threads = 128;
    cudaLaunchConfig_t launch_config{};
    launch_config.gridDim  = (n_experts + threads - 1) / threads;
    launch_config.blockDim = threads;
    launch_config.stream   = stream;
    cudaLaunchAttribute attribute{};
    attribute.id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = true;
    launch_config.attrs                                  = pdl ? &attribute : nullptr;
    launch_config.numAttrs                               = pdl ? 1 : 0;
    auto setup_kernel                                    = setup_metadata<Traits>;
    CUDA_CHECK(cudaLaunchKernelEx(&launch_config, setup_kernel, metadata, activation, activation_scales, weight.data,
                                  weight.scales, expert_bounds, dst, n, k, weight.scale_stride, n_experts, pdl));

    typename Traits::ProblemShape shapes;
    shapes.num_groups          = n_experts;
    shapes.problem_shapes      = metadata.shapes;
    shapes.host_problem_shapes = nullptr;

    cutlass::KernelHardwareInfo hardware_info{};
    hardware_info.device_id = ctx.device;
    hardware_info.sm_count  = sm_count;

    typename Gemm::Arguments arguments = {
        cutlass::gemm::GemmUniversalMode::kGrouped,
        shapes,
        { metadata.a, metadata.stride_a, metadata.b, metadata.stride_b, metadata.scale_a, metadata.layout_scale_a,
          metadata.scale_b, metadata.layout_scale_b },
        { {}, nullptr, nullptr, metadata.d, metadata.stride_d },
        hardware_info,
    };
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta  = 0.0f;

    Gemm                  gemm;
    const cutlass::Status can_implement = gemm.can_implement(arguments);
    if (can_implement != cutlass::Status::kSuccess) {
        if (require) {
            GGML_ABORT("CUTLASS MoE can_implement failed for tile 128x%dx128 swap=%d: %s", TileN, SwapAB,
                       cutlassGetStatusString(can_implement));
        }
        return false;
    }

    const size_t               workspace_size = Gemm::get_workspace_size(arguments);
    ggml_cuda_pool_alloc<char> workspace_alloc(ctx.pool());
    void *                     workspace  = workspace_size == 0 ? nullptr : workspace_alloc.alloc(workspace_size);
    const cutlass::Status      initialize = gemm.initialize(arguments, workspace);
    if (initialize != cutlass::Status::kSuccess) {
        if (require) {
            GGML_ABORT("CUTLASS MoE initialize failed for tile 128x%dx128 swap=%d: %s", TileN, SwapAB,
                       cutlassGetStatusString(initialize));
        }
        return false;
    }

    const cutlass::Status run = gemm.run(stream, nullptr, pdl);
    if (run != cutlass::Status::kSuccess) {
        if (require) {
            GGML_ABORT("CUTLASS MoE run failed for tile 128x%dx128 swap=%d: %s", TileN, SwapAB,
                       cutlassGetStatusString(run));
        }
        return false;
    }
    return true;
}

}  // namespace ggml_moe_cutlass_sm120

bool ggml_cuda_moe_cutlass_compiled() {
    return true;
}

bool ggml_cuda_moe_cutlass_gemm(ggml_backend_cuda_context &       ctx,
                                const ggml_cuda_moe_weight_view & weight,
                                const uint8_t *                   activation,
                                const uint8_t *                   activation_scales,
                                const int32_t *                   expert_bounds,
                                void *                            dst,
                                int                               n_experts,
                                int64_t                           n_rows,
                                int64_t                           n,
                                int64_t                           k,
                                int                               sm_count,
                                ggml_cuda_moe_cutlass_config      config,
                                cudaStream_t                      stream,
                                bool                              require) {
    using namespace ggml_moe_cutlass_sm120;
    GGML_ASSERT(weight.layout == ggml_cuda_moe_weight_layout::cutlass);
    GGML_ASSERT(weight.scales != nullptr);
    GGML_ASSERT(n <= INT_MAX && k <= INT_MAX && n_rows <= INT_MAX);
    GGML_ASSERT(config.tile_n == 32 || config.tile_n == 64 || config.tile_n == 128);

    if (config.tile_n == 32) {
        return config.swap_ab ?
                   run_grouped_gemm<32, true>(ctx, weight, activation, activation_scales, expert_bounds, dst, n_experts,
                                              (int) n, (int) k, sm_count, stream, config.pdl, require) :
                   run_grouped_gemm<32, false>(ctx, weight, activation, activation_scales, expert_bounds, dst,
                                               n_experts, (int) n, (int) k, sm_count, stream, config.pdl, require);
    }
    if (config.tile_n == 64) {
        return config.swap_ab ?
                   run_grouped_gemm<64, true>(ctx, weight, activation, activation_scales, expert_bounds, dst, n_experts,
                                              (int) n, (int) k, sm_count, stream, config.pdl, require) :
                   run_grouped_gemm<64, false>(ctx, weight, activation, activation_scales, expert_bounds, dst,
                                               n_experts, (int) n, (int) k, sm_count, stream, config.pdl, require);
    }
    return config.swap_ab ? run_grouped_gemm<128, true>(ctx, weight, activation, activation_scales, expert_bounds, dst,
                                                        n_experts, (int) n, (int) k, sm_count, stream, config.pdl,
                                                        require) :
                            run_grouped_gemm<128, false>(ctx, weight, activation, activation_scales, expert_bounds, dst,
                                                         n_experts, (int) n, (int) k, sm_count, stream, config.pdl,
                                                         require);
}

#else

bool ggml_cuda_moe_cutlass_compiled() {
    return false;
}

size_t ggml_cuda_moe_cutlass_activation_size(int64_t n_rows, int64_t n_cols) {
    GGML_UNUSED_VARS(n_rows, n_cols);
    return 0;
}

size_t ggml_cuda_moe_cutlass_scale_size(int64_t n_rows, int n_experts, int64_t n_cols) {
    GGML_UNUSED_VARS(n_rows, n_experts, n_cols);
    return 0;
}

void ggml_cuda_moe_cutlass_quantize_broadcast(const float *   src,
                                              const int32_t * ids,
                                              const int32_t * ids_src1,
                                              const int32_t * expert_bounds,
                                              uint8_t *       dst,
                                              uint8_t *       scales,
                                              int64_t         n_cols,
                                              int64_t         n_cols_padded,
                                              int64_t         stride_token,
                                              int64_t         n_tokens,
                                              int             n_experts,
                                              int             n_expert_used,
                                              int64_t         ids_stride,
                                              cudaStream_t    stream) {
    GGML_UNUSED_VARS(src, ids, ids_src1, expert_bounds, dst, scales, n_cols, n_cols_padded, stride_token, n_tokens,
                     n_experts, n_expert_used, ids_stride, stream);
    GGML_ABORT("the CUTLASS MoE backend was not compiled");
}

bool ggml_cuda_moe_cutlass_quantize_broadcast_cta(const float *   src,
                                                  const int32_t * ids,
                                                  const int32_t * ids_src1,
                                                  const int32_t * expert_bounds,
                                                  uint8_t *       dst,
                                                  uint8_t *       scales,
                                                  int64_t         n_cols,
                                                  int64_t         n_cols_padded,
                                                  int64_t         stride_token,
                                                  int64_t         n_tokens,
                                                  int             n_experts,
                                                  int             n_expert_used,
                                                  int64_t         ids_stride,
                                                  cudaStream_t    stream) {
    GGML_UNUSED_VARS(src, ids, ids_src1, expert_bounds, dst, scales, n_cols, n_cols_padded, stride_token, n_tokens,
                     n_experts, n_expert_used, ids_stride, stream);
    return false;
}

void ggml_cuda_moe_cutlass_quantize_routes(const float *   src,
                                           const int32_t * ids,
                                           const int32_t * ids_src1,
                                           const int32_t * expert_bounds,
                                           uint8_t *       dst,
                                           uint8_t *       scales,
                                           int64_t         n_cols,
                                           int64_t         n_cols_padded,
                                           int64_t         n_tokens,
                                           int             n_experts,
                                           int             n_expert_used,
                                           int64_t         ids_stride,
                                           cudaStream_t    stream) {
    GGML_UNUSED_VARS(src, ids, ids_src1, expert_bounds, dst, scales, n_cols, n_cols_padded, n_tokens, n_experts,
                     n_expert_used, ids_stride, stream);
    GGML_ABORT("the CUTLASS MoE backend was not compiled");
}

void ggml_cuda_moe_cutlass_scatter(const void *    src,
                                   const int32_t * ids_dst,
                                   float *         dst,
                                   int64_t         n_cols,
                                   int64_t         n_rows,
                                   cudaStream_t    stream) {
    GGML_UNUSED_VARS(src, ids_dst, dst, n_cols, n_rows, stream);
    GGML_ABORT("the CUTLASS MoE backend was not compiled");
}

void ggml_cuda_moe_cutlass_w13_epilogue(const void *    gate_up,
                                        const float *   bias,
                                        const int32_t * ids,
                                        const int32_t * ids_dst,
                                        const int32_t * expert_bounds,
                                        uint8_t *       dst,
                                        uint8_t *       scales,
                                        int64_t         n_ff,
                                        int64_t         n_ff_padded,
                                        int64_t         n_rows,
                                        int             n_experts,
                                        int             n_expert_used,
                                        int64_t         ids_stride,
                                        cudaStream_t    stream) {
    GGML_UNUSED_VARS(gate_up, bias, ids, ids_dst, expert_bounds, dst, scales, n_ff, n_ff_padded, n_rows, n_experts,
                     n_expert_used, ids_stride, stream);
    GGML_ABORT("the CUTLASS MoE backend was not compiled");
}

bool ggml_cuda_moe_cutlass_w13_epilogue_cta(const void *    gate_up,
                                            const float *   bias,
                                            const int32_t * ids,
                                            const int32_t * ids_dst,
                                            const int32_t * row_expert,
                                            const int32_t * expert_bounds,
                                            uint8_t *       dst,
                                            uint8_t *       scales,
                                            int64_t         n_ff,
                                            int64_t         n_ff_padded,
                                            int64_t         n_rows,
                                            int             n_experts,
                                            int             n_expert_used,
                                            int             rows_per_cta,
                                            int64_t         ids_stride,
                                            cudaStream_t    stream) {
    GGML_UNUSED_VARS(gate_up, bias, ids, ids_dst, row_expert, expert_bounds, dst, scales, n_ff, n_ff_padded, n_rows,
                     n_experts, n_expert_used, rows_per_cta, ids_stride, stream);
    return false;
}

void ggml_cuda_moe_cutlass_w2_finalize(const void *    down,
                                       const float *   bias,
                                       const float *   weights,
                                       const int32_t * ids,
                                       const int32_t * ids_src1,
                                       float *         dst,
                                       int64_t         n_embd,
                                       int64_t         n_tokens,
                                       int             n_expert_used,
                                       int64_t         ids_stride,
                                       cudaStream_t    stream) {
    GGML_UNUSED_VARS(down, bias, weights, ids, ids_src1, dst, n_embd, n_tokens, n_expert_used, ids_stride, stream);
    GGML_ABORT("the CUTLASS MoE backend was not compiled");
}

bool ggml_cuda_moe_cutlass_gemm(ggml_backend_cuda_context &       ctx,
                                const ggml_cuda_moe_weight_view & weight,
                                const uint8_t *                   activation,
                                const uint8_t *                   activation_scales,
                                const int32_t *                   expert_bounds,
                                void *                            dst,
                                int                               n_experts,
                                int64_t                           n_rows,
                                int64_t                           n,
                                int64_t                           k,
                                int                               sm_count,
                                ggml_cuda_moe_cutlass_config      config,
                                cudaStream_t                      stream,
                                bool                              require) {
    GGML_UNUSED_VARS(ctx, weight, activation, activation_scales, expert_bounds, dst, n_experts, n_rows, n, k, sm_count,
                     config, stream, require);
    return false;
}

#endif
