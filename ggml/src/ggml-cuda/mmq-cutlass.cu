#include "mmid.cuh"
#include "mmq-cutlass.cuh"
#include "moe-mmq.cuh"
#include "unary.cuh"

#include <cstdlib>

#ifdef GGML_CUDA_CUTLASS
#    include <cuda_bf16.h>
#    include <cuda_fp8.h>

#    include <algorithm>
#    include <array>
#    include <climits>
#    include <type_traits>

static int moe_cutlass_nvfp4_tile_n(const char * name, int64_t n_rows, int n_experts) {
    const char * value = std::getenv(name);
    if (value != nullptr && value[0] != '\0') {
        const int result = std::atoi(value);
        if (result == 32 || result == 64 || result == 128) {
            return result;
        }
        GGML_ABORT("%s must be 32, 64, or 128", name);
    }
    const int64_t rows_per_expert = (n_rows + n_experts - 1) / n_experts;
    return rows_per_expert <= 32 ? 32 : rows_per_expert <= 64 ? 64 : 128;
}

static bool moe_cutlass_pdl_requested() {
    const char * value = std::getenv("GGML_CUDA_MOE_MMQ_CUTLASS_PDL");
    return value != nullptr && std::atoi(value) != 0;
}

static bool moe_cutlass_swap_requested(const char * name) {
    const char * value = std::getenv(name);
    return value == nullptr || std::atoi(value) != 0;
}

static __device__ __forceinline__ uint8_t cutlass_mxfp8_scale(float amax) {
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

static __device__ __forceinline__ float cutlass_half_warp_amax(float value0, float value1) {
    float amax = fmaxf(fabsf(value0), fabsf(value1));
#    pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 16));
    }
    return amax;
}

static __device__ __forceinline__ int64_t cutlass_scale_offset(int row, int k_block, int padded_k_blocks) {
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
                                                                  int             padded_k_blocks,
                                                                  bool            route_groups) {
    if (route_groups) {
        return scales + (int64_t) row * 128 * padded_k_blocks + cutlass_scale_offset(0, k_block, padded_k_blocks);
    }
    const int64_t start = ((int64_t) expert_bounds[expert] + (int64_t) expert * 127) / 128 * 128;
    return scales + start * padded_k_blocks +
           cutlass_scale_offset(row - expert_bounds[expert], k_block, padded_k_blocks);
}

static __global__ void cutlass_quantize_mxfp8(const float * __restrict__ src,
                                              uint8_t * __restrict__ dst,
                                              uint8_t * __restrict__ scales,
                                              int64_t n_cols,
                                              int64_t n_cols_padded,
                                              int64_t stride_row) {
    constexpr int warps             = 8;
    const int     row               = blockIdx.x;
    const int     warp              = threadIdx.x / WARP_SIZE;
    const int     lane              = threadIdx.x % WARP_SIZE;
    const int     half              = lane / 16;
    const int     pair_lane         = lane % 16;
    const int     scale_blocks      = n_cols_padded / WARP_SIZE;
    const int     scale_block_pairs = scale_blocks / 2;

    for (int pair = warp; pair < scale_block_pairs; pair += warps) {
        const int     scale_block = 2 * pair + half;
        const int64_t k           = (int64_t) scale_block * WARP_SIZE + pair_lane * 2;
        float2        value       = { 0.0f, 0.0f };
        if (k + 1 < n_cols) {
            value = *reinterpret_cast<const float2 *>(src + (int64_t) row * stride_row + k);
        } else {
            if (k < n_cols) {
                value.x = src[(int64_t) row * stride_row + k];
            }
            if (k + 1 < n_cols) {
                value.y = src[(int64_t) row * stride_row + k + 1];
            }
        }

        const float         amax      = cutlass_half_warp_amax(value.x, value.y);
        const uint8_t       scale     = cutlass_mxfp8_scale(amax);
        const float         inv_scale = amax == 0.0f ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(scale));
        const __nv_fp8_e4m3 q0(value.x * inv_scale);
        const __nv_fp8_e4m3 q1(value.y * inv_scale);
        *reinterpret_cast<uint16_t *>(dst + (int64_t) row * n_cols_padded + k) =
            (uint16_t) q0.__x | ((uint16_t) q1.__x << 8);
        if (pair_lane == 0) {
            scales[cutlass_scale_offset(row, scale_block, scale_blocks)] = scale;
        }
    }
}

static __global__ void cutlass_quantize_nvfp4(const float * __restrict__ src,
                                              uint8_t * __restrict__ dst,
                                              uint8_t * __restrict__ scales,
                                              int64_t n_cols,
                                              int64_t n_cols_padded,
                                              int64_t stride_row) {
    constexpr int  warps             = 8;
    const int      row               = blockIdx.x;
    const int      warp              = threadIdx.x / WARP_SIZE;
    const int      lane              = threadIdx.x % WARP_SIZE;
    const int      half              = lane / QK_NVFP4_SUB;
    const int      half_lane         = lane % QK_NVFP4_SUB;
    const unsigned mask              = half == 0 ? 0x0000FFFFu : 0xFFFF0000u;
    const int      scale_blocks      = n_cols_padded / QK_NVFP4_SUB;
    const int      scale_block_pairs = (scale_blocks + 1) / 2;

    for (int pair = warp; pair < scale_block_pairs; pair += warps) {
        const int scale_block = 2 * pair + half;
        if (scale_block >= scale_blocks) {
            continue;
        }

        const int64_t k     = (int64_t) scale_block * QK_NVFP4_SUB + half_lane;
        const float   value = k < n_cols ? src[(int64_t) row * stride_row + k] : 0.0f;
        float         amax  = fabsf(value);
#    pragma unroll
        for (int offset = QK_NVFP4_SUB / 2; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(mask, amax, offset, QK_NVFP4_SUB));
        }

        const uint8_t scale_code = ggml_cuda_fp32_to_ue4m3(amax / 6.0f);
        const float   scale      = ggml_cuda_ue4m3_to_fp32(scale_code);
        const float   inv_scale  = scale > 0.0f ? 0.5f / scale : 0.0f;
        const uint8_t quantized  = ggml_cuda_float_to_fp4_e2m1(value, inv_scale);
        const uint8_t next       = (uint8_t) __shfl_down_sync(mask, (unsigned) quantized, 1, QK_NVFP4_SUB);

        if ((half_lane & 1) == 0) {
            dst[(int64_t) row * (n_cols_padded / 2) + (int64_t) scale_block * (QK_NVFP4_SUB / 2) + half_lane / 2] =
                quantized | (next << 4);
        }
        if (half_lane == 0) {
            scales[cutlass_scale_offset(row, scale_block, scale_blocks)] = scale_code;
        }
    }
}

static __global__ void cutlass_nvfp4_swiglu_quantize(const __nv_bfloat16 * __restrict__ gate_up,
                                                     const float * __restrict__ gate_scale,
                                                     const float * __restrict__ up_scale,
                                                     uint8_t * __restrict__ dst,
                                                     uint8_t * __restrict__ scales,
                                                     int64_t n_ff,
                                                     int64_t n_ff_padded) {
    constexpr int  warps             = 8;
    const int      row               = blockIdx.x;
    const int      warp              = threadIdx.x / WARP_SIZE;
    const int      lane              = threadIdx.x % WARP_SIZE;
    const int      half              = lane / QK_NVFP4_SUB;
    const int      half_lane         = lane % QK_NVFP4_SUB;
    const unsigned mask              = half == 0 ? 0x0000FFFFu : 0xFFFF0000u;
    const int      scale_blocks      = n_ff_padded / QK_NVFP4_SUB;
    const int      scale_block_pairs = (scale_blocks + 1) / 2;
    const float    gate_multiplier   = gate_scale == nullptr ? 1.0f : gate_scale[0];
    const float    up_multiplier     = up_scale == nullptr ? 1.0f : up_scale[0];

    for (int pair = warp; pair < scale_block_pairs; pair += warps) {
        const int scale_block = 2 * pair + half;
        if (scale_block >= scale_blocks) {
            continue;
        }

        const int64_t k     = (int64_t) scale_block * QK_NVFP4_SUB + half_lane;
        float         value = 0.0f;
        if (k < n_ff) {
            const float gate = __fmul_rn(__bfloat162float(gate_up[(int64_t) row * 2 * n_ff + k]), gate_multiplier);
            const float up   = __fmul_rn(__bfloat162float(gate_up[(int64_t) row * 2 * n_ff + n_ff + k]), up_multiplier);
            value            = __fmul_rn(up, ggml_cuda_op_silu_single(gate));
        }

        float amax = fabsf(value);
#    pragma unroll
        for (int offset = QK_NVFP4_SUB / 2; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(mask, amax, offset, QK_NVFP4_SUB));
        }

        const uint8_t scale_code = ggml_cuda_fp32_to_ue4m3(amax / 6.0f);
        const float   scale      = ggml_cuda_ue4m3_to_fp32(scale_code);
        const float   inv_scale  = scale > 0.0f ? 0.5f / scale : 0.0f;
        const uint8_t quantized  = ggml_cuda_float_to_fp4_e2m1(value, inv_scale);
        const uint8_t next       = (uint8_t) __shfl_down_sync(mask, (unsigned) quantized, 1, QK_NVFP4_SUB);
        if ((half_lane & 1) == 0) {
            dst[(int64_t) row * (n_ff_padded / 2) + (int64_t) scale_block * (QK_NVFP4_SUB / 2) + half_lane / 2] =
                quantized | (next << 4);
        }
        if (half_lane == 0) {
            scales[cutlass_scale_offset(row, scale_block, scale_blocks)] = scale_code;
        }
    }
}

static __global__ void cutlass_mxfp8_swiglu_quantize(const __nv_bfloat16 * __restrict__ gate_up,
                                                     const float * __restrict__ gate_scale,
                                                     const float * __restrict__ up_scale,
                                                     uint8_t * __restrict__ dst,
                                                     uint8_t * __restrict__ scales,
                                                     int64_t n_ff,
                                                     int64_t n_ff_padded) {
    constexpr int warps             = 8;
    const int     row               = blockIdx.x;
    const int     warp              = threadIdx.x / WARP_SIZE;
    const int     lane              = threadIdx.x % WARP_SIZE;
    const int     half              = lane / 16;
    const int     pair_lane         = lane % 16;
    const int     scale_blocks      = n_ff_padded / WARP_SIZE;
    const int     scale_block_pairs = scale_blocks / 2;
    const float   gate_multiplier   = gate_scale == nullptr ? 1.0f : gate_scale[0];
    const float   up_multiplier     = up_scale == nullptr ? 1.0f : up_scale[0];

    for (int pair = warp; pair < scale_block_pairs; pair += warps) {
        const int     scale_block = 2 * pair + half;
        const int64_t k           = (int64_t) scale_block * WARP_SIZE + pair_lane * 2;
        float2        value       = { 0.0f, 0.0f };
        if (k < n_ff) {
            const float gate = __fmul_rn(__bfloat162float(gate_up[(int64_t) row * 2 * n_ff + k]), gate_multiplier);
            const float up   = __fmul_rn(__bfloat162float(gate_up[(int64_t) row * 2 * n_ff + n_ff + k]), up_multiplier);
            value.x          = __fmul_rn(up, ggml_cuda_op_silu_single(gate));
        }
        if (k + 1 < n_ff) {
            const float gate = __fmul_rn(__bfloat162float(gate_up[(int64_t) row * 2 * n_ff + k + 1]), gate_multiplier);
            const float up =
                __fmul_rn(__bfloat162float(gate_up[(int64_t) row * 2 * n_ff + n_ff + k + 1]), up_multiplier);
            value.y = __fmul_rn(up, ggml_cuda_op_silu_single(gate));
        }

        const float         amax      = cutlass_half_warp_amax(value.x, value.y);
        const uint8_t       scale     = cutlass_mxfp8_scale(amax);
        const float         inv_scale = amax == 0.0f ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(scale));
        const __nv_fp8_e4m3 q0(value.x * inv_scale);
        const __nv_fp8_e4m3 q1(value.y * inv_scale);
        *reinterpret_cast<uint16_t *>(dst + (int64_t) row * n_ff_padded + k) =
            (uint16_t) q0.__x | ((uint16_t) q1.__x << 8);
        if (pair_lane == 0) {
            scales[cutlass_scale_offset(row, scale_block, scale_blocks)] = scale;
        }
    }
}

static __global__ void cutlass_finalize_bf16(const __nv_bfloat16 * __restrict__ src,
                                             const float * __restrict__ scale,
                                             float * __restrict__ dst,
                                             int64_t n) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) {
        return;
    }
    const float multiplier = scale == nullptr ? 1.0f : scale[0];
    dst[index]             = __fmul_rn(__bfloat162float(src[index]), multiplier);
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
                                                      int64_t ids_stride,
                                                      bool    route_groups) {
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

    const uint8_t       scale     = cutlass_mxfp8_scale(amax);
    const float         inv_scale = amax == 0.0f ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(scale));
    const __nv_fp8_e4m3 quantized(value * inv_scale);
    const int           padded_k_blocks = n_cols_padded / WARP_SIZE;

    for (int slot = 0; slot < n_expert_used; ++slot) {
        const int64_t route                    = token * n_expert_used + slot;
        const int     expert                   = ids[token * ids_stride + slot];
        const int     row                      = ids_src1[route];
        dst[(int64_t) row * n_cols_padded + k] = quantized.__x;
        if (lane == 0) {
            *moe_cutlass_scale_ptr(scales, expert_bounds, expert, row, k_block, padded_k_blocks, route_groups) = scale;
        }
    }
}

template <int n_expert_used>
static __global__ void moe_cutlass_quantize_broadcast_cta(const float * __restrict__ src,
                                                          const int32_t * __restrict__ ids,
                                                          int32_t * __restrict__ ids_src1,
                                                          int32_t * __restrict__ ids_dst,
                                                          int32_t * __restrict__ row_expert,
                                                          const int32_t * __restrict__ expert_bounds,
                                                          uint8_t * __restrict__ dst,
                                                          uint8_t * __restrict__ scales,
                                                          int64_t n_cols,
                                                          int64_t n_cols_padded,
                                                          int64_t stride_token,
                                                          int64_t ids_stride,
                                                          bool    route_groups) {
    constexpr int  warps = 8;
    __shared__ int route_rows[32];
    __shared__ int route_experts[32];

    const int64_t token = blockIdx.x;
    if (threadIdx.x < (unsigned) n_expert_used) {
        const int     slot   = threadIdx.x;
        const int64_t route  = token * n_expert_used + slot;
        const int     expert = ids[token * ids_stride + slot];
        const int     row    = route_groups ? (int) route : ids_src1[route];
        route_rows[slot]     = row;
        route_experts[slot]  = expert;
        if (route_groups) {
            ids_src1[route] = row;
            ids_dst[row]    = (int32_t) route;
            row_expert[row] = expert;
        }
    }
    __syncthreads();

    const int warp            = threadIdx.x / WARP_SIZE;
    const int lane            = threadIdx.x % WARP_SIZE;
    const int half            = lane / 16;
    const int pair_lane       = lane % 16;
    const int padded_k_blocks = n_cols_padded / WARP_SIZE;
    const int paired_k_blocks = padded_k_blocks / 2;

    for (int pair_block = warp; pair_block < paired_k_blocks; pair_block += warps) {
        const int     k_block = pair_block * 2 + half;
        const int64_t k       = (int64_t) k_block * WARP_SIZE + pair_lane * 2;
        float2        value   = { 0.0f, 0.0f };
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

        const float         amax      = cutlass_half_warp_amax(value.x, value.y);
        const uint8_t       scale     = cutlass_mxfp8_scale(amax);
        const float         inv_scale = amax == 0.0f ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(scale));
        const __nv_fp8_e4m3 q0(value.x * inv_scale);
        const __nv_fp8_e4m3 q1(value.y * inv_scale);
        const uint16_t      packed = (uint16_t) q0.__x | ((uint16_t) q1.__x << 8);

#    pragma unroll
        for (int slot = 0; slot < n_expert_used; ++slot) {
            const int row                                                          = route_rows[slot];
            *reinterpret_cast<uint16_t *>(dst + (int64_t) row * n_cols_padded + k) = packed;
            if (pair_lane == 0) {
                *moe_cutlass_scale_ptr(scales, expert_bounds, route_experts[slot], row, k_block, padded_k_blocks,
                                       route_groups) = scale;
            }
        }
    }
}

template <int n_expert_used>
static __global__ void moe_cutlass_quantize_nvfp4_broadcast_cta(const float * __restrict__ src,
                                                                const int32_t * __restrict__ ids,
                                                                const int32_t * __restrict__ ids_src1,
                                                                const int32_t * __restrict__ expert_bounds,
                                                                uint8_t * __restrict__ dst,
                                                                uint8_t * __restrict__ scales,
                                                                int64_t n_cols,
                                                                int64_t n_cols_padded,
                                                                int64_t stride_token) {
    constexpr int  warps = 8;
    __shared__ int route_rows[n_expert_used];
    __shared__ int route_experts[n_expert_used];

    const int token = blockIdx.x;
    if (threadIdx.x < n_expert_used) {
        const int slot      = threadIdx.x;
        const int route     = token * n_expert_used + slot;
        route_rows[slot]    = ids_src1[route];
        route_experts[slot] = ids[route];
    }
    __syncthreads();

    const int      warp              = threadIdx.x / WARP_SIZE;
    const int      lane              = threadIdx.x % WARP_SIZE;
    const int      half              = lane / QK_NVFP4_SUB;
    const int      half_lane         = lane % QK_NVFP4_SUB;
    const unsigned mask              = half == 0 ? 0x0000FFFFu : 0xFFFF0000u;
    const int      scale_blocks      = n_cols_padded / QK_NVFP4_SUB;
    const int      scale_block_pairs = (scale_blocks + 1) / 2;

    for (int pair = warp; pair < scale_block_pairs; pair += warps) {
        const int scale_block = 2 * pair + half;
        if (scale_block >= scale_blocks) {
            continue;
        }

        const int64_t k     = (int64_t) scale_block * QK_NVFP4_SUB + half_lane;
        const float   value = k < n_cols ? src[(int64_t) token * stride_token + k] : 0.0f;
        float         amax  = fabsf(value);
#    pragma unroll
        for (int offset = QK_NVFP4_SUB / 2; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(mask, amax, offset, QK_NVFP4_SUB));
        }

        const uint8_t scale_code = ggml_cuda_fp32_to_ue4m3(amax / 6.0f);
        const float   scale      = ggml_cuda_ue4m3_to_fp32(scale_code);
        const float   inv_scale  = scale > 0.0f ? 0.5f / scale : 0.0f;
        const uint8_t quantized  = ggml_cuda_float_to_fp4_e2m1(value, inv_scale);
        const uint8_t next       = (uint8_t) __shfl_down_sync(mask, (unsigned) quantized, 1, QK_NVFP4_SUB);

        if ((half_lane & 1) == 0) {
            const uint8_t packed = quantized | (next << 4);
#    pragma unroll
            for (int slot = 0; slot < n_expert_used; ++slot) {
                dst[(int64_t) route_rows[slot] * (n_cols_padded / 2) + (int64_t) scale_block * (QK_NVFP4_SUB / 2) +
                    half_lane / 2] = packed;
            }
        }
        if (half_lane == 0) {
#    pragma unroll
            for (int slot = 0; slot < n_expert_used; ++slot) {
                *moe_cutlass_scale_ptr(scales, expert_bounds, route_experts[slot], route_rows[slot], scale_block,
                                       scale_blocks, false) = scale_code;
            }
        }
    }
}

static bool moe_cutlass_quantize_nvfp4_broadcast(const float *   src,
                                                 const int32_t * ids,
                                                 const int32_t * ids_src1,
                                                 const int32_t * expert_bounds,
                                                 uint8_t *       dst,
                                                 uint8_t *       scales,
                                                 int64_t         n_cols,
                                                 int64_t         n_cols_padded,
                                                 int64_t         stride_token,
                                                 int64_t         n_tokens,
                                                 int             n_expert_used,
                                                 cudaStream_t    stream) {
    if (n_tokens <= 0 || n_tokens > UINT_MAX) {
        return false;
    }
    if (n_expert_used == 4) {
        moe_cutlass_quantize_nvfp4_broadcast_cta<4><<<(unsigned) n_tokens, 256, 0, stream>>>(
            src, ids, ids_src1, expert_bounds, dst, scales, n_cols, n_cols_padded, stride_token);
    } else if (n_expert_used == 8) {
        moe_cutlass_quantize_nvfp4_broadcast_cta<8><<<(unsigned) n_tokens, 256, 0, stream>>>(
            src, ids, ids_src1, expert_bounds, dst, scales, n_cols, n_cols_padded, stride_token);
    } else {
        return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

static __global__ void moe_cutlass_stage_routes(const int32_t * __restrict__ ids,
                                                const float * __restrict__ weights,
                                                int32_t * __restrict__ staged_ids,
                                                float * __restrict__ staged_weights,
                                                int n_routes,
                                                int n_expert_used,
                                                int ids_stride,
                                                int weights_route_stride,
                                                int weights_token_stride) {
    const int route = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= n_routes) {
        return;
    }
    const int token       = route / n_expert_used;
    const int slot        = route - token * n_expert_used;
    staged_ids[route]     = ids[(int64_t) token * ids_stride + slot];
    staged_weights[route] = weights[(int64_t) token * weights_token_stride + (int64_t) slot * weights_route_stride];
}

static __global__ void moe_cutlass_nvfp4_w13_epilogue(const __nv_bfloat16 * __restrict__ gate_up,
                                                      const int32_t * __restrict__ row_expert,
                                                      const int32_t * __restrict__ expert_bounds,
                                                      const float * __restrict__ gate_scale,
                                                      const float * __restrict__ up_scale,
                                                      uint8_t * __restrict__ dst,
                                                      uint8_t * __restrict__ scales,
                                                      int n_ff,
                                                      int n_ff_padded,
                                                      int n_rows) {
    constexpr int warps = 8;
    const int     row   = blockIdx.x;
    if (row >= n_rows) {
        return;
    }
    const int      expert            = row_expert[row];
    const int      warp              = threadIdx.x / WARP_SIZE;
    const int      lane              = threadIdx.x % WARP_SIZE;
    const int      half              = lane / QK_NVFP4_SUB;
    const int      half_lane         = lane % QK_NVFP4_SUB;
    const unsigned mask              = half == 0 ? 0x0000FFFFu : 0xFFFF0000u;
    const int      scale_blocks      = n_ff_padded / QK_NVFP4_SUB;
    const int      scale_block_pairs = (scale_blocks + 1) / 2;

    for (int pair = warp; pair < scale_block_pairs; pair += warps) {
        const int scale_block = 2 * pair + half;
        if (scale_block >= scale_blocks) {
            continue;
        }

        const int k     = scale_block * QK_NVFP4_SUB + half_lane;
        float     value = 0.0f;
        if (k < n_ff) {
            const float gate = __fmul_rn(__bfloat162float(gate_up[(int64_t) row * 2 * n_ff + k]), gate_scale[expert]);
            const float up =
                __fmul_rn(__bfloat162float(gate_up[(int64_t) row * 2 * n_ff + n_ff + k]), up_scale[expert]);
            value = __fmul_rn(up, ggml_cuda_op_silu_single(gate));
        }

        float amax = fabsf(value);
#    pragma unroll
        for (int offset = QK_NVFP4_SUB / 2; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(mask, amax, offset, QK_NVFP4_SUB));
        }

        const uint8_t scale_code = ggml_cuda_fp32_to_ue4m3(amax / 6.0f);
        const float   scale      = ggml_cuda_ue4m3_to_fp32(scale_code);
        const float   inv_scale  = scale > 0.0f ? 0.5f / scale : 0.0f;
        const uint8_t quantized  = ggml_cuda_float_to_fp4_e2m1(value, inv_scale);
        const uint8_t next       = (uint8_t) __shfl_down_sync(mask, (unsigned) quantized, 1, QK_NVFP4_SUB);
        if ((half_lane & 1) == 0) {
            dst[(int64_t) row * (n_ff_padded / 2) + (int64_t) scale_block * (QK_NVFP4_SUB / 2) + half_lane / 2] =
                quantized | (next << 4);
        }
        if (half_lane == 0) {
            *moe_cutlass_scale_ptr(scales, expert_bounds, expert, row, scale_block, scale_blocks, false) = scale_code;
        }
    }
}

static __global__ void moe_cutlass_nvfp4_w2_finalize(const __nv_bfloat16 * __restrict__ down,
                                                     const int32_t * __restrict__ ids,
                                                     const int32_t * __restrict__ ids_src1,
                                                     const float * __restrict__ down_scale,
                                                     const float * __restrict__ weights,
                                                     float * __restrict__ dst,
                                                     int n_embd,
                                                     int n_tokens,
                                                     int n_expert_used) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (int64_t) n_tokens * n_embd) {
        return;
    }

    const int token  = index / n_embd;
    const int col    = index - (int64_t) token * n_embd;
    float     result = 0.0f;
    for (int slot = 0; slot < n_expert_used; ++slot) {
        const int route  = token * n_expert_used + slot;
        const int expert = ids[route];
        const int row    = ids_src1[route];
        float     value  = __bfloat162float(down[(int64_t) row * n_embd + col]);
        value            = __fmul_rn(value, down_scale[expert]);
        value            = __fmul_rn(value, weights[route]);
        result           = __fadd_rn(result, value);
    }
    dst[index] = result;
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
                                                int64_t ids_stride,
                                                bool    route_groups) {
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

    const uint8_t       scale     = cutlass_mxfp8_scale(amax);
    const float         inv_scale = amax == 0.0f ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(scale));
    const __nv_fp8_e4m3 quantized(value * inv_scale);
    dst[row * n_ff_padded + k] = quantized.__x;
    if (lane == 0) {
        const int padded_k_blocks = n_ff_padded / WARP_SIZE;
        *moe_cutlass_scale_ptr(scales, expert_bounds, expert, row, k_block, padded_k_blocks, route_groups) = scale;
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
                                                    int64_t ids_stride,
                                                    bool    route_groups) {
    constexpr int warps = 8;
    static_assert(warps % rows_per_cta == 0, "rows_per_cta must divide the CTA warp count");
    constexpr int  warps_per_row = warps / rows_per_cta;
    __shared__ int expert_shared[rows_per_cta];

    const int64_t first_row = (int64_t) blockIdx.x * rows_per_cta;
    if (threadIdx.x < rows_per_cta && first_row + threadIdx.x < n_rows) {
        const int64_t row = first_row + threadIdx.x;
        if (row_expert != nullptr) {
            expert_shared[threadIdx.x] = row_expert[row];
        } else {
            const int64_t route        = ids_dst[row];
            const int64_t token        = route / n_expert_used;
            const int     slot         = route - token * n_expert_used;
            expert_shared[threadIdx.x] = ids[token * ids_stride + slot];
        }
    }
    __syncthreads();

    const int     cta_warp   = threadIdx.x / WARP_SIZE;
    const int     row_in_cta = cta_warp / warps_per_row;
    const int     row_warp   = cta_warp % warps_per_row;
    const int64_t row        = first_row + row_in_cta;
    if (row >= n_rows) {
        return;
    }

    const int     expert          = expert_shared[row_in_cta];
    const int     lane            = threadIdx.x % WARP_SIZE;
    const int     half            = lane / 16;
    const int     pair_lane       = lane % 16;
    const int64_t stride          = 2 * n_ff;
    const int     padded_k_blocks = n_ff_padded / WARP_SIZE;
    const int     paired_k_blocks = padded_k_blocks / 2;

    for (int pair_block = row_warp; pair_block < paired_k_blocks; pair_block += warps_per_row) {
        const int     k_block = pair_block * 2 + half;
        const int64_t k       = (int64_t) k_block * WARP_SIZE + pair_lane * 2;
        float2        value   = { 0.0f, 0.0f };
        if (k + 1 < n_ff) {
            const __nv_bfloat162 gate_pair = *reinterpret_cast<const __nv_bfloat162 *>(gate_up + row * stride + k);
            const __nv_bfloat162 up_pair = *reinterpret_cast<const __nv_bfloat162 *>(gate_up + row * stride + n_ff + k);
            const float2         gate_value = __bfloat1622float2(gate_pair);
            const float2         up_value   = __bfloat1622float2(up_pair);
            const float2         gate_bias  = *reinterpret_cast<const float2 *>(bias + (int64_t) expert * stride + k);
            const float2 up_bias = *reinterpret_cast<const float2 *>(bias + (int64_t) expert * stride + n_ff + k);
            const float  gate0   = __fadd_rn(gate_value.x, gate_bias.x);
            const float  gate1   = __fadd_rn(gate_value.y, gate_bias.y);
            const float  up0     = __fadd_rn(up_value.x, up_bias.x);
            const float  up1     = __fadd_rn(up_value.y, up_bias.y);
            value.x              = moe_cutlass_swiglu_oai(gate0, up0);
            value.y              = moe_cutlass_swiglu_oai(gate1, up1);
        } else {
            if (k < n_ff) {
                const float gate =
                    __fadd_rn(__bfloat162float(gate_up[row * stride + k]), bias[(int64_t) expert * stride + k]);
                const float up = __fadd_rn(__bfloat162float(gate_up[row * stride + n_ff + k]),
                                           bias[(int64_t) expert * stride + n_ff + k]);
                value.x        = moe_cutlass_swiglu_oai(gate, up);
            }
            if (k + 1 < n_ff) {
                const float gate =
                    __fadd_rn(__bfloat162float(gate_up[row * stride + k + 1]), bias[(int64_t) expert * stride + k + 1]);
                const float up = __fadd_rn(__bfloat162float(gate_up[row * stride + n_ff + k + 1]),
                                           bias[(int64_t) expert * stride + n_ff + k + 1]);
                value.y        = moe_cutlass_swiglu_oai(gate, up);
            }
        }

        const float         amax      = cutlass_half_warp_amax(value.x, value.y);
        const uint8_t       scale     = cutlass_mxfp8_scale(amax);
        const float         inv_scale = amax == 0.0f ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(scale));
        const __nv_fp8_e4m3 q0(value.x * inv_scale);
        const __nv_fp8_e4m3 q1(value.y * inv_scale);
        const uint16_t      packed                                 = (uint16_t) q0.__x | ((uint16_t) q1.__x << 8);
        *reinterpret_cast<uint16_t *>(dst + row * n_ff_padded + k) = packed;
        if (pair_lane == 0) {
            *moe_cutlass_scale_ptr(scales, expert_bounds, expert, row, k_block, padded_k_blocks, route_groups) = scale;
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

size_t ggml_cuda_cutlass_activation_size(ggml_type type, int64_t n_rows, int64_t n_cols) {
    GGML_ASSERT(type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4);
    GGML_ASSERT(n_rows > 0 && n_cols > 0 && n_cols % 128 == 0);
    return type == GGML_TYPE_NVFP4 ? (size_t) n_rows * n_cols / 2 : (size_t) n_rows * n_cols;
}

size_t ggml_cuda_cutlass_scale_size(ggml_type type, int64_t n_rows, int64_t n_cols) {
    GGML_ASSERT(type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4);
    GGML_ASSERT(n_rows > 0 && n_cols > 0 && n_cols % 128 == 0);
    const int scale_vector_size = type == GGML_TYPE_NVFP4 ? QK_NVFP4_SUB : QK_MXFP4;
    return (size_t) GGML_PAD(n_rows, 128) * (n_cols / scale_vector_size);
}

bool ggml_cuda_cutlass_quantize(const float * src,
                                uint8_t *     dst,
                                uint8_t *     scales,
                                ggml_type     type,
                                int64_t       n_cols,
                                int64_t       n_cols_padded,
                                int64_t       stride_row,
                                int64_t       n_rows,
                                cudaStream_t  stream) {
    if ((type != GGML_TYPE_MXFP4 && type != GGML_TYPE_NVFP4) || n_cols <= 0 || n_cols % 2 != 0 ||
        n_cols_padded < n_cols || n_cols_padded % 128 != 0 || stride_row < n_cols || stride_row % 2 != 0 ||
        n_rows <= 0 || n_rows > UINT_MAX) {
        return false;
    }

    constexpr int threads = 256;
    CUDA_CHECK(cudaMemsetAsync(scales, 0, ggml_cuda_cutlass_scale_size(type, n_rows, n_cols_padded), stream));
    if (type == GGML_TYPE_NVFP4) {
        cutlass_quantize_nvfp4<<<(unsigned) n_rows, threads, 0, stream>>>(src, dst, scales, n_cols, n_cols_padded,
                                                                          stride_row);
    } else {
        cutlass_quantize_mxfp8<<<(unsigned) n_rows, threads, 0, stream>>>(src, dst, scales, n_cols, n_cols_padded,
                                                                          stride_row);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

size_t ggml_cuda_moe_cutlass_scale_size(int64_t n_rows, int n_experts, int64_t n_cols, bool route_groups) {
    const int64_t padded_k_blocks = GGML_PAD((n_cols + WARP_SIZE - 1) / WARP_SIZE, 4);
    if (route_groups) {
        return (size_t) n_rows * 128 * padded_k_blocks;
    }
    const int64_t padded_rows = GGML_PAD(n_rows + (int64_t) n_experts * 127, 128);
    return (size_t) padded_rows * padded_k_blocks;
}

static size_t moe_cutlass_nvfp4_scale_size(int64_t n_rows, int n_experts, int64_t n_cols) {
    GGML_ASSERT(n_rows > 0 && n_experts > 0 && n_cols > 0 && n_cols % 128 == 0);
    const int64_t scale_blocks = n_cols / QK_NVFP4_SUB;
    const int64_t padded_rows  = GGML_PAD(n_rows + (int64_t) n_experts * 127, 128);
    return (size_t) padded_rows * scale_blocks;
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
                                              bool            route_groups,
                                              cudaStream_t    stream) {
    GGML_ASSERT(n_cols_padded >= n_cols && n_cols_padded % 128 == 0);
    GGML_ASSERT(route_groups || expert_bounds != nullptr);
    const int64_t k_blocks = n_cols_padded / WARP_SIZE;
    CUDA_CHECK(cudaMemsetAsync(
        scales, 0, ggml_cuda_moe_cutlass_scale_size(n_tokens * n_expert_used, n_experts, n_cols_padded, route_groups),
        stream));
    moe_cutlass_quantize_broadcast<<<dim3(n_tokens, k_blocks, 1), WARP_SIZE, 0, stream>>>(
        src, ids, ids_src1, expert_bounds, dst, scales, n_cols, n_cols_padded, stride_token, n_expert_used, ids_stride,
        route_groups);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_moe_cutlass_quantize_broadcast_cta(const float *   src,
                                                  const int32_t * ids,
                                                  int32_t *       ids_src1,
                                                  int32_t *       ids_dst,
                                                  int32_t *       row_expert,
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
                                                  bool            route_groups,
                                                  cudaStream_t    stream) {
    GGML_ASSERT(route_groups || expert_bounds != nullptr);
    GGML_ASSERT(!route_groups || (ids_dst != nullptr && row_expert != nullptr));
    if (n_cols <= 0 || n_cols % 2 != 0 || n_cols_padded < n_cols || n_cols_padded % 128 != 0 || stride_token % 2 != 0 ||
        n_tokens <= 0 || n_tokens > UINT_MAX || n_expert_used != 4) {
        return false;
    }

    constexpr int threads = 256;
    CUDA_CHECK(cudaMemsetAsync(
        scales, 0, ggml_cuda_moe_cutlass_scale_size(n_tokens * n_expert_used, n_experts, n_cols_padded, route_groups),
        stream));
    moe_cutlass_quantize_broadcast_cta<4><<<(unsigned) n_tokens, threads, 0, stream>>>(
        src, ids, ids_src1, ids_dst, row_expert, expert_bounds, dst, scales, n_cols, n_cols_padded, stride_token,
        ids_stride, route_groups);
    CUDA_CHECK(cudaGetLastError());
    return true;
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
                                        bool            route_groups,
                                        cudaStream_t    stream) {
    GGML_ASSERT(n_ff_padded >= n_ff && n_ff_padded % 128 == 0);
    GGML_ASSERT(route_groups || expert_bounds != nullptr);
    const int64_t k_blocks = n_ff_padded / WARP_SIZE;
    CUDA_CHECK(cudaMemsetAsync(scales, 0,
                               ggml_cuda_moe_cutlass_scale_size(n_rows, n_experts, n_ff_padded, route_groups), stream));
    moe_cutlass_w13_epilogue<<<dim3(n_rows, k_blocks, 1), WARP_SIZE, 0, stream>>>(
        (const __nv_bfloat16 *) gate_up, bias, ids, ids_dst, expert_bounds, dst, scales, n_ff, n_ff_padded,
        n_expert_used, ids_stride, route_groups);
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
                                            bool            route_groups,
                                            cudaStream_t    stream) {
    GGML_ASSERT(route_groups || expert_bounds != nullptr);
    if (n_ff <= 0 || n_ff % 2 != 0 || n_ff_padded < n_ff || n_ff_padded % 128 != 0 || n_rows <= 0 ||
        n_rows > UINT_MAX || n_expert_used <= 0 || n_expert_used > 32 ||
        (rows_per_cta != 1 && rows_per_cta != 4 && rows_per_cta != 8)) {
        return false;
    }

    constexpr int threads = 256;
    CUDA_CHECK(cudaMemsetAsync(scales, 0,
                               ggml_cuda_moe_cutlass_scale_size(n_rows, n_experts, n_ff_padded, route_groups), stream));
    const unsigned blocks = (unsigned) ((n_rows + rows_per_cta - 1) / rows_per_cta);
    if (rows_per_cta == 1) {
        moe_cutlass_w13_epilogue_cta<1><<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16 *) gate_up, bias, ids, ids_dst, row_expert, expert_bounds, dst, scales, n_ff,
            n_ff_padded, n_rows, n_expert_used, ids_stride, route_groups);
    } else if (rows_per_cta == 4) {
        moe_cutlass_w13_epilogue_cta<4><<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16 *) gate_up, bias, ids, ids_dst, row_expert, expert_bounds, dst, scales, n_ff,
            n_ff_padded, n_rows, n_expert_used, ids_stride, route_groups);
    } else {
        moe_cutlass_w13_epilogue_cta<8><<<blocks, threads, 0, stream>>>(
            (const __nv_bfloat16 *) gate_up, bias, ids, ids_dst, row_expert, expert_bounds, dst, scales, n_ff,
            n_ff_padded, n_rows, n_expert_used, ids_stride, route_groups);
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

namespace ggml_cutlass_sm120 {

using namespace cute;

using Activation    = cutlass::float_e4m3_t;
using Weight        = cutlass::float_e2m1_t;
using DefaultOutput = cutlass::bfloat16_t;

static_assert(sizeof(DefaultOutput) == sizeof(__nv_bfloat16));

template <int TileN, bool SwapAB> struct mxfp_kernel_traits {
    static constexpr bool swap_ab         = SwapAB;
    static constexpr int  tile_n          = TileN;
    static constexpr int  activation_bits = 8;

    using Scale            = cutlass::float_ue8m0_t;
    using Output           = DefaultOutput;
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

template <int TileN, bool SwapAB, typename OutputType = DefaultOutput> struct nvfp4_kernel_traits {
    static constexpr bool swap_ab         = SwapAB;
    static constexpr int  tile_n          = TileN;
    static constexpr int  activation_bits = 4;

    using Scale            = cutlass::float_ue4m3_t;
    using Output           = OutputType;
    using ElementInput     = cutlass::nv_float4_t<Weight>;
    using ElementA         = ElementInput;
    using ElementB         = ElementInput;
    using ElementAMainloop = ElementA;
    using ElementBMainloop = ElementB;
    using LayoutA          = cutlass::layout::RowMajor;
    using LayoutB          = cutlass::layout::ColumnMajor;
    using LayoutD          = std::conditional_t<SwapAB, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
    using TileShape        = Shape<_128, Int<TileN>, _128>;
    using ClusterShape     = Shape<_1, _1, _1>;

    static constexpr int alignment_a = 32;
    static constexpr int alignment_b = 32;
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
    static_assert(scale_granularity == QK_NVFP4_SUB);
};

template <int TileN, typename OutputType> struct dense_mxfp_kernel_traits {
    static constexpr int tile_n          = TileN;
    static constexpr int activation_bits = 8;

    using Scale            = cutlass::float_ue8m0_t;
    using Output           = OutputType;
    using ElementA         = cutlass::float_e4m3_t;
    using ElementB         = cutlass::float_e2m1_t;
    using ElementAMainloop = cute::tuple<ElementA, Scale>;
    using ElementBMainloop = cute::tuple<ElementB, Scale>;
    using LayoutA          = cutlass::layout::RowMajor;
    using LayoutB          = cutlass::layout::ColumnMajor;
    using LayoutD          = cutlass::layout::RowMajor;
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
        LayoutD,
        alignment_d,
        Output,
        LayoutD,
        alignment_d,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
    using StageCount         = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassBlockScaledTensorOp,
        ElementAMainloop,
        LayoutA,
        alignment_a,
        ElementBMainloop,
        LayoutB,
        alignment_b,
        float,
        TileShape,
        ClusterShape,
        StageCount,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
    using ProblemShape = Shape<int, int, int, int>;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA    = typename GemmKernel::StrideA;
    using StrideB    = typename GemmKernel::StrideB;
    using StrideC    = typename GemmKernel::StrideC;
    using StrideD    = typename GemmKernel::StrideD;
    using LayoutSFA  = typename CollectiveMainloop::LayoutSFA;
    using LayoutSFB  = typename CollectiveMainloop::LayoutSFB;
    using BlockScaleConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;

    static constexpr int scale_granularity = CollectiveMainloop::TiledMma::SFVecSize;
    static_assert(scale_granularity == QK_MXFP4);
};

template <int TileN, typename OutputType> struct dense_nvfp4_kernel_traits {
    static constexpr int tile_n          = TileN;
    static constexpr int activation_bits = 4;

    using Scale        = cutlass::float_ue4m3_t;
    using Output       = OutputType;
    using ElementInput = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using ElementA     = ElementInput;
    using ElementB     = ElementInput;
    using LayoutA      = cutlass::layout::RowMajor;
    using LayoutB      = cutlass::layout::ColumnMajor;
    using LayoutD      = cutlass::layout::RowMajor;
    using TileShape    = Shape<_128, Int<TileN>, _128>;
    using ClusterShape = Shape<_1, _1, _1>;

    static constexpr int alignment_a = 32;
    static constexpr int alignment_b = 32;
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
        LayoutD,
        alignment_d,
        Output,
        LayoutD,
        alignment_d,
        cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
    using StageCount         = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm120,
        cutlass::arch::OpClassBlockScaledTensorOp,
        ElementA,
        LayoutA,
        alignment_a,
        ElementB,
        LayoutB,
        alignment_b,
        float,
        TileShape,
        ClusterShape,
        StageCount,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
    using ProblemShape = Shape<int, int, int, int>;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA    = typename GemmKernel::StrideA;
    using StrideB    = typename GemmKernel::StrideB;
    using StrideC    = typename GemmKernel::StrideC;
    using StrideD    = typename GemmKernel::StrideD;
    using LayoutSFA  = typename CollectiveMainloop::LayoutSFA;
    using LayoutSFB  = typename CollectiveMainloop::LayoutSFB;
    using BlockScaleConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;

    static constexpr int scale_granularity = CollectiveMainloop::TiledMma::SFVecSize;
    static_assert(scale_granularity == QK_NVFP4_SUB);
};

template <typename Traits>
static bool run_dense_gemm(ggml_backend_cuda_context &      ctx,
                           const ggml_cuda_cutlass_weight & weight,
                           const uint8_t *                  activation,
                           const uint8_t *                  activation_scales,
                           void *                           dst,
                           int                              m,
                           int                              n,
                           int                              k,
                           cudaStream_t                     stream,
                           bool                             require) {
    using Gemm             = typename Traits::Gemm;
    using BlockScaleConfig = typename Traits::BlockScaleConfig;

    const auto stride_a   = cutlass::make_cute_packed_stride(typename Traits::StrideA{}, make_shape(m, k, 1));
    const auto stride_b   = cutlass::make_cute_packed_stride(typename Traits::StrideB{}, make_shape(n, k, 1));
    const auto stride_c   = cutlass::make_cute_packed_stride(typename Traits::StrideC{}, make_shape(m, n, 1));
    const auto stride_d   = cutlass::make_cute_packed_stride(typename Traits::StrideD{}, make_shape(m, n, 1));
    const auto shape      = make_shape(m, n, k, 1);
    const auto layout_sfa = BlockScaleConfig::tile_atom_to_shape_SFA(shape);
    const auto layout_sfb = BlockScaleConfig::tile_atom_to_shape_SFB(shape);

    typename Gemm::Arguments arguments = {
        cutlass::gemm::GemmUniversalMode::kGemm,
        shape,
        {
          reinterpret_cast<const typename Gemm::ElementA *>(activation),
          stride_a, reinterpret_cast<const typename Gemm::ElementB *>(weight.data),
          stride_b, reinterpret_cast<const typename Traits::Scale *>(activation_scales),
          layout_sfa, reinterpret_cast<const typename Traits::Scale *>(weight.scales),
          layout_sfb, },
        {
          {},
          nullptr, stride_c,
          reinterpret_cast<typename Traits::Output *>(dst),
          stride_d, },
    };
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta  = 0.0f;

    Gemm                  gemm;
    const cutlass::Status can_implement = gemm.can_implement(arguments);
    if (can_implement != cutlass::Status::kSuccess) {
        if (require) {
            GGML_ABORT("CUTLASS dense can_implement failed for tile 128x%dx128: %s", Traits::tile_n,
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
            GGML_ABORT("CUTLASS dense initialize failed for tile 128x%dx128: %s", Traits::tile_n,
                       cutlassGetStatusString(initialize));
        }
        return false;
    }

    const cutlass::Status run = gemm.run(stream);
    if (run != cutlass::Status::kSuccess) {
        if (require) {
            GGML_ABORT("CUTLASS dense run failed for tile 128x%dx128: %s", Traits::tile_n, cutlassGetStatusString(run));
        }
        return false;
    }
    return true;
}

template <typename Output>
static bool dispatch_dense_gemm(ggml_backend_cuda_context &      ctx,
                                const ggml_cuda_cutlass_weight & weight,
                                const uint8_t *                  activation,
                                const uint8_t *                  activation_scales,
                                void *                           dst,
                                int                              m,
                                int                              n,
                                int                              k,
                                cudaStream_t                     stream,
                                bool                             require) {
    if (weight.type == GGML_TYPE_MXFP4) {
        return run_dense_gemm<dense_mxfp_kernel_traits<128, Output>>(ctx, weight, activation, activation_scales, dst, m,
                                                                     n, k, stream, require);
    }
    if (weight.type == GGML_TYPE_NVFP4) {
        return run_dense_gemm<dense_nvfp4_kernel_traits<128, Output>>(ctx, weight, activation, activation_scales, dst,
                                                                      m, n, k, stream, require);
    }
    if (require) {
        GGML_ABORT("CUTLASS dense does not support weight type %s", ggml_type_name(weight.type));
    }
    return false;
}

template <typename Traits> struct grouped_metadata {
    typename Traits::ProblemShape::UnderlyingProblemShape * shapes;
    const typename Traits::Gemm::ElementA **                a;
    const typename Traits::Gemm::ElementB **                b;
    typename Traits::Output **                              d;
    typename Traits::StrideA *                              stride_a;
    typename Traits::StrideB *                              stride_b;
    typename Traits::StrideD *                              stride_d;
    const typename Traits::Scale **                         scale_a;
    const typename Traits::Scale **                         scale_b;
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
    result.a              = take<const typename Traits::Gemm::ElementA *>(base, size, groups);
    result.b              = take<const typename Traits::Gemm::ElementB *>(base, size, groups);
    result.d              = take<typename Traits::Output *>(base, size, groups);
    result.stride_a       = take<typename Traits::StrideA>(base, size, groups);
    result.stride_b       = take<typename Traits::StrideB>(base, size, groups);
    result.stride_d       = take<typename Traits::StrideD>(base, size, groups);
    result.scale_a        = take<const typename Traits::Scale *>(base, size, groups);
    result.scale_b        = take<const typename Traits::Scale *>(base, size, groups);
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
                                      const int32_t *          group_bounds,
                                      const int32_t *          row_group,
                                      void *                   dst,
                                      int                      n,
                                      int                      k,
                                      int                      weight_scale_stride,
                                      int                      groups,
                                      bool                     route_groups,
                                      bool                     pdl) {
    using Problem          = typename Traits::ProblemShape::UnderlyingProblemShape;
    using BlockScaleConfig = typename Traits::BlockScaleConfig;

    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups) {
        return;
    }

#    if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if (pdl) {
        asm volatile("griddepcontrol.wait;");
        asm volatile("griddepcontrol.launch_dependents;");
    }
#    endif

    const int     weight_group    = route_groups ? row_group[group] : group;
    const int     row_begin       = route_groups ? group : group_bounds[weight_group];
    const int     m               = route_groups ? 1 : group_bounds[weight_group + 1] - row_begin;
    const int     padded_n        = GGML_PAD(n, 128);
    const int     padded_k        = GGML_PAD(k, 128);
    const int     padded_k_blocks = padded_k / Traits::scale_granularity;
    const int64_t activation_scale_begin =
        route_groups ? (int64_t) group * 128 * padded_k_blocks :
                       ((int64_t) row_begin + (int64_t) weight_group * 127) / 128 * 128 * padded_k_blocks;

    if constexpr (Traits::swap_ab) {
        metadata.shapes[group] = Problem(n, m, k);
        metadata.a[group] =
            reinterpret_cast<const typename Traits::Gemm::ElementA *>(weights + (int64_t) weight_group * n * k / 2);
        metadata.b[group] = reinterpret_cast<const typename Traits::Gemm::ElementB *>(
            activation + (int64_t) row_begin * k * Traits::activation_bits / 8);
        metadata.stride_a[group] = cutlass::make_cute_packed_stride(typename Traits::StrideA{}, make_shape(n, k, 1));
        metadata.stride_b[group] = cutlass::make_cute_packed_stride(typename Traits::StrideB{}, make_shape(m, k, 1));
        metadata.stride_d[group] = cutlass::make_cute_packed_stride(typename Traits::StrideD{}, make_shape(n, m, 1));
        metadata.scale_a[group]  = reinterpret_cast<const typename Traits::Scale *>(
            weight_scales + (int64_t) weight_group * weight_scale_stride);
        metadata.scale_b[group] =
            reinterpret_cast<const typename Traits::Scale *>(activation_scales + activation_scale_begin);
        const auto shape               = make_shape(padded_n, m, padded_k, 1);
        metadata.layout_scale_a[group] = BlockScaleConfig::tile_atom_to_shape_SFA(shape);
        metadata.layout_scale_b[group] = BlockScaleConfig::tile_atom_to_shape_SFB(shape);
    } else {
        metadata.shapes[group] = Problem(m, n, k);
        metadata.a[group]      = reinterpret_cast<const typename Traits::Gemm::ElementA *>(
            activation + (int64_t) row_begin * k * Traits::activation_bits / 8);
        metadata.b[group] =
            reinterpret_cast<const typename Traits::Gemm::ElementB *>(weights + (int64_t) weight_group * n * k / 2);
        metadata.stride_a[group] = cutlass::make_cute_packed_stride(typename Traits::StrideA{}, make_shape(m, k, 1));
        metadata.stride_b[group] = cutlass::make_cute_packed_stride(typename Traits::StrideB{}, make_shape(n, k, 1));
        metadata.stride_d[group] = cutlass::make_cute_packed_stride(typename Traits::StrideD{}, make_shape(m, n, 1));
        metadata.scale_a[group] =
            reinterpret_cast<const typename Traits::Scale *>(activation_scales + activation_scale_begin);
        metadata.scale_b[group] = reinterpret_cast<const typename Traits::Scale *>(
            weight_scales + (int64_t) weight_group * weight_scale_stride);
        const auto shape               = make_shape(m, padded_n, padded_k, 1);
        metadata.layout_scale_a[group] = BlockScaleConfig::tile_atom_to_shape_SFA(shape);
        metadata.layout_scale_b[group] = BlockScaleConfig::tile_atom_to_shape_SFB(shape);
    }
    metadata.d[group] = reinterpret_cast<typename Traits::Output *>(dst) + (int64_t) row_begin * n;
}

template <typename Traits>
static bool run_grouped_gemm(ggml_backend_cuda_context &      ctx,
                             const ggml_cuda_cutlass_weight & weight,
                             const uint8_t *                  activation,
                             const uint8_t *                  activation_scales,
                             const int32_t *                  group_bounds,
                             const int32_t *                  row_group,
                             void *                           dst,
                             int                              group_count,
                             int64_t                          n_rows,
                             int                              n,
                             int                              k,
                             int                              sm_count,
                             cudaStream_t                     stream,
                             bool                             pdl,
                             bool                             route_groups,
                             bool                             require) {
    using Gemm = typename Traits::Gemm;

    const int groups        = route_groups ? (int) n_rows : group_count;
    size_t    metadata_size = 0;
    make_metadata<Traits>(nullptr, groups, metadata_size);
    ggml_cuda_pool_alloc<char> metadata_alloc(ctx.pool());
    char *                     metadata_data = metadata_alloc.alloc(metadata_size);
    grouped_metadata<Traits>   metadata      = make_metadata<Traits>(metadata_data, groups, metadata_size);

    constexpr int      threads = 128;
    cudaLaunchConfig_t launch_config{};
    launch_config.gridDim  = (groups + threads - 1) / threads;
    launch_config.blockDim = threads;
    launch_config.stream   = stream;
    cudaLaunchAttribute attribute{};
    attribute.id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute.val.programmaticStreamSerializationAllowed = true;
    launch_config.attrs                                  = pdl ? &attribute : nullptr;
    launch_config.numAttrs                               = pdl ? 1 : 0;
    auto setup_kernel                                    = setup_metadata<Traits>;
    CUDA_CHECK(cudaLaunchKernelEx(&launch_config, setup_kernel, metadata, activation, activation_scales, weight.data,
                                  weight.scales, group_bounds, row_group, dst, n, k, weight.scale_stride, groups,
                                  route_groups, pdl));

    typename Traits::ProblemShape shapes;
    shapes.num_groups          = groups;
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
            GGML_ABORT("CUTLASS grouped can_implement failed for tile 128x%dx128 swap=%d: %s", Traits::tile_n,
                       Traits::swap_ab, cutlassGetStatusString(can_implement));
        }
        return false;
    }

    const size_t               workspace_size = Gemm::get_workspace_size(arguments);
    ggml_cuda_pool_alloc<char> workspace_alloc(ctx.pool());
    void *                     workspace  = workspace_size == 0 ? nullptr : workspace_alloc.alloc(workspace_size);
    const cutlass::Status      initialize = gemm.initialize(arguments, workspace);
    if (initialize != cutlass::Status::kSuccess) {
        if (require) {
            GGML_ABORT("CUTLASS grouped initialize failed for tile 128x%dx128 swap=%d: %s", Traits::tile_n,
                       Traits::swap_ab, cutlassGetStatusString(initialize));
        }
        return false;
    }

    const cutlass::Status run = gemm.run(stream, nullptr, pdl);
    if (run != cutlass::Status::kSuccess) {
        if (require) {
            GGML_ABORT("CUTLASS grouped run failed for tile 128x%dx128 swap=%d: %s", Traits::tile_n, Traits::swap_ab,
                       cutlassGetStatusString(run));
        }
        return false;
    }
    return true;
}

template <template <int, bool> class Traits>
static bool dispatch_grouped_gemm(ggml_backend_cuda_context &      ctx,
                                  const ggml_cuda_cutlass_weight & weight,
                                  const uint8_t *                  activation,
                                  const uint8_t *                  activation_scales,
                                  const int32_t *                  group_bounds,
                                  const int32_t *                  row_group,
                                  void *                           dst,
                                  int                              group_count,
                                  int64_t                          n_rows,
                                  int                              n,
                                  int                              k,
                                  int                              sm_count,
                                  ggml_cuda_cutlass_config         config,
                                  cudaStream_t                     stream,
                                  bool                             require) {
    if (config.tile_n == 32) {
        return config.swap_ab ?
                   run_grouped_gemm<Traits<32, true>>(ctx, weight, activation, activation_scales, group_bounds,
                                                      row_group, dst, group_count, n_rows, n, k, sm_count, stream,
                                                      config.pdl, config.route_groups, require) :
                   run_grouped_gemm<Traits<32, false>>(ctx, weight, activation, activation_scales, group_bounds,
                                                       row_group, dst, group_count, n_rows, n, k, sm_count, stream,
                                                       config.pdl, config.route_groups, require);
    }
    if (config.tile_n == 64) {
        return config.swap_ab ?
                   run_grouped_gemm<Traits<64, true>>(ctx, weight, activation, activation_scales, group_bounds,
                                                      row_group, dst, group_count, n_rows, n, k, sm_count, stream,
                                                      config.pdl, config.route_groups, require) :
                   run_grouped_gemm<Traits<64, false>>(ctx, weight, activation, activation_scales, group_bounds,
                                                       row_group, dst, group_count, n_rows, n, k, sm_count, stream,
                                                       config.pdl, config.route_groups, require);
    }
    return config.swap_ab ?
               run_grouped_gemm<Traits<128, true>>(ctx, weight, activation, activation_scales, group_bounds, row_group,
                                                   dst, group_count, n_rows, n, k, sm_count, stream, config.pdl,
                                                   config.route_groups, require) :
               run_grouped_gemm<Traits<128, false>>(ctx, weight, activation, activation_scales, group_bounds, row_group,
                                                    dst, group_count, n_rows, n, k, sm_count, stream, config.pdl,
                                                    config.route_groups, require);
}

}  // namespace ggml_cutlass_sm120

bool ggml_cuda_cutlass_compiled() {
    return true;
}

bool ggml_cuda_cutlass_grouped_gemm(ggml_backend_cuda_context &      ctx,
                                    const ggml_cuda_cutlass_weight & weight,
                                    const uint8_t *                  activation,
                                    const uint8_t *                  activation_scales,
                                    const int32_t *                  group_bounds,
                                    const int32_t *                  row_group,
                                    void *                           dst,
                                    int                              groups,
                                    int64_t                          n_rows,
                                    int64_t                          n,
                                    int64_t                          k,
                                    int                              sm_count,
                                    ggml_cuda_cutlass_config         config,
                                    cudaStream_t                     stream,
                                    bool                             require) {
    using namespace ggml_cutlass_sm120;
    GGML_ASSERT(weight.data != nullptr && weight.scales != nullptr && activation != nullptr &&
                activation_scales != nullptr && dst != nullptr);
    GGML_ASSERT(groups > 0 && n > 0 && k > 0 && n_rows > 0);
    GGML_ASSERT(n <= INT_MAX && k <= INT_MAX && n_rows <= INT_MAX && k == weight.k);
    GGML_ASSERT(config.tile_n == 32 || config.tile_n == 64 || config.tile_n == 128);
    GGML_ASSERT(!config.route_groups || row_group != nullptr);
    GGML_ASSERT(config.route_groups || group_bounds != nullptr);

    if (weight.type == GGML_TYPE_MXFP4) {
        return dispatch_grouped_gemm<mxfp_kernel_traits>(ctx, weight, activation, activation_scales, group_bounds,
                                                         row_group, dst, groups, n_rows, (int) n, (int) k, sm_count,
                                                         config, stream, require);
    }
    if (weight.type == GGML_TYPE_NVFP4) {
        return dispatch_grouped_gemm<nvfp4_kernel_traits>(ctx, weight, activation, activation_scales, group_bounds,
                                                          row_group, dst, groups, n_rows, (int) n, (int) k, sm_count,
                                                          config, stream, require);
    }
    if (require) {
        GGML_ABORT("CUTLASS does not support weight type %s", ggml_type_name(weight.type));
    }
    return false;
}

bool ggml_cuda_cutlass_mul_mat(ggml_backend_cuda_context & ctx,
                               const ggml_tensor *         src0,
                               const ggml_tensor *         src1,
                               ggml_tensor *               dst) {
    using namespace ggml_cutlass_sm120;

    const char * disable = std::getenv("GGML_CUDA_CUTLASS_DISABLE");
    if (disable != nullptr && std::atoi(disable) != 0) {
        return false;
    }
    if ((src0->type != GGML_TYPE_MXFP4 && src0->type != GGML_TYPE_NVFP4) || src1->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32 || src0->ne[2] != 1 || src0->ne[3] != 1 || !ggml_is_contiguous(src0) ||
        !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = ggml_nelements(src1) / k;
    if (k <= 0 || n <= 0 || m < 256 || m > INT_MAX || n > INT_MAX || k > INT_MAX - 127 ||
        ggml_nelements(src1) != k * m || ggml_nelements(dst) != n * m || src1->ne[0] != k || dst->ne[0] != n) {
        return false;
    }

    const auto & device_info = ggml_cuda_info().devices[ctx.device];
    if (!blackwell_mma_available(device_info.cc)) {
        return false;
    }
    const ggml_backend_buffer_type_t buffer_type = ggml_backend_cuda_buffer_type(ctx.device);
    const ggml_tensor *              tensors[]   = { src0, src1, dst };
    for (const ggml_tensor * tensor : tensors) {
        if (tensor->buffer == nullptr || ggml_backend_buffer_get_type(tensor->buffer) != buffer_type) {
            return false;
        }
    }

    cudaStream_t            stream = ctx.stream();
    cudaStreamCaptureStatus capture_status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    if (capture_status == cudaStreamCaptureStatusNone && ctx.cutlass_weight_stream == nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.cutlass_weight_stream, cudaStreamNonBlocking));
    }
    cudaStream_t repack_stream = capture_status == cudaStreamCaptureStatusNone ? ctx.cutlass_weight_stream : stream;

    ggml_cuda_cutlass_weight weight;
    if (!ggml_cuda_cutlass_repack_weight(ctx, src0, weight, repack_stream, false)) {
        return false;
    }

    const int64_t                 k_padded = weight.k;
    ggml_cuda_pool_alloc<uint8_t> activation(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> activation_scales(ctx.pool());
    uint8_t * activation_data       = activation.alloc(ggml_cuda_cutlass_activation_size(src0->type, m, k_padded));
    uint8_t * activation_scale_data = activation_scales.alloc(ggml_cuda_cutlass_scale_size(src0->type, m, k_padded));
    if (!ggml_cuda_cutlass_quantize((const float *) src1->data, activation_data, activation_scale_data, src0->type, k,
                                    k_padded, src1->nb[1] / sizeof(float), m, stream)) {
        return false;
    }

    ggml_cuda_cutlass_weight_wait_ready(weight, stream);
    const bool launched = dispatch_dense_gemm<float>(ctx, weight, activation_data, activation_scale_data, dst->data,
                                                     (int) m, (int) n, (int) k_padded, stream, false);
    return launched;
}

bool ggml_cuda_cutlass_ffn(ggml_backend_cuda_context & ctx, const ggml_cuda_cutlass_ffn_args & args) {
    using namespace ggml_cutlass_sm120;

    const char * disable = std::getenv("GGML_CUDA_CUTLASS_DISABLE");
    if (disable != nullptr && std::atoi(disable) != 0) {
        return false;
    }
    if (args.gate == nullptr || args.up == nullptr || args.down == nullptr || args.input == nullptr ||
        args.dst == nullptr || (args.gate->type != GGML_TYPE_MXFP4 && args.gate->type != GGML_TYPE_NVFP4) ||
        args.up->type != args.gate->type || args.down->type != args.gate->type ||
        !ggml_are_same_shape(args.gate, args.up) || args.input->type != GGML_TYPE_F32 ||
        args.dst->type != GGML_TYPE_F32 || args.gate->ne[2] != 1 || args.gate->ne[3] != 1 || args.down->ne[2] != 1 ||
        args.down->ne[3] != 1 || args.down->ne[0] != args.gate->ne[1] || !ggml_is_contiguous(args.gate) ||
        !ggml_is_contiguous(args.up) || !ggml_is_contiguous(args.down) || !ggml_is_contiguous(args.input) ||
        !ggml_is_contiguous(args.dst)) {
        return false;
    }

    const int64_t n_embd = args.gate->ne[0];
    const int64_t n_ff   = args.gate->ne[1];
    const int64_t n_out  = args.down->ne[1];
    const int64_t n_rows = ggml_nelements(args.input) / n_embd;
    if (n_embd <= 0 || n_ff <= 0 || n_out <= 0 || n_rows < 256 || n_rows > INT_MAX || n_embd > INT_MAX - 127 ||
        n_ff > (INT_MAX - 127) / 2 || n_out > INT_MAX || args.input->ne[0] != n_embd ||
        ggml_nelements(args.input) != n_rows * n_embd || args.dst->ne[0] != n_out ||
        ggml_nelements(args.dst) != n_rows * n_out) {
        return false;
    }

    const ggml_tensor * scales[] = { args.gate_scale, args.up_scale, args.down_scale };
    for (const ggml_tensor * scale : scales) {
        if (scale != nullptr &&
            (scale->type != GGML_TYPE_F32 || ggml_nelements(scale) != 1 || !ggml_is_contiguous(scale))) {
            return false;
        }
    }

    const auto & device_info = ggml_cuda_info().devices[ctx.device];
    if (!blackwell_mma_available(device_info.cc)) {
        return false;
    }
    const ggml_backend_buffer_type_t buffer_type = ggml_backend_cuda_buffer_type(ctx.device);
    const ggml_tensor *              tensors[]   = {
        args.gate, args.up, args.down, args.input, args.gate_scale, args.up_scale, args.down_scale, args.dst,
    };
    for (const ggml_tensor * tensor : tensors) {
        if (tensor != nullptr &&
            (tensor->buffer == nullptr || ggml_backend_buffer_get_type(tensor->buffer) != buffer_type)) {
            return false;
        }
    }

    cudaStream_t            stream = ctx.stream();
    cudaStreamCaptureStatus capture_status;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    if (capture_status == cudaStreamCaptureStatusNone && ctx.cutlass_weight_stream == nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.cutlass_weight_stream, cudaStreamNonBlocking));
    }
    cudaStream_t repack_stream = capture_status == cudaStreamCaptureStatusNone ? ctx.cutlass_weight_stream : stream;

    ggml_cuda_cutlass_weight w13_weight;
    ggml_cuda_cutlass_weight w2_weight;
    if (!ggml_cuda_cutlass_repack_weight_pair(ctx, args.gate, args.up, w13_weight, repack_stream) ||
        !ggml_cuda_cutlass_repack_weight(ctx, args.down, w2_weight, repack_stream, false)) {
        return false;
    }

    ggml_cuda_pool_alloc<uint8_t> w13_activation(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> w13_scales(ctx.pool());
    uint8_t *                     w13_activation_data =
        w13_activation.alloc(ggml_cuda_cutlass_activation_size(args.gate->type, n_rows, w13_weight.k));
    uint8_t * w13_scale_data = w13_scales.alloc(ggml_cuda_cutlass_scale_size(args.gate->type, n_rows, w13_weight.k));
    if (!ggml_cuda_cutlass_quantize((const float *) args.input->data, w13_activation_data, w13_scale_data,
                                    args.gate->type, n_embd, w13_weight.k, args.input->nb[1] / sizeof(float), n_rows,
                                    stream)) {
        return false;
    }

    ggml_cuda_pool_alloc<__nv_bfloat16> w13_output(ctx.pool(), (size_t) n_rows * 2 * n_ff);
    ggml_cuda_cutlass_weight_wait_ready(w13_weight, stream);
    if (!dispatch_dense_gemm<cutlass::bfloat16_t>(ctx, w13_weight, w13_activation_data, w13_scale_data,
                                                  w13_output.get(), (int) n_rows, (int) (2 * n_ff), (int) w13_weight.k,
                                                  stream, false)) {
        return false;
    }

    ggml_cuda_pool_alloc<uint8_t> w2_activation(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> w2_scales(ctx.pool());
    uint8_t *                     w2_activation_data =
        w2_activation.alloc(ggml_cuda_cutlass_activation_size(args.down->type, n_rows, w2_weight.k));
    uint8_t * w2_scale_data = w2_scales.alloc(ggml_cuda_cutlass_scale_size(args.down->type, n_rows, w2_weight.k));
    CUDA_CHECK(
        cudaMemsetAsync(w2_scale_data, 0, ggml_cuda_cutlass_scale_size(args.down->type, n_rows, w2_weight.k), stream));

    constexpr int threads = 256;
    if (args.down->type == GGML_TYPE_NVFP4) {
        cutlass_nvfp4_swiglu_quantize<<<(unsigned) n_rows, threads, 0, stream>>>(
            w13_output.get(), args.gate_scale == nullptr ? nullptr : (const float *) args.gate_scale->data,
            args.up_scale == nullptr ? nullptr : (const float *) args.up_scale->data, w2_activation_data, w2_scale_data,
            n_ff, w2_weight.k);
    } else {
        cutlass_mxfp8_swiglu_quantize<<<(unsigned) n_rows, threads, 0, stream>>>(
            w13_output.get(), args.gate_scale == nullptr ? nullptr : (const float *) args.gate_scale->data,
            args.up_scale == nullptr ? nullptr : (const float *) args.up_scale->data, w2_activation_data, w2_scale_data,
            n_ff, w2_weight.k);
    }
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_pool_alloc<__nv_bfloat16> w2_output(ctx.pool(), (size_t) n_rows * n_out);
    ggml_cuda_cutlass_weight_wait_ready(w2_weight, stream);
    if (!dispatch_dense_gemm<cutlass::bfloat16_t>(ctx, w2_weight, w2_activation_data, w2_scale_data, w2_output.get(),
                                                  (int) n_rows, (int) n_out, (int) w2_weight.k, stream, false)) {
        return false;
    }

    const int64_t output_size = n_rows * n_out;
    cutlass_finalize_bf16<<<(output_size + threads - 1) / threads, threads, 0, stream>>>(
        w2_output.get(), args.down_scale == nullptr ? nullptr : (const float *) args.down_scale->data,
        (float *) args.dst->data, output_size);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

static bool moe_cutlass_nvfp4_prefill(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_cutlass_nvfp4_args & args) {
    if (!ggml_cuda_moe_cutlass_prefill_requested()) {
        return false;
    }

    if (args.gate == nullptr || args.up == nullptr || args.down == nullptr || args.input == nullptr ||
        args.ids == nullptr || args.gate_scale == nullptr || args.up_scale == nullptr || args.down_scale == nullptr ||
        args.weights == nullptr || args.dst == nullptr) {
        return false;
    }
    const int64_t n_experts_64     = args.gate->ne[2];
    const int64_t n_expert_used_64 = args.ids->ne[0];
    const int64_t n_embd_64        = args.gate->ne[0];
    const int64_t n_ff_64          = args.gate->ne[1];
    const int64_t n_tokens         = args.ids->ne[1];
    if (n_experts_64 <= 0 || n_experts_64 > 256 || (n_expert_used_64 != 4 && n_expert_used_64 != 8) ||
        n_expert_used_64 > n_experts_64 || n_embd_64 <= 0 || n_embd_64 > INT_MAX - 127 || n_ff_64 <= 0 ||
        n_ff_64 > (INT_MAX - 127) / 2 || n_tokens < 256 || n_tokens > INT_MAX / n_expert_used_64) {
        return false;
    }
    const int n_experts     = (int) n_experts_64;
    const int n_expert_used = (int) n_expert_used_64;
    const int n_embd        = (int) n_embd_64;
    const int n_ff          = (int) n_ff_64;
    const int n_rows        = (int) n_tokens * n_expert_used;

    const bool valid_shape =
        args.gate->type == GGML_TYPE_NVFP4 && args.gate->ne[0] == n_embd && args.gate->ne[1] == n_ff &&
        args.gate->ne[2] == n_experts && args.gate->ne[3] == 1 && args.up->type == GGML_TYPE_NVFP4 &&
        ggml_are_same_shape(args.gate, args.up) && args.down->type == GGML_TYPE_NVFP4 && args.down->ne[0] == n_ff &&
        args.down->ne[1] == n_embd && args.down->ne[2] == n_experts && args.down->ne[3] == 1 &&
        args.input->type == GGML_TYPE_F32 && args.input->ne[0] == n_embd && args.input->ne[1] == 1 &&
        args.input->ne[2] == n_tokens && args.input->ne[3] == 1 && args.ids->type == GGML_TYPE_I32 &&
        args.ids->ne[0] == n_expert_used && args.ids->ne[1] == n_tokens && args.ids->ne[2] == 1 &&
        args.ids->ne[3] == 1 && args.gate_scale->type == GGML_TYPE_F32 && args.gate_scale->ne[0] == n_experts &&
        args.gate_scale->ne[1] == 1 && args.gate_scale->ne[2] == 1 && args.gate_scale->ne[3] == 1 &&
        args.up_scale->type == GGML_TYPE_F32 && args.up_scale->ne[0] == n_experts && args.up_scale->ne[1] == 1 &&
        args.up_scale->ne[2] == 1 && args.up_scale->ne[3] == 1 && args.down_scale->type == GGML_TYPE_F32 &&
        args.down_scale->ne[0] == n_experts && args.down_scale->ne[1] == 1 && args.down_scale->ne[2] == 1 &&
        args.down_scale->ne[3] == 1 && args.weights->type == GGML_TYPE_F32 && args.weights->ne[0] == 1 &&
        args.weights->ne[1] == n_expert_used && args.weights->ne[2] == n_tokens && args.weights->ne[3] == 1 &&
        args.dst->type == GGML_TYPE_F32 && args.dst->ne[0] == n_embd && args.dst->ne[1] == n_tokens &&
        args.dst->ne[2] == 1 && args.dst->ne[3] == 1;
    if (!valid_shape) {
        return false;
    }

    const ggml_backend_buffer_type_t          buffer_type = ggml_backend_cuda_buffer_type(ctx.device);
    const std::array<const ggml_tensor *, 10> tensors     = {
        args.gate,       args.up,       args.down,       args.input,   args.ids,
        args.gate_scale, args.up_scale, args.down_scale, args.weights, args.dst,
    };
    const bool valid_buffers = std::all_of(tensors.begin(), tensors.end(), [buffer_type](const ggml_tensor * tensor) {
        return tensor->buffer != nullptr && ggml_backend_buffer_get_type(tensor->buffer) == buffer_type;
    });
    const bool valid_layout =
        ggml_is_contiguous(args.gate) && ggml_is_contiguous(args.up) && ggml_is_contiguous(args.down) &&
        ggml_is_contiguous(args.input) && ggml_is_contiguous_rows(args.ids) && args.ids->nb[0] == sizeof(int32_t) &&
        args.ids->nb[1] >= ggml_row_size(args.ids->type, args.ids->ne[0]) && ggml_is_contiguous(args.gate_scale) &&
        ggml_is_contiguous(args.up_scale) && ggml_is_contiguous(args.down_scale) && ggml_is_contiguous(args.weights) &&
        ggml_is_contiguous(args.dst);
    if (!valid_buffers || !valid_layout) {
        return false;
    }

    const auto & device_info = ggml_cuda_info().devices[ctx.device];
    if (!blackwell_mma_available(device_info.cc)) {
        return false;
    }

    cudaStream_t             stream = ctx.stream();
    ggml_cuda_cutlass_weight w13_weight;
    ggml_cuda_cutlass_weight w2_weight;
    {
        if (!ggml_cuda_cutlass_repack_weight_pair(ctx, args.gate, args.up, w13_weight, stream) ||
            !ggml_cuda_cutlass_repack_weight(ctx, args.down, w2_weight, stream, false)) {
            return false;
        }
    }

    ggml_cuda_pool_alloc<int32_t> staged_ids(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<float>   staged_weights(ctx.pool(), n_rows);
    constexpr int                 stage_threads        = 256;
    const int                     ids_stride           = args.ids->nb[1] / sizeof(int32_t);
    const int                     weights_route_stride = args.weights->nb[1] / sizeof(float);
    const int                     weights_token_stride = args.weights->nb[2] / sizeof(float);
    const int                     n_blocks = ggml_cuda_mm_ids_prefix_block_count((int) n_tokens, n_expert_used);
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), n_experts + 1);
    ggml_cuda_pool_alloc<int32_t> row_expert(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> block_counts(ctx.pool(), (size_t) n_blocks * n_experts);
    ggml_cuda_pool_alloc<int32_t> block_offsets(ctx.pool(), (size_t) n_blocks * n_experts);
    {
        moe_cutlass_stage_routes<<<(n_rows + stage_threads - 1) / stage_threads, stage_threads, 0, stream>>>(
            (const int32_t *) args.ids->data, (const float *) args.weights->data, staged_ids.get(),
            staged_weights.get(), n_rows, n_expert_used, ids_stride, weights_route_stride, weights_token_stride);
        CUDA_CHECK(cudaGetLastError());
        if (!ggml_cuda_launch_mm_ids_prefix(staged_ids.get(), ids_src1.get(), ids_dst.get(), expert_bounds.get(),
                                            row_expert.get(), block_counts.get(), block_offsets.get(), n_experts,
                                            (int) n_tokens, n_expert_used, n_expert_used, stream)) {
            return false;
        }
    }

    const int    w13_k               = (int) w13_weight.k;
    const int    w13_n               = 2 * n_ff;
    const size_t w13_activation_size = ggml_cuda_cutlass_activation_size(GGML_TYPE_NVFP4, n_rows, w13_k);
    const size_t w13_scale_size      = moe_cutlass_nvfp4_scale_size(n_rows, n_experts, w13_k);
    ggml_cuda_pool_alloc<uint8_t> w13_activation(ctx.pool(), w13_activation_size);
    ggml_cuda_pool_alloc<uint8_t> w13_activation_scales(ctx.pool(), w13_scale_size);
    {
        CUDA_CHECK(cudaMemsetAsync(w13_activation_scales.get(), 0, w13_scale_size, stream));
        if (!moe_cutlass_quantize_nvfp4_broadcast((const float *) args.input->data, staged_ids.get(), ids_src1.get(),
                                                  expert_bounds.get(), w13_activation.get(),
                                                  w13_activation_scales.get(), n_embd, w13_k,
                                                  args.input->nb[2] / sizeof(float), n_tokens, n_expert_used, stream)) {
            return false;
        }
    }

    const ggml_cuda_cutlass_config w13_config = {
        moe_cutlass_nvfp4_tile_n("GGML_CUDA_MOE_MMQ_CUTLASS_W13_TILE_N", n_rows, n_experts),
        moe_cutlass_swap_requested("GGML_CUDA_MOE_MMQ_CUTLASS_W13_SWAP_AB"),
        moe_cutlass_pdl_requested(),
        false,
    };
    ggml_cuda_pool_alloc<__nv_bfloat16> w13_output(ctx.pool(), (size_t) n_rows * w13_n);
    {
        ggml_cuda_cutlass_weight_wait_ready(w13_weight, stream);
        if (!ggml_cuda_cutlass_grouped_gemm(ctx, w13_weight, w13_activation.get(), w13_activation_scales.get(),
                                            expert_bounds.get(), row_expert.get(), w13_output.get(), n_experts, n_rows,
                                            w13_n, w13_k, device_info.nsm, w13_config, stream, false)) {
            return false;
        }
    }

    const int                     w2_k               = (int) w2_weight.k;
    const int                     w2_n               = n_embd;
    const size_t                  w2_activation_size = ggml_cuda_cutlass_activation_size(GGML_TYPE_NVFP4, n_rows, w2_k);
    const size_t                  w2_scale_size      = moe_cutlass_nvfp4_scale_size(n_rows, n_experts, w2_k);
    ggml_cuda_pool_alloc<uint8_t> w2_activation(ctx.pool(), w2_activation_size);
    ggml_cuda_pool_alloc<uint8_t> w2_activation_scales(ctx.pool(), w2_scale_size);
    {
        CUDA_CHECK(cudaMemsetAsync(w2_activation_scales.get(), 0, w2_scale_size, stream));
        moe_cutlass_nvfp4_w13_epilogue<<<n_rows, 256, 0, stream>>>(
            w13_output.get(), row_expert.get(), expert_bounds.get(), (const float *) args.gate_scale->data,
            (const float *) args.up_scale->data, w2_activation.get(), w2_activation_scales.get(), n_ff, w2_k, n_rows);
        CUDA_CHECK(cudaGetLastError());
    }

    const ggml_cuda_cutlass_config w2_config = {
        moe_cutlass_nvfp4_tile_n("GGML_CUDA_MOE_MMQ_CUTLASS_W2_TILE_N", n_rows, n_experts),
        moe_cutlass_swap_requested("GGML_CUDA_MOE_MMQ_CUTLASS_W2_SWAP_AB"),
        moe_cutlass_pdl_requested(),
        false,
    };
    ggml_cuda_pool_alloc<__nv_bfloat16> w2_output(ctx.pool(), (size_t) n_rows * w2_n);
    {
        ggml_cuda_cutlass_weight_wait_ready(w2_weight, stream);
        if (!ggml_cuda_cutlass_grouped_gemm(ctx, w2_weight, w2_activation.get(), w2_activation_scales.get(),
                                            expert_bounds.get(), row_expert.get(), w2_output.get(), n_experts, n_rows,
                                            w2_n, w2_k, device_info.nsm, w2_config, stream, false)) {
            return false;
        }
    }

    constexpr int finalize_threads = 256;
    const int64_t output_size      = n_tokens * n_embd;
    {
        moe_cutlass_nvfp4_w2_finalize<<<(output_size + finalize_threads - 1) / finalize_threads, finalize_threads, 0,
                                        stream>>>(w2_output.get(), staged_ids.get(), ids_src1.get(),
                                                  (const float *) args.down_scale->data, staged_weights.get(),
                                                  (float *) args.dst->data, n_embd, (int) n_tokens, n_expert_used);
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}

bool ggml_cuda_moe_cutlass_nvfp4(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_cutlass_nvfp4_args & args) {
    return moe_cutlass_nvfp4_prefill(ctx, args);
}

#else

bool ggml_cuda_cutlass_compiled() {
    return false;
}

bool ggml_cuda_cutlass_mul_mat(ggml_backend_cuda_context & ctx,
                               const ggml_tensor *         src0,
                               const ggml_tensor *         src1,
                               ggml_tensor *               dst) {
    GGML_UNUSED_VARS(ctx, src0, src1, dst);
    return false;
}

bool ggml_cuda_cutlass_ffn(ggml_backend_cuda_context & ctx, const ggml_cuda_cutlass_ffn_args & args) {
    GGML_UNUSED_VARS(ctx, args);
    return false;
}

size_t ggml_cuda_cutlass_activation_size(ggml_type type, int64_t n_rows, int64_t n_cols) {
    GGML_UNUSED_VARS(type, n_rows, n_cols);
    return 0;
}

size_t ggml_cuda_cutlass_scale_size(ggml_type type, int64_t n_rows, int64_t n_cols) {
    GGML_UNUSED_VARS(type, n_rows, n_cols);
    return 0;
}

bool ggml_cuda_cutlass_quantize(const float * src,
                                uint8_t *     dst,
                                uint8_t *     scales,
                                ggml_type     type,
                                int64_t       n_cols,
                                int64_t       n_cols_padded,
                                int64_t       stride_row,
                                int64_t       n_rows,
                                cudaStream_t  stream) {
    GGML_UNUSED_VARS(src, dst, scales, type, n_cols, n_cols_padded, stride_row, n_rows, stream);
    return false;
}

bool ggml_cuda_moe_cutlass_nvfp4(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_cutlass_nvfp4_args & args) {
    GGML_UNUSED_VARS(ctx, args);
    return false;
}

size_t ggml_cuda_moe_cutlass_scale_size(int64_t n_rows, int n_experts, int64_t n_cols, bool route_groups) {
    GGML_UNUSED_VARS(n_rows, n_experts, n_cols, route_groups);
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
                                              bool            route_groups,
                                              cudaStream_t    stream) {
    GGML_UNUSED_VARS(src, ids, ids_src1, expert_bounds, dst, scales, n_cols, n_cols_padded, stride_token, n_tokens,
                     n_experts, n_expert_used, ids_stride, route_groups, stream);
    GGML_ABORT("the CUTLASS MoE backend was not compiled");
}

bool ggml_cuda_moe_cutlass_quantize_broadcast_cta(const float *   src,
                                                  const int32_t * ids,
                                                  int32_t *       ids_src1,
                                                  int32_t *       ids_dst,
                                                  int32_t *       row_expert,
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
                                                  bool            route_groups,
                                                  cudaStream_t    stream) {
    GGML_UNUSED_VARS(src, ids, ids_src1, ids_dst, row_expert, expert_bounds, dst, scales, n_cols, n_cols_padded,
                     stride_token, n_tokens, n_experts, n_expert_used, ids_stride, route_groups, stream);
    return false;
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
                                        bool            route_groups,
                                        cudaStream_t    stream) {
    GGML_UNUSED_VARS(gate_up, bias, ids, ids_dst, expert_bounds, dst, scales, n_ff, n_ff_padded, n_rows, n_experts,
                     n_expert_used, ids_stride, route_groups, stream);
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
                                            bool            route_groups,
                                            cudaStream_t    stream) {
    GGML_UNUSED_VARS(gate_up, bias, ids, ids_dst, row_expert, expert_bounds, dst, scales, n_ff, n_ff_padded, n_rows,
                     n_experts, n_expert_used, rows_per_cta, ids_stride, route_groups, stream);
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

bool ggml_cuda_cutlass_grouped_gemm(ggml_backend_cuda_context &      ctx,
                                    const ggml_cuda_cutlass_weight & weight,
                                    const uint8_t *                  activation,
                                    const uint8_t *                  activation_scales,
                                    const int32_t *                  group_bounds,
                                    const int32_t *                  row_group,
                                    void *                           dst,
                                    int                              groups,
                                    int64_t                          n_rows,
                                    int64_t                          n,
                                    int64_t                          k,
                                    int                              sm_count,
                                    ggml_cuda_cutlass_config         config,
                                    cudaStream_t                     stream,
                                    bool                             require) {
    GGML_UNUSED_VARS(ctx, weight, activation, activation_scales, group_bounds, row_group, dst, groups, n_rows, n, k,
                     sm_count, config, stream, require);
    return false;
}

#endif
