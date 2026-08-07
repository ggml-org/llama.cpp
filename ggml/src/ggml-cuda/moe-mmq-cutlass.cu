#include "moe-mmq-cutlass.cuh"
#include "mmid.cuh"
#include "moe-mmq.cuh"
#include "unary.cuh"
#ifdef GGML_CUDA_MOE_PROFILE
#include "moe-profile.cuh"
#endif

#include <cstdlib>

bool ggml_cuda_moe_cutlass_decode_fused_requested() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CUDA_MOE_MMQ_CUTLASS_DECODE_FUSED");
        return value != nullptr && std::atoi(value) != 0;
    }();
    return enabled;
}

#ifdef GGML_CUDA_CUTLASS_MOE
#    include <cuda_bf16.h>
#    include <cuda_fp8.h>

#    include <algorithm>
#    include <array>
#    include <atomic>
#    include <climits>
#    include <type_traits>

static bool moe_cutlass_decode_output_f32_requested() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CUDA_MOE_MMQ_CUTLASS_DECODE_OUTPUT_F32");
        return value != nullptr && std::atoi(value) != 0;
    }();
    return enabled;
}

static bool moe_cutlass_nvfp4_prefill_log_requested() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_CUDA_MOE_MMQ_CUTLASS_NVFP4_PREFILL_LOG");
        return value != nullptr && std::atoi(value) != 0;
    }();
    return enabled;
}

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
    return (int64_t) m_tile * m_tile_stride + (int64_t) k_tile * k_tile_stride + outer_m * 16 +
           inner_m * 4 + inner_k;
}

static __device__ __forceinline__ uint8_t * moe_cutlass_scale_ptr(uint8_t *       scales,
                                                                  const int32_t * expert_bounds,
                                                                  int             expert,
                                                                  int             row,
                                                                  int             k_block,
                                                                  int             padded_k_blocks,
                                                                  bool            route_groups) {
    if (route_groups) {
        return scales + (int64_t) row * 128 * padded_k_blocks +
               moe_cutlass_scale_offset(0, k_block, padded_k_blocks);
    }
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
    constexpr int warps = 8;
    __shared__ int route_rows[32];
    __shared__ int route_experts[32];

    const int64_t token = blockIdx.x;
    if (threadIdx.x < (unsigned) n_expert_used) {
        const int     slot   = threadIdx.x;
        const int64_t route  = token * n_expert_used + slot;
        const int     expert = ids[token * ids_stride + slot];
        const int     row    = route_groups ? (int) route : ids_src1[route];
        route_rows[slot]      = row;
        route_experts[slot]   = expert;
        if (route_groups) {
            ids_src1[route]  = row;
            ids_dst[row]     = (int32_t) route;
            row_expert[row]  = expert;
        }
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
                    scales, expert_bounds, route_experts[slot], row, k_block, padded_k_blocks, route_groups) = scale;
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
                                                   int64_t ids_stride,
                                                   bool    route_groups) {
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
        *moe_cutlass_scale_ptr(scales, expert_bounds, expert, row, k_block, padded_k_blocks, route_groups) = scale;
    }
}

static __global__ void moe_cutlass_quantize_nvfp4_routes(const float * __restrict__ src,
                                                          const int32_t * __restrict__ ids_dst,
                                                          uint8_t * __restrict__ dst,
                                                          uint8_t * __restrict__ scales,
                                                          int64_t n_cols,
                                                          int64_t n_cols_padded,
                                                          int64_t stride_route,
                                                          int64_t stride_token,
                                                          int     n_expert_used) {
    const int row         = blockIdx.x;
    const int scale_block = blockIdx.y;
    const int lane        = threadIdx.x;
    if (lane >= QK_NVFP4_SUB) {
        return;
    }

    const int route = ids_dst[row];
    const int token = route / n_expert_used;
    const int slot  = route - token * n_expert_used;
    const int64_t k = (int64_t) scale_block * QK_NVFP4_SUB + lane;
    const float value = k < n_cols ? src[(int64_t) token * stride_token + (int64_t) slot * stride_route + k] : 0.0f;

    float amax = fabsf(value);
#    pragma unroll
    for (int mask = QK_NVFP4_SUB / 2; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFF, amax, mask, QK_NVFP4_SUB));
    }

    const uint8_t scale_code = ggml_cuda_fp32_to_ue4m3(amax / 6.0f);
    const float   scale      = ggml_cuda_ue4m3_to_fp32(scale_code);
    const float   inv_scale  = scale > 0.0f ? 0.5f / scale : 0.0f;
    const uint8_t quantized  = ggml_cuda_float_to_fp4_e2m1(value, inv_scale);
    const uint8_t next       = (uint8_t) __shfl_down_sync(0xFFFF, (unsigned) quantized, 1, QK_NVFP4_SUB);
    if ((lane & 1) == 0) {
        dst[(int64_t) row * (n_cols_padded / 2) + (int64_t) scale_block * (QK_NVFP4_SUB / 2) + lane / 2] =
            quantized | (next << 4);
    }
    if (lane == 0) {
        const int padded_scale_blocks = n_cols_padded / QK_NVFP4_SUB;
        scales[(int64_t) row * 128 * padded_scale_blocks +
               moe_cutlass_scale_offset(0, scale_block, padded_scale_blocks)] = scale_code;
    }
}

static __global__ void moe_cutlass_quantize_nvfp4_broadcast_routes(const float * __restrict__ src,
                                                                    uint8_t * __restrict__ dst,
                                                                    uint8_t * __restrict__ scales,
                                                                    int64_t n_cols,
                                                                    int64_t n_cols_padded,
                                                                    int n_rows) {
    const int scale_block = blockIdx.x;
    const int lane        = threadIdx.x;
    const int64_t k       = (int64_t) scale_block * QK_NVFP4_SUB + lane;
    const float value     = k < n_cols ? src[k] : 0.0f;

    float amax = fabsf(value);
#pragma unroll
    for (int mask = QK_NVFP4_SUB / 2; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFF, amax, mask, QK_NVFP4_SUB));
    }

    const uint8_t scale_code = ggml_cuda_fp32_to_ue4m3(amax / 6.0f);
    const float   scale      = ggml_cuda_ue4m3_to_fp32(scale_code);
    const float   inv_scale  = scale > 0.0f ? 0.5f / scale : 0.0f;
    const uint8_t quantized  = ggml_cuda_float_to_fp4_e2m1(value, inv_scale);
    const uint8_t next       = (uint8_t) __shfl_down_sync(0xFFFF, (unsigned) quantized, 1, QK_NVFP4_SUB);
    if ((lane & 1) == 0) {
        const uint8_t packed = quantized | (next << 4);
        for (int row = 0; row < n_rows; ++row) {
            dst[(int64_t) row * (n_cols_padded / 2) +
                (int64_t) scale_block * (QK_NVFP4_SUB / 2) + lane / 2] = packed;
        }
    }
    if (lane == 0) {
        const int padded_scale_blocks = n_cols_padded / QK_NVFP4_SUB;
        for (int row = 0; row < n_rows; ++row) {
            scales[(int64_t) row * 128 * padded_scale_blocks +
                   moe_cutlass_scale_offset(0, scale_block, padded_scale_blocks)] = scale_code;
        }
    }
}

template <int n_expert_used>
static __global__ void moe_cutlass_quantize_nvfp4_broadcast_cta(
        const float * __restrict__ src,
        const int32_t * __restrict__ ids,
        const int32_t * __restrict__ ids_src1,
        const int32_t * __restrict__ expert_bounds,
        uint8_t * __restrict__ dst,
        uint8_t * __restrict__ scales,
        int64_t n_cols,
        int64_t n_cols_padded,
        int64_t stride_token) {
    constexpr int warps = 8;
    __shared__ int route_rows[n_expert_used];
    __shared__ int route_experts[n_expert_used];

    const int token = blockIdx.x;
    if (threadIdx.x < n_expert_used) {
        const int slot = threadIdx.x;
        const int route = token * n_expert_used + slot;
        route_rows[slot] = ids_src1[route];
        route_experts[slot] = ids[route];
    }
    __syncthreads();

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int half = lane / QK_NVFP4_SUB;
    const int half_lane = lane % QK_NVFP4_SUB;
    const unsigned mask = half == 0 ? 0x0000FFFFu : 0xFFFF0000u;
    const int scale_blocks = n_cols_padded / QK_NVFP4_SUB;
    const int scale_block_pairs = (scale_blocks + 1) / 2;

    for (int pair = warp; pair < scale_block_pairs; pair += warps) {
        const int scale_block = 2 * pair + half;
        if (scale_block >= scale_blocks) {
            continue;
        }

        const int64_t k = (int64_t) scale_block * QK_NVFP4_SUB + half_lane;
        const float value = k < n_cols ? src[(int64_t) token * stride_token + k] : 0.0f;
        float amax = fabsf(value);
#pragma unroll
        for (int offset = QK_NVFP4_SUB / 2; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(mask, amax, offset, QK_NVFP4_SUB));
        }

        const uint8_t scale_code = ggml_cuda_fp32_to_ue4m3(amax / 6.0f);
        const float scale = ggml_cuda_ue4m3_to_fp32(scale_code);
        const float inv_scale = scale > 0.0f ? 0.5f / scale : 0.0f;
        const uint8_t quantized = ggml_cuda_float_to_fp4_e2m1(value, inv_scale);
        const uint8_t next = (uint8_t) __shfl_down_sync(mask, (unsigned) quantized, 1, QK_NVFP4_SUB);

        if ((half_lane & 1) == 0) {
            const uint8_t packed = quantized | (next << 4);
#pragma unroll
            for (int slot = 0; slot < n_expert_used; ++slot) {
                dst[(int64_t) route_rows[slot] * (n_cols_padded / 2) +
                    (int64_t) scale_block * (QK_NVFP4_SUB / 2) + half_lane / 2] = packed;
            }
        }
        if (half_lane == 0) {
#pragma unroll
            for (int slot = 0; slot < n_expert_used; ++slot) {
                *moe_cutlass_scale_ptr(scales, expert_bounds, route_experts[slot], route_rows[slot], scale_block,
                                       scale_blocks, false) = scale_code;
            }
        }
    }
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
    const int token = route / n_expert_used;
    const int slot = route - token * n_expert_used;
    staged_ids[route] = ids[(int64_t) token * ids_stride + slot];
    staged_weights[route] = weights[(int64_t) token * weights_token_stride +
                                    (int64_t) slot * weights_route_stride];
}

static __global__ void moe_cutlass_nvfp4_decode_w13_epilogue(
        const __nv_bfloat16 * __restrict__ gate_up,
        const int32_t * __restrict__ ids,
        const float * __restrict__ gate_scale,
        const float * __restrict__ up_scale,
        uint8_t * __restrict__ dst,
        uint8_t * __restrict__ scales,
        int n_ff) {
    const int row         = blockIdx.x;
    const int scale_block = blockIdx.y;
    const int lane        = threadIdx.x;
    const int k           = scale_block * QK_NVFP4_SUB + lane;
    const int expert      = ids[row];

    float value = 0.0f;
    if (k < n_ff) {
        const float gate = __fmul_rn(
            __bfloat162float(gate_up[(int64_t) row * 2 * n_ff + k]), gate_scale[expert]);
        const float up = __fmul_rn(
            __bfloat162float(gate_up[(int64_t) row * 2 * n_ff + n_ff + k]), up_scale[expert]);
        value = __fmul_rn(up, ggml_cuda_op_silu_single(gate));
    }

    float amax = fabsf(value);
#pragma unroll
    for (int mask = QK_NVFP4_SUB / 2; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFF, amax, mask, QK_NVFP4_SUB));
    }

    const uint8_t scale_code = ggml_cuda_fp32_to_ue4m3(amax / 6.0f);
    const float   scale      = ggml_cuda_ue4m3_to_fp32(scale_code);
    const float   inv_scale  = scale > 0.0f ? 0.5f / scale : 0.0f;
    const uint8_t quantized  = ggml_cuda_float_to_fp4_e2m1(value, inv_scale);
    const uint8_t next       = (uint8_t) __shfl_down_sync(0xFFFF, (unsigned) quantized, 1, QK_NVFP4_SUB);
    if ((lane & 1) == 0) {
        dst[(int64_t) row * (n_ff / 2) + (int64_t) scale_block * (QK_NVFP4_SUB / 2) + lane / 2] =
            quantized | (next << 4);
    }
    if (lane == 0) {
        const int scale_blocks = n_ff / QK_NVFP4_SUB;
        scales[(int64_t) row * 128 * scale_blocks +
               moe_cutlass_scale_offset(0, scale_block, scale_blocks)] = scale_code;
    }
}

static __global__ void moe_cutlass_nvfp4_w13_epilogue(
        const __nv_bfloat16 * __restrict__ gate_up,
        const int32_t * __restrict__ row_expert,
        const int32_t * __restrict__ expert_bounds,
        const float * __restrict__ gate_scale,
        const float * __restrict__ up_scale,
        uint8_t * __restrict__ dst,
        uint8_t * __restrict__ scales,
        int n_ff,
        int n_rows) {
    constexpr int warps = 8;
    const int row = blockIdx.x;
    if (row >= n_rows) {
        return;
    }
    const int expert = row_expert[row];
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int half = lane / QK_NVFP4_SUB;
    const int half_lane = lane % QK_NVFP4_SUB;
    const unsigned mask = half == 0 ? 0x0000FFFFu : 0xFFFF0000u;
    const int scale_blocks = n_ff / QK_NVFP4_SUB;
    const int scale_block_pairs = (scale_blocks + 1) / 2;

    for (int pair = warp; pair < scale_block_pairs; pair += warps) {
        const int scale_block = 2 * pair + half;
        if (scale_block >= scale_blocks) {
            continue;
        }

        const int k = scale_block * QK_NVFP4_SUB + half_lane;
        const float gate = __fmul_rn(
            __bfloat162float(gate_up[(int64_t) row * 2 * n_ff + k]), gate_scale[expert]);
        const float up = __fmul_rn(
            __bfloat162float(gate_up[(int64_t) row * 2 * n_ff + n_ff + k]), up_scale[expert]);
        const float value = __fmul_rn(up, ggml_cuda_op_silu_single(gate));

        float amax = fabsf(value);
#pragma unroll
        for (int offset = QK_NVFP4_SUB / 2; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(mask, amax, offset, QK_NVFP4_SUB));
        }

        const uint8_t scale_code = ggml_cuda_fp32_to_ue4m3(amax / 6.0f);
        const float scale = ggml_cuda_ue4m3_to_fp32(scale_code);
        const float inv_scale = scale > 0.0f ? 0.5f / scale : 0.0f;
        const uint8_t quantized = ggml_cuda_float_to_fp4_e2m1(value, inv_scale);
        const uint8_t next = (uint8_t) __shfl_down_sync(mask, (unsigned) quantized, 1, QK_NVFP4_SUB);
        if ((half_lane & 1) == 0) {
            dst[(int64_t) row * (n_ff / 2) +
                (int64_t) scale_block * (QK_NVFP4_SUB / 2) + half_lane / 2] = quantized | (next << 4);
        }
        if (half_lane == 0) {
            *moe_cutlass_scale_ptr(
                scales, expert_bounds, expert, row, scale_block, scale_blocks, false) = scale_code;
        }
    }
}

static __global__ void moe_cutlass_nvfp4_decode_w2_finalize(
        const __nv_bfloat16 * __restrict__ down,
        const int32_t * __restrict__ ids,
        const float * __restrict__ down_scale,
        const float * __restrict__ weights,
        float * __restrict__ dst,
        int n_embd,
        int n_expert_used,
        int weights_stride) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_embd) {
        return;
    }

    float result = 0.0f;
    for (int slot = 0; slot < n_expert_used; ++slot) {
        const int expert = ids[slot];
        float value = __bfloat162float(down[(int64_t) slot * n_embd + col]);
        value = __fmul_rn(value, down_scale[expert]);
        value = __fmul_rn(value, weights[(int64_t) slot * weights_stride]);
        result = __fadd_rn(result, value);
    }
    dst[col] = result;
}

static __global__ void moe_cutlass_nvfp4_w2_finalize(
        const __nv_bfloat16 * __restrict__ down,
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

    const int token = index / n_embd;
    const int col = index - (int64_t) token * n_embd;
    float result = 0.0f;
    for (int slot = 0; slot < n_expert_used; ++slot) {
        const int route = token * n_expert_used + slot;
        const int expert = ids[route];
        const int row = ids_src1[route];
        float value = __bfloat162float(down[(int64_t) row * n_embd + col]);
        value = __fmul_rn(value, down_scale[expert]);
        value = __fmul_rn(value, weights[route]);
        result = __fadd_rn(result, value);
    }
    dst[index] = result;
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

static __global__ void moe_cutlass_scatter_f32(const float * __restrict__ src,
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
    dst[(int64_t) ids_dst[row] * n_cols + col] = src[index];
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

    const uint8_t       scale     = moe_cutlass_mxfp8_scale(amax);
    const float         inv_scale = amax == 0.0f ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(scale));
    const __nv_fp8_e4m3 quantized(value * inv_scale);
    dst[row * n_ff_padded + k] = quantized.__x;
    if (lane == 0) {
        const int padded_k_blocks                                                            = n_ff_padded / WARP_SIZE;
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

size_t ggml_cuda_moe_cutlass_activation_size(int64_t n_rows, int64_t n_cols) {
    return (size_t) n_rows * n_cols;
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
    const int64_t padded_rows = GGML_PAD(n_rows + (int64_t) n_experts * 127, 128);
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
    if (n_cols <= 0 || n_cols % 2 != 0 || n_cols_padded < n_cols || n_cols_padded % 128 != 0 ||
        stride_token % 2 != 0 || n_tokens <= 0 || n_tokens > UINT_MAX || n_expert_used != 4) {
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
                                           bool            route_groups,
                                           cudaStream_t    stream) {
    GGML_ASSERT(n_cols_padded >= n_cols && n_cols_padded % 128 == 0);
    GGML_ASSERT(route_groups || expert_bounds != nullptr);
    const int64_t k_blocks = n_cols_padded / WARP_SIZE;
    CUDA_CHECK(cudaMemsetAsync(
        scales, 0, ggml_cuda_moe_cutlass_scale_size(n_tokens * n_expert_used, n_experts, n_cols_padded, route_groups),
        stream));
    moe_cutlass_quantize_routes<<<dim3(n_tokens, k_blocks, n_expert_used), WARP_SIZE, 0, stream>>>(
        src, ids, ids_src1, expert_bounds, dst, scales, n_cols, n_cols_padded, n_expert_used, ids_stride, route_groups);
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
                                        bool            route_groups,
                                        cudaStream_t    stream) {
    GGML_ASSERT(n_ff_padded >= n_ff && n_ff_padded % 128 == 0);
    GGML_ASSERT(route_groups || expert_bounds != nullptr);
    const int64_t k_blocks = n_ff_padded / WARP_SIZE;
    CUDA_CHECK(cudaMemsetAsync(
        scales, 0, ggml_cuda_moe_cutlass_scale_size(n_rows, n_experts, n_ff_padded, route_groups), stream));
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
    CUDA_CHECK(cudaMemsetAsync(
        scales, 0, ggml_cuda_moe_cutlass_scale_size(n_rows, n_experts, n_ff_padded, route_groups), stream));
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

namespace ggml_moe_cutlass_sm120 {

using namespace cute;

using Activation = cutlass::float_e4m3_t;
using Weight     = cutlass::float_e2m1_t;
using DefaultOutput = cutlass::bfloat16_t;

template <int TileN, bool SwapAB> struct mxfp_kernel_traits {
    static constexpr bool swap_ab = SwapAB;
    static constexpr int  tile_n  = TileN;
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
    static constexpr bool swap_ab = SwapAB;
    static constexpr int  tile_n  = TileN;
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
                                      const int32_t *          expert_bounds,
                                      const int32_t *          row_expert,
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

    const int expert    = route_groups ? row_expert[group] : group;
    const int row_begin = route_groups ? group : expert_bounds[expert];
    const int m         = route_groups ? 1 : expert_bounds[expert + 1] - row_begin;
    const int     padded_n               = GGML_PAD(n, 128);
    const int     padded_k               = GGML_PAD(k, 128);
    const int     padded_k_blocks        = padded_k / Traits::scale_granularity;
    const int64_t activation_scale_begin = route_groups ? (int64_t) group * 128 * padded_k_blocks :
        ((int64_t) row_begin + (int64_t) expert * 127) / 128 * 128 * padded_k_blocks;

    if constexpr (Traits::swap_ab) {
        metadata.shapes[group] = Problem(n, m, k);
        metadata.a[group] =
            reinterpret_cast<const typename Traits::Gemm::ElementA *>(weights + (int64_t) expert * n * k / 2);
        metadata.b[group] = reinterpret_cast<const typename Traits::Gemm::ElementB *>(
            activation + (int64_t) row_begin * k * Traits::activation_bits / 8);
        metadata.stride_a[group] = cutlass::make_cute_packed_stride(typename Traits::StrideA{}, make_shape(n, k, 1));
        metadata.stride_b[group] = cutlass::make_cute_packed_stride(typename Traits::StrideB{}, make_shape(m, k, 1));
        metadata.stride_d[group] = cutlass::make_cute_packed_stride(typename Traits::StrideD{}, make_shape(n, m, 1));
        metadata.scale_a[group] =
            reinterpret_cast<const typename Traits::Scale *>(weight_scales + (int64_t) expert * weight_scale_stride);
        metadata.scale_b[group] =
            reinterpret_cast<const typename Traits::Scale *>(activation_scales + activation_scale_begin);
        const auto shape                = make_shape(padded_n, m, padded_k, 1);
        metadata.layout_scale_a[group] = BlockScaleConfig::tile_atom_to_shape_SFA(shape);
        metadata.layout_scale_b[group] = BlockScaleConfig::tile_atom_to_shape_SFB(shape);
    } else {
        metadata.shapes[group] = Problem(m, n, k);
        metadata.a[group] = reinterpret_cast<const typename Traits::Gemm::ElementA *>(
            activation + (int64_t) row_begin * k * Traits::activation_bits / 8);
        metadata.b[group] =
            reinterpret_cast<const typename Traits::Gemm::ElementB *>(weights + (int64_t) expert * n * k / 2);
        metadata.stride_a[group] = cutlass::make_cute_packed_stride(typename Traits::StrideA{}, make_shape(m, k, 1));
        metadata.stride_b[group] = cutlass::make_cute_packed_stride(typename Traits::StrideB{}, make_shape(n, k, 1));
        metadata.stride_d[group] = cutlass::make_cute_packed_stride(typename Traits::StrideD{}, make_shape(m, n, 1));
        metadata.scale_a[group] =
            reinterpret_cast<const typename Traits::Scale *>(activation_scales + activation_scale_begin);
        metadata.scale_b[group] =
            reinterpret_cast<const typename Traits::Scale *>(weight_scales + (int64_t) expert * weight_scale_stride);
        const auto shape                = make_shape(m, padded_n, padded_k, 1);
        metadata.layout_scale_a[group] = BlockScaleConfig::tile_atom_to_shape_SFA(shape);
        metadata.layout_scale_b[group] = BlockScaleConfig::tile_atom_to_shape_SFB(shape);
    }
    metadata.d[group] = reinterpret_cast<typename Traits::Output *>(dst) + (int64_t) row_begin * n;
}

template <typename Traits>
static __global__ void setup_routed_metadata(grouped_metadata<Traits> metadata,
                                              const uint8_t *          activation,
                                              const uint8_t *          activation_scales,
                                              const char *             weights,
                                              const uint8_t *          weight_scales,
                                              const int32_t *          ids,
                                              void *                   dst,
                                              int                      n,
                                              int                      k,
                                              int                      weight_scale_stride,
                                              int                      groups) {
    using Problem          = typename Traits::ProblemShape::UnderlyingProblemShape;
    using BlockScaleConfig = typename Traits::BlockScaleConfig;

    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups) {
        return;
    }

    const int expert                  = ids[group];
    const int padded_n                = GGML_PAD(n, 128);
    const int padded_k                = GGML_PAD(k, 128);
    const int padded_k_blocks         = padded_k / Traits::scale_granularity;
    const int64_t activation_scale_begin = (int64_t) group * 128 * padded_k_blocks;

    if constexpr (Traits::swap_ab) {
        metadata.shapes[group] = Problem(n, 1, k);
        metadata.a[group] =
            reinterpret_cast<const typename Traits::Gemm::ElementA *>(weights + (int64_t) expert * n * k / 2);
        metadata.b[group] = reinterpret_cast<const typename Traits::Gemm::ElementB *>(
            activation + (int64_t) group * k * Traits::activation_bits / 8);
        metadata.stride_a[group] = cutlass::make_cute_packed_stride(typename Traits::StrideA{}, make_shape(n, k, 1));
        metadata.stride_b[group] = cutlass::make_cute_packed_stride(typename Traits::StrideB{}, make_shape(1, k, 1));
        metadata.stride_d[group] = cutlass::make_cute_packed_stride(typename Traits::StrideD{}, make_shape(n, 1, 1));
        metadata.scale_a[group] =
            reinterpret_cast<const typename Traits::Scale *>(weight_scales + (int64_t) expert * weight_scale_stride);
        metadata.scale_b[group] =
            reinterpret_cast<const typename Traits::Scale *>(activation_scales + activation_scale_begin);
        const auto shape = make_shape(padded_n, 1, padded_k, 1);
        metadata.layout_scale_a[group] = BlockScaleConfig::tile_atom_to_shape_SFA(shape);
        metadata.layout_scale_b[group] = BlockScaleConfig::tile_atom_to_shape_SFB(shape);
    } else {
        metadata.shapes[group] = Problem(1, n, k);
        metadata.a[group] = reinterpret_cast<const typename Traits::Gemm::ElementA *>(
            activation + (int64_t) group * k * Traits::activation_bits / 8);
        metadata.b[group] =
            reinterpret_cast<const typename Traits::Gemm::ElementB *>(weights + (int64_t) expert * n * k / 2);
        metadata.stride_a[group] = cutlass::make_cute_packed_stride(typename Traits::StrideA{}, make_shape(1, k, 1));
        metadata.stride_b[group] = cutlass::make_cute_packed_stride(typename Traits::StrideB{}, make_shape(n, k, 1));
        metadata.stride_d[group] = cutlass::make_cute_packed_stride(typename Traits::StrideD{}, make_shape(1, n, 1));
        metadata.scale_a[group] =
            reinterpret_cast<const typename Traits::Scale *>(activation_scales + activation_scale_begin);
        metadata.scale_b[group] =
            reinterpret_cast<const typename Traits::Scale *>(weight_scales + (int64_t) expert * weight_scale_stride);
        const auto shape = make_shape(1, padded_n, padded_k, 1);
        metadata.layout_scale_a[group] = BlockScaleConfig::tile_atom_to_shape_SFA(shape);
        metadata.layout_scale_b[group] = BlockScaleConfig::tile_atom_to_shape_SFB(shape);
    }
    metadata.d[group] = reinterpret_cast<typename Traits::Output *>(dst) + (int64_t) group * n;
}

template <typename Traits>
static bool run_routed_gemm(ggml_backend_cuda_context &       ctx,
                            const ggml_cuda_moe_weight_view & weight,
                            const uint8_t *                   activation,
                            const uint8_t *                   activation_scales,
                            const int32_t *                   ids,
                            void *                            dst,
                            int                               groups,
                            int                               n,
                            int                               k,
                            int                               sm_count,
                            cudaStream_t                      stream) {
    using Gemm = typename Traits::Gemm;

    size_t metadata_size = 0;
    make_metadata<Traits>(nullptr, groups, metadata_size);
    ggml_cuda_pool_alloc<char> metadata_alloc(ctx.pool());
    char * metadata_data = metadata_alloc.alloc(metadata_size);
    grouped_metadata<Traits> metadata = make_metadata<Traits>(metadata_data, groups, metadata_size);

    constexpr int threads = 32;
    setup_routed_metadata<Traits><<<(groups + threads - 1) / threads, threads, 0, stream>>>(
        metadata, activation, activation_scales, weight.data, weight.scales, ids, dst, n, k, weight.scale_stride,
        groups);
    CUDA_CHECK(cudaGetLastError());

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

    Gemm gemm;
    const cutlass::Status can_implement = gemm.can_implement(arguments);
    if (can_implement != cutlass::Status::kSuccess) {
        GGML_ABORT("CUTLASS routed MoE can_implement failed: %s", cutlassGetStatusString(can_implement));
    }

    const size_t workspace_size = Gemm::get_workspace_size(arguments);
    ggml_cuda_pool_alloc<char> workspace_alloc(ctx.pool());
    void * workspace = workspace_size == 0 ? nullptr : workspace_alloc.alloc(workspace_size);
    const cutlass::Status initialize = gemm.initialize(arguments, workspace);
    if (initialize != cutlass::Status::kSuccess) {
        GGML_ABORT("CUTLASS routed MoE initialize failed: %s", cutlassGetStatusString(initialize));
    }
    const cutlass::Status run = gemm.run(stream, nullptr, false);
    if (run != cutlass::Status::kSuccess) {
        GGML_ABORT("CUTLASS routed MoE run failed: %s", cutlassGetStatusString(run));
    }
    return true;
}

template <typename Traits>
static bool run_grouped_gemm(ggml_backend_cuda_context &       ctx,
                             const ggml_cuda_moe_weight_view & weight,
                             const uint8_t *                   activation,
                             const uint8_t *                   activation_scales,
                             const int32_t *                   expert_bounds,
                             const int32_t *                   row_expert,
                             void *                            dst,
                             int                               n_experts,
                             int64_t                           n_rows,
                             int                               n,
                             int                               k,
                             int                               sm_count,
                             cudaStream_t                      stream,
                             bool                              pdl,
                             bool                              route_groups,
                             bool                              require) {
    using Gemm = typename Traits::Gemm;

    const int groups = route_groups ? (int) n_rows : n_experts;
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
                                  weight.scales, expert_bounds, row_expert, dst, n, k, weight.scale_stride, groups,
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
            GGML_ABORT("CUTLASS MoE can_implement failed for tile 128x%dx128 swap=%d: %s", Traits::tile_n,
                       Traits::swap_ab,
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
            GGML_ABORT("CUTLASS MoE initialize failed for tile 128x%dx128 swap=%d: %s", Traits::tile_n,
                       Traits::swap_ab,
                       cutlassGetStatusString(initialize));
        }
        return false;
    }

    const cutlass::Status run = gemm.run(stream, nullptr, pdl);
    if (run != cutlass::Status::kSuccess) {
        if (require) {
            GGML_ABORT("CUTLASS MoE run failed for tile 128x%dx128 swap=%d: %s", Traits::tile_n, Traits::swap_ab,
                       cutlassGetStatusString(run));
        }
        return false;
    }
    return true;
}

template <template <int, bool> class Traits>
static bool dispatch_grouped_gemm(ggml_backend_cuda_context &       ctx,
                                  const ggml_cuda_moe_weight_view & weight,
                                  const uint8_t *                   activation,
                                  const uint8_t *                   activation_scales,
                                  const int32_t *                   expert_bounds,
                                  const int32_t *                   row_expert,
                                  void *                            dst,
                                  int                               n_experts,
                                  int64_t                           n_rows,
                                  int                               n,
                                  int                               k,
                                  int                               sm_count,
                                  ggml_cuda_moe_cutlass_config      config,
                                  cudaStream_t                      stream,
                                  bool                              require) {
    if (config.tile_n == 32) {
        return config.swap_ab ?
                   run_grouped_gemm<Traits<32, true>>(ctx, weight, activation, activation_scales, expert_bounds,
                                                       row_expert, dst, n_experts, n_rows, n, k, sm_count, stream,
                                                       config.pdl, config.route_groups, require) :
                   run_grouped_gemm<Traits<32, false>>(ctx, weight, activation, activation_scales, expert_bounds,
                                                        row_expert, dst, n_experts, n_rows, n, k, sm_count, stream,
                                                        config.pdl, config.route_groups, require);
    }
    if (config.tile_n == 64) {
        return config.swap_ab ?
                   run_grouped_gemm<Traits<64, true>>(ctx, weight, activation, activation_scales, expert_bounds,
                                                       row_expert, dst, n_experts, n_rows, n, k, sm_count, stream,
                                                       config.pdl, config.route_groups, require) :
                   run_grouped_gemm<Traits<64, false>>(ctx, weight, activation, activation_scales, expert_bounds,
                                                        row_expert, dst, n_experts, n_rows, n, k, sm_count, stream,
                                                        config.pdl, config.route_groups, require);
    }
    return config.swap_ab ?
               run_grouped_gemm<Traits<128, true>>(ctx, weight, activation, activation_scales, expert_bounds,
                                                    row_expert, dst, n_experts, n_rows, n, k, sm_count, stream,
                                                    config.pdl, config.route_groups, require) :
               run_grouped_gemm<Traits<128, false>>(ctx, weight, activation, activation_scales, expert_bounds,
                                                     row_expert, dst, n_experts, n_rows, n, k, sm_count, stream,
                                                     config.pdl, config.route_groups, require);
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
                                const int32_t *                   row_expert,
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
    GGML_ASSERT(!config.route_groups || row_expert != nullptr);
    GGML_ASSERT(config.route_groups || expert_bounds != nullptr);

    if (weight.type == GGML_TYPE_MXFP4) {
        return dispatch_grouped_gemm<mxfp_kernel_traits>(ctx, weight, activation, activation_scales, expert_bounds,
                                                         row_expert, dst, n_experts, n_rows, (int) n, (int) k,
                                                         sm_count, config, stream, require);
    }
    if (weight.type == GGML_TYPE_NVFP4) {
        return dispatch_grouped_gemm<nvfp4_kernel_traits>(ctx, weight, activation, activation_scales, expert_bounds,
                                                          row_expert, dst, n_experts, n_rows, (int) n, (int) k,
                                                          sm_count, config, stream, require);
    }
    if (require) {
        GGML_ABORT("CUTLASS MoE does not support weight type %s", ggml_type_name(weight.type));
    }
    return false;
}

static bool moe_cutlass_nvfp4_decode(
        ggml_backend_cuda_context &              ctx,
        const ggml_cuda_moe_cutlass_nvfp4_args & args) {
    if (!ggml_cuda_moe_cutlass_decode_fused_requested()) {
        return false;
    }

    constexpr int n_experts     = 256;
    constexpr int n_expert_used = 8;
    constexpr int n_embd        = 2048;
    constexpr int n_ff          = 512;

    const bool valid_shape =
        args.gate->type == GGML_TYPE_NVFP4 && args.gate->ne[0] == n_embd && args.gate->ne[1] == n_ff &&
        args.gate->ne[2] == n_experts && args.gate->ne[3] == 1 &&
        args.up->type == GGML_TYPE_NVFP4 && ggml_are_same_shape(args.gate, args.up) &&
        args.down->type == GGML_TYPE_NVFP4 && args.down->ne[0] == n_ff && args.down->ne[1] == n_embd &&
        args.down->ne[2] == n_experts && args.down->ne[3] == 1 &&
        args.input->type == GGML_TYPE_F32 && args.input->ne[0] == n_embd && args.input->ne[1] == 1 &&
        args.input->ne[2] == 1 && args.input->ne[3] == 1 &&
        args.ids->type == GGML_TYPE_I32 && args.ids->ne[0] == n_expert_used && args.ids->ne[1] == 1 &&
        args.ids->ne[2] == 1 && args.ids->ne[3] == 1 &&
        args.gate_scale->type == GGML_TYPE_F32 && args.gate_scale->ne[0] == n_experts &&
        args.gate_scale->ne[1] == 1 && args.gate_scale->ne[2] == 1 && args.gate_scale->ne[3] == 1 &&
        args.up_scale->type == GGML_TYPE_F32 && args.up_scale->ne[0] == n_experts &&
        args.up_scale->ne[1] == 1 && args.up_scale->ne[2] == 1 && args.up_scale->ne[3] == 1 &&
        args.down_scale->type == GGML_TYPE_F32 && args.down_scale->ne[0] == n_experts &&
        args.down_scale->ne[1] == 1 && args.down_scale->ne[2] == 1 && args.down_scale->ne[3] == 1 &&
        args.weights->type == GGML_TYPE_F32 && args.weights->ne[0] == 1 &&
        args.weights->ne[1] == n_expert_used && args.weights->ne[2] == 1 && args.weights->ne[3] == 1 &&
        args.dst->type == GGML_TYPE_F32 && args.dst->ne[0] == n_embd && args.dst->ne[1] == 1 &&
        args.dst->ne[2] == 1 && args.dst->ne[3] == 1;
    if (!valid_shape) {
        return false;
    }

    const ggml_backend_buffer_type_t buffer_type = ggml_backend_cuda_buffer_type(ctx.device);
    const std::array<const ggml_tensor *, 10> tensors = {
        args.gate, args.up, args.down, args.input, args.ids, args.gate_scale, args.up_scale, args.down_scale,
        args.weights, args.dst,
    };
    const bool valid_buffers = std::all_of(tensors.begin(), tensors.end(), [buffer_type](const ggml_tensor * tensor) {
        return tensor->buffer != nullptr && ggml_backend_buffer_get_type(tensor->buffer) == buffer_type;
    });
    const bool valid_layout = ggml_is_contiguous(args.gate) && ggml_is_contiguous(args.up) &&
        ggml_is_contiguous(args.down) && ggml_is_contiguous(args.input) && args.ids->nb[0] == sizeof(int32_t) &&
        ggml_is_contiguous(args.gate_scale) && ggml_is_contiguous(args.up_scale) &&
        ggml_is_contiguous(args.down_scale) && ggml_is_contiguous(args.weights) && ggml_is_contiguous(args.dst);
    if (!valid_buffers || !valid_layout) {
        return false;
    }

    const auto & device_info = ggml_cuda_info().devices[ctx.device];
    if (!blackwell_mma_available(device_info.cc)) {
        return false;
    }
#ifdef USE_CUDA_GRAPH
    if (ctx.any_cuda_graph_enabled()) {
        GGML_ABORT("the fused CUTLASS NVFP4 decode path requires CUDA graphs to be disabled");
    }
#endif

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<int32_t> staged_ids(ctx.pool(), n_expert_used);
    ggml_cuda_pool_alloc<float> staged_weights(ctx.pool(), n_expert_used);
    CUDA_CHECK(cudaMemcpyAsync(
        staged_ids.get(), args.ids->data, n_expert_used * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        staged_weights.get(), args.weights->data, n_expert_used * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    const int32_t * ids = staged_ids.get();

    constexpr int w13_k            = n_embd;
    constexpr int w13_n            = 2 * n_ff;
    constexpr int w13_scale_blocks = w13_k / QK_NVFP4_SUB;
    ggml_cuda_pool_alloc<uint8_t> w13_activation(ctx.pool(), (size_t) n_expert_used * w13_k / 2);
    ggml_cuda_pool_alloc<uint8_t> w13_activation_scales(
        ctx.pool(), (size_t) n_expert_used * 128 * w13_scale_blocks);
    {
#ifdef GGML_CUDA_MOE_PROFILE
        const ggml_cuda_moe_profile_scope profile_scope("ffn_moe.cutlass_decode_a1_quant");
#endif
        CUDA_CHECK(cudaMemsetAsync(
            w13_activation_scales.get(), 0, (size_t) n_expert_used * 128 * w13_scale_blocks, stream));
        moe_cutlass_quantize_nvfp4_broadcast_routes<<<w13_scale_blocks, QK_NVFP4_SUB, 0, stream>>>(
            (const float *) args.input->data, w13_activation.get(), w13_activation_scales.get(), w13_k, w13_k,
            n_expert_used);
        CUDA_CHECK(cudaGetLastError());
    }

    ggml_cuda_moe_weight_view w13_weight;
    {
#ifdef GGML_CUDA_MOE_PROFILE
        const ggml_cuda_moe_profile_scope profile_scope("ffn_moe.cutlass_decode_w13_repack");
#endif
        if (!ggml_cuda_moe_repack_weight_pair(ctx, args.gate, args.up, w13_weight, stream)) {
            GGML_ABORT("the fused CUTLASS NVFP4 decode path could not repack W13");
        }
    }
    ggml_cuda_pool_alloc<__nv_bfloat16> w13_output(ctx.pool(), (size_t) n_expert_used * w13_n);
    {
#ifdef GGML_CUDA_MOE_PROFILE
        const ggml_cuda_moe_profile_scope profile_scope("ffn_moe.cutlass_decode_w13");
#endif
        ggml_cuda_moe_weight_wait_ready(w13_weight, stream);
        ggml_moe_cutlass_sm120::run_routed_gemm<
            ggml_moe_cutlass_sm120::nvfp4_kernel_traits<32, true>>(
                ctx, w13_weight, w13_activation.get(), w13_activation_scales.get(), ids, w13_output.get(),
                n_expert_used, w13_n, w13_k, device_info.nsm, stream);
        ggml_cuda_moe_weight_mark_used(w13_weight, stream);
    }

    constexpr int w2_k            = n_ff;
    constexpr int w2_n            = n_embd;
    constexpr int w2_scale_blocks = w2_k / QK_NVFP4_SUB;
    ggml_cuda_pool_alloc<uint8_t> w2_activation(ctx.pool(), (size_t) n_expert_used * w2_k / 2);
    ggml_cuda_pool_alloc<uint8_t> w2_activation_scales(
        ctx.pool(), (size_t) n_expert_used * 128 * w2_scale_blocks);
    {
#ifdef GGML_CUDA_MOE_PROFILE
        const ggml_cuda_moe_profile_scope profile_scope("ffn_moe.cutlass_decode_w13_epilogue");
#endif
        CUDA_CHECK(cudaMemsetAsync(
            w2_activation_scales.get(), 0, (size_t) n_expert_used * 128 * w2_scale_blocks, stream));
        moe_cutlass_nvfp4_decode_w13_epilogue<<<
            dim3(n_expert_used, w2_scale_blocks, 1), QK_NVFP4_SUB, 0, stream>>>(
                w13_output.get(), ids, (const float *) args.gate_scale->data, (const float *) args.up_scale->data,
                w2_activation.get(), w2_activation_scales.get(), n_ff);
        CUDA_CHECK(cudaGetLastError());
    }

    ggml_cuda_moe_weight_view w2_weight;
    {
#ifdef GGML_CUDA_MOE_PROFILE
        const ggml_cuda_moe_profile_scope profile_scope("ffn_moe.cutlass_decode_w2_repack");
#endif
        if (!ggml_cuda_moe_repack_weight(
                ctx, args.down, ggml_cuda_moe_weight_layout::cutlass, w2_weight, stream, 2, false)) {
            GGML_ABORT("the fused CUTLASS NVFP4 decode path could not repack W2");
        }
    }
    ggml_cuda_pool_alloc<__nv_bfloat16> w2_output(ctx.pool(), (size_t) n_expert_used * w2_n);
    {
#ifdef GGML_CUDA_MOE_PROFILE
        const ggml_cuda_moe_profile_scope profile_scope("ffn_moe.cutlass_decode_w2");
#endif
        ggml_cuda_moe_weight_wait_ready(w2_weight, stream);
        ggml_moe_cutlass_sm120::run_routed_gemm<
            ggml_moe_cutlass_sm120::nvfp4_kernel_traits<32, true>>(
                ctx, w2_weight, w2_activation.get(), w2_activation_scales.get(), ids, w2_output.get(),
                n_expert_used, w2_n, w2_k, device_info.nsm, stream);
        ggml_cuda_moe_weight_mark_used(w2_weight, stream);
    }

    constexpr int threads = 256;
    {
#ifdef GGML_CUDA_MOE_PROFILE
        const ggml_cuda_moe_profile_scope profile_scope("ffn_moe.cutlass_decode_w2_finalize");
#endif
        moe_cutlass_nvfp4_decode_w2_finalize<<<(n_embd + threads - 1) / threads, threads, 0, stream>>>(
            w2_output.get(), ids, (const float *) args.down_scale->data, staged_weights.get(),
            (float *) args.dst->data, n_embd, n_expert_used, 1);
        CUDA_CHECK(cudaGetLastError());
    }

    if (ggml_cuda_moe_cutlass_decode_log_requested()) {
        static std::atomic_flag logged = ATOMIC_FLAG_INIT;
        if (!logged.test_and_set(std::memory_order_relaxed)) {
            GGML_LOG_INFO("MoE CUTLASS NVFP4 fused decode dispatch: experts=256 top=8 hidden=2048 ffn=512\n");
        }
    }
    return true;
}

static bool moe_cutlass_nvfp4_prefill(
        ggml_backend_cuda_context &              ctx,
        const ggml_cuda_moe_cutlass_nvfp4_args & args) {
    if (!ggml_cuda_moe_cutlass_prefill_requested()) {
        return false;
    }

    constexpr int n_experts = 256;
    constexpr int n_expert_used = 8;
    constexpr int n_embd = 2048;
    constexpr int n_ff = 512;
    const int64_t n_tokens = args.ids->ne[1];
    if (n_tokens < 256 || n_tokens > INT_MAX / n_expert_used) {
        return false;
    }
    const int n_rows = (int) n_tokens * n_expert_used;

    const bool valid_shape =
        args.gate->type == GGML_TYPE_NVFP4 && args.gate->ne[0] == n_embd && args.gate->ne[1] == n_ff &&
        args.gate->ne[2] == n_experts && args.gate->ne[3] == 1 &&
        args.up->type == GGML_TYPE_NVFP4 && ggml_are_same_shape(args.gate, args.up) &&
        args.down->type == GGML_TYPE_NVFP4 && args.down->ne[0] == n_ff && args.down->ne[1] == n_embd &&
        args.down->ne[2] == n_experts && args.down->ne[3] == 1 &&
        args.input->type == GGML_TYPE_F32 && args.input->ne[0] == n_embd && args.input->ne[1] == 1 &&
        args.input->ne[2] == n_tokens && args.input->ne[3] == 1 &&
        args.ids->type == GGML_TYPE_I32 && args.ids->ne[0] == n_expert_used && args.ids->ne[1] == n_tokens &&
        args.ids->ne[2] == 1 && args.ids->ne[3] == 1 &&
        args.gate_scale->type == GGML_TYPE_F32 && args.gate_scale->ne[0] == n_experts &&
        args.gate_scale->ne[1] == 1 && args.gate_scale->ne[2] == 1 && args.gate_scale->ne[3] == 1 &&
        args.up_scale->type == GGML_TYPE_F32 && args.up_scale->ne[0] == n_experts &&
        args.up_scale->ne[1] == 1 && args.up_scale->ne[2] == 1 && args.up_scale->ne[3] == 1 &&
        args.down_scale->type == GGML_TYPE_F32 && args.down_scale->ne[0] == n_experts &&
        args.down_scale->ne[1] == 1 && args.down_scale->ne[2] == 1 && args.down_scale->ne[3] == 1 &&
        args.weights->type == GGML_TYPE_F32 && args.weights->ne[0] == 1 &&
        args.weights->ne[1] == n_expert_used && args.weights->ne[2] == n_tokens && args.weights->ne[3] == 1 &&
        args.dst->type == GGML_TYPE_F32 && args.dst->ne[0] == n_embd && args.dst->ne[1] == n_tokens &&
        args.dst->ne[2] == 1 && args.dst->ne[3] == 1;
    if (!valid_shape) {
        return false;
    }

    const ggml_backend_buffer_type_t buffer_type = ggml_backend_cuda_buffer_type(ctx.device);
    const std::array<const ggml_tensor *, 10> tensors = {
        args.gate, args.up, args.down, args.input, args.ids, args.gate_scale, args.up_scale, args.down_scale,
        args.weights, args.dst,
    };
    const bool valid_buffers = std::all_of(tensors.begin(), tensors.end(), [buffer_type](const ggml_tensor * tensor) {
        return tensor->buffer != nullptr && ggml_backend_buffer_get_type(tensor->buffer) == buffer_type;
    });
    const bool valid_layout = ggml_is_contiguous(args.gate) && ggml_is_contiguous(args.up) &&
        ggml_is_contiguous(args.down) && ggml_is_contiguous(args.input) &&
        ggml_is_contiguous_rows(args.ids) && args.ids->nb[0] == sizeof(int32_t) &&
        args.ids->nb[1] >= ggml_row_size(args.ids->type, args.ids->ne[0]) &&
        ggml_is_contiguous(args.gate_scale) && ggml_is_contiguous(args.up_scale) &&
        ggml_is_contiguous(args.down_scale) && ggml_is_contiguous(args.weights) && ggml_is_contiguous(args.dst);
    if (!valid_buffers || !valid_layout) {
        return false;
    }

    const auto & device_info = ggml_cuda_info().devices[ctx.device];
    if (!blackwell_mma_available(device_info.cc)) {
        return false;
    }

    cudaStream_t stream = ctx.stream();
    ggml_cuda_moe_weight_view w13_weight;
    ggml_cuda_moe_weight_view w2_weight;
    {
#ifdef GGML_CUDA_MOE_PROFILE
        const ggml_cuda_moe_profile_scope profile_scope("ffn_moe.cutlass_nvfp4_weight_repack");
#endif
        if (!ggml_cuda_moe_repack_weight_pair(ctx, args.gate, args.up, w13_weight, stream) ||
            !ggml_cuda_moe_repack_weight(
                ctx, args.down, ggml_cuda_moe_weight_layout::cutlass, w2_weight, stream, 2, false, true)) {
            return false;
        }
    }

    ggml_cuda_pool_alloc<int32_t> staged_ids(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<float> staged_weights(ctx.pool(), n_rows);
    constexpr int stage_threads = 256;
    const int ids_stride = args.ids->nb[1] / sizeof(int32_t);
    const int weights_route_stride = args.weights->nb[1] / sizeof(float);
    const int weights_token_stride = args.weights->nb[2] / sizeof(float);
    const int n_blocks = ggml_cuda_mm_ids_prefix_block_count((int) n_tokens, n_expert_used);
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), n_experts + 1);
    ggml_cuda_pool_alloc<int32_t> row_expert(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> block_counts(ctx.pool(), (size_t) n_blocks * n_experts);
    ggml_cuda_pool_alloc<int32_t> block_offsets(ctx.pool(), (size_t) n_blocks * n_experts);
    {
#ifdef GGML_CUDA_MOE_PROFILE
        const ggml_cuda_moe_profile_scope profile_scope("ffn_moe.cutlass_nvfp4_schedule");
#endif
        moe_cutlass_stage_routes<<<(n_rows + stage_threads - 1) / stage_threads, stage_threads, 0, stream>>>(
            (const int32_t *) args.ids->data, (const float *) args.weights->data, staged_ids.get(),
            staged_weights.get(), n_rows, n_expert_used, ids_stride, weights_route_stride, weights_token_stride);
        CUDA_CHECK(cudaGetLastError());
        if (!ggml_cuda_launch_mm_ids_prefix(
                staged_ids.get(), ids_src1.get(), ids_dst.get(), expert_bounds.get(), row_expert.get(),
                block_counts.get(), block_offsets.get(), n_experts, (int) n_tokens, n_expert_used, n_expert_used,
                stream)) {
            return false;
        }
    }

    constexpr int w13_k = n_embd;
    constexpr int w13_n = 2 * n_ff;
    const size_t w13_activation_size = (size_t) n_rows * w13_k / 2;
    const size_t w13_scale_size = moe_cutlass_nvfp4_scale_size(n_rows, n_experts, w13_k);
    ggml_cuda_pool_alloc<uint8_t> w13_activation(ctx.pool(), w13_activation_size);
    ggml_cuda_pool_alloc<uint8_t> w13_activation_scales(ctx.pool(), w13_scale_size);
    {
#ifdef GGML_CUDA_MOE_PROFILE
        const ggml_cuda_moe_profile_scope profile_scope("ffn_moe.cutlass_nvfp4_a1_quant");
#endif
        CUDA_CHECK(cudaMemsetAsync(w13_activation_scales.get(), 0, w13_scale_size, stream));
        moe_cutlass_quantize_nvfp4_broadcast_cta<n_expert_used><<<(unsigned) n_tokens, 256, 0, stream>>>(
            (const float *) args.input->data, staged_ids.get(), ids_src1.get(), expert_bounds.get(),
            w13_activation.get(), w13_activation_scales.get(), n_embd, w13_k,
            args.input->nb[2] / sizeof(float));
        CUDA_CHECK(cudaGetLastError());
    }

    const ggml_cuda_moe_cutlass_config w13_config = {
        moe_cutlass_nvfp4_tile_n("GGML_CUDA_MOE_MMQ_CUTLASS_W13_TILE_N", n_rows, n_experts),
        moe_cutlass_swap_requested("GGML_CUDA_MOE_MMQ_CUTLASS_W13_SWAP_AB"),
        moe_cutlass_pdl_requested(),
        false,
    };
    ggml_cuda_pool_alloc<__nv_bfloat16> w13_output(ctx.pool(), (size_t) n_rows * w13_n);
    {
#ifdef GGML_CUDA_MOE_PROFILE
        const ggml_cuda_moe_profile_scope profile_scope("ffn_moe.cutlass_nvfp4_w13");
#endif
        ggml_cuda_moe_weight_wait_ready(w13_weight, stream);
        if (!ggml_cuda_moe_cutlass_gemm(
                ctx, w13_weight, w13_activation.get(), w13_activation_scales.get(), expert_bounds.get(),
                row_expert.get(), w13_output.get(), n_experts, n_rows, w13_n, w13_k, device_info.nsm, w13_config,
                stream, false)) {
            return false;
        }
        ggml_cuda_moe_weight_mark_used(w13_weight, stream);
    }

    constexpr int w2_k = n_ff;
    constexpr int w2_n = n_embd;
    const size_t w2_activation_size = (size_t) n_rows * w2_k / 2;
    const size_t w2_scale_size = moe_cutlass_nvfp4_scale_size(n_rows, n_experts, w2_k);
    ggml_cuda_pool_alloc<uint8_t> w2_activation(ctx.pool(), w2_activation_size);
    ggml_cuda_pool_alloc<uint8_t> w2_activation_scales(ctx.pool(), w2_scale_size);
    {
#ifdef GGML_CUDA_MOE_PROFILE
        const ggml_cuda_moe_profile_scope profile_scope("ffn_moe.cutlass_nvfp4_w13_epilogue");
#endif
        CUDA_CHECK(cudaMemsetAsync(w2_activation_scales.get(), 0, w2_scale_size, stream));
        moe_cutlass_nvfp4_w13_epilogue<<<n_rows, 256, 0, stream>>>(
            w13_output.get(), row_expert.get(), expert_bounds.get(), (const float *) args.gate_scale->data,
            (const float *) args.up_scale->data, w2_activation.get(), w2_activation_scales.get(), n_ff, n_rows);
        CUDA_CHECK(cudaGetLastError());
    }

    const ggml_cuda_moe_cutlass_config w2_config = {
        moe_cutlass_nvfp4_tile_n("GGML_CUDA_MOE_MMQ_CUTLASS_W2_TILE_N", n_rows, n_experts),
        moe_cutlass_swap_requested("GGML_CUDA_MOE_MMQ_CUTLASS_W2_SWAP_AB"),
        moe_cutlass_pdl_requested(),
        false,
    };
    ggml_cuda_pool_alloc<__nv_bfloat16> w2_output(ctx.pool(), (size_t) n_rows * w2_n);
    {
#ifdef GGML_CUDA_MOE_PROFILE
        const ggml_cuda_moe_profile_scope profile_scope("ffn_moe.cutlass_nvfp4_w2");
#endif
        ggml_cuda_moe_weight_wait_ready(w2_weight, stream);
        if (!ggml_cuda_moe_cutlass_gemm(
                ctx, w2_weight, w2_activation.get(), w2_activation_scales.get(), expert_bounds.get(),
                row_expert.get(), w2_output.get(), n_experts, n_rows, w2_n, w2_k, device_info.nsm, w2_config,
                stream, false)) {
            return false;
        }
        ggml_cuda_moe_weight_mark_used(w2_weight, stream);
    }

    constexpr int finalize_threads = 256;
    const int64_t output_size = n_tokens * n_embd;
    {
#ifdef GGML_CUDA_MOE_PROFILE
        const ggml_cuda_moe_profile_scope profile_scope("ffn_moe.cutlass_nvfp4_w2_epilogue");
#endif
        moe_cutlass_nvfp4_w2_finalize<<<
            (output_size + finalize_threads - 1) / finalize_threads, finalize_threads, 0, stream>>>(
                w2_output.get(), staged_ids.get(), ids_src1.get(), (const float *) args.down_scale->data,
                staged_weights.get(), (float *) args.dst->data, n_embd, (int) n_tokens, n_expert_used);
        CUDA_CHECK(cudaGetLastError());
    }

    if (moe_cutlass_nvfp4_prefill_log_requested()) {
        static std::atomic_flag logged = ATOMIC_FLAG_INIT;
        if (!logged.test_and_set(std::memory_order_relaxed)) {
            GGML_LOG_INFO(
                "MoE CUTLASS NVFP4 prefill dispatch: tokens=%lld experts=256 top=8 hidden=2048 ffn=512 "
                "w13-tile=%d w2-tile=%d\n",
                (long long) n_tokens, w13_config.tile_n, w2_config.tile_n);
        }
    }
    return true;
}

bool ggml_cuda_moe_cutlass_nvfp4(
        ggml_backend_cuda_context &              ctx,
        const ggml_cuda_moe_cutlass_nvfp4_args & args) {
    if (args.ids->ne[1] == 1) {
        return moe_cutlass_nvfp4_decode(ctx, args);
    }
    return moe_cutlass_nvfp4_prefill(ctx, args);
}

bool ggml_cuda_moe_cutlass_decode_mul_mat_id(ggml_backend_cuda_context & ctx,
                                             const ggml_tensor *         src0,
                                             const ggml_tensor *         src1,
                                             const ggml_tensor *         ids,
                                             ggml_tensor *               dst) {
    if (ggml_cuda_moe_cutlass_decode_fused_requested()) {
        const bool fused_candidate = src0->type == GGML_TYPE_NVFP4 && src0->ne[2] == 256 &&
            ids->type == GGML_TYPE_I32 && ids->ne[0] == 8 && ids->ne[1] == 1 &&
            src1->type == GGML_TYPE_F32 && src1->ne[2] == 1 &&
            ((src0->ne[0] == 2048 && src0->ne[1] == 512) ||
             (src0->ne[0] == 512 && src0->ne[1] == 2048));
        if (fused_candidate) {
            GGML_ABORT("a target NVFP4 decode graph did not match the fused CUTLASS path");
        }
        return false;
    }
    if (src0->type != GGML_TYPE_NVFP4) {
        static std::atomic_flag logged = ATOMIC_FLAG_INIT;
        if (ggml_cuda_moe_cutlass_decode_log_requested() && ids->type == GGML_TYPE_I32 && src0->ne[2] > 1 &&
            !logged.test_and_set(std::memory_order_relaxed)) {
            GGML_LOG_INFO(
                "MoE CUTLASS NVFP4 decode reject: weight type=%s shape=[%lld,%lld,%lld,%lld] "
                "ids=[%lld,%lld,%lld,%lld]\n",
                ggml_type_name(src0->type), (long long) src0->ne[0], (long long) src0->ne[1],
                (long long) src0->ne[2], (long long) src0->ne[3], (long long) ids->ne[0],
                (long long) ids->ne[1], (long long) ids->ne[2], (long long) ids->ne[3]);
        }
        return false;
    }

    const auto reject = [&ctx, src0]() {
        if (ggml_cuda_moe_weight_is_inplace_repacked(ctx, src0)) {
            GGML_ABORT("a CUTLASS NVFP4 weight cannot fall back to the canonical CUDA path");
        }
        return false;
    };

    constexpr int n_experts     = 256;
    constexpr int n_expert_used = 8;
    const int64_t n_tokens      = ids->ne[1];
    const int64_t n_rows        = n_tokens * n_expert_used;
    const int64_t k             = src0->ne[0];
    const int64_t n             = src0->ne[1];
    const bool valid_expert_shape = (k == 2048 && (n == 512 || n == 1024)) || (k == 512 && n == 2048);
    const bool valid_shape = valid_expert_shape && src0->ne[2] == n_experts && src0->ne[3] == 1 &&
                             ids->type == GGML_TYPE_I32 &&
                             ids->ne[0] == n_expert_used && ids->ne[2] == 1 && ids->ne[3] == 1 &&
                             n_tokens >= 1 && n_tokens <= 8 && src1->type == GGML_TYPE_F32 && src1->ne[0] == k &&
                             (src1->ne[1] == 1 || src1->ne[1] == n_expert_used) && src1->ne[2] == n_tokens &&
                             src1->ne[3] == 1 && dst->type == GGML_TYPE_F32 && dst->ne[0] == n &&
                             dst->ne[1] == n_expert_used && dst->ne[2] == n_tokens && dst->ne[3] == 1;
    if (!valid_shape) {
        static std::atomic_flag logged = ATOMIC_FLAG_INIT;
        if (ggml_cuda_moe_cutlass_decode_log_requested() &&
            !logged.test_and_set(std::memory_order_relaxed)) {
            GGML_LOG_INFO(
                "MoE CUTLASS NVFP4 decode reject: weight=[%lld,%lld,%lld,%lld] "
                "input=%s[%lld,%lld,%lld,%lld] ids=%s[%lld,%lld,%lld,%lld] "
                "output=%s[%lld,%lld,%lld,%lld]\n",
                (long long) src0->ne[0], (long long) src0->ne[1], (long long) src0->ne[2],
                (long long) src0->ne[3], ggml_type_name(src1->type), (long long) src1->ne[0],
                (long long) src1->ne[1], (long long) src1->ne[2], (long long) src1->ne[3],
                ggml_type_name(ids->type), (long long) ids->ne[0], (long long) ids->ne[1],
                (long long) ids->ne[2], (long long) ids->ne[3], ggml_type_name(dst->type),
                (long long) dst->ne[0], (long long) dst->ne[1], (long long) dst->ne[2],
                (long long) dst->ne[3]);
        }
        return reject();
    }

    const ggml_backend_buffer_type_t buffer_type = ggml_backend_cuda_buffer_type(ctx.device);
    const std::array<const ggml_tensor *, 4> tensors = { src0, src1, ids, dst };
    const bool valid_buffers = std::all_of(tensors.begin(), tensors.end(), [buffer_type](const ggml_tensor * tensor) {
        return tensor->buffer != nullptr && ggml_backend_buffer_get_type(tensor->buffer) == buffer_type;
    });
    const bool valid_layout = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
                              ggml_is_contiguous_rows(ids) &&
                              ids->nb[1] >= ggml_row_size(ids->type, ids->ne[0]) && ggml_is_contiguous(dst);
    if (!valid_buffers || !valid_layout) {
        static std::atomic_flag logged = ATOMIC_FLAG_INIT;
        if (ggml_cuda_moe_cutlass_decode_log_requested() &&
            !logged.test_and_set(std::memory_order_relaxed)) {
            GGML_LOG_INFO(
                "MoE CUTLASS NVFP4 decode reject: buffers=[%s,%s,%s,%s] expected=%s "
                "contiguous=[%d,%d,%d,%d] ids-stride=%zu ids-row=%zu\n",
                src0->buffer ? ggml_backend_buffer_name(src0->buffer) : "null",
                src1->buffer ? ggml_backend_buffer_name(src1->buffer) : "null",
                ids->buffer ? ggml_backend_buffer_name(ids->buffer) : "null",
                dst->buffer ? ggml_backend_buffer_name(dst->buffer) : "null",
                ggml_backend_buft_name(buffer_type), ggml_is_contiguous(src0), ggml_is_contiguous(src1),
                ggml_is_contiguous_rows(ids), ggml_is_contiguous(dst), ids->nb[1],
                ggml_row_size(ids->type, ids->ne[0]));
        }
        return reject();
    }

    const auto & device_info = ggml_cuda_info().devices[ctx.device];
    if (!blackwell_mma_available(device_info.cc)) {
        GGML_ABORT("the CUTLASS NVFP4 decode path requires an SM120-family CUDA device");
    }
#ifdef USE_CUDA_GRAPH
    if (ctx.any_cuda_graph_enabled()) {
        GGML_ABORT("the CUTLASS NVFP4 decode path requires CUDA graphs to be disabled");
    }
#endif

    cudaStream_t stream = ctx.stream();
    const int    ids_stride = ids->nb[1] / ggml_element_size(ids);
    const int    n_blocks   = ggml_cuda_mm_ids_prefix_block_count(n_tokens, n_expert_used);
    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), n_experts + 1);
    ggml_cuda_pool_alloc<int32_t> row_expert(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<int32_t> block_counts(ctx.pool(), (size_t) n_blocks * n_experts);
    ggml_cuda_pool_alloc<int32_t> block_offsets(ctx.pool(), (size_t) n_blocks * n_experts);
    if (!ggml_cuda_launch_mm_ids_prefix(
            (const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(), row_expert.get(),
            block_counts.get(), block_offsets.get(), n_experts, n_tokens, n_expert_used, ids_stride, stream)) {
        GGML_ABORT("the CUTLASS NVFP4 decode path could not build the expert schedule");
    }

    const int64_t padded_k          = GGML_PAD(k, 128);
    const int64_t scale_blocks      = padded_k / QK_NVFP4_SUB;
    const size_t  activation_size   = (size_t) n_rows * padded_k / 2;
    const size_t  scale_size        = (size_t) n_rows * 128 * scale_blocks;
    ggml_cuda_pool_alloc<uint8_t> activation(ctx.pool(), activation_size);
    ggml_cuda_pool_alloc<uint8_t> activation_scales(ctx.pool(), scale_size);
    CUDA_CHECK(cudaMemsetAsync(activation_scales.get(), 0, scale_size, stream));
    const int64_t stride_route = src1->ne[1] == 1 ? 0 : src1->nb[1] / ggml_element_size(src1);
    const int64_t stride_token = src1->nb[2] / ggml_element_size(src1);
    moe_cutlass_quantize_nvfp4_routes<<<dim3(n_rows, scale_blocks, 1), WARP_SIZE, 0, stream>>>(
        (const float *) src1->data, ids_dst.get(), activation.get(), activation_scales.get(), k, padded_k,
        stride_route, stride_token, n_expert_used);
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_moe_weight_view weight;
    if (!ggml_cuda_moe_repack_weight(
            ctx, src0, ggml_cuda_moe_weight_layout::cutlass, weight, stream, 2, true)) {
        GGML_ABORT("the CUTLASS NVFP4 decode path could not repack the expert weight");
    }
    ggml_cuda_moe_weight_wait_ready(weight, stream);

    const bool output_f32 = moe_cutlass_decode_output_f32_requested();
    ggml_cuda_pool_alloc<uint8_t> compact(
        ctx.pool(), (size_t) n_rows * n * (output_f32 ? sizeof(float) : sizeof(uint16_t)));
    const ggml_cuda_moe_cutlass_config config = { 32, true, false, true };
    const bool gemm_ok = output_f32 ?
        ggml_moe_cutlass_sm120::run_grouped_gemm<
            ggml_moe_cutlass_sm120::nvfp4_kernel_traits<32, true, float>>(
                ctx, weight, activation.get(), activation_scales.get(), expert_bounds.get(), row_expert.get(),
                compact.get(), n_experts, n_rows, (int) n, (int) padded_k, device_info.nsm, stream, false, true, true) :
        ggml_cuda_moe_cutlass_gemm(
            ctx, weight, activation.get(), activation_scales.get(), expert_bounds.get(), row_expert.get(),
            compact.get(), n_experts, n_rows, n, padded_k, device_info.nsm, config, stream, true);
    if (!gemm_ok) {
        GGML_ABORT("the required CUTLASS NVFP4 decode GEMM failed");
    }
    ggml_cuda_moe_weight_mark_used(weight, stream);
    if (output_f32) {
        constexpr int threads = 256;
        const int64_t count = n_rows * n;
        moe_cutlass_scatter_f32<<<(count + threads - 1) / threads, threads, 0, stream>>>(
            (const float *) compact.get(), ids_dst.get(), (float *) dst->data, n, n_rows);
        CUDA_CHECK(cudaGetLastError());
    } else {
        ggml_cuda_moe_cutlass_scatter(compact.get(), ids_dst.get(), (float *) dst->data, n, n_rows, stream);
    }

    if (ggml_cuda_moe_cutlass_decode_log_requested()) {
        const int shape = k == 2048 && n == 512 ? 0 : k == 2048 && n == 1024 ? 1 : 2;
        static bool logged[3][9] = {};
        if (!logged[shape][n_tokens]) {
            GGML_LOG_INFO(
                "MoE CUTLASS NVFP4 decode dispatch: tokens=%lld experts=%d top=%d K=%lld N=%lld output=%s\n",
                (long long) n_tokens, n_experts, n_expert_used, (long long) k, (long long) n,
                output_f32 ? "f32" : "bf16");
            logged[shape][n_tokens] = true;
        }
    }
    return true;
}

#else

bool ggml_cuda_moe_cutlass_compiled() {
    return false;
}

bool ggml_cuda_moe_cutlass_nvfp4(
        ggml_backend_cuda_context &              ctx,
        const ggml_cuda_moe_cutlass_nvfp4_args & args) {
    GGML_UNUSED_VARS(ctx, args);
    return false;
}

bool ggml_cuda_moe_cutlass_decode_mul_mat_id(ggml_backend_cuda_context & ctx,
                                             const ggml_tensor *         src0,
                                             const ggml_tensor *         src1,
                                             const ggml_tensor *         ids,
                                             ggml_tensor *               dst) {
    GGML_UNUSED_VARS(ctx, src1, ids, dst);
    if (src0->type != GGML_TYPE_NVFP4) {
        return false;
    }
    GGML_ABORT("the CUTLASS MoE backend was not compiled");
    return false;
}

size_t ggml_cuda_moe_cutlass_activation_size(int64_t n_rows, int64_t n_cols) {
    GGML_UNUSED_VARS(n_rows, n_cols);
    return 0;
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
                                           bool            route_groups,
                                           cudaStream_t    stream) {
    GGML_UNUSED_VARS(src, ids, ids_src1, expert_bounds, dst, scales, n_cols, n_cols_padded, n_tokens, n_experts,
                     n_expert_used, ids_stride, route_groups, stream);
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

bool ggml_cuda_moe_cutlass_gemm(ggml_backend_cuda_context &       ctx,
                                const ggml_cuda_moe_weight_view & weight,
                                const uint8_t *                   activation,
                                const uint8_t *                   activation_scales,
                                const int32_t *                   expert_bounds,
                                const int32_t *                   row_expert,
                                void *                            dst,
                                int                               n_experts,
                                int64_t                           n_rows,
                                int64_t                           n,
                                int64_t                           k,
                                int                               sm_count,
                                ggml_cuda_moe_cutlass_config      config,
                                cudaStream_t                      stream,
                                bool                              require) {
    GGML_UNUSED_VARS(ctx, weight, activation, activation_scales, expert_bounds, row_expert, dst, n_experts, n_rows, n,
                     k, sm_count, config, stream, require);
    return false;
}

#endif
