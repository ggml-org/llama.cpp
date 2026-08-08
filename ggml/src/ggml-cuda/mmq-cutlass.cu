#include "mmid.cuh"
#include "mmq-cutlass.cuh"
#include "unary.cuh"

bool ggml_cuda_cutlass_mul_mat_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    if (!ggml_cuda_cutlass_weight_supported(src0) || src1 == nullptr || dst == nullptr ||
        src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || src0->ne[2] != 1 || src0->ne[3] != 1 ||
        src0->ne[0] <= 0 || src0->ne[0] > INT_MAX - 127 || src0->ne[1] <= 0 || src0->ne[1] > INT_MAX ||
        src1->ne[0] != src0->ne[0] || !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) ||
        !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t n_elements = ggml_nelements(src1);
    const int64_t n_rows = n_elements / src0->ne[0];
    return n_elements == n_rows * src0->ne[0] && n_rows > 0 && n_rows <= INT_MAX &&
        dst->ne[0] == src0->ne[1] && ggml_nelements(dst) == n_rows * src0->ne[1];
}

bool ggml_cuda_cutlass_mul_mat_id_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, const ggml_tensor * dst) {
    if (!ggml_cuda_cutlass_weight_supported(src0) || src1 == nullptr || ids == nullptr || dst == nullptr ||
        src1->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32 ||
        src0->ne[0] <= 0 || src0->ne[0] > INT_MAX - 127 || src0->ne[1] <= 0 || src0->ne[1] > INT_MAX ||
        src0->ne[2] <= 0 || src0->ne[2] > 256 || src0->ne[3] != 1 || src1->ne[0] != src0->ne[0] ||
        src1->ne[1] <= 0 || src1->ne[1] > INT_MAX || src1->ne[2] <= 0 || src1->ne[2] > INT_MAX ||
        src1->ne[3] != 1 || (ids->ne[0] != 4 && ids->ne[0] != 8) || ids->ne[0] > src0->ne[2] ||
        ids->ne[1] != src1->ne[2] || ids->ne[2] != 1 || ids->ne[3] != 1 || dst->ne[0] != src0->ne[1] ||
        ids->nb[0] != sizeof(int32_t) || ids->nb[1] % sizeof(int32_t) != 0 ||
        ids->nb[1] / sizeof(int32_t) > INT_MAX || src1->nb[1] == 0 || src1->nb[1] % sizeof(float) != 0 ||
        src1->nb[2] % sizeof(float) != 0 || src1->nb[2] % src1->nb[1] != 0 ||
        src1->nb[1] / sizeof(float) > INT_MAX || src1->nb[2] / sizeof(float) > INT_MAX ||
        src1->nb[2] / src1->nb[1] > INT_MAX || !ggml_is_contiguous(src0) ||
        !ggml_is_contiguous_rows(src1) || !ggml_is_contiguous_rows(ids) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t n_rows = ids->ne[0] * ids->ne[1];
    if (n_rows <= 0 || n_rows > INT_MAX || src0->ne[0] > INT_MAX ||
        ggml_nelements(dst) != n_rows * src0->ne[1]) {
        return false;
    }

    const ggml_cuda_mm_ids_plan plan = {
        (int) src0->ne[2],
        (int) ids->ne[1],
        (int) ids->ne[0],
        (int) src1->ne[1],
        (int) (ids->nb[1] / sizeof(int32_t)),
        (int) (src1->nb[2] / src1->nb[1]),
        ggml_cuda_mm_ids_src1_map::source_to_compact,
        true,
        true,
    };
    ggml_cuda_mm_ids_plan_requirements requirements;
    return ggml_cuda_mm_ids_get_requirements(plan, requirements) && requirements.block_counts_count != 0;
}

bool ggml_cuda_cutlass_enabled() {
    return ggml_cuda_cutlass_compiled();
}

#ifdef GGML_CUDA_CUTLASS
#    include <cuda_bf16.h>
#    include <cuda_fp8.h>

#    include <algorithm>
#    include <array>
#    include <climits>
#    include <type_traits>

struct cutlass_grouped_gemm_config {
    int tile_n;
};

static cutlass_grouped_gemm_config select_grouped_gemm_config(int64_t n_rows, int n_experts) {
    const int64_t rows_per_expert = (n_rows + n_experts - 1) / n_experts;
    return { rows_per_expert <= 32 ? 32 : rows_per_expert <= 64 ? 64 : 128 };
}

static __device__ __forceinline__ uint8_t cutlass_mxfp8_scale(float amax) {
    if (!(amax > 0.0f)) {
        return 0;
    }

    constexpr float e4m3_max = 448.0f;
    const int       exponent = __float2int_ru(log2f(amax / e4m3_max));
    return (uint8_t) max(0, min(254, exponent + 127));
}

static __device__ __forceinline__ float cutlass_half_warp_amax(float value0, float value1) {
    float amax = fmaxf(fabsf(value0), fabsf(value1));
#    pragma unroll
    for (int mask = 8; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 16));
    }
    return amax;
}

static __device__ __forceinline__ uint8_t * moe_cutlass_scale_ptr(uint8_t *       scales,
                                                                  const int32_t * expert_bounds,
                                                                  int             expert,
                                                                  int             row,
                                                                  int             k_block,
                                                                  int             padded_k_blocks) {
    const int64_t start = ((int64_t) expert_bounds[expert] + (int64_t) expert * 127) / 128 * 128;
    return scales + start * padded_k_blocks +
           ggml_cuda_cutlass_blockscaled_scale_offset(row - expert_bounds[expert], k_block, padded_k_blocks);
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
            scales[ggml_cuda_cutlass_blockscaled_scale_offset(row, scale_block, scale_blocks)] = scale;
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
            scales[ggml_cuda_cutlass_blockscaled_scale_offset(row, scale_block, scale_blocks)] = scale_code;
        }
    }
}

static __global__ void cutlass_nvfp4_swiglu_quantize(const __nv_bfloat16 * __restrict__ gate,
                                                      const __nv_bfloat16 * __restrict__ up,
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
            const int64_t gate_index = up == nullptr ? (int64_t) row * 2 * n_ff + k : (int64_t) row * n_ff + k;
            const float gate_value = __fmul_rn(__bfloat162float(gate[gate_index]), gate_multiplier);
            const float up_value = __fmul_rn(
                __bfloat162float(up == nullptr ? gate[(int64_t) row * 2 * n_ff + n_ff + k] : up[gate_index]),
                up_multiplier);
            value = __fmul_rn(up_value, ggml_cuda_op_silu_single(gate_value));
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
            scales[ggml_cuda_cutlass_blockscaled_scale_offset(row, scale_block, scale_blocks)] = scale_code;
        }
    }
}

static __global__ void cutlass_mxfp8_swiglu_quantize(const __nv_bfloat16 * __restrict__ gate,
                                                      const __nv_bfloat16 * __restrict__ up,
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
            const int64_t gate_index = up == nullptr ? (int64_t) row * 2 * n_ff + k : (int64_t) row * n_ff + k;
            const float gate_value = __fmul_rn(__bfloat162float(gate[gate_index]), gate_multiplier);
            const float up_value = __fmul_rn(
                __bfloat162float(up == nullptr ? gate[(int64_t) row * 2 * n_ff + n_ff + k] : up[gate_index]),
                up_multiplier);
            value.x = __fmul_rn(up_value, ggml_cuda_op_silu_single(gate_value));
        }
        if (k + 1 < n_ff) {
            const int64_t gate_index = up == nullptr ? (int64_t) row * 2 * n_ff + k + 1 :
                                                       (int64_t) row * n_ff + k + 1;
            const float gate_value = __fmul_rn(__bfloat162float(gate[gate_index]), gate_multiplier);
            const float up_value = __fmul_rn(
                __bfloat162float(up == nullptr ? gate[(int64_t) row * 2 * n_ff + n_ff + k + 1] : up[gate_index]),
                up_multiplier);
            value.y = __fmul_rn(up_value, ggml_cuda_op_silu_single(gate_value));
        }

        const float         amax      = cutlass_half_warp_amax(value.x, value.y);
        const uint8_t       scale     = cutlass_mxfp8_scale(amax);
        const float         inv_scale = amax == 0.0f ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(scale));
        const __nv_fp8_e4m3 q0(value.x * inv_scale);
        const __nv_fp8_e4m3 q1(value.y * inv_scale);
        *reinterpret_cast<uint16_t *>(dst + (int64_t) row * n_ff_padded + k) =
            (uint16_t) q0.__x | ((uint16_t) q1.__x << 8);
        if (pair_lane == 0) {
            scales[ggml_cuda_cutlass_blockscaled_scale_offset(row, scale_block, scale_blocks)] = scale;
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

template <int n_expert_used>
static __global__ void moe_cutlass_quantize_nvfp4_broadcast_cta(const float * __restrict__ src,
                                                                 const int32_t * __restrict__ ids,
                                                                 const int32_t * __restrict__ ids_src1,
                                                                const int32_t * __restrict__ expert_bounds,
                                                                uint8_t * __restrict__ dst,
                                                                uint8_t * __restrict__ scales,
                                                                 int64_t n_cols,
                                                                 int64_t n_cols_padded,
                                                                 int64_t stride_token,
                                                                 int64_t ids_stride) {
    constexpr int  warps = 8;
    __shared__ int route_rows[n_expert_used];
    __shared__ int route_experts[n_expert_used];

    const int token = blockIdx.x;
    if (threadIdx.x < n_expert_used) {
        const int slot      = threadIdx.x;
        const int route     = token * n_expert_used + slot;
        route_rows[slot]    = ids_src1[route];
        route_experts[slot] = ids[(int64_t) token * ids_stride + slot];
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
                                       scale_blocks) = scale_code;
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
                                                  int64_t         ids_stride,
                                                  cudaStream_t    stream) {
    if (n_tokens <= 0 || n_tokens > UINT_MAX) {
        return false;
    }
    if (n_expert_used == 4) {
        moe_cutlass_quantize_nvfp4_broadcast_cta<4><<<(unsigned) n_tokens, 256, 0, stream>>>(
            src, ids, ids_src1, expert_bounds, dst, scales, n_cols, n_cols_padded, stride_token, ids_stride);
    } else if (n_expert_used == 8) {
        moe_cutlass_quantize_nvfp4_broadcast_cta<8><<<(unsigned) n_tokens, 256, 0, stream>>>(
            src, ids, ids_src1, expert_bounds, dst, scales, n_cols, n_cols_padded, stride_token, ids_stride);
    } else {
        return false;
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

static __global__ void cutlass_quantize_mmid_mxfp8(const float * __restrict__ src,
                                                    const int32_t * __restrict__ ids_dst,
                                                    const int32_t * __restrict__ row_expert,
                                                    const int32_t * __restrict__ expert_bounds,
                                                    uint8_t * __restrict__ dst,
                                                    uint8_t * __restrict__ scales,
                                                    int n_cols,
                                                    int n_cols_padded,
                                                    int stride_channel,
                                                    int stride_token,
                                                    int n_channels,
                                                    int n_expert_used) {
    constexpr int warps = 8;
    const int row   = blockIdx.x;
    const int route = ids_dst[row];
    const int token = route / n_expert_used;
    const int slot  = route - token * n_expert_used;
    const float * source = src + (int64_t) token * stride_token +
        (int64_t) (slot % n_channels) * stride_channel;
    const int warp              = threadIdx.x / WARP_SIZE;
    const int lane              = threadIdx.x % WARP_SIZE;
    const int half              = lane / 16;
    const int pair_lane         = lane % 16;
    const int scale_blocks      = n_cols_padded / WARP_SIZE;
    const int scale_block_pairs = scale_blocks / 2;

    for (int pair = warp; pair < scale_block_pairs; pair += warps) {
        const int scale_block = 2 * pair + half;
        const int k = scale_block * WARP_SIZE + pair_lane * 2;
        float2 value = { 0.0f, 0.0f };
        if (k + 1 < n_cols) {
            value = *reinterpret_cast<const float2 *>(source + k);
        } else if (k < n_cols) {
            value.x = source[k];
        }
        const float amax = cutlass_half_warp_amax(value.x, value.y);
        const uint8_t scale = cutlass_mxfp8_scale(amax);
        const float inv_scale = amax == 0.0f ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(scale));
        const __nv_fp8_e4m3 q0(value.x * inv_scale);
        const __nv_fp8_e4m3 q1(value.y * inv_scale);
        *reinterpret_cast<uint16_t *>(dst + (int64_t) row * n_cols_padded + k) =
            (uint16_t) q0.__x | ((uint16_t) q1.__x << 8);
        if (pair_lane == 0) {
            *moe_cutlass_scale_ptr(scales, expert_bounds, row_expert[row], row, scale_block, scale_blocks) =
                scale;
        }
    }
}

static __global__ void cutlass_quantize_mmid_nvfp4(const float * __restrict__ src,
                                                    const int32_t * __restrict__ ids_dst,
                                                    const int32_t * __restrict__ row_expert,
                                                    const int32_t * __restrict__ expert_bounds,
                                                    uint8_t * __restrict__ dst,
                                                    uint8_t * __restrict__ scales,
                                                    int n_cols,
                                                    int n_cols_padded,
                                                    int stride_channel,
                                                    int stride_token,
                                                    int n_channels,
                                                    int n_expert_used) {
    constexpr int warps = 8;
    const int row   = blockIdx.x;
    const int route = ids_dst[row];
    const int token = route / n_expert_used;
    const int slot  = route - token * n_expert_used;
    const float * source = src + (int64_t) token * stride_token +
        (int64_t) (slot % n_channels) * stride_channel;
    const int warp              = threadIdx.x / WARP_SIZE;
    const int lane              = threadIdx.x % WARP_SIZE;
    const int half              = lane / QK_NVFP4_SUB;
    const int half_lane         = lane % QK_NVFP4_SUB;
    const unsigned mask         = half == 0 ? 0x0000FFFFu : 0xFFFF0000u;
    const int scale_blocks      = n_cols_padded / QK_NVFP4_SUB;
    const int scale_block_pairs = (scale_blocks + 1) / 2;

    for (int pair = warp; pair < scale_block_pairs; pair += warps) {
        const int scale_block = 2 * pair + half;
        if (scale_block >= scale_blocks) {
            continue;
        }
        const int k = scale_block * QK_NVFP4_SUB + half_lane;
        const float value = k < n_cols ? source[k] : 0.0f;
        float amax = fabsf(value);
#    pragma unroll
        for (int offset = QK_NVFP4_SUB / 2; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(mask, amax, offset, QK_NVFP4_SUB));
        }
        const uint8_t scale_code = ggml_cuda_fp32_to_ue4m3(amax / 6.0f);
        const float scale = ggml_cuda_ue4m3_to_fp32(scale_code);
        const float inv_scale = scale > 0.0f ? 0.5f / scale : 0.0f;
        const uint8_t quantized = ggml_cuda_float_to_fp4_e2m1(value, inv_scale);
        const uint8_t next = (uint8_t) __shfl_down_sync(mask, (unsigned) quantized, 1, QK_NVFP4_SUB);
        if ((half_lane & 1) == 0) {
            dst[(int64_t) row * (n_cols_padded / 2) +
                (int64_t) scale_block * (QK_NVFP4_SUB / 2) + half_lane / 2] = quantized | (next << 4);
        }
        if (half_lane == 0) {
            *moe_cutlass_scale_ptr(scales, expert_bounds, row_expert[row], row, scale_block, scale_blocks) =
                scale_code;
        }
    }
}

static __global__ void cutlass_scatter_mmid_bf16(const __nv_bfloat16 * __restrict__ src,
                                                  const int32_t * __restrict__ ids_dst,
                                                  float * __restrict__ dst,
                                                  int n_cols,
                                                  int64_t n_elements) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n_elements) {
        return;
    }
    const int row = index / n_cols;
    const int col = index - (int64_t) row * n_cols;
    dst[(int64_t) ids_dst[row] * n_cols + col] = __bfloat162float(src[index]);
}

static __global__ void moe_cutlass_nvfp4_w13_epilogue(const __nv_bfloat16 * __restrict__ gate,
                                                       const __nv_bfloat16 * __restrict__ up,
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
            const int64_t gate_index = up == nullptr ? (int64_t) row * 2 * n_ff + k : (int64_t) row * n_ff + k;
            const float gate_value = __fmul_rn(__bfloat162float(gate[gate_index]), gate_scale[expert]);
            const float up_value = __fmul_rn(
                __bfloat162float(up == nullptr ? gate[(int64_t) row * 2 * n_ff + n_ff + k] :
                                                 up[(int64_t) row * n_ff + k]),
                up_scale[expert]);
            value = __fmul_rn(up_value, ggml_cuda_op_silu_single(gate_value));
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
            *moe_cutlass_scale_ptr(scales, expert_bounds, expert, row, scale_block, scale_blocks) = scale_code;
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
                                                      int n_expert_used,
                                                      int ids_stride,
                                                      int weights_route_stride,
                                                      int weights_token_stride) {
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (int64_t) n_tokens * n_embd) {
        return;
    }

    const int token  = index / n_embd;
    const int col    = index - (int64_t) token * n_embd;
    float     result = 0.0f;
    for (int slot = 0; slot < n_expert_used; ++slot) {
        const int route  = token * n_expert_used + slot;
        const int expert = ids[(int64_t) token * ids_stride + slot];
        const int row    = ids_src1[route];
        float     value  = __bfloat162float(down[(int64_t) row * n_embd + col]);
        value            = __fmul_rn(value, down_scale[expert]);
        value            = __fmul_rn(
            value, weights[(int64_t) token * weights_token_stride + (int64_t) slot * weights_route_stride]);
        result           = __fadd_rn(result, value);
    }
    dst[index] = result;
}

static size_t ggml_cuda_cutlass_activation_size(ggml_type type, int64_t n_rows, int64_t n_cols) {
    GGML_ASSERT(type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4);
    GGML_ASSERT(n_rows > 0 && n_cols > 0 && n_cols % 128 == 0);
    return type == GGML_TYPE_NVFP4 ? (size_t) n_rows * n_cols / 2 : (size_t) n_rows * n_cols;
}

static size_t ggml_cuda_cutlass_scale_size(ggml_type type, int64_t n_rows, int64_t n_cols) {
    GGML_ASSERT(type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4);
    GGML_ASSERT(n_rows > 0 && n_cols > 0 && n_cols % 128 == 0);
    const int scale_vector_size = type == GGML_TYPE_NVFP4 ? QK_NVFP4_SUB : QK_MXFP4;
    return (size_t) GGML_PAD(n_rows, 128) * (n_cols / scale_vector_size);
}

static bool ggml_cuda_cutlass_quantize(const float * src,
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

static size_t cutlass_grouped_scale_size(
        ggml_type type, int64_t n_rows, int n_experts, int64_t n_cols) {
    GGML_ASSERT(type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4);
    const int64_t scale_block_size = type == GGML_TYPE_NVFP4 ? QK_NVFP4_SUB : QK_MXFP4;
    const int64_t padded_k_blocks  = GGML_PAD((n_cols + scale_block_size - 1) / scale_block_size, 4);
    const int64_t padded_rows = GGML_PAD(n_rows + (int64_t) n_experts * 127, 128);
    return (size_t) padded_rows * padded_k_blocks;
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

struct mxfp_format_traits {
    static constexpr int activation_bits  = 8;
    static constexpr int scale_granularity = QK_MXFP4;

    using Scale      = cutlass::float_ue8m0_t;
    using Activation = cutlass::float_e4m3_t;
    using Weight     = cutlass::float_e2m1_t;

    template <typename Element>
    using MainloopElement = cute::tuple<Element, Scale>;

    template <typename Element>
    static constexpr int alignment = 128 / cutlass::sizeof_bits<Element>::value;
};

struct nvfp4_format_traits {
    static constexpr int activation_bits  = 4;
    static constexpr int scale_granularity = QK_NVFP4_SUB;

    using Scale      = cutlass::float_ue4m3_t;
    using Activation = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using Weight     = Activation;

    template <typename Element>
    using MainloopElement = Element;

    template <typename Element>
    static constexpr int alignment = 32;
};

template <typename Layout, bool Grouped>
using cutlass_layout_t = std::conditional_t<Grouped, Layout *, Layout>;

template <typename GemmKernel, typename CollectiveMainloop, bool Grouped>
struct cutlass_kernel_access;

template <typename GemmKernel, typename CollectiveMainloop>
struct cutlass_kernel_access<GemmKernel, CollectiveMainloop, false> {
    using StrideA   = typename GemmKernel::StrideA;
    using StrideB   = typename GemmKernel::StrideB;
    using StrideC   = typename GemmKernel::StrideC;
    using StrideD   = typename GemmKernel::StrideD;
    using LayoutSFA = typename CollectiveMainloop::LayoutSFA;
    using LayoutSFB = typename CollectiveMainloop::LayoutSFB;
};

template <typename GemmKernel, typename CollectiveMainloop>
struct cutlass_kernel_access<GemmKernel, CollectiveMainloop, true> {
    using StrideA   = typename GemmKernel::InternalStrideA;
    using StrideB   = typename GemmKernel::InternalStrideB;
    using StrideC   = void;
    using StrideD   = typename GemmKernel::InternalStrideD;
    using LayoutSFA = typename CollectiveMainloop::InternalLayoutSFA;
    using LayoutSFB = typename CollectiveMainloop::InternalLayoutSFB;
};

template <typename Format, int TileN, bool Grouped, bool SwapAB, typename OutputType>
struct blockscaled_kernel_traits {
    static constexpr bool swap_ab         = SwapAB;
    static constexpr int  tile_n          = TileN;
    static constexpr int  activation_bits = Format::activation_bits;
    static constexpr int  scale_granularity = Format::scale_granularity;

    using Scale            = typename Format::Scale;
    using Output           = OutputType;
    using ElementA         = std::conditional_t<SwapAB, typename Format::Weight, typename Format::Activation>;
    using ElementB         = std::conditional_t<SwapAB, typename Format::Activation, typename Format::Weight>;
    using ElementAMainloop = typename Format::template MainloopElement<ElementA>;
    using ElementBMainloop = typename Format::template MainloopElement<ElementB>;
    using LayoutABase      = cutlass::layout::RowMajor;
    using LayoutBBase      = cutlass::layout::ColumnMajor;
    using LayoutDBase      = std::conditional_t<SwapAB, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
    using LayoutA          = cutlass_layout_t<LayoutABase, Grouped>;
    using LayoutB          = cutlass_layout_t<LayoutBBase, Grouped>;
    using LayoutD          = cutlass_layout_t<LayoutDBase, Grouped>;
    using TileShape        = Shape<_128, Int<TileN>, _128>;
    using ClusterShape     = Shape<_1, _1, _1>;

    static constexpr int alignment_a = Format::template alignment<ElementA>;
    static constexpr int alignment_b = Format::template alignment<ElementB>;
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
    using ProblemShape = std::conditional_t<
        Grouped, cutlass::gemm::GroupProblemShape<Shape<int, int, int>>, Shape<int, int, int, int>>;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm       = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using Access     = cutlass_kernel_access<GemmKernel, CollectiveMainloop, Grouped>;
    using StrideA    = typename Access::StrideA;
    using StrideB    = typename Access::StrideB;
    using StrideC    = typename Access::StrideC;
    using StrideD    = typename Access::StrideD;
    using LayoutSFA  = typename Access::LayoutSFA;
    using LayoutSFB  = typename Access::LayoutSFB;
    using BlockScaleConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;

    static_assert(CollectiveMainloop::TiledMma::SFVecSize == scale_granularity);
};

template <int TileN, bool SwapAB>
using mxfp_kernel_traits = blockscaled_kernel_traits<mxfp_format_traits, TileN, true, SwapAB, DefaultOutput>;

template <int TileN, bool SwapAB, typename OutputType = DefaultOutput>
using nvfp4_kernel_traits = blockscaled_kernel_traits<nvfp4_format_traits, TileN, true, SwapAB, OutputType>;

template <int TileN, typename OutputType>
using dense_mxfp_kernel_traits = blockscaled_kernel_traits<mxfp_format_traits, TileN, false, false, OutputType>;

template <int TileN, typename OutputType>
using dense_nvfp4_kernel_traits = blockscaled_kernel_traits<nvfp4_format_traits, TileN, false, false, OutputType>;

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

template <typename T> static T * take(char * base, size_t & offset, int count) {
    offset     = GGML_PAD(offset, alignof(T) > 128 ? alignof(T) : 128);
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
    size                  = GGML_PAD(size, (size_t) 128);
    return result;
}

template <typename Traits>
static __global__ void setup_metadata(grouped_metadata<Traits> metadata,
                                      const uint8_t *          activation,
                                      const uint8_t *          activation_scales,
                                      const char *             weights,
                                      const uint8_t *          weight_scales,
                                      const int32_t *          group_bounds,
                                      void *                   dst,
                                      int                      n,
                                      int                      k,
                                      int                      weight_scale_stride,
                                      int                      groups,
                                      bool                     pdl) {
    using Problem          = typename Traits::ProblemShape::UnderlyingProblemShape;
    using BlockScaleConfig = typename Traits::BlockScaleConfig;

    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if (group >= groups) {
        return;
    }

    if (pdl) {
        ggml_cuda_pdl_sync();
        ggml_cuda_pdl_lc();
    }

    const int     row_begin       = group_bounds[group];
    const int     m               = group_bounds[group + 1] - row_begin;
    const int     padded_n        = GGML_PAD(n, 128);
    const int     padded_k        = GGML_PAD(k, 128);
    const int     padded_k_blocks = padded_k / Traits::scale_granularity;
    const int64_t activation_scale_begin =
        ((int64_t) row_begin + (int64_t) group * 127) / 128 * 128 * padded_k_blocks;

    if constexpr (Traits::swap_ab) {
        metadata.shapes[group] = Problem(n, m, k);
        metadata.a[group] =
            reinterpret_cast<const typename Traits::Gemm::ElementA *>(weights + (int64_t) group * n * k / 2);
        metadata.b[group] = reinterpret_cast<const typename Traits::Gemm::ElementB *>(
            activation + (int64_t) row_begin * k * Traits::activation_bits / 8);
        metadata.stride_a[group] = cutlass::make_cute_packed_stride(typename Traits::StrideA{}, make_shape(n, k, 1));
        metadata.stride_b[group] = cutlass::make_cute_packed_stride(typename Traits::StrideB{}, make_shape(m, k, 1));
        metadata.stride_d[group] = cutlass::make_cute_packed_stride(typename Traits::StrideD{}, make_shape(n, m, 1));
        metadata.scale_a[group]  = reinterpret_cast<const typename Traits::Scale *>(
            weight_scales + (int64_t) group * weight_scale_stride);
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
            reinterpret_cast<const typename Traits::Gemm::ElementB *>(weights + (int64_t) group * n * k / 2);
        metadata.stride_a[group] = cutlass::make_cute_packed_stride(typename Traits::StrideA{}, make_shape(m, k, 1));
        metadata.stride_b[group] = cutlass::make_cute_packed_stride(typename Traits::StrideB{}, make_shape(n, k, 1));
        metadata.stride_d[group] = cutlass::make_cute_packed_stride(typename Traits::StrideD{}, make_shape(m, n, 1));
        metadata.scale_a[group] =
            reinterpret_cast<const typename Traits::Scale *>(activation_scales + activation_scale_begin);
        metadata.scale_b[group] = reinterpret_cast<const typename Traits::Scale *>(
            weight_scales + (int64_t) group * weight_scale_stride);
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
                             void *                           dst,
                             int                              group_count,
                             int                              n,
                             int                              k,
                             int                              sm_count,
                             cudaStream_t                     stream,
                             bool                             require) {
    using Gemm = typename Traits::Gemm;

    const int groups        = group_count;
    size_t    metadata_size = 0;
    make_metadata<Traits>(nullptr, groups, metadata_size);
    ggml_cuda_pool_alloc<char> metadata_alloc(ctx.pool());
    char *                     metadata_data = metadata_alloc.alloc(metadata_size);
    grouped_metadata<Traits>   metadata      = make_metadata<Traits>(metadata_data, groups, metadata_size);

    constexpr int threads = 128;
    auto setup_kernel = setup_metadata<Traits>;
    const bool use_pdl = ggml_cuda_kernel_should_use_pdl(reinterpret_cast<const void *>(setup_kernel));
#    if defined(GGML_CUDA_USE_PDL)
    if (use_pdl) {
        const ggml_cuda_kernel_launch_params params(
            dim3((groups + threads - 1) / threads), dim3(threads), 0, stream);
        ggml_cuda_pdl_config launch_config(params);
        CUDA_CHECK(cudaLaunchKernelEx(
            &launch_config.cfg, setup_kernel, metadata, activation, activation_scales, weight.data, weight.scales,
            group_bounds, dst, n, k, weight.scale_stride, groups, use_pdl));
    } else
#    endif
    {
        setup_kernel<<<(groups + threads - 1) / threads, threads, 0, stream>>>(
            metadata, activation, activation_scales, weight.data, weight.scales, group_bounds, dst, n, k,
            weight.scale_stride, groups, use_pdl);
        CUDA_CHECK(cudaGetLastError());
    }

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

    const cutlass::Status run = gemm.run(stream, nullptr, use_pdl);
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
                                  void *                           dst,
                                  int                              group_count,
                                  int                              n,
                                  int                              k,
                                  int                              sm_count,
                                  cutlass_grouped_gemm_config     config,
                                  cudaStream_t                     stream,
    bool                             require) {
    if (config.tile_n == 32) {
        return run_grouped_gemm<Traits<32, true>>(
            ctx, weight, activation, activation_scales, group_bounds, dst, group_count, n, k, sm_count, stream, require);
    }
    if (config.tile_n == 64) {
        return run_grouped_gemm<Traits<64, true>>(
            ctx, weight, activation, activation_scales, group_bounds, dst, group_count, n, k, sm_count, stream, require);
    }
    return run_grouped_gemm<Traits<128, true>>(
        ctx, weight, activation, activation_scales, group_bounds, dst, group_count, n, k, sm_count, stream, require);
}

}  // namespace ggml_cutlass_sm120

bool ggml_cuda_cutlass_compiled() {
    return true;
}

static bool ggml_cuda_cutlass_grouped_gemm(ggml_backend_cuda_context &      ctx,
                                           const ggml_cuda_cutlass_weight & weight,
                                           const uint8_t *                  activation,
                                           const uint8_t *                  activation_scales,
                                           const int32_t *                  group_bounds,
                                           void *                           dst,
                                           int                              groups,
                                           int64_t                          n_rows,
                                           int64_t                          n,
                                           int64_t                          k,
                                           int                              sm_count,
                                           cutlass_grouped_gemm_config     config,
                                           cudaStream_t                     stream,
                                           bool                             require) {
    using namespace ggml_cutlass_sm120;
    GGML_ASSERT(weight.data != nullptr && weight.scales != nullptr && activation != nullptr &&
                activation_scales != nullptr && dst != nullptr);
    GGML_ASSERT(groups > 0 && n > 0 && k > 0 && n_rows > 0);
    GGML_ASSERT(n <= INT_MAX && k <= INT_MAX && n_rows <= INT_MAX && k == weight.k);
    GGML_ASSERT(config.tile_n == 32 || config.tile_n == 64 || config.tile_n == 128);
    GGML_ASSERT(group_bounds != nullptr);

    if (weight.type == GGML_TYPE_MXFP4) {
        return dispatch_grouped_gemm<mxfp_kernel_traits>(ctx, weight, activation, activation_scales, group_bounds,
                                                         dst, groups, (int) n, (int) k, sm_count, config,
                                                         stream, require);
    }
    if (weight.type == GGML_TYPE_NVFP4) {
        return dispatch_grouped_gemm<nvfp4_kernel_traits>(ctx, weight, activation, activation_scales, group_bounds,
                                                          dst, groups, (int) n, (int) k, sm_count, config,
                                                          stream, require);
    }
    if (require) {
        GGML_ABORT("CUTLASS does not support weight type %s", ggml_type_name(weight.type));
    }
    return false;
}

bool ggml_cuda_cutlass_mul_mat_id(ggml_backend_cuda_context & ctx,
                                  const ggml_tensor *         src0,
                                  const ggml_tensor *         src1,
                                  const ggml_tensor *         ids,
                                  ggml_tensor *               dst) {
    ggml_cuda_cutlass_weight weight;
    if (!ggml_cuda_cutlass_mul_mat_id_supported(src0, src1, ids, dst) ||
        !ggml_cuda_cutlass_weight_from_tensor(src0, weight)) {
        return false;
    }

    const int64_t n_rows_64 = ids->ne[0] * ids->ne[1];
    const size_t ids_stride_size = ids->nb[1] / sizeof(int32_t);
    const size_t stride_channel_size = src1->nb[1] / sizeof(float);
    const size_t stride_token_size = src1->nb[2] / sizeof(float);

    const int n_experts     = (int) src0->ne[2];
    const int n_expert_used = (int) ids->ne[0];
    const int n_tokens      = (int) ids->ne[1];
    const int n_rows        = (int) n_rows_64;
    const int n             = (int) src0->ne[1];
    const int k             = (int) weight.k;
    const int ids_stride    = (int) ids_stride_size;

    const ggml_cuda_mm_ids_plan ids_plan = {
        n_experts,
        n_tokens,
        n_expert_used,
        (int) src1->ne[1],
        ids_stride,
        (int) (src1->nb[2] / src1->nb[1]),
        ggml_cuda_mm_ids_src1_map::source_to_compact,
        true,
        true,
    };
    ggml_cuda_mm_ids_plan_requirements requirements;
    if (!ggml_cuda_mm_ids_get_requirements(ids_plan, requirements)) {
        return false;
    }

    ggml_cuda_mm_ids_plan_storage ids_storage(ctx.pool(), requirements);
    const ggml_cuda_mm_ids_plan_view ids_view = ids_storage.view();
    cudaStream_t stream = ctx.stream();
    if (!ggml_cuda_launch_mm_ids_plan((const int32_t *) ids->data, ids_plan, ids_view, stream)) {
        return false;
    }

    const size_t activation_size = ggml_cuda_cutlass_activation_size(src0->type, n_rows, k);
    const size_t scales_size = cutlass_grouped_scale_size(src0->type, n_rows, n_experts, k);
    ggml_cuda_pool_alloc<uint8_t> activation(ctx.pool(), activation_size);
    ggml_cuda_pool_alloc<uint8_t> activation_scales(ctx.pool(), scales_size);
    CUDA_CHECK(cudaMemsetAsync(activation_scales.get(), 0, scales_size, stream));
    if (src0->type == GGML_TYPE_MXFP4) {
        cutlass_quantize_mmid_mxfp8<<<n_rows, 256, 0, stream>>>(
            (const float *) src1->data, ids_storage.ids_dst.get(), ids_storage.row_expert.get(),
            ids_storage.expert_bounds.get(), activation.get(),
            activation_scales.get(), (int) src0->ne[0], k, (int) stride_channel_size, (int) stride_token_size,
            (int) src1->ne[1], n_expert_used);
    } else {
        cutlass_quantize_mmid_nvfp4<<<n_rows, 256, 0, stream>>>(
            (const float *) src1->data, ids_storage.ids_dst.get(), ids_storage.row_expert.get(),
            ids_storage.expert_bounds.get(), activation.get(),
            activation_scales.get(), (int) src0->ne[0], k, (int) stride_channel_size, (int) stride_token_size,
            (int) src1->ne[1], n_expert_used);
    }
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_pool_alloc<__nv_bfloat16> sorted_output(ctx.pool(), (size_t) n_rows * n);
    const cutlass_grouped_gemm_config gemm_config = select_grouped_gemm_config(n_rows, n_experts);
    const auto & device_info = ggml_cuda_info().devices[ctx.device];
    if (!ggml_cuda_cutlass_grouped_gemm(
            ctx, weight, activation.get(), activation_scales.get(), ids_storage.expert_bounds.get(),
            sorted_output.get(), n_experts, n_rows, n, k, device_info.nsm, gemm_config, stream, true)) {
        return false;
    }

    constexpr int threads = 256;
    const int64_t n_elements = (int64_t) n_rows * n;
    cutlass_scatter_mmid_bf16<<<(n_elements + threads - 1) / threads, threads, 0, stream>>>(
        sorted_output.get(), ids_storage.ids_dst.get(), (float *) dst->data, n, n_elements);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool ggml_cuda_cutlass_mul_mat(ggml_backend_cuda_context & ctx,
                               const ggml_tensor *         src0,
                               const ggml_tensor *         src1,
                               ggml_tensor *               dst) {
    using namespace ggml_cutlass_sm120;

    ggml_cuda_cutlass_weight weight;
    if (!ggml_cuda_cutlass_mul_mat_supported(src0, src1, dst) ||
        !ggml_cuda_cutlass_weight_from_tensor(src0, weight)) {
        return false;
    }

    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = ggml_nelements(src1) / k;

    const auto & device_info = ggml_cuda_info().devices[ctx.device];
    if (!blackwell_mma_available(device_info.cc)) {
        return false;
    }
    const ggml_backend_buffer_type_t buffer_type = ggml_backend_cuda_buffer_type(ctx.device);
    if (src0->buffer == nullptr || src1->buffer == nullptr || dst->buffer == nullptr ||
        !ggml_backend_buft_is_cuda_cutlass(ggml_backend_buffer_get_type(src0->buffer)) ||
        ggml_backend_buffer_get_type(src1->buffer) != buffer_type ||
        ggml_backend_buffer_get_type(dst->buffer) != buffer_type) {
        return false;
    }

    cudaStream_t stream = ctx.stream();

    const int64_t                 k_padded = weight.k;
    ggml_cuda_pool_alloc<uint8_t> activation(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> activation_scales(ctx.pool());
    uint8_t * activation_data       = activation.alloc(ggml_cuda_cutlass_activation_size(src0->type, m, k_padded));
    uint8_t * activation_scale_data = activation_scales.alloc(ggml_cuda_cutlass_scale_size(src0->type, m, k_padded));
    if (!ggml_cuda_cutlass_quantize((const float *) src1->data, activation_data, activation_scale_data, src0->type, k,
                                    k_padded, src1->nb[1] / sizeof(float), m, stream)) {
        return false;
    }

    const bool launched = dispatch_dense_gemm<float>(ctx, weight, activation_data, activation_scale_data, dst->data,
                                                     (int) m, (int) n, (int) k_padded, stream, true);
    return launched;
}

bool ggml_cuda_cutlass_ffn(ggml_backend_cuda_context & ctx, const ggml_cuda_cutlass_ffn_args & args) {
    using namespace ggml_cutlass_sm120;

    if (!ggml_cuda_cutlass_enabled() || args.gate == nullptr || args.up == nullptr || args.down == nullptr ||
        args.input == nullptr ||
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
    ggml_cuda_cutlass_weight gate_weight;
    ggml_cuda_cutlass_weight up_weight;
    ggml_cuda_cutlass_weight down_weight;
    if (!ggml_cuda_cutlass_weight_from_tensor(args.gate, gate_weight) ||
        !ggml_cuda_cutlass_weight_from_tensor(args.up, up_weight) ||
        !ggml_cuda_cutlass_weight_from_tensor(args.down, down_weight)) {
        return false;
    }

    const ggml_backend_buffer_type_t buffer_type = ggml_backend_cuda_buffer_type(ctx.device);
    const ggml_tensor *              tensors[]   = {
        args.input, args.gate_scale, args.up_scale, args.down_scale, args.dst,
    };
    for (const ggml_tensor * tensor : tensors) {
        if (tensor != nullptr &&
            (tensor->buffer == nullptr || ggml_backend_buffer_get_type(tensor->buffer) != buffer_type)) {
            return false;
        }
    }
    cudaStream_t stream = ctx.stream();
    const int64_t w13_k = gate_weight.k;
    ggml_cuda_pool_alloc<uint8_t> w13_activation(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> w13_scales(ctx.pool());
    uint8_t *                     w13_activation_data =
        w13_activation.alloc(ggml_cuda_cutlass_activation_size(args.gate->type, n_rows, w13_k));
    uint8_t * w13_scale_data = w13_scales.alloc(ggml_cuda_cutlass_scale_size(args.gate->type, n_rows, w13_k));
    if (!ggml_cuda_cutlass_quantize((const float *) args.input->data, w13_activation_data, w13_scale_data,
                                    args.gate->type, n_embd, w13_k, args.input->nb[1] / sizeof(float), n_rows, stream)) {
        return false;
    }

    ggml_cuda_pool_alloc<__nv_bfloat16> w13_output(ctx.pool(), (size_t) n_rows * 2 * n_ff);
    __nv_bfloat16 * gate_output = w13_output.get();
    __nv_bfloat16 * up_output   = gate_output + (size_t) n_rows * n_ff;
    if (!dispatch_dense_gemm<cutlass::bfloat16_t>(ctx, gate_weight, w13_activation_data, w13_scale_data, gate_output,
                                                   (int) n_rows, (int) n_ff, (int) w13_k, stream, true) ||
        !dispatch_dense_gemm<cutlass::bfloat16_t>(ctx, up_weight, w13_activation_data, w13_scale_data, up_output,
                                                   (int) n_rows, (int) n_ff, (int) w13_k, stream, true)) {
        return false;
    }

    ggml_cuda_pool_alloc<uint8_t> w2_activation(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> w2_scales(ctx.pool());
    uint8_t *                     w2_activation_data =
        w2_activation.alloc(ggml_cuda_cutlass_activation_size(args.down->type, n_rows, down_weight.k));
    uint8_t * w2_scale_data = w2_scales.alloc(ggml_cuda_cutlass_scale_size(args.down->type, n_rows, down_weight.k));
    CUDA_CHECK(
        cudaMemsetAsync(w2_scale_data, 0, ggml_cuda_cutlass_scale_size(args.down->type, n_rows, down_weight.k), stream));

    constexpr int threads = 256;
    if (args.down->type == GGML_TYPE_NVFP4) {
        cutlass_nvfp4_swiglu_quantize<<<(unsigned) n_rows, threads, 0, stream>>>(
            gate_output, up_output, args.gate_scale == nullptr ? nullptr : (const float *) args.gate_scale->data,
            args.up_scale == nullptr ? nullptr : (const float *) args.up_scale->data, w2_activation_data, w2_scale_data,
            n_ff, down_weight.k);
    } else {
        cutlass_mxfp8_swiglu_quantize<<<(unsigned) n_rows, threads, 0, stream>>>(
            gate_output, up_output, args.gate_scale == nullptr ? nullptr : (const float *) args.gate_scale->data,
            args.up_scale == nullptr ? nullptr : (const float *) args.up_scale->data, w2_activation_data, w2_scale_data,
            n_ff, down_weight.k);
    }
    CUDA_CHECK(cudaGetLastError());

    ggml_cuda_pool_alloc<__nv_bfloat16> w2_output(ctx.pool(), (size_t) n_rows * n_out);
    if (!dispatch_dense_gemm<cutlass::bfloat16_t>(ctx, down_weight, w2_activation_data, w2_scale_data, w2_output.get(),
                                                   (int) n_rows, (int) n_out, (int) down_weight.k, stream, true)) {
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
    if (!ggml_cuda_cutlass_enabled()) {
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

    ggml_cuda_cutlass_weight gate_weight;
    ggml_cuda_cutlass_weight up_weight;
    ggml_cuda_cutlass_weight down_weight;
    if (!ggml_cuda_cutlass_weight_from_tensor(args.gate, gate_weight) ||
        !ggml_cuda_cutlass_weight_from_tensor(args.up, up_weight) ||
        !ggml_cuda_cutlass_weight_from_tensor(args.down, down_weight)) {
        return false;
    }

    const ggml_backend_buffer_type_t         buffer_type = ggml_backend_cuda_buffer_type(ctx.device);
    const std::array<const ggml_tensor *, 7> tensors     = {
        args.input, args.ids, args.gate_scale, args.up_scale, args.down_scale, args.weights, args.dst,
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
    ggml_cuda_pool_alloc<int32_t> staged_ids(ctx.pool(), n_rows);
    ggml_cuda_pool_alloc<float> staged_weights(ctx.pool(), n_rows);
    const size_t route_row_size = (size_t) n_expert_used * sizeof(int32_t);
    CUDA_CHECK(cudaMemcpy2DAsync(staged_ids.get(), route_row_size, args.ids->data, args.ids->nb[1],
                                 route_row_size, n_tokens, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(staged_weights.get(), args.weights->data, (size_t) n_rows * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));

    const int ids_stride           = n_expert_used;
    const int weights_route_stride = 1;
    const int weights_token_stride = n_expert_used;
    const ggml_cuda_mm_ids_plan ids_plan = {
        n_experts,
        (int) n_tokens,
        n_expert_used,
        1,
        ids_stride,
        n_expert_used,
        ggml_cuda_mm_ids_src1_map::source_to_compact,
        true,
        true,
    };
    ggml_cuda_mm_ids_plan_requirements ids_requirements;
    if (!ggml_cuda_mm_ids_get_requirements(ids_plan, ids_requirements)) {
        return false;
    }
    ggml_cuda_mm_ids_plan_storage ids_storage(ctx.pool(), ids_requirements);
    const ggml_cuda_mm_ids_plan_view ids_view = ids_storage.view();
    if (!ggml_cuda_launch_mm_ids_plan(staged_ids.get(), ids_plan, ids_view, stream)) {
        return false;
    }

    const int    w13_k               = (int) gate_weight.k;
    const size_t w13_activation_size = ggml_cuda_cutlass_activation_size(GGML_TYPE_NVFP4, n_rows, w13_k);
    const size_t w13_scale_size = cutlass_grouped_scale_size(GGML_TYPE_NVFP4, n_rows, n_experts, w13_k);
    ggml_cuda_pool_alloc<uint8_t> w13_activation(ctx.pool(), w13_activation_size);
    ggml_cuda_pool_alloc<uint8_t> w13_activation_scales(ctx.pool(), w13_scale_size);
    {
        CUDA_CHECK(cudaMemsetAsync(w13_activation_scales.get(), 0, w13_scale_size, stream));
        if (!moe_cutlass_quantize_nvfp4_broadcast((const float *) args.input->data,
                                                   staged_ids.get(), ids_storage.ids_src1.get(),
                                                   ids_storage.expert_bounds.get(), w13_activation.get(),
                                                   w13_activation_scales.get(), n_embd, w13_k,
                                                   args.input->nb[2] / sizeof(float), n_tokens, n_expert_used,
                                                   ids_stride, stream)) {
            return false;
        }
    }

    const cutlass_grouped_gemm_config w13_config = select_grouped_gemm_config(n_rows, n_experts);
    ggml_cuda_pool_alloc<__nv_bfloat16> w13_output(ctx.pool(), (size_t) n_rows * 2 * n_ff);
    __nv_bfloat16 * gate_output = w13_output.get();
    __nv_bfloat16 * up_output   = gate_output + (size_t) n_rows * n_ff;
    {
        if (!ggml_cuda_cutlass_grouped_gemm(
                ctx, gate_weight, w13_activation.get(), w13_activation_scales.get(),
                ids_storage.expert_bounds.get(), gate_output, n_experts, n_rows, n_ff, w13_k, device_info.nsm,
                w13_config, stream, true) ||
            !ggml_cuda_cutlass_grouped_gemm(
                ctx, up_weight, w13_activation.get(), w13_activation_scales.get(),
                ids_storage.expert_bounds.get(), up_output, n_experts, n_rows, n_ff, w13_k, device_info.nsm,
                w13_config, stream, true)) {
            return false;
        }
    }

    const int                     w2_k               = (int) down_weight.k;
    const int                     w2_n               = n_embd;
    const size_t                  w2_activation_size = ggml_cuda_cutlass_activation_size(GGML_TYPE_NVFP4, n_rows, w2_k);
    const size_t w2_scale_size = cutlass_grouped_scale_size(GGML_TYPE_NVFP4, n_rows, n_experts, w2_k);
    ggml_cuda_pool_alloc<uint8_t> w2_activation(ctx.pool(), w2_activation_size);
    ggml_cuda_pool_alloc<uint8_t> w2_activation_scales(ctx.pool(), w2_scale_size);
    {
        CUDA_CHECK(cudaMemsetAsync(w2_activation_scales.get(), 0, w2_scale_size, stream));
        moe_cutlass_nvfp4_w13_epilogue<<<n_rows, 256, 0, stream>>>(
            gate_output, up_output, ids_storage.row_expert.get(), ids_storage.expert_bounds.get(),
            (const float *) args.gate_scale->data, (const float *) args.up_scale->data, w2_activation.get(),
            w2_activation_scales.get(), n_ff, w2_k, n_rows);
        CUDA_CHECK(cudaGetLastError());
    }

    const cutlass_grouped_gemm_config w2_config = select_grouped_gemm_config(n_rows, n_experts);
    ggml_cuda_pool_alloc<__nv_bfloat16> w2_output(ctx.pool(), (size_t) n_rows * w2_n);
    {
        if (!ggml_cuda_cutlass_grouped_gemm(ctx, down_weight, w2_activation.get(), w2_activation_scales.get(),
                                            ids_storage.expert_bounds.get(), w2_output.get(), n_experts, n_rows, w2_n,
                                            w2_k, device_info.nsm, w2_config, stream, false)) {
            return false;
        }
    }

    constexpr int finalize_threads = 256;
    const int64_t output_size      = n_tokens * n_embd;
    {
        moe_cutlass_nvfp4_w2_finalize<<<(output_size + finalize_threads - 1) / finalize_threads, finalize_threads, 0,
                                        stream>>>(w2_output.get(), staged_ids.get(),
                                                  ids_storage.ids_src1.get(),
                                                  (const float *) args.down_scale->data,
                                                  staged_weights.get(), (float *) args.dst->data,
                                                  n_embd, (int) n_tokens, n_expert_used, ids_stride,
                                                  weights_route_stride, weights_token_stride);
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

bool ggml_cuda_cutlass_mul_mat_id(ggml_backend_cuda_context & ctx,
                                  const ggml_tensor *         src0,
                                  const ggml_tensor *         src1,
                                  const ggml_tensor *         ids,
                                  ggml_tensor *               dst) {
    GGML_UNUSED_VARS(ctx, src0, src1, ids, dst);
    return false;
}

bool ggml_cuda_cutlass_ffn(ggml_backend_cuda_context & ctx, const ggml_cuda_cutlass_ffn_args & args) {
    GGML_UNUSED_VARS(ctx, args);
    return false;
}

bool ggml_cuda_moe_cutlass_nvfp4(ggml_backend_cuda_context & ctx, const ggml_cuda_moe_cutlass_nvfp4_args & args) {
    GGML_UNUSED_VARS(ctx, args);
    return false;
}

#endif
