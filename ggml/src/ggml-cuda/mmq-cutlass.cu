#include "mmq-cutlass.cuh"

#include <climits>

bool ggml_cuda_cutlass_mul_mat_supported(
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    constexpr int output_alignment = 4;

    if (!ggml_cuda_cutlass_weight_supported(src0) || src1 == nullptr || dst == nullptr ||
        src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        src0->ne[2] != 1 || src0->ne[3] != 1 ||
        src0->ne[0] <= 0 || src0->ne[0] > INT_MAX - 127 ||
        src0->ne[1] <= 0 || src0->ne[1] > INT_MAX || src0->ne[1] % output_alignment != 0 ||
        src1->ne[0] != src0->ne[0] ||
        !ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t n_elements = ggml_nelements(src1);
    const int64_t n_rows     = n_elements / src0->ne[0];
    return n_elements == n_rows * src0->ne[0] &&
        n_rows > 0 && n_rows <= INT_MAX &&
        dst->ne[0] == src0->ne[1] &&
        ggml_nelements(dst) == n_rows * src0->ne[1];
}

#ifdef GGML_CUDA_CUTLASS

#    include <cuda_fp8.h>

#    include "cute/tensor.hpp"
#    include "cutlass/cutlass.h"
#    include "cutlass/detail/sm100_blockscaled_layout.hpp"
#    include "cutlass/epilogue/collective/collective_builder.hpp"
#    include "cutlass/gemm/collective/collective_builder.hpp"
#    include "cutlass/gemm/device/gemm_universal_adapter.h"
#    include "cutlass/gemm/kernel/gemm_universal.hpp"
#    include "cutlass/util/packed_stride.hpp"

static __device__ __forceinline__ uint8_t cutlass_mxfp8_scale(float amax) {
    if (!(amax > 0.0f)) {
        return 0;
    }

    constexpr float e4m3_max = 448.0f;
    const int exponent = __float2int_ru(log2f(amax / e4m3_max));
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

static __global__ void cutlass_quantize_mxfp8(
        const float * __restrict__ src,
        uint8_t * __restrict__ dst,
        uint8_t * __restrict__ scales,
        int64_t n_cols,
        int64_t n_cols_padded,
        int64_t stride_row) {
    constexpr int warps = 8;
    const int row       = blockIdx.x;
    const int warp      = threadIdx.x / WARP_SIZE;
    const int lane      = threadIdx.x % WARP_SIZE;
    const int half      = lane / 16;
    const int pair_lane = lane % 16;
    const int scale_blocks      = n_cols_padded / WARP_SIZE;
    const int scale_block_pairs = scale_blocks / 2;

    for (int pair = warp; pair < scale_block_pairs; pair += warps) {
        const int scale_block = 2 * pair + half;
        const int64_t k       = (int64_t) scale_block * WARP_SIZE + pair_lane * 2;
        float2 value          = { 0.0f, 0.0f };
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

        const float amax         = cutlass_half_warp_amax(value.x, value.y);
        const uint8_t scale      = cutlass_mxfp8_scale(amax);
        const float inv_scale    = amax == 0.0f ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(scale));
        const __nv_fp8_e4m3 q0(value.x * inv_scale);
        const __nv_fp8_e4m3 q1(value.y * inv_scale);
        *reinterpret_cast<uint16_t *>(dst + (int64_t) row * n_cols_padded + k) =
            (uint16_t) q0.__x | ((uint16_t) q1.__x << 8);
        if (pair_lane == 0) {
            scales[ggml_cuda_cutlass_blockscaled_scale_offset(row, scale_block, scale_blocks)] = scale;
        }
    }
}

static __global__ void cutlass_quantize_nvfp4(
        const float * __restrict__ src,
        uint8_t * __restrict__ dst,
        uint8_t * __restrict__ scales,
        int64_t n_cols,
        int64_t n_cols_padded,
        int64_t stride_row) {
    constexpr int warps = 8;
    const int row       = blockIdx.x;
    const int warp      = threadIdx.x / WARP_SIZE;
    const int lane      = threadIdx.x % WARP_SIZE;
    const int half      = lane / QK_NVFP4_SUB;
    const int half_lane = lane % QK_NVFP4_SUB;
    const unsigned mask = half == 0 ? 0x0000FFFFu : 0xFFFF0000u;
    const int scale_blocks      = n_cols_padded / QK_NVFP4_SUB;
    const int scale_block_pairs = (scale_blocks + 1) / 2;

    for (int pair = warp; pair < scale_block_pairs; pair += warps) {
        const int scale_block = 2 * pair + half;
        if (scale_block >= scale_blocks) {
            continue;
        }

        const int64_t k     = (int64_t) scale_block * QK_NVFP4_SUB + half_lane;
        const float value   = k < n_cols ? src[(int64_t) row * stride_row + k] : 0.0f;
        float amax          = fabsf(value);
#    pragma unroll
        for (int offset = QK_NVFP4_SUB / 2; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(mask, amax, offset, QK_NVFP4_SUB));
        }

        const uint8_t scale_code = ggml_cuda_fp32_to_ue4m3(amax / 6.0f);
        const float scale        = ggml_cuda_ue4m3_to_fp32(scale_code);
        const float inv_scale    = scale > 0.0f ? 0.5f / scale : 0.0f;
        const uint8_t quantized  = ggml_cuda_float_to_fp4_e2m1(value, inv_scale);
        const uint8_t next       = (uint8_t) __shfl_down_sync(mask, (unsigned) quantized, 1, QK_NVFP4_SUB);

        if ((half_lane & 1) == 0) {
            dst[(int64_t) row * (n_cols_padded / 2) +
                (int64_t) scale_block * (QK_NVFP4_SUB / 2) + half_lane / 2] =
                quantized | (next << 4);
        }
        if (half_lane == 0) {
            scales[ggml_cuda_cutlass_blockscaled_scale_offset(row, scale_block, scale_blocks)] = scale_code;
        }
    }
}

static size_t cutlass_activation_size(ggml_type type, int64_t n_rows, int64_t n_cols) {
    GGML_ASSERT(type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4);
    GGML_ASSERT(n_rows > 0 && n_cols > 0 && n_cols % 128 == 0);
    return type == GGML_TYPE_NVFP4 ? (size_t) n_rows * n_cols / 2 : (size_t) n_rows * n_cols;
}

static size_t cutlass_scale_size(ggml_type type, int64_t n_rows, int64_t n_cols) {
    GGML_ASSERT(type == GGML_TYPE_MXFP4 || type == GGML_TYPE_NVFP4);
    GGML_ASSERT(n_rows > 0 && n_cols > 0 && n_cols % 128 == 0);
    const int scale_values = type == GGML_TYPE_NVFP4 ? QK_NVFP4_SUB : QK_MXFP4;
    return (size_t) GGML_PAD(n_rows, 128) * (n_cols / scale_values);
}

static bool cutlass_quantize(
        const float * src,
        uint8_t * dst,
        uint8_t * scales,
        ggml_type type,
        int64_t n_cols,
        int64_t n_cols_padded,
        int64_t stride_row,
        int64_t n_rows,
        cudaStream_t stream) {
    if ((type != GGML_TYPE_MXFP4 && type != GGML_TYPE_NVFP4) ||
        n_cols <= 0 || n_cols % 2 != 0 ||
        n_cols_padded < n_cols || n_cols_padded % 128 != 0 ||
        stride_row < n_cols || stride_row % 2 != 0 ||
        n_rows <= 0 || n_rows > UINT_MAX) {
        return false;
    }

    constexpr int threads = 256;
    CUDA_CHECK(cudaMemsetAsync(scales, 0, cutlass_scale_size(type, n_rows, n_cols_padded), stream));
    if (type == GGML_TYPE_NVFP4) {
        cutlass_quantize_nvfp4<<<(unsigned) n_rows, threads, 0, stream>>>(
            src, dst, scales, n_cols, n_cols_padded, stride_row);
    } else {
        cutlass_quantize_mxfp8<<<(unsigned) n_rows, threads, 0, stream>>>(
            src, dst, scales, n_cols, n_cols_padded, stride_row);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

namespace ggml_cutlass_sm120 {

using namespace cute;

struct mxfp_format_traits {
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
    static constexpr int scale_granularity = QK_NVFP4_SUB;

    using Scale      = cutlass::float_ue4m3_t;
    using Activation = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using Weight     = Activation;

    template <typename Element>
    using MainloopElement = Element;

    template <typename Element>
    static constexpr int alignment = 32;
};

template <typename Format>
struct blockscaled_kernel_traits {
    using Output           = float;
    using ElementA         = typename Format::Activation;
    using ElementB         = typename Format::Weight;
    using ElementAMainloop = typename Format::template MainloopElement<ElementA>;
    using ElementBMainloop = typename Format::template MainloopElement<ElementB>;
    using LayoutA          = cutlass::layout::RowMajor;
    using LayoutB          = cutlass::layout::ColumnMajor;
    using LayoutD          = cutlass::layout::RowMajor;
    using TileShape        = Shape<_128, _128, _128>;
    using ClusterShape     = Shape<_1, _1, _1>;

    static constexpr int alignment_a = Format::template alignment<ElementA>;
    static constexpr int alignment_b = Format::template alignment<ElementB>;
    static constexpr int alignment_d = 128 / cutlass::sizeof_bits<Output>::value;

    using EpilogueBuilder = cutlass::epilogue::collective::CollectiveBuilder<
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
        cutlass::epilogue::collective::EpilogueScheduleAuto>;
    using CollectiveEpilogue = typename EpilogueBuilder::CollectiveOp;
    using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>;
    using MainloopBuilder = cutlass::gemm::collective::CollectiveBuilder<
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
        cutlass::gemm::collective::KernelScheduleAuto>;
    using CollectiveMainloop = typename MainloopBuilder::CollectiveOp;
    using ProblemShape = Shape<int, int, int, int>;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using StrideC = typename GemmKernel::StrideC;
    using StrideD = typename GemmKernel::StrideD;
    using Scale = typename Format::Scale;
    using BlockScaleConfig = typename CollectiveMainloop::Sm1xxBlkScaledConfig;

    static_assert(CollectiveMainloop::TiledMma::SFVecSize == Format::scale_granularity);
};

template <typename Traits>
static bool run_dense_gemm(
        ggml_backend_cuda_context & ctx,
        const ggml_cuda_cutlass_weight & weight,
        const uint8_t * activation,
        const uint8_t * activation_scales,
        void * dst,
        int m,
        int n,
        int k,
        cudaStream_t stream) {
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
            stride_a,
            reinterpret_cast<const typename Gemm::ElementB *>(weight.values),
            stride_b,
            reinterpret_cast<const typename Traits::Scale *>(activation_scales),
            layout_sfa,
            reinterpret_cast<const typename Traits::Scale *>(weight.scales),
            layout_sfb,
        },
        {
            {},
            nullptr,
            stride_c,
            reinterpret_cast<typename Traits::Output *>(dst),
            stride_d,
        },
    };
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta  = 0.0f;

    Gemm gemm;
    const cutlass::Status can_implement = gemm.can_implement(arguments);
    if (can_implement != cutlass::Status::kSuccess) {
        GGML_LOG_ERROR("%s: can_implement failed: %s\n", __func__, cutlassGetStatusString(can_implement));
        return false;
    }

    const size_t workspace_size = Gemm::get_workspace_size(arguments);
    ggml_cuda_pool_alloc<char> workspace_alloc(ctx.pool());
    void * workspace = workspace_size == 0 ? nullptr : workspace_alloc.alloc(workspace_size);
    const cutlass::Status initialize = gemm.initialize(arguments, workspace);
    if (initialize != cutlass::Status::kSuccess) {
        GGML_LOG_ERROR("%s: initialize failed: %s\n", __func__, cutlassGetStatusString(initialize));
        return false;
    }

    const cutlass::Status run = gemm.run(stream);
    if (run != cutlass::Status::kSuccess) {
        GGML_LOG_ERROR("%s: run failed: %s\n", __func__, cutlassGetStatusString(run));
        return false;
    }
    return true;
}

static bool dispatch_dense_gemm(
        ggml_backend_cuda_context & ctx,
        const ggml_cuda_cutlass_weight & weight,
        const uint8_t * activation,
        const uint8_t * activation_scales,
        void * dst,
        int m,
        int n,
        int k,
        cudaStream_t stream) {
    if (weight.type == GGML_TYPE_MXFP4) {
        return run_dense_gemm<blockscaled_kernel_traits<mxfp_format_traits>>(
            ctx, weight, activation, activation_scales, dst, m, n, k, stream);
    }
    if (weight.type == GGML_TYPE_NVFP4) {
        return run_dense_gemm<blockscaled_kernel_traits<nvfp4_format_traits>>(
            ctx, weight, activation, activation_scales, dst, m, n, k, stream);
    }
    return false;
}

} // namespace ggml_cutlass_sm120

bool ggml_cuda_cutlass_compiled() {
    return true;
}

bool ggml_cuda_cutlass_mul_mat(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    using namespace ggml_cutlass_sm120;

    ggml_cuda_cutlass_weight weight;
    if (!ggml_cuda_cutlass_mul_mat_supported(src0, src1, dst) ||
        !ggml_cuda_cutlass_weight_from_tensor(src0, weight) ||
        src1->buffer == nullptr || dst->buffer == nullptr) {
        return false;
    }

    const auto & device_info = ggml_cuda_info().devices[ctx.device];
    if (!blackwell_mma_available(device_info.cc)) {
        return false;
    }

    const ggml_backend_buffer_type_t buffer_type = ggml_backend_cuda_buffer_type(ctx.device);
    if (ggml_backend_buffer_get_type(src1->buffer) != buffer_type ||
        ggml_backend_buffer_get_type(dst->buffer) != buffer_type) {
        return false;
    }

    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t m = ggml_nelements(src1) / k;
    const int64_t k_padded = weight.k;
    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<uint8_t> activation(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> activation_scales(ctx.pool());
    uint8_t * activation_data = activation.alloc(cutlass_activation_size(src0->type, m, k_padded));
    uint8_t * scale_data = activation_scales.alloc(cutlass_scale_size(src0->type, m, k_padded));
    if (!cutlass_quantize(
            (const float *) src1->data,
            activation_data,
            scale_data,
            src0->type,
            k,
            k_padded,
            src1->nb[1] / sizeof(float),
            m,
            stream)) {
        return false;
    }

    return dispatch_dense_gemm(
        ctx, weight, activation_data, scale_data, dst->data,
        (int) m, (int) n, (int) k_padded, stream);
}

#else

bool ggml_cuda_cutlass_compiled() {
    return false;
}

bool ggml_cuda_cutlass_mul_mat(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, src0, src1, dst);
    return false;
}

#endif
