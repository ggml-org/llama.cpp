

#include "op_impl.hpp"

#include <hexagon_types.h>
#include <HTP/core/intrinsics.h>

#include "op_mul_mat.hpp"

namespace {

template <HVX_Vector (*_OpIntrinsic)(HVX_Vector, HVX_Vector)>
inline void vec_op_f32_f32(const float * src0, const float * src1, size_t count, float * dst) {
    HVX_Vector * iptr0     = ((HVX_Vector *) src0);
    HVX_Vector * iptr0_end = ((HVX_Vector *) src0) + (count / hexagon::kFloatsPerVector);
    HVX_Vector * iptr1     = ((HVX_Vector *) src1);
    HVX_Vector * optr      = ((HVX_Vector *) dst);
    HVX_Vector   prev0     = *iptr0++;
    HVX_Vector   prev1     = *iptr1++;

    // TODO: prefetch or just use VTCM?
    while (iptr0 < iptr0_end) {
        HVX_Vector curr0 = *iptr0++;
        HVX_Vector curr1 = *iptr1++;
        HVX_Vector s0    = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);
        HVX_Vector s1    = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);
        *optr++          = Q6_Vsf_equals_Vqf32(_OpIntrinsic(s0, s1));
        prev0            = curr0;
        prev1            = curr1;
    }

    if ((iptr0_end - ((HVX_Vector *) src0)) > 0) {
        // handle the last vector
        // see also:
        //   https://github.com/UbiquitousLearning/mllm/blob/babf4410352ce8730824c87699c025a0d4ce3a6f/src/backends/qnn/LLaMAOpPackageHtp/LLaMAPackage/src/ops/LLaMAMul.cpp#L147
        //   or qualcomm sdk libs\qhl_hvx\src\qhblas_hvx\qhblas_hvx_aw_vector_add_ah.c
        bool       iptr0_aligned = hexagon::is_addr_aligned(iptr0);
        HVX_Vector curr0         = iptr0_aligned ? prev0 : *iptr0;
        iptr0                    = iptr0_aligned ? iptr0 : iptr0 + 1;
        bool       iptr1_aligned = hexagon::is_addr_aligned(iptr1);
        HVX_Vector curr1         = iptr1_aligned ? prev1 : *iptr1;
        iptr1                    = iptr1_aligned ? iptr1 : iptr1 + 1;
        HVX_Vector s0            = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);
        HVX_Vector s1            = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);
        *optr++                  = Q6_Vsf_equals_Vqf32(_OpIntrinsic(s0, s1));
        prev0                    = curr0;
        prev1                    = curr1;
    }

    const size_t leftover       = count % hexagon::kFloatsPerVector;
    const size_t leftover_bytes = leftover * sizeof(float);
    if (leftover > 0) {
        // handle the leftover elements
        HVX_Vector curr0 =
            (leftover_bytes + hexagon::unaligned_bytes(iptr0) > hexagon::kBytesPerVector) ? *iptr0 : prev0;
        curr0 = Q6_V_valign_VVR(curr0, prev0, (size_t) src0);

        HVX_Vector curr1 =
            (leftover_bytes + hexagon::unaligned_bytes(iptr1) > hexagon::kBytesPerVector) ? *iptr1 : prev1;
        curr1 = Q6_V_valign_VVR(curr1, prev1, (size_t) src1);

        q6op_vstu_variable_ARV(optr, leftover_bytes, Q6_Vsf_equals_Vqf32(_OpIntrinsic(curr0, curr1)));
    }
}

inline HVX_Vector vadd_f32_f32(HVX_Vector a, HVX_Vector b) {
    return Q6_Vqf32_vadd_VsfVsf(a, b);
}

inline HVX_Vector vsub_f32_f32(HVX_Vector a, HVX_Vector b) {
    return Q6_Vqf32_vsub_VsfVsf(a, b);
}

inline HVX_Vector vmul_f32_f32(HVX_Vector a, HVX_Vector b) {
    return Q6_Vqf32_vmpy_VsfVsf(a, b);
}

template <typename _TySrc, typename _TyDst, void (*_RowFunc)(const _TySrc *, const _TySrc *, size_t, _TyDst *)>
bool element_wise_op(hexagon::tensor * out) {
    if (!out) {
        return false;
    }

    auto * src0 = out->get_src(0);
    auto * src1 = out->get_src(1);
    if (!src0 || !src1) {
        return true;  // skip if no src
    }

    if (src0->get_ne(0) != src1->get_ne(0)) {
        // TODO: handle this case
        DEVICE_LOG_ERROR("src0[0] and src1[0] not match: %ld vs %ld\n", (long) src0->get_ne(0), (long) src1->get_ne(0));
        return false;
    }

    static_assert(DEVICE_TENSOR_MAX_DIMS == 4, "element_wise_op requires max dims 4");

    const auto * src0_ptr = reinterpret_cast<const uint8_t *>(src0->get_data());
    const auto * src1_ptr = reinterpret_cast<const uint8_t *>(src1->get_data());
    auto *       dst_ptr  = reinterpret_cast<uint8_t *>(out->get_data());
    for (int64_t i3 = 0; i3 < out->get_ne(3); i3++) {
        const auto * src0_cube = src0_ptr + i3 * src0->get_nb(3);
        const auto * src1_cube = src1_ptr + (i3 % src1->get_ne(3)) * src1->get_nb(3);
        auto *       dst_cube  = dst_ptr + i3 * out->get_nb(3);
        for (int64_t i2 = 0; i2 < out->get_ne(2); i2++) {
            const auto * src0_plane = src0_cube + i2 * src0->get_nb(2);
            const auto * src1_plane = src1_cube + (i2 % src1->get_ne(2)) * src1->get_nb(2);
            auto *       dst_plane  = dst_cube + i2 * out->get_nb(2);
            for (int64_t i1 = 0; i1 < out->get_ne(1); i1++) {
                // TODO: prefetch row?
                auto * src0_row = src0_plane + i1 * src0->get_nb(1);
                auto * src1_row = src1_plane + (i1 % src1->get_ne(1)) * src1->get_nb(1);
                auto * dst_row  = reinterpret_cast<float *>(dst_plane + i1 * out->get_nb(1));
                _RowFunc(reinterpret_cast<const _TySrc *>(src0_row), reinterpret_cast<const _TySrc *>(src1_row),
                         static_cast<size_t>(out->get_ne(0)), reinterpret_cast<_TyDst *>(dst_row));
            }
        }
    }

    return true;
}

bool is_element_wise_op_supported(const npu_device_tensor_spec & src0, const npu_device_tensor_spec & src1,
                                  const npu_device_tensor_spec & dst, npu_device_tensor_op op) {
    if (op != NPU_OP_ADD && op != NPU_OP_SUB && op != NPU_OP_MUL) {
        DEVICE_LOG_DEBUG("Unsupported element wise op: %s\n", hexagon::op_get_name(op));
        return false;
    }

    if (src0.ne[0] != src1.ne[0]) {
        DEVICE_LOG_DEBUG("src0.ne[0] and src1.ne[0] not match: %ld vs %ld\n", (long) src0.ne[0], (long) src1.ne[0]);
        return false;
    }

    for (size_t i = 0; i < DEVICE_TENSOR_MAX_DIMS; ++i) {
        if (src0.ne[i] != dst.ne[i]) {
            DEVICE_LOG_DEBUG("src0.ne[%zu] and dst.ne[%zu] not match: %lld vs %lld\n", i, i, (long long) src0.ne[i],
                             (long long) dst.ne[i]);
            return false;
        }
    }

    return true;
}

struct op_capabilities {
    npu_device_tensor_op               op;
    hexagon::compute_func_type         compute_func;
    hexagon::op_is_supported_func_type is_supported;
};

constexpr const op_capabilities kOpCapabilities[] = {
    { NPU_OP_MUL_MAT, hexagon::mul_mat_f32, hexagon::is_mul_mat_supported },
    { NPU_OP_ADD, element_wise_op<float, float, vec_op_f32_f32<vadd_f32_f32>>, is_element_wise_op_supported },
    { NPU_OP_SUB, element_wise_op<float, float, vec_op_f32_f32<vsub_f32_f32>>, is_element_wise_op_supported },
    { NPU_OP_MUL, element_wise_op<float, float, vec_op_f32_f32<vmul_f32_f32>>, is_element_wise_op_supported },
};

static_assert(kOpCapabilities[NPU_OP_MUL_MAT].compute_func == hexagon::mul_mat_f32,
              "kOpArray[NPU_OP_MUL_MAT] != mul_mat_f32");

static_assert(std::size(kOpCapabilities) == NPU_OP_COUNT);
static_assert(kOpCapabilities[NPU_OP_MUL_MAT].op == NPU_OP_MUL_MAT, "kOpArray[NPU_OP_MUL_MAT].op != NPU_OP_MUL_MAT");
static_assert(kOpCapabilities[NPU_OP_MUL].op == NPU_OP_MUL, "kOpArray[NPU_OP_MUL].op != NPU_OP_MUL");

}  // namespace

namespace hexagon {

compute_func_type get_compute_func(npu_device_tensor_op op) {
    if (op >= NPU_OP_COUNT) {
        return nullptr;
    }

    return kOpCapabilities[op].compute_func;
}

bool support_op(const npu_device_tensor_spec & src0, const npu_device_tensor_spec & src1,
                const npu_device_tensor_spec & dst, npu_device_tensor_op op) {
    if (get_compute_func(op) == nullptr) {
        DEVICE_LOG_ERROR("Unsupported op: %s, get_compute_func failed\n", op_get_name(op));
        return false;
    }

    auto is_supported_func = kOpCapabilities[op].is_supported;
    if (!is_supported_func || !is_supported_func(src0, src1, dst, op)) {
        DEVICE_LOG_ERROR("Unsupported op: %s, is_supported_func failed\n", op_get_name(op));
        return false;
    }

    return true;
}

}  // namespace hexagon
