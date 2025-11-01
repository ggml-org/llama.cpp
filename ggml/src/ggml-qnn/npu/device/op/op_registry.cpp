#include "op_registry.hpp"

#include "op_eltwise.hpp"
#include "op_flash_attn.hpp"
#include "op_glu.hpp"
#include "op_mul_mat.hpp"
#include "op_rope.hpp"
#include "op_rows.hpp"

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace {

struct op_capabilities {
    npu_device_tensor_op                op;
    hexagon::op_is_supported_func_type  is_supported;
    hexagon::op_required_sync_func_type requires_thread_barrier_func;
    hexagon::compute_func_type          compute_funcs[NPU_DATA_TYPE_COUNT];
};

constexpr const op_capabilities kOpCapabilities[] = {
    {
     NPU_OP_MUL_MAT,                            hexagon::is_mul_mat_supported,
     hexagon::is_mul_mat_required_sync,
     {
            hexagon::mul_mat_f32,  // NPU_DATA_TYPE_F32
            nullptr,               // NPU_DATA_TYPE_F16
        }, },
    {
     NPU_OP_ADD,                                        hexagon::is_element_wise_op_supported,
     hexagon::is_element_wise_op_required_sync,
     {
            hexagon::element_wise_op<hexagon::vec_op_f32_f32<hexagon::vadd_f32_f32>>,  // NPU_DATA_TYPE_F32
            hexagon::element_wise_op<hexagon::vec_op_f16_f16<hexagon::vadd_f16_f16>>,  // NPU_DATA_TYPE_F16
        }, },
    {
     NPU_OP_SUB, hexagon::is_element_wise_op_supported,
     hexagon::is_element_wise_op_required_sync,
     {
            hexagon::element_wise_op<hexagon::vec_op_f32_f32<hexagon::vsub_f32_f32>>,  // NPU_DATA_TYPE_F32
            hexagon::element_wise_op<hexagon::vec_op_f16_f16<hexagon::vsub_f16_f16>>,  // NPU_DATA_TYPE_F16
        }, },
    {
     NPU_OP_MUL,                                hexagon::is_element_wise_op_supported,
     hexagon::is_element_wise_op_required_sync,
     {
            hexagon::element_wise_op<hexagon::vec_op_f32_f32<hexagon::vmul_f32_f32>>,  // NPU_DATA_TYPE_F32
            hexagon::element_wise_op<hexagon::vec_op_f16_f16<hexagon::vmul_f16_f16>>,  // NPU_DATA_TYPE_F16
        }, },
    {
     NPU_OP_RMS_NORM,                                        hexagon::is_unary_op_supported,
     hexagon::is_unary_op_required_sync,
     {
            hexagon::unary_op<hexagon::rms_norm_vec_f32>,  // NPU_DATA_TYPE_F32
            nullptr,                                       // NPU_DATA_TYPE_F16
        }, },
    {
     NPU_OP_FLASH_ATTN, hexagon::is_flash_attn_supported,
     hexagon::is_flash_attn_required_sync,
     {
            hexagon::flash_attn_f32,  // NPU_DATA_TYPE_F32
            nullptr,                  // NPU_DATA_TYPE_F16
        }, },
    {
     NPU_OP_ROPE,                           hexagon::is_rope_supported,
     hexagon::is_rope_required_sync,
     {
            hexagon::rope_f32,  // NPU_DATA_TYPE_F32
            nullptr,            // NPU_DATA_TYPE_F16
        }, },
    {
     NPU_OP_GLU,                                        hexagon::is_glu_op_supported,
     hexagon::is_glu_required_sync,
     {
            hexagon::glu_f32,  // NPU_DATA_TYPE_F32
            hexagon::glu_f16,  // NPU_DATA_TYPE_F16
        }, },
    {
     NPU_OP_GET_ROWS,      hexagon::is_rows_supported,
     hexagon::is_rows_required_sync,
     {
            hexagon::get_rows_f32,  // NPU_DATA_TYPE_F32
            nullptr,                // NPU_DATA_TYPE_F16
        }, },
    {
     NPU_OP_SET_ROWS,                               hexagon::is_rows_supported,
     hexagon::is_rows_required_sync,
     {
            hexagon::set_rows_generic,  // NPU_DATA_TYPE_F32
            hexagon::set_rows_generic,  // NPU_DATA_TYPE_F16
            nullptr,                    // NPU_DATA_TYPE_I32
            nullptr,                    // NPU_DATA_TYPE_I64
            hexagon::set_rows_generic,  // NPU_DATA_TYPE_Q8_0
            hexagon::set_rows_generic,  // NPU_DATA_TYPE_Q4_0
            nullptr,                    // TODO: figure out why failed on NPU_DATA_TYPE_Q4_K
        }, },
    {
     NPU_OP_CPY,                                        hexagon::is_unary_op_supported,
     hexagon::is_unary_op_required_sync,
     {
            nullptr,                                                                     // NPU_DATA_TYPE_F32
            hexagon::unary_op<hexagon::unary_vec_op_f16_f32<hexagon::vequals_f16_f32>>,  // NPU_DATA_TYPE_F16
        }, },
};

static_assert(kOpCapabilities[NPU_OP_MUL_MAT].compute_funcs[NPU_DATA_TYPE_F32] == hexagon::mul_mat_f32,
              "kOpArray[NPU_OP_MUL_MAT] != mul_mat_f32");

static_assert(std::size(kOpCapabilities) == NPU_OP_COUNT);
static_assert(kOpCapabilities[NPU_OP_MUL_MAT].op == NPU_OP_MUL_MAT, "kOpArray[NPU_OP_MUL_MAT].op != NPU_OP_MUL_MAT");
static_assert(kOpCapabilities[NPU_OP_MUL].op == NPU_OP_MUL, "kOpArray[NPU_OP_MUL].op != NPU_OP_MUL");
static_assert(kOpCapabilities[NPU_OP_RMS_NORM].op == NPU_OP_RMS_NORM,
              "kOpArray[NPU_OP_RMS_NORM].op != NPU_OP_RMS_NORM");
static_assert(kOpCapabilities[NPU_OP_FLASH_ATTN].op == NPU_OP_FLASH_ATTN,
              "kOpArray[NPU_OP_FLASH_ATTN].op != NPU_OP_FLASH_ATTN");
static_assert(kOpCapabilities[NPU_OP_ROPE].op == NPU_OP_ROPE, "kOpArray[NPU_OP_ROPE].op != NPU_OP_ROPE");
static_assert(kOpCapabilities[NPU_OP_GLU].op == NPU_OP_GLU, "kOpArray[NPU_OP_GLU].op != NPU_OP_GLU");
static_assert(kOpCapabilities[NPU_OP_GET_ROWS].op == NPU_OP_GET_ROWS,
              "kOpArray[NPU_OP_GET_ROWS].op != NPU_OP_GET_ROWS");
static_assert(kOpCapabilities[NPU_OP_SET_ROWS].op == NPU_OP_SET_ROWS,
              "kOpArray[NPU_OP_SET_ROWS].op != NPU_OP_SET_ROWS");

hexagon::compute_func_type get_compute_func_impl(npu_device_tensor_op op, npu_device_tensor_data_type type) {
    if (op >= NPU_OP_COUNT) {
        return nullptr;
    }

    return kOpCapabilities[op].compute_funcs[type];
}

}  // namespace

namespace hexagon {

compute_func_type get_compute_func(tensor * dst) {
    return get_compute_func_impl(dst->get_op(), dst->get_type());
}

bool requires_thread_barrier(npu_device_tensor_op       prev_op,
                             const npu_device_ne_type & prev_ne,
                             npu_device_tensor_op       op,
                             const npu_device_ne_type & ne) {
    if (op >= NPU_OP_COUNT) {
        return false;
    }

    auto requires_thread_barrier_func = kOpCapabilities[op].requires_thread_barrier_func;
    return requires_thread_barrier_func && requires_thread_barrier_func(prev_op, prev_ne, op, ne);
}

bool support_op(const npu_device_tensor_op_spec * op_spec,
                const npu_device_tensor_spec *    dst,
                const npu_device_tensor_spec *    srcs,
                size_t                            src_len) {
    if (!op_spec) {
        DEVICE_LOG_ERROR("[hexagon-npu]invalid op_spec\n");
        return false;
    }

    const auto op                = op_spec->op;
    auto       is_supported_func = kOpCapabilities[op].is_supported;
    if (!is_supported_func || !is_supported_func(op_spec, dst, srcs, src_len)) {
        DEVICE_LOG_DEBUG("[%s]unsupported, is_supported_func return false\n", op_get_name(op));
        return false;
    }

    if (get_compute_func_impl(op, dst->type) == nullptr) {
        DEVICE_LOG_DEBUG("[%s]unsupported, get_compute_func failed, type: %s\n", op_get_name(op),
                         get_type_name(dst->type));
        return false;
    }

    return true;
}

}  // namespace hexagon
