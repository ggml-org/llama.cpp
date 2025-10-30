#include "util.hpp"

#include <remote.h>

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#undef GGML_COMMON_DECL_CPP

static_assert(sizeof(npu_device_block_q4_k) == sizeof(block_q4_K), "npu_device_block_q4_k size mismatch");
static_assert(sizeof(npu_device_block_q4_0) == sizeof(block_q4_0), "npu_device_block_q4_0 size mismatch");
static_assert(sizeof(npu_device_block_q8_0) == sizeof(block_q8_0), "npu_device_block_q8_0 size mismatch");
static_assert(QUANT_K_SCALE_SIZE == K_SCALE_SIZE, "QUANT_K_SCALE_SIZE size mismatch");
static_assert(QUANT_K_BLOCK_SIZE == QK_K, "QUANT_K_BLOCK_SIZE size mismatch");
static_assert(QUANT_BLOCK_SIZE == QK4_0, "QUANT_BLOCK_SIZE size mismatch");

static_assert(NPU_ROPE_TYPE_NEOX == GGML_ROPE_TYPE_NEOX, "NPU_ROPE_TYPE_NEOX mismatch");
static_assert(NPU_ROPE_TYPE_MROPE == GGML_ROPE_TYPE_MROPE, "NPU_ROPE_TYPE_MROPE mismatch");
static_assert(NPU_ROPE_TYPE_VISION == GGML_ROPE_TYPE_VISION, "NPU_ROPE_TYPE_VISION mismatch");

namespace hexagon {

enum npu_device_tensor_op op_to_npu_op(ggml_op op) {
    switch (op) {
        case GGML_OP_MUL_MAT:
            return NPU_OP_MUL_MAT;
        case GGML_OP_ADD:
            return NPU_OP_ADD;
        case GGML_OP_SUB:
            return NPU_OP_SUB;
        case GGML_OP_MUL:
            return NPU_OP_MUL;
        case GGML_OP_RMS_NORM:
            return NPU_OP_RMS_NORM;
        case GGML_OP_FLASH_ATTN_EXT:
            return NPU_OP_FLASH_ATTN;
        case GGML_OP_ROPE:
            return NPU_OP_ROPE;
        case GGML_OP_GLU:
            return NPU_OP_GLU;
        case GGML_OP_GET_ROWS:
            return NPU_OP_GET_ROWS;
        case GGML_OP_SET_ROWS:
            return NPU_OP_SET_ROWS;
        case GGML_OP_CPY:
            return NPU_OP_CPY;
        default:
            return NPU_OP_COUNT;
    }
}

const char * get_npu_op_desc(enum npu_device_tensor_op op) {
    switch (op) {
        case NPU_OP_MUL_MAT:
            return ggml_op_name(GGML_OP_MUL_MAT);
        case NPU_OP_ADD:
            return ggml_op_name(GGML_OP_ADD);
        case NPU_OP_SUB:
            return ggml_op_name(GGML_OP_SUB);
        case NPU_OP_MUL:
            return ggml_op_name(GGML_OP_MUL);
        case NPU_OP_RMS_NORM:
            return ggml_op_name(GGML_OP_RMS_NORM);
        case NPU_OP_FLASH_ATTN:
            return ggml_op_name(GGML_OP_FLASH_ATTN_EXT);
        case NPU_OP_ROPE:
            return ggml_op_name(GGML_OP_ROPE);
        case NPU_OP_GLU:
            return ggml_op_name(GGML_OP_GLU);
        case NPU_OP_GET_ROWS:
            return ggml_op_name(GGML_OP_GET_ROWS);
        case NPU_OP_SET_ROWS:
            return ggml_op_name(GGML_OP_SET_ROWS);
        case NPU_OP_CPY:
            return ggml_op_name(GGML_OP_CPY);
        default:
            return "UNKNOWN";
    }
}

enum npu_device_tensor_data_type type_to_npu_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return NPU_DATA_TYPE_F32;
        case GGML_TYPE_F16:
            return NPU_DATA_TYPE_F16;
        case GGML_TYPE_I32:
            return NPU_DATA_TYPE_I32;
        case GGML_TYPE_I64:
            return NPU_DATA_TYPE_I64;
        case GGML_TYPE_Q4_K:
            return NPU_DATA_TYPE_Q4_K;
        case GGML_TYPE_Q4_0:
            return NPU_DATA_TYPE_Q4_0;
        case GGML_TYPE_Q8_0:
            return NPU_DATA_TYPE_Q8_0;
        default:
            return NPU_DATA_TYPE_COUNT;
    }
}

hexagon_dsp_arch get_dsp_arch(common::rpc_interface_ptr rpc_interface, uint32_t domain_id) {
    if (!rpc_interface || !rpc_interface->is_valid()) {
        return NONE;
    }

    remote_dsp_capability dsp_caps = {};
    dsp_caps.domain                = domain_id;
    dsp_caps.attribute_ID          = ARCH_VER;
    auto ret = rpc_interface->remote_handle_control(DSPRPC_GET_DSP_INFO, &dsp_caps, sizeof(dsp_caps));
    if (ret != AEE_SUCCESS) {
        LOG_ERROR("failed to get DSP arch: %d\n", ret);
        return NONE;
    }

    LOG_DEBUG("get DSP arch: 0x%x\n", (int) dsp_caps.capability);
    auto arch = dsp_caps.capability & 0xFF;
    switch (arch) {
        case 0x68:
            return V68;
        case 0x69:
            return V69;
        case 0x73:
            return V73;
        case 0x75:
            return V75;
        case 0x79:
            return V79;
        default:
            LOG_ERROR("unknown DSP arch: %x\n", arch);
            return NONE;
    }
}

const char * get_dsp_arch_desc(hexagon_dsp_arch arch) {
    switch (arch) {
        case V68:
            return "V68";
        case V69:
            return "V69";
        case V73:
            return "V73";
        case V75:
            return "V75";
        case V79:
            return "V79";
        case NONE:
        default:
            return "UnknownArch";
    }
}

void enable_unsigned_dsp_module(common::rpc_interface_ptr rpc_interface, uint32_t domain_id) {
    if (!rpc_interface || !rpc_interface->is_valid()) {
        return;
    }

    remote_rpc_control_unsigned_module data = {};
    data.domain                             = domain_id;
    data.enable                             = 1;
    auto ret = rpc_interface->remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &data, sizeof(data));
    if (ret != AEE_SUCCESS) {
        LOG_ERROR("failed to enable unsigned DSP module: 0x%x\n", ret);
    }
}

void set_fast_rpc_stack_size(common::rpc_interface_ptr rpc_interface, uint32_t domain_id, uint32_t stack_size) {
    constexpr const uint32_t FASTRPC_THREAD_PARAMS = 1;

    if (!rpc_interface || !rpc_interface->is_valid()) {
        return;
    }

    remote_rpc_thread_params tp = {};
    tp.domain                   = domain_id;
    tp.prio                     = -1;
    tp.stack_size               = stack_size;
    auto ret                    = rpc_interface->remote_session_control(FASTRPC_THREAD_PARAMS, &tp, sizeof(tp));
    if (ret != AEE_SUCCESS) {
        LOG_ERROR("failed to set fast RPC stack size: 0x%x\n", ret);
    }
}

void get_op_tensor_desc(const ggml_tensor * dst, char * out, size_t max_len) {
    if (dst == nullptr) {
        snprintf(out, max_len, "null");
        return;
    }

    constexpr const auto print_tensor = [](const ggml_tensor * tensor, char * out, size_t max_len) {
        auto dims = ggml_n_dims(tensor);

        switch (dims) {
            default:
            case 4:
                snprintf(out, max_len, "%s[%ldx%ldx%ldx%ld]", ggml_type_name(tensor->type), (long) tensor->ne[0],
                         (long) tensor->ne[1], (long) tensor->ne[2], (long) tensor->ne[3]);
                break;
            case 3:
                snprintf(out, max_len, "%s[%ldx%ldx%ld]", ggml_type_name(tensor->type), (long) tensor->ne[0],
                         (long) tensor->ne[1], (long) tensor->ne[2]);
                break;
            case 2:
                snprintf(out, max_len, "%s[%ldx%ld]", ggml_type_name(tensor->type), (long) tensor->ne[0],
                         (long) tensor->ne[1]);
                break;
            case 1:
                snprintf(out, max_len, "%s[%ld]", ggml_type_name(tensor->type), (long) tensor->ne[0]);
                break;
        }
    };

    constexpr const auto get_src_tensor_count = [](const ggml_tensor * tensor) -> size_t {
        for (size_t i = 0; i < GGML_MAX_SRC; ++i) {
            if (!tensor->src[i]) {
                return i;
            }
        }

        return GGML_MAX_SRC;
    };

    char dst_desc[256];
    print_tensor(dst, dst_desc, sizeof(dst_desc));
    switch (get_src_tensor_count(dst)) {
        case 4:
            {
                char src0_desc[256];
                print_tensor(dst->src[0], src0_desc, sizeof(src0_desc));
                char src1_desc[256];
                print_tensor(dst->src[1], src1_desc, sizeof(src1_desc));
                char src2_desc[256];
                print_tensor(dst->src[2], src2_desc, sizeof(src2_desc));
                char src3_desc[256];
                print_tensor(dst->src[3], src3_desc, sizeof(src3_desc));
                snprintf(out, max_len, "dst: %s, src0: %s, src1: %s, src2: %s, src3: %s", dst_desc, src0_desc,
                         src1_desc, src2_desc, src3_desc);
                return;
            }
        case 3:
            {
                char src0_desc[256];
                print_tensor(dst->src[0], src0_desc, sizeof(src0_desc));
                char src1_desc[256];
                print_tensor(dst->src[1], src1_desc, sizeof(src1_desc));
                char src2_desc[256];
                print_tensor(dst->src[2], src2_desc, sizeof(src2_desc));
                snprintf(out, max_len, "dst: %s, src0: %s, src1: %s, src2: %s", dst_desc, src0_desc, src1_desc,
                         src2_desc);
                return;
            }
        case 2:
            {
                char src0_desc[256];
                print_tensor(dst->src[0], src0_desc, sizeof(src0_desc));
                char src1_desc[256];
                print_tensor(dst->src[1], src1_desc, sizeof(src1_desc));
                snprintf(out, max_len, "dst: %s, src0: %s, src1: %s", dst_desc, src0_desc, src1_desc);
                return;
            }
        case 1:
            {
                char src0_desc[256];
                print_tensor(dst->src[0], src0_desc, sizeof(src0_desc));
                snprintf(out, max_len, "dst: %s, src0: %s", dst_desc, src0_desc);
                return;
            }
        default:
            snprintf(out, max_len, "dst: %s", dst_desc);
            return;
    }
}

}  // namespace hexagon
