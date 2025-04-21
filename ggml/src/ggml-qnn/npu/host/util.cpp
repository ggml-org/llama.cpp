#include "util.hpp"

#include <remote.h>

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
        default:
            return NPU_OP_COUNT;
    }
}

enum npu_device_tensor_data_type type_to_npu_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return NPU_DATA_TYPE_F32;
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

}  // namespace hexagon
