#include "npu_quant_policy.h"

#include "ggml-impl.h"
#include "ggml.h"

#include <cstring>
#include <mutex>
#include <string>

ExtraQuantType ggml_openvino_get_npu_requant_type() {
    const std::string policy = ggml_openvino_getenv_str("GGML_OPENVINO_NPU_REQUANT_POLICY", "group-128");
    if (policy == "group-128") {
        return ExtraQuantType::Q4_0_128;
    }
    if (policy == "channel-wise") {
        return ExtraQuantType::Q4_0_C;
    }
    GGML_ABORT("Unknown GGML_OPENVINO_NPU_REQUANT_POLICY: %s", policy.c_str());
}

std::optional<ExtraQuantType> ggml_openvino_npu_token_embd_requant_type(const ggml_tensor * tensor) {
    // NPU widens a Q6_K embedding table to f16, which doubles its footprint versus the
    // int8 form used everywhere else. GGML_OPENVINO_TOKEN_EMBD_I8 opts back into int8, and
    // GGML_OPENVINO_TOKEN_EMBD_I4 into group-128 int4 (only worth it for a tied embedding,
    // where the table is also the lm_head weight - see gather_compressed_rows).
    if (tensor->type == GGML_TYPE_Q6_K) {
        // q4_0_c: per-channel u4, q4_0_128: group 128, default: F16
        if (const char * q = ggml_openvino_getenv_str("GGML_OPENVINO_NPU_TOKEN_EMBD_QUANT")) {
            ExtraQuantType type = ExtraQuantType::F16;
            if (strcmp(q, "q4_0_c") == 0) {
                type = ExtraQuantType::Q4_0_C;
            } else if (strcmp(q, "q4_0_128") == 0) {
                type = ExtraQuantType::Q4_0_128;
            }
            static std::once_flag logged;
            std::call_once(logged, [&] {
                GGML_LOG_INFO("ggml-openvino: NPU token_embd.weight requant type: %s",
                              (type == ExtraQuantType::F16 ? "F16" : q));
            });
            return type;
        }
        if (ggml_openvino_getenv_int("GGML_OPENVINO_TOKEN_EMBD_I4")) {
            return ExtraQuantType::Q4_0_128;
        }
        if (ggml_openvino_getenv_int("GGML_OPENVINO_TOKEN_EMBD_I8") == 0) {
            return ExtraQuantType::F16;
        }
    }
    return std::nullopt;
}

std::optional<ExtraQuantType> ggml_openvino_npu_weight_requant_type(const ggml_tensor * tensor) {
    // Q4_0 is already u4, so regrouping it to 128-element blocks only saves scale storage
    // while round-tripping the whole tensor through f32. Taking the direct extract path
    // instead does compile, but the resulting group-32 graph hangs the NPU at inference
    // (ZE_RESULT_ERROR_DEVICE_LOST, driver TDR), so it stays off by default.
    if (tensor->type == GGML_TYPE_Q4_0 && ggml_openvino_getenv_int("GGML_OPENVINO_NPU_KEEP_Q4_0")) {
        return std::nullopt;
    }
    return ggml_openvino_get_npu_requant_type();
}
