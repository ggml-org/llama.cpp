#pragma once

// NPU weight requantization rules. ggml_openvino_get_requant_type() (ggml-openvino-extra.cpp) delegates
// the NPU-specific decisions here.

#include "ggml-openvino/ggml-openvino-extra.h"

#include <optional>

struct ggml_tensor;

// Layout selected by GGML_OPENVINO_NPU_REQUANT_POLICY (group-128 | channel-wise).
ExtraQuantType ggml_openvino_get_npu_requant_type();

// token_embd.weight on NPU. nullopt: no NPU-specific rule, use the common default.
std::optional<ExtraQuantType> ggml_openvino_npu_token_embd_requant_type(const ggml_tensor * tensor);

// Any other weight on NPU. nullopt: keep the GGUF layout (no requantization).
std::optional<ExtraQuantType> ggml_openvino_npu_weight_requant_type(const ggml_tensor * tensor);
