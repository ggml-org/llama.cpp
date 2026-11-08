#pragma once
#include "ggml-openvino-extra.h"  // For ExtraQuantType
#include "ggml.h"

#include <openvino/core/node_output.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/runtime/tensor.hpp>

static constexpr size_t GGML_QUANTIZATION_GROUP_SIZE = 32;

inline const char * extra_quant_type_name(ExtraQuantType t) {
    switch (t) {
    case ExtraQuantType::F16:
        return "F16";
    case ExtraQuantType::Q4_0_C:
        return "Q4_0_C";
    case ExtraQuantType::Q4_0_128:
        return "Q4_0_128";
    case ExtraQuantType::Q8_0_C:
        return "Q8_0_C";
    case ExtraQuantType::Q8_0_32:
        return "Q8_0_32";
    case ExtraQuantType::Q8_1_C:
        return "Q8_1_C";
    case ExtraQuantType::Q4_0_64:
        return "Q4_0_64";
    case ExtraQuantType::Q4_1_64:
        return "Q4_1_64";
    default:
        return "unknown";
    }
}

// Result from process_weight_tensor containing the weight node and tensors.
// For quantized weights, also contains the extracted layout and scale/zp tensors.
struct OvWeight {
    std::shared_ptr<ov::Node> weight_node;
    ggml_openvino_extracted_layout layout;  // Only meaningful for quantized (layout.total_size > 0)
    ov::Tensor weights;
    ov::Tensor scales;
    ov::Tensor zp;

    bool is_quantized() const { return layout.scales_size > 0; }
};

// Process weight tensor and create an OpenVINO weight node
// Handles F16/F32/BF16 and quantized weights, with optional requantization
// If output_base_ptr is nullptr, allocates internal buffers (for decoder use)
// If output_base_ptr is provided, uses pre-allocated buffers at specified offsets (for backend buffer use)
// Returns OvWeight with the weight node and optional quantized tensors
OvWeight process_weight_tensor(
    const ggml_tensor * tensor,
    const void * data,                 // Source data pointer (may differ from tensor->data)
    void * output_base_ptr = nullptr,  // Base pointer for output buffers (or nullptr for internal allocation)
    bool use_bias = false);            // Use an exact f16 zero point (vs. a rounded integer one);
                                       // always used for for_gather_matmul (3D MoE expert) weights
                                       // regardless of this flag, and also settable explicitly for
                                       // test-backend-ops.
