#pragma once

#include <cstdlib>
#include <memory>
#include <optional>
#include <openvino/core/node.hpp>
#include <openvino/runtime/tensor.hpp>
#include <string>
#include "ggml.h"

// ExtraQuantType enum - defines requantization target formats
enum class ExtraQuantType { F16, Q4_0_C, Q8_1_C, Q4_0_128, Q8_0_C, Q8_0_32 };

// =====================================================
// Global Device Configuration (singleton)
// =====================================================
// Initialized once during backend init from GGML_OPENVINO_DEVICE env var

struct ggml_openvino_device_config {
    std::string device_name = "CPU";
    bool is_npu = false;
    bool initialized = false;

    void init() {
        if (initialized) return;
        const char* env = std::getenv("GGML_OPENVINO_DEVICE");
        if (env) {
            device_name = env;
            is_npu = (device_name == "NPU");
        }
        initialized = true;
    }
};

// Get the global device config singleton
inline ggml_openvino_device_config& ggml_openvino_get_device_config() {
    static ggml_openvino_device_config config;
    return config;
}

// Initialize device config (call during backend init)
inline void ggml_openvino_init_device_config() {
    ggml_openvino_get_device_config().init();
}

// Get the device name
inline const std::string& ggml_openvino_get_device_name() {
    return ggml_openvino_get_device_config().device_name;
}

// Check if running on NPU
inline bool ggml_openvino_is_npu() {
    return ggml_openvino_get_device_config().is_npu;
}

// Get requantization type for a tensor type (returns nullopt if no requant needed)
inline std::optional<ExtraQuantType> ggml_openvino_get_requant_type(ggml_type type) {
    if (!ggml_openvino_is_npu()) {
        return std::nullopt;
    }
    // NPU requantization rules
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_K:
            return ExtraQuantType::Q4_0_128;
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q5_K:
            return ExtraQuantType::F16;
        default:
            return std::nullopt;
    }
}

// =====================================================
// OpenVINO Tensor Extra Types
// =====================================================
// These types are stored in tensor->extra by the OpenVINO backend buffer.
// They allow:
// 1. Pre-built ov::Constant nodes for weights (avoiding memcpy during graph construction)
// 2. ov::Tensor wrappers for KV cache / compute tensors (for direct use with infer_request)

// Base class for OpenVINO tensor extra data
struct ggml_openvino_extra_base {
    enum class Type { WEIGHT, QUANTIZED_WEIGHT, TENSOR };
    Type type;
    virtual ~ggml_openvino_extra_base() = default;
protected:
    explicit ggml_openvino_extra_base(Type t) : type(t) {}
};

// Extra data for F16/F32/BF16 weight tensors - stores the pre-built ov::Constant node
struct ggml_openvino_weight_extra : public ggml_openvino_extra_base {
    std::shared_ptr<ov::Node> constant;  // Pre-built OpenVINO Constant node

    explicit ggml_openvino_weight_extra(std::shared_ptr<ov::Node> c)
        : ggml_openvino_extra_base(Type::WEIGHT), constant(std::move(c)) {}
};

// Extra data for quantized weight tensors - stores extracted weights/scales/biases and ov::Constant
struct ggml_openvino_quantized_weight_extra : public ggml_openvino_extra_base {
    ov::Tensor weights;   // U4 or U8 extracted weights
    ov::Tensor scales;    // F16 scales
    ov::Tensor biases;    // F16 biases (zero points)
    std::shared_ptr<ov::Node> constant;  // Pre-built OpenVINO weight subgraph

    ggml_openvino_quantized_weight_extra(ov::Tensor w, ov::Tensor s, ov::Tensor b, std::shared_ptr<ov::Node> c)
        : ggml_openvino_extra_base(Type::QUANTIZED_WEIGHT),
          weights(std::move(w)), scales(std::move(s)), biases(std::move(b)), constant(std::move(c)) {}
};

// Extra data for KV cache / compute tensors - stores ov::Tensor for infer_request
struct ggml_openvino_tensor_extra : public ggml_openvino_extra_base {
    std::shared_ptr<ov::Tensor> tensor;  // For direct use with infer_request

    explicit ggml_openvino_tensor_extra(std::shared_ptr<ov::Tensor> t)
        : ggml_openvino_extra_base(Type::TENSOR), tensor(std::move(t)) {}
};

// =====================================================
// Extracted Size Calculation for Quantized Tensors
// =====================================================
// For quantized tensors, we need extra space to store extracted weights, scales, and biases.
// Returns the total size needed in the buffer for extracted data.

struct ggml_openvino_extracted_layout {
    size_t total_size;        // Total bytes needed
    size_t weights_offset;    // Offset to weights in buffer
    size_t weights_size;      // Size of weights in bytes
    size_t scales_offset;     // Offset to scales in buffer
    size_t scales_size;       // Size of scales in bytes
    size_t biases_offset;     // Offset to biases in buffer
    size_t biases_size;       // Size of biases in bytes
    bool is_u4;               // true for U4 weights, false for U8
    int64_t weights_per_block;// weights per scale/bias block

    // Requantization info
    bool is_requant;                              // true if this tensor needs requantization
    std::optional<ExtraQuantType> requant_type;   // target requant type if is_requant
};

// Calculate the buffer layout for extracted quantized data
inline ggml_openvino_extracted_layout ggml_openvino_get_extracted_layout(const ggml_tensor * tensor) {
    ggml_openvino_extracted_layout layout = {};

    if (!ggml_is_quantized(tensor->type)) {
        return layout;
    }

    // Only handle 2D weight tensors
    if (tensor->ne[2] != 1 || tensor->ne[3] != 1) {
        return layout;
    }

    int64_t n_elements = ggml_nelements(tensor);
    const size_t alignment = 64;  // Good for SIMD

    // Check if requantization is needed (NPU-specific)
    auto requant_type = ggml_openvino_get_requant_type(tensor->type);
    if (requant_type.has_value()) {
        layout.is_requant = true;
        layout.requant_type = requant_type;

        // Special case: requant to F16 - just store F16 weights, no scales/biases
        if (requant_type.value() == ExtraQuantType::F16) {
            layout.weights_size = n_elements * sizeof(uint16_t);  // F16 = 2 bytes
            layout.total_size = layout.weights_size;
            layout.weights_offset = 0;
            // No scales/biases for F16
            return layout;
        }

        // Requant to different quantized format (e.g., Q4_0_128)
        switch (requant_type.value()) {
            case ExtraQuantType::Q4_0_128:
                layout.is_u4 = true;
                layout.weights_per_block = 128;
                break;
            case ExtraQuantType::Q8_0_32:
                layout.is_u4 = false;
                layout.weights_per_block = 32;
                break;
            default:
                // Unsupported requant type - fall through to normal extraction
                layout.is_requant = false;
                layout.requant_type = std::nullopt;
                break;
        }

        if (layout.is_requant) {
            // Calculate sizes for requantized format
            layout.weights_size = layout.is_u4 ? (n_elements / 2) : n_elements;
            int64_t n_blocks = n_elements / layout.weights_per_block;
            layout.scales_size = n_blocks * sizeof(uint16_t);
            layout.biases_size = n_blocks * sizeof(uint16_t);

            layout.weights_offset = 0;
            layout.scales_offset = ((layout.weights_size + alignment - 1) / alignment) * alignment;
            layout.biases_offset = layout.scales_offset + ((layout.scales_size + alignment - 1) / alignment) * alignment;
            layout.total_size = layout.biases_offset + layout.biases_size;
            layout.total_size = std::max(layout.total_size, ggml_nbytes(tensor));
            return layout;
        }
    }

    // Normal extraction (no requant) - determine format based on tensor type
    switch (tensor->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_K:
            layout.is_u4 = true;
            layout.weights_per_block = 32;
            break;
        case GGML_TYPE_Q8_0:
            layout.is_u4 = false;
            layout.weights_per_block = 32;
            break;
        case GGML_TYPE_Q6_K:
            layout.is_u4 = false;
            layout.weights_per_block = 16;
            break;
        case GGML_TYPE_Q5_K:
            layout.is_u4 = false;
            layout.weights_per_block = 32;
            break;
        default:
            // Unsupported quantization type
            return layout;
    }

    // Calculate sizes
    // Weights: U4 = n_elements/2 bytes, U8 = n_elements bytes
    layout.weights_size = layout.is_u4 ? (n_elements / 2) : n_elements;

    // Scales and biases: F16 per block
    int64_t n_blocks = n_elements / layout.weights_per_block;
    layout.scales_size = n_blocks * sizeof(uint16_t);  // F16 = 2 bytes
    layout.biases_size = n_blocks * sizeof(uint16_t);  // F16 = 2 bytes

    // Layout in buffer: [weights | scales | biases] with alignment
    layout.weights_offset = 0;
    layout.scales_offset = ((layout.weights_size + alignment - 1) / alignment) * alignment;
    layout.biases_offset = layout.scales_offset + ((layout.scales_size + alignment - 1) / alignment) * alignment;
    layout.total_size = layout.biases_offset + layout.biases_size;

    return layout;
}
