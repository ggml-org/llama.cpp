#include "ggml-openvino-extra.h"

#include "ggml-impl.h"

ov::Core & ov_singleton_core() {
    static ov::Core core;
    return core;
}

// =====================================================
// Device Configuration Implementations
// =====================================================

void ggml_openvino_device_config::init() {
    if (initialized) {
        return;
    }
    device_name = getenv("GGML_OPENVINO_DEVICE") ? getenv("GGML_OPENVINO_DEVICE") : "CPU";
    auto available_devices = ov_singleton_core().get_available_devices();
    if (std::find(available_devices.begin(), available_devices.end(), device_name) == available_devices.end()) {
        GGML_LOG_WARN("GGML OpenVINO Backend: device %s is not available, fallback to CPU\n", device_name.c_str());
        device_name = "CPU";
    }
    is_npu = (device_name == "NPU");
    initialized = true;
}

// Get the global device config singleton
ggml_openvino_device_config & ggml_openvino_get_device_config() {
    static ggml_openvino_device_config config;
    return config;
}

// Initialize device config (call during backend init)
void ggml_openvino_init_device_config() {
    ggml_openvino_get_device_config().init();
}

// Get the device name
const std::string & ggml_openvino_get_device_name() {
    return ggml_openvino_get_device_config().device_name;
}

// Check if running on NPU
bool ggml_openvino_is_npu() {
    return ggml_openvino_get_device_config().is_npu;
}

// Get requantization type for a tensor type (returns nullopt if no requant needed)
std::optional<ExtraQuantType> ggml_openvino_get_requant_type(ggml_type type) {
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
// Extracted Layout Calculation
// =====================================================

ggml_openvino_extracted_layout ggml_openvino_get_extracted_layout(const ggml_tensor * tensor) {
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
            layout.biases_offset =
                layout.scales_offset + ((layout.scales_size + alignment - 1) / alignment) * alignment;
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
