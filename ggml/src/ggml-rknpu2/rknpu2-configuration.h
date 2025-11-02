#pragma once

#include "ggml.h"

#include <rknn_matmul_api.h>

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <cstdint>

namespace rknpu2_configuration {

// Forward declaration for packing function pointer
struct ggml_tensor;

/**
 * @brief Defines a function signature for packing a segment of the weight matrix (B)
 * from a standard row-major layout into the device-specific native layout.
 *
 * @param dst Pointer to the destination buffer for the packed data.
 * @param src Pointer to the source data in row-major layout (ggml format).
 * @param K The number of columns in the source matrix (inner dimension).
 * @param N_total The total number of rows in the source matrix.
 * @param n_offset The row offset from which to start packing.
 * @param n_segment The number of rows to pack.
 */
using PackingFunction = std::function<void(
    uint8_t* dst,
    const uint8_t* src,
    int K,
    int N_total,
    int n_offset,
    int n_segment
)>;

/**
 * @brief An abstract representation of data types used on the NPU side.
 * This helps decouple the main logic from specific ggml or rknn types.
 */
enum Rknpu2NpuType {
    NPU_TYPE_INT4,
    NPU_TYPE_INT8,
    NPU_TYPE_INT16,
    NPU_TYPE_INT32,
    NPU_TYPE_FP16,
    NPU_TYPE_FP32,
};

/**
 * @brief Describes a single supported matrix multiplication operation on the NPU.
 *
 * This struct maps a combination of ggml tensor types for weights (src0) and
 * activations (src1) to a specific RKNN matmul type. It also specifies the
 * hardware alignment requirements for the K and N dimensions of the matrices.
 */
struct Rknpu2Operation {
    ggml_type type_w;           // Weight tensor type
    ggml_type type_a;           // Activation tensor type
    
    Rknpu2NpuType npu_type_a;   // The target data type for activations on the NPU
    Rknpu2NpuType npu_type_c;   // The data type of the result from the NPU
    rknn_matmul_type mm_type;   // Corresponding RKNN operation type
    
    int k_align;                // Required alignment for the K dimension
    int n_align;                // Required alignment for the N dimension
    PackingFunction pack_func;  // Function to pack the weight matrix for this op
};

/**
 * @brief Holds the complete hardware configuration for a specific Rockchip NPU.
 *
 * This includes the device name, number of available NPU cores, and a list of
 * all matrix multiplication operations supported by the hardware.
 */
struct Rknpu2DeviceConfig {
    std::string device_name;
    int core_count;
    std::vector<Rknpu2Operation> supported_ops;

    /**
     * @brief Finds the corresponding operation configuration for the given weight type.
     * @param w_type The ggml_type of the weight tensor.
     * @return A pointer to the Rknpu2Operation if found, otherwise nullptr.
     */
    const Rknpu2Operation* find_op_support(ggml_type w_type) const {
        for (const auto& op : supported_ops) {
            if (op.type_w == w_type) {
                return &op;
            }
        }
        return nullptr;
    }
};

/**
 * @brief Manages and provides access to hardware configurations for different Rockchip NPUs.
 *
 * This is a singleton class that should be initialized once. It allows selecting
 * a specific device configuration (e.g., for "RK3588") which can then be
 * accessed globally to drive backend logic.
 */
class Rknpu2ConfigManager {
public:
    /**
     * @brief Get the singleton instance of the manager.
     * @return Reference to the Rknpu2ConfigManager instance.
     */
    static Rknpu2ConfigManager& get_instance();

    /**
     * @brief Selects the active device configuration.
     * @param device_name The name of the device to activate (e.g., "RK3588").
     * @return True if the configuration was found and set, false otherwise.
     */
    bool select_device(const std::string& device_name);

    /**
     * @brief Get the currently active device configuration.
     * @return A constant reference to the active Rknpu2DeviceConfig.
     */
    const Rknpu2DeviceConfig& get_current_config() const;

private:
    // Private constructor, destructor, and copy operators to enforce singleton pattern
    Rknpu2ConfigManager();
    ~Rknpu2ConfigManager() = default;
    Rknpu2ConfigManager(const Rknpu2ConfigManager&) = delete;
    Rknpu2ConfigManager& operator=(const Rknpu2ConfigManager&) = delete;

    std::map<std::string, Rknpu2DeviceConfig> device_configs;
    const Rknpu2DeviceConfig* current_config = nullptr;
};

} // namespace rknpu2_configuration