#include "ggml-impl.h"

#include "rknpu2-configuration.h"

#include <arm_neon.h>
#include <cstdlib>
#include <sstream>

namespace {
    // Function for parsing ENV variable
    std::vector<std::string> split_string(const std::string& str, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(str);
        while (std::getline(tokenStream, token, delimiter)) {
            if(!token.empty()) tokens.push_back(token);
        }
        return tokens;
    }

    // Function for parsing complex integer lists
    std::vector<int> parse_int_list(const std::string& str) {
        std::vector<int> result;
        if (str.empty()) return result;
        std::stringstream ss(str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            if (token.empty()) continue;
            auto dash_pos = token.find('-');
            if (dash_pos != std::string::npos) {
                int start = std::strtol(token.substr(0, dash_pos).c_str(), nullptr, 10);
                int end = std::strtol(token.substr(dash_pos + 1).c_str(), nullptr, 10);
                for (int i = start; i <= end; ++i) result.push_back(i);
            } else {
                result.push_back(std::strtol(token.c_str(), nullptr, 10));
            }
        }
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        return result;
    }
} // anonymous namespace

namespace rknpu2_configuration {

Rknpu2ConfigManager& Rknpu2ConfigManager::get_instance() {
    static Rknpu2ConfigManager instance;
    return instance;
}

const std::vector<std::string>* Rknpu2DeviceConfig::get_active_pattern(int tensor_type) const {
    auto it = default_patterns.find(tensor_type);
    if (it == default_patterns.end()) {
        return nullptr;
    }

    if (use_custom_pattern && !custom_hybrid_pattern.empty()) {
        return &custom_hybrid_pattern;
    }

    return &it->second;
}

const Rknpu2HardwarePipeline* Rknpu2DeviceConfig::resolve_op_support(const struct ggml_tensor* w_tensor) const {
    if (!w_tensor) return nullptr;

    auto find_pipeline = [this](const std::string& name) -> const Rknpu2HardwarePipeline* {
        for (const auto& pipe : hardware_pipelines) {
            if (pipe.pipeline_name == name) return &pipe;
        }
        return nullptr;
    };

    // Retrieve active quantization pattern based on tensor type (or custom ENV variable)
    const std::vector<std::string>* pattern_ptr = get_active_pattern((int)w_tensor->type);

    // If no pattern is registered for this type and no global override exists, reject operation
    if (!pattern_ptr || pattern_ptr->empty()) {
        return nullptr;
    }

    const auto& pattern = *pattern_ptr;

    // Acquiring the lock on the pattern mutex for thread-safe tensor tracking
    std::lock_guard<std::mutex> lock(*pattern_mutex);

    // Retrieving the unique tensor name
    std::string name = w_tensor->name;
    if (name.empty()) {
        name = "ptr_" + std::to_string(reinterpret_cast<uintptr_t>(w_tensor));
    }

    // Assigning the next sequence number if this tensor is seen for the first time
    if (tensor_sequence_map.find(name) == tensor_sequence_map.end()) {
        tensor_sequence_map[name] = global_tensor_counter++;
    }

    // Selecting the pipeline cyclically based on the defined pattern
    int seq_id = tensor_sequence_map[name];
    size_t pattern_idx = seq_id % pattern.size();

    const std::string& selected_pipeline = pattern[pattern_idx];
    const auto* pipeline = find_pipeline(selected_pipeline);

    // If no hardware pipeline exists with this name, reject operation
    if (!pipeline) {
        return nullptr;
    }

    return pipeline;
}

Rknpu2ConfigManager::Rknpu2ConfigManager() {
    // Reading custom hybrid pattern ENV variable
    const char* env_pattern = std::getenv("RKNPU_HYBRID");
    bool use_custom_pattern = false;
    std::vector<std::string> custom_pattern;

    if (env_pattern != nullptr) {
        custom_pattern = split_string(env_pattern, ',');
        use_custom_pattern = true;
    }

    // Reading specific active cores ENV variable
    const char* env_cores = std::getenv("RKNPU_CORES");
    std::vector<int> custom_cores;
    if (env_cores != nullptr) {
        custom_cores = parse_int_list(env_cores);
    }

    // --- Define RK3588 Configuration ---
    Rknpu2DeviceConfig rk3588_config;
    rk3588_config.device_name = "RK3588";
    rk3588_config.active_cores = custom_cores.empty() ? std::vector<int>{0, 1, 2} : custom_cores;
    rk3588_config.max_k_limit = 8192;
    rk3588_config.hardware_pipelines = {
        {
            /* .pipeline_name = */ "W16A16_STANDARD",
            /* .npu_type_a    = */ NPU_TYPE_FP16,
            /* .npu_type_b    = */ NPU_TYPE_FP16,
            /* .npu_type_c    = */ NPU_TYPE_FP32,
            /* .mm_type       = */ RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 16,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ false
        },
        {
            /* .pipeline_name = */ "W16A16_HADAMARD",
            /* .npu_type_a    = */ NPU_TYPE_FP16,
            /* .npu_type_b    = */ NPU_TYPE_FP16,
            /* .npu_type_c    = */ NPU_TYPE_FP32,
            /* .mm_type       = */ RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 16,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ true
        },
        {
            /* .pipeline_name = */ "W8A8_STANDARD",
            /* .npu_type_a    = */ NPU_TYPE_INT8,
            /* .npu_type_b    = */ NPU_TYPE_INT8,
            /* .npu_type_c    = */ NPU_TYPE_INT32,
            /* .mm_type       = */ RKNN_INT8_MM_INT8_TO_INT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 32,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ false
        },
        {
            /* .pipeline_name = */ "W8A8_HADAMARD",
            /* .npu_type_a    = */ NPU_TYPE_INT8,
            /* .npu_type_b    = */ NPU_TYPE_INT8,
            /* .npu_type_c    = */ NPU_TYPE_INT32,
            /* .mm_type       = */ RKNN_INT8_MM_INT8_TO_INT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 32,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ true
        },
        {
            /* .pipeline_name = */ "W4A4_STANDARD",
            /* .npu_type_a    = */ NPU_TYPE_INT4,
            /* .npu_type_b    = */ NPU_TYPE_INT4,
            /* .npu_type_c    = */ NPU_TYPE_INT16,
            /* .mm_type       = */ RKNN_INT4_MM_INT4_TO_INT16,
            /* .k_align       = */ 32,
            /* .n_align       = */ 64,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ false
        },
        {
            /* .pipeline_name = */ "W4A4_HADAMARD",
            /* .npu_type_a    = */ NPU_TYPE_INT4,
            /* .npu_type_b    = */ NPU_TYPE_INT4,
            /* .npu_type_c    = */ NPU_TYPE_INT16,
            /* .mm_type       = */ RKNN_INT4_MM_INT4_TO_INT16,
            /* .k_align       = */ 32,
            /* .n_align       = */ 64,
            /* .effective_k   = */ 0,
            /* .use_hadamard  = */ true
        }
    };

    // Assigning custom variables
    rk3588_config.use_custom_pattern = use_custom_pattern;
    rk3588_config.custom_hybrid_pattern = custom_pattern;

    // Defining default quantization sequences for each supported ggml_type
    rk3588_config.default_patterns[(int)GGML_TYPE_F16]  = {"W16A16_STANDARD"};
    rk3588_config.default_patterns[(int)GGML_TYPE_Q8_0] = {"W8A8_STANDARD"};
    rk3588_config.default_patterns[(int)GGML_TYPE_Q6_K] = {"W8A8_STANDARD", "W4A4_HADAMARD"};
    rk3588_config.default_patterns[(int)GGML_TYPE_Q4_0] = {"W4A4_HADAMARD"};

    device_configs["RK3588"] = rk3588_config;

    // --- Define RK3576 Configuration (Placeholder) ---
    // Rknpu2DeviceConfig rk3576_config;
    // ... fill config for RK3576 ...
    // device_configs["RK3576"] = rk3576_config;

    // --- Define RK3566 Configuration (Placeholder) ---
    // Rknpu2DeviceConfig rk3566_config;
    // ... fill config for RK3566 ...
    // device_configs["RK3566"] = rk3566_config;

    // Select a default device
    if (!device_configs.empty()) {
        select_device(device_configs.begin()->first);
    }
}

bool Rknpu2ConfigManager::select_device(const std::string& device_name) {
    auto it = device_configs.find(device_name);
    if (it != device_configs.end()) {
        current_config = &it->second;
        return true;
    }
    return false;
}

const Rknpu2DeviceConfig& Rknpu2ConfigManager::get_current_config() const {
    GGML_ASSERT(current_config != nullptr && "No device configuration selected or available.");
    return *current_config;
}

} // namespace rknpu2_configuration