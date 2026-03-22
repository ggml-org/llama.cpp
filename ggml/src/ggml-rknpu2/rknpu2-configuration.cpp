#include "ggml-impl.h"

#include "rknpu2-configuration.h"

#include <arm_neon.h>
#include <cstdlib>
#include <sstream>

// --- Anonymous namespace for chip-specific packing functions ---

namespace {

// Packing KxN FP16 (row-major: idx [k,n] -> k*N + n) into native RKNN for RK3588: (N/16, K/32, 16, 32)
void pack_B_rk3588_fp16(
    uint8_t* dst_u8, const uint8_t* src_u8,
    int K, int N_total, int n_offset, int n_segment) {

    auto dst = reinterpret_cast<uint16_t*>(dst_u8);
    auto src = reinterpret_cast<const uint16_t*>(src_u8);

    GGML_ASSERT(K % 32 == 0 && N_total > 0 && K > 0);
    GGML_ASSERT(n_offset % 16 == 0 && n_segment % 16 == 0 && n_offset + n_segment <= N_total);

    const size_t s0 = (size_t)(K / 32) * 16 * 32;
    const size_t s1 = 16 * 32;
    const size_t s2 = 32;

    for (int i = 0; i < n_segment / 16; ++i) {
        for (int j = 0; j < K / 32; ++j) {
            const size_t dst_block = (size_t) i * s0 + (size_t) j * s1;
            for (int ii = 0; ii < 16; ++ii) {
                const size_t n_global = (size_t)n_offset + (size_t)i * 16 + (size_t)ii;
                
                const uint16_t * src_ptr = src + n_global * K + j * 32;
                uint16_t * dst_ptr = dst + dst_block + ii * s2;

                uint16x8_t d0 = vld1q_u16(src_ptr + 0);
                uint16x8_t d1 = vld1q_u16(src_ptr + 8);
                uint16x8_t d2 = vld1q_u16(src_ptr + 16);
                uint16x8_t d3 = vld1q_u16(src_ptr + 24);

                vst1q_u16(dst_ptr + 0, d0);
                vst1q_u16(dst_ptr + 8, d1);
                vst1q_u16(dst_ptr + 16, d2);
                vst1q_u16(dst_ptr + 24, d3);
            }
        }
    }
}

// Packing KxN INT8 (row-major) into native RKNN for RK3588: (N/32, K/32, 32, 32)
void pack_B_rk3588_int8(
    uint8_t* dst_u8, const uint8_t* src_u8,
    int K, int N_total, int n_offset, int n_segment) {

    auto dst = reinterpret_cast<int8_t*>(dst_u8);
    auto src = reinterpret_cast<const int8_t*>(src_u8);

    GGML_ASSERT(K % 32 == 0 && N_total > 0 && K > 0);
    GGML_ASSERT(n_offset % 32 == 0 && n_segment % 32 == 0 && n_offset + n_segment <= N_total);

    const size_t s0 = (size_t)(K / 32) * 32 * 32;
    const size_t s1 = 32 * 32;
    const size_t s2 = 32;

    for (int i = 0; i < n_segment / 32; ++i) {
        for (int j = 0; j < K / 32; ++j) {
            const size_t dst_block = (size_t) i * s0 + (size_t) j * s1;
            for (int ii = 0; ii < 32; ++ii) {
                const size_t n_global = (size_t)n_offset + (size_t)i * 32 + (size_t)ii;

                const int8_t* src_ptr = src + n_global * K + j * 32;
                int8_t* dst_ptr = dst + dst_block + ii * s2;

                int8x16_t d0 = vld1q_s8(src_ptr);
                int8x16_t d1 = vld1q_s8(src_ptr + 16);

                vst1q_s8(dst_ptr, d0);
                vst1q_s8(dst_ptr + 16, d1);
            }
        }
    }
}

// Packing KxN INT4 (row-major) into native RKNN for RK3588: (N/64, K/32, 64, 32)
void pack_B_rk3588_int4(
    uint8_t * dst, const uint8_t * src,
    int K, int N_total, int n_offset, int n_segment) {

    GGML_ASSERT(K % 32 == 0 && N_total > 0 && K > 0);
    GGML_ASSERT(n_offset % 64 == 0 && n_segment % 64 == 0 && n_offset + n_segment <= N_total);

    const size_t s0 = (size_t)(K / 32) * 64 * (32 / 2);
    const size_t s1 = 64 * (32 / 2);
    const size_t s2 = (32 / 2); 

    const size_t src_row_stride_bytes = (size_t)K / 2;

    for (int i = 0; i < n_segment / 64; ++i) {
        for (int j = 0; j < K / 32; ++j) {
            const size_t dst_block = (size_t) i * s0 + (size_t) j * s1;
            for (int ii = 0; ii < 64; ++ii) {
                const size_t n_global = (size_t)n_offset + (size_t)i * 64 + (size_t)ii;

                const uint8_t* src_ptr = src + n_global * src_row_stride_bytes + (j * 32) / 2;
                uint8_t* dst_ptr = dst + dst_block + ii * s2;

                uint8x16_t d0 = vld1q_u8(src_ptr);
                vst1q_u8(dst_ptr, d0);
            }
        }
    }
}

} // anonymous namespace

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
    const char* env_pattern = std::getenv("HYBRID_PATTERN");
    bool use_custom_pattern = false;
    std::vector<std::string> custom_pattern;

    if (env_pattern != nullptr) {
        custom_pattern = split_string(env_pattern, ',');
        use_custom_pattern = true;
    }

    // --- Define RK3588 Configuration ---
    Rknpu2DeviceConfig rk3588_config;
    rk3588_config.device_name = "RK3588";
    rk3588_config.core_count = 3;
    rk3588_config.hardware_pipelines = {
        {
            /* .pipeline_name = */ "FP16_STANDARD",
            /* .npu_type_a    = */ NPU_TYPE_FP16,
            /* .npu_type_c    = */ NPU_TYPE_FP32,
            /* .mm_type       = */ RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 16,
            /* .pack_func     = */ pack_B_rk3588_fp16,
            /* .use_hadamard  = */ false
        },
        {
            /* .pipeline_name = */ "FP16_HADAMARD",
            /* .npu_type_a    = */ NPU_TYPE_FP16,
            /* .npu_type_c    = */ NPU_TYPE_FP32,
            /* .mm_type       = */ RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 16,
            /* .pack_func     = */ pack_B_rk3588_fp16,
            /* .use_hadamard  = */ true
        },
        {
            /* .pipeline_name = */ "INT8_STANDARD",
            /* .npu_type_a    = */ NPU_TYPE_INT8,
            /* .npu_type_c    = */ NPU_TYPE_INT32,
            /* .mm_type       = */ RKNN_INT8_MM_INT8_TO_INT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 32,
            /* .pack_func     = */ pack_B_rk3588_int8,
            /* .use_hadamard  = */ false
        },
        {
            /* .pipeline_name = */ "INT8_HADAMARD",
            /* .npu_type_a    = */ NPU_TYPE_INT8,
            /* .npu_type_c    = */ NPU_TYPE_INT32,
            /* .mm_type       = */ RKNN_INT8_MM_INT8_TO_INT32,
            /* .k_align       = */ 32,
            /* .n_align       = */ 32,
            /* .pack_func     = */ pack_B_rk3588_int8,
            /* .use_hadamard  = */ true
        },
        {
            /* .pipeline_name = */ "INT4_STANDARD",
            /* .npu_type_a    = */ NPU_TYPE_INT4,
            /* .npu_type_c    = */ NPU_TYPE_INT16,
            /* .mm_type       = */ RKNN_INT4_MM_INT4_TO_INT16,
            /* .k_align       = */ 32,
            /* .n_align       = */ 64,
            /* .pack_func     = */ pack_B_rk3588_int4,
            /* .use_hadamard  = */ false
        },
        {
            /* .pipeline_name = */ "INT4_HADAMARD",
            /* .npu_type_a    = */ NPU_TYPE_INT4,
            /* .npu_type_c    = */ NPU_TYPE_INT16,
            /* .mm_type       = */ RKNN_INT4_MM_INT4_TO_INT16,
            /* .k_align       = */ 32,
            /* .n_align       = */ 64,
            /* .pack_func     = */ pack_B_rk3588_int4,
            /* .use_hadamard  = */ true
        }
    };
    
    // Assigning custom variables
    rk3588_config.use_custom_pattern = use_custom_pattern;
    rk3588_config.custom_hybrid_pattern = custom_pattern;

    // Defining default quantization sequences for each supported ggml_type
    rk3588_config.default_patterns[(int)GGML_TYPE_F16]  = {"FP16_STANDARD"};
    rk3588_config.default_patterns[(int)GGML_TYPE_Q8_0] = {"INT8_STANDARD"};
    rk3588_config.default_patterns[(int)GGML_TYPE_Q6_K] = {"INT8_STANDARD", "INT4_HADAMARD"};
    rk3588_config.default_patterns[(int)GGML_TYPE_Q4_0] = {"INT4_HADAMARD"};

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