#pragma once

#include <string>

namespace sycl_bench {

enum class DpasType {
    INT8,
    FP16,
    BF16,
};

enum class DpasAccType {
    INT32,
    FP32,
};

enum class DpasMemoryPattern {
    DIRECT_GLOBAL,
    SLM_BUFFER,
    REG_PREFETCH,
    DOUBLE_BUFFER,
    LSC_STREAMING,
    LSC_PREFETCH,
    LSC_PREFETCH_2,
    LSC_PREFETCH_3,
    LSC_PREFETCH_4,
    LSC_PREFETCH_5,
    LSC_PREFETCH_6,
    LSC_PREFETCH_8,
    LSC_PREFETCH_10,
};

enum class DpasGrfMode {
    GRF_128,
    GRF_256,
};

inline const char * dpas_type_name(DpasType type) {
    switch (type) {
        case DpasType::INT8: return "int8";
        case DpasType::FP16: return "fp16";
        case DpasType::BF16: return "bf16";
        default: return "unknown";
    }
}

inline const char * dpas_acc_name(DpasAccType type) {
    switch (type) {
        case DpasAccType::INT32: return "int32";
        case DpasAccType::FP32: return "float";
        default: return "unknown";
    }
}

inline const char * dpas_memory_pattern_name(DpasMemoryPattern pattern) {
    switch (pattern) {
        case DpasMemoryPattern::DIRECT_GLOBAL: return "direct_global";
        case DpasMemoryPattern::SLM_BUFFER: return "slm_buffer";
        case DpasMemoryPattern::REG_PREFETCH: return "reg_prefetch";
        case DpasMemoryPattern::DOUBLE_BUFFER: return "double_buffer";
        case DpasMemoryPattern::LSC_STREAMING: return "lsc_streaming";
        case DpasMemoryPattern::LSC_PREFETCH: return "lsc_prefetch";
        case DpasMemoryPattern::LSC_PREFETCH_2: return "lsc_prefetch2";
        case DpasMemoryPattern::LSC_PREFETCH_3: return "lsc_prefetch3";
        case DpasMemoryPattern::LSC_PREFETCH_4: return "lsc_prefetch4";
        case DpasMemoryPattern::LSC_PREFETCH_5: return "lsc_prefetch5";
        case DpasMemoryPattern::LSC_PREFETCH_6: return "lsc_prefetch6";
        case DpasMemoryPattern::LSC_PREFETCH_8: return "lsc_prefetch8";
        case DpasMemoryPattern::LSC_PREFETCH_10: return "lsc_prefetch10";
        default: return "unknown";
    }
}

inline const char * dpas_grf_mode_name(DpasGrfMode mode) {
    switch (mode) {
        case DpasGrfMode::GRF_128: return "128";
        case DpasGrfMode::GRF_256: return "256";
        default: return "unknown";
    }
}

inline bool parse_dpas_type(const std::string & input, DpasType & out) {
    if (input == "int8") { out = DpasType::INT8; return true; }
    if (input == "fp16") { out = DpasType::FP16; return true; }
    if (input == "bf16") { out = DpasType::BF16; return true; }
    return false;
}

inline bool parse_dpas_acc_type(const std::string & input, DpasAccType & out) {
    if (input == "int32") { out = DpasAccType::INT32; return true; }
    if (input == "float" || input == "fp32") { out = DpasAccType::FP32; return true; }
    return false;
}

inline bool parse_dpas_memory_pattern(const std::string & input, DpasMemoryPattern & out) {
    if (input == "direct_global") { out = DpasMemoryPattern::DIRECT_GLOBAL; return true; }
    if (input == "slm_buffer") { out = DpasMemoryPattern::SLM_BUFFER; return true; }
    if (input == "reg_prefetch") { out = DpasMemoryPattern::REG_PREFETCH; return true; }
    if (input == "double_buffer") { out = DpasMemoryPattern::DOUBLE_BUFFER; return true; }
    if (input == "lsc_streaming") { out = DpasMemoryPattern::LSC_STREAMING; return true; }
    if (input == "lsc_prefetch") { out = DpasMemoryPattern::LSC_PREFETCH; return true; }
    if (input == "lsc_prefetch2" || input == "lsc_prefetch_2") { out = DpasMemoryPattern::LSC_PREFETCH_2; return true; }
    if (input == "lsc_prefetch3" || input == "lsc_prefetch_3") { out = DpasMemoryPattern::LSC_PREFETCH_3; return true; }
    if (input == "lsc_prefetch4" || input == "lsc_prefetch_4") { out = DpasMemoryPattern::LSC_PREFETCH_4; return true; }
    if (input == "lsc_prefetch5" || input == "lsc_prefetch_5") { out = DpasMemoryPattern::LSC_PREFETCH_5; return true; }
    if (input == "lsc_prefetch6" || input == "lsc_prefetch_6") { out = DpasMemoryPattern::LSC_PREFETCH_6; return true; }
    if (input == "lsc_prefetch8" || input == "lsc_prefetch_8") { out = DpasMemoryPattern::LSC_PREFETCH_8; return true; }
    if (input == "lsc_prefetch10" || input == "lsc_prefetch_10") { out = DpasMemoryPattern::LSC_PREFETCH_10; return true; }
    return false;
}

inline bool parse_dpas_grf_mode(const std::string & input, DpasGrfMode & out) {
    if (input == "128") { out = DpasGrfMode::GRF_128; return true; }
    if (input == "256") { out = DpasGrfMode::GRF_256; return true; }
    return false;
}

}  // namespace sycl_bench
