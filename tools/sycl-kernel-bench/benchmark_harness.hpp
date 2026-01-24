#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include "ggml-backend.h"
#include "ggml-sycl.h"
#include "ggml.h"

#include "ggml-sycl/ggml-sycl-bench.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/presets.hpp"

#include "data_generators.hpp"
#include "dpas_config.hpp"
#include "kernel_registry.hpp"
#include "kernels/reference/reference_kernels.hpp"
#include "kernels/dpas_exploration/dpas_kernels.hpp"
#include "kernels/tier1/mmvq_tier1.hpp"
#include "kernels/tier2/mmvq_tier2.hpp"
#include "kernels/tier3/mmvq_tier3.hpp"
#include "kernels/tier4/mmvq_tier4.hpp"
#include "roofline.hpp"

namespace sycl_bench {

enum class MemoryMode {
    USM_DEVICE,
    USM_SHARED,
    BUFFER,
};

enum class DpasTuneMetric {
    THROUGHPUT,
    BANDWIDTH,
};

struct BenchmarkConfig {
    std::string    kernel_name = "mmvq_aos";
    ggml_type      quant_type  = GGML_TYPE_Q4_0;
    ggml_layout_mode layout    = GGML_LAYOUT_AOS;
    KernelKind     kernel_kind = KernelKind::MMVQ;
    int64_t        batch_size  = 1;
    int64_t        dim_m       = 4096;
    int64_t        dim_n       = 4096;
    int64_t        dim_k       = 4096;
    int           warmup_iterations  = 10;
    int           measure_iterations = 100;
    MemoryMode    memory_mode = MemoryMode::USM_DEVICE;
    bool          validate    = false;
    bool          include_percentiles = false;
    bool          include_ref_metrics = false;
    size_t        transfer_bytes = 0;
    int64_t       roofline_elements = 0;
    int           roofline_ops = 0;
    double        abs_tol     = 1e-2;
    double        rel_tol     = 1e-2;
    int           device_id   = -1;
    std::string   dpas_config_name;
    DpasType      dpas_type_a = DpasType::INT8;
    DpasType      dpas_type_b = DpasType::INT8;
    DpasAccType   dpas_type_acc = DpasAccType::INT32;
    DpasMemoryPattern dpas_memory_pattern = DpasMemoryPattern::DIRECT_GLOBAL;
    DpasGrfMode   dpas_grf_mode = DpasGrfMode::GRF_128;
    int           dpas_repeat = 8;
    int           dpas_n_tile_repeats = 1;
    bool          dpas_misaligned = false;
    bool          dpas_device_opt = false;
    bool          dpas_autotune = false;
    bool          dpas_autotune_force = false;
    DpasTuneMetric dpas_autotune_metric = DpasTuneMetric::THROUGHPUT;
    std::string   dpas_autotune_cache = "benchmark_results/dpas_tuning_cache.jsonl";
    int           dpas_autotune_override_ntiles = 0;
    int           dpas_autotune_override_prefetch = 0;
    bool          dpas_memory_explicit = false;
    bool          dpas_ntiles_explicit = false;
    bool          dpas_grf_explicit = false;
    bool          dpas_acc_explicit = false;
};

struct DeviceInfo {
    std::string name;
    std::string vendor;
    std::string driver_version;
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    uint32_t max_compute_units = 0;
    uint32_t max_clock_mhz = 0;
    size_t local_mem_size = 0;
    size_t max_work_group_size = 0;
    std::vector<size_t> sub_group_sizes;
};

struct DpasTuneKey {
    std::string device_key;
    int64_t dim_m = 0;
    int64_t dim_n = 0;
    int64_t dim_k = 0;
    int repeat = 0;
    DpasType type_a = DpasType::INT8;
    DpasType type_b = DpasType::INT8;
    DpasAccType type_acc = DpasAccType::INT32;
    DpasGrfMode grf_mode = DpasGrfMode::GRF_128;
    bool misaligned = false;
    std::string metric;
};

struct DpasTuneResult {
    int ntiles = 0;
    int prefetch_dist = 0;
    double tops = 0.0;
    double bandwidth = 0.0;
    bool valid = false;
    DpasMemoryPattern memory_pattern = DpasMemoryPattern::LSC_PREFETCH;
};

struct BenchmarkResult {
    double throughput_tps = 0.0;
    double latency_us     = 0.0;
    double latency_std    = 0.0;
    double latency_p50_us = 0.0;
    double latency_p90_us = 0.0;
    double latency_p99_us = 0.0;
    double bandwidth_gbps = 0.0;
    double variance_pct   = 0.0;
    double xmx_util_pct   = 0.0;
    double ref_total_us   = 0.0;
    double ref_dequant_us = 0.0;
    double ref_gemm_us    = 0.0;
    double ref_scale_us   = 0.0;
    double ref_tflops     = 0.0;
    double ref_tops       = 0.0;
    double ref_bandwidth_gbps = 0.0;
    double ref_arith_intensity = 0.0;
    double throughput_tops = 0.0;
    double latency_ns = 0.0;
};

struct BenchmarkOutput {
    BenchmarkConfig config;
    BenchmarkResult result;
    bool            ok = false;
    std::string     error;
    double          max_abs_error  = 0.0;
    double          mean_abs_error = 0.0;
};

inline size_t align_up(size_t value, size_t alignment) {
    if (alignment == 0) {
        return value;
    }
    const size_t rem = value % alignment;
    return rem == 0 ? value : (value + alignment - rem);
}

inline std::string json_escape_string(const std::string & input) {
    std::string out;
    out.reserve(input.size());
    for (char c : input) {
        if (c == '\"' || c == '\\') {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    return out;
}

inline DeviceInfo query_device_info(const sycl::device & device) {
    DeviceInfo info{};
    try { info.name = device.get_info<sycl::info::device::name>(); } catch (...) {}
    try { info.vendor = device.get_info<sycl::info::device::vendor>(); } catch (...) {}
    try { info.driver_version = device.get_info<sycl::info::device::driver_version>(); } catch (...) {}
    try { info.vendor_id = device.get_info<sycl::info::device::vendor_id>(); } catch (...) {}
    try { info.device_id = device.get_info<sycl::ext::intel::info::device::device_id>(); } catch (...) {}
    try { info.max_compute_units = device.get_info<sycl::info::device::max_compute_units>(); } catch (...) {}
    try { info.max_clock_mhz = device.get_info<sycl::info::device::max_clock_frequency>(); } catch (...) {}
    try { info.local_mem_size = device.get_info<sycl::info::device::local_mem_size>(); } catch (...) {}
    try { info.max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>(); } catch (...) {}
    try { info.sub_group_sizes = device.get_info<sycl::info::device::sub_group_sizes>(); } catch (...) {}
    return info;
}

inline std::string make_device_key(const DeviceInfo & info) {
    std::ostringstream oss;
    oss << info.name << "|" << info.vendor_id << "|" << info.device_id << "|" << info.driver_version;
    return oss.str();
}

inline const char * dpas_tune_metric_name(DpasTuneMetric metric) {
    switch (metric) {
        case DpasTuneMetric::BANDWIDTH: return "bandwidth";
        case DpasTuneMetric::THROUGHPUT: return "throughput";
        default: return "throughput";
    }
}

inline bool parse_dpas_tune_metric(const std::string & input, DpasTuneMetric & out) {
    if (input == "throughput" || input == "tops") {
        out = DpasTuneMetric::THROUGHPUT;
        return true;
    }
    if (input == "bandwidth" || input == "bw") {
        out = DpasTuneMetric::BANDWIDTH;
        return true;
    }
    return false;
}

inline int dpas_prefetch_dist_from_pattern(DpasMemoryPattern pattern) {
    switch (pattern) {
        case DpasMemoryPattern::LSC_PREFETCH: return 1;
        case DpasMemoryPattern::LSC_PREFETCH_2: return 2;
        case DpasMemoryPattern::LSC_PREFETCH_3: return 3;
        case DpasMemoryPattern::LSC_PREFETCH_4: return 4;
        case DpasMemoryPattern::LSC_PREFETCH_5: return 5;
        case DpasMemoryPattern::LSC_PREFETCH_6: return 6;
        case DpasMemoryPattern::LSC_PREFETCH_8: return 8;
        case DpasMemoryPattern::LSC_PREFETCH_10: return 10;
        default: return 0;
    }
}

inline DpasMemoryPattern dpas_prefetch_pattern_from_dist(int dist) {
    switch (dist) {
        case 1: return DpasMemoryPattern::LSC_PREFETCH;
        case 2: return DpasMemoryPattern::LSC_PREFETCH_2;
        case 3: return DpasMemoryPattern::LSC_PREFETCH_3;
        case 4: return DpasMemoryPattern::LSC_PREFETCH_4;
        case 5: return DpasMemoryPattern::LSC_PREFETCH_5;
        case 6: return DpasMemoryPattern::LSC_PREFETCH_6;
        case 8: return DpasMemoryPattern::LSC_PREFETCH_8;
        case 10: return DpasMemoryPattern::LSC_PREFETCH_10;
        default: return DpasMemoryPattern::LSC_PREFETCH;
    }
}

inline size_t dpas_slm_total_bytes(const BenchmarkConfig & config,
                                   int ntiles,
                                   bool double_buffer) {
    const int k_per = dpas_k_per_tile(config.dpas_type_a, config.dpas_type_b);
    if (k_per <= 0) {
        return 0;
    }
    const size_t slm_a_bytes = static_cast<size_t>(config.dpas_repeat) * static_cast<size_t>(k_per) *
                               ((config.dpas_type_a == DpasType::INT8) ? sizeof(int8_t) : sizeof(uint16_t));
    const size_t slm_b_bytes = static_cast<size_t>(k_per) * 16u *
                               ((config.dpas_type_b == DpasType::INT8) ? sizeof(int8_t) : sizeof(uint16_t));
    const size_t slm_pad = 64;
    const size_t slm_b_offset = align_up(slm_a_bytes, slm_pad);
    const size_t slm_b_stride = align_up(slm_b_bytes, slm_pad);
    const size_t tile_bytes = slm_b_offset + slm_b_stride * static_cast<size_t>(ntiles);
    return double_buffer ? tile_bytes * 2 : tile_bytes;
}

inline int dpas_max_slm_ntiles(const BenchmarkConfig & config,
                               const DeviceInfo & info,
                               bool double_buffer) {
    const size_t slm_limit = info.local_mem_size;
    if (slm_limit == 0) {
        return 1;
    }
    for (int ntiles : {4, 2, 1}) {
        if (dpas_slm_total_bytes(config, ntiles, double_buffer) <= slm_limit) {
            return ntiles;
        }
    }
    return 1;
}

inline int dpas_heuristic_ntiles(const BenchmarkConfig & config,
                                 const DeviceInfo & info) {
    int ntiles = 1;
    switch (config.dpas_repeat) {
        case 1: ntiles = 8; break;
        case 2: ntiles = 4; break;
        case 4: ntiles = 8; break;
        case 8: ntiles = 4; break;
        default:
            if (info.max_compute_units >= 128) {
                ntiles = 4;
            } else if (info.max_compute_units >= 96) {
                ntiles = 2;
            }
            break;
    }
    const int64_t n_tiles = config.dim_n / 16;
    while (ntiles > 1 && (n_tiles % ntiles) != 0) {
        ntiles >>= 1;
    }
    if (config.dpas_memory_pattern == DpasMemoryPattern::SLM_BUFFER ||
        config.dpas_memory_pattern == DpasMemoryPattern::DOUBLE_BUFFER) {
        const bool double_buffer = (config.dpas_memory_pattern == DpasMemoryPattern::DOUBLE_BUFFER);
        const int slm_cap = dpas_max_slm_ntiles(config, info, double_buffer);
        if (slm_cap < ntiles) {
            ntiles = slm_cap;
            while (ntiles > 1 && (n_tiles % ntiles) != 0) {
                ntiles >>= 1;
            }
        }
    }
    return ntiles;
}

inline DpasMemoryPattern dpas_heuristic_memory_pattern(const BenchmarkConfig & config,
                                                       const DeviceInfo & info) {
    if (config.dpas_repeat >= 4) {
        return DpasMemoryPattern::LSC_STREAMING;
    }
    if (config.dpas_repeat > 0 && config.dpas_repeat <= 2) {
        return DpasMemoryPattern::LSC_PREFETCH_2;
    }
    if (info.max_compute_units >= 128) {
        return DpasMemoryPattern::LSC_PREFETCH_3;
    }
    if (info.max_compute_units >= 96) {
        return DpasMemoryPattern::LSC_PREFETCH_2;
    }
    return DpasMemoryPattern::LSC_PREFETCH;
}

struct DpasDimTuningEntry {
    int repeat = 0;
    int64_t dim_n_tiles = 0;
    int ntiles = 0;
    DpasMemoryPattern pattern = DpasMemoryPattern::DIRECT_GLOBAL;
    DpasGrfMode grf = DpasGrfMode::GRF_128;
    DpasAccType acc = DpasAccType::INT32;
};

inline bool dpas_device_is_b580(const DeviceInfo & info) {
    if (info.device_id == 0xe20b) {
        return true;
    }
    if (info.name.find("B580") != std::string::npos) {
        return true;
    }
    return false;
}

inline bool dpas_lookup_dim_tuning(const BenchmarkConfig & config,
                                   const DeviceInfo & info,
                                   DpasDimTuningEntry & out) {
    if (config.dim_n <= 0 || (config.dim_n % 16) != 0) {
        return false;
    }
    const int64_t dim_n_tiles = config.dim_n / 16;
    const DpasDimTuningEntry default_table[] = {
        {2, 256, 4, DpasMemoryPattern::LSC_STREAMING, DpasGrfMode::GRF_128, DpasAccType::FP32},
        {2, 512, 8, DpasMemoryPattern::LSC_PREFETCH_2, DpasGrfMode::GRF_256, DpasAccType::FP32},
        {4, 128, 8, DpasMemoryPattern::LSC_STREAMING, DpasGrfMode::GRF_128, DpasAccType::FP32},
        {4, 256, 8, DpasMemoryPattern::LSC_STREAMING, DpasGrfMode::GRF_256, DpasAccType::FP32},
        {4, 512, 4, DpasMemoryPattern::LSC_PREFETCH_2, DpasGrfMode::GRF_128, DpasAccType::INT32},
        {4, 1024, 8, DpasMemoryPattern::LSC_STREAMING, DpasGrfMode::GRF_128, DpasAccType::INT32},
        {8, 256, 4, DpasMemoryPattern::LSC_STREAMING, DpasGrfMode::GRF_256, DpasAccType::FP32},
        {8, 512, 4, DpasMemoryPattern::LSC_STREAMING, DpasGrfMode::GRF_128, DpasAccType::FP32},
        {8, 1024, 4, DpasMemoryPattern::LSC_STREAMING, DpasGrfMode::GRF_256, DpasAccType::INT32},
    };
    const DpasDimTuningEntry b580_table[] = {
        {4, 128, 8, DpasMemoryPattern::LSC_STREAMING, DpasGrfMode::GRF_128, DpasAccType::FP32},
        {4, 256, 8, DpasMemoryPattern::LSC_STREAMING, DpasGrfMode::GRF_128, DpasAccType::FP32},
        {8, 128, 4, DpasMemoryPattern::LSC_STREAMING, DpasGrfMode::GRF_128, DpasAccType::FP32},
        {8, 256, 4, DpasMemoryPattern::LSC_STREAMING, DpasGrfMode::GRF_128, DpasAccType::FP32},
    };

    const DpasDimTuningEntry * table = default_table;
    size_t table_size = sizeof(default_table) / sizeof(default_table[0]);
    if (dpas_device_is_b580(info)) {
        table = b580_table;
        table_size = sizeof(b580_table) / sizeof(b580_table[0]);
    }

    for (size_t i = 0; i < table_size; ++i) {
        const auto & entry = table[i];
        if (entry.repeat == config.dpas_repeat && entry.dim_n_tiles == dim_n_tiles) {
            out = entry;
            return true;
        }
    }
    return false;
}

inline bool json_extract_string(const std::string & line, const char * key, std::string & out) {
    const std::string needle = std::string("\"") + key + "\":";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    pos += needle.size();
    while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) {
        ++pos;
    }
    if (pos >= line.size() || line[pos] != '\"') {
        return false;
    }
    ++pos;
    std::string value;
    while (pos < line.size()) {
        char c = line[pos++];
        if (c == '\\' && pos < line.size()) {
            value.push_back(line[pos++]);
            continue;
        }
        if (c == '\"') {
            break;
        }
        value.push_back(c);
    }
    out = value;
    return true;
}

inline bool json_extract_int64(const std::string & line, const char * key, int64_t & out) {
    const std::string needle = std::string("\"") + key + "\":";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    pos += needle.size();
    while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) {
        ++pos;
    }
    const char * start = line.c_str() + pos;
    char * end = nullptr;
    const long long val = std::strtoll(start, &end, 10);
    if (start == end) {
        return false;
    }
    out = static_cast<int64_t>(val);
    return true;
}

inline bool json_extract_double(const std::string & line, const char * key, double & out) {
    const std::string needle = std::string("\"") + key + "\":";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    pos += needle.size();
    while (pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos]))) {
        ++pos;
    }
    const char * start = line.c_str() + pos;
    char * end = nullptr;
    const double val = std::strtod(start, &end);
    if (start == end) {
        return false;
    }
    out = val;
    return true;
}

inline bool json_extract_dims3(const std::string & line, int64_t & m, int64_t & n, int64_t & k) {
    const std::string needle = "\"dims\":[";
    size_t pos = line.find(needle);
    if (pos == std::string::npos) {
        return false;
    }
    pos += needle.size();
    const char * start = line.c_str() + pos;
    char * end = nullptr;
    const long long v0 = std::strtoll(start, &end, 10);
    if (!end || *end != ',') {
        return false;
    }
    const long long v1 = std::strtoll(end + 1, &end, 10);
    if (!end || *end != ',') {
        return false;
    }
    const long long v2 = std::strtoll(end + 1, &end, 10);
    if (!end || *end != ']') {
        return false;
    }
    m = v0;
    n = v1;
    k = v2;
    return true;
}

inline bool load_dpas_tuning_cache(const std::string & path,
                                   const DpasTuneKey & key,
                                   DpasTuneResult & out) {
    std::ifstream file(path);
    if (!file) {
        return false;
    }
    std::string line;
    bool found = false;
    DpasTuneResult best{};
    while (std::getline(file, line)) {
        if (line.empty() || line[0] != '{') {
            continue;
        }
        std::string device_key;
        int64_t dim_m = 0, dim_n = 0, dim_k = 0, repeat = 0, ntiles = 0, prefetch = 0;
        std::string type_a, type_b, acc, grf, memory, metric;
        int64_t misaligned = 0;
        double tops = 0.0;
        double bandwidth = 0.0;
        if (!json_extract_string(line, "device_key", device_key)) {
            continue;
        }
        if (device_key != key.device_key) {
            continue;
        }
        if (!json_extract_dims3(line, dim_m, dim_n, dim_k)) {
            continue;
        }
        if (dim_m != key.dim_m || dim_n != key.dim_n || dim_k != key.dim_k) {
            continue;
        }
        if (!json_extract_int64(line, "repeat", repeat) || repeat != key.repeat) {
            continue;
        }
        if (!json_extract_string(line, "type_a", type_a) ||
            !json_extract_string(line, "type_b", type_b) ||
            !json_extract_string(line, "acc", acc) ||
            !json_extract_string(line, "grf", grf)) {
            continue;
        }
        if (type_a != dpas_type_name(key.type_a) ||
            type_b != dpas_type_name(key.type_b) ||
            acc != dpas_acc_name(key.type_acc)) {
            continue;
        }
        if (grf != dpas_grf_mode_name(key.grf_mode)) {
            continue;
        }
        if (!json_extract_int64(line, "misaligned", misaligned) || (misaligned != 0) != key.misaligned) {
            continue;
        }
        if (!json_extract_string(line, "metric", metric)) {
            metric = "throughput";
        }
        if (!key.metric.empty() && metric != key.metric) {
            continue;
        }
        if (!json_extract_int64(line, "ntiles", ntiles) ||
            !json_extract_int64(line, "prefetch", prefetch)) {
            continue;
        }
        if (!json_extract_string(line, "memory", memory)) {
            memory.clear();
        }
        json_extract_double(line, "tops", tops);
        json_extract_double(line, "bandwidth", bandwidth);
        DpasTuneResult candidate{};
        candidate.ntiles = static_cast<int>(ntiles);
        candidate.prefetch_dist = static_cast<int>(prefetch);
        candidate.tops = tops;
        candidate.bandwidth = bandwidth;
        if (!memory.empty()) {
            DpasMemoryPattern pattern{};
            if (parse_dpas_memory_pattern(memory, pattern)) {
                candidate.memory_pattern = pattern;
            } else {
                candidate.memory_pattern = dpas_prefetch_pattern_from_dist(candidate.prefetch_dist);
            }
        } else {
            candidate.memory_pattern = dpas_prefetch_pattern_from_dist(candidate.prefetch_dist);
        }
        candidate.valid = true;
        const bool prefer_bandwidth = (key.metric == "bandwidth");
        const double candidate_score = prefer_bandwidth ? candidate.bandwidth : candidate.tops;
        const double best_score = prefer_bandwidth ? best.bandwidth : best.tops;
        if (!found || candidate_score > best_score) {
            best = candidate;
            found = true;
        }
    }
    if (found) {
        out = best;
    }
    return found;
}

inline bool append_dpas_tuning_cache(const std::string & path,
                                     const DeviceInfo & info,
                                     const DpasTuneKey & key,
                                     const DpasTuneResult & result) {
    std::ofstream file(path, std::ios::app);
    if (!file) {
        return false;
    }
    file << "{\"device\":\"" << json_escape_string(info.name) << "\","
         << "\"vendor\":\"" << json_escape_string(info.vendor) << "\","
         << "\"driver\":\"" << json_escape_string(info.driver_version) << "\","
         << "\"vendor_id\":" << info.vendor_id << ","
         << "\"device_id\":" << info.device_id << ","
         << "\"device_key\":\"" << json_escape_string(key.device_key) << "\","
         << "\"dims\":[" << key.dim_m << "," << key.dim_n << "," << key.dim_k << "],"
         << "\"repeat\":" << key.repeat << ","
         << "\"type_a\":\"" << dpas_type_name(key.type_a) << "\","
         << "\"type_b\":\"" << dpas_type_name(key.type_b) << "\","
         << "\"acc\":\"" << dpas_acc_name(key.type_acc) << "\","
         << "\"grf\":\"" << dpas_grf_mode_name(key.grf_mode) << "\","
         << "\"misaligned\":" << (key.misaligned ? 1 : 0) << ","
         << "\"metric\":\"" << json_escape_string(key.metric) << "\","
         << "\"ntiles\":" << result.ntiles << ","
         << "\"prefetch\":" << result.prefetch_dist << ","
         << "\"memory\":\"" << dpas_memory_pattern_name(result.memory_pattern) << "\","
         << "\"tops\":" << result.tops << ","
         << "\"bandwidth\":" << result.bandwidth << "}\n";
    return true;
}

struct DeviceBuffer {
    void *                 ptr = nullptr;
    size_t                 size = 0;
    ggml_backend_buffer_t  buffer = nullptr;
    MemoryMode             mode = MemoryMode::USM_DEVICE;
};

class BenchmarkHarness {
public:
    bool run(const BenchmarkConfig & config, BenchmarkOutput & out) const;

private:
    static bool run_reference(const BenchmarkConfig & config,
                              sycl::queue & queue,
                              BenchmarkOutput & out);
    static bool run_mmq(const BenchmarkConfig & config,
                        sycl::queue & queue,
                        BenchmarkOutput & out);
    static bool run_dpas(const BenchmarkConfig & config,
                         sycl::queue & queue,
                         BenchmarkOutput & out);
    static bool apply_dpas_tuning(BenchmarkConfig & config,
                                  sycl::queue & queue,
                                  std::string & error);
    static bool run_dpas_autotune(const BenchmarkConfig & base,
                                  sycl::queue & queue,
                                  DpasTuneResult & best,
                                  std::string & error);
    static DeviceBuffer allocate_buffer(size_t bytes,
                                        MemoryMode mode,
                                        sycl::queue & queue,
                                        int device_id,
                                        std::string & error);
    static void free_buffer(DeviceBuffer & buffer, sycl::queue & queue);
    static bool copy_to_device(DeviceBuffer & buffer,
                               const void * src,
                               size_t bytes,
                               sycl::queue & queue,
                               std::string & error);
    static bool copy_to_host(void * dst,
                             const DeviceBuffer & buffer,
                             size_t bytes,
                             sycl::queue & queue,
                             std::string & error);
    static bool get_event_timing(const sycl::event & event,
                                 uint64_t & start,
                                 uint64_t & end,
                                 std::string & error);
    static bool dequantize_weights_row(ggml_type type,
                                       const void * src,
                                       float * dst,
                                       int64_t k,
                                       std::string & error);
    static void dequantize_q8_1_row_aos(const block_q8_1 * src, float * dst, int64_t k);
    static void dequantize_q8_1_row_soa(const uint8_t * src, float * dst, int64_t k, int64_t k_padded);
};

inline const char * memory_mode_name(MemoryMode mode) {
    switch (mode) {
        case MemoryMode::USM_DEVICE: return "USM_DEVICE";
        case MemoryMode::USM_SHARED: return "USM_SHARED";
        case MemoryMode::BUFFER: return "BUFFER";
        default: return "UNKNOWN";
    }
}

inline DeviceBuffer BenchmarkHarness::allocate_buffer(size_t bytes,
                                                      MemoryMode mode,
                                                      sycl::queue & queue,
                                                      int device_id,
                                                      std::string & error) {
    DeviceBuffer out;
    out.size = bytes;
    out.mode = mode;
    if (bytes == 0) {
        error = "buffer size is 0";
        return out;
    }

    switch (mode) {
        case MemoryMode::USM_DEVICE:
            out.ptr = sycl::malloc_device(bytes, queue);
            if (!out.ptr) {
                error = "sycl::malloc_device failed";
            }
            break;
        case MemoryMode::USM_SHARED:
            out.ptr = sycl::malloc_shared(bytes, queue);
            if (!out.ptr) {
                error = "sycl::malloc_shared failed";
            }
            break;
        case MemoryMode::BUFFER:
            {
                ggml_backend_buffer_type_t buft = ggml_backend_sycl_buffer_type(device_id);
                if (!buft) {
                    error = "ggml_backend_sycl_buffer_type returned null";
                    break;
                }
                out.buffer = ggml_backend_buft_alloc_buffer(buft, bytes);
                if (!out.buffer) {
                    error = "ggml_backend_buft_alloc_buffer failed";
                    break;
                }
                out.ptr = ggml_backend_buffer_get_base(out.buffer);
                if (!out.ptr) {
                    error = "ggml_backend_buffer_get_base returned null";
                    break;
                }
            }
            break;
    }
    return out;
}

inline void BenchmarkHarness::free_buffer(DeviceBuffer & buffer, sycl::queue & queue) {
    if (buffer.buffer) {
        ggml_backend_buffer_free(buffer.buffer);
        buffer.buffer = nullptr;
        buffer.ptr = nullptr;
        buffer.size = 0;
        return;
    }
    if (buffer.ptr) {
        sycl::free(buffer.ptr, queue);
        buffer.ptr = nullptr;
        buffer.size = 0;
    }
}

inline bool BenchmarkHarness::copy_to_device(DeviceBuffer & buffer,
                                             const void * src,
                                             size_t bytes,
                                             sycl::queue & queue,
                                             std::string & error) {
    if (!buffer.ptr || bytes == 0) {
        return true;
    }
    if (bytes > buffer.size) {
        error = "copy_to_device: source bytes exceed buffer size";
        return false;
    }
    try {
        queue.memcpy(buffer.ptr, src, bytes);
    } catch (const std::exception & e) {
        error = e.what();
        return false;
    }
    return true;
}

inline bool BenchmarkHarness::copy_to_host(void * dst,
                                           const DeviceBuffer & buffer,
                                           size_t bytes,
                                           sycl::queue & queue,
                                           std::string & error) {
    if (!buffer.ptr || bytes == 0) {
        return true;
    }
    if (bytes > buffer.size) {
        error = "copy_to_host: requested bytes exceed buffer size";
        return false;
    }
    try {
        queue.memcpy(dst, buffer.ptr, bytes);
    } catch (const std::exception & e) {
        error = e.what();
        return false;
    }
    return true;
}

inline bool BenchmarkHarness::get_event_timing(const sycl::event & event,
                                               uint64_t & start,
                                               uint64_t & end,
                                               std::string & error) {
    try {
        start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
        end   = event.get_profiling_info<sycl::info::event_profiling::command_end>();
    } catch (const std::exception & e) {
        error = e.what();
        return false;
    }
    if (start == 0 || end == 0 || end <= start) {
        error = "profiling info unavailable (queue missing enable_profiling?)";
        return false;
    }
    return true;
}

inline bool BenchmarkHarness::dequantize_weights_row(ggml_type type,
                                                     const void * src,
                                                     float * dst,
                                                     int64_t k,
                                                     std::string & error) {
    switch (type) {
        case GGML_TYPE_Q4_0: dequantize_row_q4_0((const block_q4_0 *) src, dst, k); break;
        case GGML_TYPE_Q4_1: dequantize_row_q4_1((const block_q4_1 *) src, dst, k); break;
        case GGML_TYPE_Q5_0: dequantize_row_q5_0((const block_q5_0 *) src, dst, k); break;
        case GGML_TYPE_Q5_1: dequantize_row_q5_1((const block_q5_1 *) src, dst, k); break;
        case GGML_TYPE_Q8_0: dequantize_row_q8_0((const block_q8_0 *) src, dst, k); break;
        case GGML_TYPE_MXFP4: dequantize_row_mxfp4((const block_mxfp4 *) src, dst, k); break;
        case GGML_TYPE_Q2_K: dequantize_row_q2_K((const block_q2_K *) src, dst, k); break;
        case GGML_TYPE_Q3_K: dequantize_row_q3_K((const block_q3_K *) src, dst, k); break;
        case GGML_TYPE_Q4_K: dequantize_row_q4_K((const block_q4_K *) src, dst, k); break;
        case GGML_TYPE_Q5_K: dequantize_row_q5_K((const block_q5_K *) src, dst, k); break;
        case GGML_TYPE_Q6_K: dequantize_row_q6_K((const block_q6_K *) src, dst, k); break;
        default:
            error = "unsupported quant type for dequantize";
            return false;
    }
    return true;
}

inline void BenchmarkHarness::dequantize_q8_1_row_aos(const block_q8_1 * src, float * dst, int64_t k) {
    const int64_t blocks = k / QK8_1;
    for (int64_t b = 0; b < blocks; ++b) {
        const block_q8_1 & block = src[b];
        const sycl::half2 ds = *reinterpret_cast<const sycl::half2 *>(&block.ds);
        const float d = static_cast<float>(ds.x());
        for (int i = 0; i < QK8_1; ++i) {
            dst[b * QK8_1 + i] = d * static_cast<float>(block.qs[i]);
        }
    }
}

inline void BenchmarkHarness::dequantize_q8_1_row_soa(const uint8_t * src, float * dst, int64_t k, int64_t k_padded) {
    const int64_t blocks = k / QK8_1;
    const int8_t * qs = reinterpret_cast<const int8_t *>(src);
    const uint8_t * ds = src + k;

    for (int64_t b = 0; b < blocks; ++b) {
        const sycl::half2 ds_vals = *reinterpret_cast<const sycl::half2 *>(ds + b * sizeof(sycl::half2));
        const float d = static_cast<float>(ds_vals.x());
        for (int i = 0; i < QK8_1; ++i) {
            dst[b * QK8_1 + i] = d * static_cast<float>(qs[b * QK8_1 + i]);
        }
    }

    if (k_padded > k) {
        std::fill(dst + k, dst + k_padded, 0.0f);
    }
}

inline bool BenchmarkHarness::run_reference(const BenchmarkConfig & config,
                                            sycl::queue & queue,
                                            BenchmarkOutput & out) {
    ReferenceMetrics metrics{};
    std::string error;

    if (config.dim_m <= 0 || config.dim_k <= 0 || config.batch_size <= 0) {
        out.error = "Invalid dimensions.";
        return false;
    }

    const int64_t m = config.dim_m;
    const int64_t n = config.batch_size;
    const int64_t k = config.dim_k;
    const int64_t k_padded = GGML_PAD(k, QK8_1);

    switch (config.kernel_kind) {
        case KernelKind::ONEDNN_FP16_GEMM: {
            GeneratedWeights weights;
            if (!generate_quantized_weights(config.quant_type, GGML_LAYOUT_AOS, m, k, false, weights)) {
                out.error = "Failed to generate quantized weights for FP16 GEMM baseline.";
                return false;
            }
            GeneratedActivations activations = generate_activations(n, k, k_padded, false, true, false);
            if (!run_onednn_fp16_gemm(weights, activations, m, n, k, config.quant_type,
                                      config.warmup_iterations, config.measure_iterations,
                                      queue, metrics, error)) {
                out.error = error;
                return false;
            }
            break;
        }
        case KernelKind::ONEDNN_INT8_GEMM: {
            GeneratedWeights weights;
            if (!generate_quantized_weights(config.quant_type, GGML_LAYOUT_AOS, m, k, false, weights)) {
                out.error = "Failed to generate quantized weights for INT8 GEMM baseline.";
                return false;
            }
            GeneratedActivations activations = generate_activations(n, k, k_padded, true, false, false);
            if (!run_onednn_int8_gemm(weights, activations, m, n, k, config.quant_type,
                                      config.warmup_iterations, config.measure_iterations,
                                      queue, metrics, error)) {
                out.error = error;
                return false;
            }
            break;
        }
        case KernelKind::MEMORY_BANDWIDTH: {
            size_t bytes = config.transfer_bytes;
            if (bytes == 0) {
                bytes = static_cast<size_t>(config.batch_size) * (1ull << 30);
            }
            if (!run_memory_bandwidth(bytes, config.warmup_iterations, config.measure_iterations,
                                      queue, metrics, error)) {
                out.error = error;
                return false;
            }
            break;
        }
        case KernelKind::ROOFLINE_COMPUTE: {
            size_t elements = (config.roofline_elements > 0)
                ? static_cast<size_t>(config.roofline_elements)
                : static_cast<size_t>(config.dim_m) * static_cast<size_t>(config.dim_k);
            int ops = config.roofline_ops > 0 ? config.roofline_ops : 1000;
            if (!run_roofline_compute(elements, ops, config.warmup_iterations, config.measure_iterations,
                                      queue, metrics, error)) {
                out.error = error;
                return false;
            }
            break;
        }
        default:
            out.error = "Unsupported reference kernel kind.";
            return false;
    }

    out.result.latency_us = metrics.total_us;
    out.result.latency_std = 0.0;
    out.result.throughput_tps = 0.0;
    out.result.bandwidth_gbps = metrics.bandwidth_gbps;
    out.result.ref_total_us = metrics.total_us;
    out.result.ref_dequant_us = metrics.dequant_us;
    out.result.ref_gemm_us = metrics.gemm_us;
    out.result.ref_scale_us = metrics.scale_us;
    out.result.ref_tflops = metrics.tflops;
    out.result.ref_tops = metrics.tops;
    out.result.ref_bandwidth_gbps = metrics.bandwidth_gbps;
    out.result.ref_arith_intensity = metrics.arithmetic_intensity;

    out.ok = true;
    return true;
}

inline bool BenchmarkHarness::run_dpas(const BenchmarkConfig & config,
                                       sycl::queue & queue,
                                       BenchmarkOutput & out) {
    int device_id = config.device_id;
    if (device_id < 0) {
        device_id = get_current_device_id();
    }
    if (config.dim_m <= 0 || config.dim_n <= 0 || config.dim_k <= 0) {
        out.error = "Invalid dimensions for DPAS benchmark.";
        return false;
    }
    if (config.dpas_repeat != 1 && config.dpas_repeat != 2 &&
        config.dpas_repeat != 4 && config.dpas_repeat != 8) {
        out.error = "DPAS repeat must be 1,2,4,8.";
        return false;
    }
    if (config.dpas_n_tile_repeats != 1 && config.dpas_n_tile_repeats != 2 &&
        config.dpas_n_tile_repeats != 4 && config.dpas_n_tile_repeats != 8) {
        out.error = "DPAS n-tile repeats must be 1, 2, 4, or 8.";
        return false;
    }
    if ((config.dpas_memory_pattern == DpasMemoryPattern::SLM_BUFFER ||
         config.dpas_memory_pattern == DpasMemoryPattern::DOUBLE_BUFFER) &&
        config.dpas_n_tile_repeats > 4) {
        out.error = "DPAS SLM patterns only support n-tile repeats = 1, 2, or 4.";
        return false;
    }
    if (config.dpas_type_a != config.dpas_type_b) {
        out.error = "DPAS exploration requires matching A/B types.";
        return false;
    }
    if (config.dpas_type_a != DpasType::INT8 && config.dpas_type_acc == DpasAccType::INT32) {
        out.error = "INT32 accumulation only supported with INT8 inputs.";
        return false;
    }

    const int k_per = dpas_k_per_tile(config.dpas_type_a, config.dpas_type_b);
    if (k_per <= 0) {
        out.error = "Unable to determine DPAS K tile for requested types.";
        return false;
    }
    if ((config.dim_k % k_per) != 0 || (config.dim_m % config.dpas_repeat) != 0 || (config.dim_n % 16) != 0) {
        out.error = "DPAS dims must be multiples of repeat, 16, and K tile.";
        return false;
    }
    if (((config.dim_n / 16) % config.dpas_n_tile_repeats) != 0) {
        out.error = "DPAS dim_n/16 must be divisible by n-tile repeats.";
        return false;
    }

    const int64_t m_tiles = config.dim_m / config.dpas_repeat;
    const int64_t n_tiles = config.dim_n / 16;
    const int64_t k_tiles = config.dim_k / k_per;

    const size_t a_tile_elems = static_cast<size_t>(config.dpas_repeat) * static_cast<size_t>(k_per);
    const size_t b_tile_elems = static_cast<size_t>(k_per) * 16u;
    const size_t a_elems = static_cast<size_t>(m_tiles) * static_cast<size_t>(k_tiles) * a_tile_elems;
    const size_t b_elems = static_cast<size_t>(n_tiles) * static_cast<size_t>(k_tiles) * b_tile_elems;
    const size_t c_elems = static_cast<size_t>(config.dim_m) * static_cast<size_t>(config.dim_n);

    std::string error;
    const size_t misalign = config.dpas_misaligned ? 1u : 0u;

    auto alloc = [&](size_t bytes, DeviceBuffer & buf) -> bool {
        buf = allocate_buffer(bytes + misalign, config.memory_mode, queue, device_id, error);
        if (!error.empty()) {
            out.error = error;
            return false;
        }
        return true;
    };

    auto copy_to = [&](void * dst, const void * src, size_t bytes) -> bool {
        if (!dst || bytes == 0) {
            return true;
        }
        queue.memcpy(dst, src, bytes);
        return true;
    };

    auto fill_int8 = [&](std::vector<int8_t> & vec, int8_t base) {
        for (size_t i = 0; i < vec.size(); ++i) {
            vec[i] = static_cast<int8_t>(base + static_cast<int8_t>(i % 7));
        }
    };

    auto fill_fp32 = [&](auto & vec, float base) {
        using ElemT = std::decay_t<decltype(vec[0])>;
        for (size_t i = 0; i < vec.size(); ++i) {
            vec[i] = static_cast<ElemT>(base + static_cast<float>((i % 13) - 6) * 0.01f);
        }
    };

    auto pack_vnni = [&](auto * dst, const auto * src, size_t rows, size_t cols, int pack) {
        size_t idx = 0;
        for (size_t k0 = 0; k0 < rows; k0 += static_cast<size_t>(pack)) {
            for (size_t col = 0; col < cols; ++col) {
                for (int p = 0; p < pack; ++p) {
                    dst[idx++] = src[(k0 + static_cast<size_t>(p)) * cols + col];
                }
            }
        }
    };

    auto make_inputs = [&](auto type_tag_a, auto type_tag_b, auto type_tag_c) -> bool {
        using TA = decltype(type_tag_a);
        using TB = decltype(type_tag_b);
        using TC = decltype(type_tag_c);

        std::vector<TA> a_host(a_elems);
        std::vector<TB> b_host(b_elems);
        std::vector<TC> c_host(c_elems);

        if constexpr (std::is_same_v<TA, int8_t>) {
            fill_int8(a_host, 1);
        } else {
            fill_fp32(a_host, 0.5f);
        }

        const int elem_bits = static_cast<int>(sizeof(TB) * 8);
        const int pack = (elem_bits > 0) ? (32 / elem_bits) : 1;

        const size_t per_tile_k = static_cast<size_t>(k_per);
        const size_t per_tile_n = 16;
        std::vector<TB> b_tile(per_tile_k * per_tile_n);
        for (size_t i = 0; i < b_tile.size(); ++i) {
            if constexpr (std::is_same_v<TB, int8_t>) {
                b_tile[i] = static_cast<int8_t>(1 + (i % 7));
            } else {
                b_tile[i] = static_cast<TB>(0.25f + static_cast<float>((i % 11) - 5) * 0.01f);
            }
        }

        for (int64_t tile_n = 0; tile_n < n_tiles; ++tile_n) {
            for (int64_t tile_k = 0; tile_k < k_tiles; ++tile_k) {
                const size_t offset = static_cast<size_t>((tile_n * k_tiles + tile_k) * b_tile_elems);
                pack_vnni(b_host.data() + offset, b_tile.data(), per_tile_k, per_tile_n, pack);
            }
        }

        std::fill(c_host.begin(), c_host.end(), static_cast<TC>(0));

        DeviceBuffer a_buf;
        DeviceBuffer b_buf;
        DeviceBuffer c_buf;

        if (!alloc(a_host.size() * sizeof(TA), a_buf) ||
            !alloc(b_host.size() * sizeof(TB), b_buf) ||
            !alloc(c_host.size() * sizeof(TC), c_buf)) {
            if (a_buf.ptr) free_buffer(a_buf, queue);
            if (b_buf.ptr) free_buffer(b_buf, queue);
            if (c_buf.ptr) free_buffer(c_buf, queue);
            return false;
        }

        void * a_ptr = static_cast<uint8_t *>(a_buf.ptr) + misalign;
        void * b_ptr = static_cast<uint8_t *>(b_buf.ptr) + misalign;
        void * c_ptr = c_buf.ptr;

        if (config.memory_mode == MemoryMode::BUFFER && config.dpas_misaligned) {
            free_buffer(a_buf, queue);
            free_buffer(b_buf, queue);
            free_buffer(c_buf, queue);
            out.error = "Misaligned DPAS not supported with BUFFER memory mode.";
            return false;
        }

        copy_to(a_ptr, a_host.data(), a_host.size() * sizeof(TA));
        copy_to(b_ptr, b_host.data(), b_host.size() * sizeof(TB));
        copy_to(c_ptr, c_host.data(), c_host.size() * sizeof(TC));

        queue.wait_and_throw();

        DpasBenchArgs args{};
        args.a = a_ptr;
        args.b = b_ptr;
        args.c = c_ptr;
        args.m = config.dim_m;
        args.n = config.dim_n;
        args.k = config.dim_k;
        args.type_a = config.dpas_type_a;
        args.type_b = config.dpas_type_b;
        args.type_acc = config.dpas_type_acc;
        args.memory_pattern = config.dpas_memory_pattern;
        args.grf_mode = config.dpas_grf_mode;
        args.repeat = config.dpas_repeat;
        args.n_tile_repeats = config.dpas_n_tile_repeats;
        args.misaligned = config.dpas_misaligned;
        args.stream = &queue;

        auto launch_kernel = [&](std::vector<sycl::event> * evs, std::string & launch_error) -> bool {
            if (config.kernel_name == "dpas_baseline") {
                return run_dpas_baseline(args, evs, launch_error);
            }
            if (config.kernel_name == "dpas_memory_patterns") {
                return run_dpas_memory_patterns(args, evs, launch_error);
            }
            return run_dpas_sweep(args, evs, launch_error);
        };

        for (int i = 0; i < config.warmup_iterations; ++i) {
            std::string launch_error;
            if (!launch_kernel(nullptr, launch_error)) {
                out.error = launch_error.empty() ? "Warmup DPAS launch failed." : launch_error;
                free_buffer(a_buf, queue);
                free_buffer(b_buf, queue);
                free_buffer(c_buf, queue);
                return false;
            }
        }

        queue.wait_and_throw();

        std::vector<sycl::event> events;
        events.reserve(static_cast<size_t>(config.measure_iterations));

        for (int i = 0; i < config.measure_iterations; ++i) {
            std::string launch_error;
            if (!launch_kernel(&events, launch_error)) {
                out.error = launch_error.empty() ? "Measured DPAS launch failed." : launch_error;
                free_buffer(a_buf, queue);
                free_buffer(b_buf, queue);
                free_buffer(c_buf, queue);
                return false;
            }
        }

        if (!events.empty()) {
            sycl::event::wait(events);
        }

        std::vector<double> iter_ns;
        iter_ns.reserve(events.size());
        for (const auto & ev : events) {
            uint64_t start = 0;
            uint64_t finish = 0;
            if (!get_event_timing(ev, start, finish, error)) {
                out.error = error;
                free_buffer(a_buf, queue);
                free_buffer(b_buf, queue);
                free_buffer(c_buf, queue);
                return false;
            }
            iter_ns.push_back(static_cast<double>(finish - start));
        }

        double sum = 0.0;
        for (double v : iter_ns) {
            sum += v;
        }
        const double mean_ns = (iter_ns.empty()) ? 0.0 : sum / static_cast<double>(iter_ns.size());
        double variance = 0.0;
        for (double v : iter_ns) {
            const double diff = v - mean_ns;
            variance += diff * diff;
        }
        const double std_ns = (iter_ns.size() > 1) ? std::sqrt(variance / static_cast<double>(iter_ns.size() - 1)) : 0.0;

        const double mean_s = mean_ns * 1e-9;
        const double ops = 2.0 * static_cast<double>(config.dim_m) * static_cast<double>(config.dim_n) * static_cast<double>(config.dim_k);
        const double tops = (mean_s > 0.0) ? (ops / mean_s / 1.0e12) : 0.0;

        const double bytes_moved = static_cast<double>(a_host.size() * sizeof(TA) +
                                                       b_host.size() * sizeof(TB) +
                                                       c_host.size() * sizeof(TC));
        const double bandwidth = (mean_s > 0.0) ? (bytes_moved / mean_s / 1.0e9) : 0.0;

        out.result.throughput_tops = tops;
        out.result.latency_ns = mean_ns;
        out.result.latency_us = mean_ns * 1e-3;
        out.result.latency_std = std_ns * 1e-3;
        out.result.bandwidth_gbps = bandwidth;
        out.ok = true;

        free_buffer(a_buf, queue);
        free_buffer(b_buf, queue);
        free_buffer(c_buf, queue);
        return true;
    };

    if (config.dpas_type_a == DpasType::INT8) {
        if (config.dpas_type_acc == DpasAccType::INT32) {
            return make_inputs(int8_t{}, int8_t{}, int{});
        }
        return make_inputs(int8_t{}, int8_t{}, float{});
    }
    if (config.dpas_type_a == DpasType::FP16) {
        return make_inputs(sycl::half{}, sycl::half{}, float{});
    }
    if (config.dpas_type_a == DpasType::BF16) {
        return make_inputs(sycl::ext::oneapi::bfloat16{}, sycl::ext::oneapi::bfloat16{}, float{});
    }

    out.error = "Unsupported DPAS type.";
    return false;
}

inline bool BenchmarkHarness::apply_dpas_tuning(BenchmarkConfig & config,
                                                sycl::queue & queue,
                                                std::string & error) {
    if (!config.dpas_device_opt && !config.dpas_autotune) {
        return true;
    }
    if (config.kernel_name != "dpas_memory_patterns") {
        return true;
    }
    const DeviceInfo info = query_device_info(queue.get_device());
    const std::string device_key = make_device_key(info);
    const int prefetch_override = config.dpas_autotune_override_prefetch;
    const int ntiles_override = config.dpas_autotune_override_ntiles;
    bool tuned = false;

    if (config.dpas_memory_explicit || config.dpas_ntiles_explicit) {
        std::fprintf(stderr,
                     "[DPAS tune] Skipping device tuning; explicit --dpas-memory/--dpas-ntiles provided.\n");
        return true;
    }

    DpasTuneKey key{};
    key.device_key = device_key;
    key.dim_m = config.dim_m;
    key.dim_n = config.dim_n;
    key.dim_k = config.dim_k;
    key.repeat = config.dpas_repeat;
    key.type_a = config.dpas_type_a;
    key.type_b = config.dpas_type_b;
    key.type_acc = config.dpas_type_acc;
    key.grf_mode = config.dpas_grf_mode;
    key.misaligned = config.dpas_misaligned;
    key.metric = dpas_tune_metric_name(config.dpas_autotune_metric);

    if (config.dpas_autotune) {
        DpasTuneResult cached{};
        if (!config.dpas_autotune_force &&
            load_dpas_tuning_cache(config.dpas_autotune_cache, key, cached)) {
            config.dpas_n_tile_repeats = cached.ntiles;
            config.dpas_memory_pattern = cached.memory_pattern;
            const bool prefer_bandwidth = (key.metric == "bandwidth");
            const double metric_value = prefer_bandwidth ? cached.bandwidth : cached.tops;
            std::fprintf(stderr,
                         "[DPAS tune] Cache hit (%s): metric=%s ntiles=%d prefetch=%d memory=%s %s=%.2f\n",
                         config.dpas_autotune_cache.c_str(),
                         key.metric.c_str(),
                         cached.ntiles,
                         cached.prefetch_dist,
                         dpas_memory_pattern_name(cached.memory_pattern),
                         prefer_bandwidth ? "bandwidth" : "tops",
                         metric_value);
            tuned = true;
        } else {
            DpasTuneResult best{};
            if (run_dpas_autotune(config, queue, best, error)) {
                config.dpas_n_tile_repeats = best.ntiles;
                config.dpas_memory_pattern = best.memory_pattern;
                append_dpas_tuning_cache(config.dpas_autotune_cache, info, key, best);
                const bool prefer_bandwidth = (key.metric == "bandwidth");
                const double metric_value = prefer_bandwidth ? best.bandwidth : best.tops;
                std::fprintf(stderr,
                             "[DPAS tune] Autotune picked metric=%s ntiles=%d prefetch=%d memory=%s %s=%.2f (cache=%s)\n",
                             key.metric.c_str(),
                             best.ntiles,
                             best.prefetch_dist,
                             dpas_memory_pattern_name(best.memory_pattern),
                             prefer_bandwidth ? "bandwidth" : "tops",
                             metric_value,
                             config.dpas_autotune_cache.c_str());
                tuned = true;
            } else if (!error.empty()) {
                std::fprintf(stderr, "[DPAS tune] Autotune failed: %s\n", error.c_str());
                error.clear();
            }
        }
    }

    if (!tuned && config.dpas_device_opt) {
        DpasDimTuningEntry entry{};
        if (dpas_lookup_dim_tuning(config, info, entry)) {
            if (!config.dpas_ntiles_explicit) {
                config.dpas_n_tile_repeats = entry.ntiles;
            }
            if (!config.dpas_memory_explicit) {
                config.dpas_memory_pattern = entry.pattern;
            }
            if (!config.dpas_grf_explicit) {
                config.dpas_grf_mode = entry.grf;
            }
            if (!config.dpas_acc_explicit &&
                config.dpas_type_a == DpasType::INT8 &&
                config.dpas_type_b == DpasType::INT8) {
                config.dpas_type_acc = entry.acc;
            }
            const int prefetch_dist = dpas_prefetch_dist_from_pattern(config.dpas_memory_pattern);
            std::fprintf(stderr,
                         "[DPAS tune] Heuristic table: ntiles=%d prefetch=%d memory=%s grf=%s acc=%s (repeat=%d, dim_n=%lld)\n",
                         config.dpas_n_tile_repeats,
                         prefetch_dist,
                         dpas_memory_pattern_name(config.dpas_memory_pattern),
                         dpas_grf_mode_name(config.dpas_grf_mode),
                         dpas_acc_name(config.dpas_type_acc),
                         config.dpas_repeat,
                         static_cast<long long>(config.dim_n));
            tuned = true;
        }
    }

    if (!tuned && config.dpas_device_opt) {
        int ntiles = dpas_heuristic_ntiles(config, info);
        config.dpas_memory_pattern = dpas_heuristic_memory_pattern(config, info);
        int prefetch_dist = dpas_prefetch_dist_from_pattern(config.dpas_memory_pattern);
        config.dpas_n_tile_repeats = ntiles;
        std::fprintf(stderr,
                     "[DPAS tune] Heuristic ntiles=%d prefetch=%d memory=%s (repeat=%d, EU=%u, SLM=%zu KB)\n",
                     ntiles,
                     prefetch_dist,
                     dpas_memory_pattern_name(config.dpas_memory_pattern),
                     config.dpas_repeat,
                     info.max_compute_units,
                     info.local_mem_size / 1024);
    }

    if (ntiles_override > 0) {
        config.dpas_n_tile_repeats = ntiles_override;
        std::fprintf(stderr, "[DPAS tune] Override ntiles=%d\n", ntiles_override);
    }
    if (prefetch_override > 0) {
        config.dpas_memory_pattern = dpas_prefetch_pattern_from_dist(prefetch_override);
        std::fprintf(stderr, "[DPAS tune] Override prefetch=%d\n", prefetch_override);
    }

    return true;
}

inline bool BenchmarkHarness::run_dpas_autotune(const BenchmarkConfig & base,
                                                sycl::queue & queue,
                                                DpasTuneResult & best,
                                                std::string & error) {
    BenchmarkConfig config = base;
    config.validate = false;
    config.include_percentiles = false;
    config.include_ref_metrics = false;
    config.warmup_iterations = std::min(config.warmup_iterations, 5);
    config.measure_iterations = std::min(config.measure_iterations, 20);
    config.dpas_memory_pattern = DpasMemoryPattern::LSC_PREFETCH;

    const int64_t n_tiles = config.dim_n / 16;
    const int ntiles_candidates[] = {1, 2, 4, 8};
    struct PatternCandidate {
        DpasMemoryPattern pattern;
        int prefetch_dist;
    };
    const PatternCandidate pattern_candidates[] = {
        {DpasMemoryPattern::LSC_STREAMING, 0},
        {DpasMemoryPattern::LSC_PREFETCH, 1},
        {DpasMemoryPattern::LSC_PREFETCH_2, 2},
        {DpasMemoryPattern::LSC_PREFETCH_3, 3},
        {DpasMemoryPattern::LSC_PREFETCH_4, 4},
        {DpasMemoryPattern::LSC_PREFETCH_5, 5},
        {DpasMemoryPattern::LSC_PREFETCH_6, 6},
        {DpasMemoryPattern::LSC_PREFETCH_8, 8},
        {DpasMemoryPattern::LSC_PREFETCH_10, 10},
    };

    // Precompile all candidates to reduce JIT bias during the timed sweep.
    {
        BenchmarkConfig warm = config;
        warm.warmup_iterations = 0;
        warm.measure_iterations = 1;
        for (const auto & candidate : pattern_candidates) {
            warm.dpas_memory_pattern = candidate.pattern;
            for (int ntiles : ntiles_candidates) {
                if (n_tiles % ntiles != 0) {
                    continue;
                }
                warm.dpas_n_tile_repeats = ntiles;
                BenchmarkOutput tmp;
                run_dpas(warm, queue, tmp);
            }
        }
    }

    auto median_value = [](std::vector<double> values) -> double {
        if (values.empty()) {
            return 0.0;
        }
        const size_t mid = values.size() / 2;
        std::nth_element(values.begin(), values.begin() + mid, values.end());
        double med = values[mid];
        if ((values.size() & 1u) == 0u && mid > 0) {
            const double lower = *std::max_element(values.begin(), values.begin() + mid);
            med = 0.5 * (med + lower);
        }
        return med;
    };

    const int sweep_samples = 5;
    DpasTuneResult best_local{};
    for (const auto & candidate : pattern_candidates) {
        config.dpas_memory_pattern = candidate.pattern;
        for (int ntiles : ntiles_candidates) {
            if (n_tiles % ntiles != 0) {
                continue;
            }
            config.dpas_n_tile_repeats = ntiles;
            std::vector<double> tops_samples;
            std::vector<double> bw_samples;
            tops_samples.reserve(sweep_samples);
            bw_samples.reserve(sweep_samples);
            for (int sample = 0; sample < sweep_samples; ++sample) {
                BenchmarkOutput tmp;
                if (!run_dpas(config, queue, tmp)) {
                    continue;
                }
                tops_samples.push_back(tmp.result.throughput_tops);
                bw_samples.push_back(tmp.result.bandwidth_gbps);
            }
            if (tops_samples.empty()) {
                continue;
            }
            const double tops = median_value(tops_samples);
            const double bandwidth = median_value(bw_samples);
            const double score = (base.dpas_autotune_metric == DpasTuneMetric::BANDWIDTH) ? bandwidth : tops;
            const double best_score =
                (base.dpas_autotune_metric == DpasTuneMetric::BANDWIDTH) ? best_local.bandwidth : best_local.tops;
            if (!best_local.valid || score > best_score) {
                best_local.valid = true;
                best_local.ntiles = ntiles;
                best_local.prefetch_dist = candidate.prefetch_dist;
                best_local.tops = tops;
                best_local.bandwidth = bandwidth;
                best_local.memory_pattern = candidate.pattern;
            }
        }
    }
    if (!best_local.valid) {
        error = "No valid DPAS autotune candidate.";
        return false;
    }
    best = best_local;
    return true;
}

inline bool BenchmarkHarness::run(const BenchmarkConfig & config, BenchmarkOutput & out) const {
    out = {};
    out.config = config;

    const int device_count = ggml_backend_sycl_get_device_count();
    if (device_count <= 0) {
        out.error = "No SYCL device found. Set ONEAPI_DEVICE_SELECTOR.";
        return false;
    }

    int device_id = config.device_id;
    if (device_id < 0) {
        device_id = get_current_device_id();
    }
    if (device_id < 0 || device_id >= device_count) {
        out.error = "Invalid device id.";
        return false;
    }

    dpct::select_device(static_cast<unsigned int>(device_id));
    auto & device = dpct::get_device(device_id);
    auto & base_queue = device.default_queue();
    sycl::async_handler handler = [](sycl::exception_list exceptions) {
        for (const auto & e : exceptions) {
            try {
                std::rethrow_exception(e);
            } catch (const std::exception & ex) {
                std::fprintf(stderr, "SYCL async exception: %s\n", ex.what());
            }
        }
    };
    sycl::queue queue(base_queue.get_context(),
                      device,
                      handler,
                      sycl::property_list{ sycl::property::queue::enable_profiling{} });

    if (config.kernel_kind == KernelKind::DPAS_EXPLORATION) {
        BenchmarkConfig tuned = config;
        std::string tune_error;
        if (!apply_dpas_tuning(tuned, queue, tune_error)) {
            out.error = tune_error;
            return false;
        }
        out.config = tuned;
        return run_dpas(tuned, queue, out);
    }

    if (config.kernel_kind == KernelKind::MMQ) {
        return run_mmq(config, queue, out);
    }

    if (config.kernel_kind != KernelKind::MMVQ) {
        return run_reference(config, queue, out);
    }

    if (!kernel_supports_layout(config.quant_type, config.layout)) {
        out.error = "Kernel layout not supported for quant type.";
        return false;
    }

    if (config.dim_m <= 0 || config.dim_k <= 0 || config.batch_size <= 0) {
        out.error = "Invalid dimensions.";
        return false;
    }

    const int64_t k_padded = GGML_PAD(config.dim_k, MATRIX_ROW_PADDING);
    const size_t activation_bytes =
        static_cast<size_t>(config.batch_size) * static_cast<size_t>(k_padded) * sizeof(block_q8_1) / QK8_1;
    const size_t output_bytes =
        static_cast<size_t>(config.batch_size) * static_cast<size_t>(config.dim_m) * sizeof(float);

    GeneratedWeights weights;
    if (!generate_quantized_weights(config.quant_type,
                                    config.layout,
                                    config.dim_m,
                                    config.dim_k,
                                    config.validate,
                                    weights)) {
        out.error = "Failed to generate quantized weights for requested layout/type.";
        return false;
    }

    const bool activations_soa = (config.layout == GGML_LAYOUT_SOA || config.layout == GGML_LAYOUT_COALESCED);
    GeneratedActivations activations = generate_activations(config.batch_size,
                                                            config.dim_k,
                                                            k_padded,
                                                            config.validate,
                                                            config.validate,
                                                            activations_soa);

    const size_t weight_bytes = (config.layout == GGML_LAYOUT_AOS) ? weights.bytes_aos : weights.bytes_layout;
    const size_t total_bytes = weight_bytes + activation_bytes + output_bytes;

    size_t free_mem = 0;
    size_t total_mem = 0;
    ggml_backend_sycl_get_device_memory(device_id, &free_mem, &total_mem);
    if (total_mem > 0 && total_bytes > total_mem) {
        char msg[256];
        std::snprintf(msg,
                      sizeof(msg),
                      "Required memory (%.2f GB) exceeds device memory (%.2f GB).",
                      total_bytes / (1024.0 * 1024.0 * 1024.0),
                      total_mem / (1024.0 * 1024.0 * 1024.0));
        out.error = msg;
        return false;
    }

    std::string error;
    DeviceBuffer weight_buf = allocate_buffer(weight_bytes, config.memory_mode, queue, device_id, error);
    if (!error.empty()) {
        out.error = error;
        return false;
    }
    DeviceBuffer act_buf = allocate_buffer(activation_bytes, config.memory_mode, queue, device_id, error);
    if (!error.empty()) {
        free_buffer(weight_buf, queue);
        out.error = error;
        return false;
    }
    DeviceBuffer out_buf = allocate_buffer(output_bytes, config.memory_mode, queue, device_id, error);
    if (!error.empty()) {
        free_buffer(weight_buf, queue);
        free_buffer(act_buf, queue);
        out.error = error;
        return false;
    }

    const void * host_weights = (config.layout == GGML_LAYOUT_AOS) ? weights.aos.data() : weights.layout.data();
    if (!copy_to_device(weight_buf, host_weights, weight_bytes, queue, error)) {
        free_buffer(weight_buf, queue);
        free_buffer(act_buf, queue);
        free_buffer(out_buf, queue);
        out.error = error;
        return false;
    }
    if (!copy_to_device(act_buf, activations.q8_1.data(), activation_bytes, queue, error)) {
        free_buffer(weight_buf, queue);
        free_buffer(act_buf, queue);
        free_buffer(out_buf, queue);
        out.error = error;
        return false;
    }

    queue.wait_and_throw();

    ggml_sycl::mmvq_bench_args args{};
    args.weight_type = config.quant_type;
    args.layout = config.layout;
    args.weights = weight_buf.ptr;
    args.layout_base = (config.layout == GGML_LAYOUT_AOS) ? nullptr : weight_buf.ptr;
    args.activations = act_buf.ptr;
    args.output = static_cast<float *>(out_buf.ptr);
    args.ncols = config.dim_k;
    args.nrows = config.dim_m;
    args.batch = config.batch_size;
    args.row_low = 0;
    args.row_high = config.dim_m;
    args.src1_padded_col_size = k_padded;
    args.dst_row_stride = config.dim_m;
    args.device_id = device_id;
    args.stream = &queue;

    auto zero_output_if_needed = [&]() {
        if (config.kernel_name == "mmvq_xmx_multi_wg") {
            queue.memset(out_buf.ptr, 0, output_bytes).wait();
        }
    };

    auto launch_kernel = [&](std::vector<sycl::event> * evs, std::string & launch_error) -> bool {
        if (config.kernel_name == "mmvq_aos_baseline") {
            return run_mmvq_aos_baseline(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_soa_baseline") {
            return run_mmvq_soa_baseline(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_coalesced") {
            return run_mmvq_coalesced(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_slm_cached") {
            return run_mmvq_slm_cached(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_prefetch") {
            return run_mmvq_prefetch(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_wide_load") {
            return run_mmvq_wide_load(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_esimd_block_load") {
            return run_mmvq_esimd_block_load(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_esimd_slm") {
            return run_mmvq_esimd_slm(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_xmx_tile_8x8") {
            return run_mmvq_xmx_tile_8x8(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_xmx_tile_16x16") {
            return run_mmvq_xmx_tile_16x16(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_xmx_aos_direct") {
            return run_mmvq_xmx_aos_direct(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_xmx_soa_direct") {
            return run_mmvq_xmx_soa_direct(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_xmx_double_buffer") {
            return run_mmvq_xmx_double_buffer(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_esimd_dpas_1x16x32") {
            return run_mmvq_esimd_dpas_1x16x32(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_esimd_dpas_8x16x32") {
            return run_mmvq_esimd_dpas_8x16x32(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_esimd_dpas_chained") {
            return run_mmvq_esimd_dpas_chained(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_xmx_tile_64x64") {
            return run_mmvq_xmx_tile_64x64(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_xmx_register_accum") {
            return run_mmvq_xmx_register_accum(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_xmx_multi_wg") {
            return run_mmvq_xmx_multi_wg(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_xmx_persistent") {
            return run_mmvq_xmx_persistent(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_esimd_large_tile") {
            return run_mmvq_esimd_large_tile(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_esimd_persistent") {
            return run_mmvq_esimd_persistent(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_esimd_lsc_prefetch") {
            return run_mmvq_esimd_lsc_prefetch(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_hybrid_adaptive") {
            return run_mmvq_hybrid_adaptive(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_xmx_fused") {
            return run_mmvq_xmx_fused(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_coalesced_xmx_aligned") {
            return run_mmvq_coalesced_xmx_aligned(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_esimd_hybrid") {
            return run_mmvq_esimd_hybrid(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_esimd_cooperative") {
            return run_mmvq_esimd_cooperative(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_q4_0_specialized") {
            return run_mmvq_q4_0_specialized(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_q6_k_specialized") {
            return run_mmvq_q6_k_specialized(args, evs, launch_error);
        }
        if (config.kernel_name == "mmvq_mxfp4_native") {
            return run_mmvq_mxfp4_native(args, evs, launch_error);
        }
        return ggml_sycl::ggml_sycl_mmvq_bench_launch(args, evs);
    };

    for (int i = 0; i < config.warmup_iterations; ++i) {
        zero_output_if_needed();
        std::string launch_error;
        if (!launch_kernel(nullptr, launch_error)) {
            out.error = launch_error.empty() ? "Warmup kernel launch failed." : launch_error;
            free_buffer(weight_buf, queue);
            free_buffer(act_buf, queue);
            free_buffer(out_buf, queue);
            return false;
        }
    }

    queue.wait_and_throw();

    std::vector<sycl::event> events;
    events.reserve(static_cast<size_t>(config.measure_iterations) * static_cast<size_t>(config.batch_size));
    std::vector<size_t> iter_offsets;
    iter_offsets.reserve(static_cast<size_t>(config.measure_iterations));

    for (int i = 0; i < config.measure_iterations; ++i) {
        iter_offsets.push_back(events.size());
        zero_output_if_needed();
        std::string launch_error;
        if (!launch_kernel(&events, launch_error)) {
            out.error = launch_error.empty() ? "Measured kernel launch failed." : launch_error;
            free_buffer(weight_buf, queue);
            free_buffer(act_buf, queue);
            free_buffer(out_buf, queue);
            return false;
        }
    }

    if (!events.empty()) {
        sycl::event::wait(events);
    }

    std::vector<double> iter_ns;
    iter_ns.reserve(iter_offsets.size());

    for (size_t i = 0; i < iter_offsets.size(); ++i) {
        const size_t begin = iter_offsets[i];
        const size_t end = (i + 1 < iter_offsets.size()) ? iter_offsets[i + 1] : events.size();
        if (end <= begin) {
            out.error = "No events captured for iteration.";
            free_buffer(weight_buf, queue);
            free_buffer(act_buf, queue);
            free_buffer(out_buf, queue);
            return false;
        }
        uint64_t min_start = 0;
        uint64_t max_end = 0;
        for (size_t j = begin; j < end; ++j) {
            uint64_t start = 0;
            uint64_t finish = 0;
            if (!get_event_timing(events[j], start, finish, error)) {
                out.error = error;
                free_buffer(weight_buf, queue);
                free_buffer(act_buf, queue);
                free_buffer(out_buf, queue);
                return false;
            }
            if (min_start == 0 || start < min_start) {
                min_start = start;
            }
            if (finish > max_end) {
                max_end = finish;
            }
        }
        iter_ns.push_back(static_cast<double>(max_end - min_start));
    }

    double sum = 0.0;
    for (double v : iter_ns) {
        sum += v;
    }
    const double mean_ns = (iter_ns.empty()) ? 0.0 : sum / static_cast<double>(iter_ns.size());

    double variance = 0.0;
    for (double v : iter_ns) {
        const double diff = v - mean_ns;
        variance += diff * diff;
    }
    const double std_ns = (iter_ns.size() > 1) ? std::sqrt(variance / static_cast<double>(iter_ns.size() - 1)) : 0.0;

    std::vector<double> sorted_ns = iter_ns;
    std::sort(sorted_ns.begin(), sorted_ns.end());
    auto percentile_ns = [&](double pct) -> double {
        if (sorted_ns.empty()) {
            return 0.0;
        }
        if (sorted_ns.size() == 1) {
            return sorted_ns[0];
        }
        const double pos = (pct / 100.0) * static_cast<double>(sorted_ns.size() - 1);
        const size_t idx = static_cast<size_t>(pos);
        const size_t idx2 = std::min(idx + 1, sorted_ns.size() - 1);
        const double frac = pos - static_cast<double>(idx);
        return sorted_ns[idx] + (sorted_ns[idx2] - sorted_ns[idx]) * frac;
    };

    const double mean_s = mean_ns * 1e-9;
    out.result.latency_us = mean_ns * 1e-3;
    out.result.latency_std = std_ns * 1e-3;
    out.result.latency_p50_us = percentile_ns(50.0) * 1e-3;
    out.result.latency_p90_us = percentile_ns(90.0) * 1e-3;
    out.result.latency_p99_us = percentile_ns(99.0) * 1e-3;
    out.result.variance_pct = (mean_ns > 0.0) ? (std_ns / mean_ns) * 100.0 : 0.0;
    out.result.throughput_tps = (mean_s > 0.0) ? (static_cast<double>(config.batch_size) / mean_s) : 0.0;

    const RooflineEstimate roofline = estimate_mmvq_roofline(config.dim_m,
                                                             config.dim_k,
                                                             config.batch_size,
                                                             weight_bytes,
                                                             activation_bytes,
                                                             output_bytes);
    out.result.bandwidth_gbps = (mean_s > 0.0) ? (roofline.bytes / mean_s) / 1e9 : 0.0;
    out.result.xmx_util_pct = 0.0;

    if (config.validate) {
        std::vector<float> host_output(config.dim_m * config.batch_size);
        if (!copy_to_host(host_output.data(), out_buf, output_bytes, queue, error)) {
            out.error = error;
            free_buffer(weight_buf, queue);
            free_buffer(act_buf, queue);
            free_buffer(out_buf, queue);
            return false;
        }
        queue.wait_and_throw();

        const double abs_tol = config.abs_tol;
        const double rel_tol = config.rel_tol;
        std::vector<float> weights_deq(config.dim_m * config.dim_k);
        std::vector<float> act_deq(config.batch_size * k_padded);

        for (int64_t row = 0; row < config.dim_m; ++row) {
            const uint8_t * row_ptr = weights.aos.data() + row * ggml_row_size(config.quant_type, config.dim_k);
            if (!dequantize_weights_row(config.quant_type, row_ptr, weights_deq.data() + row * config.dim_k,
                                        config.dim_k, error)) {
                out.error = error;
                free_buffer(weight_buf, queue);
                free_buffer(act_buf, queue);
                free_buffer(out_buf, queue);
                return false;
            }
        }

        const size_t row_bytes = (size_t) (k_padded / QK8_1) * sizeof(block_q8_1);
        for (int64_t b = 0; b < config.batch_size; ++b) {
            const uint8_t * src = activations.q8_1.data() + (size_t) b * row_bytes;
            if (activations_soa) {
                dequantize_q8_1_row_soa(src, act_deq.data() + b * k_padded, config.dim_k, k_padded);
            } else {
                const block_q8_1 * src_blocks = reinterpret_cast<const block_q8_1 *>(src);
                dequantize_q8_1_row_aos(src_blocks, act_deq.data() + b * k_padded, config.dim_k);
                if (k_padded > config.dim_k) {
                    std::fill(act_deq.data() + b * k_padded + config.dim_k,
                              act_deq.data() + (b + 1) * k_padded, 0.0f);
                }
            }
        }

        double max_err = 0.0;
        double sum_err = 0.0;
        double max_ref_abs = 0.0;
        const size_t total_vals = static_cast<size_t>(config.dim_m * config.batch_size);
        for (int64_t b = 0; b < config.batch_size; ++b) {
            const float * act = act_deq.data() + b * k_padded;
            for (int64_t row = 0; row < config.dim_m; ++row) {
                const float * w = weights_deq.data() + row * config.dim_k;
                float sum_row = 0.0f;
                for (int64_t k = 0; k < config.dim_k; ++k) {
                    sum_row += w[k] * act[k];
                }
                const size_t idx = (size_t) b * (size_t) config.dim_m + (size_t) row;
                const double ref_val = static_cast<double>(sum_row);
                const double diff = std::fabs(ref_val - static_cast<double>(host_output[idx]));
                sum_err += diff;
                max_err = std::max(max_err, diff);
                max_ref_abs = std::max(max_ref_abs, std::fabs(ref_val));
            }
        }

        out.max_abs_error = max_err;
        out.mean_abs_error = (total_vals > 0) ? sum_err / static_cast<double>(total_vals) : 0.0;

        const double tol = abs_tol + rel_tol * max_ref_abs;
        if (max_err > tol) {
            char msg[256];
            std::snprintf(msg, sizeof(msg),
                          "Validation failed: max_abs_error=%.6f (tol=%.6f, abs=%.6f, rel=%.6f).",
                          max_err, tol, abs_tol, rel_tol);
            out.error = msg;
            free_buffer(weight_buf, queue);
            free_buffer(act_buf, queue);
            free_buffer(out_buf, queue);
            return false;
        }
    }

    free_buffer(weight_buf, queue);
    free_buffer(act_buf, queue);
    free_buffer(out_buf, queue);

    out.ok = true;
    return true;
}

inline bool BenchmarkHarness::run_mmq(const BenchmarkConfig & config,
                                      sycl::queue & queue,
                                      BenchmarkOutput & out) {
    const auto mmq_supports_type = [](ggml_type type) -> bool {
        switch (type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
                return true;
            default:
                return false;
        }
    };

    if (!mmq_supports_type(config.quant_type)) {
        out.error = "MMQ does not support requested quant type.";
        return false;
    }

    if (!kernel_supports_layout(config.quant_type, config.layout)) {
        out.error = "Kernel layout not supported for quant type.";
        return false;
    }

    if (config.dim_m <= 0 || config.dim_k <= 0 || config.batch_size <= 0) {
        out.error = "Invalid dimensions.";
        return false;
    }

    int device_id = get_current_device_id();
    if (device_id < 0) {
        out.error = "Invalid device id.";
        return false;
    }

    const int64_t k_padded = GGML_PAD(config.dim_k, MATRIX_ROW_PADDING);
    const size_t activation_bytes =
        static_cast<size_t>(config.batch_size) * static_cast<size_t>(k_padded) * sizeof(block_q8_1) / QK8_1;
    const size_t output_bytes =
        static_cast<size_t>(config.batch_size) * static_cast<size_t>(config.dim_m) * sizeof(float);

    GeneratedWeights weights;
    if (!generate_quantized_weights(config.quant_type,
                                    config.layout,
                                    config.dim_m,
                                    config.dim_k,
                                    config.validate,
                                    weights)) {
        out.error = "Failed to generate quantized weights for requested layout/type.";
        return false;
    }

    const bool activations_soa = false;
    GeneratedActivations activations = generate_activations(config.batch_size,
                                                            config.dim_k,
                                                            k_padded,
                                                            config.validate,
                                                            config.validate,
                                                            activations_soa);

    const size_t weight_bytes = (config.layout == GGML_LAYOUT_AOS) ? weights.bytes_aos : weights.bytes_layout;
    const size_t total_bytes = weight_bytes + activation_bytes + output_bytes;

    size_t free_mem = 0;
    size_t total_mem = 0;
    ggml_backend_sycl_get_device_memory(device_id, &free_mem, &total_mem);
    if (total_mem > 0 && total_bytes > total_mem) {
        char msg[256];
        std::snprintf(msg,
                      sizeof(msg),
                      "Required memory (%.2f GB) exceeds device memory (%.2f GB).",
                      total_bytes / (1024.0 * 1024.0 * 1024.0),
                      total_mem / (1024.0 * 1024.0 * 1024.0));
        out.error = msg;
        return false;
    }

    std::string error;
    DeviceBuffer weight_buf = allocate_buffer(weight_bytes, config.memory_mode, queue, device_id, error);
    if (!error.empty()) {
        out.error = error;
        return false;
    }
    DeviceBuffer act_buf = allocate_buffer(activation_bytes, config.memory_mode, queue, device_id, error);
    if (!error.empty()) {
        free_buffer(weight_buf, queue);
        out.error = error;
        return false;
    }
    DeviceBuffer out_buf = allocate_buffer(output_bytes, config.memory_mode, queue, device_id, error);
    if (!error.empty()) {
        free_buffer(weight_buf, queue);
        free_buffer(act_buf, queue);
        out.error = error;
        return false;
    }

    const void * host_weights = (config.layout == GGML_LAYOUT_AOS) ? weights.aos.data() : weights.layout.data();
    if (!copy_to_device(weight_buf, host_weights, weight_bytes, queue, error)) {
        free_buffer(weight_buf, queue);
        free_buffer(act_buf, queue);
        free_buffer(out_buf, queue);
        out.error = error;
        return false;
    }
    if (!copy_to_device(act_buf, activations.q8_1.data(), activation_bytes, queue, error)) {
        free_buffer(weight_buf, queue);
        free_buffer(act_buf, queue);
        free_buffer(out_buf, queue);
        out.error = error;
        return false;
    }

    queue.wait_and_throw();

    ggml_sycl::mmq_bench_args args{};
    args.weight_type = config.quant_type;
    args.layout = config.layout;
    args.weights = weight_buf.ptr;
    args.layout_base = (config.layout == GGML_LAYOUT_AOS) ? nullptr : weight_buf.ptr;
    args.activations = act_buf.ptr;
    args.output = static_cast<float *>(out_buf.ptr);
    args.ncols = config.dim_k;
    args.nrows = config.dim_m;
    args.batch = config.batch_size;
    args.row_low = 0;
    args.row_high = config.dim_m;
    args.src1_padded_row_size = k_padded;
    args.dst_row_stride = config.dim_m;
    args.device_id = device_id;
    args.stream = &queue;

    auto launch_kernel = [&](std::vector<sycl::event> * evs, std::string & launch_error) -> bool {
        if (!ggml_sycl::ggml_sycl_mmq_bench_launch(args, evs)) {
            launch_error = "MMQ bench launch failed.";
            return false;
        }
        return true;
    };

    for (int i = 0; i < config.warmup_iterations; ++i) {
        std::string launch_error;
        if (!launch_kernel(nullptr, launch_error)) {
            out.error = launch_error.empty() ? "Warmup kernel launch failed." : launch_error;
            free_buffer(weight_buf, queue);
            free_buffer(act_buf, queue);
            free_buffer(out_buf, queue);
            return false;
        }
    }

    queue.wait_and_throw();

    std::vector<sycl::event> events;
    events.reserve(static_cast<size_t>(config.measure_iterations));
    std::vector<size_t> iter_offsets;
    iter_offsets.reserve(static_cast<size_t>(config.measure_iterations));

    for (int i = 0; i < config.measure_iterations; ++i) {
        iter_offsets.push_back(events.size());
        std::string launch_error;
        if (!launch_kernel(&events, launch_error)) {
            out.error = launch_error.empty() ? "Measured kernel launch failed." : launch_error;
            free_buffer(weight_buf, queue);
            free_buffer(act_buf, queue);
            free_buffer(out_buf, queue);
            return false;
        }
    }

    if (!events.empty()) {
        sycl::event::wait(events);
    }

    std::vector<double> iter_ns;
    iter_ns.reserve(iter_offsets.size());

    for (size_t i = 0; i < iter_offsets.size(); ++i) {
        const size_t begin = iter_offsets[i];
        const size_t end = (i + 1 < iter_offsets.size()) ? iter_offsets[i + 1] : events.size();
        if (end <= begin) {
            out.error = "No events captured for iteration.";
            free_buffer(weight_buf, queue);
            free_buffer(act_buf, queue);
            free_buffer(out_buf, queue);
            return false;
        }
        uint64_t min_start = 0;
        uint64_t max_end = 0;
        for (size_t j = begin; j < end; ++j) {
            uint64_t start = 0;
            uint64_t finish = 0;
            if (!get_event_timing(events[j], start, finish, error)) {
                out.error = error;
                free_buffer(weight_buf, queue);
                free_buffer(act_buf, queue);
                free_buffer(out_buf, queue);
                return false;
            }
            if (min_start == 0 || start < min_start) {
                min_start = start;
            }
            if (finish > max_end) {
                max_end = finish;
            }
        }
        iter_ns.push_back(static_cast<double>(max_end - min_start));
    }

    double sum = 0.0;
    for (double v : iter_ns) {
        sum += v;
    }
    const double mean_ns = (iter_ns.empty()) ? 0.0 : sum / static_cast<double>(iter_ns.size());

    double variance = 0.0;
    for (double v : iter_ns) {
        const double diff = v - mean_ns;
        variance += diff * diff;
    }
    const double std_ns = (iter_ns.size() > 1) ? std::sqrt(variance / static_cast<double>(iter_ns.size() - 1)) : 0.0;

    std::vector<double> sorted_ns = iter_ns;
    std::sort(sorted_ns.begin(), sorted_ns.end());
    auto percentile_ns = [&](double pct) -> double {
        if (sorted_ns.empty()) {
            return 0.0;
        }
        if (sorted_ns.size() == 1) {
            return sorted_ns[0];
        }
        const double pos = (pct / 100.0) * static_cast<double>(sorted_ns.size() - 1);
        const size_t idx = static_cast<size_t>(pos);
        const size_t idx2 = std::min(idx + 1, sorted_ns.size() - 1);
        const double frac = pos - static_cast<double>(idx);
        return sorted_ns[idx] + (sorted_ns[idx2] - sorted_ns[idx]) * frac;
    };

    const double mean_s = mean_ns * 1e-9;
    out.result.latency_us = mean_ns * 1e-3;
    out.result.latency_std = std_ns * 1e-3;
    out.result.latency_p50_us = percentile_ns(50.0) * 1e-3;
    out.result.latency_p90_us = percentile_ns(90.0) * 1e-3;
    out.result.latency_p99_us = percentile_ns(99.0) * 1e-3;
    out.result.variance_pct = (mean_ns > 0.0) ? (std_ns / mean_ns) * 100.0 : 0.0;
    out.result.throughput_tps = (mean_s > 0.0) ? (static_cast<double>(config.batch_size) / mean_s) : 0.0;

    const RooflineEstimate roofline = estimate_mmvq_roofline(config.dim_m,
                                                             config.dim_k,
                                                             config.batch_size,
                                                             weight_bytes,
                                                             activation_bytes,
                                                             output_bytes);
    out.result.bandwidth_gbps = (mean_s > 0.0) ? (roofline.bytes / mean_s) / 1e9 : 0.0;
    out.result.xmx_util_pct = 0.0;

    if (config.validate) {
        std::vector<float> host_output(config.dim_m * config.batch_size);
        if (!copy_to_host(host_output.data(), out_buf, output_bytes, queue, error)) {
            out.error = error;
            free_buffer(weight_buf, queue);
            free_buffer(act_buf, queue);
            free_buffer(out_buf, queue);
            return false;
        }
        queue.wait_and_throw();

        const double abs_tol = config.abs_tol;
        const double rel_tol = config.rel_tol;
        std::vector<float> weights_deq(config.dim_m * config.dim_k);
        std::vector<float> act_deq(config.batch_size * k_padded);

        for (int64_t row = 0; row < config.dim_m; ++row) {
            const uint8_t * row_ptr = weights.aos.data() + row * ggml_row_size(config.quant_type, config.dim_k);
            if (!dequantize_weights_row(config.quant_type, row_ptr, weights_deq.data() + row * config.dim_k,
                                        config.dim_k, error)) {
                out.error = error;
                free_buffer(weight_buf, queue);
                free_buffer(act_buf, queue);
                free_buffer(out_buf, queue);
                return false;
            }
        }

        const size_t row_bytes = (size_t) (k_padded / QK8_1) * sizeof(block_q8_1);
        for (int64_t b = 0; b < config.batch_size; ++b) {
            const uint8_t * src = activations.q8_1.data() + (size_t) b * row_bytes;
            const block_q8_1 * src_blocks = reinterpret_cast<const block_q8_1 *>(src);
            dequantize_q8_1_row_aos(src_blocks, act_deq.data() + b * k_padded, config.dim_k);
            if (k_padded > config.dim_k) {
                std::fill(act_deq.data() + b * k_padded + config.dim_k,
                          act_deq.data() + (b + 1) * k_padded, 0.0f);
            }
        }

        double max_err = 0.0;
        double sum_err = 0.0;
        double max_ref_abs = 0.0;
        const size_t total_vals = static_cast<size_t>(config.dim_m * config.batch_size);
        for (int64_t b = 0; b < config.batch_size; ++b) {
            const float * act = act_deq.data() + b * k_padded;
            for (int64_t row = 0; row < config.dim_m; ++row) {
                const float * w = weights_deq.data() + row * config.dim_k;
                float sum_row = 0.0f;
                for (int64_t k = 0; k < config.dim_k; ++k) {
                    sum_row += w[k] * act[k];
                }
                const size_t idx = (size_t) b * (size_t) config.dim_m + (size_t) row;
                const double ref_val = static_cast<double>(sum_row);
                const double diff = std::fabs(ref_val - static_cast<double>(host_output[idx]));
                sum_err += diff;
                max_err = std::max(max_err, diff);
                max_ref_abs = std::max(max_ref_abs, std::fabs(ref_val));
            }
        }

        out.max_abs_error = max_err;
        out.mean_abs_error = (total_vals > 0) ? sum_err / static_cast<double>(total_vals) : 0.0;

        const double tol = abs_tol + rel_tol * max_ref_abs;
        if (max_err > tol) {
            char msg[256];
            std::snprintf(msg, sizeof(msg),
                          "Validation failed: max_abs_error=%.6f (tol=%.6f, abs=%.6f, rel=%.6f).",
                          max_err, tol, abs_tol, rel_tol);
            out.error = msg;
            free_buffer(weight_buf, queue);
            free_buffer(act_buf, queue);
            free_buffer(out_buf, queue);
            return false;
        }
    }

    free_buffer(weight_buf, queue);
    free_buffer(act_buf, queue);
    free_buffer(out_buf, queue);

    out.ok = true;
    return true;
}

}  // namespace sycl_bench
