#include "host-memory.h"

#include "ggml-quants.h"
#include "hrx-interop-utils.h"
#include "hrx_runtime.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

namespace ggml::hrx {
namespace {

static constexpr size_t kMaxInlineUploadBytes = 63 * 1024;
static constexpr size_t kLargeHostUploadBytes = 1024 * 1024;

static bool mapped_download_staging_enabled() {
    static const bool enabled = [] {
        const char * mode = std::getenv("GGML_HRX_WEIGHT_DOWNLOAD_STAGING");
        return mode == nullptr || mode[0] == '\0' || std::strcmp(mode, "mapped") == 0;
    }();
    return enabled;
}

struct SymmetricI4Block {
    std::array<ggml_fp16_t, 8>             scales   = {};
    std::array<std::array<uint8_t, 16>, 8> payloads = {};
};

static bool checked_multiply(size_t lhs, size_t rhs, size_t & result) {
    if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

template <typename Work> static bool run_worker_threads(size_t thread_count, Work && work) {
    std::atomic<bool>        worker_failed = false;
    std::vector<std::thread> workers;
    auto                     join_workers = [&]() {
        for (std::thread & worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    };

    try {
        workers.reserve(thread_count);
        for (size_t thread = 0; thread < thread_count; ++thread) {
            workers.emplace_back([&, thread]() {
                try {
                    work(thread);
                } catch (...) {
                    worker_failed.store(true, std::memory_order_relaxed);
                }
            });
        }
    } catch (...) {
        join_workers();
        return false;
    }
    join_workers();
    return !worker_failed.load(std::memory_order_relaxed);
}

static void dequantize_block(ggml_type type, const uint8_t * source, float * destination) {
    switch (type) {
        case GGML_TYPE_Q4_K:
            dequantize_row_q4_K(reinterpret_cast<const block_q4_K *>(source), destination, QK_K);
            break;
        case GGML_TYPE_Q5_K:
            dequantize_row_q5_K(reinterpret_cast<const block_q5_K *>(source), destination, QK_K);
            break;
        case GGML_TYPE_IQ4_XS:
            dequantize_row_iq4_xs(reinterpret_cast<const block_iq4_xs *>(source), destination, QK_K);
            break;
        case GGML_TYPE_Q6_K:
            dequantize_row_q6_K(reinterpret_cast<const block_q6_K *>(source), destination, QK_K);
            break;
        default:
            break;
    }
}

static bool quantize_symmetric_value(float value, float scale, int quant_min, int quant_max, int & quantized) {
    if (!std::isfinite(value) || !std::isfinite(scale) || scale <= 0.0f) {
        return false;
    }
    float quotient = value / scale;
    if (std::isnan(quotient)) {
        return false;
    }
    quotient  = std::clamp(quotient, static_cast<float>(quant_min), static_cast<float>(quant_max));
    quantized = static_cast<int>(std::nearbyint(quotient));
    return true;
}

static bool fit_symmetric_scale(const float * values,
                                size_t        value_count,
                                int           quant_min,
                                int           quant_max,
                                float &       encoded_scale) {
    float positive_max = 0.0f;
    float negative_max = 0.0f;
    bool  nonzero      = false;
    for (size_t element = 0; element < value_count; ++element) {
        if (!std::isfinite(values[element])) {
            return false;
        }
        nonzero      = nonzero || values[element] != 0.0f;
        positive_max = std::max(positive_max, values[element]);
        negative_max = std::max(negative_max, -values[element]);
    }
    if (!nonzero) {
        encoded_scale = 0.0f;
        return true;
    }

    float scale = std::max(positive_max / quant_max, negative_max / -quant_min);
    if (!std::isfinite(scale) || scale <= 0.0f) {
        return false;
    }
    for (int iteration = 0; iteration < 2; ++iteration) {
        double numerator   = 0.0;
        double denominator = 0.0;
        for (size_t element = 0; element < value_count; ++element) {
            int quantized = 0;
            if (!quantize_symmetric_value(values[element], scale, quant_min, quant_max, quantized)) {
                return false;
            }
            numerator += static_cast<double>(values[element]) * quantized;
            denominator += static_cast<double>(quantized) * quantized;
        }
        if (denominator > 0.0) {
            scale = static_cast<float>(numerator / denominator);
            if (!std::isfinite(scale) || scale <= 0.0f) {
                return false;
            }
        }
    }

    encoded_scale = ggml_fp16_to_fp32(ggml_fp32_to_fp16(scale));
    return std::isfinite(encoded_scale) && encoded_scale > 0.0f;
}

static bool fit_shared_symmetric_scale(const std::array<std::array<float, QK_K>, 4> & values,
                                       size_t                                         row_count,
                                       size_t                                         group,
                                       int                                            quant_min,
                                       int                                            quant_max,
                                       bool                                           multistart,
                                       float &                                        encoded_scale) {
    float positive_max = 0.0f;
    float negative_max = 0.0f;
    bool  nonzero      = false;
    for (size_t row = 0; row < row_count; ++row) {
        const float * group_values = values[row].data() + group * 32;
        for (size_t element = 0; element < 32; ++element) {
            if (!std::isfinite(group_values[element])) {
                return false;
            }
            nonzero      = nonzero || group_values[element] != 0.0f;
            positive_max = std::max(positive_max, group_values[element]);
            negative_max = std::max(negative_max, -group_values[element]);
        }
    }
    if (!nonzero) {
        encoded_scale = 0.0f;
        return true;
    }

    const float initial_scale = std::max(positive_max / quant_max, negative_max / -quant_min);
    if (!std::isfinite(initial_scale) || initial_scale <= 0.0f) {
        return false;
    }

    auto refit = [&](float & scale) {
        const int iteration_count = multistart ? 4 : 2;
        for (int iteration = 0; iteration < iteration_count; ++iteration) {
            double numerator   = 0.0;
            double denominator = 0.0;
            for (size_t row = 0; row < row_count; ++row) {
                const float * group_values = values[row].data() + group * 32;
                for (size_t element = 0; element < 32; ++element) {
                    int quantized = 0;
                    if (!quantize_symmetric_value(group_values[element], scale, quant_min, quant_max, quantized)) {
                        return false;
                    }
                    numerator += static_cast<double>(group_values[element]) * quantized;
                    denominator += static_cast<double>(quantized) * quantized;
                }
            }
            if (denominator > 0.0) {
                scale = static_cast<float>(numerator / denominator);
                if (!std::isfinite(scale) || scale <= 0.0f) {
                    return false;
                }
            }
        }
        return true;
    };

    if (!multistart) {
        float scale = initial_scale;
        if (!refit(scale)) {
            return false;
        }
        encoded_scale = ggml_fp16_to_fp32(ggml_fp32_to_fp16(scale));
        return std::isfinite(encoded_scale) && encoded_scale > 0.0f;
    }

    constexpr std::array<float, 7> kInitialScaleRatios = { 0.50f, 0.60f, 0.70f, 0.80f, 0.90f, 1.00f, 1.10f };
    double                         best_error          = std::numeric_limits<double>::infinity();
    float                          best_scale          = 0.0f;
    for (float ratio : kInitialScaleRatios) {
        float scale = initial_scale * ratio;
        if (!refit(scale)) {
            return false;
        }
        scale        = ggml_fp16_to_fp32(ggml_fp32_to_fp16(scale));
        double error = 0.0;
        for (size_t row = 0; row < row_count; ++row) {
            const float * group_values = values[row].data() + group * 32;
            for (size_t element = 0; element < 32; ++element) {
                int quantized = 0;
                if (!quantize_symmetric_value(group_values[element], scale, quant_min, quant_max, quantized)) {
                    return false;
                }
                const double delta =
                    static_cast<double>(group_values[element]) - static_cast<double>(quantized) * scale;
                error += delta * delta;
            }
        }
        if (error < best_error) {
            best_error = error;
            best_scale = scale;
        }
    }

    encoded_scale = best_scale;
    return std::isfinite(encoded_scale) && encoded_scale > 0.0f;
}

static bool quantize_symmetric_i4_k64(const float * values, SymmetricI4Block & output) {
    for (size_t pair = 0; pair < output.scales.size() / 2; ++pair) {
        const float * pair_values = values + pair * 64;
        float         scale       = 0.0f;
        if (!fit_symmetric_scale(pair_values, 64, -8, 7, scale)) {
            return false;
        }
        if (scale == 0.0f) {
            continue;
        }

        const ggml_fp16_t encoded_scale = ggml_fp32_to_fp16(scale);
        output.scales[pair * 2]         = encoded_scale;
        output.scales[pair * 2 + 1]     = encoded_scale;
        for (size_t half = 0; half < 2; ++half) {
            const size_t  logical_group = pair * 2 + half;
            const float * group_values  = pair_values + half * 32;
            for (size_t element_pair = 0; element_pair < 16; ++element_pair) {
                int low  = 0;
                int high = 0;
                if (!quantize_symmetric_value(group_values[element_pair * 2], scale, -8, 7, low) ||
                    !quantize_symmetric_value(group_values[element_pair * 2 + 1], scale, -8, 7, high)) {
                    return false;
                }
                output.payloads[logical_group][element_pair] =
                    static_cast<uint8_t>((low & 0x0F) | ((high & 0x0F) << 4));
            }
        }
    }
    return true;
}

static bool quantize_symmetric_i4_k32(const float * values, SymmetricI4Block & output) {
    for (size_t group = 0; group < output.scales.size(); ++group) {
        const float * group_values = values + group * 32;
        float         scale        = 0.0f;
        if (!fit_symmetric_scale(group_values, 32, -8, 7, scale)) {
            return false;
        }
        if (scale == 0.0f) {
            continue;
        }
        output.scales[group] = ggml_fp32_to_fp16(scale);
        for (size_t element_pair = 0; element_pair < 16; ++element_pair) {
            int low  = 0;
            int high = 0;
            if (!quantize_symmetric_value(group_values[element_pair * 2], scale, -8, 7, low) ||
                !quantize_symmetric_value(group_values[element_pair * 2 + 1], scale, -8, 7, high)) {
                return false;
            }
            output.payloads[group][element_pair] = static_cast<uint8_t>((low & 0x0F) | ((high & 0x0F) << 4));
        }
    }
    return true;
}

using SymmetricI4Quantize = bool (*)(const float *, SymmetricI4Block &);

static Status materialize_symmetric_i4_row64(const HostWeightSource & source,
                                             SymmetricI4Quantize      quantize,
                                             std::vector<uint8_t> &   output) {
    Status status;
    if (source.source_type != GGML_TYPE_Q4_K && source.source_type != GGML_TYPE_Q5_K &&
        source.source_type != GGML_TYPE_Q6_K && source.source_type != GGML_TYPE_IQ4_XS) {
        status.log("layout %s does not support GGML type %d", source.layout.c_str(),
                   static_cast<int>(source.source_type));
        return status;
    }
    if (source.input_size <= 0 || source.input_size % QK_K != 0 || source.output_size <= 0 ||
        source.output_size % 64 != 0) {
        status.log("layout %s requires K divisible by %d and rows divisible by 64, got K=%lld rows=%lld",
                   source.layout.c_str(), QK_K, static_cast<long long>(source.input_size),
                   static_cast<long long>(source.output_size));
        return status;
    }

    const size_t row_bytes             = ggml_row_size(source.source_type, source.input_size);
    size_t       expected_source_bytes = 0;
    if (!checked_multiply(row_bytes, static_cast<size_t>(source.output_size), expected_source_bytes) ||
        source.length != expected_source_bytes) {
        status.log("layout %s source length %zu does not match expected %zu", source.layout.c_str(), source.length,
                   expected_source_bytes);
        return status;
    }

    constexpr size_t materialized_row_bytes = 144;
    constexpr size_t row_group              = 64;
    const size_t     logical_row_count      = static_cast<size_t>(source.output_size);
    const size_t     padded_row_count       = (logical_row_count + row_group - 1) / row_group * row_group;
    const size_t     block_count            = static_cast<size_t>(source.input_size / QK_K);
    size_t           expected_output_bytes  = 0;
    if (!checked_multiply(padded_row_count, block_count, expected_output_bytes) ||
        !checked_multiply(expected_output_bytes, materialized_row_bytes, expected_output_bytes) ||
        source.materialized_length != expected_output_bytes) {
        status.log("layout %s materialized length %zu does not match expected %zu", source.layout.c_str(),
                   source.materialized_length, expected_output_bytes);
        return status;
    }

    output.assign(expected_output_bytes, 0);
    const auto *     source_bytes       = static_cast<const uint8_t *>(source.host_data) + source.offset;
    constexpr size_t field_count        = 9;
    constexpr size_t field_bytes        = 16;
    constexpr size_t block_out_bytes    = field_count * field_bytes;
    const size_t     source_block_bytes = ggml_type_size(source.source_type);
    const size_t thread_count = std::min<size_t>(logical_row_count, std::max(1u, std::thread::hardware_concurrency()));
    std::atomic<bool> conversion_failed = false;

    auto convert_rows = [&](size_t row_begin, size_t row_end) {
        std::array<float, QK_K> decoded = {};
        for (size_t row = row_begin; row < row_end; ++row) {
            if (conversion_failed.load(std::memory_order_relaxed)) {
                return;
            }
            const size_t    group      = row / row_group;
            const size_t    lane       = row % row_group;
            const uint8_t * source_row = source_bytes + row * row_bytes;
            for (size_t block = 0; block < block_count; ++block) {
                dequantize_block(source.source_type, source_row + block * source_block_bytes, decoded.data());
                SymmetricI4Block converted;
                if (!quantize(decoded.data(), converted)) {
                    conversion_failed.store(true, std::memory_order_relaxed);
                    return;
                }
                const size_t group_base = (group * block_count + block) * row_group * block_out_bytes;
                uint8_t *    header     = output.data() + group_base + lane * field_bytes;
                for (size_t group = 0; group < converted.scales.size(); ++group) {
                    std::memcpy(header + group * sizeof(ggml_fp16_t), &converted.scales[group], sizeof(ggml_fp16_t));
                }
                for (size_t logical_group = 0; logical_group < converted.payloads.size(); ++logical_group) {
                    uint8_t * payload =
                        output.data() + group_base + (1 + logical_group) * row_group * field_bytes + lane * field_bytes;
                    std::memcpy(payload, converted.payloads[logical_group].data(), field_bytes);
                }
            }
        }
    };

    if (!run_worker_threads(thread_count, [&](size_t thread) {
            const size_t row_begin = logical_row_count * thread / thread_count;
            const size_t row_end   = logical_row_count * (thread + 1) / thread_count;
            convert_rows(row_begin, row_end);
        })) {
        output.clear();
        status.log("layout %s host materialization worker failed", source.layout.c_str());
        return status;
    }
    if (conversion_failed.load(std::memory_order_relaxed)) {
        output.clear();
        status.log("layout %s cannot represent non-finite or out-of-range symmetric weights", source.layout.c_str());
    }
    return status;
}

static Status materialize_symmetric_i4_k32_row64(const HostWeightSource & source, std::vector<uint8_t> & output) {
    return materialize_symmetric_i4_row64(source, quantize_symmetric_i4_k32, output);
}

static Status materialize_symmetric_i4_k64_row64(const HostWeightSource & source, std::vector<uint8_t> & output) {
    return materialize_symmetric_i4_row64(source, quantize_symmetric_i4_k64, output);
}

static Status materialize_symmetric_k32_eightgroups_shared4(const HostWeightSource & source,
                                                            int                      quant_bits,
                                                            std::vector<uint8_t> &   output) {
    Status status;
    if (source.source_type != GGML_TYPE_Q4_K && source.source_type != GGML_TYPE_Q5_K &&
        source.source_type != GGML_TYPE_Q6_K && source.source_type != GGML_TYPE_IQ4_XS) {
        status.log("layout %s does not support GGML type %d", source.layout.c_str(),
                   static_cast<int>(source.source_type));
        return status;
    }
    if (source.input_size <= 0 || source.input_size % QK_K != 0 || source.output_size <= 0) {
        status.log("layout %s requires K divisible by %d and positive rows, got K=%lld rows=%lld",
                   source.layout.c_str(), QK_K, static_cast<long long>(source.input_size),
                   static_cast<long long>(source.output_size));
        return status;
    }

    constexpr size_t shared_rows  = 4;
    const size_t     logical_rows = static_cast<size_t>(source.output_size);
    if (logical_rows > std::numeric_limits<size_t>::max() - 255) {
        status.log("layout %s row count overflows row-group sizing", source.layout.c_str());
        return status;
    }
    const size_t materialized_row_bytes = 4 + 8 * 32 * static_cast<size_t>(quant_bits) / 8;
    const size_t unpadded_bytes =
        logical_rows * static_cast<size_t>(source.input_size / QK_K) * materialized_row_bytes;
    const size_t dynamic_row_group = ((logical_rows + 255) / 256) * 32;
    const size_t row_group =
        quant_bits == 4 && unpadded_bytes < size_t{ 16 } * 1024 * 1024 ? 32 :
        quant_bits == 4 && unpadded_bytes <= size_t{ 32 } * 1024 * 1024 && source.output_size > source.input_size ? 96 :
                                                                                                                   dynamic_row_group;
    if (logical_rows > std::numeric_limits<size_t>::max() - (row_group - 1)) {
        status.log("layout %s row count overflows row-group padding", source.layout.c_str());
        return status;
    }
    const size_t physical_rows = (logical_rows + row_group - 1) / row_group * row_group;

    const size_t row_bytes             = ggml_row_size(source.source_type, source.input_size);
    size_t       expected_source_bytes = 0;
    if (!checked_multiply(row_bytes, logical_rows, expected_source_bytes) || source.length != expected_source_bytes) {
        status.log("layout %s source length %zu does not match expected %zu", source.layout.c_str(), source.length,
                   expected_source_bytes);
        return status;
    }

    if (quant_bits != 2 && quant_bits != 4) {
        status.log("layout %s requires 2-bit or 4-bit symmetric materialization", source.layout.c_str());
        return status;
    }

    const int    quant_min              = -(1 << (quant_bits - 1));
    const int    quant_max              = (1 << (quant_bits - 1)) - 1;
    const size_t field_bytes            = 32 * static_cast<size_t>(quant_bits) / 8;
    const size_t encoded_row_bytes = 4 + 8 * field_bytes;
    const size_t block_count            = static_cast<size_t>(source.input_size / QK_K);
    size_t       expected_output_bytes  = 0;
    if (!checked_multiply(physical_rows, block_count, expected_output_bytes) ||
        !checked_multiply(expected_output_bytes, encoded_row_bytes, expected_output_bytes) ||
        source.materialized_length != expected_output_bytes) {
        status.log("layout %s materialized length %zu does not match expected %zu", source.layout.c_str(),
                   source.materialized_length, expected_output_bytes);
        return status;
    }

    output.assign(expected_output_bytes, uint8_t{ 0 });
    const auto * source_bytes        = static_cast<const uint8_t *>(source.host_data) + source.offset;
    const size_t scale_plane_bytes   = row_group / shared_rows * 16;
    const size_t payload_plane_bytes = row_group * field_bytes;
    const size_t payload_block_bytes = 8 * payload_plane_bytes;
    const size_t block_out_bytes     = scale_plane_bytes + payload_block_bytes;
    const size_t row_group_bytes     = block_count * block_out_bytes;
    const size_t source_block_bytes  = ggml_type_size(source.source_type);
    const size_t logical_cohorts     = (logical_rows + shared_rows - 1) / shared_rows;
    const size_t thread_count = std::min<size_t>(logical_cohorts, std::max(1u, std::thread::hardware_concurrency()));
    std::atomic<bool> conversion_failed = false;

    auto convert_cohorts = [&](size_t cohort_begin, size_t cohort_end) {
        std::array<std::array<float, QK_K>, shared_rows> decoded = {};
        for (size_t cohort = cohort_begin; cohort < cohort_end; ++cohort) {
            if (conversion_failed.load(std::memory_order_relaxed)) {
                return;
            }
            const size_t first_row    = cohort * shared_rows;
            const size_t row_count    = std::min(shared_rows, logical_rows - first_row);
            const size_t row_group_id = first_row / row_group;
            const size_t cohort_lane  = first_row % row_group / shared_rows;
            for (size_t block = 0; block < block_count; ++block) {
                for (size_t row = 0; row < row_count; ++row) {
                    const uint8_t * source_row = source_bytes + (first_row + row) * row_bytes;
                    dequantize_block(source.source_type, source_row + block * source_block_bytes, decoded[row].data());
                }

                const size_t group_base         = row_group_id * row_group_bytes;
                const size_t payload_block_base = group_base + block * payload_block_bytes;
                const size_t scale_region       = group_base + block_count * payload_block_bytes;
                uint8_t *    header = output.data() + scale_region + block * scale_plane_bytes + cohort_lane * 16;
                std::array<float, 8> scales = {};
                for (size_t group = 0; group < scales.size(); ++group) {
                    float      scale      = 0.0f;
                    const bool multistart = source.layout == kSymmetricI4K32EightGroupsShared4MultistartLayout;
                    if (!fit_shared_symmetric_scale(decoded, row_count, group, quant_min, quant_max, multistart,
                                                    scale)) {
                        conversion_failed.store(true, std::memory_order_relaxed);
                        return;
                    }
                    const ggml_fp16_t encoded_scale = ggml_fp32_to_fp16(scale);
                    std::memcpy(header + group * sizeof(encoded_scale), &encoded_scale, sizeof(encoded_scale));
                    scales[group] = scale;
                }

                for (size_t row = 0; row < row_count; ++row) {
                    const size_t row_lane = (first_row + row) % row_group;
                    for (size_t group = 0; group < scales.size(); ++group) {
                        uint8_t * payload =
                            output.data() + payload_block_base + group * payload_plane_bytes + row_lane * field_bytes;
                        const float * group_values = decoded[row].data() + group * 32;
                        const float   scale        = scales[group];
                        if (scale == 0.0f) {
                            continue;
                        }
                        if (quant_bits == 4) {
                            for (size_t element_pair = 0; element_pair < 16; ++element_pair) {
                                int low  = 0;
                                int high = 0;
                                if (!quantize_symmetric_value(group_values[element_pair * 2], scale, quant_min,
                                                              quant_max, low) ||
                                    !quantize_symmetric_value(group_values[element_pair * 2 + 1], scale, quant_min,
                                                              quant_max, high)) {
                                    conversion_failed.store(true, std::memory_order_relaxed);
                                    return;
                                }
                                payload[element_pair] = static_cast<uint8_t>((low & 0x0F) | ((high & 0x0F) << 4));
                            }
                        } else {
                            for (size_t element_quartet = 0; element_quartet < 8; ++element_quartet) {
                                uint8_t packed = 0;
                                for (size_t element = 0; element < 4; ++element) {
                                    const size_t index     = element_quartet * 4 + element;
                                    int          quantized = 0;
                                    if (!quantize_symmetric_value(group_values[index], scale, quant_min, quant_max,
                                                                  quantized)) {
                                        conversion_failed.store(true, std::memory_order_relaxed);
                                        return;
                                    }
                                    packed |= static_cast<uint8_t>((quantized & 0x03) << (element * 2));
                                }
                                payload[element_quartet] = packed;
                            }
                        }
                    }
                }
            }
        }
    };

    if (!run_worker_threads(thread_count, [&](size_t thread) {
            const size_t cohort_begin = logical_cohorts * thread / thread_count;
            const size_t cohort_end   = logical_cohorts * (thread + 1) / thread_count;
            convert_cohorts(cohort_begin, cohort_end);
        })) {
        output.clear();
        status.log("layout %s host materialization worker failed", source.layout.c_str());
        return status;
    }
    if (conversion_failed.load(std::memory_order_relaxed)) {
        output.clear();
        status.log("layout %s cannot represent non-finite or out-of-range symmetric weights", source.layout.c_str());
    }
    return status;
}

static Status materialize_symmetric_i4_k32_eightgroups_shared4(const HostWeightSource & source,
                                                               std::vector<uint8_t> &   output) {
    return materialize_symmetric_k32_eightgroups_shared4(source, 4, output);
}

static Status materialize_symmetric_i5_k32(const HostWeightSource & source, std::vector<uint8_t> & output) {
    Status status;
    if (source.source_type != GGML_TYPE_Q5_K) {
        status.log("layout %s does not support GGML type %d", source.layout.c_str(),
                   static_cast<int>(source.source_type));
        return status;
    }
    if (source.input_size <= 0 || source.input_size % QK_K != 0 || source.output_size <= 0 ||
        source.output_size % 64 != 0) {
        status.log("layout %s requires K divisible by %d and rows divisible by 64, got K=%lld rows=%lld",
                   source.layout.c_str(), QK_K, static_cast<long long>(source.input_size),
                   static_cast<long long>(source.output_size));
        return status;
    }

    const size_t row_bytes      = ggml_row_size(GGML_TYPE_Q5_K, source.input_size);
    const size_t row_count      = static_cast<size_t>(source.output_size);
    const size_t block_count    = static_cast<size_t>(source.input_size / QK_K);
    size_t       expected_bytes = 0;
    if (!checked_multiply(row_bytes, row_count, expected_bytes) || source.length != expected_bytes ||
        source.materialized_length != expected_bytes) {
        status.log("layout %s has inconsistent source/materialized lengths for K=%lld rows=%lld", source.layout.c_str(),
                   static_cast<long long>(source.input_size), static_cast<long long>(source.output_size));
        return status;
    }

    output.assign(expected_bytes, uint8_t{ 0 });
    const auto * source_bytes = static_cast<const uint8_t *>(source.host_data) + source.offset;
    const size_t thread_count =
        std::min(row_count, static_cast<size_t>(std::max(1u, std::thread::hardware_concurrency())));
    std::atomic<bool> conversion_failed = false;

    auto convert_rows = [&](size_t row_begin, size_t row_end) {
        std::array<float, QK_K>                values = {};
        std::array<std::array<uint8_t, 32>, 8> codes  = {};
        for (size_t row = row_begin; row < row_end; ++row) {
            const uint8_t * source_row = source_bytes + row * row_bytes;
            uint8_t *       output_row = output.data() + row * row_bytes;
            for (size_t block = 0; block < block_count; ++block) {
                dequantize_row_q5_K(reinterpret_cast<const block_q5_K *>(source_row) + block, values.data(), QK_K);
                uint8_t * record = output_row + block * sizeof(block_q5_K);
                for (size_t group = 0; group < 8; ++group) {
                    const float * group_values = values.data() + group * 32;
                    float         positive_max = 0.0f;
                    float         negative_max = 0.0f;
                    for (size_t element = 0; element < 32; ++element) {
                        const float value = group_values[element];
                        if (!std::isfinite(value)) {
                            conversion_failed.store(true, std::memory_order_relaxed);
                            return;
                        }
                        positive_max = std::max(positive_max, value);
                        negative_max = std::max(negative_max, -value);
                    }
                    float scale = std::max(positive_max / 15.0f, negative_max / 16.0f);
                    if (scale == 0.0f) {
                        scale = 1.0f;
                    }
                    for (int iteration = 0; iteration < 2; ++iteration) {
                        double numerator   = 0.0;
                        double denominator = 0.0;
                        for (size_t element = 0; element < 32; ++element) {
                            int quantized = 0;
                            if (!quantize_symmetric_value(group_values[element], scale, -16, 15, quantized)) {
                                conversion_failed.store(true, std::memory_order_relaxed);
                                return;
                            }
                            numerator += static_cast<double>(group_values[element]) * quantized;
                            denominator += static_cast<double>(quantized) * quantized;
                        }
                        if (denominator > 0.0) {
                            scale = static_cast<float>(numerator / denominator);
                        }
                        if (!std::isfinite(scale) || scale <= 0.0f) {
                            conversion_failed.store(true, std::memory_order_relaxed);
                            return;
                        }
                    }

                    const ggml_fp16_t encoded_scale = ggml_fp32_to_fp16(scale);
                    std::memcpy(record + group * sizeof(encoded_scale), &encoded_scale, sizeof(encoded_scale));
                    scale = ggml_fp16_to_fp32(encoded_scale);
                    for (size_t element = 0; element < 32; ++element) {
                        int quantized = 0;
                        if (!quantize_symmetric_value(group_values[element], scale, -16, 15, quantized)) {
                            conversion_failed.store(true, std::memory_order_relaxed);
                            return;
                        }
                        codes[group][element] = static_cast<uint8_t>(quantized) & uint8_t{ 31 };
                    }
                }

                uint8_t * high   = record + 16;
                uint8_t * packed = record + 48;
                for (size_t element = 0; element < 32; ++element) {
                    uint8_t high_byte = 0;
                    for (size_t group = 0; group < 8; ++group) {
                        high_byte |= static_cast<uint8_t>(((codes[group][element] >> 4) & 1) << group);
                    }
                    high[element] = high_byte;
                }
                for (size_t pair = 0; pair < 4; ++pair) {
                    const size_t group0 = pair * 2;
                    const size_t group1 = group0 + 1;
                    for (size_t element = 0; element < 32; ++element) {
                        packed[pair * 32 + element] =
                            static_cast<uint8_t>((codes[group0][element] & 15) | ((codes[group1][element] & 15) << 4));
                    }
                }
            }
        }
    };

    if (!run_worker_threads(thread_count, [&](size_t thread) {
            convert_rows(row_count * thread / thread_count, row_count * (thread + 1) / thread_count);
        })) {
        output.clear();
        status.log("layout %s host materialization worker failed", source.layout.c_str());
        return status;
    }
    if (conversion_failed.load(std::memory_order_relaxed)) {
        output.clear();
        status.log("layout %s cannot represent non-finite symmetric weights", source.layout.c_str());
    }
    return status;
}

static Status materialize_symmetric_i8_k256_row64(const HostWeightSource & source, std::vector<uint8_t> & output) {
    Status status;
    if (source.source_type != GGML_TYPE_Q5_K) {
        status.log("layout %s does not support GGML type %d", source.layout.c_str(),
                   static_cast<int>(source.source_type));
        return status;
    }
    if (source.input_size <= 0 || source.input_size % QK_K != 0 || source.output_size <= 0 ||
        source.output_size % 64 != 0) {
        status.log("layout %s requires K divisible by %d and rows divisible by 64, got K=%lld rows=%lld",
                   source.layout.c_str(), QK_K, static_cast<long long>(source.input_size),
                   static_cast<long long>(source.output_size));
        return status;
    }

    const size_t     row_bytes         = ggml_row_size(GGML_TYPE_Q5_K, source.input_size);
    const size_t     row_count         = static_cast<size_t>(source.output_size);
    const size_t     block_count       = static_cast<size_t>(source.input_size / QK_K);
    constexpr size_t row_group         = 64;
    constexpr size_t scale_plane_bytes = row_group * sizeof(ggml_fp16_t);
    constexpr size_t field_bytes       = row_group * 64;
    constexpr size_t record_bytes      = scale_plane_bytes + 4 * field_bytes;
    constexpr size_t row_record_bytes  = record_bytes / row_group;

    size_t expected_source_bytes = 0;
    size_t expected_output_bytes = 0;
    if (!checked_multiply(row_bytes, row_count, expected_source_bytes) || source.length != expected_source_bytes ||
        !checked_multiply(row_count, block_count, expected_output_bytes) ||
        !checked_multiply(expected_output_bytes, row_record_bytes, expected_output_bytes) ||
        source.materialized_length != expected_output_bytes) {
        status.log("layout %s has inconsistent source/materialized lengths for K=%lld rows=%lld", source.layout.c_str(),
                   static_cast<long long>(source.input_size), static_cast<long long>(source.output_size));
        return status;
    }

    output.resize(expected_output_bytes);
    const auto * source_bytes = static_cast<const uint8_t *>(source.host_data) + source.offset;
    const size_t thread_count =
        std::min(row_count, static_cast<size_t>(std::max(1u, std::thread::hardware_concurrency())));
    std::atomic<bool> conversion_failed = false;

    auto convert_rows = [&](size_t row_begin, size_t row_end) {
        std::array<float, QK_K> values = {};
        for (size_t row = row_begin; row < row_end; ++row) {
            const size_t    row_group_id = row / row_group;
            const size_t    lane         = row % row_group;
            const uint8_t * source_row   = source_bytes + row * row_bytes;
            for (size_t block = 0; block < block_count; ++block) {
                dequantize_row_q5_K(reinterpret_cast<const block_q5_K *>(source_row) + block, values.data(), QK_K);

                float positive_max = 0.0f;
                float negative_max = 0.0f;
                for (float value : values) {
                    if (!std::isfinite(value)) {
                        conversion_failed.store(true, std::memory_order_relaxed);
                        return;
                    }
                    positive_max = std::max(positive_max, value);
                    negative_max = std::max(negative_max, -value);
                }
                float scale = std::max(positive_max, negative_max) / 127.0f;
                if (scale == 0.0f) {
                    scale = 1.0f;
                }
                for (int iteration = 0; iteration < 2; ++iteration) {
                    double numerator   = 0.0;
                    double denominator = 0.0;
                    for (float value : values) {
                        int quantized = 0;
                        if (!quantize_symmetric_value(value, scale, -127, 127, quantized)) {
                            conversion_failed.store(true, std::memory_order_relaxed);
                            return;
                        }
                        numerator += static_cast<double>(value) * quantized;
                        denominator += static_cast<double>(quantized) * quantized;
                    }
                    if (denominator > 0.0) {
                        scale = static_cast<float>(numerator / denominator);
                    }
                    if (!std::isfinite(scale) || scale <= 0.0f) {
                        conversion_failed.store(true, std::memory_order_relaxed);
                        return;
                    }
                }

                const ggml_fp16_t encoded_scale = ggml_fp32_to_fp16(scale);
                scale                           = ggml_fp16_to_fp32(encoded_scale);
                const size_t record_base        = (row_group_id * block_count + block) * record_bytes;
                std::memcpy(output.data() + record_base + lane * sizeof(encoded_scale), &encoded_scale,
                            sizeof(encoded_scale));
                for (size_t chunk = 0; chunk < 4; ++chunk) {
                    auto * payload = reinterpret_cast<int8_t *>(output.data() + record_base + scale_plane_bytes +
                                                                chunk * field_bytes + lane * 64);
                    for (size_t element = 0; element < 64; ++element) {
                        int quantized = 0;
                        if (!quantize_symmetric_value(values[chunk * 64 + element], scale, -127, 127, quantized)) {
                            conversion_failed.store(true, std::memory_order_relaxed);
                            return;
                        }
                        payload[element] = static_cast<int8_t>(quantized);
                    }
                }
            }
        }
    };

    if (!run_worker_threads(thread_count, [&](size_t thread) {
            convert_rows(row_count * thread / thread_count, row_count * (thread + 1) / thread_count);
        })) {
        output.clear();
        status.log("layout %s host materialization worker failed", source.layout.c_str());
        return status;
    }
    if (conversion_failed.load(std::memory_order_relaxed)) {
        output.clear();
        status.log("layout %s cannot represent non-finite symmetric weights", source.layout.c_str());
    }
    return status;
}

// Transpose native Q4_K records in 16-byte fields. No scale or code is
// reconstructed: applying the inverse permutation restores every source byte.
static Status materialize_q4_k_packed_k256_row64(const HostWeightSource & source, std::vector<uint8_t> & output) {
    Status status;
    if (source.source_type != GGML_TYPE_Q4_K || source.input_size <= 0 || source.input_size % QK_K != 0 ||
        source.output_size <= 0 || source.output_size % 64 != 0) {
        status.log("layout %s requires Q4_K, K divisible by %d, and rows divisible by 64", source.layout.c_str(),
                   QK_K);
        return status;
    }

    constexpr size_t row_group = 64;
    constexpr size_t field_bytes = 16;
    constexpr size_t field_count = sizeof(block_q4_K) / field_bytes;
    static_assert(sizeof(block_q4_K) == 144);
    const size_t blocks = static_cast<size_t>(source.input_size / QK_K);
    const size_t rows = static_cast<size_t>(source.output_size);
    size_t bytes = 0;
    if (!checked_multiply(ggml_row_size(GGML_TYPE_Q4_K, source.input_size), rows, bytes) ||
        source.length != bytes || source.materialized_length != bytes) {
        status.log("layout %s has inconsistent source/materialized lengths", source.layout.c_str());
        return status;
    }

    output.resize(bytes);
    const auto * input = static_cast<const uint8_t *>(source.host_data) + source.offset;
    const size_t groups = rows / row_group;
    const size_t thread_count = std::min<size_t>(groups, std::max(1u, std::thread::hardware_concurrency()));
    if (!run_worker_threads(thread_count, [&](size_t thread) {
        const size_t begin = groups * thread / thread_count;
        const size_t end = groups * (thread + 1) / thread_count;
        for (size_t group = begin; group < end; ++group) {
            for (size_t block = 0; block < blocks; ++block) {
                for (size_t field = 0; field < field_count; ++field) {
                    for (size_t lane = 0; lane < row_group; ++lane) {
                        const size_t src = (((group * row_group + lane) * blocks + block) * field_count + field) *
                                           field_bytes;
                        const size_t dst = (((group * blocks + block) * field_count + field) * row_group + lane) *
                                           field_bytes;
                        std::memcpy(output.data() + dst, input + src, field_bytes);
                    }
                }
            }
        }
    })) {
        output.clear();
        status.log("layout %s host materialization worker failed", source.layout.c_str());
    }
    return status;
}

static Status materialize_q6_k_i8_k32_row64(const HostWeightSource & source, std::vector<uint8_t> & output) {
    Status status;
    if (source.source_type != GGML_TYPE_Q6_K) {
        status.log("layout %s does not support GGML type %d", source.layout.c_str(),
                   static_cast<int>(source.source_type));
        return status;
    }
    if (source.input_size <= 0 || source.input_size % QK_K != 0 || source.output_size <= 0 ||
        source.output_size % 64 != 0) {
        status.log("layout %s requires K divisible by %d and rows divisible by 64, got K=%lld rows=%lld",
                   source.layout.c_str(), QK_K, static_cast<long long>(source.input_size),
                   static_cast<long long>(source.output_size));
        return status;
    }

    static_assert(sizeof(block_q6_K) == 210);
    constexpr size_t row_group         = 64;
    constexpr size_t group_count       = 8;
    constexpr size_t group_elements    = 32;
    constexpr size_t d_plane_bytes     = row_group * sizeof(ggml_fp16_t);
    constexpr size_t scale_plane_bytes = group_count * row_group * 2;
    constexpr size_t code_plane_bytes  = group_count * row_group * group_elements;
    constexpr size_t tile_block_bytes  = d_plane_bytes + scale_plane_bytes + code_plane_bytes;
    static_assert(tile_block_bytes == row_group * 274);

    const size_t block_count           = static_cast<size_t>(source.input_size / QK_K);
    const size_t row_bytes             = block_count * sizeof(block_q6_K);
    size_t       expected_source_bytes = 0;
    size_t       expected_output_bytes = 0;
    if (!checked_multiply(row_bytes, static_cast<size_t>(source.output_size), expected_source_bytes) ||
        !checked_multiply(static_cast<size_t>(source.output_size), block_count, expected_output_bytes) ||
        !checked_multiply(expected_output_bytes, size_t{ 274 }, expected_output_bytes) ||
        source.length != expected_source_bytes || source.materialized_length != expected_output_bytes) {
        status.log("layout %s source/materialized lengths %zu/%zu do not match expected %zu/%zu", source.layout.c_str(),
                   source.length, source.materialized_length, expected_source_bytes, expected_output_bytes);
        return status;
    }

    output.resize(expected_output_bytes);
    const auto * source_bytes       = static_cast<const uint8_t *>(source.host_data) + source.offset;
    const size_t output_group_count = static_cast<size_t>(source.output_size) / row_group;
    const size_t thread_count = std::min<size_t>(output_group_count, std::max(1u, std::thread::hardware_concurrency()));

    auto convert_groups = [&](size_t group_begin, size_t group_end) {
        for (size_t output_group = group_begin; output_group < group_end; ++output_group) {
            for (size_t block = 0; block < block_count; ++block) {
                const size_t tile_base   = (output_group * block_count + block) * tile_block_bytes;
                uint8_t *    d_plane     = output.data() + tile_base;
                int8_t *     scale_plane = reinterpret_cast<int8_t *>(d_plane + d_plane_bytes);
                int8_t *     code_plane  = scale_plane + scale_plane_bytes;
                for (size_t lane = 0; lane < row_group; ++lane) {
                    const size_t row          = output_group * row_group + lane;
                    const auto * source_block = reinterpret_cast<const block_q6_K *>(source_bytes + row * row_bytes +
                                                                                     block * sizeof(block_q6_K));
                    std::memcpy(d_plane + lane * sizeof(ggml_fp16_t), &source_block->d, sizeof(ggml_fp16_t));
                    for (size_t quant_group = 0; quant_group < group_count; ++quant_group) {
                        int8_t * scales            = scale_plane + (quant_group * row_group + lane) * 2;
                        scales[0]                  = source_block->scales[quant_group * 2];
                        scales[1]                  = source_block->scales[quant_group * 2 + 1];
                        int8_t *     codes         = code_plane + (quant_group * row_group + lane) * group_elements;
                        const size_t half128       = quant_group / 4;
                        const size_t group_in_half = quant_group % 4;
                        for (size_t element = 0; element < group_elements; ++element) {
                            const size_t  low_index  = half128 * 64 + (group_in_half % 2) * 32 + element;
                            const size_t  high_index = half128 * 32 + element;
                            const uint8_t low =
                                static_cast<uint8_t>((source_block->ql[low_index] >> ((group_in_half / 2) * 4)) & 0x0F);
                            const uint8_t high =
                                static_cast<uint8_t>((source_block->qh[high_index] >> (group_in_half * 2)) & 0x03);
                            codes[element] = static_cast<int8_t>((low | (high << 4)) - 32);
                        }
                    }
                }
            }
        }
    };

    if (!run_worker_threads(thread_count, [&](size_t thread) {
            const size_t group_begin = output_group_count * thread / thread_count;
            const size_t group_end   = output_group_count * (thread + 1) / thread_count;
            convert_groups(group_begin, group_end);
        })) {
        output.clear();
        status.log("layout %s host materialization worker failed", source.layout.c_str());
    }
    return status;
}

static Status materialize_q6_k_packed_k256_row64_scalerow(const HostWeightSource & source,
                                                          std::vector<uint8_t> &   output) {
    Status status;
    if (source.source_type != GGML_TYPE_Q6_K) {
        status.log("layout %s does not support GGML type %d", source.layout.c_str(),
                   static_cast<int>(source.source_type));
        return status;
    }
    if (source.input_size <= 0 || source.input_size % QK_K != 0 || source.output_size <= 0 ||
        source.output_size % 64 != 0) {
        status.log("layout %s requires K divisible by %d and rows divisible by 64, got K=%lld rows=%lld",
                   source.layout.c_str(), QK_K, static_cast<long long>(source.input_size),
                   static_cast<long long>(source.output_size));
        return status;
    }

    static_assert(sizeof(block_q6_K) == 210);
    constexpr size_t row_group         = 64;
    constexpr size_t half_count        = 2;
    constexpr size_t raw_field_count   = 3;
    constexpr size_t raw_field_bytes   = 32;
    constexpr size_t scale_group_count = 8;
    constexpr size_t d_plane_bytes     = row_group * sizeof(ggml_fp16_t);
    constexpr size_t scale_plane_bytes = scale_group_count * row_group * half_count;
    constexpr size_t raw_plane_bytes   = half_count * raw_field_count * row_group * raw_field_bytes;
    constexpr size_t tile_block_bytes  = d_plane_bytes + scale_plane_bytes + raw_plane_bytes;
    static_assert(tile_block_bytes == row_group * sizeof(block_q6_K));

    const size_t block_count           = static_cast<size_t>(source.input_size / QK_K);
    const size_t row_bytes             = block_count * sizeof(block_q6_K);
    size_t       expected_source_bytes = 0;
    if (!checked_multiply(row_bytes, static_cast<size_t>(source.output_size), expected_source_bytes) ||
        source.length != expected_source_bytes || source.materialized_length != expected_source_bytes) {
        status.log("layout %s source/materialized lengths %zu/%zu do not match expected %zu", source.layout.c_str(),
                   source.length, source.materialized_length, expected_source_bytes);
        return status;
    }

    output.resize(expected_source_bytes);
    const auto * source_bytes       = static_cast<const uint8_t *>(source.host_data) + source.offset;
    const size_t output_group_count = static_cast<size_t>(source.output_size) / row_group;
    const size_t thread_count = std::min<size_t>(output_group_count, std::max(1u, std::thread::hardware_concurrency()));

    auto convert_groups = [&](size_t group_begin, size_t group_end) {
        for (size_t output_group = group_begin; output_group < group_end; ++output_group) {
            for (size_t block = 0; block < block_count; ++block) {
                const size_t tile_base   = (output_group * block_count + block) * tile_block_bytes;
                uint8_t *    d_plane     = output.data() + tile_base;
                int8_t *     scale_plane = reinterpret_cast<int8_t *>(d_plane + d_plane_bytes);
                uint8_t *    raw_plane   = reinterpret_cast<uint8_t *>(scale_plane + scale_plane_bytes);
                for (size_t lane = 0; lane < row_group; ++lane) {
                    const size_t row          = output_group * row_group + lane;
                    const auto * source_block = reinterpret_cast<const block_q6_K *>(source_bytes + row * row_bytes +
                                                                                     block * sizeof(block_q6_K));
                    std::memcpy(d_plane + lane * sizeof(ggml_fp16_t), &source_block->d, sizeof(ggml_fp16_t));
                    for (size_t quant_group = 0; quant_group < scale_group_count; ++quant_group) {
                        for (size_t half = 0; half < half_count; ++half) {
                            const size_t destination_index =
                                (lane * half_count + half) * scale_group_count + quant_group;
                            scale_plane[destination_index] = source_block->scales[quant_group * half_count + half];
                        }
                    }
                    for (size_t half = 0; half < half_count; ++half) {
                        uint8_t * ql0 = raw_plane + ((half * raw_field_count) * row_group + lane) * raw_field_bytes;
                        uint8_t * ql1 = ql0 + row_group * raw_field_bytes;
                        uint8_t * qh  = ql1 + row_group * raw_field_bytes;
                        std::memcpy(ql0, source_block->ql + half * 64, raw_field_bytes);
                        std::memcpy(ql1, source_block->ql + half * 64 + raw_field_bytes, raw_field_bytes);
                        std::memcpy(qh, source_block->qh + half * raw_field_bytes, raw_field_bytes);
                    }
                }
            }
        }
    };

    if (!run_worker_threads(thread_count, [&](size_t thread) {
            const size_t group_begin = output_group_count * thread / thread_count;
            const size_t group_end   = output_group_count * (thread + 1) / thread_count;
            convert_groups(group_begin, group_end);
        })) {
        output.clear();
        status.log("layout %s host materialization worker failed", source.layout.c_str());
    }
    return status;
}

static Status materialize_q6_k_symmetric_i2_packed_k256_row64_scalerow(const HostWeightSource & source,
                                                                       std::vector<uint8_t> &   output) {
    Status status;
    if (source.source_type != GGML_TYPE_Q6_K || source.input_size <= 0 || source.input_size % QK_K != 0 ||
        source.output_size <= 0 || source.output_size % 64 != 0 ||
        static_cast<uint64_t>(source.output_size) > std::numeric_limits<size_t>::max() - 255) {
        status.log(
            "layout %s requires Q6_K, K divisible by %d, and rows divisible by 64, got type=%d K=%lld "
            "rows=%lld",
            source.layout.c_str(), QK_K, static_cast<int>(source.source_type),
            static_cast<long long>(source.input_size), static_cast<long long>(source.output_size));
        return status;
    }

    const size_t logical_rows    = static_cast<size_t>(source.output_size);
    const size_t row_group       = ((logical_rows + 255) / 256) * 32;
    const size_t physical_rows   = (logical_rows + row_group - 1) / row_group * row_group;
    const size_t block_count     = static_cast<size_t>(source.input_size / QK_K);
    size_t       symmetric_bytes = 0;
    size_t       packed_bytes    = 0;
    if (!checked_multiply(physical_rows, block_count, symmetric_bytes) ||
        !checked_multiply(symmetric_bytes, size_t{ 68 }, symmetric_bytes) ||
        !checked_multiply(logical_rows, block_count, packed_bytes) ||
        !checked_multiply(packed_bytes, sizeof(block_q6_K), packed_bytes) ||
        symmetric_bytes > std::numeric_limits<size_t>::max() - packed_bytes || source.length != packed_bytes ||
        source.materialized_length != symmetric_bytes + packed_bytes) {
        status.log("layout %s source/materialized lengths %zu/%zu are inconsistent with K=%lld rows=%lld",
                   source.layout.c_str(), source.length, source.materialized_length,
                   static_cast<long long>(source.input_size), static_cast<long long>(source.output_size));
        return status;
    }

    HostWeightSource symmetric_source    = source;
    symmetric_source.layout              = kSymmetricI2K32EightGroupsShared4Layout;
    symmetric_source.materialized_length = symmetric_bytes;
    status                               = materialize_symmetric_k32_eightgroups_shared4(symmetric_source, 2, output);
    if (!status.success()) {
        return status;
    }
    output.resize(source.materialized_length);

    HostWeightSource packed_source    = source;
    packed_source.layout              = kQ6KPackedK256Row64ScaleRowLayout;
    packed_source.materialized_length = packed_bytes;
    std::vector<uint8_t> packed;
    status = materialize_q6_k_packed_k256_row64_scalerow(packed_source, packed);
    if (!status.success()) {
        output.clear();
        return status;
    }
    std::memcpy(output.data() + symmetric_bytes, packed.data(), packed.size());
    return status;
}

static Status materialize_weight(const HostWeightSource & source,
                                 const void *&            upload_data,
                                 size_t &                 upload_size,
                                 std::vector<uint8_t> &   transformed) {
    Status status;
    if (source.layout == kNativeWeightLayout) {
        upload_data = static_cast<const uint8_t *>(source.host_data) + source.offset;
        upload_size = source.length;
        return status;
    }
    if (source.layout == kSymmetricI4K32EightGroupsShared4Layout ||
        source.layout == kSymmetricI4K32EightGroupsShared4MultistartLayout) {
        status = materialize_symmetric_i4_k32_eightgroups_shared4(source, transformed);
        if (status.success()) {
            upload_data = transformed.data();
            upload_size = transformed.size();
        }
        return status;
    }
    if (source.layout == kSymmetricI4K32Row64Layout) {
        status = materialize_symmetric_i4_k32_row64(source, transformed);
        if (status.success()) {
            upload_data = transformed.data();
            upload_size = transformed.size();
        }
        return status;
    }
    if (source.layout == kSymmetricI4K64Row64Layout) {
        status = materialize_symmetric_i4_k64_row64(source, transformed);
        if (status.success()) {
            upload_data = transformed.data();
            upload_size = transformed.size();
        }
        return status;
    }
    if (source.layout == kQ5KSymmetricI5K32Layout) {
        status = materialize_symmetric_i5_k32(source, transformed);
        if (status.success()) {
            upload_data = transformed.data();
            upload_size = transformed.size();
        }
        return status;
    }
    if (source.layout == kQ5KSymmetricI8K256Row64Layout) {
        status = materialize_symmetric_i8_k256_row64(source, transformed);
        if (status.success()) {
            upload_data = transformed.data();
            upload_size = transformed.size();
        }
        return status;
    }
    if (source.layout == kQ4KPackedK256Row64Layout) {
        status = materialize_q4_k_packed_k256_row64(source, transformed);
        if (status.success()) {
            upload_data = transformed.data();
            upload_size = transformed.size();
        }
        return status;
    }
    if (source.layout == kQ6KI8K32Row64Layout) {
        status = materialize_q6_k_i8_k32_row64(source, transformed);
        if (status.success()) {
            upload_data = transformed.data();
            upload_size = transformed.size();
        }
        return status;
    }
    if (source.layout == kQ6KPackedK256Row64ScaleRowLayout) {
        status = materialize_q6_k_packed_k256_row64_scalerow(source, transformed);
        if (status.success()) {
            upload_data = transformed.data();
            upload_size = transformed.size();
        }
        return status;
    }
    if (source.layout == kQ6KSymmetricI2PackedK256Row64ScaleRowLayout) {
        status = materialize_q6_k_symmetric_i2_packed_k256_row64_scalerow(source, transformed);
        if (status.success()) {
            upload_data = transformed.data();
            upload_size = transformed.size();
        }
        return status;
    }
    status.log("unknown host weight layout %s", source.layout.c_str());
    return status;
}

static Status allocate_device_buffer(hrx_device_t device, size_t size, hrx_buffer_t & buffer) {
    Status status;
    if (device == nullptr) {
        status.log("missing HRX device for host memory allocation");
        return status;
    }
    if (size == 0) {
        status.log("cannot allocate an empty host memory device buffer");
        return status;
    }
    hrx_buffer_params_t params = {
        HRX_MEMORY_TYPE_DEVICE_LOCAL,
        HRX_MEMORY_ACCESS_ALL,
        HRX_BUFFER_USAGE_DEFAULT,
        0,
    };
    if (ErrorResult error =
            take_status(hrx_allocator_allocate_buffer(hrx_device_allocator(device), params, size, &buffer))) {
        status.log("allocate host memory device buffer: %s", error->c_str());
    }
    return status;
}

}  // namespace

Status allocate_mapped_host_staging_buffer(hrx_device_t device, size_t size, hrx_buffer_t & buffer, void *& host_data) {
    Status status;
    buffer    = nullptr;
    host_data = nullptr;
    if (device == nullptr) {
        status.log("missing HRX device for host staging allocation");
        return status;
    }
    if (size == 0) {
        status.log("cannot allocate an empty host staging buffer");
        return status;
    }
    hrx_buffer_params_t params = {
        HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
        HRX_MEMORY_ACCESS_ALL,
        HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SCOPED | HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
        0,
    };
    if (ErrorResult error =
            take_status(hrx_allocator_allocate_buffer(hrx_device_allocator(device), params, size, &buffer))) {
        status.log("allocate mapped HRX host staging buffer: %s", error->c_str());
        return status;
    }
    if (ErrorResult error = take_status(hrx_buffer_map(buffer, HRX_MAP_READ | HRX_MAP_WRITE, 0, size, &host_data))) {
        status.log("map HRX host staging buffer: %s", error->c_str());
        hrx_buffer_release(buffer);
        buffer = nullptr;
    }
    return status;
}

HostTransferManager::~HostTransferManager() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mapped_download_buffer_ != nullptr) {
        hrx_buffer_release(mapped_download_buffer_);
    }
}

Status HostTransferManager::upload_synchronous(hrx_stream_t stream,
                                               const void * host_source,
                                               hrx_buffer_t destination,
                                               size_t       offset,
                                               size_t       size) {
    Status status;
    if (size == 0) {
        return status;
    }
    if (stream == nullptr || host_source == nullptr || destination == nullptr) {
        status.log("invalid HRX host upload");
        return status;
    }
    hrx_device_t device = nullptr;
    if (ErrorResult error = take_status(hrx_stream_get_device(stream, &device))) {
        status.log("query HRX upload device failed: %s", error->c_str());
        return status;
    }
    if (ErrorResult error = take_status(hrx_stream_synchronize(stream))) {
        status.log("synchronize before HRX host upload failed: %s", error->c_str());
        return status;
    }
    if (ErrorResult error = take_status(hrx_synchronous_h2d(device, host_source, destination, offset, size))) {
        status.log("synchronous HRX host upload failed: %s", error->c_str());
        return status;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.uploads;
    stats_.upload_bytes += size;
    return status;
}

Status HostTransferManager::upload_async(hrx_stream_t stream,
                                         const void * host_source,
                                         hrx_buffer_t destination,
                                         size_t       offset,
                                         size_t       size) {
    Status status;
    if (size == 0) {
        return status;
    }
    if (stream == nullptr || host_source == nullptr || destination == nullptr) {
        status.log("invalid HRX host upload");
        return status;
    }

    // Inline stream updates are intended for small payloads. Use bulk H2D for
    // large host-staged tensors to avoid hundreds of update commands.
    if (size >= kLargeHostUploadBytes) {
        return upload_synchronous(stream, host_source, destination, offset, size);
    }

    const uint8_t * host_bytes = static_cast<const uint8_t *>(host_source);
    size_t          uploaded   = 0;
    while (uploaded < size) {
        const size_t remaining  = size - uploaded;
        const size_t chunk_size = remaining < kMaxInlineUploadBytes ? remaining : kMaxInlineUploadBytes;
        if (ErrorResult error = take_status(
                hrx_stream_update_buffer(stream, host_bytes + uploaded, chunk_size, destination, offset + uploaded))) {
            status.log("HRX async host upload failed: %s", error->c_str());
            return status;
        }
        uploaded += chunk_size;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.uploads;
    stats_.upload_bytes += size;
    return status;
}

Status HostTransferManager::download_synchronous(hrx_stream_t stream,
                                                 hrx_buffer_t source,
                                                 size_t       offset,
                                                 void *       host_destination,
                                                 size_t       size) {
    Status status;
    if (size == 0) {
        return status;
    }
    if (stream == nullptr || source == nullptr || host_destination == nullptr) {
        status.log("invalid HRX host download");
        return status;
    }
    hrx_device_t device = nullptr;
    if (ErrorResult error = take_status(hrx_stream_get_device(stream, &device))) {
        status.log("query HRX download device failed: %s", error->c_str());
        return status;
    }
    if (ErrorResult error = take_status(hrx_stream_synchronize(stream))) {
        status.log("synchronize before HRX host download failed: %s", error->c_str());
        return status;
    }
    if (ErrorResult error = take_status(hrx_synchronous_d2h(device, source, offset, host_destination, size))) {
        status.log("synchronous HRX host download failed: %s", error->c_str());
        return status;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.downloads;
    stats_.download_bytes += size;
    return status;
}

Status HostTransferManager::download_to_mapped_synchronous(hrx_device_t              device,
                                                           hrx_stream_t              stream,
                                                           hrx_buffer_t              source,
                                                           size_t                    offset,
                                                           size_t                    size,
                                                           HostMappedDownloadLease & lease) {
    Status status;
    lease = {};
    if (device == nullptr || stream == nullptr || source == nullptr || size == 0) {
        status.log("invalid mapped HRX host download");
        return status;
    }
    lease.lock = std::unique_lock<std::mutex>(mutex_);
    if (mapped_download_capacity_ < size) {
        size_t capacity = std::max<size_t>(mapped_download_capacity_, 1);
        while (capacity < size && capacity <= std::numeric_limits<size_t>::max() / 2) {
            capacity *= 2;
        }
        if (capacity < size) {
            capacity = size;
        }
        hrx_buffer_t new_buffer = nullptr;
        void *       new_data   = nullptr;
        status = allocate_mapped_host_staging_buffer(device, capacity, new_buffer, new_data);
        if (!status.success()) {
            return status;
        }
        if (mapped_download_buffer_ != nullptr) {
            hrx_buffer_release(mapped_download_buffer_);
        }
        mapped_download_buffer_   = new_buffer;
        mapped_download_data_     = new_data;
        mapped_download_capacity_ = capacity;
    }
    if (ErrorResult error =
            take_status(hrx_stream_copy_buffer(stream, source, offset, mapped_download_buffer_, 0, size))) {
        status.log("mapped host weight download failed: %s", error->c_str());
        return status;
    }
    if (ErrorResult error = take_status(hrx_stream_synchronize(stream))) {
        status.log("synchronize mapped host weight download failed: %s", error->c_str());
        return status;
    }
    ++stats_.downloads;
    stats_.download_bytes += size;
    lease.host_data = mapped_download_data_;
    return status;
}

HostTransferStats HostTransferManager::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void HostTransferManager::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    stats_ = {};
}

struct HostWeightLease::Entry {
    ~Entry() {
        if (buffer != nullptr) {
            hrx_buffer_release(buffer);
        }
    }

    hrx_buffer_t buffer = nullptr;
    size_t       length = 0;
    std::string  layout;
};

HostWeightLease::HostWeightLease(std::shared_ptr<Entry> entry) : entry_(std::move(entry)) {}

bool HostWeightLease::valid() const {
    return entry_ != nullptr && entry_->buffer != nullptr;
}

hrx_buffer_t HostWeightLease::buffer() const {
    return valid() ? entry_->buffer : nullptr;
}

size_t HostWeightLease::length() const {
    return entry_ != nullptr ? entry_->length : 0;
}

const std::string & HostWeightLease::layout() const {
    static const std::string empty;
    return entry_ != nullptr ? entry_->layout : empty;
}

HostWeightCache::~HostWeightCache() {
    clear();
}

size_t HostWeightCache::SourceKeyHash::operator()(const SourceKey & key) const {
    uint64_t hash = UINT64_C(1469598103934665603);
    auto     mix  = [&](uint64_t value) {
        hash ^= value;
        hash *= UINT64_C(1099511628211);
    };
    mix(key.identity);
    mix(key.generation);
    mix(static_cast<uint64_t>(key.capacity));
    mix(static_cast<uint64_t>(key.offset));
    mix(static_cast<uint64_t>(key.length));
    mix(static_cast<uint64_t>(key.materialized_length));
    for (unsigned char byte : key.layout) {
        mix(byte);
    }
    mix(static_cast<uint64_t>(key.source_type));
    mix(static_cast<uint64_t>(key.input_size));
    mix(static_cast<uint64_t>(key.output_size));
    return static_cast<size_t>(hash);
}

HostWeightAcquireResult HostWeightCache::acquire(hrx_device_t             device,
                                                 hrx_stream_t             stream,
                                                 HostTransferManager &    transfers,
                                                 const HostWeightSource & source) {
    HostWeightAcquireResult result;
    if (stream == nullptr) {
        result.status.log("host weight residency requires an HRX stream");
        return result;
    }
    const bool has_host_source   = source.host_data != nullptr;
    const bool has_device_source = source.device_buffer != nullptr;
    if (has_host_source == has_device_source || source.identity == 0 || source.generation == 0 || source.length == 0 ||
        source.offset > source.capacity || source.length > source.capacity - source.offset) {
        result.status.log("invalid host weight source");
        return result;
    }
    if (source.layout.empty()) {
        result.status.log("host weight source has no layout");
        return result;
    }

    const size_t materialized_length =
        source.layout == kNativeWeightLayout ? source.length : source.materialized_length;
    if (materialized_length == 0) {
        result.status.log("host weight source has an empty materialized layout");
        return result;
    }

    const SourceKey key{ source.identity,   source.generation,   source.capacity, source.offset,
                         source.length,     materialized_length, source.layout,   source.source_type,
                         source.input_size, source.output_size };
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  found = entries_.find(key);
        if (found != entries_.end()) {
            ++stats_.hits;
            result.lease = HostWeightLease(found->second);
            return result;
        }
    }

    HostWeightSource        materialization_source = source;
    HostMappedDownloadLease mapped_canonical;
    std::vector<uint8_t>    canonical;
    if (has_device_source) {
        if (source.layout == kNativeWeightLayout) {
            result.status.log("native device weights do not require host materialization");
            return result;
        }
        if (!mapped_download_staging_enabled()) {
            canonical.resize(source.length);
        }
        void * canonical_data = canonical.data();
        if (mapped_download_staging_enabled()) {
            result.status = transfers.download_to_mapped_synchronous(
                device, stream, source.device_buffer, source.offset, source.length, mapped_canonical);
            canonical_data = mapped_canonical.host_data;
        } else {
            result.status = transfers.download_synchronous(stream, source.device_buffer, source.offset, canonical_data,
                                                           source.length);
        }
        if (!result.status.success()) {
            return result;
        }
        materialization_source.host_data     = canonical_data;
        materialization_source.device_buffer = nullptr;
        materialization_source.capacity      = source.length;
        materialization_source.offset        = 0;
    }

    const void *         upload_data = nullptr;
    size_t               upload_size = 0;
    std::vector<uint8_t> transformed;
    result.status = materialize_weight(materialization_source, upload_data, upload_size, transformed);
    if (mapped_canonical.lock.owns_lock()) {
        mapped_canonical.lock.unlock();
    }
    if (!result.status.success()) {
        return result;
    }
    if (upload_size != materialized_length) {
        result.status.log("host weight layout %s produced %zu bytes, expected %zu", source.layout.c_str(), upload_size,
                          materialized_length);
        return result;
    }

    auto entry    = std::make_shared<HostWeightLease::Entry>();
    entry->length = materialized_length;
    entry->layout = source.layout;
    result.status = allocate_device_buffer(device, materialized_length, entry->buffer);
    if (!result.status.success()) {
        return result;
    }
    result.status = transfers.upload_synchronous(stream, upload_data, entry->buffer, 0, upload_size);
    if (!result.status.success()) {
        return result;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  inserted = entries_.emplace(key, entry);
        if (!inserted.second) {
            ++stats_.hits;
            result.lease = HostWeightLease(inserted.first->second);
            return result;
        }
        ++stats_.misses;
        stats_.allocation_count = entries_.size();
        stats_.resident_bytes += materialized_length;
    }
    result.lease = HostWeightLease(std::move(entry));
    return result;
}

HostWeightCacheStats HostWeightCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void HostWeightCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    stats_ = {};
}

HostStagingBuffer::~HostStagingBuffer() {
    clear();
}

HostStagingBuffer::HostStagingBuffer(HostStagingBuffer && other) noexcept {
    *this = std::move(other);
}

HostStagingBuffer & HostStagingBuffer::operator=(HostStagingBuffer && other) noexcept {
    if (this == &other) {
        return *this;
    }
    clear();
    buffer          = other.buffer;
    host_data       = other.host_data;
    source_host_buffer = std::move(other.source_host_buffer);
    value           = other.value;
    length          = other.length;
    upload          = other.upload;
    download        = other.download;
    other.buffer    = nullptr;
    other.host_data = nullptr;
    other.value     = -1;
    other.length    = 0;
    other.upload    = false;
    other.download  = false;
    return *this;
}

void HostStagingBuffer::clear() {
    if (buffer != nullptr) {
        hrx_buffer_release(buffer);
        buffer = nullptr;
    }
    host_data = nullptr;
    source_host_buffer = HostBufferRef{};
    value     = -1;
    length    = 0;
    upload    = false;
    download  = false;
}

Status allocate_host_staging_buffer(hrx_device_t device, size_t size, HostStagingBuffer & staging) {
    staging.clear();
    Status status = allocate_device_buffer(device, size, staging.buffer);
    if (status.success()) {
        staging.length = size;
    }
    return status;
}

}  // namespace ggml::hrx
