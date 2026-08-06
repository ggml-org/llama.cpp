#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

enum class ggml_cuda_rdna2_bf16_hidden_option {
    disabled,
    enabled,
    invalid,
};

inline ggml_cuda_rdna2_bf16_hidden_option ggml_cuda_parse_rdna2_bf16_hidden_option(const char * value) {
    if (value == nullptr || std::strcmp(value, "0") == 0) {
        return ggml_cuda_rdna2_bf16_hidden_option::disabled;
    }
    if (std::strcmp(value, "1") == 0) {
        return ggml_cuda_rdna2_bf16_hidden_option::enabled;
    }
    return ggml_cuda_rdna2_bf16_hidden_option::invalid;
}

enum class ggml_cuda_rdna2_bf16_hidden_activation {
    disabled,
    enabled,
    invalid_option,
    requires_hip,
    requires_nccl,
    requires_explicit_nccl,
    requires_four_distinct_rdna2,
};

inline ggml_cuda_rdna2_bf16_hidden_activation ggml_cuda_validate_rdna2_bf16_hidden_activation(
        ggml_cuda_rdna2_bf16_hidden_option option,
        bool hip_compiled,
        bool nccl_compiled,
        const char * allreduce_backend,
        bool candidate_topology) {
    if (option == ggml_cuda_rdna2_bf16_hidden_option::invalid) {
        return ggml_cuda_rdna2_bf16_hidden_activation::invalid_option;
    }
    if (option == ggml_cuda_rdna2_bf16_hidden_option::disabled) {
        return ggml_cuda_rdna2_bf16_hidden_activation::disabled;
    }
    if (!hip_compiled) {
        return ggml_cuda_rdna2_bf16_hidden_activation::requires_hip;
    }
    if (!nccl_compiled) {
        return ggml_cuda_rdna2_bf16_hidden_activation::requires_nccl;
    }
    if (allreduce_backend == nullptr || std::strcmp(allreduce_backend, "nccl") != 0) {
        return ggml_cuda_rdna2_bf16_hidden_activation::requires_explicit_nccl;
    }
    if (!candidate_topology) {
        return ggml_cuda_rdna2_bf16_hidden_activation::requires_four_distinct_rdna2;
    }
    return ggml_cuda_rdna2_bf16_hidden_activation::enabled;
}

struct ggml_cuda_allreduce_topology_device {
    int logical_id;
    int physical_id;
    int share_count;
    bool rdna2;

    ggml_cuda_allreduce_topology_device(
            int logical_id = -1, int physical_id = -1, int share_count = 0, bool rdna2 = false)
        : logical_id(logical_id), physical_id(physical_id), share_count(share_count), rdna2(rdna2) {}
};

inline bool ggml_cuda_is_four_distinct_rdna2_topology(
        const ggml_cuda_allreduce_topology_device * devices,
        size_t n_devices,
        int exposed_device_count,
        int physical_device_count) {
    if (devices == nullptr || n_devices != 4) {
        return false;
    }
    for (size_t i = 0; i < n_devices; ++i) {
        const auto & device = devices[i];
        if (device.logical_id < 0 || device.logical_id >= exposed_device_count ||
                device.physical_id < 0 || device.physical_id >= physical_device_count ||
                device.share_count != 1 || !device.rdna2) {
            return false;
        }
        for (size_t j = 0; j < i; ++j) {
            if (devices[j].logical_id == device.logical_id || devices[j].physical_id == device.physical_id) {
                return false;
            }
        }
    }
    return true;
}

inline bool ggml_cuda_any_allreduce_force_flag(
        const uint32_t * flags, size_t n_flags, uint32_t force_mask) {
    for (size_t i = 0; i < n_flags; ++i) {
        if ((flags[i] & force_mask) != 0) {
            return true;
        }
    }
    return false;
}

enum class ggml_cuda_allreduce_precision {
    forced_fp32,
    legacy_fp32,
    candidate_bf16,
    legacy_bf16,
};

struct ggml_cuda_allreduce_precision_input {
    bool candidate_enabled  = false;
    bool candidate_topology = false;
    bool all_f32            = false;
    bool all_contiguous     = false;
    bool all_same_shape     = false;
    bool force_fp32         = false;

    size_t  n_backends = 0;
    int64_t nelements  = 0;
    int64_t ne[4]      = { 0, 0, 0, 0 };
};

// This is deliberately a shape-scoped experimental selector, not a generic
// BF16 policy. DSV4 one-token hidden reductions have this exact shape; all
// other shapes retain the existing size-based precision heuristic.
inline bool ggml_cuda_is_rdna2_bf16_hidden_shape(const ggml_cuda_allreduce_precision_input & input) {
    return input.candidate_topology &&
        input.n_backends == 4 &&
        input.nelements == 4096 &&
        input.ne[0] == 4096 && input.ne[1] == 1 && input.ne[2] == 1 && input.ne[3] == 1 &&
        input.all_f32 && input.all_contiguous && input.all_same_shape;
}

inline bool ggml_cuda_allreduce_is_small_by_default(size_t n_backends, int64_t nelements) {
    return (n_backends <= 2 && nelements < 32768) ||
        (n_backends == 3 && nelements < 131072) ||
        (n_backends >= 4 && nelements < 262144);
}

inline ggml_cuda_allreduce_precision ggml_cuda_select_allreduce_precision(
        const ggml_cuda_allreduce_precision_input & input) {
    if (input.force_fp32) {
        return ggml_cuda_allreduce_precision::forced_fp32;
    }
    if (input.candidate_enabled && ggml_cuda_is_rdna2_bf16_hidden_shape(input)) {
        return ggml_cuda_allreduce_precision::candidate_bf16;
    }
    if (ggml_cuda_allreduce_is_small_by_default(input.n_backends, input.nelements)) {
        return ggml_cuda_allreduce_precision::legacy_fp32;
    }
    return ggml_cuda_allreduce_precision::legacy_bf16;
}

struct ggml_cuda_allreduce_audit_counters {
    uint64_t allreduce_calls           = 0;
    uint64_t zero_element_calls        = 0;
    uint64_t candidate_eligible_calls  = 0;
    uint64_t candidate_bf16_calls      = 0;
    uint64_t candidate_disabled_calls  = 0;
    uint64_t force_fp32_calls          = 0;
    uint64_t force_candidate_conflicts = 0;
    uint64_t legacy_fp32_calls         = 0;
    uint64_t legacy_bf16_calls         = 0;
};

inline void ggml_cuda_audit_record_call(ggml_cuda_allreduce_audit_counters & counters, bool zero_elements) {
    ++counters.allreduce_calls;
    counters.zero_element_calls += zero_elements;
}

inline void ggml_cuda_audit_record_decision(
        ggml_cuda_allreduce_audit_counters & counters,
        bool candidate_eligible,
        bool candidate_enabled,
        bool force_fp32,
        ggml_cuda_allreduce_precision precision) {
    counters.candidate_eligible_calls += candidate_eligible;
    counters.candidate_disabled_calls += candidate_eligible && !candidate_enabled &&
        precision == ggml_cuda_allreduce_precision::legacy_fp32;
    counters.force_fp32_calls += force_fp32;
    counters.force_candidate_conflicts += candidate_eligible && force_fp32;
    counters.candidate_bf16_calls += precision == ggml_cuda_allreduce_precision::candidate_bf16;
    counters.legacy_fp32_calls += precision == ggml_cuda_allreduce_precision::legacy_fp32;
    counters.legacy_bf16_calls += precision == ggml_cuda_allreduce_precision::legacy_bf16;
}

inline uint64_t ggml_cuda_audit_nonzero_decision_count(const ggml_cuda_allreduce_audit_counters & counters) {
    return counters.force_fp32_calls + counters.candidate_bf16_calls +
        counters.legacy_fp32_calls + counters.legacy_bf16_calls;
}

inline bool ggml_cuda_append_allreduce_audit_line(const char * path, const std::string & line) {
    if (path == nullptr || path[0] == '\0') {
        return false;
    }
    FILE * file = std::fopen(path, "a");
    if (file == nullptr) {
        return false;
    }
    const size_t written = std::fwrite(line.data(), 1, line.size(), file);
    const bool flush_failed = std::fflush(file) != 0;
    const bool stream_failed = std::ferror(file) != 0;
    const bool close_failed = std::fclose(file) != 0;
    return written == line.size() && !flush_failed && !stream_failed && !close_failed;
}