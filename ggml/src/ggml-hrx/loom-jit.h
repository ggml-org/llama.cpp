// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hrx_runtime.h"

#include <array>
#include <cstddef>
#include <cstdint>

struct ggml_hrx_loom_jit_amdgpu;

enum class ggml_hrx_loom_jit_source_format {
    Text,
    Bytecode,
};
inline constexpr auto GGML_HRX_LOOM_JIT_SOURCE_FORMAT_TEXT     = ggml_hrx_loom_jit_source_format::Text;
inline constexpr auto GGML_HRX_LOOM_JIT_SOURCE_FORMAT_BYTECODE = ggml_hrx_loom_jit_source_format::Bytecode;

struct ggml_hrx_loom_jit_amdgpu_options {
    const char * processor           = nullptr;
    const char * identifier          = nullptr;
    const char * sanitizer           = nullptr;
    const char * sanitizer_reporting = nullptr;
};

struct ggml_hrx_loom_jit_config_binding {
    const char * key   = nullptr;
    const char * value = nullptr;
};

struct ggml_hrx_loom_jit_source {
    const void *                    source_data       = nullptr;
    size_t                          source_size       = 0;
    ggml_hrx_loom_jit_source_format source_format     = ggml_hrx_loom_jit_source_format::Text;
    const char *                    source_identifier = nullptr;
};

struct ggml_hrx_loom_jit_launch_config {
    std::array<uint32_t, 3> workgroup_count         = {};
    std::array<uint32_t, 3> workgroup_size          = {};
    uint32_t                subgroup_size           = 0;
    uint64_t                workgroup_storage_bytes = 0;
    size_t                  workload_argument_count = 0;
    uint32_t                fields                  = 0;
};

struct ggml_hrx_loom_jit_compile_options {
    const void *                             source_data             = nullptr;
    size_t                                   source_size             = 0;
    ggml_hrx_loom_jit_source_format          source_format           = ggml_hrx_loom_jit_source_format::Text;
    const char *                             source_identifier       = nullptr;
    const char *                             root_symbol             = nullptr;
    const char *                             launch_config_symbol    = nullptr;
    const char *                             module_name             = nullptr;
    const char *                             artifact_identifier     = nullptr;
    const ggml_hrx_loom_jit_source *         dependencies            = nullptr;
    size_t                                   dependency_count        = 0;
    const ggml_hrx_loom_jit_config_binding * config_bindings         = nullptr;
    size_t                                   config_binding_count    = 0;
    const int64_t *                          workload_arguments      = nullptr;
    size_t                                   workload_argument_count = 0;
    bool                                     evaluate_launch_config  = false;
};

struct ggml_hrx_loom_jit_compile_result {
    ggml_hrx_loom_jit_compile_result() = default;
    ~ggml_hrx_loom_jit_compile_result();
    ggml_hrx_loom_jit_compile_result(const ggml_hrx_loom_jit_compile_result &)             = delete;
    ggml_hrx_loom_jit_compile_result & operator=(const ggml_hrx_loom_jit_compile_result &) = delete;
    ggml_hrx_loom_jit_compile_result(ggml_hrx_loom_jit_compile_result && other) noexcept;
    ggml_hrx_loom_jit_compile_result & operator=(ggml_hrx_loom_jit_compile_result && other) noexcept;

    void reset();

    void *                          hsaco_data               = nullptr;
    size_t                          hsaco_size               = 0;
    char *                          manifest_json            = nullptr;
    size_t                          manifest_json_size       = 0;
    char *                          compile_report_json      = nullptr;
    size_t                          compile_report_json_size = 0;
    char *                          final_module_text        = nullptr;
    size_t                          final_module_text_size   = 0;
    ggml_hrx_loom_jit_launch_config launch_config;
};

hrx_status_t ggml_hrx_loom_jit_amdgpu_create(const ggml_hrx_loom_jit_amdgpu_options * options,
                                             ggml_hrx_loom_jit_amdgpu **              out_jit);

void ggml_hrx_loom_jit_amdgpu_release(ggml_hrx_loom_jit_amdgpu * jit);

hrx_status_t ggml_hrx_loom_jit_amdgpu_compile(ggml_hrx_loom_jit_amdgpu *                jit,
                                              const ggml_hrx_loom_jit_compile_options * options,
                                              ggml_hrx_loom_jit_compile_result *        out_result);
