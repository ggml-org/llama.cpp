#pragma once

#include "hrx_runtime.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// Shapes go to kernels as launch scalars. Thus each kernel and config pair compiles only one time.

struct ggml_hrx_loom_jit;
struct ggml_hrx_loom_launch_program;

struct ggml_hrx_loom_source {
    const void * data       = nullptr;
    size_t       size       = 0;
    const char * identifier = nullptr;
};

struct ggml_hrx_loom_config_binding {
    const char * key   = nullptr;
    const char * value = nullptr;
};

struct ggml_hrx_loom_compile_options {
    ggml_hrx_loom_source                 source;
    const ggml_hrx_loom_source *         deps     = nullptr;
    size_t                               n_deps   = 0;
    const ggml_hrx_loom_config_binding * config   = nullptr;
    size_t                               n_config = 0;
    const char *                         symbol   = nullptr;
};

struct ggml_hrx_loom_launch_config {
    uint32_t workgroup_count[3] = { 0, 0, 0 };
    uint32_t workgroup_size[3]  = { 0, 0, 0 };
    uint32_t subgroup_size      = 0;
};

struct ggml_hrx_loom_compile_result {
    std::vector<uint8_t>           hsaco;
    ggml_hrx_loom_launch_program * launch = nullptr; // On success, the caller owns this.
};

// processor: the device architecture. kernel_target: the AMDGPU target of @ggml_wave32 and @ggml_wave64.
hrx_status_t ggml_hrx_loom_jit_create(const char * processor, const char * kernel_target, ggml_hrx_loom_jit ** out_jit);
void         ggml_hrx_loom_jit_free(ggml_hrx_loom_jit * jit);

// Not thread safe. Callers must lock each jit.
hrx_status_t ggml_hrx_loom_compile(ggml_hrx_loom_jit *                   jit,
                                   const ggml_hrx_loom_compile_options * options,
                                   ggml_hrx_loom_compile_result *        out_result);

void         ggml_hrx_loom_launch_program_free(ggml_hrx_loom_launch_program * program);
hrx_status_t ggml_hrx_loom_launch_program_eval(ggml_hrx_loom_launch_program * program,
                                               const int64_t *                workload,
                                               size_t                         n_workload,
                                               ggml_hrx_loom_launch_config *  out_config);
