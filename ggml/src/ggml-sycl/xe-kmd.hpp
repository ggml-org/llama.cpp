//
// MIT license
// Copyright (C) 2026 Raudbjorn/ggml-llama.cpp contributors
// SPDX-License-Identifier: MIT
//
// Process-wide runtime defaults for Intel GPUs bound to the xe kernel driver.
// No SYCL includes: the checks run before the SYCL runtime is initialized and
// are unit-tested on hosts without a GPU.

#pragma once

#include <string>

// True when an Intel (vendor 0x8086) render node under drm_sysfs_root is bound
// to the "xe" kernel driver. drm_sysfs_root is "/sys/class/drm" in production;
// tests pass a fake tree. bdf receives the PCI address of the first match.
// Never throws; any filesystem error reads as "not found".
bool ggml_sycl_intel_gpu_on_xe(const std::string & drm_sysfs_root, std::string * bdf);

// Value UR_L0_USE_COPY_ENGINE (Level Zero adapter v1) should be set to, or
// nullptr when a non-empty user setting (that variable or its
// SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE alias) stands or no xe-bound Intel GPU was
// found. An empty value counts as unset.
const char * ggml_sycl_xe_copy_engine_default(bool intel_on_xe, const char * ur_value, const char * pi_value);

// Same decision for the Level Zero v2 adapter, which ignores the variables
// above and keeps copies off the copy engines with
// UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1. An explicit v1 setting (either of the
// variables above) also disables this default: the user is managing copy-engine
// policy and a v1 override must not be silently undone on a v2 system.
const char * ggml_sycl_xe_copy_offload_default(bool intel_on_xe, const char * v2_value, const char * ur_value,
                                                const char * pi_value);

// Applies both defaults to the process environment for the given sysfs root.
// Exposed for the unit test; production code calls the once-only wrapper.
// Returns true when at least one variable was set.
bool ggml_sycl_apply_xe_kmd_defaults_in(const std::string & drm_sysfs_root);

// Applies the defaults once per process. Must run before the first SYCL runtime
// call because the Level Zero adapter reads the variables at initialization.
void ggml_sycl_apply_xe_kmd_defaults();
