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
bool ggml_sycl_intel_gpu_on_xe(const std::string & drm_sysfs_root, std::string * bdf);

// Value UR_L0_USE_COPY_ENGINE should be set to, or nullptr when the user's own
// setting (UR_L0_USE_COPY_ENGINE or its SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE alias)
// stands or no xe-bound Intel GPU was found.
const char * ggml_sycl_xe_copy_engine_default(bool intel_on_xe, const char * ur_value, const char * pi_value);

// Applies the defaults once per process. Must run before the first SYCL runtime
// call because the Level Zero adapter reads the variable at initialization.
void ggml_sycl_apply_xe_kmd_defaults();
