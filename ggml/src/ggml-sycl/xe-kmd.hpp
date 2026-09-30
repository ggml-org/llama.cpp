//
// MIT license
// Copyright (C) 2026 Raudbjorn/ggml-llama.cpp contributors
// SPDX-License-Identifier: MIT
//
// Process-wide runtime defaults for DG2 (Arc A-series) GPUs bound to the xe kernel
// driver. No SYCL includes: the checks run before the SYCL runtime is initialized
// and are unit-tested on hosts without a GPU.

#pragma once

#include <cstdint>
#include <string>

// True when an Intel (vendor 0x8086) render node under drm_sysfs_root is bound to
// the "xe" kernel driver. drm_sysfs_root is "/sys/class/drm" in production; tests
// pass a fake tree. bdf receives the PCI address and device_id the PCI device id of
// the first match. Never throws; any filesystem error reads as "not found".
bool ggml_sycl_intel_gpu_on_xe(const std::string & drm_sysfs_root, std::string * bdf, uint16_t * device_id);

// DG2 / Alchemist (Arc A-series, Flex 140/170): PCI device ids 0x5690-0x56ff. The
// blitter defect was established on this family only.
bool ggml_sycl_is_dg2(uint16_t device_id);

// True when the variable is present with a non-empty value. Empty counts as unset.
bool ggml_sycl_env_is_set(const char * value);
// True when the variable is present and its value is exactly "0". Shared with the
// private copy-queue check in ggml-sycl.cpp so both sides agree on what "off" means.
bool ggml_sycl_env_is_zero(const char * value);

// True when the user asked for copy engines: any Unified Runtime Level Zero v1 variable
// (UR_L0_USE_COPY_ENGINE, the UR_L0_USE_COPY_ENGINE_FOR_* family, the SYCL_PI aliases)
// with a value other than "0", or UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=0. Values that
// agree with the defaults below do not count; empty values count as unset.
bool ggml_sycl_xe_user_wants_copy_engine(const char * (*get_env)(const char *));

// Value UR_L0_USE_COPY_ENGINE (v1 adapter) should be set to, or nullptr when the
// variable (or its SYCL_PI alias) is already set or the user wants copy engines.
const char * ggml_sycl_xe_copy_engine_default(bool dg2_on_xe, bool v1_set, bool user_wants_copy_engine);

// Value UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD (v2 adapter) should be set to, or nullptr
// when that variable is already set or the user wants copy engines.
const char * ggml_sycl_xe_copy_offload_default(bool dg2_on_xe, bool v2_set, bool user_wants_copy_engine);

// Applies both defaults to the process environment for the given sysfs root unless
// GGML_SYCL_XE_COPY_ENGINE_DEFAULT=0. Exposed for the unit test; production code
// calls the once-only wrapper. Returns true when at least one variable was set.
bool ggml_sycl_apply_xe_kmd_defaults_in(const std::string & drm_sysfs_root);

// Applies the defaults once per process. Must run before the first SYCL runtime
// call because the Level Zero adapters read the variables at initialization.
void ggml_sycl_apply_xe_kmd_defaults();

// True when the hook above set UR_L0_USE_COPY_ENGINE in this process (as opposed to
// the user), so the private copy-queue diagnostic can say where the setting came from.
bool ggml_sycl_xe_kmd_defaults_applied();
