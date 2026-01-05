//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

// moe-xmx-fused.hpp - Fused XMX MoE GEMM kernel with persistent work-groups
#pragma once

#include "common.hpp"
#include "moe-xmx.hpp"  // For MoEXMXConfig and preprocessing
#include <sycl/sycl.hpp>

#if SYCL_XMX_MOE_AVAILABLE

namespace moe_xmx_fused {

using namespace sycl::ext::oneapi::experimental::matrix;

// Fused kernel configuration
struct FusedMoEConfig {
    int num_persistent_wgs = 0;       // nsm * 2 (from device info)
    int wg_size = 256;                // 256 default
    int tiles_m = 4;                  // 4 (from XMXCapabilities)
    int tiles_n = 4;                  // 4 (from XMXCapabilities)
    size_t slm_size = 65536;          // Device SLM budget

    static FusedMoEConfig from_device(int device_id) {
        const auto& dev_info = ggml_sycl_info().devices[device_id];
        const auto& xmx = dev_info.xmx_caps;

        FusedMoEConfig cfg;
        cfg.num_persistent_wgs = dev_info.nsm * 2;  // 2 WGs per XeCore
        cfg.wg_size = std::min(256, ggml_sycl_info().max_work_group_sizes[device_id]);
        cfg.tiles_m = xmx.optimal_tiles_m > 0 ? xmx.optimal_tiles_m : 4;
        cfg.tiles_n = xmx.optimal_tiles_n > 0 ? xmx.optimal_tiles_n : 4;
        cfg.slm_size = xmx.slm_size > 0 ? xmx.slm_size : 65536;
        return cfg;
    }
};

} // namespace moe_xmx_fused

#endif // SYCL_XMX_MOE_AVAILABLE
