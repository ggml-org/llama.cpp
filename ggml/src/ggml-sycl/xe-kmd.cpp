//
// MIT license
// Copyright (C) 2026 Raudbjorn/ggml-llama.cpp contributors
// SPDX-License-Identifier: MIT
//

#include "xe-kmd.hpp"

#include "ggml-impl.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>

namespace fs = std::filesystem;

static std::string read_first_token(const fs::path & p) {
    std::ifstream f(p);
    std::string   s;
    f >> s;
    return s;
}

bool ggml_sycl_intel_gpu_on_xe(const std::string & drm_sysfs_root, std::string * bdf) {
#ifdef _WIN32
    GGML_UNUSED(drm_sysfs_root);
    GGML_UNUSED(bdf);
    return false;
#else
    std::error_code ec;
    if (!fs::is_directory(drm_sysfs_root, ec)) {
        return false;
    }
    for (const auto & entry : fs::directory_iterator(drm_sysfs_root, ec)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("renderD", 0) != 0) {
            continue;
        }
        const fs::path dev = entry.path() / "device";
        if (read_first_token(dev / "vendor") != "0x8086") {
            continue;
        }
        const fs::path drv = fs::read_symlink(dev / "driver", ec);
        if (ec || drv.filename().string() != "xe") {
            ec.clear();
            continue;
        }
        if (bdf != nullptr) {
            const fs::path target = fs::read_symlink(dev, ec);
            *bdf = ec ? name : target.filename().string();
        }
        return true;
    }
    return false;
#endif
}

const char * ggml_sycl_xe_copy_engine_default(bool intel_on_xe, const char * ur_value, const char * pi_value) {
    if (!intel_on_xe) {
        return nullptr;
    }
    // An explicit user value always wins, including engine ranges such as "0:0".
    if ((ur_value != nullptr && ur_value[0] != '\0') || (pi_value != nullptr && pi_value[0] != '\0')) {
        return nullptr;
    }
    return "0";
}

void ggml_sycl_apply_xe_kmd_defaults() {
    static std::once_flag once;
    std::call_once(once, [] {
#ifndef _WIN32
        std::string bdf;
        const bool  on_xe = ggml_sycl_intel_gpu_on_xe("/sys/class/drm", &bdf);
        const char * value = ggml_sycl_xe_copy_engine_default(on_xe, getenv("UR_L0_USE_COPY_ENGINE"),
                                                              getenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE"));
        if (value == nullptr) {
            return;
        }
        // Copies on the blitter queue hang or reset the bcs engine on DG2 with the
        // xe KMD (NEO 26.35 unbinds the blitter's command buffer while its job is
        // pending; docs/backend/SYCL.md, Known Issues). Route copies to the compute
        // queue unless the user asked otherwise.
        setenv("UR_L0_USE_COPY_ENGINE", value, 0);
        GGML_LOG_INFO("%s: Intel GPU %s is bound to the xe kernel driver: defaulting UR_L0_USE_COPY_ENGINE=%s "
                      "(copies on the compute queue; blitter copies hang/reset on DG2, see docs/backend/SYCL.md). "
                      "Set UR_L0_USE_COPY_ENGINE=1 to override; --prefetch-experts-slots needs that override.\n",
                      __func__, bdf.c_str(), value);
#endif
    });
}
