//
// MIT license
// Copyright (C) 2026 Raudbjorn/ggml-llama.cpp contributors
// SPDX-License-Identifier: MIT
//

#include "xe-kmd.hpp"

#include "ggml-impl.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <system_error>

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
    // Explicit iterator: the range-for form throws from operator++ on an I/O
    // error, and this probe must never take down device discovery.
    fs::directory_iterator it(drm_sysfs_root, ec);
    if (ec) {
        return false;
    }
    for (; it != fs::directory_iterator(); it.increment(ec)) {
        if (ec) {
            return false;
        }
        const std::string name = it->path().filename().string();
        if (name.rfind("renderD", 0) != 0) {
            continue;
        }
        const fs::path dev = it->path() / "device";
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
            ec.clear();
        }
        return true;
    }
    return false;
#endif
}

static bool env_is_set(const char * v) {
    return v != nullptr && v[0] != '\0';
}

const char * ggml_sycl_xe_copy_engine_default(bool intel_on_xe, const char * ur_value, const char * pi_value) {
    if (!intel_on_xe) {
        return nullptr;
    }
    // An explicit user value always wins, including engine ranges such as "0:0".
    if (env_is_set(ur_value) || env_is_set(pi_value)) {
        return nullptr;
    }
    return "0";
}

const char * ggml_sycl_xe_copy_offload_default(bool intel_on_xe, const char * v2_value, const char * ur_value,
                                                const char * pi_value) {
    if (!intel_on_xe || env_is_set(v2_value) || env_is_set(ur_value) || env_is_set(pi_value)) {
        return nullptr;
    }
    return "1";
}

bool ggml_sycl_apply_xe_kmd_defaults_in(const std::string & drm_sysfs_root) {
#ifdef _WIN32
    GGML_UNUSED(drm_sysfs_root);
    return false;
#else
    std::string bdf;
    const bool  on_xe = ggml_sycl_intel_gpu_on_xe(drm_sysfs_root, &bdf);
    const char * v1 = ggml_sycl_xe_copy_engine_default(on_xe, getenv("UR_L0_USE_COPY_ENGINE"),
                                                       getenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE"));
    const char * v2 = ggml_sycl_xe_copy_offload_default(on_xe, getenv("UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD"),
                                                        getenv("UR_L0_USE_COPY_ENGINE"),
                                                        getenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE"));
    if (v1 == nullptr && v2 == nullptr) {
        return false;
    }
    // Copies on the copy engine hang or reset the bcs engine on DG2 with the xe
    // KMD: NEO 26.35 answers the failing userptr binds of read-only file mappings
    // with an eviction sweep that unbinds the blitter's command buffer while its
    // job is pending (docs/backend/SYCL.md, Known Issues). Keep copies on the
    // compute queue unless the user asked otherwise. overwrite=1 because the
    // decision above already treated an empty value as unset.
    if (v1 != nullptr) {
        setenv("UR_L0_USE_COPY_ENGINE", v1, 1);
    }
    if (v2 != nullptr) {
        setenv("UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD", v2, 1);
    }
    GGML_LOG_INFO("%s: Intel GPU %s is bound to the xe kernel driver: defaulting %s%s%s "
                  "(copies on the compute queue; blitter copies hang/reset on DG2, see docs/backend/SYCL.md). "
                  "Override with UR_L0_USE_COPY_ENGINE=1 (Level Zero v1 adapter) or "
                  "UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=0 (v2 adapter); that can bring the blitter failure back. "
                  "--prefetch-experts-slots needs the override.\n",
                  __func__, bdf.c_str(), v1 != nullptr ? "UR_L0_USE_COPY_ENGINE=0" : "",
                  (v1 != nullptr && v2 != nullptr) ? " and " : "",
                  v2 != nullptr ? "UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1" : "");
    return true;
#endif
}

void ggml_sycl_apply_xe_kmd_defaults() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_sycl_apply_xe_kmd_defaults_in("/sys/class/drm"); });
}
