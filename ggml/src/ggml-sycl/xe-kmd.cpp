//
// MIT license
// Copyright (C) 2026 Raudbjorn/ggml-llama.cpp contributors
// SPDX-License-Identifier: MIT
//

#include "xe-kmd.hpp"

#include "ggml-impl.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

static std::atomic<bool> g_xe_kmd_defaults_applied{ false };

// sysfs vocabulary and the driver this hook targets.
static constexpr const char * DRM_SYSFS_ROOT      = "/sys/class/drm";
static constexpr const char * RENDER_NODE_PREFIX  = "renderD";
static constexpr const char * INTEL_PCI_VENDOR    = "0x8086";
static constexpr const char * XE_KMD_NAME         = "xe";
static constexpr uint16_t     DG2_PCI_ID_FIRST    = 0x5690;
static constexpr uint16_t     DG2_PCI_ID_LAST     = 0x56ff;
// Level Zero adapter variables (v1 name, SYCL_PI alias) that express copy-engine policy.
static constexpr const char * UR_COPY_ENGINE      = "UR_L0_USE_COPY_ENGINE";
static constexpr const char * PI_COPY_ENGINE      = "SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE";
static constexpr const char * UR_COPY_ENGINE_INO  = "UR_L0_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE";
static constexpr const char * PI_COPY_ENGINE_INO  = "SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE";
static constexpr const char * UR_COPY_ENGINE_D2D  = "UR_L0_USE_COPY_ENGINE_FOR_D2D_COPY";
static constexpr const char * UR_COPY_ENGINE_FILL = "UR_L0_USE_COPY_ENGINE_FOR_FILL";
static constexpr const char * UR_V2_NO_OFFLOAD    = "UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD";
static constexpr const char * GGML_XE_DEFAULT_SW  = "GGML_SYCL_XE_COPY_ENGINE_DEFAULT";

static std::string read_first_token(const fs::path & p) {
    std::ifstream f(p);
    std::string   s;
    f >> s;
    return s;
}

bool ggml_sycl_intel_gpu_on_xe(const std::string & drm_sysfs_root, std::string * bdf, uint16_t * device_id) {
#ifdef _WIN32
    GGML_UNUSED(drm_sysfs_root);
    GGML_UNUSED(bdf);
    GGML_UNUSED(device_id);
    return false;
#else
    std::error_code ec;
    if (!fs::is_directory(drm_sysfs_root, ec)) {
        GGML_LOG_DEBUG("%s: %s is not a directory (%s); no xe probe\n", __func__, drm_sysfs_root.c_str(),
                       ec ? ec.message().c_str() : "absent");
        return false;
    }
    // Explicit iterator: the range-for form throws from operator++ on an I/O
    // error, and this probe must never take down device discovery.
    fs::directory_iterator it(drm_sysfs_root, ec);
    if (ec) {
        GGML_LOG_DEBUG("%s: cannot read %s: %s\n", __func__, drm_sysfs_root.c_str(), ec.message().c_str());
        return false;
    }
    for (; it != fs::directory_iterator(); it.increment(ec)) {
        if (ec) {
            GGML_LOG_DEBUG("%s: directory iteration failed in %s: %s\n", __func__, drm_sysfs_root.c_str(),
                           ec.message().c_str());
            return false;
        }
        const std::string name = it->path().filename().string();
        if (name.rfind(RENDER_NODE_PREFIX, 0) != 0) {
            continue;
        }
        const fs::path dev = it->path() / "device";
        if (read_first_token(dev / "vendor") != INTEL_PCI_VENDOR) {
            continue;
        }
        const fs::path drv = fs::read_symlink(dev / "driver", ec);
        if (ec || drv.filename().string() != XE_KMD_NAME) {
            ec.clear();
            continue;
        }
        if (bdf != nullptr) {
            const fs::path target = fs::read_symlink(dev, ec);
            *bdf = ec ? name : target.filename().string();
            ec.clear();
        }
        if (device_id != nullptr) {
            *device_id = (uint16_t) strtoul(read_first_token(dev / "device").c_str(), nullptr, 16);
        }
        return true;
    }
    return false;
#endif
}

bool ggml_sycl_is_dg2(uint16_t device_id) {
    return device_id >= DG2_PCI_ID_FIRST && device_id <= DG2_PCI_ID_LAST;
}

bool ggml_sycl_env_is_set(const char * v) {
    return v != nullptr && v[0] != '\0';
}

bool ggml_sycl_env_is_zero(const char * v) {
    return v != nullptr && strcmp(v, "0") == 0;
}

bool ggml_sycl_xe_user_wants_copy_engine(const char * (*get_env)(const char *)) {
    for (const char * n : { UR_COPY_ENGINE, PI_COPY_ENGINE, UR_COPY_ENGINE_INO, PI_COPY_ENGINE_INO,
                            UR_COPY_ENGINE_D2D, UR_COPY_ENGINE_FILL }) {
        const char * v = get_env(n);
        if (ggml_sycl_env_is_set(v) && !ggml_sycl_env_is_zero(v)) {
            return true;
        }
    }
    // The v2 variable has the opposite sense: 0 keeps copy offload on.
    return ggml_sycl_env_is_zero(get_env(UR_V2_NO_OFFLOAD));
}

const char * ggml_sycl_xe_copy_engine_default(bool dg2_on_xe, bool v1_set, bool user_wants_copy_engine) {
    return (dg2_on_xe && !v1_set && !user_wants_copy_engine) ? "0" : nullptr;
}

const char * ggml_sycl_xe_copy_offload_default(bool dg2_on_xe, bool v2_set, bool user_wants_copy_engine) {
    return (dg2_on_xe && !v2_set && !user_wants_copy_engine) ? "1" : nullptr;
}

// The adapter prefers the UR name over its SYCL_PI alias and parses it with stoi(), so a
// present-but-empty UR variable next to a non-empty alias throws inside the adapter and
// leaves no GPU. Copy the alias value into the empty UR variable in that case.
static void mirror_alias_into_empty_ur(const char * ur_name, const char * pi_name) {
    const char * ur = getenv(ur_name);
    const char * pi = getenv(pi_name);
    if (ur != nullptr && ur[0] == '\0' && ggml_sycl_env_is_set(pi)) {
        setenv(ur_name, pi, 1);
    }
}

// The v1 adapter parses the FOR_* variables with stoi() in a static initializer, so an
// empty value aborts the process when the adapter is loaded; UR_L0_USE_COPY_ENGINE is
// parsed the same way at the first queue creation. Empty means unset to this hook, so
// drop such values instead of leaving the abort in place.
static void unset_if_empty(const char * name) {
    const char * v = getenv(name);
    if (v != nullptr && v[0] == '\0') {
        unsetenv(name);
        GGML_LOG_WARN("%s: unset empty %s (the Level Zero adapter aborts on an empty value)\n", __func__, name);
    }
}

static const char * plain_getenv(const char * name) {
    return getenv(name);
}

bool ggml_sycl_apply_xe_kmd_defaults_in(const std::string & drm_sysfs_root) {
#ifdef _WIN32
    GGML_UNUSED(drm_sysfs_root);
    return false;
#else
    // Kill switch until a compute-runtime release carries the read-only userptr
    // retry (docs/research/patches/0001-neo-retry-userptr-bind-readonly-on-eperm.patch);
    // revisit the default when that happens.
    g_xe_kmd_defaults_applied.store(false);
    if (ggml_sycl_env_is_zero(getenv(GGML_XE_DEFAULT_SW))) {
        return false;
    }
    mirror_alias_into_empty_ur(UR_COPY_ENGINE, PI_COPY_ENGINE);
    mirror_alias_into_empty_ur(UR_COPY_ENGINE_INO, PI_COPY_ENGINE_INO);
    for (const char * n : { UR_COPY_ENGINE, PI_COPY_ENGINE, UR_COPY_ENGINE_INO, PI_COPY_ENGINE_INO,
                            UR_COPY_ENGINE_D2D, UR_COPY_ENGINE_FILL, UR_V2_NO_OFFLOAD }) {
        unset_if_empty(n);
    }
    std::string bdf;
    uint16_t    device_id = 0;
    const bool  dg2_on_xe = ggml_sycl_intel_gpu_on_xe(drm_sysfs_root, &bdf, &device_id) && ggml_sycl_is_dg2(device_id);
    const bool  wants     = ggml_sycl_xe_user_wants_copy_engine(plain_getenv);
    const bool  v1_set    = ggml_sycl_env_is_set(getenv(UR_COPY_ENGINE)) || ggml_sycl_env_is_set(getenv(PI_COPY_ENGINE));
    const bool  v2_set    = ggml_sycl_env_is_set(getenv(UR_V2_NO_OFFLOAD));
    const char * v1 = ggml_sycl_xe_copy_engine_default(dg2_on_xe, v1_set, wants);
    const char * v2 = ggml_sycl_xe_copy_offload_default(dg2_on_xe, v2_set, wants);
    if (v1 == nullptr && v2 == nullptr) {
        return false;
    }
    // Copies on the copy engine hang or reset the bcs engine on DG2 with the xe
    // KMD: NEO 26.35 answers the failing userptr binds of read-only file mappings
    // with an eviction sweep that unbinds the blitter's command buffer while its
    // job is pending (docs/backend/SYCL.md, Known Issues). Keep copies on the
    // compute queue unless the user asked otherwise. Empty values were dropped
    // above, so overwrite=1 only ever replaces nothing.
    //
    // setenv() is not thread-safe against concurrent getenv() in other threads;
    // this runs from the first SYCL initialization, normally at program start,
    // and child processes inherit the values.
    if (v1 != nullptr) {
        setenv(UR_COPY_ENGINE, v1, 1);
    }
    if (v2 != nullptr) {
        setenv(UR_V2_NO_OFFLOAD, v2, 1);
    }
    g_xe_kmd_defaults_applied.store(v1 != nullptr);
    GGML_LOG_INFO("%s: DG2 GPU %s (0x%04x) on the xe kernel driver: ggml-sycl set %s%s%s so copies stay on the "
                  "compute queue (blitter copies hang/reset, docs/backend/SYCL.md). GGML_SYCL_XE_COPY_ENGINE_DEFAULT=0 "
                  "restores the adapter defaults; UR_L0_USE_COPY_ENGINE=1 (v1 adapter) or "
                  "UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=0 (v2) re-enable copy engines and can bring the failure back. "
                  "--prefetch-experts-slots needs the v1 override and is refused on the v2 adapter regardless.\n",
                  __func__, bdf.c_str(), device_id, v1 != nullptr ? "UR_L0_USE_COPY_ENGINE=0" : "",
                  (v1 != nullptr && v2 != nullptr) ? " and " : "",
                  v2 != nullptr ? "UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1" : "");
    return true;
#endif
}

void ggml_sycl_apply_xe_kmd_defaults() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_sycl_apply_xe_kmd_defaults_in(DRM_SYSFS_ROOT); });
}

// Single point of application: runs when libggml-sycl is loaded (or before main for a
// static link), ahead of every SYCL entry point in this library. ggml-sycl has no
// static-storage SYCL objects, so nothing in this library can initialize the runtime
// earlier. An application that initializes SYCL before loading ggml-sycl is out of reach.
namespace {
struct ggml_sycl_xe_kmd_loader {
    ggml_sycl_xe_kmd_loader() { ggml_sycl_apply_xe_kmd_defaults(); }
} const g_ggml_sycl_xe_kmd_loader [[maybe_unused]];
}  // namespace

bool ggml_sycl_xe_kmd_defaults_applied() {
    return g_xe_kmd_defaults_applied.load();
}
