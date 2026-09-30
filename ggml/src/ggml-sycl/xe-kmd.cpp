//
// MIT license
// Copyright (C) 2026 Raudbjorn/ggml-llama.cpp contributors
// SPDX-License-Identifier: MIT
//

#include "xe-kmd.hpp"

#include "common.hpp"
#include "ggml-impl.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

static std::atomic<bool> g_xe_kmd_defaults_applied{ false };

// sysfs vocabulary and the driver this hook targets.
static constexpr const char * DRM_SYSFS_ROOT     = "/sys/class/drm";
static constexpr const char * RENDER_NODE_PREFIX = "renderD";
static constexpr const char * INTEL_PCI_VENDOR   = "0x8086";
static constexpr const char * XE_KMD_NAME        = "xe";
static constexpr uint16_t     DG2_PCI_ID_FIRST   = 0x5690;
static constexpr uint16_t     DG2_PCI_ID_LAST    = 0x56ff;

// Level Zero adapter variables that express copy-engine policy: the v1 names with their
// SYCL_PI aliases (all eight are in the v1 adapter binary), and the v2 adapter's own
// variable, whose sense is inverted (0 keeps copy offload on).
struct ggml_sycl_xe_copy_var {
    const char * ur;
    const char * pi;
};
static constexpr ggml_sycl_xe_copy_var V1_COPY_VARS[] = {
    { "UR_L0_USE_COPY_ENGINE", "SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE" },
    { "UR_L0_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE", "SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE" },
    { "UR_L0_USE_COPY_ENGINE_FOR_D2D_COPY", "SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_D2D_COPY" },
    { "UR_L0_USE_COPY_ENGINE_FOR_FILL", "SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_FILL" },
};
static constexpr const char * UR_COPY_ENGINE     = "UR_L0_USE_COPY_ENGINE";
static constexpr const char * PI_COPY_ENGINE     = "SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE";
static constexpr const char * UR_V2_NO_OFFLOAD   = "UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD";
static constexpr const char * GGML_XE_DEFAULT_SW = "GGML_SYCL_XE_COPY_ENGINE_DEFAULT";
static constexpr const char * COPY_ENGINE_OFF    = "0";  // value for UR_L0_USE_COPY_ENGINE
static constexpr const char * COPY_OFFLOAD_OFF   = "1";  // value for UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD

// Lines produced while the process is still being loaded. ggml_sycl_xe_kmd_flush_log()
// emits them from the backend initialization, through the logger the application has
// installed by then (--log-disable, --log-file and custom callbacks all apply).
struct ggml_sycl_xe_pending_line {
    ggml_log_level level;
    std::string    text;
};

static std::vector<ggml_sycl_xe_pending_line> & pending_log() {
    static std::vector<ggml_sycl_xe_pending_line> lines;
    return lines;
}

static std::string fmt(const char * format, ...) GGML_ATTRIBUTE_FORMAT(1, 2);

static std::string fmt(const char * format, ...) {
    char    buf[1536];
    va_list ap;
    va_start(ap, format);
    vsnprintf(buf, sizeof(buf), format, ap);
    va_end(ap);
    return buf;
}

static void log_later(ggml_log_level level, std::string text) {
    pending_log().push_back({ level, std::move(text) });
}

void ggml_sycl_xe_kmd_flush_log() {
    for (const auto & line : pending_log()) {
        ggml_log_internal(line.level, "%s", line.text.c_str());
    }
    pending_log().clear();
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
    for (const auto & var : V1_COPY_VARS) {
        for (const char * n : { var.ur, var.pi }) {
            const char * v = get_env(n);
            if (ggml_sycl_env_is_set(v) && !ggml_sycl_env_is_zero(v)) {
                return true;
            }
        }
    }
    return ggml_sycl_env_is_zero(get_env(UR_V2_NO_OFFLOAD));
}

bool ggml_sycl_xe_apply_default(bool dg2_on_xe, bool var_set, bool user_wants_copy_engine) {
    return dg2_on_xe && !var_set && !user_wants_copy_engine;
}

#ifndef _WIN32

static std::string read_first_token(const fs::path & p) {
    std::ifstream f(p);
    std::string   s;
    f >> s;
    return s;
}

// Walks the render nodes and returns the first Intel node bound to xe whose PCI device
// id passes accept(). A missing root (no DRM sysfs at all) is silent; I/O errors while
// reading an existing tree are logged at debug level.
static bool find_intel_gpu_on_xe(const std::string & root, bool (*accept)(uint16_t), std::string * bdf,
                                 uint16_t * device_id) {
    std::error_code ec;
    if (!fs::is_directory(root, ec)) {
        return false;
    }
    // Explicit iterator: the range-for form throws from operator++ on an I/O
    // error, and this probe must never take down device discovery.
    fs::directory_iterator it(root, ec);
    if (ec) {
        log_later(GGML_LOG_LEVEL_DEBUG, fmt("%s: cannot read %s: %s\n", __func__, root.c_str(), ec.message().c_str()));
        return false;
    }
    for (; it != fs::directory_iterator(); it.increment(ec)) {
        if (ec) {
            log_later(GGML_LOG_LEVEL_DEBUG,
                      fmt("%s: directory iteration failed in %s: %s\n", __func__, root.c_str(), ec.message().c_str()));
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
        const uint16_t id = (uint16_t) strtoul(read_first_token(dev / "device").c_str(), nullptr, 16);
        if (!accept(id)) {
            continue;
        }
        if (bdf != nullptr) {
            const fs::path target = fs::read_symlink(dev, ec);
            *bdf = ec ? name : target.filename().string();
            ec.clear();
        }
        if (device_id != nullptr) {
            *device_id = id;
        }
        return true;
    }
    return false;
}

static bool accept_any_device(uint16_t) {
    return true;
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
        log_later(GGML_LOG_LEVEL_WARN,
                  fmt("ggml_sycl_apply_xe_kmd_defaults: unset empty %s (the Level Zero adapter aborts on an empty "
                      "value)\n",
                      name));
    }
}

static const char * plain_getenv(const char * name) {
    return getenv(name);
}

// Name and value of the first copy-engine variable the user set, for the log line
// that says the defaults were left alone.
static const char * first_explicit_copy_var(std::string & value) {
    for (const auto & var : V1_COPY_VARS) {
        for (const char * n : { var.ur, var.pi }) {
            if (ggml_sycl_env_is_set(getenv(n))) {
                value = getenv(n);
                return n;
            }
        }
    }
    if (ggml_sycl_env_is_set(getenv(UR_V2_NO_OFFLOAD))) {
        value = getenv(UR_V2_NO_OFFLOAD);
        return UR_V2_NO_OFFLOAD;
    }
    return nullptr;
}

#endif  // _WIN32

bool ggml_sycl_intel_gpu_on_xe(const std::string & drm_sysfs_root, std::string * bdf, uint16_t * device_id) {
#ifdef _WIN32
    GGML_UNUSED(drm_sysfs_root);
    GGML_UNUSED(bdf);
    GGML_UNUSED(device_id);
    return false;
#else
    return find_intel_gpu_on_xe(drm_sysfs_root, accept_any_device, bdf, device_id);
#endif
}

bool ggml_sycl_dg2_gpu_on_xe(const std::string & drm_sysfs_root, std::string * bdf, uint16_t * device_id) {
#ifdef _WIN32
    GGML_UNUSED(drm_sysfs_root);
    GGML_UNUSED(bdf);
    GGML_UNUSED(device_id);
    return false;
#else
    return find_intel_gpu_on_xe(drm_sysfs_root, ggml_sycl_is_dg2, bdf, device_id);
#endif
}

bool ggml_sycl_apply_xe_kmd_defaults_in(const std::string & drm_sysfs_root) {
#ifdef _WIN32
    GGML_UNUSED(drm_sysfs_root);
    return false;
#else
    g_xe_kmd_defaults_applied.store(false);
    // Kill switch until a compute-runtime release carries the read-only userptr
    // retry (docs/research/patches/0001-neo-retry-userptr-bind-readonly-on-eperm.patch);
    // revisit the default when that happens.
    if (ggml_sycl_get_env(GGML_XE_DEFAULT_SW, 1) == 0) {
        log_later(GGML_LOG_LEVEL_DEBUG, fmt("%s: disabled by %s=0\n", __func__, GGML_XE_DEFAULT_SW));
        return false;
    }
    std::string bdf;
    uint16_t    device_id = 0;
    if (!ggml_sycl_dg2_gpu_on_xe(drm_sysfs_root, &bdf, &device_id)) {
        return false;  // any other GPU or driver: the environment is not touched
    }
    for (const auto & var : V1_COPY_VARS) {
        mirror_alias_into_empty_ur(var.ur, var.pi);
    }
    for (const auto & var : V1_COPY_VARS) {
        unset_if_empty(var.ur);
        unset_if_empty(var.pi);
    }
    unset_if_empty(UR_V2_NO_OFFLOAD);
    const bool wants  = ggml_sycl_xe_user_wants_copy_engine(plain_getenv);
    const bool v1_set = ggml_sycl_env_is_set(getenv(UR_COPY_ENGINE)) || ggml_sycl_env_is_set(getenv(PI_COPY_ENGINE));
    const bool v2_set = ggml_sycl_env_is_set(getenv(UR_V2_NO_OFFLOAD));
    const bool v1     = ggml_sycl_xe_apply_default(true, v1_set, wants);
    const bool v2     = ggml_sycl_xe_apply_default(true, v2_set, wants);
    if (!v1 && !v2) {
        std::string  value;
        const char * name = first_explicit_copy_var(value);
        log_later(GGML_LOG_LEVEL_INFO,
                  fmt("%s: DG2 GPU %s (0x%04x) on the xe kernel driver: copy engines left as configured by %s=%s; "
                      "the blitter failure described in docs/backend/SYCL.md can occur\n",
                      __func__, bdf.c_str(), device_id, name != nullptr ? name : "the environment", value.c_str()));
        return false;
    }
    // Copies on the copy engine hang or reset the bcs engine on DG2 with the xe
    // KMD: NEO 26.35 answers the failing userptr binds of read-only file mappings
    // with an eviction sweep that unbinds the blitter's command buffer while its
    // job is pending (docs/backend/SYCL.md, Known Issues). Keep copies on the
    // compute queue unless the user asked otherwise. Empty values were dropped
    // above, so overwrite=1 only ever replaces nothing.
    if (v1) {
        setenv(UR_COPY_ENGINE, COPY_ENGINE_OFF, 1);
    }
    if (v2) {
        setenv(UR_V2_NO_OFFLOAD, COPY_OFFLOAD_OFF, 1);
    }
    g_xe_kmd_defaults_applied.store(v1);
    log_later(GGML_LOG_LEVEL_INFO,
              fmt("%s: DG2 GPU %s (0x%04x) on the xe kernel driver: ggml-sycl set %s%s%s so copies stay on the "
                  "compute queue (blitter copies hang/reset, docs/backend/SYCL.md). GGML_SYCL_XE_COPY_ENGINE_DEFAULT=0 "
                  "restores the adapter defaults; UR_L0_USE_COPY_ENGINE=1 (v1 adapter, every copy engine) or "
                  "UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=0 (v2) re-enable copy engines and can bring the failure back. "
                  "--prefetch-experts-slots needs the v1 override and is refused on the v2 adapter regardless.\n",
                  __func__, bdf.c_str(), device_id, v1 ? "UR_L0_USE_COPY_ENGINE=0" : "", (v1 && v2) ? " and " : "",
                  v2 ? "UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1" : ""));
    return true;
#endif
}

void ggml_sycl_apply_xe_kmd_defaults() {
    static std::once_flag once;
    std::call_once(once, [] { ggml_sycl_apply_xe_kmd_defaults_in(DRM_SYSFS_ROOT); });
}

// Single point of application: runs when libggml-sycl is loaded (or before main for a
// static link), ahead of every SYCL entry point in this library, and cannot be
// re-invoked. Consequences: an application that initializes SYCL before loading
// ggml-sycl is out of reach; one that changes these variables after load must set
// UR_L0_USE_COPY_ENGINE itself (a FOR_* value set later is not seen); and a host that
// dlopen()s the backend while other threads call getenv() gets the usual setenv race.
namespace {
struct ggml_sycl_xe_kmd_loader {
    ggml_sycl_xe_kmd_loader() { ggml_sycl_apply_xe_kmd_defaults(); }
} const g_ggml_sycl_xe_kmd_loader [[maybe_unused]];
}  // namespace

bool ggml_sycl_xe_kmd_defaults_applied() {
    return g_xe_kmd_defaults_applied.load();
}
