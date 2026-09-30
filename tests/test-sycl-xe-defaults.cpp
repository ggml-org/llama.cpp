// CPU-only test for the xe KMD runtime defaults (ggml/src/ggml-sycl/xe-kmd.cpp).
// Builds a fake /sys/class/drm tree; never touches the SYCL runtime. Registered
// on Linux only (tests/CMakeLists.txt): the probe is a stub elsewhere.

#include "ggml-sycl/xe-kmd.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

#include <unistd.h>

namespace fs = std::filesystem;

static int g_failures = 0;

#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            g_failures++;                                                      \
        }                                                                      \
    } while (0)

// <root>/renderD<idx>/device -> <root>/devices/<bdf>/ with vendor and driver link.
static void add_node(const fs::path & root, int idx, const char * bdf, const char * vendor, const char * driver) {
    const fs::path devdir = root / "devices" / bdf;
    fs::create_directories(devdir);
    std::ofstream(devdir / "vendor") << vendor << "\n";
    if (driver != nullptr) {
        const fs::path drvdir = root / "drivers" / driver;
        fs::create_directories(drvdir);
        fs::create_directory_symlink(drvdir, devdir / "driver");
    }
    const fs::path node = root / ("renderD" + std::to_string(idx));
    fs::create_directories(node);
    fs::create_directory_symlink(devdir, node / "device");
}

int main() {
    const fs::path base = fs::temp_directory_path() / ("ggml-xe-kmd-test-" + std::to_string(getpid()));
    fs::remove_all(base);

    std::string bdf;

    // Missing root.
    CHECK(!ggml_sycl_intel_gpu_on_xe((base / "missing").string(), &bdf));

    // Intel on i915 only.
    {
        const fs::path root = base / "i915";
        add_node(root, 128, "0000:03:00.0", "0x8086", "i915");
        CHECK(!ggml_sycl_intel_gpu_on_xe(root.string(), &bdf));
    }
    // Non-Intel on xe (never happens, but the vendor gate must hold).
    {
        const fs::path root = base / "amd";
        add_node(root, 128, "0000:0d:00.0", "0x1002", "xe");
        CHECK(!ggml_sycl_intel_gpu_on_xe(root.string(), &bdf));
    }
    // Intel node without a driver link (unbound device).
    {
        const fs::path root = base / "unbound";
        add_node(root, 128, "0000:03:00.0", "0x8086", nullptr);
        CHECK(!ggml_sycl_intel_gpu_on_xe(root.string(), &bdf));
    }
    // AMD iGPU on amdgpu plus Intel dGPU on xe: found, with the right address.
    {
        const fs::path root = base / "mixed";
        add_node(root, 128, "0000:0d:00.0", "0x1002", "amdgpu");
        add_node(root, 129, "0000:03:00.0", "0x8086", "xe");
        bdf.clear();
        CHECK(ggml_sycl_intel_gpu_on_xe(root.string(), &bdf));
        CHECK(bdf == "0000:03:00.0");
        CHECK(ggml_sycl_intel_gpu_on_xe(root.string(), nullptr));
    }

    // Env decision: only an unset user value on xe yields "0".
    const char * v = ggml_sycl_xe_copy_engine_default(true, nullptr, nullptr);
    CHECK(v != nullptr && strcmp(v, "0") == 0);
    CHECK(ggml_sycl_xe_copy_engine_default(true, "1", nullptr) == nullptr);
    CHECK(ggml_sycl_xe_copy_engine_default(true, "0:0", nullptr) == nullptr);
    CHECK(ggml_sycl_xe_copy_engine_default(true, nullptr, "1") == nullptr);
    CHECK(ggml_sycl_xe_copy_engine_default(true, "", "") != nullptr);
    CHECK(ggml_sycl_xe_copy_engine_default(false, nullptr, nullptr) == nullptr);
    const char * o = ggml_sycl_xe_copy_offload_default(true, nullptr);
    CHECK(o != nullptr && strcmp(o, "1") == 0);
    CHECK(ggml_sycl_xe_copy_offload_default(true, "0") == nullptr);
    CHECK(ggml_sycl_xe_copy_offload_default(false, nullptr) == nullptr);

    // Applying against the fake trees must change the process environment as decided,
    // including an empty existing value, and must leave explicit values alone.
    const std::string xe_root = (base / "mixed").string();
    const std::string i915_root = (base / "i915").string();
    setenv("UR_L0_USE_COPY_ENGINE", "", 1);
    unsetenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE");
    unsetenv("UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD");
    CHECK(ggml_sycl_apply_xe_kmd_defaults_in(xe_root));
    CHECK(getenv("UR_L0_USE_COPY_ENGINE") != nullptr && strcmp(getenv("UR_L0_USE_COPY_ENGINE"), "0") == 0);
    CHECK(getenv("UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD") != nullptr &&
          strcmp(getenv("UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD"), "1") == 0);
    setenv("UR_L0_USE_COPY_ENGINE", "1", 1);
    setenv("UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD", "0", 1);
    CHECK(!ggml_sycl_apply_xe_kmd_defaults_in(xe_root));
    CHECK(strcmp(getenv("UR_L0_USE_COPY_ENGINE"), "1") == 0);
    CHECK(strcmp(getenv("UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD"), "0") == 0);
    unsetenv("UR_L0_USE_COPY_ENGINE");
    unsetenv("UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD");
    CHECK(!ggml_sycl_apply_xe_kmd_defaults_in(i915_root));
    CHECK(getenv("UR_L0_USE_COPY_ENGINE") == nullptr);
    CHECK(getenv("UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD") == nullptr);

    fs::remove_all(base);

    if (g_failures == 0) {
        printf("test-sycl-xe-defaults: OK\n");
        return 0;
    }
    printf("test-sycl-xe-defaults: %d failure(s)\n", g_failures);
    return 1;
}
