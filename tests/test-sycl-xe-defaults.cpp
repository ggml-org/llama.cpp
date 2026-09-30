// CPU-only test for the xe KMD runtime defaults (ggml/src/ggml-sycl/xe-kmd.cpp).
// Builds a fake /sys/class/drm tree; never touches the SYCL runtime. Registered
// on Linux only (tests/CMakeLists.txt): the probe is a stub elsewhere.

#include "ggml-sycl/xe-kmd.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
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

// <root>/renderD<idx>/device -> <root>/devices/<bdf>/ with vendor, device id and driver link.
static void add_node(const fs::path & root, int idx, const char * bdf, const char * vendor, const char * device,
                     const char * driver) {
    const fs::path devdir = root / "devices" / bdf;
    fs::create_directories(devdir);
    std::ofstream(devdir / "vendor") << vendor << "\n";
    std::ofstream(devdir / "device") << device << "\n";
    if (driver != nullptr) {
        const fs::path drvdir = root / "drivers" / driver;
        fs::create_directories(drvdir);
        fs::create_directory_symlink(drvdir, devdir / "driver");
    }
    const fs::path node = root / ("renderD" + std::to_string(idx));
    fs::create_directories(node);
    fs::create_directory_symlink(devdir, node / "device");
}

static std::map<std::string, std::string> g_fake_env;

static const char * fake_getenv(const char * name) {
    auto it = g_fake_env.find(name);
    return it == g_fake_env.end() ? nullptr : it->second.c_str();
}

static const char * env(const char * name) {
    return getenv(name);
}

static void clear_copy_env() {
    for (const char * n : { "UR_L0_USE_COPY_ENGINE", "UR_L0_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE",
                            "UR_L0_USE_COPY_ENGINE_FOR_D2D_COPY", "UR_L0_USE_COPY_ENGINE_FOR_FILL",
                            "UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD", "SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE",
                            "SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE",
                            "SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_D2D_COPY", "SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_FILL",
                            "GGML_SYCL_XE_COPY_ENGINE_DEFAULT" }) {
        unsetenv(n);
    }
}

int main() {
    const fs::path base = fs::temp_directory_path() / ("ggml-xe-kmd-test-" + std::to_string(getpid()));
    fs::remove_all(base);

    std::string bdf;
    uint16_t    id = 0;

    // Missing root.
    CHECK(!ggml_sycl_intel_gpu_on_xe((base / "missing").string(), &bdf, &id));

    // Intel DG2 on i915 only.
    {
        const fs::path root = base / "i915";
        add_node(root, 128, "0000:03:00.0", "0x8086", "0x56a0", "i915");
        CHECK(!ggml_sycl_intel_gpu_on_xe(root.string(), &bdf, &id));
    }
    // Non-Intel on xe (never happens, but the vendor gate must hold).
    {
        const fs::path root = base / "amd";
        add_node(root, 128, "0000:0d:00.0", "0x1002", "0x164e", "xe");
        CHECK(!ggml_sycl_intel_gpu_on_xe(root.string(), &bdf, &id));
    }
    // Intel node without a driver link (unbound device).
    {
        const fs::path root = base / "unbound";
        add_node(root, 128, "0000:03:00.0", "0x8086", "0x56a0", nullptr);
        CHECK(!ggml_sycl_intel_gpu_on_xe(root.string(), &bdf, &id));
    }
    // AMD iGPU on amdgpu plus Intel DG2 on xe: found, with address and device id.
    {
        const fs::path root = base / "mixed";
        add_node(root, 128, "0000:0d:00.0", "0x1002", "0x164e", "amdgpu");
        add_node(root, 129, "0000:03:00.0", "0x8086", "0x56a0", "xe");
        bdf.clear();
        id = 0;
        CHECK(ggml_sycl_intel_gpu_on_xe(root.string(), &bdf, &id));
        CHECK(bdf == "0000:03:00.0");
        CHECK(id == 0x56a0);
        CHECK(ggml_sycl_intel_gpu_on_xe(root.string(), nullptr, nullptr));
    }
    // Battlemage on xe: found as Intel-on-xe, but not DG2.
    {
        const fs::path root = base / "bmg";
        add_node(root, 128, "0000:03:00.0", "0x8086", "0xe20b", "xe");
        id = 0;
        CHECK(ggml_sycl_intel_gpu_on_xe(root.string(), nullptr, &id));
        CHECK(id == 0xe20b);
        CHECK(!ggml_sycl_is_dg2(id));
        CHECK(!ggml_sycl_dg2_gpu_on_xe(root.string(), nullptr, nullptr));
    }
    // Two Intel nodes on xe in either order: the DG2 is found whichever is listed first.
    for (const bool dg2_first : { false, true }) {
        const fs::path root = base / (dg2_first ? "dg2-first" : "bmg-first");
        add_node(root, dg2_first ? 128 : 129, "0000:03:00.0", "0x8086", "0x56a0", "xe");
        add_node(root, dg2_first ? 129 : 128, "0000:0a:00.0", "0x8086", "0xe20b", "xe");
        bdf.clear();
        id = 0;
        CHECK(ggml_sycl_dg2_gpu_on_xe(root.string(), &bdf, &id));
        CHECK(bdf == "0000:03:00.0");
        CHECK(id == 0x56a0);
        CHECK(ggml_sycl_intel_gpu_on_xe(root.string(), nullptr, nullptr));
    }

    // DG2 id range: A770/A750/A380 desktop, A770M/A730M mobile, Flex 170/140.
    CHECK(ggml_sycl_is_dg2(0x56a0));
    CHECK(ggml_sycl_is_dg2(0x5690));
    CHECK(ggml_sycl_is_dg2(0x56c0));
    CHECK(!ggml_sycl_is_dg2(0x4680));  // Alder Lake iGPU
    CHECK(!ggml_sycl_is_dg2(0x7d55));  // Meteor Lake
    CHECK(!ggml_sycl_is_dg2(0x0bd5));  // PVC

    // Copy-engine policy: a value other than "0" on a v1-family variable, or 0 on the v2
    // variable, means the user wants copy engines. "0" on v1 agrees with the default and
    // does not count; empty counts as unset.
    g_fake_env.clear();
    CHECK(!ggml_sycl_xe_user_wants_copy_engine(fake_getenv));
    g_fake_env["UR_L0_USE_COPY_ENGINE"] = "";
    CHECK(!ggml_sycl_xe_user_wants_copy_engine(fake_getenv));
    g_fake_env["UR_L0_USE_COPY_ENGINE"] = "0";
    CHECK(!ggml_sycl_xe_user_wants_copy_engine(fake_getenv));
    g_fake_env["UR_L0_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE"] = "1";
    CHECK(ggml_sycl_xe_user_wants_copy_engine(fake_getenv));
    g_fake_env.clear();
    g_fake_env["SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE"] = "0:0";
    CHECK(ggml_sycl_xe_user_wants_copy_engine(fake_getenv));
    g_fake_env.clear();
    g_fake_env["UR_L0_USE_COPY_ENGINE_FOR_D2D_COPY"] = "1";
    CHECK(ggml_sycl_xe_user_wants_copy_engine(fake_getenv));
    g_fake_env.clear();
    g_fake_env["SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_FILL"] = "1";
    CHECK(ggml_sycl_xe_user_wants_copy_engine(fake_getenv));
    g_fake_env.clear();
    g_fake_env["UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD"] = "1";
    CHECK(!ggml_sycl_xe_user_wants_copy_engine(fake_getenv));
    g_fake_env["UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD"] = "0";
    CHECK(ggml_sycl_xe_user_wants_copy_engine(fake_getenv));
    g_fake_env.clear();

    // Decision: (dg2_on_xe, variable already set, user wants copy engines).
    CHECK(ggml_sycl_xe_apply_default(true, false, false));
    CHECK(!ggml_sycl_xe_apply_default(true, true, false));
    CHECK(!ggml_sycl_xe_apply_default(true, false, true));
    CHECK(!ggml_sycl_xe_apply_default(false, false, false));

    // Applying against the fake trees changes the process environment as decided.
    const std::string dg2_root  = (base / "bmg-first").string();  // DG2 behind a Battlemage node
    const std::string bmg_root  = (base / "bmg").string();
    const std::string i915_root = (base / "i915").string();
    auto env_eq = [](const char * name, const char * want) {
        const char * got = env(name);
        return got != nullptr && strcmp(got, want) == 0;
    };
    // Nothing set: both defaults. An empty v1 value counts as unset and is replaced.
    clear_copy_env();
    setenv("UR_L0_USE_COPY_ENGINE", "", 1);
    CHECK(ggml_sycl_apply_xe_kmd_defaults_in(dg2_root));
    CHECK(ggml_sycl_xe_kmd_defaults_applied());
    CHECK(env_eq("UR_L0_USE_COPY_ENGINE", "0"));
    CHECK(env_eq("UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD", "1"));
    // A FOR_* variable asking for copy engines: nothing is set on either adapter.
    clear_copy_env();
    setenv("UR_L0_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE", "1", 1);
    CHECK(!ggml_sycl_apply_xe_kmd_defaults_in(dg2_root));
    CHECK(!ggml_sycl_xe_kmd_defaults_applied());
    CHECK(env("UR_L0_USE_COPY_ENGINE") == nullptr);
    CHECK(env("UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD") == nullptr);
    // v1 explicitly on: nothing is set, the value stays.
    clear_copy_env();
    setenv("UR_L0_USE_COPY_ENGINE", "1", 1);
    CHECK(!ggml_sycl_apply_xe_kmd_defaults_in(dg2_root));
    CHECK(env_eq("UR_L0_USE_COPY_ENGINE", "1"));
    CHECK(env("UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD") == nullptr);
    // v1 explicitly off agrees with the default: the v2 default is still applied.
    clear_copy_env();
    setenv("UR_L0_USE_COPY_ENGINE", "0", 1);
    CHECK(ggml_sycl_apply_xe_kmd_defaults_in(dg2_root));
    CHECK(!ggml_sycl_xe_kmd_defaults_applied());  // the user set the v1 variable, not the hook
    CHECK(env_eq("UR_L0_USE_COPY_ENGINE", "0"));
    CHECK(env_eq("UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD", "1"));
    // v2 explicitly off agrees with the default: the v1 default is still applied.
    clear_copy_env();
    setenv("UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD", "1", 1);
    CHECK(ggml_sycl_apply_xe_kmd_defaults_in(dg2_root));
    CHECK(ggml_sycl_xe_kmd_defaults_applied());
    CHECK(env_eq("UR_L0_USE_COPY_ENGINE", "0"));
    CHECK(env_eq("UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD", "1"));
    // v2 explicitly keeping copy offload: nothing is set.
    clear_copy_env();
    setenv("UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD", "0", 1);
    CHECK(!ggml_sycl_apply_xe_kmd_defaults_in(dg2_root));
    CHECK(env("UR_L0_USE_COPY_ENGINE") == nullptr);
    CHECK(env_eq("UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD", "0"));
    // Empty UR variable next to a set SYCL_PI alias: the alias is copied into the UR
    // variable (the adapter would otherwise parse the empty string), nothing else is set.
    clear_copy_env();
    setenv("UR_L0_USE_COPY_ENGINE", "", 1);
    setenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE", "0:0", 1);
    CHECK(!ggml_sycl_apply_xe_kmd_defaults_in(dg2_root));
    CHECK(env_eq("UR_L0_USE_COPY_ENGINE", "0:0"));
    CHECK(env("UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD") == nullptr);
    // SYCL_PI alias off alone: the v1 variable is left to the alias, the v2 default applies.
    clear_copy_env();
    setenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE", "0", 1);
    CHECK(ggml_sycl_apply_xe_kmd_defaults_in(dg2_root));
    CHECK(env("UR_L0_USE_COPY_ENGINE") == nullptr);
    CHECK(env_eq("UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD", "1"));
    // Empty FOR_* variables abort the adapter at load; they are dropped on every GPU.
    clear_copy_env();
    setenv("UR_L0_USE_COPY_ENGINE_FOR_D2D_COPY", "", 1);
    setenv("UR_L0_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE", "", 1);
    CHECK(ggml_sycl_apply_xe_kmd_defaults_in(dg2_root));
    CHECK(env("UR_L0_USE_COPY_ENGINE_FOR_D2D_COPY") == nullptr);
    CHECK(env("UR_L0_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE") == nullptr);
    CHECK(env_eq("UR_L0_USE_COPY_ENGINE", "0"));
    // Off DG2/xe nothing is touched, not even an empty value or an alias to mirror.
    clear_copy_env();
    setenv("UR_L0_USE_COPY_ENGINE_FOR_FILL", "", 1);
    CHECK(!ggml_sycl_apply_xe_kmd_defaults_in(bmg_root));
    CHECK(env("UR_L0_USE_COPY_ENGINE_FOR_FILL") != nullptr && env("UR_L0_USE_COPY_ENGINE_FOR_FILL")[0] == '\0');
    clear_copy_env();
    setenv("UR_L0_USE_COPY_ENGINE", "", 1);
    setenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE", "1", 1);
    CHECK(!ggml_sycl_apply_xe_kmd_defaults_in(i915_root));
    CHECK(env("UR_L0_USE_COPY_ENGINE") != nullptr && env("UR_L0_USE_COPY_ENGINE")[0] == '\0');
    CHECK(env_eq("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE", "1"));
    CHECK(env("UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD") == nullptr);
    // Empty UR in-order variable next to its set alias: mirrored, and the alias asks for
    // copy engines, so no default is applied.
    clear_copy_env();
    setenv("UR_L0_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE", "", 1);
    setenv("SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE", "1", 1);
    CHECK(!ggml_sycl_apply_xe_kmd_defaults_in(dg2_root));
    CHECK(env_eq("UR_L0_USE_COPY_ENGINE_FOR_IN_ORDER_QUEUE", "1"));
    CHECK(env("UR_L0_USE_COPY_ENGINE") == nullptr);
    // Not DG2, or not xe: nothing set.
    clear_copy_env();
    CHECK(!ggml_sycl_apply_xe_kmd_defaults_in(bmg_root));
    CHECK(env("UR_L0_USE_COPY_ENGINE") == nullptr);
    CHECK(!ggml_sycl_apply_xe_kmd_defaults_in(i915_root));
    CHECK(env("UR_L0_USE_COPY_ENGINE") == nullptr);
    // Kill switch.
    clear_copy_env();
    setenv("GGML_SYCL_XE_COPY_ENGINE_DEFAULT", "0", 1);
    CHECK(!ggml_sycl_apply_xe_kmd_defaults_in(dg2_root));
    CHECK(!ggml_sycl_xe_kmd_defaults_applied());
    CHECK(env("UR_L0_USE_COPY_ENGINE") == nullptr);
    clear_copy_env();
    ggml_sycl_xe_kmd_flush_log();  // show what the hook would have logged at backend init

    fs::remove_all(base);

    if (g_failures == 0) {
        printf("test-sycl-xe-defaults: OK\n");
        return 0;
    }
    printf("test-sycl-xe-defaults: %d failure(s)\n", g_failures);
    return 1;
}
