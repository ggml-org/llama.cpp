// Unit tests for the free host-memory estimate helpers in ggml-cpu-mem.cpp.
// All tests use fixture files in a temporary directory; no real /proc or /sys access is needed.

#include "../ggml/src/ggml-cpu/ggml-cpu-mem.h"

#include <cstdio>
#include <filesystem>
#include <random>
#include <string>

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); return false; } } while (0)

namespace {

const size_t GiB = 1024ull * 1024 * 1024;

struct fixture {
    std::string dir;

    fixture() {
        dir = (std::filesystem::temp_directory_path() / ("test-host-mem-" + std::to_string(std::random_device{}()))).string();
        std::error_code ec;
        if (!std::filesystem::create_directories(dir, ec)) {
            fprintf(stderr, "cannot create fixture dir %s: %s\n", dir.c_str(), ec.message().c_str());
            dir.clear();
        }
    }

    ~fixture() {
        if (!dir.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(dir, ec);
            if (ec) {
                fprintf(stderr, "cleanup failed: %s: %s\n", dir.c_str(), ec.message().c_str());
            }
        }
    }

    void write(const char * name, const char * content) const {
        const std::filesystem::path path = std::filesystem::path(dir) / name;
        // cgroup v1 fixtures live one level down (memory/...)
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        FILE * f = fopen(path.string().c_str(), "w");
        if (f == nullptr) {
            fprintf(stderr, "cannot write fixture %s\n", path.string().c_str());
            return;
        }
        fputs(content, f);
        fclose(f);
    }

    std::string path(const char * name) const {
        return (std::filesystem::path(dir) / name).string();
    }
};

bool test_meminfo() {
    fixture fx;
    size_t free_mem = 123;

    // MemAvailable preferred
    fx.write("meminfo", "MemTotal: 32768 kB\nMemFree: 1024 kB\nMemAvailable: 8192 kB\n");
    CHECK(ggml_cpu_meminfo_free(fx.path("meminfo").c_str(), &free_mem));
    CHECK(free_mem == 8192 * 1024);

    // MemFree fallback without MemAvailable
    fx.write("meminfo", "MemTotal: 32768 kB\nMemFree: 2048 kB\n");
    CHECK(ggml_cpu_meminfo_free(fx.path("meminfo").c_str(), &free_mem));
    CHECK(free_mem == 2048 * 1024);

    // failure paths: unusable file, missing file; out param untouched
    fx.write("meminfo", "MemTotal: 32768 kB\n");
    free_mem = 123;
    CHECK(!ggml_cpu_meminfo_free(fx.path("meminfo").c_str(), &free_mem));
    CHECK(free_mem == 123);
    CHECK(!ggml_cpu_meminfo_free(fx.path("nonexistent").c_str(), &free_mem));
    CHECK(free_mem == 123);

    printf("%s: ok\n", __func__);
    return true;
}

bool test_cgroup() {
    fixture fx;
    size_t remaining = 0;

    // v2: 4 GiB limit, 1 GiB used
    fx.write("memory.max", "4294967296\n");
    fx.write("memory.current", "1073741824\n");
    CHECK(ggml_cpu_cgroup_remaining(fx.dir.c_str(), 32 * GiB, &remaining));
    CHECK(remaining == 3 * GiB);

    // v2: usage above limit clamps to 0
    fx.write("memory.current", "5368709120\n");
    CHECK(ggml_cpu_cgroup_remaining(fx.dir.c_str(), 32 * GiB, &remaining));
    CHECK(remaining == 0);

    // v2: "max" means no limit
    fx.write("memory.max", "max\n");
    CHECK(!ggml_cpu_cgroup_remaining(fx.dir.c_str(), 32 * GiB, &remaining));

    // v1 fallback, 8 GiB limit, 2 GiB used
    fx.write("memory/memory.limit_in_bytes", "8589934592\n");
    fx.write("memory/memory.usage_in_bytes", "2147483648\n");
    CHECK(ggml_cpu_cgroup_remaining(fx.dir.c_str(), 32 * GiB, &remaining));
    CHECK(remaining == 6 * GiB);

    // v1 default sentinel limit (above physical total): no constraint
    fx.write("memory/memory.limit_in_bytes", "9223372036854771712\n");
    CHECK(!ggml_cpu_cgroup_remaining(fx.dir.c_str(), 32 * GiB, &remaining));

    printf("%s: ok\n", __func__);
    return true;
}

bool test_estimate() {
    fixture fx;
    size_t free_mem = 0;

    // both sources: the tighter one wins (cgroup 4 GiB < meminfo 10 GiB)
    fx.write("meminfo", "MemAvailable: 10485760 kB\n");
    fx.write("memory.max", "4294967296\n");
    fx.write("memory.current", "0\n");
    CHECK(ggml_cpu_host_free_estimate(fx.path("meminfo").c_str(), fx.dir.c_str(), 32 * GiB, &free_mem));
    CHECK(free_mem == 4 * GiB);

    // meminfo only
    fx.write("memory.max", "max\n");
    CHECK(ggml_cpu_host_free_estimate(fx.path("meminfo").c_str(), fx.dir.c_str(), 32 * GiB, &free_mem));
    CHECK(free_mem == 10 * GiB);

    // cgroup only
    fx.write("memory.max", "5368709120\n");
    fx.write("memory.current", "1073741824\n");
    CHECK(ggml_cpu_host_free_estimate(fx.path("nonexistent").c_str(), fx.dir.c_str(), 32 * GiB, &free_mem));
    CHECK(free_mem == 4 * GiB);

    // neither source: keep previous behavior
    fx.write("memory.max", "max\n");
    CHECK(!ggml_cpu_host_free_estimate(fx.path("nonexistent").c_str(), fx.dir.c_str(), 32 * GiB, &free_mem));

    printf("%s: ok\n", __func__);
    return true;
}

} // namespace

int main() {
    int failures = 0;
    failures += !test_meminfo();
    failures += !test_cgroup();
    failures += !test_estimate();
    if (failures != 0) {
        fprintf(stderr, "%d test(s) failed\n", failures);
        return 1;
    }
    printf("all tests passed\n");
    return 0;
}
