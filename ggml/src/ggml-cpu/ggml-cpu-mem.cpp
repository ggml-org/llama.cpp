#include "ggml-cpu-mem.h"

#include <cinttypes>
#include <cstdio>
#include <limits>

namespace {

// reads a single unsigned value from a one-line file ("max" and other non-numeric contents fail the conversion); returns false on failure
bool read_u64_file(const char * path, uint64_t * out) {
    FILE * file = fopen(path, "r");
    if (file == nullptr) {
        return false;
    }
    unsigned long long value = 0;
    const bool ok = fscanf(file, "%llu", &value) == 1;
    fclose(file);
    if (!ok) {
        return false;
    }
    *out = value;
    return true;
}

void kib_to_bytes(uint64_t kib, size_t * out) {
    const uint64_t max_kib = std::numeric_limits<size_t>::max() / 1024;
    *out = kib > max_kib ? std::numeric_limits<size_t>::max() : (size_t) kib * 1024;
}

bool cgroup_remaining_at(const char * limit_path, const char * usage_path, size_t total, size_t * out_remaining) {
    uint64_t limit = 0;
    uint64_t usage = 0;
    if (!read_u64_file(limit_path, &limit) || !read_u64_file(usage_path, &usage)) {
        return false;
    }
    // no effective constraint: unset ("max" fails to parse above), or at/above the physical total (e.g. the cgroup v1 default)
    if (limit >= total) {
        return false;
    }
    *out_remaining = usage < limit ? (size_t) (limit - usage) : 0;
    return true;
}

} // namespace

bool ggml_cpu_meminfo_free(const char * path, size_t * out_free) {
    FILE * file = fopen(path, "r");
    if (file == nullptr) {
        return false;
    }

    uint64_t available_kib = 0;
    uint64_t free_kib      = 0;
    bool     has_available = false;
    bool     has_free      = false;

    char line[256];
    while (fgets(line, sizeof(line), file) != nullptr) {
        unsigned long long value = 0;
        if (sscanf(line, "MemAvailable: %llu kB", &value) == 1) {
            available_kib = value;
            has_available = true;
        } else if (sscanf(line, "MemFree: %llu kB", &value) == 1) {
            free_kib = value;
            has_free  = true;
        }
    }
    fclose(file);

    // MemAvailable is the kernel estimate and is preferred; MemFree is the safe fallback on kernels without it
    if (has_available) {
        kib_to_bytes(available_kib, out_free);
        return true;
    }
    if (has_free) {
        kib_to_bytes(free_kib, out_free);
        return true;
    }
    return false;
}

bool ggml_cpu_cgroup_remaining(const char * dir, size_t total, size_t * out_remaining) {
    char limit[4096];
    char usage[4096];

    // cgroup v2
    snprintf(limit, sizeof(limit), "%s/memory.max", dir);
    snprintf(usage, sizeof(usage), "%s/memory.current", dir);
    if (cgroup_remaining_at(limit, usage, total, out_remaining)) {
        return true;
    }

    // cgroup v1
    snprintf(limit, sizeof(limit), "%s/memory/memory.limit_in_bytes", dir);
    snprintf(usage, sizeof(usage), "%s/memory/memory.usage_in_bytes", dir);
    return cgroup_remaining_at(limit, usage, total, out_remaining);
}

bool ggml_cpu_host_free_estimate(const char * meminfo_path, const char * cgroup_dir, size_t total, size_t * out_free) {
    size_t meminfo_free = 0;
    size_t cgroup_free  = 0;
    const bool has_meminfo = ggml_cpu_meminfo_free(meminfo_path, &meminfo_free);
    const bool has_cgroup  = ggml_cpu_cgroup_remaining(cgroup_dir, total, &cgroup_free);

    if (has_meminfo && has_cgroup) {
        *out_free = meminfo_free < cgroup_free ? meminfo_free : cgroup_free;
        return true;
    }
    if (has_meminfo || has_cgroup) {
        *out_free = has_meminfo ? meminfo_free : cgroup_free;
        return true;
    }
    return false;
}
