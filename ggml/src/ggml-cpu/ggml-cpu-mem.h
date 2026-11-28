// Free host-memory estimate helpers for the CPU backend.
// Factored out of ggml-cpu.cpp and parameterized by file paths so that the parsing and combination logic can be unit-tested with fixture files.
// Nothing here is Linux-specific by itself; only the paths passed by the caller are ("/proc/meminfo", "/sys/fs/cgroup").

#pragma once

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Reads MemAvailable (preferred) or MemFree from a /proc/meminfo-style file and returns the value in bytes.
// Returns false when the file is unreadable or neither field is present; out_free is not modified in that case.
bool ggml_cpu_meminfo_free(const char * path, size_t * out_free);

// Reads the cgroup memory limit and current usage from a cgroup mount directory (v2: memory.max + memory.current; v1: memory/memory.limit_in_bytes + memory/memory.usage_in_bytes) and returns the remaining bytes.
// Returns false when no effective limit is configured ("max", a limit at or above the physical total, unreadable files); a cgroup without a limit does not constrain the estimate.
// total is the physical memory size in bytes.
bool ggml_cpu_cgroup_remaining(const char * dir, size_t total, size_t * out_remaining);

// Combined free-memory estimate: the smaller of the meminfo free memory and the cgroup remaining memory, using whichever source is available.
// Returns false when neither source produced a value; the caller should keep its previous behavior in that case.
bool ggml_cpu_host_free_estimate(const char * meminfo_path, const char * cgroup_dir, size_t total, size_t * out_free);

#ifdef __cplusplus
}
#endif
