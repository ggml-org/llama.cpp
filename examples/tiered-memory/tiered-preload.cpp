#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using mmap_fn = void * (*)(void *, size_t, int, int, int, off_t);
using munmap_fn = int (*)(void *, size_t);
using cuda_host_register_fn = int (*)(void *, size_t, unsigned int);
using cuda_host_unregister_fn = int (*)(void *);

constexpr unsigned int cuda_host_register_portable = 0x01;

struct range {
    uint64_t offset = 0;
    uint64_t size = 0;
};

struct registration {
    void * ptr = nullptr;
    size_t size = 0;
};

std::once_flag init_once;
std::mutex registrations_mutex;
std::unordered_map<std::string, std::vector<range>> ranges_by_path;
std::vector<registration> registrations;

mmap_fn real_mmap = nullptr;
munmap_fn real_munmap = nullptr;
cuda_host_register_fn cuda_host_register = nullptr;
cuda_host_unregister_fn cuda_host_unregister = nullptr;

bool enabled = false;
bool warned_cuda = false;

std::string canonical_path(const std::string & path) {
    char * resolved = realpath(path.c_str(), nullptr);
    if (!resolved) {
        return path;
    }
    std::string result(resolved);
    std::free(resolved);
    return result;
}

std::string fd_path(int fd) {
    char link_path[64];
    std::snprintf(link_path, sizeof(link_path), "/proc/self/fd/%d", fd);

    std::vector<char> buffer(4096);
    const ssize_t len = readlink(link_path, buffer.data(), buffer.size() - 1);
    if (len <= 0) {
        return {};
    }
    buffer[static_cast<size_t>(len)] = '\0';

    std::string result(buffer.data());
    const std::string deleted = " (deleted)";
    if (result.size() > deleted.size() && result.compare(result.size() - deleted.size(), deleted.size(), deleted) == 0) {
        result.resize(result.size() - deleted.size());
    }
    return canonical_path(result);
}

void load_cuda_symbols() {
    cuda_host_register = reinterpret_cast<cuda_host_register_fn>(dlsym(RTLD_DEFAULT, "cudaHostRegister"));
    cuda_host_unregister = reinterpret_cast<cuda_host_unregister_fn>(dlsym(RTLD_DEFAULT, "cudaHostUnregister"));

    if (cuda_host_register && cuda_host_unregister) {
        return;
    }

    const char * candidates[] = {
        "libcudart.so",
        "libcudart.so.13",
        "libcudart.so.12",
        "libcudart.so.11.0",
    };

    for (const char * candidate : candidates) {
        void * handle = dlopen(candidate, RTLD_LAZY | RTLD_LOCAL);
        if (!handle) {
            continue;
        }
        cuda_host_register = reinterpret_cast<cuda_host_register_fn>(dlsym(handle, "cudaHostRegister"));
        cuda_host_unregister = reinterpret_cast<cuda_host_unregister_fn>(dlsym(handle, "cudaHostUnregister"));
        if (cuda_host_register && cuda_host_unregister) {
            return;
        }
    }
}

void load_manifest(const char * manifest_path) {
    std::ifstream input(manifest_path);
    if (!input) {
        std::fprintf(stderr, "llama-tiered-preload: cannot open manifest %s\n", manifest_path);
        return;
    }

    std::string line;
    std::string model_path;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (line.rfind("model\t", 0) == 0) {
            model_path = canonical_path(line.substr(6));
            continue;
        }
        if (line.rfind("dram\t", 0) != 0 || model_path.empty()) {
            continue;
        }

        const size_t first_tab = line.find('\t', 5);
        const size_t second_tab = first_tab == std::string::npos ? std::string::npos : line.find('\t', first_tab + 1);
        if (first_tab == std::string::npos || second_tab == std::string::npos) {
            continue;
        }

        try {
            const uint64_t offset = std::stoull(line.substr(5, first_tab - 5));
            const uint64_t size = std::stoull(line.substr(first_tab + 1, second_tab - first_tab - 1));
            if (size != 0) {
                ranges_by_path[model_path].push_back({offset, size});
            }
        } catch (...) {
            std::fprintf(stderr, "llama-tiered-preload: invalid manifest line: %s\n", line.c_str());
        }
    }

    for (auto & entry : ranges_by_path) {
        auto & ranges = entry.second;
        std::sort(ranges.begin(), ranges.end(), [](const range & a, const range & b) {
            return a.offset < b.offset;
        });

        std::vector<range> merged;
        for (const auto & current : ranges) {
            if (merged.empty() || current.offset > merged.back().offset + merged.back().size) {
                merged.push_back(current);
            } else {
                const uint64_t end = std::max(merged.back().offset + merged.back().size, current.offset + current.size);
                merged.back().size = end - merged.back().offset;
            }
        }
        ranges = std::move(merged);
    }

    enabled = !ranges_by_path.empty();
}

void initialize() {
    real_mmap = reinterpret_cast<mmap_fn>(dlsym(RTLD_NEXT, "mmap"));
    real_munmap = reinterpret_cast<munmap_fn>(dlsym(RTLD_NEXT, "munmap"));
    if (!real_mmap || !real_munmap) {
        std::fprintf(stderr, "llama-tiered-preload: failed to resolve mmap/munmap\n");
        return;
    }

    const char * manifest_path = std::getenv("LLAMA_TIERED_MANIFEST");
    if (!manifest_path || manifest_path[0] == '\0') {
        return;
    }

    load_manifest(manifest_path);
    if (enabled) {
        load_cuda_symbols();
        std::fprintf(stderr, "llama-tiered-preload: loaded %zu model mapping(s)\n", ranges_by_path.size());
    }
}

void ensure_initialized() {
    std::call_once(init_once, initialize);
}

void register_intersections(void * mapping, size_t length, off_t mapping_offset, const std::vector<range> & ranges) {
    if (!cuda_host_register || !cuda_host_unregister || mapping == MAP_FAILED || length == 0) {
        if (!warned_cuda && enabled) {
            warned_cuda = true;
            std::fprintf(stderr, "llama-tiered-preload: CUDA runtime registration symbols unavailable; continuing pageable\n");
        }
        return;
    }

    const uint64_t map_begin = static_cast<uint64_t>(mapping_offset);
    const uint64_t map_end = map_begin + length;
    const size_t page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));

    for (const auto & current : ranges) {
        const uint64_t range_begin = current.offset;
        const uint64_t range_end = current.offset + current.size;
        const uint64_t begin = std::max(map_begin, range_begin);
        const uint64_t end = std::min(map_end, range_end);
        if (begin >= end) {
            continue;
        }

        uintptr_t ptr_value = reinterpret_cast<uintptr_t>(mapping) + static_cast<uintptr_t>(begin - map_begin);
        uintptr_t aligned_begin = ptr_value & ~(static_cast<uintptr_t>(page_size) - 1);
        uintptr_t aligned_end = (reinterpret_cast<uintptr_t>(mapping) + static_cast<uintptr_t>(end - map_begin) + page_size - 1) &
                ~(static_cast<uintptr_t>(page_size) - 1);
        if (aligned_end <= aligned_begin) {
            continue;
        }

        void * ptr = reinterpret_cast<void *>(aligned_begin);
        const size_t size = aligned_end - aligned_begin;
        const int status = cuda_host_register(ptr, size, cuda_host_register_portable);
        if (status != 0) {
            std::fprintf(stderr, "llama-tiered-preload: cudaHostRegister(%p, %zu) failed with status %d\n", ptr, size, status);
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(registrations_mutex);
            registrations.push_back({ptr, size});
        }
        std::fprintf(stderr, "llama-tiered-preload: registered %.2f MiB at %p\n", size / 1024.0 / 1024.0, ptr);
    }
}

void unregister_overlaps(void * addr, size_t length) {
    if (!cuda_host_unregister || length == 0) {
        return;
    }

    const uintptr_t begin = reinterpret_cast<uintptr_t>(addr);
    const uintptr_t end = begin + length;

    std::lock_guard<std::mutex> lock(registrations_mutex);
    auto it = registrations.begin();
    while (it != registrations.end()) {
        const uintptr_t reg_begin = reinterpret_cast<uintptr_t>(it->ptr);
        const uintptr_t reg_end = reg_begin + it->size;
        if (reg_begin < end && begin < reg_end) {
            const int status = cuda_host_unregister(it->ptr);
            if (status != 0) {
                std::fprintf(stderr, "llama-tiered-preload: cudaHostUnregister(%p) failed with status %d\n", it->ptr, status);
            }
            it = registrations.erase(it);
        } else {
            ++it;
        }
    }
}

} // namespace

extern "C" void * mmap(void * addr, size_t length, int prot, int flags, int fd, off_t offset) {
    ensure_initialized();
    if (!real_mmap) {
        errno = ENOSYS;
        return MAP_FAILED;
    }

    std::string path;
    const std::vector<range> * selected_ranges = nullptr;
    if (enabled && fd >= 0 && (flags & MAP_ANONYMOUS) == 0) {
        path = fd_path(fd);
        const auto found = ranges_by_path.find(path);
        if (found != ranges_by_path.end()) {
            selected_ranges = &found->second;
            prot |= PROT_WRITE;
            flags = (flags & ~MAP_SHARED) | MAP_PRIVATE;
        }
    }

    void * result = real_mmap(addr, length, prot, flags, fd, offset);
    if (selected_ranges && result != MAP_FAILED) {
        register_intersections(result, length, offset, *selected_ranges);
    }
    return result;
}

extern "C" void * mmap64(void * addr, size_t length, int prot, int flags, int fd, off64_t offset) {
    return mmap(addr, length, prot, flags, fd, static_cast<off_t>(offset));
}

extern "C" int munmap(void * addr, size_t length) {
    ensure_initialized();
    if (!real_munmap) {
        errno = ENOSYS;
        return -1;
    }
    unregister_overlaps(addr, length);
    return real_munmap(addr, length);
}
