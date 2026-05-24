#if defined(__gnu_linux__) && !defined(_GNU_SOURCE)
    #define _GNU_SOURCE  // for posix_memalign
#endif

#include "pinned.h"
#include "ggml-impl.h"

#include <string.h>

#if defined(__gnu_linux__)
    #include <sys/mman.h>
    #include <stdlib.h>
    #include <unistd.h>
#elif defined(_WIN32)
    #include <windows.h>
#endif

#define GGML_PINNED_ALIGNMENT 4096

void * ggml_cpu_pinned_alloc(size_t size) {
    if (size == 0) return NULL;
    void * ptr = NULL;

#if defined(__gnu_linux__)
    // Method 1: mmap with MAP_LOCKED (locks pages at creation)
    ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
               MAP_SHARED | MAP_ANONYMOUS | MAP_LOCKED, -1, 0);
    if (ptr != MAP_FAILED) {
        GGML_LOG_INFO("pinned: mmap(MAP_LOCKED) %zu bytes\n", size);
        return ptr;
    }

    // Method 2: posix_memalign + mlock
    if (posix_memalign(&ptr, GGML_PINNED_ALIGNMENT, size) == 0) {
        if (mlock(ptr, size) == 0) {
            GGML_LOG_INFO("pinned: posix_memalign + mlock %zu bytes\n", size);
            return ptr;
        }
        free(ptr);
        ptr = NULL;
    }

    // Method 3: plain malloc fallback
    GGML_LOG_WARN("pinned: MAP_LOCKED and mlock failed, using malloc (pages not locked)\n");
    ptr = malloc(size);
    if (ptr) return ptr;

#elif defined(_WIN32)
    ptr = VirtualAlloc(NULL, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (ptr && VirtualLock(ptr, size)) {
        GGML_LOG_INFO("pinned: VirtualLock %zu bytes\n", size);
        return ptr;
    }
    if (ptr) VirtualFree(ptr, 0, MEM_RELEASE);

    GGML_LOG_WARN("pinned: VirtualLock failed, using malloc\n");
    ptr = malloc(size);
    if (ptr) return ptr;
#else
    // Generic fallback
    GGML_LOG_WARN("pinned: platform not supported, using malloc\n");
    ptr = malloc(size);
#endif

    if (!ptr) {
        GGML_LOG_ERROR("pinned: allocation failed for %zu bytes\n", size);
    }
    return ptr;
}

void ggml_cpu_pinned_free(void * ptr, size_t size) {
    if (!ptr) return;

#if defined(__gnu_linux__)
    // Try munmap first — if it succeeds, we allocated via mmap
    int ret = munmap(ptr, size);
    if (ret == 0) return;

    // Otherwise it was posix_memalign/malloc — just free
    free(ptr);

#elif defined(_WIN32)
    // VirtualFree with MEM_RELEASE handles both locked and unlocked
    if (!VirtualFree(ptr, 0, MEM_RELEASE)) {
        free(ptr);
    }
#else
    free(ptr);
#endif
}
