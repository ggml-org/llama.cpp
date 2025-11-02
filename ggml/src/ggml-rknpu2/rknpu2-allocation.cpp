#include "rknpu2-allocation.h"

#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <cerrno>
#include <cstdio>

// --- Anonymous namespace for implementation details ---

namespace {

// --- DMA-Heap specific structures and IOCTL commands ---

struct dma_heap_allocation_data {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};

#define DMA_HEAP_IOC_MAGIC 'H'
#define DMA_HEAP_IOCTL_ALLOC _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct dma_heap_allocation_data)

} // anonymous namespace

namespace rknpu2_allocation {

DmaBuffer alloc(size_t size) {
    DmaBuffer buffer;
    buffer.size = size;

    const char* path = "/dev/dma_heap/system";
    int dma_heap_fd = open(path, O_RDWR);
    if (dma_heap_fd < 0) {
        fprintf(stderr, "RKNPU_DMA_ALLOC: Failed to open %s: %s\n", path, strerror(errno));
        return buffer;
    }

    dma_heap_allocation_data buf_data;
    memset(&buf_data, 0, sizeof(buf_data));
    buf_data.len = size;
    buf_data.fd_flags = O_CLOEXEC | O_RDWR;

    if (ioctl(dma_heap_fd, DMA_HEAP_IOCTL_ALLOC, &buf_data) < 0) {
        fprintf(stderr, "RKNPU_DMA_ALLOC: ioctl DMA_HEAP_IOCTL_ALLOC failed: %s\n", strerror(errno));
        close(dma_heap_fd);
        return buffer;
    }

    close(dma_heap_fd);

    buffer.fd = buf_data.fd;
    buffer.virt_addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, buffer.fd, 0);

    if (buffer.virt_addr == MAP_FAILED) {
        fprintf(stderr, "RKNPU_DMA_ALLOC: mmap failed: %s\n", strerror(errno));
        close(buffer.fd);
        buffer.fd = -1;
        buffer.virt_addr = nullptr;
    }

    return buffer;
}

void free(const DmaBuffer& buffer) {
    if (buffer.virt_addr != nullptr) {
        munmap(buffer.virt_addr, buffer.size);
    }
    if (buffer.fd >= 0) {
        close(buffer.fd);
    }
}

} // namespace rknpu2_allocation