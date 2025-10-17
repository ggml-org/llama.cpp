#pragma once

#include <cstddef>
#include <cstdint>

// --- Structs ---
struct dma_heap_allocation_data {
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};

// --- IOCTL-commands ---

#define DMA_HEAP_IOC_MAGIC 'H'
#define DMA_HEAP_IOCTL_ALLOC _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct dma_heap_allocation_data)

// Information of allocated block
struct DmaBuffer {
    int fd = -1;
    void* virt_addr = nullptr;
    size_t size = 0;
};

// --- Functions ---

DmaBuffer dma_alloc(size_t size);

void dma_free(const DmaBuffer& buffer);
