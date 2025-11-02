#pragma once

#include <cstddef>
#include <cstdint>

/**
 * @brief Provides a simple interface for allocating and freeing physically contiguous
 * memory using the Linux DMA-Heap subsystem, which is required for zero-copy
 * operations with the Rockchip NPU.
 */
namespace rknpu2_allocation {

/**
 * @brief Represents a block of memory allocated from the DMA-Heap.
 *
 * This struct holds the file descriptor, virtual address, and size of the
 * allocated buffer. It is the handle used for all operations.
 */
struct DmaBuffer {
    int fd = -1;                // File descriptor for the DMA buffer
    void* virt_addr = nullptr;  // User-space virtual address mapped to the buffer
    size_t size = 0;            // Size of the allocation in bytes
};

/**
 * @brief Allocates a physically contiguous block of memory from the system DMA heap.
 *
 * @param size The number of bytes to allocate.
 * @return A DmaBuffer struct. If allocation fails, `fd` will be -1 and
 *         `virt_addr` will be nullptr.
 */
DmaBuffer alloc(size_t size);

/**
 * @brief Frees a previously allocated DMA buffer.
 *
 * This function unmaps the virtual address and closes the file descriptor.
 *
 * @param buffer The DmaBuffer to be freed.
 */
void free(const DmaBuffer& buffer);

} // namespace rknpu2_allocation