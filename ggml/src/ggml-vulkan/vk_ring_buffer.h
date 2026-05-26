#pragma once

#include <vulkan/vulkan.hpp>
#include "vk_allocator.h"
#include <cstdint>
#include <atomic>
#include <deque>
#include <mutex>

namespace ggml_vk {

// Staging ring buffer for efficient host↔device transfers.
// Uses a circular buffer with fence-based tracking to avoid stalls.
//
// NOTE: For non-HOST_COHERENT memory types, callers must manually flush/invalidate
// mapped memory ranges via VMA (vmaFlushAllocation / vmaInvalidateAllocation) before
// GPU access. The Allocation struct contains the buffer and offset needed to identify
// the flushed range. Use _allocator->handle() to obtain the VmaAllocator.
class StagingRingBuffer {
public:
    struct Allocation {
        vk::Buffer buffer{};
        uint64_t offset{};
        uint64_t size{};
        void* mapped_ptr{};
        uint64_t fence_value{};
    };

    StagingRingBuffer(VkAllocator* allocator,
                      vk::Device device,
                      uint64_t total_size,
                      bool host_to_device  // true=upload, false=readback
    );
    ~StagingRingBuffer();

    StagingRingBuffer(const StagingRingBuffer&) = delete;
    StagingRingBuffer& operator=(const StagingRingBuffer&) = delete;

    // Allocate a chunk from the ring. Blocks if no space available.
    // Call finish() after the GPU work using this allocation completes.
    [[nodiscard]] Allocation allocate(uint64_t size, uint64_t alignment = 256);

    // Try to allocate a chunk from the ring without blocking.
    // Returns Allocation{} (size==0) if there is no space.
    [[nodiscard]] Allocation try_allocate(uint64_t size, uint64_t alignment = 256);

    // Signal that the GPU has finished using the given fence value.
    // This advances the tail pointer.
    void finish(uint64_t fence_value);

    // Get current fence value (monotonically increasing)
    [[nodiscard]] uint64_t current_fence() const { return _fence_counter.load(std::memory_order_relaxed); }

    // Advance fence and return new value (for use with timeline semaphores)
    [[nodiscard]] uint64_t advance_fence();

    [[nodiscard]] vk::Buffer buffer() const { return _buffer; }
    [[nodiscard]] uint64_t total_size() const { return _total_size; }

private:
    VkAllocator* _allocator;
    vk::Device _device;
    vk::Buffer _buffer{};
    VmaAllocation _vma_allocation{};
    void* _mapped_ptr{};
    bool _owns_map = false;
    uint64_t _total_size{};
    bool _host_to_device{};

    // Ring buffer state
    std::atomic<uint64_t> _head{0};   // next write position (monotonic)
    std::atomic<uint64_t> _tail{0};   // next free position (monotonic)
    std::atomic<uint64_t> _fence_counter{0};

    // Per-allocation fence tracking
    // We store fence values for outstanding allocations
    struct FenceEntry {
        uint64_t end;          // logical end position (_head after allocation)
        uint64_t fence_value;
    };
    std::deque<FenceEntry> _outstanding;
    std::mutex _mutex;
};

} // namespace ggml_vk
