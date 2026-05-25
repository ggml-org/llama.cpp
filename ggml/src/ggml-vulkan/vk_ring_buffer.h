#pragma once

#include <vulkan/vulkan.hpp>
#include "vk_allocator.h"
#include <cstdint>
#include <atomic>

namespace ggml_vk {

// Staging ring buffer for efficient host↔device transfers.
// Uses a circular buffer with fence-based tracking to avoid stalls.
class StagingRingBuffer {
public:
    struct Allocation {
        vk::Buffer buffer{};
        uint64_t offset{};
        uint64_t size{};
        void* mapped_ptr{};
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

    // Signal that the GPU has finished using the given fence value.
    // This advances the tail pointer.
    void finish(uint64_t fence_value);

    // Get current fence value (monotonically increasing)
    [[nodiscard]] uint64_t current_fence() const { return _fence_counter.load(); }

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
    uint64_t _total_size{};
    bool _host_to_device{};

    // Ring buffer state
    std::atomic<uint64_t> _head{0};   // next write position
    std::atomic<uint64_t> _tail{0};   // next free position (all before this are free)
    std::atomic<uint64_t> _fence_counter{0};

    // Per-allocation fence tracking
    // We store fence values for outstanding allocations
    struct FenceEntry {
        uint64_t offset;
        uint64_t size;
        uint64_t fence_value;
    };
    std::vector<FenceEntry> _outstanding;
    std::mutex _mutex;
};

} // namespace ggml_vk
