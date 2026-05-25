#include "vk_ring_buffer.h"
#include "vk_mem_alloc.h"
#include <algorithm>
#include <stdexcept>
#include <thread>

namespace ggml_vk {

StagingRingBuffer::StagingRingBuffer(VkAllocator* allocator,
                                     vk::Device device,
                                     uint64_t total_size,
                                     bool host_to_device)
    : _allocator(allocator)
    , _device(device)
    , _total_size(total_size)
    , _host_to_device(host_to_device) {

    vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eTransferSrc |
                                  vk::BufferUsageFlagBits::eTransferDst;
    AccessType access = host_to_device ? AccessType::kUpload : AccessType::kReadBack;

    auto buf = _allocator->allocate_buffer(total_size, usage, access);
    _buffer = buf.buffer;
    _vma_allocation = buf.allocation;

    // Map persistently
    VmaAllocationInfo alloc_info{};
    vmaGetAllocationInfo(_allocator->handle(), _vma_allocation, &alloc_info);
    _mapped_ptr = alloc_info.pMappedData;
    if (!_mapped_ptr) {
        // Map if not already host-visible
        VkResult res = vmaMapMemory(_allocator->handle(), _vma_allocation, &_mapped_ptr);
        if (res != VK_SUCCESS) {
            _allocator->destroy_buffer({_buffer, _vma_allocation});
            _buffer = nullptr;
            _vma_allocation = nullptr;
            throw std::runtime_error("StagingRingBuffer: failed to map memory");
        }
        _owns_map = true;
    }
}

StagingRingBuffer::~StagingRingBuffer() {
    if (_owns_map && _mapped_ptr) {
        vmaUnmapMemory(_allocator->handle(), _vma_allocation);
    }
    if (_buffer) {
        _allocator->destroy_buffer({_buffer, _vma_allocation});
    }
}

StagingRingBuffer::Allocation StagingRingBuffer::allocate(uint64_t size, uint64_t alignment) {
    if (size > _total_size) [[unlikely]] {
        // allocation exceeds total size
        return Allocation{};
    }

    // Align size
    alignment = std::max<uint64_t>(alignment, 256);
    uint64_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

    std::unique_lock<std::mutex> lock(_mutex);
    while (true) {
        uint64_t head = _head.load(std::memory_order_relaxed);
        uint64_t tail = _tail.load(std::memory_order_relaxed);

        // Align head
        uint64_t aligned_head = (head + alignment - 1) & ~(alignment - 1);

        // If the allocation would wrap around the physical end of the buffer,
        // skip the tail waste and start at the next boundary.
        uint64_t offset = aligned_head % _total_size;
        if (offset + aligned_size > _total_size) {
            uint64_t next_boundary = ((aligned_head / _total_size) + 1) * _total_size;
            aligned_head = next_boundary;
            offset = 0;
        }

        if (aligned_head + aligned_size <= tail + _total_size) {
            // Enough space
            _head.store(aligned_head + aligned_size, std::memory_order_release);

            Allocation alloc{};
            alloc.buffer = _buffer;
            alloc.offset = offset;
            alloc.size = size;
            alloc.mapped_ptr = static_cast<uint8_t*>(_mapped_ptr) + offset;
            alloc.fence_value = advance_fence();

            _outstanding.push_back({aligned_head + aligned_size, alloc.fence_value});
            return alloc;
        }

        // Not enough space, wait for GPU
        lock.unlock();
        std::this_thread::yield();
        lock.lock();
    }
}

StagingRingBuffer::Allocation StagingRingBuffer::try_allocate(uint64_t size, uint64_t alignment) {
    if (size > _total_size) [[unlikely]] {
        return Allocation{};
    }

    alignment = std::max<uint64_t>(alignment, 256);
    uint64_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

    std::lock_guard<std::mutex> lock(_mutex);
    uint64_t head = _head.load(std::memory_order_relaxed);
    uint64_t tail = _tail.load(std::memory_order_relaxed);

    uint64_t aligned_head = (head + alignment - 1) & ~(alignment - 1);
    uint64_t offset = aligned_head % _total_size;
    if (offset + aligned_size > _total_size) {
        uint64_t next_boundary = ((aligned_head / _total_size) + 1) * _total_size;
        aligned_head = next_boundary;
        offset = 0;
    }

    if (aligned_head + aligned_size <= tail + _total_size) {
        _head.store(aligned_head + aligned_size, std::memory_order_release);

        Allocation alloc{};
        alloc.buffer = _buffer;
        alloc.offset = offset;
        alloc.size = size;
        alloc.mapped_ptr = static_cast<uint8_t*>(_mapped_ptr) + offset;
        alloc.fence_value = advance_fence();

        _outstanding.push_back({aligned_head + aligned_size, alloc.fence_value});
        return alloc;
    }

    return Allocation{};
}

void StagingRingBuffer::finish(uint64_t fence_value) {
    std::lock_guard<std::mutex> lock(_mutex);

    // Advance tail past all completed fences.
    // Allocations are added in fence order, so completed entries form a prefix.
    uint64_t new_tail = _tail.load(std::memory_order_relaxed);
    while (!_outstanding.empty() && _outstanding.front().fence_value <= fence_value) {
        new_tail = _outstanding.front().end;
        _outstanding.pop_front();
    }
    _tail.store(new_tail, std::memory_order_release);
}

uint64_t StagingRingBuffer::advance_fence() {
    return _fence_counter.fetch_add(1, std::memory_order_acq_rel) + 1;
}

} // namespace ggml_vk
