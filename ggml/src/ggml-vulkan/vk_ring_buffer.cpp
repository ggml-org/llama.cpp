#include "vk_ring_buffer.h"
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
        vmaMapMemory(_allocator->handle(), _vma_allocation, &_mapped_ptr);
    }
}

StagingRingBuffer::~StagingRingBuffer() {
    if (_mapped_ptr) {
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

    // Spin until enough space
    while (true) {
        uint64_t head = _head.load(std::memory_order_acquire);
        uint64_t tail = _tail.load(std::memory_order_acquire);

        // Align head
        uint64_t aligned_head = (head + alignment - 1) & ~(alignment - 1);

        if (aligned_head + aligned_size <= _total_size) {
            // Fits in remaining space
            uint64_t new_head = aligned_head + aligned_size;
            if (_head.compare_exchange_weak(head, new_head,
                                            std::memory_order_release,
                                            std::memory_order_relaxed)) {
                Allocation alloc{};
                alloc.buffer = _buffer;
                alloc.offset = aligned_head;
                alloc.size = size;
                alloc.mapped_ptr = static_cast<uint8_t*>(_mapped_ptr) + aligned_head;
                return alloc;
            }
        } else if (tail >= aligned_size) {
            // Wrap around
            uint64_t new_head = aligned_size;
            if (_head.compare_exchange_weak(head, new_head,
                                            std::memory_order_release,
                                            std::memory_order_relaxed)) {
                Allocation alloc{};
                alloc.buffer = _buffer;
                alloc.offset = 0;
                alloc.size = size;
                alloc.mapped_ptr = _mapped_ptr;
                return alloc;
            }
        } else {
            // Not enough space, wait for GPU
            std::this_thread::yield();
        }
    }
}

void StagingRingBuffer::finish(uint64_t fence_value) {
    std::lock_guard<std::mutex> lock(_mutex);

    // Advance tail past all completed fences
    uint64_t new_tail = _tail.load();
    for (auto it = _outstanding.begin(); it != _outstanding.end(); ) {
        if (it->fence_value <= fence_value) {
            new_tail = std::max(new_tail, it->offset + it->size);
            it = _outstanding.erase(it);
        } else {
            ++it;
        }
    }
    _tail.store(new_tail, std::memory_order_release);
}

uint64_t StagingRingBuffer::advance_fence() {
    return _fence_counter.fetch_add(1, std::memory_order_acq_rel) + 1;
}

} // namespace ggml_vk
