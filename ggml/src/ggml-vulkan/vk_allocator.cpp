#define VMA_STATIC_VULKAN_FUNCTIONS 1
#define VMA_IMPLEMENTATION 1
#include "vk_mem_alloc.h"

#include "vk_allocator.h"
#include <vulkan/vulkan.hpp>

namespace ggml_vk {

VkAllocator::VkAllocator(vk::Instance instance,
                         vk::PhysicalDevice physical_device,
                         vk::Device device,
                         bool buffer_device_address_enabled)
    : _allocator(nullptr) {

    VmaAllocatorCreateInfo create_info{};
    create_info.flags = buffer_device_address_enabled
                            ? VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT
                            : 0;
    create_info.physicalDevice = physical_device;
    create_info.device = device;
    create_info.instance = instance;
    create_info.vulkanApiVersion = VK_API_VERSION_1_3;
    create_info.preferredLargeHeapBlockSize = 0;
    create_info.pAllocationCallbacks = nullptr;
    create_info.pDeviceMemoryCallbacks = nullptr;

    if (vmaCreateAllocator(&create_info, &_allocator) != VK_SUCCESS) {
        throw std::runtime_error("VkAllocator: failed to create VMA allocator");
    }
}

VkAllocator::~VkAllocator() {
    if (_allocator) {
        vmaDestroyAllocator(_allocator);
    }
}

VkAllocator::VkAllocator(VkAllocator&& other) noexcept
    : _allocator(other._allocator) {
    other._allocator = nullptr;
}

VkAllocator& VkAllocator::operator=(VkAllocator&& other) noexcept {
    if (this != &other) {
        if (_allocator) {
            vmaDestroyAllocator(_allocator);
        }
        _allocator = other._allocator;
        other._allocator = nullptr;
    }
    return *this;
}

AllocatedBuffer VkAllocator::allocate_buffer(size_t byte_size,
                                              vk::BufferUsageFlags usage,
                                              AccessType access) {
    if (byte_size == 0) {
        return {};
    }

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = byte_size;
    buffer_info.usage = static_cast<VkBufferUsageFlags>(usage);

    VmaAllocationCreateInfo alloc_info{};
    alloc_info.flags = VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT;
    switch (access) {
        case AccessType::kReadBack:
            alloc_info.usage = VMA_MEMORY_USAGE_GPU_TO_CPU;
            break;
        case AccessType::kUpload:
            alloc_info.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
            break;
        case AccessType::kNone:
        default:
            alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            break;
    }

    AllocatedBuffer result;
    VkResult res = vmaCreateBuffer(
        _allocator, &buffer_info, &alloc_info,
        &reinterpret_cast<VkBuffer&>(result.buffer),
        &result.allocation, nullptr);
    if (res != VK_SUCCESS) {
        throw vk::OutOfDeviceMemoryError("VMA allocate_buffer failed");
    }
    return result;
}

void VkAllocator::destroy_buffer(AllocatedBuffer const& buf) {
    if (buf.buffer) {
        vmaDestroyBuffer(_allocator, buf.buffer, buf.allocation);
    }
}

AllocatedImage VkAllocator::allocate_image(vk::ImageType dimension,
                                            vk::Format format,
                                            uint32_t width, uint32_t height, uint32_t depth,
                                            uint32_t mip_levels,
                                            vk::ImageUsageFlags usage) {
    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = static_cast<VkImageType>(dimension);
    image_info.format = static_cast<VkFormat>(format);
    image_info.extent = VkExtent3D{width, height, depth};
    image_info.mipLevels = mip_levels;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.usage = static_cast<VkImageUsageFlags>(usage);
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo alloc_info{};
    alloc_info.flags = VMA_ALLOCATION_CREATE_STRATEGY_BEST_FIT_BIT;
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    AllocatedImage result;
    VkResult res = vmaCreateImage(
        _allocator, &image_info, &alloc_info,
        &reinterpret_cast<VkImage&>(result.image),
        &result.allocation, nullptr);
    if (res != VK_SUCCESS) {
        throw vk::OutOfDeviceMemoryError("VMA allocate_image failed");
    }
    return result;
}

void VkAllocator::destroy_image(AllocatedImage const& img) {
    if (img.image) {
        vmaDestroyImage(_allocator, img.image, img.allocation);
    }
}

size_t VkAllocator::defragment() {
    if (!_allocator) return 0;
    // Run defragmentation if possible
    size_t bytes_moved = 0;
    vmaDefragmentationBegin(_allocator, nullptr);
    return bytes_moved;
}

} // namespace ggml_vk
