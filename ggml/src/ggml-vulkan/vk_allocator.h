#pragma once

#include <vulkan/vulkan.hpp>

struct VmaAllocator_T;
typedef VmaAllocator_T* VmaAllocator;
struct VmaAllocation_T;
typedef VmaAllocation_T* VmaAllocation;

namespace ggml_vk {

struct AllocatedBuffer {
    vk::Buffer buffer{};
    VmaAllocation allocation{};
};

struct AllocatedImage {
    vk::Image image{};
    VmaAllocation allocation{};
};

enum class AccessType {
    kNone,
    kUpload,
    kReadBack
};

class VkAllocator {
public:
    VkAllocator(vk::Instance instance,
                vk::PhysicalDevice physical_device,
                vk::Device device,
                bool buffer_device_address_enabled);
    ~VkAllocator();

    VkAllocator(const VkAllocator&) = delete;
    VkAllocator& operator=(const VkAllocator&) = delete;
    VkAllocator(VkAllocator&&) noexcept;
    VkAllocator& operator=(VkAllocator&&) noexcept;

    [[nodiscard]] AllocatedBuffer allocate_buffer(size_t byte_size,
                                                   vk::BufferUsageFlags usage,
                                                   AccessType access);
    void destroy_buffer(AllocatedBuffer const& buf);
    [[nodiscard]] AllocatedImage allocate_image(vk::ImageType dimension,
                                                 vk::Format format,
                                                 uint32_t width, uint32_t height, uint32_t depth,
                                                 uint32_t mip_levels,
                                                 vk::ImageUsageFlags usage);
    void destroy_image(AllocatedImage const& img);

    [[nodiscard]] VmaAllocator handle() const { return _allocator; }

    size_t defragment();

private:
    VmaAllocator _allocator{};
};

} // namespace ggml_vk
