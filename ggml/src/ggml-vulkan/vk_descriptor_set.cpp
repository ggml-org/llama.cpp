#include "vk_descriptor_set.h"
#include <stdexcept>

namespace ggml_vk {

DescriptorManager::DescriptorManager(vk::Device device,
                                     uint32_t max_bindless_buffers,
                                     uint32_t max_per_dispatch_sets)
    : _device(device)
    , _max_bindless_buffers(max_bindless_buffers) {

    // -- Create bindless descriptor pool and set --
    {
        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = max_bindless_buffers;

        VkDescriptorPoolCreateInfo pool_ci{};
        pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_ci.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
        pool_ci.maxSets = 1;
        pool_ci.poolSizeCount = 1;
        pool_ci.pPoolSizes = &pool_size;

        VkDescriptorPool vk_pool;
        if (vkCreateDescriptorPool(device, &pool_ci, nullptr, &vk_pool) != VK_SUCCESS) {
            throw std::runtime_error("DescriptorManager: failed to create bindless pool");
        }
        _bindless_pool = vk_pool;

        // Bindless layout
        VkDescriptorBindingFlags binding_flags =
            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
            VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;

        VkDescriptorSetLayoutBindingFlagsCreateInfo flags_ci{};
        flags_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        flags_ci.bindingCount = 1;
        flags_ci.pBindingFlags = &binding_flags;

        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = max_bindless_buffers;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layout_ci{};
        layout_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_ci.pNext = &flags_ci;
        layout_ci.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
        layout_ci.bindingCount = 1;
        layout_ci.pBindings = &binding;

        VkDescriptorSetLayout vk_layout;
        if (vkCreateDescriptorSetLayout(device, &layout_ci, nullptr, &vk_layout) != VK_SUCCESS) {
            throw std::runtime_error("DescriptorManager: failed to create bindless layout");
        }
        _bindless_layout = vk_layout;

        VkDescriptorSetAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = _bindless_pool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &vk_layout;

        VkDescriptorSet vk_set;
        if (vkAllocateDescriptorSets(device, &alloc_info, &vk_set) != VK_SUCCESS) {
            throw std::runtime_error("DescriptorManager: failed to allocate bindless set");
        }
        _bindless_set = vk_set;
    }

    // Pre-create first per-dispatch pool
    ensure_pool();
}

DescriptorManager::~DescriptorManager() {
    for (auto& pool : _pools) {
        if (pool) {
            vkDestroyDescriptorPool(_device, pool, nullptr);
        }
    }
    if (_bindless_set) {
        // Sets are freed implicitly when pool is destroyed
        vkDestroyDescriptorPool(_device, _bindless_pool, nullptr);
    }
    if (_bindless_layout) {
        vkDestroyDescriptorSetLayout(_device, _bindless_layout, nullptr);
    }
}

uint32_t DescriptorManager::register_buffer(vk::Buffer buffer,
                                            uint64_t offset,
                                            uint64_t range) {
    uint32_t index = _bindless_count++;
    if (index >= _max_bindless_buffers) {
        throw std::runtime_error("DescriptorManager: bindless buffer limit exceeded");
    }

    VkDescriptorBufferInfo buf_info{};
    buf_info.buffer = buffer;
    buf_info.offset = offset;
    buf_info.range = range;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = _bindless_set;
    write.dstBinding = 0;
    write.dstArrayElement = index;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &buf_info;

    vkUpdateDescriptorSets(_device, 1, &write, 0, nullptr);
    return index;
}

void DescriptorManager::update_buffer(uint32_t index,
                                       vk::Buffer buffer,
                                       uint64_t offset,
                                       uint64_t range) {
    VkDescriptorBufferInfo buf_info{};
    buf_info.buffer = buffer;
    buf_info.offset = offset;
    buf_info.range = range;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = _bindless_set;
    write.dstBinding = 0;
    write.dstArrayElement = index;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &buf_info;

    vkUpdateDescriptorSets(_device, 1, &write, 0, nullptr);
}

vk::DescriptorSet DescriptorManager::allocate_set(vk::DescriptorSetLayout layout) {
    ensure_pool();

    VkDescriptorSetLayout vk_layout = layout;
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = _pools[_current_pool_idx];
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &vk_layout;

    VkDescriptorSet vk_set;
    VkResult res = vkAllocateDescriptorSets(_device, &alloc_info, &vk_set);
    if (res == VK_ERROR_OUT_OF_POOL_MEMORY) {
        // Move to next pool
        _current_pool_idx++;
        ensure_pool();
        alloc_info.descriptorPool = _pools[_current_pool_idx];
        res = vkAllocateDescriptorSets(_device, &alloc_info, &vk_set);
    }
    if (res != VK_SUCCESS) {
        throw std::runtime_error("DescriptorManager: failed to allocate descriptor set");
    }

    _sets_allocated_in_current_pool++;
    return vk_set;
}

void DescriptorManager::write_buffers(vk::DescriptorSet set,
                                       uint32_t first_binding,
                                       vk::ArrayProxy<const vk::DescriptorBufferInfo> buffer_infos) {
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(buffer_infos.size());

    for (uint32_t i = 0; i < buffer_infos.size(); i++) {
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = first_binding + i;
        write.dstArrayElement = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = reinterpret_cast<const VkDescriptorBufferInfo*>(&buffer_infos.data()[i]);
        writes.push_back(write);
    }

    if (!writes.empty()) {
        vkUpdateDescriptorSets(_device,
                               static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }
}

void DescriptorManager::reset_pools() {
    // Reset all pools for reuse
    for (auto& pool : _pools) {
        if (pool) {
            vkResetDescriptorPool(_device, pool, 0);
        }
    }
    _current_pool_idx = 0;
    _sets_allocated_in_current_pool = 0;
}

void DescriptorManager::ensure_pool() {
    while (_current_pool_idx >= _pools.size()) {
        VkDescriptorPoolSize pool_sizes[2];
        pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_sizes[0].descriptorCount = kMaxDescriptorsPerPool;
        pool_sizes[1].type = VK_DESCRIPTOR_TYPE_SAMPLER;
        pool_sizes[1].descriptorCount = 64;

        VkDescriptorPoolCreateInfo pool_ci{};
        pool_ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_ci.maxSets = kMaxSetsPerPool;
        pool_ci.poolSizeCount = 2;
        pool_ci.pPoolSizes = pool_sizes;

        VkDescriptorPool pool;
        if (vkCreateDescriptorPool(_device, &pool_ci, nullptr, &pool) != VK_SUCCESS) {
            throw std::runtime_error("DescriptorManager: failed to create pool");
        }
        _pools.push_back(pool);
    }
    _sets_allocated_in_current_pool = 0;
}

} // namespace ggml_vk
