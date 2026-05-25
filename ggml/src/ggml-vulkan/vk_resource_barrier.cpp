#include "vk_resource_barrier.h"
#include <cstring>

namespace ggml_vk {

namespace {

VkPipelineStageFlags2 usage_to_stage(ResourceBarrier::Usage usage) {
    switch (usage) {
        case ResourceBarrier::Usage::kComputeRead:
            return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        case ResourceBarrier::Usage::kComputeWrite:
            return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        case ResourceBarrier::Usage::kCopySource:
            return VK_PIPELINE_STAGE_2_COPY_BIT;
        case ResourceBarrier::Usage::kCopyDest:
            return VK_PIPELINE_STAGE_2_COPY_BIT;
        case ResourceBarrier::Usage::kHostRead:
            return VK_PIPELINE_STAGE_2_HOST_BIT;
        case ResourceBarrier::Usage::kHostWrite:
            return VK_PIPELINE_STAGE_2_HOST_BIT;
        default:
            return VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    }
}

VkAccessFlags2 usage_to_access(ResourceBarrier::Usage usage) {
    switch (usage) {
        case ResourceBarrier::Usage::kComputeRead:
            return VK_ACCESS_2_SHADER_READ_BIT;
        case ResourceBarrier::Usage::kComputeWrite:
            return VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        case ResourceBarrier::Usage::kCopySource:
            return VK_ACCESS_2_TRANSFER_READ_BIT;
        case ResourceBarrier::Usage::kCopyDest:
            return VK_ACCESS_2_TRANSFER_WRITE_BIT;
        case ResourceBarrier::Usage::kHostRead:
            return VK_ACCESS_2_HOST_READ_BIT;
        case ResourceBarrier::Usage::kHostWrite:
            return VK_ACCESS_2_HOST_WRITE_BIT;
        default:
            return VK_ACCESS_2_SHADER_READ_BIT;
    }
}

} // namespace

void ResourceBarrier::record(VkBuffer buffer, uint64_t offset, uint64_t size, Usage usage) {
    // Round offset down to alignment for tracking granularity
    uint64_t aligned_offset = offset & ~uint64_t(255);
    BufferKey key{buffer, aligned_offset};

    auto& state = _states[key];
    VkPipelineStageFlags2 new_stage = usage_to_stage(usage);
    VkAccessFlags2 new_access = usage_to_access(usage);
    bool is_write = (usage == Usage::kComputeWrite ||
                     usage == Usage::kCopyDest ||
                     usage == Usage::kHostWrite);

    if (state.stage == 0) {
        // First time seeing this buffer
        _ordered_keys.push_back(key);
        state.stage = new_stage;
        state.access = new_access;
        state.written = is_write;
    } else {
        // Combine: we need barrier if prior was write OR new is write
        state.stage |= new_stage;
        state.access |= new_access;
        state.written = state.written || is_write;
    }
}

void ResourceBarrier::emit(VkCommandBuffer cmd_buffer) {
    if (_states.empty()) return;

    // For simplicity, we emit a single full barrier per recorded buffer.
    // A more advanced implementation would track per-buffer transitions
    // and only emit barriers for actual conflicts.
    std::vector<VkBufferMemoryBarrier2> barriers;
    barriers.reserve(_ordered_keys.size());

    for (const auto& key : _ordered_keys) {
        const auto& state = _states[key];

        // If this buffer was written, ensure all subsequent reads see the write
        if (state.written) {
            VkBufferMemoryBarrier2 barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                   VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT |
                                    VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                   VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                                   VK_PIPELINE_STAGE_2_HOST_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT |
                                    VK_ACCESS_2_SHADER_WRITE_BIT |
                                    VK_ACCESS_2_TRANSFER_READ_BIT |
                                    VK_ACCESS_2_HOST_READ_BIT;
            barrier.buffer = key.buffer;
            barrier.offset = key.offset;
            barrier.size = VK_WHOLE_SIZE;
            barriers.push_back(barrier);
        }
    }

    if (!barriers.empty()) {
        VkDependencyInfo dep_info{};
        dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep_info.bufferMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
        dep_info.pBufferMemoryBarriers = barriers.data();

        vkCmdPipelineBarrier2(cmd_buffer, &dep_info);
    }

    // Reset for next batch
    reset();
}

void ResourceBarrier::reset() {
    _states.clear();
    _ordered_keys.clear();
}

} // namespace ggml_vk
