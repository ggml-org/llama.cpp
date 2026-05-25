#include "vk_resource_barrier.h"
#include <algorithm>
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

static constexpr VkAccessFlags2 kWriteAccess =
    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
    VK_ACCESS_2_TRANSFER_WRITE_BIT |
    VK_ACCESS_2_HOST_WRITE_BIT;

} // namespace

void ResourceBarrier::record(VkBuffer buffer, uint64_t offset, uint64_t size, Usage usage) {
    BufferKey key{buffer, offset};

    auto& state = _states[key];
    VkPipelineStageFlags2 stage = usage_to_stage(usage);
    VkAccessFlags2 access = usage_to_access(usage);
    bool is_write = (access & kWriteAccess) != 0;

    if (state.after_stage == 0 && state.before_stage == 0) {
        // First time seeing this buffer in the current graph
        _ordered_keys.push_back(key);
    }

    state.after_stage |= stage;
    state.after_access |= access;
    state.written = state.written || is_write;
    state.after_size = std::max(state.after_size, size);
}

void ResourceBarrier::emit(VkCommandBuffer cmd_buffer) {
    if (_states.empty()) return;

    std::vector<VkBufferMemoryBarrier2> barriers;
    barriers.reserve(_ordered_keys.size());

    for (const auto& key : _ordered_keys) {
        auto& state = _states[key];

        bool prev_has_access = state.before_stage != 0 || state.before_access != 0;
        bool curr_has_access = state.after_stage != 0 || state.after_access != 0;
        bool prev_is_write = (state.before_access & kWriteAccess) != 0;
        bool curr_is_write = state.written;
        bool needs_barrier = (prev_is_write || curr_is_write) && prev_has_access && curr_has_access;

        if (needs_barrier) {
            VkBufferMemoryBarrier2 barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
            barrier.srcStageMask = state.before_stage;
            barrier.srcAccessMask = state.before_access;
            barrier.dstStageMask = state.after_stage;
            barrier.dstAccessMask = state.after_access;
            barrier.buffer = key.buffer;
            barrier.offset = key.offset;
            barrier.size = std::max(state.before_size, state.after_size);
            if (barrier.size > 0) {
                barriers.push_back(barrier);
            }
        }

        // Promote current after state to before state for the next emit
        state.before_stage = state.after_stage;
        state.before_access = state.after_access;
        state.before_size = state.after_size;
        state.after_stage = 0;
        state.after_access = 0;
        state.after_size = 0;
        state.written = false;
    }

    if (!barriers.empty()) {
        VkDependencyInfo dep_info{};
        dep_info.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dep_info.bufferMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
        dep_info.pBufferMemoryBarriers = barriers.data();

        vkCmdPipelineBarrier2(cmd_buffer, &dep_info);
    }
}

void ResourceBarrier::reset() {
    _states.clear();
    _ordered_keys.clear();
}

} // namespace ggml_vk
