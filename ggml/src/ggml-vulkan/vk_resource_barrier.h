#pragma once

#include <vulkan/vulkan.hpp>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace ggml_vk {

// Tracks resource states across dispatches within a single command buffer.
// Automatically inserts VkPipelineBarrier2 between incompatible access patterns.
// Enables single-submission graph execution (one vkQueueSubmit per graph).
class ResourceBarrier {
public:
    // How a resource is being used in a dispatch
    enum class Usage : uint32_t {
        kComputeRead,        // storage buffer read in compute
        kComputeWrite,       // storage buffer write in compute
        kCopySource,          // transfer read
        kCopyDest,            // transfer write
        kHostRead,            // host read access
        kHostWrite,           // host write access
    };

    struct BufferState {
        VkPipelineStageFlags2 before_stage{0};
        VkAccessFlags2 before_access{0};
        VkPipelineStageFlags2 after_stage{0};
        VkAccessFlags2 after_access{0};
        uint64_t before_size{0};
        uint64_t after_size{0};
        bool written{false};
    };

    ResourceBarrier() = default;
    ~ResourceBarrier() = default;

    // Record a usage for a buffer (identified by VkBuffer + offset + size).
    // Call once per argument per dispatch before cmd is recorded.
    void record(VkBuffer buffer, uint64_t offset, uint64_t size, Usage usage);

    // Emit all pending barriers via vkCmdPipelineBarrier2.
    // Call between dispatches inside the command buffer.
    void emit(VkCommandBuffer cmd_buffer);

    // Reset all tracked state (call at start of each graph).
    void reset();

    // Number of recorded entries
    [[nodiscard]] size_t tracked_count() const { return _states.size(); }

private:
    // Buffer key: buffer handle + exact offset
    struct BufferKey {
        VkBuffer buffer;
        uint64_t offset;
        bool operator==(const BufferKey& other) const {
            return buffer == other.buffer && offset == other.offset;
        }
    };
    struct BufferKeyHash {
        size_t operator()(const BufferKey& k) const {
            return std::hash<VkBuffer>{}(k.buffer) ^ std::hash<uint64_t>{}(k.offset);
        }
    };

    std::unordered_map<BufferKey, BufferState, BufferKeyHash> _states;
    std::vector<BufferKey> _ordered_keys; // maintain insert order for barrier emission
};

} // namespace ggml_vk
