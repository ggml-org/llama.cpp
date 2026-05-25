#pragma once

#include <vulkan/vulkan.hpp>
#include <string>
#include <vector>
#include <cstdint>

namespace ggml_vk {

// Persistent VkPipelineCache wrapper.
// Caches compiled PSOs to disk to avoid recompilation on subsequent launches.
class PipelineCache {
public:
    PipelineCache(vk::Device device,
                  vk::PhysicalDeviceProperties const& props,
                  const std::string& cache_file_path);
    ~PipelineCache();

    PipelineCache(const PipelineCache&) = delete;
    PipelineCache& operator=(const PipelineCache&) = delete;

    // Returns the VkPipelineCache handle for use in pipeline creation.
    [[nodiscard]] vk::PipelineCache handle() const { return _cache; }

    // Explicit save (also auto-saved on destruction).
    void save();

    // Merge external cache data (e.g., from a different run or bundled cache).
    void merge(vk::ArrayProxy<const vk::PipelineCache> caches);

private:
    vk::Device _device;
    std::string _cache_file_path;
    vk::PipelineCache _cache{};
};

} // namespace ggml_vk
