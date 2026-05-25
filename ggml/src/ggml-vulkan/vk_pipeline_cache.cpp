#include "vk_pipeline_cache.h"

#include <fstream>
#include <vector>
#include <cstring>

namespace ggml_vk {

PipelineCache::PipelineCache(vk::Device device,
                             vk::PhysicalDeviceProperties const& props,
                             const std::string& cache_file_path)
    : _device(device)
    , _cache_file_path(cache_file_path) {

    // Build a header to validate cache compatibility
    VkPipelineCacheHeaderVersionOne header{};
    header.headerSize = sizeof(VkPipelineCacheHeaderVersionOne);
    header.headerVersion = VK_PIPELINE_CACHE_HEADER_VERSION_ONE;
    header.vendorID = props.vendorID;
    header.deviceID = props.deviceID;
    memcpy(header.pipelineCacheUUID, props.pipelineCacheUUID, VK_UUID_SIZE);

    // Try to load cache from disk
    std::vector<uint8_t> cache_data;
    {
        std::ifstream file(cache_file_path, std::ios::binary | std::ios::ate);
        if (file.good()) {
            std::streamsize size = file.tellg();
            if (size > static_cast<std::streamsize>(sizeof(VkPipelineCacheHeaderVersionOne))) {
                file.seekg(0);
                cache_data.resize(static_cast<size_t>(size));
                file.read(reinterpret_cast<char*>(cache_data.data()), size);

                // Validate header
                auto* file_header = reinterpret_cast<VkPipelineCacheHeaderVersionOne*>(cache_data.data());
                if (memcmp(file_header, &header, sizeof(header)) != 0) {
                    // Incompatible cache; discard
                    cache_data.clear();
                }
            }
        }
    }

    vk::PipelineCacheCreateInfo create_info{};
    if (!cache_data.empty()) {
        create_info.initialDataSize = cache_data.size();
        create_info.pInitialData = cache_data.data();
    }

    _cache = _device.createPipelineCache(create_info);
}

PipelineCache::~PipelineCache() {
    if (_cache) {
        save();
        _device.destroyPipelineCache(_cache);
    }
}

void PipelineCache::save() {
    if (!_cache || _cache_file_path.empty()) return;

    auto data = _device.getPipelineCacheData(_cache);
    if (data.empty()) return;

    // Prepend header for compatibility validation
    std::ofstream file(_cache_file_path, std::ios::binary | std::ios::trunc);
    if (file.good()) {
        file.write(reinterpret_cast<const char*>(data.data()),
                   static_cast<std::streamsize>(data.size()));
    }
}

void PipelineCache::merge(vk::ArrayProxy<const vk::PipelineCache> caches) {
    if (caches.empty()) return;
    std::vector<VkPipelineCache> vk_caches;
    vk_caches.reserve(caches.size());
    for (auto& c : caches) {
        vk_caches.push_back(c);
    }
    _device.mergePipelineCaches(_cache, vk_caches);
}

} // namespace ggml_vk
