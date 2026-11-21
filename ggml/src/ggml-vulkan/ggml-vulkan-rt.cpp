#include "ggml-vulkan-common.h"

#include <cstdlib>

struct vk_rt_kv_push_constants {
    uint32_t kv, hsk, hsv, page_size, pages, window;
    uint32_t k_stride, k_head_stride, v_stride, v_head_stride, q_head_stride;
    float scale, error_budget, padding;
    uint32_t exact_bounds, dense_mode, light_summary;
    uint32_t block_cols, cache_summary, unit_box;
};

static_assert(sizeof(vk_rt_kv_push_constants) == 80, "RT KV push constants");

static void ggml_vk_rt_check(VkResult result, const char * name) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(name) + ": " + vk::to_string(static_cast<vk::Result>(result)));
    }
}

struct vk_rt_kv_scene {
    vk_device device;
    PFN_vkDestroyAccelerationStructureKHR destroy_as {};
    uint32_t capacity {}, hsk {}, hsv {}, heads {}, page_size {}, pages {};
    vk_buffer bounds, metadata, indices, instances, scratch;
    vk_buffer vertices, triangles, query_state;
    std::vector<vk_buffer> storage;
    std::vector<VkAccelerationStructureKHR> bottom;
    VkAccelerationStructureKHR top {};
    bool built {};
    vk_buffer_ref source;
    uint64_t source_offset {}, source_generation {};
    uint32_t k_stride {}, k_head_stride {};
    uint32_t query_count {};
    vk_buffer summary_state;
    bool summary_initialized {}, invalidated {};

    ~vk_rt_kv_scene() {
        if (top) {
            destroy_as(device->device, top, nullptr);
        }
        for (auto as : bottom) {
            destroy_as(device->device, as, nullptr);
        }
    }
};

struct vk_rt_kv_context {
    vk_device device;
    PFN_vkCreateAccelerationStructureKHR create_as {};
    PFN_vkDestroyAccelerationStructureKHR destroy_as {};
    PFN_vkGetAccelerationStructureBuildSizesKHR get_sizes {};
    PFN_vkCmdBuildAccelerationStructuresKHR build_as {};
    PFN_vkGetAccelerationStructureDeviceAddressKHR as_address {};
    uint32_t scratch_alignment {};
    static constexpr uint32_t query_threads = 128;
    uint32_t page_size {}, window {128};
    float error_budget {1.0e-4f}, padding {0.3f};
    uint32_t exact_bounds {};
    uint32_t approximate {}, dense_mode {}, light_summary {}, refit {};
    uint32_t tile_mask {}, unit_box {};
    uint64_t max_primitives {}, max_instances {};
    bool use_subgroups {};
    vk::DescriptorSetLayout descriptor_layout;
    vk::PipelineLayout pipeline_layout;
    vk::ShaderModule summary_module, select_module;
    std::map<std::array<uint32_t, 4>, std::array<vk::Pipeline, 4>> pipelines;
    std::vector<vk::DescriptorPool> pools;
    std::vector<vk::DescriptorSet> sets;
    uint32_t set_idx {};
    vk_buffer stats;
    std::vector<std::unique_ptr<vk_rt_kv_scene>> scenes;
    std::unique_ptr<vk_rt_kv_scene> unit_scene;

    ~vk_rt_kv_context() {
        scenes.clear();
        for (auto pool : pools) {
            device->device.destroyDescriptorPool(pool);
        }
        for (auto & entry : pipelines) {
            for (auto pipeline : entry.second) {
                if (pipeline) { device->device.destroyPipeline(pipeline); }
            }
        }
        device->device.destroyShaderModule(summary_module);
        device->device.destroyShaderModule(select_module);
        device->device.destroyPipelineLayout(pipeline_layout);
        device->device.destroyDescriptorSetLayout(descriptor_layout);
    }
};

static uint32_t ggml_vk_rt_uint(const char * name, uint32_t value, uint32_t limit) {
    const char * text = getenv(name);
    if (!text) {
        return value;
    }
    char * end = nullptr;
    unsigned long parsed = strtoul(text, &end, 10);
    if (end == text || *end != '\0' || parsed > limit) {
        throw std::runtime_error(std::string("Invalid ") + name);
    }
    return static_cast<uint32_t>(parsed);
}

static float ggml_vk_rt_float(const char * name, float value, float limit) {
    const char * text = getenv(name);
    if (!text) {
        return value;
    }
    char * end = nullptr;
    float parsed = strtof(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(parsed) || parsed < 0.0f || parsed > limit) {
        throw std::runtime_error(std::string("Invalid ") + name);
    }
    return parsed;
}

static vk_buffer ggml_vk_rt_buffer(vk_device & device, size_t size, vk::BufferUsageFlags usage = {}) {
    return ggml_vk_create_buffer_check(device, size, vk::MemoryPropertyFlagBits::eDeviceLocal, {}, usage);
}

static void ggml_vk_rt_pipeline(vk_rt_kv_context & rt, const void * data, size_t size, vk::ShaderModule & module, vk::Pipeline & pipeline, const std::array<uint32_t, 6> & constants) {
    if (!module) {
        module = rt.device->device.createShaderModule(vk::ShaderModuleCreateInfo({}, size, static_cast<const uint32_t *>(data)));
    }
    std::array<vk::SpecializationMapEntry, 6> entries;
    for (uint32_t i = 0; i < entries.size(); ++i) {
        entries[i] = vk::SpecializationMapEntry(i, i * uint32_t(sizeof(uint32_t)), sizeof(uint32_t));
    }
    const vk::SpecializationInfo local_info(uint32_t(entries.size()), entries.data(), sizeof(constants), constants.data());
    const vk::PipelineShaderStageCreateInfo stage({}, vk::ShaderStageFlagBits::eCompute, module, "main", &local_info);
    pipeline = rt.device->device.createComputePipeline({}, vk::ComputePipelineCreateInfo({}, stage, rt.pipeline_layout)).value;
}

static std::shared_ptr<vk_rt_kv_context> ggml_vk_rt_context(vk_device & device) {
    auto rt = std::make_shared<vk_rt_kv_context>();
    rt->device = device;
    rt->create_as = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(vkGetDeviceProcAddr(device->device, "vkCreateAccelerationStructureKHR"));
    rt->destroy_as = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(vkGetDeviceProcAddr(device->device, "vkDestroyAccelerationStructureKHR"));
    rt->get_sizes = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(vkGetDeviceProcAddr(device->device, "vkGetAccelerationStructureBuildSizesKHR"));
    rt->build_as = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(vkGetDeviceProcAddr(device->device, "vkCmdBuildAccelerationStructuresKHR"));
    rt->as_address = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(vkGetDeviceProcAddr(device->device, "vkGetAccelerationStructureDeviceAddressKHR"));
    if (!rt->create_as || !rt->destroy_as || !rt->get_sizes || !rt->build_as || !rt->as_address) {
        throw std::runtime_error("RT KV acceleration structure functions are unavailable");
    }

    VkPhysicalDeviceAccelerationStructurePropertiesKHR properties {};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 properties2 {};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &properties;
    vkGetPhysicalDeviceProperties2(device->physical_device, &properties2);
    rt->scratch_alignment = properties.minAccelerationStructureScratchOffsetAlignment;
    rt->max_primitives = properties.maxPrimitiveCount;
    rt->max_instances = properties.maxInstanceCount;
    rt->page_size = ggml_vk_rt_uint("GGML_VK_RT_KV_PAGE", 0, 64);
    if (rt->page_size != 0 && rt->page_size != 8 && rt->page_size != 16 && rt->page_size != 32 && rt->page_size != 64) {
        throw std::runtime_error("GGML_VK_RT_KV_PAGE must be 0, 8, 16, 32 or 64");
    }
    rt->window = ggml_vk_rt_uint("GGML_VK_RT_KV_WINDOW", 128, 65536);
    rt->error_budget = ggml_vk_rt_float("GGML_VK_RT_KV_ERROR", 1.0e-4f, 1.0f);
    rt->padding = ggml_vk_rt_float("GGML_VK_RT_KV_PADDING", 0.3f, 0.9f);
    rt->exact_bounds = ggml_vk_rt_uint("GGML_VK_RT_KV_EXACT", 0, 1);
    rt->approximate = ggml_vk_rt_uint("GGML_VK_RT_KV_APPROX", 0, 1);
    rt->dense_mode = ggml_vk_rt_uint("GGML_VK_RT_KV_DENSE", 0, 2);
    rt->light_summary = rt->approximate && !ggml_vk_rt_uint("GGML_VK_RT_KV_FULL_SUMMARY", 0, 1);
    rt->refit = ggml_vk_rt_uint("GGML_VK_RT_KV_REFIT", 0, 1);
    rt->tile_mask = ggml_vk_rt_uint("GGML_VK_RT_KV_MASK", 1, 1);

    std::array<vk::DescriptorSetLayoutBinding, 10> bindings;
    for (uint32_t i = 0; i < 10; ++i) {
        bindings[i] = { i, i == 8 ? vk::DescriptorType::eAccelerationStructureKHR : vk::DescriptorType::eStorageBuffer,
                       1, vk::ShaderStageFlagBits::eCompute };
    }
    rt->descriptor_layout = device->device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo({}, bindings));
    const vk::PushConstantRange range(vk::ShaderStageFlagBits::eCompute, 0, sizeof(vk_rt_kv_push_constants));
    rt->pipeline_layout = device->device.createPipelineLayout(vk::PipelineLayoutCreateInfo({}, rt->descriptor_layout, range));
    rt->use_subgroups = device->subgroup_ballot && device->subgroup_size >= 32 && device->subgroup_size <= rt->query_threads && rt->query_threads % device->subgroup_size == 0;
    rt->unit_box = rt->approximate && rt->use_subgroups && ggml_vk_rt_uint("GGML_VK_RT_KV_UNIT_BOX", 1, 1);

    rt->stats = ggml_vk_rt_buffer(device, 8 * sizeof(uint32_t));
    ggml_vk_buffer_memset(rt->stats, 0, 0, rt->stats->size);
    GGML_LOG_INFO("ggml_vulkan: RT KV page_override=%u window=%u truncation_budget=%g exact=%u approximate=%u dense=%u light=%u refit=%u tile_mask=%u unit_box=%u\n", rt->page_size, rt->window, rt->error_budget, rt->exact_bounds, rt->approximate, rt->dense_mode, rt->light_summary, rt->refit, rt->tile_mask, rt->unit_box);
    return rt;
}

static VkAccelerationStructureKHR ggml_vk_rt_as(vk_rt_kv_context & rt, vk_rt_kv_scene & scene, VkAccelerationStructureBuildGeometryInfoKHR & info, uint32_t count, size_t & scratch_size) {
    VkAccelerationStructureBuildSizesInfoKHR sizes {};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    rt.get_sizes(rt.device->device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &info, &count, &sizes);
    scratch_size = std::max(scratch_size, static_cast<size_t>(std::max(sizes.buildScratchSize, sizes.updateScratchSize)));
    auto storage = ggml_vk_rt_buffer(rt.device, sizes.accelerationStructureSize, vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR);
    scene.storage.push_back(storage);
    VkAccelerationStructureCreateInfoKHR create {};
    create.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    create.type = info.type;
    create.buffer = storage->buffer;
    create.size = sizes.accelerationStructureSize;
    VkAccelerationStructureKHR as {};
    ggml_vk_rt_check(rt.create_as(rt.device->device, &create, nullptr, &as), "RT KV acceleration structure");
    return as;
}

static VkAccelerationStructureGeometryKHR ggml_vk_rt_bottom_geometry(const vk_rt_kv_scene & scene, uint32_t head) {
    VkAccelerationStructureGeometryKHR geometry {};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    if (scene.vertices) {
        geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        auto & triangles = geometry.geometry.triangles;
        triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        triangles.vertexData.deviceAddress = scene.vertices->bda_addr;
        triangles.vertexStride = 3 * sizeof(float);
        triangles.maxVertex = 7;
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress = scene.triangles->bda_addr;
        return geometry;
    }
    geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    geometry.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    geometry.geometry.aabbs.data.deviceAddress = scene.bounds->bda_addr + head * scene.pages * sizeof(VkAabbPositionsKHR);
    geometry.geometry.aabbs.stride = sizeof(VkAabbPositionsKHR);
    return geometry;
}

static VkAccelerationStructureGeometryKHR ggml_vk_rt_top_geometry(const vk_rt_kv_scene & scene) {
    VkAccelerationStructureGeometryKHR geometry {};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.data.deviceAddress = scene.instances->bda_addr;
    return geometry;
}

static VkAccelerationStructureBuildGeometryInfoKHR ggml_vk_rt_build_info(const vk_rt_kv_context & rt, const vk_rt_kv_scene & scene, VkAccelerationStructureTypeKHR type, const VkAccelerationStructureGeometryKHR & geometry) {
    VkAccelerationStructureBuildGeometryInfoKHR info {};
    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    info.type = type;
    info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    if (rt.refit && !scene.vertices) { info.flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR; }
    info.geometryCount = 1;
    info.pGeometries = &geometry;
    return info;
}

static void ggml_vk_rt_allocate_scene(vk_rt_kv_context & rt, vk_rt_kv_scene & scene) {
    const auto build_usage = vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
    scene.instances = ggml_vk_rt_buffer(rt.device, scene.heads * sizeof(VkAccelerationStructureInstanceKHR), build_usage);
    size_t scratch_size = 0;
    std::vector<VkAccelerationStructureInstanceKHR> instances(scene.heads);
    for (uint32_t head = 0; head < scene.heads; ++head) {
        auto geometry = ggml_vk_rt_bottom_geometry(scene, head);
        auto info = ggml_vk_rt_build_info(rt, scene, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, geometry);
        auto as = ggml_vk_rt_as(rt, scene, info, scene.pages, scratch_size);
        scene.bottom.push_back(as);
        VkAccelerationStructureDeviceAddressInfoKHR address {};
        address.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        address.accelerationStructure = as;
        for (uint32_t axis = 0; axis < 3; ++axis) {
            instances[head].transform.matrix[axis][axis] = 1.0f;
        }
        instances[head].instanceCustomIndex = head;
        instances[head].mask = 1u << (head % 8);
        instances[head].accelerationStructureReference = rt.as_address(rt.device->device, &address);
    }
    ggml_vk_buffer_write(scene.instances, 0, instances.data(), scene.heads * sizeof(VkAccelerationStructureInstanceKHR));
    auto geometry = ggml_vk_rt_top_geometry(scene);
    auto info = ggml_vk_rt_build_info(rt, scene, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, geometry);
    scene.top = ggml_vk_rt_as(rt, scene, info, scene.heads, scratch_size);
    scene.scratch = ggml_vk_rt_buffer(rt.device, scratch_size + rt.scratch_alignment);
}

static vk_rt_kv_scene & ggml_vk_rt_scene(vk_rt_kv_context & rt, const ggml_tensor * k, uint32_t hsv, const vk_subbuffer & source, const vk_rt_kv_params & params) {
    const uint32_t hsk = uint32_t(k->ne[0]);
    const uint32_t heads = uint32_t(k->ne[2]);
    for (auto it = rt.scenes.begin(); it != rt.scenes.end();) {
        auto & scene = *it;
        const auto source_buffer = scene->source.lock();
        if (!source_buffer) {
            it = rt.scenes.erase(it);
            continue;
        }
        if (source_buffer == source.buffer && scene->source_offset == source.offset &&
            scene->hsk == hsk && scene->hsv == hsv && scene->heads == heads &&
            scene->k_stride == k->nb[1] && scene->k_head_stride == k->nb[2] && scene->page_size == params.page_size && scene->query_count == params.query_count &&
            scene->capacity == params.capacity && bool(scene->summary_state) == params.cache_summary) {
            return *scene;
        }
        ++it;
    }
    auto scene = std::make_unique<vk_rt_kv_scene>();
    scene->device = rt.device;
    scene->destroy_as = rt.destroy_as;
    scene->source = source.buffer;
    scene->source_offset = source.offset;
    scene->k_stride = uint32_t(k->nb[1]);
    scene->k_head_stride = uint32_t(k->nb[2]);
    scene->capacity = params.capacity;
    scene->hsk = hsk;
    scene->hsv = hsv;
    scene->heads = heads;
    scene->page_size = params.page_size;
    scene->query_count = params.query_count;
    scene->pages = CEIL_DIV(scene->capacity, scene->page_size);
    const auto build_usage = vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
    if (!rt.unit_box) {
        scene->bounds = ggml_vk_rt_buffer(rt.device, size_t(heads) * scene->pages * sizeof(VkAabbPositionsKHR), build_usage);
    }
    scene->metadata = ggml_vk_rt_buffer(rt.device, size_t(heads) * scene->pages * (rt.light_summary ? 8 : 2 * hsk + 8) * sizeof(float));
    scene->indices = ggml_vk_rt_buffer(rt.device, size_t(heads) * (CEIL_DIV(scene->capacity, 16 * scene->page_size) + 4) * sizeof(uint32_t));
    if (rt.unit_box) {
        scene->query_state = ggml_vk_rt_buffer(rt.device, size_t(heads) * params.query_state_words * sizeof(uint32_t));
    } else {
        ggml_vk_rt_allocate_scene(rt, *scene);
    }
    if (params.cache_summary) {
        const size_t words = 2 + scene->pages + size_t(heads) * scene->pages * (1 + CEIL_DIV(scene->page_size, 32));
        scene->summary_state = ggml_vk_rt_buffer(rt.device, words * sizeof(uint32_t));
        source.buffer->rt_kv_source = true;
    }
    rt.scenes.push_back(std::move(scene));
    return *rt.scenes.back();
}

static void ggml_vk_rt_barrier(VkCommandBuffer cmd, VkPipelineStageFlags src, VkPipelineStageFlags dst, VkAccessFlags src_access, VkAccessFlags dst_access) {
    VkMemoryBarrier barrier {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = src_access;
    barrier.dstAccessMask = dst_access;
    vkCmdPipelineBarrier(cmd, src, dst, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

static void ggml_vk_rt_build_scene(vk_rt_kv_context & rt, vk_rt_kv_scene & scene, VkCommandBuffer cmd) {
    const bool update = rt.refit && !scene.vertices && scene.built;
    const VkDeviceAddress scratch = ((scene.scratch->bda_addr + rt.scratch_alignment - 1) / rt.scratch_alignment) * rt.scratch_alignment;
    auto build_as = [&](VkAccelerationStructureTypeKHR type, const VkAccelerationStructureGeometryKHR & geometry, VkAccelerationStructureKHR as, uint32_t count) {
        auto build = ggml_vk_rt_build_info(rt, scene, type, geometry);
        build.mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        build.srcAccelerationStructure = update ? as : VK_NULL_HANDLE;
        build.dstAccelerationStructure = as;
        build.scratchData.deviceAddress = scratch;
        VkAccelerationStructureBuildRangeInfoKHR range { count, 0, 0, 0 };
        const auto * ranges = &range;
        rt.build_as(cmd, 1, &build, &ranges);
    };
    for (uint32_t head = 0; head < scene.heads; ++head) {
        build_as(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, ggml_vk_rt_bottom_geometry(scene, head), scene.bottom[head], scene.pages);
        ggml_vk_rt_barrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                          VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);
    }
    build_as(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, ggml_vk_rt_top_geometry(scene), scene.top, scene.heads);
    scene.built = true;
}

static vk_rt_kv_scene & ggml_vk_rt_unit_scene(vk_rt_kv_context & rt) {
    if (rt.unit_scene) { return *rt.unit_scene; }
    auto scene = std::make_unique<vk_rt_kv_scene>();
    scene->device = rt.device;
    scene->destroy_as = rt.destroy_as;
    scene->heads = 1;
    scene->pages = 12;
    const float vertices[] = {
        -1, -1, -1,  1, -1, -1,  1,  1, -1, -1,  1, -1,
        -1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1,
    };
    const uint32_t triangles[] = {
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
        0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2,
        0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5,
    };
    const auto build_usage = vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
    scene->vertices = ggml_vk_rt_buffer(rt.device, sizeof(vertices), build_usage);
    scene->triangles = ggml_vk_rt_buffer(rt.device, sizeof(triangles), build_usage);
    ggml_vk_buffer_write(scene->vertices, 0, vertices, sizeof(vertices));
    ggml_vk_buffer_write(scene->triangles, 0, triangles, sizeof(triangles));
    ggml_vk_rt_allocate_scene(rt, *scene);
    rt.unit_scene = std::move(scene);
    return *rt.unit_scene;
}

static vk::DescriptorSet ggml_vk_rt_set(vk_rt_kv_context & rt, const std::array<vk_subbuffer, 8> & buffers, VkAccelerationStructureKHR as, const vk_subbuffer & summary_state) {
    if (rt.set_idx == rt.sets.size()) {
        const std::array<vk::DescriptorPoolSize, 2> sizes {{
            { vk::DescriptorType::eStorageBuffer, 9 * 64 },
            { vk::DescriptorType::eAccelerationStructureKHR, 64 },
        }};
        rt.pools.push_back(rt.device->device.createDescriptorPool(vk::DescriptorPoolCreateInfo({}, 64, sizes)));
        std::array<vk::DescriptorSetLayout, 64> layouts;
        layouts.fill(rt.descriptor_layout);
        const auto sets = rt.device->device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo(rt.pools.back(), layouts));
        rt.sets.insert(rt.sets.end(), sets.begin(), sets.end());
    }
    const auto set = rt.sets[rt.set_idx++];
    std::array<vk::DescriptorBufferInfo, 9> infos;
    std::array<vk::WriteDescriptorSet, 10> writes;
    for (uint32_t i = 0; i < 9; ++i) {
        infos[i] = i == 8 ? summary_state : buffers[i];
        writes[i] = { set, i == 8 ? 9 : i, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &infos[i] };
    }
    const vk::AccelerationStructureKHR acceleration_structure(as);
    const vk::WriteDescriptorSetAccelerationStructureKHR acceleration(1, &acceleration_structure);
    writes[9] = { set, 8, 0, 1, vk::DescriptorType::eAccelerationStructureKHR };
    writes[9].pNext = &acceleration;
    rt.device->device.updateDescriptorSets(writes, {});
    return set;
}

bool ggml_vk_rt_kv_prepare(ggml_backend_vk_context * ctx, const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v, const ggml_tensor * mask, const ggml_tensor * sinks, const vk_fa_tuning_params & tuning, bool f32acc, float max_bias, float logit_softcap, vk_rt_kv_params & params) {
    const bool supported = tuning.path == FA_COOPMAT2 && f32acc && ctx->device->buffer_device_address && mask && !sinks &&
           q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16 && mask->type == GGML_TYPE_F16 &&
           q->ne[1] == 1 && q->ne[3] == 1 && k->ne[3] == 1 && v->ne[3] == 1 &&
           k->ne[1] >= 512 && k->ne[0] > 0 && v->ne[0] > 0 &&
           k->ne[2] > 0 && k->ne[2] == v->ne[2] &&
           q->ne[2] >= k->ne[2] && q->ne[2] % k->ne[2] == 0 &&
           mask->ne[0] == k->ne[1] && mask->ne[1] >= 1 &&
           mask->ne[2] == 1 && mask->ne[3] == 1 && mask->nb[0] == sizeof(ggml_fp16_t) &&
           ((k->nb[1] | k->nb[2] | v->nb[1] | v->nb[2]) % (4 * sizeof(ggml_fp16_t))) == 0 &&
           max_bias == 0.0f && logit_softcap == 0.0f;
    if (!supported) {
        return false;
    }
    // The selectors use 32-bit element offsets.
    const ggml_tensor * tensors[] = { q, k, v };
    for (const auto * tensor : tensors) {
        const size_t element_size = ggml_type_size(tensor->type);
        uint64_t last = 0;
        for (uint32_t dim = 0; dim < 4; ++dim) {
            if (tensor->ne[dim] <= 0 || uint64_t(tensor->ne[dim]) > UINT32_MAX) {
                return false;
            }
            const uint64_t stride = tensor->nb[dim] / element_size;
            const uint64_t count = uint64_t(tensor->ne[dim] - 1);
            if (tensor->nb[dim] % element_size != 0 || stride > UINT32_MAX || (stride != 0 && count > (UINT32_MAX - last) / stride)) {
                return false;
            }
            last += count * stride;
        }
    }
    if (!ctx->rt_kv) { ctx->rt_kv = ggml_vk_rt_context(ctx->device); }
    const auto & rt = *ctx->rt_kv;
    params.page_size = rt.page_size ? rt.page_size : std::min(32u, tuning.block_cols);
    if (!rt.tile_mask || params.page_size < 8 || tuning.block_cols % params.page_size != 0) {
        return false;
    }
    params.query_count = uint32_t(q->ne[2] / k->ne[2]);
    const auto & limits = ctx->device->properties.limits;
    if (params.query_count > rt.query_threads || uint64_t(k->ne[2]) > std::min(limits.maxComputeWorkGroupCount[0], limits.maxComputeWorkGroupCount[1]) || uint64_t(k->ne[2]) > rt.max_instances) {
        return false;
    }

    const ggml_tensor * root = k;
    while (root->view_src) { root = root->view_src; }
    params.cache_summary = rt.unit_box && rt.light_summary && root->buffer && root->buffer->buft == &rt.device->buffer_type && ggml_is_contiguous(root) &&
        root->ne[2] == 1 && root->ne[3] == 1 && root->ne[1] >= k->ne[1] && uint64_t(root->ne[1]) <= UINT32_MAX &&
        root->nb[1] == k->nb[1] && root->ne[0] == k->ne[0] * k->ne[2] && k->nb[2] == size_t(k->ne[0]) * sizeof(ggml_fp16_t);
    if (params.cache_summary) {
        params.cache_summary = uint64_t(root->ne[1]) <= UINT32_MAX / (root->nb[1] / sizeof(ggml_fp16_t)) &&
            vk_tensor_offset(root) + root->view_offs == vk_tensor_offset(k) + k->view_offs;
    }
    // The root tensor carries the KV allocation selected by llama.cpp.
    uint64_t capacity = params.cache_summary ? uint64_t(root->ne[1]) : 512;
    while (capacity < uint64_t(k->ne[1])) { capacity *= 2; }
    params.capacity = capacity <= UINT32_MAX ? uint32_t(capacity) : uint32_t(k->ne[1]);
    const uint64_t pages = CEIL_DIV(uint64_t(params.capacity), params.page_size);
    if ((!rt.unit_box && pages > rt.max_primitives) || pages > uint64_t(ctx->device->properties.limits.maxComputeWorkGroupCount[0]) * rt.query_threads) {
        return false;
    }
    if (!params.cache_summary && pages > ctx->device->properties.limits.maxComputeWorkGroupCount[0]) {
        return false;
    }
    params.bitmap_words = uint32_t(CEIL_DIV(pages, 32));
    params.query_state_words = CEIL_DIV(8 + 3 * params.query_count, 32) * 32 + params.bitmap_words + 2 * uint32_t(CEIL_DIV(pages, rt.query_threads));
    const uint64_t query_floats = uint64_t(params.query_count) * (rt.approximate ? 3 : k->ne[0]);
    const uint64_t bitmap_floats = uint64_t(params.bitmap_words) * (rt.unit_box ? 1 : 1 + params.query_count);
    const uint64_t shared_size = (query_floats + bitmap_floats + 2 * rt.query_threads + 8) * sizeof(uint32_t);
    if (shared_size > ctx->device->properties.limits.maxComputeSharedMemorySize) {
        return false;
    }
    const uint64_t heads = uint64_t(k->ne[2]);
    const uint64_t buffer_limit = std::min<uint64_t>(ctx->device->properties.limits.maxStorageBufferRange, std::min(ctx->device->max_buffer_size, ctx->device->max_memory_allocation_size));
    const uint64_t metadata_size = heads * pages * (rt.light_summary ? 8 : 2 * uint64_t(k->ne[0]) + 8) * sizeof(float);
    const uint64_t indices_size = heads * (CEIL_DIV(uint64_t(params.capacity), 16 * params.page_size) + 4) * sizeof(uint32_t);
    const uint64_t state_size = heads * params.query_state_words * sizeof(uint32_t);
    const uint64_t summary_size = (2 + pages + heads * pages * (1 + CEIL_DIV(params.page_size, 32))) * sizeof(uint32_t);
    return metadata_size <= buffer_limit && indices_size <= buffer_limit && state_size <= buffer_limit && summary_size <= buffer_limit;
}

static std::array<vk::Pipeline, 4> & ggml_vk_rt_pipelines(vk_rt_kv_context & rt, const ggml_tensor * k, const vk_rt_kv_params & params) {
    const std::array<uint32_t, 4> key { uint32_t(k->ne[0]), params.query_count, params.bitmap_words, params.page_size };
    auto result = rt.pipelines.try_emplace(key);
    auto & pipelines = result.first->second;
    if (result.second) {
        std::array<uint32_t, 6> constants { rt.query_threads, rt.approximate, rt.unit_box ? 1u : 0u, params.query_count, uint32_t(k->ne[0]), params.bitmap_words };
        constants[0] = rt.light_summary ? params.page_size : rt.query_threads;
        ggml_vk_rt_pipeline(rt, fa_rt_summary_data, fa_rt_summary_len, rt.summary_module, pipelines[0], constants);
        constants[0] = rt.query_threads;
        ggml_vk_rt_pipeline(rt, rt.unit_box ? fa_rt_select_unit_data : rt.use_subgroups ? fa_rt_select_subgroup_data : fa_rt_select_data,
                            rt.unit_box ? fa_rt_select_unit_len : rt.use_subgroups ? fa_rt_select_subgroup_len : fa_rt_select_len,
                            rt.select_module, pipelines[1], constants);
        if (rt.unit_box) {
            constants[2] = 2;
            ggml_vk_rt_pipeline(rt, fa_rt_select_unit_data, fa_rt_select_unit_len, rt.select_module, pipelines[2], constants);
            constants[2] = 3;
            ggml_vk_rt_pipeline(rt, fa_rt_select_unit_data, fa_rt_select_unit_len, rt.select_module, pipelines[3], constants);
        }
    }
    return pipelines;
}

vk_subbuffer ggml_vk_rt_kv_select(ggml_backend_vk_context * ctx, vk_context & subctx, const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v, const ggml_tensor * mask, float scale, uint32_t block_cols, const vk_rt_kv_params & params) {
    auto & rt = *ctx->rt_kv;
    auto k_buf = ggml_vk_tensor_subbuffer(ctx, k);
    auto & scene = ggml_vk_rt_scene(rt, k, uint32_t(v->ne[0]), k_buf, params);
    const auto & pipelines = ggml_vk_rt_pipelines(rt, k, params);
    auto & trace_scene = rt.unit_box ? ggml_vk_rt_unit_scene(rt) : scene;
    const vk_rt_kv_push_constants pc {
        uint32_t(k->ne[1]), scene.hsk, scene.hsv, scene.page_size, scene.pages, rt.window,
        uint32_t(k->nb[1] / 2), uint32_t(k->nb[2] / 2), uint32_t(v->nb[1] / 2), uint32_t(v->nb[2] / 2), uint32_t(q->nb[2] / 4),
        scale, rt.error_budget, rt.padding, rt.exact_bounds, rt.dense_mode, rt.light_summary,
        block_cols, uint32_t(params.cache_summary), rt.unit_box,
    };
    auto q_buf = ggml_vk_tensor_subbuffer(ctx, q);
    auto v_buf = ggml_vk_tensor_subbuffer(ctx, v);
    auto m_buf = ggml_vk_tensor_subbuffer(ctx, mask);
    vk_subbuffer metadata { scene.metadata, 0, scene.metadata->size };
    vk_subbuffer bounds = rt.unit_box ? metadata : vk_subbuffer { scene.bounds, 0, scene.bounds->size };
    vk_subbuffer indices { scene.indices, 0, scene.indices->size };
    vk_subbuffer stats { rt.stats, 0, rt.stats->size };
    vk_subbuffer query_state = rt.unit_box ? vk_subbuffer { scene.query_state, 0, scene.query_state->size } : metadata;
    ggml_vk_sync_buffers(ctx, subctx);
    vk::CommandBuffer cmd = subctx->s->buffer->buf;
    vk_subbuffer summary_state = scene.summary_state ? vk_subbuffer { scene.summary_state, 0, scene.summary_state->size } : metadata;
    if (scene.summary_state) {
        const uint64_t generation = k_buf.buffer->rt_kv_generation.load(std::memory_order_relaxed);
        const size_t seen_offset = (2 + scene.pages) * sizeof(uint32_t);
        if (!scene.summary_initialized) {
            const uint32_t header[] = { scene.page_size, scene.pages };
            vkCmdFillBuffer(cmd, scene.summary_state->buffer, seen_offset, scene.summary_state->size - seen_offset, UINT32_MAX);
            vkCmdFillBuffer(cmd, scene.summary_state->buffer, 2 * sizeof(uint32_t), scene.pages * sizeof(uint32_t), 0);
            vkCmdUpdateBuffer(cmd, scene.summary_state->buffer, 0, sizeof(header), header);
            scene.summary_initialized = true;
            scene.invalidated = true;
        } else if (scene.invalidated || scene.source_generation != generation) {
            vkCmdFillBuffer(cmd, scene.summary_state->buffer, seen_offset, scene.summary_state->size - seen_offset, UINT32_MAX);
        }
        if (scene.invalidated || scene.source_generation != generation) {
            ggml_vk_rt_barrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        }
        scene.invalidated = false;
        scene.source_generation = generation;
    }
    if (rt.unit_box && !trace_scene.built) {
        ggml_vk_rt_build_scene(rt, trace_scene, cmd);
        ggml_vk_rt_barrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    }
    if (rt.dense_mode != 1 && !scene.summary_state) {
        auto summary_set = ggml_vk_rt_set(rt, { k_buf, v_buf, m_buf, metadata, bounds, indices, stats, metadata }, trace_scene.top, summary_state);
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines[0]);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, rt.pipeline_layout, 0, summary_set, {});
        cmd.pushConstants(rt.pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
        cmd.dispatch(scene.pages, scene.heads, 1);
        if (rt.unit_box) {
            ggml_vk_rt_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        } else {
            ggml_vk_rt_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                              VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                              VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);

            ggml_vk_rt_build_scene(rt, scene, cmd);
            ggml_vk_rt_barrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_SHADER_WRITE_BIT,
                              VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT);
        }
    }

    auto select_set = ggml_vk_rt_set(rt, { q_buf, k_buf, v_buf, m_buf, metadata, indices, stats, query_state }, trace_scene.top, summary_state);
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines[1]);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, rt.pipeline_layout, 0, select_set, {});
    cmd.pushConstants(rt.pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
    cmd.dispatch(scene.heads, 1, 1);
    if (rt.unit_box) {
        ggml_vk_rt_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines[2]);
        cmd.dispatch(CEIL_DIV(scene.pages, rt.query_threads), scene.heads, 1);
        ggml_vk_rt_barrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipelines[3]);
        cmd.dispatch(scene.heads, 1, 1);
    }
    ggml_vk_sync_buffers(ctx, subctx);
    return indices;
}

static vk_subbuffer ggml_vk_rt_kv_update(ggml_backend_vk_context * ctx, const ggml_tensor * dst, bool incremental) {
    if (!ctx->rt_kv || !dst->buffer || dst->buffer->buft != &ctx->device->buffer_type) {
        return {};
    }
    const auto * buf_ctx = static_cast<const ggml_backend_vk_buffer_context *>(dst->buffer->context);
    if (!buf_ctx->dev_buffer || !buf_ctx->dev_buffer->rt_kv_source) {
        return {};
    }
    const size_t offset = vk_tensor_offset(dst) + dst->view_offs;
    const size_t size = ggml_nbytes(dst);
    vk_subbuffer state {};
    for (auto & scene : ctx->rt_kv->scenes) {
        if (!scene->summary_state || scene->source.lock() != buf_ctx->dev_buffer ||
            offset >= scene->source_offset + uint64_t(scene->capacity) * scene->k_stride || scene->source_offset >= offset + size) {
            continue;
        }
        if (incremental && !state.buffer && scene->summary_initialized && offset == scene->source_offset && dst->type == GGML_TYPE_F16 && ggml_is_contiguous(dst) &&
            dst->ne[2] == 1 && dst->ne[3] == 1 && dst->ne[1] == scene->capacity && dst->nb[1] == scene->k_stride) {
            state = { scene->summary_state, 0, scene->summary_state->size };
        } else {
            scene->invalidated = true;
        }
    }
    return state;
}

void ggml_vk_rt_kv_invalidate(ggml_backend_vk_context * ctx, const ggml_tensor * dst) {
    ggml_vk_rt_kv_update(ctx, dst, false);
}

vk_subbuffer ggml_vk_rt_kv_write(ggml_backend_vk_context * ctx, const ggml_tensor * dst) {
    return ggml_vk_rt_kv_update(ctx, dst, true);
}

void ggml_vk_rt_kv_reset(ggml_backend_vk_context * ctx) {
    if (ctx->rt_kv) {
        ctx->rt_kv->set_idx = 0;
    }
}

void ggml_vk_rt_kv_free(ggml_backend_vk_context * ctx) {
    if (ctx->rt_kv && getenv("GGML_VK_RT_KV_STATS")) {
        uint32_t stats[8] {};
        ggml_vk_buffer_read(ctx->rt_kv->stats, 0, stats, sizeof(stats));
        GGML_LOG_INFO("ggml_vulkan: RT KV calls=%u rows=%u selected=%u fallback=%u rays=%u hits=%u\n",
                      stats[0], stats[1], stats[2], stats[3], stats[4], stats[5]);
    }
    ctx->rt_kv.reset();
}
