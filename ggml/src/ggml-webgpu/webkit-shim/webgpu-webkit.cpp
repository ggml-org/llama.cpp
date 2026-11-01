#include "WebGPU.h"

#include "webgpu-webkit-bridge.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <utility>
#include <vector>

#define WEBKIT_FORWARD_HANDLE(name, type) \
    extern "C" void name(void * handle) { ::name(static_cast<type>(handle)); }

WEBKIT_FORWARD_HANDLE(wgpuAdapterRelease, WGPUAdapter)
WEBKIT_FORWARD_HANDLE(wgpuBindGroupAddRef, WGPUBindGroup)
WEBKIT_FORWARD_HANDLE(wgpuBindGroupRelease, WGPUBindGroup)
WEBKIT_FORWARD_HANDLE(wgpuBindGroupLayoutRelease, WGPUBindGroupLayout)
WEBKIT_FORWARD_HANDLE(wgpuBufferAddRef, WGPUBuffer)
WEBKIT_FORWARD_HANDLE(wgpuBufferDestroy, WGPUBuffer)
WEBKIT_FORWARD_HANDLE(wgpuBufferRelease, WGPUBuffer)
WEBKIT_FORWARD_HANDLE(wgpuBufferUnmap, WGPUBuffer)
WEBKIT_FORWARD_HANDLE(wgpuCommandBufferAddRef, WGPUCommandBuffer)
WEBKIT_FORWARD_HANDLE(wgpuCommandBufferRelease, WGPUCommandBuffer)
WEBKIT_FORWARD_HANDLE(wgpuCommandEncoderRelease, WGPUCommandEncoder)
WEBKIT_FORWARD_HANDLE(wgpuComputePassEncoderEnd, WGPUComputePassEncoder)
WEBKIT_FORWARD_HANDLE(wgpuComputePassEncoderRelease, WGPUComputePassEncoder)
WEBKIT_FORWARD_HANDLE(wgpuComputePipelineAddRef, WGPUComputePipeline)
WEBKIT_FORWARD_HANDLE(wgpuComputePipelineRelease, WGPUComputePipeline)
WEBKIT_FORWARD_HANDLE(wgpuDeviceAddRef, WGPUDevice)
WEBKIT_FORWARD_HANDLE(wgpuDeviceRelease, WGPUDevice)
WEBKIT_FORWARD_HANDLE(wgpuInstanceRelease, WGPUInstance)
WEBKIT_FORWARD_HANDLE(wgpuPipelineLayoutRelease, WGPUPipelineLayout)
WEBKIT_FORWARD_HANDLE(wgpuQueueRelease, WGPUQueue)
WEBKIT_FORWARD_HANDLE(wgpuSamplerAddRef, WGPUSampler)
WEBKIT_FORWARD_HANDLE(wgpuSamplerRelease, WGPUSampler)
WEBKIT_FORWARD_HANDLE(wgpuShaderModuleAddRef, WGPUShaderModule)
WEBKIT_FORWARD_HANDLE(wgpuShaderModuleRelease, WGPUShaderModule)
WEBKIT_FORWARD_HANDLE(wgpuSurfaceRelease, WGPUSurface)
WEBKIT_FORWARD_HANDLE(wgpuTextureViewAddRef, WGPUTextureView)
WEBKIT_FORWARD_HANDLE(wgpuTextureViewRelease, WGPUTextureView)

#undef WEBKIT_FORWARD_HANDLE

extern "C" void * wgpuDeviceGetQueue(void * device) {
    return ::wgpuDeviceGetQueue(static_cast<WGPUDevice>(device));
}

extern "C" void * wgpuComputePipelineGetBindGroupLayout(void * pipeline, uint32_t group_index) {
    return ::wgpuComputePipelineGetBindGroupLayout(static_cast<WGPUComputePipeline>(pipeline), group_index);
}

extern "C" uint64_t wgpuBufferGetSize(void * buffer) {
    return ::wgpuBufferGetCurrentSize(static_cast<WGPUBuffer>(buffer));
}

extern "C" void const * wgpuBufferGetConstMappedRange(void * buffer, size_t offset, size_t size) {
    return ::wgpuBufferGetMappedRange(static_cast<WGPUBuffer>(buffer), offset, size).data();
}

extern "C" void wgpuCommandEncoderCopyBufferToBuffer(void * encoder, void * source, uint64_t source_offset,
                                                       void * destination, uint64_t destination_offset, uint64_t size) {
    ::wgpuCommandEncoderCopyBufferToBuffer(static_cast<WGPUCommandEncoder>(encoder), static_cast<WGPUBuffer>(source),
                                           source_offset, static_cast<WGPUBuffer>(destination), destination_offset, size);
}

extern "C" void wgpuComputePassEncoderDispatchWorkgroups(void * pass, uint32_t x, uint32_t y, uint32_t z) {
    ::wgpuComputePassEncoderDispatchWorkgroups(static_cast<WGPUComputePassEncoder>(pass), x, y, z);
}

extern "C" void wgpuComputePassEncoderSetPipeline(void * pass, void * pipeline) {
    ::wgpuComputePassEncoderSetPipeline(static_cast<WGPUComputePassEncoder>(pass),
                                        static_cast<WGPUComputePipeline>(pipeline));
}

extern "C" void wgpuQueueSubmit(void * queue, size_t command_count, void * const * commands) {
    ::wgpuQueueSubmit(static_cast<WGPUQueue>(queue), command_count,
                      reinterpret_cast<WGPUCommandBuffer const *>(commands));
}

extern "C" void wgpuQueueWriteBuffer(void * queue, void * buffer, uint64_t offset, void const * data, size_t size) {
    ::wgpuQueueWriteBuffer(static_cast<WGPUQueue>(queue), static_cast<WGPUBuffer>(buffer), offset,
                           { const_cast<uint8_t *>(static_cast<uint8_t const *>(data)), size });
}

extern "C" void * webkit_create_instance() {
    WGPUInstanceDescriptor descriptor{};
    return ::wgpuCreateInstance(&descriptor);
}

extern "C" void webkit_instance_process_events(void * instance) {
    ::wgpuInstanceProcessEvents(static_cast<WGPUInstance>(instance));
}

extern "C" void webkit_instance_request_adapter(void * instance, int power_preference,
                                                    webkit_adapter_callback callback, void * userdata) {
    WGPURequestAdapterOptions options{};
    options.powerPreference = static_cast<WGPUPowerPreference>(power_preference);
    ::wgpuInstanceRequestAdapter(static_cast<WGPUInstance>(instance), &options,
                                 [](WGPURequestAdapterStatus status, WGPUAdapter adapter, const char * message, void * context) {
                                     auto * args = static_cast<std::pair<webkit_adapter_callback, void *> *>(context);
                                     args->first(static_cast<int>(status), adapter, message, args->second);
                                     delete args;
                                 }, new std::pair<webkit_adapter_callback, void *>(callback, userdata));
}

extern "C" void webkit_adapter_get_info(void * adapter, webkit_adapter_info * info) {
    WGPUAdapterInfo webkit_info{};
    ::wgpuAdapterGetInfo(static_cast<WGPUAdapter>(adapter), &webkit_info);
    info->vendor_id         = webkit_info.vendorID;
    info->device_id         = webkit_info.deviceID;
    info->adapter_type      = static_cast<uint32_t>(webkit_info.adapterType);
    info->backend_type      = static_cast<uint32_t>(webkit_info.backendType);
    info->subgroup_min_size = webkit_info.subgroupMinSize;
    info->subgroup_max_size = webkit_info.subgroupMaxSize;
    info->vendor            = webkit_info.vendorName;
    info->architecture      = webkit_info.architecture;
    info->device            = webkit_info.name;
    info->description       = webkit_info.driverDescription;
}

extern "C" bool webkit_adapter_get_limits(void * adapter, webkit_limits * limits) {
    WGPUSupportedLimits supported{};
    if (!::wgpuAdapterGetLimits(static_cast<WGPUAdapter>(adapter), &supported)) {
        return false;
    }
    limits->max_buffer_size                     = supported.limits.maxBufferSize;
    limits->max_storage_buffer_binding_size     = supported.limits.maxStorageBufferBindingSize;
    limits->max_compute_invocations_per_workgroup = supported.limits.maxComputeInvocationsPerWorkgroup;
    limits->max_compute_workgroup_storage_size  = supported.limits.maxComputeWorkgroupStorageSize;
    limits->max_compute_workgroups_per_dimension = supported.limits.maxComputeWorkgroupsPerDimension;
    limits->min_storage_buffer_offset_alignment = supported.limits.minStorageBufferOffsetAlignment;
    limits->min_uniform_buffer_offset_alignment = supported.limits.minUniformBufferOffsetAlignment;
    return true;
}

static WGPUFeatureName webkit_feature_name(int feature) {
    switch (feature) {
        case WEBKIT_FEATURE_SHADER_F16:      return WGPUFeatureName_ShaderF16;
        case WEBKIT_FEATURE_SUBGROUPS:       return WGPUFeatureName_Subgroups;
        case WEBKIT_FEATURE_TIMESTAMP_QUERY: return WGPUFeatureName_TimestampQuery;
        default:                             return WGPUFeatureName_Undefined;
    }
}

extern "C" bool webkit_adapter_has_feature(void * adapter, int feature) {
    const WGPUFeatureName name = webkit_feature_name(feature);
    return name != WGPUFeatureName_Undefined && ::wgpuAdapterHasFeature(static_cast<WGPUAdapter>(adapter), name);
}

extern "C" void webkit_adapter_request_device(void * adapter, const int * features, size_t feature_count,
                                                 webkit_device_callback callback, void * userdata) {
    std::vector<WGPUFeatureName> webkit_features;
    webkit_features.reserve(feature_count);
    for (size_t i = 0; i < feature_count; ++i) {
        const WGPUFeatureName feature = webkit_feature_name(features[i]);
        if (feature == WGPUFeatureName_Undefined) {
            callback(1, nullptr, "Unsupported Dawn feature", userdata);
            return;
        }
        webkit_features.push_back(feature);
    }

    WGPUSupportedLimits supported{};
    if (!::wgpuAdapterGetLimits(static_cast<WGPUAdapter>(adapter), &supported)) {
        callback(1, nullptr, "Failed to get WebKit limits", userdata);
        return;
    }
    WGPURequiredLimits required{ supported.limits };
    WGPUDeviceDescriptor descriptor{};
    descriptor.requiredFeatureCount = webkit_features.size();
    descriptor.requiredFeatures     = webkit_features.data();
    descriptor.requiredLimits       = &required;
    ::wgpuAdapterRequestDevice(static_cast<WGPUAdapter>(adapter), &descriptor,
                               [](WGPURequestDeviceStatus status, WGPUDevice device, const char * message, void * context) {
                                   auto * args = static_cast<std::pair<webkit_device_callback, void *> *>(context);
                                   if (device) {
                                       ::wgpuDeviceSetUncapturedErrorCallback(device,
                                           [](WGPUErrorType type, const char * error, void *) {
                                               std::fprintf(stderr, "WebKit WebGPU error %d: %s\n", static_cast<int>(type), error);
                                           }, nullptr);
                                   }
                                   args->first(static_cast<int>(status), device, message, args->second);
                                   delete args;
                               }, new std::pair<webkit_device_callback, void *>(callback, userdata));
}

extern "C" void * webkit_device_create_buffer(void * device, const char * label, size_t label_size,
                                                uint64_t usage, uint64_t size, bool mapped_at_creation) {
    WGPUBufferDescriptor descriptor{};
    descriptor.label            = { label, label_size };
    descriptor.usage            = usage;
    descriptor.size             = size;
    descriptor.mappedAtCreation = mapped_at_creation;
    return ::wgpuDeviceCreateBuffer(static_cast<WGPUDevice>(device), &descriptor);
}

extern "C" void webkit_buffer_map_async(void * buffer, uint64_t mode, size_t offset, size_t size,
                                          webkit_status_callback callback, void * userdata) {
    ::wgpuBufferMapAsync(static_cast<WGPUBuffer>(buffer), mode, offset, size,
                         [](WGPUMapAsyncStatus status, void * context) {
                             auto * args = static_cast<std::pair<webkit_status_callback, void *> *>(context);
                             args->first(static_cast<int>(status), args->second);
                             delete args;
                         }, new std::pair<webkit_status_callback, void *>(callback, userdata));
}

extern "C" void webkit_queue_on_submitted_work_done(void * queue, webkit_status_callback callback, void * userdata) {
    ::wgpuQueueOnSubmittedWorkDone(static_cast<WGPUQueue>(queue),
                                   [](WGPUQueueWorkDoneStatus status, void * context) {
                                       auto * args = static_cast<std::pair<webkit_status_callback, void *> *>(context);
                                       args->first(static_cast<int>(status), args->second);
                                       delete args;
                                   }, new std::pair<webkit_status_callback, void *>(callback, userdata));
}

extern "C" void * webkit_device_create_shader_module(void * device, const char * code, size_t code_size) {
    WGPUShaderSourceWGSL source{};
    source.chain.sType = WGPUSType_ShaderSourceWGSL;
    source.code        = { code, code_size };
    WGPUShaderModuleDescriptor descriptor{};
    descriptor.nextInChain = &source.chain;
    return ::wgpuDeviceCreateShaderModule(static_cast<WGPUDevice>(device), &descriptor);
}

extern "C" void * webkit_device_create_compute_pipeline(void * device, const char * label, size_t label_size,
                                                           void * shader_module, const char * entry_point, size_t entry_point_size,
                                                           const webkit_constant * constants, size_t constant_count) {
    std::string entry_point_string(entry_point, entry_point_size);
    std::vector<std::string> keys;
    std::vector<WGPUConstantEntry> webkit_constants;
    keys.reserve(constant_count);
    webkit_constants.reserve(constant_count);
    for (size_t i = 0; i < constant_count; ++i) {
        keys.emplace_back(constants[i].key, constants[i].key_size);
        webkit_constants.push_back({ keys.back().c_str(), constants[i].value });
    }
    WGPUComputePipelineDescriptor descriptor{};
    descriptor.label                 = { label, label_size };
    WGPUPipelineLayoutDescriptor layout_descriptor{};
    WGPUPipelineLayout auto_layout = ::wgpuDeviceCreatePipelineLayout(static_cast<WGPUDevice>(device), &layout_descriptor);
    descriptor.layout                = auto_layout;
    descriptor.compute.module        = static_cast<WGPUShaderModule>(shader_module);
    descriptor.compute.entryPoint    = entry_point_string.c_str();
    descriptor.compute.constantCount = webkit_constants.size();
    descriptor.compute.constants     = webkit_constants.data();
    auto pipeline = ::wgpuDeviceCreateComputePipeline(static_cast<WGPUDevice>(device), &descriptor);
    ::wgpuPipelineLayoutRelease(auto_layout);
    return pipeline;
}

extern "C" void * webkit_device_create_bind_group(void * device, void * layout, const webkit_bind_entry * entries,
                                                    size_t entry_count) {
    std::vector<WGPUBindGroupEntry> webkit_entries(entry_count);
    for (size_t i = 0; i < entry_count; ++i) {
        webkit_entries[i].binding = entries[i].binding;
        webkit_entries[i].buffer  = static_cast<WGPUBuffer>(entries[i].buffer);
        webkit_entries[i].offset  = entries[i].offset;
        webkit_entries[i].size    = entries[i].size;
    }
    WGPUBindGroupDescriptor descriptor{};
    descriptor.layout     = static_cast<WGPUBindGroupLayout>(layout);
    descriptor.entryCount = webkit_entries.size();
    descriptor.entries    = webkit_entries.data();
    return ::wgpuDeviceCreateBindGroup(static_cast<WGPUDevice>(device), &descriptor);
}

extern "C" void * webkit_device_create_command_encoder(void * device, const char * label, size_t label_size) {
    WGPUCommandEncoderDescriptor descriptor{};
    descriptor.label = { label, label_size };
    return ::wgpuDeviceCreateCommandEncoder(static_cast<WGPUDevice>(device), &descriptor);
}

extern "C" void * webkit_command_encoder_begin_compute_pass(void * encoder) {
    WGPUComputePassDescriptor descriptor{};
    return ::wgpuCommandEncoderBeginComputePass(static_cast<WGPUCommandEncoder>(encoder), &descriptor);
}

extern "C" void * webkit_command_encoder_finish(void * encoder) {
    WGPUCommandBufferDescriptor descriptor{};
    return ::wgpuCommandEncoderFinish(static_cast<WGPUCommandEncoder>(encoder), &descriptor);
}

extern "C" void webkit_compute_pass_set_bind_group(void * pass, uint32_t group_index, void * bind_group,
                                                     const uint32_t * offsets, size_t offset_count) {
    std::optional<WTF::Vector<uint32_t>> webkit_offsets;
    if (offset_count) {
        webkit_offsets.emplace();
        webkit_offsets->append(std::span<const uint32_t>(offsets, offset_count));
    }
    ::wgpuComputePassEncoderSetBindGroup(static_cast<WGPUComputePassEncoder>(pass), group_index,
                                         static_cast<WGPUBindGroup>(bind_group), std::move(webkit_offsets));
}
