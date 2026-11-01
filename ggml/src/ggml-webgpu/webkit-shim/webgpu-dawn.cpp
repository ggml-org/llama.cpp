#include "webgpu-webkit-bridge.h"

#include <webgpu/webgpu.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

struct webkit_future {
    std::atomic<bool> done = false;
};

static std::atomic<uint64_t> next_future_id = 1;
static std::mutex future_mutex;
static std::unordered_map<uint64_t, std::shared_ptr<webkit_future>> futures;

static WGPUFuture make_future(std::shared_ptr<webkit_future> state) {
    WGPUFuture future = { next_future_id++ };
    std::lock_guard<std::mutex> lock(future_mutex);
    futures.emplace(future.id, std::move(state));
    return future;
}

static WGPUStringView string_view(const char * string) {
    return { string, string ? std::strlen(string) : 0 };
}

extern "C" WGPUInstance wgpuCreateInstance(const WGPUInstanceDescriptor * descriptor) {
    if (descriptor && descriptor->requiredLimits) {
        return nullptr;
    }
    return static_cast<WGPUInstance>(webkit_create_instance());
}

struct adapter_callback_context {
    WGPURequestAdapterCallbackInfo callback_info;
    std::shared_ptr<webkit_future> future;
};

static void on_adapter(int status, void * adapter, const char * message, void * userdata) {
    std::unique_ptr<adapter_callback_context> context(static_cast<adapter_callback_context *>(userdata));
    WGPURequestAdapterStatus dawn_status = WGPURequestAdapterStatus_Error;
    if (status == 0) {
        dawn_status = WGPURequestAdapterStatus_Success;
    } else if (status == 1) {
        dawn_status = WGPURequestAdapterStatus_Unavailable;
    }
    context->callback_info.callback(dawn_status, static_cast<WGPUAdapter>(adapter), string_view(message),
                                    context->callback_info.userdata1, context->callback_info.userdata2);
    context->future->done = true;
}

extern "C" WGPUFuture wgpuInstanceRequestAdapter(WGPUInstance instance, const WGPURequestAdapterOptions * options,
                                                    WGPURequestAdapterCallbackInfo callback_info) {
    if (!callback_info.callback || callback_info.mode != WGPUCallbackMode_AllowSpontaneous) {
        return {};
    }
    auto state = std::make_shared<webkit_future>();
    WGPUFuture future = make_future(state);
    auto * context = new adapter_callback_context{ callback_info, state };
    const int power_preference = options ? static_cast<int>(options->powerPreference) : 0;
    webkit_instance_request_adapter(instance, power_preference, on_adapter, context);
    return future;
}

extern "C" WGPUWaitStatus wgpuInstanceWaitAny(WGPUInstance instance, size_t future_count, WGPUFutureWaitInfo * wait_infos,
                                                uint64_t timeout_ns) {
    const auto start = std::chrono::steady_clock::now();
    for (;;) {
        webkit_instance_process_events(instance);
        bool completed = false;
        {
            std::lock_guard<std::mutex> lock(future_mutex);
            for (size_t i = 0; i < future_count; ++i) {
                const auto it = futures.find(wait_infos[i].future.id);
                wait_infos[i].completed = it != futures.end() && it->second->done;
                completed |= wait_infos[i].completed != 0;
                if (wait_infos[i].completed) {
                    futures.erase(it);
                }
            }
        }
        if (completed) {
            return WGPUWaitStatus_Success;
        }
        if (timeout_ns != UINT64_MAX &&
            std::chrono::steady_clock::now() - start >= std::chrono::nanoseconds(timeout_ns)) {
            return WGPUWaitStatus_TimedOut;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

extern "C" WGPUBool wgpuAdapterHasFeature(WGPUAdapter adapter, WGPUFeatureName feature) {
    switch (feature) {
        case WGPUFeatureName_ShaderF16:
            return webkit_adapter_has_feature(adapter, WEBKIT_FEATURE_SHADER_F16);
        case WGPUFeatureName_Subgroups:
            return webkit_adapter_has_feature(adapter, WEBKIT_FEATURE_SUBGROUPS);
        case WGPUFeatureName_TimestampQuery:
            return webkit_adapter_has_feature(adapter, WEBKIT_FEATURE_TIMESTAMP_QUERY);
        default:
            return false;
    }
}

extern "C" WGPUBool wgpuInstanceHasWGSLLanguageFeature(WGPUInstance, WGPUWGSLLanguageFeatureName) {
    return false;
}

static int webkit_feature_from_dawn(WGPUFeatureName feature) {
    switch (feature) {
        case WGPUFeatureName_ShaderF16:      return WEBKIT_FEATURE_SHADER_F16;
        case WGPUFeatureName_Subgroups:       return WEBKIT_FEATURE_SUBGROUPS;
        case WGPUFeatureName_TimestampQuery: return WEBKIT_FEATURE_TIMESTAMP_QUERY;
        default:                             return -1;
    }
}

struct device_callback_context {
    WGPURequestDeviceCallbackInfo callback_info;
    std::shared_ptr<webkit_future> future;
};

static void on_device(int status, void * device, const char * message, void * userdata) {
    std::unique_ptr<device_callback_context> context(static_cast<device_callback_context *>(userdata));
    const WGPURequestDeviceStatus dawn_status = status == 0 ? WGPURequestDeviceStatus_Success : WGPURequestDeviceStatus_Error;
    context->callback_info.callback(dawn_status, static_cast<WGPUDevice>(device), string_view(message),
                                    context->callback_info.userdata1, context->callback_info.userdata2);
    context->future->done = true;
}

extern "C" WGPUFuture wgpuAdapterRequestDevice(WGPUAdapter adapter, const WGPUDeviceDescriptor * descriptor,
                                                  WGPURequestDeviceCallbackInfo callback_info) {
    if (!callback_info.callback || callback_info.mode != WGPUCallbackMode_AllowSpontaneous) {
        return {};
    }
    std::vector<int> features;
    if (descriptor) {
        features.reserve(descriptor->requiredFeatureCount);
        for (size_t i = 0; i < descriptor->requiredFeatureCount; ++i) {
            features.push_back(webkit_feature_from_dawn(descriptor->requiredFeatures[i]));
        }
    }
    auto state = std::make_shared<webkit_future>();
    WGPUFuture future = make_future(state);
    auto * context = new device_callback_context{ callback_info, state };
    webkit_adapter_request_device(adapter, features.data(), features.size(), on_device, context);
    return future;
}

extern "C" WGPUStatus wgpuAdapterGetLimits(WGPUAdapter adapter, WGPULimits * limits) {
    webkit_limits webkit{};
    if (!webkit_adapter_get_limits(adapter, &webkit)) {
        return WGPUStatus_Error;
    }
    limits->maxBufferSize                      = webkit.max_buffer_size;
    limits->maxStorageBufferBindingSize       = webkit.max_storage_buffer_binding_size;
    limits->maxComputeInvocationsPerWorkgroup = webkit.max_compute_invocations_per_workgroup;
    limits->maxComputeWorkgroupStorageSize    = webkit.max_compute_workgroup_storage_size;
    limits->maxComputeWorkgroupsPerDimension  = webkit.max_compute_workgroups_per_dimension;
    limits->minStorageBufferOffsetAlignment   = webkit.min_storage_buffer_offset_alignment;
    limits->minUniformBufferOffsetAlignment   = webkit.min_uniform_buffer_offset_alignment;
    return WGPUStatus_Success;
}

static WGPUStringView copy_string(const char * string) {
    return string_view(string ? strdup(string) : nullptr);
}

extern "C" WGPUStatus wgpuAdapterGetInfo(WGPUAdapter adapter, WGPUAdapterInfo * info) {
    webkit_adapter_info webkit{};
    webkit_adapter_get_info(adapter, &webkit);
    info->vendor          = copy_string(webkit.vendor);
    info->architecture    = copy_string(webkit.architecture);
    info->device          = copy_string(webkit.device);
    info->description     = copy_string(webkit.description);
    info->vendorID        = webkit.vendor_id;
    info->deviceID        = webkit.device_id;
    info->adapterType     = static_cast<WGPUAdapterType>(webkit.adapter_type + 1);
    info->backendType     = WGPUBackendType_Metal;
    info->subgroupMinSize = webkit.subgroup_min_size;
    info->subgroupMaxSize = webkit.subgroup_max_size;
    return WGPUStatus_Success;
}

extern "C" void wgpuAdapterInfoFreeMembers(WGPUAdapterInfo info) {
    std::free(const_cast<char *>(info.vendor.data));
    std::free(const_cast<char *>(info.architecture.data));
    std::free(const_cast<char *>(info.device.data));
    std::free(const_cast<char *>(info.description.data));
}

extern "C" void wgpuAdapterPropertiesSubgroupMatrixConfigsFreeMembers(WGPUAdapterPropertiesSubgroupMatrixConfigs) {
}

extern "C" WGPUBuffer wgpuDeviceCreateBuffer(WGPUDevice device, const WGPUBufferDescriptor * descriptor) {
    return static_cast<WGPUBuffer>(webkit_device_create_buffer(device, descriptor->label.data, descriptor->label.length,
                                                                 descriptor->usage, descriptor->size, descriptor->mappedAtCreation));
}

struct map_callback_context {
    WGPUBufferMapCallbackInfo callback_info;
    std::shared_ptr<webkit_future> future;
};

static void on_buffer_map(int status, void * userdata) {
    std::unique_ptr<map_callback_context> context(static_cast<map_callback_context *>(userdata));
    const WGPUMapAsyncStatus dawn_status = status == 0 ? WGPUMapAsyncStatus_Success : WGPUMapAsyncStatus_Error;
    context->callback_info.callback(dawn_status, string_view(""), context->callback_info.userdata1,
                                    context->callback_info.userdata2);
    context->future->done = true;
}

extern "C" WGPUFuture wgpuBufferMapAsync(WGPUBuffer buffer, WGPUMapMode mode, size_t offset, size_t size,
                                           WGPUBufferMapCallbackInfo callback_info) {
    auto state = std::make_shared<webkit_future>();
    WGPUFuture future = make_future(state);
    auto * context = new map_callback_context{ callback_info, state };
    webkit_buffer_map_async(buffer, mode, offset, size, on_buffer_map, context);
    return future;
}

struct queue_callback_context {
    WGPUQueueWorkDoneCallbackInfo callback_info;
    std::shared_ptr<webkit_future> future;
};

static void on_queue_done(int status, void * userdata) {
    std::unique_ptr<queue_callback_context> context(static_cast<queue_callback_context *>(userdata));
    const WGPUQueueWorkDoneStatus dawn_status = status == 0 ? WGPUQueueWorkDoneStatus_Success : WGPUQueueWorkDoneStatus_Error;
    context->callback_info.callback(dawn_status, string_view(""), context->callback_info.userdata1,
                                    context->callback_info.userdata2);
    context->future->done = true;
}

extern "C" WGPUFuture wgpuQueueOnSubmittedWorkDone(WGPUQueue queue, WGPUQueueWorkDoneCallbackInfo callback_info) {
    auto state = std::make_shared<webkit_future>();
    WGPUFuture future = make_future(state);
    auto * context = new queue_callback_context{ callback_info, state };
    webkit_queue_on_submitted_work_done(queue, on_queue_done, context);
    return future;
}

static size_t view_length(WGPUStringView view) {
    if (!view.data) {
        return 0;
    }
    return view.length == WGPU_STRLEN ? std::strlen(view.data) : view.length;
}

extern "C" WGPUShaderModule wgpuDeviceCreateShaderModule(WGPUDevice device, const WGPUShaderModuleDescriptor * descriptor) {
    if (!descriptor || !descriptor->nextInChain || descriptor->nextInChain->sType != WGPUSType_ShaderSourceWGSL) {
        return nullptr;
    }
    const auto * source = reinterpret_cast<const WGPUShaderSourceWGSL *>(descriptor->nextInChain);
    return static_cast<WGPUShaderModule>(webkit_device_create_shader_module(device, source->code.data,
                                                                             view_length(source->code)));
}

extern "C" WGPUComputePipeline wgpuDeviceCreateComputePipeline(WGPUDevice device,
                                                                  const WGPUComputePipelineDescriptor * descriptor) {
    if (!descriptor || descriptor->layout || descriptor->compute.nextInChain) {
        return nullptr;
    }
    std::vector<webkit_constant> constants(descriptor->compute.constantCount);
    for (size_t i = 0; i < constants.size(); ++i) {
        const WGPUConstantEntry & entry = descriptor->compute.constants[i];
        constants[i] = { entry.key.data, view_length(entry.key), entry.value };
    }
    return static_cast<WGPUComputePipeline>(webkit_device_create_compute_pipeline(
        device, descriptor->label.data, view_length(descriptor->label), descriptor->compute.module,
        descriptor->compute.entryPoint.data, view_length(descriptor->compute.entryPoint), constants.data(), constants.size()));
}

extern "C" WGPUBindGroup wgpuDeviceCreateBindGroup(WGPUDevice device, const WGPUBindGroupDescriptor * descriptor) {
    if (!descriptor) {
        return nullptr;
    }
    std::vector<webkit_bind_entry> entries(descriptor->entryCount);
    for (size_t i = 0; i < entries.size(); ++i) {
        const WGPUBindGroupEntry & entry = descriptor->entries[i];
        if (entry.nextInChain || entry.sampler || entry.textureView || !entry.buffer) {
            return nullptr;
        }
        entries[i] = { entry.binding, entry.buffer, entry.offset, entry.size };
    }
    return static_cast<WGPUBindGroup>(webkit_device_create_bind_group(device, descriptor->layout,
                                                                       entries.data(), entries.size()));
}

extern "C" WGPUCommandEncoder wgpuDeviceCreateCommandEncoder(WGPUDevice device,
                                                                const WGPUCommandEncoderDescriptor * descriptor) {
    const WGPUStringView label = descriptor ? descriptor->label : WGPUStringView{};
    return static_cast<WGPUCommandEncoder>(webkit_device_create_command_encoder(device, label.data, view_length(label)));
}

extern "C" WGPUComputePassEncoder wgpuCommandEncoderBeginComputePass(WGPUCommandEncoder encoder,
                                                                        const WGPUComputePassDescriptor * descriptor) {
    if (descriptor && descriptor->timestampWrites) {
        return nullptr;
    }
    return static_cast<WGPUComputePassEncoder>(webkit_command_encoder_begin_compute_pass(encoder));
}

extern "C" WGPUCommandBuffer wgpuCommandEncoderFinish(WGPUCommandEncoder encoder,
                                                         const WGPUCommandBufferDescriptor *) {
    return static_cast<WGPUCommandBuffer>(webkit_command_encoder_finish(encoder));
}

extern "C" void wgpuComputePassEncoderSetBindGroup(WGPUComputePassEncoder pass, uint32_t group_index,
                                                     WGPUBindGroup bind_group, size_t offset_count,
                                                     const uint32_t * offsets) {
    webkit_compute_pass_set_bind_group(pass, group_index, bind_group, offsets, offset_count);
}
