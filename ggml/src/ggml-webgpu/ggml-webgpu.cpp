#include "ggml-webgpu.h"

#include <webgpu/webgpu_cpp.h>
#include <webgpu/webgpu_cpp_print.h>

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <iostream>
#include <vector>

#ifdef GGML_WEBGPU_DEBUG
#define WEBGPU_LOG_DEBUG(msg) std::cout << msg << std::endl
#else
#define WEBGPU_LOG_DEBUG(msg) ((void) 0)
#endif // GGML_WEBGPU_DEBUG

// TODO: find a better way to get the memory available
#define WEBGPU_MAX_BUFFERS 32

// When registering the backend, we initialize the WebGPU instance.
struct webgpu_reg_context {
    wgpu::Instance instance;
    size_t device_count;
    const char * name;
};

// When getting the (ggml) device, we create a WebGPU adapter and its associated WebGPU device.
struct webgpu_device_context {
    // An adapter can only be used to create one device
    wgpu::Adapter adapter;
    wgpu::Device device;
    wgpu::StringView device_name;
    wgpu::StringView device_desc;
};

struct webgpu_backend_context {
    wgpu::Device device;
};

static ggml_backend_i ggml_backend_webgpu_i = {
    /* .get_name                = */ NULL,
    /* .free                    = */ NULL,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ NULL,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

static ggml_guid_t ggml_backend_webgpu_guid(void) {
    static const char * guid_str = "__ggml_webgpu :)";
    return reinterpret_cast<ggml_guid_t>((void *)guid_str);
}

static const char * ggml_backend_webgpu_device_get_name(ggml_backend_dev_t dev) {
    webgpu_device_context * ctx = static_cast<webgpu_device_context *>(dev->context);
    return ctx->device_name.data;
}

static const char * ggml_backend_webgpu_device_get_description(ggml_backend_dev_t dev) {
    webgpu_device_context * ctx = static_cast<webgpu_device_context *>(dev->context);
    return ctx->device_desc.data;
}

static void ggml_backend_webgpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    webgpu_device_context * ctx = static_cast<webgpu_device_context *>(dev->context);
    wgpu::Limits limits;
    ctx->device.GetLimits(&limits);
    // TODO: what do we actually want to return here?
    *free = limits.maxBufferSize * WEBGPU_MAX_BUFFERS;
    *total = limits.maxBufferSize * WEBGPU_MAX_BUFFERS;
}

static enum ggml_backend_dev_type ggml_backend_webgpu_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_webgpu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_webgpu_device_get_name(dev);
    props->description = ggml_backend_webgpu_device_get_description(dev);
    props->type        = ggml_backend_webgpu_device_get_type(dev);
    ggml_backend_webgpu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ true, // maybe? not sure what this means yet
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

// TODO: Does this need to be thread safe? Is it only called once?
static ggml_backend_t ggml_backend_webgpu_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);

    webgpu_device_context * dev_ctx = static_cast<webgpu_device_context *>(dev->context);

    static webgpu_backend_context backend_ctx;
    backend_ctx.device = dev_ctx->device;

    static ggml_backend backend = {
        /* .guid      = */ ggml_backend_webgpu_guid(),
        /* .interface = */ ggml_backend_webgpu_i,
        /* .device    = */ dev,
        /* .context   = */ &backend_ctx,
    };
    return &backend;
}

static ggml_backend_buffer_type_t ggml_backend_webgpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    static struct ggml_backend_buffer_type ggml_backend_buffer_type_webgpu = {
        /* .iface = */ {
            /* .get_name         = */ NULL,
            /* .alloc_buffer     = */ NULL,
            /* .get_alignment    = */ NULL,
            /* .get_max_size     = */ NULL,
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ NULL,
        },
        /* .device  = */ dev,
        /* .context = */ NULL,
    };
    return &ggml_backend_buffer_type_webgpu;
}

static struct ggml_backend_device_i ggml_backend_webgpu_device_i = {
    /* .get_name             = */ ggml_backend_webgpu_device_get_name,
    /* .get_description      = */ ggml_backend_webgpu_device_get_description,
    /* .get_memory           = */ ggml_backend_webgpu_device_get_memory,
    /* .get_type             = */ ggml_backend_webgpu_device_get_type,
    /* .get_props            = */ ggml_backend_webgpu_device_get_props,
    /* .init_backend         = */ ggml_backend_webgpu_device_init,
    /* .get_buffer_type      = */ ggml_backend_webgpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ NULL,
    /* .supports_buft        = */ NULL,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

static const char * ggml_backend_webgpu_reg_get_name(ggml_backend_reg_t reg) {
    webgpu_reg_context * ctx = static_cast<webgpu_reg_context *>(reg->context);
    return ctx->name;
}

static size_t ggml_backend_webgpu_reg_get_device_count(ggml_backend_reg_t reg) {
    webgpu_reg_context * ctx = static_cast<webgpu_reg_context *>(reg->context);
    return ctx->device_count;
}

// TODO: Does this need to be thread safe? Is it only called once?
// Only one device is supported for now
static ggml_backend_dev_t ggml_backend_webgpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    WEBGPU_LOG_DEBUG("ggml_backend_reg_get_device()");

    webgpu_reg_context * reg_ctx = static_cast<webgpu_reg_context *>(reg->context);
    static webgpu_device_context device_ctx;

    wgpu::RequestAdapterOptions options = {};
    auto callback = [](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, const char *message, void *userdata) {
        if (status != wgpu::RequestAdapterStatus::Success) {
            GGML_LOG_ERROR("ggml_webgpu: Failed to get an adapter: %s\n", message);
            return;
        }
        *static_cast<wgpu::Adapter *>(userdata) = adapter;
    };
    auto callbackMode = wgpu::CallbackMode::WaitAnyOnly;
    void *userdata = &device_ctx.adapter;
    reg_ctx->instance.WaitAny(reg_ctx->instance.RequestAdapter(&options, callbackMode, callback, userdata), UINT64_MAX);
    GGML_ASSERT(device_ctx.adapter != nullptr);

    wgpu::DeviceDescriptor deviceDescriptor;
    deviceDescriptor.SetDeviceLostCallback(wgpu::CallbackMode::AllowSpontaneous, 
        [](const wgpu::Device& device, wgpu::DeviceLostReason reason, wgpu::StringView message) {
            GGML_UNUSED(device);
            GGML_LOG_ERROR("ggml_webgpu: Device lost! Reason: %d, Message: %s\n", static_cast<int>(reason), message.data);
    });
    deviceDescriptor.SetUncapturedErrorCallback(
        [](const wgpu::Device& device, wgpu::ErrorType reason, wgpu::StringView message) {
            GGML_UNUSED(device);
            GGML_LOG_ERROR("ggml_webgpu: Device error! Reason: %d, Message: %s\n", static_cast<int>(reason), message.data);
    });
    reg_ctx->instance.WaitAny(device_ctx.adapter.RequestDevice(&deviceDescriptor, callbackMode,
        [](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView message) {
            if (status != wgpu::RequestDeviceStatus::Success) {
                GGML_LOG_ERROR("ggml_webgpu: Failed to get a device: %s\n", message.data);
                return;
            }
            device_ctx.device = std::move(device);
        }),
        UINT64_MAX
    );
    GGML_ASSERT(device_ctx.device != nullptr);

    wgpu::AdapterInfo info{};
    device_ctx.adapter.GetInfo(&info);
    device_ctx.device_name = info.device;
    device_ctx.device_desc = info.description;
    GGML_LOG_INFO("ggml_webgpu: adapter_info: vendor_id: %u | vendor: %s | architecture: %s | device_id: %u | name: %s | device_desc: %s\n", 
        info.vendorID, info.vendor.data, info.architecture.data, info.deviceID, info.device.data, info.description.data);

    static ggml_backend_device device = {
        /* .iface   = */ ggml_backend_webgpu_device_i,
        /* .reg     = */ reg,
        /* .context = */ &device_ctx,
    };
    return &device;
}

static const struct ggml_backend_reg_i ggml_backend_webgpu_reg_i = {
    /* .get_name         = */ ggml_backend_webgpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_webgpu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_webgpu_reg_get_device,
    /* .get_proc_address = */ NULL,
};

// TODO: Does this need to be thread safe? Is it only called once?
ggml_backend_reg_t ggml_backend_webgpu_reg() {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_reg()");

    static webgpu_reg_context ctx;
    ctx.name = GGML_WEBGPU_NAME;
    ctx.device_count = 1;

    wgpu::InstanceDescriptor instanceDescriptor{};
    instanceDescriptor.capabilities.timedWaitAnyEnable = true;
    ctx.instance = wgpu::CreateInstance(&instanceDescriptor);
    GGML_ASSERT(ctx.instance != nullptr);

    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_webgpu_reg_i,
        /* .context     = */ &ctx,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_webgpu_reg)