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


struct webgpu_context {
    wgpu::Instance instance;
    // an adapter can only be used to create one device
    wgpu::Adapter adapter;
    // we only support one device for now
    wgpu::Device device;
};

static bool webgpu_context_initialized = false;
static webgpu_context webgpu_ctx;

static void ggml_webgpu_context_init() {
    if (webgpu_context_initialized) {
        return;
    }
    WEBGPU_LOG_DEBUG("ggml_webgpu_context_init()");

    wgpu::InstanceDescriptor instanceDescriptor{};
    instanceDescriptor.capabilities.timedWaitAnyEnable = true;
    webgpu_ctx.instance = wgpu::CreateInstance(&instanceDescriptor);
    GGML_ASSERT(webgpu_ctx.instance != nullptr);

    wgpu::RequestAdapterOptions options = {};
    wgpu::Adapter adapter;

    auto callback = [](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, const char *message, void *userdata) {
        if (status != wgpu::RequestAdapterStatus::Success) {
            GGML_LOG_ERROR("ggml_webgpu: Failed to get an adapter: %s\n", message);
            return;
        }
        *static_cast<wgpu::Adapter *>(userdata) = adapter;
    };

    auto callbackMode = wgpu::CallbackMode::WaitAnyOnly;
    void *userdata = &webgpu_ctx.adapter;
    webgpu_ctx.instance.WaitAny(webgpu_ctx.instance.RequestAdapter(&options, callbackMode, callback, userdata), UINT64_MAX);
    GGML_ASSERT(webgpu_ctx.adapter != nullptr);

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
    webgpu_ctx.instance.WaitAny(webgpu_ctx.adapter.RequestDevice(&deviceDescriptor, callbackMode,
        [](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView message) {
            if (status != wgpu::RequestDeviceStatus::Success) {
                GGML_LOG_ERROR("ggml_webgpu: Failed to get a device: %s\n", message.data);
                return;
            }
            webgpu_ctx.device = std::move(device);
        }),
        UINT64_MAX
    );
    GGML_ASSERT(webgpu_ctx.device != nullptr);

    wgpu::DawnAdapterPropertiesPowerPreference power_props{};
    wgpu::AdapterInfo info{};
    info.nextInChain = &power_props;
    webgpu_ctx.adapter.GetInfo(&info);
    GGML_LOG_INFO("ggml_webgpu: adapter_info: vendor_id: %u | vendor: %s | architecture: %s | device_id: %u | name: %s | device_desc: %s\n", 
        info.vendorID, info.vendor.data, info.architecture.data, info.deviceID, info.device.data, info.description.data);

    webgpu_context_initialized = true;
}

static ggml_backend_i ggml_backend_webgpu_interface = {
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
    static ggml_guid guid = { 0x9a, 0x5f, 0x3c, 0x2d, 0xb7, 0x1e, 0x47, 0xa1, 0x92, 0xcf, 0x16, 0x44, 0x58, 0xee, 0x90, 0x2b };
    return &guid;
}

// necessary??
ggml_backend_t ggml_backend_webgpu_init() {
    ggml_backend_t webgpu_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_webgpu_guid(),
        /* .interface = */ ggml_backend_webgpu_interface,
        /* .device    = */ NULL,
        /* .context   = */ NULL,
    };

    return webgpu_backend;
}

static const char * ggml_backend_webgpu_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_WEBGPU_NAME;
}

// Stub for now
static size_t ggml_backend_webgpu_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

// Stub for now
static ggml_backend_dev_t ggml_backend_webgpu_reg_get_device(ggml_backend_reg_t reg, size_t device) {
    static std::vector<ggml_backend_dev_t> devices;
    return devices[device];

}

static const struct ggml_backend_reg_i ggml_backend_webgpu_reg_i = {
    /* .get_name         = */ ggml_backend_webgpu_reg_get_name,
    /* .get_device_count = */ ggml_backend_webgpu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_webgpu_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_webgpu_reg() {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_webgpu_reg_i,
        /* .context     = */ nullptr,
    };
    // need to init webgpu here
    ggml_webgpu_context_init();
    return &reg;
}
