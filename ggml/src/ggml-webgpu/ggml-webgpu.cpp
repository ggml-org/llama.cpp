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

// TODO: copied from Vulkan for now, not sure what it's used for
static void * const webgpu_ptr_base = (void *)(uintptr_t) 0x1000;  // NOLINT

/* Struct definitions */

// All the base objects needed to run operations on a WebGPU device
struct webgpu_context_struct {
    wgpu::Instance instance;
    wgpu::Adapter adapter;
    wgpu::Device device;
    // TODO: initialize
    wgpu::Queue queue;
    wgpu::Limits limits;
    // TODO: initialize
    wgpu::ComputePipeline memset_pipeline;
};

typedef std::shared_ptr<webgpu_context_struct> webgpu_context;

struct ggml_backend_webgpu_reg_context {
    webgpu_context webgpu_ctx;

    size_t device_count;
    const char * name;
};

struct ggml_backend_webgpu_device_context {
    webgpu_context webgpu_ctx;

    std::string device_name;
    std::string device_desc;
};

struct ggml_backend_webgpu_context {
    webgpu_context webgpu_ctx;

    std::string name;
};

struct ggml_backend_webgpu_buffer_context {
    webgpu_context webgpu_ctx;

    wgpu::Buffer buffer;

    ggml_backend_webgpu_buffer_context(webgpu_context ctx, wgpu::Buffer buf) :
        webgpu_ctx(ctx), buffer(buf) {
    }
};

/* End struct definitions */

/** GGML Backend Interface */

static const char * ggml_backend_webgpu_name(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *)backend->context;
    return ctx->name.c_str();
}

static void ggml_backend_webgpu_free(ggml_backend_t backend) {
    ggml_backend_webgpu_context * ctx = (ggml_backend_webgpu_context *)backend->context;
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_free(" << ctx->name << ")");

    // TODO: cleanup
}

static ggml_backend_i ggml_backend_webgpu_i = {
    /* .get_name                = */ ggml_backend_webgpu_name,
    /* .free                    = */ ggml_backend_webgpu_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL, // TODO
    /* .graph_compute           = */ NULL,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

/* End GGML Backend Interface */

/* GGML Backend Buffer Interface */

static void ggml_backend_webgpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_webgpu_buffer_context * ctx = static_cast<ggml_backend_webgpu_buffer_context *>(buffer->context);
    ctx->buffer.Destroy();
    delete ctx;
}

// TODO: what to return here?
static void * ggml_backend_webgpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return webgpu_ptr_base;
}

// TODO
static void ggml_backend_webgpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_memset_tensor(" << buffer << ", " << tensor << ", " << value << ", " << offset << ", " << size << ")");
}

// TODO
static void ggml_backend_webgpu_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_set_tensor(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size << ")");
}

// TODO
static void ggml_backend_webgpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_get_tensor(" << buffer << ", " << tensor << ", " << data << ", " << offset << ", " << size << ")");
}

// TODO
static bool ggml_backend_webgpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);

    return true;
}

// TODO
static void ggml_backend_webgpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static ggml_backend_buffer_i ggml_backend_webgpu_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_webgpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_webgpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // TODO: should we implement this?
    /* .memset_tensor   = */ ggml_backend_webgpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_webgpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_webgpu_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_webgpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_webgpu_buffer_clear,
    /* .reset           = */ NULL,
};

/* End GGML Backend Buffer Interface */

/* GGML Backend Buffer Type Interface */

static const char * ggml_backend_webgpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    return ctx->device_name.c_str();
}

static ggml_backend_buffer_t ggml_backend_webgpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_buffer_type_alloc_buffer(" << size << ")");
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);

    wgpu::BufferDescriptor buf_desc;
    buf_desc.mappedAtCreation = false;
    buf_desc.size = size;
    buf_desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
    // TODO: error handling
    wgpu::Buffer buf = ctx->webgpu_ctx->device.CreateBuffer(&buf_desc);

    ggml_backend_webgpu_buffer_context * buf_ctx = new ggml_backend_webgpu_buffer_context(ctx->webgpu_ctx, buf);

    return ggml_backend_buffer_init(buft, ggml_backend_webgpu_buffer_interface, buf_ctx, size);
}

static size_t ggml_backend_webgpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    return ctx->webgpu_ctx->limits.minStorageBufferOffsetAlignment;
}

static size_t ggml_backend_webgpu_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    return ctx->webgpu_ctx->limits.maxBufferSize;
}

/* End GGML Backend Buffer Type Interface */

/* GGML Backend Host Buffer Type Interface */

static const char * ggml_backend_webgpu_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_WEBGPU_NAME "_Host";
}

// WebGPU doesn't specify a memory map alignment like Vulkan, so we use the same value as the storage buffer alignment
static size_t ggml_backend_webgpu_host_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(buft->device->context);
    return ctx->webgpu_ctx->limits.minStorageBufferOffsetAlignment;
}

/* End GGML Backend Host Buffer Type Interface */

/* GGML Backend Device Interface */

static const char * ggml_backend_webgpu_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    return ctx->device_name.c_str();
}

static const char * ggml_backend_webgpu_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    return ctx->device_desc.c_str();
}

static void ggml_backend_webgpu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_webgpu_device_context * ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);
    // TODO: what do we actually want to return here?
    *free = ctx->webgpu_ctx->limits.maxBufferSize * WEBGPU_MAX_BUFFERS;
    *total = ctx->webgpu_ctx->limits.maxBufferSize * WEBGPU_MAX_BUFFERS;
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

static ggml_guid_t ggml_backend_webgpu_guid(void) {
    static const char * guid_str = "__ggml_webgpu :)";
    return reinterpret_cast<ggml_guid_t>((void *)guid_str);
}

// TODO: Does this need to be thread safe? Is it only called once?
// Implementation in GGML Backend Interface section
static ggml_backend_t ggml_backend_webgpu_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);

    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_device_init()");

    ggml_backend_webgpu_device_context * dev_ctx = static_cast<ggml_backend_webgpu_device_context *>(dev->context);

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
    webgpu_context webgpu_ctx = dev_ctx->webgpu_ctx;
    dev_ctx->webgpu_ctx->instance.WaitAny(dev_ctx->webgpu_ctx->adapter.RequestDevice(&deviceDescriptor, wgpu::CallbackMode::WaitAnyOnly,
        [webgpu_ctx](wgpu::RequestDeviceStatus status, wgpu::Device device, wgpu::StringView message) {
            if (status != wgpu::RequestDeviceStatus::Success) {
                GGML_LOG_ERROR("ggml_webgpu: Failed to get a device: %s\n", message.data);
                return;
            }
            webgpu_ctx->device = device;
        }),
        UINT64_MAX
    );
    GGML_ASSERT(dev_ctx->webgpu_ctx->device != nullptr);

    static ggml_backend_webgpu_context backend_ctx;
    backend_ctx.name = GGML_WEBGPU_NAME + std::string(": ") + dev_ctx->device_name;
    backend_ctx.webgpu_ctx = dev_ctx->webgpu_ctx;

    static ggml_backend backend = {
        /* .guid      = */ ggml_backend_webgpu_guid(),
        /* .interface = */ ggml_backend_webgpu_i,
        /* .device    = */ dev,
        /* .context   = */ &backend_ctx,
    };
    return &backend;
}

// Implementation in GGML Backend Buffer Type Interface section
static ggml_backend_buffer_type_t ggml_backend_webgpu_device_get_buffer_type(ggml_backend_dev_t dev) {
    static struct ggml_backend_buffer_type ggml_backend_webgpu_buffer_type = {
        /* .iface = */ {
            /* .get_name         = */ ggml_backend_webgpu_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_webgpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_webgpu_buffer_type_get_alignment,
            /* .get_max_size     = */ ggml_backend_webgpu_buffer_type_get_max_size,
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ NULL, // defaults to false
        },
        /* .device  = */ dev,
        /* .context = */ NULL,
    };

    return &ggml_backend_webgpu_buffer_type;
}


// Implementation in GGML Backend Host Buffer Type Interface
static ggml_backend_buffer_type_t ggml_backend_webgpu_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    static struct ggml_backend_buffer_type ggml_backend_webgpu_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_webgpu_host_buffer_type_name,
            /* .alloc_buffer     = */ NULL, // TODO
            /* .get_alignment    = */ ggml_backend_webgpu_host_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ dev,
        /* .context  = */ NULL,
    };

    return &ggml_backend_webgpu_buffer_type_host;
}

static bool ggml_backend_webgpu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return  buft->iface.get_name == ggml_backend_webgpu_buffer_type_get_name;
}

static bool ggml_backend_webgpu_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);

    // what should we support first?
    switch (op->op) {
        default:
            return false;
    }
}

static struct ggml_backend_device_i ggml_backend_webgpu_device_i = {
    /* .get_name             = */ ggml_backend_webgpu_device_get_name,
    /* .get_description      = */ ggml_backend_webgpu_device_get_description,
    /* .get_memory           = */ ggml_backend_webgpu_device_get_memory,
    /* .get_type             = */ ggml_backend_webgpu_device_get_type,
    /* .get_props            = */ ggml_backend_webgpu_device_get_props,
    /* .init_backend         = */ ggml_backend_webgpu_device_init,
    /* .get_buffer_type      = */ ggml_backend_webgpu_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_webgpu_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_webgpu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_webgpu_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

/* End GGML Backend Device Interface */

/* GGML Backend Registration Interface */

static const char * ggml_backend_webgpu_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    return ctx->name;
}

static size_t ggml_backend_webgpu_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_webgpu_reg_context * ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);
    return ctx->device_count;
}

// TODO: Does this need to be thread safe? Is it only called once?
// Only one device is supported for now
// Implementation in GGML Backend Device Interface section
static ggml_backend_dev_t ggml_backend_webgpu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    WEBGPU_LOG_DEBUG("ggml_backend_reg_get_device()");

    ggml_backend_webgpu_reg_context * reg_ctx = static_cast<ggml_backend_webgpu_reg_context *>(reg->context);

    webgpu_context ctx = reg_ctx->webgpu_ctx;

    wgpu::RequestAdapterOptions options = {};
    auto callback = [](wgpu::RequestAdapterStatus status, wgpu::Adapter adapter, const char *message, void *userdata) {
        if (status != wgpu::RequestAdapterStatus::Success) {
            GGML_LOG_ERROR("ggml_webgpu: Failed to get an adapter: %s\n", message);
            return;
        }
        *static_cast<wgpu::Adapter *>(userdata) = adapter;
    };
    void *userdata = &ctx->adapter;
    ctx->instance.WaitAny(ctx->instance.RequestAdapter(&options, wgpu::CallbackMode::WaitAnyOnly, callback, userdata), UINT64_MAX);
    GGML_ASSERT(ctx->adapter != nullptr);

    ctx->adapter.GetLimits(&ctx->limits);

    wgpu::AdapterInfo info{};
    ctx->adapter.GetInfo(&info);

    static ggml_backend_webgpu_device_context device_ctx;
    device_ctx.webgpu_ctx = ctx;
    device_ctx.device_name = std::string(info.device.data);
    device_ctx.device_desc = std::string(info.description.data);

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

/* End GGML Backend Registration Interface */

// TODO: Does this need to be thread safe? Is it only called once?
ggml_backend_reg_t ggml_backend_webgpu_reg() {
    WEBGPU_LOG_DEBUG("ggml_backend_webgpu_reg()");

    webgpu_context webgpu_ctx = std::make_shared<webgpu_context_struct>();

    static ggml_backend_webgpu_reg_context ctx;
    ctx.webgpu_ctx = webgpu_ctx;
    ctx.name = GGML_WEBGPU_NAME;
    ctx.device_count = 1;


    wgpu::InstanceDescriptor instanceDescriptor{};
    instanceDescriptor.capabilities.timedWaitAnyEnable = true;
    webgpu_ctx->instance = wgpu::CreateInstance(&instanceDescriptor);
    GGML_ASSERT(webgpu_ctx->instance != nullptr);

    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_webgpu_reg_i,
        /* .context     = */ &ctx,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_webgpu_reg)