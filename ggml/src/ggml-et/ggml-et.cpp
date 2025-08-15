#include "ggml-et.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"

#include <device-layer/IDeviceLayer.h>
#include <runtime/IRuntime.h>
#include <cstring>

/*
  ET Driver.

  `ggml_et_driver()` handles both the device layer and the runtime,
  for doing actual operations on devices.
*/

static struct ggml_et_driver {
    std::shared_ptr<dev::IDeviceLayer> device_layer;
    std::shared_ptr<rt::IRuntime> runtime;
} _drv;

static bool ggml_et_driver_init() {
    if (_drv.runtime != nullptr) {
	assert(_drv.device_layer != nullptr);
    } else {
	try {
	    _drv.device_layer = dev::IDeviceLayer::createPcieDeviceLayer();
	    _drv.runtime = rt::IRuntime::create(_drv.device_layer);
	    GGML_LOG_INFO("ET: FOUND %d devices!\n", _drv.device_layer->getDevicesCount());
	} catch (const std::exception& e) {
	    GGML_LOG_ERROR("ggml_et: %s", e.what());
	    if (_drv.device_layer != nullptr)
		_drv.device_layer.reset();
	    if (_drv.runtime != nullptr)
		_drv.runtime.reset();
	    return false;
	}
    }
    return true;
}

static std::shared_ptr<dev::IDeviceLayer> ggml_et_devicelayer() {
    return _drv.device_layer;
}

static std::shared_ptr<rt::IRuntime> ggml_et_runtime() {
    return _drv.runtime;
}

struct ggml_backend_et_buffer_type_context {
    int devidx;
    std::string name;
};

struct ggml_backend_et_buffer_context {
    int devidx;
    void * data;
    size_t size;
};

struct ggml_backend_et_context {
    int devidx;
};

struct ggml_backend_et_device_context {
    int devidx;
    std::string name;
    std::string desc;
    size_t total_mem;
    ggml_backend_buffer_type_t buftype;
};

static void ggml_backend_et_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_et_buffer_context * ctx = (ggml_backend_et_buffer_context *)buffer->context;
    if (ctx->data != nullptr) {
        free(ctx->data);
    }
    delete ctx;
}

static void * ggml_backend_et_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_et_buffer_context * ctx = (ggml_backend_et_buffer_context *)buffer->context;
    return ctx->data;
}

static enum ggml_status ggml_backend_et_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_et_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
    GGML_UNUSED(data);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
}

static void ggml_backend_et_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
    GGML_UNUSED(data);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
}

static bool ggml_backend_et_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    return false;
}

static void ggml_backend_et_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static const struct ggml_backend_buffer_i ggml_backend_et_buffer_i = {
    /* .free_buffer     = */ ggml_backend_et_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_et_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_et_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_et_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_et_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_et_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_et_buffer_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_et_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_ET_NAME;
}

static ggml_backend_buffer_t ggml_backend_et_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_et_buffer_type_context * btctx = (ggml_backend_et_buffer_type_context *)buft->context;

    ggml_backend_et_buffer_context * ctx = new ggml_backend_et_buffer_context;
    ctx->devidx = btctx->devidx;
    ctx->data = malloc(size);
    ctx->size = size;

    if (ctx->data == nullptr) {
        delete ctx;
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_et_buffer_i, ctx, size);
}

static size_t ggml_backend_et_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_MEM_ALIGN;
}

static size_t ggml_backend_et_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return SIZE_MAX;
}

static size_t ggml_backend_et_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    return ggml_nbytes_pad(tensor);
}

static bool ggml_backend_et_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

static const struct ggml_backend_buffer_type_i ggml_backend_et_buffer_type_i = {
    /* .get_name         = */ ggml_backend_et_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_et_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_et_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_et_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_et_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_et_buffer_type_is_host,
};

static const char * ggml_backend_et_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return GGML_ET_NAME;
}

static void ggml_backend_et_free(ggml_backend_t backend) {
    ggml_backend_et_context * et_ctx = (ggml_backend_et_context *)backend->context;
    delete et_ctx;
    delete backend;
}

static ggml_backend_buffer_type_t ggml_backend_et_get_default_buffer_type(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    // Return CPU buffer type to ensure all tensors allocated on CPU-accessible memory
    return ggml_backend_cpu_buffer_type();
}

static void ggml_backend_et_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_UNUSED(backend);
    GGML_UNUSED(tensor);
    GGML_UNUSED(data);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
}

static void ggml_backend_et_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_UNUSED(backend);
    GGML_UNUSED(tensor);
    GGML_UNUSED(data);
    GGML_UNUSED(offset);
    GGML_UNUSED(size);
}

static bool ggml_backend_et_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(backend_src);
    GGML_UNUSED(backend_dst);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
    return false;
}

static void ggml_backend_et_synchronize(ggml_backend_t backend) {
    GGML_UNUSED(backend);
}

static enum ggml_status ggml_backend_et_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);
    GGML_UNUSED(cgraph);
    // Return success but perform no computation - fallback to other backends
    return GGML_STATUS_SUCCESS;
}

static bool ggml_backend_et_supports_op(ggml_backend_t backend, const ggml_tensor * op) {
    GGML_UNUSED(backend);
    GGML_UNUSED(op);
    return false;
}

static bool ggml_backend_et_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(backend);
    // Only support host (CPU) buffer types to avoid allocation issues
    return ggml_backend_buft_is_host(buft);
}

static bool ggml_backend_et_offload_op(ggml_backend_t backend, const ggml_tensor * op) {
    GGML_UNUSED(backend);
    GGML_UNUSED(op);
    return false;
}

static bool ggml_backend_et_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    return false;
}

static bool ggml_backend_et_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    // Only support host (CPU) buffer types
    return ggml_backend_buft_is_host(buft);
}

static bool ggml_backend_et_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    return false;
}

static const struct ggml_backend_i ggml_backend_et_i = {
    /* .get_name                = */ ggml_backend_et_get_name,
    /* .free                    = */ ggml_backend_et_free,
    /* .set_tensor_async        = */ ggml_backend_et_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_et_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_et_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_et_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_et_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
};

static const char * ggml_backend_et_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_ET_NAME;
}

static const char * ggml_backend_et_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)dev->context;
    return dev_ctx->desc.c_str();
}

static void ggml_backend_et_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)dev->context;
    // Currently getFreeMemory is not available on a runtime without server.
    // For now, report total memory as free.
    *free = dev_ctx->total_mem;
    *total = dev_ctx->total_mem;
}

static enum ggml_backend_dev_type ggml_backend_et_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_et_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    GGML_UNUSED(dev);
    props->name        = ggml_backend_et_device_get_name(dev);
    props->description = ggml_backend_et_device_get_description(dev);
    props->type        = ggml_backend_et_device_get_type(dev);
    ggml_backend_et_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_et_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)dev->context;
    return ggml_backend_et_init(dev_ctx->devidx);
}

static ggml_backend_buffer_type_t ggml_backend_et_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    // Return CPU buffer type to ensure all tensors allocated on CPU-accessible memory
    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_type_t ggml_backend_et_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    // Return CPU buffer type for host buffer type
    return ggml_backend_cpu_buffer_type();
}

static const struct ggml_backend_device_i ggml_backend_et_device_i = {
    /* .get_name          = */ ggml_backend_et_device_get_name,
    /* .get_description   = */ ggml_backend_et_device_get_description,
    /* .get_memory        = */ ggml_backend_et_device_get_memory,
    /* .get_type          = */ ggml_backend_et_device_get_type,
    /* .get_props         = */ ggml_backend_et_device_get_props,
    /* .init_backend      = */ ggml_backend_et_device_init_backend,
    /* .get_buffer_type   = */ ggml_backend_et_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_et_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op       = */ ggml_backend_et_device_supports_op,
    /* .supports_buft     = */ ggml_backend_et_device_supports_buft,
    /* .offload_op        = */ ggml_backend_et_device_offload_op,
    /* .event_new         = */ NULL,
    /* .event_free        = */ NULL,
    /* .event_synchronize = */ NULL,
};


/*
  Backend Registry.
*/

struct ggml_backend_et_reg_ctx {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_et_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_ET_NAME;
}

static size_t ggml_backend_et_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_et_reg_ctx * ctx = (ggml_backend_et_reg_ctx *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_et_reg_get_device(ggml_backend_reg_t reg, size_t devidx) {
    ggml_backend_et_reg_ctx * ctx = (ggml_backend_et_reg_ctx *)reg->context;
    if (devidx >= ctx->devices.size()) {
        return nullptr;
    }
    return ctx->devices[devidx];
}

static void * ggml_backend_et_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return nullptr;
}

static const struct ggml_backend_reg_i ggml_backend_et_reg_i = {
    /* .get_name         = */ ggml_backend_et_reg_get_name,
    /* .get_device_count = */ ggml_backend_et_reg_get_device_count,
    /* .get_device       = */ ggml_backend_et_reg_get_device,
    /* .get_proc_address = */ ggml_backend_et_get_proc_address,
};

ggml_backend_reg_t ggml_backend_et_reg(void) {
    static ggml_backend_reg_t _reg = []() -> ggml_backend_reg_t {
	ggml_backend_et_reg_ctx * ctx = new ggml_backend_et_reg_ctx;

	if (!ggml_et_driver_init())
	    return nullptr;

	std::vector<rt::DeviceId> rtids = ggml_et_runtime()->getDevices();

        for (int i = 0; i < ggml_et_devicelayer()->getDevicesCount(); i++) {
	    ggml_backend_dev_t dev = new ggml_backend_device {
		/* .iface   = */ ggml_backend_et_device_i,
		/* .reg     = */ _reg,
		/* .context = */ nullptr // Set later
	    };

	    rt::DeviceId rtid = rtids[i];
	    rt::DeviceProperties prop = ggml_et_runtime()->getDeviceProperties(rtid);

	    // Create device context.
	    ggml_backend_et_device_context * dev_ctx = new ggml_backend_et_device_context;
	    dev_ctx->devidx = i;
	    dev_ctx->name = GGML_ET_NAME + std::to_string(i);
	    dev_ctx->desc = "ET device " + std::to_string(i);
	    dev_ctx->total_mem = static_cast<size_t>(prop.memorySize_);
	    // Add buffer type for device to device context.
	    ggml_backend_et_buffer_type_context * bufty_ctx = new ggml_backend_et_buffer_type_context;
	    bufty_ctx->devidx = i;
	    bufty_ctx->name = GGML_ET_NAME + std::to_string(i);
	    dev_ctx->buftype = new ggml_backend_buffer_type {
		/* .iface   = */ ggml_backend_et_buffer_type_i,
		/* .device  = */ dev,
		/* .context = */ bufty_ctx
	    };
	    dev->context = dev_ctx;

	    ctx->devices.push_back(dev);
	}

	ggml_backend_reg_t r = new ggml_backend_reg {
	    /* .api_version = */ GGML_BACKEND_API_VERSION,
	    /* .iface       = */ ggml_backend_et_reg_i,
	    /* .context     = */ ctx,
	};
	return r;
    }();

    return _reg;
}

ggml_guid_t ggml_backend_et_guid(void) {
    static ggml_guid guid = { 0x4b, 0xe0, 0x72, 0x88, 0xc0, 0xf6, 0x29, 0xb4, 0x79, 0x9f, 0x70, 0x68, 0x71, 0x0f, 0x6d, 0xc8 };
    return &guid;
}

ggml_backend_t ggml_backend_et_init(size_t devidx) {
    if (!ggml_et_driver_init())
	return nullptr;

    if (devidx >= (size_t)ggml_backend_et_get_device_count()) {
        return nullptr;
    }

    ggml_backend_et_context * ctx = new ggml_backend_et_context;
    ctx->devidx = (int)devidx;

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_et_guid(),
        /* .iface   = */ ggml_backend_et_i,
        /* .device  = */ ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_et(ggml_backend_t backend) {
    return backend != nullptr && backend->iface.get_name == ggml_backend_et_get_name;
}

int ggml_backend_et_get_device_count(void) {
    return ggml_backend_et_reg_get_device_count(ggml_backend_et_reg());
}

void ggml_backend_et_get_device_description(int devidx, char * description, size_t description_size) {
    if (devidx < 0 || devidx >= ggml_backend_et_get_device_count()) {
        snprintf(description, description_size, "ET Device %d (invalid)", devidx);
        return;
    }

    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)dev->context;
    snprintf(description, description_size, "%s", dev_ctx->desc.c_str());
}

void ggml_backend_et_get_device_memory(int devidx, size_t * free, size_t * total) {
    if (devidx < 0 || devidx >= ggml_backend_et_get_device_count()) {
        *free = 0;
        *total = 0;
        return;
    }

    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx);
    ggml_backend_et_device_get_memory(dev, free, total);
}

ggml_backend_buffer_type_t ggml_backend_et_buffer_type(size_t dev_num) {
    if (dev_num >= (size_t)ggml_backend_et_get_device_count()) {
        return nullptr;
    }

    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), dev_num);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *)dev->context;
    return dev_ctx->buftype;
}

ggml_backend_buffer_type_t ggml_backend_et_host_buffer_type(void) {
    static ggml_backend_buffer_type host_buffer_type = {
        /* .iface   = */ ggml_backend_et_buffer_type_i,
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    return &host_buffer_type;
}
