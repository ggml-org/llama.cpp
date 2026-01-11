
#include "buffer.hpp"
#include "common.hpp"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "host_device.hpp"
#include "profiler.hpp"

#include <memory>
#include <string>

namespace {

hexagon::npu_device * get_device_object(ggml_backend_dev_t device) {
    return reinterpret_cast<hexagon::npu_device *>(device->context);
}

hexagon::npu_device * get_device_object(ggml_backend_t backend) {
    return get_device_object(backend->device);
}

const char * backend_dev_get_name(ggml_backend_dev_t dev) {
    auto * dev_obj = get_device_object(dev);
    GGML_ASSERT(dev_obj != nullptr);
    return dev_obj->get_name();
}

const char * backend_dev_get_description(ggml_backend_dev_t dev) {
    auto * dev_obj = get_device_object(dev);
    GGML_ASSERT(dev_obj != nullptr);
    return dev_obj->get_description();
}

bool backend_dev_is_npu_device(ggml_backend_dev_t dev) {
    return dev->iface.get_name == backend_dev_get_name;
}

void backend_dev_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    *free  = common::get_system_free_memory_in_bytes();
    *total = common::get_system_total_memory_in_bytes();
}

enum ggml_backend_dev_type backend_dev_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);

    // TODO: figure out why the GGML_BACKEND_DEVICE_TYPE_ACCEL type will miss some ops
    return GGML_BACKEND_DEVICE_TYPE_IGPU;
}

void backend_dev_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    GGML_ASSERT(get_device_object(dev) != nullptr);
    props->name        = backend_dev_get_name(dev);
    props->description = backend_dev_get_description(dev);
    props->type        = backend_dev_get_type(dev);
    backend_dev_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {};
}

ggml_backend_t backend_dev_init_backend(ggml_backend_dev_t dev, const char * params) {
    auto * dev_obj = get_device_object(dev);
    GGML_ASSERT(dev_obj != nullptr);
    if (!dev_obj->init_device()) {
        LOG_ERROR("[%s]Failed to init device\n", backend_dev_get_name(dev));
        return nullptr;
    }

    SCOPED_PERFORMANCE_TRACKER("[hexagon-npu][%p]backend_dev_init_backend", (void *) dev_obj);
    return new hexagon::npu_backend(dev);
}

ggml_backend_buffer_type_t backend_dev_get_buffer_type(ggml_backend_dev_t dev) {
    auto * dev_obj = get_device_object(dev);
    GGML_ASSERT(dev_obj != nullptr);
    return dev_obj->get_default_buffer_type(dev);
}

ggml_backend_buffer_t backend_dev_buffer_from_host_ptr(ggml_backend_dev_t dev,
                                                       void *             ptr,
                                                       size_t             size,
                                                       size_t             max_tensor_size) {
    // TODO: should we use the device memory here?
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

bool backend_dev_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    if (!backend_dev_is_npu_device(dev)) {
        return false;
    }

    auto * dev_obj = get_device_object(dev);
    GGML_ASSERT(dev_obj != nullptr);

    SCOPED_PERFORMANCE_TRACKER("[hexagon-npu][%p]backend_dev_supports_op", (void *) dev_obj);
    return dev_obj->supports_op(op);
}

bool backend_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!backend_dev_is_npu_device(dev)) {
        return false;
    }

    auto * dev_obj = get_device_object(dev);
    GGML_ASSERT(dev_obj != nullptr);

    return dev_obj->supports_buft(buft);
}

bool backend_dev_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    if (!backend_dev_is_npu_device(dev)) {
        return false;
    }

    auto * dev_obj = get_device_object(dev);
    GGML_ASSERT(dev_obj != nullptr);

    SCOPED_PERFORMANCE_TRACKER("[hexagon-npu][%p]backend_dev_offload_op", (void *) dev_obj);
    return dev_obj->offload_op(op);
}

constexpr const ggml_backend_device_i npu_device_interface = {
    /* .get_name             = */ backend_dev_get_name,
    /* .get_description      = */ backend_dev_get_description,
    /* .get_memory           = */ backend_dev_get_memory,
    /* .get_type             = */ backend_dev_get_type,
    /* .get_props            = */ backend_dev_get_props,
    /* .init_backend         = */ backend_dev_init_backend,
    /* .get_buffer_type      = */ backend_dev_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ backend_dev_buffer_from_host_ptr,
    /* .supports_op          = */ backend_dev_supports_op,
    /* .supports_buft        = */ backend_dev_supports_buft,
    /* .offload_op           = */ backend_dev_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

class npu_device_proxy : public backend_device_proxy {
  public:
    explicit npu_device_proxy(backend_index_type device) { _device = std::make_unique<hexagon::npu_device>(device); }

    const ggml_backend_device_i & get_iface() const { return npu_device_interface; }

    void * get_context() { return _device.get(); }

  private:
    std::unique_ptr<hexagon::npu_device> _device;

    DISABLE_COPY(npu_device_proxy);
    DISABLE_MOVE(npu_device_proxy);
};

}  // namespace

backend_device_proxy_ptr create_hexagon_backend_context(backend_index_type device) {
    if (device < QNN_BACKEND_COUNT || device >= TOTAL_BACKEND_COUNT) {
        return backend_device_proxy_ptr();
    }

    return std::make_shared<npu_device_proxy>(device);
}
