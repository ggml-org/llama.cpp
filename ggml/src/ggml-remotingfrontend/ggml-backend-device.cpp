#include "ggml-remoting.h"

#define DEV_TO_GPU(name) \
  ((struct ggml_backend_remoting_device_context *) (name)->context)->gpu

static const char *
ggml_backend_remoting_device_get_name(ggml_backend_dev_t dev) {
  IMPLEMENTED;

  struct virtgpu *gpu = DEV_TO_GPU(dev);

  return apir_device_get_name(gpu);
}

static const char *
ggml_backend_remoting_device_get_description(ggml_backend_dev_t dev) {
  IMPLEMENTED;

  struct virtgpu *gpu = DEV_TO_GPU(dev);

  return apir_device_get_description(gpu);
}

static enum ggml_backend_dev_type
ggml_backend_remoting_device_get_type(ggml_backend_dev_t dev) {
  IMPLEMENTED;

  struct virtgpu *gpu = DEV_TO_GPU(dev);

  return (enum ggml_backend_dev_type) apir_device_get_type(gpu);
}

static void
ggml_backend_remoting_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
  IMPLEMENTED;

  struct virtgpu *gpu = DEV_TO_GPU(dev);

  return apir_device_get_memory(gpu, free, total);
}

static bool
ggml_backend_remoting_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
  IMPLEMENTED;

  struct virtgpu *gpu = DEV_TO_GPU(dev);

  return apir_device_supports_op(gpu, op);
}

static bool
ggml_backend_remoting_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
  UNUSED(dev);
  UNUSED(buft);

  NOT_IMPLEMENTED;

  return true;
}

static bool
ggml_backend_remoting_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
  const int min_batch_size = 32;

  NOT_IMPLEMENTED;

  return (op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS) ||
    (op->ne[2] >= min_batch_size && op->op == GGML_OP_MUL_MAT_ID);

  UNUSED(dev);
}

static void
ggml_backend_remoting_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
  IMPLEMENTED;

  struct virtgpu *gpu = DEV_TO_GPU(dev);

  props->name        = ggml_backend_remoting_device_get_name(dev);
  props->description = ggml_backend_remoting_device_get_description(dev);
  props->type        = ggml_backend_remoting_device_get_type(dev);
  ggml_backend_remoting_device_get_memory(dev, &props->memory_free, &props->memory_total);

  apir_device_get_props(gpu,
			&props->caps.async,
			&props->caps.host_buffer,
			&props->caps.buffer_from_host_ptr,
			&props->caps.events
    );

  INFO("%s: async=%d, host_buffer=%d, buffer_from_host_ptr=%d, events=%d",
    __func__, props->caps.async, props->caps.host_buffer,
       props->caps.buffer_from_host_ptr, props->caps.events);
}

ggml_backend_buffer_type_t
ggml_backend_remoting_device_get_buffer_type(ggml_backend_dev_t dev) {
  IMPLEMENTED;

  struct virtgpu *gpu = DEV_TO_GPU(dev);

  apir_buffer_type_context_t ctx = apir_device_get_buffer_type(gpu);

  static struct ggml_backend_buffer_type buft {
    /* .iface    = */ ggml_backend_remoting_buffer_type_interface,
    /* .device   = */ dev,
    /* .context  = */ ctx,
  };

  return &buft;
}

static ggml_backend_buffer_t ggml_backend_remoting_device_buffer_from_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
  UNUSED(dev);
  UNUSED(ptr);
  UNUSED(size);
  UNUSED(max_tensor_size);

  NOT_IMPLEMENTED;
  STOP_HERE;

  return nullptr;
}

const struct ggml_backend_device_i ggml_backend_remoting_device_i = {
  /* .get_name             = */ ggml_backend_remoting_device_get_name,
  /* .get_description      = */ ggml_backend_remoting_device_get_description,
  /* .get_memory           = */ ggml_backend_remoting_device_get_memory,
  /* .get_type             = */ ggml_backend_remoting_device_get_type,
  /* .get_props            = */ ggml_backend_remoting_device_get_props,
  /* .init_backend         = */ ggml_backend_remoting_device_init,
  /* .get_buffer_type      = */ ggml_backend_remoting_device_get_buffer_type,
  /* .get_host_buffer_type = */ NULL,
  /* .buffer_from_host_ptr = */ ggml_backend_remoting_device_buffer_from_ptr,
  /* .supports_op          = */ ggml_backend_remoting_device_supports_op,
  /* .supports_buft        = */ ggml_backend_remoting_device_supports_buft,
  /* .offload_op           = */ ggml_backend_remoting_device_offload_op,
  /* .event_new            = */ NULL,
  /* .event_free           = */ NULL,
  /* .event_synchronize    = */ NULL,
};
