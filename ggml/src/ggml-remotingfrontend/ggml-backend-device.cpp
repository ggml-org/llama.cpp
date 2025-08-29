#include "ggml-remoting.h"

static const char *
ggml_backend_remoting_device_get_name(ggml_backend_dev_t dev) {
  IMPLEMENTED_ONCE;

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
  IMPLEMENTED_ONCE;
  struct virtgpu *gpu = DEV_TO_GPU(dev);

  static enum ggml_backend_dev_type type;
  static bool has_type = false;
  if (!has_type) {
    has_type = true;
    type = (enum ggml_backend_dev_type) apir_device_get_type(gpu);
  }

  return type;
}

static void
ggml_backend_remoting_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
  IMPLEMENTED;

  struct virtgpu *gpu = DEV_TO_GPU(dev);

  return apir_device_get_memory(gpu, free, total);
}

static bool
ggml_backend_remoting_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
#if USE_ALWAYS_TRUE_SUPPORTS_OP == 1
  /* ggml-rpc cheats it like this */
  /* with the current implementation of serialize_tensor, the src/view aren't properly passed */
  UNUSED(dev);
  UNUSED(op);

  return true;
#elif USE_METAL_GUEST_SUPPORTS_OP == 1
  UNUSED(dev);

  struct ggml_backend_remoting_device_context *device_ctx = GET_DEVICE_CONTEXT();

  return ggml_metal_device_supports_op(device_ctx->metal_dev_ctx, op);
#else
  struct virtgpu *gpu = DEV_TO_GPU(dev);

  return apir_device_supports_op(gpu, op);
#endif
}

static bool
ggml_backend_remoting_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
  //IMPLEMENTED_ONCE;

#if 1
  bool supported = buft->device == dev;
  if (!supported) {
    //WARNING("%s: unsupported buffer type (%s). Double check.", __func__, buft->iface.get_name(buft));
  }

  return supported;
#else
  UNUSED(dev);
  UNUSED(buft);

  return true;
#endif
}

static bool
ggml_backend_remoting_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
  //IMPLEMENTED_ONCE;

  UNUSED(dev);
  UNUSED(op);

  // related to supports_buft, need to confirm

  return false; // same as ggml-metal
}

static void
ggml_backend_remoting_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
  IMPLEMENTED;

  props->name        = ggml_backend_remoting_device_get_name(dev);
  props->description = ggml_backend_remoting_device_get_description(dev);
  props->type        = ggml_backend_remoting_device_get_type(dev);
  ggml_backend_remoting_device_get_memory(dev, &props->memory_free, &props->memory_total);

#if 0
  struct virtgpu *gpu = DEV_TO_GPU(dev);
  apir_device_get_props(gpu,
			&props->caps.async,
			&props->caps.host_buffer,
			&props->caps.buffer_from_host_ptr,
			&props->caps.events
    );
#else
  // ignore the actual backend answers and set it as we provide it in
  // the API Remoting frontend
  props->caps.async = false;
  props->caps.host_buffer = false;
  props->caps.buffer_from_host_ptr = false;
  props->caps.events = false;
#endif

  INFO("%s: async=%d, host_buffer=%d!, buffer_from_host_ptr=%d!, events=%d",
    __func__, props->caps.async, props->caps.host_buffer,
       props->caps.buffer_from_host_ptr, props->caps.events);
}

ggml_backend_buffer_type_t
ggml_backend_remoting_device_get_buffer_type(ggml_backend_dev_t dev) {
  IMPLEMENTED_ONCE;

  struct virtgpu *gpu = DEV_TO_GPU(dev);

  apir_buffer_type_host_handle_t ctx = apir_device_get_buffer_type(gpu);

  static struct ggml_backend_buffer_type buft {
    /* .iface    = */ ggml_backend_remoting_buffer_type_interface,
    /* .device   = */ dev,
    /* .context  = */ (void *) ctx,
  };

  return &buft;
}

static ggml_backend_buffer_type_t
ggml_backend_remoting_device_get_buffer_from_ptr_type(ggml_backend_dev_t dev) {
  IMPLEMENTED_ONCE;

  struct virtgpu *gpu = DEV_TO_GPU(dev);

  apir_buffer_type_host_handle_t ctx = apir_device_get_buffer_type(gpu);

  static struct ggml_backend_buffer_type buft {
    /* .iface    = */ ggml_backend_remoting_buffer_from_ptr_type_interface,
    /* .device   = */ dev,
    /* .context  = */ (void *) ctx,
  };

  return &buft;
}

static ggml_backend_buffer_t
ggml_backend_remoting_device_buffer_from_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {

  struct virtgpu *gpu = DEV_TO_GPU(dev);

  struct ggml_backend_remoting_buffer_context *context = (struct ggml_backend_remoting_buffer_context *) malloc(sizeof(*context));
  if (!context) {
    FATAL("Couldn't allocate the buffer context ...");
  }

  context->gpu = gpu;
  context->apir_context = apir_device_buffer_from_ptr(gpu, size, max_tensor_size);
  context->base = ptr;
  context->is_from_ptr = true;

  ggml_backend_buffer_t buffer = ggml_backend_buffer_init(ggml_backend_remoting_device_get_buffer_from_ptr_type(dev), ggml_backend_remoting_buffer_from_ptr_interface, (void *) context, size);

  INFO("#");
  INFO("# %s(%p, %llx) --> %p", __func__, ptr, size, buffer);
  INFO("#\n");

  return buffer;
}

static ggml_backend_buffer_type_t
ggml_backend_remoting_device_get_host_buffer_type(ggml_backend_dev_t dev) {
  IMPLEMENTED_ONCE;

  static struct ggml_backend_buffer_type host_bufft = {
    /* .iface    = */ ggml_backend_remoting_host_buffer_type_interface,
    /* .device   = */ dev,
    /* .context  = */ nullptr,
  };

  return &host_bufft;
}

const struct ggml_backend_device_i ggml_backend_remoting_device_interface = {
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
