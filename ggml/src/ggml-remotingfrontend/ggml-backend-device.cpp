#include "ggml-remoting.h"

static const char *ggml_backend_remoting_device_get_name(ggml_backend_dev_t dev) {
  IMPLEMENTED;

  struct virtgpu *gpu = ((struct ggml_backend_remoting_device_context *) dev->context)->gpu;

  return apir_get_device_name(gpu);
}

static const char *ggml_backend_remoting_device_get_description(ggml_backend_dev_t dev) {
  IMPLEMENTED;

  struct virtgpu *gpu = ((struct ggml_backend_remoting_device_context *) dev->context)->gpu;

  return apir_get_device_description(gpu);
}

static enum ggml_backend_dev_type ggml_backend_remoting_device_get_type(ggml_backend_dev_t dev) {
  IMPLEMENTED;

  struct virtgpu *gpu = ((struct ggml_backend_remoting_device_context *) dev->context)->gpu;

  return (enum ggml_backend_dev_type) apir_get_device_type(gpu);
}

static void ggml_backend_remoting_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
  IMPLEMENTED;

  struct virtgpu *gpu = ((struct ggml_backend_remoting_device_context *) dev->context)->gpu;

  return apir_get_device_memory(gpu, free, total);
}

static bool ggml_backend_remoting_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
  UNUSED(dev);
  UNUSED(op);

  //NOT_IMPLEMENTED; // to chatty

  return true;
}

static bool ggml_backend_remoting_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
  UNUSED(dev);
  UNUSED(buft);

  NOT_IMPLEMENTED;

  return true;
}

static bool ggml_backend_remoting_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
  const int min_batch_size = 32;

  NOT_IMPLEMENTED;

  return (op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS) ||
    (op->ne[2] >= min_batch_size && op->op == GGML_OP_MUL_MAT_ID);

  UNUSED(dev);
}

static ggml_backend_buffer_type_t ggml_backend_remoting_device_get_host_buffer_type(ggml_backend_dev_t dev) {
  UNUSED(dev);

  // NOT_IMPLEMENTED; // too chatty

  return ggml_backend_remoting_host_buffer_type();
}


static void ggml_backend_remoting_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {

  IMPLEMENTED;
  props->name        = ggml_backend_remoting_device_get_name(dev);
  props->description = ggml_backend_remoting_device_get_description(dev);
  props->type        = ggml_backend_remoting_device_get_type(dev);
  ggml_backend_remoting_device_get_memory(dev, &props->memory_free, &props->memory_total);
  props->caps = {
    /* .async                 = */ false,
    /* .host_buffer           = */ true,
    /* .buffer_from_host_ptr  = */ false,
    /* .events                = */ false,
  };
}

const struct ggml_backend_device_i ggml_backend_remoting_device_i = {
  /* .get_name             = */ ggml_backend_remoting_device_get_name,
  /* .get_description      = */ ggml_backend_remoting_device_get_description,
  /* .get_memory           = */ ggml_backend_remoting_device_get_memory,
  /* .get_type             = */ ggml_backend_remoting_device_get_type,
  /* .get_props            = */ ggml_backend_remoting_device_get_props,
  /* .init_backend         = */ ggml_backend_remoting_device_init,
  /* .get_buffer_type      = */ ggml_backend_remoting_device_get_buffer_type,
  /* .get_host_buffer_type = */ ggml_backend_remoting_device_get_host_buffer_type,
  /* .buffer_from_host_ptr = */ NULL,
  /* .supports_op          = */ ggml_backend_remoting_device_supports_op,
  /* .supports_buft        = */ ggml_backend_remoting_device_supports_buft,
  /* .offload_op           = */ ggml_backend_remoting_device_offload_op,
  /* .event_new            = */ NULL,
  /* .event_free           = */ NULL,
  /* .event_synchronize    = */ NULL,
};
