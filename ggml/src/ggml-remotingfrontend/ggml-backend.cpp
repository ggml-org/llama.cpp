#include "ggml-remoting.h"

static const char * ggml_backend_remoting_get_name(ggml_backend_t backend) {
  UNUSED(backend);

  //IMPLEMENTED_ONCE;

  return "API Remoting backend";
}

static void ggml_backend_remoting_free(ggml_backend_t backend) {
  IMPLEMENTED;

  delete backend;
}

struct timer_data graph_compute_timer = {0, 0, 0, "compute_timer"};

static ggml_status ggml_backend_remoting_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
  struct virtgpu *gpu = DEV_TO_GPU(backend->device);

  IMPLEMENTED_ONCE;

  start_timer(&graph_compute_timer);

  ggml_status status = apir_backend_graph_compute(gpu, cgraph);

  stop_timer(&graph_compute_timer);

  return status;
}

static void ggml_backend_remoting_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph) {
    struct virtgpu *gpu = DEV_TO_GPU(backend->device);
#if true
    UNUSED(gpu);
    UNUSED(cgraph);

    NOT_IMPLEMENTED;
#else
    start_timer(&graph_compute_timer);
    
    apir_backend_graph_optimize(gpu, cgraph);
    
    stop_timer(&graph_compute_timer);
#endif
}

static ggml_backend_i ggml_backend_remoting_interface = {
  /* .get_name                = */ ggml_backend_remoting_get_name,
  /* .free                    = */ ggml_backend_remoting_free,
  /* .set_tensor_async        = */ NULL,  // ggml_backend_remoting_set_tensor_async,
  /* .get_tensor_async        = */ NULL,  // ggml_backend_remoting_get_tensor_async,
  /* .cpy_tensor_async        = */ NULL,  // ggml_backend_remoting_cpy_tensor_async,
  /* .synchronize             = */ NULL,  // ggml_backend_remoting_synchronize,
  /* .graph_plan_create       = */ NULL,
  /* .graph_plan_free         = */ NULL,
  /* .graph_plan_update       = */ NULL,
  /* .graph_plan_compute      = */ NULL,
  /* .graph_compute           = */ ggml_backend_remoting_graph_compute,
  /* .event_record            = */ NULL,
  /* .event_wait              = */ NULL,
  /* .graph_optimize          = */ ggml_backend_remoting_graph_optimize,
};

static ggml_guid_t ggml_backend_remoting_guid() {
  static ggml_guid guid = { 0xb8, 0xf7, 0x4f, 0x86, 0x14, 0x03, 0x86, 0x02, 0x91, 0xc8, 0xdd, 0xe9, 0x02, 0x3f, 0xc0, 0x2b };

  return &guid;
}


ggml_backend_t ggml_backend_remoting_device_init(ggml_backend_dev_t dev, const char * params) {
  UNUSED(params);
  IMPLEMENTED;

  ggml_backend_remoting_device_context * ctx = (ggml_backend_remoting_device_context *)dev->context;

  ggml_backend_t remoting_backend = new ggml_backend {
    /* .guid      = */ ggml_backend_remoting_guid(),
    /* .interface = */ ggml_backend_remoting_interface,
    /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_remoting_frontend_reg(), ctx->device),
    /* .context   = */ ctx,
  };

  return remoting_backend;
}
