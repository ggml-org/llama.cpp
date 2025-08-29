#include <mutex>
#include <iostream>

#include "ggml-remoting.h"
#include "ggml-metal-remoting.h"

static struct virtgpu *apir_initialize() {
  static struct virtgpu *apir_gpu_instance = NULL;
  static bool apir_initialized = false;

  if (apir_initialized) {
    return apir_gpu_instance;
  }
  apir_initialized = true;

  apir_gpu_instance = create_virtgpu();
  if (!apir_gpu_instance) {
    FATAL("failed to initialize the virtgpu :/");
    return NULL;
  }

  apir_initialized = true;

  return apir_gpu_instance;
}

static int ggml_backend_remoting_get_device_count() {
  IMPLEMENTED;

  struct virtgpu *gpu = apir_initialize();
  if (!gpu) {
    WARNING("apir_initialize failed :/");
    return 0;
  }

  return apir_device_get_count(gpu);
}

static size_t ggml_backend_remoting_reg_get_device_count(ggml_backend_reg_t reg) {
  UNUSED(reg);

  IMPLEMENTED;

  return ggml_backend_remoting_get_device_count();
}

static std::vector<ggml_backend_dev_t> devices;

ggml_backend_dev_t ggml_backend_remoting_get_device(size_t device) {
  GGML_ASSERT(device < devices.size());
  return devices[device];
}

static void ggml_backend_remoting_reg_init_devices(ggml_backend_reg_t reg) {
  IMPLEMENTED;

  if (devices.size() > 0) {
    INFO("%s: already initialized", __func__);
  }

  struct virtgpu *gpu = apir_initialize();
  if (!gpu) {
    FATAL("apir_initialize failed :/");
    return;
  }

  static bool initialized = false;

  {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (!initialized) {

      for (int i = 0; i < ggml_backend_remoting_get_device_count(); i++) {
        ggml_backend_remoting_device_context *ctx = new ggml_backend_remoting_device_context;
        char desc[256] = "API Remoting device";

        ctx->device = i;
        ctx->name = GGML_REMOTING_FRONTEND_NAME + std::to_string(i);
        ctx->description = desc;
	ctx->gpu = gpu;

	ggml_backend_dev_t dev = new ggml_backend_device {
	  /* .iface   = */ ggml_backend_remoting_device_interface,
	  /* .reg     = */ reg,
	  /* .context = */ ctx,
	};

	ctx->metal_dev_ctx = get_metal_dev_context(dev);

        devices.push_back(dev);
      }
      initialized = true;
    }
  }
}

static ggml_backend_dev_t ggml_backend_remoting_reg_get_device(ggml_backend_reg_t reg, size_t device) {
  UNUSED(reg);

  IMPLEMENTED;

  return ggml_backend_remoting_get_device(device);
}

static const char *ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
  UNUSED(reg);

  return GGML_REMOTING_FRONTEND_NAME;
}

static const struct ggml_backend_reg_i ggml_backend_remoting_reg_i = {
  /* .get_name         = */ ggml_backend_remoting_reg_get_name,
  /* .get_device_count = */ ggml_backend_remoting_reg_get_device_count,
  /* .get_device       = */ ggml_backend_remoting_reg_get_device,
  /* .get_proc_address = */ NULL,
};


static void showTime() {
  show_timer(&graph_compute_timer);
  show_timer(&get_tensor_timer);
  show_timer(&set_tensor_timer);
  show_timer(&wait_host_reply_timer);

  if (get_tensor_from_ptr_timer.count) {
    show_timer(&get_tensor_from_ptr_timer);
    show_timer(&set_tensor_from_ptr_timer);
  }
}

ggml_backend_reg_t ggml_backend_remoting_frontend_reg() {
  struct virtgpu *gpu = apir_initialize();
  if (!gpu) {
    FATAL("apir_initialize failed :/");
    return NULL;
  }

  static ggml_backend_reg reg = {
    /* .api_version = */ GGML_BACKEND_API_VERSION,
    /* .iface       = */ ggml_backend_remoting_reg_i,
    /* .context     = */ gpu,
  };

  static bool initialized = false;
  if (initialized) {
    return &reg;
  }
  initialized = true;

  ggml_backend_remoting_reg_init_devices(&reg);

  int cr = atexit(showTime);
  GGML_ASSERT(cr == 0);

  MESSAGE("%s: initialzed", __func__);

  return &reg;
}
