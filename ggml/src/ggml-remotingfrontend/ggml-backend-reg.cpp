#include <mutex>
#include <iostream>

#include "ggml-remoting.h"

static struct virtgpu *apir_gpu_instance = NULL;

static int apir_initialize() {
  static bool apir_initialized = false;

  if (apir_initialized) {
    if (!apir_gpu_instance) {
      return 0;
    }
    return 1;
  }
  apir_initialized = true;

  apir_gpu_instance = create_virtgpu();
  if (!apir_gpu_instance) {
    FATAL("failed to initialize the virtgpu :/");
    return 0;
  }

  apir_initialized = true;

  return 1;
}

static int ggml_backend_remoting_get_device_count() {
  if (!apir_initialize()) {
    WARNING("apir_initialize failed :/");
    return 0;
  }
  IMPLEMENTED;

  return apir_get_device_count(apir_gpu_instance);
}

static size_t ggml_backend_remoting_reg_get_device_count(ggml_backend_reg_t reg) {
  UNUSED(reg);

  IMPLEMENTED;

  return ggml_backend_remoting_get_device_count();
}

static ggml_backend_dev_t ggml_backend_remoting_reg_get_device(ggml_backend_reg_t reg, size_t device) {
  static std::vector<ggml_backend_dev_t> devices;

  IMPLEMENTED;

  static bool initialized = false;

  {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (!initialized) {

      for (size_t i = 0; i < ggml_backend_remoting_reg_get_device_count(reg); i++) {
        ggml_backend_remoting_device_context * ctx = new ggml_backend_remoting_device_context;
        char desc[256] = "API Remoting device";

        ctx->device = i;
        ctx->name = GGML_REMOTING_FRONTEND_NAME + std::to_string(i);
        ctx->description = desc;
        devices.push_back(new ggml_backend_device {
            /* .iface   = */ ggml_backend_remoting_device_i,
            /* .reg     = */ reg,
            /* .context = */ ctx,
          });
      }
      initialized = true;
    }
  }

  GGML_ASSERT(device < devices.size());
  return devices[device];
}

static const char * ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
  UNUSED(reg);
  printf("reached %s\n", __func__);
  //thks_bye();
  return GGML_REMOTING_FRONTEND_NAME;
}

static const struct ggml_backend_reg_i ggml_backend_remoting_reg_i = {
  /* .get_name         = */ ggml_backend_remoting_reg_get_name,
  /* .get_device_count = */ ggml_backend_remoting_reg_get_device_count,
  /* .get_device       = */ ggml_backend_remoting_reg_get_device,
  /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_remoting_frontend_reg() {
  static ggml_backend_reg reg = {
    /* .api_version = */ GGML_BACKEND_API_VERSION,
    /* .iface       = */ ggml_backend_remoting_reg_i,
    /* .context     = */ nullptr,
  };

  RMT_LOG_DEBUG("ggml_backend_remoting_frontend_reg() hello :wave:");
  return &reg;
}
