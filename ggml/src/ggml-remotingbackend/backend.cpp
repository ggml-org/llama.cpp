#include <iostream>
#include <dlfcn.h>

#include "ggml-remoting-backend.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"

#include "backend-internal.h"

#define UNUSED GGML_UNUSED

static size_t ggml_backend_remoting_reg_get_device_count(ggml_backend_reg_t reg) {
  UNUSED(reg);
  return 0;
}

static const char * ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
  UNUSED(reg);
  return GGML_REMOTING_BACKEND_NAME;
}

static ggml_backend_dev_t ggml_backend_remoting_reg_get_device(ggml_backend_reg_t reg, size_t device) {
  UNUSED(reg);
  UNUSED(device);

  return NULL;
}

static const struct ggml_backend_reg_i ggml_backend_remoting_reg_i = {
    /* .get_name         = */ ggml_backend_remoting_reg_get_name,
    /* .get_device_count = */ ggml_backend_remoting_reg_get_device_count,
    /* .get_device       = */ ggml_backend_remoting_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_remoting_backend_reg() {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_remoting_reg_i,
        /* .context     = */ nullptr,
    };

    LOG("%s, hello :wave:", __func__);

    return &reg;
}

typedef ggml_backend_reg_t (*backend_reg_fct_t)(void);

#define METAL_LIBRARY_PATH "/Users/kevinpouget/remoting/llama_cpp/build.remoting-backend/bin/libggml-metal.dylib"
#define ENTRYPOINT_FCT_NAME "ggml_backend_metal_reg"

extern "C" {
  void ggml_backend_remoting_backend_say_hello() {
    LOG("%s: hello :wave: \\o/", __func__);

    void * library_handle = dlopen(METAL_LIBRARY_PATH, RTLD_LAZY);

    if (!library_handle) {
      FATAL("Cannot open library: %s\n", dlerror());
      return;
    }

    backend_reg_fct_t entrypoint_fct = (backend_reg_fct_t) dlsym(library_handle, ENTRYPOINT_FCT_NAME);
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
      FATAL("Cannot load symbol: %s\n", dlsym_error);
      return;
    }

    ggml_backend_reg_t reg = entrypoint_fct();
    LOG("%s: --> %s", __func__, reg->iface.get_name(reg));

    dlclose(library_handle);
  }
}
