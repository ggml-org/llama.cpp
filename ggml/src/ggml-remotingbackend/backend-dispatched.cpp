#include <cstdint>
#include "backend-internal.h"
#include "backend-dispatched.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-remoting-backend.h"

#include "ggml-metal.h"

ggml_backend_reg_t reg = NULL;
ggml_backend_dev_t dev = NULL;
ggml_backend_t bck = NULL;

uint32_t backend_dispatch_initialize(void *ggml_backend_reg_fct_p, void *ggml_backend_init_fct_p) {
  if (reg != NULL) {
    FATAL("%s: already initialized :/", __func__);
  }
  ggml_backend_reg_t (* ggml_backend_reg_fct)(void) = (ggml_backend_reg_t (*)()) ggml_backend_reg_fct_p;

  reg = ggml_backend_reg_fct();
  if (reg == NULL) {
    FATAL("%s: backend registration failed :/", __func__);
  }

  if (reg->iface.get_device_count(reg)) {
    dev = reg->iface.get_device(reg, 0);
  }

  ggml_backend_t (* ggml_backend_fct)(void) = (ggml_backend_t (*)()) ggml_backend_init_fct_p;

  bck = ggml_backend_fct();
  if (!bck) {
    ERROR("%s: backend initialization failed :/", __func__);
    return APIR_BACKEND_INITIALIZE_BACKEND_FAILED;
  }

  return APIR_BACKEND_INITIALIZE_SUCCESSS;
}

static size_t ggml_backend_remoting_reg_get_device_count(ggml_backend_reg_t reg) {
  UNUSED(reg);

  NOT_IMPLEMENTED;

  return 0;
}

static const char *ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
  UNUSED(reg);

  NOT_IMPLEMENTED;

  return GGML_REMOTING_BACKEND_NAME;
}

static ggml_backend_dev_t ggml_backend_remoting_reg_get_device(ggml_backend_reg_t reg, size_t device) {
  UNUSED(reg);
  UNUSED(device);

  NOT_IMPLEMENTED;

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

    INFO("%s, hello :wave:", __func__);

    return &reg;
}
