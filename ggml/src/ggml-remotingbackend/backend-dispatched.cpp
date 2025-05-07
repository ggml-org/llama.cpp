#include <cstdint>
#include "backend-internal.h"
#include "backend-dispatched.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-remoting-backend.h"

static ggml_backend_reg_t reg = NULL;

uint32_t backend_dispatch_initialize(void *ggml_backend_reg_fct_p) {
  if (reg != NULL) {
    FATAL("%s: already initialized :/", __func__);
  }
  ggml_backend_reg_t (* ggml_backend_reg_fct)(void) = (ggml_backend_reg_t (*)()) ggml_backend_reg_fct_p;

  reg = ggml_backend_reg_fct();

  return APIR_BACKEND_INITIALIZE_SUCCESSS;

}

static size_t ggml_backend_remoting_reg_get_device_count(ggml_backend_reg_t reg) {
  UNUSED(reg);
  return 0;
}

static const char *ggml_backend_remoting_reg_get_name(ggml_backend_reg_t reg) {
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

    INFO("%s, hello :wave:", __func__);

    return &reg;
}

uint32_t backend_reg_get_device_count(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec) {
  UNUSED(dec);

  int32_t dev_count = reg->iface.get_device_count(reg);
  vn_encode_int32_t(enc, &dev_count);

  return 0;
}
