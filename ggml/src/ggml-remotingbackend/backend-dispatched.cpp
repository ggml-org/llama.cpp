#include <cstdint>
#include "backend-internal.h"
#include "backend-dispatched.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-remoting-backend.h"

#include "ggml-metal.h"

static ggml_backend_reg_t reg = NULL;
static ggml_backend_dev_t dev = NULL;
static ggml_backend_t bck = NULL;

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

uint32_t backend_device_get_name(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec) {
  UNUSED(dec);

  const char *string = dev->iface.get_name(dev);

  const size_t string_size = strlen(string) + 1;
  vn_encode_array_size(enc, string_size);
  vn_encode_char_array(enc, string, string_size);

  return 0;
}

uint32_t
backend_device_get_description(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec) {
  UNUSED(dec);

  const char *string = dev->iface.get_description(dev);

  const size_t string_size = strlen(string) + 1;
  vn_encode_array_size(enc, string_size);
  vn_encode_char_array(enc, string, string_size);

  return 0;
}

uint32_t
backend_device_get_type(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec) {
  UNUSED(dec);

  uint32_t type = dev->iface.get_type(dev);
  vn_encode_uint32_t(enc, &type);

  return 0;
}

uint32_t
backend_device_get_memory(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec) {
  UNUSED(dec);

  size_t free, total;
  dev->iface.get_memory(dev, &free, &total);

  vn_encode_size_t(enc, &free);
  vn_encode_size_t(enc, &total);

  return 0;
}

uint32_t
backend_device_supports_op(struct vn_cs_encoder *enc, struct vn_cs_decoder *dec) {
  const ggml_tensor *op = vn_decode_ggml_tensor_inplace(dec);

  bool supports_op = dev->iface.supports_op(dev, op);

  vn_encode_bool_t(enc, &supports_op);

  return 0;
}
