#include <iostream>
#include <dlfcn.h>

#include "ggml-remoting-backend.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"

#include "backend-internal.h"
#include "shared/apir_backend.h"
#include "shared/venus_cs.h"

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

    INFO("%s, hello :wave:", __func__);

    return &reg;
}

typedef ggml_backend_reg_t (*backend_reg_fct_t)(void);

#define GGML_BACKEND_METAL_LIBRARY_PATH "/Users/kevinpouget/remoting/llama_cpp/build.remoting-backend/bin/libggml-metal.dylib"
#define GGML_BACKEND_METAL_REG_FCT_NAME "ggml_backend_metal_reg"

static void *backend_library_handle = NULL;

extern "C" {
  void apir_backend_deinit(void) {
    if (backend_library_handle) {
      INFO("%s: The GGML backend library was loaded. Unloading it.", __func__);
      dlclose(backend_library_handle);
    }

    INFO("%s: bye-bye", __func__);
  }

  uint32_t apir_backend_initialize() {
    INFO("%s: hello :wave: \\o/", __func__);

    backend_library_handle = dlopen(GGML_BACKEND_METAL_LIBRARY_PATH, RTLD_LAZY);

    if (!backend_library_handle) {
      ERROR("Cannot open library: %s\n", dlerror());

      return APIR_BACKEND_INITIALIZE_CANNOT_OPEN_GGML_LIBRARY;
    }

    backend_reg_fct_t entrypoint_fct = (backend_reg_fct_t) dlsym(backend_library_handle, GGML_BACKEND_METAL_REG_FCT_NAME);
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
      ERROR("Cannot load symbol: %s\n", dlsym_error);

      return APIR_BACKEND_INITIALIZE_MISSING_GGML_SYMBOLS;
    }

    ggml_backend_reg_t reg = entrypoint_fct();
    INFO("%s: --> %s", __func__, reg->iface.get_name(reg));

    return APIR_BACKEND_INITIALIZE_SUCCESSS;
  }

  uint32_t apir_backend_dispatcher(uint32_t cmd_type,
				   char *dec_cur, const char *dec_end,
				   char *enc_cur, const char *enc_end,
				   char **enc_cur_after) {
    INFO("%s: --> %d | %p | %p ", __func__, cmd_type, dec_cur, enc_cur);

    struct vn_cs_encoder _enc = {
      .cur = enc_cur,
      .end = enc_end,
    };
    struct vn_cs_encoder *enc = &_enc;

    struct vn_cs_decoder _dec = {
      .cur = dec_cur,
      .end = dec_end,
    };
    struct vn_cs_decoder *dec = &_dec;

    int32_t arg1, arg2, arg3;
    vn_decode_int32_t(dec, &arg1);
    vn_decode_int32_t(dec, &arg2);
    vn_decode_int32_t(dec, &arg3);

    INFO("%s: ARGS %d %d %d\n", __func__, arg1, arg2, arg3);

    int32_t resp1 = 1;
    int32_t resp2 = 2;
    int32_t resp3 = 3;
    int32_t resp4 = 4;
    vn_encode_int32_t(enc, &resp1);
    vn_encode_int32_t(enc, &resp2);
    vn_encode_int32_t(enc, &resp3);
    vn_encode_int32_t(enc, &resp4);
    *enc_cur_after = enc->cur;

    return 0;
  }
}
