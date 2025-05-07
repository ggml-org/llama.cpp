#include <iostream>
#include <dlfcn.h>

#include <ggml-backend.h>

#include "backend-utils.h"
#include "backend-internal.h"
#include "backend-dispatched.h"

#include "shared/apir_backend.h"
#include "shared/venus_cs.h"

#define GGML_BACKEND_LIBRARY_PATH "/Users/kevinpouget/remoting/llama_cpp/build.remoting-backend/bin/libggml-metal.dylib"
#define GGML_BACKEND_REG_FCT_NAME "ggml_backend_metal_reg"

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

    backend_library_handle = dlopen(GGML_BACKEND_LIBRARY_PATH, RTLD_LAZY);

    if (!backend_library_handle) {
      ERROR("Cannot open library: %s\n", dlerror());

      return APIR_BACKEND_INITIALIZE_CANNOT_OPEN_GGML_LIBRARY;
    }

    void *ggml_backend_reg_fct = dlsym(backend_library_handle, GGML_BACKEND_REG_FCT_NAME);
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
      ERROR("Cannot load symbol: %s\n", dlsym_error);

      return APIR_BACKEND_INITIALIZE_MISSING_GGML_SYMBOLS;
    }

    return backend_dispatch_initialize(ggml_backend_reg_fct);
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


    if (cmd_type > APIR_BACKEND_DISPATCH_TABLE_COUNT) {
      ERROR("Received an invalid dispatch index (%d > %d)\n",
	    cmd_type, APIR_BACKEND_DISPATCH_TABLE_COUNT);
      return APIR_BACKEND_FORWARD_INDEX_INVALID;
    }

    backend_dispatch_t forward_fct = apir_backend_dispatch_table[cmd_type];
    uint32_t ret = forward_fct(enc, dec);

    *enc_cur_after = enc->cur;

    return ret;
  }
}
