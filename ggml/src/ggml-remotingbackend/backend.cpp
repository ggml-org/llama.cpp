#include <iostream>
#include <dlfcn.h>

#include <ggml-backend.h>

#include "backend-utils.h"
#include "backend-internal.h"
#include "backend-dispatched.h"

#include "shared/api_remoting.h"
#include "shared/apir_backend.h"
#include "shared/venus_cs.h"

#define GGML_BACKEND_LIBRARY_PATH_ENV "APIR_LLAMA_CPP_GGML_LIBRARY_PATH"
#define GGML_BACKEND_LIBRARY_REG_ENV "APIR_LLAMA_CPP_GGML_LIBRARY_REG"
#define GGML_BACKEND_LIBRARY_INIT_ENV "APIR_LLAMA_CPP_GGML_LIBRARY_INIT"

#define GGML_BACKEND_LIBRARY_METAL_DEVICE_CONTEXT "ggml_backend_metal_get_device_context"

static void *backend_library_handle = NULL;

extern "C" {
  void apir_backend_deinit(void) {
    auto buffers = get_track_backend_buffers();
    for (const auto& buffer: buffers) {
      untrack_backend_buffer(buffer);
      buffer->iface.free_buffer(buffer);
    }

    if (dev) {
      size_t free, total;
      dev->iface.get_memory(dev, &free, &total);
      INFO("%s: free memory: %ld MB", __func__, (size_t) free/1024/1024);
    }

    show_timer(&graph_compute_timer);
    show_timer(&set_tensor_timer);
    show_timer(&get_tensor_timer);
    /* *** */

    if (backend_library_handle) {
      INFO("%s: The GGML backend library was loaded. Unloading it.", __func__);
      dlclose(backend_library_handle);
    }

    INFO("%s: bye-bye", __func__);
  }

  ApirLoadLibraryReturnCode apir_backend_initialize() {
    const char* dlsym_error;

    const char* library_name = getenv(GGML_BACKEND_LIBRARY_PATH_ENV);
    const char* library_reg = getenv(GGML_BACKEND_LIBRARY_REG_ENV);
    const char* library_init = getenv(GGML_BACKEND_LIBRARY_INIT_ENV);

    INFO("%s: loading %s (%s|%s)", __func__, library_name, library_reg, library_init);

    if (!library_name) {
      ERROR("cannot open the GGML library: env var '%s' not defined\n", GGML_BACKEND_LIBRARY_PATH_ENV);

      return APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    backend_library_handle = dlopen(library_name, RTLD_LAZY);

    if (!backend_library_handle) {
      ERROR("cannot open the GGML library: %s", dlerror());

      return APIR_LOAD_LIBRARY_CANNOT_OPEN;
    }

    if (!library_reg) {
      ERROR("cannot register the GGML library: env var '%s' not defined", GGML_BACKEND_LIBRARY_REG_ENV);

      return APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    void *ggml_backend_reg_fct = dlsym(backend_library_handle, library_reg);
    dlsym_error = dlerror();
    if (dlsym_error) {
      ERROR("cannot find the GGML backend registration symbol '%s' (from %s): %s",
	    library_reg, GGML_BACKEND_LIBRARY_REG_ENV, dlsym_error);

      return APIR_LOAD_LIBRARY_SYMBOL_MISSING;
    }

    if (!library_init) {
      ERROR("cannot initialize the GGML library: env var '%s' not defined", library_init);

      return APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    void *ggml_backend_init_fct = dlsym(backend_library_handle, library_init);
    dlsym_error = dlerror();
    if (dlsym_error) {
      ERROR("cannot find the GGML backend init symbol '%s' (from %s): %s",
	    library_init, GGML_BACKEND_LIBRARY_INIT_ENV, dlsym_error);

      return APIR_LOAD_LIBRARY_SYMBOL_MISSING;
    }

    ggml_backend_metal_get_device_context_fct = (void (*)(ggml_backend_dev_t, bool *, bool *, bool *)) dlsym(backend_library_handle, GGML_BACKEND_LIBRARY_METAL_DEVICE_CONTEXT);
    dlsym_error = dlerror();
    if (dlsym_error) {
      ERROR("cannot find the GGML device context symbol '%s': %s\n",
	    GGML_BACKEND_LIBRARY_METAL_DEVICE_CONTEXT, dlsym_error);

      return APIR_LOAD_LIBRARY_SYMBOL_MISSING;
    }

    uint32_t ret = backend_dispatch_initialize(ggml_backend_reg_fct, ggml_backend_init_fct);

    return (ApirLoadLibraryReturnCode) (APIR_LOAD_LIBRARY_INIT_BASE_INDEX + ret);
  }

  uint32_t apir_backend_dispatcher(uint32_t cmd_type, struct virgl_apir_context *ctx,
				   char *dec_cur, const char *dec_end,
				   char *enc_cur, const char *enc_end,
				   char **enc_cur_after) {
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


    if (cmd_type >= APIR_BACKEND_DISPATCH_TABLE_COUNT) {
      ERROR("Received an invalid dispatch index (%d >= %d)\n",
	    cmd_type, APIR_BACKEND_DISPATCH_TABLE_COUNT);
      return APIR_BACKEND_FORWARD_INDEX_INVALID;
    }

#if 0
    static long long count = 0;
    INFO("[%lld] Calling %s", count, backend_dispatch_command_name((ApirBackendCommandType) cmd_type));
    count += 1;
#endif
    backend_dispatch_t forward_fct = apir_backend_dispatch_table[cmd_type];
    uint32_t ret = forward_fct(enc, dec, ctx);

    *enc_cur_after = enc->cur;

    return ret;
  }
}
