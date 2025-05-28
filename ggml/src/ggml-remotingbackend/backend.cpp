#include <iostream>
#include <dlfcn.h>

#include <ggml-backend.h>

#include "backend-utils.h"
#include "backend-internal.h"
#include "backend-dispatched.h"

#include "shared/apir_backend.h"
#include "shared/venus_cs.h"

#define USE_METAL 1

#if USE_METAL
#define GGML_BACKEND_LIBRARY_PATH "/Users/kevinpouget/remoting/llama_cpp/build.remoting-backend/bin/libggml-metal.dylib"
#define GGML_BACKEND_REG_FCT_NAME "ggml_backend_metal_reg"
#define GGML_BACKEND_INIT_FCT_NAME "ggml_backend_metal_init"
#else
#define GGML_BACKEND_LIBRARY_PATH "/Users/kevinpouget/remoting/llama_cpp/build.remoting-backend/bin/libggml-vulkan.dylib"
#define GGML_BACKEND_REG_FCT_NAME "ggml_backend_vk_reg"
#define GGML_BACKEND_INIT_FCT_NAME "ggml_backend_vk_init"
#endif

static void *backend_library_handle = NULL;

extern "C" {
  void apir_backend_deinit(void) {
    auto buffers = get_track_backend_buffers();
    for (const auto& buffer: buffers) {
      untrack_backend_buffer(buffer);
      buffer->iface.free_buffer(buffer);
    }

    size_t free, total;
    dev->iface.get_memory(dev, &free, &total);
    WARNING("%s: free memory: %ld MB\n", __func__, (size_t) free/1024/1024);

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

  uint32_t apir_backend_initialize() {
    const char* dlsym_error;

    INFO("%s: hello " GGML_BACKEND_REG_FCT_NAME " :wave: \\o/", __func__);

    backend_library_handle = dlopen(GGML_BACKEND_LIBRARY_PATH, RTLD_LAZY);

    if (!backend_library_handle) {
      ERROR("Cannot open library: %s\n", dlerror());

      return APIR_BACKEND_INITIALIZE_CANNOT_OPEN_GGML_LIBRARY;
    }

    void *ggml_backend_reg_fct = dlsym(backend_library_handle, GGML_BACKEND_REG_FCT_NAME);
    dlsym_error = dlerror();
    if (dlsym_error) {
      ERROR("Cannot load symbol: %s\n", dlsym_error);

      return APIR_BACKEND_INITIALIZE_MISSING_GGML_SYMBOLS;
    }

    void *ggml_backend_init_fct = dlsym(backend_library_handle, GGML_BACKEND_INIT_FCT_NAME);
    dlsym_error = dlerror();
    if (dlsym_error) {
      ERROR("Cannot load symbol: %s\n", dlsym_error);

      return APIR_BACKEND_INITIALIZE_MISSING_GGML_SYMBOLS;
    }

    INFO("#");
#if APIR_ALLOC_FROM_HOST_PTR
    INFO("# USING ALLOC_FROM_HOST_PTR");
#else
    INFO("# USING ALLOC_BUFFER");
#endif
    INFO("#");

    return backend_dispatch_initialize(ggml_backend_reg_fct, ggml_backend_init_fct);
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


    if (cmd_type > APIR_BACKEND_DISPATCH_TABLE_COUNT) {
      ERROR("Received an invalid dispatch index (%d > %d)\n",
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
