#include "backend-dispatched.h"
#include "backend-virgl-apir.h"

#include "shared/api_remoting.h"
#include "shared/apir_backend.h"
#include "shared/apir_cs.h"

#include <dlfcn.h>
#include <ggml-backend.h>

#include <iostream>

#define APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV "APIR_LLAMA_CPP_GGML_LIBRARY_PATH"
#define APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV  "APIR_LLAMA_CPP_GGML_LIBRARY_REG"
#define APIR_LLAMA_CPP_LOG_TO_FILE_ENV       "APIR_LLAMA_CPP_LOG_TO_FILE"

#define GGML_DEFAULT_BACKEND_REG "ggml_backend_init"

static void * backend_library_handle = NULL;
static FILE * apir_logfile = NULL;

static void log_to_file_callback(enum ggml_log_level level, const char * text, void * user_data) {
    FILE * logfile = (FILE *)user_data;
    fprintf(logfile, "[%d] %s", level, text);
    fflush(logfile);
}

extern "C" {
void apir_backend_deinit(uint32_t virgl_ctx_id) {
    GGML_UNUSED(virgl_ctx_id);

    auto buffers = apir_get_track_backend_buffers();
    for (const auto & buffer : buffers) {
        apir_untrack_backend_buffer(buffer);
        buffer->iface.free_buffer(buffer);
    }

    if (dev) {
        size_t free, total;
        dev->iface.get_memory(dev, &free, &total);
        GGML_LOG_INFO("%s: free memory: %ld MB\n", __func__, (size_t) free / 1024 / 1024);
    }

    if (backend_library_handle) {
        GGML_LOG_INFO("%s: The GGML backend library was loaded. Unloading it.\n", __func__);
        dlclose(backend_library_handle);
        backend_library_handle = NULL;
    }

    if (apir_logfile) {
        fclose(apir_logfile);
        apir_logfile = NULL;
    }
}

#define APIR_GGML_LIBRARY_PATH_KEY "ggml.library.path"
#define APIR_GGML_LIBRARY_REG_KEY "ggml.library.reg"

ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, struct virgl_apir_callbacks *virgl_cbs) {
    const char * dlsym_error;

    const char * apir_log_to_file = getenv(APIR_LLAMA_CPP_LOG_TO_FILE_ENV);
    if (apir_log_to_file) {
        apir_logfile = fopen(apir_log_to_file, "w");
        if (apir_logfile) {
            ggml_log_set(log_to_file_callback, apir_logfile);
        } else {
            GGML_LOG_INFO("Could not open the log file at '%s'\n", apir_log_to_file);
        }
    }

    const char * library_name = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_PATH_KEY);
    const char * virgl_library_reg = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_REG_KEY);
    const char * library_reg = virgl_library_reg ? virgl_library_reg : GGML_DEFAULT_BACKEND_REG;

    if (!library_name) {
        GGML_LOG_ERROR("cannot open the GGML library: env var '%s' not defined\n", APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV);

        return APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    backend_library_handle = dlopen(library_name, RTLD_LAZY);

    if (!backend_library_handle) {
        GGML_LOG_ERROR("cannot open the GGML library: %s\n", dlerror());

        return APIR_LOAD_LIBRARY_CANNOT_OPEN;
    }

    if (!library_reg) {
        GGML_LOG_ERROR("cannot register the GGML library: env var '%s' not defined\n", APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV);

        return APIR_LOAD_LIBRARY_ENV_VAR_MISSING;
    }

    void * ggml_backend_reg_fct = dlsym(backend_library_handle, library_reg);
    dlsym_error                 = dlerror();
    if (dlsym_error) {
        GGML_LOG_ERROR("cannot find the GGML backend registration symbol '%s' (from %s): %s\n", library_reg,
              APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV, dlsym_error);

        return APIR_LOAD_LIBRARY_SYMBOL_MISSING;
    }

    uint32_t ret = backend_dispatch_initialize(ggml_backend_reg_fct);

    return (ApirLoadLibraryReturnCode) (APIR_LOAD_LIBRARY_INIT_BASE_INDEX + ret);
}

uint32_t apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                 virgl_apir_callbacks * virgl_cbs,
                                 uint32_t               cmd_type,
                                 char *                 dec_cur,
                                 const char *           dec_end,
                                 char *                 enc_cur,
                                 const char *           enc_end,
                                 char **                enc_cur_after) {
    apir_encoder enc = {
        .cur   = enc_cur,
        .start = enc_cur,
        .end   = enc_end,
        .fatal = false,
    };

    apir_decoder dec = {
        .cur   = dec_cur,
        .end   = dec_end,
        .fatal = false,
    };

    virgl_apir_context ctx = {
        .ctx_id = virgl_ctx_id,
        .iface = virgl_cbs,
    };

    if (cmd_type >= APIR_BACKEND_DISPATCH_TABLE_COUNT) {
        GGML_LOG_ERROR("Received an invalid dispatch index (%d >= %d)\n", cmd_type, APIR_BACKEND_DISPATCH_TABLE_COUNT);
        return APIR_BACKEND_FORWARD_INDEX_INVALID;
    }

    backend_dispatch_t forward_fct = apir_backend_dispatch_table[cmd_type];
    uint32_t           ret         = forward_fct(&enc, &dec, &ctx);

    *enc_cur_after = enc.cur;

    return ret;
}
}
