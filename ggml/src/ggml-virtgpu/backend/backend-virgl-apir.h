#pragma once

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "shared/api_remoting.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

extern ggml_backend_reg_t reg;
extern ggml_backend_dev_t dev;
extern ggml_backend_t     bck;

struct virgl_apir_callbacks {
    const char * (*get_config)(uint32_t virgl_ctx_id, const char * key);
    void * (*get_shmem_ptr)(uint32_t virgl_ctx_id, uint32_t res_id);
};

extern "C" {
ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, struct virgl_apir_callbacks *virgl_cbs);
void                      apir_backend_deinit(uint32_t virgl_ctx_id);
uint32_t                  apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                                  virgl_apir_callbacks * virgl_cbs,
                                                  uint32_t               cmd_type,
                                                  char *                 dec_cur,
                                                  const char *           dec_end,
                                                  char *                 enc_cur,
                                                  const char *           enc_end,
                                                  char **                enc_cur_after);
}
