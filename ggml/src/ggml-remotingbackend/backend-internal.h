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

extern "C" {
ApirLoadLibraryReturnCode apir_backend_initialize();
void                      apir_backend_deinit(void);
uint32_t                  apir_backend_dispatcher(uint32_t                    cmd_type,
                                                  struct virgl_apir_context * ctx,
                                                  char *                      dec_cur,
                                                  const char *                dec_end,
                                                  char *                      enc_cur,
                                                  const char *                enc_end,
                                                  char **                     enc_cur_after);
}
