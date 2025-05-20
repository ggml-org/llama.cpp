#include <cstdio>
#include <cstdarg>
#include <cstdlib>

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"

extern ggml_backend_reg_t reg;
extern ggml_backend_dev_t dev;
extern ggml_backend_t bck;

#define NOT_IMPLEMENTED							\
  do {									\
    static bool first = true;						\
    if (first) {							\
      printf("\nWARN: ###\nWARN: ### reached unimplemented function %s\nWARN: ###\n\n", __func__); \
      first = false;							\
    }									\
  } while(0)

extern "C" {
  uint32_t apir_backend_initialize();
  void apir_backend_deinit(void);
  uint32_t apir_backend_dispatcher(uint32_t cmd_type, struct virgl_apir_context *ctx,
				   char *dec_cur, const char *dec_end,
				   char *enc_cur, const char *enc_end,
				   char **enc_cur_after);
}
