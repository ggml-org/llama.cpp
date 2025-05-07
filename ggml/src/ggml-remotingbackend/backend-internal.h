#include <cstdio>
#include <cstdarg>
#include <cstdlib>

extern "C" {
  uint32_t apir_backend_initialize();
  void apir_backend_deinit(void);
  uint32_t apir_backend_dispatcher(uint32_t cmd_type,
				   char *dec_cur, const char *dec_end,
				   char *enc_cur, const char *enc_end,
				   char **enc_cur_after);
}
