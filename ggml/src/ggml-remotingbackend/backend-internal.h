#include <cstdio>
#include <cstdarg>

static inline void INFO(const char* fmt, ...) {
  printf("INFO: ");
  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);

  printf("\n");
}

static inline void ERROR(const char* fmt, ...) {
  printf("ERROR: ");
  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);

  printf("\n");
}

static inline void FATAL(const char* fmt, ...) {
  printf("FATAL: ");
  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);

  printf("\n");

  if (!fmt)
    return; // avoid the noreturn attribute

  exit(1);
}

extern "C" {
  uint32_t apir_backend_initialize();
  void apir_backend_deinit(void);
  uint32_t apir_backend_dispatcher(uint32_t cmd_type,
				   char *dec_cur, const char *dec_end,
				   char *enc_cur, const char *enc_end,
				   char **enc_cur_after);
}
