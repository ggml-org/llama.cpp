#include <cstdio>
#include <cstdarg>

static inline void LOG(const char* fmt, ...) {
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
  void ggml_backend_remoting_backend_say_hello();
}
