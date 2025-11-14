#pragma once

#include <cstdarg>
#include <cstdio>
#include <cassert>

#include <ggml.h>

#define UNUSED GGML_UNUSED
#define APIR_LLAMA_CPP_LOG_TO_FILE_ENV "APIR_LLAMA_CPP_LOG_TO_FILE"

static FILE *
get_log_dest(void)
{
   static FILE *dest = NULL;
   if (dest) {
      return dest;
   }
   const char *apir_log_to_file = getenv(APIR_LLAMA_CPP_LOG_TO_FILE_ENV);
   if (!apir_log_to_file) {
      dest = stderr;
      return dest;
   }

   dest = fopen(apir_log_to_file, "w");

   return dest;
}

#define APIR_VA_PRINT(prefix, format)               \
   do {                                             \
      FILE *dest = get_log_dest();                  \
      fprintf(dest, prefix);                        \
      va_list argptr;                               \
      va_start(argptr, format);                     \
      vfprintf(dest, format, argptr);               \
      fprintf(dest, "\n");                          \
      va_end(argptr);                               \
      fflush(dest);                                 \
   } while (0)

inline void
INFO(const char *format, ...) {
  APIR_VA_PRINT("INFO: ", format);
}

inline void
WARNING(const char *format, ...) {
  APIR_VA_PRINT("WARNING: ", format);
}

inline void
ERROR(const char *format, ...) {
  APIR_VA_PRINT("ERROR: ", format);
}

[[noreturn]] inline void
FATAL(const char *format, ...) {
  APIR_VA_PRINT("FORMAT: ", format);
  abort();
}
