#pragma once

#include <cstdarg>
#include <cstdio>
#include <cassert>

#include <ggml.h>

#define UNUSED GGML_UNUSED

inline void
INFO(const char *format, ...) {
  va_list argptr;
  va_start(argptr, format);
  vfprintf(stderr, format, argptr);
  fprintf(stderr, "\n");
  va_end(argptr);
}

inline void
WARNING(const char *format, ...) {
  fprintf(stderr, "WARNING: ");

  va_list argptr;
  va_start(argptr, format);
  vfprintf(stderr, format, argptr);
  fprintf(stderr, "\n");
  va_end(argptr);
}

inline void
ERROR(const char *format, ...) {
  fprintf(stderr, "ERROR: ");

  va_list argptr;
  va_start(argptr, format);
  vfprintf(stderr, format, argptr);
  fprintf(stderr, "\n");
  va_end(argptr);
}

inline void
FATAL(const char *format, ...) {
  fprintf(stderr, "FATAL: ");

  va_list argptr;
  va_start(argptr, format);
  vfprintf(stderr, format, argptr);
  fprintf(stderr, "\n");
  va_end(argptr);
  if (format)
    assert(false);
}
