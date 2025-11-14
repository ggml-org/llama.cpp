#pragma once

#include <cstdint>
#include <cassert>
#include <cstddef>
#include <ctime>
#include <cerrno>
#include <atomic>
#include <cstdlib>
#include <cstdarg>
#include <cstdio>

#define unlikely(x) __builtin_expect(!!(x), 0)
#define likely(x) __builtin_expect(!!(x), 1)

#ifndef UNUSED
#define UNUSED(x) (void)(x)
#endif

/** Checks is a value is a power of two. Does not handle zero. */
#define IS_POT(v) (((v) & ((v) - 1)) == 0)

/** Checks is a value is a power of two. Zero handled. */
#define IS_POT_NONZERO(v) ((v) != 0 && IS_POT(v))

/** Align a value to a power of two */
#define ALIGN_POT(x, pot_align) (((x) + (pot_align) - 1) & ~((pot_align) - 1))

#define p_atomic_read(_v) __atomic_load_n((_v), __ATOMIC_ACQUIRE)

void thks_bye();
void breakpoint();

#ifndef NDEBUG
inline void
INFO(const char *format, ...) {
  fprintf(stderr, "INFO: ");

  va_list argptr;
  va_start(argptr, format);
  vfprintf(stderr, format, argptr);
  fprintf(stderr, "\n");
  va_end(argptr);
}
#else
inline void
INFO(...) {}
#endif

inline void
MESSAGE(const char *format, ...) {
  fprintf(stderr, "APIR: ");

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

  abort();
}

static inline bool
util_is_power_of_two_nonzero64(uint64_t v)
{
   return IS_POT_NONZERO(v);
}

static inline uint64_t
align64(uint64_t value, uint64_t alignment)
{
   assert(util_is_power_of_two_nonzero64(alignment));
   return ALIGN_POT(value, alignment);
}

struct list_head
{
    struct list_head *prev;
    struct list_head *next;
};

struct util_sparse_array {
   size_t elem_size;
   unsigned node_size_log2;

   uintptr_t root;
};

void *util_sparse_array_get(struct util_sparse_array *arr, uint64_t idx);
void util_sparse_array_init(struct util_sparse_array *arr,
			    size_t elem_size, size_t node_size);

inline void
os_time_sleep(int64_t usecs)
{
   struct timespec time;
   time.tv_sec = usecs / 1000000;
   time.tv_nsec = (usecs % 1000000) * 1000;
   while (clock_nanosleep(CLOCK_MONOTONIC, 0, &time, &time) == EINTR);
}
