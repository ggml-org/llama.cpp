/*
   The latest version of this library is available on GitHub;
   https://github.com/sheredom/subprocess.h
*/

/*
   This is free and unencumbered software released into the public domain.

   Anyone is free to copy, modify, publish, use, compile, sell, or
   distribute this software, either in source code form or as a compiled
   binary, for any purpose, commercial or non-commercial, and by any
   means.

   In jurisdictions that recognize copyright laws, the author or authors
   of this software dedicate any and all copyright interest in the
   software to the public domain. We make this dedication for the benefit
   of the public at large and to the detriment of our heirs and
   successors. We intend this dedication to be an overt act of
   relinquishment in perpetuity of all present and future rights to this
   software under copyright law.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
   IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
   OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
   ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
   OTHER DEALINGS IN THE SOFTWARE.

   For more information, please refer to <http://unlicense.org/>
*/

#ifndef SHEREDOM_SUBPROCESS_H_INCLUDED
#define SHEREDOM_SUBPROCESS_H_INCLUDED

#if defined(_MSC_VER)
#pragma warning(push, 1)

/* disable warning: '__cplusplus' is not defined as a preprocessor macro,
 * replacing with '0' for '#if/#elif' */
#pragma warning(disable : 4668)
#endif

#include <stdio.h>
#include <string.h>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#if defined(__TINYC__)
#define SUBPROCESS_ATTRIBUTE(a) __attribute((a))
#else
#define SUBPROCESS_ATTRIBUTE(a) __attribute__((a))
#endif

#if defined(_MSC_VER)
#define subprocess_pure
#define subprocess_weak __inline
#define subprocess_tls __declspec(thread)
#elif defined(__MINGW32__)
#define subprocess_pure SUBPROCESS_ATTRIBUTE(pure)
#define subprocess_weak static SUBPROCESS_ATTRIBUTE(used)
#define subprocess_tls __thread
#elif defined(__clang__) || defined(__GNUC__) || defined(__TINYC__)
#define subprocess_pure SUBPROCESS_ATTRIBUTE(pure)
#define subprocess_weak SUBPROCESS_ATTRIBUTE(weak)
#define subprocess_tls __thread
#else
#error Non clang, non gcc, non MSVC compiler found!
#endif

struct subprocess_s;

enum subprocess_option_e {
  // stdout and stderr are the same FILE.
  subprocess_option_combined_stdout_stderr = 0x1,

  // The child process should inherit the environment variables of the parent.
  subprocess_option_inherit_environment = 0x2,

  // Enable asynchronous reading of stdout/stderr before it has completed.
  subprocess_option_enable_async = 0x4,

  // Enable the child process to be spawned with no window visible if supported
  // by the platform.
  subprocess_option_no_window = 0x8,

  // Search for program names in the PATH variable. Always enabled on Windows.
  // Note: this will **not** search for paths in any provided custom environment
  // and instead uses the PATH of the spawning process.
  subprocess_option_search_user_path = 0x10,

  // Make subprocess_read_stdout and subprocess_read_stderr return immediately
  // with 0 if no data is available. Requires subprocess_option_enable_async.
  subprocess_option_enable_async_no_wait = 0x20
};

// Error codes returned by subprocess_create and subprocess_create_ex.
// subprocess_error_success is always zero; all errors are non-zero.
enum subprocess_error_e {
  subprocess_error_success = 0,
  subprocess_error_unknown = -1,
  subprocess_error_invalid_options = -2,
  subprocess_error_invalid_environment = -3,
  subprocess_error_not_found = -4,
  subprocess_error_permission_denied = -5,
  subprocess_error_no_memory = -6,
  subprocess_error_pipe = -7,
  subprocess_error_spawn = -8
};

#if defined(__cplusplus)
extern "C" {
#endif

/// @brief Create a process.
/// @param command_line An array of strings for the command line to execute for
/// this process. The last element must be NULL to signify the end of the array.
/// On Windows these strings are interpreted as UTF-8 and passed to the Unicode
/// process creation APIs. The memory backing this parameter only needs to
/// persist until this function returns.
/// @param options A bit field of subprocess_option_e's to pass.
/// @param out_process The newly created process.
/// @return On success zero is returned. On failure a non-zero
/// `subprocess_error_e` value is returned; inspect `errno` on POSIX platforms
/// or `GetLastError()` on Windows for the platform-specific failure reason.
subprocess_weak int subprocess_create(const char *const command_line[],
                                      int options,
                                      struct subprocess_s *const out_process);

/// @brief Create a process (extended create).
/// @param command_line An array of strings for the command line to execute for
/// this process. The last element must be NULL to signify the end of the array.
/// On Windows these strings are interpreted as UTF-8 and passed to the Unicode
/// process creation APIs. The memory backing this parameter only needs to
/// persist until this function returns.
/// @param options A bit field of subprocess_option_e's to pass.
/// @param environment An optional array of strings for the environment to use
/// for a child process (each element of the form FOO=BAR). The last element
/// must be NULL to signify the end of the array. On Windows these strings are
/// interpreted as UTF-8.
/// @param process_cwd The current working directory of the newly created
/// process. If NULL, will be the same as the parent process. On Windows this
/// string is interpreted as UTF-8.
/// @param out_process The newly created process.
/// @return On success zero is returned. On failure a non-zero
/// `subprocess_error_e` value is returned; inspect `errno` on POSIX platforms
/// or `GetLastError()` on Windows for the platform-specific failure reason.
///
/// If `options` contains `subprocess_option_inherit_environment`, then
/// `environment` must be NULL.
subprocess_weak int
subprocess_create_ex(const char *const command_line[], int options,
                     const char *const environment[],
                     const char *const process_cwd,
                     struct subprocess_s *const out_process);

/// @brief Get the standard input file for a process.
/// @param process The process to query.
/// @return The file for standard input of the process.
///
/// The file returned can be written to by the parent process to feed data to
/// the standard input of the process.
subprocess_pure subprocess_weak FILE *
subprocess_stdin(const struct subprocess_s *const process);

/// @brief Get the standard output file for a process.
/// @param process The process to query.
/// @return The file for standard output of the process.
///
/// The file returned can be read from by the parent process to read data from
/// the standard output of the child process.
subprocess_pure subprocess_weak FILE *
subprocess_stdout(const struct subprocess_s *const process);

/// @brief Get the standard error file for a process.
/// @param process The process to query.
/// @return The file for standard error of the process.
///
/// The file returned can be read from by the parent process to read data from
/// the standard error of the child process.
///
/// If the process was created with the subprocess_option_combined_stdout_stderr
/// option bit set, this function will return NULL, and the subprocess_stdout
/// function should be used for both the standard output and error combined.
subprocess_pure subprocess_weak FILE *
subprocess_stderr(const struct subprocess_s *const process);

/// @brief Wait for a process to finish execution.
/// @param process The process to wait for.
/// @param out_return_code The return code of the returned process (can be
/// NULL).
/// @return On success zero is returned.
///
/// Joining a process will close the stdin pipe to the process.
subprocess_weak int subprocess_join(struct subprocess_s *const process,
                                    int *const out_return_code);

/// @brief Destroy a previously created process.
/// @param process The process to destroy.
/// @return On success zero is returned.
///
/// If the process to be destroyed had not finished execution, it may out live
/// the parent process.
subprocess_weak int subprocess_destroy(struct subprocess_s *const process);

/// @brief Terminate a previously created process.
/// @param process The process to terminate.
/// @return On success zero is returned.
///
/// If the process to be destroyed had not finished execution, it will be
/// terminated (i.e killed).
subprocess_weak int subprocess_terminate(struct subprocess_s *const process);

/// @brief Read the standard output from the child process.
/// @param process The process to read from.
/// @param buffer The buffer to read into.
/// @param size The maximum number of bytes to read.
/// @return The number of bytes actually read into buffer. Can only be 0 if the
/// process has complete, or if the process was created with
/// `subprocess_option_enable_async_no_wait` and no data is currently available.
///
/// The only safe way to read from the standard output of a process during it's
/// execution is to use the `subprocess_option_enable_async` option in
/// conjunction with this method.
subprocess_weak unsigned
subprocess_read_stdout(struct subprocess_s *const process, char *const buffer,
                       unsigned size);

/// @brief Read the standard error from the child process.
/// @param process The process to read from.
/// @param buffer The buffer to read into.
/// @param size The maximum number of bytes to read.
/// @return The number of bytes actually read into buffer. Can only be 0 if the
/// process has complete, or if the process was created with
/// `subprocess_option_enable_async_no_wait` and no data is currently available.
///
/// The only safe way to read from the standard error of a process during it's
/// execution is to use the `subprocess_option_enable_async` option in
/// conjunction with this method.
subprocess_weak unsigned
subprocess_read_stderr(struct subprocess_s *const process, char *const buffer,
                       unsigned size);

/// @brief Returns if the subprocess is currently still alive and executing.
/// @param process The process to check.
/// @return If the process is still alive non-zero is returned.
subprocess_weak int subprocess_alive(struct subprocess_s *const process);

#if defined(__cplusplus)
#define SUBPROCESS_CAST(type, x) static_cast<type>(x)
#define SUBPROCESS_PTR_CAST(type, x) reinterpret_cast<type>(x)
#define SUBPROCESS_CONST_CAST(type, x) const_cast<type>(x)
#define SUBPROCESS_NULL NULL
#else
#define SUBPROCESS_CAST(type, x) ((type)(x))
#define SUBPROCESS_PTR_CAST(type, x) ((type)(x))
#define SUBPROCESS_CONST_CAST(type, x) ((type)(x))
#define SUBPROCESS_NULL 0
#endif

#if !defined(_WIN32)
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <stdlib.h>
#if defined(__APPLE__)
#include <AvailabilityMacros.h>
#endif
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#if defined(_WIN32)

#include <wchar.h>

#if defined(__clang__)
#if __has_warning("-Wc++-keyword")
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++-keyword"
#endif
#endif

typedef wchar_t subprocess_wchar_t;

#if defined(__clang__)
#if __has_warning("-Wc++-keyword")
#pragma clang diagnostic pop
#endif
#endif

#if (_MSC_VER < 1920)
#ifdef _WIN64
typedef __int64 subprocess_intptr_t;
typedef unsigned __int64 subprocess_size_t;
#else
typedef int subprocess_intptr_t;
typedef unsigned int subprocess_size_t;
#endif
#else
#include <inttypes.h>

typedef intptr_t subprocess_intptr_t;
typedef size_t subprocess_size_t;
#endif

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-identifier"
#endif

typedef struct _PROCESS_INFORMATION *LPPROCESS_INFORMATION;
typedef struct _SECURITY_ATTRIBUTES *LPSECURITY_ATTRIBUTES;
typedef struct _STARTUPINFOW *LPSTARTUPINFOW;
typedef struct _OVERLAPPED *LPOVERLAPPED;

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#ifdef _MSC_VER
#pragma warning(push, 1)
#endif
#ifdef __MINGW32__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif

struct subprocess_subprocess_information_s {
  void *hProcess;
  void *hThread;
  unsigned long dwProcessId;
  unsigned long dwThreadId;
};

struct subprocess_security_attributes_s {
  unsigned long nLength;
  void *lpSecurityDescriptor;
  int bInheritHandle;
};

struct subprocess_startup_info_s {
  unsigned long cb;
  subprocess_wchar_t *lpReserved;
  subprocess_wchar_t *lpDesktop;
  subprocess_wchar_t *lpTitle;
  unsigned long dwX;
  unsigned long dwY;
  unsigned long dwXSize;
  unsigned long dwYSize;
  unsigned long dwXCountChars;
  unsigned long dwYCountChars;
  unsigned long dwFillAttribute;
  unsigned long dwFlags;
  unsigned short wShowWindow;
  unsigned short cbReserved2;
  unsigned char *lpReserved2;
  void *hStdInput;
  void *hStdOutput;
  void *hStdError;
};

struct subprocess_overlapped_s {
  uintptr_t Internal;
  uintptr_t InternalHigh;
  union {
    struct {
      unsigned long Offset;
      unsigned long OffsetHigh;
    } DUMMYSTRUCTNAME;
    void *Pointer;
  } DUMMYUNIONNAME;

  void *hEvent;
};

#ifdef __clang__
#pragma clang diagnostic pop
#endif
#ifdef __MINGW32__
#pragma GCC diagnostic pop
#endif
#ifdef _MSC_VER
#pragma warning(pop)
#endif

__declspec(dllimport) unsigned long __stdcall GetLastError(void);
__declspec(dllimport) void __stdcall SetLastError(unsigned long);
__declspec(dllimport) int __stdcall SetHandleInformation(void *, unsigned long,
                                                         unsigned long);
__declspec(dllimport) int __stdcall CreatePipe(void **, void **,
                                               LPSECURITY_ATTRIBUTES,
                                               unsigned long);
__declspec(dllimport) void *__stdcall CreateNamedPipeA(
    const char *, unsigned long, unsigned long, unsigned long, unsigned long,
    unsigned long, unsigned long, LPSECURITY_ATTRIBUTES);
__declspec(dllimport) int __stdcall ReadFile(void *, void *, unsigned long,
                                             unsigned long *, LPOVERLAPPED);
__declspec(dllimport) unsigned long __stdcall GetCurrentProcessId(void);
__declspec(dllimport) unsigned long __stdcall GetCurrentThreadId(void);
__declspec(dllimport) void *__stdcall CreateFileA(const char *, unsigned long,
                                                  unsigned long,
                                                  LPSECURITY_ATTRIBUTES,
                                                  unsigned long, unsigned long,
                                                  void *);
__declspec(dllimport) void *__stdcall CreateEventA(LPSECURITY_ATTRIBUTES, int,
                                                   int, const char *);
__declspec(dllimport) int __stdcall CreateProcessW(
    const subprocess_wchar_t *, subprocess_wchar_t *, LPSECURITY_ATTRIBUTES,
    LPSECURITY_ATTRIBUTES, int, unsigned long, void *,
    const subprocess_wchar_t *, LPSTARTUPINFOW, LPPROCESS_INFORMATION);
__declspec(dllimport) int __stdcall MultiByteToWideChar(
    unsigned int, unsigned long, const char *, int, subprocess_wchar_t *, int);
__declspec(dllimport) int __stdcall CloseHandle(void *);
__declspec(dllimport) unsigned long __stdcall WaitForSingleObject(
    void *, unsigned long);
__declspec(dllimport) int __stdcall GetExitCodeProcess(
    void *, unsigned long *lpExitCode);
__declspec(dllimport) int __stdcall TerminateProcess(void *, unsigned int);
__declspec(dllimport) unsigned long __stdcall WaitForMultipleObjects(
    unsigned long, void *const *, int, unsigned long);
__declspec(dllimport) int __stdcall GetOverlappedResult(void *, LPOVERLAPPED,
                                                        unsigned long *, int);
__declspec(dllimport) int __stdcall PeekNamedPipe(void *, void *, unsigned long,
                                                  unsigned long *,
                                                  unsigned long *,
                                                  unsigned long *);

#if defined(_DLL)
#define SUBPROCESS_DLLIMPORT __declspec(dllimport)
#else
#define SUBPROCESS_DLLIMPORT
#endif

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-identifier"
#endif

SUBPROCESS_DLLIMPORT int __cdecl _fileno(FILE *);
SUBPROCESS_DLLIMPORT int __cdecl _open_osfhandle(subprocess_intptr_t, int);
SUBPROCESS_DLLIMPORT subprocess_intptr_t __cdecl _get_osfhandle(int);
SUBPROCESS_DLLIMPORT int __cdecl _close(int);

#ifndef __MINGW32__
void *__cdecl _alloca(subprocess_size_t);
#else
#include <malloc.h>
#endif

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#else
typedef size_t subprocess_size_t;
#endif

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpadded"
#endif
struct subprocess_s {
  FILE *stdin_file;
  FILE *stdout_file;
  FILE *stderr_file;

#if defined(_WIN32)
  void *hProcess;
  void *hStdInput;
  void *hEventOutput;
  void *hEventError;
#else
  pid_t child;
  int return_status;
#endif

  int alive;
  int no_wait;
};
#ifdef __clang__
#pragma clang diagnostic pop
#endif

#if defined(_WIN32)
subprocess_weak int subprocess_error_from_windows_error(unsigned long error);
int subprocess_error_from_windows_error(unsigned long error) {
  enum {
    errorFileNotFound = 2,
    errorPathNotFound = 3,
    errorTooManyOpenFiles = 4,
    errorAccessDenied = 5,
    errorInvalidHandle = 6,
    errorNotEnoughMemory = 8,
    errorOutOfMemory = 14,
    errorInvalidDrive = 15,
    errorBadPathname = 161,
    errorDirectory = 267
  };

  switch (error) {
  case errorFileNotFound:
  case errorPathNotFound:
  case errorInvalidDrive:
  case errorBadPathname:
  case errorDirectory:
    return subprocess_error_not_found;
  case errorAccessDenied:
    return subprocess_error_permission_denied;
  case errorTooManyOpenFiles:
  case errorNotEnoughMemory:
  case errorOutOfMemory:
    return subprocess_error_no_memory;
  case errorInvalidHandle:
    return subprocess_error_pipe;
  default:
    return subprocess_error_unknown;
  }
}
#else
subprocess_weak int subprocess_error_from_errno(int error);
int subprocess_error_from_errno(int error) {
  switch (error) {
  case ENOENT:
  case ENOTDIR:
    return subprocess_error_not_found;
  case EACCES:
  case EPERM:
    return subprocess_error_permission_denied;
  case EMFILE:
  case ENFILE:
  case ENOMEM:
    return subprocess_error_no_memory;
  default:
    return subprocess_error_unknown;
  }
}
#endif

#if defined(__clang__)
#if __has_warning("-Wunsafe-buffer-usage")
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#endif
#endif

#if defined(_WIN32)
subprocess_weak int subprocess_create_named_pipe_helper(void **rd, void **wr);
subprocess_weak void subprocess_close_handle(void **handle);
void subprocess_close_handle(void **handle) {
  const void *const invalidHandleValue =
      SUBPROCESS_PTR_CAST(void *, ~(SUBPROCESS_CAST(subprocess_intptr_t, 0)));

  if (*handle && (invalidHandleValue != *handle)) {
    CloseHandle(*handle);
  }

  *handle = SUBPROCESS_NULL;
}

int subprocess_create_named_pipe_helper(void **rd, void **wr) {
  const unsigned long pipeAccessInbound = 0x00000001;
  const unsigned long fileFlagOverlapped = 0x40000000;
  const unsigned long pipeTypeByte = 0x00000000;
  const unsigned long pipeWait = 0x00000000;
  const unsigned long genericWrite = 0x40000000;
  const unsigned long openExisting = 3;
  const unsigned long fileAttributeNormal = 0x00000080;
  const void *const invalidHandleValue =
      SUBPROCESS_PTR_CAST(void *, ~(SUBPROCESS_CAST(subprocess_intptr_t, 0)));
  struct subprocess_security_attributes_s saAttr = {sizeof(saAttr),
                                                    SUBPROCESS_NULL, 1};
  char name[256] = {0};
#if defined(__clang__)
#if __has_warning("-Wunique-object-duplication")
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunique-object-duplication"
#endif
#endif
  static subprocess_tls long index = 0;
#if defined(__clang__)
#if __has_warning("-Wunique-object-duplication")
#pragma clang diagnostic pop
#endif
#endif
  const long unique = index++;

  *rd = SUBPROCESS_NULL;
  *wr = SUBPROCESS_NULL;

#if defined(_MSC_VER) && _MSC_VER < 1900
#pragma warning(push, 1)
#pragma warning(disable : 4996)
  _snprintf(name, sizeof(name) - 1,
            "\\\\.\\pipe\\sheredom_subprocess_h.%08lx.%08lx.%ld",
            GetCurrentProcessId(), GetCurrentThreadId(), unique);
#pragma warning(pop)
#else
  snprintf(name, sizeof(name) - 1,
           "\\\\.\\pipe\\sheredom_subprocess_h.%08lx.%08lx.%ld",
           GetCurrentProcessId(), GetCurrentThreadId(), unique);
#endif

  *rd =
      CreateNamedPipeA(name, pipeAccessInbound | fileFlagOverlapped,
                       pipeTypeByte | pipeWait, 1, 4096, 4096, SUBPROCESS_NULL,
                       SUBPROCESS_PTR_CAST(LPSECURITY_ATTRIBUTES, &saAttr));

  if (invalidHandleValue == *rd) {
    return -1;
  }

  *wr = CreateFileA(name, genericWrite, SUBPROCESS_NULL,
                    SUBPROCESS_PTR_CAST(LPSECURITY_ATTRIBUTES, &saAttr),
                    openExisting, fileAttributeNormal, SUBPROCESS_NULL);

  if (invalidHandleValue == *wr) {
    subprocess_close_handle(rd);
    return -1;
  }

  return 0;
}
#endif

int subprocess_create(const char *const commandLine[], int options,
                      struct subprocess_s *const out_process) {
  return subprocess_create_ex(commandLine, options, SUBPROCESS_NULL,
                              SUBPROCESS_NULL, out_process);
}

int subprocess_create_ex(const char *const commandLine[], int options,
                         const char *const environment[],
                         const char *const process_cwd,
                         struct subprocess_s *const out_process) {
#if defined(_WIN32)
  int fd;
  int async_no_wait;
  void *rd = SUBPROCESS_NULL;
  void *wr = SUBPROCESS_NULL;
  char *commandLineCombined;
  subprocess_wchar_t *commandLineCombinedWide = SUBPROCESS_NULL;
  subprocess_wchar_t *process_cwd_wide = SUBPROCESS_NULL;
  subprocess_size_t len;
  int wide_len;
  int i, j;
  int need_quoting;
  unsigned long flags = 0;
  unsigned long last_error = 0;
  int result = subprocess_error_unknown;
  const unsigned int codePageUtf8 = 65001;
  const unsigned long mbErrInvalidChars = 0x00000008;
  const unsigned long startFUseStdHandles = 0x00000100;
  const unsigned long handleFlagInherit = 0x00000001;
  const unsigned long createNoWindow = 0x08000000;
  const unsigned long createUnicodeEnvironment = 0x00000400;
  struct subprocess_subprocess_information_s processInfo = {SUBPROCESS_NULL,
                                                            SUBPROCESS_NULL, 0,
                                                            0};
  struct subprocess_security_attributes_s saAttr = {sizeof(saAttr),
                                                    SUBPROCESS_NULL, 1};
  subprocess_wchar_t empty_environment[2] = {0, 0};
  subprocess_wchar_t *used_environment = SUBPROCESS_NULL;
  struct subprocess_startup_info_s startInfo = {0,
                                                SUBPROCESS_NULL,
                                                SUBPROCESS_NULL,
                                                SUBPROCESS_NULL,
                                                0,
                                                0,
                                                0,
                                                0,
                                                0,
                                                0,
                                                0,
                                                0,
                                                0,
                                                0,
                                                SUBPROCESS_NULL,
                                                SUBPROCESS_NULL,
                                                SUBPROCESS_NULL,
                                                SUBPROCESS_NULL};

  async_no_wait = subprocess_option_enable_async_no_wait ==
                  (options & subprocess_option_enable_async_no_wait);

  if (async_no_wait && (subprocess_option_enable_async !=
                        (options & subprocess_option_enable_async))) {
    return subprocess_error_invalid_options;
  }

  startInfo.cb = sizeof(startInfo);
  startInfo.dwFlags = startFUseStdHandles;

  if (subprocess_option_no_window == (options & subprocess_option_no_window)) {
    flags |= createNoWindow;
  }

  memset(out_process, 0, sizeof(*out_process));

  if (subprocess_option_inherit_environment !=
      (options & subprocess_option_inherit_environment)) {
    flags |= createUnicodeEnvironment;

    if (SUBPROCESS_NULL == environment) {
      used_environment = empty_environment;
    } else {
      // We always end with two null terminators. MultiByteToWideChar includes
      // each environment string's null terminator, so start with one extra.
      len = 1;

      for (i = 0; environment[i]; i++) {
        wide_len = MultiByteToWideChar(codePageUtf8, mbErrInvalidChars,
                                       environment[i], -1, SUBPROCESS_NULL, 0);
        if (0 == wide_len) {
          result = subprocess_error_from_windows_error(GetLastError());
          if (subprocess_error_unknown == result) {
            result = subprocess_error_spawn;
          }
          goto cleanup;
        }

        len += SUBPROCESS_CAST(subprocess_size_t, wide_len);
      }

      if (((SUBPROCESS_CAST(subprocess_size_t, -1)) /
           sizeof(subprocess_wchar_t)) < len) {
        result = subprocess_error_no_memory;
        goto cleanup;
      }

      used_environment = SUBPROCESS_CAST(
          subprocess_wchar_t *, _alloca(len * sizeof(subprocess_wchar_t)));
      if (!used_environment) {
        result = subprocess_error_no_memory;
        goto cleanup;
      }

      // Re-use len for the insertion position.
      len = 0;

      for (i = 0; environment[i]; i++) {
        wide_len = MultiByteToWideChar(codePageUtf8, mbErrInvalidChars,
                                       environment[i], -1, SUBPROCESS_NULL, 0);
        if (0 == wide_len) {
          result = subprocess_error_from_windows_error(GetLastError());
          if (subprocess_error_unknown == result) {
            result = subprocess_error_spawn;
          }
          goto cleanup;
        }

        if (0 == MultiByteToWideChar(codePageUtf8, mbErrInvalidChars,
                                     environment[i], -1,
                                     &used_environment[len], wide_len)) {
          result = subprocess_error_from_windows_error(GetLastError());
          if (subprocess_error_unknown == result) {
            result = subprocess_error_spawn;
          }
          goto cleanup;
        }

        len += SUBPROCESS_CAST(subprocess_size_t, wide_len);
      }

      // End with the second null terminator.
      used_environment[len++] = 0;
    }
  } else {
    if (SUBPROCESS_NULL != environment) {
      return subprocess_error_invalid_environment;
    }
  }

  if (!CreatePipe(&rd, &wr, SUBPROCESS_PTR_CAST(LPSECURITY_ATTRIBUTES, &saAttr),
                  0)) {
    result = subprocess_error_pipe;
    goto cleanup;
  }

  if (!SetHandleInformation(wr, handleFlagInherit, 0)) {
    result = subprocess_error_pipe;
    goto cleanup;
  }

  fd = _open_osfhandle(SUBPROCESS_PTR_CAST(subprocess_intptr_t, wr), 0);
  if (-1 == fd) {
    result = subprocess_error_pipe;
    goto cleanup;
  }
  wr = SUBPROCESS_NULL;

  out_process->stdin_file = _fdopen(fd, "wb");
  if (SUBPROCESS_NULL == out_process->stdin_file) {
    _close(fd);
    goto cleanup;
  }

  startInfo.hStdInput = rd;
  rd = SUBPROCESS_NULL;

  if (options & subprocess_option_enable_async) {
    if (subprocess_create_named_pipe_helper(&rd, &wr)) {
      result = subprocess_error_pipe;
      goto cleanup;
    }
  } else {
    if (!CreatePipe(&rd, &wr,
                    SUBPROCESS_PTR_CAST(LPSECURITY_ATTRIBUTES, &saAttr), 0)) {
      result = subprocess_error_pipe;
      goto cleanup;
    }
  }

  if (!SetHandleInformation(rd, handleFlagInherit, 0)) {
    result = subprocess_error_pipe;
    goto cleanup;
  }

  fd = _open_osfhandle(SUBPROCESS_PTR_CAST(subprocess_intptr_t, rd), 0);
  if (-1 == fd) {
    result = subprocess_error_pipe;
    goto cleanup;
  }
  rd = SUBPROCESS_NULL;

  out_process->stdout_file = _fdopen(fd, "rb");
  if (SUBPROCESS_NULL == out_process->stdout_file) {
    _close(fd);
    goto cleanup;
  }

  startInfo.hStdOutput = wr;
  wr = SUBPROCESS_NULL;

  if (subprocess_option_combined_stdout_stderr ==
      (options & subprocess_option_combined_stdout_stderr)) {
    out_process->stderr_file = out_process->stdout_file;
    startInfo.hStdError = startInfo.hStdOutput;
  } else {
    if (options & subprocess_option_enable_async) {
      if (subprocess_create_named_pipe_helper(&rd, &wr)) {
        result = subprocess_error_pipe;
        goto cleanup;
      }
    } else {
      if (!CreatePipe(&rd, &wr,
                      SUBPROCESS_PTR_CAST(LPSECURITY_ATTRIBUTES, &saAttr), 0)) {
        result = subprocess_error_pipe;
        goto cleanup;
      }
    }

    if (!SetHandleInformation(rd, handleFlagInherit, 0)) {
      result = subprocess_error_pipe;
      goto cleanup;
    }

    fd = _open_osfhandle(SUBPROCESS_PTR_CAST(subprocess_intptr_t, rd), 0);
    if (-1 == fd) {
      result = subprocess_error_pipe;
      goto cleanup;
    }
    rd = SUBPROCESS_NULL;

    out_process->stderr_file = _fdopen(fd, "rb");
    if (SUBPROCESS_NULL == out_process->stderr_file) {
      _close(fd);
      goto cleanup;
    }

    startInfo.hStdError = wr;
    wr = SUBPROCESS_NULL;
  }

  if (options & subprocess_option_enable_async) {
    out_process->hEventOutput =
        CreateEventA(SUBPROCESS_PTR_CAST(LPSECURITY_ATTRIBUTES, &saAttr), 1, 1,
                     SUBPROCESS_NULL);
    out_process->hEventError =
        CreateEventA(SUBPROCESS_PTR_CAST(LPSECURITY_ATTRIBUTES, &saAttr), 1, 1,
                     SUBPROCESS_NULL);
    if (!out_process->hEventOutput || !out_process->hEventError) {
      result = subprocess_error_from_windows_error(GetLastError());
      if (subprocess_error_unknown == result) {
        result = subprocess_error_no_memory;
      }
      goto cleanup;
    }
  } else {
    out_process->hEventOutput = SUBPROCESS_NULL;
    out_process->hEventError = SUBPROCESS_NULL;
  }

  // Combine commandLine together into a single string
  len = 0;
  for (i = 0; commandLine[i]; i++) {
    // for the trailing \0
    len++;

    // Quote the argument if it has a space in it
    if (strpbrk(commandLine[i], "\t\v ") != SUBPROCESS_NULL ||
        commandLine[i][0] == SUBPROCESS_NULL)
      len += 2;

    for (j = 0; '\0' != commandLine[i][j]; j++) {
      switch (commandLine[i][j]) {
      default:
        break;
      case '\\':
        if (commandLine[i][j + 1] == '"') {
          len++;
        }

        break;
      case '"':
        len++;
        break;
      }
      len++;
    }
  }

  commandLineCombined = SUBPROCESS_CAST(char *, _alloca(len));

  if (!commandLineCombined) {
    result = subprocess_error_no_memory;
    goto cleanup;
  }

  // Gonna re-use len to store the write index into commandLineCombined
  len = 0;

  for (i = 0; commandLine[i]; i++) {
    if (0 != i) {
      commandLineCombined[len++] = ' ';
    }

    need_quoting = strpbrk(commandLine[i], "\t\v ") != SUBPROCESS_NULL ||
                   commandLine[i][0] == SUBPROCESS_NULL;
    if (need_quoting) {
      commandLineCombined[len++] = '"';
    }

    for (j = 0; '\0' != commandLine[i][j]; j++) {
      switch (commandLine[i][j]) {
      default:
        break;
      case '\\':
        if (commandLine[i][j + 1] == '"') {
          commandLineCombined[len++] = '\\';
        }

        break;
      case '"':
        commandLineCombined[len++] = '\\';
        break;
      }

      commandLineCombined[len++] = commandLine[i][j];
    }
    if (need_quoting) {
      commandLineCombined[len++] = '"';
    }
  }

  commandLineCombined[len] = '\0';

  wide_len = MultiByteToWideChar(codePageUtf8, mbErrInvalidChars,
                                 commandLineCombined, -1, SUBPROCESS_NULL, 0);
  if (0 == wide_len) {
    result = subprocess_error_from_windows_error(GetLastError());
    if (subprocess_error_unknown == result) {
      result = subprocess_error_spawn;
    }
    goto cleanup;
  }

  commandLineCombinedWide = SUBPROCESS_CAST(
      subprocess_wchar_t *,
      _alloca(SUBPROCESS_CAST(subprocess_size_t, wide_len) *
              sizeof(subprocess_wchar_t)));
  if (!commandLineCombinedWide) {
    result = subprocess_error_no_memory;
    goto cleanup;
  }

  if (0 == MultiByteToWideChar(codePageUtf8, mbErrInvalidChars,
                               commandLineCombined, -1,
                               commandLineCombinedWide, wide_len)) {
    result = subprocess_error_from_windows_error(GetLastError());
    if (subprocess_error_unknown == result) {
      result = subprocess_error_spawn;
    }
    goto cleanup;
  }

  if (process_cwd) {
    wide_len = MultiByteToWideChar(codePageUtf8, mbErrInvalidChars, process_cwd,
                                   -1, SUBPROCESS_NULL, 0);
    if (0 == wide_len) {
      result = subprocess_error_from_windows_error(GetLastError());
      if (subprocess_error_unknown == result) {
        result = subprocess_error_spawn;
      }
      goto cleanup;
    }

    process_cwd_wide = SUBPROCESS_CAST(
        subprocess_wchar_t *,
        _alloca(SUBPROCESS_CAST(subprocess_size_t, wide_len) *
                sizeof(subprocess_wchar_t)));
    if (!process_cwd_wide) {
      result = subprocess_error_no_memory;
      goto cleanup;
    }

    if (0 == MultiByteToWideChar(codePageUtf8, mbErrInvalidChars, process_cwd,
                                 -1, process_cwd_wide, wide_len)) {
      result = subprocess_error_from_windows_error(GetLastError());
      if (subprocess_error_unknown == result) {
        result = subprocess_error_spawn;
      }
      goto cleanup;
    }
  }

  if (!CreateProcessW(
          SUBPROCESS_NULL,
          commandLineCombinedWide, // command line
          SUBPROCESS_NULL,         // process security attributes
          SUBPROCESS_NULL,         // primary thread security attributes
          1,                       // handles are inherited
          flags,                   // creation flags
          used_environment,        // used environment
          process_cwd_wide,        // use specified current directory
          SUBPROCESS_PTR_CAST(LPSTARTUPINFOW,
                              &startInfo), // STARTUPINFO pointer
          SUBPROCESS_PTR_CAST(LPPROCESS_INFORMATION, &processInfo))) {
    result = subprocess_error_from_windows_error(GetLastError());
    if (subprocess_error_unknown == result) {
      result = subprocess_error_spawn;
    }
    goto cleanup;
  }

  out_process->hProcess = processInfo.hProcess;
  processInfo.hProcess = SUBPROCESS_NULL;

  out_process->hStdInput = startInfo.hStdInput;
  startInfo.hStdInput = SUBPROCESS_NULL;

  // We don't need the handle of the primary thread in the called process.
  CloseHandle(processInfo.hThread);
  processInfo.hThread = SUBPROCESS_NULL;

  if (SUBPROCESS_NULL != startInfo.hStdOutput) {
    CloseHandle(startInfo.hStdOutput);

    if (startInfo.hStdError != startInfo.hStdOutput) {
      CloseHandle(startInfo.hStdError);
    }

    startInfo.hStdOutput = SUBPROCESS_NULL;
    startInfo.hStdError = SUBPROCESS_NULL;
  }

  out_process->alive = 1;
  out_process->no_wait = async_no_wait;

  return 0;

cleanup:
  last_error = GetLastError();

  if (subprocess_error_unknown == result) {
    result = subprocess_error_from_windows_error(last_error);
  }

  if (out_process->stdin_file) {
    fclose(out_process->stdin_file);
    out_process->stdin_file = SUBPROCESS_NULL;
  }

  if (out_process->stdout_file) {
    fclose(out_process->stdout_file);
    if (out_process->stderr_file &&
        (out_process->stdout_file != out_process->stderr_file)) {
      fclose(out_process->stderr_file);
    }
    out_process->stdout_file = SUBPROCESS_NULL;
    out_process->stderr_file = SUBPROCESS_NULL;
  }

  subprocess_close_handle(&rd);
  subprocess_close_handle(&wr);
  subprocess_close_handle(&startInfo.hStdInput);

  if (startInfo.hStdOutput == startInfo.hStdError) {
    subprocess_close_handle(&startInfo.hStdOutput);
    startInfo.hStdError = SUBPROCESS_NULL;
  } else {
    subprocess_close_handle(&startInfo.hStdOutput);
    subprocess_close_handle(&startInfo.hStdError);
  }

  subprocess_close_handle(&out_process->hEventOutput);
  subprocess_close_handle(&out_process->hEventError);
  subprocess_close_handle(&processInfo.hThread);
  subprocess_close_handle(&processInfo.hProcess);

  SetLastError(last_error);

  return result;
#else
  int stdinfd[2] = {-1, -1};
  int stdoutfd[2] = {-1, -1};
  int stderrfd[2] = {-1, -1};
  int fd, fd_flags;
  int async_no_wait;
  int actions_created = 0;
  int result = subprocess_error_unknown;
  int saved_errno = 0;
  int posix_error;
  pid_t child = 0;
  extern char **environ;
  char *const empty_environment[1] = {SUBPROCESS_NULL};
  posix_spawn_file_actions_t actions;
  char *const *used_environment;

  async_no_wait = subprocess_option_enable_async_no_wait ==
                  (options & subprocess_option_enable_async_no_wait);

  if (async_no_wait && (subprocess_option_enable_async !=
                        (options & subprocess_option_enable_async))) {
    errno = EINVAL;
    return subprocess_error_invalid_options;
  }

  if (subprocess_option_inherit_environment ==
      (options & subprocess_option_inherit_environment)) {
    if (SUBPROCESS_NULL != environment) {
      errno = EINVAL;
      return subprocess_error_invalid_environment;
    }
  }

  memset(out_process, 0, sizeof(*out_process));

  if (0 != pipe(stdinfd)) {
    saved_errno = errno;
    result = subprocess_error_pipe;
    goto cleanup;
  }

  if (0 != pipe(stdoutfd)) {
    saved_errno = errno;
    result = subprocess_error_pipe;
    goto cleanup;
  }

  if (subprocess_option_combined_stdout_stderr !=
      (options & subprocess_option_combined_stdout_stderr)) {
    if (0 != pipe(stderrfd)) {
      saved_errno = errno;
      result = subprocess_error_pipe;
      goto cleanup;
    }
  }

  if (environment) {
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-qual"
#pragma clang diagnostic ignored "-Wold-style-cast"
#endif
    used_environment = SUBPROCESS_CONST_CAST(char *const *, environment);
#ifdef __clang__
#pragma clang diagnostic pop
#endif
  } else if (subprocess_option_inherit_environment ==
             (options & subprocess_option_inherit_environment)) {
    used_environment = environ;
  } else {
    used_environment = empty_environment;
  }

  posix_error = posix_spawn_file_actions_init(&actions);
  if (0 != posix_error) {
    saved_errno = posix_error;
    result = subprocess_error_from_errno(posix_error);
    if (subprocess_error_unknown == result) {
      result = subprocess_error_spawn;
    }
    goto cleanup;
  }
  actions_created = 1;

  // Set working directory
  if (process_cwd) {
#if defined(__APPLE__) && MAC_OS_X_VERSION_MIN_REQUIRED >= 260000
    posix_error = posix_spawn_file_actions_addchdir(&actions, process_cwd);
#else
#if defined(__APPLE__) && defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    posix_error = posix_spawn_file_actions_addchdir_np(&actions, process_cwd);
#if defined(__APPLE__) && defined(__clang__)
#pragma clang diagnostic pop
#endif
#endif
    if (0 != posix_error) {
      saved_errno = posix_error;
      result = subprocess_error_from_errno(posix_error);
      if (subprocess_error_unknown == result) {
        result = subprocess_error_spawn;
      }
      goto cleanup;
    }
  }

  // Close the stdin write end
  posix_error = posix_spawn_file_actions_addclose(&actions, stdinfd[1]);
  if (0 != posix_error) {
    saved_errno = posix_error;
    result = subprocess_error_from_errno(posix_error);
    if (subprocess_error_unknown == result) {
      result = subprocess_error_spawn;
    }
    goto cleanup;
  }

  // Map the read end to stdin
  posix_error =
      posix_spawn_file_actions_adddup2(&actions, stdinfd[0], STDIN_FILENO);
  if (0 != posix_error) {
    saved_errno = posix_error;
    result = subprocess_error_from_errno(posix_error);
    if (subprocess_error_unknown == result) {
      result = subprocess_error_spawn;
    }
    goto cleanup;
  }

  // Close the stdout read end
  posix_error = posix_spawn_file_actions_addclose(&actions, stdoutfd[0]);
  if (0 != posix_error) {
    saved_errno = posix_error;
    result = subprocess_error_from_errno(posix_error);
    if (subprocess_error_unknown == result) {
      result = subprocess_error_spawn;
    }
    goto cleanup;
  }

  // Map the write end to stdout
  posix_error =
      posix_spawn_file_actions_adddup2(&actions, stdoutfd[1], STDOUT_FILENO);
  if (0 != posix_error) {
    saved_errno = posix_error;
    result = subprocess_error_from_errno(posix_error);
    if (subprocess_error_unknown == result) {
      result = subprocess_error_spawn;
    }
    goto cleanup;
  }

  if (subprocess_option_combined_stdout_stderr ==
      (options & subprocess_option_combined_stdout_stderr)) {
    posix_error = posix_spawn_file_actions_adddup2(&actions, STDOUT_FILENO,
                                                   STDERR_FILENO);
    if (0 != posix_error) {
      saved_errno = posix_error;
      result = subprocess_error_from_errno(posix_error);
      if (subprocess_error_unknown == result) {
        result = subprocess_error_spawn;
      }
      goto cleanup;
    }
  } else {
    // Close the stderr read end
    posix_error = posix_spawn_file_actions_addclose(&actions, stderrfd[0]);
    if (0 != posix_error) {
      saved_errno = posix_error;
      result = subprocess_error_from_errno(posix_error);
      if (subprocess_error_unknown == result) {
        result = subprocess_error_spawn;
      }
      goto cleanup;
    }
    // Map the write end to stdout
    posix_error = posix_spawn_file_actions_adddup2(&actions, stderrfd[1],
                                                   STDERR_FILENO);
    if (0 != posix_error) {
      saved_errno = posix_error;
      result = subprocess_error_from_errno(posix_error);
      if (subprocess_error_unknown == result) {
        result = subprocess_error_spawn;
      }
      goto cleanup;
    }
  }

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-qual"
#pragma clang diagnostic ignored "-Wold-style-cast"
#endif
  if (subprocess_option_search_user_path ==
      (options & subprocess_option_search_user_path)) {
    posix_error = posix_spawnp(&child, commandLine[0], &actions,
                               SUBPROCESS_NULL,
                               SUBPROCESS_CONST_CAST(char *const *, commandLine),
                               used_environment);
    if (0 != posix_error) {
      saved_errno = posix_error;
      result = subprocess_error_from_errno(posix_error);
      if (subprocess_error_unknown == result) {
        result = subprocess_error_spawn;
      }
      goto cleanup;
    }
  } else {
    posix_error = posix_spawn(&child, commandLine[0], &actions,
                              SUBPROCESS_NULL,
                              SUBPROCESS_CONST_CAST(char *const *, commandLine),
                              used_environment);
    if (0 != posix_error) {
      saved_errno = posix_error;
      result = subprocess_error_from_errno(posix_error);
      if (subprocess_error_unknown == result) {
        result = subprocess_error_spawn;
      }
      goto cleanup;
    }
  }
#ifdef __clang__
#pragma clang diagnostic pop
#endif

  // Close the stdin read end
  close(stdinfd[0]);
  stdinfd[0] = -1;
  // Store the stdin write end
  out_process->stdin_file = fdopen(stdinfd[1], "wb");
  if (SUBPROCESS_NULL == out_process->stdin_file) {
    saved_errno = errno;
    result = subprocess_error_from_errno(saved_errno);
    goto cleanup;
  }
  stdinfd[1] = -1;

  // Close the stdout write end
  close(stdoutfd[1]);
  stdoutfd[1] = -1;
  // Store the stdout read end
  out_process->stdout_file = fdopen(stdoutfd[0], "rb");
  if (SUBPROCESS_NULL == out_process->stdout_file) {
    saved_errno = errno;
    result = subprocess_error_from_errno(saved_errno);
    goto cleanup;
  }
  stdoutfd[0] = -1;

  // Set non blocking if we are async and asked not to wait.
  if (async_no_wait) {
    fd = fileno(out_process->stdout_file);
    fd_flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, fd_flags | O_NONBLOCK);
  }

  if (subprocess_option_combined_stdout_stderr ==
      (options & subprocess_option_combined_stdout_stderr)) {
    out_process->stderr_file = out_process->stdout_file;
  } else {
    // Close the stderr write end
    close(stderrfd[1]);
    stderrfd[1] = -1;
    // Store the stderr read end
    out_process->stderr_file = fdopen(stderrfd[0], "rb");
    if (SUBPROCESS_NULL == out_process->stderr_file) {
      saved_errno = errno;
      result = subprocess_error_from_errno(saved_errno);
      goto cleanup;
    }
    stderrfd[0] = -1;

    // Set non blocking if we are async and asked not to wait.
    if (async_no_wait) {
      fd = fileno(out_process->stderr_file);
      fd_flags = fcntl(fd, F_GETFL, 0);
      fcntl(fd, F_SETFL, fd_flags | O_NONBLOCK);
    }
  }

  // Store the child's pid
  out_process->child = child;
  child = 0;

  out_process->alive = 1;
  out_process->no_wait = async_no_wait;

  result = 0;

cleanup:
  if ((0 != result) && (0 == saved_errno)) {
    saved_errno = errno;
  }

  if ((subprocess_error_unknown == result) && (0 != saved_errno)) {
    result = subprocess_error_from_errno(saved_errno);
  }

  if (actions_created) {
    posix_spawn_file_actions_destroy(&actions);
  }

  if (0 != result) {
    if (child) {
      kill(child, 9);
      waitpid(child, SUBPROCESS_NULL, 0);
    }

    if (out_process->stdin_file) {
      fclose(out_process->stdin_file);
      out_process->stdin_file = SUBPROCESS_NULL;
    }

    if (out_process->stdout_file) {
      fclose(out_process->stdout_file);
      if (out_process->stderr_file &&
          (out_process->stdout_file != out_process->stderr_file)) {
        fclose(out_process->stderr_file);
      }
      out_process->stdout_file = SUBPROCESS_NULL;
      out_process->stderr_file = SUBPROCESS_NULL;
    }
  }

  if (-1 != stdinfd[0]) {
    close(stdinfd[0]);
  }
  if (-1 != stdinfd[1]) {
    close(stdinfd[1]);
  }
  if (-1 != stdoutfd[0]) {
    close(stdoutfd[0]);
  }
  if (-1 != stdoutfd[1]) {
    close(stdoutfd[1]);
  }
  if (-1 != stderrfd[0]) {
    close(stderrfd[0]);
  }
  if (-1 != stderrfd[1]) {
    close(stderrfd[1]);
  }

  if ((0 != result) && (0 != saved_errno)) {
    errno = saved_errno;
  }

  return result;
#endif
}

FILE *subprocess_stdin(const struct subprocess_s *const process) {
  return process->stdin_file;
}

FILE *subprocess_stdout(const struct subprocess_s *const process) {
  return process->stdout_file;
}

FILE *subprocess_stderr(const struct subprocess_s *const process) {
  if (process->stdout_file != process->stderr_file) {
    return process->stderr_file;
  } else {
    return SUBPROCESS_NULL;
  }
}

int subprocess_join(struct subprocess_s *const process,
                    int *const out_return_code) {
#if defined(_WIN32)
  const unsigned long infinite = 0xFFFFFFFF;

  if (process->stdin_file) {
    fclose(process->stdin_file);
    process->stdin_file = SUBPROCESS_NULL;
  }

  if (process->hStdInput) {
    CloseHandle(process->hStdInput);
    process->hStdInput = SUBPROCESS_NULL;
  }

  WaitForSingleObject(process->hProcess, infinite);

  if (out_return_code) {
    if (!GetExitCodeProcess(
            process->hProcess,
            SUBPROCESS_PTR_CAST(unsigned long *, out_return_code))) {
      return -1;
    }
  }

  process->alive = 0;

  return 0;
#else
  int status;

  if (process->stdin_file) {
    fclose(process->stdin_file);
    process->stdin_file = SUBPROCESS_NULL;
  }

  if (process->child) {
    if (process->child != waitpid(process->child, &status, 0)) {
      return -1;
    }

    process->child = 0;

    if (WIFEXITED(status)) {
      process->return_status = WEXITSTATUS(status);
    } else {
      process->return_status = EXIT_FAILURE;
    }

    process->alive = 0;
  }

  if (out_return_code) {
    *out_return_code = process->return_status;
  }

  return 0;
#endif
}

int subprocess_destroy(struct subprocess_s *const process) {
  if (process->stdin_file) {
    fclose(process->stdin_file);
    process->stdin_file = SUBPROCESS_NULL;
  }

  if (process->stdout_file) {
    fclose(process->stdout_file);

    if (process->stdout_file != process->stderr_file) {
      fclose(process->stderr_file);
    }

    process->stdout_file = SUBPROCESS_NULL;
    process->stderr_file = SUBPROCESS_NULL;
  }

#if defined(_WIN32)
  if (process->hProcess) {
    CloseHandle(process->hProcess);
    process->hProcess = SUBPROCESS_NULL;

    if (process->hStdInput) {
      CloseHandle(process->hStdInput);
    }

    if (process->hEventOutput) {
      CloseHandle(process->hEventOutput);
    }

    if (process->hEventError) {
      CloseHandle(process->hEventError);
    }
  }
#endif

  return 0;
}

int subprocess_terminate(struct subprocess_s *const process) {
#if defined(_WIN32)
  unsigned int killed_process_exit_code;
  int success_terminate;
  int windows_call_result;

  killed_process_exit_code = 99;
  windows_call_result =
      TerminateProcess(process->hProcess, killed_process_exit_code);
  success_terminate = (windows_call_result == 0) ? 1 : 0;
  return success_terminate;
#else
  int result;
  result = kill(process->child, 9);
  return result;
#endif
}

unsigned subprocess_read_stdout(struct subprocess_s *const process,
                                char *const buffer, unsigned size) {
#if defined(_WIN32)
  void *handle;
  unsigned long bytes_available = 0;
  unsigned long bytes_read = 0;
  struct subprocess_overlapped_s overlapped = {0, 0, {{0, 0}}, SUBPROCESS_NULL};
  overlapped.hEvent = process->hEventOutput;

  handle = SUBPROCESS_PTR_CAST(void *,
                               _get_osfhandle(_fileno(process->stdout_file)));

  if (process->no_wait) {
    if (!PeekNamedPipe(handle, SUBPROCESS_NULL, 0, SUBPROCESS_NULL,
                       &bytes_available, SUBPROCESS_NULL)) {
      return 0;
    }

    if (0 == bytes_available) {
      return 0;
    }
  }

  if (!ReadFile(handle, buffer, size, &bytes_read,
                SUBPROCESS_PTR_CAST(LPOVERLAPPED, &overlapped))) {
    const unsigned long errorIoPending = 997;
    unsigned long error = GetLastError();

    // Means we've got an async read!
    if (error == errorIoPending) {
      const uintptr_t statusPending = 0x00000103;

      const int wait = statusPending == overlapped.Internal;

      if (!GetOverlappedResult(handle,
                               SUBPROCESS_PTR_CAST(LPOVERLAPPED, &overlapped),
                               &bytes_read, wait)) {
        const unsigned long errorHandleEOF = 38;
        const unsigned long errorBrokenPipe = 109;
        const unsigned long errorIoIncomplete = 996;
        error = GetLastError();

        if ((errorHandleEOF != error) && (errorBrokenPipe != error) &&
            (errorIoIncomplete != error)) {
          return 0;
        }
      }
    }
  }

  return SUBPROCESS_CAST(unsigned, bytes_read);
#else
  const int fd = fileno(process->stdout_file);
  const ssize_t bytes_read = read(fd, buffer, size);

  if (bytes_read < 0) {
    return 0;
  }

  return SUBPROCESS_CAST(unsigned, bytes_read);
#endif
}

unsigned subprocess_read_stderr(struct subprocess_s *const process,
                                char *const buffer, unsigned size) {
#if defined(_WIN32)
  void *handle;
  unsigned long bytes_available = 0;
  unsigned long bytes_read = 0;
  struct subprocess_overlapped_s overlapped = {0, 0, {{0, 0}}, SUBPROCESS_NULL};
  overlapped.hEvent = process->hEventError;

  handle = SUBPROCESS_PTR_CAST(void *,
                               _get_osfhandle(_fileno(process->stderr_file)));

  if (process->no_wait) {
    if (!PeekNamedPipe(handle, SUBPROCESS_NULL, 0, SUBPROCESS_NULL,
                       &bytes_available, SUBPROCESS_NULL)) {
      return 0;
    }

    if (0 == bytes_available) {
      return 0;
    }
  }

  if (!ReadFile(handle, buffer, size, &bytes_read,
                SUBPROCESS_PTR_CAST(LPOVERLAPPED, &overlapped))) {
    const unsigned long errorIoPending = 997;
    unsigned long error = GetLastError();

    // Means we've got an async read!
    if (error == errorIoPending) {
      if (!GetOverlappedResult(handle,
                               SUBPROCESS_PTR_CAST(LPOVERLAPPED, &overlapped),
                               &bytes_read, 1)) {
        const unsigned long errorIoIncomplete = 996;
        const unsigned long errorHandleEOF = 38;
        error = GetLastError();

        if ((error != errorIoIncomplete) && (error != errorHandleEOF)) {
          return 0;
        }
      }
    }
  }

  return SUBPROCESS_CAST(unsigned, bytes_read);
#else
  const int fd = fileno(process->stderr_file);
  const ssize_t bytes_read = read(fd, buffer, size);

  if (bytes_read < 0) {
    return 0;
  }

  return SUBPROCESS_CAST(unsigned, bytes_read);
#endif
}

int subprocess_alive(struct subprocess_s *const process) {
  int is_alive = SUBPROCESS_CAST(int, process->alive);

  if (!is_alive) {
    return 0;
  }
#if defined(_WIN32)
  {
    const unsigned long zero = 0x0;
    const unsigned long wait_object_0 = 0x00000000L;

    is_alive = wait_object_0 != WaitForSingleObject(process->hProcess, zero);
  }
#else
  {
    int status;
    is_alive = 0 == waitpid(process->child, &status, WNOHANG);

    // If the process was successfully waited on we need to cleanup now.
    if (!is_alive) {
      if (WIFEXITED(status)) {
        process->return_status = WEXITSTATUS(status);
      } else {
        process->return_status = EXIT_FAILURE;
      }

      // Since we've already successfully waited on the process, we need to wipe
      // the child now.
      process->child = 0;

      if (subprocess_join(process, SUBPROCESS_NULL)) {
        return -1;
      }
    }
  }
#endif

  if (!is_alive) {
    process->alive = 0;
  }

  return is_alive;
}

#if defined(__clang__)
#if __has_warning("-Wunsafe-buffer-usage")
#pragma clang diagnostic pop
#endif
#endif

#if defined(__cplusplus)
} // extern "C"
#endif

#endif /* SHEREDOM_SUBPROCESS_H_INCLUDED */
