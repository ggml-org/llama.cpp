
#pragma once

#include <AEEStdErr.h>
#include <rpcmem.h>
#include <remote.h>
#include <dspqueue.h>

#if defined(_WIN32) && !defined(__MINGW32__)
#    ifdef GGML_BACKEND_BUILD
#        define HTPDRV_API __declspec(dllexport) extern
#    else
#        define HTPDRV_API __declspec(dllimport) extern
#    endif
#else
#    define HTPDRV_API __attribute__ ((visibility ("default"))) extern
#endif

#ifdef _WIN32
#   pragma clang diagnostic ignored "-Wdeprecated-declaration"
#endif

#ifdef __cplusplus
extern "C" {
#endif

//
// Driver interface entry point
//

HTPDRV_API int htpdrv_init(void);

#ifdef __cplusplus
}
#endif
