// sample drv interface

#pragma once

#include <AEEStdErr.h>
#include <remote.h>
#include <dspqueue.h>

#ifdef __QAIC_REMOTE
#undef __QAIC_REMOTE
#endif //__QAIC_REMOTE
#define __QAIC_REMOTE(ff) m_##ff

// Temp disable
#ifdef _WIN32
#   pragma clang diagnostic ignored "-Wdeprecated-declaration"
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void *(*RpcMemAllocFn_t)(int, uint32_t, int);

typedef void *(*RpcMemAllocFn2_t)(int, uint32_t, int);

typedef void (*RpcMemFreeFn_t)(void *);

typedef int (*RpcMemToFdFn_t)(void *);

typedef AEEResult (*DspQueueCreateFn_t)(int, uint32_t, uint32_t, uint32_t,
                                        dspqueue_callback_t, dspqueue_callback_t,
                                        void *, dspqueue_t *);

typedef AEEResult (*DspQueueCloseFn_t)(dspqueue_t);

typedef AEEResult (*DspQueueExportFn_t)(dspqueue_t, uint64_t *);

typedef AEEResult (*DspQueueWriteFn_t)(dspqueue_t, uint32_t, uint32_t,
                                       struct dspqueue_buffer *, uint32_t,
                                       const uint8_t *, uint32_t);

                                       typedef AEEResult (*DspQueueReadFn_t)(dspqueue_t, uint32_t *, uint32_t,
                                      uint32_t *, struct dspqueue_buffer *,
                                      uint32_t, uint32_t *, uint8_t *, uint32_t);

typedef int (*FastRpcMmapFn_t)(int, int, void *, int, size_t, enum fastrpc_map_flags);

typedef int (*FastRpcMunmapFn_t)(int, int, void *, size_t);


typedef int (*RemoteHandle64OpenFn_t)(const char *, remote_handle64 *);

typedef int (*RemoteHandle64InvokeFn_t)(remote_handle64, uint32_t, remote_arg *);

typedef int (*RemoteHandleControlFn_t)(uint32_t, void *, uint32_t);

typedef int (*RemoteHandle64ControlFn_t)(remote_handle64, uint32_t, void *, uint32_t);

typedef int (*RemoteHandle64CloseFn_t)(remote_handle64);

typedef int (*RemoteSessionControlFn_t)(uint32_t, void *, uint32_t);

typedef int (*RemoteSystemRequestFn_t)(system_req_payload *);

extern RpcMemAllocFn_t htpdrv_rpcmem_alloc;

extern RpcMemAllocFn_t htpdrv_rpcmem_alloc2;

extern RpcMemFreeFn_t htpdrv_rpcmem_free;

extern RpcMemToFdFn_t htpdrv_rpcmem_to_fd;

extern FastRpcMmapFn_t htpdrv_fastrpc_mmap;

extern FastRpcMunmapFn_t htpdrv_fastrpc_munmap;
 
extern DspQueueCreateFn_t htpdrv_dspqueue_create;

extern DspQueueCloseFn_t htpdrv_dspqueue_close;

extern DspQueueExportFn_t htpdrv_dspqueue_export;

extern DspQueueWriteFn_t htpdrv_dspqueue_write;

extern DspQueueReadFn_t htpdrv_dspqueue_read;

extern RemoteHandle64OpenFn_t htpdrv_remote_handle64_open;

extern RemoteHandle64InvokeFn_t htpdrv_remote_handle64_invoke;

extern RemoteHandleControlFn_t htpdrv_remote_handle_control;

extern RemoteHandle64ControlFn_t htpdrv_remote_handle64_control;

extern RemoteHandle64CloseFn_t htpdrv_remote_handle64_close;

extern RemoteSessionControlFn_t htpdrv_remote_session_control;

extern RemoteSystemRequestFn_t htpdrv_remote_system_request;

// CDSPRPC Driver interface
int htpdrv_initialize(void);

#ifdef __cplusplus
}
#endif

