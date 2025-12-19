
#pragma once

#include <AEEStdErr.h>
#include <remote.h>
#include <dspqueue.h>

#ifdef GGML_BACKEND_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef GGML_BACKEND_BUILD
#            define HTP_DRV_API __declspec(dllexport) extern
#        else
#            define HTP_DRV_API __declspec(dllimport) extern
#        endif
#    else
#        define HTP_DRV_API __attribute__ ((visibility ("default"))) extern
#    endif
#else
#    define HTP_DRV_API extern
#endif

#ifdef _WIN32
#   pragma clang diagnostic ignored "-Wdeprecated-declaration"
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#endif

#ifdef __cplusplus
extern "C" {
#endif

//
// Driver API
//

extern void * rpcmem_alloc(int heapid, uint32_t flags, int size);
extern void * rpcmem_alloc2(int heapid, uint32_t flags, size_t size);
extern void   rpcmem_free(void * po);
extern int    rpcmem_to_fd(void * po);

extern int fastrpc_mmap(int domain, int fd, void * addr, int offset, size_t length, enum fastrpc_map_flags flags);
extern int fastrpc_munmap(int domain, int fd, void * addr, size_t length);

extern AEEResult dspqueue_create(int                 domain,
                                 uint32_t            flags,
                                 uint32_t            req_queue_size,
                                 uint32_t            resp_queue_size,
                                 dspqueue_callback_t packet_callback,
                                 dspqueue_callback_t error_callback,
                                 void *              callback_context,
                                 dspqueue_t *        queue);
extern AEEResult dspqueue_close(dspqueue_t queue);
extern AEEResult dspqueue_export(dspqueue_t queue, uint64_t * queue_id);
extern AEEResult dspqueue_write(dspqueue_t               queue,
                                uint32_t                 flags,
                                uint32_t                 num_buffers,
                                struct dspqueue_buffer * buffers,
                                uint32_t                 message_length,
                                const uint8_t *          message,
                                uint32_t                 timeout_us);

extern AEEResult dspqueue_read(dspqueue_t               queue,
                               uint32_t *               flags,
                               uint32_t                 max_buffers,
                               uint32_t *               num_buffers,
                               struct dspqueue_buffer * buffers,
                               uint32_t                 max_message_length,
                               uint32_t *               message_length,
                               uint8_t *                message,
                               uint32_t                 timeout_us);

extern int remote_handle64_open(const char * name, remote_handle64 * ph);
extern int remote_handle64_invoke(remote_handle64 h, uint32_t dwScalars, remote_arg * pra);
extern int remote_handle64_close(remote_handle64 h);
extern int remote_handle_control(uint32_t req, void * data, uint32_t datalen);
extern int remote_handle64_control(remote_handle64 h, uint32_t req, void * data, uint32_t datalen);
extern int remote_session_control(uint32_t req, void * data, uint32_t datalen);
extern int remote_system_request(system_req_payload * req);

// Driver interface entry point
int htpdrv_init(void);

#ifdef __cplusplus
}
#endif
