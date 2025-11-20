#ifndef HTP_UTILS_H
#define HTP_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <AEEStdErr.h>
#include <inttypes.h>
#include <remote.h>
#include <stdbool.h>

/* Offset to differentiate HLOS and Hexagon error codes.
   Stores the value of AEE_EOFFSET for Hexagon. */
#ifndef DSP_OFFSET
#    define DSP_OFFSET 0x80000400
#endif

/* Errno for connection reset by peer. */
#ifndef ECONNRESET
#    ifdef __hexagon__
#        define ECONNRESET 104
#    endif
#endif

/* Abstraction of different OS specific sleep APIs.
   SLEEP accepts input in seconds. */
#ifndef SLEEP
#    ifdef __hexagon__
#        define SLEEP(x)                      \
            { /* Do nothing for simulator. */ \
            }
#    else
#        ifdef _WINDOWS
#            define SLEEP(x) Sleep(1000 * x) /* Sleep accepts input in milliseconds. */
#        else
#            define SLEEP(x) sleep(x)        /* sleep accepts input in seconds. */
#        endif
#    endif
#endif

/* Include windows specific header files. */
#ifdef _WINDOWS
#    include <sysinfoapi.h>
#    include <windows.h>
#    define _CRT_SECURE_NO_WARNINGS         1
#    define _WINSOCK_DEPRECATED_NO_WARNINGS 1
/* Including this file for custom implementation of getopt function. */
#    include "getopt_custom.h"
#endif

/* Includes and defines for all HLOS except windows */
#if !defined(__hexagon__) && !defined(_WINDOWS)
#    include "unistd.h"

#    include <sys/time.h>
#endif

/* Includes and defines for Hexagon and all HLOS except Windows. */
#if !defined(_WINDOWS)
/* Weak reference to remote symbol for compilation. */
#    pragma weak remote_session_control
#    pragma weak remote_handle_control
#    pragma weak remote_handle64_control
#    pragma weak fastrpc_mmap
#    pragma weak fastrpc_munmap
#    pragma weak rpcmem_alloc2
#endif

#if !defined(_WINDOWS)
#    pragma weak remote_system_request
#endif
/**
 * Wrapper for FastRPC Capability API: query DSP support.
 *
 * @param[out]  domain pointer to supported domain.
 * @return      0          if query is successful.
 *              non-zero   if error, return value points to the error.
 */
int get_dsp_support(int * domain);

/**
 * Wrapper for FastRPC Capability API: query VTCM information.
 *
 * @param[in]   domain value of domain in the queried.
 * @param[out]  capability capability value of the attribute queried.
 * @param[in]   attr value of the attribute to the queried.
 * @return      0          if query is successful.
 *              non-zero   if error, return value points to the error.
 */
int get_vtcm_info(int domain, uint32_t * capability, uint32_t attr);

/**
 * Wrapper for FastRPC Capability API: query unsigned pd support on CDSP domain.
 *
 * @return      true          if unsigned pd is supported.
 *              false         if unsigned pd is not supported, capability query failed.
 */

bool get_unsignedpd_support(void);

/**
 * Wrapper for FastRPC Capability API: query unsigned pd support.
 *
 * @param[in]   domain value of domain in the queried.
 * @return      true          if unsigned pd is supported.
 *              false         if unsigned pd is not supported, capability query failed.
 */

bool is_unsignedpd_supported(int domain_id);

/**
 * is_valid_domain_id API: query a domain id is valid.
 *
 * @param[in]   domain value of domain in the queried.
 * @param[in]   compute_only value of domain is only compared with CDSP domains supported by the target when enabled.
 * @return      true          if value of domain is valid.
 *              false         if value of domain is not valid.
 */

bool is_valid_domain_id(int domain_id, int compute_only);

/**
 * get_domain API: get domain struct from domain value.
 *
 * @param[in]  domain value of a domain
 * @return     Returns domain struct of the domain if it is supported or else
 *             returns NULL.
 *
 */

domain * get_domain(int domain_id);

/**
 * get_domains_info API: get information for all the domains available on the device
 *
 * @param[in]  domain_type pointer to domain type
 * @param[in]  num_domains pointer to number of domains
 * @param[in]  domains_info pointer to save discovered domains information.
 * @return     0 if query is successful.
 *              non-zero if error, return value points to the error.
 *
 * It is user's responsibility to free the memory used to store the domains info whose address is present in domains_info before closing the application.
 *
 */

int get_domains_info(char * domain_type, int * num_domains, fastrpc_domain ** domains_info);

/**
 * get_effective_domain_id API: get effective domain id for given session id
 *
 * @param[in]  domain_name pointer to domain name
 * @param[in]  session_id
 * @param[in]  effec_domain_id pointer to save obtained effective domain id.
 * @return     0 if query is successful.
 *              non-zero if error, return value points to the error.
 *
 */

int get_effective_domain_id(char * domain_name, int session_id, int * effec_domain_id);

/**
 * is_async_fastrpc_supported API: query a domain id has async fastrpc supported or not
 *
 * @param[in]  domain_id value of a domain
 * @return     Returns true or false stating support of Async FastRPC
 *
 */

bool is_async_fastrpc_supported(int domain_id);

/**
 * is_status_notification_supported API: query the DSP for STATUS_NOTIFICATION_SUPPORT information
 *
 * @param[in]  domain_id value of a domain
 * @return     Returns true or false stating status notification support information
 *
 */
bool is_status_notification_supported(int domain_id);

/**
 * get_hmx_support_info API: query the DSP for HMX SUPPORT information
 *
 * @param[in]   domain_id value of a domain
 * @param[out]  capability capability value of the attribute queried.
 * @param[in]   attr value of the attribute to the queried.
 * @return      0 if query is successful.
 *              non-zero if error, return value points to the error.
 *
 */
int get_hmx_support_info(int domain, uint32_t * capability, uint32_t attr);

/**
 * get_hex_arch_ver API: query the Hexagon processor architecture version information
 *
 * @param[in]   domain_id value of a domain
 * @param[out]  Arch version (73, 75, ...)
 * @return      0 if query is successful.
 *              non-zero if error, return value points to the error.
 *
 */
int get_hex_arch_ver(int domain, int * arch);

/**
 * get_hvx_support_info API: query the DSP for HVX SUPPORT information
 *
 * @param[in]   domain_id value of a domain
 * @param[out]  capability capability value of the attribute queried.
 * @param[in]   attr value of the attribute to the queried.
 * @return      0 if query is successful.
 *              non-zero if error, return value points to the error.
 *
 */
int get_hvx_support_info(int domain, uint32_t * capability, uint32_t attr);

#ifdef __cplusplus
}
#endif

#endif  //DSP_CAPABILITIES_UTILS_H
