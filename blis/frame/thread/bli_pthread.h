/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2018, Southern Methodist University
   Copyright (C) 2018, The University of Texas at Austin

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name(s) of the copyright holder(s) nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef BLIS_PTHREAD_H
#define BLIS_PTHREAD_H

// -- Type and macro definitions -----------------------------------------------

#if defined(BLIS_DISABLE_SYSTEM)

// This branch defines a pthread-like API, bli_pthread_*(), and implements it
// in terms of "dummy" code that doesn't depend on POSIX threads or any other
// threading mechanism. See issue #454 to see the use case that prompted this
// feature.
// NOTE: THIS CODE DOES NOT IMPLEMENT THREADING AND IS NOT THREAD-SAFE!

// -- pthread types --

typedef int bli_pthread_t;
typedef int bli_pthread_attr_t;
typedef int bli_pthread_mutex_t;
typedef int bli_pthread_mutexattr_t;
typedef int bli_pthread_cond_t;
typedef int bli_pthread_condattr_t;
typedef int bli_pthread_once_t;

typedef int bli_pthread_barrier_t;
typedef int bli_pthread_barrierattr_t;

// -- pthreads macros --

#define BLIS_PTHREAD_MUTEX_INITIALIZER 0
#define BLIS_PTHREAD_COND_INITIALIZER  0
#define BLIS_PTHREAD_ONCE_INIT         0

#elif defined(_MSC_VER) // !defined(BLIS_DISABLE_SYSTEM)

// This branch defines a pthread-like API, bli_pthread_*(), and implements it
// in terms of Windows API calls.

// -- pthread types --

typedef struct bli_pthread_s
{
    HANDLE handle;
    void* retval;
} bli_pthread_t;
typedef void bli_pthread_attr_t;
typedef SRWLOCK bli_pthread_mutex_t;
typedef void bli_pthread_mutexattr_t;
typedef CONDITION_VARIABLE bli_pthread_cond_t;
typedef void bli_pthread_condattr_t;
typedef INIT_ONCE bli_pthread_once_t;
typedef struct bli_pthread_barrier_s
{
    bli_pthread_mutex_t mutex;
    bli_pthread_cond_t  cond;
    int                 count;
    int                 tripCount;
} bli_pthread_barrier_t;
typedef void bli_pthread_barrierattr_t;

// -- pthreads macros --

#define BLIS_PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT
#define BLIS_PTHREAD_ONCE_INIT         INIT_ONCE_STATIC_INIT
#define BLIS_PTHREAD_COND_INITIALIZER  CONDITION_VARIABLE_INIT

#else // !defined(BLIS_DISABLE_SYSTEM) && !defined(_MSC_VER)

#include <pthread.h>

// This branch defines a pthreads-like API, bli_pthreads_*(), and implements it
// in terms of the corresponding pthreads_*() types, macros, and function calls.

// -- pthread types --

typedef pthread_t           bli_pthread_t;
typedef pthread_attr_t      bli_pthread_attr_t;
typedef pthread_mutex_t     bli_pthread_mutex_t;
typedef pthread_mutexattr_t bli_pthread_mutexattr_t;
typedef pthread_cond_t      bli_pthread_cond_t;
typedef pthread_condattr_t  bli_pthread_condattr_t;
typedef pthread_once_t      bli_pthread_once_t;

#if defined(__APPLE__)

// For OS X, we must define the barrier types ourselves since Apple does
// not implement barriers in their variant of pthreads.

typedef void bli_pthread_barrierattr_t;

typedef struct bli_pthread_barrier_s
{
	bli_pthread_mutex_t mutex;
	bli_pthread_cond_t  cond;
	int                 count;
	int                 tripCount;
} bli_pthread_barrier_t;

#else

// For other non-Windows OSes (primarily Linux), we can define the barrier
// types in terms of existing pthreads barrier types since we expect they
// will be provided by the pthreads implementation.

typedef pthread_barrier_t      bli_pthread_barrier_t;
typedef pthread_barrierattr_t  bli_pthread_barrierattr_t;

#endif

// -- pthreads macros --

#define BLIS_PTHREAD_MUTEX_INITIALIZER PTHREAD_MUTEX_INITIALIZER
#define BLIS_PTHREAD_COND_INITIALIZER  PTHREAD_COND_INITIALIZER
#define BLIS_PTHREAD_ONCE_INIT         PTHREAD_ONCE_INIT

#endif

// -- Function definitions -----------------------------------------------------

// -- pthread_create(), pthread_join() --

BLIS_EXPORT_BLIS int bli_pthread_create
     (
       bli_pthread_t*            thread,
       const bli_pthread_attr_t* attr,
       void*                   (*start_routine)(void*),
       void*                     arg
     );

BLIS_EXPORT_BLIS int bli_pthread_join
     (
       bli_pthread_t thread,
       void**        retval
     );

// -- pthread_mutex_*() --

BLIS_EXPORT_BLIS int bli_pthread_mutex_init
     (
       bli_pthread_mutex_t*           mutex,
       const bli_pthread_mutexattr_t* attr
     );

BLIS_EXPORT_BLIS int bli_pthread_mutex_destroy
     (
       bli_pthread_mutex_t* mutex
     );

BLIS_EXPORT_BLIS int bli_pthread_mutex_lock
     (
       bli_pthread_mutex_t* mutex
     );

BLIS_EXPORT_BLIS int bli_pthread_mutex_trylock
     (
       bli_pthread_mutex_t* mutex
     );

BLIS_EXPORT_BLIS int bli_pthread_mutex_unlock
     (
       bli_pthread_mutex_t* mutex
     );

// -- pthread_cond_*() --

BLIS_EXPORT_BLIS int bli_pthread_cond_init
     (
       bli_pthread_cond_t*           cond,
       const bli_pthread_condattr_t* attr
     );

BLIS_EXPORT_BLIS int bli_pthread_cond_destroy
     (
       bli_pthread_cond_t* cond
     );

BLIS_EXPORT_BLIS int bli_pthread_cond_wait
     (
       bli_pthread_cond_t*  cond,
       bli_pthread_mutex_t* mutex
     );

BLIS_EXPORT_BLIS int bli_pthread_cond_broadcast
     (
       bli_pthread_cond_t* cond
     );

// -- pthread_once() --

BLIS_EXPORT_BLIS void bli_pthread_once
     (
       bli_pthread_once_t* once,
       void              (*init)(void)
     );

#if 0
// NOTE: This part of the API is disabled because (1) we don't actually need
// _self() or _equal() yet, and (2) when we do try to include these functions,
// AppVeyor for some reason fails on all the Windows/clang builds with the
// error:
//    libblis.a(bli_pthread.o) : error LNK2019: unresolved external symbol
//     __imp_CompareObjectHandles referenced in function bli_pthread_equal

// -- pthread_self() --

BLIS_EXPORT_BLIS bli_pthread_t bli_pthread_self
     (
       void
     );

// -- pthread_equal() --

BLIS_EXPORT_BLIS int bli_pthread_equal
     (
       bli_pthread_t t1,
       bli_pthread_t t2
     );
#endif

// -- pthread_barrier_*() --

BLIS_EXPORT_BLIS int bli_pthread_barrier_init
     (
       bli_pthread_barrier_t*           barrier,
       const bli_pthread_barrierattr_t* attr,
       unsigned int                     count
     );

BLIS_EXPORT_BLIS int bli_pthread_barrier_destroy
     (
       bli_pthread_barrier_t* barrier
     );

BLIS_EXPORT_BLIS int bli_pthread_barrier_wait
     (
       bli_pthread_barrier_t* barrier
     );

// -- Non-standard extensions --------------------------------------------------

// -- pthread_switch --

typedef struct bli_pthread_switch_s
{
    int                 status;
    bli_pthread_mutex_t mutex;
} bli_pthread_switch_t;

#define BLIS_PTHREAD_SWITCH_INIT { .status = 0, \
                                   .mutex  = BLIS_PTHREAD_MUTEX_INITIALIZER }

int bli_pthread_switch_on
     (
       bli_pthread_switch_t* sw,
       int                 (*init)(void)
     );

int bli_pthread_switch_off
     (
       bli_pthread_switch_t* sw,
       int                 (*deinit)(void)
     );

#endif // BLIS_PTHREAD_H
