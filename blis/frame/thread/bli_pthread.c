/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2018, Southern Methodist University
   Copyright (C) 2018, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

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

#include "blis.h"

#if defined(BLIS_DISABLE_SYSTEM)

// This branch defines a pthread-like API, bli_pthread_*(), and implements it
// in terms of "dummy" code that doesn't depend on POSIX threads or any other
// threading mechanism. See issue #454 to see the use case that prompted this
// feature.
// NOTE: THIS CODE DOES NOT IMPLEMENT THREADING AND IS NOT THREAD-SAFE!

// -- pthread_create(), pthread_join() --

int bli_pthread_create
     (
       bli_pthread_t*            thread,
       const bli_pthread_attr_t* attr,
       void*                   (*start_routine)(void*),
       void*                     arg
     )
{
	//return pthread_create( thread, attr, start_routine, arg );
	start_routine( arg );
	return 0;
}

int bli_pthread_join
     (
       bli_pthread_t thread,
       void**        retval
     )
{
	//return pthread_join( thread, retval );
	return 0;
}

// -- pthread_mutex_*() --

int bli_pthread_mutex_init
     (
       bli_pthread_mutex_t*           mutex,
       const bli_pthread_mutexattr_t* attr
     )
{
	//return pthread_mutex_init( mutex, attr );
	return 0;
}

int bli_pthread_mutex_destroy
     (
       bli_pthread_mutex_t* mutex
     )
{
	//return pthread_mutex_destroy( mutex );
	return 0;
}

int bli_pthread_mutex_lock
     (
       bli_pthread_mutex_t* mutex
     )
{
	//return pthread_mutex_lock( mutex );
	return 0;
}

int bli_pthread_mutex_trylock
     (
       bli_pthread_mutex_t* mutex
     )
{
	//return pthread_mutex_trylock( mutex );
	return 0;
}

int bli_pthread_mutex_unlock
     (
       bli_pthread_mutex_t* mutex
     )
{
	//return pthread_mutex_unlock( mutex );
	return 0;
}

// -- pthread_cond_*() --

int bli_pthread_cond_init
     (
       bli_pthread_cond_t*           cond,
       const bli_pthread_condattr_t* attr
     )
{
	//return pthread_cond_init( cond, attr );
	return 0;
}

int bli_pthread_cond_destroy
     (
       bli_pthread_cond_t* cond
     )
{
	//return pthread_cond_destroy( cond );
	return 0;
}

int bli_pthread_cond_wait
     (
       bli_pthread_cond_t*  cond,
       bli_pthread_mutex_t* mutex
     )
{
	//return pthread_cond_wait( cond, mutex );
	return 0;
}

int bli_pthread_cond_broadcast
     (
       bli_pthread_cond_t* cond
     )
{
	//return pthread_cond_broadcast( cond );
	return 0;
}

// -- pthread_once() --

void bli_pthread_once
     (
       bli_pthread_once_t* once,
       void              (*init)(void)
     )
{
	//pthread_once( once, init );
	init();
}

#if 0
// NOTE: This part of the API is disabled because (1) we don't actually need
// _self() or _equal() yet, and (2) when we do try to include these functions,
// AppVeyor for some reason fails on all the Windows/clang builds with the
// error:
//    libblis.a(bli_pthread.o) : error LNK2019: unresolved external symbol
//     __imp_CompareObjectHandles referenced in function bli_pthread_equal

// -- pthread_self() --

bli_pthread_t bli_pthread_self
     (
       void
     )
{
	return 0;
}

// -- pthread_equal() --

int bli_pthread_equal
     (
       bli_pthread_t t1,
       bli_pthread_t t2
     )
{
	// We don't bother comparing t1 and t2 since we must, by definition, be
	// executing the same thread if there is not threading mechanism on the
	// system.
	return 1;
}
#endif

#elif defined(_MSC_VER) // !defined(BLIS_DISABLE_SYSTEM)

#include <errno.h>

// This branch defines a pthread-like API, bli_pthread_*(), and implements it
// in terms of Windows API calls.

// -- pthread_create(), pthread_join() --

typedef struct bli_thread_param
{
	void* (*start_routine)( void* );
	void*   param;
	void**  retval;

} bli_thread_param;

static DWORD bli_thread_func
     (
       void* param_
     )
{
	bli_thread_param* param = param_;
	*param->retval = param->start_routine( param->param );
	return 0;
}

int bli_pthread_create
     (
       bli_pthread_t*            thread,
       const bli_pthread_attr_t* attr,
       void*                   (*start_routine)(void*),
       void*                     arg
     )
{
	if ( attr ) return EINVAL;
	bli_thread_param param = { start_routine, arg, &thread->retval };
	thread->handle = CreateThread( NULL, 0, bli_thread_func, &param, 0, NULL );
	if ( !thread->handle ) return EAGAIN;
	return 0;
}

int bli_pthread_join
     (
       bli_pthread_t thread,
       void**        retval
     )
{
	if ( !WaitForSingleObject( thread.handle, INFINITE ) ) return EAGAIN;
	if ( retval ) *retval = thread.retval;
	return 0;
}

// -- pthread_mutex_*() --

int bli_pthread_mutex_init
     (
       bli_pthread_mutex_t*           mutex,
       const bli_pthread_mutexattr_t* attr
     )
{
	if ( attr ) return EINVAL;
	InitializeSRWLock( mutex );
	return 0;
}

int bli_pthread_mutex_destroy
     (
       bli_pthread_mutex_t* mutex
     )
{
	return 0;
}

int bli_pthread_mutex_lock
     (
       bli_pthread_mutex_t* mutex
     )
{
	AcquireSRWLockExclusive( mutex );
	return 0;
}

int bli_pthread_mutex_trylock
     (
       bli_pthread_mutex_t* mutex
     )
{
	return TryAcquireSRWLockExclusive( mutex ) ? 0 : EBUSY;
}

int bli_pthread_mutex_unlock
     (
       bli_pthread_mutex_t* mutex
     )
{
	ReleaseSRWLockExclusive( mutex );
	return 0;
}

// -- pthread_cond_*() --

int bli_pthread_cond_init
     (
       bli_pthread_cond_t*           cond,
       const bli_pthread_condattr_t* attr
     )
{
	if ( attr ) return EINVAL;
	InitializeConditionVariable( cond );
	return 0;
}

int bli_pthread_cond_destroy
     (
       bli_pthread_cond_t* cond
     )
{
	( void )cond;
	return 0;
}

int bli_pthread_cond_wait
     (
       bli_pthread_cond_t*  cond,
       bli_pthread_mutex_t* mutex
     )
{
	if ( !SleepConditionVariableSRW( cond, mutex, INFINITE, 0 ) ) return EAGAIN;
	return 0;
}

int bli_pthread_cond_broadcast
     (
       bli_pthread_cond_t* cond
     )
{
	WakeAllConditionVariable( cond );
	return 0;
}

// -- pthread_once() --

static BOOL bli_init_once_wrapper
     (
       bli_pthread_once_t* once,
       void*               param,
       void**              context
     )
{
	( void )once;
	( void )context;
	typedef void (*callback)( void );
	((callback)param)();
	return TRUE;
}

void bli_pthread_once
     (
       bli_pthread_once_t* once,
       void              (*init)(void)
     )
{
	InitOnceExecuteOnce( once, bli_init_once_wrapper, init, NULL );
}

#if 0
// NOTE: This part of the API is disabled because (1) we don't actually need
// _self() or _equal() yet, and (2) when we do try to include these functions,
// AppVeyor for some reason fails on all the Windows/clang builds with the
// error:
//    libblis.a(bli_pthread.o) : error LNK2019: unresolved external symbol
//     __imp_CompareObjectHandles referenced in function bli_pthread_equal

// -- pthread_self() --

bli_pthread_t bli_pthread_self
     (
       void
     )
{
	bli_pthread_t t;

	// Note: BLIS will only ever use bli_pthread_self() in conjunction with
	// bli_pthread_equal(), and thus setting the .retval field is unnecessary.
	// Despite this, we set it to NULL anyway.
	t.handle = GetCurrentThread();
	t.retval = NULL;

	return t;
}

// -- pthread_equal() --

int bli_pthread_equal
     (
       bli_pthread_t t1,
       bli_pthread_t t2
     )
{
	return ( int )CompareObjectHandles( t1.handle, t2.handle );
}
#endif

#else // !defined(BLIS_DISABLE_SYSTEM) && !defined(_MSC_VER)

// This branch defines a pthreads-like API, bli_pthreads_*(), and implements it
// in terms of the corresponding pthreads_*() types, macros, and function calls.
// This branch is compiled for Linux and other non-Windows environments where
// we assume that *some* implementation of pthreads is provided (although it
// may lack barriers--see below).

// -- pthread_create(), pthread_join() --

int bli_pthread_create
     (
       bli_pthread_t*            thread,
       const bli_pthread_attr_t* attr,
       void*                   (*start_routine)(void*),
       void*                     arg
     )
{
	return pthread_create( thread, attr, start_routine, arg );
}

int bli_pthread_join
     (
       bli_pthread_t thread,
       void**        retval
     )
{
	return pthread_join( thread, retval );
}

// -- pthread_mutex_*() --

int bli_pthread_mutex_init
     (
       bli_pthread_mutex_t*           mutex,
       const bli_pthread_mutexattr_t* attr
     )
{
	return pthread_mutex_init( mutex, attr );
}

int bli_pthread_mutex_destroy
     (
       bli_pthread_mutex_t* mutex
     )
{
	return pthread_mutex_destroy( mutex );
}

int bli_pthread_mutex_lock
     (
       bli_pthread_mutex_t* mutex
     )
{
	return pthread_mutex_lock( mutex );
}

int bli_pthread_mutex_trylock
     (
       bli_pthread_mutex_t* mutex
     )
{
	return pthread_mutex_trylock( mutex );
}

int bli_pthread_mutex_unlock
     (
       bli_pthread_mutex_t* mutex
     )
{
	return pthread_mutex_unlock( mutex );
}

// -- pthread_cond_*() --

int bli_pthread_cond_init
     (
       bli_pthread_cond_t*           cond,
       const bli_pthread_condattr_t* attr
     )
{
	return pthread_cond_init( cond, attr );
}

int bli_pthread_cond_destroy
     (
       bli_pthread_cond_t* cond
     )
{
	return pthread_cond_destroy( cond );
}

int bli_pthread_cond_wait
     (
       bli_pthread_cond_t*  cond,
       bli_pthread_mutex_t* mutex
     )
{
	return pthread_cond_wait( cond, mutex );
}

int bli_pthread_cond_broadcast
     (
       bli_pthread_cond_t* cond
     )
{
	return pthread_cond_broadcast( cond );
}

// -- pthread_once() --

void bli_pthread_once
     (
       bli_pthread_once_t* once,
       void              (*init)(void)
     )
{
	pthread_once( once, init );
}

#if 0
// NOTE: This part of the API is disabled because (1) we don't actually need
// _self() or _equal() yet, and (2) when we do try to include these functions,
// AppVeyor for some reason fails on all the Windows/clang builds with the
// error:
//    libblis.a(bli_pthread.o) : error LNK2019: unresolved external symbol
//     __imp_CompareObjectHandles referenced in function bli_pthread_equal

// -- pthread_self() --

bli_pthread_t bli_pthread_self
     (
       void
     )
{
	return pthread_self();
}

// -- pthread_equal() --

int bli_pthread_equal
     (
       bli_pthread_t t1,
       bli_pthread_t t2
     )
{
	return pthread_equal( t1, t2 );
}
#endif

#endif // !defined(BLIS_DISABLE_SYSTEM) && !defined(_MSC_VER)




// -- pthread_barrier_*() --

#if defined(BLIS_DISABLE_SYSTEM)

int bli_pthread_barrier_init
     (
       bli_pthread_barrier_t*           barrier,
       const bli_pthread_barrierattr_t* attr,
       unsigned int                     count
     )
{
	//return pthread_barrier_init( barrier, attr, count );
	return 0;
}

int bli_pthread_barrier_destroy
     (
       bli_pthread_barrier_t* barrier
     )
{
	//return pthread_barrier_destroy( barrier );
	return 0;
}

int bli_pthread_barrier_wait
     (
       bli_pthread_barrier_t* barrier
     )
{
	//return pthread_barrier_wait( barrier );
	return 0;
}

#elif defined(__APPLE__) || defined(_MSC_VER) // !defined(BLIS_DISABLE_SYSTEM)

#include <errno.h>

// For OS X and Windows, we define barriers ourselves in terms of the rest
// of the API, though for slightly different reasons: For Windows, we must
// define barriers because we are defining *everything* from scratch. For
// OS X, we must define barriers because Apple chose to omit barriers from
// their implementation of POSIX threads (since barriers are actually
// optional to the POSIX standard).

int bli_pthread_barrier_init
     (
       bli_pthread_barrier_t*           barrier,
       const bli_pthread_barrierattr_t* attr,
       unsigned int                     count )
{
	if ( attr ) return EINVAL;
	if ( count == 0 ) return EINVAL;

	int err;
	if ( (err = bli_pthread_mutex_init( &barrier->mutex, 0 )) != 0 ) return err;
	if ( (err = bli_pthread_cond_init( &barrier->cond, 0 )) != 0 )
	{
		bli_pthread_mutex_destroy( &barrier->mutex );
		return err;
	}
	barrier->tripCount = count;
	barrier->count = 0;

	return 0;
}

int bli_pthread_barrier_destroy
     (
       bli_pthread_barrier_t *barrier
     )
{
	bli_pthread_cond_destroy( &barrier->cond );
	bli_pthread_mutex_destroy( &barrier->mutex );
	return 0;
}

int bli_pthread_barrier_wait
     (
       bli_pthread_barrier_t *barrier
     )
{
	bli_pthread_mutex_lock( &barrier->mutex );
	++(barrier->count);
	if ( barrier->count >= barrier->tripCount )
	{
		barrier->count = 0;
		bli_pthread_cond_broadcast( &barrier->cond );
		bli_pthread_mutex_unlock( &barrier->mutex );
		return 1;
	}
	else
	{
		bli_pthread_cond_wait( &barrier->cond, &(barrier->mutex) );
		bli_pthread_mutex_unlock( &barrier->mutex );
		return 0;
	}
}

#else // !defined(BLIS_DISABLE_SYSTEM) && !defined(__APPLE__) && !defined(_MSC_VER)

// Linux environments implement the pthread_barrier* sub-API. So, if we're
// on Linux, we can simply call those functions, just as we did before for
// the other functions.

int bli_pthread_barrier_init
     (
       bli_pthread_barrier_t*           barrier,
       const bli_pthread_barrierattr_t* attr,
       unsigned int                     count
     )
{
	return pthread_barrier_init( barrier, attr, count );
}

int bli_pthread_barrier_destroy
     (
       bli_pthread_barrier_t* barrier
     )
{
	return pthread_barrier_destroy( barrier );
}

int bli_pthread_barrier_wait
     (
       bli_pthread_barrier_t* barrier
     )
{
	return pthread_barrier_wait( barrier );
}

#endif

// -- Non-standard extensions --------------------------------------------------

// -- pthread_switch --

//
// Note that bli_pthread_switch_t has the following properties:
//
// 1. Access to a switch is protected by a mutex specific to that switch, and
//    therefore state changes are thread-safe.
//
// 2. An initialized switch always starts in the "off" state.
//
// 3. Calling _switch_on() when the switch is already "on" results in an early
//    return (no action); similar for _switch_off() when it is already "off".
//
// 4. The _switch_on() and _switch_off() functions each return an error code
//    that is equal to the return value of their user-supplied functions,
//    provided the function in question was actually called rather than being
//    skipped. When a function call is skipped (as in (3) above), the return
//    value from _switch_on() and/or _switch_off() is 0 (success).
//
//    Note that the user-supplied functions must abide by the convention that a
//    return value of 0 indicates success and all other values indicate failure
//    (of some kind). The switch and the user-supplied function must agree on
//    how "success" is conveyed because the switch must know whether to toggle
//    its state after inspecting the return value of the user-supplied function.
//

int bli_pthread_switch_on
     (
       bli_pthread_switch_t* sw,
       int                 (*init)(void)
     )
{
	// NOTE: This function assumes that init() will return 0 on success;
	// otherwise, it will return some other integer. If the function
	// partially succeeds (in such a way that it must be called again in
	// order to complete), it should treat that outcome as failure and
	// return a non-zero value.

	// Initialize the return value with the error code for success.
	int r_val = 0;

	// Proceed only if the switch is currently off; otherwise, we return with
	// an error code of 0.
	if ( sw->status == 0 )
	{
		// Wait for and acquire the switch's lock.
		bli_pthread_mutex_lock( &sw->mutex );

		// Check the status of the switch once more now that we've acquired the
		// lock. Proceed with calling the init() function only if the switch
		// is still off; otherwise, release the lock with an error code of 0.
		if ( sw->status == 0 )
		{
			// Call the init() function and catch its return value in r_val.
			r_val = init();

			// If the init() function succeeded, turn the switch on;
			// otherwise, leave the switch off.
			if ( r_val == 0 )
				sw->status = 1;
		}

		// Release the switch's lock.
		bli_pthread_mutex_unlock( &sw->mutex );
	}

	return r_val;
}

int bli_pthread_switch_off
     (
       bli_pthread_switch_t* sw,
       int                 (*deinit)(void)
     )
{
	// NOTE: This function assumes that deinit() will return 0 on success;
	// otherwise, it will return some other integer. If the function
	// partially succeeds (in such a way that it must be called again in
	// order to complete), it should treat that outcome as failure and
	// return a non-zero value.

	// Initialize the return value with the error code for success.
	int r_val = 0;

	// Proceed only if the switch is currently on; otherwise, we return with
	// an error code of 0.
	if ( sw->status == 1 )
	{
		// Wait for and acquire the switch's lock.
		bli_pthread_mutex_lock( &sw->mutex );

		// Check the status of the switch once more now that we've acquired the
		// lock. Proceed with calling the deinit() function only if the switch
		// is still on; otherwise, release the lock with an error code of 0.
		if ( sw->status == 1 )
		{
			// Call the deinit() function and catch its return value in r_val.
			r_val = deinit();

			// If the deinit() function succeeded, turn the switch off;
			// otherwise, leave the switch on.
			if ( r_val == 0 )
				sw->status = 0;
		}

		// Release the switch's lock.
		bli_pthread_mutex_unlock( &sw->mutex );
	}

	return r_val;
}

