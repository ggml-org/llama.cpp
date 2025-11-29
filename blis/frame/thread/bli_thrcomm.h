/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
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

#ifndef BLIS_THRCOMM_H
#define BLIS_THRCOMM_H

// Define barrier_t, which is specific to the tree barrier in the OpenMP
// implementation. This needs to be done first since it is (potentially)
// used within the definition of thrcomm_t below.

#ifdef BLIS_ENABLE_OPENMP
#ifdef BLIS_TREE_BARRIER
struct barrier_s
{
	int               arity;
	struct barrier_s* dad;

	// We insert a cache line of padding here to eliminate false sharing between
	// the fields above and fields below.
	char   padding1[ BLIS_CACHE_LINE_SIZE ];

	dim_t             count;

	// We insert a cache line of padding here to eliminate false sharing between
	// the fields above and fields below.
	char   padding2[ BLIS_CACHE_LINE_SIZE ];

	gint_t            signal;

	// We insert a cache line of padding here to eliminate false sharing between
	// this struct and the next one.
	char   padding3[ BLIS_CACHE_LINE_SIZE ];
};
typedef struct barrier_s barrier_t;
#endif
#endif

// Define hpx_barrier_t, which is specific to the barrier used in HPX
// implementation. This needs to be done first since it is (potentially)
// used within the definition of thrcomm_t below.

#ifdef BLIS_ENABLE_HPX
typedef struct hpx_barrier_t
{
	void* handle;
} hpx_barrier_t;
#endif

// Define the thrcomm_t structure, which will be common to all threading
// implementations.

typedef struct thrcomm_s
{
	// -- Fields common to all threading implementations --

	void*       sent_object;
	dim_t       n_threads;
	timpl_t     ti;

	// We insert a cache line of padding here to eliminate false sharing between
	// the fields above and fields below.
	char   padding1[ BLIS_CACHE_LINE_SIZE ];

	// NOTE: barrier_sense was originally a gint_t-based bool_t, but upon
	// redefining bool_t as bool we discovered that some gcc __atomic built-ins
	// don't allow the use of bool for the variables being operated upon.
	// (Specifically, this was observed of __atomic_fetch_xor(), but it likely
	// applies to all other related built-ins.) Thus, we get around this by
	// redefining barrier_sense as a gint_t.
	//volatile gint_t  barrier_sense;
	gint_t barrier_sense;

	// We insert a cache line of padding here to eliminate false sharing between
	// the fields above and fields below.
	char   padding2[ BLIS_CACHE_LINE_SIZE ];

	dim_t  barrier_threads_arrived;

	// We insert a cache line of padding here to eliminate false sharing between
	// the fields above and whatever data structures follow.
	char   padding3[ BLIS_CACHE_LINE_SIZE ];

	// -- Fields specific to OpenMP --

	#ifdef BLIS_ENABLE_OPENMP
	#ifdef BLIS_TREE_BARRIER
	// This field is only needed if the tree barrier implementation is being
	// compiled. The non-tree barrier code does not use it.
	barrier_t** barriers;
	#endif
	#endif

	// -- Fields specific to pthreads --

	#ifdef BLIS_ENABLE_PTHREADS
	#ifdef BLIS_USE_PTHREAD_BARRIER
	// This field is only needed if the pthread_barrier_t implementation is
	// being compiled. The non-pthread_barrier_t code does not use it.
	bli_pthread_barrier_t barrier;
	#endif
	#endif

	// -- Fields specific to HPX --

	#ifdef BLIS_ENABLE_HPX
	hpx_barrier_t barrier;
	#endif

} thrcomm_t;





// Include definitions (mostly thrcomm_t) specific to the method of
// multithreading.
#include "bli_thrcomm_single.h"
#include "bli_thrcomm_openmp.h"
#include "bli_thrcomm_pthreads.h"
#include "bli_thrcomm_hpx.h"

// Define a function pointer type for each of the functions that are
// "overloaded" by each method of multithreading.
typedef void (*thrcomm_init_ft)( dim_t nt, thrcomm_t* comm );
typedef void (*thrcomm_cleanup_ft)( thrcomm_t* comm );
typedef void (*thrcomm_barrier_ft)( dim_t tid, thrcomm_t* comm );


// thrcomm_t query (field only)

BLIS_INLINE dim_t bli_thrcomm_num_threads( thrcomm_t* comm )
{
	return comm->n_threads;
}

BLIS_INLINE timpl_t bli_thrcomm_thread_impl( thrcomm_t* comm )
{
	return comm->ti;
}


// Threading method-agnostic function prototypes.
BLIS_EXPORT_BLIS thrcomm_t* bli_thrcomm_create( timpl_t ti, pool_t* sba_pool, dim_t n_threads );
BLIS_EXPORT_BLIS void       bli_thrcomm_free( pool_t* sba_pool, thrcomm_t* comm );

// Threading method-specific function prototypes.
// NOTE: These are the prototypes to the dispatcher functions and thus they
// require the timpl_t as an argument. The threading-specific functions can
// (and do) omit the timpl_t from their function signatures since their
// threading implementation is intrinsically known.
BLIS_EXPORT_BLIS void  bli_thrcomm_init( timpl_t ti, dim_t n_threads, thrcomm_t* comm );
BLIS_EXPORT_BLIS void  bli_thrcomm_cleanup( thrcomm_t* comm );
BLIS_EXPORT_BLIS void  bli_thrcomm_barrier( dim_t thread_id, thrcomm_t* comm );

// Other function prototypes.
BLIS_EXPORT_BLIS void* bli_thrcomm_bcast( dim_t inside_id, void* to_send, thrcomm_t* comm );
void                   bli_thrcomm_barrier_atomic( dim_t thread_id, thrcomm_t* comm );

#endif

