/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.
   Copyright (C) 2022, Southern Methodist University

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

#define BLIS_NUM_STATIC_COMMS 80

void bli_thrinfo_attach_sub_node( thrinfo_t* sub_node, thrinfo_t* t )
{
	dim_t next = 0;
	for ( ; next < BLIS_MAX_SUB_NODES; next++ )
	{
		if ( bli_thrinfo_sub_node( next, t ) == NULL )
			break;
	}

	if ( next == BLIS_MAX_SUB_NODES )
		bli_abort();

	bli_thrinfo_set_sub_node( next, sub_node, t );
}

thrinfo_t* bli_thrinfo_create_root
     (
       thrcomm_t* comm,
       dim_t      thread_id,
       pool_t*    sba_pool,
       pba_t*     pba
     )
{
	return bli_thrinfo_create
	(
	  comm,
	  thread_id,
	  1,
	  0,
	  FALSE,
	  sba_pool,
	  pba
	);
}

thrinfo_t* bli_thrinfo_create
     (
       thrcomm_t* comm,
       dim_t      thread_id,
       dim_t      n_way,
       dim_t      work_id,
       bool       free_comm,
       pool_t*    sba_pool,
       pba_t*     pba
     )
{
	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_thrinfo_create(): " );
	#endif

	thrinfo_t* thread = bli_sba_acquire( sba_pool, sizeof( thrinfo_t ) );

	bli_thrinfo_set_comm( comm, thread );
	bli_thrinfo_set_thread_id( thread_id, thread );
	bli_thrinfo_set_n_way( n_way, thread );
	bli_thrinfo_set_work_id( work_id, thread );
	bli_thrinfo_set_free_comm( free_comm, thread );
	bli_thrinfo_set_sba_pool( sba_pool, thread );
	bli_thrinfo_set_pba( pba, thread );
	bli_mem_clear( bli_thrinfo_mem( thread ) );

	for ( dim_t i = 0; i < BLIS_MAX_SUB_NODES; i++ )
		bli_thrinfo_set_sub_node( i, NULL, thread );

	return thread;
}

void bli_thrinfo_free
     (
       thrinfo_t* thread
     )
{
	if ( thread == NULL ) return;

	pool_t* sba_pool   = bli_thrinfo_sba_pool( thread );
	mem_t*  cntl_mem_p = bli_thrinfo_mem( thread );
	pba_t*  pba        = bli_thrinfo_pba( thread );

	// Recursively free all children of the current thrinfo_t.
	for ( dim_t i = 0; i < BLIS_MAX_SUB_NODES; i++ )
	{
		thrinfo_t* thrinfo_sub_node = bli_thrinfo_sub_node( i, thread );
		if ( thrinfo_sub_node != NULL )
			bli_thrinfo_free( thrinfo_sub_node );
	}

	// Free the communicators, but only if the current thrinfo_t struct
	// is marked as needing them to be freed. The most common example of
	// thrinfo_t nodes NOT marked as needing their comms freed are those
	// associated with packm thrinfo_t nodes.
	if ( bli_thrinfo_needs_free_comm( thread ) )
	{
		// The ochief always frees his communicator.
		if ( bli_thrinfo_am_chief( thread ) )
			bli_thrcomm_free( sba_pool, bli_thrinfo_comm( thread ) );
	}

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_thrinfo_free(): " );
	#endif

	// Free any allocated memory from the pba.
	if ( bli_mem_is_alloc( cntl_mem_p ) && bli_thrinfo_am_chief( thread ) )
	{
		bli_pba_release
		(
		  pba,
		  cntl_mem_p
		);
	}

	// Free the thrinfo_t struct.
	bli_sba_release( sba_pool, thread );
}

// -----------------------------------------------------------------------------

thrinfo_t* bli_thrinfo_split
     (
       dim_t      n_way,
       thrinfo_t* thread_par
     )
{
	      thrcomm_t* parent_comm        = bli_thrinfo_comm( thread_par );
	const timpl_t    ti                 = bli_thrcomm_thread_impl( parent_comm );
	const dim_t      parent_num_threads = bli_thrinfo_num_threads( thread_par );
	const dim_t      parent_thread_id   = bli_thrinfo_thread_id( thread_par );
	      pool_t*    sba_pool           = bli_thrinfo_sba_pool( thread_par );
	      pba_t*     pba                = bli_thrinfo_pba( thread_par );

	// Sanity check: make sure the number of threads in the parent's
	// communicator is divisible by the number of new sub-groups.
	if ( parent_num_threads % n_way != 0 )
	{
		printf( "Assertion failed: parent_num_threads %% n_way != 0\n" );
		bli_abort();
	}

	// Compute:
	// - the number of threads inside the new child comm,
	// - the current thread's id within the new communicator,
	// - the current thread's work id, given the ways of parallelism
	//   to be obtained within the next loop.
	const dim_t child_num_threads = parent_num_threads / n_way;
	const dim_t child_thread_id   = parent_thread_id % child_num_threads;
	const dim_t child_work_id     = parent_thread_id / child_num_threads;

	thrcomm_t*  static_comms[ BLIS_NUM_STATIC_COMMS ];
	thrcomm_t** new_comms = NULL;
	thrcomm_t*  my_comm = NULL;
	bool        free_comm = FALSE;

	if ( n_way == 1 )
	{
		my_comm = parent_comm;
	}
	else if ( n_way == parent_num_threads )
	{
		my_comm = &BLIS_SINGLE_COMM;
	}
	else
	{
		// The parent's chief thread creates a temporary array of thrcomm_t
		// pointers.
		if ( bli_thrinfo_am_chief( thread_par ) )
		{
			err_t r_val;

			if ( n_way > BLIS_NUM_STATIC_COMMS )
				new_comms = bli_malloc_intl( n_way * sizeof( thrcomm_t* ), &r_val );
			else
				new_comms = static_comms;
		}

		// Broadcast the temporary array to all threads in the parent's
		// communicator.
		new_comms = bli_thrinfo_broadcast( thread_par, new_comms );

		// Chiefs in the child communicator allocate the communicator
		// object and store it in the array element corresponding to the
		// parent's work id.
		if ( child_thread_id == 0 )
			new_comms[ child_work_id ] = bli_thrcomm_create( ti, sba_pool, child_num_threads );

		bli_thrinfo_barrier( thread_par );

		my_comm = new_comms[ child_work_id ];
		free_comm = TRUE;
	}

	// All threads create a new thrinfo_t node using the communicator
	// that was created by their chief, as identified by parent_work_id.
	thrinfo_t* thread_chl = bli_thrinfo_create
	(
	  my_comm,
	  child_thread_id,
	  n_way,
	  child_work_id,
	  free_comm,
	  sba_pool,
	  pba
	);

	bli_thrinfo_barrier( thread_par );

	// The parent's chief thread frees the temporary array of thrcomm_t
	// pointers.
	if ( bli_thrinfo_am_chief( thread_par ) &&
	     new_comms != static_comms )
	{
		bli_free_intl( new_comms );
	}

	return thread_chl;
}

void bli_thrinfo_print
     (
       thrinfo_t* thread
     )
{
	printf( " lvl   nt  tid nway wkid free\n" );
	bli_thrinfo_print_sub( thread, 0 );
}

void bli_thrinfo_print_sub
     (
       thrinfo_t* thread,
       gint_t     level
     )
{
	if ( thread == NULL ) return;

	printf( "%4ld %4ld %4ld %4ld %4ld %4ld\n",
	        ( unsigned long )level,
	        ( unsigned long )bli_thrinfo_num_threads( thread ),
	        ( unsigned long )bli_thrinfo_thread_id( thread ),
	        ( unsigned long )bli_thrinfo_n_way( thread ),
	        ( unsigned long )bli_thrinfo_work_id( thread ),
	        ( unsigned long )bli_thrinfo_needs_free_comm( thread ));

	for ( dim_t i = 0; i < BLIS_MAX_SUB_NODES; i++ )
		bli_thrinfo_print_sub( bli_thrinfo_sub_node( i, thread ), level+1 );
}

