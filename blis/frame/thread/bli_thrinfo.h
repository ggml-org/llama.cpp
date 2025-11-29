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

#ifndef BLIS_THRINFO_H
#define BLIS_THRINFO_H

// Thread info structure definition
struct thrinfo_s
{
	// The thread communicator for the other threads sharing the same work
	// at this level.
	thrcomm_t*         comm;

	// Our thread id within the thread communicator.
	dim_t              thread_id;

	// The number of communicators which are "siblings" of our communicator.
	dim_t              n_way;

	// An id to identify what we're working on. This is the same for all threads
	// in the same communicator, and 0 <= work_id < n_way.
	dim_t              work_id;

	// When freeing, should the communicators in this node be freed? Usually,
	// this is field is true, but when nodes are created that share the same
	// communicators as other nodes (such as with packm nodes), this is set
	// to false.
	bool               free_comm;

	// The small block pool.
	pool_t*            sba_pool;

	// The packing block allocator.
	pba_t*             pba;

	// Storage for allocated memory obtained from the packing block allocator.
	mem_t              mem;

	// Child thread info nodes.
	struct thrinfo_s*  sub_nodes[ BLIS_MAX_SUB_NODES ];
};
typedef struct thrinfo_s thrinfo_t;

//
// thrinfo_t functions
//

// thrinfo_t query (field only)

BLIS_INLINE dim_t bli_thrinfo_num_threads( const thrinfo_t* t )
{
	return (t->comm)->n_threads;
}

BLIS_INLINE dim_t bli_thrinfo_thread_id( const thrinfo_t* t )
{
	return t->thread_id;
}

BLIS_INLINE dim_t bli_thrinfo_n_way( const thrinfo_t* t )
{
	return t->n_way;
}

BLIS_INLINE dim_t bli_thrinfo_work_id( const thrinfo_t* t )
{
	return t->work_id;
}

BLIS_INLINE thrcomm_t* bli_thrinfo_comm( const thrinfo_t* t )
{
	return t->comm;
}

BLIS_INLINE bool bli_thrinfo_needs_free_comm( const thrinfo_t* t )
{
	return t->free_comm;
}

BLIS_INLINE pool_t* bli_thrinfo_sba_pool( const thrinfo_t* t )
{
	return t->sba_pool;
}

BLIS_INLINE pba_t* bli_thrinfo_pba( const thrinfo_t* t )
{
	return t->pba;
}

BLIS_INLINE mem_t* bli_thrinfo_mem( thrinfo_t* t )
{
	return &t->mem;
}

BLIS_INLINE thrinfo_t* bli_thrinfo_sub_node( dim_t which, const thrinfo_t* t )
{
	return t->sub_nodes[ which ];
}

// thrinfo_t query (complex)

BLIS_INLINE bool bli_thrinfo_am_chief( const thrinfo_t* t )
{
	return t->thread_id == 0;
}

// thrinfo_t modification

BLIS_INLINE void bli_thrinfo_set_comm( thrcomm_t* comm, thrinfo_t* t )
{
	t->comm = comm;
}

BLIS_INLINE void bli_thrinfo_set_thread_id( dim_t thread_id, thrinfo_t* t )
{
	t->thread_id = thread_id;
}

BLIS_INLINE void bli_thrinfo_set_n_way( dim_t n_way, thrinfo_t* t )
{
	t->n_way = n_way;
}

BLIS_INLINE void bli_thrinfo_set_work_id( dim_t work_id, thrinfo_t* t )
{
	t->work_id = work_id;
}

BLIS_INLINE void bli_thrinfo_set_free_comm( bool free_comm, thrinfo_t* t )
{
	t->free_comm = free_comm;
}

BLIS_INLINE void bli_thrinfo_set_sba_pool( pool_t* sba_pool, thrinfo_t* t )
{
	t->sba_pool = sba_pool;
}

BLIS_INLINE void bli_thrinfo_set_pba( pba_t* pba, thrinfo_t* t )
{
	t->pba = pba;
}

BLIS_INLINE void bli_thrinfo_set_sub_node( dim_t which, thrinfo_t* sub_node, thrinfo_t* t )
{
	t->sub_nodes[ which ] = sub_node;
}

void bli_thrinfo_attach_sub_node( thrinfo_t* sub_node, thrinfo_t* t );

// other thrinfo_t-related functions

BLIS_INLINE void* bli_thrinfo_broadcast( const thrinfo_t* t, void* p )
{
	return bli_thrcomm_bcast( t->thread_id, p, t->comm );
}

BLIS_INLINE void bli_thrinfo_barrier( const thrinfo_t* t )
{
	bli_thrcomm_barrier( t->thread_id, t->comm );
}


//
// Prototypes for level-3 thrinfo functions not specific to any operation.
//

thrinfo_t* bli_thrinfo_create_root
     (
       thrcomm_t* comm,
       dim_t      thread_id,
       pool_t*    sba_pool,
       pba_t*     pba
     );

thrinfo_t* bli_thrinfo_create
     (
       thrcomm_t* comm,
       dim_t      thread_id,
       dim_t      n_way,
       dim_t      work_id,
       bool       free_comm,
       pool_t*    sba_pool,
       pba_t*     pba
     );

BLIS_EXPORT_BLIS void bli_thrinfo_free
     (
       thrinfo_t* thread
     );

// -----------------------------------------------------------------------------

thrinfo_t* bli_thrinfo_split
     (
       dim_t      n_way,
       thrinfo_t* thread_par
     );

void bli_thrinfo_print
     (
       thrinfo_t* thread
     );

void bli_thrinfo_print_sub
     (
       thrinfo_t* thread,
       gint_t     level
     );

#endif
