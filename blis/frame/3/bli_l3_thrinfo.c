/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.
   Copyright (C) 2024, Southern Methodist University

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

thrinfo_t* bli_l3_thrinfo_create
     (
             dim_t       id,
             thrcomm_t*  gl_comm,
             array_t*    array,
       const rntm_t*     rntm,
       const cntl_t*     cntl
     )
{
	pool_t* sba_pool = bli_sba_array_elem( id, array );

	// Create the root thrinfo_t node.
	thrinfo_t* root = bli_thrinfo_create_root
	(
	  gl_comm,
	  id,
	  sba_pool,
	  bli_pba_query()
	);

	bli_l3_thrinfo_grow( root, rntm, cntl );

	return root;
}

void bli_l3_thrinfo_grow
     (
             thrinfo_t*  thread,
       const rntm_t*     rntm,
       const cntl_t*     cntl
     )
{
	// For leaf nodes, create one more node in the thread control tree
	// which splits the remianing threads into single-thread teams.
	if ( bli_cntl_is_leaf( cntl ) )
	{
		dim_t      n_way      = bli_thrinfo_num_threads( thread );
		thrinfo_t* thread_sub = bli_thrinfo_split( n_way, thread );

		bli_thrinfo_attach_sub_node( thread_sub, thread );

		return;
	}

	// Create a thread control tree sub-node for each non-NULL control tree
	// sub-node. Note that the "ways" encoded in the control tree for each
	// sub-node control parallelism for the parent node, not the sub-node. This
	// is necessary because some nodes need to parallelize differently when
	// calling different sub-nodes (e.g. the ic loop of trsm_l).
	for ( dim_t i = 0; i < BLIS_MAX_SUB_NODES; i++ )
	{
		const cntl_t* sub_node = bli_cntl_sub_node( i, cntl );
		if ( sub_node == NULL )
			return;

		dim_t      ways       = bli_cntl_ways( i, cntl );
		dim_t      n_way      = bli_rntm_total_ways_for( ways, rntm );
		thrinfo_t* thread_sub = bli_thrinfo_split( n_way, thread );

		bli_thrinfo_attach_sub_node( thread_sub, thread );
		bli_l3_thrinfo_grow( thread_sub, rntm, sub_node );
	}
}

// -----------------------------------------------------------------------------

thrinfo_t* bli_l3_sup_thrinfo_create
     (
             dim_t      id,
             thrcomm_t* gl_comm,
             pool_t*    sba_pool,
       const rntm_t*    rntm
     )
{
	// Create the root thrinfo_t node.
	thrinfo_t* root = bli_thrinfo_create_root
	(
	  gl_comm,
	  id,
	  sba_pool,
	  bli_pba_query()
	);

	const dim_t n_way_jc = bli_rntm_ways_for( BLIS_NC, rntm );
	const dim_t n_way_pc = bli_rntm_ways_for( BLIS_KC, rntm );
	const dim_t n_way_ic = bli_rntm_ways_for( BLIS_MC, rntm );
	const dim_t n_way_jr = bli_rntm_ways_for( BLIS_NR, rntm );
	const dim_t n_way_ir = bli_rntm_ways_for( BLIS_MR, rntm );

	thrinfo_t* thread_jc = bli_thrinfo_split( n_way_jc,      root );
	thrinfo_t* thread_pc = bli_thrinfo_split( n_way_pc, thread_jc );
	thrinfo_t* thread_pb = bli_thrinfo_split(        1, thread_pc );
	thrinfo_t* thread_ic = bli_thrinfo_split( n_way_ic, thread_pb );
	thrinfo_t* thread_pa = bli_thrinfo_split(        1, thread_ic );
	thrinfo_t* thread_jr = bli_thrinfo_split( n_way_jr, thread_pa );
	thrinfo_t* thread_ir = bli_thrinfo_split( n_way_ir, thread_jr );

	bli_thrinfo_set_sub_node( 0, thread_jc,      root );
	bli_thrinfo_set_sub_node( 0, thread_pc, thread_jc );
	bli_thrinfo_set_sub_node( 0, thread_pb, thread_pc );
	bli_thrinfo_set_sub_node( 0, thread_ic, thread_pb );
	bli_thrinfo_set_sub_node( 0, thread_pa, thread_ic );
	bli_thrinfo_set_sub_node( 0, thread_jr, thread_pa );
	bli_thrinfo_set_sub_node( 0, thread_ir, thread_jr );

	return root;
}

void bli_l3_sup_thrinfo_update
     (
       const rntm_t*     rntm,
             thrinfo_t** root
     )
{
	thrcomm_t* gl_comm  = bli_thrinfo_comm( *root );
	dim_t      tid      = bli_thrinfo_thread_id( *root );
	pool_t*    sba_pool = bli_thrinfo_sba_pool( *root );
	dim_t      nt       = bli_thrinfo_num_threads( *root );

	// Return early in single-threaded execution
	// since the thread control tree may not have been
	// allocated normally
	if ( nt == 1 ) return;

	bli_thrinfo_free( *root );
	*root = bli_l3_sup_thrinfo_create( tid, gl_comm, sba_pool, rntm );
}

// -----------------------------------------------------------------------------

void bli_l3_thrinfo_print_gemm_paths
     (
       thrinfo_t** threads
     )
{
	// In order to query the number of threads, we query the only thread we
	// know exists: thread 0.
	dim_t n_threads = bli_thrinfo_num_threads( threads[0] );

	// For the purposes of printing the "header" information that is common
	// to the various instances of a thrinfo_t (ie: across all threads), we
	// choose the last thread in case the problem is so small that there is
	// only an "edge" case, which will always be assigned to the last thread
	// (at least for higher levels of partitioning).
	thrinfo_t* jc_info  = threads[n_threads-1];
	thrinfo_t* pc_info  = NULL;
	thrinfo_t* pb_info  = NULL;
	thrinfo_t* ic_info  = NULL;
	thrinfo_t* pa_info  = NULL;
	thrinfo_t* jr_info  = NULL;
	thrinfo_t* ir_info  = NULL;

	// Initialize the n_ways and n_threads fields of each thrinfo_t "level"
	// to -1. More than likely, these will all be overwritten with meaningful
	// values, but in case some thrinfo_t trees are not fully built (see
	// next commnet), these will be the placeholder values.
	dim_t jc_way = -1, pc_way = -1, pb_way = -1, ic_way = -1,
	                   pa_way = -1, jr_way = -1, ir_way = -1;

	dim_t jc_nt = -1,  pc_nt = -1,  pb_nt = -1,  ic_nt = -1,
	                   pa_nt = -1,  jr_nt = -1,  ir_nt = -1;

	// NOTE: We must check each thrinfo_t pointer for NULLness. Certain threads
	// may not fully build their thrinfo_t structures--specifically when the
	// dimension being parallelized is not large enough for each thread to have
	// even one unit of work (where as unit is usually a single micropanel's
	// width, MR or NR).

	if ( !jc_info ) goto print_header;

	jc_way  = bli_thrinfo_n_way( jc_info );
	jc_nt   = bli_thrinfo_num_threads( jc_info );
	pc_info = bli_thrinfo_sub_node( 0, jc_info );

	if ( !pc_info ) goto print_header;

	pc_way  = bli_thrinfo_n_way( pc_info );
	pc_nt   = bli_thrinfo_num_threads( pc_info );
	pb_info = bli_thrinfo_sub_node( 0, pc_info );

	if ( !pb_info ) goto print_header;

	pb_way  = bli_thrinfo_n_way( pb_info );
	pb_nt   = bli_thrinfo_num_threads( pb_info );
	ic_info = bli_thrinfo_sub_node( 0, pb_info );

	if ( !ic_info ) goto print_header;

	ic_way  = bli_thrinfo_n_way( ic_info );
	ic_nt   = bli_thrinfo_num_threads( ic_info );
	pa_info = bli_thrinfo_sub_node( 0, ic_info );

	if ( !pa_info ) goto print_header;

	pa_way  = bli_thrinfo_n_way( pa_info );
	pa_nt   = bli_thrinfo_num_threads( pa_info );
	jr_info = bli_thrinfo_sub_node( 0, pa_info );

	if ( !jr_info ) goto print_header;

	jr_way  = bli_thrinfo_n_way( jr_info );
	jr_nt   = bli_thrinfo_num_threads( jr_info );
	ir_info = bli_thrinfo_sub_node( 0, jr_info );

	if ( !ir_info ) goto print_header;

	ir_way  = bli_thrinfo_n_way( ir_info );
	ir_nt   = bli_thrinfo_num_threads( ir_info );

	print_header:

	printf( "            jc   kc   pb   ic   pa   jr   ir\n" );
	printf( "xx_nt:    %4ld %4ld %4ld %4ld %4ld %4ld %4ld\n",
	        ( unsigned long )jc_nt,
	        ( unsigned long )pc_nt,
	        ( unsigned long )pb_nt,
	        ( unsigned long )ic_nt,
	        ( unsigned long )pa_nt,
	        ( unsigned long )jr_nt,
	        ( unsigned long )ir_nt );
	printf( "xx_way:   %4ld %4ld %4ld %4ld %4ld %4ld %4ld\n",
	        ( unsigned long )jc_way,
	        ( unsigned long )pc_way,
	        ( unsigned long )pb_way,
	        ( unsigned long )ic_way,
	        ( unsigned long )pa_way,
	        ( unsigned long )jr_way,
	        ( unsigned long )ir_way );
	printf( "============================================\n" );

	for ( dim_t gl_id = 0; gl_id < n_threads; ++gl_id )
	{
		jc_info = threads[gl_id];

		dim_t jc_comm_id = -1, pc_comm_id = -1, pb_comm_id = -1, ic_comm_id = -1,
		                       pa_comm_id = -1, jr_comm_id = -1, ir_comm_id = -1;

		dim_t jc_work_id = -1, pc_work_id = -1, pb_work_id = -1, ic_work_id = -1,
		                       pa_work_id = -1, jr_work_id = -1, ir_work_id = -1;

		if ( !jc_info ) goto print_thrinfo;

		jc_comm_id = bli_thrinfo_thread_id( jc_info );
		jc_work_id = bli_thrinfo_work_id( jc_info );
		pc_info    = bli_thrinfo_sub_node( 0, jc_info );

		if ( !pc_info ) goto print_thrinfo;

		pc_comm_id = bli_thrinfo_thread_id( pc_info );
		pc_work_id = bli_thrinfo_work_id( pc_info );
		pb_info    = bli_thrinfo_sub_node( 0, pc_info );

		if ( !pb_info ) goto print_thrinfo;

		pb_comm_id = bli_thrinfo_thread_id( pb_info );
		pb_work_id = bli_thrinfo_work_id( pb_info );
		ic_info    = bli_thrinfo_sub_node( 0, pb_info );

		if ( !ic_info ) goto print_thrinfo;

		ic_comm_id = bli_thrinfo_thread_id( ic_info );
		ic_work_id = bli_thrinfo_work_id( ic_info );
		pa_info    = bli_thrinfo_sub_node( 0, ic_info );

		if ( !pa_info ) goto print_thrinfo;

		pa_comm_id = bli_thrinfo_thread_id( pa_info );
		pa_work_id = bli_thrinfo_work_id( pa_info );
		jr_info    = bli_thrinfo_sub_node( 0, pa_info );

		if ( !jr_info ) goto print_thrinfo;

		jr_comm_id = bli_thrinfo_thread_id( jr_info );
		jr_work_id = bli_thrinfo_work_id( jr_info );
		ir_info    = bli_thrinfo_sub_node( 0, jr_info );

		if ( !ir_info ) goto print_thrinfo;

		ir_comm_id = bli_thrinfo_thread_id( ir_info );
		ir_work_id = bli_thrinfo_work_id( ir_info );

		print_thrinfo:

		printf( "comm ids: %4ld %4ld %4ld %4ld %4ld %4ld %4ld\n",
		        ( long )jc_comm_id,
		        ( long )pc_comm_id,
		        ( long )pb_comm_id,
		        ( long )ic_comm_id,
		        ( long )pa_comm_id,
		        ( long )jr_comm_id,
		        ( long )ir_comm_id );
		printf( "work ids: %4ld %4ld %4ld %4ld %4ld %4ld %4ld\n",
		        ( long )jc_work_id,
		        ( long )pc_work_id,
		        ( long )pb_work_id,
		        ( long )ic_work_id,
		        ( long )pa_work_id,
		        ( long )jr_work_id,
		        ( long )ir_work_id );
		printf( "--------------------------------------------\n" );
	}

}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

void bli_l3_thrinfo_print_trsm_paths
     (
       thrinfo_t** threads
     )
{
	// In order to query the number of threads, we query the only thread we
	// know exists: thread 0.
	dim_t n_threads = bli_thrinfo_num_threads( threads[0] );

	// For the purposes of printing the "header" information that is common
	// to the various instances of a thrinfo_t (ie: across all threads), we
	// choose the last thread in case the problem is so small that there is
	// only an "edge" case, which will always be assigned to the last thread
	// (at least for higher levels of partitioning).
	thrinfo_t* jc_info  = threads[n_threads-1];
	thrinfo_t* pc_info  = NULL;
	thrinfo_t* pb_info  = NULL;
	thrinfo_t* ic_info  = NULL;
	thrinfo_t* pa_info  = NULL; thrinfo_t* pa_info0 = NULL;
	thrinfo_t* jr_info  = NULL; thrinfo_t* jr_info0 = NULL;
	thrinfo_t* ir_info  = NULL; thrinfo_t* ir_info0 = NULL;

	// Initialize the n_ways and n_threads fields of each thrinfo_t "level"
	// to -1. More than likely, these will all be overwritten with meaningful
	// values, but in case some thrinfo_t trees are not fully built (see
	// next commnet), these will be the placeholder values.
	dim_t jc_way = -1, pc_way  = -1, pb_way  = -1, ic_way  = -1,
	                   pa_way  = -1, jr_way  = -1, ir_way  = -1,
	                   pa_way0 = -1, jr_way0 = -1, ir_way0 = -1;

	dim_t jc_nt = -1,  pc_nt   = -1, pb_nt   = -1, ic_nt   = -1,
	                   pa_nt   = -1, jr_nt   = -1, ir_nt   = -1,
	                   pa_nt0  = -1, jr_nt0  = -1, ir_nt0  = -1;

	// NOTE: We must check each thrinfo_t pointer for NULLness. Certain threads
	// may not fully build their thrinfo_t structures--specifically when the
	// dimension being parallelized is not large enough for each thread to have
	// even one unit of work (where as unit is usually a single micropanel's
	// width, MR or NR).

	if ( !jc_info ) goto print_header;

	jc_way   = bli_thrinfo_n_way( jc_info );
	jc_nt    = bli_thrinfo_num_threads( jc_info );
	pc_info  = bli_thrinfo_sub_node( 0, jc_info );

	if ( !pc_info ) goto print_header;

	pc_way   = bli_thrinfo_n_way( pc_info );
	pc_nt    = bli_thrinfo_num_threads( pc_info );
	pb_info  = bli_thrinfo_sub_node( 0, pc_info );

	if ( !pb_info ) goto print_header;

	pb_way   = bli_thrinfo_n_way( pb_info );
	pb_nt    = bli_thrinfo_num_threads( pb_info );
	ic_info  = bli_thrinfo_sub_node( 0, pb_info );

	if ( !ic_info ) goto print_header;

	ic_way   = bli_thrinfo_n_way( ic_info );
	ic_nt    = bli_thrinfo_num_threads( ic_info );
	pa_info  = bli_thrinfo_sub_node( 1, ic_info );
	pa_info0 = bli_thrinfo_sub_node( 0, ic_info );

	// check_header_prenode:

	if ( !pa_info0 ) goto check_header_node;

	pa_way0  = bli_thrinfo_n_way( pa_info0 );
	pa_nt0   = bli_thrinfo_num_threads( pa_info0 );
	jr_info0 = bli_thrinfo_sub_node( 0, pa_info0 );

	if ( !jr_info0 ) goto check_header_node;

	jr_way0  = bli_thrinfo_n_way( jr_info0 );
	jr_nt0   = bli_thrinfo_num_threads( jr_info0 );
	ir_info0 = bli_thrinfo_sub_node( 0, jr_info0 );

	if ( !ir_info0 ) goto check_header_node;

	ir_way0  = bli_thrinfo_n_way( ir_info0 );
	ir_nt0   = bli_thrinfo_num_threads( ir_info0 );

	check_header_node:

	if ( !pa_info ) goto print_header;

	pa_way  = bli_thrinfo_n_way( pa_info );
	pa_nt   = bli_thrinfo_num_threads( pa_info );
	jr_info = bli_thrinfo_sub_node( 0, pa_info );

	if ( !jr_info ) goto print_header;

	jr_way  = bli_thrinfo_n_way( jr_info );
	jr_nt   = bli_thrinfo_num_threads( jr_info );
	ir_info = bli_thrinfo_sub_node( 0, jr_info );

	if ( !ir_info ) goto print_header;

	ir_way  = bli_thrinfo_n_way( ir_info );
	ir_nt   = bli_thrinfo_num_threads( ir_info );

	print_header:

	printf( "            jc   kc   pb   ic     pa     jr     ir\n" );
	printf( "xx_nt:    %4ld %4ld %4ld %4ld  %2ld|%2ld  %2ld|%2ld  %2ld|%2ld\n",
	        ( long )jc_nt,
	        ( long )pc_nt,
	        ( long )pb_nt,
	        ( long )ic_nt,
	        ( long )pa_nt0, ( long )pa_nt,
	        ( long )jr_nt0, ( long )jr_nt,
	        ( long )ir_nt0, ( long )ir_nt );
	printf( "xx_way:   %4ld %4ld %4ld %4ld  %2ld|%2ld  %2ld|%2ld  %2ld|%2ld\n",
	        ( long )jc_way,
	        ( long )pc_way,
	        ( long )pb_way,
	        ( long )ic_way,
	        ( long )pa_way0, ( long )pa_way,
	        ( long )jr_way0, ( long )jr_way,
	        ( long )ir_way0, ( long )ir_way );
	printf( "==================================================\n" );


	for ( dim_t gl_id = 0; gl_id < n_threads; ++gl_id )
	{
		jc_info = threads[gl_id];

		// NOTE: This cpp branch contains code that is safe to execute
		// for small problems that are parallelized enough that one or
		// more threads gets no work.

		dim_t jc_comm_id = -1, pc_comm_id  = -1, pb_comm_id  = -1, ic_comm_id  = -1,
		                       pa_comm_id  = -1, jr_comm_id  = -1, ir_comm_id  = -1,
		                       pa_comm_id0 = -1, jr_comm_id0 = -1, ir_comm_id0 = -1;

		dim_t jc_work_id = -1, pc_work_id  = -1, pb_work_id  = -1, ic_work_id  = -1,
		                       pa_work_id  = -1, jr_work_id  = -1, ir_work_id  = -1,
		                       pa_work_id0 = -1, jr_work_id0 = -1, ir_work_id0 = -1;

		if ( !jc_info ) goto print_thrinfo;

		jc_comm_id = bli_thrinfo_thread_id( jc_info );
		jc_work_id = bli_thrinfo_work_id( jc_info );
		pc_info    = bli_thrinfo_sub_node( 0, jc_info );

		if ( !pc_info ) goto print_thrinfo;

		pc_comm_id = bli_thrinfo_thread_id( pc_info );
		pc_work_id = bli_thrinfo_work_id( pc_info );
		pb_info    = bli_thrinfo_sub_node( 0, pc_info );

		if ( !pb_info ) goto print_thrinfo;

		pb_comm_id = bli_thrinfo_thread_id( pb_info );
		pb_work_id = bli_thrinfo_work_id( pb_info );
		ic_info    = bli_thrinfo_sub_node( 0, pb_info );

		if ( !ic_info ) goto print_thrinfo;

		ic_comm_id = bli_thrinfo_thread_id( ic_info );
		ic_work_id = bli_thrinfo_work_id( ic_info );
		pa_info    = bli_thrinfo_sub_node( 1, ic_info );
		pa_info0   = bli_thrinfo_sub_node( 0, ic_info );

		// check_thrinfo_prenode:

		if ( !pa_info0 ) goto check_thrinfo_node;

		pa_comm_id0 = bli_thrinfo_thread_id( pa_info0 );
		pa_work_id0 = bli_thrinfo_work_id( pa_info0 );
		jr_info0    = bli_thrinfo_sub_node( 0, pa_info0 );

		if ( !jr_info0 ) goto check_thrinfo_node;

		jr_comm_id0 = bli_thrinfo_thread_id( jr_info0 );
		jr_work_id0 = bli_thrinfo_work_id( jr_info0 );
		ir_info0    = bli_thrinfo_sub_node( 0, jr_info0 );

		if ( !ir_info0 ) goto check_thrinfo_node;

		ir_comm_id0 = bli_thrinfo_thread_id( ir_info0 );
		ir_work_id0 = bli_thrinfo_work_id( ir_info0 );

		check_thrinfo_node:

		if ( !pa_info ) goto print_thrinfo;

		pa_comm_id = bli_thrinfo_thread_id( pa_info );
		pa_work_id = bli_thrinfo_work_id( pa_info );
		jr_info    = bli_thrinfo_sub_node( 0, pa_info );

		if ( !jr_info ) goto print_thrinfo;

		jr_comm_id = bli_thrinfo_thread_id( jr_info );
		jr_work_id = bli_thrinfo_work_id( jr_info );
		ir_info    = bli_thrinfo_sub_node( 0, jr_info );

		if ( !ir_info ) goto print_thrinfo;

		ir_comm_id = bli_thrinfo_thread_id( ir_info );
		ir_work_id = bli_thrinfo_work_id( ir_info );

		print_thrinfo:

		printf( "comm ids: %4ld %4ld %4ld %4ld  %2ld|%2ld  %2ld|%2ld  %2ld|%2ld\n",
		        ( long )jc_comm_id,
		        ( long )pc_comm_id,
		        ( long )pb_comm_id,
		        ( long )ic_comm_id,
		        ( long )pa_comm_id0, ( long )pa_comm_id,
		        ( long )jr_comm_id0, ( long )jr_comm_id,
		        ( long )ir_comm_id0, ( long )ir_comm_id );
		printf( "work ids: %4ld %4ld %4ld %4ld  %2ld|%2ld  %2ld|%2ld  %2ld|%2ld\n",
		        ( long )jc_work_id,
		        ( long )pc_work_id,
		        ( long )pb_work_id,
		        ( long )ic_work_id,
		        ( long )pa_work_id0, ( long )pa_work_id,
		        ( long )jr_work_id0, ( long )jr_work_id,
		        ( long )ir_work_id0, ( long )ir_work_id );
		printf( "--------------------------------------------------\n" );
	}
}

// -----------------------------------------------------------------------------

void bli_l3_thrinfo_free_paths
     (
       thrinfo_t** threads
     )
{
	dim_t n_threads = bli_thrinfo_num_threads( threads[0] );
	dim_t i;

	for ( i = 0; i < n_threads; ++i )
		bli_thrinfo_free( threads[i] );

	bli_free_intl( threads );
}

