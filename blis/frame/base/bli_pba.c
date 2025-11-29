/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2016, Hewlett Packard Enterprise Development LP
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

// Statically initialize the mutex within the packing block allocator object.
static pba_t global_pba = { .mutex = BLIS_PTHREAD_MUTEX_INITIALIZER };

// -----------------------------------------------------------------------------

pba_t* bli_pba_query( void )
{
    return &global_pba;
}

void bli_pba_init
     (
       const cntx_t* cntx
     )
{
	pba_t* pba = bli_pba_query();

	const siz_t align_size = BLIS_POOL_ADDR_ALIGN_SIZE_GEN;
	malloc_ft   malloc_fp  = BLIS_MALLOC_POOL;
	free_ft     free_fp    = BLIS_FREE_POOL;

	// These fields are used for general-purpose allocation (ie: buf_type
	// equal to BLIS_BUFFER_FOR_GEN_USE) within bli_pba_acquire_m().
	bli_pba_set_align_size( align_size, pba );
	bli_pba_set_malloc_fp( malloc_fp, pba );
	bli_pba_set_free_fp( free_fp, pba );

	// The mutex field of pba is initialized statically above. This
	// keeps bli_pba_init() simpler and removes the possibility of
	// something going wrong during mutex initialization.

#ifdef BLIS_ENABLE_PBA_POOLS
	bli_pba_init_pools( cntx, pba );
#endif
}

void bli_pba_finalize
     (
       void
     )
{
	pba_t* pba = bli_pba_query();

#ifdef BLIS_ENABLE_PBA_POOLS
	bli_pba_finalize_pools( pba );
#endif

	// The mutex field of pba is initialized statically above, and
	// therefore never destroyed.

	bli_pba_set_malloc_fp( NULL, pba );
	bli_pba_set_free_fp( NULL, pba );
}

void bli_pba_acquire_m
     (
       pba_t*    pba,
       siz_t     req_size,
       packbuf_t buf_type,
       mem_t*    mem
     )
{
	// If the internal memory pools for packing block allocator are disabled,
	// we spoof the buffer type as BLIS_BUFFER_FOR_GEN_USE to induce the
	// immediate usage of bli_pba_malloc().
#ifndef BLIS_ENABLE_PBA_POOLS
	buf_type = BLIS_BUFFER_FOR_GEN_USE;

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_pba_acquire_m(): bli_fmalloc_align(): size %ld\n",
	        ( long )req_size );
	#endif
#endif

	if ( buf_type == BLIS_BUFFER_FOR_GEN_USE )
	{
		malloc_ft malloc_fp  = bli_pba_malloc_fp( pba );
		siz_t     align_size = bli_pba_align_size( pba );

		// For general-use buffer requests, dynamically allocating memory
		// is assumed to be sufficient.
		err_t r_val;
		void* buf = bli_fmalloc_align( malloc_fp, req_size, align_size, &r_val );

		// Initialize the mem_t object with:
		// - the address of the memory block,
		// - the buffer type (a packbuf_t value),
		// - the size of the requested region,
		// - the pba_t from which the mem_t entry was acquired.
		// NOTE: We initialize the pool field to NULL since this block did not
		// come from a memory pool.
		bli_mem_set_buffer( buf, mem );
		bli_mem_set_buf_type( buf_type, mem );
		bli_mem_set_pool( NULL, mem );
		bli_mem_set_size( req_size, mem );
	}
	else
	{
		// This branch handles cases where the memory block needs to come
		// from an internal memory pool, in which blocks are allocated once
		// and then recycled.

		// Map the requested packed buffer type to a zero-based index, which
		// we then use to select the corresponding memory pool.
		dim_t   pi   = bli_packbuf_index( buf_type );
		pool_t* pool = bli_pba_pool( pi, pba );

		// Extract the address of the pblk_t struct within the mem_t.
		pblk_t* pblk = bli_mem_pblk( mem );

		// Acquire the mutex associated with the pba object.
		bli_pba_lock( pba );

		// BEGIN CRITICAL SECTION
		{

			// Checkout a block from the pool. If the pool's blocks are too
			// small, it will be reinitialized with blocks large enough to
			// accommodate the requested block size. If the pool is exhausted,
			// either because it is still empty or because all blocks have
			// been checked out already, additional blocks will be allocated
			// automatically, as-needed. Note that the addresses are stored
			// directly into the mem_t struct since pblk is the address of
			// the struct's pblk_t field.
			bli_pool_checkout_block( req_size, pblk, pool );

		}
		// END CRITICAL SECTION

		// Release the mutex associated with the pba object.
		bli_pba_unlock( pba );

		// Query the block_size from the pblk_t. This will be at least
		// req_size, perhaps larger.
		siz_t block_size = bli_pblk_block_size( pblk );

		// Initialize the mem_t object with:
		// - the buffer type (a packbuf_t value),
		// - the address of the memory pool to which it belongs,
		// - the size of the contiguous memory block (NOT the size of the
		//   requested region),
		// - the pba_t from which the mem_t entry was acquired.
		// The actual (aligned) address is already stored in the mem_t
		// struct's pblk_t field.
		bli_mem_set_buf_type( buf_type, mem );
		bli_mem_set_pool( pool, mem );
		bli_mem_set_size( block_size, mem );
	}
}


void bli_pba_release
     (
       pba_t* pba,
       mem_t* mem
     )
{
	// Extract the buffer type so we know what kind of memory was allocated.
	packbuf_t buf_type = bli_mem_buf_type( mem );

#ifndef BLIS_ENABLE_PBA_POOLS
	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_pba_release(): bli_ffree_align(): size %ld\n",
	        ( long )bli_mem_size( mem ) );
	#endif
#endif

	if ( buf_type == BLIS_BUFFER_FOR_GEN_USE )
	{
		free_ft free_fp = bli_pba_free_fp( pba );
		void*   buf     = bli_mem_buffer( mem );

		// For general-use buffers, we dynamically allocate memory, and so
		// here we need to free it.
		bli_ffree_align( free_fp, buf );
	}
	else
	{
		// Extract the address of the pool from which the memory was
		// allocated.
		pool_t* pool = bli_mem_pool( mem );

		// Extract the address of the pblk_t struct within the mem_t struct.
		pblk_t* pblk = bli_mem_pblk( mem );

		// Acquire the mutex associated with the pba object.
		bli_pba_lock( pba );

		// BEGIN CRITICAL SECTION
		{

			// Check the block back into the pool.
			bli_pool_checkin_block( pblk, pool );

		}
		// END CRITICAL SECTION

		// Release the mutex associated with the pba object.
		bli_pba_unlock( pba );
	}

	// Clear the mem_t object so that it appears unallocated. This clears:
	// - the pblk_t struct's fields (ie: the buffer addresses)
	// - the pool field
	// - the size field
	// - the pba field
	// NOTE: We do not clear the buf_type field since there is no
	// "uninitialized" value for packbuf_t.
	bli_mem_clear( mem );
}


#if 0
void bli_pba_acquire_v
     (
       pba_t* pba,
       siz_t  req_size,
       mem_t* mem
     )
{
	bli_pba_acquire_m
	(
	  pba,
	  req_size,
	  BLIS_BUFFER_FOR_GEN_USE,
	  mem
	);
}
#endif


siz_t bli_pba_pool_size
     (
       const pba_t*    pba,
             packbuf_t buf_type
     )
{
	siz_t r_val;

	if ( buf_type == BLIS_BUFFER_FOR_GEN_USE )
	{
		// We don't (yet) track the amount of general-purpose
		// memory that is currently allocated.
		r_val = 0;
	}
	else
	{
		dim_t   pool_index;
		pool_t* pool;

		// Acquire the pointer to the pool corresponding to the buf_type
		// provided.
		pool_index = bli_packbuf_index( buf_type );
		pool       = bli_pba_pool( pool_index, ( pba_t* )pba );

		// Compute the pool "size" as the product of the block size
		// and the number of blocks in the pool.
		r_val = bli_pool_block_size( pool ) *
		        bli_pool_num_blocks( pool );
	}

	return r_val;
}

// -----------------------------------------------------------------------------

void bli_pba_init_pools
     (
       const cntx_t* cntx,
             pba_t*  pba
     )
{
	// Map each of the packbuf_t values to an index starting at zero.
	const dim_t index_a      = bli_packbuf_index( BLIS_BUFFER_FOR_A_BLOCK );
	const dim_t index_b      = bli_packbuf_index( BLIS_BUFFER_FOR_B_PANEL );
	const dim_t index_c      = bli_packbuf_index( BLIS_BUFFER_FOR_C_PANEL );

	// Alias the pool addresses to convenient identifiers.
	pool_t*     pool_a       = bli_pba_pool( index_a, pba );
	pool_t*     pool_b       = bli_pba_pool( index_b, pba );
	pool_t*     pool_c       = bli_pba_pool( index_c, pba );

	// Start with empty pools.
	const dim_t num_blocks_a = 0;
	const dim_t num_blocks_b = 0;
	const dim_t num_blocks_c = 0;

	siz_t       block_size_a = 0;
	siz_t       block_size_b = 0;
	siz_t       block_size_c = 0;

	// For blocks of A and panels of B, start off with block_ptrs arrays that
	// are of a decent length. For C, we can start off with an empty array.
	const dim_t block_ptrs_len_a = 80;
	const dim_t block_ptrs_len_b = 80;
	const dim_t block_ptrs_len_c = 0;

	// Use the address alignment sizes designated (at configure-time) for pools.
	const siz_t align_size_a = BLIS_POOL_ADDR_ALIGN_SIZE_A;
	const siz_t align_size_b = BLIS_POOL_ADDR_ALIGN_SIZE_B;
	const siz_t align_size_c = BLIS_POOL_ADDR_ALIGN_SIZE_C;

	// Use the offsets from the above alignments.
	const siz_t offset_size_a = BLIS_POOL_ADDR_OFFSET_SIZE_A;
	const siz_t offset_size_b = BLIS_POOL_ADDR_OFFSET_SIZE_B;
	const siz_t offset_size_c = BLIS_POOL_ADDR_OFFSET_SIZE_C;

	// Use the malloc() and free() designated (at configure-time) for pools.
	malloc_ft malloc_fp  = BLIS_MALLOC_POOL;
	free_ft   free_fp    = BLIS_FREE_POOL;

	// Determine the block size for each memory pool.
	bli_pba_compute_pool_block_sizes( &block_size_a,
	                                  &block_size_b,
	                                  &block_size_c,
	                                  cntx );

	// Initialize the memory pools for A, B, and C.
	bli_pool_init( num_blocks_a, block_ptrs_len_a, block_size_a, align_size_a,
	               offset_size_a, malloc_fp, free_fp, pool_a );
	bli_pool_init( num_blocks_b, block_ptrs_len_b, block_size_b, align_size_b,
	               offset_size_b, malloc_fp, free_fp, pool_b );
	bli_pool_init( num_blocks_c, block_ptrs_len_c, block_size_c, align_size_c,
	               offset_size_c, malloc_fp, free_fp, pool_c );
}

void bli_pba_finalize_pools
     (
       pba_t* pba
     )
{
	// Map each of the packbuf_t values to an index starting at zero.
	dim_t   index_a = bli_packbuf_index( BLIS_BUFFER_FOR_A_BLOCK );
	dim_t   index_b = bli_packbuf_index( BLIS_BUFFER_FOR_B_PANEL );
	dim_t   index_c = bli_packbuf_index( BLIS_BUFFER_FOR_C_PANEL );

	// Alias the pool addresses to convenient identifiers.
	pool_t* pool_a  = bli_pba_pool( index_a, pba );
	pool_t* pool_b  = bli_pba_pool( index_b, pba );
	pool_t* pool_c  = bli_pba_pool( index_c, pba );

	// Finalize the memory pools for A, B, and C.
	bli_pool_finalize( pool_a, FALSE );
	bli_pool_finalize( pool_b, FALSE );
	bli_pool_finalize( pool_c, FALSE );
}

// -----------------------------------------------------------------------------

void bli_pba_compute_pool_block_sizes
     (
             siz_t*  bs_a,
             siz_t*  bs_b,
             siz_t*  bs_c,
       const cntx_t* cntx
     )
{
	siz_t bs_cand_a = 0;
	siz_t bs_cand_b = 0;
	siz_t bs_cand_c = 0;

	// Compute pool block sizes for each datatype and find the maximum
	// size for each pool. This is done so that new pools do not need
	// to be allocated if the user switches datatypes.
	for ( num_t dt = BLIS_DT_LO; dt <= BLIS_DT_HI; ++dt )
	{
		siz_t bs_dt_a;
		siz_t bs_dt_b;
		siz_t bs_dt_c;

		bli_pba_compute_pool_block_sizes_dt( dt,
		                                     &bs_dt_a,
		                                     &bs_dt_b,
		                                     &bs_dt_c,
		                                     cntx );

		bs_cand_a = bli_max( bs_dt_a, bs_cand_a );
		bs_cand_b = bli_max( bs_dt_b, bs_cand_b );
		bs_cand_c = bli_max( bs_dt_c, bs_cand_c );
	}

	// Save the results.
	*bs_a = bs_cand_a;
	*bs_b = bs_cand_b;
	*bs_c = bs_cand_c;
}

// -----------------------------------------------------------------------------

void bli_pba_compute_pool_block_sizes_dt
     (
             num_t   dt,
             siz_t*  bs_a,
             siz_t*  bs_b,
             siz_t*  bs_c,
       const cntx_t* cntx
     )
{
	//
	// Find the larger of the two register blocksizes.
	//

	// Query the mr and nr blksz_t objects for the given method of
	// execution.
	const blksz_t* mr = bli_cntx_get_blksz( BLIS_MR, cntx );
	const blksz_t* nr = bli_cntx_get_blksz( BLIS_NR, cntx );

	// Extract the mr and nr values specific to the current datatype.
	dim_t mr_dt = bli_blksz_get_def( dt, mr );
	dim_t nr_dt = bli_blksz_get_def( dt, nr );

	// Find the maximum of mr and nr.
	dim_t max_mnr_dt = bli_max( mr_dt, nr_dt );

	//
	// Define local maximum cache blocksizes.
	//

	// Query the mc, kc, and nc blksz_t objects for native execution.
	const blksz_t* mc = bli_cntx_get_blksz( BLIS_MC, cntx );
	const blksz_t* kc = bli_cntx_get_blksz( BLIS_KC, cntx );
	const blksz_t* nc = bli_cntx_get_blksz( BLIS_NC, cntx );

	// Extract the maximum mc, kc, and nc values specific to the current
	// datatype.
	dim_t mc_max_dt = bli_blksz_get_max( dt, mc );
	dim_t kc_max_dt = bli_blksz_get_max( dt, kc );
	dim_t nc_max_dt = bli_blksz_get_max( dt, nc );

	// Add max(mr,nr) to kc to make room for the nudging of kc at
	// runtime to be a multiple of mr or nr for triangular operations
	// trmm, trmm3, and trsm.
	kc_max_dt += max_mnr_dt;

	//
	// Compute scaling factors.
	//

	// Compute integer scaling factors (numerator and denominator) used
	// to account for situations when the packing register blocksizes are
	// larger than the regular register blocksizes.

	// In order to compute the scaling factors, we first have to determine
	// whether ( packmr / mr ) is greater than ( packnr / nr ). This is
	// needed ONLY because the amount of space allocated for a block of A
	// and a panel of B needs to be such that MR and NR can be swapped (ie:
	// A is packed with NR and B is packed with MR). This transformation is
	// needed for right-side trsm when inducing an algorithm that (a) has
	// favorable access patterns for column-stored C and (b) allows the
	// macro-kernel to reuse the existing left-side fused gemmtrsm micro-
	// kernels. We avoid integer division by cross-multiplying:
	//
	//   ( packmr / mr )      >= ( packnr / nr )
	//   ( packmr / mr ) * nr >=   packnr
	//     packmr * nr        >=   packnr * mr
	//
	// So, if packmr * nr >= packnr * mr, then we will use packmr and mr as
	// our scaling factors. Otherwise, we'll use packnr and nr.

	dim_t packmr_dt = bli_blksz_get_max( dt, mr );
	dim_t packnr_dt = bli_blksz_get_max( dt, nr );

	dim_t scale_num_dt;
	dim_t scale_den_dt;

	if ( packmr_dt * nr_dt >=
	     packnr_dt * mr_dt ) { scale_num_dt = packmr_dt;
	                           scale_den_dt =     mr_dt; }
	else                     { scale_num_dt = packnr_dt;
	                           scale_den_dt =     nr_dt; }

	//
	// Compute pool block dimensions.
	//

	dim_t pool_mc_dt = ( mc_max_dt * scale_num_dt ) / scale_den_dt;
	dim_t left_mc_dt = ( mc_max_dt * scale_num_dt ) % scale_den_dt;

	dim_t pool_nc_dt = ( nc_max_dt * scale_num_dt ) / scale_den_dt;
	dim_t left_nc_dt = ( nc_max_dt * scale_num_dt ) % scale_den_dt;

	dim_t pool_kc_dt = ( kc_max_dt );

	if ( left_mc_dt > 0 ) pool_mc_dt += 1;
	if ( left_nc_dt > 0 ) pool_nc_dt += 1;

	//
	// Compute pool block sizes
	//

	siz_t size_dt = bli_dt_size( dt );

	// We add an extra micro-panel of space to the block sizes for A and B
	// just to be sure any pre-loading performed by the micro-kernel does
	// not cause a segmentation fault.
	dim_t max_packmnr_dt = bli_max( packmr_dt, packnr_dt );

	*bs_a = ( pool_mc_dt + max_packmnr_dt ) * pool_kc_dt * size_dt;
	*bs_b = ( pool_nc_dt + max_packmnr_dt ) * pool_kc_dt * size_dt;
	*bs_c = ( pool_mc_dt                  ) * pool_nc_dt * size_dt;
}
