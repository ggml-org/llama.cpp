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

#include "blis.h"

//#define BLIS_ENABLE_MEM_TRACING

void bli_pool_init
     (
       siz_t     num_blocks,
       siz_t     block_ptrs_len,
       siz_t     block_size,
       siz_t     align_size,
       siz_t     offset_size,
       malloc_ft malloc_fp,
       free_ft   free_fp,
       pool_t*   pool
     )
{
	err_t r_val;

	// Make sure that block_ptrs_len is at least num_blocks.
	block_ptrs_len = bli_max( block_ptrs_len, num_blocks );

	// Handle the case where block_ptrs_len is zero, we explicitly set it to 1,
	// to avoid any malloc() with zero size, whose behavior is not fixed, and
	// also to prevent from falling into any further memory corruption bug.
	block_ptrs_len = ( block_ptrs_len == 0 ) ? 1 : block_ptrs_len;

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_pool_init(): allocating block_ptrs (length %d): ",
	        ( int )block_ptrs_len );
	#endif

	// Allocate the block_ptrs array.
	// FGVZ: Do we want to call malloc_fp() for internal data structures as
	// well as pool blocks? If so, don't forget to s/bli_free_intl/free_fp/g.
	pblk_t* block_ptrs
	=
	bli_malloc_intl( block_ptrs_len * sizeof( pblk_t ), &r_val );

	// Allocate and initialize each entry in the block_ptrs array.
	for ( dim_t i = 0; i < num_blocks; ++i )
	{
		#ifdef BLIS_ENABLE_MEM_TRACING
		printf( "bli_pool_init(): allocating block %d of size %d (align %d, offset %d).\n",
		        ( int )i, ( int )block_size, ( int )align_size, ( int )offset_size );
		fflush( stdout );
		#endif

		bli_pool_alloc_block
		(
		  block_size,
		  align_size,
		  offset_size,
		  malloc_fp,
		  &(block_ptrs[i])
		);
	}

	// NOTE: The semantics of top_index approximate a stack, where a "full"
	// stack (no blocks checked out) is one where top_index == 0 and an empty
	// stack (all blocks checked out) one where top_index == num_blocks.
	// (Here, num_blocks tracks the number of blocks currently allocated as
	// part of the pool.) This "orientation" of the stack was chosen
	// intentionally, in contrast to one where top_index == -1 means the
	// stack is empty and top_index = num_blocks - 1 means the stack is
	// full. The chosen scheme allows one to conceptualize the stack as a
	// number line in which blocks are checked out from lowest to highest,
	// and additional blocks are added at the higher end.

	// Initialize the pool_t structure.
	bli_pool_set_block_ptrs( block_ptrs, pool );
	bli_pool_set_block_ptrs_len( block_ptrs_len, pool );
	bli_pool_set_top_index( 0, pool );
	bli_pool_set_num_blocks( num_blocks, pool );
	bli_pool_set_block_size( block_size, pool );
	bli_pool_set_align_size( align_size, pool );
	bli_pool_set_offset_size( offset_size, pool );
	bli_pool_set_malloc_fp( malloc_fp, pool );
	bli_pool_set_free_fp( free_fp, pool );
}

void bli_pool_finalize
     (
       pool_t* pool,
       bool    reinit
     )
{
	// NOTE: This implementation assumes that either:
	// - all blocks have been checked in by all threads, or
	// - some subset of blocks have been checked in and the caller
	//   is bli_pool_reinit().

	// Query the block_ptrs array.
	pblk_t* block_ptrs = bli_pool_block_ptrs( pool );

	// Query the total number of blocks currently allocated.
	const siz_t num_blocks = bli_pool_num_blocks( pool );

	// Query the top_index of the pool.
	const siz_t top_index = bli_pool_top_index( pool );

	// Sanity check: The top_index should be zero.
	// NOTE: This sanity check is disabled when called from bli_pool_reinit()
	// because it is currently implemented in terms of bli_pool_finalize() followed by
	// bli_pool_init(). If that _reinit() takes place when some blocks are
	// checked out, then we would expect top_index != 0, and therefore this
	// check is not universally appropriate.
	if ( top_index != 0 && !reinit )
	{
		printf( "bli_pool_finalize(): final top_index == %d (expected 0); block_size: %d.\n",
		        ( int )top_index, ( int )bli_pool_block_size( pool ) );
		printf( "bli_pool_finalize(): Implication: not all blocks were checked back in!\n" );
		bli_abort();
	}

	// Query the free() function pointer for the pool.
	free_ft free_fp = bli_pool_free_fp( pool );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_pool_finalize(): freeing %d blocks of size %d (align %d, offset %d).\n",
	        ( int )num_blocks, ( int )bli_pool_block_size( pool ),
	                           ( int )bli_pool_align_size( pool ),
	                           ( int )bli_pool_offset_size( pool ) );
	fflush( stdout );
	#endif

	// Query the offset size of the pool.
	const siz_t offset_size = bli_pool_offset_size( pool );

	// Free the individual blocks currently in the pool.
	for ( dim_t i = 0; i < num_blocks; ++i )
	{
		#ifdef BLIS_ENABLE_MEM_TRACING
		printf( "bli_pool_finalize(): block %d: ", ( int )i );
		#endif

		bli_pool_free_block( offset_size, free_fp, &(block_ptrs[i]) );
	}

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_pool_finalize(): freeing block_ptrs (length %d): ",
	        ( int )( bli_pool_block_ptrs_len( pool ) ) );
	#endif

	// Free the block_ptrs array.
	bli_free_intl( block_ptrs );

	// This explicit clearing of the pool_t struct is not strictly
	// necessary and so it has been commented out.
#if 0
	// Clear the contents of the pool_t struct.
	bli_pool_set_block_ptrs( NULL, pool );
	bli_pool_set_block_ptrs_len( 0, pool );
	bli_pool_set_num_blocks( 0, pool );
	bli_pool_set_top_index( 0, pool );
	bli_pool_set_block_size( 0, pool );
	bli_pool_set_align_size( 0, pool );
	bli_pool_set_offset_size( 0, pool );
#endif
}

void bli_pool_reinit
     (
       siz_t   num_blocks_new,
       siz_t   block_ptrs_len_new,
       siz_t   block_size_new,
       siz_t   align_size_new,
       siz_t   offset_size_new,
       pool_t* pool
     )
{
	// Preserve the pointers to malloc() and free() provided when the pool
	// was first initialized.
	malloc_ft malloc_fp = bli_pool_malloc_fp( pool );
	free_ft   free_fp   = bli_pool_free_fp( pool );

	// Finalize the pool as it is currently configured. If some blocks
	// are still checked out to threads, those blocks are not freed
	// here, and instead will be freed when the threads attempt to check
	// those blocks back into the pool. (This condition can be detected
	// since the block size is encoded into each pblk, which is copied
	// upon checkout.)
	bli_pool_finalize( pool, TRUE );

	// Reinitialize the pool with the new parameters, in particular,
	// the new block size.
	bli_pool_init
	(
	  num_blocks_new,
	  block_ptrs_len_new,
	  block_size_new,
	  align_size_new,
	  offset_size_new,
	  malloc_fp,
	  free_fp,
	  pool
	);
}

void bli_pool_checkout_block
     (
       siz_t   req_size,
       pblk_t* block,
       pool_t* pool
     )
{
	// If the requested block size is smaller than what the pool was
	// initialized with, reinitialize the pool to contain blocks of the
	// requested size.
	if ( bli_pool_block_size( pool ) < req_size )
	{
		const siz_t num_blocks_new     = bli_pool_num_blocks( pool );
		const siz_t block_ptrs_len_new = bli_pool_block_ptrs_len( pool );
		const siz_t align_size_new     = bli_pool_align_size( pool );
		const siz_t offset_size_new    = bli_pool_offset_size( pool );

		#ifdef BLIS_ENABLE_MEM_TRACING
		printf( "bli_pool_checkout_block(): old block size %d < req size %d; "
		        "reiniting.\n",
		        ( int )bli_pool_block_size( pool ), ( int )req_size );
		fflush( stdout );
		#endif

		bli_pool_reinit
		(
		  num_blocks_new,
		  block_ptrs_len_new,
		  req_size,
		  align_size_new,
		  offset_size_new,
		  pool
		);
	}

	// If the pool is exhausted, add a block.
	if ( bli_pool_is_exhausted( pool ) )
	{
		#ifdef BLIS_ENABLE_MEM_TRACING
		printf( "bli_pool_checkout_block(): pool is exhausted (block size %d); "
		        "growing by 1.\n", ( int )bli_pool_block_size( pool ) );
		fflush( stdout );
		#endif

		bli_pool_grow( 1, pool );
	}

	// At this point, at least one block is guaranteed to be available.

	// Query the block_ptrs array.
	pblk_t* block_ptrs = bli_pool_block_ptrs( pool );

	// Query the top_index of the pool.
	const siz_t top_index = bli_pool_top_index( pool );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_pool_checkout_block(): checking out block %d of size %d "
	        "(align %d).\n",
	        ( int )top_index, ( int )bli_pool_block_size( pool ),
	                          ( int )bli_pool_align_size( pool ) );
	fflush( stdout );
	#endif

	// Copy the pblk_t at top_index to the caller's pblk_t struct.
	*block = block_ptrs[ top_index ];

	// Notice that we don't actually need to clear the contents of
	// block_ptrs[top_index]. It will get overwritten eventually when
	// the block is checked back in.
	bli_pblk_clear( &block_ptrs[top_index] );

	// Increment the pool's top_index.
	bli_pool_set_top_index( top_index + 1, pool );
}

void bli_pool_checkin_block
     (
       pblk_t* block,
       pool_t* pool
     )
{
	// If the pblk_t being checked in was allocated with a different block
	// size than is currently in use in the pool, we simply free it and
	// return. These "orphaned" blocks are no longer of use because the pool
	// has since been reinitialized to a different (larger) block size.
	if ( bli_pblk_block_size( block ) != bli_pool_block_size( pool ) )
	{
		// Query the offset size of the pool.
		const siz_t offset_size = bli_pool_offset_size( pool );

		// Query the free() function pointer for the pool.
		free_ft free_fp = bli_pool_free_fp( pool );

		bli_pool_free_block( offset_size, free_fp, block );
		return;
	}

	// Query the block_ptrs array.
	pblk_t* block_ptrs = bli_pool_block_ptrs( pool );

	// Query the top_index of the pool.
	const siz_t top_index = bli_pool_top_index( pool );

	// Check for double-free and other conditions which may prematurely
	// exhaust the memory pool.
	if ( top_index == 0 ) bli_abort();

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_pool_checkin_block(): checking in block %d of size %d "
	        "(align %d, offset %d).\n",
	        ( int )top_index - 1, ( int )bli_pool_block_size( pool ),
	                              ( int )bli_pool_align_size( pool ),
	                              ( int )bli_pool_offset_size( pool ) );
	fflush( stdout );
	#endif

	// Copy the caller's pblk_t struct to the block at top_index - 1.
	block_ptrs[ top_index - 1 ] = *block;

	// Decrement the pool's top_index.
	bli_pool_set_top_index( top_index - 1, pool );
}

void bli_pool_grow
     (
       siz_t   num_blocks_add,
       pool_t* pool
     )
{
	err_t r_val;

	// If the requested increase is zero, return early.
	if ( num_blocks_add == 0 ) return;

	// Query the allocated length of the block_ptrs array and also the
	// total number of blocks currently allocated.
	const siz_t block_ptrs_len_cur = bli_pool_block_ptrs_len( pool );
	const siz_t num_blocks_cur     = bli_pool_num_blocks( pool );

	// Compute the total number of allocated blocks that will exist
	// after we grow the pool.
	const siz_t num_blocks_new = num_blocks_cur + num_blocks_add;

	// If adding num_blocks_add new blocks will exceed the current capacity
	// of the block_ptrs array, we need to first put in place a new (larger)
	// array.
	if ( block_ptrs_len_cur < num_blocks_new )
	{
		// To prevent this from happening often, we double the current
		// length of the block_ptrs array.
		// Sanity: make sure that the block_ptrs_len_new will be at least
		// num_blocks_new, in case doubling the block_ptrs_len_cur is not enough.
		// Example 1:
		// - block_ptrs_len_cur == num_blocks_cur == 0 and num_blocks_add = 1
		// - So doubling: 2 * block_ptrs_len_cur = 0, whereas 1 is expected
		// Example 2:
		// - block_ptrs_len_cur == num_blocks_cur == 10 and num_blocks_add = 30
		// - So doubling: 2 * block_ptrs_len_cur = 20, whereas 40 is expected
		const siz_t block_ptrs_len_new = bli_max( (2 * block_ptrs_len_cur), num_blocks_new );

		#ifdef BLIS_ENABLE_MEM_TRACING
		printf( "bli_pool_grow(): growing block_ptrs_len (%d -> %d): ",
		        ( int )block_ptrs_len_cur, ( int )block_ptrs_len_new );
		#endif

		// Query the current block_ptrs array.
		pblk_t* block_ptrs_cur = bli_pool_block_ptrs( pool );

		// Allocate a new block_ptrs array.
		// FGVZ: Do we want to call malloc_fp() for internal data structures as
		// well as pool blocks? If so, don't forget to s/bli_free_intl/free_fp/g.
		pblk_t* block_ptrs_new
		=
		bli_malloc_intl( block_ptrs_len_new * sizeof( pblk_t ), &r_val );

		// Copy the contents of the old block_ptrs array to the new/resized
		// array. Notice that we copy the entire array, including elements
		// corresponding to blocks that have been checked out. Those elements
		// were set to NULL upon checkout, and so it's important to copy them
		// into the new block_ptrs array.
		for ( dim_t i = 0; i < num_blocks_cur; ++i )
		{
			block_ptrs_new[i] = block_ptrs_cur[i];
		}

		#ifdef BLIS_ENABLE_MEM_TRACING
		printf( "bli_pool_grow(): freeing prev block_ptrs: " );
		#endif

		// Free the old block_ptrs array.
		bli_free_intl( block_ptrs_cur );

		// Update the pool_t struct with the new block_ptrs array and
		// record its allocated length.
		bli_pool_set_block_ptrs( block_ptrs_new, pool );
		bli_pool_set_block_ptrs_len( block_ptrs_len_new, pool );
	}

	// At this point, we are guaranteed to have enough unused elements
	// in the block_ptrs array to accommodate an additional num_blocks_add
	// blocks.

	// Query the current block_ptrs array (which was mabye just resized).
	pblk_t* block_ptrs = bli_pool_block_ptrs( pool );

	// Query the block size and alignment size of the pool.
	const siz_t block_size  = bli_pool_block_size( pool );
	const siz_t align_size  = bli_pool_align_size( pool );
	const siz_t offset_size = bli_pool_offset_size( pool );

	// Query the malloc() function pointer for the pool.
	malloc_ft malloc_fp = bli_pool_malloc_fp( pool );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_pool_grow(): growing pool from (%d -> %d).\n",
	        ( int )num_blocks_cur, ( int )num_blocks_new );
	fflush( stdout );
	#endif

	// Allocate the requested additional blocks in the resized array.
	for ( dim_t i = num_blocks_cur; i < num_blocks_new; ++i )
	{
		bli_pool_alloc_block
		(
		  block_size,
		  align_size,
		  offset_size,
		  malloc_fp,
		  &(block_ptrs[i])
		);
	}

	// Update the pool_t struct with the new number of allocated blocks.
	// Notice that top_index remains unchanged, as do the block_size and
	// align_size fields.
	bli_pool_set_num_blocks( num_blocks_new, pool );
}

void bli_pool_shrink
     (
       siz_t   num_blocks_sub,
       pool_t* pool
     )
{
	// If the requested decrease is zero, return early.
	if ( num_blocks_sub == 0 ) return;

	// Query the total number of blocks currently allocated.
	const siz_t num_blocks = bli_pool_num_blocks( pool );

	// Query the top_index of the pool.
	const siz_t top_index = bli_pool_top_index( pool );

	// Compute the number of blocks available to be checked out
	// (and thus available for removal).
	const siz_t num_blocks_avail = num_blocks - top_index;

	// If the requested decrease is more than the number of available
	// blocks in the pool, only remove the number of blocks actually
	// available.
	num_blocks_sub = bli_min( num_blocks_sub, num_blocks_avail );

	// Query the block_ptrs array.
	pblk_t* block_ptrs = bli_pool_block_ptrs( pool );

	// Compute the new total number of blocks.
	const siz_t num_blocks_new = num_blocks - num_blocks_sub;

	// Query the offset size of the pool.
	const siz_t offset_size = bli_pool_offset_size( pool );

	// Query the free() function pointer for the pool.
	free_ft free_fp = bli_pool_free_fp( pool );

	// Free the individual blocks.
	for ( dim_t i = num_blocks_new; i < num_blocks; ++i )
	{
		bli_pool_free_block( offset_size, free_fp, &(block_ptrs[i]) );
	}

	// Update the pool_t struct.
	bli_pool_set_num_blocks( num_blocks_new, pool );

	// Note that after shrinking the pool, num_blocks < block_ptrs_len.
	// This means the pool can grow again by num_blocks_sub before
	// a re-allocation of block_ptrs is triggered.
}

void bli_pool_alloc_block
     (
       siz_t     block_size,
       siz_t     align_size,
       siz_t     offset_size,
       malloc_ft malloc_fp,
       pblk_t*   block
     )
{
	err_t r_val;

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_pool_alloc_block(): calling fmalloc_align(): size %d (align %d, offset %d)\n",
	        ( int )block_size, ( int )align_size, ( int )offset_size );
	fflush( stdout );
	#endif

	// Allocate the block via the bli_fmalloc_align() wrapper, which performs
	// alignment logic and opaquely saves the original pointer so that it can
	// be recovered when it's time to free the block. Note that we have to
	// add offset_size to the number of bytes requested since we will skip
	// that many bytes at the beginning of the allocated memory.
	void* buf
	=
	bli_fmalloc_align( malloc_fp, block_size + offset_size, align_size, &r_val );

#if 0
	// NOTE: This code is disabled because it is not needed, since
	// bli_fmalloc_align() is guaranteed to return an aligned address.

	// Advance the pointer to achieve the necessary alignment, if it is not
	// already aligned.
	if ( bli_is_unaligned_to( ( siz_t )buf_sys, ( siz_t )align_size ) )
	{
		// C99's stdint.h guarantees that a void* can be safely cast to a
		// uintptr_t and then back to a void*, hence the casting of buf_sys
		// and align_size to uintptr_t. buf_align is initially cast to char*
		// to allow pointer arithmetic in units of bytes, and then advanced
		// to the next nearest alignment boundary, and finally cast back to
		// void* before being stored. Notice that the arithmetic works even
		// if the alignment value is not a power of two.
		buf_align = ( void* )(   ( char*     )buf_align +
		                       ( ( uintptr_t )align_size -
		                         ( uintptr_t )buf_sys %
		                         ( uintptr_t )align_size )
		                     );
	}
#endif

	// Advance the pointer by offset_size bytes.
	buf = ( void* )( ( char* )buf + offset_size );

	// Save the results in the pblk_t structure.
	bli_pblk_set_buf( buf, block );
	bli_pblk_set_block_size( block_size, block );
}

void bli_pool_free_block
     (
       siz_t            offset_size,
       free_ft          free_fp,
       pblk_t* block
     )
{
	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_pool_free_block(): calling ffree_align(): size %d.\n",
	        ( int )bli_pblk_block_size( block ) );
	fflush( stdout );
	#endif

	// Extract the pblk_t buffer, which is the aligned address returned from
	// bli_fmalloc_align() when the block was allocated.
	void* buf = bli_pblk_buf( block );

	// Undo the pointer advancement by offset_size bytes performed previously
	// by bli_pool_alloc_block().
	buf = ( void* )( ( char* )buf - offset_size );

	// Free the block via the bli_ffree_align() wrapper, which recovers the
	// original pointer that was returned by the pool's malloc() function when
	// the block was allocated.
	bli_ffree_align( free_fp, buf );
}

void bli_pool_print
     (
       const pool_t* pool
     )
{
	pblk_t* block_ptrs     = bli_pool_block_ptrs( pool );
	siz_t   block_ptrs_len = bli_pool_block_ptrs_len( pool );
	siz_t   top_index      = bli_pool_top_index( pool );
	siz_t   num_blocks     = bli_pool_num_blocks( pool );
	siz_t   block_size     = bli_pool_block_size( pool );
	siz_t   align_size     = bli_pool_align_size( pool );
	siz_t   offset_size    = bli_pool_offset_size( pool );

	printf( "pool struct ---------------\n" );
	printf( "  block_ptrs:      %p\n", block_ptrs );
	printf( "  block_ptrs_len:  %d\n", ( int )block_ptrs_len );
	printf( "  top_index:       %d\n", ( int )top_index );
	printf( "  num_blocks:      %d\n", ( int )num_blocks );
	printf( "  block_size:      %d\n", ( int )block_size );
	printf( "  align_size:      %d\n", ( int )align_size );
	printf( "  offset_size:     %d\n", ( int )offset_size );
	printf( "  pblks   sys    align\n" );

	for ( dim_t i = 0; i < num_blocks; ++i )
	{
		printf( "  %d: %p\n", ( int )i, bli_pblk_buf( &block_ptrs[i] ) );
	}
}

void bli_pblk_print
     (
       const pblk_t* pblk
     )
{
	void* buf = bli_pblk_buf( pblk );

	printf( "pblk struct ---------------\n" );
	printf( "  block address (aligned): %p\n", buf );
}

