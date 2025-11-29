/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

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

void bli_apool_init
     (
       apool_t* apool
     )
{
	err_t r_val;

	// NOTE: The apool_t is only used in one place; it is the type used to
	// define the sba. We've switched to static initialization of the mutex
	// field to remove one more thing that could possibly go wrong during
	// library initialization.

	// Query the mutex from the apool_t.
	//bli_pthread_mutex_t* mutex = bli_apool_mutex( apool );

	// Initialize the mutex.
	//*mutex = BLIS_PTHREAD_MUTEX_INITIALIZER;
	//bli_pthread_mutex_init( mutex, NULL );

	// We choose to start with:
	// - an empty pool
	// - an initial block_ptrs_len of 8
	// - a single element in each initial array_t (though this is moot with
	//   num_blocks = 0).
	const siz_t num_blocks     = 0;
	      siz_t block_ptrs_len = 8;
	const siz_t num_elem       = 1;

	// NOTE: Unlike in the bli_pool API, apool_t allocates block_ptrs as an
	// array of array_t* instead of an array of pblk_t. Why? We don't need to
	// track the size of each block, thus we don't need the block_size field
	// of pblk_t. That leaves only the void* field, and since we know apool_t
	// will always contain "blocks" that are really array_t structs, we can
	// make block_ptrs an array of array_t*.

	// We formally set the block_size and align_size fields of the underlying
	// pool, even though they won't be queried. (They are used from hard-coded
	// values in bli_apool_alloc_block().)
	const siz_t block_size = sizeof( array_t );
	const siz_t align_size = 64;

	// Query the underlying pool_t from the apool_t.
	pool_t* pool = bli_apool_pool( apool );

	// Set the default array_t length of the apool_t.
	bli_apool_set_def_array_len( num_elem, apool );

	// -------------------------------------------------------------------------

	// Make sure that block_ptrs_len is at least num_blocks.
	block_ptrs_len = bli_max( block_ptrs_len, num_blocks );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_apool_init(): allocating block_ptrs (length %d): ",
	        ( int )block_ptrs_len );
	#endif

	// Allocate the block_ptrs array.
	array_t** block_ptrs
	=
	bli_malloc_intl( block_ptrs_len * sizeof( array_t* ), &r_val );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_apool_init(): allocating %d array_t.\n", ( int )num_blocks );
	fflush( stdout );
	#endif

	// Allocate and initialize each entry in the block_ptrs array.
	for ( dim_t i = 0; i < num_blocks; ++i )
	{
		// Pass in num_elem so the function knows how many elements to
		// initially have in each array_t.
		bli_apool_alloc_block
		(
		  num_elem,
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
	// NOTE: We don't use the malloc_fp and free_fp fields at the apool_t
	// level. Nevertheless, we set them to NULL.
	bli_pool_set_block_ptrs( block_ptrs, pool );
	bli_pool_set_block_ptrs_len( block_ptrs_len, pool );
	bli_pool_set_top_index( 0, pool );
	bli_pool_set_num_blocks( num_blocks, pool );
	bli_pool_set_block_size( block_size, pool );
	bli_pool_set_align_size( align_size, pool );
	bli_pool_set_malloc_fp( NULL, pool );
	bli_pool_set_free_fp( NULL, pool );
}

void bli_apool_alloc_block
     (
       siz_t     num_elem,
       array_t** array_p
     )
{
	err_t r_val;

	// Since the apool_t is defined as a pool of array_t, we can hard-code
	// the block_size parameter.
	const siz_t block_size = sizeof( array_t );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_apool_alloc_block(): allocating array_t: " );
	#endif

	// Allocate the array_t via the bli_fmalloc_align() wrapper, which performs
	// alignment logic and opaquely saves the original pointer so that it can
	// be recovered when it's time to free the block.
	array_t* array = bli_malloc_intl( block_size, &r_val );

	// Initialize an array_t struct within the newly allocated memory region.
	bli_array_init( num_elem, sizeof( pool_t* ), array );

	// Save the pointer in the caller's array_t*.
	*array_p = array;
}

void bli_apool_free_block
     (
       array_t* array
     )
{
	const siz_t    num_elem = bli_array_num_elem( array );
	      pool_t** buf      = bli_array_buf( array );

	// Step through the array and finalize each pool_t.
	for ( dim_t i = 0; i < num_elem; ++i )
	{
		pool_t* pool = buf[ i ];

		#ifdef BLIS_ENABLE_MEM_TRACING
		printf( "bli_apool_free_block(): freeing pool_t %d within array_t.\n",
		        ( int )i );
		fflush( stdout );
		#endif

		// Finalize and free the current pool_t, if it was created/allocated.
		if ( pool != NULL )
		{
			// Finalize the pool.
			bli_pool_finalize( pool, FALSE );

			#ifdef BLIS_ENABLE_MEM_TRACING
			printf( "bli_apool_free_block(): pool_t %d: ", ( int )i );
			#endif

			// Free the pool_t struct.
			bli_free_intl( pool );
		}
	}

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_apool_free_block(): " );
	#endif

	// Free the array buffer.
	bli_array_finalize( array );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_apool_free_block(): freeing array_t: " );
	#endif

	// Free the array.
	bli_free_intl( array );
}

void bli_apool_finalize
     (
       apool_t* apool
     )
{
	// NOTE: Since the apool_t's mutex is now initialized statically, we no
	// longer need to explicitly destroy it.

	// Query the mutex from the apool_t.
	//bli_pthread_mutex_t* mutex = bli_apool_mutex( apool );

	// Destroy the mutex.
	//bli_pthread_mutex_destroy( mutex );

	// Query the underlying pool_t and mutex from the apool_t.
	pool_t* pool = bli_apool_pool( apool );

	// ----------------------------------------------------------------

	// Query the block_ptrs array.
	array_t** block_ptrs = bli_pool_block_ptrs( pool );

	// Query the total number of blocks currently allocated.
	siz_t num_blocks = bli_pool_num_blocks( pool );

	// Query the top_index of the pool.
	siz_t top_index = bli_pool_top_index( pool );

	// Sanity check: The top_index should be zero.
	if ( top_index != 0 ) bli_abort();

	// Free the individual blocks (each an array_t) currently in the pool.
	for ( dim_t i = 0; i < num_blocks; ++i )
	{
		#ifdef BLIS_ENABLE_MEM_TRACING
		printf( "bli_apool_finalize(): freeing array_t %d within apool_t.\n",
		        ( int )i );
		fflush( stdout );
		#endif

		bli_apool_free_block( block_ptrs[i] );
	}

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_apool_finalize(): freeing block_ptrs (length %d): ",
	        ( int )( bli_pool_block_ptrs_len( pool ) ) );
	#endif

	// Free the block_ptrs array.
	bli_free_intl( block_ptrs );
}

array_t* bli_apool_checkout_array
     (
       siz_t    n_threads,
       apool_t* apool
     )
{
	// Acquire the apool_t's mutex.
	bli_apool_lock( apool );

	// ----------------------------------------------------------------------------

	// NOTE: Unlike with the bli_pool API, we do not need to handle potential
	// reinitialization since the apool_t's block_size (corresponding to the
	// size of an array_t struct) will never grow.

	// If the apool_t is exhausted, add a block (e.g. an array_t).
	if ( bli_apool_is_exhausted( apool ) )
	{
		#ifdef BLIS_ENABLE_MEM_TRACING
		printf( "bli_apool_checkout_block(): apool_t is exhausted; "
		        "growing by 1 array_t.\n" );
		fflush( stdout );
		#endif

		bli_apool_grow( 1, apool );
	}

	// At this point, at least one array_t is guaranteed to be available.

	// Query the underlying pool_t from the apool_t.
	pool_t* pool = bli_apool_pool( apool );

	// Query the block_ptrs array.
	array_t** block_ptrs = bli_pool_block_ptrs( pool );

	// Query the top_index of the pool.
	const siz_t top_index = bli_pool_top_index( pool );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_apool_checkout_array(): checking out array_t %d.\n",
	        ( int )top_index );
	fflush( stdout );
	#endif

	// Select the array_t* at top_index to return to the caller.
	array_t* array = block_ptrs[ top_index ];

	// Increment the pool's top_index.
	bli_pool_set_top_index( top_index + 1, pool );

	// ----------------------------------------------------------------------------

	// Release the apool_t's mutex.
	bli_apool_unlock( apool );

	// Resize the array_t according to the number of threads specified by the
	// caller. (We need one element in the array_t per thread.)
	bli_array_resize( n_threads, array );

	// Return the selected array_t*.
	return array;
}

void bli_apool_checkin_array
     (
       array_t* array,
       apool_t* apool
     )
{
	// Acquire the apool_t's mutex.
	bli_apool_lock( apool );

	// Query the underlying pool_t from the apool_t.
	pool_t* pool = bli_apool_pool( apool );

	// ----------------------------------------------------------------------------

	// NOTE: Unlike with the bli_pool API, we do not need to handle potential
	// freeing of the blocks upon checkin due to the block_size having since
	// changed due to reinitialization since the apool's block_size will never
	// change.

	// Query the block_ptrs array.
	array_t** block_ptrs = bli_pool_block_ptrs( pool );

	// Query the top_index of the pool.
	const siz_t top_index = bli_pool_top_index( pool );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_apool_checkin_block(): checking in array_t %d.\n",
	        ( int )top_index - 1 );
	fflush( stdout );
	#endif

	// Copy the caller's array_t address to the element at top_index - 1.
	block_ptrs[ top_index - 1 ] = array;

	// Decrement the pool's top_index.
	bli_pool_set_top_index( top_index - 1, pool );

	// ----------------------------------------------------------------------------

	// Release the apool_t's mutex.
	bli_apool_unlock( apool );
}

pool_t* bli_apool_array_elem
     (
       siz_t    index,
       array_t* array
     )
{
	err_t r_val;

	// Query the array element corresponding to index.
	// NOTE: If we knew that the array_t contained elements of size
	// sizeof( void* ) or sizeof( whatever ), we could return the *value*
	// stored in the array. But since array_t is general-purpose, it can't
	// return the element itself. So instead, bli_array_elem() returns the
	// address of the element in the array. Since the elements that apool_t
	// stores in the array_t are pool_t*, that means that the function is
	// actually returning the address of a pool_t*, or pool_t**, hence the
	// dereferencing below.
	pool_t** pool_p = bli_array_elem( index, array );
	pool_t*  pool   = *pool_p;

	// If the element is NULL, then it means a pool_t has not yet been created
	// and allocated for the given index (thread id).
	if ( pool == NULL )
	{
		// Settle on the parameters to use when initializing the pool_t for
		// the current index within the array_t.
		const siz_t num_blocks     = 1;
		const siz_t block_ptrs_len = 25;
		const siz_t align_size     = 16;
		const siz_t offset_size    = 0;
		malloc_ft   malloc_fp      = BLIS_MALLOC_POOL;
		free_ft     free_fp        = BLIS_FREE_POOL;

		// Each small block pool should contain blocks large enough to
		// accommodate any of the data structures for which they will be
		// used.
		const siz_t sizes[]    = { sizeof( cntl_t ),
		                           sizeof( thrcomm_t ),
		                           sizeof( thrinfo_t ) };
		const siz_t n_sizes    = sizeof( sizes ) / sizeof( sizes[0] );
		      siz_t block_size = 0;

		// Find the largest of the sizes above and use that as the block_size
		// for the pool.
		for ( dim_t i = 0; i < n_sizes; ++i )
		{
			if ( block_size < sizes[i] ) block_size = sizes[i];
		}

		#ifdef BLIS_ENABLE_MEM_TRACING
		printf( "bli_apool_array_elem(): pool_t for tid %d is NULL; allocating pool_t.\n",
		        ( int )index );
		printf( "bli_apool_array_elem(): allocating pool_t: " );
		#endif

		// Allocate the pool_t.
		pool = bli_malloc_intl( sizeof( pool_t ), &r_val );

		// Initialize the pool_t.
		bli_pool_init
		(
		  num_blocks,
		  block_ptrs_len,
		  block_size,
		  align_size,
		  offset_size,
		  malloc_fp,
		  free_fp,
		  pool
		);

		// Update the array element with the address to the new pool_t.
		// NOTE: We pass in the address of the pool_t* since the bli_array
		// API is generalized for arbitrarily-sized elements, and therefore
		// it must always take the address of the data, rather than the
		// value (which it can only do if the elem size were fixed).
		bli_array_set_elem( &pool, index, array );
	}

	// The array element is now guaranteed to refer to an allocated and
	// initialized pool_t.

	// Return the array element.
	return pool;
}

void bli_apool_grow
     (
       siz_t    num_blocks_add,
       apool_t* apool
     )
{
	err_t r_val;

	// If the requested increase is zero, return early.
	if ( num_blocks_add == 0 ) return;

	// Query the underlying pool_t from the apool_t.
	pool_t* pool = bli_apool_pool( apool );

	// Query the default initial array length from the apool_t.
	const siz_t num_elem = bli_apool_def_array_len( apool );

	// ----------------------------------------------------------------------------

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
		const siz_t block_ptrs_len_new = 2 * block_ptrs_len_cur;

		// Query the current block_ptrs array.
		array_t** block_ptrs_cur = bli_pool_block_ptrs( pool );

		#ifdef BLIS_ENABLE_MEM_TRACING
		printf( "bli_apool_grow(): growing block_ptrs_len (%d -> %d): ",
		        ( int )block_ptrs_len_cur, ( int )block_ptrs_len_new );
		#endif

		// Allocate a new block_ptrs array.
		array_t** block_ptrs_new
		=
		bli_malloc_intl( block_ptrs_len_new * sizeof( array_t* ), &r_val );

		// Query the top_index of the pool.
		const siz_t top_index = bli_pool_top_index( pool );

		// Copy the contents of the old block_ptrs array to the new/resized
		// array. Notice that we can begin with top_index since all entries
		// from 0 to top_index-1 have been (and are currently) checked out
		// to threads.
		for ( dim_t i = top_index; i < num_blocks_cur; ++i )
		{
			block_ptrs_new[i] = block_ptrs_cur[i];
		}

		#ifdef BLIS_ENABLE_MEM_TRACING
		printf( "bli_apool_grow(): freeing prev block_ptrs: " );
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

	// Query the current block_ptrs array (which was maybe just resized).
	array_t** block_ptrs = bli_pool_block_ptrs( pool );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_apool_grow(): growing apool_t (%d -> %d).\n",
	        ( int )num_blocks_cur, ( int )num_blocks_new );
	fflush( stdout );
	#endif

	// Allocate the requested additional blocks in the resized array.
	for ( dim_t i = num_blocks_cur; i < num_blocks_new; ++i )
	{
		bli_apool_alloc_block
		(
		  num_elem,
		  &(block_ptrs[i])
		);
	}

	// Update the pool_t struct with the new number of allocated blocks.
	// Notice that top_index remains unchanged, as do the block_size and
	// align_size fields.
	bli_pool_set_num_blocks( num_blocks_new, pool );
}

