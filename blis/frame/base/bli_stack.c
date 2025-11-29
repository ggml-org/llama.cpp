/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2023, Southern Methodist University

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

err_t bli_stack_init
     (
       siz_t   elem_size,
       siz_t   block_len,
       siz_t   max_blocks,
       siz_t   initial_size,
       stck_t* stack
     )
{
	if ( stack == NULL )
		return BLIS_NULL_POINTER;

	if ( initial_size > max_blocks * block_len )
		return BLIS_OUT_OF_BOUNDS;

	// Set up an initial state which cannot store any elements
	stack->elem_size = elem_size;
	stack->block_len = block_len;
	stack->max_blocks = 0;
	stack->size = 0;
	stack->capacity = 0;

	if ( bli_pthread_mutex_init( &stack->lock, NULL ) != 0 )
		return BLIS_LOCK_FAILURE;

	err_t error;
	stack->blocks = ( void** )bli_malloc_intl( sizeof( void* ) * max_blocks, &error );
	if ( error != BLIS_SUCCESS )
		return error;

	// Set this to a non-zero value only after successfully
	// allocating the blocks array. This way on failure, we
	// always get a valid stck_t, even if it can't actually contain
	// any elements.
	stack->max_blocks = max_blocks;

	// Determine how many blocks are required to store the intial capacity
	siz_t len = block_len;
	siz_t num_blocks = ( initial_size + len - 1 ) / len;

	// Allocate the new blocks one by one. If an allocation fails,
	// the stack state will still be valid for as many blocks as were
	// successfully allocated. This requires only updating the stack
	// capacity *after* successful allocation.
	for ( siz_t block = 0; block < num_blocks; block++ )
	{
		stack->blocks[ block ] = bli_malloc_intl( len * stack->elem_size, &error );
		if ( error != BLIS_SUCCESS )
			return error;

		stack->capacity += len;
	}

	stack->size = initial_size;

	return BLIS_SUCCESS;
}

err_t bli_stack_finalize( stck_t* stack )
{
	siz_t len = stack->block_len;
	siz_t num_blocks = ( stack->capacity + len - 1 ) / len;

	for ( siz_t block = num_blocks; block --> 0; )
		bli_free_intl( stack->blocks[ block ] );

	bli_free_intl( stack->blocks );

	stack->size = 0;
	stack->capacity = 0;
	stack->max_blocks = 0;

	return BLIS_SUCCESS;
}

err_t bli_stack_get( siz_t i, void** elem, const stck_t* stack )
{
	if ( elem == NULL )
		return BLIS_NULL_POINTER;

	if ( stack == NULL )
	{
		*elem = NULL;
		return BLIS_NULL_POINTER;
	}

	if ( /* i < 0 || */ i >= bli_stack_size( stack ) )
	{
		*elem = NULL;
		return BLIS_OUT_OF_BOUNDS;
	}

	// Calculate the position of the requested element using
	// an O(1) addressing algorithm. Note that all information used
	// here can never change even during stack pushes in other threads.
	siz_t block = i / stack->block_len;
	siz_t i_in_block = i % stack->block_len;
	*elem = ( void* )( ( char* )stack->blocks[ block ] + i_in_block * stack->elem_size );

	return BLIS_SUCCESS;
}

err_t bli_stack_push( siz_t* i, stck_t* stack )
{
	if ( i == NULL || stack == NULL )
		return BLIS_NULL_POINTER;

	// While normal access doesn't require locking, we *do* have to
	// lock to update the size and capacity.
	if ( bli_pthread_mutex_lock( &stack->lock ) != 0 )
		return BLIS_LOCK_FAILURE;

	// Check if we will need to allocate some extra space.
	if ( stack->size + 1 > stack->capacity )
	{
		// Determine how many blocks are required to store the new capacity.
		// A default growth factor of 1.5 (3/2) is used; the check against
		// stack->size + 1 ensures that we grow the size even if the initial
		// capacity is zero. Also don't grow the capacity beyond the maximum
		// number of blocks (unless the stack is completely full and we
		// return an error code below).
		siz_t len = stack->block_len;
		siz_t num_blocks_orig = ( stack->capacity + len - 1) / len;
		siz_t new_capacity = bli_max( stack->size + 1,
		                              bli_min( ( stack->capacity * 3 ) / 2,
		                                       stack->max_blocks * len
		                                     )
		                            );
		siz_t num_blocks_new = ( new_capacity + len - 1 ) / len;

		// If too many blocks are required we must fail.
		if ( num_blocks_new > stack->max_blocks )
		{
			bli_pthread_mutex_unlock( &stack->lock );
			return BLIS_OUT_OF_BOUNDS;
		}

		// Allocate the new blocks one by one. If an allocation fails,
		// the stack state will still be valid for as many blocks as were
		// successfully allocated. This requires only updating the stack
		// capacity *after* successful allocation.
		err_t error;
		for ( siz_t block = num_blocks_orig; block < num_blocks_new; block++ )
		{
			stack->blocks[ block ] = bli_malloc_intl( len * stack->elem_size, &error );
			if ( error != BLIS_SUCCESS )
			{
				bli_pthread_mutex_unlock( &stack->lock );
				return error;
			}

			stack->capacity += len;
		}
	}

	// Save the position of the end of the stack and finally increase the size.
	*i = stack->size;
	stack->size += 1;

	bli_pthread_mutex_unlock( &stack->lock );

	return BLIS_SUCCESS;
}

