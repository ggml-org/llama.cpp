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

//#define BLIS_ENABLE_MEM_TRACING

void bli_array_init
     (
       siz_t    num_elem,
       siz_t    elem_size,
       array_t* array
     )
{
	err_t r_val;

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_array_init(): allocating array [%d * %d]: ",
	        ( int )num_elem, ( int )elem_size );
	#endif

	// Compute the total size (in bytes) of the array.
	const size_t array_size = num_elem * elem_size;

	// Allocate the array buffer.
	void* buf = bli_malloc_intl( array_size, &r_val );

	// Initialize the array elements to zero. THIS IS IMPORANT because
	// consumer threads will use the NULL-ness of the array elements to
	// determine if the corresponding block (data structure) needs to be
	// created/allocated and initialized.
	memset( buf, 0, array_size );

	// Initialize the array_t structure.
	bli_array_set_buf( buf, array );
	bli_array_set_num_elem( num_elem, array );
	bli_array_set_elem_size( elem_size, array );
}

void bli_array_resize
     (
       siz_t    num_elem_new,
       array_t* array
     )
{
	err_t r_val;

	// Query the number of elements in the array.
	const siz_t num_elem_prev = bli_array_num_elem( array );

	// If the new requested size (number of elements) is less than or equal to
	// the current size, no action is needed; return early.
	if ( num_elem_new <= num_elem_prev ) return;

	// At this point, we know that num_elem_prev < num_elem_new, which means
	// we need to proceed with the resizing.

	// Query the size of each element in the array.
	const siz_t elem_size = bli_array_elem_size( array );

	// Compute the total size (in bytes) of the array before and after resizing.
	const size_t array_size_prev = num_elem_prev * elem_size;
	const size_t array_size_new  = num_elem_new  * elem_size;

	// Query the previous array buffer.
	void* buf_prev = bli_array_buf( array );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_array_resize(): allocating array [%d * %d]: ",
	        ( int )num_elem_new, ( int )elem_size );
	#endif

	// Allocate a new array buffer.
	char* buf_new = bli_malloc_intl( array_size_new, &r_val );

	// Copy the previous array contents to the new array.
	memcpy( buf_new, buf_prev, array_size_prev );

	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_array_resize(): freeing array [%d * %d]: ",
	        ( int )num_elem_prev, ( int )elem_size );
	#endif

	// Now that the elements have been copied over to the new buffer, we can
	// free the previous array buffer.
	bli_free_intl( buf_prev );

	// Initialize the new elements' contents to zero. (Note that we advance
	// the new buffer address by the size of the previous array so that we
	// arrive at the first byte of the new segment.)
	memset( &buf_new[ array_size_prev ], 0, array_size_new - array_size_prev );

	// Update the array_t structure.
	// NOTE: The array elem_size field does not need updating.
	bli_array_set_buf( buf_new, array );
	bli_array_set_num_elem( num_elem_new, array );
}

void bli_array_finalize
     (
       array_t* array
     )
{
	#ifdef BLIS_ENABLE_MEM_TRACING
	printf( "bli_array_finalize(): freeing buf (length %d): ",
	        ( int )bli_array_num_elem( array ) );
	#endif

	// Query the buffer from the array.
	void* buf = bli_array_buf( array );

	// Free the buffer.
	bli_free_intl( buf );
}

void* bli_array_elem
     (
             siz_t    index,
       const array_t* array
     )
{
	// Query the number of elements in the array.
	const siz_t num_elem = bli_array_num_elem( array );

	// Sanity check: disallow access beyond the bounds of the array.
	if ( num_elem <= index ) bli_abort();

	// Query the size of each element in the array.
	const siz_t elem_size = bli_array_elem_size( array );

	// Query the buffer from the array, but store it as a char* so we can use
	// it to easily perform byte pointer arithmetic.
	char* buf = bli_array_buf( array );

	// Advance the pointer by (index * elem_size) bytes.
	buf += index * elem_size;

	// Return the address of the element computed above.
	return ( void* )buf;
}

void bli_array_set_elem
     (
       void*    elem,
       siz_t    index,
       array_t* array
     )
{
	// Query the size of each element in the array.
	const siz_t elem_size = bli_array_elem_size( array );

	// Query the buffer from the array as a char*.
	char* buf = bli_array_buf( array );

// memcpy() is the only safe way to copy data of unknown type
#if 0
	if ( elem_size == sizeof( void* ) )
	{
		#ifdef BLIS_ENABLE_MEM_TRACING
		printf( "bli_array_set_elem(): elem_size is %d; setting index %d.\n",
		        ( int )elem_size, ( int )index );
		fflush( stdout );
		#endif

		// Special case: Handle elem_size = sizeof( void* ) without calling
		// memcpy().
		void** buf_vvp  = ( void** )buf;
		void** elem_vvp = ( void** )elem;

		buf_vvp[ index ] = *elem_vvp;
	}
	else
	{
#endif
		// General case: Copy the elem_size bytes from elem to buf at the
		// element index specified by index.
		memcpy( &buf[ index * elem_size ], elem, ( size_t )elem_size );
#if 0
	}
#endif
}

