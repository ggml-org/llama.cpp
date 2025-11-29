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

#ifndef BLIS_APOOL_H
#define BLIS_APOOL_H

// -- Locked pool-of-arrays type --

/*
typedef struct apool_s
{
	bli_pthread_mutex_t mutex;
	pool_t              pool;

	siz_t               def_array_len;

} apool_t;
*/


// apool entry query

BLIS_INLINE pool_t* bli_apool_pool( apool_t* apool )
{
	return &(apool->pool);
}

BLIS_INLINE bli_pthread_mutex_t* bli_apool_mutex( apool_t* apool )
{
	return &(apool->mutex);
}

BLIS_INLINE siz_t bli_apool_def_array_len( const apool_t* pool )
{
	return pool->def_array_len;
}

BLIS_INLINE bool bli_apool_is_exhausted( const apool_t* apool )
{
	return bli_pool_is_exhausted( &apool->pool );
}

// apool action

BLIS_INLINE void bli_apool_lock( apool_t* apool )
{
	bli_pthread_mutex_lock( bli_apool_mutex( apool ) );
}

BLIS_INLINE void bli_apool_unlock( apool_t* apool )
{
	bli_pthread_mutex_unlock( bli_apool_mutex( apool ) );
}

// apool entry modification

BLIS_INLINE void bli_apool_set_def_array_len( siz_t def_array_len, apool_t* pool ) \
{
	pool->def_array_len = def_array_len;
}

// -----------------------------------------------------------------------------

void bli_apool_init
     (
       apool_t* apool
     );
void bli_apool_finalize
     (
       apool_t* apool
     );

array_t* bli_apool_checkout_array
     (
       siz_t    n_threads,
       apool_t* apool
     );
void bli_apool_checkin_array
     (
       array_t* array,
       apool_t* apool
     );

pool_t* bli_apool_array_elem
     (
       siz_t    index,
       array_t* array
     );

void bli_apool_grow
     (
       siz_t    num_blocks_add,
       apool_t* apool
     );

void bli_apool_alloc_block
     (
       siz_t     num_elem,
       array_t** array_p
     );
void bli_apool_free_block
     (
       array_t* array
     );


#endif

