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

#ifndef BLIS_POOL_H
#define BLIS_POOL_H

// -- Pool block type --

/*
typedef struct pblk_s
{
	void*     buf;
	siz_t     block_size;

} pblk_t;
*/

// -- Pool type --

/*
typedef struct pool_s
{
	void*     block_ptrs;
	siz_t     block_ptrs_len;

	siz_t     top_index;
	siz_t     num_blocks;

	siz_t     block_size;
	siz_t     align_size;

	malloc_ft malloc_fp;
	free_ft   free_fp;

} pool_t;
*/


// Pool block query

BLIS_INLINE void* bli_pblk_buf( const pblk_t* pblk )
{
	return pblk->buf;
}

BLIS_INLINE siz_t bli_pblk_block_size( const pblk_t* pblk )
{
	return pblk->block_size;
}

// Pool block modification

BLIS_INLINE void bli_pblk_set_buf( void* buf, pblk_t* pblk )
{
	pblk->buf = buf;
}

BLIS_INLINE void bli_pblk_set_block_size( siz_t block_size, pblk_t* pblk )
{
	pblk->block_size = block_size;
}

//
// -- pool block initialization ------------------------------------------------
//

// NOTE: This initializer macro must be updated whenever fields are added or
// removed from the pblk_t type definition. An alternative to the initializer is
// calling bli_pblk_clear() at runtime.

#define BLIS_PBLK_INITIALIZER \
        { \
          /* .buf        = */ NULL, \
          /* .block_size = */ 0, \
        }  \

BLIS_INLINE void bli_pblk_clear( pblk_t* pblk )
{
	bli_pblk_set_buf( NULL, pblk );
	bli_pblk_set_block_size( 0, pblk );
}


// Pool entry query

BLIS_INLINE void* bli_pool_block_ptrs( const pool_t* pool )
{
	return pool->block_ptrs;
}

BLIS_INLINE siz_t bli_pool_block_ptrs_len( const pool_t* pool )
{
	return pool->block_ptrs_len;
}

BLIS_INLINE siz_t bli_pool_num_blocks( const pool_t* pool )
{
	return pool->num_blocks;
}

BLIS_INLINE siz_t bli_pool_block_size( const pool_t* pool )
{
	return pool->block_size;
}

BLIS_INLINE siz_t bli_pool_align_size( const pool_t* pool )
{
	return pool->align_size;
}

BLIS_INLINE siz_t bli_pool_offset_size( const pool_t* pool )
{
	return pool->offset_size;
}

BLIS_INLINE malloc_ft bli_pool_malloc_fp( const pool_t* pool )
{
	return pool->malloc_fp;
}

BLIS_INLINE free_ft bli_pool_free_fp( const pool_t* pool )
{
	return pool->free_fp;
}

BLIS_INLINE siz_t bli_pool_top_index( const pool_t* pool )
{
	return pool->top_index;
}

BLIS_INLINE bool bli_pool_is_exhausted( const pool_t* pool )
{
	return ( bool )
	       ( bli_pool_top_index( pool ) == bli_pool_num_blocks( pool ) );
}

// Pool entry modification

BLIS_INLINE void bli_pool_set_block_ptrs( void* block_ptrs, pool_t* pool ) \
{
	pool->block_ptrs = block_ptrs;
}

BLIS_INLINE void bli_pool_set_block_ptrs_len( siz_t block_ptrs_len, pool_t* pool ) \
{
	pool->block_ptrs_len = block_ptrs_len;
}

BLIS_INLINE void bli_pool_set_num_blocks( siz_t num_blocks, pool_t* pool ) \
{
	pool->num_blocks = num_blocks;
}

BLIS_INLINE void bli_pool_set_block_size( siz_t block_size, pool_t* pool ) \
{
	pool->block_size = block_size;
}

BLIS_INLINE void bli_pool_set_align_size( siz_t align_size, pool_t* pool ) \
{
	pool->align_size = align_size;
}

BLIS_INLINE void bli_pool_set_offset_size( siz_t offset_size, pool_t* pool ) \
{
	pool->offset_size = offset_size;
}

BLIS_INLINE void bli_pool_set_malloc_fp( malloc_ft malloc_fp, pool_t* pool ) \
{
	pool->malloc_fp = malloc_fp;
}

BLIS_INLINE void bli_pool_set_free_fp( free_ft free_fp, pool_t* pool ) \
{
	pool->free_fp = free_fp;
}

BLIS_INLINE void bli_pool_set_top_index( siz_t top_index, pool_t* pool ) \
{
	pool->top_index = top_index;
}

// -----------------------------------------------------------------------------

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
     );
void bli_pool_finalize
     (
       pool_t* pool,
       bool    reinit
     );
void bli_pool_reinit
     (
       siz_t   num_blocks_new,
       siz_t   block_ptrs_len_new,
       siz_t   block_size_new,
       siz_t   align_size_new,
       siz_t   offset_size_new,
       pool_t* pool
     );

void bli_pool_checkout_block
     (
       siz_t   req_size,
       pblk_t* block,
       pool_t* pool
     );
void bli_pool_checkin_block
     (
       pblk_t* block,
       pool_t* pool
     );

void bli_pool_grow
     (
       siz_t   num_blocks_add,
       pool_t* pool
     );
void bli_pool_shrink
     (
       siz_t   num_blocks_sub,
       pool_t* pool
     );

void bli_pool_alloc_block
     (
       siz_t     block_size,
       siz_t     align_size,
       siz_t     offset_size,
       malloc_ft malloc_fp,
       pblk_t*   block
     );
void bli_pool_free_block
     (
       siz_t   offset_size,
       free_ft free_fp,
       pblk_t* block
     );

void bli_pool_print
     (
       const pool_t* pool
     );
void bli_pblk_print
     (
       const pblk_t* pblk
     );

#endif

