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


#ifndef BLIS_MEM_H
#define BLIS_MEM_H


// mem_t object type (defined in bli_type_defs.h)

/*
typedef struct mem_s
{
	pblk_t    pblk;
	packbuf_t buf_type;
	pool_t*   pool;
	siz_t     size;
} mem_t;

typedef struct pblk_s
{
	void*     buf;
	siz_t     block_size;
} pblk_t;
*/

//
// -- mem_t query --------------------------------------------------------------
//

BLIS_INLINE pblk_t* bli_mem_pblk( mem_t* mem )
{
	return &(mem->pblk);
}

BLIS_INLINE void* bli_mem_buffer( const mem_t* mem )
{
	return bli_pblk_buf( bli_mem_pblk( ( mem_t* )mem ) );
}

BLIS_INLINE packbuf_t bli_mem_buf_type( const mem_t* mem )
{
	return mem->buf_type;
}

BLIS_INLINE pool_t* bli_mem_pool( const mem_t* mem )
{
	return mem->pool;
}

BLIS_INLINE siz_t bli_mem_size( const mem_t* mem )
{
	return mem->size;
}

BLIS_INLINE bool bli_mem_is_alloc( const mem_t* mem )
{
	return ( bool )
	       ( bli_mem_buffer( mem ) != NULL );
}

BLIS_INLINE bool bli_mem_is_unalloc( const mem_t* mem )
{
	return ( bool )
	       ( bli_mem_buffer( mem ) == NULL );
}


//
// -- mem_t modification -------------------------------------------------------
//

BLIS_INLINE void bli_mem_set_pblk( pblk_t* pblk, mem_t* mem )
{
	mem->pblk = *pblk;
}

BLIS_INLINE void bli_mem_set_buffer( void* buf, mem_t* mem )
{
	bli_pblk_set_buf( buf, &(mem->pblk) );
}

BLIS_INLINE void bli_mem_set_buf_type( packbuf_t buf_type, mem_t* mem )
{
	mem->buf_type = buf_type;
}

BLIS_INLINE void bli_mem_set_pool( pool_t* pool, mem_t* mem )
{
	mem->pool = pool;
}

BLIS_INLINE void bli_mem_set_size( siz_t size, mem_t* mem )
{
	mem->size = size;
}

//
// -- mem_t initialization -----------------------------------------------------
//

// NOTE: This initializer macro must be updated whenever fields are added or
// removed from the mem_t type definition. An alternative to the initializer is
// calling bli_mem_clear() at runtime.

#define BLIS_MEM_INITIALIZER \
        { \
          /* .pblk     = */ BLIS_PBLK_INITIALIZER, \
          /* .buf_type = */ BLIS_BUFFER_FOR_GEN_USE, \
          /* .pool     = */ NULL, \
          /* .size     = */ 0, \
        }


BLIS_INLINE void bli_mem_clear( mem_t* mem )
{
	bli_mem_set_buffer( NULL, mem );
	bli_mem_set_buf_type( BLIS_BUFFER_FOR_GEN_USE, mem );
	bli_mem_set_pool( NULL, mem );
	bli_mem_set_size( 0, mem );
}


#endif
