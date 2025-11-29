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

#ifndef BLIS_ARRAY_H
#define BLIS_ARRAY_H

// -- Array type --

/*
typedef struct array_s
{
	void*     buf;

	siz_t     num_elem;
	siz_t     elem_size;

} array_t;
*/


// Array entry query

BLIS_INLINE void* bli_array_buf( const array_t* array )
{
	return array->buf;
}

BLIS_INLINE siz_t bli_array_num_elem( const array_t* array )
{
	return array->num_elem;
}

BLIS_INLINE siz_t bli_array_elem_size( const array_t* array )
{
	return array->elem_size;
}

// Array entry modification

BLIS_INLINE void bli_array_set_buf( void* buf, array_t* array ) \
{
	array->buf = buf;
}

BLIS_INLINE void bli_array_set_num_elem( siz_t num_elem, array_t* array ) \
{
	array->num_elem = num_elem;
}

BLIS_INLINE void bli_array_set_elem_size( siz_t elem_size, array_t* array ) \
{
	array->elem_size = elem_size;
}

// -----------------------------------------------------------------------------

void bli_array_init
     (
       siz_t    num_elem,
       siz_t    elem_size,
       array_t* array
     );
void bli_array_resize
     (
       siz_t    num_elem_new,
       array_t* array
     );
void bli_array_finalize
     (
       array_t* array
     );

void* bli_array_elem
     (
             siz_t    index,
       const array_t* array
     );
void bli_array_set_elem
     (
       void*    elem,
       siz_t    index,
       array_t* array
     );

#endif

