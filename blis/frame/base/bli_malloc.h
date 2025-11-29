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

// Typedef function pointer types for malloc() and free() substitutes.
//typedef void* (*malloc_ft) ( size_t size );
//typedef void  (*free_ft)   ( void*  p    );

// -----------------------------------------------------------------------------

#if 0
BLIS_EXPORT_BLIS void* bli_malloc_pool( size_t size );
BLIS_EXPORT_BLIS void   bli_free_pool( void* p );
#endif

void* bli_malloc_intl( size_t size, err_t* r_val );
void* bli_calloc_intl( size_t size, err_t* r_val );
void  bli_free_intl( void* p );

BLIS_EXPORT_BLIS void* bli_malloc_user( size_t size, err_t* r_val );
BLIS_EXPORT_BLIS void  bli_free_user( void* p );

// -----------------------------------------------------------------------------

void* bli_fmalloc_align( malloc_ft f, size_t size, size_t align_size, err_t* r_val );
void  bli_ffree_align( free_ft f, void* p );

void* bli_fmalloc_noalign( malloc_ft f, size_t size, err_t* r_val );
void  bli_ffree_noalign( free_ft f, void* p );

void  bli_fmalloc_align_check( malloc_ft f, size_t size, size_t align_size );
void  bli_fmalloc_post_check( void* p );

