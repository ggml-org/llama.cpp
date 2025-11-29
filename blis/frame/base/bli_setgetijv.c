/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

typedef void (*setijv_fp)
     (
       double ar,
       double ai,
       dim_t  i,
       void*  x, inc_t incx
     );
static setijv_fp GENARRAY(ftypes_setijv,setijv);

err_t bli_setijv
     (
             double ar,
             double ai,
             dim_t  i,
       const obj_t* x
     )
{
	dim_t n    = bli_obj_vector_dim( x );
	dim_t incx = bli_obj_vector_inc( x );
	num_t dt   = bli_obj_dt( x );

	// Return error if i is beyond bounds of the vector.
	if ( i < 0 || n <= i ) return BLIS_FAILURE;

	// Don't modify scalar constants.
	if ( dt == BLIS_CONSTANT ) return BLIS_FAILURE;

	// Query the pointer to the buffer at the adjusted offsets.
	void* x_p = bli_obj_buffer_at_off( x );

	// Index into the function pointer array.
	setijv_fp f = ftypes_setijv[ dt ];

	// Invoke the type-specific function.
	f
	(
	  ar,
	  ai,
	  i,
	  x_p, incx
	);

	return BLIS_SUCCESS;
}

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       double ar, \
       double ai, \
       dim_t  i, \
       void*  x, inc_t incx  \
     ) \
{ \
	ctype* restrict x_cast = ( ctype* )x; \
\
	ctype* restrict x_i = x_cast + (i  )*incx; \
\
	bli_tsets( z,ch, ar, ai, *x_i ); \
}

INSERT_GENTFUNC_BASIC( setijv )

// -----------------------------------------------------------------------------

typedef void (*getijv_fp)
     (
             dim_t   i,
       const void*   x, inc_t incx,
             double* ar,
             double* ai
     );
static getijv_fp GENARRAY(ftypes_getijv,getijv);

err_t bli_getijv
      (
              dim_t   i,
        const obj_t*  x,
              double* ar,
              double* ai
      )
{
	dim_t n    = bli_obj_vector_dim( x );
	dim_t incx = bli_obj_vector_inc( x );
	num_t dt   = bli_obj_dt( x );

	// Return error if i is beyond bounds of the vector.
	if ( i < 0 || n <= i ) return BLIS_FAILURE;

	// Disallow access into scalar constants.
	if ( dt == BLIS_CONSTANT ) return BLIS_FAILURE;

	// Query the pointer to the buffer at the adjusted offsets.
	void* x_p = bli_obj_buffer_at_off( x );

	// Index into the function pointer array.
	getijv_fp f = ftypes_getijv[ dt ];

	// Invoke the type-specific function.
	f
	(
	  i,
	  x_p, incx,
	  ar,
	  ai
	);

	return BLIS_SUCCESS;
}

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
             dim_t   i, \
       const void*   x, inc_t incx, \
             double* ar, \
             double* ai  \
     ) \
{ \
	const ctype* restrict x_cast = ( const ctype* )x; \
\
	const ctype* restrict x_i = x_cast + (i  )*incx; \
\
	bli_tgets( ch,z, *x_i, *ar, *ai ); \
}

INSERT_GENTFUNC_BASIC( getijv )

