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

typedef void (*setijm_fp)
     (
       double ar,
       double ai,
       dim_t  i,
       dim_t  j,
       void*  b, inc_t rs, inc_t cs
     );
static setijm_fp GENARRAY(ftypes_setijm,setijm);

err_t bli_setijm
     (
             double ar,
             double ai,
             dim_t  i,
             dim_t  j,
       const obj_t* b
     )
{
	dim_t m  = bli_obj_length( b );
	dim_t n  = bli_obj_width( b );
	dim_t rs = bli_obj_row_stride( b );
	dim_t cs = bli_obj_col_stride( b );
	num_t dt = bli_obj_dt( b );

	// Return error if i or j is beyond bounds of the matrix/vector.
	if ( i < 0 || m <= i ) return BLIS_FAILURE;
	if ( j < 0 || n <= j ) return BLIS_FAILURE;

	// Don't modify scalar constants.
	if ( dt == BLIS_CONSTANT ) return BLIS_FAILURE;

	// Query the pointer to the buffer at the adjusted offsets.
	void* b_p = bli_obj_buffer_at_off( b );

	// Index into the function pointer array.
	setijm_fp f = ftypes_setijm[ dt ];

	// Invoke the type-specific function.
	f
	(
	  ar,
	  ai,
	  i,
	  j,
	  b_p, rs, cs
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
       dim_t  j, \
       void*  b, inc_t rs, inc_t cs  \
     ) \
{ \
	ctype* b_cast = ( ctype* )b; \
\
	ctype* b_ij = b_cast + (i  )*rs + (j  )*cs; \
\
	bli_tsets( z,ch, ar, ai, *b_ij ); \
}

INSERT_GENTFUNC_BASIC( setijm )

// -----------------------------------------------------------------------------

typedef void (*getijm_fp)
     (
             dim_t   i,
             dim_t   j,
       const void*   b, inc_t rs, inc_t cs,
             double* ar,
             double* ai
     );
static getijm_fp GENARRAY(ftypes_getijm,getijm);

err_t bli_getijm
      (
              dim_t   i,
              dim_t   j,
        const obj_t*  b,
              double* ar,
              double* ai
      )
{
	dim_t m  = bli_obj_length( b );
	dim_t n  = bli_obj_width( b );
	dim_t rs = bli_obj_row_stride( b );
	dim_t cs = bli_obj_col_stride( b );
	num_t dt = bli_obj_dt( b );

	// Return error if i or j is beyond bounds of the matrix/vector.
	if ( i < 0 || m <= i ) return BLIS_FAILURE;
	if ( j < 0 || n <= j ) return BLIS_FAILURE;

	// Disallow access into scalar constants.
	if ( dt == BLIS_CONSTANT ) return BLIS_FAILURE;

	// Query the pointer to the buffer at the adjusted offsets.
	void* b_p = bli_obj_buffer_at_off( b );

	// Index into the function pointer array.
	getijm_fp f = ftypes_getijm[ dt ];

	// Invoke the type-specific function.
	f
	(
	  i,
	  j,
	  b_p, rs, cs,
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
             dim_t   j, \
       const void*   b, inc_t rs, inc_t cs, \
             double* ar, \
             double* ai  \
     ) \
{ \
	const ctype* b_cast = ( const ctype* )b; \
\
	const ctype* b_ij = b_cast + (i  )*rs + (j  )*cs; \
\
	bli_tgets( ch,z, *b_ij, *ar, *ai ); \
}

INSERT_GENTFUNC_BASIC( getijm )

