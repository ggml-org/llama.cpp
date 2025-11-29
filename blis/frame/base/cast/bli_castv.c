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

// NOTE: This is one of the few functions in BLIS that is defined
// with heterogeneous type support. This is done so that we have
// an operation that can be used to typecast (copy-cast) a matrix
// of one datatype to a scalar of another datatype.

typedef void (*FUNCPTR_T)
     (
             conj_t conjx,
             dim_t  n,
       const void*  x, inc_t inc_x,
             void*  y, inc_t inc_y
     );

static FUNCPTR_T GENARRAY2_ALL(ftypes,castv);

//
// Define object-based interface.
//

void bli_castv
     (
       const obj_t* x,
       const obj_t* y
     )
{
	const num_t  dt_x  = bli_obj_dt( x );
	const num_t  dt_y  = bli_obj_dt( y );

	const conj_t conjx = bli_obj_conj_status( x );

	const dim_t  n     = bli_obj_vector_dim( x );

	const void*  buf_x = bli_obj_buffer_at_off( x );
	const inc_t  inc_x = bli_obj_vector_inc( x );

	      void*  buf_y = bli_obj_buffer_at_off( y );
	const inc_t  inc_y = bli_obj_vector_inc( y );

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_castv_check( x, y );

#if 0
	if ( bli_obj_dt( x ) == bli_obj_dt( y ) )
	{
		// If x and y share the same datatype, we can simply use copyv.
		bli_copyv( x, y );
		return;
	}
#endif

	// Index into the type combination array to extract the correct
	// function pointer.
	ftypes[dt_x][dt_y]
	(
	  conjx,
	  n,
	  buf_x, inc_x,
	  buf_y, inc_y
	);
}

// -----------------------------------------------------------------------------

//
// Define BLAS-like interfaces with typed operands.
//

#undef  GENTFUNC2
#define GENTFUNC2( ctype_x, ctype_y, chx, chy, opname ) \
\
void PASTEMAC(chx,chy,opname) \
     ( \
             conj_t conjx, \
             dim_t  n, \
       const void*  x, inc_t incx, \
             void*  y, inc_t incy  \
     ) \
{ \
	const ctype_x* restrict x1 = x; \
	      ctype_y* restrict y1 = y; \
	      dim_t             i; \
\
	if ( bli_is_conj( conjx ) ) \
	{ \
		if ( incx == 1 && incy == 1 ) \
		{ \
			for ( i = 0; i < n; ++i ) \
			{ \
				bli_tcopyjs( chx,chy, x1[i], y1[i] ); \
			} \
		} \
		else \
		{ \
			for ( i = 0; i < n; ++i ) \
			{ \
				bli_tcopyjs( chx,chy, *x1, *y1 ); \
\
				x1 += incx; \
				y1 += incy; \
			} \
		} \
	} \
	else \
	{ \
		if ( incx == 1 && incy == 1 ) \
		{ \
			for ( i = 0; i < n; ++i ) \
			{ \
				bli_tcopys( chx,chy, x1[i], y1[i] ); \
			} \
		} \
		else \
		{ \
			for ( i = 0; i < n; ++i ) \
			{ \
				bli_tcopys( chx,chy, *x1, *y1 ); \
\
				x1 += incx; \
				y1 += incy; \
			} \
		} \
	} \
}

INSERT_GENTFUNC2_BASIC( castv )
INSERT_GENTFUNC2_MIX_DP( castv )

// -----------------------------------------------------------------------------

//
// Define object-based _check() function.
//

void bli_castv_check
     (
       const obj_t* x,
       const obj_t* y
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_floating_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( y );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_vector_object( x );
	bli_check_error_code( e_val );

	e_val = bli_check_vector_object( y );
	bli_check_error_code( e_val );

	e_val = bli_check_equal_vector_lengths( x, y );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( x );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( y );
	bli_check_error_code( e_val );
}

