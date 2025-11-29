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

extern ger_t* ger_cntl_bs_ke_row;
extern ger_t* ger_cntl_bs_ke_col;
extern ger_t* ger_cntl_ge_row;
extern ger_t* ger_cntl_ge_col;

void bli_ger_front
     (
       obj_t*  alpha,
       obj_t*  x,
       obj_t*  y,
       obj_t*  a,
       cntx_t* cntx
     )
{
	ger_t*  ger_cntl;
	num_t   dt_targ_x;
	num_t   dt_targ_y;
	//num_t   dt_targ_a;
	bool    x_has_unit_inc;
	bool    y_has_unit_inc;
	bool    a_has_unit_inc;
	obj_t   alpha_local;
	num_t   dt_alpha;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_ger_check( alpha, x, y, a );


	// Query the target datatypes of each object.
	dt_targ_x = bli_obj_target_dt( x );
	dt_targ_y = bli_obj_target_dt( y );
	//dt_targ_a = bli_obj_target_dt( a );

	// Determine whether each operand with unit stride.
	x_has_unit_inc = ( bli_obj_vector_inc( x ) == 1 );
	y_has_unit_inc = ( bli_obj_vector_inc( y ) == 1 );
	a_has_unit_inc = ( bli_obj_is_row_stored( a ) ||
	                   bli_obj_is_col_stored( a ) );


	// Create an object to hold a copy-cast of alpha. Notice that we use
	// the type union of the target datatypes of x and y to prevent any
	// unnecessary loss of information during the computation.
	dt_alpha = bli_dt_union( dt_targ_x, dt_targ_y );
	bli_obj_scalar_init_detached_copy_of( dt_alpha,
	                                      BLIS_NO_CONJUGATE,
	                                      alpha,
	                                      &alpha_local );


	// If all operands have unit stride, we choose a control tree for calling
	// the unblocked implementation directly without any blocking.
	if ( x_has_unit_inc &&
	     y_has_unit_inc &&
	     a_has_unit_inc )
	{
		// Use different control trees depending on storage of the matrix
		// operand.
		if ( bli_obj_is_row_stored( a ) ) ger_cntl = ger_cntl_bs_ke_row;
		else                               ger_cntl = ger_cntl_bs_ke_col;
	}
	else
	{
		// Mark objects with unit stride as already being packed. This prevents
		// unnecessary packing from happening within the blocked algorithm.
		if ( x_has_unit_inc ) bli_obj_set_pack_schema( BLIS_PACKED_VECTOR, x );
		if ( y_has_unit_inc ) bli_obj_set_pack_schema( BLIS_PACKED_VECTOR, y );
		if ( a_has_unit_inc ) bli_obj_set_pack_schema( BLIS_PACKED_UNSPEC, a );

		// Here, we make a similar choice as above, except that (1) we look
		// at storage tilt, and (2) we choose a tree that performs blocking.
		if ( bli_obj_is_row_tilted( a ) ) ger_cntl = ger_cntl_ge_row;
		else                               ger_cntl = ger_cntl_ge_col;
	}

	// Invoke the internal back-end with the copy-cast scalar and the
	// chosen control tree.
	bli_ger_int( BLIS_NO_CONJUGATE,
	             BLIS_NO_CONJUGATE,
	             &alpha_local,
	             x,
	             y,
	             a,
	             cntx,
	             ger_cntl );
}


//
// Define BLAS-like interfaces with homogeneous-typed operands.
//
#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       conj_t  conjx, \
       conj_t  conjy, \
       dim_t   m, \
       dim_t   n, \
       ctype*  alpha, \
       ctype*  x, inc_t incx, \
       ctype*  y, inc_t incy, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       cntx_t* cntx  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	obj_t       alphao, xo, yo, ao; \
\
	dim_t       m_x; \
	dim_t       m_y; \
	inc_t       rs_x, cs_x; \
	inc_t       rs_y, cs_y; \
\
	bli_set_dims_with_trans( BLIS_NO_TRANSPOSE, m, n, &m_x, &m_y ); \
\
	rs_x = incx; cs_x = m_x * incx; \
	rs_y = incy; cs_y = m_y * incy; \
\
	bli_obj_create_1x1_with_attached_buffer( dt, alpha, &alphao ); \
\
	bli_obj_create_with_attached_buffer( dt, m_x, 1, x, rs_x, cs_x, &xo ); \
	bli_obj_create_with_attached_buffer( dt, m_y, 1, y, rs_y, cs_y, &yo ); \
	bli_obj_create_with_attached_buffer( dt, m,   n, a, rs_a, cs_a, &ao ); \
\
	bli_obj_set_conj( conjx, &xo ); \
	bli_obj_set_conj( conjy, &yo ); \
\
	PASTEMAC0(opname)( &alphao, \
	                   &xo, \
	                   &yo, \
	                   &ao, \
	                   cntx ); \
}

INSERT_GENTFUNC_BASIC0( ger_front )

