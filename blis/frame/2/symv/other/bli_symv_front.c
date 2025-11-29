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

extern hemv_t* hemv_cntl_bs_ke_lrow_ucol;
extern hemv_t* hemv_cntl_bs_ke_lcol_urow;
extern hemv_t* hemv_cntl_ge_lrow_ucol;
extern hemv_t* hemv_cntl_ge_lcol_urow;

void bli_symv_front
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  x,
       obj_t*  beta,
       obj_t*  y,
       cntx_t* cntx
     )
{
	hemv_t* hemv_cntl;
	num_t   dt_targ_a;
	num_t   dt_targ_x;
	num_t   dt_targ_y;
	bool    a_has_unit_inc;
	bool    x_has_unit_inc;
	bool    y_has_unit_inc;
	obj_t   alpha_local;
	obj_t   beta_local;
	num_t   dt_alpha;
	num_t   dt_beta;

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_symv_check( alpha, a, x, beta, y );


	// Query the target datatypes of each object.
	dt_targ_a = bli_obj_target_dt( a );
	dt_targ_x = bli_obj_target_dt( x );
	dt_targ_y = bli_obj_target_dt( y );

	// Determine whether each operand with unit stride.
	a_has_unit_inc = ( bli_obj_is_row_stored( a ) ||
	                   bli_obj_is_col_stored( a ) );
	x_has_unit_inc = ( bli_obj_vector_inc( x ) == 1 );
	y_has_unit_inc = ( bli_obj_vector_inc( y ) == 1 );


	// Create an object to hold a copy-cast of alpha. Notice that we use
	// the type union of the target datatypes of a and x to prevent any
	// unnecessary loss of information during the computation.
	dt_alpha = bli_dt_union( dt_targ_a, dt_targ_x );
	bli_obj_scalar_init_detached_copy_of( dt_alpha,
	                                      BLIS_NO_CONJUGATE,
	                                      alpha,
	                                      &alpha_local );

	// Create an object to hold a copy-cast of beta. Notice that we use
	// the datatype of y. Here's why: If y is real and beta is complex,
	// there is no reason to keep beta_local in the complex domain since
	// the complex part of beta*y will not be stored. If y is complex and
	// beta is real then beta is harmlessly promoted to complex.
	dt_beta = dt_targ_y;
	bli_obj_scalar_init_detached_copy_of( dt_beta,
	                                      BLIS_NO_CONJUGATE,
	                                      beta,
	                                      &beta_local );


	// If all operands have unit stride, we choose a control tree for calling
	// the unblocked implementation directly without any blocking.
	if ( a_has_unit_inc &&
	     x_has_unit_inc &&
	     y_has_unit_inc )
	{
		// We use two control trees to handle the four cases corresponding to
		// combinations of upper/lower triangular storage and row/column-storage.
		// The row-stored lower triangular and column-stored upper triangular
		// trees are identical. Same for the remaining two trees.
		if ( bli_obj_is_lower( a ) )
		{
			if ( bli_obj_is_row_stored( a ) ) hemv_cntl = hemv_cntl_bs_ke_lrow_ucol;
			else                               hemv_cntl = hemv_cntl_bs_ke_lcol_urow;
		}
		else // if ( bli_obj_is_upper( a ) )
		{
			if ( bli_obj_is_row_stored( a ) ) hemv_cntl = hemv_cntl_bs_ke_lcol_urow;
			else                               hemv_cntl = hemv_cntl_bs_ke_lrow_ucol;
		}
	}
	else
	{
		// Mark objects with unit stride as already being packed. This prevents
		// unnecessary packing from happening within the blocked algorithm.
		if ( a_has_unit_inc ) bli_obj_set_pack_schema( BLIS_PACKED_UNSPEC, a );
		if ( x_has_unit_inc ) bli_obj_set_pack_schema( BLIS_PACKED_VECTOR, x );
		if ( y_has_unit_inc ) bli_obj_set_pack_schema( BLIS_PACKED_VECTOR, y );

		// Here, we make a similar choice as above, except that (1) we look
		// at storage tilt, and (2) we choose a tree that performs blocking.
		if ( bli_obj_is_lower( a ) )
		{
			if ( bli_obj_is_row_tilted( a ) ) hemv_cntl = hemv_cntl_ge_lrow_ucol;
			else                               hemv_cntl = hemv_cntl_ge_lcol_urow;
		}
		else // if ( bli_obj_is_upper( a ) )
		{
			if ( bli_obj_is_row_tilted( a ) ) hemv_cntl = hemv_cntl_ge_lcol_urow;
			else                               hemv_cntl = hemv_cntl_ge_lrow_ucol;
		}
	}


	// Invoke the internal back-end with the copy-casts of scalars and the
	// chosen control tree. Set conjh to BLIS_NO_CONJUGATE to invoke the
	// symmetric (and not Hermitian) algorithms.
	bli_hemv_int( BLIS_NO_CONJUGATE,
	              &alpha_local,
	              a,
	              x,
	              &beta_local,
	              y,
	              cntx,
	              hemv_cntl );
}


//
// Define BLAS-like interfaces with homogeneous-typed operands.
//
#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       uplo_t  uploa, \
       conj_t  conja, \
       conj_t  conjx, \
       dim_t   m, \
       ctype*  alpha, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       ctype*  x, inc_t incx, \
       ctype*  beta, \
       ctype*  y, inc_t incy, \
       cntx_t* cntx  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	obj_t       alphao, ao, xo, betao, yo; \
\
	inc_t       rs_x, cs_x; \
	inc_t       rs_y, cs_y; \
\
	rs_x = incx; cs_x = m * incx; \
	rs_y = incy; cs_y = m * incy; \
\
	bli_obj_create_1x1_with_attached_buffer( dt, alpha, &alphao ); \
	bli_obj_create_1x1_with_attached_buffer( dt, beta,  &betao  ); \
\
	bli_obj_create_with_attached_buffer( dt, m, m, a, rs_a, cs_a, &ao ); \
	bli_obj_create_with_attached_buffer( dt, m, 1, x, rs_x, cs_x, &xo ); \
	bli_obj_create_with_attached_buffer( dt, m, 1, y, rs_y, cs_y, &yo ); \
\
	bli_obj_set_uplo( uploa, &ao ); \
	bli_obj_set_conj( conja, &ao ); \
	bli_obj_set_conj( conjx, &xo ); \
\
	bli_obj_set_struc( BLIS_SYMMETRIC, &ao ); \
\
	PASTEMAC0(opname)( &alphao, \
	                   &ao, \
	                   &xo, \
	                   &betao, \
	                   &yo, \
	                   cntx ); \
}

INSERT_GENTFUNC_BASIC0( symv_front )

