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
             trans_t transa,
             dim_t   m,
             dim_t   n,
       const void*   a, inc_t rs_a, inc_t cs_a,
             void*   b, inc_t rs_b, inc_t cs_b
     );

static FUNCPTR_T GENARRAY2_ALL(ftypes,castm);

//
// Define object-based interface.
//

void bli_castm
     (
       const obj_t* a,
       const obj_t* b
     )
{
	const num_t   dt_a   = bli_obj_dt( a );
	const num_t   dt_b   = bli_obj_dt( b );

	const trans_t transa = bli_obj_conjtrans_status( a );

	const dim_t   m      = bli_obj_length( b );
	const dim_t   n      = bli_obj_width( b );

	const void*   buf_a  = bli_obj_buffer_at_off( a );
	const inc_t   rs_a   = bli_obj_row_stride( a );
	const inc_t   cs_a   = bli_obj_col_stride( a );

	      void*   buf_b  = bli_obj_buffer_at_off( b );
	const inc_t   rs_b   = bli_obj_row_stride( b );
	const inc_t   cs_b   = bli_obj_col_stride( b );

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_castm_check( a, b );

#if 0
	if ( bli_obj_dt( a ) == bli_obj_dt( b ) )
	{
		// If a and b share the same datatype, we can simply use copym.
		bli_copym( a, b );
		return;
	}
#endif

	// Index into the type combination array to extract the correct
	// function pointer.
	ftypes[dt_a][dt_b]
	(
	  transa,
	  m,
	  n,
	  buf_a, rs_a, cs_a,
	  buf_b, rs_b, cs_b
	);
}

// -----------------------------------------------------------------------------

//
// Define BLAS-like interfaces with typed operands.
//

#undef  GENTFUNC2
#define GENTFUNC2( ctype_a, ctype_b, cha, chb, opname ) \
\
void PASTEMAC(cha,chb,opname) \
     ( \
             trans_t transa, \
             dim_t   m, \
             dim_t   n, \
       const void*   a, inc_t rs_a, inc_t cs_a, \
             void*   b, inc_t rs_b, inc_t cs_b  \
     ) \
{ \
	const ctype_a* restrict a_cast = a; \
	      ctype_b* restrict b_cast = b; \
	      conj_t            conja; \
	      dim_t             n_iter; \
	      dim_t             n_elem; \
	      inc_t             lda, inca; \
	      inc_t             ldb, incb; \
	      dim_t             j, i; \
\
	/* Set various loop parameters. */ \
	bli_set_dims_incs_2m \
	( \
	  transa, \
	  m,       n,       rs_a,  cs_a, rs_b,  cs_b, \
	  &n_elem, &n_iter, &inca, &lda, &incb, &ldb  \
	); \
\
	/* Extract the conjugation component from the transa parameter. */ \
	conja = bli_extract_conj( transa ); \
\
	if ( bli_is_conj( conja ) ) \
	{ \
		if ( inca == 1 && incb == 1 ) \
		{ \
			for ( j = 0; j < n_iter; ++j ) \
			{ \
				const ctype_a* restrict a1 = a_cast + (j  )*lda + (0  )*inca; \
				      ctype_b* restrict b1 = b_cast + (j  )*ldb + (0  )*incb; \
\
				for ( i = 0; i < n_elem; ++i ) \
				{ \
					bli_tcopyjs( cha,chb, a1[i], b1[i] ); \
				} \
			} \
		} \
		else \
		{ \
			for ( j = 0; j < n_iter; ++j ) \
			{ \
				const ctype_a* restrict a1 = a_cast + (j  )*lda + (0  )*inca; \
				      ctype_b* restrict b1 = b_cast + (j  )*ldb + (0  )*incb; \
\
				for ( i = 0; i < n_elem; ++i ) \
				{ \
					bli_tcopyjs( cha,chb, *a1, *b1 ); \
\
					a1 += inca; \
					b1 += incb; \
				} \
			} \
		} \
	} \
	else \
	{ \
		if ( inca == 1 && incb == 1 ) \
		{ \
			for ( j = 0; j < n_iter; ++j ) \
			{ \
				const ctype_a* restrict a1 = a_cast + (j  )*lda + (0  )*inca; \
				      ctype_b* restrict b1 = b_cast + (j  )*ldb + (0  )*incb; \
\
				for ( i = 0; i < n_elem; ++i ) \
				{ \
					bli_tcopys( cha,chb, a1[i], b1[i] ); \
				} \
			} \
		} \
		else \
		{ \
			for ( j = 0; j < n_iter; ++j ) \
			{ \
				const ctype_a* restrict a1 = a_cast + (j  )*lda + (0  )*inca; \
				      ctype_b* restrict b1 = b_cast + (j  )*ldb + (0  )*incb; \
\
				for ( i = 0; i < n_elem; ++i ) \
				{ \
					bli_tcopys( cha,chb, *a1, *b1 ); \
\
					a1 += inca; \
					b1 += incb; \
				} \
			} \
		} \
	} \
}

INSERT_GENTFUNC2_BASIC( castm )
INSERT_GENTFUNC2_MIX_DP( castm )

// -----------------------------------------------------------------------------

//
// Define object-based _check() function.
//

void bli_castm_check
     (
       const obj_t* a,
       const obj_t* b
     )
{
	err_t e_val;

	// Check object datatypes.

	e_val = bli_check_floating_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_floating_object( b );
	bli_check_error_code( e_val );

	// Check structure.
	// NOTE: We enforce general structure for now in order to simplify the
	// implementation.

	bli_check_general_object( a );
	bli_check_error_code( e_val );

	bli_check_general_object( b );
	bli_check_error_code( e_val );

	// Check object dimensions.

	e_val = bli_check_matrix_object( a );
	bli_check_error_code( e_val );

	e_val = bli_check_matrix_object( b );
	bli_check_error_code( e_val );

	e_val = bli_check_conformal_dims( a, b );
	bli_check_error_code( e_val );

	// Check object buffers (for non-NULLness).

	e_val = bli_check_object_buffer( a );
	bli_check_error_code( e_val );

	e_val = bli_check_object_buffer( b );
	bli_check_error_code( e_val );
}

