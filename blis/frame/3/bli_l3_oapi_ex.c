/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, The University of Texas at Austin
   Copyright (C) 2024, Southern Methodist University

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


//
// Define object-based interfaces (expert).
//

// If a sandbox was enabled, we forgo defining bli_gemm_ex() since it will be
// defined in the sandbox environment.
#ifdef BLIS_ENABLE_SANDBOX
void PASTEMAC(gemm_def,BLIS_OAPI_EX_SUF)
#else
void PASTEMAC(gemm,BLIS_OAPI_EX_SUF)
#endif
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	// Check the operands.
	if ( bli_error_checking_is_enabled() )
		bli_gemm_check( alpha, a, b, beta, c, cntx );

	// Check for zero dimensions, alpha == 0, or other conditions which
	// mean that we don't actually have to perform a full l3 operation.
	if ( bli_l3_return_early_if_trivial( alpha, a, b, beta, c ) == BLIS_SUCCESS )
		return;

	// Execute the small/unpacked oapi handler. If it finds that the problem
	// does not fall within the thresholds that define "small", or for some
	// other reason decides not to use the small/unpacked implementation,
	// the function returns with BLIS_FAILURE, which causes execution to
	// proceed towards the conventional implementation.
	if ( bli_gemmsup( alpha, a, b, beta, c, cntx, rntm ) == BLIS_SUCCESS )
		return;

	// Default to using native execution.
	num_t dt = bli_obj_dt( c );
	ind_t im = BLIS_NAT;

	// If each matrix operand has a complex storage datatype, try to get an
	// induced method (if one is available and enabled). NOTE: Allowing
	// precisions to vary while using 1m, which is what we do here, is unique
	// to gemm; other level-3 operations use 1m only if all storage datatypes
	// are equal (and they ignore the computation precision).
	if ( bli_obj_is_complex( c ) &&
	     bli_obj_is_complex( a ) &&
	     bli_obj_is_complex( b ) )
	{
		// Find the highest priority induced method that is both enabled and
		// available for the current operation. (If an induced method is
		// available but not enabled, or simply unavailable, BLIS_NAT will
		// be returned here.)
		im = bli_gemmind_find_avail( dt );
	}

	// If necessary, obtain a valid context from the gks using the induced
	// method id determined above.
	if ( cntx == NULL ) cntx = bli_gks_query_cntx();

#if 0
#ifdef BLIS_ENABLE_SMALL_MATRIX
	// Only handle small problems separately for homogeneous datatypes.
	if ( bli_obj_dt( a ) == bli_obj_dt( b ) &&
	     bli_obj_dt( a ) == bli_obj_dt( c ) &&
	     bli_obj_comp_prec( c ) == bli_obj_prec( c ) )
	{
		err_t status = bli_gemm_small( alpha, a, b, beta, c, cntx, cntl );
		if ( status == BLIS_SUCCESS ) return;
	}
#endif
#endif

	// Alias A, B, and C in case we need to apply transformations.
	obj_t a_local;
	obj_t b_local;
	obj_t c_local;
	bli_obj_alias_submatrix( a, &a_local );
	bli_obj_alias_submatrix( b, &b_local );
	bli_obj_alias_submatrix( c, &c_local );

	gemm_cntl_t cntl;
	bli_gemm_cntl_init
	(
	  im,
	  BLIS_GEMM,
	  alpha,
	  &a_local,
	  &b_local,
	  beta,
	  &c_local,
	  cntx,
	  &cntl
	);

	// Invoke the internal back-end via the thread handler.
	bli_l3_thread_decorator
	(
	  &a_local,
	  &b_local,
	  &c_local,
	  cntx,
	  ( cntl_t* )&cntl,
	  rntm
	);
}


void PASTEMAC(gemmt,BLIS_OAPI_EX_SUF)
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	// Check the operands.
	if ( bli_error_checking_is_enabled() )
		bli_gemmt_check( alpha, a, b, beta, c, cntx );

	// Check for zero dimensions, alpha == 0, or other conditions which
	// mean that we don't actually have to perform a full l3 operation.
	if ( bli_l3_return_early_if_trivial( alpha, a, b, beta, c ) == BLIS_SUCCESS )
		return;

	// Default to using native execution.
	num_t dt = bli_obj_dt( c );
	ind_t im = BLIS_NAT;

	// If all matrix operands are complex and of the same storage datatype, try
	// to get an induced method (if one is available and enabled).
	if ( bli_obj_dt( a ) == bli_obj_dt( c ) &&
	     bli_obj_dt( b ) == bli_obj_dt( c ) &&
	     bli_obj_is_complex( c ) )
	{
		// Find the highest priority induced method that is both enabled and
		// available for the current operation. (If an induced method is
		// available but not enabled, or simply unavailable, BLIS_NAT will
		// be returned here.)
		im = bli_gemmtind_find_avail( dt );
	}

	// If necessary, obtain a valid context from the gks using the induced
	// method id determined above.
	if ( cntx == NULL ) cntx = bli_gks_query_cntx();

	// Alias A, B, and C in case we need to apply transformations.
	obj_t a_local;
	obj_t b_local;
	obj_t c_local;
	bli_obj_alias_submatrix( a, &a_local );
	bli_obj_alias_submatrix( b, &b_local );
	bli_obj_alias_submatrix( c, &c_local );

	gemm_cntl_t cntl;
	bli_gemm_cntl_init
	(
	  im,
	  BLIS_GEMMT,
	  alpha,
	  &a_local,
	  &b_local,
	  beta,
	  &c_local,
	  cntx,
	  &cntl
	);

	// Invoke the internal back-end via the thread handler.
	bli_l3_thread_decorator
	(
	  &a_local,
	  &b_local,
	  &c_local,
	  cntx,
	  ( cntl_t* )&cntl,
	  rntm
	);
}


void PASTEMAC(her2k,BLIS_OAPI_EX_SUF)
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_her2k_check( alpha, a, b, beta, c, cntx );

	obj_t alphah;
	obj_t ah;
	obj_t bh;
	bli_obj_alias_with_conj( BLIS_CONJUGATE, alpha, &alphah );
	bli_obj_alias_with_trans( BLIS_CONJ_TRANSPOSE, a, &ah );
	bli_obj_alias_with_trans( BLIS_CONJ_TRANSPOSE, b, &bh );

	// Invoke gemmt twice, using beta only the first time.
	PASTEMAC(gemmt,BLIS_OAPI_EX_SUF)(   alpha, a, &bh,      beta, c, cntx, rntm );
	PASTEMAC(gemmt,BLIS_OAPI_EX_SUF)( &alphah, b, &ah, &BLIS_ONE, c, cntx, rntm );

	// The Hermitian rank-2k product was computed as alpha*A*B'+alpha'*B*A', even for
	// the diagonal elements. Mathematically, the imaginary components of
	// diagonal elements of a Hermitian rank-2k product should always be
	// zero. However, in practice, they sometimes accumulate meaningless
	// non-zero values. To prevent this, we explicitly set those values
	// to zero before returning.
	bli_setid( &BLIS_ZERO, c );
}


void PASTEMAC(syr2k,BLIS_OAPI_EX_SUF)
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_syr2k_check( alpha, a, b, beta, c, cntx );

	obj_t at;
	obj_t bt;
	bli_obj_alias_with_trans( BLIS_TRANSPOSE, a, &at );
	bli_obj_alias_with_trans( BLIS_TRANSPOSE, b, &bt );

	// Invoke gemmt twice, using beta only the first time.
	PASTEMAC(gemmt,BLIS_OAPI_EX_SUF)( alpha, a, &bt,      beta, c, cntx, rntm );
	PASTEMAC(gemmt,BLIS_OAPI_EX_SUF)( alpha, b, &at, &BLIS_ONE, c, cntx, rntm );
}


void PASTEMAC(herk,BLIS_OAPI_EX_SUF)
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_herk_check( alpha, a, beta, c, cntx );

	obj_t ah;
	bli_obj_alias_with_trans( BLIS_CONJ_TRANSPOSE, a, &ah );

	PASTEMAC(gemmt,BLIS_OAPI_EX_SUF)( alpha, a, &ah, beta, c, cntx, rntm );

	// The Hermitian rank-k product was computed as Re(alpha)*A*A', even for the
	// diagonal elements. Mathematically, the imaginary components of
	// diagonal elements of a Hermitian rank-k product should always be
	// zero. However, in practice, they sometimes accumulate meaningless
	// non-zero values. To prevent this, we explicitly set those values
	// to zero before returning.
	bli_setid( &BLIS_ZERO, c );
}


void PASTEMAC(syrk,BLIS_OAPI_EX_SUF)
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bli_syrk_check( alpha, a, beta, c, cntx );

	obj_t at;
	bli_obj_alias_with_trans( BLIS_TRANSPOSE, a, &at );

	PASTEMAC(gemmt,BLIS_OAPI_EX_SUF)( alpha, a, &at, beta, c, cntx, rntm );
}


void PASTEMAC(hemm,BLIS_OAPI_EX_SUF)
     (
             side_t  side,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	// Check the operands.
	if ( bli_error_checking_is_enabled() )
		bli_hemm_check( side, alpha, a, b, beta, c, cntx );

	// Check for zero dimensions, alpha == 0, or other conditions which
	// mean that we don't actually have to perform a full l3 operation.
	if ( bli_l3_return_early_if_trivial( alpha, a, b, beta, c ) == BLIS_SUCCESS )
		return;

	// Default to using native execution.
	num_t dt = bli_obj_dt( c );
	ind_t im = BLIS_NAT;

	// If all matrix operands are complex and of the same storage datatype, try
	// to get an induced method (if one is available and enabled).
	if ( bli_obj_dt( a ) == bli_obj_dt( c ) &&
	     bli_obj_dt( b ) == bli_obj_dt( c ) &&
	     bli_obj_is_complex( c ) )
	{
		// Find the highest priority induced method that is both enabled and
		// available for the current operation. (If an induced method is
		// available but not enabled, or simply unavailable, BLIS_NAT will
		// be returned here.)
		im = bli_hemmind_find_avail( dt );
	}

	// If necessary, obtain a valid context from the gks using the induced
	// method id determined above.
	if ( cntx == NULL ) cntx = bli_gks_query_cntx();

	// Alias A, B, and C in case we need to apply transformations.
	obj_t a_local;
	obj_t b_local;
	obj_t c_local;
	bli_obj_alias_submatrix( a, &a_local );
	bli_obj_alias_submatrix( b, &b_local );
	bli_obj_alias_submatrix( c, &c_local );

	// If the Hermitian/symmetric matrix A is being multiplied from the right,
	// swap A and B so that the Hermitian/symmetric matrix will actually be on
	// the right.
	if ( bli_is_right( side ) )
	{
		bli_obj_swap( &a_local, &b_local );
	}

	gemm_cntl_t cntl;
	bli_gemm_cntl_init
	(
	  im,
	  BLIS_HEMM,
	  alpha,
	  &a_local,
	  &b_local,
	  beta,
	  &c_local,
	  cntx,
	  &cntl
	);

	// Invoke the internal back-end.
	bli_l3_thread_decorator
	(
	  &a_local,
	  &b_local,
	  &c_local,
	  cntx,
	  ( cntl_t* )&cntl,
	  rntm
	);
}


void PASTEMAC(symm,BLIS_OAPI_EX_SUF)
     (
             side_t  side,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	// Check the operands.
	if ( bli_error_checking_is_enabled() )
		bli_symm_check( side, alpha, a, b, beta, c, cntx );

	// Check for zero dimensions, alpha == 0, or other conditions which
	// mean that we don't actually have to perform a full l3 operation.
	if ( bli_l3_return_early_if_trivial( alpha, a, b, beta, c ) == BLIS_SUCCESS )
		return;

	// Default to using native execution.
	num_t dt = bli_obj_dt( c );
	ind_t im = BLIS_NAT;

	// If all matrix operands are complex and of the same storage datatype, try
	// to get an induced method (if one is available and enabled).
	if ( bli_obj_dt( a ) == bli_obj_dt( c ) &&
	     bli_obj_dt( b ) == bli_obj_dt( c ) &&
	     bli_obj_is_complex( c ) )
	{
		// Find the highest priority induced method that is both enabled and
		// available for the current operation. (If an induced method is
		// available but not enabled, or simply unavailable, BLIS_NAT will
		// be returned here.)
		im = bli_symmind_find_avail( dt );
	}

	// If necessary, obtain a valid context from the gks using the induced
	// method id determined above.
	if ( cntx == NULL ) cntx = bli_gks_query_cntx();

	// Alias A, B, and C in case we need to apply transformations.
	obj_t a_local;
	obj_t b_local;
	obj_t c_local;
	bli_obj_alias_submatrix( a, &a_local );
	bli_obj_alias_submatrix( b, &b_local );
	bli_obj_alias_submatrix( c, &c_local );

	// If the Hermitian/symmetric matrix A is being multiplied from the right,
	// swap A and B so that the Hermitian/symmetric matrix will actually be on
	// the right.
	if ( bli_is_right( side ) )
	{
		bli_obj_swap( &a_local, &b_local );
	}

	gemm_cntl_t cntl;
	bli_gemm_cntl_init
	(
	  im,
	  BLIS_SYMM,
	  alpha,
	  &a_local,
	  &b_local,
	  beta,
	  &c_local,
	  cntx,
	  &cntl
	);

	// Invoke the internal back-end.
	bli_l3_thread_decorator
	(
	  &a_local,
	  &b_local,
	  &c_local,
	  cntx,
	  ( cntl_t* )&cntl,
	  rntm
	);
}


void PASTEMAC(trmm3,BLIS_OAPI_EX_SUF)
     (
             side_t  side,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	// Check the operands.
	if ( bli_error_checking_is_enabled() )
		bli_trmm3_check( side, alpha, a, b, beta, c, cntx );

	// Check for zero dimensions, alpha == 0, or other conditions which
	// mean that we don't actually have to perform a full l3 operation.
	if ( bli_l3_return_early_if_trivial( alpha, a, b, beta, c ) == BLIS_SUCCESS )
		return;

	// Default to using native execution.
	num_t dt = bli_obj_dt( c );
	ind_t im = BLIS_NAT;

	// If all matrix operands are complex and of the same storage datatype, try
	// to get an induced method (if one is available and enabled).
	if ( bli_obj_dt( a ) == bli_obj_dt( c ) &&
	     bli_obj_dt( b ) == bli_obj_dt( c ) &&
	     bli_obj_is_complex( c ) )
	{
		// Find the highest priority induced method that is both enabled and
		// available for the current operation. (If an induced method is
		// available but not enabled, or simply unavailable, BLIS_NAT will
		// be returned here.)
		im = bli_trmm3ind_find_avail( dt );
	}

	// If necessary, obtain a valid context from the gks using the induced
	// method id determined above.
	if ( cntx == NULL ) cntx = bli_gks_query_cntx();

	// Alias A, B, and C so we can tweak the objects if necessary.
	obj_t a_local;
	obj_t b_local;
	obj_t c_local;
	bli_obj_alias_submatrix( a, &a_local );
	bli_obj_alias_submatrix( b, &b_local );
	bli_obj_alias_submatrix( c, &c_local );

	// If A is being multiplied from the right, swap A and B so that
	// the matrix will actually be on the right.
	if ( bli_is_right( side ) )
	{
		bli_obj_swap( &a_local, &b_local );
	}

	gemm_cntl_t cntl;
	bli_gemm_cntl_init
	(
	  im,
	  BLIS_TRMM3,
	  alpha,
	  &a_local,
	  &b_local,
	  beta,
	  &c_local,
	  cntx,
	  &cntl
	);

	// Invoke the internal back-end.
	bli_l3_thread_decorator
	(
	  &a_local,
	  &b_local,
	  &c_local,
	  cntx,
	  ( cntl_t* )&cntl,
	  rntm
	);
}


void PASTEMAC(trmm,BLIS_OAPI_EX_SUF)
     (
             side_t  side,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	// Check the operands.
	if ( bli_error_checking_is_enabled() )
		bli_trmm_check( side, alpha, a, b, cntx );

	// Check for zero dimensions, alpha == 0, or other conditions which
	// mean that we don't actually have to perform a full l3 operation.
	if ( bli_l3_return_early_if_trivial( alpha, a, b, &BLIS_ZERO, b ) == BLIS_SUCCESS )
		return;

	// Default to using native execution.
	num_t dt = bli_obj_dt( b );
	ind_t im = BLIS_NAT;

	// If all matrix operands are complex and of the same storage datatype, try
	// to get an induced method (if one is available and enabled).
	if ( bli_obj_dt( a ) == bli_obj_dt( b ) &&
	     bli_obj_is_complex( b ) )
	{
		// Find the highest priority induced method that is both enabled and
		// available for the current operation. (If an induced method is
		// available but not enabled, or simply unavailable, BLIS_NAT will
		// be returned here.)
		im = bli_trmmind_find_avail( dt );
	}

	// If necessary, obtain a valid context from the gks using the induced
	// method id determined above.
	if ( cntx == NULL ) cntx = bli_gks_query_cntx();

	// Alias A and B so we can tweak the objects if necessary.
	obj_t a_local;
	obj_t b_local;
	obj_t c_local;
	bli_obj_alias_submatrix( a, &a_local );
	bli_obj_alias_submatrix( b, &b_local );
	bli_obj_alias_submatrix( b, &c_local );

	// If A is being multiplied from the right, swap A and B so that
	// the matrix will actually be on the right.
	if ( bli_is_right( side ) )
	{
		bli_obj_swap( &a_local, &b_local );
	}

	gemm_cntl_t cntl;
	bli_gemm_cntl_init
	(
	  im,
	  BLIS_TRMM,
	  alpha,
	  &a_local,
	  &b_local,
	  &BLIS_ZERO,
	  &c_local,
	  cntx,
	  &cntl
	);

	// Invoke the internal back-end.
	bli_l3_thread_decorator
	(
	  &a_local,
	  &b_local,
	  &c_local,
	  cntx,
	  ( cntl_t* )&cntl,
	  rntm
	);
}


void PASTEMAC(trsm,BLIS_OAPI_EX_SUF)
     (
             side_t  side,
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	// Check the operands.
	if ( bli_error_checking_is_enabled() )
		bli_trsm_check( side, alpha, a, b, cntx );

	// Check for zero dimensions, alpha == 0, or other conditions which
    // mean that we don't actually have to perform a full l3 operation.
	if ( bli_l3_return_early_if_trivial( alpha, a, b, &BLIS_ZERO, b ) == BLIS_SUCCESS )
		return;

	// Default to using native execution.
	num_t dt = bli_obj_dt( b );
	ind_t im = BLIS_NAT;

	// If all matrix operands are complex and of the same storage datatype, try
	// to get an induced method (if one is available and enabled).
	if ( bli_obj_dt( a ) == bli_obj_dt( b ) &&
	     bli_obj_is_complex( b ) )
	{
		// Find the highest priority induced method that is both enabled and
		// available for the current operation. (If an induced method is
		// available but not enabled, or simply unavailable, BLIS_NAT will
		// be returned here.)
		im = bli_trsmind_find_avail( dt );
	}

	// If necessary, obtain a valid context from the gks using the induced
	// method id determined above.
	if ( cntx == NULL ) cntx = bli_gks_query_cntx();

#if 0
#ifdef BLIS_ENABLE_SMALL_MATRIX_TRSM
	gint_t status = bli_trsm_small( side, alpha, a, b, cntx, cntl );
	if ( status == BLIS_SUCCESS ) return;
#endif
#endif

	// Alias A and B so we can tweak the objects if necessary.
	obj_t a_local;
	obj_t b_local;
	obj_t c_local;
	bli_obj_alias_submatrix( a, &a_local );
	bli_obj_alias_submatrix( b, &b_local );
	bli_obj_alias_submatrix( b, &c_local );

#if 1

	// If A is being solved against from the right, transpose all operands
	// so that we can perform the computation as if A were being solved
	// from the left.
	if ( bli_is_right( side ) )
	{
		bli_toggle_side( &side );
		bli_obj_induce_trans( &a_local );
		bli_obj_induce_trans( &b_local );
		bli_obj_induce_trans( &c_local );
	}

#else

	// NOTE: Enabling this code requires that BLIS NOT be configured with
	// BLIS_RELAX_MCNR_NCMR_CONSTRAINTS defined.
#ifdef BLIS_RELAX_MCNR_NCMR_CONSTRAINTS
	#error "BLIS_RELAX_MCNR_NCMR_CONSTRAINTS must not be defined for current trsm_r implementation."
#endif

	// If A is being solved against from the right, swap A and B so that
	// the triangular matrix will actually be on the right.
	if ( bli_is_right( side ) )
	{
		bli_obj_swap( &a_local, &b_local );
	}

#endif

	trsm_cntl_t cntl;
	bli_trsm_cntl_init
	(
	  im,
	  alpha,
	  &a_local,
	  &b_local,
	  alpha,
	  &c_local,
	  cntx,
	  &cntl
	);

	// Invoke the internal back-end.
	bli_l3_thread_decorator
	(
	  &a_local,
	  &b_local,
	  &c_local,
	  cntx,
	  ( cntl_t* )&cntl,
	  rntm
	);
}
