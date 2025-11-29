/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, The University of Texas at Austin

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
// -- Define the gemmd operation's object API ----------------------------------
//

void bao_gemmd
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  d,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c
     )
{
	bao_gemmd_ex
	(
	  alpha,
	  a,
	  d,
	  b,
	  beta,
	  c,
	  NULL,
	  NULL
	);
}

void bao_gemmd_ex
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  d,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm
     )
{
	bli_init_once();

	// Initialize a local runtime with global settings if necessary. Note
	// that in the case that a runtime is passed in, we make a local copy.
	rntm_t rntm_l;
	if ( rntm == NULL ) { bli_rntm_init_from_global( &rntm_l ); rntm = &rntm_l; }
	else                { rntm_l = *rntm;                       rntm = &rntm_l; }

	// Set the .pack_a and .pack_b fields to TRUE. This is only needed because
	// this addon uses bli_thrinfo_sup_grow(), which calls
	// bli_thrinfo_sup_create_for_cntl(), which employs an optimization if
	// both fields are FALSE (as is often the case with sup). However, this
	// addon implements the "large" code path, and so both A and B must
	// always be packed. Setting the fields to TRUE will avoid the optimization
	// while this addon implementation executes (and it also reinforces the
	// fact that we *are* indeed packing A and B, albeit not in the sup context
	// originally envisioned for the .pack_a and .pack_b fields).
	bli_rntm_set_pack_a( TRUE, rntm );
	bli_rntm_set_pack_b( TRUE, rntm );

	// Obtain a valid (native) context from the gks if necessary.
	// NOTE: This must be done before calling the _check() function, since
	// that function assumes the context pointer is valid.
	if ( cntx == NULL ) cntx = ( cntx_t* )bli_gks_query_cntx();

	// Check parameters.
	if ( bli_error_checking_is_enabled() )
		bao_gemmd_check( alpha, a, d, b, beta, c, cntx );

	// -- bao_gemmd_front() ----------------------------------------------------

	obj_t a_local;
	obj_t b_local;
	obj_t c_local;

	// If C has a zero dimension, return early.
	if ( bli_obj_has_zero_dim( c ) )
	{
		return;
	}

	// If alpha is zero, or if A or B has a zero dimension, scale C by beta
	// and return early.
	if ( bli_obj_equals( alpha, &BLIS_ZERO ) ||
	     bli_obj_has_zero_dim( a ) ||
	     bli_obj_has_zero_dim( b ) )
	{
		bli_scalm( beta, c );
		return;
	}

	// Alias A, B, and C in case we need to apply transformations.
	bli_obj_alias_to( a, &a_local );
	bli_obj_alias_to( b, &b_local );
	bli_obj_alias_to( c, &c_local );

	// Induce a transposition of A if it has its transposition property set.
	// Then clear the transposition bit in the object.
	if ( bli_obj_has_trans( &a_local ) )
	{
		bli_obj_induce_trans( &a_local );
		bli_obj_set_onlytrans( BLIS_NO_TRANSPOSE, &a_local );
	}

	// Induce a transposition of B if it has its transposition property set.
	// Then clear the transposition bit in the object.
	if ( bli_obj_has_trans( &b_local ) )
	{
		bli_obj_induce_trans( &b_local );
		bli_obj_set_onlytrans( BLIS_NO_TRANSPOSE, &b_local );
	}

	// An optimization: If C is stored by rows and the micro-kernel prefers
	// contiguous columns, or if C is stored by columns and the micro-kernel
	// prefers contiguous rows, transpose the entire operation to allow the
	// micro-kernel to access elements of C in its preferred manner.
	if ( bli_cntx_dislikes_storage_of( &c_local, BLIS_GEMM_VIR_UKR, cntx ) )
	{
		bli_obj_swap( &a_local, &b_local );

		bli_obj_induce_trans( &a_local );
		bli_obj_induce_trans( &b_local );
		bli_obj_induce_trans( &c_local );
	}

	// Parse and interpret the contents of the rntm_t object to properly
	// set the ways of parallelism for each loop, and then make any
	// additional modifications necessary for the current operation.
	bli_rntm_set_ways_for_op
	(
	  BLIS_GEMM,
	  BLIS_LEFT, // ignored for gemm/hemm/symm
	  bli_obj_length( &c_local ),
	  bli_obj_width( &c_local ),
	  bli_obj_width( &a_local ),
	  rntm
	);

	// Spawn threads (if applicable), where bao_gemmd_int() is the thread entry
	// point function for each thread. This also begins the process of creating
	// the thrinfo_t tree, which contains thread communicators.
	bao_l3_thread_decorator
	(
	  bao_gemmd_int,
	  BLIS_GEMM, // operation family id
	  alpha,
	  &a_local,
	  d,
	  &b_local,
	  beta,
	  &c_local,
	  cntx,
	  rntm
	);
}

//
// -- Define the gemmd operation's thread entry point --------------------------
//

void bao_gemmd_int
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  d,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm,
       thrinfo_t* thread
     )
{
	// In this function, we choose the gemmd implementation that is executed
	// on each thread.

	// Call the block-panel algorithm.
	bao_gemmd_bp_var1
	(
	  alpha,
	  a,
	  d,
	  b,
	  beta,
	  c,
	  cntx,
	  rntm,
	  thread
	);
}

//
// -- Define the gemmd operation's typed API -----------------------------------
//

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTECH2(bao_,ch,opname) \
     ( \
       trans_t transa, \
       trans_t transb, \
       dim_t   m, \
       dim_t   n, \
       dim_t   k, \
       ctype*  alpha, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       ctype*  d, inc_t incd, \
       ctype*  b, inc_t rs_b, inc_t cs_b, \
       ctype*  beta, \
       ctype*  c, inc_t rs_c, inc_t cs_c  \
     ) \
{ \
	bli_init_once(); \
\
	/* Determine the datatype (e.g. BLIS_FLOAT, BLIS_DOUBLE, etc.) based on
	   the macro parameter 'ch' (e.g. s, d, etc). */ \
	const num_t dt = PASTEMAC(ch,type); \
\
	obj_t       alphao, ao, dd, bo, betao, co; \
\
	dim_t       m_a, n_a; \
	dim_t       m_b, n_b; \
\
	/* Adjust the dimensions of matrices A and B according to the transa and
	   transb parameters. */ \
	bli_set_dims_with_trans( transa, m, k, &m_a, &n_a ); \
	bli_set_dims_with_trans( transb, k, n, &m_b, &n_b ); \
\
	/* Create bufferless scalar objects and attach the provided scalar pointers
	   to those scalar objects. */ \
	bli_obj_create_1x1_with_attached_buffer( dt, alpha, &alphao ); \
	bli_obj_create_1x1_with_attached_buffer( dt, beta,  &betao  ); \
\
	/* Create bufferless matrix objects and attach the provided matrix pointers
	   to those matrix objects. */ \
	bli_obj_create_with_attached_buffer( dt, m_a, n_a, a, rs_a, cs_a, &ao ); \
	bli_obj_create_with_attached_buffer( dt, k,   1,   d, incd, k,    &dd ); \
	bli_obj_create_with_attached_buffer( dt, m_b, n_b, b, rs_b, cs_b, &bo ); \
	bli_obj_create_with_attached_buffer( dt, m,   n,   c, rs_c, cs_c, &co ); \
\
	/* Set the transposition/conjugation properties of the objects for matrices
	   A and B. */ \
	bli_obj_set_conjtrans( transa, &ao ); \
	bli_obj_set_conjtrans( transb, &bo ); \
\
	/* Call the object interface. */ \
	PASTECH(bao_,opname) \
	( \
	  &alphao, \
	  &ao, \
	  &dd, \
	  &bo, \
	  &betao, \
	  &co  \
	); \
}

//INSERT_GENTFUNC_BASIC0( gemmd )
GENTFUNC( float,    s, gemmd )
GENTFUNC( double,   d, gemmd )
GENTFUNC( scomplex, c, gemmd )
GENTFUNC( dcomplex, z, gemmd )

