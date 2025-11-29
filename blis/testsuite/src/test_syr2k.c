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

#include "blis.h"
#include "test_libblis.h"


// Static variables.
static char*     op_str                    = "syr2k";
static char*     o_types                   = "mmm";  // a b c
static char*     p_types                   = "uhh";  // uploc transa transb
static thresh_t  thresh[BLIS_NUM_FP_TYPES] = { { 1e-04, 1e-05 },   // warn, pass for s
                                               { 1e-04, 1e-05 },   // warn, pass for c
                                               { 1e-13, 1e-14 },   // warn, pass for d
                                               { 1e-13, 1e-14 } }; // warn, pass for z

// Local prototypes.
void libblis_test_syr2k_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     );

void libblis_test_syr2k_experiment
     (
       test_params_t* params,
       test_op_t*     op,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       unsigned int   p_cur,
       double*        perf,
       double*        resid
     );

void libblis_test_syr2k_impl
     (
       iface_t   iface,
       obj_t*    alpha,
       obj_t*    a,
       obj_t*    b,
       obj_t*    beta,
       obj_t*    c
     );

void libblis_test_syr2k_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         a,
       obj_t*         b,
       obj_t*         beta,
       obj_t*         c,
       obj_t*         c_orig,
       double*        resid
     );



void libblis_test_syr2k_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     )
{
	libblis_test_randv( tdata, params, &(op->ops->randv) );
	libblis_test_randm( tdata, params, &(op->ops->randm) );
	libblis_test_setv( tdata, params, &(op->ops->setv) );
	libblis_test_normfv( tdata, params, &(op->ops->normfv) );
	libblis_test_subv( tdata, params, &(op->ops->subv) );
	libblis_test_scalv( tdata, params, &(op->ops->scalv) );
	libblis_test_copym( tdata, params, &(op->ops->copym) );
	libblis_test_scalm( tdata, params, &(op->ops->scalm) );
	libblis_test_gemv( tdata, params, &(op->ops->gemv) );
	libblis_test_symv( tdata, params, &(op->ops->symv) );
}



void libblis_test_syr2k
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     )
{

	// Return early if this test has already been done.
	if ( libblis_test_op_is_done( op ) ) return;

	// Return early if operation is disabled.
	if ( libblis_test_op_is_disabled( op ) ||
	     libblis_test_l3_is_disabled( op ) ) return;

	// Call dependencies first.
	if ( TRUE ) libblis_test_syr2k_deps( tdata, params, op );

	// Execute the test driver for each implementation requested.
	//if ( op->front_seq == ENABLE )
	{
		libblis_test_op_driver( tdata,
		                        params,
		                        op,
		                        BLIS_TEST_SEQ_FRONT_END,
		                        op_str,
		                        p_types,
		                        o_types,
		                        thresh,
		                        libblis_test_syr2k_experiment );
	}
}



void libblis_test_syr2k_experiment
     (
       test_params_t* params,
       test_op_t*     op,
       iface_t        iface,
       char*          dc_str,
       char*          pc_str,
       char*          sc_str,
       unsigned int   p_cur,
       double*        perf,
       double*        resid
     )
{
	unsigned int n_repeats = params->n_repeats;
	unsigned int i;

	double       time_min  = DBL_MAX;
	double       time;

	num_t        dt_a, dt_b, dt_c, dt_alpha, dt_beta, dt_comp;

	dim_t        m, k;

	uplo_t       uploc;
	trans_t      transa, transb;

	obj_t        alpha, a, b, beta, c;
	obj_t        c_save;


	// Use the datatype of the first char in the datatype combination string.
	bli_param_map_char_to_blis_dt( dc_str[0], &dt_c );
	bli_param_map_char_to_blis_dt( dc_str[1], &dt_a );
	bli_param_map_char_to_blis_dt( dc_str[2], &dt_b );
	bli_param_map_char_to_blis_dt( dc_str[3], &dt_alpha );
	bli_param_map_char_to_blis_dt( dc_str[4], &dt_beta );
	bli_param_map_char_to_blis_dt( dc_str[5], &dt_comp );

	// Map the dimension specifier to actual dimensions.
	m = libblis_test_get_dim_from_prob_size( op->dim_spec[0], p_cur );
	k = libblis_test_get_dim_from_prob_size( op->dim_spec[1], p_cur );

	// Map parameter characters to BLIS constants.
	bli_param_map_char_to_blis_uplo( pc_str[0], &uploc );
	bli_param_map_char_to_blis_trans( pc_str[1], &transa );
	bli_param_map_char_to_blis_trans( pc_str[2], &transb );

	// Create test scalars.
	bli_obj_scalar_init_detached( dt_alpha, &alpha );
	bli_obj_scalar_init_detached( dt_beta, &beta );

	// Create test operands (vectors and/or matrices).
	libblis_test_mobj_create( params, dt_a, transa,
	                          sc_str[1], m, k, &a );
	libblis_test_mobj_create( params, dt_b, transb,
	                          sc_str[2], m, k, &b );
	libblis_test_mobj_create( params, dt_c, BLIS_NO_TRANSPOSE,
	                          sc_str[0], m, m, &c );
	libblis_test_mobj_create( params, dt_c, BLIS_NO_TRANSPOSE,
	                          sc_str[0], m, m, &c_save );

	// Set the computation precision of C.
	bli_obj_set_comp_prec( bli_dt_prec( dt_comp ), &c );

	// Set alpha and beta.
	{
		bli_setsc(  2.0,  0.2, &alpha );
		bli_setsc(  1.2,  0.5, &beta );
	}

	// Randomize A and B.
	libblis_test_mobj_randomize( params, TRUE, &a );
	libblis_test_mobj_randomize( params, TRUE, &b );

	// Set the structure and uplo properties of C.
	bli_obj_set_struc( BLIS_SYMMETRIC, &c );
	bli_obj_set_uplo( uploc, &c );

	// Randomize A, make it densely symmetric, and zero the unstored triangle
	// to ensure the implementation is reads only from the stored region.
	libblis_test_mobj_randomize( params, TRUE, &c );
	bli_mksymm( &c );
	bli_mktrim( &c );

	// Save C and set its structure and uplo properties.
	bli_obj_set_struc( BLIS_SYMMETRIC, &c_save );
	bli_obj_set_uplo( uploc, &c_save );
	bli_copym( &c, &c_save );

	// Apply the remaining parameters.
	bli_obj_set_conjtrans( transa, &a );
	bli_obj_set_conjtrans( transb, &b );

	// Repeat the experiment n_repeats times and record results.
	for ( i = 0; i < n_repeats; ++i )
	{
		bli_copym( &c_save, &c );

		time = bli_clock();

		libblis_test_syr2k_impl( iface, &alpha, &a, &b, &beta, &c );

		time_min = bli_clock_min_diff( time_min, time );
	}

	obj_t bt;
	bli_obj_alias_with_trans( BLIS_TRANSPOSE, &b, &bt );

	// Estimate the performance of the best experiment repeat.
	*perf = libblis_test_l3_flops( BLIS_SYR2K, &a, &bt, &c ) / time_min / FLOPS_PER_UNIT_PERF;

	// Perform checks.
	libblis_test_syr2k_check( params, &alpha, &a, &b, &beta, &c, &c_save, resid );

	// Zero out performance and residual if output matrix is empty.
	libblis_test_check_empty_problem( &c, perf, resid );

	// Free the test objects.
	bli_obj_free( &a );
	bli_obj_free( &b );
	bli_obj_free( &c );
	bli_obj_free( &c_save );
}



void libblis_test_syr2k_impl
     (
       iface_t   iface,
       obj_t*    alpha,
       obj_t*    a,
       obj_t*    b,
       obj_t*    beta,
       obj_t*    c
     )
{
	switch ( iface )
	{
		case BLIS_TEST_SEQ_FRONT_END:
		bli_syr2k( alpha, a, b, beta, c );
		break;

		default:
		libblis_test_printf_error( "Invalid interface type.\n" );
	}
}



void libblis_test_syr2k_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         a,
       obj_t*         b,
       obj_t*         beta,
       obj_t*         c,
       obj_t*         c_orig,
       double*        resid
     )
{
	uplo_t uploc   = bli_obj_uplo( c );
	num_t  dt_real = bli_obj_comp_prec( c ) | BLIS_REAL;
	num_t  dt_comp = bli_obj_comp_prec( c ) | BLIS_COMPLEX;
	num_t  dt;

	dim_t  m       = bli_obj_length( c );
	dim_t  k       = bli_obj_width_after_trans( a );

	obj_t  norm;
	obj_t  t, v, w1, w2, z;

	double junk;

	//
	// Pre-conditions:
	// - a is randomized.
	// - b is randomized.
	// - c_orig is randomized and symmetric.
	// Note:
	// - alpha and beta should have non-zero imaginary components in the
	//   complex cases in order to more fully exercise the implementation.
	//
	// Under these conditions, we assume that the implementation for
	//
	//   C := beta * C_orig + alpha * transa(A) * transb(B)^T + alpha * transb(B) * transa(A)^T
	//
	// is functioning correctly if
	//
	//   normfv( v - z )
	//
	// is negligible, where
	//
	//   v = C * t
	//   z = ( beta * C_orig + alpha * transa(A) * transb(B)^T + alpha * transb(B) * transa(A)^T ) * t
	//     = beta * C_orig * t + alpha * transa(A) * transb(B)^T * t + alpha * transb(B) * transa(A)^T * t
	//     = beta * C_orig * t + alpha * transa(A) * transb(B)^T * t + alpha * transb(B) * w2
	//     = beta * C_orig * t + alpha * transa(A) * w1              + alpha * transb(B) * w2
	//     = beta * C_orig * t + alpha * transa(A) * w1              + z
	//     = beta * C_orig * t + z
	//

	// Compute our reference checksum in the real domain if all operands
	// are real, and in the complex domain otherwise.
	if ( bli_obj_is_real( a ) &&
	     bli_obj_is_real( b ) &&
	     bli_obj_is_real( c ) ) dt = dt_real;
	else                        dt = dt_comp;

	// Project a, b, and c into the appropriate domain and computational
	// precision, and then proceed with the checking accordingly.

	obj_t a2, b2, at2, bt2, c2, c0;

	bli_obj_scalar_init_detached( dt_real, &norm );

	bli_obj_create( dt, m, 1, 0, 0, &t );
	bli_obj_create( dt, m, 1, 0, 0, &v );
	bli_obj_create( dt, k, 1, 0, 0, &w1 );
	bli_obj_create( dt, k, 1, 0, 0, &w2 );
	bli_obj_create( dt, m, 1, 0, 0, &z );

	libblis_test_vobj_randomize( params, TRUE, &t );

	// We need to zero out the imaginary part of t in order for our
	// checks to work in all cases. Otherwise, the imaginary parts
	// could affect intermediate products, depending on the order that
	// they are executed.
	bli_setiv( &BLIS_ZERO, &t );

	// Create type-casted equivalents of a, b, c_orig, and c.
	bli_obj_create( dt, m, k, 0, 0, &a2 );
	bli_obj_create( dt, k, m, 0, 0, &b2 );
	bli_obj_create( dt, m, m, 0, 0, &c2 );
	bli_obj_create( dt, m, m, 0, 0, &c0 );
	bli_obj_set_struc( BLIS_SYMMETRIC, &c2 );
	bli_obj_set_struc( BLIS_SYMMETRIC, &c0 );
	bli_obj_set_uplo( uploc, &c2 );
	bli_obj_set_uplo( uploc, &c0 );

	// Cast a, b, c_orig, and c into the datatype of our temporary objects.
	bli_castm( a,      &a2 );
	bli_castm( b,      &b2 );
	bli_castm( c_orig, &c2 );
	bli_castm( c,      &c0 );

	bli_obj_alias_with_trans( BLIS_TRANSPOSE, &a2, &at2 );
	bli_obj_alias_with_trans( BLIS_TRANSPOSE, &b2, &bt2 );

	bli_symv( &BLIS_ONE, &c0, &t, &BLIS_ZERO, &v );

	bli_gemv( &BLIS_ONE, &at2, &t, &BLIS_ZERO, &w2 );
	bli_gemv( &BLIS_ONE, &bt2, &t, &BLIS_ZERO, &w1 );
	bli_gemv( alpha, &a2, &w1, &BLIS_ZERO, &z );
	bli_gemv( alpha, &b2, &w2, &BLIS_ONE, &z );
	bli_symv( beta, &c2, &t, &BLIS_ONE, &z );
	if ( bli_obj_is_real( c ) ) bli_setiv( &BLIS_ZERO, &z );

	bli_subv( &z, &v );
	bli_normfv( &v, &norm );
	bli_getsc( &norm, resid, &junk );

	bli_obj_free( &t );
	bli_obj_free( &v );
	bli_obj_free( &w1 );
	bli_obj_free( &w2 );
	bli_obj_free( &z );

	bli_obj_free( &a2 );
	bli_obj_free( &b2 );
	bli_obj_free( &c2 );
	bli_obj_free( &c0 );
}

