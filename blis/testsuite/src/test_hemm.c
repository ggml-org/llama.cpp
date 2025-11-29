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
static char*     op_str                    = "hemm";
static char*     o_types                   = "mmm";   // a b c
static char*     p_types                   = "such";  // side uploa conja transb
static thresh_t  thresh[BLIS_NUM_FP_TYPES] = { { 1e-04, 1e-05 },   // warn, pass for s
                                               { 1e-04, 1e-05 },   // warn, pass for c
                                               { 1e-13, 1e-14 },   // warn, pass for d
                                               { 1e-13, 1e-14 } }; // warn, pass for z

// Local prototypes.
void libblis_test_hemm_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     );

void libblis_test_hemm_experiment
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

void libblis_test_hemm_impl
     (
       iface_t   iface,
       side_t    side,
       obj_t*    alpha,
       obj_t*    a,
       obj_t*    b,
       obj_t*    beta,
       obj_t*    c
     );

void libblis_test_hemm_check
     (
       test_params_t* params,
       side_t         side,
       obj_t*         alpha,
       obj_t*         a,
       obj_t*         b,
       obj_t*         beta,
       obj_t*         c,
       obj_t*         c_orig,
       double*        resid
     );



void libblis_test_hemm_deps
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
	libblis_test_hemv( tdata, params, &(op->ops->hemv) );
}



void libblis_test_hemm
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
	if ( TRUE ) libblis_test_hemm_deps( tdata, params, op );

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
		                        libblis_test_hemm_experiment );
	}
}



void libblis_test_hemm_experiment
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

	dim_t        m, n;
	dim_t        mn_side;

	side_t       side;
	uplo_t       uploa;
	conj_t       conja;
	trans_t      transb;

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
	n = libblis_test_get_dim_from_prob_size( op->dim_spec[1], p_cur );

	// Map parameter characters to BLIS constants.
	bli_param_map_char_to_blis_side( pc_str[0], &side );
	bli_param_map_char_to_blis_uplo( pc_str[1], &uploa );
	bli_param_map_char_to_blis_conj( pc_str[2], &conja );
	bli_param_map_char_to_blis_trans( pc_str[3], &transb );

	// Create test scalars.
	bli_obj_scalar_init_detached( dt_alpha, &alpha );
	bli_obj_scalar_init_detached( dt_beta, &beta );

	// Create test operands (vectors and/or matrices).
	bli_set_dim_with_side( side, m, n, &mn_side );
	libblis_test_mobj_create( params, dt_a, BLIS_NO_TRANSPOSE,
	                          sc_str[1], mn_side, mn_side, &a );
	libblis_test_mobj_create( params, dt_b, transb,
	                          sc_str[2], m,       n,       &b );
	libblis_test_mobj_create( params, dt_c, BLIS_NO_TRANSPOSE,
	                          sc_str[0], m,       n,       &c );
	libblis_test_mobj_create( params, dt_c, BLIS_NO_TRANSPOSE,
	                          sc_str[0], m,       n,       &c_save );

	// Set the computation precision of C.
	bli_obj_set_comp_prec( bli_dt_prec( dt_comp ), &c );

	// Set alpha and beta.
	{
		bli_setsc(  2.0,  0.2, &alpha );
		bli_setsc(  1.2,  0.5, &beta );
	}

	// Set the structure and uplo properties of A.
	bli_obj_set_struc( BLIS_HERMITIAN, &a );
	bli_obj_set_uplo( uploa, &a );

	// Randomize A, make it densely Hermitian, and zero the unstored triangle
	// to ensure the implementation reads only from the stored region.
	libblis_test_mobj_randomize( params, TRUE, &a );
	bli_mkherm( &a );
	bli_mktrim( &a );

	// Randomize B and C, and save C.
	libblis_test_mobj_randomize( params, TRUE, &b );
	libblis_test_mobj_randomize( params, TRUE, &c );
	bli_copym( &c, &c_save );

	// Apply the remaining parameters.
	bli_obj_set_conj( conja, &a );
	bli_obj_set_conjtrans( transb, &b );

	// Repeat the experiment n_repeats times and record results.
	for ( i = 0; i < n_repeats; ++i )
	{
		bli_copym( &c_save, &c );

		time = bli_clock();

		libblis_test_hemm_impl( iface, side, &alpha, &a, &b, &beta, &c );

		time_min = bli_clock_min_diff( time_min, time );
	}

	// Estimate the performance of the best experiment repeat.
	if ( bli_is_left( side ) )
	{
		*perf = libblis_test_l3_flops( BLIS_HEMM, &a, &b, &c ) / time_min / FLOPS_PER_UNIT_PERF;
	}
	else
	{
		*perf = libblis_test_l3_flops( BLIS_HEMM, &b, &a, &c ) / time_min / FLOPS_PER_UNIT_PERF;
	}

	// Perform checks.
	libblis_test_hemm_check( params, side, &alpha, &a, &b, &beta, &c, &c_save, resid );

	// Zero out performance and residual if output matrix is empty.
	libblis_test_check_empty_problem( &c, perf, resid );

	// Free the test objects.
	bli_obj_free( &a );
	bli_obj_free( &b );
	bli_obj_free( &c );
	bli_obj_free( &c_save );
}



void libblis_test_hemm_impl
     (
       iface_t   iface,
       side_t    side,
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
		bli_hemm( side, alpha, a, b, beta, c );
		break;

		default:
		libblis_test_printf_error( "Invalid interface type.\n" );
	}
}



void libblis_test_hemm_check
     (
       test_params_t* params,
       side_t         side,
       obj_t*         alpha,
       obj_t*         a,
       obj_t*         b,
       obj_t*         beta,
       obj_t*         c,
       obj_t*         c_orig,
       double*        resid
     )
{
	uplo_t uploa   = bli_obj_uplo( a );
	if ( bli_obj_has_trans( a ) )
		bli_toggle_uplo( &uploa );

	num_t  dt_real = bli_obj_comp_prec( c ) | BLIS_REAL;
	num_t  dt_comp = bli_obj_comp_prec( c ) | BLIS_COMPLEX;
	num_t  dt;

	dim_t  m       = bli_obj_length( c );
	dim_t  n       = bli_obj_width( c );
	dim_t  mn_side = bli_obj_length( a );

	obj_t  norm;
	obj_t  t, v, w, z;

	double junk;

	//
	// Pre-conditions:
	// - a is randomized and Hermitian.
	// - b is randomized.
	// - c_orig is randomized.
	// Note:
	// - alpha and beta should have non-zero imaginary components in the
	//   complex cases in order to more fully exercise the implementation.
	//
	// Under these conditions, we assume that the implementation for
	//
	//   C := beta * C_orig + alpha * conja(A) * transb(B)    (side = left)
	//   C := beta * C_orig + alpha * transb(B) * conja(A)    (side = right)
	//
	// is functioning correctly if
	//
	//   normfv( v - z )
	//
	// is negligible, where
	//
	//   v = C * t
	//
	//   z = ( beta * C_orig + alpha * conja(A) * transb(B) ) * t     (side = left)
	//     = beta * C_orig * t + alpha * conja(A) * transb(B) * t
	//     = beta * C_orig * t + alpha * conja(A) * w
	//     = beta * C_orig * t + z
	//
	//   z = ( beta * C_orig + alpha * transb(B) * conja(A) ) * t     (side = right)
	//     = beta * C_orig * t + alpha * transb(B) * conja(A) * t
	//     = beta * C_orig * t + alpha * transb(B) * w
	//     = beta * C_orig * t + z

	// Compute our reference checksum in the real domain if all operands
	// are real, and in the complex domain otherwise.
	if ( bli_obj_is_real( a ) &&
	     bli_obj_is_real( b ) &&
	     bli_obj_is_real( c ) ) dt = dt_real;
	else                        dt = dt_comp;

	// Project a, b, and c into the appropriate domain and computational
	// precision, and then proceed with the checking accordingly.

	obj_t a2, b2, c2, c0;

	bli_obj_scalar_init_detached( dt_real, &norm );

	if ( bli_is_left( side ) )
	{
		bli_obj_create( dt, n, 1, 0, 0, &t );
		bli_obj_create( dt, m, 1, 0, 0, &v );
		bli_obj_create( dt, m, 1, 0, 0, &w );
		bli_obj_create( dt, m, 1, 0, 0, &z );
	}
	else // else if ( bli_is_right( side ) )
	{
		bli_obj_create( dt, n, 1, 0, 0, &t );
		bli_obj_create( dt, m, 1, 0, 0, &v );
		bli_obj_create( dt, n, 1, 0, 0, &w );
		bli_obj_create( dt, m, 1, 0, 0, &z );
	}

	libblis_test_vobj_randomize( params, TRUE, &t );

	// We need to zero out the imaginary part of t in order for our
	// checks to work in all cases. Otherwise, the imaginary parts
	// could affect intermediate products, depending on the order that
	// they are executed.
	bli_setiv( &BLIS_ZERO, &t );

	// Create type-casted equivalents of a, b, c_orig, and c.
	bli_obj_create( dt, mn_side, mn_side, 0, 0, &a2 );
	bli_obj_create( dt, m, n, 0, 0, &b2 );
	bli_obj_create( dt, m, n, 0, 0, &c2 );
	bli_obj_create( dt, m, n, 0, 0, &c0 );
	bli_obj_set_struc( BLIS_HERMITIAN, &a2 );
	bli_obj_set_uplo( uploa, &a2 );

	// Cast a, b, c_orig, and c into the datatype of our temporary objects.
	bli_castm( a,      &a2 );
	bli_castm( b,      &b2 );
	bli_castm( c_orig, &c2 );
	bli_castm( c,      &c0 );

	bli_gemv( &BLIS_ONE, &c0, &t, &BLIS_ZERO, &v );

	if ( bli_is_left( side ) )
	{
		bli_gemv( &BLIS_ONE, &b2, &t, &BLIS_ZERO, &w );
		bli_hemv( alpha, &a2, &w, &BLIS_ZERO, &z );
	}
	else // else if ( bli_is_right( side ) )
	{
		bli_hemv( &BLIS_ONE, &a2, &t, &BLIS_ZERO, &w );
		bli_gemv( alpha, &b2, &w, &BLIS_ZERO, &z );
	}

	bli_gemv( beta, &c2, &t, &BLIS_ONE, &z );
	if ( bli_obj_is_real( c ) ) bli_setiv( &BLIS_ZERO, &z );

	bli_subv( &z, &v );
	bli_normfv( &v, &norm );
	bli_getsc( &norm, resid, &junk );

	bli_obj_free( &t );
	bli_obj_free( &v );
	bli_obj_free( &w );
	bli_obj_free( &z );

	bli_obj_free( &a2 );
	bli_obj_free( &b2 );
	bli_obj_free( &c2 );
	bli_obj_free( &c0 );
}

