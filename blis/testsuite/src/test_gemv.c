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
static char*     op_str                    = "gemv";
static char*     o_types                   = "mvv"; // a x y
static char*     p_types                   = "hc";  // transa conjx
static thresh_t  thresh[BLIS_NUM_FP_TYPES] = { { 1e-04, 1e-05 },   // warn, pass for s
                                               { 1e-04, 1e-05 },   // warn, pass for c
                                               { 1e-13, 1e-14 },   // warn, pass for d
                                               { 1e-13, 1e-14 } }; // warn, pass for z

// Local prototypes.
void libblis_test_gemv_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     );

void libblis_test_gemv_experiment
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

void libblis_test_gemv_impl
     (
       iface_t   iface,
       obj_t*    alpha,
       obj_t*    a,
       obj_t*    x,
       obj_t*    beta,
       obj_t*    y
     );

void libblis_test_gemv_check
     (
       test_params_t* params,
       obj_t*         kappa,
       obj_t*         alpha,
       obj_t*         a,
       obj_t*         x,
       obj_t*         beta,
       obj_t*         y,
       obj_t*         y_orig,
       double*        resid
     );



void libblis_test_gemv_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     )
{
	libblis_test_randv( tdata, params, &(op->ops->randv) );
	libblis_test_normfv( tdata, params, &(op->ops->normfv) );
	libblis_test_subv( tdata, params, &(op->ops->subv) );
	libblis_test_copyv( tdata, params, &(op->ops->copyv) );
	libblis_test_scalv( tdata, params, &(op->ops->scalv) );
	libblis_test_copym( tdata, params, &(op->ops->copym) );
}



void libblis_test_gemv
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
	     libblis_test_l2_is_disabled( op ) ) return;

	// Call dependencies first.
	if ( TRUE ) libblis_test_gemv_deps( tdata, params, op );

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
		                        libblis_test_gemv_experiment );
	}
}



void libblis_test_gemv_experiment
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

	num_t        datatype;

	dim_t        m, n;

	trans_t      transa;
	conj_t       conjx;

	obj_t        kappa;
	obj_t        alpha, a, x, beta, y;
	obj_t        y_save;


	// Use the datatype of the first char in the datatype combination string.
	bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

	// Map the dimension specifier to actual dimensions.
	m = libblis_test_get_dim_from_prob_size( op->dim_spec[0], p_cur );
	n = libblis_test_get_dim_from_prob_size( op->dim_spec[1], p_cur );

	// Map parameter characters to BLIS constants.
	bli_param_map_char_to_blis_trans( pc_str[0], &transa );
	bli_param_map_char_to_blis_conj( pc_str[1], &conjx );

	// Create test scalars.
	bli_obj_scalar_init_detached( datatype, &kappa );
	bli_obj_scalar_init_detached( datatype, &alpha );
	bli_obj_scalar_init_detached( datatype, &beta );

	// Create test operands (vectors and/or matrices).
	libblis_test_mobj_create( params, datatype, transa,
	                          sc_str[0], m, n, &a );
	libblis_test_vobj_create( params, datatype,
	                          sc_str[1], n,    &x );
	libblis_test_vobj_create( params, datatype,
	                          sc_str[2], m,    &y );
	libblis_test_vobj_create( params, datatype,
	                          sc_str[2], m,    &y_save );

	// Set alpha and beta.
	if ( bli_obj_is_real( &y ) )
	{
		bli_setsc(  2.0,  0.0, &alpha );
		bli_setsc( -1.0,  0.0, &beta );
	}
	else
	{
		bli_setsc(  1.0,  2.0, &alpha );
		bli_setsc(  1.0, -1.0, &beta );
	}

	// Initialize diagonal of matrix A.
	bli_setsc( 2.0, -1.0, &kappa );
	bli_setm( &BLIS_ZERO, &a );
	bli_setd( &kappa, &a );

	// Randomize x and y, and save y.
	libblis_test_vobj_randomize( params, TRUE, &x );
	libblis_test_vobj_randomize( params, TRUE, &y );
	bli_copyv( &y, &y_save );

	// Apply the parameters.
	bli_obj_set_conjtrans( transa, &a );
	bli_obj_set_conj( conjx, &x );

	// Repeat the experiment n_repeats times and record results. 
	for ( i = 0; i < n_repeats; ++i )
	{
		bli_copym( &y_save, &y );

		time = bli_clock();

		libblis_test_gemv_impl( iface, &alpha, &a, &x, &beta, &y );

		time_min = bli_clock_min_diff( time_min, time );
	}

	// Estimate the performance of the best experiment repeat.
	*perf = ( 2.0 * m * n ) / time_min / FLOPS_PER_UNIT_PERF;
	if ( bli_obj_is_complex( &y ) ) *perf *= 4.0;

	// Perform checks.
	libblis_test_gemv_check( params, &kappa, &alpha, &a, &x, &beta, &y, &y_save, resid );

	// Zero out performance and residual if output vector is empty.
	libblis_test_check_empty_problem( &y, perf, resid );

	// Free the test objects.
	bli_obj_free( &a );
	bli_obj_free( &x );
	bli_obj_free( &y );
	bli_obj_free( &y_save );
}



void libblis_test_gemv_impl
     (
       iface_t   iface,
       obj_t*    alpha,
       obj_t*    a,
       obj_t*    x,
       obj_t*    beta,
       obj_t*    y
     )
{
	switch ( iface )
	{
		case BLIS_TEST_SEQ_FRONT_END:
		bli_gemv( alpha, a, x, beta, y );
		break;

		default:
		libblis_test_printf_error( "Invalid interface type.\n" );
	}
}



void libblis_test_gemv_check
     (
       test_params_t* params,
       obj_t*         kappa,
       obj_t*         alpha,
       obj_t*         a,
       obj_t*         x,
       obj_t*         beta,
       obj_t*         y,
       obj_t*         y_orig,
       double*        resid
     )
{
	num_t  dt      = bli_obj_dt( y );
	num_t  dt_real = bli_obj_dt_proj_to_real( y );

	conj_t conja   = bli_obj_conj_status( a );

	dim_t  n_x     = bli_obj_vector_dim( x );
	dim_t  m_y     = bli_obj_vector_dim( y );

	dim_t  min_m_n = bli_min( m_y, n_x );

	obj_t  x_temp, y_temp;
	obj_t  kappac, norm;
	obj_t  xT_temp, yT_temp, yT;

	double junk;

	//
	// Pre-conditions:
	// - a is initialized to kappa along the diagonal.
	// - x is randomized.
	// - y_orig is randomized.
	// Note:
	// - alpha, beta, and kappa should have non-zero imaginary components in
	//   the complex cases in order to more fully exercise the implementation.
	//
	// Under these conditions, we assume that the implementation for
	//
	//   y := beta * y_orig + alpha * transa(A) * conjx(x)
	//
	// is functioning correctly if
	//
	//   normfv( y - z )
	//
	// is negligible, where
	//
	//   z = beta * y_orig + alpha * conja(kappa) * x
	//

	bli_obj_scalar_init_detached_copy_of( dt, conja, kappa, &kappac );
	bli_obj_scalar_init_detached( dt_real, &norm );

	bli_obj_create( dt, n_x, 1, 0, 0, &x_temp );
	bli_obj_create( dt, m_y, 1, 0, 0, &y_temp );

	bli_copyv( x,      &x_temp );
	bli_copyv( y_orig, &y_temp );

	bli_acquire_vpart_f2b( BLIS_SUBPART1, 0, min_m_n, 
	                       &x_temp, &xT_temp );
	bli_acquire_vpart_f2b( BLIS_SUBPART1, 0, min_m_n, 
	                       &y_temp, &yT_temp );
	bli_acquire_vpart_f2b( BLIS_SUBPART1, 0, min_m_n, 
	                       y, &yT );

	bli_scalv( &kappac, &xT_temp );
	bli_scalv( beta, &yT_temp );
	bli_axpyv( alpha, &xT_temp, &yT_temp );

	bli_subv( &yT_temp, &yT );
	bli_normfv( &yT, &norm );
	bli_getsc( &norm, resid, &junk );

	bli_obj_free( &x_temp );
	bli_obj_free( &y_temp );
}

