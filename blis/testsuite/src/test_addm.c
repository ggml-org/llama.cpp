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
static char*     op_str                    = "addm";
static char*     o_types                   = "mm";  // x y
static char*     p_types                   = "h";   // transx
static thresh_t  thresh[BLIS_NUM_FP_TYPES] = { { 1e-04, 1e-05 },   // warn, pass for s
                                               { 1e-04, 1e-05 },   // warn, pass for c
                                               { 1e-13, 1e-14 },   // warn, pass for d
                                               { 1e-13, 1e-14 } }; // warn, pass for z

// Local prototypes.
void libblis_test_addm_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     );

void libblis_test_addm_experiment
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

void libblis_test_addm_impl
     (
       iface_t   iface,
       obj_t*    x,
       obj_t*    y
     );

void libblis_test_addm_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         beta,
       obj_t*         x,
       obj_t*         y,
       double*        resid
     );



void libblis_test_addm_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     )
{
	libblis_test_setm( tdata, params, &(op->ops->setm) );
	libblis_test_normfm( tdata, params, &(op->ops->normfm) );
}



void libblis_test_addm
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
	     libblis_test_l1m_is_disabled( op ) ) return;

	// Call dependencies first.
	if ( TRUE ) libblis_test_addm_deps( tdata, params, op );

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
		                        libblis_test_addm_experiment );
	}
}



void libblis_test_addm_experiment
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
	double       time_min  = DBL_MAX;
	double       time;

	num_t        datatype;

	dim_t        m, n;

	trans_t      transx;

	obj_t        alpha, beta;
	obj_t        x, y;


	// Use the datatype of the first char in the datatype combination string.
	bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

	// Map the dimension specifier to actual dimensions.
	m = libblis_test_get_dim_from_prob_size( op->dim_spec[0], p_cur );
	n = libblis_test_get_dim_from_prob_size( op->dim_spec[1], p_cur );

	// Map parameter characters to BLIS constants.
	bli_param_map_char_to_blis_trans( pc_str[0], &transx );

	// Create test scalars.
	bli_obj_scalar_init_detached( datatype, &alpha );
	bli_obj_scalar_init_detached( datatype, &beta );

	// Create test operands (vectors and/or matrices).
	libblis_test_mobj_create( params, datatype, transx,
	                          sc_str[0], m, n, &x );
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          sc_str[1], m, n, &y );

	// Initialize alpha and beta.
	bli_setsc( -1.0, -1.0, &alpha );
	bli_setsc(  3.0,  3.0, &beta );

	// Randomize x.
	bli_setm( &alpha, &x );
	bli_setm( &beta,  &y );

	// Apply the parameters.
	bli_obj_set_conjtrans( transx, &x );

	// Disable repeats since bli_copym() is not yet tested.
	//for ( i = 0; i < n_repeats; ++i )
	{
		time = bli_clock();

		libblis_test_addm_impl( iface, &x, &y );

		time_min = bli_clock_min_diff( time_min, time );
	}

	// Estimate the performance of the best experiment repeat.
	*perf = ( 1.0 * m * n ) / time_min / FLOPS_PER_UNIT_PERF;
	if ( bli_obj_is_complex( &x ) ) *perf *= 2.0;

	// Perform checks.
	libblis_test_addm_check( params, &alpha, &beta, &x, &y, resid );

	// Zero out performance and residual if output matrix is empty.
	libblis_test_check_empty_problem( &y, perf, resid );

	// Free the test objects.
	bli_obj_free( &x );
	bli_obj_free( &y );
}



void libblis_test_addm_impl
     (
       iface_t   iface,
       obj_t*    x,
       obj_t*    y
     )
{
	switch ( iface )
	{
		case BLIS_TEST_SEQ_FRONT_END:
		bli_addm( x, y );
		break;

		default:
		libblis_test_printf_error( "Invalid interface type.\n" );
	}
}



void libblis_test_addm_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         beta,
       obj_t*         x,
       obj_t*         y,
       double*        resid
     )
{
	num_t  dt      = bli_obj_dt( y );
	num_t  dt_real = bli_obj_dt_proj_to_real( y );
	dim_t  m       = bli_obj_length( y );
	dim_t  n       = bli_obj_width( y );

	conj_t conjx   = bli_obj_conj_status( x );

	obj_t  aplusb;
	obj_t  alpha_conj;
	obj_t  norm_r, m_r, n_r, temp_r;

	double junk;

	//
	// Pre-conditions:
	// - x is set to alpha.
	// - y_orig is set to beta.
	// Note:
	// - alpha and beta should have non-zero imaginary components in the
	//   complex cases in order to more fully exercise the implementation.
	//
	// Under these conditions, we assume that the implementation for
	//
	//   y := y_orig + conjx(x)
	//
	// is functioning correctly if
	//
	//   normfm(y) - sqrt( absqsc( beta + conjx(alpha) ) * m * n )
	//
	// is negligible.
	//

	bli_obj_scalar_init_detached( dt,      &aplusb );
	bli_obj_scalar_init_detached( dt_real, &temp_r );
	bli_obj_scalar_init_detached( dt_real, &norm_r );
	bli_obj_scalar_init_detached( dt_real, &m_r );
	bli_obj_scalar_init_detached( dt_real, &n_r );

	bli_obj_scalar_init_detached_copy_of( dt, conjx, alpha, &alpha_conj );

	bli_normfm( y, &norm_r );

	bli_copysc( beta, &aplusb );
	bli_addsc( &alpha_conj, &aplusb );

	bli_setsc( ( double )m, 0.0, &m_r );
	bli_setsc( ( double )n, 0.0, &n_r );

	bli_absqsc( &aplusb, &temp_r );
	bli_mulsc( &m_r, &temp_r );
	bli_mulsc( &n_r, &temp_r );
	bli_sqrtsc( &temp_r, &temp_r );
	bli_subsc( &temp_r, &norm_r );

	bli_getsc( &norm_r, resid, &junk );
}

