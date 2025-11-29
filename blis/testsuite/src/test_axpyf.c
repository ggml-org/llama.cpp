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
static char*     op_str                    = "axpyf";
static char*     o_types                   = "mvv";  // A x y
static char*     p_types                   = "cc";   // conja conjx
static thresh_t  thresh[BLIS_NUM_FP_TYPES] = { { 1e-04, 1e-05 },   // warn, pass for s
                                               { 1e-04, 1e-05 },   // warn, pass for c
                                               { 1e-13, 1e-14 },   // warn, pass for d
                                               { 1e-13, 1e-14 } }; // warn, pass for z

// Local prototypes.
void libblis_test_axpyf_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     );

void libblis_test_axpyf_experiment
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

void libblis_test_axpyf_impl
     (
       iface_t   iface,
       obj_t*    alpha,
       obj_t*    a,
       obj_t*    x,
       obj_t*    y,
       cntx_t*   cntx
     );

void libblis_test_axpyf_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         a,
       obj_t*         x,
       obj_t*         y,
       obj_t*         y_orig,
       double*        resid
     );



void libblis_test_axpyf_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     )
{
	libblis_test_randv( tdata, params, &(op->ops->randv) );
	libblis_test_randm( tdata, params, &(op->ops->randm) );
	libblis_test_normfv( tdata, params, &(op->ops->normfv) );
	libblis_test_subv( tdata, params, &(op->ops->subv) );
	libblis_test_copyv( tdata, params, &(op->ops->copyv) );
	libblis_test_axpyv( tdata, params, &(op->ops->axpyv) );
}



void libblis_test_axpyf
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
	     libblis_test_l1f_is_disabled( op ) ) return;

	// Call dependencies first.
	if ( TRUE ) libblis_test_axpyf_deps( tdata, params, op );

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
		                        libblis_test_axpyf_experiment );
	}
}



void libblis_test_axpyf_experiment
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

	dim_t        m, b_n;

	conj_t       conja, conjx;

	obj_t        alpha, a, x, y;
	obj_t        y_save;

	cntx_t*      cntx;


	// Query a context.
	cntx = ( cntx_t* )bli_gks_query_cntx();

	// Use the datatype of the first char in the datatype combination string.
	bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

	// Map the dimension specifier to an actual dimension.
	m = libblis_test_get_dim_from_prob_size( op->dim_spec[0], p_cur );

	// Query the operation's fusing factor for the current datatype.
	b_n = bli_cntx_get_blksz_def_dt( datatype, BLIS_AF, cntx );

	// Store the fusing factor so that the driver can retrieve the value
	// later when printing results.
	op->dim_aux[0] = b_n;

	// Map parameter characters to BLIS constants.
	bli_param_map_char_to_blis_conj( pc_str[0], &conja );
	bli_param_map_char_to_blis_conj( pc_str[1], &conjx );

	// Create test scalars.
	bli_obj_scalar_init_detached( datatype, &alpha );

	// Create test operands (vectors and/or matrices).
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                                            sc_str[0], m, b_n, &a );
	libblis_test_vobj_create( params, datatype, sc_str[1], b_n, &x );
	libblis_test_vobj_create( params, datatype, sc_str[2], m, &y );
	libblis_test_vobj_create( params, datatype, sc_str[2], m, &y_save );

	// Set alpha.
	if ( bli_obj_is_real( &y ) )
	{
		bli_setsc( -1.0,  0.0, &alpha );
	}
	else
	{
		bli_setsc(  0.0, -1.0, &alpha );
	}

	// Randomize A, x, and y, and save y.
	libblis_test_mobj_randomize( params, FALSE, &a );
	libblis_test_vobj_randomize( params, FALSE, &x );
	libblis_test_vobj_randomize( params, FALSE, &y );
	bli_copyv( &y, &y_save );

	// Apply the parameters.
	bli_obj_set_conj( conja, &a );
	bli_obj_set_conj( conjx, &x );

	// Repeat the experiment n_repeats times and record results.
	for ( i = 0; i < n_repeats; ++i )
	{
		bli_copyv( &y_save, &y );

		time = bli_clock();

		libblis_test_axpyf_impl( iface,
		                         &alpha, &a, &x, &y,
		                         cntx );

		time_min = bli_clock_min_diff( time_min, time );
	}

	// Estimate the performance of the best experiment repeat.
	*perf = ( 2.0 * m * b_n ) / time_min / FLOPS_PER_UNIT_PERF;
	if ( bli_obj_is_complex( &y ) ) *perf *= 4.0;

	// Perform checks.
	libblis_test_axpyf_check( params, &alpha, &a, &x, &y, &y_save, resid );

	// Zero out performance and residual if output vector is empty.
	libblis_test_check_empty_problem( &y, perf, resid );

	// Free the test objects.
	bli_obj_free( &a );
	bli_obj_free( &x );
	bli_obj_free( &y );
	bli_obj_free( &y_save );
}



void libblis_test_axpyf_impl
     (
       iface_t   iface,
       obj_t*    alpha,
       obj_t*    a,
       obj_t*    x,
       obj_t*    y,
       cntx_t*   cntx
     )
{
	switch ( iface )
	{
		case BLIS_TEST_SEQ_FRONT_END:
		bli_axpyf_ex( alpha, a, x, y, cntx, NULL );
		break;

		default:
		libblis_test_printf_error( "Invalid interface type.\n" );
	}
}



void libblis_test_axpyf_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         a,
       obj_t*         x,
       obj_t*         y,
       obj_t*         y_orig,
       double*        resid
     )
{
	num_t  dt      = bli_obj_dt( y );
	num_t  dt_real = bli_obj_dt_proj_to_real( y );

	dim_t  m       = bli_obj_vector_dim( y );
	dim_t  b_n     = bli_obj_width( a );

	dim_t  i;

	obj_t  a1, chi1, v;
	obj_t  alpha_chi1;
	obj_t  norm;

	double junk;

	//
	// Pre-conditions:
	// - a is randomized.
	// - x is randomized.
	// - y is randomized.
	// Note:
	// - alpha should have a non-zero imaginary component in the complex
	//   cases in order to more fully exercise the implementation.
	//
	// Under these conditions, we assume that the implementation for
	//
	//   y := y_orig + alpha * conja(A) * conjx(x)
	//
	// is functioning correctly if
	//
	//   normfv( y - v )
	//
	// is negligible, where v contains y as computed by repeated calls to
	// axpyv.
	//

	bli_obj_scalar_init_detached( dt_real, &norm );
	bli_obj_scalar_init_detached( dt,      &alpha_chi1 );

	bli_obj_create( dt, m,   1, 0, 0, &v );

	bli_copyv( y_orig, &v );

	for ( i = 0; i < b_n; ++i )
	{
		bli_acquire_mpart_l2r( BLIS_SUBPART1, i, 1, a, &a1 );
		bli_acquire_vpart_f2b( BLIS_SUBPART1, i, 1, x, &chi1 );

		bli_copysc( &chi1, &alpha_chi1 );
		bli_mulsc( alpha, &alpha_chi1 );

		bli_axpyv( &alpha_chi1, &a1, &v );
	}

	bli_subv( y, &v );
	bli_normfv( &v, &norm );
	bli_getsc( &norm, resid, &junk );

	bli_obj_free( &v );
}

