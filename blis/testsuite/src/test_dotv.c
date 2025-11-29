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
static char*     op_str                    = "dotv";
static char*     o_types                   = "vv";  // x y
static char*     p_types                   = "cc";  // conjx conjy
static thresh_t  thresh[BLIS_NUM_FP_TYPES] = { { 1e-04, 1e-05 },   // warn, pass for s
                                               { 1e-04, 1e-05 },   // warn, pass for c
                                               { 1e-13, 1e-14 },   // warn, pass for d
                                               { 1e-13, 1e-14 } }; // warn, pass for z

// Local prototypes.
void libblis_test_dotv_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     );

void libblis_test_dotv_experiment
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

void libblis_test_dotv_impl
     (
       iface_t   iface,
       obj_t*    x,
       obj_t*    y,
       obj_t*    rho
     );

void libblis_test_dotv_check
     (
       test_params_t* params,
       obj_t*         x,
       obj_t*         y,
       obj_t*         rho,
       double*        resid
     );



void libblis_test_dotv_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     )
{
	libblis_test_randv( tdata, params, &(op->ops->randv) );
	libblis_test_normfv( tdata, params, &(op->ops->normfv) );
	libblis_test_copyv( tdata, params, &(op->ops->copyv) );
}



void libblis_test_dotv
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
	     libblis_test_l1v_is_disabled( op ) ) return;

	// Call dependencies first.
	if ( TRUE ) libblis_test_dotv_deps( tdata, params, op );

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
		                        libblis_test_dotv_experiment );
	}
}



void libblis_test_dotv_experiment
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

	dim_t        m;

	conj_t       conjx, conjy, conjconjxy;

	obj_t        x, y, rho;


	// Use the datatype of the first char in the datatype combination string.
	bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

	// Map the dimension specifier to an actual dimension.
	m = libblis_test_get_dim_from_prob_size( op->dim_spec[0], p_cur );

	// Map parameter characters to BLIS constants.
	bli_param_map_char_to_blis_conj( pc_str[0], &conjx );
	bli_param_map_char_to_blis_conj( pc_str[1], &conjy );

	// Create test scalars.
	bli_obj_scalar_init_detached( datatype, &rho );

	// Create test operands (vectors and/or matrices).
	libblis_test_vobj_create( params, datatype, sc_str[0], m, &x );
	libblis_test_vobj_create( params, datatype, sc_str[1], m, &y );

	// Randomize x.
	libblis_test_vobj_randomize( params, TRUE, &x );

	// Determine whether to make a copy of x with or without conjugation.
	// 
	//  conjx conjy  ~conjx^conjy   y is initialized as
	//  n     n      c              y = conj(x)
	//  n     c      n              y = x
	//  c     n      n              y = x
	//  c     c      c              y = conj(x)
	//
	conjconjxy = bli_apply_conj( conjx, conjy );
	conjconjxy = bli_conj_toggled( conjconjxy );
	bli_obj_set_conj( conjconjxy, &x );
	bli_copyv( &x, &y );

	// Apply the parameters.
	bli_obj_set_conj( conjx, &x );
	bli_obj_set_conj( conjy, &y );

	// Repeat the experiment n_repeats times and record results. 
	for ( i = 0; i < n_repeats; ++i )
	{
		bli_copysc( &BLIS_MINUS_ONE, &rho );

		time = bli_clock();

		libblis_test_dotv_impl( iface, &x, &y, &rho );

		time_min = bli_clock_min_diff( time_min, time );
	}

	// Estimate the performance of the best experiment repeat.
	*perf = ( 2.0 * m ) / time_min / FLOPS_PER_UNIT_PERF;
	if ( bli_obj_is_complex( &y ) ) *perf *= 4.0;

	// Perform checks.
	libblis_test_dotv_check( params, &x, &y, &rho, resid );

	// Zero out performance and residual if output scalar is empty.
	libblis_test_check_empty_problem( &rho, perf, resid );

	// Free the test objects.
	bli_obj_free( &x );
	bli_obj_free( &y );
}



void libblis_test_dotv_impl
     (
       iface_t   iface,
       obj_t*    x,
       obj_t*    y,
       obj_t*    rho
     )
{
	switch ( iface )
	{
		case BLIS_TEST_SEQ_FRONT_END:
		bli_dotv( x, y, rho );
		break;

		default:
		libblis_test_printf_error( "Invalid interface type.\n" );
	}
}



void libblis_test_dotv_check
     (
       test_params_t* params,
       obj_t*         x,
       obj_t*         y,
       obj_t*         rho,
       double*        resid
     )
{
	num_t  dt_real = bli_obj_dt_proj_to_real( y );

	obj_t  rho_r, rho_i;
	obj_t  norm_x, norm_xy;

	double zero;
	double junk;

	//
	// Pre-conditions:
	// - x is randomized.
	// - y is equal to conj(conjx(conjy(x))).
	//
	// Under these conditions, we assume that the implementation for
	//
	//   rho := conjx(x^T) conjy(y)
	//
	// is functioning correctly if
	//
	//   sqrtsc( rho.real ) - normfv( x )
	//
	// and
	//
	//   rho.imag
	//
	// are negligible.
	//

	bli_obj_scalar_init_detached( dt_real, &rho_r );
	bli_obj_scalar_init_detached( dt_real, &rho_i );
	bli_obj_scalar_init_detached( dt_real, &norm_x );
	bli_obj_scalar_init_detached( dt_real, &norm_xy );

	bli_normfv( x, &norm_x );

	bli_unzipsc( rho, &rho_r, &rho_i );

	bli_sqrtsc( &rho_r, &norm_xy );

	bli_subsc( &norm_x, &norm_xy );
	bli_getsc( &norm_xy, resid, &junk );
	bli_getsc( &rho_i,   &zero, &junk );

	*resid = bli_fmaxabs( *resid, zero );
}

