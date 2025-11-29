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
static char*     op_str                    = "dotaxpyv";
static char*     o_types                   = "vvv";  // x y z
static char*     p_types                   = "ccc";  // conjxt conjx conjy
static thresh_t  thresh[BLIS_NUM_FP_TYPES] = { { 1e-04, 1e-05 },   // warn, pass for s
                                               { 1e-04, 1e-05 },   // warn, pass for c
                                               { 1e-13, 1e-14 },   // warn, pass for d
                                               { 1e-13, 1e-14 } }; // warn, pass for z

// Local prototypes.
void libblis_test_dotaxpyv_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     );

void libblis_test_dotaxpyv_experiment
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

void libblis_test_dotaxpyv_impl
     (
       iface_t   iface,
       obj_t*    alpha,
       obj_t*    xt,
       obj_t*    x,
       obj_t*    y,
       obj_t*    rho,
       obj_t*    z,
       cntx_t*   cntx
     );

void libblis_test_dotaxpyv_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         xt,
       obj_t*         x,
       obj_t*         y,
       obj_t*         rho,
       obj_t*         z,
       obj_t*         z_orig,
       double*        resid
     );



void libblis_test_dotaxpyv_deps
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
	libblis_test_dotv( tdata, params, &(op->ops->dotv) );
	libblis_test_axpyv( tdata, params, &(op->ops->axpyv) );
}



void libblis_test_dotaxpyv
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
	if ( TRUE ) libblis_test_dotaxpyv_deps( tdata, params, op );

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
		                        libblis_test_dotaxpyv_experiment );
	}
}



void libblis_test_dotaxpyv_experiment
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

	conj_t       conjxt, conjx, conjy;
	conj_t       conjconjxty;

	obj_t        alpha, xt, x, y, rho, z;
	obj_t        z_save;

	cntx_t*      cntx;


	// Query a context.
	cntx = ( cntx_t* )bli_gks_query_cntx();

	// Use the datatype of the first char in the datatype combination string.
	bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

	// Map the dimension specifier to an actual dimension.
	m = libblis_test_get_dim_from_prob_size( op->dim_spec[0], p_cur );

	// Map parameter characters to BLIS constants.
	bli_param_map_char_to_blis_conj( pc_str[0], &conjxt );
	bli_param_map_char_to_blis_conj( pc_str[1], &conjx );
	bli_param_map_char_to_blis_conj( pc_str[2], &conjy );

	// Create test scalars.
	bli_obj_scalar_init_detached( datatype, &alpha );
	bli_obj_scalar_init_detached( datatype, &rho );

	// Create test operands (vectors and/or matrices).
	libblis_test_vobj_create( params, datatype, sc_str[0], m, &x );
	libblis_test_vobj_create( params, datatype, sc_str[1], m, &y );
	libblis_test_vobj_create( params, datatype, sc_str[2], m, &z );
	libblis_test_vobj_create( params, datatype, sc_str[2], m, &z_save );

	// Set alpha.
	if ( bli_obj_is_real( &z ) )
	{
		bli_setsc( -0.8,  0.0, &alpha );
	}
	else
	{
		bli_setsc(  0.7, -0.1, &alpha );
	}

	// Randomize x and z, and save z.
	libblis_test_vobj_randomize( params, TRUE, &x );
	libblis_test_vobj_randomize( params, TRUE, &z );
	bli_copyv( &z, &z_save );

	// Create an alias to x for xt. (Note that it doesn't actually need to be
	// transposed.)
	bli_obj_alias_to( &x, &xt );

	// Determine whether to make a copy of x with or without conjugation.
	//
	//  conjx conjy  ~conjx^conjy   y is initialized as
	//  n     n      c              y = conj(x)
	//  n     c      n              y = x
	//  c     n      n              y = x
	//  c     c      c              y = conj(x)
	//
	conjconjxty = bli_apply_conj( conjxt, conjy );
	conjconjxty = bli_conj_toggled( conjconjxty );
	bli_obj_set_conj( conjconjxty, &xt );
	bli_copyv( &xt, &y );

	// Apply the parameters.
	bli_obj_set_conj( conjxt, &xt );
	bli_obj_set_conj( conjx,  &x );
	bli_obj_set_conj( conjy,  &y );

	// Repeat the experiment n_repeats times and record results.
	for ( i = 0; i < n_repeats; ++i )
	{
		bli_copysc( &BLIS_MINUS_ONE, &rho );
		bli_copyv( &z_save, &z );

		time = bli_clock();

		libblis_test_dotaxpyv_impl( iface,
		                            &alpha, &xt, &x, &y, &rho, &z,
		                            cntx );

		time_min = bli_clock_min_diff( time_min, time );
	}

	// Estimate the performance of the best experiment repeat.
	*perf = ( 2.0 * m + 2.0 * m ) / time_min / FLOPS_PER_UNIT_PERF;
	if ( bli_obj_is_complex( &z ) ) *perf *= 4.0;

	// Perform checks.
	libblis_test_dotaxpyv_check( params, &alpha, &xt, &x, &y, &rho, &z, &z_save, resid );

	// Zero out performance and residual if output vector is empty.
	libblis_test_check_empty_problem( &z, perf, resid );

	// Free the test objects.
	bli_obj_free( &x );
	bli_obj_free( &y );
	bli_obj_free( &z );
	bli_obj_free( &z_save );
}



void libblis_test_dotaxpyv_impl
     (
       iface_t   iface,
       obj_t*    alpha,
       obj_t*    xt,
       obj_t*    x,
       obj_t*    y,
       obj_t*    rho,
       obj_t*    z,
       cntx_t*   cntx
     )
{
	switch ( iface )
	{
		case BLIS_TEST_SEQ_FRONT_END:
		bli_dotaxpyv_ex( alpha, xt, x, y, rho, z, cntx, NULL );
		break;

		default:
		libblis_test_printf_error( "Invalid interface type.\n" );
	}
}



void libblis_test_dotaxpyv_check
     (
       test_params_t* params,
       obj_t*         alpha,
       obj_t*         xt,
       obj_t*         x,
       obj_t*         y,
       obj_t*         rho,
       obj_t*         z,
       obj_t*         z_orig,
       double*        resid
     )
{
	num_t  dt      = bli_obj_dt( z );
	num_t  dt_real = bli_obj_dt_proj_to_real( z );

	dim_t  m       = bli_obj_vector_dim( z );

	obj_t  rho_temp;

	obj_t  z_temp;
	obj_t  norm_z;

	double resid1, resid2;
	double junk;

	//
	// Pre-conditions:
	// - x is randomized.
	// - y is randomized.
	// - z_orig is randomized.
	// - xt is an alias to x.
	// Note:
	// - alpha should have a non-zero imaginary component in the complex
	//   cases in order to more fully exercise the implementation.
	//
	// Under these conditions, we assume that the implementation for
	//
	//   rho := conjxt(x^T) conjy(y)
	//   z := z_orig + alpha * conjx(x)
	//
	// is functioning correctly if
	//
	//   ( rho - rho_temp )
	//
	// and
	//
	//   normfv( z - z_temp )
	//
	// are negligible, where rho_temp and z_temp contain rho and z as
	// computed by dotv and axpyv, respectively.
	//

	bli_obj_scalar_init_detached( dt,      &rho_temp );
	bli_obj_scalar_init_detached( dt_real, &norm_z );

	bli_obj_create( dt, m, 1, 0, 0, &z_temp );
	bli_copyv( z_orig, &z_temp );


	bli_dotv( xt, y, &rho_temp );
	bli_axpyv( alpha, x, &z_temp );


	bli_subsc( rho, &rho_temp );
	bli_getsc( &rho_temp, &resid1, &junk );

	bli_subv( &z_temp, z );
	bli_normfv( z, &norm_z );
	bli_getsc( &norm_z, &resid2, &junk );

	*resid = bli_fmaxabs( resid1, resid2 );

	bli_obj_free( &z_temp );
}

