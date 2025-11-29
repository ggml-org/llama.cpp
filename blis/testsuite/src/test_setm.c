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
static char*     op_str                    = "setm";
static char*     o_types                   = "m";  // x
static char*     p_types                   = "";   // (no parameters)
static thresh_t  thresh[BLIS_NUM_FP_TYPES] = { { 1e-04, 1e-05 },   // warn, pass for s
                                               { 1e-04, 1e-05 },   // warn, pass for c
                                               { 1e-13, 1e-14 },   // warn, pass for d
                                               { 1e-13, 1e-14 } }; // warn, pass for z

// Local prototypes.
void libblis_test_setm_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     );

void libblis_test_setm_experiment
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

void libblis_test_setm_impl
     (
       iface_t   iface,
       obj_t*    beta,
       obj_t*    x
     );

void libblis_test_setm_check
     (
       test_params_t* params,
       obj_t*         beta,
       obj_t*         x,
       double*        resid
     );



void libblis_test_setm_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     )
{
	libblis_test_randv( tdata, params, &(op->ops->randm) );
}



void libblis_test_setm
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
	if ( TRUE ) libblis_test_setm_deps( tdata, params, op );

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
		                        libblis_test_setm_experiment );
	}
}



void libblis_test_setm_experiment
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

	obj_t        beta;
	obj_t        x;


	// Use the datatype of the first char in the datatype combination string.
	bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

	// Map the dimension specifier to actual dimensions.
	m = libblis_test_get_dim_from_prob_size( op->dim_spec[0], p_cur );
	n = libblis_test_get_dim_from_prob_size( op->dim_spec[1], p_cur );

	// Map parameter characters to BLIS constants.


	// Create test scalars.
	bli_obj_scalar_init_detached( datatype, &beta );

	// Create test operands (vectors and/or matrices).
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          sc_str[0], m, n, &x );

	// Initialize beta to unit.
	bli_copysc( &BLIS_ONE, &beta );

	// Randomize x.
	libblis_test_mobj_randomize( params, FALSE, &x );

	// Repeat the experiment n_repeats times and record results.
	for ( i = 0; i < n_repeats; ++i )
	{
		time = bli_clock();

		libblis_test_setm_impl( iface, &beta, &x );

		time_min = bli_clock_min_diff( time_min, time );
	}

	// Estimate the performance of the best experiment repeat.
	*perf = ( 1.0 * m * n ) / time_min / FLOPS_PER_UNIT_PERF;
	if ( bli_obj_is_complex( &x ) ) *perf *= 2.0;

	// Perform checks.
	libblis_test_setm_check( params, &beta, &x, resid );

	// Zero out performance and residual if output matrix is empty.
	libblis_test_check_empty_problem( &x, perf, resid );

	// Free the test objects.
	bli_obj_free( &x );
}



void libblis_test_setm_impl
     (
       iface_t   iface,
       obj_t*    beta,
       obj_t*    x
     )
{
	switch ( iface )
	{
		case BLIS_TEST_SEQ_FRONT_END:
		bli_setm( beta, x );
		break;

		default:
		libblis_test_printf_error( "Invalid interface type.\n" );
	}
}



void libblis_test_setm_check
     (
       test_params_t* params,
       obj_t*         beta,
       obj_t*         x,
       double*        resid
     )
{
	num_t dt_x     = bli_obj_dt( x );
	dim_t m_x      = bli_obj_length( x );
	dim_t n_x      = bli_obj_width( x );
	inc_t rs_x     = bli_obj_row_stride( x );
	inc_t cs_x     = bli_obj_col_stride( x );
	void* buf_x    = bli_obj_buffer_at_off( x );
	void* buf_beta = bli_obj_buffer_for_1x1( dt_x, beta );
	dim_t i, j;

	*resid = 0.0;

	//
	// The easiest way to check that setm was successful is to confirm
	// that each element of x is equal to beta.
	//

	if      ( bli_obj_is_float( x ) )
	{
		float*    beta_cast  = buf_beta;
		float*    buf_x_cast = buf_x;
		float*    chi1;

		for ( j = 0; j < n_x; ++j )
		{
			for ( i = 0; i < m_x; ++i )
			{
				chi1 = buf_x_cast + (i  )*rs_x + (j  )*cs_x;

				if ( !bli_seq( *chi1, *beta_cast ) ) { *resid = 1.0; return; }
			}
		}
	}
	else if ( bli_obj_is_double( x ) )
	{
		double*   beta_cast  = buf_beta;
		double*   buf_x_cast = buf_x;
		double*   chi1;

		for ( j = 0; j < n_x; ++j )
		{
			for ( i = 0; i < m_x; ++i )
			{
				chi1 = buf_x_cast + (i  )*rs_x + (j  )*cs_x;

				if ( !bli_deq( *chi1, *beta_cast ) ) { *resid = 1.0; return; }
			}
		}
	}
	else if ( bli_obj_is_scomplex( x ) )
	{
		scomplex* beta_cast  = buf_beta;
		scomplex* buf_x_cast = buf_x;
		scomplex* chi1;

		for ( j = 0; j < n_x; ++j )
		{
			for ( i = 0; i < m_x; ++i )
			{
				chi1 = buf_x_cast + (i  )*rs_x + (j  )*cs_x;

				if ( !bli_teqs( c,c,c, *chi1, *beta_cast ) ) { *resid = 1.0; return; }
			}
		}
	}
	else // if ( bli_obj_is_dcomplex( x ) )
	{
		dcomplex* beta_cast  = buf_beta;
		dcomplex* buf_x_cast = buf_x;
		dcomplex* chi1;

		for ( j = 0; j < n_x; ++j )
		{
			for ( i = 0; i < m_x; ++i )
			{
				chi1 = buf_x_cast + (i  )*rs_x + (j  )*cs_x;

				if ( !bli_teqs( z,z,z, *chi1, *beta_cast ) ) { *resid = 1.0; return; }
			}
		}
	}
}

