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
static char*     op_str                    = "randm";
static char*     o_types                   = "m";  // a
static char*     p_types                   = "";   // (no parameters)
static thresh_t  thresh[BLIS_NUM_FP_TYPES] = { { 1e-04, 1e-05 },   // warn, pass for s
                                               { 1e-04, 1e-05 },   // warn, pass for c
                                               { 1e-13, 1e-14 },   // warn, pass for d
                                               { 1e-13, 1e-14 } }; // warn, pass for z

// Local prototypes.
void libblis_test_randm_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     );

void libblis_test_randm_experiment
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

void libblis_test_randm_impl
     (
       iface_t   iface,
       obj_t*    x
     );

void libblis_test_randm_check
     (
       test_params_t* params,
       obj_t*         x,
       double*        resid
     );



void libblis_test_randm_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     )
{
	// No dependencies.
}



void libblis_test_randm
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
	     libblis_test_util_is_disabled( op ) ) return;

	// Call dependencies first.
	if ( TRUE ) libblis_test_randm_deps( tdata, params, op );

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
		                        libblis_test_randm_experiment );
	}
}



void libblis_test_randm_experiment
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

	char         x_store;

	obj_t        x;


	// Use the datatype of the first char in the datatype combination string.
	bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

	// Map the dimension specifier to actual dimensions.
	m = libblis_test_get_dim_from_prob_size( op->dim_spec[0], p_cur );
	n = libblis_test_get_dim_from_prob_size( op->dim_spec[1], p_cur );

	// Map parameter characters to BLIS constants.

	// Extract the storage character for each operand.
	x_store = sc_str[0];

	// Create the test objects.
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE, x_store, m, n, &x );

	// Repeat the experiment n_repeats times and record results.
	for ( i = 0; i < n_repeats; ++i )
	{
		time = bli_clock();

		libblis_test_randm_impl( iface, &x );

		time_min = bli_clock_min_diff( time_min, time );
	}

	// Estimate the performance of the best experiment repeat.
	*perf = ( 2.0 * m * n ) / time_min / FLOPS_PER_UNIT_PERF;
	if ( bli_obj_is_complex( &x ) ) *perf *= 2.0;

	// Perform checks.
	// For randm(), we don't return a meaningful residual/diff, since we can't
	// really say for sure what is "random" and what is not, so instead we
	// manually perform some checks that will fail under some scenarios whic
	// we consider to be likely.
	libblis_test_randm_check( params, &x, resid );

	// Zero out performance and residual if input matrix is empty.
	libblis_test_check_empty_problem( &x, perf, resid );

	// Free the test objects.
	bli_obj_free( &x );
}



void libblis_test_randm_impl
     (
       iface_t   iface,
       obj_t*    x
     )
{
	switch ( iface )
	{
		case BLIS_TEST_SEQ_FRONT_END:
		bli_randm( x );
		break;

		default:
		libblis_test_printf_error( "Invalid interface type.\n" );
	}
}



void libblis_test_randm_check
     (
       test_params_t* params,
       obj_t*         x,
       double*        resid
     )
{
	num_t  dt_real = bli_obj_dt_proj_to_real( x );
	dim_t  m_x     = bli_obj_length( x );
	dim_t  n_x     = bli_obj_width( x );
	obj_t  sum;

	//
	// The two most likely ways that randm would fail is if all elements
	// were zero, or if all elements were greater than or equal to one.
	// We check both of these conditions by computing the sum of the
	// absolute values of the elements of x.
	//

	*resid = 0.0;

	bli_obj_scalar_init_detached( dt_real, &sum );

	bli_absumm( x, &sum );

	if ( bli_is_float( dt_real ) )
	{
		float*  sum_x = bli_obj_buffer_at_off( &sum );

		if      ( *sum_x == *bli_d0         ) *resid = 1.0;
		else if ( *sum_x >= 2.0 * m_x * n_x ) *resid = 2.0;
	}
	else // if ( bli_is_double( dt_real ) )
	{
		double* sum_x = bli_obj_buffer_at_off( &sum );

		if      ( *sum_x == *bli_d0         ) *resid = 1.0;
		else if ( *sum_x >= 2.0 * m_x * n_x ) *resid = 2.0;
	}
}




#define FUNCPTR_T absumm_fp

typedef void (*FUNCPTR_T)(
                           dim_t  m,
                           dim_t  n,
                           void*  x, inc_t rs_x, inc_t cs_x,
                           void*  sum_x
                         );

static FUNCPTR_T GENARRAY(ftypes,absumm);


void bli_absumm
     (
       obj_t* x,
       obj_t* sum_x
     )
{
	num_t     dt        = bli_obj_dt( x );

	dim_t     m         = bli_obj_length( x );
	dim_t     n         = bli_obj_width( x );

	void*     buf_x     = bli_obj_buffer_at_off( x );
	inc_t     rs_x      = bli_obj_row_stride( x );
	inc_t     cs_x      = bli_obj_col_stride( x );

	void*     buf_sum_x = bli_obj_buffer_at_off( sum_x );

	FUNCPTR_T f;


	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt];

	// Invoke the function.
	f( m,
	   n,
	   buf_x, rs_x, cs_x,
	   buf_sum_x );
}


#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname)( \
                           dim_t  m, \
                           dim_t  n, \
                           void*  x, inc_t rs_x, inc_t cs_x, \
                           void*  sum_x  \
                         ) \
{ \
	ctype*   x_cast     = x; \
	ctype_r* sum_x_cast = sum_x; \
	ctype_r  abs_chi1; \
	ctype_r  sum; \
	dim_t    i, j; \
\
	bli_tset0s( chr, sum ); \
\
	for ( j = 0; j < n; j++ ) \
	{ \
		for ( i = 0; i < m; i++ ) \
		{ \
			ctype* chi1 = x_cast + (i  )*rs_x + (j  )*cs_x; \
\
			bli_tabval2s( ch,chr,chr, *chi1, abs_chi1 ); \
			bli_tadds( chr,chr,chr, abs_chi1, sum ); \
		} \
	} \
\
	bli_tcopys( chr,chr, sum, *sum_x_cast ); \
}

INSERT_GENTFUNCR_BASIC( absumm )

