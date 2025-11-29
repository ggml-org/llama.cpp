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
static char*     op_str                    = "amaxv";
static char*     o_types                   = "v";  // x
static char*     p_types                   = "";   // (no parameters)
static thresh_t  thresh[BLIS_NUM_FP_TYPES] = { { 1e-04, 1e-05 },   // warn, pass for s
                                               { 1e-04, 1e-05 },   // warn, pass for c
                                               { 1e-13, 1e-14 },   // warn, pass for d
                                               { 1e-13, 1e-14 } }; // warn, pass for z

// Local prototypes.
void libblis_test_amaxv_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     );

void libblis_test_amaxv_experiment
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

void libblis_test_amaxv_impl
     (
       iface_t   iface,
       obj_t*    x,
       obj_t*    index
     );

void libblis_test_amaxv_check
     (
       test_params_t* params,
       obj_t*         x,
       obj_t*         index,
       double*        resid
     );

void bli_amaxv_test
     (
       obj_t*  x,
       obj_t*  index
     );



void libblis_test_amaxv_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     )
{
	libblis_test_randv( tdata, params, &(op->ops->randv) );
}



void libblis_test_amaxv
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
	if ( TRUE ) libblis_test_amaxv_deps( tdata, params, op );

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
		                        libblis_test_amaxv_experiment );
	}
}



void libblis_test_amaxv_experiment
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

	obj_t        x;
	obj_t        index;


	// Use the datatype of the first char in the datatype combination string.
	bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

	// Map the dimension specifier to an actual dimension.
	m = libblis_test_get_dim_from_prob_size( op->dim_spec[0], p_cur );

	// Map parameter characters to BLIS constants.


	// Create test scalars.
	bli_obj_scalar_init_detached( BLIS_INT, &index );

	// Create test operands (vectors and/or matrices).
	libblis_test_vobj_create( params, datatype, sc_str[0], m, &x );

	// Randomize x.
	libblis_test_vobj_randomize( params, FALSE, &x );

	// Repeat the experiment n_repeats times and record results.
	for ( i = 0; i < n_repeats; ++i )
	{
		time = bli_clock();

		libblis_test_amaxv_impl( iface, &x, &index );

		time_min = bli_clock_min_diff( time_min, time );
	}

	// Estimate the performance of the best experiment repeat.
	*perf = ( 1.0 * m ) / time_min / FLOPS_PER_UNIT_PERF;
	if ( bli_obj_is_complex( &x ) ) *perf *= 2.0;

	// Perform checks.
	libblis_test_amaxv_check( params, &x, &index, resid );

	// Zero out performance and residual if input vector is empty.
	libblis_test_check_empty_problem( &x, perf, resid );

	// Free the test objects.
	bli_obj_free( &x );
}



void libblis_test_amaxv_impl
     (
       iface_t   iface,
       obj_t*    x,
       obj_t*    index
     )
{
	switch ( iface )
	{
		case BLIS_TEST_SEQ_FRONT_END:
		bli_amaxv( x, index );
		break;

		default:
		libblis_test_printf_error( "Invalid interface type.\n" );
	}
}



void libblis_test_amaxv_check
     (
       test_params_t* params,
       obj_t*         x,
       obj_t*         index,
       double*        resid
     )
{
	obj_t index_test;
	obj_t chi_i;
	obj_t chi_i_test;
	dim_t i;
	dim_t i_test;

	double i_d, junk;
	double i_d_test;

	//
	// Pre-conditions:
	// - x is randomized.
	//
	// Under these conditions, we assume that the implementation for
	//
	//   index := amaxv( x )
	//
	// is functioning correctly if
	//
	//   x[ index ] = max( x )
	//
	// where max() is implemented via the bli_?amaxv_test() function.
	//

	// The following two calls have already been made by the caller. That
	// is, the index object has already been created and the library's
	// amaxv implementation has already been tested.
	//bli_obj_scalar_init_detached( BLIS_INT, &index );
	//bli_amaxv( x, &index );
	bli_getsc( index, &i_d, &junk ); i = i_d;

	// If x is length 0, then we can't access any elements, and so we
	// return early with a good residual.
	if ( bli_obj_vector_dim( x ) == 0 ) { *resid = 0.0; return; }

	bli_acquire_vi( i, x, &chi_i );

	bli_obj_scalar_init_detached( BLIS_INT, &index_test );
	bli_amaxv_test( x, &index_test );
	bli_getsc( &index_test, &i_d_test, &junk ); i_test = i_d_test;
	bli_acquire_vi( i_test, x, &chi_i_test );

	// Verify that the values referenced by index and index_test are equal.
	if ( bli_obj_equals( &chi_i, &chi_i_test ) ) *resid = 0.0;
	else                                         *resid = 1.0;
}

// -----------------------------------------------------------------------------

//
// Prototype BLAS-like interfaces with typed operands for a local amaxv test
// operation
//

#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
void PASTEMAC(ch,opname) \
     ( \
       dim_t           n, \
       ctype* restrict x, inc_t incx, \
       dim_t* restrict index  \
     ); \

INSERT_GENTPROT_BASIC( amaxv_test )


//
// Prototype function pointer query interface.
//

#undef  GENPROT
#define GENPROT( tname, opname ) \
\
PASTECH(tname,_vft) \
PASTEMAC(opname,_qfp)( num_t dt );

GENPROT( amaxv, amaxv_test )


//
// Define function pointer query interfaces.
//

#undef  GENFRONT
#define GENFRONT( tname, opname ) \
\
GENARRAY_FPA( PASTECH(tname,_vft), \
              opname ); \
\
PASTECH(tname,_vft) \
PASTEMAC(opname,_qfp)( num_t dt ) \
{ \
    return PASTECH(opname,_fpa)[ dt ]; \
}

GENFRONT( amaxv, amaxv_test )


//
// Define object-based interface for a local amaxv test operation.
//

#undef  GENFRONT
#define GENFRONT( tname, opname ) \
\
void PASTEMAC(opname) \
     ( \
       obj_t*  x, \
       obj_t*  index  \
     ) \
{ \
    num_t     dt        = bli_obj_dt( x ); \
\
    dim_t     n         = bli_obj_vector_dim( x ); \
    void*     buf_x     = bli_obj_buffer_at_off( x ); \
    inc_t     incx      = bli_obj_vector_inc( x ); \
\
    void*     buf_index = bli_obj_buffer_at_off( index ); \
\
/*
	FGVZ: Disabling this code since bli_amaxv_check() is supposed to be a
	non-public API function, and therefore unavailable unless all symbols
	are scheduled to be exported at configure-time (which is not currently
	the default behavior).

    if ( bli_error_checking_is_enabled() ) \
        bli_amaxv_check( x, index ); \
*/ \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(tname,_vft) f = \
	PASTEMAC(opname,_qfp)( dt ); \
\
	f \
	( \
       n, \
       buf_x, incx, \
       buf_index  \
    ); \
}

GENFRONT( amaxv, amaxv_test )


//
// Define BLAS-like interfaces with typed operands for a local amaxv test
// operation.
// NOTE: This is based on a simplified version of the bli_?amaxv_ref()
// reference kernel.
//

#undef  GENTFUNCR
#define GENTFUNCR( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       dim_t*   index  \
     ) \
{ \
	ctype_r* minus_one = PASTEMAC(chr,m1); \
	dim_t*   zero_i    = PASTEMAC(i,0); \
\
	ctype_r  chi1_r; \
	ctype_r  chi1_i; \
	ctype_r  abs_chi1; \
	ctype_r  abs_chi1_max; \
	dim_t    index_l; \
	dim_t    i; \
\
	/* If the vector length is zero, return early. This directly emulates
	   the behavior of netlib BLAS's i?amax() routines. */ \
	if ( bli_zero_dim1( n ) ) \
	{ \
		bli_tcopys( i,i, *zero_i, *index ); \
		return; \
	} \
\
	/* Initialize the index of the maximum absolute value to zero. */ \
	bli_tcopys( i,i, *zero_i, index_l ); \
\
	/* Initialize the maximum absolute value search candidate with
	   -1, which is guaranteed to be less than all values we will
	   compute. */ \
	bli_tcopys( chr,chr, *minus_one, abs_chi1_max ); \
\
	{ \
		for ( i = 0; i < n; ++i ) \
		{ \
			ctype* chi1 = x + (i  )*incx; \
\
			/* Get the real and imaginary components of chi1. */ \
			bli_tgets( ch,chr, *chi1, chi1_r, chi1_i ); \
\
			/* Replace chi1_r and chi1_i with their absolute values. */ \
			bli_tabval2s( chr,chr,chr, chi1_r, chi1_r ); \
			bli_tabval2s( chr,chr,chr, chi1_i, chi1_i ); \
\
			/* Add the real and imaginary absolute values together. */ \
			bli_tset0s( chr, abs_chi1 ); \
			bli_tadds( chr,chr,chr, chi1_r, abs_chi1 ); \
			bli_tadds( chr,chr,chr, chi1_i, abs_chi1 ); \
\
			/* If the absolute value of the current element exceeds that of
			   the previous largest, save it and its index. If NaN is
			   encountered, then treat it the same as if it were a valid
			   value that was smaller than any previously seen. This
			   behavior mimics that of LAPACK's ?lange(). */ \
			if ( abs_chi1_max < abs_chi1 || PASTEMAC(chr,isnan)( abs_chi1 ) ) \
			{ \
				abs_chi1_max = abs_chi1; \
				index_l       = i; \
			} \
		} \
	} \
\
	/* Store the final index to the output variable. */ \
	bli_tcopys( i,i, index_l, *index ); \
}

INSERT_GENTFUNCR_BASIC( amaxv_test )

