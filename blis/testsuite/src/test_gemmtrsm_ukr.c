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
static char*     op_str                    = "gemmtrsm_ukr";
static char*     o_types                   = "m";  // c11
static char*     p_types                   = "u";  // uploa
static thresh_t  thresh[BLIS_NUM_FP_TYPES] = { { 1e-04, 1e-05 },   // warn, pass for s
                                               { 1e-04, 1e-05 },   // warn, pass for c
                                               { 1e-13, 1e-14 },   // warn, pass for d
                                               { 1e-13, 1e-14 } }; // warn, pass for z

// Local prototypes.
void libblis_test_gemmtrsm_ukr_deps
     (
       thread_data_t* tdata,
       test_params_t* params,
       test_op_t*     op
     );

void libblis_test_gemmtrsm_ukr_experiment
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

void libblis_test_gemmtrsm_ukr_impl
     (
       iface_t   iface,
       side_t    side,
       obj_t*    alpha,
       obj_t*    a1x,
       obj_t*    a11,
       obj_t*    bx1,
       obj_t*    b11,
       obj_t*    c11,
       cntx_t*   cntx
     );

void libblis_test_gemmtrsm_ukr_check
     (
       test_params_t* params,
       side_t         side,
       obj_t*         alpha,
       obj_t*         a1x,
       obj_t*         a11,
       obj_t*         bx1,
       obj_t*         b11,
       obj_t*         c11,
       double*        resid
     );

void bli_gemmtrsm_ukr_make_subparts
     (
       dim_t  k,
       obj_t* a,
       obj_t* b,
       obj_t* a1x,
       obj_t* a11,
       obj_t* bx1,
       obj_t* b11
     );


void libblis_test_gemmtrsm_ukr_deps
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
	libblis_test_trsv( tdata, params, &(op->ops->trsv) );
}



void libblis_test_gemmtrsm_ukr
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
	     libblis_test_l3ukr_is_disabled( op ) ) return;

	// Call dependencies first.
	if ( TRUE ) libblis_test_gemmtrsm_ukr_deps( tdata, params, op );

	// Execute the test driver for each implementation requested.
	//if ( op->front_seq == ENABLE )
	{
		libblis_test_op_driver( tdata,
		                        params,
		                        op,
		                        BLIS_TEST_SEQ_UKERNEL,
		                        op_str,
		                        p_types,
		                        o_types,
		                        thresh,
		                        libblis_test_gemmtrsm_ukr_experiment );
	}
}


// Import the register blocksizes used by the micro-kernel(s).
extern blksz_t* gemm_mr;
extern blksz_t* gemm_nr;
extern blksz_t* gemm_kr;

void libblis_test_gemmtrsm_ukr_experiment
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

	dim_t        m, n, k;

	char         sc_a = 'c';
	char         sc_b = 'r';

	side_t       side = BLIS_LEFT;
	uplo_t       uploa;

	obj_t        alpha;
	obj_t        a_big, a, b;
	obj_t        a11, a1x, b11, bx1, c11;
	obj_t        ap, bp;
	obj_t        a1xp, a11p, bx1p, b11p;
	obj_t        c11_save;

	cntx_t*      cntx;


	// Query a context.
	cntx = ( cntx_t* )bli_gks_query_cntx();

	// Use the datatype of the first char in the datatype combination string.
	bli_param_map_char_to_blis_dt( dc_str[0], &datatype );

	// Map the dimension specifier to actual dimensions.
	k = libblis_test_get_dim_from_prob_size( op->dim_spec[0], p_cur );

	// Fix m and n to MR and NR, respectively.
	m = bli_cntx_get_blksz_def_dt( datatype, BLIS_MR, cntx );
	n = bli_cntx_get_blksz_def_dt( datatype, BLIS_NR, cntx );

	// Store the register blocksizes so that the driver can retrieve the
	// values later when printing results.
	op->dim_aux[0] = m;
	op->dim_aux[1] = n;

	// Map parameter characters to BLIS constants.
	bli_param_map_char_to_blis_uplo( pc_str[0], &uploa );

	// Create test scalars.
	bli_obj_scalar_init_detached( datatype, &alpha );

	// Create test operands (vectors and/or matrices).
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          sc_a,      k+m, k+m, &a_big );
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          sc_b,      k+m, n,   &b );
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          sc_str[0], m,   n,   &c11 );
	libblis_test_mobj_create( params, datatype, BLIS_NO_TRANSPOSE,
	                          sc_str[0], m,   n,   &c11_save );

	// Set alpha.
	if ( bli_obj_is_real( &b ) )
	{
		bli_setsc(  2.0,  0.0, &alpha );
	}
	else
	{
		bli_setsc(  2.0,  0.0, &alpha );
	}

	// Set the structure, uplo, and diagonal offset properties of A.
	bli_obj_set_struc( BLIS_TRIANGULAR, &a_big );
	bli_obj_set_uplo( uploa, &a_big );

	// Randomize A and make it densely triangular.
	libblis_test_mobj_randomize( params, TRUE, &a_big );
	libblis_test_mobj_load_diag( params, &a_big );

	// Normalize B and save.
	libblis_test_mobj_randomize( params, TRUE, &b );

	// Locate A1x/A11 (lower) or Ax1/A11 (upper), and then locate the
	// corresponding B11 block of B.
	if ( bli_obj_is_lower( &a_big ) )
	{
		bli_acquire_mpart_t2b( BLIS_SUBPART1, k, m, &a_big, &a );
		bli_acquire_mpart_l2r( BLIS_SUBPART0, k, m, &a, &a1x );
		bli_acquire_mpart_l2r( BLIS_SUBPART1, k, m, &a, &a11 );
		bli_acquire_mpart_t2b( BLIS_SUBPART0, k, m, &b, &bx1 );
		bli_acquire_mpart_t2b( BLIS_SUBPART1, k, m, &b, &b11 );
	}
	else
	{
		bli_acquire_mpart_t2b( BLIS_SUBPART1, 0, m, &a_big, &a );
		bli_acquire_mpart_l2r( BLIS_SUBPART1, 0, m, &a, &a11 );
		bli_acquire_mpart_l2r( BLIS_SUBPART2, 0, m, &a, &a1x );
		bli_acquire_mpart_t2b( BLIS_SUBPART1, 0, m, &b, &b11 );
		bli_acquire_mpart_t2b( BLIS_SUBPART2, 0, m, &b, &bx1 );
	}

	bli_obj_set_struc( BLIS_GENERAL, &a1x );

	// Copy B11 to C11, and save.
	bli_copym( &b11, &c11 );
	bli_copym( &c11, &c11_save );

	// Create pack objects for a and b, and pack them to ap and bp,
	// respectively.
	thrinfo_t* thread_a = libblis_test_pobj_create
	(
	  BLIS_MR,
	  BLIS_MR,
	  BLIS_BBM,
	  BLIS_INVERT_DIAG,
	  BLIS_PACKED_PANELS,
	  BLIS_BUFFER_FOR_A_BLOCK,
	  &a, &ap,
	  cntx
	);

	// Set the diagonal offset of ap.
	if ( bli_is_lower( uploa ) ) { bli_obj_set_diag_offset( k, &ap ); }
	else                         { bli_obj_set_diag_offset( 0, &ap ); }

	// Set the uplo field of ap since the default for packed objects is
	// BLIS_DENSE, and the _make_subparts() routine needs this information
	// to know how to initialize the subpartitions.
	bli_obj_set_uplo( uploa, &ap );

#if 0
bli_printm( "a", &a, "%5.2f", "" );
bli_printm( "ap", &ap, "%5.2f", "" );
#endif

	thrinfo_t* thread_b = NULL;

	// Repeat the experiment n_repeats times and record results.
	for ( i = 0; i < n_repeats; ++i )
	{
		bli_copym( &c11_save, &c11 );

		// Transpose B to B^T for packing.
		bli_obj_induce_trans( &b );

		thread_b = libblis_test_pobj_create
		(
		  BLIS_NR,
		  BLIS_MR,
		  BLIS_BBN,
		  BLIS_NO_INVERT_DIAG,
		  BLIS_PACKED_PANELS,
		  BLIS_BUFFER_FOR_B_PANEL,
		  &b, &bp,
		  cntx
		);

		// Transpose B^T back to B and Bp^T back to Bp.
		bli_obj_induce_trans( &b );
		bli_obj_induce_trans( &bp );

		// Create subpartitions from the a and b panels.
		bli_gemmtrsm_ukr_make_subparts( k, &ap, &bp,
		                                &a1xp, &a11p, &bx1p, &b11p );

		// Set the uplo field of a11p since the default for packed objects is
		// BLIS_DENSE, and the _ukernel() wrapper needs this information to
		// know which set of micro-kernels (lower or upper) to choose from.
		bli_obj_set_uplo( uploa, &a11p );

		time = bli_clock();

		libblis_test_gemmtrsm_ukr_impl( iface, side, &alpha,
		                                &a1xp, &a11p, &bx1p, &b11p, &c11,
		                                cntx );

		time_min = bli_clock_min_diff( time_min, time );

		// On the last pass, we must keep the packed B buffer checked out in order
		// to perform the correctness check later.
		if ( i < n_repeats - 1 )
		{
			// Free the thread control tree nodes and release their cached mem_t entries
			// back to the memory broker.
			bli_thrinfo_free( thread_b );
		}
	}

	// Estimate the performance of the best experiment repeat.
	*perf = ( 2.0 * m * n * k + 1.0 * m * m * n ) / time_min / FLOPS_PER_UNIT_PERF;
	if ( bli_obj_is_complex( &b ) ) *perf *= 4.0;

	// Perform checks.
	libblis_test_gemmtrsm_ukr_check( params, side, &alpha,
	                                 &a1x, &a11, &bx1, &b11, &c11, resid );

	// Zero out performance and residual if output matrix is empty.
	//libblis_test_check_empty_problem( &c11, perf, resid );

	// Free the thread control tree nodes and release their cached mem_t entries
	// back to the pba.
	bli_thrinfo_free( thread_a );
	if ( thread_b )
	    bli_thrinfo_free( thread_b );

	// Free the test objects.
	bli_obj_free( &a_big );
	bli_obj_free( &b );
	bli_obj_free( &c11 );
	bli_obj_free( &c11_save );
}



void libblis_test_gemmtrsm_ukr_impl
     (
       iface_t   iface,
       side_t    side,
       obj_t*    alpha,
       obj_t*    a1x,
       obj_t*    a11,
       obj_t*    bx1,
       obj_t*    b11,
       obj_t*    c11,
       cntx_t*   cntx
     )
{
	switch ( iface )
	{
		case BLIS_TEST_SEQ_UKERNEL:
		bli_gemmtrsm_ukernel( alpha, a1x, a11, bx1, b11, c11, cntx );
		break;

		default:
		libblis_test_printf_error( "Invalid interface type.\n" );
	}
}



void libblis_test_gemmtrsm_ukr_check
     (
       test_params_t* params,
       side_t         side,
       obj_t*         alpha,
       obj_t*         a1x,
       obj_t*         a11,
       obj_t*         bx1,
       obj_t*         b11,
       obj_t*         c11,
       double*        resid
     )
{
	num_t  dt      = bli_obj_dt( c11 );
	num_t  dt_real = bli_obj_dt_proj_to_real( c11 );

	dim_t  m       = bli_obj_length( c11 );
	dim_t  n       = bli_obj_width( c11 );
	dim_t  k       = bli_obj_width( a1x );

	obj_t  norm;
	obj_t  t, v, w, z;

	double junk;

	//
	// Pre-conditions:
	// - a1x, a11, bx1, b11 are randomized; a11 is triangular.
	// - side == BLIS_LEFT.
	//
	// Under these conditions, we assume that the implementation for
	//
	//   C11 := inv(A11) * ( alpha * B11 - A1x * Bx1 )       (side = left)
	//
	// is functioning correctly if
	//
	//   normfv( v - z )
	//
	// is negligible, where
	//
	//   v = C11 * t
	//
	//   z = ( inv(A11) * ( alpha * B11 - A1x * Bx1 ) ) * t
	//     = inv(A11) * ( alpha * B11 * t - A1x * Bx1 * t )
	//     = inv(A11) * ( alpha * B11 * t - A1x * w )
	//

	bli_obj_scalar_init_detached( dt_real, &norm );

	if ( bli_is_left( side ) )
	{
		bli_obj_create( dt, n, 1, 0, 0, &t );
		bli_obj_create( dt, m, 1, 0, 0, &v );
		bli_obj_create( dt, k, 1, 0, 0, &w );
		bli_obj_create( dt, m, 1, 0, 0, &z );
	}
	else // else if ( bli_is_left( side ) )
	{
		// BLIS does not currently support right-side micro-kernels.
		bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
	}

	libblis_test_vobj_randomize( params, TRUE, &t );

	bli_gemv( &BLIS_ONE, c11, &t, &BLIS_ZERO, &v );

#if 0
bli_printm( "a11", a11, "%5.2f", "" );
#endif

	if ( bli_is_left( side ) )
	{
		bli_gemv( &BLIS_ONE, bx1, &t, &BLIS_ZERO, &w );
		bli_gemv( alpha, b11, &t, &BLIS_ZERO, &z );
		bli_gemv( &BLIS_MINUS_ONE, a1x, &w, &BLIS_ONE, &z );
		bli_trsv( &BLIS_ONE, a11, &z );
	}
	else // else if ( bli_is_left( side ) )
	{
		// BLIS does not currently support right-side micro-kernels.
		bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED );
	}

	bli_subv( &z, &v );
	bli_normfv( &v, &norm );
	bli_getsc( &norm, resid, &junk );

	bli_obj_free( &t );
	bli_obj_free( &v );
	bli_obj_free( &w );
	bli_obj_free( &z );
}



void bli_gemmtrsm_ukr_make_subparts
     (
       dim_t  k,
       obj_t* a,
       obj_t* b,
       obj_t* a1x,
       obj_t* a11,
       obj_t* bx1,
       obj_t* b11
     )
{
	dim_t mr = bli_obj_length( a );
	dim_t nr = bli_obj_width( b );

	dim_t off_a1x, off_a11;
	dim_t off_bx1, off_b11;

	if ( bli_obj_is_lower( a ) )
	{
		off_a1x = 0;
		off_a11 = k;
		off_bx1 = 0;
		off_b11 = k;
	}
	else
	{
		off_a1x = mr;
		off_a11 = 0;
		off_bx1 = mr;
		off_b11 = 0;
	}

	bli_obj_init_subpart_from( a, a1x );
	bli_obj_set_dims( mr, k, a1x );
	bli_obj_inc_offs( 0, off_a1x, a1x );

	bli_obj_init_subpart_from( a, a11 );
	bli_obj_set_dims( mr, mr, a11 );
	bli_obj_inc_offs( 0, off_a11, a11 );

	bli_obj_init_subpart_from( b, bx1 );
	bli_obj_set_dims( k, nr, bx1 );
	bli_obj_inc_offs( off_bx1, 0, bx1 );

	bli_obj_init_subpart_from( b, b11 );
	bli_obj_set_dims( mr, nr, b11 );
	bli_obj_inc_offs( off_b11, 0, b11 );

	// Mark a1x as having general structure (which overwrites the triangular
	// property it inherited from a).
	bli_obj_set_struc( BLIS_GENERAL, a1x );

	// Set the diagonal offset of a11 to 0 (which overwrites the diagonal
	// offset value it inherited from a).
	bli_obj_set_diag_offset( 0, a11 );
}

