/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.
   Copyright (C) 2024, Southern Methodist University

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


static packm_ker_ft GENARRAY2_MIXP(packm_struc_cxk,packm_struc_cxk);

void bli_gemm_var_cntl_init_node
     (
       void_fp          var_func,
       num_t            dt_comp,
       num_t            dt_out,
       gemm_ukr_ft      ukr,
       gemm_ukr_ft      real_ukr,
       bool             row_pref,
       dim_t            mr,
       dim_t            nr,
       dim_t            mr_scale,
       dim_t            nr_scale,
       gemm_var_cntl_t* cntl
     )
{
	// Initialize the gemm_var_cntl_t struct.
	cntl->dt_comp  = dt_comp;
	cntl->dt_out   = dt_out;
	cntl->ukr      = ukr;
	cntl->real_ukr = real_ukr;
	cntl->row_pref = row_pref;
	cntl->mr       = mr;
	cntl->nr       = nr;
	cntl->mr_scale = mr_scale;
	cntl->nr_scale = nr_scale;

	bli_cntl_init_node
	(
	  var_func,
	  &cntl->cntl
	);
}

bool bli_gemm_cntl_init
     (
             ind_t        im,
             opid_t       family,
       const obj_t*       alpha,
             obj_t*       a,
             obj_t*       b,
       const obj_t*       beta,
             obj_t*       c,
       const cntx_t*      cntx,
             gemm_cntl_t* cntl
     )
{
	      bool   a_is_real = bli_obj_is_real( a );
	      bool   b_is_real = bli_obj_is_real( b );
	      bool   c_is_real = bli_obj_is_real( c );
	const bool   induced   = im != BLIS_NAT ||
	                         a_is_real != b_is_real ||
	                         a_is_real != c_is_real ||
	                         b_is_real != c_is_real;
	const prec_t comp_prec = bli_obj_comp_prec( c );
	const num_t  dt_c      = bli_obj_dt( c );
	const num_t  dt_comp   = ( induced ? BLIS_REAL : bli_dt_domain( dt_c ) ) | comp_prec;
	const bool   row_pref  = bli_cntx_get_ukr_prefs_dt( dt_comp, BLIS_GEMM_UKR_ROW_PREF, cntx );

	// An optimization: If C is stored by rows and the micro-kernel prefers
	// contiguous columns, or if C is stored by columns and the micro-kernel
	// prefers contiguous rows, transpose the entire operation to allow the
	// micro-kernel to access elements of C in its preferred manner.
	bool needs_swap = (  row_pref && bli_obj_is_col_tilted( c ) ) ||
	                  ( !row_pref && bli_obj_is_row_tilted( c ) );

	// NOTE: This case casts right-side symm/hemm/trmm/trmm3 in terms of left side.
	// This may be necessary when the current subconfiguration uses a gemm microkernel
	// that assumes that the packing kernel will have already duplicated
	// (broadcast) element of B in the packed copy of B. Supporting
	// duplication within the logic that packs micropanels from symmetric
	// matrices is ugly, but technically supported. This can
	// lead to the microkernel being executed on an output matrix with the
	// microkernel's general stride IO case (unless the microkernel supports
	// both both row and column IO cases as well). As a
	// consequence, those subconfigurations need a way to force the symmetric
	// matrix to be on the left (and thus the general matrix to the on the
	// right). So our solution is that in those cases, the subconfigurations
	// simply #define BLIS_DISABLE_{SYMM,HEMM,TRMM,TRMM3}_RIGHT.

	// If A is being multiplied from the right, transpose all operands
	// so that we can perform the computation as if A were being multiplied
	// from the left.
#ifdef BLIS_DISABLE_SYMM_RIGHT
	if ( family == BLIS_SYMM ) needs_swap = bli_obj_is_symmetric( b );
#endif
#ifdef BLIS_DISABLE_HEMM_RIGHT
	if ( family == BLIS_HEMM ) needs_swap = bli_obj_is_hermitian( b );
#endif
#ifdef BLIS_DISABLE_TRMM_RIGHT
	if ( family == BLIS_TRMM ) needs_swap = bli_obj_is_triangular( b );
#endif
#ifdef BLIS_DISABLE_TRMM3_RIGHT
	if ( family == BLIS_TRMM3 ) needs_swap = bli_obj_is_triangular( b );
#endif

	if ( a_is_real && !b_is_real && !c_is_real )
	{
		// C := R * C *must* be swapped for column-preferring kernels
		needs_swap = !row_pref;
	}
	else if ( !a_is_real && b_is_real && !c_is_real )
	{
		// C := C * R *must* be swapped for row-preferring kernels
		needs_swap = row_pref;
	}

	// Swap the A and B operands if required. This transforms the operation
	// C = alpha A B + beta C into C^T = alpha B^T A^T + beta C^T.
	if ( needs_swap )
	{
		bli_obj_swap( a, b );

		bli_obj_induce_trans( a );
		bli_obj_induce_trans( b );
		bli_obj_induce_trans( c );

		bool tmp = a_is_real;
		a_is_real = b_is_real;
		b_is_real = tmp;
	}

	const num_t dt_a  = bli_obj_dt( a );
	const num_t dt_b  = bli_obj_dt( b );
	const num_t dt_ap = bli_dt_domain( dt_a ) | comp_prec;
	const num_t dt_bp = bli_dt_domain( dt_b ) | comp_prec;

	// Cast alpha and beta to the computational precision.
	// Alpha should be complex if any of A, B, or C are.
	obj_t alpha_cast, beta_cast;
	dom_t alpha_dom = bli_obj_is_complex( a ) ||
	                  bli_obj_is_complex( b ) ||
	                  bli_obj_is_complex( c ) ? BLIS_COMPLEX : BLIS_REAL;
	bli_obj_scalar_init_detached_copy_of( alpha_dom | comp_prec,
	                                      BLIS_NO_CONJUGATE,
	                                      alpha,
	                                      &alpha_cast );
	// Cast beta to the type of C, since we will need to
	// ignore the imaginary part of beta for real C.
	bli_obj_scalar_init_detached_copy_of( dt_c,
	                                      BLIS_NO_CONJUGATE,
	                                      beta,
	                                      &beta_cast );

	// Cast the scalars of A and B to the computational precision
	bli_obj_scalar_cast_to( BLIS_COMPLEX | comp_prec, a );
	bli_obj_scalar_cast_to( BLIS_COMPLEX | comp_prec, b );

	// If alpha is non-unit, typecast and apply it to the scalar attached
	// to B, unless alpha is complex and A is complex while B is not.
	if ( bli_obj_is_complex( &alpha_cast ) &&
	     bli_obj_is_complex( a ) &&
	     bli_obj_is_real( b ) )
	{
		if ( !bli_obj_equals( &alpha_cast, &BLIS_ONE ) )
			bli_obj_scalar_apply_scalar( &alpha_cast, a );
	}
	else
	{
		if ( !bli_obj_equals( &alpha_cast, &BLIS_ONE ) )
			bli_obj_scalar_apply_scalar( &alpha_cast, b );
	}

	// If beta is non-unit, typecast and apply it to the scalar attached
	// to C.
	if ( !bli_obj_equals( &beta_cast, &BLIS_ONE ) )
		bli_obj_scalar_apply_scalar( &beta_cast, c );

	void_fp     macro_kernel_fp = bli_gemm_ker_var2;
	gemm_ukr_ft gemm_ukr        = bli_cntx_get_ukr2_dt( dt_comp, dt_c, BLIS_GEMM_UKR, cntx );
	gemm_ukr_ft real_gemm_ukr   = bli_cntx_get_ukr_dt( dt_comp, BLIS_GEMM_UKR, cntx );

	// Set the macrokernel function pointer based on the operation family
	// and struc/uplo properties.
#ifdef BLIS_ENABLE_JRIR_TLB
	if ( family == BLIS_GEMMT )
	{
		macro_kernel_fp = bli_obj_is_lower( c ) ? bli_gemmt_l_ker_var2b
		                                        : bli_gemmt_u_ker_var2b;
	}
	else if ( family == BLIS_TRMM || family == BLIS_TRMM3 )
	{
		if ( bli_obj_is_triangular( a ) )
			macro_kernel_fp = bli_obj_is_lower( a ) ? bli_trmm_ll_ker_var2b
			                                        : bli_trmm_lu_ker_var2b;
		else /* if ( bli_obj_is_triangular( b ) ) */
			macro_kernel_fp = bli_obj_is_lower( b ) ? bli_trmm_rl_ker_var2b
			                                        : bli_trmm_ru_ker_var2b;
	}
#else
	if ( family == BLIS_GEMMT )
	{
		macro_kernel_fp = bli_obj_is_lower( c ) ? bli_gemmt_l_ker_var2
		                                        : bli_gemmt_u_ker_var2;
	}
	else if ( family == BLIS_TRMM || family == BLIS_TRMM3 )
	{
		if ( bli_obj_is_triangular( a ) )
			macro_kernel_fp = bli_obj_is_lower( a ) ? bli_trmm_ll_ker_var2
			                                        : bli_trmm_lu_ker_var2;
		else /* if ( bli_obj_is_triangular( b ) ) */
			macro_kernel_fp = bli_obj_is_lower( b ) ? bli_trmm_rl_ker_var2
			                                        : bli_trmm_ru_ker_var2;
	}
#endif

	const bool         trmm_r        = family == BLIS_TRMM && bli_obj_is_triangular( b );
	const bool         a_lo_tri      = bli_obj_is_triangular( a ) && bli_obj_is_lower( a );
	const bool         b_up_tri      = bli_obj_is_triangular( b ) && bli_obj_is_upper( b );
	      pack_t       schema_a      = BLIS_PACKED_PANELS;
	      pack_t       schema_b      = BLIS_PACKED_PANELS;
	const packm_ker_ft packm_a_ukr   = packm_struc_cxk[ dt_a ][ dt_ap ];
	const packm_ker_ft packm_b_ukr   = packm_struc_cxk[ dt_b ][ dt_bp ];
	const dim_t        mr_def        = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_MR, cntx );
	const dim_t        mr_pack       = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_MR, cntx );
	const dim_t        mr_bcast      = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_BBM, cntx );
	      dim_t        mr_scale      = 1;
	      dim_t        mr_pack_scale = 1;
	const dim_t        nr_def        = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_NR, cntx );
	const dim_t        nr_pack       = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_NR, cntx );
	const dim_t        nr_bcast      = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_BBN, cntx );
	      dim_t        nr_scale      = 1;
	      dim_t        nr_pack_scale = 1;
	const dim_t        kr_def        = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_KR, cntx );
	const dim_t        mc_def        = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_MC, cntx );
	const dim_t        mc_max        = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_MC, cntx );
	      dim_t        mc_scale      = 1;
	const dim_t        nc_def        = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_NC, cntx );
	const dim_t        nc_max        = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_NC, cntx );
	      dim_t        nc_scale      = 1;
	const dim_t        kc_def        = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_KC, cntx );
	const dim_t        kc_max        = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_KC, cntx );
	      dim_t        kc_scale      = 1;

	if ( im == BLIS_1M )
	{
		if ( !row_pref )
		{
			schema_a = BLIS_PACKED_PANELS_1E;
			schema_b = BLIS_PACKED_PANELS_1R;
			mr_scale = 2;
			mc_scale = 2;
			mr_pack_scale = 1; //don't divide PACKMR by 2 since we are also doubling k
		}
		else
		{
			schema_a = BLIS_PACKED_PANELS_1R;
			schema_b = BLIS_PACKED_PANELS_1E;
			nr_scale = 2;
			nc_scale = 2;
			nr_pack_scale = 1; //don't divide PACKNR by 2 since we are also doubling k
		}

		kc_scale = 2;
		gemm_ukr = bli_cntx_get_ukr2_dt( dt_comp, bli_dt_proj_to_real( dt_c ), BLIS_GEMM1M_UKR, cntx );
	}
	else if ( (  c_is_real &&  a_is_real &&  b_is_real ) ||
	          ( !c_is_real && !a_is_real && !b_is_real ) )
	{
		// C_real += A_real * B_real
		// C_complex += A_complex * B_complex
		// Nothing to do.
	}
	else if ( ( !c_is_real && !a_is_real &&  b_is_real ) ||
	          ( !c_is_real &&  a_is_real && !b_is_real ) )
	{
		// C_complex += A_complex * B_real
		// C_complex += A_real * B_complex

		// Pack the complex input operand as normal, except that
		// the (rescaled) real-domain block sizes are used.

		if ( !row_pref )
		{
			// We transpose the operation above to make sure that
			// the complex matrix is on the right side for the storage
			// preference of the microkernel, but things sometimes go
			// wrong.
			if ( a_is_real ) bli_abort();
			mc_scale = 2;
			mr_scale = 2;
			mr_pack_scale = 2;
		}
		else
		{
			// We transpose the operation above to make sure that
			// the complex matrix is on the right side for the storage
			// preference of the microkernel, but things sometimes go
			// wrong.
			if ( b_is_real ) bli_abort();
			nc_scale = 2;
			nr_scale = 2;
			nr_pack_scale = 2;
		}

		// A microkernel wrapper is necessary for cases where C is general-stored
		// or does not match the storage preference of the real-domain
		// gemm microkernel, or when beta is complex.

		gemm_ukr = bli_cntx_get_ukr2_dt( dt_comp, bli_dt_proj_to_real( dt_c ), BLIS_GEMM_CCR_UKR, cntx );
	}
	else if (  c_is_real && !a_is_real && !b_is_real )
	{
		// C_real += A_complex * B_complex

		// Pack both A and B in the 1r format and use 1/2
		// of the real-domain KC block size since twice as
		// many values will be packed. One of the matrices
		// needs to be conjugated to get the right sign
		// on the imaginary components.

		schema_a = BLIS_PACKED_PANELS_1R;
		schema_b = BLIS_PACKED_PANELS_1R;
		kc_scale = 2;
		bli_obj_toggle_conj( a );

		// A microkernel wrapper is necessary only to scale k by 2
		// due to the 1r packing schema (or if type conversion is required).
		// Any complex values of alpha will be applied during packing,
		// so the real-domain microkernel can do everything directly.

		gemm_ukr = bli_cntx_get_ukr2_dt( dt_comp, bli_dt_proj_to_real( dt_c ), BLIS_GEMM_RCC_UKR, cntx );
	}
	else if ( !c_is_real &&  a_is_real &&  b_is_real )
	{
		// C_complex += A_real * B_real

		// A microkernel wrapper is always needed to store
		// only the real part of the AB product, but also deal
		// with potentially complex alpha and beta scalars.

		gemm_ukr = bli_cntx_get_ukr2_dt( dt_comp, bli_dt_proj_to_real( dt_c ), BLIS_GEMM_CRR_UKR, cntx );
	}
	else if ( (  c_is_real && !a_is_real &&  b_is_real ) ||
	          (  c_is_real &&  a_is_real && !b_is_real ) )
	{
		// C_real += A_complex * B_real
		// C_real += A_real * B_complex

		// Pack only the real part of the complex operand.
		// If alpha is also complex then it will be applied
		// during packing.

		if ( a_is_real )
		{
			schema_b = BLIS_PACKED_PANELS_RO;
		}
		else
		{
			schema_a = BLIS_PACKED_PANELS_RO;
		}
	}

	//printf("MR: %lld/%lld,  %lld/%lld\n", mr_def, mr_scale, mr_pack, mr_pack_scale);
	//printf("NR: %lld/%lld,  %lld/%lld\n", nr_def, nr_scale, nr_pack, nr_pack_scale);

	// Create two nodes for the macro-kernel.
	bli_cntl_init_node
	(
	  NULL,         // variant function pointer not used
	  &cntl->ir_loop
	);

	bli_gemm_var_cntl_init_node
	(
	  macro_kernel_fp,
	  dt_comp,
	  dt_c,
	  gemm_ukr,
	  real_gemm_ukr,
	  row_pref,
	  mr_def / mr_scale,
	  nr_def / nr_scale,
	  mr_scale,
	  nr_scale,
	  &cntl->ker
	);
	bli_cntl_attach_sub_node
	(
	  BLIS_THREAD_NR,
	  ( cntl_t* )&cntl->ir_loop,
	  ( cntl_t* )&cntl->ker
	);

	// Give the gemm kernel control tree node to the
	// virtual microkernel as the parameters, so that e.g.
	// the 1m virtual microkernel can look up the real-domain
	// micro-kernel and its parameters.
	bli_gemm_var_cntl_set_params( &cntl->ker, ( cntl_t* )&cntl->ker );

	// Create a node for packing matrix A.
	bli_packm_def_cntl_init_node
	(
	  bli_l3_packa, // pack the left-hand operand
	  dt_a,
	  dt_ap,
	  dt_comp,
	  packm_a_ukr,
	  mr_def / mr_scale,
	  mr_pack / mr_pack_scale,
	  mr_bcast,
	  mr_scale,
	  mr_pack_scale,
	  kr_def,
	  FALSE,
	  FALSE,
	  FALSE,
	  schema_a,
	  BLIS_BUFFER_FOR_A_BLOCK,
	  &cntl->pack_a
	);
	bli_cntl_attach_sub_node
	(
	  BLIS_THREAD_NONE,
	  ( cntl_t* )&cntl->ker,
	  ( cntl_t* )&cntl->pack_a
	);

	// Create a node for partitioning the m dimension by MC.
	bli_part_cntl_init_node
	(
	  bli_gemm_blk_var1,
	  dt_comp,
	  mc_def / mc_scale,
	  mc_max / mc_scale,
	  mc_scale,
	  mr_def / mr_scale,
	  mr_scale,
	  a_lo_tri ? BLIS_BWD
	           : BLIS_FWD,
	  bli_obj_is_triangular( a ) || bli_obj_is_upper_or_lower( c ),
	  &cntl->part_ic
	);
	bli_cntl_attach_sub_node
	(
	  trmm_r ? BLIS_THREAD_MC | BLIS_THREAD_NC
	         : BLIS_THREAD_MC,
	  ( cntl_t* )&cntl->pack_a,
	  ( cntl_t* )&cntl->part_ic
	);

	// Create a node for packing matrix B.
	bli_packm_def_cntl_init_node
	(
	  bli_l3_packb, // pack the right-hand operand
	  dt_b,
	  dt_bp,
	  dt_comp,
	  packm_b_ukr,
	  nr_def / nr_scale,
	  nr_pack / nr_pack_scale,
	  nr_bcast,
	  nr_scale,
	  nr_pack_scale,
	  kr_def,
	  FALSE,
	  FALSE,
	  FALSE,
	  schema_b,
	  BLIS_BUFFER_FOR_B_PANEL,
	  &cntl->pack_b
	);
	bli_cntl_attach_sub_node
	(
	  BLIS_THREAD_NONE,
	  ( cntl_t* )&cntl->part_ic,
	  ( cntl_t* )&cntl->pack_b
	);

	// Create a node for partitioning the k dimension by KC.
	bli_part_cntl_init_node
	(
	  bli_gemm_blk_var3,
	  dt_comp,
	  kc_def / kc_scale,
	  kc_max / kc_scale,
	  kc_scale,
	  kr_def,
	  1,
	  ( a_lo_tri || b_up_tri ) ? BLIS_BWD
	                           : BLIS_FWD,
	  FALSE,
	  &cntl->part_pc
	);
	bli_cntl_attach_sub_node
	(
	  BLIS_THREAD_KC,
	  ( cntl_t* )&cntl->pack_b,
	  ( cntl_t* )&cntl->part_pc
	);

	// Create a node for partitioning the n dimension by NC.
	bli_part_cntl_init_node
	(
	  bli_gemm_blk_var2,
	  dt_comp,
	  nc_def / nc_scale,
	  nc_max / nc_scale,
	  nc_scale,
	  nr_def / nr_scale,
	  nr_scale,
	  b_up_tri ? BLIS_BWD
	           : BLIS_FWD,
	  bli_obj_is_triangular( b ) || bli_obj_is_upper_or_lower( c ),
	  &cntl->part_jc
	);
	bli_cntl_attach_sub_node
	(
	  trmm_r ? BLIS_THREAD_NONE
	         : BLIS_THREAD_NC,
	  ( cntl_t* )&cntl->part_pc,
	  ( cntl_t* )&cntl->part_jc
	);

	bli_gemm_cntl_finalize
	(
	  family,
	  a,
	  b,
	  c,
	  cntl
	);

	return needs_swap;
}

void bli_gemm_cntl_finalize
     (
             opid_t       family,
       const obj_t*       a,
       const obj_t*       b,
       const obj_t*       c,
             gemm_cntl_t* cntl
     )
{
	( void )c;

	const dim_t ic_mult = bli_part_cntl_blksz_mult( ( cntl_t* )&cntl->part_ic );
	const dim_t jc_mult = bli_part_cntl_blksz_mult( ( cntl_t* )&cntl->part_jc );

	//
	// Ensure that:
	//
	// 1. KC is a multiple of MR (NR) if A (B) is triangular, hermitian, or symmetric.
	//    KC is always rounded up.
	//
	// 2. MC and NR are multiples of MR and NR, respectively. MC and NC are always
	//    rounded down.
	//

	// Nudge the default and maximum kc blocksizes up to the nearest
	// multiple of MR if A is Hermitian, symmetric, or triangular or
	// NR if B is Hermitian, symmetric, or triangular. If neither case
	// applies, then we leave the blocksizes unchanged. For trsm we
	// always use MR (rather than sometimes using NR) because even
	// when the triangle is on the right, packing of that matrix uses
	// MR, since only left-side trsm micro-kernels are supported.
	if ( !bli_obj_root_is_general( a ) || family == BLIS_TRSM )
	{
		bli_part_cntl_align_blksz_to_mult( ic_mult, true, ( cntl_t* )&cntl->part_pc );
	}
	else if ( !bli_obj_root_is_general( b ) )
	{
		bli_part_cntl_align_blksz_to_mult( jc_mult, true, ( cntl_t* )&cntl->part_pc );
	}

	bli_part_cntl_align_blksz( false, ( cntl_t* )&cntl->part_ic );
	bli_part_cntl_align_blksz( false, ( cntl_t* )&cntl->part_jc );
}

