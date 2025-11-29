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

void bli_trsm_var_cntl_init_node
     (
       void_fp          var_func,
       num_t            dt_comp,
       num_t            dt_out,
       gemmtrsm_ukr_ft  gemmtrsm_ukr,
       gemm_ukr_ft      gemm_ukr,
       gemm_ukr_ft      real_gemm_ukr,
       bool             row_pref,
       dim_t            mr,
       dim_t            nr,
       dim_t            mr_pack,
       dim_t            nr_pack,
       dim_t            mr_bcast,
       dim_t            nr_bcast,
       dim_t            mr_scale,
       dim_t            nr_scale,
       trsm_var_cntl_t* cntl
     )
{
	// Initialize the embedded gemm_var_cntl_t struct.
	bli_gemm_var_cntl_init_node
	(
	  var_func,
	  dt_comp,
	  dt_out,
	  gemm_ukr,
	  real_gemm_ukr,
	  row_pref,
	  mr,
	  nr,
	  mr_scale,
	  nr_scale,
	  ( gemm_var_cntl_t* )cntl
	);

	// Initialize the trsm_var_cntl_t struct.
	cntl->gemmtrsm_ukr  = gemmtrsm_ukr;
	cntl->mr_pack       = mr_pack;
	cntl->nr_pack       = nr_pack;
	cntl->mr_bcast      = mr_bcast;
	cntl->nr_bcast      = nr_bcast;
}

void bli_trsm_cntl_init
     (
             ind_t        im,
       const obj_t*       alpha,
             obj_t*       a,
             obj_t*       b,
       const obj_t*       beta,
             obj_t*       c,
       const cntx_t*      cntx,
             trsm_cntl_t* cntl
     )
{
	if ( bli_obj_is_triangular( a ) )
		bli_trsm_l_cntl_init( im, alpha, a, b, beta, c, cntx, cntl );
	else
		bli_check_error_code(BLIS_NOT_YET_IMPLEMENTED);
		//bli_trsm_r_cntl_init( im, alpha, a, b, beta, c, cntx, cntl );
}

void bli_trsm_l_cntl_init
     (
             ind_t        im,
       const obj_t*       alpha,
             obj_t*       a,
             obj_t*       b,
       const obj_t*       beta,
             obj_t*       c,
       const cntx_t*      cntx,
             trsm_cntl_t* cntl
     )
{
	const prec_t           comp_prec      = bli_obj_comp_prec( c );
	const num_t            dt_a           = bli_obj_dt( a );
	const num_t            dt_b           = bli_obj_dt( b );
	const num_t            dt_c           = bli_obj_dt( c );
	const num_t            dt_ap          = bli_dt_domain( dt_a ) | comp_prec;
	const num_t            dt_bp          = bli_dt_domain( dt_b ) | comp_prec;
	const num_t            dt_comp        = ( im == BLIS_1M ? BLIS_REAL
	                                                        : bli_dt_domain( dt_c )
	                                        ) | comp_prec;

	const void_fp          macro_kernel_p = bli_obj_is_lower( a ) ? bli_trsm_ll_ker_var2
	                                                              : bli_trsm_lu_ker_var2;
	      gemmtrsm_ukr_ft  gemmtrsm_ukr   = bli_obj_is_lower( a )
	                                        ? bli_cntx_get_ukr_dt( dt_comp, BLIS_GEMMTRSM_L_UKR, cntx )
	                                        : bli_cntx_get_ukr_dt( dt_comp, BLIS_GEMMTRSM_U_UKR, cntx );
	      gemm_ukr_ft      gemm_ukr       = bli_cntx_get_ukr_dt( dt_comp, BLIS_GEMM_UKR, cntx );
	      gemm_ukr_ft      real_gemm_ukr  = NULL;
	const dir_t            direct         = bli_obj_is_lower( a ) ? BLIS_FWD
	                                                              : BLIS_BWD;
	const bool             row_pref       = bli_cntx_get_ukr_prefs_dt( dt_comp, BLIS_GEMM_UKR_ROW_PREF, cntx );
	      pack_t           schema_a       = BLIS_PACKED_PANELS;
	      pack_t           schema_b       = BLIS_PACKED_PANELS;
	const packm_ker_ft     packm_a_ukr    = packm_struc_cxk[ dt_a ][ dt_ap ];
	const packm_ker_ft     packm_b_ukr    = packm_struc_cxk[ dt_b ][ dt_bp ];
	const dim_t            mr_def         = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_MR, cntx );
	const dim_t            mr_pack        = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_MR, cntx );
	const dim_t            mr_bcast       = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_BBM, cntx );
	      dim_t            mr_scale       = 1;
	      dim_t            mr_pack_scale  = 1;
	const dim_t            nr_def         = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_NR, cntx );
	const dim_t            nr_pack        = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_NR, cntx );
	const dim_t            nr_bcast       = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_BBN, cntx );
	      dim_t            nr_scale       = 1;
	      dim_t            nr_pack_scale  = 1;
	const dim_t            kr_def         = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_KR, cntx );
	const dim_t            mc_def         = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_MC, cntx );
	const dim_t            mc_max         = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_MC, cntx );
	      dim_t            mc_scale       = 1;
	const dim_t            nc_def         = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_NC, cntx );
	const dim_t            nc_max         = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_NC, cntx );
	      dim_t            nc_scale       = 1;
	const dim_t            kc_def         = bli_cntx_get_blksz_def_dt( dt_comp, BLIS_KC, cntx );
	const dim_t            kc_max         = bli_cntx_get_blksz_max_dt( dt_comp, BLIS_KC, cntx );
	      dim_t            kc_scale       = 1;

	if ( im == BLIS_1M )
	{
		if ( ! row_pref )
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
		real_gemm_ukr = gemm_ukr;
		gemm_ukr = bli_cntx_get_ukr_dt( dt_comp, BLIS_GEMM1M_UKR, cntx );
		gemmtrsm_ukr = bli_obj_is_lower( a )
		               ? bli_cntx_get_ukr_dt( dt_comp, BLIS_GEMMTRSM1M_L_UKR, cntx )
		               : bli_cntx_get_ukr_dt( dt_comp, BLIS_GEMMTRSM1M_U_UKR, cntx );
	}

	// If alpha is non-unit, typecast and apply it to the scalar attached
	// to B, unless it happens to be triangular.
	if ( bli_obj_root_is_triangular( b ) )
	{
		if ( !bli_obj_equals( alpha, &BLIS_ONE ) )
			bli_obj_scalar_apply_scalar( alpha, a );
	}
	else // if ( bli_obj_root_is_triangular( b ) )
	{
		if ( !bli_obj_equals( alpha, &BLIS_ONE ) )
			bli_obj_scalar_apply_scalar( alpha, b );
	}

	// If beta is non-unit, typecast and apply it to the scalar attached
	// to C.
	if ( !bli_obj_equals( beta, &BLIS_ONE ) )
		bli_obj_scalar_apply_scalar( beta, c );

	//
	// Create nodes for packing A and the macro-kernel (gemm branch).
	//

	bli_cntl_init_node
	(
	  NULL,         // variant function pointer not used
	  &cntl->ir_loop_gemm
	);

	bli_trsm_var_cntl_init_node
	(
	  macro_kernel_p,
	  dt_comp,
	  dt_c,
	  gemmtrsm_ukr,
	  gemm_ukr,
	  real_gemm_ukr,
	  row_pref,
	  mr_def / mr_scale,
	  nr_def / nr_scale,
	  mr_pack,
	  nr_pack,
	  mr_bcast,
	  nr_bcast,
	  mr_scale,
	  nr_scale,
	  &cntl->gemm_ker
	);
	bli_cntl_attach_sub_node
	(
	  BLIS_THREAD_MR | BLIS_THREAD_NR,
	  ( cntl_t* )&cntl->ir_loop_gemm,
	  ( cntl_t* )&cntl->gemm_ker
	);

	// Give the trsm kernel control tree node to the
	// virtual microkernel as the parameters, so that e.g.
	// the 1m virtual microkernel can look up the real-domain
	// micro-kernel and its parameters.
	bli_trsm_var_cntl_set_params( &cntl->gemm_ker, ( cntl_t* )&cntl->gemm_ker );

	// Create a node for packing matrix A.
	bli_packm_def_cntl_init_node
	(
	  bli_l3_packa, // trsm operation's packm function for A.
	  dt_a,
	  dt_ap,
	  dt_comp,
	  packm_a_ukr,
	  mr_def / mr_scale,
	  mr_pack,
	  mr_bcast,
	  mr_scale,
	  mr_pack_scale,
	  mr_def / mr_scale,
	  FALSE,        // do NOT invert diagonal
	  TRUE,         // reverse iteration if upper?
	  FALSE,        // reverse iteration if lower?
	  schema_a,
	  BLIS_BUFFER_FOR_A_BLOCK,
	  &cntl->pack_a_gemm
	);
	bli_cntl_attach_sub_node
	(
	  BLIS_THREAD_NONE,
	  ( cntl_t* )&cntl->gemm_ker,
	  ( cntl_t* )&cntl->pack_a_gemm
	);

	//
	// Create nodes for packing A and the macro-kernel (trsm branch).
	//

	bli_cntl_init_node
	(
	  NULL,
	  &cntl->ir_loop_trsm
	);

	bli_trsm_var_cntl_init_node
	(
	  macro_kernel_p,
	  dt_comp,
	  dt_c,
	  gemmtrsm_ukr,
	  gemm_ukr,
	  real_gemm_ukr,
	  row_pref,
	  mr_def / mr_scale,
	  nr_def / nr_scale,
	  mr_pack,
	  nr_pack,
	  mr_bcast,
	  nr_bcast,
	  mr_scale,
	  nr_scale,
	  &cntl->trsm_ker
	);
	bli_cntl_attach_sub_node
	(
	  BLIS_THREAD_MC | BLIS_THREAD_KC | BLIS_THREAD_NR,
	  ( cntl_t* )&cntl->ir_loop_trsm,
	  ( cntl_t* )&cntl->trsm_ker
	);

	// Give the trsm kernel control tree node to the
	// virtual microkernel as the parameters, so that e.g.
	// the 1m virtual microkernel can look up the real-domain
	// micro-kernel and its parameters.
	bli_trsm_var_cntl_set_params( &cntl->trsm_ker, ( cntl_t* )&cntl->trsm_ker );

	// Create a node for packing matrix A.
	bli_packm_def_cntl_init_node
	(
	  bli_l3_packa, // trsm operation's packm function for A.
	  dt_a,
	  dt_ap,
	  dt_comp,
	  packm_a_ukr,
	  mr_def / mr_scale,
	  mr_pack,
	  mr_bcast,
	  mr_scale,
	  mr_pack_scale,
	  mr_def / mr_scale,
#ifdef BLIS_ENABLE_TRSM_PREINVERSION
	  TRUE,         // invert diagonal
#else
	  FALSE,        // do NOT invert diagonal
#endif
	  TRUE,         // reverse iteration if upper?
	  FALSE,        // reverse iteration if lower?
	  schema_a,
	  BLIS_BUFFER_FOR_A_BLOCK,
	  &cntl->pack_a_trsm
	);
	bli_cntl_attach_sub_node
	(
	  BLIS_THREAD_NONE,
	  ( cntl_t* )&cntl->trsm_ker,
	  ( cntl_t* )&cntl->pack_a_trsm
	);

	// -------------------------------------------------------------------------

	// Create a node for partitioning the m dimension by MC.
	// NOTE: We attach the gemm sub-tree as the main branch.
	bli_part_cntl_init_node
	(
	  bli_trsm_blk_var1,
	  dt_comp,
	  mc_def / mc_scale,
	  mc_max / mc_scale,
	  mc_scale,
	  mr_def / mr_scale,
	  mr_scale,
	  direct,
	  FALSE,
	  &cntl->part_ic
	);
	bli_cntl_attach_sub_node
	(
	  BLIS_THREAD_NONE,
	  ( cntl_t* )&cntl->pack_a_trsm,
	  ( cntl_t* )&cntl->part_ic
	);
	bli_cntl_attach_sub_node
	(
	  BLIS_THREAD_MC | BLIS_THREAD_KC,
	  ( cntl_t* )&cntl->pack_a_gemm,
	  ( cntl_t* )&cntl->part_ic
	);

	// -------------------------------------------------------------------------

	// Create a node for packing matrix B.
	bli_packm_def_cntl_init_node
	(
	  bli_l3_packb,
	  dt_b,
	  dt_bp,
	  dt_comp,
	  packm_b_ukr,
	  nr_def / nr_scale,
	  nr_pack,
	  nr_bcast,
	  nr_scale,
	  nr_pack_scale,
	  mr_def / mr_scale,
	  FALSE,        // do NOT invert diagonal
	  FALSE,        // reverse iteration if upper?
	  FALSE,        // reverse iteration if lower?
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
	  bli_trsm_blk_var3,
	  dt_comp,
	  kc_def / kc_scale,
	  kc_max / kc_scale,
	  kc_scale,
	  kr_def,
	  1,
	  direct,
	  FALSE,
	  &cntl->part_pc
	);
	bli_cntl_attach_sub_node
	(
	  BLIS_THREAD_NONE,
	  ( cntl_t* )&cntl->pack_b,
	  ( cntl_t* )&cntl->part_pc
	);

	// Create a node for partitioning the n dimension by NC.
	bli_part_cntl_init_node
	(
	  bli_trsm_blk_var2,
	  dt_comp,
	  nc_def / nc_scale,
	  nc_max / nc_scale,
	  nc_scale,
	  nr_def / nr_scale,
	  nr_scale,
	  BLIS_FWD,
	  FALSE,
	  &cntl->part_jc
	);
	bli_cntl_attach_sub_node
	(
	  BLIS_THREAD_NC,
	  ( cntl_t* )&cntl->part_pc,
	  ( cntl_t* )&cntl->part_jc
	);

	bli_trsm_cntl_finalize( cntl );
}

#if 0

void bli_trsm_r_cntl_init
     (
             ind_t        im,
       const obj_t*       alpha,
             obj_t*       a,
             obj_t*       b,
       const obj_t*       beta,
             obj_t*       c,
       const cntx_t*      cntx,
             trsm_cntl_t* cntl
     )
{
	const num_t            dt_a           = bli_obj_dt( a );
	const num_t            dt_b           = bli_obj_dt( b );
	const num_t            dt_ap          = bli_obj_target_dt( a );
	const num_t            dt_bp          = bli_obj_target_dt( b );
	const num_t            dt_exec        = bli_obj_exec_dt( c );

	const void_fp          macro_kernel_p = bli_obj_is_lower( b ) ? bli_trsm_rl_ker_var2 : bli_trsm_ru_ker_var2;
	const gemmtrsm_ukr_vft gemmtrsm_ukr   = bli_obj_is_lower( b )
	    ? bli_cntx_get_ukr_dt( dt_exec, BLIS_GEMMTRSM_L_UKR, cntx )
	    : bli_cntx_get_ukr_dt( dt_exec, BLIS_GEMMTRSM_U_UKR, cntx );
	const gemm_ukr_vft     gemm_ukr       = bli_cntx_get_ukr_dt( dt_exec, BLIS_GEMM_UKR, cntx );

	const dir_t            direct         = bli_obj_is_lower( b ) ? BLIS_BWD : BLIS_FWD;
	const dim_t            ic_alg         = bli_cntx_get_blksz_def_dt( dt_exec, BLIS_MC, cntx );
	const dim_t            ic_max         = bli_cntx_get_blksz_max_dt( dt_exec, BLIS_MC, cntx );
	const dim_t            ic_mult        = bli_cntx_get_blksz_def_dt( dt_exec, BLIS_NR, cntx ); //note: different!
	      dim_t            pc_alg         = bli_cntx_get_blksz_def_dt( dt_exec, BLIS_KC, cntx );
	      dim_t            pc_max         = bli_cntx_get_blksz_max_dt( dt_exec, BLIS_KC, cntx );
	const dim_t            pc_mult        = bli_cntx_get_blksz_def_dt( dt_exec, BLIS_KR, cntx );
	const dim_t            jc_alg         = bli_cntx_get_blksz_def_dt( dt_exec, BLIS_NC, cntx );
	const dim_t            jc_max         = bli_cntx_get_blksz_max_dt( dt_exec, BLIS_NC, cntx );
	const dim_t            jc_mult        = bli_cntx_get_blksz_def_dt( dt_exec, BLIS_MR, cntx ); //note: different!

	const dim_t            bmult_m_def    = bli_cntx_get_blksz_def_dt(   dt_ap, BLIS_NR, cntx );
	const dim_t            bmult_m_pack   = bli_cntx_get_blksz_max_dt(   dt_ap, BLIS_NR, cntx );
	const dim_t            bmult_n_def    = bli_cntx_get_blksz_def_dt(   dt_bp, BLIS_MR, cntx );
	const dim_t            bmult_n_pack   = bli_cntx_get_blksz_max_dt(   dt_bp, BLIS_MR, cntx );
	const dim_t            bmult_k_def    = bmult_n_def;

	bli_l3_adjust_kc
	(
	  BLIS_TRSM,
	  a,
	  b,
	  &pc_alg,
	  &pc_max,
	  ic_mult,
	  jc_mult
	);

	// Create two nodes for the macro-kernel.
	bli_cntl_init_node
	(
	  NULL,         // variant function pointer not used
	  &cntl->ir_loop_trsm
	);

	bli_trsm_var_cntl_init_node
	(
	  macro_kernel_p,
	  gemmtrsm_ukr,
	  gemm_ukr,
	  &cntl->trsm_ker
	);
	bli_cntl_attach_sub_node
	(
	  BLIS_THREAD_NONE,
	  ( cntl_t* )&cntl->ir_loop_trsm,
	  ( cntl_t* )&cntl->trsm_ker
	);

	// Create a node for packing matrix A.
	bli_packm_def_cntl_init_node
	(
	  bli_l3_packa,
	  dt_a,
	  dt_ap,
	  bmult_m_def,
	  bmult_m_pack,
	  bmult_k_def,
	  FALSE,   // do NOT invert diagonal
	  FALSE,   // reverse iteration if upper?
	  FALSE,   // reverse iteration if lower?
	  schema_a, // normally BLIS_PACKED_PANELS
	  BLIS_BUFFER_FOR_A_BLOCK,
	  &cntl->pack_a_trsm
	);
	bli_cntl_attach_sub_node
	(
	  BLIS_THREAD_NONE,
	  ( cntl_t* )&cntl->trsm_ker,
	  ( cntl_t* )&cntl->pack_a_trsm
	);

	// Create a node for partitioning the m dimension by MC.
	bli_part_cntl_init_node
	(
	  bli_trsm_blk_var1,
	  ic_alg,
	  ic_max,
	  ic_mult,
	  BLIS_FWD,
	  FALSE,
	  &cntl->part_ic
	);
	bli_cntl_attach_sub_node
	(
	  BLIS_THREAD_MC | BLIS_THREAD_KC | BLIS_THREAD_NC | BLIS_THREAD_MR | BLIS_THREAD_NR,
	  ( cntl_t* )&cntl->pack_a_trsm,
	  ( cntl_t* )&cntl->part_ic
	);

	// Create a node for packing matrix B.
	bli_packm_def_cntl_init_node
	(
	  bli_l3_packb,
	  dt_b,
	  dt_bp,
	  bmult_n_def,
	  bmult_n_pack,
	  bmult_k_def,
	  TRUE,    // do NOT invert diagonal
	  FALSE,   // reverse iteration if upper?
	  TRUE,    // reverse iteration if lower?
	  schema_b, // normally BLIS_PACKED_PANELS
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
	  bli_trsm_blk_var3,
	  pc_alg,
	  pc_max,
	  pc_mult,
	  direct,
	  FALSE,
	  &cntl->part_pc
	);
	bli_cntl_attach_sub_node
	(
	  BLIS_THREAD_NONE,
	  ( cntl_t* )&cntl->pack_b,
	  ( cntl_t* )&cntl->part_pc
	);

	// Create a node for partitioning the n dimension by NC.
	bli_part_cntl_init_node
	(
	  bli_trsm_blk_var2,
	  jc_alg,
	  jc_max,
	  jc_mult,
	  direct,
	  FALSE,
	  &cntl->part_jc
	);
	bli_cntl_attach_sub_node
	(
	  BLIS_THREAD_NONE,
	  ( cntl_t* )&cntl->part_pc,
	  ( cntl_t* )&cntl->part_jc
	);
}

#endif

void bli_trsm_cntl_finalize
     (
       trsm_cntl_t* cntl
     )
{
	const dim_t ic_mult = bli_part_cntl_blksz_mult( ( cntl_t* )&cntl->part_ic );

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
	bli_part_cntl_align_blksz_to_mult( ic_mult, true, ( cntl_t* )&cntl->part_pc );

	bli_part_cntl_align_blksz( false, ( cntl_t* )&cntl->part_ic );
	bli_part_cntl_align_blksz( false, ( cntl_t* )&cntl->part_jc );
}

