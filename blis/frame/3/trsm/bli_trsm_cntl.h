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

struct trsm_var_cntl_s
{
	gemm_var_cntl_t gemm; //this field must be present and come first
	gemmtrsm_ukr_ft gemmtrsm_ukr;
	dim_t           mr_pack;
	dim_t           nr_pack;
	dim_t           mr_bcast;
	dim_t           nr_bcast;
};
typedef struct trsm_var_cntl_s trsm_var_cntl_t;

// -----------------------------------------------------------------------------

BLIS_INLINE gemmtrsm_ukr_ft bli_trsm_var_cntl_gemmtrsm_ukr( const cntl_t* cntl )
{
	return ( ( const trsm_var_cntl_t* ) cntl )->gemmtrsm_ukr;
}

BLIS_INLINE gemm_ukr_ft bli_trsm_var_cntl_gemm_ukr( const cntl_t* cntl )
{
	return bli_gemm_var_cntl_ukr( cntl );
}

BLIS_INLINE gemm_ukr_ft bli_trsm_var_cntl_real_gemm_ukr( const cntl_t* cntl )
{
	return bli_gemm_var_cntl_real_ukr( cntl );
}

BLIS_INLINE bool bli_trsm_var_cntl_row_pref( const cntl_t* cntl )
{
	return bli_gemm_var_cntl_row_pref( cntl );
}

BLIS_INLINE const void* bli_trsm_var_cntl_params( const cntl_t* cntl )
{
	return bli_gemm_var_cntl_params( cntl );
}

BLIS_INLINE const void* bli_trsm_var_cntl_real_params( const cntl_t* cntl )
{
	return bli_gemm_var_cntl_real_params( cntl );
}

BLIS_INLINE dim_t bli_trsm_var_cntl_mr( const cntl_t* cntl )
{
	return bli_gemm_var_cntl_mr( cntl );
}

BLIS_INLINE dim_t bli_trsm_var_cntl_nr( const cntl_t* cntl )
{
	return bli_gemm_var_cntl_nr( cntl );
}

BLIS_INLINE dim_t bli_trsm_var_cntl_mr_pack( const cntl_t* cntl )
{
	return ( ( const trsm_var_cntl_t* ) cntl )->mr_pack;
}

BLIS_INLINE dim_t bli_trsm_var_cntl_nr_pack( const cntl_t* cntl )
{
	return ( ( const trsm_var_cntl_t* ) cntl )->nr_pack;
}

BLIS_INLINE dim_t bli_trsm_var_cntl_mr_bcast( const cntl_t* cntl )
{
	return ( ( const trsm_var_cntl_t* ) cntl )->mr_bcast;
}

BLIS_INLINE dim_t bli_trsm_var_cntl_nr_bcast( const cntl_t* cntl )
{
	return ( ( const trsm_var_cntl_t* ) cntl )->nr_bcast;
}

// -----------------------------------------------------------------------------

BLIS_INLINE void bli_trsm_var_cntl_set_gemmtrsm_ukr( const func2_t* ukr, cntl_t* cntl_ )
{
	trsm_var_cntl_t* cntl = ( trsm_var_cntl_t* )cntl_;
	num_t dt_comp = cntl->gemm.dt_comp;
	num_t dt_out = cntl->gemm.dt_out;
	cntl->gemmtrsm_ukr = ( gemmtrsm_ukr_ft )bli_func2_get_dt( dt_comp, dt_out, ukr );
}

BLIS_INLINE err_t bli_trsm_var_cntl_set_gemmtrsm_ukr_simple( const func_t* ukr, cntl_t* cntl_ )
{
	trsm_var_cntl_t* cntl = ( trsm_var_cntl_t* )cntl_;
	num_t dt_comp = cntl->gemm.dt_comp;
	num_t dt_out = cntl->gemm.dt_out;
	if ( dt_comp != dt_out )
		return BLIS_INCONSISTENT_DATATYPES;
	cntl->gemmtrsm_ukr = ( gemmtrsm_ukr_ft )bli_func_get_dt( dt_comp, ukr );
	return BLIS_SUCCESS;
}

BLIS_INLINE void bli_trsm_var_cntl_set_gemm_ukr( const func2_t* ukr, cntl_t* cntl )
{
	bli_gemm_var_cntl_set_ukr( ukr, cntl );
}

BLIS_INLINE void bli_trsm_var_cntl_set_real_gemm_ukr( const func2_t* ukr, cntl_t* cntl )
{
	bli_gemm_var_cntl_set_real_ukr( ukr, cntl );
}

BLIS_INLINE err_t bli_trsm_var_cntl_set_gemm_ukr_simple( const func_t* ukr, cntl_t* cntl )
{
	return bli_gemm_var_cntl_set_ukr_simple( ukr, cntl );
}

BLIS_INLINE err_t bli_trsm_var_cntl_set_real_gemm_ukr_simple( const func_t* ukr, cntl_t* cntl )
{
	return bli_gemm_var_cntl_set_real_ukr_simple( ukr, cntl );
}

BLIS_INLINE void bli_trsm_var_cntl_set_row_pref( const mbool_t* row_pref, cntl_t* cntl )
{
	bli_gemm_var_cntl_set_row_pref( row_pref, cntl );
}

BLIS_INLINE void bli_trsm_var_cntl_set_params( const void* params, cntl_t* cntl )
{
	bli_gemm_var_cntl_set_params( params, cntl );
}

BLIS_INLINE void bli_trsm_var_cntl_set_real_params( const void* params, cntl_t* cntl )
{
	bli_gemm_var_cntl_set_real_params( params, cntl );
}

BLIS_INLINE void bli_trsm_var_cntl_set_mr( dim_t mr, cntl_t* cntl )
{
	bli_gemm_var_cntl_set_mr( mr, cntl );
}

BLIS_INLINE void bli_trsm_var_cntl_set_nr( dim_t nr, cntl_t* cntl )
{
	bli_gemm_var_cntl_set_nr( nr, cntl );
}

BLIS_INLINE void bli_trsm_var_cntl_set_mr_pack( dim_t mr_pack, cntl_t* cntl )
{
	( ( trsm_var_cntl_t* ) cntl )->mr_pack = mr_pack;
}

BLIS_INLINE void bli_trsm_var_cntl_set_nr_pack( dim_t nr_pack, cntl_t* cntl )
{
	( ( trsm_var_cntl_t* ) cntl )->nr_pack = nr_pack;
}

BLIS_INLINE void bli_trsm_var_cntl_set_mr_bcast( dim_t mr_bcast, cntl_t* cntl )
{
	( ( trsm_var_cntl_t* ) cntl )->mr_bcast = mr_bcast;
}

BLIS_INLINE void bli_trsm_var_cntl_set_nr_bcast( dim_t nr_bcast, cntl_t* cntl )
{
	( ( trsm_var_cntl_t* ) cntl )->nr_bcast = nr_bcast;
}

// -----------------------------------------------------------------------------

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
     );

// -----------------------------------------------------------------------------

struct trsm_cntl_s
{
         part_cntl_t part_jc;
         part_cntl_t part_pc;
    packm_def_cntl_t pack_b;
         part_cntl_t part_ic;
    packm_def_cntl_t pack_a_trsm;
     trsm_var_cntl_t trsm_ker;
    packm_def_cntl_t pack_a_gemm;
     trsm_var_cntl_t gemm_ker;
              cntl_t ir_loop_gemm;
              cntl_t ir_loop_trsm;
};
typedef struct trsm_cntl_s trsm_cntl_t;

// -----------------------------------------------------------------------------

BLIS_EXPORT_BLIS void bli_trsm_cntl_init
     (
             ind_t        im,
       const obj_t*       alpha,
             obj_t*       a,
             obj_t*       b,
       const obj_t*       beta,
             obj_t*       c,
       const cntx_t*      cntx,
             trsm_cntl_t* cntl
     );

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
     );

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
     );

BLIS_EXPORT_BLIS void bli_trsm_cntl_finalize
     (
       trsm_cntl_t* cntl
     );

// -----------------------------------------------------------------------------

BLIS_INLINE gemmtrsm_ukr_ft bli_trsm_cntl_gemmtrsm_ukr( trsm_cntl_t* cntl )
{
	return bli_trsm_var_cntl_gemmtrsm_ukr( ( cntl_t* )&cntl->trsm_ker );
}

BLIS_INLINE gemm_ukr_ft bli_trsm_cntl_gemm_ukr( trsm_cntl_t* cntl )
{
	gemm_ukr_ft real_ukr = bli_trsm_var_cntl_real_gemm_ukr( ( cntl_t* )&cntl->trsm_ker );
	return real_ukr ? real_ukr : bli_trsm_var_cntl_gemm_ukr( ( cntl_t* )&cntl->trsm_ker );
}

BLIS_INLINE bool bli_trsm_cntl_row_pref( trsm_cntl_t* cntl )
{
	return bli_trsm_var_cntl_row_pref( ( cntl_t* )&cntl->trsm_ker );
}

BLIS_INLINE const void* bli_trsm_cntl_params( trsm_cntl_t* cntl )
{
	gemm_ukr_ft real_ukr = bli_trsm_var_cntl_real_gemm_ukr( ( cntl_t* )&cntl->trsm_ker );
	return real_ukr ? bli_trsm_var_cntl_real_params( ( cntl_t* )&cntl->trsm_ker )
	                : bli_trsm_var_cntl_params( ( cntl_t* )&cntl->trsm_ker );
}

BLIS_INLINE l3_var_oft bli_trsm_cntl_var( trsm_cntl_t* cntl )
{
	return ( l3_var_oft )bli_cntl_var_func( ( cntl_t* )&cntl->trsm_ker );
}

BLIS_INLINE packm_ker_ft bli_trsm_cntl_packa_ukr( trsm_cntl_t* cntl )
{
	return bli_packm_def_cntl_ukr( ( cntl_t* )&cntl->pack_a_trsm );
}

BLIS_INLINE pack_t bli_trsm_cntl_packa_schema( trsm_cntl_t* cntl )
{
	return bli_packm_def_cntl_pack_schema( ( cntl_t* )&cntl->pack_a_trsm );
}

BLIS_INLINE const void* bli_trsm_cntl_packa_params( trsm_cntl_t* cntl )
{
	return bli_packm_def_cntl_ukr_params( ( cntl_t* )&cntl->pack_a_trsm );
}

BLIS_INLINE packm_var_oft bli_trsm_cntl_packa_var( trsm_cntl_t* cntl )
{
	return bli_packm_cntl_variant( ( cntl_t* )&cntl->pack_a_trsm );
}

BLIS_INLINE packm_ker_ft bli_trsm_cntl_packb_ukr( trsm_cntl_t* cntl )
{
	return bli_packm_def_cntl_ukr( ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE pack_t bli_trsm_cntl_packb_schema( trsm_cntl_t* cntl )
{
	return bli_packm_def_cntl_pack_schema( ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE const void* bli_trsm_cntl_packb_params( trsm_cntl_t* cntl )
{
	return bli_packm_def_cntl_ukr_params( ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE packm_var_oft bli_trsm_cntl_packb_var( trsm_cntl_t* cntl )
{
	return bli_packm_cntl_variant( ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE dim_t bli_trsm_cntl_mr_def( trsm_cntl_t* cntl )
{
	return bli_packm_def_cntl_bmult_m_def( ( cntl_t* )&cntl->pack_a_trsm );
}

BLIS_INLINE dim_t bli_trsm_cntl_mr_pack( trsm_cntl_t* cntl )
{
	return bli_packm_def_cntl_bmult_m_pack( ( cntl_t* )&cntl->pack_a_trsm );
}

BLIS_INLINE dim_t bli_trsm_cntl_nr_def( trsm_cntl_t* cntl )
{
	return bli_packm_def_cntl_bmult_m_def( ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE dim_t bli_trsm_cntl_nr_pack( trsm_cntl_t* cntl )
{
	return bli_packm_def_cntl_bmult_m_pack( ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE dim_t bli_trsm_cntl_mc_def( trsm_cntl_t* cntl )
{
	return bli_part_cntl_blksz_alg( ( cntl_t* )&cntl->part_ic );
}

BLIS_INLINE dim_t bli_trsm_cntl_mc_max( trsm_cntl_t* cntl )
{
	return bli_part_cntl_blksz_max( ( cntl_t* )&cntl->part_ic );
}

BLIS_INLINE dim_t bli_trsm_cntl_nc_def( trsm_cntl_t* cntl )
{
	return bli_part_cntl_blksz_alg( ( cntl_t* )&cntl->part_jc );
}

BLIS_INLINE dim_t bli_trsm_cntl_nc_max( trsm_cntl_t* cntl )
{
	return bli_part_cntl_blksz_max( ( cntl_t* )&cntl->part_jc );
}

BLIS_INLINE dim_t bli_trsm_cntl_kc_def( trsm_cntl_t* cntl )
{
	return bli_part_cntl_blksz_alg( ( cntl_t* )&cntl->part_pc );
}

BLIS_INLINE dim_t bli_trsm_cntl_kc_max( trsm_cntl_t* cntl )
{
	return bli_part_cntl_blksz_max( ( cntl_t* )&cntl->part_pc );
}

// -----------------------------------------------------------------------------

BLIS_INLINE void bli_trsm_cntl_set_gemmtrsm_ukr( const func2_t* ukr, trsm_cntl_t* cntl )
{
	bli_trsm_var_cntl_set_gemmtrsm_ukr( ukr, ( cntl_t* )&cntl->trsm_ker );
	bli_trsm_var_cntl_set_gemmtrsm_ukr( ukr, ( cntl_t* )&cntl->gemm_ker );
}

BLIS_INLINE void bli_trsm_cntl_set_gemmtrsm_ukr_simple( const func_t* ukr, trsm_cntl_t* cntl )
{
	bli_trsm_var_cntl_set_gemmtrsm_ukr_simple( ukr, ( cntl_t* )&cntl->trsm_ker );
	bli_trsm_var_cntl_set_gemmtrsm_ukr_simple( ukr, ( cntl_t* )&cntl->gemm_ker );
}

BLIS_INLINE void bli_trsm_cntl_set_gemm_ukr( const func2_t* ukr, trsm_cntl_t* cntl )
{
	if ( bli_trsm_var_cntl_real_gemm_ukr( ( cntl_t* )&cntl->trsm_ker ) )
	{
		bli_trsm_var_cntl_set_real_gemm_ukr( ukr, ( cntl_t* )&cntl->trsm_ker );
		bli_trsm_var_cntl_set_real_gemm_ukr( ukr, ( cntl_t* )&cntl->gemm_ker );
	}
	else
	{
		bli_trsm_var_cntl_set_gemm_ukr( ukr, ( cntl_t* )&cntl->trsm_ker );
		bli_trsm_var_cntl_set_gemm_ukr( ukr, ( cntl_t* )&cntl->gemm_ker );
	}
}

BLIS_INLINE void bli_trsm_cntl_set_gemm_ukr_simple( const func_t* ukr, trsm_cntl_t* cntl )
{
	if ( bli_trsm_var_cntl_real_gemm_ukr( ( cntl_t* )&cntl->trsm_ker ) )
	{
		bli_trsm_var_cntl_set_real_gemm_ukr_simple( ukr, ( cntl_t* )&cntl->trsm_ker );
		bli_trsm_var_cntl_set_real_gemm_ukr_simple( ukr, ( cntl_t* )&cntl->gemm_ker );
	}
	else
	{
		bli_trsm_var_cntl_set_gemm_ukr_simple( ukr, ( cntl_t* )&cntl->trsm_ker );
		bli_trsm_var_cntl_set_gemm_ukr_simple( ukr, ( cntl_t* )&cntl->gemm_ker );
	}
}

BLIS_INLINE void bli_trsm_cntl_set_row_pref( const mbool_t* row_pref, trsm_cntl_t* cntl )
{
	bli_trsm_var_cntl_set_row_pref( row_pref, ( cntl_t* )&cntl->trsm_ker );
	bli_trsm_var_cntl_set_row_pref( row_pref, ( cntl_t* )&cntl->gemm_ker );
}

BLIS_INLINE void bli_trsm_cntl_set_params( const void* params, trsm_cntl_t* cntl )
{
	if ( bli_trsm_var_cntl_real_gemm_ukr( ( cntl_t* )&cntl->trsm_ker ) )
	{
		bli_trsm_var_cntl_set_real_params( params, ( cntl_t* )&cntl->trsm_ker );
		bli_trsm_var_cntl_set_real_params( params, ( cntl_t* )&cntl->gemm_ker );
	}
	else
	{
		bli_trsm_var_cntl_set_params( params, ( cntl_t* )&cntl->trsm_ker );
		bli_trsm_var_cntl_set_params( params, ( cntl_t* )&cntl->gemm_ker );
	}
}

BLIS_INLINE void bli_trsm_cntl_set_var( l3_var_oft var, trsm_cntl_t* cntl )
{
	bli_cntl_set_var_func( ( void_fp )var, ( cntl_t* )&cntl->trsm_ker );
	bli_cntl_set_var_func( ( void_fp )var, ( cntl_t* )&cntl->gemm_ker );
}

BLIS_INLINE void bli_trsm_cntl_set_packa_ukr( const func2_t* ukr, trsm_cntl_t* cntl )
{
	bli_packm_def_cntl_set_ukr( ukr, ( cntl_t* )&cntl->pack_a_gemm );
	bli_packm_def_cntl_set_ukr( ukr, ( cntl_t* )&cntl->pack_a_trsm );
}

BLIS_INLINE void bli_trsm_cntl_set_packa_ukr_simple( const func_t* ukr, trsm_cntl_t* cntl )
{
	bli_packm_def_cntl_set_ukr_simple( ukr, ( cntl_t* )&cntl->pack_a_gemm );
	bli_packm_def_cntl_set_ukr_simple( ukr, ( cntl_t* )&cntl->pack_a_trsm );
}

BLIS_INLINE void bli_trsm_cntl_set_packa_schema( pack_t schema, trsm_cntl_t* cntl )
{
	bli_packm_def_cntl_set_pack_schema( schema, ( cntl_t* )&cntl->pack_a_gemm );
	bli_packm_def_cntl_set_pack_schema( schema, ( cntl_t* )&cntl->pack_a_trsm );
}

BLIS_INLINE void bli_trsm_cntl_set_packa_params( const void* params, trsm_cntl_t* cntl )
{
	bli_packm_def_cntl_set_ukr_params( params, ( cntl_t* )&cntl->pack_a_gemm );
	bli_packm_def_cntl_set_ukr_params( params, ( cntl_t* )&cntl->pack_a_trsm );
	bli_packm_cntl_set_variant_params( params, ( cntl_t* )&cntl->pack_a_gemm );
	bli_packm_cntl_set_variant_params( params, ( cntl_t* )&cntl->pack_a_trsm );
}

BLIS_INLINE void bli_trsm_cntl_set_packa_var( packm_var_oft var, trsm_cntl_t* cntl )
{
	bli_packm_cntl_set_variant( var, ( cntl_t* )&cntl->pack_a_gemm );
	bli_packm_cntl_set_variant( var, ( cntl_t* )&cntl->pack_a_trsm );
}

BLIS_INLINE void bli_trsm_cntl_set_packb_ukr( const func2_t* ukr, trsm_cntl_t* cntl )
{
	bli_packm_def_cntl_set_ukr( ukr, ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE void bli_trsm_cntl_set_packb_ukr_simple( const func_t* ukr, trsm_cntl_t* cntl )
{
	bli_packm_def_cntl_set_ukr_simple( ukr, ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE void bli_trsm_cntl_set_packb_schema( pack_t schema, trsm_cntl_t* cntl )
{
	bli_packm_def_cntl_set_pack_schema( schema, ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE void bli_trsm_cntl_set_packb_params( const void* params, trsm_cntl_t* cntl )
{
	bli_packm_def_cntl_set_ukr_params( params, ( cntl_t* )&cntl->pack_b );
	bli_packm_cntl_set_variant_params( params, ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE void bli_trsm_cntl_set_packb_var( packm_var_oft var, trsm_cntl_t* cntl )
{
	bli_packm_cntl_set_variant( var, ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE void bli_trsm_cntl_set_mr( const blksz_t* mr, trsm_cntl_t* cntl )
{
	num_t dt = cntl->gemm_ker.gemm.dt_comp;
	dim_t mr_dt = bli_blksz_get_def( dt, mr );
	dim_t mr_pack_dt = bli_blksz_get_max( dt, mr );
	bli_packm_def_cntl_set_bmult_m( mr, ( cntl_t* )&cntl->pack_a_trsm );
	bli_packm_def_cntl_set_bmult_m( mr, ( cntl_t* )&cntl->pack_a_gemm );
	bli_part_cntl_set_blksz_mult( mr, ( cntl_t* )&cntl->part_ic );
	bli_trsm_var_cntl_set_mr( mr_dt, ( cntl_t* )&cntl->gemm_ker );
	bli_trsm_var_cntl_set_mr( mr_dt, ( cntl_t* )&cntl->trsm_ker );
	bli_trsm_var_cntl_set_mr_pack( mr_pack_dt, ( cntl_t* )&cntl->gemm_ker );
	bli_trsm_var_cntl_set_mr_pack( mr_pack_dt, ( cntl_t* )&cntl->trsm_ker );
}

BLIS_INLINE void bli_trsm_cntl_set_mr_bcast( const blksz_t* mr_bcast, trsm_cntl_t* cntl )
{
	num_t dt = cntl->gemm_ker.gemm.dt_comp;
	dim_t mr_bcast_dt = bli_blksz_get_def( dt, mr_bcast );
	bli_packm_def_cntl_set_bmult_m_bcast( mr_bcast, ( cntl_t* )&cntl->pack_a_trsm );
	bli_packm_def_cntl_set_bmult_m_bcast( mr_bcast, ( cntl_t* )&cntl->pack_a_gemm );
	bli_trsm_var_cntl_set_mr_bcast( mr_bcast_dt, ( cntl_t* )&cntl->gemm_ker );
	bli_trsm_var_cntl_set_mr_bcast( mr_bcast_dt, ( cntl_t* )&cntl->trsm_ker );
}

BLIS_INLINE void bli_trsm_cntl_set_nr( const blksz_t* nr, trsm_cntl_t* cntl )
{
	num_t dt = cntl->gemm_ker.gemm.dt_comp;
	dim_t nr_dt = bli_blksz_get_def( dt, nr );
	dim_t nr_pack_dt = bli_blksz_get_max( dt, nr );
	bli_packm_def_cntl_set_bmult_m( nr, ( cntl_t* )&cntl->pack_b );
	bli_part_cntl_set_blksz_mult( nr, ( cntl_t* )&cntl->part_jc );
	bli_trsm_var_cntl_set_nr( nr_dt, ( cntl_t* )&cntl->gemm_ker );
	bli_trsm_var_cntl_set_nr( nr_dt, ( cntl_t* )&cntl->trsm_ker );
	bli_trsm_var_cntl_set_nr_pack( nr_pack_dt, ( cntl_t* )&cntl->gemm_ker );
	bli_trsm_var_cntl_set_nr_pack( nr_pack_dt, ( cntl_t* )&cntl->trsm_ker );
}

BLIS_INLINE void bli_trsm_cntl_set_nr_bcast( const blksz_t* nr_bcast, trsm_cntl_t* cntl )
{
	num_t dt = cntl->gemm_ker.gemm.dt_comp;
	dim_t nr_bcast_dt = bli_blksz_get_def( dt, nr_bcast );
	bli_packm_def_cntl_set_bmult_m_bcast( nr_bcast, ( cntl_t* )&cntl->pack_b );
	bli_trsm_var_cntl_set_nr_bcast( nr_bcast_dt, ( cntl_t* )&cntl->gemm_ker );
	bli_trsm_var_cntl_set_nr_bcast( nr_bcast_dt, ( cntl_t* )&cntl->trsm_ker );
}

BLIS_INLINE void bli_trsm_cntl_set_mc( const blksz_t* mc, trsm_cntl_t* cntl )
{
	bli_part_cntl_set_blksz( mc, ( cntl_t* )&cntl->part_ic );
}

BLIS_INLINE void bli_trsm_cntl_set_nc( const blksz_t* nc, trsm_cntl_t* cntl )
{
	bli_part_cntl_set_blksz( nc, ( cntl_t* )&cntl->part_jc );
}

BLIS_INLINE void bli_trsm_cntl_set_kc( const blksz_t* kc, trsm_cntl_t* cntl )
{
	bli_part_cntl_set_blksz( kc, ( cntl_t* )&cntl->part_pc );
}

