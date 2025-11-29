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


// -----------------------------------------------------------------------------

struct gemm_var_cntl_s
{
	cntl_t      cntl; //this field must be present and come first
	num_t       dt_comp;
	num_t       dt_out;
	gemm_ukr_ft ukr;
	gemm_ukr_ft real_ukr;
	const void* params;
	const void* real_params;
	dim_t       mr;
	dim_t       nr;
	dim_t       mr_scale;
	dim_t       nr_scale;
	bool        row_pref;
};
typedef struct gemm_var_cntl_s gemm_var_cntl_t;

// -----------------------------------------------------------------------------

BLIS_INLINE gemm_ukr_ft bli_gemm_var_cntl_ukr( const cntl_t* cntl )
{
	return ( ( const gemm_var_cntl_t* ) cntl )->ukr;
}

BLIS_INLINE gemm_ukr_ft bli_gemm_var_cntl_real_ukr( const cntl_t* cntl )
{
	return ( ( const gemm_var_cntl_t* ) cntl )->real_ukr;
}

BLIS_INLINE bool bli_gemm_var_cntl_row_pref( const cntl_t* cntl )
{
	return ( ( const gemm_var_cntl_t* ) cntl )->row_pref;
}

BLIS_INLINE const void* bli_gemm_var_cntl_params( const cntl_t* cntl )
{
	return ( ( const gemm_var_cntl_t* ) cntl )->params;
}

BLIS_INLINE const void* bli_gemm_var_cntl_real_params( const cntl_t* cntl )
{
	return ( ( const gemm_var_cntl_t* ) cntl )->real_params;
}

BLIS_INLINE dim_t bli_gemm_var_cntl_mr( const cntl_t* cntl )
{
	return ( ( const gemm_var_cntl_t* ) cntl )->mr;
}

BLIS_INLINE dim_t bli_gemm_var_cntl_nr( const cntl_t* cntl )
{
	return ( ( const gemm_var_cntl_t* ) cntl )->nr;
}

BLIS_INLINE num_t bli_gemm_var_cntl_comp_dt( const cntl_t* cntl )
{
	return ( ( const gemm_var_cntl_t* ) cntl )->dt_comp;
}

// -----------------------------------------------------------------------------

BLIS_INLINE void bli_gemm_var_cntl_set_ukr( const func2_t* ukr, cntl_t* cntl_ )
{
	gemm_var_cntl_t* cntl = ( gemm_var_cntl_t* )cntl_;
	num_t dt_comp = cntl->dt_comp;
	num_t dt_out = cntl->dt_out;
	cntl->ukr = ( gemm_ukr_ft )bli_func2_get_dt( dt_comp, dt_out, ukr );
}

BLIS_INLINE void bli_gemm_var_cntl_set_real_ukr( const func2_t* ukr, cntl_t* cntl_ )
{
	gemm_var_cntl_t* cntl = ( gemm_var_cntl_t* )cntl_;
	num_t dt_comp = cntl->dt_comp;
	num_t dt_out = cntl->dt_out;
	cntl->real_ukr = ( gemm_ukr_ft )bli_func2_get_dt( dt_comp, dt_out, ukr );
}

BLIS_INLINE err_t bli_gemm_var_cntl_set_ukr_simple( const func_t* ukr, cntl_t* cntl_ )
{
	gemm_var_cntl_t* cntl = ( gemm_var_cntl_t* )cntl_;
	num_t dt_comp = cntl->dt_comp;
	num_t dt_out = cntl->dt_out;
	if ( dt_comp != dt_out )
		return BLIS_INCONSISTENT_DATATYPES;
	cntl->ukr = ( gemm_ukr_ft )bli_func_get_dt( dt_comp, ukr );
	return BLIS_SUCCESS;
}

BLIS_INLINE err_t bli_gemm_var_cntl_set_real_ukr_simple( const func_t* ukr, cntl_t* cntl_ )
{
	gemm_var_cntl_t* cntl = ( gemm_var_cntl_t* )cntl_;
	num_t dt_comp = cntl->dt_comp;
	num_t dt_out = cntl->dt_out;
	if ( dt_comp != dt_out )
		return BLIS_INCONSISTENT_DATATYPES;
	cntl->real_ukr = ( gemm_ukr_ft )bli_func_get_dt( dt_comp, ukr );
	return BLIS_SUCCESS;
}

BLIS_INLINE void bli_gemm_var_cntl_set_row_pref( const mbool_t* row_pref, cntl_t* cntl_ )
{
	gemm_var_cntl_t* cntl = ( gemm_var_cntl_t* )cntl_;
	num_t dt_comp = cntl->dt_comp;
	cntl->row_pref = bli_mbool_get_dt( dt_comp, row_pref );
}

BLIS_INLINE void bli_gemm_var_cntl_set_params( const void* params, cntl_t* cntl )
{
	( ( gemm_var_cntl_t* ) cntl )->params = params;
}

BLIS_INLINE void bli_gemm_var_cntl_set_real_params( const void* params, cntl_t* cntl )
{
	( ( gemm_var_cntl_t* ) cntl )->real_params = params;
}

BLIS_INLINE void bli_gemm_var_cntl_set_mr( dim_t mr, cntl_t* cntl )
{
	( ( gemm_var_cntl_t* ) cntl )->mr = mr / ( ( gemm_var_cntl_t* ) cntl )->mr_scale;
}

BLIS_INLINE void bli_gemm_var_cntl_set_nr( dim_t nr, cntl_t* cntl )
{
	( ( gemm_var_cntl_t* ) cntl )->nr = nr / ( ( gemm_var_cntl_t* ) cntl )->nr_scale;
}

BLIS_INLINE void bli_gemm_var_cntl_set_comp_dt( num_t dt, const cntl_t* cntl )
{
	( ( gemm_var_cntl_t* ) cntl )->dt_comp = dt;
}

// -----------------------------------------------------------------------------

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
     );

// -----------------------------------------------------------------------------

struct gemm_cntl_s
{
         part_cntl_t part_jc;
         part_cntl_t part_pc;
    packm_def_cntl_t pack_b;
         part_cntl_t part_ic;
    packm_def_cntl_t pack_a;
     gemm_var_cntl_t ker;
              cntl_t ir_loop;
};
typedef struct gemm_cntl_s gemm_cntl_t;

// -----------------------------------------------------------------------------

BLIS_EXPORT_BLIS bool bli_gemm_cntl_init
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
     );

BLIS_EXPORT_BLIS void bli_gemm_cntl_finalize
     (
             opid_t       family,
       const obj_t*       a,
       const obj_t*       b,
       const obj_t*       c,
             gemm_cntl_t* cntl
     );

// -----------------------------------------------------------------------------

BLIS_INLINE gemm_ukr_ft bli_gemm_cntl_ukr( gemm_cntl_t* cntl )
{
	gemm_ukr_ft real_ukr = bli_gemm_var_cntl_real_ukr( ( cntl_t* )&cntl->ker );
	return real_ukr ? real_ukr : bli_gemm_var_cntl_ukr( ( cntl_t* )&cntl->ker );
}

BLIS_INLINE bool bli_gemm_cntl_row_pref( gemm_cntl_t* cntl )
{
	return bli_gemm_var_cntl_row_pref( ( cntl_t* )&cntl->ker );
}

BLIS_INLINE const void* bli_gemm_cntl_params( gemm_cntl_t* cntl )
{
	gemm_ukr_ft real_ukr = bli_gemm_var_cntl_real_ukr( ( cntl_t* )&cntl->ker );
	return real_ukr ? bli_gemm_var_cntl_real_params( ( cntl_t* )&cntl->ker )
	                : bli_gemm_var_cntl_params( ( cntl_t* )&cntl->ker );
}

BLIS_INLINE l3_var_oft bli_gemm_cntl_var( gemm_cntl_t* cntl )
{
	return ( l3_var_oft )bli_cntl_var_func( ( cntl_t* )&cntl->ker );
}

BLIS_INLINE packm_ker_ft bli_gemm_cntl_packa_ukr( gemm_cntl_t* cntl )
{
	return bli_packm_def_cntl_ukr( ( cntl_t* )&cntl->pack_a );
}

BLIS_INLINE pack_t bli_gemm_cntl_packa_schema( gemm_cntl_t* cntl )
{
	return bli_packm_def_cntl_pack_schema( ( cntl_t* )&cntl->pack_a );
}

BLIS_INLINE const void* bli_gemm_cntl_packa_params( gemm_cntl_t* cntl )
{
	return bli_packm_def_cntl_ukr_params( ( cntl_t* )&cntl->pack_a );
}

BLIS_INLINE packm_var_oft bli_gemm_cntl_packa_var( gemm_cntl_t* cntl )
{
	return bli_packm_cntl_variant( ( cntl_t* )&cntl->pack_a );
}

BLIS_INLINE packm_ker_ft bli_gemm_cntl_packb_ukr( gemm_cntl_t* cntl )
{
	return bli_packm_def_cntl_ukr( ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE pack_t bli_gemm_cntl_packb_schema( gemm_cntl_t* cntl )
{
	return bli_packm_def_cntl_pack_schema( ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE const void* bli_gemm_cntl_packb_params( gemm_cntl_t* cntl )
{
	return bli_packm_def_cntl_ukr_params( ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE packm_var_oft bli_gemm_cntl_packb_var( gemm_cntl_t* cntl )
{
	return bli_packm_cntl_variant( ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE dim_t bli_gemm_cntl_mr_def( gemm_cntl_t* cntl )
{
	return bli_packm_def_cntl_bmult_m_def( ( cntl_t* )&cntl->pack_a );
}

BLIS_INLINE dim_t bli_gemm_cntl_mr_pack( gemm_cntl_t* cntl )
{
	return bli_packm_def_cntl_bmult_m_pack( ( cntl_t* )&cntl->pack_a );
}

BLIS_INLINE dim_t bli_gemm_cntl_nr_def( gemm_cntl_t* cntl )
{
	return bli_packm_def_cntl_bmult_m_def( ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE dim_t bli_gemm_cntl_nr_pack( gemm_cntl_t* cntl )
{
	return bli_packm_def_cntl_bmult_m_pack( ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE dim_t bli_gemm_cntl_mc_def( gemm_cntl_t* cntl )
{
	return bli_part_cntl_blksz_alg( ( cntl_t* )&cntl->part_ic );
}

BLIS_INLINE dim_t bli_gemm_cntl_mc_max( gemm_cntl_t* cntl )
{
	return bli_part_cntl_blksz_max( ( cntl_t* )&cntl->part_ic );
}

BLIS_INLINE dim_t bli_gemm_cntl_nc_def( gemm_cntl_t* cntl )
{
	return bli_part_cntl_blksz_alg( ( cntl_t* )&cntl->part_jc );
}

BLIS_INLINE dim_t bli_gemm_cntl_nc_max( gemm_cntl_t* cntl )
{
	return bli_part_cntl_blksz_max( ( cntl_t* )&cntl->part_jc );
}

BLIS_INLINE dim_t bli_gemm_cntl_kc_def( gemm_cntl_t* cntl )
{
	return bli_part_cntl_blksz_alg( ( cntl_t* )&cntl->part_pc );
}

BLIS_INLINE dim_t bli_gemm_cntl_kc_max( gemm_cntl_t* cntl )
{
	return bli_part_cntl_blksz_max( ( cntl_t* )&cntl->part_pc );
}

// -----------------------------------------------------------------------------

BLIS_INLINE void bli_gemm_cntl_set_ukr( const func2_t* ukr, gemm_cntl_t* cntl )
{
	if ( bli_gemm_var_cntl_real_ukr( ( cntl_t* )&cntl->ker ) )
	{
		bli_gemm_var_cntl_set_real_ukr( ukr, ( cntl_t* )&cntl->ker );
	}
	else
	{
		bli_gemm_var_cntl_set_ukr( ukr, ( cntl_t* )&cntl->ker );
	}
}

BLIS_INLINE void bli_gemm_cntl_set_ukr_simple( const func_t* ukr, gemm_cntl_t* cntl )
{
	if ( bli_gemm_var_cntl_real_ukr( ( cntl_t* )&cntl->ker ) )
	{
		bli_gemm_var_cntl_set_real_ukr_simple( ukr, ( cntl_t* )&cntl->ker );
	}
	else
	{
		bli_gemm_var_cntl_set_ukr_simple( ukr, ( cntl_t* )&cntl->ker );
	}
}

BLIS_INLINE void bli_gemm_cntl_set_row_pref( const mbool_t* row_pref, gemm_cntl_t* cntl )
{
	bli_gemm_var_cntl_set_row_pref( row_pref, ( cntl_t* )&cntl->ker );
}

BLIS_INLINE void bli_gemm_cntl_set_params( const void* params, gemm_cntl_t* cntl )
{
	if ( bli_gemm_var_cntl_real_ukr( ( cntl_t* )&cntl->ker ) )
	{
		bli_gemm_var_cntl_set_real_params( params, ( cntl_t* )&cntl->ker );
	}
	else
	{
		bli_gemm_var_cntl_set_params( params, ( cntl_t* )&cntl->ker );
	}
}

BLIS_INLINE void bli_gemm_cntl_set_var( l3_var_oft var, gemm_cntl_t* cntl )
{
	bli_cntl_set_var_func( ( void_fp )var, ( cntl_t* )&cntl->ker );
}

BLIS_INLINE void bli_gemm_cntl_set_packa_ukr( const func2_t* ukr, gemm_cntl_t* cntl )
{
	bli_packm_def_cntl_set_ukr( ukr, ( cntl_t* )&cntl->pack_a );
}

BLIS_INLINE void bli_gemm_cntl_set_packa_ukr_simple( const func_t* ukr, gemm_cntl_t* cntl )
{
	bli_packm_def_cntl_set_ukr_simple( ukr, ( cntl_t* )&cntl->pack_a );
}

BLIS_INLINE void bli_gemm_cntl_set_packa_schema( pack_t schema, gemm_cntl_t* cntl )
{
	bli_packm_def_cntl_set_pack_schema( schema, ( cntl_t* )&cntl->pack_a );
}

BLIS_INLINE void bli_gemm_cntl_set_packa_params( const void* params, gemm_cntl_t* cntl )
{
	bli_packm_def_cntl_set_ukr_params( params, ( cntl_t* )&cntl->pack_a );
	bli_packm_cntl_set_variant_params( params, ( cntl_t* )&cntl->pack_a );
}

BLIS_INLINE void bli_gemm_cntl_set_packa_var( packm_var_oft var, gemm_cntl_t* cntl )
{
	bli_packm_cntl_set_variant( var, ( cntl_t* )&cntl->pack_a );
}

BLIS_INLINE void bli_gemm_cntl_set_packb_ukr( const func2_t* ukr, gemm_cntl_t* cntl )
{
	bli_packm_def_cntl_set_ukr( ukr, ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE void bli_gemm_cntl_set_packb_ukr_simple( const func_t* ukr, gemm_cntl_t* cntl )
{
	bli_packm_def_cntl_set_ukr_simple( ukr, ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE void bli_gemm_cntl_set_packb_schema( pack_t schema, gemm_cntl_t* cntl )
{
	bli_packm_def_cntl_set_pack_schema( schema, ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE void bli_gemm_cntl_set_packb_params( const void* params, gemm_cntl_t* cntl )
{
	bli_packm_def_cntl_set_ukr_params( params, ( cntl_t* )&cntl->pack_b );
	bli_packm_cntl_set_variant_params( params, ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE void bli_gemm_cntl_set_packb_var( packm_var_oft var, gemm_cntl_t* cntl )
{
	bli_packm_cntl_set_variant( var, ( cntl_t* )&cntl->pack_b );
}

BLIS_INLINE void bli_gemm_cntl_set_mr( const blksz_t* mr, gemm_cntl_t* cntl )
{
	num_t dt = cntl->ker.dt_comp;
	dim_t mr_dt = bli_blksz_get_def( dt, mr );
	bli_packm_def_cntl_set_bmult_m( mr, ( cntl_t* )&cntl->pack_a );
	bli_part_cntl_set_blksz_mult( mr, ( cntl_t* )&cntl->part_ic );
	bli_gemm_var_cntl_set_mr( mr_dt, ( cntl_t* )&cntl->ker );
}

BLIS_INLINE void bli_gemm_cntl_set_nr( const blksz_t* nr, gemm_cntl_t* cntl )
{
	num_t dt = cntl->ker.dt_comp;
	dim_t nr_dt = bli_blksz_get_def( dt, nr );
	bli_packm_def_cntl_set_bmult_m( nr, ( cntl_t* )&cntl->pack_b );
	bli_part_cntl_set_blksz_mult( nr, ( cntl_t* )&cntl->part_jc );
	bli_gemm_var_cntl_set_nr( nr_dt, ( cntl_t* )&cntl->ker );
}

BLIS_INLINE void bli_gemm_cntl_set_mc( const blksz_t* mc, gemm_cntl_t* cntl )
{
	bli_part_cntl_set_blksz( mc, ( cntl_t* )&cntl->part_ic );
}

BLIS_INLINE void bli_gemm_cntl_set_nc( const blksz_t* nc, gemm_cntl_t* cntl )
{
	bli_part_cntl_set_blksz( nc, ( cntl_t* )&cntl->part_jc );
}

BLIS_INLINE void bli_gemm_cntl_set_kc( const blksz_t* kc, gemm_cntl_t* cntl )
{
	bli_part_cntl_set_blksz( kc, ( cntl_t* )&cntl->part_pc );
}

