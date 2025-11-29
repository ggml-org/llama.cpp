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


struct packm_cntl_s
{
	cntl_t        cntl; // cntl field must be present and come first.
	packm_var_oft var;
	const void*   params;
};
typedef struct packm_cntl_s packm_cntl_t;

// -----------------------------------------------------------------------------

BLIS_INLINE packm_var_oft bli_packm_cntl_variant( const cntl_t* cntl )
{
	return ( ( const packm_cntl_t* ) cntl )->var;
}

BLIS_INLINE const void* bli_packm_cntl_variant_params( const cntl_t* cntl )
{
	return ( ( const packm_cntl_t* ) cntl )->params;
}

// -----------------------------------------------------------------------------

BLIS_INLINE void bli_packm_cntl_set_variant( packm_var_oft var, cntl_t* cntl )
{
	( ( packm_cntl_t* ) cntl )->var = var;
}

BLIS_INLINE void bli_packm_cntl_set_variant_params( const void* params, cntl_t* cntl )
{
	( ( packm_cntl_t* ) cntl )->params = params;
}

// -----------------------------------------------------------------------------

struct packm_def_cntl_s
{
	packm_cntl_t cntl; // cntl field must be present and come first.
	num_t        dt_orig;
	num_t        dt_pack;
	num_t        dt_bmult;
	packm_ker_ft ukr;
	dim_t        bmult_m_def;
	dim_t        bmult_m_pack;
	dim_t        bmult_m_bcast;
	dim_t        bmult_m_scale;
	dim_t        bmult_m_pack_scale;
	dim_t        bmult_n_def;
	bool         does_invert_diag;
	bool         rev_iter_if_upper;
	bool         rev_iter_if_lower;
	pack_t       pack_schema;
	packbuf_t    pack_buf_type;
	const void*  params;
};
typedef struct packm_def_cntl_s packm_def_cntl_t;

// -----------------------------------------------------------------------------

BLIS_INLINE dim_t bli_packm_def_cntl_bmult_m_def( const cntl_t* cntl )
{
	return ( ( const packm_def_cntl_t* ) cntl )->bmult_m_def;
}

BLIS_INLINE dim_t bli_packm_def_cntl_bmult_m_pack( const cntl_t* cntl )
{
	return ( ( const packm_def_cntl_t* ) cntl )->bmult_m_pack;
}

BLIS_INLINE dim_t bli_packm_def_cntl_bmult_m_bcast( const cntl_t* cntl )
{
	return ( ( const packm_def_cntl_t* ) cntl )->bmult_m_bcast;
}

BLIS_INLINE dim_t bli_packm_def_cntl_bmult_n_def( const cntl_t* cntl )
{
	return ( ( const packm_def_cntl_t* ) cntl )->bmult_n_def;
}

BLIS_INLINE bool bli_packm_def_cntl_does_invert_diag( const cntl_t* cntl )
{
	return ( ( const packm_def_cntl_t* ) cntl )->does_invert_diag;
}

BLIS_INLINE bool bli_packm_def_cntl_rev_iter_if_upper( const cntl_t* cntl )
{
	return ( ( const packm_def_cntl_t* ) cntl )->rev_iter_if_upper;
}

BLIS_INLINE bool bli_packm_def_cntl_rev_iter_if_lower( const cntl_t* cntl )
{
	return ( ( const packm_def_cntl_t* ) cntl )->rev_iter_if_lower;
}

BLIS_INLINE pack_t bli_packm_def_cntl_pack_schema( const cntl_t* cntl )
{
	return ( ( const packm_def_cntl_t* ) cntl )->pack_schema;
}

BLIS_INLINE num_t bli_packm_def_cntl_target_dt( const cntl_t* cntl )
{
	return ( ( const packm_def_cntl_t* ) cntl )->dt_pack;
}

BLIS_INLINE packbuf_t bli_packm_def_cntl_pack_buf_type( const cntl_t* cntl )
{
	return ( ( const packm_def_cntl_t* ) cntl )->pack_buf_type;
}

BLIS_INLINE packm_ker_ft bli_packm_def_cntl_ukr( const cntl_t* cntl )
{
	return ( ( const packm_def_cntl_t* ) cntl )->ukr;
}

BLIS_INLINE const void* bli_packm_def_cntl_ukr_params( const cntl_t* cntl )
{
	return ( ( const packm_def_cntl_t* ) cntl )->params;
}

// -----------------------------------------------------------------------------

BLIS_INLINE void bli_packm_def_cntl_set_bmult_m( const blksz_t* bmult_m, cntl_t* cntl_ )
{
	packm_def_cntl_t* cntl = ( packm_def_cntl_t* )cntl_;
	num_t dt = cntl->dt_bmult;
	cntl->bmult_m_def = bli_blksz_get_def( dt, bmult_m ) / cntl->bmult_m_scale;
	cntl->bmult_m_pack = bli_blksz_get_max( dt, bmult_m ) / cntl->bmult_m_pack_scale;
}

BLIS_INLINE void bli_packm_def_cntl_set_bmult_m_bcast( const blksz_t* bmult_m_bcast, cntl_t* cntl_ )
{
	packm_def_cntl_t* cntl = ( packm_def_cntl_t* )cntl_;
	num_t dt = cntl->dt_bmult;
	cntl->bmult_m_bcast = bli_blksz_get_def( dt, bmult_m_bcast );
}

BLIS_INLINE void bli_packm_def_cntl_set_bmult_n( const blksz_t* bmult_n, cntl_t* cntl_ )
{
	packm_def_cntl_t* cntl = ( packm_def_cntl_t* )cntl_;
	num_t dt = cntl->dt_bmult;
	cntl->bmult_n_def = bli_blksz_get_def( dt, bmult_n );
}

BLIS_INLINE void bli_packm_def_cntl_set_does_invert_diag( bool does_invert_diag, cntl_t* cntl )
{
	 ( ( packm_def_cntl_t* ) cntl )->does_invert_diag = does_invert_diag;
}

BLIS_INLINE void bli_packm_def_cntl_set_rev_iter_if_upper( bool rev_iter_if_upper, cntl_t* cntl )
{
	( ( packm_def_cntl_t* ) cntl )->rev_iter_if_upper = rev_iter_if_upper;
}

BLIS_INLINE void bli_packm_def_cntl_set_rev_iter_if_lower( bool rev_iter_if_lower, cntl_t* cntl )
{
	( ( packm_def_cntl_t* ) cntl )->rev_iter_if_lower = rev_iter_if_lower;
}

BLIS_INLINE void bli_packm_def_cntl_set_pack_schema( pack_t pack_schema, cntl_t* cntl )
{
	( ( packm_def_cntl_t* ) cntl )->pack_schema = pack_schema;
}

BLIS_INLINE void bli_packm_def_cntl_set_pack_buf_type( packbuf_t pack_buf_type, cntl_t* cntl )
{
	( ( packm_def_cntl_t* ) cntl )->pack_buf_type = pack_buf_type;
}

BLIS_INLINE void bli_packm_def_cntl_set_ukr( const func2_t* ukr, cntl_t* cntl_ )
{
	packm_def_cntl_t* cntl = ( packm_def_cntl_t* )cntl_;
	num_t dt_orig = cntl->dt_orig;
	num_t dt_pack = cntl->dt_pack;
	cntl->ukr = ( packm_ker_ft )bli_func2_get_dt( dt_orig, dt_pack, ukr );
}

BLIS_INLINE err_t bli_packm_def_cntl_set_ukr_simple( const func_t* ukr, cntl_t* cntl_ )
{
	packm_def_cntl_t* cntl = ( packm_def_cntl_t* )cntl_;
	num_t dt_orig = cntl->dt_orig;
	num_t dt_pack = cntl->dt_pack;
	if ( dt_orig != dt_pack )
		return BLIS_INCONSISTENT_DATATYPES;
	cntl->ukr = ( packm_ker_ft )bli_func_get_dt( dt_orig, ukr );
	return BLIS_SUCCESS;
}

BLIS_INLINE void bli_packm_def_cntl_set_ukr_params( const void* params, cntl_t* cntl )
{
	( ( packm_def_cntl_t* ) cntl )->params = params;
}

// -----------------------------------------------------------------------------

BLIS_EXPORT_BLIS void bli_packm_cntl_init_node
     (
       void_fp       var_func,
       packm_var_oft var,
       const void*   params,
       packm_cntl_t* cntl
     );

BLIS_EXPORT_BLIS void bli_packm_def_cntl_init_node
     (
       void_fp           var_func,
       num_t             dt_orig,
       num_t             dt_pack,
       num_t             dt_bmult,
       packm_ker_ft      ukr,
       dim_t             bmult_m_def,
       dim_t             bmult_m_pack,
       dim_t             bmult_m_bcast,
       dim_t             bmult_m_scale,
	   dim_t             bmult_m_pack_scale,
       dim_t             bmult_n_def,
       bool              does_invert_diag,
       bool              rev_iter_if_upper,
       bool              rev_iter_if_lower,
       pack_t            pack_schema,
       packbuf_t         pack_buf_type,
       packm_def_cntl_t* cntl
     );

