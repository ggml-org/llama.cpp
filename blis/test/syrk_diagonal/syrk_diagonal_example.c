/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, Southern Methodist University

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

#include "syrk_diagonal_ref.h"

/*
 * Structure which includes all additional information beyond what is
 * already stored in the obj_t structure.
 *
 * This structure is **read-only** during the operation!
 */
typedef struct packm_diag_params_t
{
	packm_blk_var1_params_t super;
	void* d;
	inc_t incd;
} packm_diag_params_t;

/*
 * Declare the pack kernel type and set up and array of
 * packing kernels, one for each data type.
 */
#undef GENTFUNC
#define GENTFUNC(ctype,ch,op) \
void PASTEMAC(ch,op) \
    ( \
       struc_t        struca, \
       diag_t         diaga, \
       uplo_t         uploa, \
       conj_t         conja, \
       pack_t         schema, \
       bool           invdiag, \
       dim_t          panel_dim, \
       dim_t          panel_len, \
       dim_t          panel_dim_max, \
       dim_t          panel_len_max, \
       dim_t          panel_dim_off, \
       dim_t          panel_len_off, \
       void* restrict kappa, \
       void* restrict a, inc_t inca, inc_t lda, \
       void* restrict p,             inc_t ldp, \
                         inc_t is_p, \
       cntx_t*        cntx, \
       void*          params \
    ) \
{ \
	packm_diag_params_t* params_cast = params; \
	ctype* restrict      a_cast      = a; \
	ctype* restrict      p_cast      = p; \
	ctype* restrict      d_cast      = params_cast->d; \
	inc_t                incd        = params_cast->incd; \
	ctype                kappa_cast  = *( ctype* )kappa; \
\
	if ( schema != BLIS_PACKED_ROW_PANELS && \
		 schema != BLIS_PACKED_COL_PANELS ) \
		bli_abort(); \
\
	/* Apply the offset */ \
	d_cast += panel_len_off * incd; \
\
	if ( conja ) \
	{ \
		for ( dim_t j = 0; j < panel_len; j++ ) \
		{ \
			ctype kappa_d; \
			PASTEMAC(ch,scal2s)( kappa_cast, d_cast[ j*incd ], kappa_d ); \
\
			for (dim_t i = 0;i < panel_dim;i++) \
				PASTEMAC(ch,scal2js)( kappa_d, a_cast[ i*inca + j*lda ], p_cast[ i + j*ldp ] ); \
\
			for (dim_t i = panel_dim;i < panel_dim_max;i++) \
				PASTEMAC(ch,set0s)( p_cast[ i + j*ldp ] ); \
		} \
	} \
	else \
	{ \
		for ( dim_t j = 0; j < panel_len; j++ ) \
		{ \
			ctype kappa_d; \
			PASTEMAC(ch,scal2s)( kappa_cast, d_cast[ j*incd ], kappa_d ); \
\
			for (dim_t i = 0;i < panel_dim;i++) \
				PASTEMAC(ch,scal2s)( kappa_d, a_cast[ i*inca + j*lda ], p_cast[ i + j*ldp ] ); \
\
			for (dim_t i = panel_dim;i < panel_dim_max;i++) \
				PASTEMAC(ch,set0s)( p_cast[ i + j*ldp ] ); \
		} \
	} \
\
	for (dim_t j = panel_len;j < panel_len_max;j++) \
		for (dim_t i = 0;i < panel_dim_max;i++) \
			PASTEMAC(ch,set0s)( p_cast[ i + j*ldp ] ); \
}

INSERT_GENTFUNC_BASIC0(packm_diag_ukr);

static packm_ker_vft GENARRAY( packm_diag_ukrs, packm_diag_ukr );

/*
 * Modify the object A to include information about the diagonal D,
 * and imbue it with special function pointers which will take care
 * of the actual work of forming (D * A^T)
 */
void attach_diagonal_factor( packm_diag_params_t* params, obj_t* d, obj_t* a )
{
	memset( params, 0, sizeof(*params) );

	// Assumes D is a column vector
	params->d = bli_obj_buffer_at_off( d );
	params->incd = bli_obj_row_stride( d );

	for ( int i = BLIS_DT_LO; i <= BLIS_DT_HI; i++ )
		params->super.ukr_fn[i][i] = packm_diag_ukrs[i];

	// Attach the parameters to the A object.
	bli_obj_set_pack_params( params, a );
}

/*
 * Implements C := alpha * A * D * A^T + beta * C
 *
 * where D is a diagonal matrix with elements taken from the "d" vector.
 */
void syrk_diag( obj_t* alpha, obj_t* a, obj_t* d, obj_t* beta, obj_t* c )
{
	obj_t ad; // this is (D * A^T)
	packm_diag_params_t params;

	bli_obj_alias_to( a, &ad );
	bli_obj_toggle_trans( &ad ); // because gemmt is A*B instead of A*B^T
	attach_diagonal_factor( &params, d, &ad );

	// Does C := alpha * A * B + beta * C using B = (D + A^T)
	bli_gemmtnat( alpha, a, &ad, beta, c, NULL, NULL );
}

int main( void )
{
	obj_t a;
	obj_t d;
	obj_t c;
	obj_t c_copy;
	obj_t norm;

	dim_t m = 10;
	dim_t k = 10;

	for ( int dt_ = BLIS_DT_LO; dt_ <= BLIS_DT_HI; dt_++ )
	for ( int upper = 0; upper <= 1; upper++ )
	for ( int transa = 0; transa <= 1; transa++ )
	for ( int transc = 0; transc <= 1; transc++ )
	{
		num_t dt = dt_;
		uplo_t uplo = upper ? BLIS_UPPER : BLIS_LOWER;

		bli_obj_create( dt, m, k, transa ? k : 1, transa ? 1 : m, &a );
		bli_obj_create( dt, k, 1,              1,          1,     &d );
		bli_obj_create( dt, m, m, transc ? m : 1, transc ? 1 : m, &c );
		bli_obj_create( dt, m, m, transc ? m : 1, transc ? 1 : m, &c_copy );
		bli_obj_set_struc( BLIS_SYMMETRIC , &c );
		bli_obj_set_struc( BLIS_SYMMETRIC , &c_copy );
		bli_obj_set_uplo( uplo , &c );
		bli_obj_set_uplo( uplo , &c_copy );
		bli_obj_create_1x1( bli_dt_proj_to_real( dt ), &norm );

		bli_randm( &a );
		bli_randm( &d );
		bli_randm( &c );
		bli_copym( &c, &c_copy );

		syrk_diag( &BLIS_ONE, &a, &d, &BLIS_ONE, &c );
		syrk_diag_ref( &BLIS_ONE, &a, &d, &BLIS_ONE, &c_copy );

		bli_subm( &c_copy, &c );
		bli_normfm( &c, &norm );

		double normr, normi;
		bli_getsc( &norm, &normr, &normi );

		printf( "dt: %d, upper: %d, transa: %d, transc: %d, norm: %g\n",
		        dt, upper, transa, transc, normr );

		bli_obj_free( &a );
		bli_obj_free( &d );
		bli_obj_free( &c );
		bli_obj_free( &c_copy );
		bli_obj_free( &norm );
	}
}
