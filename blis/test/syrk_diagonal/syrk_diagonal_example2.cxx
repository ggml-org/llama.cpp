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
 * Forward-declare the pack kernel type and set up and array of
 * packing kernels, one for each data type.
 */
template <typename T>
void packm_diag_ukr
    (
       bool           conja,
       dim_t          panel_dim,
       dim_t          panel_len,
       dim_t          panel_dim_max,
       dim_t          panel_len_max,
       void* restrict kappa,
       void* restrict d, inc_t incd,
       void* restrict a, inc_t inca, inc_t lda,
       void* restrict p,             inc_t ldp
    );

#undef GENTFUNC
#define GENTFUNC(ctype,ch,op) \
static auto PASTEMAC(ch,op) = &packm_diag_ukr<ctype>;

INSERT_GENTFUNC_BASIC0(packm_diag_ukr);

using packm_diag_ukr_vft = decltype(&packm_diag_ukr<void>);
static packm_diag_ukr_vft GENARRAY( packm_diag_ukrs, packm_diag_ukr );

/*
 * Structure which includes all additional information beyond what is
 * already stored in the obj_t structure.
 *
 * This structure is **read-only** during the operation!
 */
struct packm_diag_params_t
{
    void* d;
    inc_t incd;

    packm_diag_params_t() {}

    packm_diag_params_t( void* d, inc_t incd )
    : d(d), incd(incd) {}
};

/*
 * Selecting a different kernel based on the current architecture is
 * currently not possible, but is something we plan to support.
 */
template <typename T>
void packm_diag_ukr
    (
       bool           conja,
       dim_t          panel_dim,
       dim_t          panel_len,
       dim_t          panel_dim_max,
       dim_t          panel_len_max,
       void* restrict kappa,
       void* restrict d, inc_t incd,
       void* restrict a, inc_t inca, inc_t lda,
       void* restrict p,             inc_t ldp
    )
{
    T* restrict a_cast     = ( T* )a;
    T* restrict p_cast     = ( T* )p;
    T* restrict d_cast     = ( T* )d;
    auto        kappa_cast = *( T* )kappa;

    if ( conja )
    {
        for ( dim_t j = 0; j < panel_len; j++ )
        {
            auto kappa_d = kappa_cast * d_cast[ j*incd ];

            for (dim_t i = 0;i < panel_dim;i++)
                p_cast[ i + j*ldp ] = kappa_d * conj( a_cast[ i*inca + j*lda ] );

            for (dim_t i = panel_dim;i < panel_dim_max;i++)
                p_cast[ i + j*ldp ] = convert<T>(0.0);
        }
    }
    else
    {
        for ( dim_t j = 0; j < panel_len; j++ )
        {
            auto kappa_d = kappa_cast * d_cast[ j*incd ];

            for (dim_t i = 0;i < panel_dim;i++)
                p_cast[ i + j*ldp ] = kappa_d * a_cast[ i*inca + j*lda ];

            for (dim_t i = panel_dim;i < panel_dim_max;i++)
                p_cast[ i + j*ldp ] = convert<T>(0.0);
        }
    }

    for (dim_t j = panel_len;j < panel_len_max;j++)
        for (dim_t i = 0;i < panel_dim_max;i++)
            p_cast[ i + j*ldp ] = convert<T>(0.0);
}

void packm_diag
     (
       obj_t*   a,
       obj_t*   p,
       cntx_t*  cntx,
       rntm_t*  rntm,
       cntl_t*  cntl,
       thrinfo_t* thread
     )
{
	// We begin by copying the fields of A.
	bli_obj_alias_to( a, p );

    // Get information about data types.
	num_t dt        = bli_obj_dt( a );
	num_t dt_tar    = bli_obj_target_dt( a );
	num_t dt_scalar = bli_obj_scalar_dt( a );
	dim_t dt_size   = bli_dt_size( dt );

	if ( dt_scalar != dt || dt_tar != dt )
       bli_abort();

	// Extract various fields from the control tree.
	bszid_t bmult_id_m   = bli_cntl_packm_params_bmid_m( cntl );
	bszid_t bmult_id_n   = bli_cntl_packm_params_bmid_n( cntl );
	pack_t  schema       = bli_cntl_packm_params_pack_schema( cntl );
	dim_t   bmult_m_def  = bli_cntx_get_blksz_def_dt( dt_tar, bmult_id_m, cntx );
	dim_t   bmult_m_pack = bli_cntx_get_blksz_max_dt( dt_tar, bmult_id_m, cntx );
	dim_t   bmult_n_def  = bli_cntx_get_blksz_def_dt( dt_tar, bmult_id_n, cntx );

    if ( schema != BLIS_PACKED_ROW_PANELS &&
         schema != BLIS_PACKED_COL_PANELS )
       bli_abort();

	// Store the pack schema to the object.
	bli_obj_set_pack_schema( schema, p );

	// Clear the conjugation field from the object since matrix packing
	// in BLIS is deemed to take care of all conjugation necessary.
	bli_obj_set_conj( BLIS_NO_CONJUGATE, p );

	// If we are packing micropanels, mark P as dense.
	bli_obj_set_uplo( BLIS_DENSE, p );

	// Reset the view offsets to (0,0).
	bli_obj_set_offs( 0, 0, p );

	// Compute the dimensions padded by the dimension multiples. These
	// dimensions will be the dimensions of the packed matrices, including
	// zero-padding, and will be used by the macro- and micro-kernels.
	// We compute them by starting with the effective dimensions of A (now
	// in P) and aligning them to the dimension multiples (typically equal
	// to register blocksizes). This does waste a little bit of space for
	// level-2 operations, but that's okay with us.
	dim_t m_p     = bli_obj_length( p );
	dim_t n_p     = bli_obj_width( p );
	dim_t m_p_pad = bli_align_dim_to_mult( m_p, bmult_m_def );
	dim_t n_p_pad = bli_align_dim_to_mult( n_p, bmult_n_def );

	// Save the padded dimensions into the packed object. It is important
	// to save these dimensions since they represent the actual dimensions
	// of the zero-padded matrix.
	bli_obj_set_padded_dims( m_p_pad, n_p_pad, p );

	// The "panel stride" of a micropanel packed object is interpreted as
	// the distance between the (0,0) element of panel k and the (0,0)
	// element of panel k+1. We use the padded width computed above to
	// allow for zero-padding (if necessary/desired) along the far end
	// of each micropanel (ie: the right edge of the matrix). Zero-padding
	// can also occur along the long edge of the last micropanel if the m
	// dimension of the matrix is not a whole multiple of MR.
	inc_t ps_p = bmult_m_pack * n_p_pad;

	/* Compute the total number of iterations we'll need. */
	dim_t n_iter = m_p_pad / bmult_m_def;

	// Store the strides and panel dimension in P.
	bli_obj_set_strides( 1, bmult_m_pack, p );
	bli_obj_set_imag_stride( 1, p );
	bli_obj_set_panel_dim( bmult_m_def, p );
	bli_obj_set_panel_stride( ps_p, p );
	bli_obj_set_panel_length( bmult_m_def, p );
	bli_obj_set_panel_width( n_p, p );

	// Compute the size of the packed buffer.
	siz_t size_p = ps_p * n_iter * dt_size;
	if ( size_p == 0 ) return;

	// Update the buffer address in p to point to the buffer associated
	// with the mem_t entry acquired from the memory broker (now cached in
	// the control tree node).
	char*   p_cast         = (char*)bli_packm_alloc( size_p, rntm, cntl, thread );
	bli_obj_set_buffer( p_cast, p );

	char*   a_cast         = (char*)bli_obj_buffer_at_off( a );
	inc_t   inca           = bli_obj_row_stride( a );
	inc_t   lda            = bli_obj_col_stride( a );
	dim_t   panel_len_off  = bli_obj_col_off( a );
	conj_t  conja          = bli_obj_conj_status( a );

    auto    params         = (packm_diag_params_t*)bli_obj_pack_params( a );
    char*   d_cast         = (char*)params->d;
    inc_t   incd           = params->incd;

	obj_t   kappa_local;
	char*   kappa_cast     = (char*)bli_packm_scalar( &kappa_local, p );

	auto    packm_ker_cast = packm_diag_ukrs[ dt ];

	/* Query the number of threads and thread ids from the current thread's
	   packm thrinfo_t node. */
	const dim_t nt  = bli_thrinfo_n_way( thread );
	const dim_t tid = bli_thrinfo_work_id( thread );

	/* Determine the thread range and increment using the current thread's
	   packm thrinfo_t node. NOTE: The definition of bli_thread_range_jrir()
	   will depend on whether slab or round-robin partitioning was requested
	   at configure-time. */
	dim_t it_start, it_end, it_inc;
	bli_thread_range_jrir( thread, n_iter, 1, FALSE, &it_start, &it_end, &it_inc );

	/* Iterate over every logical micropanel in the source matrix. */
	for ( dim_t it  = 0; it < n_iter; it += 1 )
	{
		dim_t panel_dim_i = bli_min( bmult_m_def, m_p - it*bmult_m_def );

        char* d_begin     = d_cast +    panel_len_off*incd*dt_size;
		char* a_begin     = a_cast + it*  bmult_m_def*inca*dt_size;
	    char* p_begin     = p_cast + it*              ps_p*dt_size;

		if ( bli_packm_my_iter( it, it_start, it_end, tid, nt ) )
		{
    		packm_ker_cast( conja,
                            panel_dim_i,
    		                n_p,
    		                bmult_m_def,
    		                n_p_pad,
    		                kappa_cast,
                            d_begin, incd,
    		                a_begin, inca, lda,
    		                p_begin, bmult_m_pack );
        }
	}
}

/*
 * Modify the object A to include information about the diagonal D,
 * and imbue it with special function pointers which will take care
 * of the actual work of forming (D * A^T)
 */
void attach_diagonal_factor( packm_diag_params_t* params, obj_t* d, obj_t* a )
{
    // Assumes D is a column vector
    new (params) packm_diag_params_t
    (
      bli_obj_buffer_at_off( d ),
      bli_obj_row_stride( d )
    );

    // Set the custom pack function.
    bli_obj_set_pack_fn( packm_diag, a );

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
    bli_gemmt( alpha, a, &ad, beta, c );
}

int main()
{
    obj_t a;
    obj_t d;
    obj_t c;
    obj_t c_copy;
    obj_t norm;

    auto m = 10;
    auto k = 10;

    for ( int dt_ = BLIS_DT_LO; dt_ <= BLIS_DT_HI; dt_++ )
    for ( int upper = 0; upper <= 1; upper++ )
    for ( int transa = 0; transa <= 1; transa++ )
    for ( int transc = 0; transc <= 1; transc++ )
    {
        auto dt = ( num_t )dt_;
        auto uplo = upper ? BLIS_UPPER : BLIS_LOWER;

        bli_obj_create( dt, m, k, transa ? k : 1, transa ? 1 : m, &a );
        bli_obj_create( dt, k, 1,              1,              1, &d );
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

        printf("dt: %d, upper: %d, transa: %d, transc: %d, norm: %g\n",
               dt, upper, transa, transc, normr);

        bli_obj_free( &a );
        bli_obj_free( &d );
        bli_obj_free( &c );
        bli_obj_free( &c_copy );
        bli_obj_free( &norm );
    }
}
