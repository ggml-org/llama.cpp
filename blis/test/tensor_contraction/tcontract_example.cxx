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

#include "tcontract_ref.hpp"

#include <algorithm>
#include <numeric>

static constexpr dim_t BS_K = 8;

struct packm_tensor_params_t
{
    gint_t ndim_m, ndim_n;
    const dim_t *len_m, *len_n;
    const inc_t *stride_m, *stride_n;

    packm_tensor_params_t() {}

    packm_tensor_params_t( gint_t ndim_m, const dim_t* len_m, const inc_t* stride_m,
                           gint_t ndim_n, const dim_t* len_n, const inc_t* stride_n )
    : ndim_m(ndim_m), ndim_n(ndim_n),
      len_m(len_m), len_n(len_n),
      stride_m(stride_m), stride_n(stride_n) {}
};

using gemm_tensor_params_t = packm_tensor_params_t;

template <typename T>
void packm_ckx_nb
    (
       bool  conja,
       dim_t panel_dim,
       dim_t panel_len,
       dim_t panel_dim_max,
       dim_t panel_len_max,
       void* kappa,
       void* a, inc_t inca, inc_t* bsa, inc_t* scata,
       void* p, inc_t ldp
    )
{
    T* restrict a_cast     = ( T* )a;
    T* restrict p_cast     = ( T* )p;
    auto        kappa_cast = *( T* )kappa;

    if ( conja )
    {
        for ( auto j0 = 0; j0 < panel_len; j0 += BS_K, bsa += BS_K, scata += BS_K )
        {
            auto lda = *bsa;
            auto panel_len_j = std::min<dim_t>( panel_len-j0, BS_K );

            if ( lda )
            {
                T* restrict aj = a_cast + *scata;

                for ( auto j = 0; j < panel_len_j; j++ )
                {
                    for ( auto i = 0; i < panel_dim; i++ )
                        p_cast[ i ] = kappa_cast * conj( aj[ i*inca + j*lda ] );

                    for ( auto i = panel_dim; i < panel_dim_max; i++ )
                        p_cast[ i ] = convert<T>(0.0);

                    p_cast += ldp;
                }
            }
            else
            {
                for ( auto j = 0; j < panel_len_j; j++)
                {
                    for ( auto i = 0; i < panel_dim; i++)
                        p_cast[ i ] = kappa_cast * conj( a_cast[ i*inca + scata[j] ] );

                    for ( auto i = panel_dim; i < panel_dim_max; i++)
                        p_cast[ i ] = convert<T>(0.0);

                    p_cast += ldp;
                }
            }
        }
    }
    else
    {
        for ( auto j0 = 0; j0 < panel_len; j0 += BS_K, bsa += BS_K, scata += BS_K )
        {
            auto lda = *bsa;
            auto panel_len_j = std::min<dim_t>( panel_len-j0, BS_K );

            if ( lda )
            {
                T* restrict aj = a_cast + *scata;

                for ( auto j = 0; j < panel_len_j; j++ )
                {
                    for ( auto i = 0; i < panel_dim; i++ )
                        p_cast[ i ] = kappa_cast * aj[ i*inca + j*lda ];

                    for ( auto i = panel_dim; i < panel_dim_max; i++ )
                        p_cast[ i ] = convert<T>(0.0);

                    p_cast += ldp;
                }
            }
            else
            {
                for ( auto j = 0; j < panel_len_j; j++ )
                {
                    for ( auto i = 0; i < panel_dim; i++ )
                        p_cast[ i ] = kappa_cast * a_cast[ i*inca + scata[j] ];

                    for ( auto i = panel_dim; i < panel_dim_max; i++ )
                        p_cast[ i ] = convert<T>(0.0);

                    p_cast += ldp;
                }
            }
        }
    }

    for ( auto j = panel_len; j < panel_len_max; j++)
    {
        for ( auto i = 0; i < panel_dim_max; i++)
            p_cast[ i ] = convert<T>(0.0);

        p_cast += ldp;
    }
}

template <typename T>
void packm_ckx_ss
    (
       bool  conja,
       dim_t panel_dim,
       dim_t panel_len,
       dim_t panel_dim_max,
       dim_t panel_len_max,
       void* kappa,
       void* a, inc_t* inca, inc_t* scata,
       void* p, inc_t ldp
    )
{
    T* restrict a_cast     = ( T* )a;
    T* restrict p_cast     = ( T* )p;
    auto        kappa_cast = *( T* )kappa;

    if ( conja )
    {
        for (dim_t j = 0;j < panel_len;j++)
        {
            for (dim_t i = 0;i < panel_dim;i++)
                p_cast[ i ] = kappa_cast * conj( a_cast[ inca[i] + scata[j] ] );

            for (dim_t i = panel_dim;i < panel_dim_max;i++)
                p_cast[ i ] = convert<T>(0.0);

            p_cast += ldp;
        }
    }
    else
    {
        for (dim_t j = 0;j < panel_len;j++)
        {
            for (dim_t i = 0;i < panel_dim;i++)
                p_cast[ i ] = kappa_cast * a_cast[ inca[i] + scata[j] ];

            for (dim_t i = panel_dim;i < panel_dim_max;i++)
                p_cast[ i ] = convert<T>(0.0);

            p_cast += ldp;
        }
    }

    for (dim_t j = panel_len;j < panel_len_max;j++)
    {
        for (dim_t i = 0;i < panel_dim_max;i++)
            p_cast[ i ] = convert<T>(0.0);

        p_cast += ldp;
    }
}

#undef GENTFUNC
#define GENTFUNC(ctype,ch,op) \
static auto PASTEMAC(ch,op) = &packm_ckx_nb<ctype>;

INSERT_GENTFUNC_BASIC0(packm_ckx_nb);

#undef GENTFUNC
#define GENTFUNC(ctype,ch,op) \
static auto PASTEMAC(ch,op) = &packm_ckx_ss<ctype>;

INSERT_GENTFUNC_BASIC0(packm_ckx_ss);

static decltype(&packm_ckx_nb<void>) GENARRAY( packm_ckx_nb_ukrs, packm_ckx_nb );
static decltype(&packm_ckx_ss<void>) GENARRAY( packm_ckx_ss_ukrs, packm_ckx_ss );

static void fill_scatter
            (
              gint_t                ndim,
              const dim_t* restrict len,
              const inc_t* restrict stride,
              dim_t                 BS,
              inc_t                 off,
              dim_t                 size,
              inc_t* restrict       scat,
              inc_t* restrict       bs
            )
{
    if ( size == 0 ) return;

    if ( ndim == 0 )
    {
        *scat = 0;
        *bs = 0;
        return;
    }

    if ( ndim == 1 )
    {
        auto l = *len;
        auto s = *stride;
        for ( auto i = 0; i < l; i++ )
        {
            scat[i] = i*s;
            bs[i] = s;
        }
    }

    dim_t tot_len = 1;
    for ( auto i = 0; i < ndim; i++ )
        tot_len *= len[i];

    assert(off >= 0);
    assert(size >= 0);
    assert(off+size <= tot_len);

    auto len0 = len[0];
    auto stride0 = stride[0];
    auto off0 = off % len0;
    auto off1 = off / len0;
    auto size1 = ( size + off0 + len0 - 1) / len0;

    inc_t pos1 = 0;
    inc_t idx = 0;
    for_each( ndim-1, len+1, off1, size1, pos1, stride+1,
    [&]
    {
        auto pos = pos1 + off0 * stride0;
        auto len_i = std::min( len0-off0, size-idx );
        for ( auto i = 0; i < len_i; i++ )
        {
            scat[idx++] = pos;
            pos += stride0;
        }
        off0 = 0;
    });
    assert(idx == size);

    for ( idx = 0; idx < size; idx += BS )
    {
        auto len_i = std::min( BS, size-idx );
        auto s = stride0;

        for ( auto i = idx; i < idx+len_i-1; i++)
        {
            if (scat[i+1]-scat[i] != s)
            {
                s = 0;
                break;
            }
        }

        bs[idx] = s;
    }
}

void packm_tensor
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
	auto dt        = bli_obj_dt( a );
	auto dt_tar    = bli_obj_target_dt( a );
	auto dt_scalar = bli_obj_scalar_dt( a );
	auto dt_size   = bli_dt_size( dt );

	if ( dt_scalar != dt || dt_tar != dt )
       bli_abort();

	// Extract various fields from the control tree.
	auto bmult_id_m   = bli_cntl_packm_params_bmid_m( cntl );
	auto bmult_id_n   = bli_cntl_packm_params_bmid_n( cntl );
	auto schema       = bli_cntl_packm_params_pack_schema( cntl );
	auto bmult_m_def  = bli_cntx_get_blksz_def_dt( dt_tar, bmult_id_m, cntx );
	auto bmult_m_pack = bli_cntx_get_blksz_max_dt( dt_tar, bmult_id_m, cntx );
	auto bmult_n_def  = bli_cntx_get_blksz_def_dt( dt_tar, bmult_id_n, cntx );

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
	auto m_p     = bli_obj_length( p );
	auto n_p     = bli_obj_width( p );
	auto m_p_pad = bli_align_dim_to_mult( m_p, bmult_m_def );
	auto n_p_pad = bli_align_dim_to_mult( n_p, bmult_n_def );

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
	auto ps_p = bmult_m_pack * n_p_pad;

	/* Compute the total number of iterations we'll need. */
	auto n_iter = m_p_pad / bmult_m_def;

	// Store the strides and panel dimension in P.
	bli_obj_set_strides( 1, bmult_m_pack, p );
	bli_obj_set_imag_stride( 1, p );
	bli_obj_set_panel_dim( bmult_m_def, p );
	bli_obj_set_panel_stride( ps_p, p );
	bli_obj_set_panel_length( bmult_m_def, p );
	bli_obj_set_panel_width( n_p, p );

	// Compute the size of the packed buffer.
	auto size_p = ps_p * n_iter * dt_size;
	if ( size_p == 0 ) return;

    // Compute the size of the scatter and block-scatter vectors to the total.
    // It is never necessary to add padding for alignment because:
    // 1) ps_p is always even
    // 2) dt_size is a power of two >= 4
    // 3) the alignment of the scatter vectors is at most 8
    auto scat_size = 2 * (m_p + n_p) * sizeof(inc_t);

	// Update the buffer address in p to point to the buffer associated
	// with the mem_t entry acquired from the memory broker (now cached in
	// the control tree node).
	auto p_cast = (char*)bli_packm_alloc( size_p + scat_size, rntm, cntl, thread );
	bli_obj_set_buffer( p_cast, p );

    // Get the addresses of the scatter and block-scatter vectors. These are
    // placed directly after the packed matrix buffer.
    auto  rscat          = (inc_t*)(p_cast + size_p);
    auto  rbs            = rscat + m_p;
    auto  cscat          = rbs + m_p;
    auto  cbs            = cscat + n_p;

	auto  a_cast         = (char*)bli_obj_buffer_at_off( a );
	auto  panel_dim_off  = bli_obj_row_off( a );
	auto  panel_len_off  = bli_obj_col_off( a );
	auto  conja          = bli_obj_conj_status( a );

    auto  params         = (packm_tensor_params_t*)bli_obj_pack_params( a );
    auto  ndim_m         = params->ndim_m;
    auto  ndim_n         = params->ndim_n;
    auto  len_m          = params->len_m;
    auto  len_n          = params->len_n;
    auto  stride_m       = params->stride_m;
    auto  stride_n       = params->stride_n;

	obj_t kappa_local;
	auto  kappa_cast     = (char*)bli_packm_scalar( &kappa_local, p );

	auto  packm_nb_ker   = packm_ckx_nb_ukrs[ dt ];
	auto  packm_ss_ker   = packm_ckx_ss_ukrs[ dt ];

    a_cast -= ( panel_dim_off * stride_m[0] +
                panel_len_off * stride_n[0] ) * dt_size;

    /* Fill in the scatter and block-scatter vectors. This is done single-threaded for now. */
    if ( bli_thread_am_ochief( thread ) )
    {
        fill_scatter
        (
          ndim_m,
          len_m,
          stride_m,
          bmult_m_def,
          panel_dim_off,
          m_p,
          rscat,
          rbs
        );

        fill_scatter
        (
          ndim_n,
          len_n,
          stride_n,
          BS_K,
          panel_len_off,
          n_p,
          cscat,
          cbs
        );
    }

    /* Wait for the scatter vectors to be done. */
    bli_thrinfo_barrier( thread );

	/* Query the number of threads and thread ids from the current thread's
	   packm thrinfo_t node. */
	auto nt  = bli_thrinfo_n_way( thread );
	auto tid = bli_thrinfo_work_id( thread );

	/* Determine the thread range and increment using the current thread's
	   packm thrinfo_t node. NOTE: The definition of bli_thread_range_jrir()
	   will depend on whether slab or round-robin partitioning was requested
	   at configure-time. */
	dim_t it_start, it_end, it_inc;
	bli_thread_range_jrir( thread, n_iter, 1, FALSE, &it_start, &it_end, &it_inc );

	/* Iterate over every logical micropanel in the source matrix. */
	for ( auto it  = 0; it < n_iter; it += 1 )
	if ( bli_packm_my_iter( it, it_start, it_end, tid, nt ) )
	{
		auto panel_dim_i = bli_min( bmult_m_def, m_p - it*bmult_m_def );

	    auto p_begin     = p_cast + it*ps_p*dt_size;
        auto inca        = rbs[ it*bmult_m_def ];

        if ( inca )
        {
	        auto a_begin = a_cast + rscat[ it*bmult_m_def ]*dt_size;

    		packm_nb_ker( conja,
                          panel_dim_i,
		                  n_p,
		                  bmult_m_def,
		                  n_p_pad,
		                  kappa_cast,
		                  a_begin, inca, cbs, cscat,
		                  p_begin, bmult_m_pack );
        }
        else
        {
	        auto a_begin   = a_cast;
            auto rscat_use = rscat + it*bmult_m_def;

    		packm_ss_ker( conja,
                          panel_dim_i,
		                  n_p,
		                  bmult_m_def,
		                  n_p_pad,
		                  kappa_cast,
		                  a_begin, rscat_use, cscat,
		                  p_begin, bmult_m_pack );
        }
    }
}

#undef GENTFUNC
#define GENTFUNC(ctype,ch,op) \
void PASTEMAC(ch,op) \
    ( \
      dim_t m, \
      dim_t n, \
      void* x, inc_t rs_x, inc_t cs_x, \
      void* b, \
      void* y, inc_t* rs_y, inc_t* cs_y \
    ) \
{ \
    ctype* restrict x_cast =  (ctype*)x; \
    ctype           b_cast = *(ctype*)b; \
    ctype* restrict y_cast =  (ctype*)y; \
\
    if ( PASTEMAC(ch,eq0)( b_cast ) ) \
    { \
        for ( auto i = 0; i < m; i++ ) \
        for ( auto j = 0; j < n; j++ ) \
            PASTEMAC(ch,copys)( x_cast[ i*rs_x + j*cs_x ], y_cast[ rs_y[i] + cs_y[j] ] ); \
    } \
    else \
    { \
        for ( auto i = 0; i < m; i++ ) \
        for ( auto j = 0; j < n; j++ ) \
            PASTEMAC(ch,xpbys)( x_cast[ i*rs_x + j*cs_x ], b_cast, y_cast[ rs_y[i] + cs_y[j] ] ); \
    } \
}

INSERT_GENTFUNC_BASIC0(scatter_mxn);

static decltype(&bli_sscatter_mxn) GENARRAY(scatter_mxn, scatter_mxn);

void gemm_tensor
     (
       obj_t*  a,
       obj_t*  b,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm,
       cntl_t* cntl,
       thrinfo_t* thread
     )
{
	auto dt       = bli_obj_dt( c );
    auto dt_size  = bli_dt_size( dt );

	auto m        = bli_obj_length( c );
	auto n        = bli_obj_width( c );
	auto k        = bli_obj_width( a );

	auto a_cast   = (char*)bli_obj_buffer_at_off( a );
	auto pd_a     = bli_obj_panel_dim( a );
	auto ps_a     = bli_obj_panel_stride( a );

	auto b_cast   = (char*)bli_obj_buffer_at_off( b );
	auto pd_b     = bli_obj_panel_dim( b );
	auto ps_b     = bli_obj_panel_stride( b );

	auto c_cast   = (char*)bli_obj_buffer_at_off( c );
	auto rs_c0    = bli_obj_row_stride( c );
	auto cs_c0    = bli_obj_col_stride( c );
	auto off_m    = bli_obj_row_off( c );
	auto off_n    = bli_obj_col_off( c );

    auto params   = (gemm_tensor_params_t*)bli_obj_ker_params( c );
    auto ndim_m   = params->ndim_m;
    auto ndim_n   = params->ndim_n;
    auto len_m    = params->len_m;
    auto len_n    = params->len_n;
    auto stride_m = params->stride_m;
    auto stride_n = params->stride_n;

    if ( rs_c0 != stride_m[0] || cs_c0 != stride_n[0] )
    {
        std::swap( ndim_m, ndim_n );
        std::swap( len_m, len_n );
        std::swap( stride_m, stride_n );
    }

	/* If any dimension is zero, return immediately. */
	if ( bli_zero_dim3( m, n, k ) ) return;

    c_cast -= ( off_m * stride_m[0] +
                off_n * stride_n[0] ) * dt_size;

	// Detach and multiply the scalars attached to A and B.
	// NOTE: We know that the internal scalars of A and B are already of the
	// target datatypes because the necessary typecasting would have already
	// taken place during bli_packm_init().
	obj_t scalar_a;
	obj_t scalar_b;
	bli_obj_scalar_detach( a, &scalar_a );
	bli_obj_scalar_detach( b, &scalar_b );
	bli_mulsc( &scalar_a, &scalar_b );

	// Grab the addresses of the internal scalar buffers for the scalar
	// merged above and the scalar attached to C.
	// NOTE: We know that scalar_b is of type dt due to the above code
	// that casts the scalars of A and B to dt via scalar_a and scalar_b,
	// and we know that the internal scalar in C is already of the type dt
	// due to the casting in the implementation of bli_obj_scalar_attach().
	auto alpha_cast = (char*)bli_obj_internal_scalar_buffer( &scalar_b );
	auto beta_cast  = (char*)bli_obj_internal_scalar_buffer( c );

	/* Alias some constants to simpler names. */
	auto MR = pd_a;
	auto NR = pd_b;

	/* Query the context for the micro-kernel address and cast it to its
	   function pointer type. */
	auto gemm_ukr = (gemm_ukr_vft)bli_cntx_get_l3_nat_ukr_dt( dt, BLIS_GEMM_UKR, cntx );

	/* Temporary C buffer for edge cases. Note that the strides of this
	   temporary buffer are set so that they match the storage of the
	   original C matrix. For example, if C is column-stored, ct will be
	   column-stored as well. */
	char ct[ BLIS_STACK_BUF_MAX_SIZE ] __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE)));
	auto col_pref = bli_cntx_l3_vir_ukr_prefers_cols_dt( dt, BLIS_GEMM_UKR, cntx );
	auto rs_ct    = ( col_pref ? 1 : NR );
	auto cs_ct    = ( col_pref ? MR : 1 );
	auto zero     = (char*)bli_obj_buffer_for_const( dt, &BLIS_ZERO );

	/*
	   Assumptions/assertions:
	     rs_a == 1
	     cs_a == PACKMR
	     pd_a == MR
	     ps_a == stride to next micro-panel of A
	     rs_b == PACKNR
	     cs_b == 1
	     pd_b == NR
	     ps_b == stride to next micro-panel of B
	     rs_c == (no assumptions)
	     cs_c == (no assumptions)
	*/

    auto scat_size = 2 * (m + n) * sizeof(inc_t);
    auto rscat_c   = (inc_t*)bli_packm_alloc_ex( scat_size, BLIS_BUFFER_FOR_GEN_USE, rntm, cntl, thread );
    auto rbs_c     = rscat_c + m;
    auto cscat_c   = rbs_c + m;
    auto cbs_c     = cscat_c + n;

    /* Fill in the scatter and block-scatter vectors. This is done single-threaded for now. */
    if ( bli_thread_am_ochief( thread ) )
    {
        fill_scatter
        (
          ndim_m,
          len_m,
          stride_m,
          MR,
          off_m,
          m,
          rscat_c,
          rbs_c
        );

        fill_scatter
        (
          ndim_n,
          len_n,
          stride_n,
          NR,
          off_n,
          n,
          cscat_c,
          cbs_c
        );
    }

    /* Wait for the scatter vectors to be done. */
    bli_thrinfo_barrier( thread );

	/* Compute number of primary and leftover components of the m and n
	   dimensions. */
	auto n_iter = n / NR;
	auto n_left = n % NR;

	auto m_iter = m / MR;
	auto m_left = m % MR;

	if ( n_left ) ++n_iter;
	if ( m_left ) ++m_iter;

	/* Determine some increments used to step through A, B, and C. */
	auto rstep_a = ps_a * dt_size;
	auto cstep_b = ps_b * dt_size;

    /* Save the virtual microkernel address and the params. */
	auxinfo_t aux;
    bli_auxinfo_set_ukr( (void*)gemm_ukr, &aux );
    bli_auxinfo_set_params( params, &aux );

	/* The 'thread' argument points to the thrinfo_t node for the 2nd (jr)
	   loop around the microkernel. Here we query the thrinfo_t node for the
	   1st (ir) loop around the microkernel. */
	auto caucus = bli_thrinfo_sub_node( thread );

	/* Query the number of threads and thread ids for each loop. */
	auto jr_nt  = bli_thrinfo_n_way( thread );
	auto jr_tid = bli_thrinfo_work_id( thread );
	auto ir_nt  = bli_thrinfo_n_way( caucus );
	auto ir_tid = bli_thrinfo_work_id( caucus );

	/* Determine the thread range and increment for the 2nd and 1st loops.
	   NOTE: The definition of bli_thread_range_jrir() will depend on whether
	   slab or round-robin partitioning was requested at configure-time. */
	dim_t jr_start, jr_end;
	dim_t ir_start, ir_end;
	dim_t jr_inc,   ir_inc;
	bli_thread_range_jrir( thread, n_iter, 1, FALSE, &jr_start, &jr_end, &jr_inc );
	bli_thread_range_jrir( caucus, m_iter, 1, FALSE, &ir_start, &ir_end, &ir_inc );

	/* Loop over the n dimension (NR columns at a time). */
	for ( auto j = jr_start; j < jr_end; j += jr_inc )
	{
		auto b1    = b_cast + j * cstep_b;

		auto n_cur = ( bli_is_not_edge_f( j, n_iter, n_left ) ? NR : n_left );

		/* Initialize our next panel of B to be the current panel of B. */
		auto b2 = b1;

		/* Loop over the m dimension (MR rows at a time). */
		for ( auto i = ir_start; i < ir_end; i += ir_inc )
		{
			auto a1       = a_cast  + i * rstep_a;
            auto rscat_c1 = rscat_c + i * MR;
            auto rbs_c1   = rbs_c   + i * MR;
            auto cscat_c1 = cscat_c + j * NR;
            auto cbs_c1   = cbs_c   + j * NR;

			auto m_cur    = ( bli_is_not_edge_f( i, m_iter, m_left ) ? MR : m_left );

			/* Compute the addresses of the next panels of A and B. */
			auto a2 = bli_gemm_get_next_a_upanel( a1, rstep_a, ir_inc );
			if ( bli_is_last_iter( i, ir_end, ir_tid, ir_nt ) )
			{
				a2 = a_cast;
				b2 = bli_gemm_get_next_b_upanel( b1, cstep_b, jr_inc );
				if ( bli_is_last_iter( j, jr_end, jr_tid, jr_nt ) )
					b2 = b_cast;
			}

			/* Save addresses of next panels of A and B to the auxinfo_t
			   object. */
			bli_auxinfo_set_next_a( a2, &aux );
			bli_auxinfo_set_next_b( b2, &aux );

            auto rs_c = *rbs_c1;
            auto cs_c = *cbs_c1;

            if ( rs_c && cs_c )
            {
			    auto c11 = c_cast + ( *rscat_c1 + *cscat_c1 ) * dt_size;

    			/* Invoke the gemm micro-kernel. */
    			gemm_ukr
    			(
                  m_cur,
                  n_cur,
    			  k,
    			  alpha_cast,
    			  a1,
    			  b1,
    			  beta_cast,
    			  c11, rs_c, cs_c,
    			  &aux,
    			  cntx
    			);
            }
            else
            {
    			/* Invoke the gemm micro-kernel. */
    			gemm_ukr
    			(
                  MR,
                  NR,
    			  k,
    			  alpha_cast,
    			  a1,
    			  b1,
    			  zero,
    			  &ct, rs_ct, cs_ct,
    			  &aux,
    			  cntx
    			);

    			/* Scatter to C. */
                scatter_mxn[ dt ]
                (
                    m_cur, n_cur,
                    &ct, rs_ct, cs_ct,
                    beta_cast,
                    c_cast, rscat_c1, cscat_c1
                );
            }
		}
	}
}

static bool has_unit_stride( const std::vector<inc_t>& stride )
{
    for ( auto s : stride )
        if ( s == 1 )
            return true;
    return false;
}

void tcontract( num_t dt, const std::vector<dim_t>& m, const std::vector<dim_t>& n, const std::vector<dim_t>& k,
                const void* alpha, const void* a, std::vector<inc_t> rs_a, std::vector<inc_t> cs_a,
                                   const void* b, std::vector<inc_t> rs_b, std::vector<inc_t> cs_b,
                const void*  beta,       void* c, std::vector<inc_t> rs_c, std::vector<inc_t> cs_c )
{
    if ( rs_a.size() != m.size() ||
         rs_b.size() != k.size() ||
         rs_c.size() != m.size() )
        bli_check_error_code( BLIS_INVALID_ROW_STRIDE );

    if ( cs_a.size() != k.size() ||
         cs_b.size() != n.size() ||
         cs_c.size() != n.size() )
        bli_check_error_code( BLIS_INVALID_COL_STRIDE );

    dim_t m_mat = 1;
    dim_t n_mat = 1;
    dim_t k_mat = 1;
    for ( auto& i : m ) m_mat *= i;
    for ( auto& i : n ) n_mat *= i;
    for ( auto& i : k ) k_mat *= i;

    auto& stride_m = has_unit_stride( rs_c ) ? rs_c : rs_a;
    for ( int i = 1;i < m.size(); i++ )
    for ( int j = 0;j < m.size()-i; j++ )
    if ( stride_m[j] > stride_m[j+1] )
    {
        std::swap( rs_a[j], rs_a[j+1] );
        std::swap( rs_c[j], rs_c[j+1] );
    }

    auto& stride_n = has_unit_stride( cs_c ) ? cs_c : cs_b;
    for ( int i = 1;i < n.size(); i++ )
    for ( int j = 0;j < n.size()-i; j++ )
    if ( stride_n[j] > stride_n[j+1] )
    {
        std::swap( cs_b[j], cs_b[j+1] );
        std::swap( cs_c[j], cs_c[j+1] );
    }

    auto& stride_k = has_unit_stride( cs_a ) ? cs_a : rs_b;
    for ( int i = 1;i < k.size(); i++ )
    for ( int j = 0;j < k.size()-i; j++ )
    if ( stride_k[j] > stride_k[j+1] )
    {
        std::swap( cs_a[j], cs_a[j+1] );
        std::swap( rs_b[j], rs_b[j+1] );
    }

    if ( rs_a.empty() ) rs_a.push_back( 1 );
    if ( cs_a.empty() ) cs_a.push_back( 1 );
    if ( rs_b.empty() ) rs_b.push_back( 1 );
    if ( cs_b.empty() ) cs_b.push_back( 1 );
    if ( rs_c.empty() ) rs_c.push_back( 1 );
    if ( cs_c.empty() ) cs_c.push_back( 1 );

    obj_t a_o, b_o, c_o;
    bli_obj_create_with_attached_buffer( dt, m_mat, k_mat, const_cast<void*>(a), rs_a[0], cs_a[0], &a_o );
    bli_obj_create_with_attached_buffer( dt, k_mat, n_mat, const_cast<void*>(b), rs_b[0], cs_b[0], &b_o );
    bli_obj_create_with_attached_buffer( dt, m_mat, n_mat,                   c , rs_c[0], cs_c[0], &c_o );

    packm_tensor_params_t params_a( m.size(), m.data(), rs_a.data(),
                                    k.size(), k.data(), cs_a.data() );
    packm_tensor_params_t params_b( n.size(), n.data(), cs_b.data(),
                                    k.size(), k.data(), rs_b.data() );
    gemm_tensor_params_t params_c( m.size(), m.data(), rs_c.data(),
                                   n.size(), n.data(), cs_c.data() );

    bli_obj_set_pack_fn( packm_tensor, &a_o );
    bli_obj_set_pack_fn( packm_tensor, &b_o );
    bli_obj_set_ker_fn( gemm_tensor, &c_o );
    bli_obj_set_pack_params( &params_a, &a_o );
    bli_obj_set_pack_params( &params_b, &b_o );
    bli_obj_set_ker_params( &params_c, &c_o );

    obj_t alpha_o, beta_o;
    bli_obj_create_1x1_with_attached_buffer( dt, const_cast<void*>(alpha), &alpha_o );
    bli_obj_create_1x1_with_attached_buffer( dt, const_cast<void*>(beta), &beta_o );

    rntm_t rntm;
    bli_rntm_init_from_global( &rntm );
    bli_rntm_disable_l3_sup( &rntm );

    bli_gemm_ex( &alpha_o, &a_o, &b_o, &beta_o, &c_o, NULL, &rntm );
}

int main()
{
    auto N = 5;

    gint_t ndim_a = 4;
    gint_t ndim_b = 4;
    gint_t ndim_c = 4;

    std::vector<dim_t> len_a(ndim_a, N);
    std::vector<dim_t> len_b(ndim_b, N);
    std::vector<dim_t> len_c(ndim_c, N);

    std::vector<inc_t> stride_a(ndim_a, 1);
    std::vector<inc_t> stride_b(ndim_b, 1);
    std::vector<inc_t> stride_c(ndim_c, 1);
    for ( gint_t i = 1; i < ndim_a; i++ )
        stride_a[i] = stride_a[i-1] * len_a[i - 1];
    for ( gint_t i = 1; i < ndim_b; i++ )
        stride_b[i] = stride_b[i-1] * len_b[i - 1];
    for ( gint_t i = 1; i < ndim_c; i++ )
        stride_c[i] = stride_c[i-1] * len_c[i - 1];

    std::vector<int> dim_a(ndim_a);
    std::vector<int> dim_b(ndim_b);
    std::vector<int> dim_c(ndim_c);
    std::iota(dim_a.begin(), dim_a.end(), 0);
    std::iota(dim_b.begin(), dim_b.end(), 0);
    std::iota(dim_c.begin(), dim_c.end(), 0);

    for ( int dt_ = BLIS_DT_LO; dt_ <= BLIS_DT_HI; dt_++ )
    do
    do
    do
    {
        auto dt = ( num_t )dt_;

        auto ndim_m = (ndim_a + ndim_c - ndim_b)/2;
        auto ndim_k = (ndim_a + ndim_b - ndim_c)/2;

        std::vector<dim_t> m(len_a.begin(), len_a.begin()+ndim_m);
        std::vector<dim_t> n(len_b.begin()+ndim_k, len_b.end());
        std::vector<dim_t> k(len_b.begin(), len_b.begin()+ndim_k);

        std::vector<inc_t> rs_a(stride_a.begin(), stride_a.begin()+ndim_m);
        std::vector<inc_t> cs_a(stride_a.begin()+ndim_m, stride_a.end());
        std::vector<inc_t> rs_b(stride_b.begin(), stride_b.begin()+ndim_k);
        std::vector<inc_t> cs_b(stride_b.begin()+ndim_k, stride_b.end());
        std::vector<inc_t> rs_c(stride_c.begin(), stride_c.begin()+ndim_m);
        std::vector<inc_t> cs_c(stride_c.begin()+ndim_m, stride_c.end());

        dim_t m_tot = 1;
        dim_t n_tot = 1;
        dim_t k_tot = 1;
        for ( auto i : m ) m_tot *= i;
        for ( auto i : n ) n_tot *= i;
        for ( auto i : k ) k_tot *= i;

        obj_t a, b, c, c_ref, norm;

        bli_obj_create( dt, m_tot*k_tot, 1, 1, 1, &a );
        bli_obj_create( dt, k_tot*n_tot, 1, 1, 1, &b );
        bli_obj_create( dt, m_tot*n_tot, 1, 1, 1, &c );
        bli_obj_create( dt, m_tot*n_tot, 1, 1, 1, &c_ref );
        bli_obj_create_1x1( bli_dt_proj_to_real( dt ), &norm );

        bli_randv( &a );
        bli_randv( &b );
        bli_randv( &c );
        bli_copyv( &c, &c_ref );

        tcontract( dt, m, n, k,
                   bli_obj_buffer_for_const( dt, &BLIS_ONE ),
                   bli_obj_buffer( &a ), rs_a, cs_a,
                   bli_obj_buffer( &b ), rs_b, cs_b,
                   bli_obj_buffer_for_const( dt, &BLIS_ZERO ),
                   bli_obj_buffer( &c ), rs_c, cs_c );

        tcontract_ref( dt, m, n, k,
                       bli_obj_buffer_for_const( dt, &BLIS_ONE ),
                       bli_obj_buffer( &a ), rs_a, cs_a,
                       bli_obj_buffer( &b ), rs_b, cs_b,
                       bli_obj_buffer_for_const( dt, &BLIS_ZERO ),
                       bli_obj_buffer( &c_ref ), rs_c, cs_c );

        bli_subv( &c_ref, &c );
        bli_normfv( &c, &norm );

        double normr, normi;
        bli_getsc( &norm, &normr, &normi );

        printf("dt: %d, dim_a: [%d,%d,%d,%d], dim_b: [%d,%d,%d,%d], dim_c: [%d,%d,%d,%d], norm: %g\n",
               dt, dim_a[0], dim_a[1], dim_a[2], dim_a[3],
                   dim_b[0], dim_b[1], dim_b[2], dim_b[3],
                   dim_c[0], dim_c[1], dim_c[2], dim_c[3],
               normr / std::sqrt( bli_obj_vector_dim( &c ) ) );

        bli_obj_free( &a );
        bli_obj_free( &b );
        bli_obj_free( &c );
        bli_obj_free( &c_ref );
    }
    while (std::next_permutation(dim_a.begin(), dim_a.end()));
    while (std::next_permutation(dim_b.begin(), dim_b.end()));
    while (std::next_permutation(dim_c.begin(), dim_c.end()));
}

