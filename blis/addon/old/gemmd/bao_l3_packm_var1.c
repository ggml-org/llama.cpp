/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2021, The University of Texas at Austin

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

//
// Variant 1 provides basic support for packing by calling packm_cxk().
//

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTECH2(bao_,ch,varname) \
     ( \
       trans_t          transc, \
       pack_t           schema, \
       dim_t            m, \
       dim_t            n, \
       dim_t            m_max, \
       dim_t            n_max, \
       ctype*  restrict kappa, \
       ctype*  restrict d, inc_t incd, \
       ctype*  restrict c, inc_t rs_c, inc_t cs_c, \
       ctype*  restrict p, inc_t rs_p, inc_t cs_p, \
                           dim_t pd_p, inc_t ps_p, \
       cntx_t* restrict cntx, \
       thrinfo_t* restrict thread  \
     ) \
{ \
	ctype* restrict kappa_cast = kappa; \
	ctype* restrict c_cast     = c; \
	ctype* restrict p_cast     = p; \
\
	dim_t           iter_dim; \
	dim_t           n_iter; \
	dim_t           it, ic; \
	dim_t           ic0; \
	doff_t          ic_inc; \
	dim_t           panel_len; \
	dim_t           panel_len_max; \
	dim_t           panel_dim; \
	dim_t           panel_dim_max; \
	inc_t           incc; \
	inc_t           ldc; \
	inc_t           ldp; \
	conj_t          conjc; \
\
\
	/* Extract the conjugation bit from the transposition argument. */ \
	conjc = bli_extract_conj( transc ); \
\
	/* Create flags to incidate row or column storage. Note that the
	   schema bit that encodes row or column is describing the form of
	   micro-panel, not the storage in the micro-panel. Hence the
	   mismatch in "row" and "column" semantics. */ \
	bool row_stored = bli_is_col_packed( schema ); \
	/*bool col_stored = bli_is_row_packed( schema );*/ \
\
	/* If the row storage flag indicates row storage, then we are packing
	   to column panels; otherwise, if the strides indicate column storage,
	   we are packing to row panels. */ \
	if ( row_stored ) \
	{ \
		/* Prepare to pack to row-stored column panels. */ \
		iter_dim       = n; \
		panel_len      = m; \
		panel_len_max  = m_max; \
		panel_dim_max  = pd_p; \
		incc           = cs_c; \
		ldc            = rs_c; \
		ldp            = rs_p; \
	} \
	else /* if ( col_stored ) */ \
	{ \
		/* Prepare to pack to column-stored row panels. */ \
		iter_dim       = m; \
		panel_len      = n; \
		panel_len_max  = n_max; \
		panel_dim_max  = pd_p; \
		incc           = rs_c; \
		ldc            = cs_c; \
		ldp            = cs_p; \
	} \
\
	/* Compute the total number of iterations we'll need. */ \
	n_iter = iter_dim / panel_dim_max + ( iter_dim % panel_dim_max ? 1 : 0 ); \
\
	/* Set the initial values and increments for indices related to C and P
	   based on whether reverse iteration was requested. */ \
	{ \
		ic0    = 0; \
		ic_inc = panel_dim_max; \
	} \
\
	ctype* restrict p_begin = p_cast; \
\
	/* Query the number of threads and thread ids from the current thread's
	   packm thrinfo_t node. */ \
	const dim_t nt  = bli_thrinfo_n_way( thread ); \
	const dim_t tid = bli_thrinfo_work_id( thread ); \
\
	/* Suppress warnings in case tid isn't used (ie: as in slab partitioning). */ \
	( void )nt; \
	( void )tid; \
\
	dim_t it_start, it_end, it_inc; \
\
	/* Determine the thread range and increment using the current thread's
	   packm thrinfo_t node. NOTE: The definition of bli_thread_range_jrir()
	   will depend on whether slab or round-robin partitioning was requested
	   at configure-time. */ \
	bli_thread_range_jrir( thread, n_iter, 1, FALSE, &it_start, &it_end, &it_inc ); \
\
	/* Iterate over every logical micropanel in the source matrix. */ \
	for ( ic  = ic0,    it  = 0; it < n_iter; \
	      ic += ic_inc, it += 1 ) \
	{ \
		panel_dim = bli_min( panel_dim_max, iter_dim - ic ); \
\
		ctype* restrict c_begin = c_cast   + (ic  )*incc; \
\
		ctype* restrict c_use = c_begin; \
		ctype* restrict p_use = p_begin; \
\
		/* The definition of bli_packm_my_iter() will depend on whether slab
		   or round-robin partitioning was requested at configure-time. (The
		   default is slab.) */ \
		if ( bli_packm_my_iter( it, it_start, it_end, tid, nt ) ) \
		{ \
			PASTECH2(bao_,ch,packm_cxk) \
			( \
			  conjc, \
			  schema, \
			  panel_dim, \
			  panel_dim_max, \
			  panel_len, \
			  panel_len_max, \
			  kappa_cast, \
			  d,     incd, \
			  c_use, incc, ldc, \
			  p_use,       ldp, \
			  cntx  \
			); \
		} \
\
/*
if ( !row_stored ) \
PASTEMAC(ch,fprintm)( stdout, "packm_var1: a packed", panel_dim_max, panel_len_max, \
                               p_use, rs_p, cs_p, "%5.2f", "" ); \
else \
PASTEMAC(ch,fprintm)( stdout, "packm_var1: b packed", panel_len_max, panel_dim_max, \
                               p_use, rs_p, cs_p, "%5.2f", "" ); \
*/ \
\
		p_begin += ps_p; \
	} \
}

//INSERT_GENTFUNC_BASIC0( packm_var1 )
GENTFUNC( float,    s, packm_var1 )
GENTFUNC( double,   d, packm_var1 )
GENTFUNC( scomplex, c, packm_var1 )
GENTFUNC( dcomplex, z, packm_var1 )

