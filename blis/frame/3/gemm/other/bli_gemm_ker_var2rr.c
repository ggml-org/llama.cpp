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

#define FUNCPTR_T gemm_fp

typedef void (*FUNCPTR_T)
     (
       pack_t  schema_a,
       pack_t  schema_b,
       dim_t   m,
       dim_t   n,
       dim_t   k,
       void*   alpha,
       void*   a, inc_t cs_a, inc_t is_a,
                  dim_t pd_a, inc_t ps_a,
       void*   b, inc_t rs_b, inc_t is_b,
                  dim_t pd_b, inc_t ps_b,
       void*   beta,
       void*   c, inc_t rs_c, inc_t cs_c,
       cntx_t* cntx,
       rntm_t* rntm,
       thrinfo_t* thread
     );

static FUNCPTR_T GENARRAY(ftypes,gemm_ker_var2rr);

//
// -- Macrokernel functions for round-robin partitioning -----------------------
//

void bli_gemm_ker_var2rr
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
	num_t     dt_exec   = bli_obj_exec_dt( c );

	pack_t    schema_a  = bli_obj_pack_schema( a );
	pack_t    schema_b  = bli_obj_pack_schema( b );

	dim_t     m         = bli_obj_length( c );
	dim_t     n         = bli_obj_width( c );
	dim_t     k         = bli_obj_width( a );

	void*     buf_a     = bli_obj_buffer_at_off( a );
	inc_t     cs_a      = bli_obj_col_stride( a );
	inc_t     is_a      = bli_obj_imag_stride( a );
	dim_t     pd_a      = bli_obj_panel_dim( a );
	inc_t     ps_a      = bli_obj_panel_stride( a );

	void*     buf_b     = bli_obj_buffer_at_off( b );
	inc_t     rs_b      = bli_obj_row_stride( b );
	inc_t     is_b      = bli_obj_imag_stride( b );
	dim_t     pd_b      = bli_obj_panel_dim( b );
	inc_t     ps_b      = bli_obj_panel_stride( b );

	void*     buf_c     = bli_obj_buffer_at_off( c );
	inc_t     rs_c      = bli_obj_row_stride( c );
	inc_t     cs_c      = bli_obj_col_stride( c );

	obj_t     scalar_a;
	obj_t     scalar_b;

	void*     buf_alpha;
	void*     buf_beta;

	FUNCPTR_T f;

	// Detach and multiply the scalars attached to A and B.
	bli_obj_scalar_detach( a, &scalar_a );
	bli_obj_scalar_detach( b, &scalar_b );
	bli_mulsc( &scalar_a, &scalar_b );

	// Grab the addresses of the internal scalar buffers for the scalar
	// merged above and the scalar attached to C.
	buf_alpha = bli_obj_internal_scalar_buffer( &scalar_b );
	buf_beta  = bli_obj_internal_scalar_buffer( c );

    // If 1m is being employed on a column- or row-stored matrix with a
    // real-valued beta, we can use the real domain macro-kernel, which
	// eliminates a little overhead associated with the 1m virtual
	// micro-kernel.
	if ( bli_is_1m_packed( schema_a ) )
	{
		bli_l3_ind_recast_1m_params
		(
		  dt_exec,
		  schema_a,
		  c,
		  m, n, k,
		  pd_a, ps_a,
		  pd_b, ps_b,
		  rs_c, cs_c
		);
	}

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_exec];

	// Invoke the function.
	f( schema_a,
	   schema_b,
	   m,
	   n,
	   k,
	   buf_alpha,
	   buf_a, cs_a, is_a,
	          pd_a, ps_a,
	   buf_b, rs_b, is_b,
	          pd_b, ps_b,
	   buf_beta,
	   buf_c, rs_c, cs_c,
	   cntx,
	   rntm,
	   thread );
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       pack_t  schema_a, \
       pack_t  schema_b, \
       dim_t   m, \
       dim_t   n, \
       dim_t   k, \
       void*   alpha, \
       void*   a, inc_t cs_a, inc_t is_a, \
                  dim_t pd_a, inc_t ps_a, \
       void*   b, inc_t rs_b, inc_t is_b, \
                  dim_t pd_b, inc_t ps_b, \
       void*   beta, \
       void*   c, inc_t rs_c, inc_t cs_c, \
       cntx_t* cntx, \
       rntm_t* rntm, \
       thrinfo_t* thread  \
     ) \
{ \
	const num_t     dt         = PASTEMAC(ch,type); \
\
	/* Alias some constants to simpler names. */ \
	const dim_t     MR         = pd_a; \
	const dim_t     NR         = pd_b; \
	/*const dim_t     PACKMR     = cs_a;*/ \
	/*const dim_t     PACKNR     = rs_b;*/ \
\
	/* Query the context for the micro-kernel address and cast it to its
	   function pointer type. */ \
	PASTECH(ch,gemm_ukr_ft) \
	                gemm_ukr   = bli_cntx_get_l3_vir_ukr_dt( dt, BLIS_GEMM_UKR, cntx ); \
\
	/* Temporary C buffer for edge cases. Note that the strides of this
	   temporary buffer are set so that they match the storage of the
	   original C matrix. For example, if C is column-stored, ct will be
	   column-stored as well. */ \
	ctype           ct[ BLIS_STACK_BUF_MAX_SIZE \
	                    / sizeof( ctype ) ] \
	                    __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
	const bool      col_pref    = bli_cntx_ukr_prefers_cols_dt( dt, BLIS_GEMM_VIR_UKR, cntx ); \
	const inc_t     rs_ct       = ( col_pref ? 1 : NR ); \
	const inc_t     cs_ct       = ( col_pref ? MR : 1 ); \
\
	ctype* restrict zero       = PASTEMAC(ch,0); \
	ctype* restrict a_cast     = a; \
	ctype* restrict b_cast     = b; \
	ctype* restrict c_cast     = c; \
	ctype* restrict alpha_cast = alpha; \
	ctype* restrict beta_cast  = beta; \
	ctype* restrict b1; \
	ctype* restrict c1; \
\
	dim_t           m_iter, m_left; \
	dim_t           n_iter, n_left; \
	dim_t           i, j; \
	dim_t           m_cur; \
	dim_t           n_cur; \
	inc_t           rstep_a; \
	inc_t           cstep_b; \
	inc_t           rstep_c, cstep_c; \
	auxinfo_t       aux; \
\
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
	*/ \
\
	/* If any dimension is zero, return immediately. */ \
	if ( bli_zero_dim3( m, n, k ) ) return; \
\
	/* Clear the temporary C buffer in case it has any infs or NaNs. */ \
	PASTEMAC(ch,set0s_mxn)( MR, NR, \
	                        ct, rs_ct, cs_ct ); \
\
	/* Compute number of primary and leftover components of the m and n
	   dimensions. */ \
	n_iter = n / NR; \
	n_left = n % NR; \
\
	m_iter = m / MR; \
	m_left = m % MR; \
\
	if ( n_left ) ++n_iter; \
	if ( m_left ) ++m_iter; \
\
	/* Determine some increments used to step through A, B, and C. */ \
	rstep_a = ps_a; \
\
	cstep_b = ps_b; \
\
	rstep_c = rs_c * MR; \
	cstep_c = cs_c * NR; \
\
	/* Save the pack schemas of A and B to the auxinfo_t object. */ \
	bli_auxinfo_set_schema_a( schema_a, &aux ); \
	bli_auxinfo_set_schema_b( schema_b, &aux ); \
\
	/* Save the imaginary stride of A and B to the auxinfo_t object. */ \
	bli_auxinfo_set_is_a( is_a, &aux ); \
	bli_auxinfo_set_is_b( is_b, &aux ); \
\
	/* The 'thread' argument points to the thrinfo_t node for the 2nd (jr)
	   loop around the microkernel. Here we query the thrinfo_t node for the
	   1st (ir) loop around the microkernel. */ \
	thrinfo_t* caucus = bli_thrinfo_sub_node( thread ); \
\
	/* Query the number of threads and thread ids for each loop. */ \
	dim_t jr_nt  = bli_thrinfo_n_way( thread ); \
	dim_t jr_tid = bli_thrinfo_work_id( thread ); \
	dim_t ir_nt  = bli_thrinfo_n_way( caucus ); \
	dim_t ir_tid = bli_thrinfo_work_id( caucus ); \
\
	dim_t jr_start, jr_end; \
	dim_t ir_start, ir_end; \
	dim_t jr_inc,   ir_inc; \
\
	/* Determine the thread range and increment for each thrinfo_t node. */ \
	bli_thread_range_jrir_rr( thread, n_iter, 1, FALSE, &jr_start, &jr_end, &jr_inc ); \
	bli_thread_range_jrir_rr( caucus, m_iter, 1, FALSE, &ir_start, &ir_end, &ir_inc ); \
\
	/* Loop over the n dimension (NR columns at a time). */ \
	for ( j = jr_start; j < jr_end; j += jr_inc ) \
	{ \
		ctype* restrict a1; \
		ctype* restrict c11; \
		ctype* restrict b2; \
\
		b1 = b_cast + j * cstep_b; \
		c1 = c_cast + j * cstep_c; \
\
		n_cur = ( bli_is_not_edge_f( j, n_iter, n_left ) ? NR : n_left ); \
\
		/* Initialize our next panel of B to be the current panel of B. */ \
		b2 = b1; \
\
		/* Loop over the m dimension (MR rows at a time). */ \
		for ( i = ir_start; i < ir_end; i += ir_inc ) \
		{ \
			ctype* restrict a2; \
\
			a1  = a_cast + i * rstep_a; \
			c11 = c1     + i * rstep_c; \
\
			m_cur = ( bli_is_not_edge_f( i, m_iter, m_left ) ? MR : m_left ); \
\
			/* Compute the addresses of the next panels of A and B. */ \
			a2 = bli_gemm_get_next_a_upanel( a1, rstep_a, ir_inc ); \
			if ( bli_is_last_iter_rr( i, ir_end, ir_tid, ir_nt ) ) \
			{ \
				a2 = a_cast; \
				b2 = bli_gemm_get_next_b_upanel( b1, cstep_b, jr_inc ); \
				if ( bli_is_last_iter_rr( j, jr_end, jr_tid, jr_nt ) ) \
					b2 = b_cast; \
			} \
\
			/* Save addresses of next panels of A and B to the auxinfo_t
			   object. */ \
			bli_auxinfo_set_next_a( a2, &aux ); \
			bli_auxinfo_set_next_b( b2, &aux ); \
\
			/* Handle interior and edge cases separately. */ \
			if ( m_cur == MR && n_cur == NR ) \
			{ \
				/* Invoke the gemm micro-kernel. */ \
				gemm_ukr \
				( \
				  k, \
				  alpha_cast, \
				  a1, \
				  b1, \
				  beta_cast, \
				  c11, rs_c, cs_c, \
				  &aux, \
				  cntx  \
				); \
			} \
			else \
			{ \
				/* Invoke the gemm micro-kernel. */ \
				gemm_ukr \
				( \
				  k, \
				  alpha_cast, \
				  a1, \
				  b1, \
				  zero, \
				  ct, rs_ct, cs_ct, \
				  &aux, \
				  cntx  \
				); \
\
				/* Scale the bottom edge of C and add the result from above. */ \
				PASTEMAC(ch,xpbys_mxn)( m_cur, n_cur, \
				                        ct,  rs_ct, cs_ct, \
				                        beta_cast, \
				                        c11, rs_c,  cs_c ); \
			} \
		} \
	} \
\
/*
PASTEMAC(ch,fprintm)( stdout, "gemm_ker_var2rr: b1", k, NR, b1, NR, 1, "%4.1f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemm_ker_var2rr: a1", MR, k, a1, 1, MR, "%4.1f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemm_ker_var2rr: c after", m_cur, n_cur, c11, rs_c, cs_c, "%4.1f", "" ); \
*/ \
}

INSERT_GENTFUNC_BASIC0( gemm_ker_var2rr )

