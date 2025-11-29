/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2019, Advanced Micro Devices, Inc.

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
#include "blix.h"

#define FUNCPTR_T gemmsup_fp

typedef void (*FUNCPTR_T)
     (
       bool             packa,
       bool             packb,
       conj_t           conja,
       conj_t           conjb,
       dim_t            m,
       dim_t            n,
       dim_t            k,
       void*   restrict alpha,
       void*   restrict a, inc_t rs_a, inc_t cs_a,
       void*   restrict b, inc_t rs_b, inc_t cs_b,
       void*   restrict beta,
       void*   restrict c, inc_t rs_c, inc_t cs_c,
       stor3_t          eff_id,
       cntx_t* restrict cntx,
       rntm_t* restrict rntm,
       thrinfo_t* restrict thread
     );

//
// -- var2 ---------------------------------------------------------------------
//

static FUNCPTR_T GENARRAY(ftypes_var2,gemm_ref_var2);

void blx_gemm_ref_var2
     (
       trans_t trans,
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       stor3_t eff_id,
       cntx_t* cntx,
       rntm_t* rntm,
       thrinfo_t* thread
     )
{
	const num_t    dt        = bli_obj_dt( c );

	const bool     packa     = bli_rntm_pack_a( rntm );
	const bool     packb     = bli_rntm_pack_b( rntm );

	const conj_t   conja     = bli_obj_conj_status( a );
	const conj_t   conjb     = bli_obj_conj_status( b );

	const dim_t    m         = bli_obj_length( c );
	const dim_t    n         = bli_obj_width( c );
	      dim_t    k;

	void* restrict buf_a = bli_obj_buffer_at_off( a );
	      inc_t    rs_a;
	      inc_t    cs_a;

	void* restrict buf_b = bli_obj_buffer_at_off( b );
	      inc_t    rs_b;
	      inc_t    cs_b;

	if ( bli_obj_has_notrans( a ) )
	{
		k     = bli_obj_width( a );

		rs_a  = bli_obj_row_stride( a );
		cs_a  = bli_obj_col_stride( a );
	}
	else // if ( bli_obj_has_trans( a ) )
	{
		// Assign the variables with an implicit transposition.
		k     = bli_obj_length( a );

		rs_a  = bli_obj_col_stride( a );
		cs_a  = bli_obj_row_stride( a );
	}

	if ( bli_obj_has_notrans( b ) )
	{
		rs_b  = bli_obj_row_stride( b );
		cs_b  = bli_obj_col_stride( b );
	}
	else // if ( bli_obj_has_trans( b ) )
	{
		// Assign the variables with an implicit transposition.
		rs_b  = bli_obj_col_stride( b );
		cs_b  = bli_obj_row_stride( b );
	}

	void* restrict buf_c     = bli_obj_buffer_at_off( c );
	const inc_t    rs_c      = bli_obj_row_stride( c );
	const inc_t    cs_c      = bli_obj_col_stride( c );

	void* restrict buf_alpha = bli_obj_buffer_for_1x1( dt, alpha );
	void* restrict buf_beta  = bli_obj_buffer_for_1x1( dt, beta );

	// Index into the type combination array to extract the correct
	// function pointer.
	FUNCPTR_T f = ftypes_var2[dt];

	if ( bli_is_notrans( trans ) )
	{
		// Invoke the function.
		f
		(
		  packa,
		  packb,
		  conja,
		  conjb,
		  m,
		  n,
		  k,
		  buf_alpha,
		  buf_a, rs_a, cs_a,
		  buf_b, rs_b, cs_b,
		  buf_beta,
		  buf_c, rs_c, cs_c,
		  eff_id,
		  cntx,
		  rntm,
		  thread
		);
	}
	else
	{
		bli_abort();
	}
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       bool             packa, \
       bool             packb, \
       conj_t           conja, \
       conj_t           conjb, \
       dim_t            m, \
       dim_t            n, \
       dim_t            k, \
       void*   restrict alpha, \
       void*   restrict a, inc_t rs_a, inc_t cs_a, \
       void*   restrict b, inc_t rs_b, inc_t cs_b, \
       void*   restrict beta, \
       void*   restrict c, inc_t rs_c, inc_t cs_c, \
       stor3_t          stor_id, \
       cntx_t* restrict cntx, \
       rntm_t* restrict rntm, \
       thrinfo_t* restrict thread  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	/* If m or n is zero, return immediately. */ \
	if ( bli_zero_dim2( m, n ) ) return; \
\
	/* If k < 1 or alpha is zero, scale by beta and return. */ \
	if ( k < 1 || PASTEMAC(ch,eq0)( *(( ctype* )alpha) ) ) \
	{ \
		PASTEMAC(ch,scalm) \
		( \
		  BLIS_NO_CONJUGATE, \
		  0, \
		  BLIS_NONUNIT_DIAG, \
		  BLIS_DENSE, \
		  m, n, \
		  beta, \
		  c, rs_c, cs_c \
		); \
		return; \
	} \
\
	/* Query the context for various blocksizes. NOTE: We query the
	   regular blocksizes since the sup blocksizes are not guaranteed
	   to have default values. */ \
	const dim_t NR  = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx ); \
	const dim_t MR  = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx ); \
	const dim_t NC  = bli_cntx_get_blksz_def_dt( dt, BLIS_NC, cntx ); \
	const dim_t MC  = bli_cntx_get_blksz_def_dt( dt, BLIS_MC, cntx ); \
	const dim_t KC  = bli_cntx_get_blksz_def_dt( dt, BLIS_KC, cntx ); \
\
	/* Compute partitioning step values for each matrix of each loop. */ \
	const inc_t jcstep_c = cs_c; \
	const inc_t jcstep_b = cs_b; \
\
	const inc_t pcstep_a = cs_a; \
	const inc_t pcstep_b = rs_b; \
\
	const inc_t icstep_c = rs_c; \
	const inc_t icstep_a = rs_a; \
\
	const inc_t jrstep_c = cs_c * NR; \
	const inc_t jrstep_b = cs_b * NR; \
\
	const inc_t irstep_c = rs_c * MR; \
	const inc_t irstep_a = rs_a * MR; \
\
	/* Query the context for the sup microkernel address and cast it to its
	   function pointer type. */ \
	PASTECH(ch,gemmsup_ker_ft) \
               gemmsup_ker = bli_cntx_get_l3_sup_ker_dt( dt, stor_id, cntx ); \
\
	ctype* restrict a_00       = a; \
	ctype* restrict b_00       = b; \
	ctype* restrict c_00       = c; \
	ctype* restrict alpha_cast = alpha; \
	ctype* restrict beta_cast  = beta; \
\
	/* Make local copies of beta and one scalars to prevent any unnecessary
	   sharing of cache lines between the cores' caches. */ \
	ctype           beta_local = *beta_cast; \
	ctype           one_local  = *PASTEMAC(ch,1); \
\
	auxinfo_t       aux; \
\
	/* Compute number of primary and leftover components of the JC loop. */ \
	/*const dim_t jc_iter = ( n + NC - 1 ) / NC;*/ \
	const dim_t jc_left =   n % NC; \
\
	/* Loop over the n dimension (NC rows/columns at a time). */ \
	for ( dim_t jj = 0; jj < n; jj += NC ) \
	{ \
		/* Calculate the thread's current JC block dimension. */ \
		const dim_t nc_cur = ( NC <= n - jj ? NC : jc_left ); \
\
		ctype* restrict b_jc = b_00 + jj * jcstep_b; \
		ctype* restrict c_jc = c_00 + jj * jcstep_c; \
\
		/* Compute number of primary and leftover components of the PC loop. */ \
		/*const dim_t pc_iter = ( k + KC - 1 ) / KC;*/ \
		const dim_t pc_left =   k % KC; \
\
		/* Loop over the k dimension (KC rows/columns at a time). */ \
		for ( dim_t pp = 0; pp < k; pp += KC ) \
		{ \
			/* Calculate the thread's current PC block dimension. */ \
			const dim_t kc_cur = ( KC <= k - pp ? KC : pc_left ); \
\
			ctype* restrict a_pc = a_00 + pp * pcstep_a; \
			ctype* restrict b_pc = b_jc + pp * pcstep_b; \
\
			/* Only apply beta to the first iteration of the pc loop. */ \
			ctype* restrict beta_use = ( pp == 0 ? &beta_local : &one_local ); \
\
			/*bli_auxinfo_set_ps_b( ps_b_use, &aux );*/ \
\
			/* Compute number of primary and leftover components of the IC loop. */ \
			/*const dim_t ic_iter = ( m + MC - 1 ) / MC;*/ \
			const dim_t ic_left =   m % MC; \
\
			/* Loop over the m dimension (MC rows at a time). */ \
			for ( dim_t ii = 0; ii < m; ii += MC ) \
			{ \
				/* Calculate the thread's current IC block dimension. */ \
				const dim_t mc_cur = ( MC <= m - ii ? MC : ic_left ); \
\
				ctype* restrict a_ic = a_pc + ii * icstep_a; \
				ctype* restrict c_ic = c_jc + ii * icstep_c; \
\
				/*bli_auxinfo_set_ps_a( ps_a_use, &aux );*/ \
\
				/* Compute number of primary and leftover components of the JR loop. */ \
				dim_t jr_iter = ( nc_cur + NR - 1 ) / NR; \
				dim_t jr_left =   nc_cur % NR; \
\
				/* Loop over the n dimension (NR columns at a time). */ \
				for ( dim_t j = 0; j < jr_iter; j += 1 ) \
				{ \
					const dim_t nr_cur = ( bli_is_not_edge_f( j, jr_iter, jr_left ) ? NR : jr_left ); \
\
					ctype* restrict b_jr = b_pc + j * jrstep_b; \
					ctype* restrict c_jr = c_ic + j * jrstep_c; \
\
					/* Compute number of primary and leftover components of the IR loop. */ \
					const dim_t ir_iter = ( mc_cur + MR - 1 ) / MR; \
					const dim_t ir_left =   mc_cur % MR; \
\
					/* Loop over the m dimension (MR columns at a time). */ \
					for ( dim_t i = 0; i < ir_iter; i += 1 ) \
					{ \
						const dim_t mr_cur = ( bli_is_not_edge_f( i, ir_iter, ir_left ) ? MR : ir_left ); \
\
						ctype* restrict a_ir = a_ic + i * irstep_a; \
						ctype* restrict c_ir = c_jr + i * irstep_c; \
\
						/*
						bli_auxinfo_set_next_a( a2, &aux ); \
						bli_auxinfo_set_next_b( b2, &aux ); \
						*/ \
\
						/* Invoke the kernel. */ \
						gemmsup_ker \
						( \
						  conja, \
						  conjb, \
						  mr_cur, \
						  nr_cur, \
						  kc_cur, \
						  alpha_cast, \
						  a_ir, rs_a, cs_a, \
						  b_jr, rs_b, cs_b, \
						  beta_use, \
						  c_ir, rs_c, cs_c, \
						  &aux, \
						  cntx  \
						); \
					} \
				} \
			} \
		} \
	} \
\
/*
PASTEMAC(ch,fprintm)( stdout, "gemm_ref_var2: b1", kc_cur, nr_cur, b_jr, rs_b, cs_b, "%4.1f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemm_ref_var2: a1", mr_cur, kc_cur, a_ir, rs_a, cs_a, "%4.1f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemm_ref_var2: c ", mr_cur, nr_cur, c_ir, rs_c, cs_c, "%4.1f", "" ); \
*/ \
}

INSERT_GENTFUNC_BASIC0( gemm_ref_var2 )

