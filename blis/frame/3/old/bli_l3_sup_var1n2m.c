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

#define FUNCPTR_T gemmsup_fp

typedef void (*FUNCPTR_T)
     (
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
       cntl_t* restrict cntl,
       thrinfo_t* restrict thread
     );

//
// -- var1n --------------------------------------------------------------------
//

static FUNCPTR_T GENARRAY(ftypes_var1n,gemmsup_ref_var1n);

void bli_gemmsup_ref_var1n
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
       cntl_t* cntl,
       thrinfo_t* thread
     )
{
#if 0
	obj_t at, bt;

	bli_obj_alias_to( a, &at );
	bli_obj_alias_to( b, &bt );

	// Induce transpositions on A and/or B if either object is marked for
	// transposition. We can induce "fast" transpositions since they objects
	// are guaranteed to not have structure or be packed.
	if ( bli_obj_has_trans( &at ) ) { bli_obj_induce_fast_trans( &at ); }
	if ( bli_obj_has_trans( &bt ) ) { bli_obj_induce_fast_trans( &bt ); }

	const num_t    dt_exec   = bli_obj_dt( c );

	const conj_t   conja     = bli_obj_conj_status( a );
	const conj_t   conjb     = bli_obj_conj_status( b );

	const dim_t    m         = bli_obj_length( c );
	const dim_t    n         = bli_obj_width( c );

	const dim_t    k         = bli_obj_width( &at );

	void* restrict buf_a     = bli_obj_buffer_at_off( &at );
	const inc_t    rs_a      = bli_obj_row_stride( &at );
	const inc_t    cs_a      = bli_obj_col_stride( &at );

	void* restrict buf_b     = bli_obj_buffer_at_off( &bt );
	const inc_t    rs_b      = bli_obj_row_stride( &bt );
	const inc_t    cs_b      = bli_obj_col_stride( &bt );

	void* restrict buf_c     = bli_obj_buffer_at_off( c );
	const inc_t    rs_c      = bli_obj_row_stride( c );
	const inc_t    cs_c      = bli_obj_col_stride( c );

	void* restrict buf_alpha = bli_obj_buffer_for_1x1( dt_exec, alpha );
	void* restrict buf_beta  = bli_obj_buffer_for_1x1( dt_exec, beta );

#else

	const num_t    dt_exec   = bli_obj_dt( c );

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

	void* restrict buf_alpha = bli_obj_buffer_for_1x1( dt_exec, alpha );
	void* restrict buf_beta  = bli_obj_buffer_for_1x1( dt_exec, beta );

#endif

	// Index into the type combination array to extract the correct
	// function pointer.
	FUNCPTR_T f = ftypes_var1n[dt_exec];

	if ( bli_is_notrans( trans ) )
	{
		// Invoke the function.
		f
		(
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
		  cntl,
		  thread
		);
	}
	else
	{
		// Invoke the function (transposing the operation).
		f
		(
		  conjb,             // swap the conj values.
		  conja,
		  n,                 // swap the m and n dimensions.
		  m,
		  k,
		  buf_alpha,
		  buf_b, cs_b, rs_b, // swap the positions of A and B.
		  buf_a, cs_a, rs_a, // swap the strides of A and B.
		  buf_beta,
		  buf_c, cs_c, rs_c, // swap the strides of C.
		  bli_stor3_trans( eff_id ), // transpose the stor3_t id.
		  cntx,
		  rntm,
		  cntl,
		  thread
		);
	}
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
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
       cntl_t* restrict cntl, \
       thrinfo_t* restrict thread  \
     ) \
{ \
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
	const num_t dt  = PASTEMAC(ch,type); \
\
	/* This transposition of the stor3_t id value is inherent to variant 1.
	   The reason: we assume that variant 2 is the "main" variant. The
	   consequence of this is that we assume that the millikernels that
	   iterate over m are registered to the kernel group associated with
	   the kernel preference. So, regardless of whether the mkernels are
	   row- or column-preferential, millikernels that iterate over n are
	   always placed in the slots for the opposite kernel group. */ \
	stor_id = bli_stor3_trans( stor_id ); \
\
	/* Query the context for various blocksizes. */ \
	const dim_t NR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NR, cntx ); \
	const dim_t MR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MR, cntx ); \
	const dim_t NC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NC, cntx ); \
	const dim_t MC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MC, cntx ); \
	const dim_t KC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_KC, cntx ); \
\
	dim_t KC; \
	if      ( FALSE                  ) KC = KC0; \
	else if ( stor_id == BLIS_RRC || \
	          stor_id == BLIS_CRC    ) KC = KC0; \
	else if ( m <=   MR && n <=   NR ) KC = KC0; \
	else if ( m <= 2*MR && n <= 2*NR ) KC = KC0 / 2; \
	else if ( m <= 3*MR && n <= 3*NR ) KC = (( KC0 / 3 ) / 4 ) * 4; \
	else if ( m <= 4*MR && n <= 4*NR ) KC = KC0 / 4; \
	else                               KC = (( KC0 / 5 ) / 4 ) * 4; \
\
	/* Nudge NC up to a multiple of MR and MC up to a multiple of NR. */ \
	const dim_t NC  = bli_align_dim_to_mult( NC0, MR ); \
	const dim_t MC  = bli_align_dim_to_mult( MC0, NR ); \
\
	/* Query the maximum blocksize for MR, which implies a maximum blocksize
	   extension for the final iteration. */ \
	const dim_t MRM = bli_cntx_get_l3_sup_blksz_max_dt( dt, BLIS_MR, cntx ); \
	const dim_t MRE = MRM - MR; \
\
	/* Compute partitioning step values for each matrix of each loop. */ \
	const inc_t jcstep_c = rs_c * NC; \
	const inc_t jcstep_a = rs_a * NC; \
\
	const inc_t pcstep_a = cs_a * KC; \
	const inc_t pcstep_b = rs_b * KC; \
\
	const inc_t icstep_c = cs_c * MC; \
	const inc_t icstep_b = cs_b * MC; \
\
	const inc_t jrstep_c = rs_c * MR; \
	const inc_t jrstep_a = rs_a * MR; \
\
	/*
	const inc_t irstep_c = cs_c * NR; \
	const inc_t irstep_b = cs_b * NR; \
	*/ \
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
	ctype* restrict one        = PASTEMAC(ch,1); \
\
	auxinfo_t       aux; \
\
	/* Compute number of primary and leftover components of the outer
	   dimensions.
	   NOTE: Functionally speaking, we compute jc_iter as:
	     jc_iter = m / NC; if ( jc_left ) ++jc_iter;
	   However, this is implemented as:
	     jc_iter = ( m + NC - 1 ) / NC;
	   This avoids a branch at the cost of two additional integer instructions.
	   The pc_iter, mc_iter, nr_iter, and mr_iter variables are computed in
	   similar manner. */ \
	const dim_t jc_iter = ( m + NC - 1 ) / NC; \
	const dim_t jc_left =   m % NC; \
\
	const dim_t pc_iter = ( k + KC - 1 ) / KC; \
	const dim_t pc_left =   k % KC; \
\
	const dim_t ic_iter = ( n + MC - 1 ) / MC; \
	const dim_t ic_left =   n % MC; \
\
	const dim_t jc_inc  = 1; \
	const dim_t pc_inc  = 1; \
	const dim_t ic_inc  = 1; \
	const dim_t jr_inc  = 1; \
	/*
	const dim_t ir_inc  = 1; \
	*/ \
\
	/* Loop over the m dimension (NC rows/columns at a time). */ \
	for ( dim_t jj = 0; jj < jc_iter; jj += jc_inc ) \
	{ \
		const dim_t nc_cur = ( bli_is_not_edge_f( jj, jc_iter, jc_left ) ? NC : jc_left ); \
\
		ctype* restrict a_jc = a_00 + jj * jcstep_a; \
		ctype* restrict c_jc = c_00 + jj * jcstep_c; \
\
		dim_t jr_iter = ( nc_cur + MR - 1 ) / MR; \
		dim_t jr_left =   nc_cur % MR; \
\
		/* An optimization: allow the last jr iteration to contain up to MRE
		   rows of C and A. (If MRE > MR, the mkernel has agreed to handle
		   these cases.) Note that this prevents us from declaring jr_iter and
		   jr_left as const. */ \
		if ( 1 ) \
		if ( MRE != 0 && 1 < jr_iter && jr_left != 0 && jr_left <= MRE ) \
		{ \
			jr_iter--; jr_left += MR; \
		} \
\
		/* Loop over the k dimension (KC rows/columns at a time). */ \
		for ( dim_t pp = 0; pp < pc_iter; pp += pc_inc ) \
		{ \
			const dim_t kc_cur = ( bli_is_not_edge_f( pp, pc_iter, pc_left ) ? KC : pc_left ); \
\
			ctype* restrict a_pc = a_jc + pp * pcstep_a; \
			ctype* restrict b_pc = b_00 + pp * pcstep_b; \
\
			/* Only apply beta to the first iteration of the pc loop. */ \
			ctype* restrict beta_use = ( pp == 0 ? beta_cast : one ); \
\
			/* Loop over the n dimension (MC rows at a time). */ \
			for ( dim_t ii = 0; ii < ic_iter; ii += ic_inc ) \
			{ \
				const dim_t mc_cur = ( bli_is_not_edge_f( ii, ic_iter, ic_left ) ? MC : ic_left ); \
\
				ctype* restrict b_ic = b_pc + ii * icstep_b; \
				ctype* restrict c_ic = c_jc + ii * icstep_c; \
\
				/*
				const dim_t ir_iter = ( mc_cur + NR - 1 ) / NR; \
				const dim_t ir_left =   mc_cur % NR; \
				*/ \
\
				/* Loop over the m dimension (NR columns at a time). */ \
				for ( dim_t j = 0; j < jr_iter; j += jr_inc ) \
				{ \
					const dim_t nr_cur = ( bli_is_not_edge_f( j, jr_iter, jr_left ) ? MR : jr_left ); \
\
					ctype* restrict a_jr = a_pc + j * jrstep_a; \
					ctype* restrict c_jr = c_ic + j * jrstep_c; \
\
					/* Loop over the n dimension (MR rows at a time). */ \
					{ \
						/* Invoke the gemmsup millikernel. */ \
						gemmsup_ker \
						( \
						  conja, \
						  conjb, \
						  nr_cur, /* Notice: nr_cur <= MR. */ \
						  mc_cur, /* Recall: mc_cur partitions the n dimension! */ \
						  kc_cur, \
						  alpha_cast, \
						  a_jr, rs_a, cs_a, \
						  b_ic, rs_b, cs_b, \
						  beta_use, \
						  c_jr, rs_c, cs_c, \
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
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: b1", kc_cur, nr_cur, b_jr, rs_b, cs_b, "%4.1f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: a1", mr_cur, kc_cur, a_ir, rs_a, cs_a, "%4.1f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: c ", mr_cur, nr_cur, c_ir, rs_c, cs_c, "%4.1f", "" ); \
*/ \
}

INSERT_GENTFUNC_BASIC0( gemmsup_ref_var1n )


//
// -- var2m --------------------------------------------------------------------
//

static FUNCPTR_T GENARRAY(ftypes_var2m,gemmsup_ref_var2m);

void bli_gemmsup_ref_var2m
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
       cntl_t* cntl,
       thrinfo_t* thread
     )
{
#if 0
	obj_t at, bt;

	bli_obj_alias_to( a, &at );
	bli_obj_alias_to( b, &bt );

	// Induce transpositions on A and/or B if either object is marked for
	// transposition. We can induce "fast" transpositions since they objects
	// are guaranteed to not have structure or be packed.
	if ( bli_obj_has_trans( &at ) ) { bli_obj_induce_fast_trans( &at ); }
	if ( bli_obj_has_trans( &bt ) ) { bli_obj_induce_fast_trans( &bt ); }

	const num_t    dt_exec   = bli_obj_dt( c );

	const conj_t   conja     = bli_obj_conj_status( a );
	const conj_t   conjb     = bli_obj_conj_status( b );

	const dim_t    m         = bli_obj_length( c );
	const dim_t    n         = bli_obj_width( c );

	const dim_t    k         = bli_obj_width( &at );

	void* restrict buf_a     = bli_obj_buffer_at_off( &at );
	const inc_t    rs_a      = bli_obj_row_stride( &at );
	const inc_t    cs_a      = bli_obj_col_stride( &at );

	void* restrict buf_b     = bli_obj_buffer_at_off( &bt );
	const inc_t    rs_b      = bli_obj_row_stride( &bt );
	const inc_t    cs_b      = bli_obj_col_stride( &bt );

	void* restrict buf_c     = bli_obj_buffer_at_off( c );
	const inc_t    rs_c      = bli_obj_row_stride( c );
	const inc_t    cs_c      = bli_obj_col_stride( c );

	void* restrict buf_alpha = bli_obj_buffer_for_1x1( dt_exec, alpha );
	void* restrict buf_beta  = bli_obj_buffer_for_1x1( dt_exec, beta );

#else
	const num_t    dt_exec   = bli_obj_dt( c );

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

	void* restrict buf_alpha = bli_obj_buffer_for_1x1( dt_exec, alpha );
	void* restrict buf_beta  = bli_obj_buffer_for_1x1( dt_exec, beta );

#endif

	// Index into the type combination array to extract the correct
	// function pointer.
	FUNCPTR_T f = ftypes_var2m[dt_exec];

	if ( bli_is_notrans( trans ) )
	{
		// Invoke the function.
		f
		(
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
		  cntl,
		  thread
		);
	}
	else
	{
		// Invoke the function (transposing the operation).
		f
		(
		  conjb,             // swap the conj values.
		  conja,
		  n,                 // swap the m and n dimensions.
		  m,
		  k,
		  buf_alpha,
		  buf_b, cs_b, rs_b, // swap the positions of A and B.
		  buf_a, cs_a, rs_a, // swap the strides of A and B.
		  buf_beta,
		  buf_c, cs_c, rs_c, // swap the strides of C.
		  bli_stor3_trans( eff_id ), // transpose the stor3_t id.
		  cntx,
		  rntm,
		  cntl,
		  thread
		);
	}
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
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
       cntl_t* restrict cntl, \
       thrinfo_t* restrict thread  \
     ) \
{ \
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
	const num_t dt  = PASTEMAC(ch,type); \
\
	/* Query the context for various blocksizes. */ \
	const dim_t NR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NR, cntx ); \
	const dim_t MR  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MR, cntx ); \
	const dim_t NC  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_NC, cntx ); \
	const dim_t MC  = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_MC, cntx ); \
	const dim_t KC0 = bli_cntx_get_l3_sup_blksz_def_dt( dt, BLIS_KC, cntx ); \
\
	dim_t KC; \
	if      ( stor_id == BLIS_RRR || \
	          stor_id == BLIS_CCC    ) KC = KC0; \
	else if ( stor_id == BLIS_RRC || \
	          stor_id == BLIS_CRC    ) KC = KC0; \
	else if ( m <=   MR && n <=   NR ) KC = KC0; \
	else if ( m <= 2*MR && n <= 2*NR ) KC = KC0 / 2; \
	else if ( m <= 3*MR && n <= 3*NR ) KC = (( KC0 / 3 ) / 4 ) * 4; \
	else if ( m <= 4*MR && n <= 4*NR ) KC = KC0 / 4; \
	else                               KC = (( KC0 / 5 ) / 4 ) * 4; \
\
	/* Query the maximum blocksize for NR, which implies a maximum blocksize
	   extension for the final iteration. */ \
	const dim_t NRM = bli_cntx_get_l3_sup_blksz_max_dt( dt, BLIS_NR, cntx ); \
	const dim_t NRE = NRM - NR; \
\
	/* Compute partitioning step values for each matrix of each loop. */ \
	const inc_t jcstep_c = cs_c * NC; \
	const inc_t jcstep_b = cs_b * NC; \
\
	const inc_t pcstep_a = cs_a * KC; \
	const inc_t pcstep_b = rs_b * KC; \
\
	const inc_t icstep_c = rs_c * MC; \
	const inc_t icstep_a = rs_a * MC; \
\
	const inc_t jrstep_c = cs_c * NR; \
	const inc_t jrstep_b = cs_b * NR; \
\
	/*
	const inc_t irstep_c = rs_c * MR; \
	const inc_t irstep_a = rs_a * MR; \
	*/ \
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
	ctype* restrict one        = PASTEMAC(ch,1); \
\
	auxinfo_t       aux; \
\
	/* Compute number of primary and leftover components of the outer
	   dimensions.
	   NOTE: Functionally speaking, we compute jc_iter as:
	     jc_iter = n / NC; if ( jc_left ) ++jc_iter;
	   However, this is implemented as:
	     jc_iter = ( n + NC - 1 ) / NC;
	   This avoids a branch at the cost of two additional integer instructions.
	   The pc_iter, mc_iter, nr_iter, and mr_iter variables are computed in
	   similar manner. */ \
	const dim_t jc_iter = ( n + NC - 1 ) / NC; \
	const dim_t jc_left =   n % NC; \
\
	const dim_t pc_iter = ( k + KC - 1 ) / KC; \
	const dim_t pc_left =   k % KC; \
\
	const dim_t ic_iter = ( m + MC - 1 ) / MC; \
	const dim_t ic_left =   m % MC; \
\
	const dim_t jc_inc  = 1; \
	const dim_t pc_inc  = 1; \
	const dim_t ic_inc  = 1; \
	const dim_t jr_inc  = 1; \
	/*
	const dim_t ir_inc  = 1; \
	*/ \
\
	/* Loop over the n dimension (NC rows/columns at a time). */ \
	for ( dim_t jj = 0; jj < jc_iter; jj += jc_inc ) \
	{ \
		const dim_t nc_cur = ( bli_is_not_edge_f( jj, jc_iter, jc_left ) ? NC : jc_left ); \
\
		ctype* restrict b_jc = b_00 + jj * jcstep_b; \
		ctype* restrict c_jc = c_00 + jj * jcstep_c; \
\
		dim_t jr_iter = ( nc_cur + NR - 1 ) / NR; \
		dim_t jr_left =   nc_cur % NR; \
\
		/* An optimization: allow the last jr iteration to contain up to NRE
		   columns of C and B. (If NRE > NR, the mkernel has agreed to handle
		   these cases.) Note that this prevents us from declaring jr_iter and
		   jr_left as const. */ \
		if ( 1 ) \
		if ( NRE != 0 && 1 < jr_iter && jr_left != 0 && jr_left <= NRE ) \
		{ \
			jr_iter--; jr_left += NR; \
		} \
\
		/* Loop over the k dimension (KC rows/columns at a time). */ \
		for ( dim_t pp = 0; pp < pc_iter; pp += pc_inc ) \
		{ \
			const dim_t kc_cur = ( bli_is_not_edge_f( pp, pc_iter, pc_left ) ? KC : pc_left ); \
\
			ctype* restrict a_pc = a_00 + pp * pcstep_a; \
			ctype* restrict b_pc = b_jc + pp * pcstep_b; \
\
			/* Only apply beta to the first iteration of the pc loop. */ \
			ctype* restrict beta_use = ( pp == 0 ? beta_cast : one ); \
\
			/* Loop over the m dimension (MC rows at a time). */ \
			for ( dim_t ii = 0; ii < ic_iter; ii += ic_inc ) \
			{ \
				const dim_t mc_cur = ( bli_is_not_edge_f( ii, ic_iter, ic_left ) ? MC : ic_left ); \
\
				ctype* restrict a_ic = a_pc + ii * icstep_a; \
				ctype* restrict c_ic = c_jc + ii * icstep_c; \
\
				/*
				const dim_t ir_iter = ( mc_cur + MR - 1 ) / MR; \
				const dim_t ir_left =   mc_cur % MR; \
				*/ \
\
				/* Loop over the n dimension (NR columns at a time). */ \
				for ( dim_t j = 0; j < jr_iter; j += jr_inc ) \
				{ \
					const dim_t nr_cur = ( bli_is_not_edge_f( j, jr_iter, jr_left ) ? NR : jr_left ); \
\
					ctype* restrict b_jr = b_pc + j * jrstep_b; \
					ctype* restrict c_jr = c_ic + j * jrstep_c; \
\
					/* Loop over the m dimension (MR rows at a time). */ \
					{ \
						/* Invoke the gemmsup millikernel. */ \
						gemmsup_ker \
						( \
						  conja, \
						  conjb, \
						  mc_cur, \
						  nr_cur, \
						  kc_cur, \
						  alpha_cast, \
						  a_ic, rs_a, cs_a, \
						  b_jr, rs_b, cs_b, \
						  beta_use, \
						  c_jr, rs_c, cs_c, \
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
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: b1", kc_cur, nr_cur, b_jr, rs_b, cs_b, "%4.1f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: a1", mr_cur, kc_cur, a_ir, rs_a, cs_a, "%4.1f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemmsup_ref_var2: c ", mr_cur, nr_cur, c_ir, rs_c, cs_c, "%4.1f", "" ); \
*/ \
}

INSERT_GENTFUNC_BASIC0( gemmsup_ref_var2m )

