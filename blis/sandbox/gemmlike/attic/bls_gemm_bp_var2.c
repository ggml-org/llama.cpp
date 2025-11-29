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

#define FUNCPTR_T gemm_fp

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
       cntx_t* restrict cntx,
       rntm_t* restrict rntm,
       thrinfo_t* restrict thread
     );

//
// -- gemm-like block-panel algorithm (object interface) -----------------------
//

// Define a function pointer array named ftypes and initialize its contents with
// the addresses of the typed functions defined below, bls_?gemm_bp_var2().
static FUNCPTR_T GENARRAY_PREF(ftypes,bls_,gemm_bp_var2);

void bls_gemm_bp_var2
     (
       obj_t*  alpha,
       obj_t*  a,
       obj_t*  b,
       obj_t*  beta,
       obj_t*  c,
       cntx_t* cntx,
       rntm_t* rntm,
       thrinfo_t* thread
     )
{
	const num_t    dt        = bli_obj_dt( c );

	const conj_t   conja     = bli_obj_conj_status( a );
	const conj_t   conjb     = bli_obj_conj_status( b );

	const dim_t    m         = bli_obj_length( c );
	const dim_t    n         = bli_obj_width( c );
	const dim_t    k         = bli_obj_width( a );

	void* restrict buf_a     = bli_obj_buffer_at_off( a );
	const inc_t    rs_a      = bli_obj_row_stride( a );
	const inc_t    cs_a      = bli_obj_col_stride( a );

	void* restrict buf_b     = bli_obj_buffer_at_off( b );
	const inc_t    rs_b      = bli_obj_row_stride( b );
	const inc_t    cs_b      = bli_obj_col_stride( b );

	void* restrict buf_c     = bli_obj_buffer_at_off( c );
	const inc_t    rs_c      = bli_obj_row_stride( c );
	const inc_t    cs_c      = bli_obj_col_stride( c );

	void* restrict buf_alpha = bli_obj_buffer_for_1x1( dt, alpha );
	void* restrict buf_beta  = bli_obj_buffer_for_1x1( dt, beta );

	// Index into the function pointer array to extract the correct
	// typed function pointer based on the chosen datatype.
	FUNCPTR_T f = ftypes[dt];

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
	  cntx,
	  rntm,
	  thread
	);
}

//
// -- gemm-like block-panel algorithm (typed interface) ------------------------
//

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTECH2(bls_,ch,varname) \
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
       cntx_t* restrict cntx, \
       rntm_t* restrict rntm, \
       thrinfo_t* restrict thread  \
     ) \
{ \
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Query the context for various blocksizes. */ \
	const dim_t NR  = bli_cntx_get_blksz_def_dt( dt, BLIS_NR, cntx ); \
	const dim_t MR  = bli_cntx_get_blksz_def_dt( dt, BLIS_MR, cntx ); \
	const dim_t NC  = bli_cntx_get_blksz_def_dt( dt, BLIS_NC, cntx ); \
	const dim_t MC  = bli_cntx_get_blksz_def_dt( dt, BLIS_MC, cntx ); \
	const dim_t KC  = bli_cntx_get_blksz_def_dt( dt, BLIS_KC, cntx ); \
\
	/* Query the context for the microkernel address and cast it to its
	   function pointer type. */ \
	/*
	PASTECH(ch,gemm_ukr_ft) \
               gemm_ukr = bli_cntx_get_ukr_dt( dt, BLIS_GEMM_UKR, cntx ); \
	*/ \
\
	/* Temporary C buffer for edge cases. Note that the strides of this
	   temporary buffer are set so that they match the storage of the
	   original C matrix. For example, if C is column-stored, ct will be
	   column-stored as well. */ \
	/*
	ctype       ct[ BLIS_STACK_BUF_MAX_SIZE \
	                / sizeof( ctype ) ] \
	                __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
	const bool col_pref = bli_cntx_ukr_prefers_cols_dt( dt, BLIS_GEMM_UKR, cntx ); \
	const inc_t rs_ct   = ( col_pref ? 1 : NR ); \
	const inc_t cs_ct   = ( col_pref ? MR : 1 ); \
	*/ \
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
\
	const inc_t irstep_c = rs_c * MR; \
\
	ctype* restrict a_00       = a; \
	ctype* restrict b_00       = b; \
	ctype* restrict c_00       = c; \
	ctype* restrict alpha_cast = alpha; \
	ctype* restrict beta_cast  = beta; \
\
	/* Make local copies of the scalars to prevent any unnecessary sharing of
	   cache lines between the cores' caches. */ \
	ctype           alpha_local = *alpha_cast; \
	ctype           beta_local  = *beta_cast; \
	ctype           one_local   = *PASTEMAC(ch,1); \
	/*ctype           zero_local  = *PASTEMAC(ch,0);*/ \
\
	auxinfo_t       aux; \
\
	/* Initialize a mem_t entry for A and B. Strictly speaking, this is only
	   needed for the matrix we will be packing (if any), but we do it
	   unconditionally to be safe. */ \
	mem_t mem_a = BLIS_MEM_INITIALIZER; \
	mem_t mem_b = BLIS_MEM_INITIALIZER; \
\
	/* Define an array of bszid_t ids, which will act as our substitute for
	   the cntl_t tree. */ \
	bszid_t bszids[8] = { BLIS_NC,      /* 5th loop */ \
	                      BLIS_KC,      /* 4th loop */ \
	                      BLIS_NO_PART, /* pack B */ \
	                      BLIS_MC,      /* 3rd loop */ \
	                      BLIS_NO_PART, /* pack A */ \
	                      BLIS_NR,      /* 2nd loop */ \
	                      BLIS_MR,      /* 1st loop */ \
	                      BLIS_KR };    /* microkernel loop */  \
\
	bszid_t* restrict bszids_jc = &bszids[0]; \
	bszid_t* restrict bszids_pc = &bszids[1]; \
	/*bszid_t* restrict bszids_pb = &bszids[2];*/ \
	bszid_t* restrict bszids_ic = &bszids[3]; \
	/*bszid_t* restrict bszids_pa = &bszids[4];*/ \
	bszid_t* restrict bszids_jr = &bszids[5]; \
	/*bszid_t* restrict bszids_ir = &bszids[6];*/ \
\
	thrinfo_t* restrict thread_jc = NULL; \
	thrinfo_t* restrict thread_pc = NULL; \
	thrinfo_t* restrict thread_pb = NULL; \
	thrinfo_t* restrict thread_ic = NULL; \
	thrinfo_t* restrict thread_pa = NULL; \
	thrinfo_t* restrict thread_jr = NULL; \
	thrinfo_t* restrict thread_ir = NULL; \
\
	/* Identify the current thrinfo_t node and then grow the tree. */ \
	thread_jc = thread; \
	bli_thrinfo_sup_grow( rntm, bszids_jc, thread_jc ); \
\
	/* Compute the JC loop thread range for the current thread. */ \
	dim_t jc_start, jc_end; \
	bli_thread_range_sub( thread_jc, n, NR, FALSE, &jc_start, &jc_end ); \
	const dim_t n_local = jc_end - jc_start; \
\
	/* Compute number of primary and leftover components of the JC loop. */ \
	/*const dim_t jc_iter = ( n_local + NC - 1 ) / NC;*/ \
	const dim_t jc_left =   n_local % NC; \
\
	/* Loop over the n dimension (NC rows/columns at a time). */ \
	for ( dim_t jj = jc_start; jj < jc_end; jj += NC ) \
	{ \
		/* Calculate the thread's current JC block dimension. */ \
		const dim_t nc_cur = ( NC <= jc_end - jj ? NC : jc_left ); \
\
		ctype* restrict b_jc = b_00 + jj * jcstep_b; \
		ctype* restrict c_jc = c_00 + jj * jcstep_c; \
\
		/* Identify the current thrinfo_t node and then grow the tree. */ \
		thread_pc = bli_thrinfo_sub_node( thread_jc ); \
		bli_thrinfo_sup_grow( rntm, bszids_pc, thread_pc ); \
\
		/* Compute the PC loop thread range for the current thread. */ \
		const dim_t pc_start = 0, pc_end = k; \
		const dim_t k_local = k; \
\
		/* Compute number of primary and leftover components of the PC loop. */ \
		/*const dim_t pc_iter = ( k_local + KC - 1 ) / KC;*/ \
		const dim_t pc_left =   k_local % KC; \
\
		/* Loop over the k dimension (KC rows/columns at a time). */ \
		for ( dim_t pp = pc_start; pp < pc_end; pp += KC ) \
		{ \
			/* Calculate the thread's current PC block dimension. */ \
			const dim_t kc_cur = ( KC <= pc_end - pp ? KC : pc_left ); \
\
			ctype* restrict a_pc = a_00 + pp * pcstep_a; \
			ctype* restrict b_pc = b_jc + pp * pcstep_b; \
\
			/* Only apply beta to the first iteration of the pc loop. */ \
			ctype* restrict beta_use = ( pp == 0 ? &beta_local : &one_local ); \
\
			ctype* b_use; \
			inc_t  rs_b_use, cs_b_use, ps_b_use; \
\
			/* Identify the current thrinfo_t node. Note that the thrinfo_t
			   node will have already been created by a previous call to
			   bli_thrinfo_sup_grow() since bszid_t values of BLIS_NO_PART
			   cause the tree to grow by two (e.g. to the next bszid that is
			   a normal bszid_t value). */ \
			thread_pb = bli_thrinfo_sub_node( thread_pc ); \
			/*bli_thrinfo_sup_grow( rntm, bszids_pb, thread_pb );*/ \
\
			/* Determine the packing buffer and related parameters for matrix
			   B. Then call the packm implementation. */ \
			PASTECH2(bls_,ch,packm_b) \
			( \
			  conjb, \
			  KC,     NC, \
			  kc_cur, nc_cur, NR, \
			  &one_local, \
			  b_pc,   rs_b,      cs_b, \
			  &b_use, &rs_b_use, &cs_b_use, \
			                     &ps_b_use, \
			  cntx, \
			  rntm, \
			  &mem_b, \
			  thread_pb  \
			); \
\
			/* Alias b_use so that it's clear this is our current block of
			   matrix B. */ \
			ctype* restrict b_pc_use = b_use; \
\
			/* Identify the current thrinfo_t node and then grow the tree. */ \
			thread_ic = bli_thrinfo_sub_node( thread_pb ); \
			bli_thrinfo_sup_grow( rntm, bszids_ic, thread_ic ); \
\
			/* Compute the IC loop thread range for the current thread. */ \
			dim_t ic_start, ic_end; \
			bli_thread_range_sub( thread_ic, m, MR, FALSE, &ic_start, &ic_end ); \
			const dim_t m_local = ic_end - ic_start; \
\
			/* Compute number of primary and leftover components of the IC loop. */ \
			/*const dim_t ic_iter = ( m_local + MC - 1 ) / MC;*/ \
			const dim_t ic_left =   m_local % MC; \
\
			/* Loop over the m dimension (MC rows at a time). */ \
			for ( dim_t ii = ic_start; ii < ic_end; ii += MC ) \
			{ \
				/* Calculate the thread's current IC block dimension. */ \
				const dim_t mc_cur = ( MC <= ic_end - ii ? MC : ic_left ); \
\
				ctype* restrict a_ic = a_pc + ii * icstep_a; \
				ctype* restrict c_ic = c_jc + ii * icstep_c; \
\
				ctype* a_use; \
				inc_t  rs_a_use, cs_a_use, ps_a_use; \
\
				/* Identify the current thrinfo_t node. Note that the thrinfo_t
				   node will have already been created by a previous call to
				   bli_thrinfo_sup_grow() since bszid_t values of BLIS_NO_PART
				   cause the tree to grow by two (e.g. to the next bszid that is
				   a normal bszid_t value). */ \
				thread_pa = bli_thrinfo_sub_node( thread_ic ); \
				/*bli_thrinfo_sup_grow( rntm, bszids_pa, thread_pa );*/ \
\
				/* Determine the packing buffer and related parameters for matrix
				   A. Then call the packm implementation. */ \
				PASTECH2(bls_,ch,packm_a) \
				( \
				  conja, \
				  MC,     KC, \
				  mc_cur, kc_cur, MR, \
				  &one_local, \
				  a_ic,   rs_a,      cs_a, \
				  &a_use, &rs_a_use, &cs_a_use, \
				                     &ps_a_use, \
				  cntx, \
				  rntm, \
				  &mem_a, \
				  thread_pa  \
				); \
\
				/* Alias a_use so that it's clear this is our current block of
				   matrix A. */ \
				ctype* restrict a_ic_use = a_use; \
\
				/* Identify the current thrinfo_t node and then grow the tree. */ \
				thread_jr = bli_thrinfo_sub_node( thread_pa ); \
				bli_thrinfo_sup_grow( rntm, bszids_jr, thread_jr ); \
\
				/* Query the number of threads and thread ids for the JR loop.
				   NOTE: These values are only needed when computing the next
				   micropanel of B. */ \
				const dim_t jr_nt  = bli_thrinfo_n_way( thread_jr ); \
				const dim_t jr_tid = bli_thrinfo_work_id( thread_jr ); \
\
				/* Compute number of primary and leftover components of the JR loop. */ \
				dim_t jr_iter = ( nc_cur + NR - 1 ) / NR; \
				dim_t jr_left =   nc_cur % NR; \
\
				/* Compute the JR loop thread range for the current thread. */ \
				dim_t jr_start, jr_end; \
				bli_thread_range_sub( thread_jr, jr_iter, 1, FALSE, &jr_start, &jr_end ); \
\
				/* Loop over the n dimension (NR columns at a time). */ \
				for ( dim_t j = jr_start; j < jr_end; j += 1 ) \
				{ \
					const dim_t nr_cur \
					= ( bli_is_not_edge_f( j, jr_iter, jr_left ) ? NR : jr_left ); \
\
					ctype* restrict b_jr = b_pc_use + j * ps_b_use; \
					ctype* restrict c_jr = c_ic     + j * jrstep_c; \
\
					/* Assume for now that our next panel of B to be the current panel
					   of B. */ \
					ctype* restrict b2 = b_jr; \
\
					/* Identify the current thrinfo_t node. */ \
					thread_ir = bli_thrinfo_sub_node( thread_jr ); \
\
					/* Query the number of threads and thread ids for the IR loop.
					   NOTE: These values are only needed when computing the next
					   micropanel of A. */ \
					const dim_t ir_nt  = bli_thrinfo_n_way( thread_ir ); \
					const dim_t ir_tid = bli_thrinfo_work_id( thread_ir ); \
\
					/* Compute number of primary and leftover components of the IR loop. */ \
					dim_t ir_iter = ( mc_cur + MR - 1 ) / MR; \
					dim_t ir_left =   mc_cur % MR; \
\
					/* Compute the IR loop thread range for the current thread. */ \
					dim_t ir_start, ir_end; \
					bli_thread_range_sub( thread_ir, ir_iter, 1, FALSE, &ir_start, &ir_end ); \
\
					/* Loop over the m dimension (MR rows at a time). */ \
					for ( dim_t i = ir_start; i < ir_end; i += 1 ) \
					{ \
						const dim_t mr_cur \
						= ( bli_is_not_edge_f( i, ir_iter, ir_left ) ? MR : ir_left ); \
\
						ctype* restrict a_ir = a_ic_use + i * ps_a_use; \
						ctype* restrict c_ir = c_jr     + i * irstep_c; \
\
						ctype* restrict a2; \
\
						/* Compute the addresses of the next micropanels of A and B. */ \
						a2 = bli_gemm_get_next_a_upanel( a_ir, ps_a_use, 1 ); \
						if ( bli_is_last_iter( i, ir_end, ir_tid, ir_nt ) ) \
						{ \
							a2 = a_ic_use; \
							b2 = bli_gemm_get_next_b_upanel( b_jr, ps_b_use, 1 ); \
							if ( bli_is_last_iter( j, jr_end, jr_tid, jr_nt ) ) \
								b2 = b_pc_use; \
						} \
\
						/* Save the addresses of next micropanels of A and B to the
						   auxinfo_t object. */ \
						bli_auxinfo_set_next_a( a2, &aux ); \
						bli_auxinfo_set_next_b( b2, &aux ); \
\
						/* Call a wrapper to the kernel (which handles edge cases). */ \
						PASTECH2(bls_,ch,gemm_kernel) \
						( \
						  MR, \
						  NR, \
						  mr_cur, \
						  nr_cur, \
						  kc_cur, \
						  &alpha_local, \
						  a_ir, rs_a_use, cs_a_use, \
						  b_jr, rs_b_use, cs_b_use, \
						  beta_use, \
						  c_ir, rs_c,     cs_c, \
						  &aux, \
						  cntx  \
						); \
					} \
				} \
			} \
\
			/* This barrier is needed to prevent threads from starting to pack
			   the next row panel of B before the current row panel is fully
			   computed upon. */ \
			bli_thrinfo_barrier( thread_pb ); \
		} \
	} \
\
	/* Release any memory that was acquired for packing matrices A and B. */ \
	PASTECH2(bls_,ch,packm_finalize_mem_a) \
	( \
	  rntm, \
	  &mem_a, \
	  thread_pa  \
	); \
	PASTECH2(bls_,ch,packm_finalize_mem_b) \
	( \
	  rntm, \
	  &mem_b, \
	  thread_pb  \
	); \
\
/*
PASTEMAC(ch,fprintm)( stdout, "gemm_bp_var2: a1_packed", mr_cur, kc_cur, a_ir, rs_a_use, cs_a_use, "%5.2f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemm_bp_var2: b1_packed", kc_cur, nr_cur, b_jr, rs_b_use, cs_b_use, "%5.2f", "" ); \
PASTEMAC(ch,fprintm)( stdout, "gemm_bp_var2: c ", mr_cur, nr_cur, c_ir, rs_c, cs_c, "%5.2f", "" ); \
*/ \
}

//INSERT_GENTFUNC_BASIC0( gemm_bp_var2 )
GENTFUNC( float,    s, gemm_bp_var2 )
GENTFUNC( double,   d, gemm_bp_var2 )
GENTFUNC( scomplex, c, gemm_bp_var2 )
GENTFUNC( dcomplex, z, gemm_bp_var2 )

//
// -- gemm-like microkernel wrapper --------------------------------------------
//

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTECH2(bls_,ch,varname) \
     ( \
       const dim_t         MR, \
       const dim_t         NR, \
       dim_t               mr_cur, \
       dim_t               nr_cur, \
       dim_t               kc_cur, \
       ctype*     restrict alpha, \
       ctype*     restrict a, inc_t rs_a, inc_t cs_a, \
       ctype*     restrict b, inc_t rs_b, inc_t cs_b, \
       ctype*     restrict beta, \
       ctype*     restrict c, inc_t rs_c, inc_t cs_c, \
       auxinfo_t* restrict aux, \
       cntx_t*    restrict cntx  \
     ) \
{ \
	/* Infer the datatype from the ctype. */ \
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Query the context for the microkernel address and cast it to its
	   function pointer type. */ \
	PASTECH(ch,gemm_ukr_ft) \
               gemm_ukr = bli_cntx_get_ukr_dt( dt, BLIS_GEMM_UKR, cntx ); \
\
	/* Temporary C buffer for edge cases. Note that the strides of this
	   temporary buffer are set so that they match the storage of the
	   original C matrix. For example, if C is column-stored, ct will be
	   column-stored as well. */ \
	ctype       ct[ BLIS_STACK_BUF_MAX_SIZE \
	                / sizeof( ctype ) ] \
	                __attribute__((aligned(BLIS_STACK_BUF_ALIGN_SIZE))); \
	const bool col_pref = bli_cntx_ukr_prefers_cols_dt( dt, BLIS_GEMM_UKR, cntx ); \
	const inc_t rs_ct   = ( col_pref ? 1 : NR ); \
	const inc_t cs_ct   = ( col_pref ? MR : 1 ); \
\
	ctype       zero    = *PASTEMAC(ch,0); \
\
	/* Handle interior and edge cases separately. */ \
	if ( mr_cur == MR && nr_cur == NR ) \
	{ \
		/* Invoke the gemm microkernel. */ \
		gemm_ukr \
		( \
		  kc_cur, \
		  alpha, \
		  a, \
		  b, \
		  beta, \
		  c, rs_c, cs_c, \
		  aux, \
		  cntx  \
		); \
	} \
	else \
	{ \
		/* Invoke the gemm microkernel. */ \
		gemm_ukr \
		( \
		  kc_cur, \
		  alpha, \
		  a, \
		  b, \
		  &zero, \
		  ct, rs_ct, cs_ct, \
		  aux, \
		  cntx  \
		); \
\
		/* Scale the bottom edge of C and add the result from above. */ \
		PASTEMAC(ch,xpbys_mxn) \
		( \
		  mr_cur, \
		  nr_cur, \
		  ct, rs_ct, cs_ct, \
		  beta, \
		  c,  rs_c,  cs_c \
		); \
	} \
}

//INSERT_GENTFUNC_BASIC0( gemm_kernel )
GENTFUNC( float,    s, gemm_kernel )
GENTFUNC( double,   d, gemm_kernel )
GENTFUNC( scomplex, c, gemm_kernel )
GENTFUNC( dcomplex, z, gemm_kernel )

