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

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTECH2(bao_,ch,opname) \
     ( \
       dim_t            k, \
       dim_t            n, \
       dim_t            nr, \
       cntx_t* restrict cntx, \
       rntm_t* restrict rntm, \
       mem_t*  restrict mem, \
       thrinfo_t* restrict thread  \
     ) \
{ \
	/* Set the pack buffer type so that we are obtaining memory blocks from
	   the pool dedicated to panels of B. */ \
	const packbuf_t pack_buf_type = BLIS_BUFFER_FOR_B_PANEL; \
\
	/* NOTE: This "rounding up" of the last upanel is absolutely necessary since
	   we NEED that last micropanel to have the same ldim (cs_p) as the other
	   micropanels. Why? Because the microkernel assumes that the register (MR,
	   NR) AND storage (PACKMR, PACKNR) blocksizes do not change. */ \
	const dim_t k_pack = k; \
	const dim_t n_pack = ( n / nr + ( n % nr ? 1 : 0 ) ) * nr; \
\
	/* Barrier to make sure all threads are caught up and ready to begin the
	   packm stage. */ \
	bli_thrinfo_barrier( thread ); \
\
	/* Compute the size of the memory block eneded. */ \
	siz_t size_needed = sizeof( ctype ) * k_pack * n_pack; \
\
	/* Check the mem_t entry provided by the caller. If it is unallocated,
	   then we need to acquire a block from the packed block allocator. */ \
	if ( bli_mem_is_unalloc( mem ) ) \
	{ \
		if ( bli_thread_am_ochief( thread ) ) \
		{ \
			/* Acquire directly to the chief thread's mem_t that was passed in.
			   It needs to be that mem_t struct, and not a local (temporary)
			   mem_t, since there is no barrier until after packing is finished,
			   which could allow a race condition whereby the chief thread exits
			   the current function before the other threads have a chance to
			   copy from it. (A barrier would fix that race condition, but then
			   again, I prefer to keep barriers to a minimum.) */ \
			bli_pba_acquire_m \
			( \
			  rntm, \
			  size_needed, \
			  pack_buf_type, \
			  mem  \
			); \
		} \
\
		/* Broadcast the address of the chief thread's passed-in mem_t to all
		   threads. */ \
		mem_t* mem_p = bli_thrinfo_broadcast( thread, mem ); \
\
		/* Non-chief threads: Copy the contents of the chief thread's
		   passed-in mem_t to the passed-in mem_t for this thread. (The
		   chief thread already has the mem_t, so it does not need to
		   perform any copy.) */ \
		if ( !bli_thread_am_ochief( thread ) ) \
		{ \
			*mem = *mem_p; \
		} \
	} \
	else /* if ( bli_mem_is_alloc( mem ) ) */ \
	{ \
		/* If the mem_t entry provided by the caller does NOT contain a NULL
		   buffer, then a block has already been acquired from the packed
		   block allocator and cached by the caller. */ \
\
		/* As a sanity check, we should make sure that the mem_t object isn't
		   associated with a block that is too small compared to the size of
		   the packed matrix buffer that is needed, according to the value
		   computed above. */ \
		siz_t mem_size = bli_mem_size( mem ); \
\
		if ( mem_size < size_needed ) \
		{ \
			if ( bli_thread_am_ochief( thread ) ) \
			{ \
				/* The chief thread releases the existing block associated
				   with the mem_t, and then re-acquires a new block, saving
				   the associated mem_t to its passed-in mem_t. (See coment
				   above for why the acquisition needs to be directly to
				   the chief thread's passed-in mem_t and not a local
				   (temporary) mem_t. */ \
				bli_pba_release \
				( \
				  rntm, \
				  mem \
				); \
				bli_pba_acquire_m \
				( \
				  rntm, \
				  size_needed, \
				  pack_buf_type, \
				  mem \
				); \
			} \
\
			/* Broadcast the address of the chief thread's passed-in mem_t
			   to all threads. */ \
			mem_t* mem_p = bli_thrinfo_broadcast( thread, mem ); \
\
			/* Non-chief threads: Copy the contents of the chief thread's
			   passed-in mem_t to the passed-in mem_t for this thread. (The
			   chief thread already has the mem_t, so it does not need to
			   perform any copy.) */ \
			if ( !bli_thread_am_ochief( thread ) ) \
			{ \
				*mem = *mem_p; \
			} \
		} \
		else \
		{ \
			/* If the mem_t entry is already allocated and sufficiently large,
			   then we use it as-is. No action is needed. */ \
		} \
	} \
}

//INSERT_GENTFUNC_BASIC0( packm_init_mem_b )
GENTFUNC( float,    s, packm_init_mem_b )
GENTFUNC( double,   d, packm_init_mem_b )
GENTFUNC( scomplex, c, packm_init_mem_b )
GENTFUNC( dcomplex, z, packm_init_mem_b )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTECH2(bao_,ch,opname) \
     ( \
       rntm_t* restrict rntm, \
       mem_t*  restrict mem, \
       thrinfo_t* restrict thread  \
     ) \
{ \
	if ( thread != NULL ) \
	if ( bli_thread_am_ochief( thread ) ) \
	{ \
		/* Check the mem_t entry provided by the caller. Only proceed if it
		   is allocated, which it should be. */ \
		if ( bli_mem_is_alloc( mem ) ) \
		{ \
			bli_pba_release \
			( \
			  rntm, \
			  mem \
			); \
		} \
	} \
}

//INSERT_GENTFUNC_BASIC0( packm_finalize_mem_b )
GENTFUNC( float,    s, packm_finalize_mem_b )
GENTFUNC( double,   d, packm_finalize_mem_b )
GENTFUNC( scomplex, c, packm_finalize_mem_b )
GENTFUNC( dcomplex, z, packm_finalize_mem_b )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTECH2(bao_,ch,opname) \
     ( \
       pack_t* restrict schema, \
       dim_t            k, \
       dim_t            n, \
       dim_t            nr, \
       dim_t*  restrict k_max, \
       dim_t*  restrict n_max, \
       ctype**          p, inc_t* restrict rs_p, inc_t* restrict cs_p, \
                           dim_t* restrict pd_p, inc_t* restrict ps_p, \
       mem_t*  restrict mem  \
     ) \
{ \
	/* NOTE: This "rounding up" of the last upanel is absolutely necessary since
	   we NEED that last micropanel to have the same ldim (cs_p) as the other
	   micropanels. Why? Because the microkernel assumes that the register (MR,
	   NR) AND storage (PACKMR, PACKNR) blocksizes do not change. */ \
	*k_max = k; \
	*n_max = ( n / nr + ( n % nr ? 1 : 0 ) ) * nr; \
\
	/* Determine the dimensions and strides for the packed matrix B. */ \
	{ \
		/* Pack B to row-stored column-panels. */ \
		*rs_p = nr; \
		*cs_p = 1; \
\
		*pd_p = nr; \
		*ps_p = k * nr; \
\
		/* Set the schema to "packed column panels" to indicate packing to
		   conventional row-stored column panels. */ \
		*schema = BLIS_PACKED_COL_PANELS; \
	} \
\
	/* Set the buffer address provided by the caller to point to the memory
	   associated with the mem_t entry acquired from the memory pool. */ \
	*p = bli_mem_buffer( mem ); \
}

//INSERT_GENTFUNC_BASIC0( packm_init_b )
GENTFUNC( float,    s, packm_init_b )
GENTFUNC( double,   d, packm_init_b )
GENTFUNC( scomplex, c, packm_init_b )
GENTFUNC( dcomplex, z, packm_init_b )


//
// Define BLAS-like interfaces to the variant chooser.
//

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname ) \
\
void PASTECH2(bao_,ch,opname) \
     ( \
       conj_t           conj, \
       dim_t            k_alloc, \
       dim_t            n_alloc, \
       dim_t            k, \
       dim_t            n, \
       dim_t            nr, \
       ctype*  restrict kappa, \
       ctype*  restrict d, inc_t           incd, \
       ctype*  restrict b, inc_t           rs_b, inc_t           cs_b, \
       ctype** restrict p, inc_t* restrict rs_p, inc_t* restrict cs_p, \
                                                 inc_t* restrict ps_p, \
       cntx_t* restrict cntx, \
       rntm_t* restrict rntm, \
       mem_t*  restrict mem, \
       thrinfo_t* restrict thread  \
     ) \
{ \
	pack_t schema; \
	dim_t  k_max; \
	dim_t  n_max; \
	dim_t  pd_p; \
\
	/* Prepare the packing destination buffer. */ \
	PASTECH2(bao_,ch,packm_init_mem_b) \
	( \
	  k_alloc, n_alloc, nr, \
	  cntx, \
	  rntm, \
	  mem, \
	  thread  \
	); \
\
	/* Determine the packing buffer and related parameters for matrix B. */ \
	PASTECH2(bao_,ch,packm_init_b) \
	( \
	  &schema, \
	  k, n, nr, \
	  &k_max, &n_max, \
	  p, rs_p,  cs_p, \
	     &pd_p, ps_p, \
	  mem  \
	); \
\
	/* Pack matrix B to the destination buffer chosen above. Here, the packed
	   matrix is stored to row-stored k x NR micropanels. */ \
	PASTECH2(bao_,ch,packm_var1) \
	( \
	  conj, \
	  schema, \
	  k, \
	  n, \
	  k_max, \
	  n_max, \
	  kappa, \
	  d,  incd, \
	  b,  rs_b,  cs_b, \
	  *p, *rs_p, *cs_p, \
	       pd_p, *ps_p, \
	  cntx, \
	  thread  \
	); \
\
	/* Barrier so that packing is done before computation. */ \
	bli_thrinfo_barrier( thread ); \
}

//INSERT_GENTFUNC_BASIC0( packm_b )
GENTFUNC( float,    s, packm_b )
GENTFUNC( double,   d, packm_b )
GENTFUNC( scomplex, c, packm_b )
GENTFUNC( dcomplex, z, packm_b )

