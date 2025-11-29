/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

#define FUNCPTR_T unpackv_fp

typedef void (*FUNCPTR_T)(
                           dim_t   m,
                           void*   p, inc_t incp,
                           void*   c, inc_t incc,
                           cntx_t* cntx
                         );

static FUNCPTR_T GENARRAY(ftypes,unpackv_unb_var1);


void bli_unpackv_unb_var1( obj_t*     p,
                           obj_t*     c,
                           cntx_t*    cntx,
                           unpackv_t* cntl )
{
	num_t     dt_pc     = bli_obj_dt( p );

	dim_t     dim_c     = bli_obj_vector_dim( c );

	void*     buf_p     = bli_obj_buffer_at_off( p );
	inc_t     incp      = bli_obj_vector_inc( p );

	void*     buf_c     = bli_obj_buffer_at_off( c );
	inc_t     incc      = bli_obj_vector_inc( c );

	FUNCPTR_T f;

	// Index into the type combination array to extract the correct
	// function pointer.
	f = ftypes[dt_pc];

	// Invoke the function.
	f
	(
	  dim_c,
	  buf_p, incp,
	  buf_c, incc,
	  cntx
	);
}


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t   m, \
       void*   p, inc_t incp, \
       void*   c, inc_t incc, \
       cntx_t* cntx  \
     ) \
{ \
	const num_t dt  = PASTEMAC(ch,type); \
\
	PASTECH(ch,copyv_ft) copyv_p = bli_cntx_get_ukr_dt( dt, BLIS_COPYV_KER, cntx ); \
\
	copyv_p \
	( \
	  BLIS_NO_CONJUGATE, \
	  m, \
	  p, incp, \
	  c, incc, \
	  cntx  \
	); \
}

INSERT_GENTFUNC_BASIC0( unpackv_unb_var1 )

