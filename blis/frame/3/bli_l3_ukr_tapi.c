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

#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, tname, kerid ) \
\
void PASTEMAC(ch,opname) \
     ( \
             dim_t      m, \
             dim_t      n, \
             dim_t      k, \
       const ctype*     alpha, \
       const ctype*     a, \
       const ctype*     b, \
       const ctype*     beta, \
             ctype*     c, inc_t rs_c, inc_t cs_c, \
       const auxinfo_t* data, \
       const cntx_t*    cntx  \
     ) \
{ \
	bli_init_once(); \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Query the context for the function address of the current
	   datatype's micro-kernel. */ \
	PASTECH(tname,_ukr_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	/* Invoke the typed function for the given datatype. */ \
	f \
	( \
	  m, \
	  n, \
	  k, \
	  alpha, \
	  a, \
	  b, \
	  beta, \
	  c, rs_c, cs_c, \
	  data, \
	  cntx  \
	); \
} \

INSERT_GENTFUNC_BASIC( gemm_ukernel, gemm, BLIS_GEMM_UKR )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, tname, kerid ) \
\
void PASTEMAC(ch,opname) \
     ( \
             dim_t      m, \
             dim_t      n, \
             dim_t      k, \
       const ctype*     alpha, \
       const ctype*     a1x, \
       const ctype*     a11, \
       const ctype*     bx1, \
             ctype*     b11, \
             ctype*     c11, inc_t rs_c, inc_t cs_c, \
       const auxinfo_t* data, \
       const cntx_t*    cntx  \
     ) \
{ \
	bli_init_once(); \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Query the context for the function address of the current
	   datatype's micro-kernel. */ \
	PASTECH(tname,_ukr_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	/* Invoke the typed function for the given datatype. */ \
	f \
	( \
	  m, \
	  n, \
	  k, \
	  alpha, \
	  a1x, \
	  a11, \
	  bx1, \
	  b11, \
	  c11, rs_c, cs_c, \
	  data, \
	  cntx  \
	); \
} \

INSERT_GENTFUNC_BASIC( gemmtrsm_l_ukernel, gemmtrsm, BLIS_GEMMTRSM_L_UKR )
INSERT_GENTFUNC_BASIC( gemmtrsm_u_ukernel, gemmtrsm, BLIS_GEMMTRSM_U_UKR )


#undef  GENTFUNC
#define GENTFUNC( ctype, ch, opname, tname, kerid ) \
\
void PASTEMAC(ch,opname) \
     ( \
       const ctype*     a, \
             ctype*     b, \
             ctype*     c, inc_t rs_c, inc_t cs_c, \
       const auxinfo_t* data, \
       const cntx_t*    cntx  \
     ) \
{ \
	bli_init_once(); \
\
	const num_t dt = PASTEMAC(ch,type); \
\
	/* Query the context for the function address of the current
	   datatype's micro-kernel. */ \
	PASTECH(tname,_ukr_ft) f = bli_cntx_get_ukr_dt( dt, kerid, cntx ); \
\
	/* Invoke the typed function for the given datatype. */ \
	f \
	( \
	  a, \
	  b, \
	  c, rs_c, cs_c, \
	  data, \
	  cntx  \
	); \
} \

INSERT_GENTFUNC_BASIC( trsm_l_ukernel, trsm, BLIS_TRSM_L_UKR )
INSERT_GENTFUNC_BASIC( trsm_u_ukernel, trsm, BLIS_TRSM_U_UKR )

