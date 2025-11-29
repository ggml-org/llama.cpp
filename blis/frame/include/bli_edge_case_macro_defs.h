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

#ifndef BLIS_EDGE_CASE_MACRO_DEFS_H
#define BLIS_EDGE_CASE_MACRO_DEFS_H

//
// Macros for edge-case handling within gemm microkernels.
//

// -- Setup helper macros --

#define GEMM_UKR_SETUP_CT_PRE(ch,mr,nr,row_major,alignment) \
\
	const PASTEMAC(ch,ctype)* restrict _beta   = beta; \
	      PASTEMAC(ch,ctype)* restrict _c      = c; \
	const inc_t                        _rs_c   = rs_c; \
	const inc_t                        _cs_c   = cs_c; \
	      PASTEMAC(ch,ctype)           _ct[ BLIS_STACK_BUF_MAX_SIZE \
	                                        / sizeof( PASTEMAC(ch,ctype) ) \
	                                      ] __attribute__((aligned(alignment))); \
	const inc_t                        _rs_ct  = row_major ? nr :  1; \
	const inc_t                        _cs_ct  = row_major ?  1 : mr;

#define GEMM_UKR_SETUP_CT_POST(ch) \
\
	PASTEMAC(ch,ctype) _zero; \
	bli_tset0s( ch, _zero ); \
	\
	if ( _use_ct ) \
	{ \
		c = _ct; \
		rs_c = _rs_ct; \
		cs_c = _cs_ct; \
		beta = &_zero; \
	}

// -- Setup macros --

#define GEMM_UKR_SETUP_CT(ch,mr,nr,row_major) \
\
	/* Scenario 1: the ukernel contains assembly-level support only for its
	   IO preference (e.g. only row-oriented or only column-oriented IO).
	   Use a temporary microtile for the other two cases as well as edge
	   cases. */ \
	GEMM_UKR_SETUP_CT_PRE(ch,mr,nr,row_major,1); \
	const bool _use_ct = ( row_major ? cs_c != 1 : rs_c != 1 ) || \
	                     m != mr || n != nr; \
	GEMM_UKR_SETUP_CT_POST(ch);

#define GEMM_UKR_SETUP_CT_AMBI(ch,mr,nr,row_major) \
\
	/* Scenario 2: the ukernel contains assembly-level support for its IO
	   preference as well as its opposite via in-register transpose
	   (e.g. both row- and column-oriented IO). Use a temporary microtile
	   for the general stride case as well as edge cases. */ \
	GEMM_UKR_SETUP_CT_PRE(ch,mr,nr,row_major,1); \
	const bool _use_ct = ( cs_c != 1 && rs_c != 1 ) || \
	                     m != mr || n != nr; \
	GEMM_UKR_SETUP_CT_POST(ch);

#define GEMM_UKR_SETUP_CT_ANY(ch,mr,nr,row_major) \
\
	/* Scenario 3: Similar to (2) where the assembly region also supports
	   general stride I0. Use a temporary microtile only for edge cases. */ \
	GEMM_UKR_SETUP_CT_PRE(ch,mr,nr,row_major,1); \
	const bool _use_ct = ( m != mr || n != nr ); \
	GEMM_UKR_SETUP_CT_POST(ch);

#define GEMM_UKR_SETUP_CT_ALIGNED(ch,mr,nr,row_major,alignment) \
\
	/* Scenario 4: Similar to (1), but uses temporary microtile to handle
	   cases where the pointer to the C microtile is not aligned. */ \
	GEMM_UKR_SETUP_CT_PRE(ch,mr,nr,row_major,alignment); \
	const bool _use_ct = ( row_major ? cs_c != 1 : rs_c != 1 ) || \
	                     m != mr || n != nr || \
	                     ( (uintptr_t)_c % alignment ) || \
	                     ( ( ( row_major ? _rs_c : _cs_c )*sizeof( PASTEMAC(ch,ctype) ) ) % alignment ); \
	GEMM_UKR_SETUP_CT_POST(ch);

// -- Flush macros --

#define GEMM_UKR_FLUSH_CT(ch) \
\
	/* If we actually used the temporary microtile, accumulate it to the output
	   microtile. */ \
	if ( _use_ct ) \
	{ \
		bli_txpbys_mxn \
		( \
		  ch,ch,ch,ch, \
		  m, n, \
		  _ct, _rs_ct, _cs_ct, \
		  _beta, \
		  _c,  _rs_c,  _cs_c \
		); \
	} \


//
// Macros for edge-case handling within gemmtrsm microkernels.
//

// -- Setup helper macros --

#define GEMMTRSM_UKR_SETUP_CT_PRE(ch,mr,nr,row_major,alignment) \
\
	      PASTEMAC(ch,ctype)* restrict _c      = c11; \
	const inc_t                        _rs_c   = rs_c; \
	const inc_t                        _cs_c   = cs_c; \
	      PASTEMAC(ch,ctype)           _ct[ BLIS_STACK_BUF_MAX_SIZE \
	                                        / sizeof( PASTEMAC(ch,ctype) ) \
	                                      ] __attribute__((aligned(alignment))); \
	const inc_t                        _rs_ct  = row_major ? nr :  1; \
	const inc_t                        _cs_ct  = row_major ?  1 : mr;

#define GEMMTRSM_UKR_SETUP_CT_POST(ch) \
\
	if ( _use_ct ) \
	{ \
		c11 = _ct; \
		rs_c = _rs_ct; \
		cs_c = _cs_ct; \
	}

// -- Setup macros --

#define GEMMTRSM_UKR_SETUP_CT(ch,mr,nr,row_major) \
\
	/* Scenario 1: the ukernel contains assembly-level support only for its
	   IO preference (e.g. only row-oriented or only column-oriented IO).
	   Use a temporary microtile for the other two cases as well as edge
	   cases. */ \
	GEMMTRSM_UKR_SETUP_CT_PRE(ch,mr,nr,row_major,1); \
	const bool _use_ct = ( row_major ? cs_c != 1 : rs_c != 1 ) || \
	                     m != mr || n != nr; \
	GEMMTRSM_UKR_SETUP_CT_POST(ch);

#define GEMMTRSM_UKR_SETUP_CT_AMBI(ch,mr,nr,row_major) \
\
	/* Scenario 2: the ukernel contains assembly-level support for its IO
	   preference as well as its opposite via in-register transpose
	   (e.g. both row- and column-oriented IO). Use a temporary microtile
	   for the general stride case as well as edge cases. */ \
	GEMMTRSM_UKR_SETUP_CT_PRE(ch,mr,nr,row_major,1); \
	const bool _use_ct = ( cs_c != 1 && rs_c != 1 ) || \
	                     m != mr || n != nr; \
	GEMMTRSM_UKR_SETUP_CT_POST(ch);

#define GEMMTRSM_UKR_SETUP_CT_ANY(ch,mr,nr,row_major) \
\
	/* Scenario 3: Similar to (2) where the assembly region also supports
	   general stride I0. Use a temporary microtile only for edge cases. */ \
	GEMMTRSM_UKR_SETUP_CT_PRE(ch,mr,nr,row_major,1); \
	const bool _use_ct = ( m != mr || n != nr ); \
	GEMMTRSM_UKR_SETUP_CT_POST(ch);

#define GEMMTRSM_UKR_SETUP_CT_ALIGNED(ch,mr,nr,row_major,alignment) \
\
	/* Scenario 4: Similar to (1), but uses temporary microtile to handle
	   cases where the pointer to the C microtile is not aligned. */ \
	GEMMTRSM_UKR_SETUP_CT_PRE(ch,mr,nr,row_major,alignment); \
	const bool _use_ct = ( row_major ? cs_c != 1 : rs_c != 1 ) || \
	                     m != mr || n != nr || \
	                     ( (uintptr_t)_c % alignment ) || \
	                     ( ( ( row_major ? _rs_c : _cs_c )*sizeof( PASTEMAC(ch,ctype) ) ) % alignment ); \
	GEMMTRSM_UKR_SETUP_CT_POST(ch);

// -- Flush macros --

#define GEMMTRSM_UKR_FLUSH_CT(ch) \
\
	/* If we actually used the temporary microtile, use it to overwrite the
	   output microtile. Used by trsm. */ \
	if ( _use_ct ) \
	{ \
		PASTEMAC(t,copys_mxn) \
		( \
		  ch,ch, \
		  m, n, \
		  _ct, _rs_ct, _cs_ct, \
		  _c,  _rs_c,  _cs_c \
		); \
	} \


#endif

