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

// Given the current architecture of BLIS sandboxes, bli_gemm_ex() is the
// entry point to any sandbox implementation.

// NOTE: This function is implemented functionally identically to the
// function that it overrides in frame/3/bli_l3_oapi_ex.c. This means that
// we are forgoing the option of customizing the implementations that
// underlie bli_gemm() and bli_?gemm() (which both call bli_gemm_ex()).
// Any new code defined in this sandbox directory, however, will be
// included in the BLIS.

#include "blis.h"

void bli_gemm_ex
     (
       const obj_t*  alpha,
       const obj_t*  a,
       const obj_t*  b,
       const obj_t*  beta,
       const obj_t*  c,
       const cntx_t* cntx,
       const rntm_t* rntm
     )
{
	bli_init_once();

	// A switch to easily toggle whether we use the sandbox implementation
	// of bls_gemm() as the implementation for bli_gemm(). (This allows for
	// easy testing of bls_gemm() via the testsuite.) Changing the conditional
	// to "0" will cause bli_gemm()/bli_gemm_ex() to *not* call the local
	// sandbox implementation, though that implementation may still be called
	// directly.
	if ( 1 )
	{
		bls_gemm_ex
		(
		  ( obj_t* )alpha, ( obj_t* )a, ( obj_t* )b, ( obj_t* )beta, ( obj_t* )c,
		  ( cntx_t* )cntx, ( rntm_t* )rntm
		);
	}
	else
	{
		bli_gemm_def_ex
		(
		  ( obj_t* )alpha, ( obj_t* )a, ( obj_t* )b, ( obj_t* )beta, ( obj_t* )c,
		  ( cntx_t* )cntx, ( rntm_t* )rntm
		);
	}
}

