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

// BLIS GEMM function naming scheme
#define GEMM_FUNC_NAME_(ch)    bli_ ## ch ## gemm
#define GEMM_FUNC_NAME(ch)     GEMM_FUNC_NAME_(ch)

// BLIS GEMM function prototype macro
#define GEMM_FUNC_PROT(DTYPE_IN, DTYPE_OUT, ch) \
    void GEMM_FUNC_NAME(ch) \
        ( \
            trans_t transa, \
            trans_t transb, \
            dim_t   m, \
            dim_t   n, \
            dim_t   k, \
            DTYPE_OUT*  alpha, \
            DTYPE_IN*  a, inc_t rsa, inc_t csa, \
            DTYPE_IN*  b, inc_t rsb, inc_t csb, \
            DTYPE_OUT*  beta, \
            DTYPE_OUT*  c, inc_t rsc, inc_t csc \
        ) 

// Pack routine naming scheme
#define PACK_FUNC_NAME_(ch, mat) ch ## _pack_ ## mat
#define PACK_FUNC_NAME(ch, mat)  PACK_FUNC_NAME_(ch, mat)

// Pack routine prototype
#define PACK_MACRO_PROTO(ch, DTYPE_IN) \
\
void PACK_FUNC_NAME(ch, a) \
    (  \
        dim_t MR, \
        int m, int k, \
        DTYPE_IN* ap, int rs_a, int cs_a, \
        DTYPE_IN* apack \
    ); \
\
void PACK_FUNC_NAME(ch, b) \
    ( \
        dim_t NR, \
        int k, int n, \
        DTYPE_IN* bp, int rs_b, int cs_b, \
        DTYPE_IN* bpack \
    ); 
