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

//
// Prototype BLAS-like interfaces to the variants.
//

#undef  GENTPROT
#define GENTPROT( ctype, ch, varname ) \
\
void PASTECH2(bao_,ch,varname) \
     ( \
       trans_t          transc, \
       pack_t           schema, \
       dim_t            m, \
       dim_t            n, \
       dim_t            m_max, \
       dim_t            n_max, \
       ctype*  restrict kappa, \
       ctype*  restrict d, inc_t incd, \
       ctype*  restrict c, inc_t rs_c, inc_t cs_c, \
       ctype*  restrict p, inc_t rs_p, inc_t cs_p, \
                           dim_t pd_p, inc_t ps_p, \
       cntx_t* restrict cntx, \
       thrinfo_t* restrict thread  \
     );

//INSERT_GENTPROT_BASIC0( packm_var1 )
GENTPROT( float,    s, packm_var1 )
GENTPROT( double,   d, packm_var1 )
GENTPROT( scomplex, c, packm_var1 )
GENTPROT( dcomplex, z, packm_var1 )

//INSERT_GENTPROT_BASIC0( packm_var2 )
GENTPROT( float,    s, packm_var2 )
GENTPROT( double,   d, packm_var2 )
GENTPROT( scomplex, c, packm_var2 )
GENTPROT( dcomplex, z, packm_var2 )
