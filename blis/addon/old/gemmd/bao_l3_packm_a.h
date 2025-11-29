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

#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
void PASTECH2(bao_,ch,opname) \
     ( \
       dim_t            m, \
       dim_t            k, \
       dim_t            mr, \
       cntx_t* restrict cntx, \
       rntm_t* restrict rntm, \
       mem_t*  restrict mem, \
       thrinfo_t* restrict thread  \
     ); \

//INSERT_GENTPROT_BASIC0( packm_init_mem_a )
GENTPROT( float,    s, packm_init_mem_a )
GENTPROT( double,   d, packm_init_mem_a )
GENTPROT( scomplex, c, packm_init_mem_a )
GENTPROT( dcomplex, z, packm_init_mem_a )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
void PASTECH2(bao_,ch,opname) \
     ( \
       rntm_t* restrict rntm, \
       mem_t*  restrict mem, \
       thrinfo_t* restrict thread  \
     ); \

//INSERT_GENTPROT_BASIC0( packm_finalize_mem_a )
GENTPROT( float,    s, packm_finalize_mem_a )
GENTPROT( double,   d, packm_finalize_mem_a )
GENTPROT( scomplex, c, packm_finalize_mem_a )
GENTPROT( dcomplex, z, packm_finalize_mem_a )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
void PASTECH2(bao_,ch,opname) \
     ( \
       pack_t* restrict schema, \
       dim_t            m, \
       dim_t            k, \
       dim_t            mr, \
       dim_t*  restrict m_max, \
       dim_t*  restrict k_max, \
       ctype**          p, inc_t* restrict rs_p, inc_t* restrict cs_p, \
                           dim_t* restrict pd_p, inc_t* restrict ps_p, \
       mem_t*  restrict mem  \
     ); \

//INSERT_GENTPROT_BASIC0( packm_init_a )
GENTPROT( float,    s, packm_init_a )
GENTPROT( double,   d, packm_init_a )
GENTPROT( scomplex, c, packm_init_a )
GENTPROT( dcomplex, z, packm_init_a )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
void PASTECH2(bao_,ch,opname) \
     ( \
       conj_t           conj, \
       dim_t            m_alloc, \
       dim_t            k_alloc, \
       dim_t            m, \
       dim_t            k, \
       dim_t            mr, \
       ctype*  restrict kappa, \
       ctype*  restrict d, inc_t           incd, \
       ctype*  restrict a, inc_t           rs_a, inc_t           cs_a, \
       ctype** restrict p, inc_t* restrict rs_p, inc_t* restrict cs_p, \
                                                 inc_t* restrict ps_p, \
       cntx_t* restrict cntx, \
       rntm_t* restrict rntm, \
       mem_t*  restrict mem, \
       thrinfo_t* restrict thread  \
     ); \

//INSERT_GENTPROT_BASIC0( packm_a )
GENTPROT( float,    s, packm_a )
GENTPROT( double,   d, packm_a )
GENTPROT( scomplex, c, packm_a )
GENTPROT( dcomplex, z, packm_a )

