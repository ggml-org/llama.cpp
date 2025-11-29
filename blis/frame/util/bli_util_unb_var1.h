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


//
// Prototype BLAS-like interfaces with typed operands.
//

#undef  GENTPROTR
#define GENTPROTR( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       ctype_r* asum, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     );

INSERT_GENTPROTR_BASIC( asumv_unb_var1 )


#undef  GENTPROT
#define GENTPROT( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       uplo_t  uploa, \
       dim_t   m, \
       ctype*  a, inc_t rs_a, inc_t cs_a, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     );

INSERT_GENTPROT_BASIC( mkherm_unb_var1 )
INSERT_GENTPROT_BASIC( mksymm_unb_var1 )
INSERT_GENTPROT_BASIC( mktrim_unb_var1 )


#undef  GENTPROTR
#define GENTPROTR( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       ctype_r* norm, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     );

INSERT_GENTPROTR_BASIC( norm1v_unb_var1 )
INSERT_GENTPROTR_BASIC( normfv_unb_var1 )
INSERT_GENTPROTR_BASIC( normiv_unb_var1 )


#undef  GENTPROTR
#define GENTPROTR( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       doff_t   diagoffx, \
       diag_t   diagx, \
       uplo_t   uplox, \
       dim_t    m, \
       dim_t    n, \
       ctype*   x, inc_t rs_x, inc_t cs_x, \
       ctype_r* norm, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     );

INSERT_GENTPROTR_BASIC( norm1m_unb_var1 )
INSERT_GENTPROTR_BASIC( normfm_unb_var1 )
INSERT_GENTPROTR_BASIC( normim_unb_var1 )


#undef  GENTPROT
#define GENTPROT( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t   n, \
       ctype*  x, inc_t incx, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     );

INSERT_GENTPROT_BASIC( randv_unb_var1 )
INSERT_GENTPROT_BASIC( randnv_unb_var1 )


#undef  GENTPROT
#define GENTPROT( ctype, ch, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       doff_t  diagoffx, \
       uplo_t  uplox, \
       dim_t   m, \
       dim_t   n, \
       ctype*  x, inc_t rs_x, inc_t cs_x, \
       cntx_t* cntx, \
       rntm_t* rntm  \
     );

INSERT_GENTPROT_BASIC( randm_unb_var1 )
INSERT_GENTPROT_BASIC( randnm_unb_var1 )


#undef  GENTPROTR
#define GENTPROTR( ctype, ctype_r, ch, chr, varname ) \
\
void PASTEMAC(ch,varname) \
     ( \
       dim_t    n, \
       ctype*   x, inc_t incx, \
       ctype_r* scale, \
       ctype_r* sumsq, \
       cntx_t*  cntx, \
       rntm_t*  rntm  \
     );

INSERT_GENTPROTR_BASIC( sumsqv_unb_var1 )

// -----------------------------------------------------------------------------

#undef  GENTPROT
#define GENTPROT( ctype, ch, varname ) \
\
bool PASTEMAC(ch,varname) \
     ( \
       conj_t conjx, \
       dim_t  n, \
       ctype* x, inc_t incx, \
       ctype* y, inc_t incy  \
     );

INSERT_GENTPROT_BASIC( eqv_unb_var1 )


#undef  GENTPROT
#define GENTPROT( ctype, ch, varname ) \
\
bool PASTEMAC(ch,varname) \
     ( \
       doff_t  diagoffx, \
       diag_t  diagx, \
       uplo_t  uplox, \
       trans_t transx, \
       dim_t   m, \
       dim_t   n, \
       ctype*  x, inc_t rs_x, inc_t cs_x, \
       ctype*  y, inc_t rs_y, inc_t cs_y  \
     );

INSERT_GENTPROT_BASIC( eqm_unb_var1 )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname) \
     ( \
             FILE*  file, \
       const char*  s1, \
             dim_t  n, \
       const ctype* x, inc_t incx, \
       const char*  format, \
       const char*  s2  \
     );

INSERT_GENTPROT_BASIC_I( fprintv )


#undef  GENTPROT
#define GENTPROT( ctype, ch, opname ) \
\
BLIS_EXPORT_BLIS void PASTEMAC(ch,opname) \
     ( \
             FILE*  file, \
       const char*  s1, \
             dim_t  m, \
             dim_t  n, \
       const ctype* x, inc_t rs_x, inc_t cs_x, \
       const char*  format, \
       const char*  s2  \
     );

INSERT_GENTPROT_BASIC_I( fprintm )


