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
// Prototype object-based check functions.
//

#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* x, \
       const obj_t* asum  \
     );

GENPROT( asumv )


#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* x  \
     );

GENPROT( mkherm )
GENPROT( mksymm )
GENPROT( mktrim )


#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* x, \
       const obj_t* norm  \
     );

GENPROT( norm1v )
GENPROT( normfv )
GENPROT( normiv )


#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* x, \
       const obj_t* norm  \
     );

GENPROT( norm1m )
GENPROT( normfm )
GENPROT( normim )


#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* x  \
     );

GENPROT( randv )
GENPROT( randnv )
GENPROT( randm )
GENPROT( randnm )


#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* x, \
       const obj_t* scale, \
       const obj_t* sumsq  \
     );

GENPROT( sumsqv )

// -----------------------------------------------------------------------------

#undef  GENTPROT
#define GENTPROT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* chi, \
       const obj_t* psi, \
       const bool*  is  \
     );

GENTPROT( eqsc )
GENTPROT( ltsc )
GENTPROT( lesc )
GENTPROT( gtsc )
GENTPROT( gesc )


#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* x, \
       const obj_t* y, \
       const bool*  is  \
    );

GENPROT( eqv )
GENPROT( eqm )


#undef  GENPROT
#define GENPROT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const FILE*  file, \
       const char*  s1, \
       const obj_t* x, \
       const char*  format, \
       const char*  s2  \
     );

GENPROT( fprintv )
GENPROT( fprintm )

// -----------------------------------------------------------------------------

void bli_utilv_xi_check
     (
       const obj_t* x,
       const obj_t* index
     );

void bli_utilv_xa_check
     (
       const obj_t* x,
       const obj_t* asum
     );

void bli_utilm_mkhst_check
     (
       const obj_t* a
     );

void bli_utilv_norm_check
     (
       const obj_t* x,
       const obj_t* norm
     );

void bli_utilm_norm_check
     (
       const obj_t* x,
       const obj_t* norm
     );

void bli_utilm_fprint_check
     (
       const FILE*  file,
       const char*  s1,
       const obj_t* x,
       const char*  format,
       const char*  s2
     );

void bli_utilm_rand_check
     (
       const obj_t* x
     );

void bli_utilv_sumsqv_check
     (
       const obj_t* x,
       const obj_t* scale,
       const obj_t* sumsq
     );

