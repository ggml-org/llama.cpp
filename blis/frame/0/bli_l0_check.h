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

#undef  GENTPROT
#define GENTPROT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* chi, \
       const obj_t* psi  \
     );

GENTPROT( addsc )
GENTPROT( copysc )
GENTPROT( divsc )
GENTPROT( mulsc )
GENTPROT( sqrtsc )
GENTPROT( sqrtrsc )
GENTPROT( subsc )
GENTPROT( invertsc )


#undef  GENTPROT
#define GENTPROT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* chi, \
       const obj_t* absq  \
     );

GENTPROT( absqsc )
GENTPROT( normfsc )


#undef  GENTPROT
#define GENTPROT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t*  chi, \
       const double* zeta_r, \
       const double* zeta_i  \
     );

GENTPROT( getsc )


#undef  GENTPROT
#define GENTPROT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
             double zeta_r, \
             double zeta_i, \
       const obj_t* chi  \
     );

GENTPROT( setsc )


#undef  GENTPROT
#define GENTPROT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* chi, \
       const obj_t* zeta_r, \
       const obj_t* zeta_i  \
     );

GENTPROT( unzipsc )


#undef  GENTPROT
#define GENTPROT( opname ) \
\
void PASTEMAC(opname,_check) \
     ( \
       const obj_t* zeta_r, \
       const obj_t* zeta_i, \
       const obj_t* chi  \
     );

GENTPROT( zipsc )

// -----------------------------------------------------------------------------

void bli_l0_xsc_check
     (
       const obj_t* chi
     );

void bli_l0_xxsc_check
     (
       const obj_t* chi,
       const obj_t* psi
     );

void bli_l0_xx2sc_check
     (
       const obj_t* chi,
       const obj_t* norm
     );

void bli_l0_xxbsc_check
     (
       const obj_t* chi,
       const obj_t* psi,
       const bool*  is
     );
