/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2020, Advanced Micro Devices, Inc.

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

#ifndef BLIS_BLAS_H
#define BLIS_BLAS_H


// If the CBLAS compatibility layer was enabled while the BLAS layer
// was not enabled, we must enable it here.
#if defined(BLIS_ENABLE_CBLAS) && !defined(BLIS_ENABLE_BLAS)
  #define BLIS_ENABLE_BLAS
#endif

// By default, if the BLAS compatibility layer is enabled, we define
// (include) all of the BLAS prototypes. However, if the user is
// #including "blis.h" and also #including another header that also
// declares the BLAS functions, then we provide an opportunity to
// #undefine the BLIS_ENABLE_BLAS_DEFS macro (see below).
#ifdef BLIS_ENABLE_BLAS
  #define BLIS_ENABLE_BLAS_DEFS
#else
  #undef  BLIS_ENABLE_BLAS_DEFS
#endif

// Skip prototyping all of the BLAS if the BLAS test drivers are being
// compiled.
#ifdef BLIS_VIA_BLASTEST
  #undef BLIS_ENABLE_BLAS_DEFS
#endif

// Skip prototyping all of the BLAS if the environment has defined the
// macro BLIS_DISABLE_BLAS_DEFS.
#ifdef BLIS_DISABLE_BLAS_DEFS
  #undef BLIS_ENABLE_BLAS_DEFS
#endif

// Begin including all BLAS prototypes, if appropriate.
#ifdef BLIS_ENABLE_BLAS_DEFS
  // If BLIS_ENABLE_BLAS_DEFS is defined, then we should #include the BLAS
  // prototypes.
  #include "bli_blas_defs.h"
#else
  // Even if BLAS prototypes are not to be #included into blis.h, we still
  // need to #include the prototypes when compiling BLIS.
  #ifdef BLIS_IS_BUILDING_LIBRARY
    #include "bli_blas_defs.h"
  #endif
#endif


#endif // BLIS_BLAS_H

