/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018 - 2019, Advanced Micro Devices, Inc.

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

#ifndef BLIS_LANG_DEFS_H
#define BLIS_LANG_DEFS_H


// -- Undefine restrict for C++ and C89/90 --

#ifdef __cplusplus
  // Language is C++; define restrict as nothing.
  #ifndef restrict
  #define restrict
  #endif
#elif __STDC_VERSION__ >= 199901L
  // Language is C99 (or later); do nothing since restrict is recognized.
#else
  // Language is pre-C99; define restrict as nothing.
  #ifndef restrict
  #define restrict
  #endif
#endif


// -- Define typeof() operator if using non-GNU compiler --

#ifndef __GNUC__
  #define typeof __typeof__
#else
  #ifndef typeof
  #define typeof __typeof__
  #endif
#endif


// -- BLIS Thread Local Storage Keyword --

// __thread for TLS is supported by GCC, CLANG, ICC, and IBMC.
// There is a small risk here as __GNUC__ can also be defined by some other
// compiler (other than ICC and CLANG which we know define it) that
// doesn't support __thread, as __GNUC__ is not quite unique to GCC.
// But the possibility of someone using such non-main-stream compiler
// for building BLIS is low.
#if defined(BLIS_ENABLE_TLS) && ( defined(__GNUC__)  || \
                                  defined(__clang__) || \
                                  defined(__ICC)     || \
                                  defined(__IBMC__) )
  #define BLIS_THREAD_LOCAL __thread
#else
  #define BLIS_THREAD_LOCAL
#endif


// -- BLIS constructor/destructor function attribute --

// __attribute__((constructor/destructor)) is supported by GCC only.
// There is a small risk here as __GNUC__ can also be defined by some other
// compiler (other than ICC and CLANG which we know define it) that
// doesn't support this, as __GNUC__ is not quite unique to GCC.
// But the possibility of someone using such non-main-stream compiler
// for building BLIS is low.

#if defined(__ICC) || defined(__INTEL_COMPILER)
  // ICC defines __GNUC__ but doesn't support this
  #define BLIS_ATTRIB_CTOR
  #define BLIS_ATTRIB_DTOR
#elif defined(__clang__)
  // CLANG supports __attribute__, but its documentation doesn't
  // mention support for constructor/destructor. Compiling with
  // clang and testing shows that it does support.
  #define BLIS_ATTRIB_CTOR __attribute__((constructor))
  #define BLIS_ATTRIB_DTOR __attribute__((destructor))
#elif defined(__GNUC__)
  #define BLIS_ATTRIB_CTOR __attribute__((constructor))
  #define BLIS_ATTRIB_DTOR __attribute__((destructor))
#else
  #define BLIS_ATTRIB_CTOR
  #define BLIS_ATTRIB_DTOR
#endif


#endif
