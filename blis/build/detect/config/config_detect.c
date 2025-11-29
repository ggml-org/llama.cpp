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

// NOTE: This file will likely only ever get compiled as part of the BLIS
// configure script, and therefore BLIS_CONFIGURETIME_CPUID is guaranteed to
// be #defined. However, we preserve the cpp conditional for consistency with
// the other three files mentioned above.
#ifdef BLIS_CONFIGURETIME_CPUID

  // NOTE: If you need to make any changes to this cpp branch, it's probably
  // the case that you also need to modify bli_arch.c, bli_cpuid.c, and
  // bli_env.c. Don't forget to update these other files as needed!

  // The BLIS_ENABLE_SYSTEM macro must be defined so that the correct cpp
  // branch in bli_system.h is processed. (This macro is normally defined in
  // bli_config.h.)
  #define BLIS_ENABLE_SYSTEM

  // Use C-style static inline functions for any static inline functions that
  // happen to be defined by the headers below. (This macro is normally defined
  // in bli_config_macro_defs.h.)
  #define BLIS_INLINE static

  // Since we're not building a shared library, we can forgo the use of the
  // BLIS_EXPORT_BLIS annotations by #defining them to be nothing. (This macro
  // is normally defined in bli_config_macro_defs.h.)
  #define BLIS_EXPORT_BLIS

  #include "bli_system.h"
  #include "bli_type_defs.h"
  #include "bli_arch.h"
  #include "bli_cpuid.h"
  //#include "bli_env.h"
#else
  #include "blis.h"
#endif

int main( int argc, char** argv )
{
	arch_t id     = bli_cpuid_query_id();
	const char* s = bli_arch_string( id );

	printf( "%s\n", s );

	return 0;
}

