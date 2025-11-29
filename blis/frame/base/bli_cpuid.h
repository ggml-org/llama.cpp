/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2018-2020, Advanced Micro Devices, Inc.

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

#if 0
  // Used only during standalone testing of ARM support.
  #define FALSE 0
  #define TRUE  1
  typedef enum arch_e
  {
	BLIS_ARCH_CORTEXA57 = 10,
	BLIS_ARCH_CORTEXA15 = 11,
	BLIS_ARCH_CORTEXA9  = 12,
	BLIS_ARCH_GENERIC   = 13
  } arch_t;
  typedef uint64_t bool;
  #define bli_abort abort
#endif

#ifndef BLIS_CPUID_H
#define BLIS_CPUID_H

arch_t bli_cpuid_query_id( void );

// Intel
bool bli_cpuid_is_skx( uint32_t family, uint32_t model, uint32_t features );
bool bli_cpuid_is_knl( uint32_t family, uint32_t model, uint32_t features );
bool bli_cpuid_is_haswell( uint32_t family, uint32_t model, uint32_t features );
bool bli_cpuid_is_sandybridge( uint32_t family, uint32_t model, uint32_t features );
bool bli_cpuid_is_penryn( uint32_t family, uint32_t model, uint32_t features );

// AMD
bool bli_cpuid_is_zen3( uint32_t family, uint32_t model, uint32_t features );
bool bli_cpuid_is_zen2( uint32_t family, uint32_t model, uint32_t features );
bool bli_cpuid_is_zen( uint32_t family, uint32_t model, uint32_t features );
bool bli_cpuid_is_excavator( uint32_t family, uint32_t model, uint32_t features );
bool bli_cpuid_is_steamroller( uint32_t family, uint32_t model, uint32_t features );
bool bli_cpuid_is_piledriver( uint32_t family, uint32_t model, uint32_t features );
bool bli_cpuid_is_bulldozer( uint32_t family, uint32_t model, uint32_t features );

// ARM
bool bli_cpuid_is_thunderx2( uint32_t model, uint32_t part, uint32_t features );
bool bli_cpuid_is_cortexa57( uint32_t model, uint32_t part, uint32_t features );
bool bli_cpuid_is_cortexa53( uint32_t model, uint32_t part, uint32_t features );
bool bli_cpuid_is_armsve( uint32_t model, uint32_t part, uint32_t features );
bool bli_cpuid_is_a64fx( uint32_t model, uint32_t part, uint32_t features );
bool bli_cpuid_is_cortexa15( uint32_t model, uint32_t part, uint32_t features );
bool bli_cpuid_is_cortexa9( uint32_t model, uint32_t part, uint32_t features );

uint32_t bli_cpuid_query( uint32_t* family, uint32_t* model, uint32_t* features );

// -----------------------------------------------------------------------------

//
// This section of the file was based off of cpuid.hpp from TBLIS [1].
//
// [1] https://github.com/devinamatthews/tblis
//

/*

   Copyright (C) 2017, The University of Texas at Austin
   Copyright (C) 2017, Devin Matthews

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

BLIS_INLINE bool bli_cpuid_has_features( uint32_t have, uint32_t want )
{
    return ( have & want ) == want;
}

// -----------------------------------------------------------------------------

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)

// cpuid.h is now #included in bli_cpuid.c instead of here. See issue #393
// for more information why this move was made.
//#include "cpuid.h"

void get_cpu_name( char *cpu_name );
int  vpu_count( void );


enum
{
	VENDOR_INTEL = 0,
	VENDOR_AMD,
	VENDOR_UNKNOWN
};
enum
{
	FEATURE_SSE3     = 0x0001,
	FEATURE_SSSE3    = 0x0002,
	FEATURE_SSE41    = 0x0004,
	FEATURE_SSE42    = 0x0008,
	FEATURE_AVX      = 0x0010,
	FEATURE_AVX2     = 0x0020,
	FEATURE_FMA3     = 0x0040,
	FEATURE_FMA4     = 0x0080,
	FEATURE_AVX512F  = 0x0100,
	FEATURE_AVX512DQ = 0x0200,
	FEATURE_AVX512PF = 0x0400,
	FEATURE_AVX512ER = 0x0800,
	FEATURE_AVX512CD = 0x1000,
	FEATURE_AVX512BW = 0x2000,
	FEATURE_AVX512VL = 0x4000
};

#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM) || defined(_ARCH_PPC)

char* find_string_in( char* target, char* buffer, size_t buf_len, char* filepath );

enum
{
	VENDOR_ARM = 0,
	VENDOR_IBM,
	VENDOR_UNKNOWN
};
enum
{
	MODEL_ARMV7 = 0,
	MODEL_ARMV8,
	MODEL_POWER7,
	MODEL_POWER9,
	MODEL_POWER10,
	MODEL_UNKNOWN
};
enum
{
	FEATURE_NEON = 0x01,
	FEATURE_SVE  = 0x02
};

#endif



#endif

