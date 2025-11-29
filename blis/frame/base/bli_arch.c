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
  #include "bli_env.h"
#else
  #include "blis.h"
#endif

// -----------------------------------------------------------------------------

// The arch_t id for the currently running hardware. We initialize to -1,
// which will be overwritten upon calling bli_arch_set_id().
static arch_t cached_id = -1;

arch_t bli_arch_query_id( void )
{

#ifdef BLIS_ENABLE_GKS_CACHING

	// Deep-query the arch_t id once via bli_pthread_once(). Since we are
	// constrained by the pthread interface to pthread_once(), the id must be
	// "returned" indirectly via a static variable (cached_id).
	bli_arch_set_id_once();

	// Return the id that was previously cached.
	return cached_id;

#else

	// Deep-query and return a fresh arch_t.
	return bli_arch_query_id_impl();

#endif
}

// -----------------------------------------------------------------------------

// A pthread structure used in pthread_once(). pthread_once() is guaranteed to
// execute exactly once among all threads that pass in this control object.
static bli_pthread_once_t once_id = BLIS_PTHREAD_ONCE_INIT;

void bli_arch_set_id_once( void )
{
	// When this file is being compiled as part of the configure script's
	// hardware auto-detection driver, we avoid calling the bli_pthread APIs
	// so that we aren't required to include those symbols in the executable.
#ifndef BLIS_CONFIGURETIME_CPUID
	bli_pthread_once( &once_id, bli_arch_set_id );
#endif
}

// -----------------------------------------------------------------------------

void bli_arch_set_id( void )
{
	// Deep-query the arch_t and save it in the static variable (cached_id).
	cached_id = bli_arch_query_id_impl();
}

// -----------------------------------------------------------------------------

arch_t bli_arch_query_id_impl( void )
{
	arch_t id;

	// Check the environment variable BLIS_ARCH_DEBUG to see if the user
	// requested that we echo the result of the subconfiguration selection.
	bool do_logging = bli_env_get_var( "BLIS_ARCH_DEBUG", 0 );
	bli_arch_set_logging( do_logging );

	// Check the environment variable BLIS_ARCH_TYPE to see if the user
	// requested that we use a specific subconfiguration.
	dim_t req_id = bli_env_get_var( "BLIS_ARCH_TYPE", -1 );
	const char* req_env = bli_env_get_str( "BLIS_ARCH_TYPE" );

	// If the user specifically goes out of their way to define
	// BLIS_ARCH_TYPE=-1, then we have to handle that specially.
	if ( req_env )
	{
		// strtol allows leading whitespace
		const char* p = req_env;
		while ( isspace( *p ) ) p++;

		if ( strcmp( p, "-1" ) == 0 )
		{
			req_env = NULL;
		}
	}

	// When this file is being compiled as part of the configure script's
	// hardware auto-detection driver, we avoid calling the bli_check APIs
	// so that we aren't required to include those symbols in the executable.
#ifndef BLIS_CONFIGURETIME_CPUID
	if ( req_env )
	{
		// BLIS_ARCH_TYPE was set. Cautiously check whether its value is usable.

		// A non-numerical value was supplied. Check if it matches one of the
		// configuration names.
		if ( req_id == -1 )
		{
			for ( arch_t i = 0; i < BLIS_NUM_ARCHS; i++ )
			{
				if ( strcasecmp( req_env, bli_arch_string( i ) ) == 0 )
				{
					req_id = i;
				}
			}
		}

		// If req_id was set to an invalid arch_t value (ie: outside the range
		// [0,BLIS_NUM_ARCHS-1]), output an error message and abort.
		if ( bli_error_checking_is_enabled() )
		{
			err_t e_val = bli_check_valid_arch_id( req_id );
			bli_check_error_code( e_val );
		}

		// At this point, we know that req_id is in the valid range, but we
		// don't yet know if it refers to a context that was actually
		// initialized. Query the address of an internal context data structure
		// corresponding to req_id. This pointer will be NULL if the associated
		// subconfig is not available.
		const cntx_t* req_cntx = bli_gks_lookup_id( req_id );

		// This function checks the context pointer and aborts with a useful
		// error message if the pointer is found to be NULL.
		if ( bli_error_checking_is_enabled() )
		{
			err_t e_val = bli_check_initialized_gks_cntx( req_cntx );
			bli_check_error_code( e_val );
		}

		// Finally, we can be confident that req_id (1) is in range and (2)
		// refers to a context that has been initialized.
		id = req_id;
	}
	else
#endif
	{
		// BLIS_ARCH_TYPE was unset. Proceed with normal subconfiguration
		// selection behavior.

		// Architecture families.
		#if defined BLIS_FAMILY_INTEL64 || \
		    defined BLIS_FAMILY_AMD64   || \
		    defined BLIS_FAMILY_X86_64  || \
		    defined BLIS_FAMILY_ARM64   || \
		    defined BLIS_FAMILY_ARM32   || \
		    defined BLIS_FAMILY_POWER
		id = bli_cpuid_query_id();
		#endif

		// Intel microarchitectures.
		#ifdef BLIS_FAMILY_SKX
		id = BLIS_ARCH_SKX;
		#endif
		#ifdef BLIS_FAMILY_KNL
		id = BLIS_ARCH_KNL;
		#endif
		#ifdef BLIS_FAMILY_KNC
		id = BLIS_ARCH_KNC;
		#endif
		#ifdef BLIS_FAMILY_HASWELL
		id = BLIS_ARCH_HASWELL;
		#endif
		#ifdef BLIS_FAMILY_SANDYBRIDGE
		id = BLIS_ARCH_SANDYBRIDGE;
		#endif
		#ifdef BLIS_FAMILY_PENRYN
		id = BLIS_ARCH_PENRYN;
		#endif

		// AMD microarchitectures.
		#ifdef BLIS_FAMILY_ZEN3
		id = BLIS_ARCH_ZEN3;
		#endif
		#ifdef BLIS_FAMILY_ZEN2
		id = BLIS_ARCH_ZEN2;
		#endif
		#ifdef BLIS_FAMILY_ZEN
		id = BLIS_ARCH_ZEN;
		#endif
		#ifdef BLIS_FAMILY_EXCAVATOR
		id = BLIS_ARCH_EXCAVATOR;
		#endif
		#ifdef BLIS_FAMILY_STEAMROLLER
		id = BLIS_ARCH_STEAMROLLER;
		#endif
		#ifdef BLIS_FAMILY_PILEDRIVER
		id = BLIS_ARCH_PILEDRIVER;
		#endif
		#ifdef BLIS_FAMILY_BULLDOZER
		id = BLIS_ARCH_BULLDOZER;
		#endif

		// ARM microarchitectures.
		#ifdef BLIS_FAMILY_ARMSVE
		id = BLIS_ARCH_ARMSVE;
		#endif
		#ifdef BLIS_FAMILY_A64FX
		id = BLIS_ARCH_A64FX;
		#endif

		#ifdef BLIS_FAMILY_ALTRAMAX
		id = BLIS_ARCH_ALTRAMAX;
		#endif
		#ifdef BLIS_FAMILY_ALTRA
		id = BLIS_ARCH_ALTRA;
		#endif
		#ifdef BLIS_FAMILY_FIRESTORM
		id = BLIS_ARCH_FIRESTORM;
		#endif

		#ifdef BLIS_FAMILY_THUNDERX2
		id = BLIS_ARCH_THUNDERX2;
		#endif
		#ifdef BLIS_FAMILY_CORTEXA57
		id = BLIS_ARCH_CORTEXA57;
		#endif
		#ifdef BLIS_FAMILY_CORTEXA53
		id = BLIS_ARCH_CORTEXA53;
		#endif

		#ifdef BLIS_FAMILY_CORTEXA15
		id = BLIS_ARCH_CORTEXA15;
		#endif
		#ifdef BLIS_FAMILY_CORTEXA9
		id = BLIS_ARCH_CORTEXA9;
		#endif

		// IBM microarchitectures.
		#ifdef BLIS_FAMILY_POWER10
		id = BLIS_ARCH_POWER10;
		#endif
		#ifdef BLIS_FAMILY_POWER9
		id = BLIS_ARCH_POWER9;
		#endif
		#ifdef BLIS_FAMILY_POWER7
		id = BLIS_ARCH_POWER7;
		#endif
		#ifdef BLIS_FAMILY_BGQ
		id = BLIS_ARCH_BGQ;
		#endif

		// RISC-V microarchitectures
		#ifdef BLIS_FAMILY_RV32I
		id = BLIS_ARCH_RV32I;
		#endif
		#ifdef BLIS_FAMILY_RV64I
		id = BLIS_ARCH_RV64I;
		#endif
		#ifdef BLIS_FAMILY_RV32IV
		id = BLIS_ARCH_RV32IV;
		#endif
		#ifdef BLIS_FAMILY_RV64IV
		id = BLIS_ARCH_RV64IV;
		#endif

		// SiFive microarchitectures.
		#ifdef BLIS_FAMILY_SIFIVE_RVV
		id = BLIS_ARCH_SIFIVE_RVV;
		#endif
		#ifdef BLIS_FAMILY_SIFIVE_X280
		id = BLIS_ARCH_SIFIVE_X280;
		#endif

		// Generic microarchitecture.
		#ifdef BLIS_FAMILY_GENERIC
		id = BLIS_ARCH_GENERIC;
		#endif
	}

	if ( bli_arch_get_logging() )
		fprintf( stderr, "libblis: selecting sub-configuration '%s'.\n",
				 bli_arch_string( id ) );

	#if 0
	printf( "blis_arch_query_id_impl(): id = %u\n", id );
	exit(1);
	#endif

	return id;
}

// -----------------------------------------------------------------------------

// NOTE: This string array must be kept up-to-date with the arch_t
// enumeration that is typedef'ed in bli_type_defs.h. That is, the
// index order of each string should correspond to the implied/assigned
// enum value given to the corresponding BLIS_ARCH_ value.
static const char* config_name[ BLIS_NUM_ARCHS ] =
{
    "skx",
    "knl",
    "knc",
    "haswell",
    "sandybridge",
    "penryn",

    "zen3",
    "zen2",
    "zen",
    "excavator",
    "steamroller",
    "piledriver",
    "bulldozer",

    "armsve",
    "a64fx",

	"altramax",
	"altra",
    "firestorm",

    "thunderx2",
    "cortexa57",
    "cortexa53",

    "cortexa15",
    "cortexa9",

    "power10",
    "power9",
    "power7",
    "bgq",

    "rv32i",
    "rv64i",
    "rv32iv",
    "rv64iv",

    "sifive_rvv",
    "sifive_x280",

    "generic"
};

const char* bli_arch_string( arch_t id )
{
	return config_name[ id ];
}

// -----------------------------------------------------------------------------

static bool arch_dolog = 0;

void bli_arch_set_logging( bool dolog )
{
	arch_dolog = dolog;
}

bool bli_arch_get_logging( void )
{
	return arch_dolog;
}

void bli_arch_log( const char* fmt, ... )
{
	const char prefix[] = "libblis: ";
	int  n_chars  = strlen( prefix ) + strlen( fmt ) + 1;

	if ( bli_arch_get_logging() && fmt )
	{
		char* prefix_fmt = malloc( n_chars );

		snprintf( prefix_fmt, n_chars, "%s%s", prefix, fmt );

		va_list ap;
		va_start( ap, fmt );
		vfprintf( stderr, prefix_fmt, ap );
		va_end( ap );

		free( prefix_fmt );
	}
}

