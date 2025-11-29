/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2022, The University of Texas at Austin

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

#include "blis.h"

#ifndef TEST_UTILS_H
#define TEST_UTILS_H

// Allow C++ users to include this header file in their source code. However,
// we make the extern "C" conditional on whether we're using a C++ compiler,
// since regular C compilers don't understand the extern "C" construct.
#ifdef __cplusplus
extern "C" {
#endif

// String arrays allocated using this constant will always add 1 to
// the value defined below, and so the total allocated will still be
// a nice power of two.
#define MAX_STRING_SIZE    31


extern const char* GLOB_DEF_DT_STR;
extern const char* GLOB_DEF_SC_STR;
extern const char* GLOB_DEF_IM_STR;

extern const char* GLOB_DEF_PS_STR;
extern const char* GLOB_DEF_M_STR;
extern const char* GLOB_DEF_N_STR;
extern const char* GLOB_DEF_K_STR;

extern const char* GLOB_DEF_NR_STR;

extern const char* GLOB_DEF_ALPHA_STR;
extern const char* GLOB_DEF_BETA_STR;


typedef struct params_s
{
	// Binary name.
	const char* bin;

	// Operation name.
	const char* opname;

	// Implementation name.
	const char* impl;

	// Multithreading parameters: number of threads and affinity string.
	const char* nt_str;
	long int    nt;
	const char* af_str;

	// Parameter combinations, datatype, operand storage combination,
	// and induced method.
	const char* pc_str;
	const char* dt_str;
	const char* sc_str;
	num_t dt;

	const char* im_str;
	ind_t im;
	bool  im_is_auto;

	// Problem size range and dimension specifiers.
	const char* ps_str;
	const char* m_str;
	const char* n_str;
	const char* k_str;
	long int sta;
	long int end;
	long int inc;
	long int m;
	long int n;
	long int k;

	// Number of repeats.
	const char* nr_str;
	long int nr;

	// Value of alpha and beta.
	const char* alpha_str;
	const char* beta_str;
	double alpha;
	double beta;

	// A flag controlling whether to print informational messages.
	bool verbose;

} params_t;

typedef void (*init_fp)( params_t* params );

// -----------------------------------------------------------------------------

void init_def_params( params_t* params );
void parse_cl_params( int argc, char** argv, init_fp fp, params_t* params );
void proc_params( params_t* params );

// -----------------------------------------------------------------------------

bool is_match( const char* str1, const char* str2 );
bool is_gemm( params_t* params );
bool is_hemm( params_t* params );
bool is_herk( params_t* params );
bool is_trmm( params_t* params );
bool is_trsm( params_t* params );

#ifdef __cplusplus
}
#endif

#endif
