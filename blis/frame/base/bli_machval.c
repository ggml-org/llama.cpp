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

#include "blis.h"

#define FUNCPTR_T machval_fp

typedef void (*FUNCPTR_T)(
                           machval_t mval,
                           void*     v
                         );

static FUNCPTR_T GENARRAY(ftypes,machval);



//
// Define object-based interface.
//
void bli_machval( machval_t mval,
                  obj_t*    v )
{
	num_t     dt_v  = bli_obj_dt( v );

	void*     buf_v = bli_obj_buffer_at_off( v );

	FUNCPTR_T f;

	// Index into the function pointer array.
	f = ftypes[dt_v];

	// Invoke the function.
	f( mval,
	   buf_v );
}


//
// Define BLAS-like interfaces.
//
#undef  GENTFUNCR
#define GENTFUNCR( ctype_v, ctype_vr, chv, chvr, opname, varname ) \
\
void PASTEMAC(chv,opname) \
     ( \
       machval_t mval, \
       void*     v     \
     ) \
{ \
	static ctype_vr pvals[ BLIS_NUM_MACH_PARAMS ]; \
\
	static bool     first_time = TRUE; \
\
	dim_t           val_i      = mval - BLIS_MACH_PARAM_FIRST; \
	ctype_v*        v_cast     = v; \
\
	/* If this is the first time through, call the underlying
	   code to discover each machine parameter. */ \
	if ( first_time ) \
	{ \
		char  lapack_mval; \
		dim_t m, i; \
\
		for( i = 0, m = BLIS_MACH_PARAM_FIRST; \
		     i < BLIS_NUM_MACH_PARAMS - 1; \
		     ++i, ++m ) \
		{ \
			bli_param_map_blis_to_netlib_machval( m, &lapack_mval ); \
\
			/*printf( "bli_machval: querying %u %c\n", m, lapack_mval );*/ \
\
			pvals[i] = PASTEMAC(chvr,varname)( &lapack_mval, 1 ); \
\
			/*printf( "bli_machval: got back %34.29e\n", pvals[i] ); */ \
		} \
\
		/* Store epsilon^2 in the last element. */ \
		pvals[i] = pvals[0] * pvals[0]; \
\
		first_time = FALSE; \
	} \
\
	/* Copy the requested parameter value to the output buffer, which
	   may involve a demotion from the complex to real domain. */ \
	bli_tcopys( chvr,chv, pvals[ val_i ], *v_cast ); \
}

INSERT_GENTFUNCR_BASIC( machval, lamch )

