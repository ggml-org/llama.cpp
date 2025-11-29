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

// Guard the function definitions so that they are only compiled when
// #included from files that define the object API macros.
#ifdef BLIS_ENABLE_OAPI

//
// Define object-based interfaces.
//

#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       const obj_t* x, \
       const obj_t* asum  \
       BLIS_OAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_OAPI_EX_DECLS \
\
	num_t dt    = bli_obj_dt( x ); \
\
	dim_t n     = bli_obj_vector_dim( x ); \
	void* buf_x = bli_obj_buffer_at_off( x ); \
	inc_t incx  = bli_obj_vector_inc( x ); \
\
	void* buf_asum = bli_obj_buffer_at_off( asum ); \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( x, asum ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(opname,BLIS_TAPI_EX_SUF,_vft) f = \
	PASTEMAC(opname,BLIS_TAPI_EX_SUF,_qfp)( dt ); \
\
	f \
	( \
	  n, \
	  buf_x, incx, \
	  buf_asum, \
	  cntx, \
	  rntm  \
	); \
}

GENFRONT( asumv )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       const obj_t* a  \
       BLIS_OAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_OAPI_EX_DECLS \
\
	num_t  dt    = bli_obj_dt( a ); \
\
	uplo_t uploa = bli_obj_uplo( a ); \
	dim_t  m     = bli_obj_length( a ); \
	void*  buf_a = bli_obj_buffer_at_off( a ); \
	inc_t  rs_a  = bli_obj_row_stride( a ); \
	inc_t  cs_a  = bli_obj_col_stride( a ); \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( a ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(opname,BLIS_TAPI_EX_SUF,_vft) f = \
	PASTEMAC(opname,BLIS_TAPI_EX_SUF,_qfp)( dt ); \
\
	f \
	( \
	  uploa, \
	  m, \
	  buf_a, rs_a, cs_a, \
	  cntx, \
	  rntm  \
	); \
}

GENFRONT( mkherm )
GENFRONT( mksymm )
GENFRONT( mktrim )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       const obj_t* x, \
       const obj_t* norm  \
       BLIS_OAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_OAPI_EX_DECLS \
\
	num_t dt       = bli_obj_dt( x ); \
\
	dim_t n        = bli_obj_vector_dim( x ); \
	void* buf_x    = bli_obj_buffer_at_off( x ); \
	inc_t incx     = bli_obj_vector_inc( x ); \
	void* buf_norm = bli_obj_buffer_at_off( norm ); \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( x, norm ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(opname,BLIS_TAPI_EX_SUF,_vft) f = \
	PASTEMAC(opname,BLIS_TAPI_EX_SUF,_qfp)( dt ); \
\
	f \
	( \
	  n, \
	  buf_x, incx, \
	  buf_norm, \
	  cntx, \
	  rntm  \
	); \
}

GENFRONT( norm1v )
GENFRONT( normfv )
GENFRONT( normiv )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       const obj_t* x, \
       const obj_t* norm  \
       BLIS_OAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_OAPI_EX_DECLS \
\
	num_t  dt       = bli_obj_dt( x ); \
\
	doff_t diagoffx = bli_obj_diag_offset( x ); \
	diag_t diagx    = bli_obj_diag( x ); \
	uplo_t uplox    = bli_obj_uplo( x ); \
	dim_t  m        = bli_obj_length( x ); \
	dim_t  n        = bli_obj_width( x ); \
	void*  buf_x    = bli_obj_buffer_at_off( x ); \
	inc_t  rs_x     = bli_obj_row_stride( x ); \
	inc_t  cs_x     = bli_obj_col_stride( x ); \
	void*  buf_norm = bli_obj_buffer_at_off( norm ); \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( x, norm ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(opname,BLIS_TAPI_EX_SUF,_vft) f = \
	PASTEMAC(opname,BLIS_TAPI_EX_SUF,_qfp)( dt ); \
\
	f \
	( \
	  diagoffx, \
	  diagx, \
	  uplox, \
	  m, \
	  n, \
	  buf_x, rs_x, cs_x, \
	  buf_norm, \
	  cntx, \
	  rntm  \
	); \
}

GENFRONT( norm1m )
GENFRONT( normfm )
GENFRONT( normim )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       const obj_t* x  \
       BLIS_OAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_OAPI_EX_DECLS \
\
	num_t dt    = bli_obj_dt( x ); \
\
	dim_t n     = bli_obj_vector_dim( x ); \
	void* buf_x = bli_obj_buffer_at_off( x ); \
	inc_t incx  = bli_obj_vector_inc( x ); \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( x ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(opname,BLIS_TAPI_EX_SUF,_vft) f = \
	PASTEMAC(opname,BLIS_TAPI_EX_SUF,_qfp)( dt ); \
\
	f \
	( \
	  n, \
	  buf_x, incx, \
	  cntx, \
	  rntm  \
	); \
}

GENFRONT( randv )
GENFRONT( randnv )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       const obj_t* x  \
       BLIS_OAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_OAPI_EX_DECLS \
\
	num_t  dt       = bli_obj_dt( x ); \
\
	doff_t diagoffx = bli_obj_diag_offset( x ); \
	uplo_t uplox    = bli_obj_uplo( x ); \
	dim_t  m        = bli_obj_length( x ); \
	dim_t  n        = bli_obj_width( x ); \
	void*  buf_x    = bli_obj_buffer_at_off( x ); \
	inc_t  rs_x     = bli_obj_row_stride( x ); \
	inc_t  cs_x     = bli_obj_col_stride( x ); \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( x ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(opname,BLIS_TAPI_EX_SUF,_vft) f = \
	PASTEMAC(opname,BLIS_TAPI_EX_SUF,_qfp)( dt ); \
\
	f \
	( \
	  diagoffx, \
	  uplox, \
	  m, \
	  n, \
	  buf_x, rs_x, cs_x, \
	  cntx, \
	  rntm  \
	); \
}

GENFRONT( randm )
GENFRONT( randnm )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname,EX_SUF) \
     ( \
       const obj_t* x, \
       const obj_t* scale, \
       const obj_t* sumsq  \
       BLIS_OAPI_EX_PARAMS  \
     ) \
{ \
	bli_init_once(); \
\
	BLIS_OAPI_EX_DECLS \
\
	num_t dt        = bli_obj_dt( x ); \
\
	dim_t n         = bli_obj_vector_dim( x ); \
	void* buf_x     = bli_obj_buffer_at_off( x ); \
	inc_t incx      = bli_obj_vector_inc( x ); \
	void* buf_scale = bli_obj_buffer_at_off( scale ); \
	void* buf_sumsq = bli_obj_buffer_at_off( sumsq ); \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( x, scale, sumsq ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(opname,BLIS_TAPI_EX_SUF,_vft) f = \
	PASTEMAC(opname,BLIS_TAPI_EX_SUF,_qfp)( dt ); \
\
	f \
	( \
	  n, \
	  buf_x, incx, \
	  buf_scale, \
	  buf_sumsq, \
	  cntx, \
	  rntm  \
	); \
}

GENFRONT( sumsqv )

// -----------------------------------------------------------------------------

// Operations with only basic interfaces.

#ifdef BLIS_OAPI_BASIC

#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname) \
     ( \
       const obj_t* chi, \
       const obj_t* psi, \
             bool*  is_eq  \
     ) \
{ \
	bli_init_once(); \
\
	num_t dt_chi = bli_obj_dt( chi ); \
	num_t dt_psi = bli_obj_dt( psi ); \
	num_t dt; \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( chi, psi, is_eq ); \
\
	/* Decide which datatype will be used to query the buffer from the
	   constant object (if there is one). */ \
	if ( bli_is_constant( dt_psi ) ) dt = dt_chi; \
	else                             dt = dt_psi; \
\
	/* If chi and psi are both constants, then we compare only the dcomplex
	   fields. */ \
	if ( bli_is_constant( dt ) ) dt = BLIS_DCOMPLEX; \
\
	void* buf_chi = bli_obj_buffer_for_1x1( dt, chi ); \
	void* buf_psi = bli_obj_buffer_for_1x1( dt, psi ); \
\
	/* Integer objects are handled separately. */ \
	if ( bli_is_int( dt ) ) \
	{ \
		*is_eq = bli_ieq( buf_chi, buf_psi ); \
		return; \
	} \
\
	/* Query the conj status of each object and use the two to come up with a
	   single "net" conj_t value. */ \
	conj_t conjchi = bli_obj_conj_status( chi ); \
	conj_t conjpsi = bli_obj_conj_status( psi ); \
	conj_t conj    = bli_apply_conj( conjchi, conjpsi ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(opname,_vft) f = \
	PASTEMAC(opname,_qfp)( dt ); \
\
	f \
	( \
	  conj, \
	  buf_chi, \
	  buf_psi, \
	  is_eq  \
	); \
}

GENFRONT( eqsc )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname) \
     ( \
       const obj_t* x, \
       const obj_t* y, \
             bool*  is_eq  \
     ) \
{ \
	bli_init_once(); \
\
	num_t dt    = bli_obj_dt( x ); \
\
	dim_t n     = bli_obj_vector_dim( x ); \
	void* buf_x = bli_obj_buffer_at_off( x ); \
	inc_t inc_x = bli_obj_vector_inc( x ); \
	void* buf_y = bli_obj_buffer_at_off( y ); \
	inc_t inc_y = bli_obj_vector_inc( y ); \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( x, y, is_eq ); \
\
	/* Query the conj status of each object and use the two to come up with a
	   single "net" conj_t value. */ \
	conj_t conjx = bli_obj_conj_status( x ); \
	conj_t conjy = bli_obj_conj_status( y ); \
	conj_t conj  = bli_apply_conj( conjx, conjy ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(opname,_vft) f = \
	PASTEMAC(opname,_qfp)( dt ); \
\
	f \
	( \
	  conj, \
	  n, \
	  buf_x, inc_x, \
	  buf_y, inc_y, \
	  is_eq  \
	); \
}

GENFRONT( eqv )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname) \
     ( \
       const obj_t* x, \
       const obj_t* y, \
             bool*  is_eq  \
     ) \
{ \
	bli_init_once(); \
\
	num_t  dt       = bli_obj_dt( x ); \
\
	doff_t diagoffx = bli_obj_diag_offset( x ); \
	diag_t diagx    = bli_obj_diag( x ); \
	uplo_t uplox    = bli_obj_uplo( x ); \
	dim_t  m        = bli_obj_length( y ); \
	dim_t  n        = bli_obj_width( y ); \
	void*  buf_x    = bli_obj_buffer_at_off( x ); \
	inc_t  rs_x     = bli_obj_row_stride( x ); \
	inc_t  cs_x     = bli_obj_col_stride( x ); \
	void*  buf_y    = bli_obj_buffer_at_off( y ); \
	inc_t  rs_y     = bli_obj_row_stride( y ); \
	inc_t  cs_y     = bli_obj_col_stride( y ); \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( x, y, is_eq ); \
\
	/* Query the combined trans and conj status of each object and use the two
	   to come up with a single "net" trans_t value. */ \
	trans_t transx = bli_obj_conjtrans_status( x ); \
	trans_t transy = bli_obj_conjtrans_status( y ); \
	trans_t trans  = bli_apply_trans( transy, transx ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(opname,_vft) f = \
	PASTEMAC(opname,_qfp)( dt ); \
\
	f \
	( \
	  diagoffx, \
	  diagx, \
	  uplox, \
	  trans, \
	  m, \
	  n, \
	  buf_x, rs_x, cs_x, \
	  buf_y, rs_y, cs_y, \
	  is_eq  \
	); \
}

GENFRONT( eqm )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname) \
     ( \
       const obj_t* chi, \
       const obj_t* psi, \
             bool*  is  \
     ) \
{ \
	bli_init_once(); \
\
	num_t dt_chi = bli_obj_dt( chi ); \
	num_t dt_psi = bli_obj_dt( psi ); \
	num_t dt; \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( chi, psi, is ); \
\
	/* Decide which datatype will be used to query the buffer from the
	   constant object (if there is one). */ \
	if ( bli_is_constant( dt_psi ) ) dt = dt_chi; \
	else                             dt = dt_psi; \
\
	/* If chi and psi are both constants, then we compare only the dcomplex
	   fields. */ \
	if ( bli_is_constant( dt ) ) dt = BLIS_DOUBLE; \
\
	void* buf_chi = bli_obj_buffer_for_1x1( dt, chi ); \
	void* buf_psi = bli_obj_buffer_for_1x1( dt, psi ); \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(opname,_vft) f = \
	PASTEMAC(opname,_qfp)( dt ); \
\
	f \
	( \
	  buf_chi, \
	  buf_psi, \
	  is  \
	); \
}

GENFRONT( ltsc )
GENFRONT( lesc )
GENFRONT( gtsc )
GENFRONT( gesc )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname) \
     ( \
             FILE*  file, \
       const char*  s1, \
       const obj_t* x, \
       const char*  format, \
       const char*  s2  \
     ) \
{ \
	bli_init_once(); \
\
	num_t dt    = bli_obj_dt( x ); \
\
	dim_t n     = bli_obj_vector_dim( x ); \
	void* buf_x = bli_obj_buffer_at_off( x ); \
	inc_t incx  = bli_obj_vector_inc( x ); \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( file, s1, x, format, s2 ); \
\
	/* Handle constants up front. */ \
	if ( dt == BLIS_CONSTANT ) \
	{ \
		bli_check_error_code( BLIS_NOT_YET_IMPLEMENTED ); \
	} \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(opname,_vft) f = \
	PASTEMAC(opname,_qfp)( dt ); \
\
	f \
	( \
	  file, \
	  s1, \
	  n, \
	  buf_x, incx, \
	  format, \
	  s2  \
	); \
}

GENFRONT( fprintv )


#undef  GENFRONT
#define GENFRONT( opname ) \
\
void PASTEMAC(opname) \
     ( \
             FILE*  file, \
       const char*  s1, \
       const obj_t* x, \
       const char*  format, \
       const char*  s2  \
     ) \
{ \
	bli_init_once(); \
\
	num_t dt    = bli_obj_dt( x ); \
\
	dim_t m     = bli_obj_length( x ); \
	dim_t n     = bli_obj_width( x ); \
	void* buf_x = bli_obj_buffer_at_off( x ); \
	inc_t rs_x  = bli_obj_row_stride( x ); \
	inc_t cs_x  = bli_obj_col_stride( x ); \
\
	if ( bli_error_checking_is_enabled() ) \
		PASTEMAC(opname,_check)( file, s1, x, format, s2 ); \
\
	/* Handle constants up front. */ \
	if ( dt == BLIS_CONSTANT ) \
	{ \
		const float*    sp = bli_obj_buffer_for_const( BLIS_FLOAT,    x ); \
		const double*   dp = bli_obj_buffer_for_const( BLIS_DOUBLE,   x ); \
		const scomplex* cp = bli_obj_buffer_for_const( BLIS_SCOMPLEX, x ); \
		const dcomplex* zp = bli_obj_buffer_for_const( BLIS_DCOMPLEX, x ); \
		const gint_t*   ip = bli_obj_buffer_for_const( BLIS_INT,      x ); \
\
		fprintf( file, "%s\n", s1 ); \
		fprintf( file, " float:     %9.2e\n",         bli_sreal( *sp ) ); \
		fprintf( file, " double:    %9.2e\n",         bli_dreal( *dp ) ); \
		fprintf( file, " scomplex:  %9.2e + %9.2e\n", bli_creal( *cp ), \
		                                              bli_cimag( *cp ) ); \
		fprintf( file, " dcomplex:  %9.2e + %9.2e\n", bli_zreal( *zp ), \
		                                              bli_zimag( *zp ) ); \
		fprintf( file, " int:       %ld\n",           ( long )(*ip) ); \
		fprintf( file, "\n" ); \
		return; \
	} \
\
	/* Query a type-specific function pointer, except one that uses
	   void* for function arguments instead of typed pointers. */ \
	PASTECH(opname,_vft) f = \
	PASTEMAC(opname,_qfp)( dt ); \
\
	f \
	( \
	  file, \
	  s1, \
	  m, \
	  n, \
	  buf_x, rs_x, cs_x, \
	  format, \
	  s2  \
	); \
}

GENFRONT( fprintm )


#undef  GENFRONT
#define GENFRONT( opname, varname ) \
\
void PASTEMAC(opname) \
     ( \
       const char*  s1, \
       const obj_t* x, \
       const char*  format, \
       const char*  s2  \
     ) \
{ \
	bli_init_once(); \
\
	/* Invoke the typed function. */ \
	PASTEMAC(varname) \
	( \
	  stdout, \
	  s1, \
	  x, \
	  format, \
	  s2  \
	); \
}

GENFRONT( printv, fprintv )
GENFRONT( printm, fprintm )

#endif // #ifdef BLIS_OAPI_BASIC


#endif

