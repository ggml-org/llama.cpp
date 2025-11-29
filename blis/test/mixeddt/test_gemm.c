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
    - Neither the name of The University of Texas nor the names of its
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

#include <unistd.h>
#include "blis.h"

void blas_gemm_md( obj_t* alpha, obj_t* a, obj_t* b, obj_t* beta, obj_t* c );
void blas_gemm( trans_t transa, trans_t transb, num_t dt, obj_t* ao, obj_t* alpha, obj_t* bo, obj_t* beta, obj_t* co );

//#define PRINT

int main( int argc, char** argv )
{
	obj_t    a, b, c;
	obj_t    c_save;
	obj_t*   alphao;
	obj_t*   betao;
	dim_t    m, n, k;
	dim_t    p;
	int      r;

	double   dtime;
	double   dtime_save;
	double   gflops;
	double   flopsmul;

	//bli_init();

	//bli_error_checking_level_set( BLIS_NO_ERROR_CHECKING );

	int n_repeats = 3;

	num_t dta = DTA;
	num_t dtb = DTB;
	num_t dtc = DTC;
	num_t dtx = DTX;

	const bool   a_real    = bli_is_real( dta );
	const bool   b_real    = bli_is_real( dtb );
	const bool   c_real    = bli_is_real( dtc );
	const bool   a_complex = bli_is_complex( dta );
	const bool   b_complex = bli_is_complex( dtb );
	const bool   c_complex = bli_is_complex( dtc );

	// Extract the precision component of the computation datatype.
	prec_t comp_prec = bli_dt_prec( dtx );

	dim_t p_begin = P_BEGIN;
	dim_t p_max   = P_MAX;
	dim_t p_inc   = P_INC;

	int m_input   = -1;
	int n_input   = -1;
	int k_input   = -1;

#if 0
	k_input = 256;
#endif

#if 0
	char dta_ch, dtb_ch, dtc_ch, dtx_ch;

	// Choose the char corresponding to the requested datatype.
	if      ( bli_is_float( dta ) )    dta_ch = 's';
	else if ( bli_is_double( dta ) )   dta_ch = 'd';
	else if ( bli_is_scomplex( dta ) ) dta_ch = 'c';
	else                               dta_ch = 'z';

	if      ( bli_is_float( dtb ) )    dtb_ch = 's';
	else if ( bli_is_double( dtb ) )   dtb_ch = 'd';
	else if ( bli_is_scomplex( dtb ) ) dtb_ch = 'c';
	else                               dtb_ch = 'z';

	if      ( bli_is_float( dtc ) )    dtc_ch = 's';
	else if ( bli_is_double( dtc ) )   dtc_ch = 'd';
	else if ( bli_is_scomplex( dtc ) ) dtc_ch = 'c';
	else                               dtc_ch = 'z';

	if      ( bli_is_float( dtx ) )    dtx_ch = 's';
	else                               dtx_ch = 'd';

	( void )dta_ch;
	( void )dtb_ch;
	( void )dtc_ch;
	( void )dtx_ch;
#endif

	trans_t transa = BLIS_NO_TRANSPOSE;
	trans_t transb = BLIS_NO_TRANSPOSE;


	// Begin with initializing the last entry to zero so that
	// matlab allocates space for the entire array once up-front.
	for ( p = p_begin; p + p_inc <= p_max; p += p_inc ) ;

	//printf( "data_%s_%c%c%c%cgemm_%s",      THR_STR, dtc_ch, dta_ch, dtb_ch, dtx_ch, STR );
	printf( "data_gemm_%s", STR );
	printf( "( %2lu, 1:4 ) = [ %4lu %4lu %4lu %7.2f ];\n",
	        ( unsigned long )(p - p_begin)/p_inc + 1,
	        ( unsigned long )0,
	        ( unsigned long )0,
	        ( unsigned long )0, 0.0 );

	// Adjust the flops scaling based on which domain case is being executed.
	if      ( c_real    && a_real    && b_real    ) flopsmul = 2.0;
	else if ( c_real    && a_real    && b_complex ) flopsmul = 2.0;
	else if ( c_real    && a_complex && b_real    ) flopsmul = 2.0;
	else if ( c_real    && a_complex && b_complex ) flopsmul = 4.0;
	else if ( c_complex && a_real    && b_real    ) flopsmul = 2.0;
	else if ( c_complex && a_real    && b_complex ) flopsmul = 4.0;
	else if ( c_complex && a_complex && b_real    ) flopsmul = 4.0;
	else if ( c_complex && a_complex && b_complex ) flopsmul = 8.0;


	//for ( p = p_begin; p <= p_max; p += p_inc )
	for ( p = p_max; p_begin <= p; p -= p_inc )
	{

		if ( m_input < 0 ) m = p * ( dim_t )abs(m_input);
		else               m =     ( dim_t )    m_input;
		if ( n_input < 0 ) n = p * ( dim_t )abs(n_input);
		else               n =     ( dim_t )    n_input;
		if ( k_input < 0 ) k = p * ( dim_t )abs(k_input);
		else               k =     ( dim_t )    k_input;

		bli_obj_create( dta, m, k, 0, 0, &a );
		bli_obj_create( dtb, k, n, 0, 0, &b );
		bli_obj_create( dtc, m, n, 0, 0, &c );
		bli_obj_create( dtc, m, n, 0, 0, &c_save );

		bli_obj_set_comp_prec( comp_prec, &c );

		alphao = &BLIS_ONE;
		betao  = &BLIS_ONE;

		bli_randm( &a );
		bli_randm( &b );
		bli_randm( &c );

		bli_obj_set_conjtrans( transa, &a );
		bli_obj_set_conjtrans( transb, &b );

		bli_copym( &c, &c_save );

		dtime_save = DBL_MAX;

		for ( r = 0; r < n_repeats; ++r )
		{
			bli_copym( &c_save, &c );

			dtime = bli_clock();

#ifdef PRINT
			bli_printm( "a", &a, "%4.1f", "" );
			bli_printm( "b", &b, "%4.1f", "" );
			bli_printm( "c", &c, "%4.1f", "" );
#endif

#ifdef BLIS

			bli_gemm
			(
			  alphao,
			  &a,
			  &b,
			  betao,
			  &c
			);
#else
			blas_gemm_md
			(
			  alphao,
			  &a,
			  &b,
			  betao,
			  &c
			);
#endif

#ifdef PRINT
			bli_printm( "c after", &c, "%4.1f", "" );
			exit(1);
#endif

			dtime_save = bli_clock_min_diff( dtime_save, dtime );
		}

		gflops = ( flopsmul * m * k * n ) / ( dtime_save * 1.0e9 );

		//printf( "data_%s_%c%c%c%cgemm_%s",      THR_STR, dtc_ch, dta_ch, dtb_ch, dtx_ch, STR );
		printf( "data_gemm_%s", STR );
		printf( "( %2lu, 1:4 ) = [ %4lu %4lu %4lu %7.2f ];\n",
		        ( unsigned long )(p - p_begin)/p_inc + 1,
		        ( unsigned long )m,
		        ( unsigned long )k,
		        ( unsigned long )n, gflops );

		bli_obj_free( &a );
		bli_obj_free( &b );
		bli_obj_free( &c );
		bli_obj_free( &c_save );
	}

	//bli_finalize();

	return 0;
}

void blas_gemm_md( obj_t* alpha, obj_t* a, obj_t* b, obj_t* beta, obj_t* c )
{
	trans_t transa = bli_obj_conjtrans_status( a );
	trans_t transb = bli_obj_conjtrans_status( b );

	prec_t  comp_prec = bli_obj_comp_prec( c );

	if ( bli_obj_dt( a ) == bli_obj_dt( b ) &&
	     bli_obj_dt( b ) == bli_obj_dt( c ) &&
	     //bli_obj_dt( c ) == ( num_t )comp_prec )
	     bli_obj_prec( c ) == comp_prec )
	{
		blas_gemm( transa, transb, bli_obj_dt( c ), alpha, a, b, beta, c );
		return;
	}

	num_t   dtc = bli_obj_dt( c );
	num_t   dta = bli_obj_dt( a );
	num_t   dtb = bli_obj_dt( b );

	dim_t   m = bli_obj_length( c );
	dim_t   n = bli_obj_width( c );
	dim_t   k = bli_obj_width_after_trans( a );

	obj_t*  ao = a;
	obj_t*  bo = b;
	obj_t*  co = c;

	num_t   targ_dt_c, targ_dt_a, targ_dt_b;
	dom_t   targ_dom_c, targ_dom_a, targ_dom_b;
	num_t   dt_comp;
	dom_t   comp_dom;
	obj_t   at, bt, ct;
	obj_t   ar,     cr;
	bool    needacc;
	bool    force_proj_a = FALSE;
	bool    force_proj_b = FALSE;

	

	if      (    bli_is_real( dtc ) &&    bli_is_real( dta ) &&    bli_is_real( dtb ) )
	{
		// rrr
		comp_dom = BLIS_REAL;
		targ_dom_c = BLIS_REAL; targ_dom_a = BLIS_REAL; targ_dom_b = BLIS_REAL;
		needacc = FALSE;
	}
	else if (    bli_is_real( dtc ) &&    bli_is_real( dta ) && bli_is_complex( dtb ) )
	{
		// rrc
		comp_dom = BLIS_REAL;
		targ_dom_c = BLIS_REAL; targ_dom_a = BLIS_REAL; targ_dom_b = BLIS_REAL;
		needacc = FALSE;
		force_proj_b = TRUE;
	}
	else if (    bli_is_real( dtc ) && bli_is_complex( dta ) &&    bli_is_real( dtb ) )
	{
		// rcr
		comp_dom = BLIS_REAL;
		targ_dom_c = BLIS_REAL; targ_dom_a = BLIS_REAL; targ_dom_b = BLIS_REAL;
		needacc = FALSE;
		force_proj_a = TRUE;
	}
	else if (    bli_is_real( dtc ) && bli_is_complex( dta ) && bli_is_complex( dtb ) )
	{
		// rcc
		comp_dom = BLIS_COMPLEX;
		targ_dom_c = BLIS_COMPLEX; targ_dom_a = BLIS_COMPLEX; targ_dom_b = BLIS_COMPLEX;
		needacc = TRUE;
	}
	else if ( bli_is_complex( dtc ) &&    bli_is_real( dta ) &&    bli_is_real( dtb ) )
	{
		// crr
		comp_dom = BLIS_REAL;
		targ_dom_c = BLIS_REAL; targ_dom_a = BLIS_REAL; targ_dom_b = BLIS_REAL;
		needacc = TRUE;
	}
	else if ( bli_is_complex( dtc ) &&    bli_is_real( dta ) && bli_is_complex( dtb ) )
	{
		// crc
		comp_dom = BLIS_COMPLEX;
		targ_dom_c = BLIS_COMPLEX; targ_dom_a = BLIS_COMPLEX; targ_dom_b = BLIS_COMPLEX;
		needacc = FALSE;
		force_proj_a = TRUE;
	}
	else if ( bli_is_complex( dtc ) && bli_is_complex( dta ) &&    bli_is_real( dtb ) )
	{
		// ccr
		comp_dom = BLIS_REAL;
		targ_dom_c = BLIS_COMPLEX; targ_dom_a = BLIS_COMPLEX; targ_dom_b = BLIS_REAL;
		needacc = FALSE;
	}
	else if ( bli_is_complex( dtc ) && bli_is_complex( dta ) && bli_is_complex( dtb ) )
	{
		// ccc
		comp_dom = BLIS_COMPLEX;
		targ_dom_c = BLIS_COMPLEX; targ_dom_a = BLIS_COMPLEX; targ_dom_b = BLIS_COMPLEX;
		needacc = FALSE;
	}
	else
	{
		comp_dom = BLIS_REAL;
		targ_dom_c = BLIS_REAL; targ_dom_a = BLIS_REAL; targ_dom_b = BLIS_REAL;
		needacc = FALSE;
	}

	// ----------------------------------------------------------------------------


	// Merge the computation domain with the computation precision.
	dt_comp = comp_dom | comp_prec;

	targ_dt_a = targ_dom_a | comp_prec;
	targ_dt_b = targ_dom_b | comp_prec;
	targ_dt_c = targ_dom_c | comp_prec;

	// Copy-cast A, if needed.
	if ( bli_dt_prec( dta ) != comp_prec || force_proj_a )
	{
		bli_obj_create( targ_dt_a, m, k, 0, 0, &at );
		bli_castm( ao, &at );
		ao = &at;
	}

	// Copy-cast B, if needed.
	if ( bli_dt_prec( dtb ) != comp_prec || force_proj_b )
	{
		bli_obj_create( targ_dt_b, k, n, 0, 0, &bt );
		bli_castm( bo, &bt );
		bo = &bt;
	}

	if ( bli_dt_prec( dtc ) != comp_prec )
	{
		needacc = TRUE;
	}

	// Copy-cast C, if needed.
	if ( needacc )
	{
		//bli_obj_create( dt_comp, m, n, 0, 0, &ct );
		bli_obj_create( targ_dt_c, m, n, 0, 0, &ct );
		bli_castm( c, &ct );
		co = &ct;
	}

	// ----------------------------------------------------------------------------

	if      (    bli_is_real( dtc ) &&    bli_is_real( dta ) &&    bli_is_real( dtb ) )
	{
	}
	else if (    bli_is_real( dtc ) &&    bli_is_real( dta ) && bli_is_complex( dtb ) )
	{
	}
	else if (    bli_is_real( dtc ) && bli_is_complex( dta ) &&    bli_is_real( dtb ) )
	{
	}
	else if (    bli_is_real( dtc ) && bli_is_complex( dta ) && bli_is_complex( dtb ) )
	{
	}
	else if ( bli_is_complex( dtc ) &&    bli_is_real( dta ) &&    bli_is_real( dtb ) )
	{
	}
	else if ( bli_is_complex( dtc ) &&    bli_is_real( dta ) && bli_is_complex( dtb ) )
	{
	}
	else if ( bli_is_complex( dtc ) && bli_is_complex( dta ) &&    bli_is_real( dtb ) )
	{
		inc_t rsa = bli_obj_row_stride( ao );
		inc_t csa = bli_obj_col_stride( ao );
		inc_t ma  = bli_obj_length( ao );
		inc_t na  = bli_obj_width( ao );
		siz_t ela = bli_obj_elem_size( ao );
		num_t dtap = bli_obj_dt_proj_to_real( ao );

		bli_obj_alias_to( ao, &ar ); ao = &ar;
		bli_obj_set_strides( rsa, 2*csa, ao );
		bli_obj_set_dims( 2*ma, na, ao );
		bli_obj_set_dt( dtap, ao );
		bli_obj_set_elem_size( ela/2, ao );

		inc_t rsc = bli_obj_row_stride( co );
		inc_t csc = bli_obj_col_stride( co );
		inc_t mc  = bli_obj_length( co );
		inc_t nc  = bli_obj_width( co );
		siz_t elc = bli_obj_elem_size( co );
		num_t dtcp = bli_obj_dt_proj_to_real( co );

		bli_obj_alias_to( co, &cr ); co = &cr;
		bli_obj_set_strides( rsc, 2*csc, co );
		bli_obj_set_dims( 2*mc, nc, co );
		bli_obj_set_dt( dtcp, co );
		bli_obj_set_elem_size( elc/2, co );
	}
	else if ( bli_is_complex( dtc ) && bli_is_complex( dta ) && bli_is_complex( dtb ) )
	{
	}
	else
	{
	}

	// ----------------------------------------------------------------------------


	// Call the BLAS.
	blas_gemm( transa, transb, dt_comp, alpha, ao, bo, beta, co );

	// Accumulate back to C, if needed.
	if ( needacc )
	{
		bli_castm( &ct, c );
	}


	if ( bli_dt_prec( dta ) != comp_prec || force_proj_a ) { bli_obj_free( &at ); }
	if ( bli_dt_prec( dtb ) != comp_prec || force_proj_b ) { bli_obj_free( &bt ); }
	if ( needacc )                                         { bli_obj_free( &ct ); }
}

void blas_gemm( trans_t transa, trans_t transb, num_t dt, obj_t* alpha, obj_t* a, obj_t* b, obj_t* beta, obj_t* c )
{
	char f77_transa = 'N';
	char f77_transb = 'N';

	//bli_param_map_blis_to_netlib_trans( transa, &f77_transa );
	//bli_param_map_blis_to_netlib_trans( transb, &f77_transb );

	if ( bli_is_float( dt ) )
	{
		f77_int  mm     = bli_obj_length( c );
		f77_int  kk     = bli_obj_width_after_trans( a );
		f77_int  nn     = bli_obj_width( c );
		f77_int  lda    = bli_obj_col_stride( a );
		f77_int  ldb    = bli_obj_col_stride( b );
		f77_int  ldc    = bli_obj_col_stride( c );
		float*   alphap = bli_obj_buffer_for_1x1( dt, alpha );
		float*   ap     = bli_obj_buffer( a );
		float*   bp     = bli_obj_buffer( b );
		float*   betap  = bli_obj_buffer_for_1x1( dt, beta );
		float*   cp     = bli_obj_buffer( c );

		sgemm_( &f77_transa,
		        &f77_transb,
		        &mm,
		        &nn,
		        &kk,
		        alphap,
		        ap, &lda,
		        bp, &ldb,
		        betap,
		        cp, &ldc );
	}
	else if ( bli_is_double( dt ) )
	{
		f77_int  mm     = bli_obj_length( c );
		f77_int  kk     = bli_obj_width_after_trans( a );
		f77_int  nn     = bli_obj_width( c );
		f77_int  lda    = bli_obj_col_stride( a );
		f77_int  ldb    = bli_obj_col_stride( b );
		f77_int  ldc    = bli_obj_col_stride( c );
		double*  alphap = bli_obj_buffer_for_1x1( dt, alpha );
		double*  ap     = bli_obj_buffer( a );
		double*  bp     = bli_obj_buffer( b );
		double*  betap  = bli_obj_buffer_for_1x1( dt, beta );
		double*  cp     = bli_obj_buffer( c );

		dgemm_( &f77_transa,
		        &f77_transb,
		        &mm,
		        &nn,
		        &kk,
		        alphap,
		        ap, &lda,
		        bp, &ldb,
		        betap,
		        cp, &ldc );
	}
	else if ( bli_is_scomplex( dt ) )
	{
		f77_int    mm     = bli_obj_length( c );
		f77_int    kk     = bli_obj_width_after_trans( a );
		f77_int    nn     = bli_obj_width( c );
		f77_int    lda    = bli_obj_col_stride( a );
		f77_int    ldb    = bli_obj_col_stride( b );
		f77_int    ldc    = bli_obj_col_stride( c );
		scomplex*  alphap = bli_obj_buffer_for_1x1( dt, alpha );
		scomplex*  ap     = bli_obj_buffer( a );
		scomplex*  bp     = bli_obj_buffer( b );
		scomplex*  betap  = bli_obj_buffer_for_1x1( dt, beta );
		scomplex*  cp     = bli_obj_buffer( c );

		cgemm_( &f77_transa,
		        &f77_transb,
		        &mm,
		        &nn,
		        &kk,
		        alphap,
		        ap, &lda,
		        bp, &ldb,
		        betap,
		        cp, &ldc );
	}
	else if ( bli_is_dcomplex( dt ) )
	{
		f77_int    mm     = bli_obj_length( c );
		f77_int    kk     = bli_obj_width_after_trans( a );
		f77_int    nn     = bli_obj_width( c );
		f77_int    lda    = bli_obj_col_stride( a );
		f77_int    ldb    = bli_obj_col_stride( b );
		f77_int    ldc    = bli_obj_col_stride( c );
		dcomplex*  alphap = bli_obj_buffer_for_1x1( dt, alpha );
		dcomplex*  ap     = bli_obj_buffer( a );
		dcomplex*  bp     = bli_obj_buffer( b );
		dcomplex*  betap  = bli_obj_buffer_for_1x1( dt, beta );
		dcomplex*  cp     = bli_obj_buffer( c );

		zgemm_( &f77_transa,
		        &f77_transb,
		        &mm,
		        &nn,
		        &kk,
		        alphap,
		        ap, &lda,
		        bp, &ldb,
		        betap,
		        cp, &ldc );
	}
}

