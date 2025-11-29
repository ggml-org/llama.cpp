/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2019, Advanced Micro Devices, Inc.

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


// --- BLIS to BLAS/LAPACK mappings --------------------------------------------

BLIS_EXPORT_BLIS void bli_param_map_blis_to_netlib_side( side_t side, char* blas_side );
BLIS_EXPORT_BLIS void bli_param_map_blis_to_netlib_uplo( uplo_t uplo, char* blas_uplo );
BLIS_EXPORT_BLIS void bli_param_map_blis_to_netlib_trans( trans_t trans, char* blas_trans );
BLIS_EXPORT_BLIS void bli_param_map_blis_to_netlib_diag( diag_t diag, char* blas_diag );
BLIS_EXPORT_BLIS void bli_param_map_blis_to_netlib_machval( machval_t machval, char* blas_machval );


// --- BLAS/LAPACK to BLIS mappings --------------------------------------------

// NOTE: These static functions were converted from regular functions in order
// to reduce function call overhead within the BLAS compatibility layer.

BLIS_INLINE void bli_param_map_netlib_to_blis_side( char side, side_t* blis_side )
{
	if      ( side == 'l' || side == 'L' ) *blis_side = BLIS_LEFT;
	else if ( side == 'r' || side == 'R' ) *blis_side = BLIS_RIGHT;
	else
	{
		// Instead of reporting an error to the framework, default to
		// an arbitrary value. This is needed because this function is
		// called by the BLAS compatibility layer AFTER it has already
		// checked errors and called xerbla(). If the application wants
		// to override the BLAS compatibility layer's xerbla--which
		// responds to errors with abort()--we need to also NOT call
		// abort() here, since either way it has already been dealt
		// with.
		//bli_check_error_code( BLIS_INVALID_SIDE );
		*blis_side = BLIS_LEFT;
	}
}

BLIS_INLINE void bli_param_map_netlib_to_blis_uplo( char uplo, uplo_t* blis_uplo )
{
	if      ( uplo == 'l' || uplo == 'L' ) *blis_uplo = BLIS_LOWER;
	else if ( uplo == 'u' || uplo == 'U' ) *blis_uplo = BLIS_UPPER;
	else
	{
		// See comment for bli_param_map_netlib_to_blis_side() above.
		//bli_check_error_code( BLIS_INVALID_UPLO );
		*blis_uplo = BLIS_LOWER;
	}
}

BLIS_INLINE void bli_param_map_netlib_to_blis_trans( char trans, trans_t* blis_trans )
{
	if      ( trans == 'n' || trans == 'N' ) *blis_trans = BLIS_NO_TRANSPOSE;
	else if ( trans == 't' || trans == 'T' ) *blis_trans = BLIS_TRANSPOSE;
	else if ( trans == 'c' || trans == 'C' ) *blis_trans = BLIS_CONJ_TRANSPOSE;
	else
	{
		// See comment for bli_param_map_netlib_to_blis_side() above.
		//bli_check_error_code( BLIS_INVALID_TRANS );
		*blis_trans = BLIS_NO_TRANSPOSE;
	}
}

BLIS_INLINE void bli_param_map_netlib_to_blis_diag( char diag, diag_t* blis_diag )
{
	if      ( diag == 'n' || diag == 'N' ) *blis_diag = BLIS_NONUNIT_DIAG;
	else if ( diag == 'u' || diag == 'U' ) *blis_diag = BLIS_UNIT_DIAG;
	else
	{
		// See comment for bli_param_map_netlib_to_blis_side() above.
		//bli_check_error_code( BLIS_INVALID_DIAG );
		*blis_diag = BLIS_NONUNIT_DIAG;
	}
}


// --- BLIS char to BLIS mappings ----------------------------------------------

BLIS_EXPORT_BLIS void bli_param_map_char_to_blis_side( char side, side_t* blis_side );
BLIS_EXPORT_BLIS void bli_param_map_char_to_blis_uplo( char uplo, uplo_t* blis_uplo );
BLIS_EXPORT_BLIS void bli_param_map_char_to_blis_trans( char trans, trans_t* blis_trans );
BLIS_EXPORT_BLIS void bli_param_map_char_to_blis_conj( char conj, conj_t* blis_conj );
BLIS_EXPORT_BLIS void bli_param_map_char_to_blis_diag( char diag, diag_t* blis_diag );
BLIS_EXPORT_BLIS void bli_param_map_char_to_blis_dt( char dt, num_t* blis_dt );


// --- BLIS to BLIS char mappings ----------------------------------------------

BLIS_EXPORT_BLIS void bli_param_map_blis_to_char_side( side_t blis_side, char* side );
BLIS_EXPORT_BLIS void bli_param_map_blis_to_char_uplo( uplo_t blis_uplo, char* uplo );
BLIS_EXPORT_BLIS void bli_param_map_blis_to_char_trans( trans_t blis_trans, char* trans );
BLIS_EXPORT_BLIS void bli_param_map_blis_to_char_conj( conj_t blis_conj, char* conj );
BLIS_EXPORT_BLIS void bli_param_map_blis_to_char_diag( diag_t blis_diag, char* diag );
BLIS_EXPORT_BLIS void bli_param_map_blis_to_char_dt( num_t blis_dt, char* dt );

