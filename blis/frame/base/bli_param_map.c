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

#include "blis.h"

// --- BLIS to BLAS/LAPACK mappings --------------------------------------------

void bli_param_map_blis_to_netlib_side( side_t side, char* blas_side )
{
	if      ( side == BLIS_LEFT  ) *blas_side = 'L';
	else if ( side == BLIS_RIGHT ) *blas_side = 'R';
	else
	{
		bli_check_error_code( BLIS_INVALID_SIDE );
	}
}

void bli_param_map_blis_to_netlib_uplo( uplo_t uplo, char* blas_uplo )
{
	if      ( uplo == BLIS_LOWER ) *blas_uplo = 'L';
	else if ( uplo == BLIS_UPPER ) *blas_uplo = 'U';
	else
	{
		bli_check_error_code( BLIS_INVALID_UPLO );
	}
}

void bli_param_map_blis_to_netlib_trans( trans_t trans, char* blas_trans )
{
	if      ( trans == BLIS_NO_TRANSPOSE   ) *blas_trans = 'N';
	else if ( trans == BLIS_TRANSPOSE      ) *blas_trans = 'T';
	else if ( trans == BLIS_CONJ_TRANSPOSE ) *blas_trans = 'C';
	else
	{
		bli_check_error_code( BLIS_INVALID_TRANS );
	}
}

void bli_param_map_blis_to_netlib_diag( diag_t diag, char* blas_diag )
{
	if      ( diag == BLIS_NONUNIT_DIAG ) *blas_diag = 'N';
	else if ( diag == BLIS_UNIT_DIAG    ) *blas_diag = 'U';
	else
	{
		bli_check_error_code( BLIS_INVALID_DIAG );
	}
}

void bli_param_map_blis_to_netlib_machval( machval_t machval, char* blas_machval )
{
	if      ( machval == BLIS_MACH_EPS      ) *blas_machval = 'E';
	else if ( machval == BLIS_MACH_SFMIN    ) *blas_machval = 'S';
	else if ( machval == BLIS_MACH_BASE     ) *blas_machval = 'B';
	else if ( machval == BLIS_MACH_PREC     ) *blas_machval = 'P';
	else if ( machval == BLIS_MACH_NDIGMANT ) *blas_machval = 'N';
	else if ( machval == BLIS_MACH_RND      ) *blas_machval = 'R';
	else if ( machval == BLIS_MACH_EMIN     ) *blas_machval = 'M';
	else if ( machval == BLIS_MACH_RMIN     ) *blas_machval = 'U';
	else if ( machval == BLIS_MACH_EMAX     ) *blas_machval = 'L';
	else if ( machval == BLIS_MACH_RMAX     ) *blas_machval = 'O';
	else
	{
		bli_check_error_code( BLIS_INVALID_MACHVAL );
	}
}


// --- BLAS/LAPACK to BLIS mappings --------------------------------------------

// NOTE: These functions were converted into static functions. Please see this
// file's corresponding header for those definitions.


// --- BLIS char to BLIS mappings ----------------------------------------------

void bli_param_map_char_to_blis_side( char side, side_t* blis_side )
{
	if      ( side == 'l' || side == 'L' ) *blis_side = BLIS_LEFT;
	else if ( side == 'r' || side == 'R' ) *blis_side = BLIS_RIGHT;
	else
	{
		bli_check_error_code( BLIS_INVALID_SIDE );
	}
}

void bli_param_map_char_to_blis_uplo( char uplo, uplo_t* blis_uplo )
{
	if      ( uplo == 'l' || uplo == 'L' ) *blis_uplo = BLIS_LOWER;
	else if ( uplo == 'u' || uplo == 'U' ) *blis_uplo = BLIS_UPPER;
	else if ( uplo == 'e' || uplo == 'E' ) *blis_uplo = BLIS_DENSE;
	else
	{
		bli_check_error_code( BLIS_INVALID_UPLO );
	}
}

void bli_param_map_char_to_blis_trans( char trans, trans_t* blis_trans )
{
	if      ( trans == 'n' || trans == 'N' ) *blis_trans = BLIS_NO_TRANSPOSE;
	else if ( trans == 't' || trans == 'T' ) *blis_trans = BLIS_TRANSPOSE;
	else if ( trans == 'c' || trans == 'C' ) *blis_trans = BLIS_CONJ_NO_TRANSPOSE;
	else if ( trans == 'h' || trans == 'H' ) *blis_trans = BLIS_CONJ_TRANSPOSE;
	else
	{
		bli_check_error_code( BLIS_INVALID_TRANS );
	}
}

void bli_param_map_char_to_blis_conj( char conj, conj_t* blis_conj )
{
	if      ( conj == 'n' || conj == 'N' ) *blis_conj = BLIS_NO_CONJUGATE;
	else if ( conj == 'c' || conj == 'C' ) *blis_conj = BLIS_CONJUGATE;
	else
	{
		bli_check_error_code( BLIS_INVALID_CONJ );
	}
}

void bli_param_map_char_to_blis_diag( char diag, diag_t* blis_diag )
{
	if      ( diag == 'n' || diag == 'N' ) *blis_diag = BLIS_NONUNIT_DIAG;
	else if ( diag == 'u' || diag == 'U' ) *blis_diag = BLIS_UNIT_DIAG;
	else
	{
		bli_check_error_code( BLIS_INVALID_DIAG );
	}
}

void bli_param_map_char_to_blis_dt( char dt, num_t* blis_dt )
{
	if      ( dt == 's' ) *blis_dt = BLIS_FLOAT;
	else if ( dt == 'd' ) *blis_dt = BLIS_DOUBLE;
	else if ( dt == 'c' ) *blis_dt = BLIS_SCOMPLEX;
	else if ( dt == 'z' ) *blis_dt = BLIS_DCOMPLEX;
	else if ( dt == 'i' ) *blis_dt = BLIS_INT;
	else
	{
		bli_check_error_code( BLIS_INVALID_DATATYPE );
	}
}


// --- BLIS to BLIS char mappings ----------------------------------------------

void bli_param_map_blis_to_char_side( side_t blis_side, char* side )
{
	if      ( blis_side == BLIS_LEFT  ) *side = 'l';
	else if ( blis_side == BLIS_RIGHT ) *side = 'r';
	else
	{
		bli_check_error_code( BLIS_INVALID_SIDE );
	}
}

void bli_param_map_blis_to_char_uplo( uplo_t blis_uplo, char* uplo )
{
	if      ( blis_uplo == BLIS_LOWER ) *uplo = 'l';
	else if ( blis_uplo == BLIS_UPPER ) *uplo = 'u';
	else
	{
		bli_check_error_code( BLIS_INVALID_UPLO );
	}
}

void bli_param_map_blis_to_char_trans( trans_t blis_trans, char* trans )
{
	if      ( blis_trans == BLIS_NO_TRANSPOSE      ) *trans = 'n';
	else if ( blis_trans == BLIS_TRANSPOSE         ) *trans = 't';
	else if ( blis_trans == BLIS_CONJ_NO_TRANSPOSE ) *trans = 'c';
	else if ( blis_trans == BLIS_CONJ_TRANSPOSE    ) *trans = 'h';
	else
	{
		bli_check_error_code( BLIS_INVALID_TRANS );
	}
}

void bli_param_map_blis_to_char_conj( conj_t blis_conj, char* conj )
{
	if      ( blis_conj == BLIS_NO_CONJUGATE ) *conj = 'n';
	else if ( blis_conj == BLIS_CONJUGATE    ) *conj = 'c';
	else
	{
		bli_check_error_code( BLIS_INVALID_CONJ );
	}
}

void bli_param_map_blis_to_char_diag( diag_t blis_diag, char* diag )
{
	if      ( blis_diag == BLIS_NONUNIT_DIAG ) *diag = 'n';
	else if ( blis_diag == BLIS_UNIT_DIAG    ) *diag = 'u';
	else
	{
		bli_check_error_code( BLIS_INVALID_DIAG );
	}
}

void bli_param_map_blis_to_char_dt( num_t blis_dt, char* dt )
{
	if      ( blis_dt == BLIS_FLOAT    ) *dt = 's';
	else if ( blis_dt == BLIS_DOUBLE   ) *dt = 'd';
	else if ( blis_dt == BLIS_SCOMPLEX ) *dt = 'c';
	else if ( blis_dt == BLIS_DCOMPLEX ) *dt = 'z';
	else if ( blis_dt == BLIS_INT      ) *dt = 'i';
	else
	{
		bli_check_error_code( BLIS_INVALID_DATATYPE );
	}
}

