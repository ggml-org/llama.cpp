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

#include "blis.h"

// Internal array to hold error strings.
static const char *bli_error_string[-BLIS_ERROR_CODE_MAX] =
{
	[-BLIS_INVALID_ERROR_CHECKING_LEVEL]         = "Invalid error checking level.",
	[-BLIS_UNDEFINED_ERROR_CODE]                 = "Undefined error code.",
	[-BLIS_NULL_POINTER]                         = "Encountered unexpected null pointer.",
	[-BLIS_NOT_YET_IMPLEMENTED]                  = "Requested functionality not yet implemented.",
	[-BLIS_OUT_OF_BOUNDS]                        = "Requested index is out of bounds.",
	[-BLIS_LOCK_FAILURE]                         = "Failed to obtain lock.",

	[-BLIS_INVALID_SIDE]                         = "Invalid side parameter value.",
	[-BLIS_INVALID_UPLO]                         = "Invalid uplo_t parameter value.",
	[-BLIS_INVALID_TRANS]                        = "Invalid trans_t parameter value.",
	[-BLIS_INVALID_CONJ]                         = "Invalid conj_t parameter value.",
	[-BLIS_INVALID_DIAG]                         = "Invalid diag_t parameter value.",
	[-BLIS_EXPECTED_NONUNIT_DIAG]                = "Expected object with non-unit diagonal.",

	[-BLIS_INVALID_DATATYPE]                     = "Invalid datatype value.",
	[-BLIS_EXPECTED_FLOATING_POINT_DATATYPE]     = "Expected floating-point datatype value.",
	[-BLIS_EXPECTED_NONINTEGER_DATATYPE]         = "Expected non-integer datatype value.",
	[-BLIS_EXPECTED_NONCONSTANT_DATATYPE]        = "Expected non-constant datatype value.",
	[-BLIS_EXPECTED_REAL_DATATYPE]               = "Expected real datatype value.",
	[-BLIS_EXPECTED_INTEGER_DATATYPE]            = "Expected integer datatype value.",
	[-BLIS_INCONSISTENT_DATATYPES]               = "Expected consistent datatypes (equal, or one being constant).",
	[-BLIS_EXPECTED_REAL_PROJ_OF]                = "Expected second datatype to be real projection of first.",
	[-BLIS_EXPECTED_REAL_VALUED_OBJECT]          = "Expected real-valued object (ie: if complex, imaginary component equals zero).",
	[-BLIS_INCONSISTENT_PRECISIONS]              = "Expected consistent precisions (both single or both double).",

	[-BLIS_NONCONFORMAL_DIMENSIONS]              = "Encountered non-conformal dimensions between objects.",
	[-BLIS_EXPECTED_SCALAR_OBJECT]               = "Expected scalar object.",
	[-BLIS_EXPECTED_VECTOR_OBJECT]               = "Expected vector object.",
	[-BLIS_UNEQUAL_VECTOR_LENGTHS]               = "Encountered unequal vector lengths.",
	[-BLIS_EXPECTED_SQUARE_OBJECT]               = "Expected square object.",
	[-BLIS_UNEXPECTED_OBJECT_LENGTH]             = "Unexpected object length.",
	[-BLIS_UNEXPECTED_OBJECT_WIDTH]              = "Unexpected object width.",
	[-BLIS_UNEXPECTED_VECTOR_DIM]                = "Unexpected vector dimension.",
	[-BLIS_UNEXPECTED_DIAG_OFFSET]               = "Unexpected object diagonal offset.",
	[-BLIS_NEGATIVE_DIMENSION]                   = "Encountered negative dimension.",

	[-BLIS_INVALID_ROW_STRIDE]                   = "Encountered invalid row stride relative to n dimension.",
	[-BLIS_INVALID_COL_STRIDE]                   = "Encountered invalid col stride relative to m dimension.",
	[-BLIS_INVALID_DIM_STRIDE_COMBINATION]       = "Encountered invalid stride/dimension combination.",

	[-BLIS_EXPECTED_GENERAL_OBJECT]              = "Expected general object.",
	[-BLIS_EXPECTED_HERMITIAN_OBJECT]            = "Expected Hermitian object.",
	[-BLIS_EXPECTED_SYMMETRIC_OBJECT]            = "Expected symmetric object.",
	[-BLIS_EXPECTED_TRIANGULAR_OBJECT]           = "Expected triangular object.",

	[-BLIS_EXPECTED_UPPER_OR_LOWER_OBJECT]       = "Expected upper or lower triangular object.",

	[-BLIS_INVALID_3x1_SUBPART]                  = "Encountered invalid 3x1 (vertical) subpartition label.",
	[-BLIS_INVALID_1x3_SUBPART]                  = "Encountered invalid 1x3 (horizontal) subpartition label.",
	[-BLIS_INVALID_3x3_SUBPART]                  = "Encountered invalid 3x3 (diagonal) subpartition label.",

	[-BLIS_UNEXPECTED_NULL_CONTROL_TREE]         = "Encountered unexpected null control tree node.",

	[-BLIS_PACK_SCHEMA_NOT_SUPPORTED_FOR_UNPACK] = "Pack schema not yet supported/implemented for use with unpacking.",
	[-BLIS_PACK_SCHEMA_NOT_SUPPORTED_FOR_PART]   = "Pack schema not yet supported/implemented for use with partitioning.",

	[-BLIS_EXPECTED_NONNULL_OBJECT_BUFFER]       = "Encountered object with non-zero dimensions containing null buffer.",

	[-BLIS_MALLOC_RETURNED_NULL]                 = "malloc() returned NULL; heap memory is likely exhausted.",

	[-BLIS_INVALID_PACKBUF]                      = "Invalid packbuf_t value.",
	[-BLIS_EXHAUSTED_CONTIG_MEMORY_POOL]         = "Attempted to allocate more memory from contiguous pool than is available.",
	[-BLIS_INSUFFICIENT_STACK_BUF_SIZE]          = "Configured maximum stack buffer size is insufficient for register blocksizes currently in use.",
	[-BLIS_ALIGNMENT_NOT_POWER_OF_TWO]           = "Encountered memory alignment value that is either zero or not a power of two.",
	[-BLIS_ALIGNMENT_NOT_MULT_OF_PTR_SIZE]       = "Encountered memory alignment value that is not a multiple of sizeof(void*).",

	[-BLIS_EXPECTED_OBJECT_ALIAS]                = "Expected object to be alias.",

	[-BLIS_INVALID_ARCH_ID]                      = "Invalid architecture id value.",
	[-BLIS_UNINITIALIZED_GKS_CNTX]               = "Accessed uninitialized context in gks; BLIS_ARCH_TYPE is probably set to an invalid architecture id.",

	[-BLIS_MC_DEF_NONMULTIPLE_OF_MR]             = "Default MC is non-multiple of MR for one or more datatypes.",
	[-BLIS_MC_MAX_NONMULTIPLE_OF_MR]             = "Maximum MC is non-multiple of MR for one or more datatypes.",
	[-BLIS_NC_DEF_NONMULTIPLE_OF_NR]             = "Default NC is non-multiple of NR for one or more datatypes.",
	[-BLIS_NC_MAX_NONMULTIPLE_OF_NR]             = "Maximum NC is non-multiple of NR for one or more datatypes.",
	[-BLIS_KC_DEF_NONMULTIPLE_OF_KR]             = "Default KC is non-multiple of KR for one or more datatypes.",
	[-BLIS_KC_MAX_NONMULTIPLE_OF_KR]             = "Maximum KC is non-multiple of KR for one or more datatypes.",
	[-BLIS_MR_NOT_EVEN_FOR_REAL_TYPE]            = "MR is not an even number for one or more real-domain datatypes.",
	[-BLIS_PACKMR_NOT_EVEN_FOR_REAL_TYPE]        = "PACKMR is not an even number for one or more real-domain datatypes.",
	[-BLIS_NR_NOT_EVEN_FOR_REAL_TYPE]            = "NR is not an even number for one or more real-domain datatypes.",
	[-BLIS_PACKNR_NOT_EVEN_FOR_REAL_TYPE]        = "PACKNR is not an even number for one or more real-domain datatypes.",
};

// -----------------------------------------------------------------------------

void bli_print_msg( const char* str, const char* file, guint_t line )
{
	fprintf( stderr, "\n" );
	fprintf( stderr, "libblis: %s (line %lu):\n", file, ( long unsigned int )line );
	fprintf( stderr, "libblis: %s\n", str );
	fflush( stderr );
}

void bli_abort( void )
{
	fprintf( stderr, "libblis: Aborting.\n" );
	//raise( SIGABRT );
	abort();
}

// -----------------------------------------------------------------------------

// Current error checking level.
static BLIS_THREAD_LOCAL errlev_t bli_err_chk_level = BLIS_FULL_ERROR_CHECKING;

errlev_t bli_error_checking_level( void )
{
	return bli_err_chk_level;
}

void bli_error_checking_level_set( errlev_t new_level )
{
	err_t e_val;

	e_val = bli_check_valid_error_level( new_level );
	bli_check_error_code( e_val );

	bli_err_chk_level = new_level;
}

bool bli_error_checking_is_enabled( void )
{
	return bli_error_checking_level() != BLIS_NO_ERROR_CHECKING;
}

const char* bli_error_string_for_code( gint_t code )
{
	return bli_error_string[-code];
}

