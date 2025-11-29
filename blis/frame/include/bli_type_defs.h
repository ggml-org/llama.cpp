/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2016, Hewlett Packard Enterprise Development LP
   Copyright (C) 2020, Advanced Micro Devices, Inc.

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

#ifndef BLIS_TYPE_DEFS_H
#define BLIS_TYPE_DEFS_H


//
// -- BLIS basic types ---------------------------------------------------------
//

#ifdef __cplusplus
  // For C++, include stdint.h.
  #include <cstdint>
#elif __STDC_VERSION__ >= 199901L
  // For C99 (or later), include stdint.h.
  #include <stddef.h>
  #include <stdint.h>
  #include <stdbool.h>
#else
  // When stdint.h is not available, manually typedef the types we will use.
  #ifdef _WIN32
    typedef          __int32  int32_t;
    typedef unsigned __int32 uint32_t;
    typedef          __int64  int64_t;
    typedef unsigned __int64 uint64_t;
  #else
    #error "Attempting to compile on pre-C99 system without stdint.h."
  #endif
#endif

// -- General-purpose integers --

// If BLAS integers are 64 bits, mandate that BLIS integers also be 64 bits.
// NOTE: This cpp guard will only meaningfully change BLIS's behavior on
// systems where the BLIS integer size would have been automatically selected
// to be 32 bits, since explicit selection of 32 bits is prohibited at
// configure-time (and explicit or automatic selection of 64 bits is fine
// and would have had the same result).
#if BLIS_BLAS_INT_TYPE_SIZE == 64
  #undef  BLIS_INT_TYPE_SIZE
  #define BLIS_INT_TYPE_SIZE 64
#endif

// Define integer types depending on what size integer was requested.
#if   BLIS_INT_TYPE_SIZE == 32
typedef           int32_t  gint_t;
typedef          uint32_t guint_t;
#elif BLIS_INT_TYPE_SIZE == 64
typedef           int64_t  gint_t;
typedef          uint64_t guint_t;
#else
typedef   signed long int  gint_t;
typedef unsigned long int guint_t;
#endif

// -- Boolean type --

// NOTE: bool_t is no longer used and has been replaced with C99's bool type.
//typedef bool bool_t;

// BLIS uses TRUE and FALSE macro constants as possible boolean values, but we
// define these macros in terms of true and false, respectively, which are
// defined by C99 in stdbool.h.
#ifndef TRUE
  #define TRUE  true
#endif

#ifndef FALSE
  #define FALSE false
#endif

// -- Special-purpose integers --

// This cpp guard provides a temporary hack to allow libflame
// interoperability with BLIS.
#ifndef _DEFINED_DIM_T
#define _DEFINED_DIM_T
typedef   gint_t dim_t;      // dimension type
#endif
typedef   gint_t inc_t;      // increment/stride type
typedef   gint_t doff_t;     // diagonal offset type
typedef  guint_t siz_t;      // byte size type
typedef uint32_t objbits_t;  // object information bit field

// -- Real types --

// Define the number of floating-point types supported, and the size of the
// largest type.
#define BLIS_NUM_FP_TYPES   4
#define BLIS_MAX_TYPE_SIZE  sizeof(dcomplex)

// There are some places where we need to use sizeof() inside of a C
// preprocessor #if conditional, and so here we define the various sizes
// for those purposes.
#define BLIS_SIZEOF_S      4  // sizeof(float)
#define BLIS_SIZEOF_D      8  // sizeof(double)
#define BLIS_SIZEOF_C      8  // sizeof(scomplex)
#define BLIS_SIZEOF_Z      16 // sizeof(dcomplex)

// -- Complex types --

#if defined(__cplusplus) && defined(BLIS_ENABLE_STD_COMPLEX)

	} //extern "C"

	#include <complex>

	// Typedef official C++ complex types to BLIS complex type names.

	// This cpp guard provides a temporary hack to allow libflame
	// interoperability with BLIS.
	#ifndef _DEFINED_SCOMPLEX
	#define _DEFINED_SCOMPLEX
	typedef std::complex<float> scomplex;
	#endif

	// This cpp guard provides a temporary hack to allow libflame
	// interoperability with BLIS.
	#ifndef _DEFINED_DCOMPLEX
	#define _DEFINED_DCOMPLEX
	typedef std::complex<double> dcomplex;
	#endif

	extern "C"
	{

#elif defined(BLIS_ENABLE_C99_COMPLEX)

	#if __STDC_VERSION__ >= 199901L
		#include <complex.h>

		// Typedef official C99 complex types to BLIS complex type names.

		// This cpp guard provides a temporary hack to allow libflame
		// interoperability with BLIS.
		#ifndef _DEFINED_SCOMPLEX
		#define _DEFINED_SCOMPLEX
		typedef float complex scomplex;
		#endif

		// This cpp guard provides a temporary hack to allow libflame
		// interoperability with BLIS.
		#ifndef _DEFINED_DCOMPLEX
		#define _DEFINED_DCOMPLEX
		typedef double complex dcomplex;
		#endif
	#else
		#error "Configuration requested C99 complex types, but C99 does not appear to be supported."
	#endif

#else // ifndef BLIS_ENABLE_C99_COMPLEX

	// This cpp guard provides a temporary hack to allow libflame
	// interoperability with BLIS.
	#ifndef _DEFINED_SCOMPLEX
	#define _DEFINED_SCOMPLEX
	typedef struct scomplex
	{
		float  real;
		float  imag;
	} scomplex;
	#endif

	// This cpp guard provides a temporary hack to allow libflame
	// interoperability with BLIS.
	#ifndef _DEFINED_DCOMPLEX
	#define _DEFINED_DCOMPLEX
	typedef struct dcomplex
	{
		double real;
		double imag;
	} dcomplex;
	#endif

#endif // BLIS_ENABLE_C99_COMPLEX

// -- Atom type --

// Note: atom types are used to hold "bufferless" scalar object values. Note
// that it needs to be as large as the largest possible scalar value we might
// want to hold. Thus, for now, it is a dcomplex.
typedef dcomplex atom_t;

// -- Fortran-77 types --

// Note: These types are typically only used by BLAS compatibility layer, but
// we must define them even when the compatibility layer isn't being built
// because they also occur in bli_slamch() and bli_dlamch().

// Define f77_int depending on what size of integer was requested.
#if   BLIS_BLAS_INT_TYPE_SIZE == 32
typedef int32_t   f77_int;
#elif BLIS_BLAS_INT_TYPE_SIZE == 64
typedef int64_t   f77_int;
#else
typedef long int  f77_int;
#endif

typedef char      f77_char;
typedef float     f77_float;
typedef double    f77_double;
typedef scomplex  f77_scomplex;
typedef dcomplex  f77_dcomplex;

// -- Misc. function pointer types --

// Note: This type should be used in any situation where the address of a
// *function* will be conveyed or stored prior to it being typecast back
// to the correct function type. It does not need to be used when conveying
// or storing the address of *data* (such as an array of float or double).
//typedef void (*void_fp)( void );
typedef void* void_fp;

// Typedef function pointer types for malloc() and free() substitutes.
typedef void* (*malloc_ft)( size_t size );
typedef void  (*free_ft)  ( void*  p    );


//
// -- BLIS info bit field sizes ------------------------------------------------
//

#define BLIS_DATATYPE_NUM_BITS             ( BLIS_DOMAIN_NUM_BITS + BLIS_PRECISION_NUM_BITS )
#define   BLIS_DOMAIN_NUM_BITS               1
#define   BLIS_PRECISION_NUM_BITS            2
#define BLIS_CONJTRANS_NUM_BITS            ( BLIS_TRANS_NUM_BITS + BLIS_CONJ_NUM_BITS )
#define   BLIS_TRANS_NUM_BITS                1
#define   BLIS_CONJ_NUM_BITS                 1
#define BLIS_UPLO_NUM_BITS                 ( BLIS_UPPER_NUM_BITS + BLIS_DIAG_NUM_BITS + BLIS_LOWER_NUM_BITS )
#define   BLIS_UPPER_NUM_BITS                1
#define   BLIS_DIAG_NUM_BITS                 1
#define   BLIS_LOWER_NUM_BITS                1
#define BLIS_UNIT_DIAG_NUM_BITS            1
#define BLIS_INVERT_DIAG_NUM_BITS          1
#define BLIS_PACK_SCHEMA_NUM_BITS          ( BLIS_PACK_PANEL_NUM_BITS + BLIS_PACK_FORMAT_NUM_BITS + BLIS_PACK_NUM_BITS )
#define   BLIS_PACK_PANEL_NUM_BITS           1
#define   BLIS_PACK_FORMAT_NUM_BITS          4
#define   BLIS_PACK_NUM_BITS                 1
#define BLIS_PACK_REV_IF_UPPER_NUM_BITS    1
#define BLIS_PACK_REV_IF_LOWER_NUM_BITS    1
#define BLIS_PACK_BUFFER_NUM_BITS          2
#define BLIS_STRUC_NUM_BITS                2


//
// -- BLIS info bit field offsets ----------------------------------------------
//

#define BLIS_DATATYPE_SHIFT                0
#define   BLIS_DOMAIN_SHIFT              (   BLIS_DATATYPE_SHIFT )
#define   BLIS_PRECISION_SHIFT           (   BLIS_DOMAIN_SHIFT + BLIS_DOMAIN_NUM_BITS )
#define BLIS_CONJTRANS_SHIFT             ( BLIS_DATATYPE_SHIFT + BLIS_DATATYPE_NUM_BITS )
#define   BLIS_TRANS_SHIFT               (   BLIS_CONJTRANS_SHIFT )
#define   BLIS_CONJ_SHIFT                (   BLIS_TRANS_SHIFT + BLIS_TRANS_NUM_BITS )
#define BLIS_UPLO_SHIFT                  ( BLIS_CONJTRANS_SHIFT + BLIS_CONJTRANS_NUM_BITS )
#define   BLIS_UPPER_SHIFT               (   BLIS_UPLO_SHIFT )
#define   BLIS_DIAG_SHIFT                (   BLIS_UPPER_SHIFT + BLIS_UPPER_NUM_BITS )
#define   BLIS_LOWER_SHIFT               (   BLIS_DIAG_SHIFT + BLIS_DIAG_NUM_BITS )
#define BLIS_UNIT_DIAG_SHIFT             ( BLIS_UPLO_SHIFT + BLIS_UPLO_NUM_BITS )
#define BLIS_INVERT_DIAG_SHIFT           ( BLIS_UNIT_DIAG_SHIFT + BLIS_UNIT_DIAG_NUM_BITS )
#define BLIS_PACK_SCHEMA_SHIFT           ( BLIS_INVERT_DIAG_SHIFT + BLIS_INVERT_DIAG_NUM_BITS )
#define   BLIS_PACK_PANEL_SHIFT          (   BLIS_PACK_SCHEMA_SHIFT )
#define   BLIS_PACK_FORMAT_SHIFT         (   BLIS_PACK_PANEL_SHIFT + BLIS_PACK_PANEL_NUM_BITS )
#define   BLIS_PACK_SHIFT                (   BLIS_PACK_FORMAT_SHIFT + BLIS_PACK_FORMAT_NUM_BITS )
#define BLIS_PACK_REV_IF_UPPER_SHIFT     ( BLIS_PACK_SCHEMA_SHIFT + BLIS_PACK_SCHEMA_NUM_BITS )
#define BLIS_PACK_REV_IF_LOWER_SHIFT     ( BLIS_PACK_REV_IF_UPPER_SHIFT + BLIS_PACK_REV_IF_UPPER_NUM_BITS )
#define BLIS_PACK_BUFFER_SHIFT           ( BLIS_PACK_REV_IF_LOWER_SHIFT + BLIS_PACK_REV_IF_LOWER_NUM_BITS )
#define BLIS_STRUC_SHIFT                 ( BLIS_PACK_BUFFER_SHIFT + BLIS_PACK_BUFFER_NUM_BITS )
#define BLIS_COMP_PREC_SHIFT             ( BLIS_STRUC_SHIFT + BLIS_STRUC_NUM_BITS )
#define BLIS_SCALAR_DT_SHIFT             ( BLIS_COMP_PREC_SHIFT + BLIS_PRECISION_NUM_BITS )
#define   BLIS_SCALAR_DOMAIN_SHIFT       (   BLIS_SCALAR_DT_SHIFT )
#define   BLIS_SCALAR_PREC_SHIFT         (   BLIS_SCALAR_DOMAIN_SHIFT + BLIS_DOMAIN_NUM_BITS )
// This is the total number of bits, which should always be <= 32
#define BLIS_INFO_NUM_BITS               ( BLIS_SCALAR_DT_SHIFT + BLIS_DATATYPE_NUM_BITS )

//
// -- BLIS info bit field masks ------------------------------------------------
//

#define BLIS_DATATYPE_BITS                 ( ( ( 1 << BLIS_DATATYPE_NUM_BITS          ) - 1 ) << BLIS_DATATYPE_SHIFT )
#define   BLIS_DOMAIN_BIT                  ( ( ( 1 << BLIS_DOMAIN_NUM_BITS            ) - 1 ) << BLIS_DOMAIN_SHIFT )
#define   BLIS_PRECISION_BIT               ( ( ( 1 << BLIS_PRECISION_NUM_BITS         ) - 1 ) << BLIS_PRECISION_SHIFT )
#define BLIS_CONJTRANS_BITS                ( ( ( 1 << BLIS_CONJTRANS_NUM_BITS         ) - 1 ) << BLIS_CONJTRANS_SHIFT )
#define   BLIS_TRANS_BIT                   ( ( ( 1 << BLIS_TRANS_NUM_BITS             ) - 1 ) << BLIS_TRANS_SHIFT )
#define   BLIS_CONJ_BIT                    ( ( ( 1 << BLIS_CONJ_NUM_BITS              ) - 1 ) << BLIS_CONJ_SHIFT )
#define BLIS_UPLO_BITS                     ( ( ( 1 << BLIS_UPLO_NUM_BITS              ) - 1 ) << BLIS_UPLO_SHIFT )
#define   BLIS_UPPER_BIT                   ( ( ( 1 << BLIS_UPPER_NUM_BITS             ) - 1 ) << BLIS_UPPER_SHIFT )
#define   BLIS_DIAG_BIT                    ( ( ( 1 << BLIS_DIAG_NUM_BITS              ) - 1 ) << BLIS_DIAG_SHIFT )
#define   BLIS_LOWER_BIT                   ( ( ( 1 << BLIS_LOWER_NUM_BITS             ) - 1 ) << BLIS_LOWER_SHIFT )
#define BLIS_UNIT_DIAG_BIT                 ( ( ( 1 << BLIS_UNIT_DIAG_NUM_BITS         ) - 1 ) << BLIS_UNIT_DIAG_SHIFT )
#define BLIS_INVERT_DIAG_BIT               ( ( ( 1 << BLIS_INVERT_DIAG_NUM_BITS       ) - 1 ) << BLIS_INVERT_DIAG_SHIFT )
#define BLIS_PACK_SCHEMA_BITS              ( ( ( 1 << BLIS_PACK_SCHEMA_NUM_BITS       ) - 1 ) << BLIS_PACK_SCHEMA_SHIFT )
#define   BLIS_PACK_PANEL_BIT              ( ( ( 1 << BLIS_PACK_PANEL_NUM_BITS        ) - 1 ) << BLIS_PACK_PANEL_SHIFT )
#define   BLIS_PACK_FORMAT_BITS            ( ( ( 1 << BLIS_PACK_FORMAT_NUM_BITS       ) - 1 ) << BLIS_PACK_FORMAT_SHIFT )
#define   BLIS_PACK_BIT                    ( ( ( 1 << BLIS_PACK_NUM_BITS              ) - 1 ) << BLIS_PACK_SHIFT )
#define BLIS_PACK_REV_IF_UPPER_BIT         ( ( ( 1 << BLIS_PACK_REV_IF_UPPER_NUM_BITS ) - 1 ) << BLIS_PACK_REV_IF_UPPER_SHIFT )
#define BLIS_PACK_REV_IF_LOWER_BIT         ( ( ( 1 << BLIS_PACK_REV_IF_LOWER_NUM_BITS ) - 1 ) << BLIS_PACK_REV_IF_LOWER_SHIFT )
#define BLIS_PACK_BUFFER_BITS              ( ( ( 1 << BLIS_PACK_BUFFER_NUM_BITS       ) - 1 ) << BLIS_PACK_BUFFER_SHIFT )
#define BLIS_STRUC_BITS                    ( ( ( 1 << BLIS_STRUC_NUM_BITS             ) - 1 ) << BLIS_STRUC_SHIFT )
#define BLIS_COMP_PREC_BIT                 ( ( ( 1 << BLIS_PRECISION_NUM_BITS         ) - 1 ) << BLIS_COMP_PREC_SHIFT )
#define BLIS_SCALAR_DT_BITS                ( ( ( 1 << BLIS_DATATYPE_NUM_BITS          ) - 1 ) << BLIS_SCALAR_DT_SHIFT )
#define   BLIS_SCALAR_DOMAIN_BIT           ( ( ( 1 << BLIS_DOMAIN_NUM_BITS            ) - 1 ) << BLIS_SCALAR_DOMAIN_SHIFT )
#define   BLIS_SCALAR_PREC_BIT             ( ( ( 1 << BLIS_PRECISION_NUM_BITS         ) - 1 ) << BLIS_SCALAR_PREC_SHIFT )


//
// -- BLIS enumerated type value definitions -----------------------------------
//

#define BLIS_BITVAL_REAL                  0x0
#define BLIS_BITVAL_COMPLEX               BLIS_DOMAIN_BIT
#define BLIS_BITVAL_SINGLE_PREC           0x0
#define BLIS_BITVAL_DOUBLE_PREC         ( 0x1 << BLIS_PRECISION_SHIFT )
#define   BLIS_BITVAL_FLOAT_TYPE          0x0
#define   BLIS_BITVAL_SCOMPLEX_TYPE       BLIS_DOMAIN_BIT
#define   BLIS_BITVAL_DOUBLE_TYPE         BLIS_BITVAL_DOUBLE_PREC
#define   BLIS_BITVAL_DCOMPLEX_TYPE     ( BLIS_DOMAIN_BIT | BLIS_BITVAL_DOUBLE_PREC )
#define   BLIS_BITVAL_INT_TYPE            0x04
#define   BLIS_BITVAL_CONST_TYPE          0x05
#define BLIS_BITVAL_NO_TRANS              0x0
#define BLIS_BITVAL_TRANS                 BLIS_TRANS_BIT
#define BLIS_BITVAL_NO_CONJ               0x0
#define BLIS_BITVAL_CONJ                  BLIS_CONJ_BIT
#define BLIS_BITVAL_CONJ_TRANS          ( BLIS_CONJ_BIT | BLIS_TRANS_BIT )
#define BLIS_BITVAL_ZEROS                 0x0
#define BLIS_BITVAL_UPPER               ( BLIS_UPPER_BIT | BLIS_DIAG_BIT )
#define BLIS_BITVAL_LOWER               ( BLIS_LOWER_BIT | BLIS_DIAG_BIT )
#define BLIS_BITVAL_DENSE                 BLIS_UPLO_BITS
#define BLIS_BITVAL_NONUNIT_DIAG          0x0
#define BLIS_BITVAL_UNIT_DIAG             BLIS_UNIT_DIAG_BIT
#define BLIS_BITVAL_INVERT_DIAG           BLIS_INVERT_DIAG_BIT
#define BLIS_BITVAL_NOT_PACKED            0x0
#define   BLIS_BITVAL_1E                ( 0x1  << BLIS_PACK_FORMAT_SHIFT )
#define   BLIS_BITVAL_1R                ( 0x2  << BLIS_PACK_FORMAT_SHIFT )
#define   BLIS_BITVAL_RO                ( 0x3  << BLIS_PACK_FORMAT_SHIFT )
#define   BLIS_BITVAL_PACKED_UNSPEC     ( BLIS_PACK_BIT                                         )
#define   BLIS_BITVAL_PACKED_PANELS     ( BLIS_PACK_BIT                   | BLIS_PACK_PANEL_BIT )
#define   BLIS_BITVAL_PACKED_PANELS_1E  ( BLIS_PACK_BIT | BLIS_BITVAL_1E  | BLIS_PACK_PANEL_BIT )
#define   BLIS_BITVAL_PACKED_PANELS_1R  ( BLIS_PACK_BIT | BLIS_BITVAL_1R  | BLIS_PACK_PANEL_BIT )
#define   BLIS_BITVAL_PACKED_PANELS_RO  ( BLIS_PACK_BIT | BLIS_BITVAL_RO  | BLIS_PACK_PANEL_BIT )
#define BLIS_BITVAL_PACK_FWD_IF_UPPER     0x0
#define BLIS_BITVAL_PACK_REV_IF_UPPER     BLIS_PACK_REV_IF_UPPER_BIT
#define BLIS_BITVAL_PACK_FWD_IF_LOWER     0x0
#define BLIS_BITVAL_PACK_REV_IF_LOWER     BLIS_PACK_REV_IF_LOWER_BIT
#define BLIS_BITVAL_BUFFER_FOR_A_BLOCK    0x0
#define BLIS_BITVAL_BUFFER_FOR_B_PANEL  ( 0x1 << BLIS_PACK_BUFFER_SHIFT )
#define BLIS_BITVAL_BUFFER_FOR_C_PANEL  ( 0x2 << BLIS_PACK_BUFFER_SHIFT )
#define BLIS_BITVAL_BUFFER_FOR_GEN_USE  ( 0x3 << BLIS_PACK_BUFFER_SHIFT )
#define BLIS_BITVAL_GENERAL               0x0
#define BLIS_BITVAL_HERMITIAN           ( 0x1 << BLIS_STRUC_SHIFT )
#define BLIS_BITVAL_SYMMETRIC           ( 0x2 << BLIS_STRUC_SHIFT )
#define BLIS_BITVAL_TRIANGULAR          ( 0x3 << BLIS_STRUC_SHIFT )


//
// -- BLIS enumerated type definitions -----------------------------------------
//

// -- Operational parameter types --

typedef enum trans_e
{
	BLIS_NO_TRANSPOSE      = 0x0,
	BLIS_TRANSPOSE         = BLIS_BITVAL_TRANS,
	BLIS_CONJ_NO_TRANSPOSE = BLIS_BITVAL_CONJ,
	BLIS_CONJ_TRANSPOSE    = BLIS_BITVAL_CONJ_TRANS
} trans_t;

typedef enum conj_e
{
	BLIS_NO_CONJUGATE      = 0x0,
	BLIS_CONJUGATE         = BLIS_BITVAL_CONJ
} conj_t;

typedef enum uplo_e
{
	BLIS_ZEROS             = BLIS_BITVAL_ZEROS,
	BLIS_LOWER             = BLIS_BITVAL_LOWER,
	BLIS_UPPER             = BLIS_BITVAL_UPPER,
	BLIS_DENSE             = BLIS_BITVAL_DENSE
} uplo_t;

typedef enum side_e
{
	BLIS_LEFT              = 0x0,
	BLIS_RIGHT
} side_t;

typedef enum diag_e
{
	BLIS_NONUNIT_DIAG      = 0x0,
	BLIS_UNIT_DIAG         = BLIS_BITVAL_UNIT_DIAG
} diag_t;

typedef enum invdiag_e
{
	BLIS_NO_INVERT_DIAG    = 0x0,
	BLIS_INVERT_DIAG       = BLIS_BITVAL_INVERT_DIAG
} invdiag_t;

typedef enum struc_e
{
	BLIS_GENERAL           = BLIS_BITVAL_GENERAL,
	BLIS_HERMITIAN         = BLIS_BITVAL_HERMITIAN,
	BLIS_SYMMETRIC         = BLIS_BITVAL_SYMMETRIC,
	BLIS_TRIANGULAR        = BLIS_BITVAL_TRIANGULAR
} struc_t;


// -- Data type --

typedef enum num_e
{
	BLIS_FLOAT             = BLIS_BITVAL_FLOAT_TYPE,
	BLIS_DOUBLE            = BLIS_BITVAL_DOUBLE_TYPE,
	BLIS_SCOMPLEX          = BLIS_BITVAL_SCOMPLEX_TYPE,
	BLIS_DCOMPLEX          = BLIS_BITVAL_DCOMPLEX_TYPE,
	BLIS_INT               = BLIS_BITVAL_INT_TYPE,
	BLIS_CONSTANT          = BLIS_BITVAL_CONST_TYPE,
	BLIS_DT_LO             = BLIS_FLOAT,
	BLIS_DT_HI             = BLIS_DCOMPLEX
} num_t;

typedef enum dom_e
{
	BLIS_REAL              = BLIS_BITVAL_REAL,
	BLIS_COMPLEX           = BLIS_BITVAL_COMPLEX
} dom_t;

typedef enum prec_e
{
	BLIS_SINGLE_PREC       = BLIS_BITVAL_SINGLE_PREC,
	BLIS_DOUBLE_PREC       = BLIS_BITVAL_DOUBLE_PREC
} prec_t;


// -- Pack schema type --

typedef enum pack_e
{
	BLIS_NOT_PACKED       = BLIS_BITVAL_NOT_PACKED,
	BLIS_PACKED_UNSPEC    = BLIS_BITVAL_PACKED_UNSPEC,
	BLIS_PACKED_VECTOR    = BLIS_BITVAL_PACKED_UNSPEC,
	BLIS_PACKED_MATRIX    = BLIS_BITVAL_PACKED_UNSPEC,
	BLIS_PACKED_PANELS    = BLIS_BITVAL_PACKED_PANELS,
	BLIS_PACKED_PANELS_1E = BLIS_BITVAL_PACKED_PANELS_1E,
	BLIS_PACKED_PANELS_1R = BLIS_BITVAL_PACKED_PANELS_1R,
	BLIS_PACKED_PANELS_RO = BLIS_BITVAL_PACKED_PANELS_RO,

	// BLIS_NUM_PACK_SCHEMA_TYPES must be last!
	// We start with BLIS_PACKED_PANELS.
	BLIS_NUM_PACK_SCHEMA_TYPES_,
	BLIS_NUM_PACK_SCHEMA_TYPES = ((( BLIS_NUM_PACK_SCHEMA_TYPES_ - BLIS_PACKED_PANELS - 1 ) >> BLIS_PACK_FORMAT_SHIFT ) + 1)
} pack_t;


// -- Pack order type --

typedef enum packord_e
{
	BLIS_PACK_FWD_IF_UPPER = BLIS_BITVAL_PACK_FWD_IF_UPPER,
	BLIS_PACK_REV_IF_UPPER = BLIS_BITVAL_PACK_REV_IF_UPPER,

	BLIS_PACK_FWD_IF_LOWER = BLIS_BITVAL_PACK_FWD_IF_LOWER,
	BLIS_PACK_REV_IF_LOWER = BLIS_BITVAL_PACK_REV_IF_LOWER
} packord_t;


// -- Pack buffer type --

typedef enum packbuf_e
{
	BLIS_BUFFER_FOR_A_BLOCK = BLIS_BITVAL_BUFFER_FOR_A_BLOCK,
	BLIS_BUFFER_FOR_B_PANEL = BLIS_BITVAL_BUFFER_FOR_B_PANEL,
	BLIS_BUFFER_FOR_C_PANEL = BLIS_BITVAL_BUFFER_FOR_C_PANEL,
	BLIS_BUFFER_FOR_GEN_USE = BLIS_BITVAL_BUFFER_FOR_GEN_USE
} packbuf_t;


// -- Partitioning direction --

typedef enum dir_e
{
	BLIS_FWD,
	BLIS_BWD
} dir_t;


// -- Subpartition type --

typedef enum subpart_e
{
	BLIS_SUBPART0,
	BLIS_SUBPART1,
	BLIS_SUBPART2,
	BLIS_SUBPART1AND0,
	BLIS_SUBPART1AND2,
	BLIS_SUBPART1A,
	BLIS_SUBPART1B,
	BLIS_SUBPART00,
	BLIS_SUBPART10,
	BLIS_SUBPART20,
	BLIS_SUBPART01,
	BLIS_SUBPART11,
	BLIS_SUBPART21,
	BLIS_SUBPART02,
	BLIS_SUBPART12,
	BLIS_SUBPART22
} subpart_t;


// -- Matrix dimension type --

typedef enum mdim_e
{
	BLIS_M = 0,
	BLIS_N = 1
} mdim_t;


// -- Machine parameter types --

typedef enum machval_e
{
	BLIS_MACH_EPS = 0,
	BLIS_MACH_SFMIN,
	BLIS_MACH_BASE,
	BLIS_MACH_PREC,
	BLIS_MACH_NDIGMANT,
	BLIS_MACH_RND,
	BLIS_MACH_EMIN,
	BLIS_MACH_RMIN,
	BLIS_MACH_EMAX,
	BLIS_MACH_RMAX,
	BLIS_MACH_EPS2,

	// BLIS_NUM_MACH_PARAMS must be last!
	BLIS_NUM_MACH_PARAMS
} machval_t;

#define BLIS_MACH_PARAM_FIRST  BLIS_MACH_EPS
#define BLIS_MACH_PARAM_LAST   BLIS_MACH_EPS2


// -- Induced method types --

typedef enum ind_e
{
	BLIS_1M        = 0,
	BLIS_NAT,

	BLIS_IND_FIRST = 0,
	BLIS_IND_LAST  = BLIS_NAT,

	// BLIS_NUM_IND_METHODS must be last!
	BLIS_NUM_IND_METHODS
} ind_t;

// These are used in bli_l3_*_oapi.c to construct the ind_t values from
// the induced method substrings that go into function names.
#define bli_1m   BLIS_1M
#define bli_nat  BLIS_NAT


// -- Threading implementation type --

typedef enum timpl_e
{
	BLIS_SINGLE = 0,
	BLIS_OPENMP,
	BLIS_POSIX,
	BLIS_HPX,

	// BLIS_NUM_THREAD_IMPLS must be last!
	BLIS_NUM_THREAD_IMPLS

} timpl_t;


// -- Kernel ID types --

// Encode the number of independent type parameters in the high
// bits of the kernel ID. This lets us identify kernel IDs as the
// appropriate type while also using them as linear indices after
// masking out these bits.
#define BLIS_NTYPE_KER_SHIFT 28
#define BLIS_NTYPE_KER_BITS  (0xFu << BLIS_NTYPE_KER_SHIFT)
#define BLIS_1TYPE_KER       (  0u << BLIS_NTYPE_KER_SHIFT)
#define BLIS_2TYPE_KER       (  1u << BLIS_NTYPE_KER_SHIFT)
#define BLIS_3TYPE_KER       (  2u << BLIS_NTYPE_KER_SHIFT)

#define bli_ker_idx( ker )   ( kerid_t )((ker) & ~BLIS_NTYPE_KER_BITS)
#define bli_ker_ntype( ker ) ((((ker) & BLIS_NTYPE_KER_BITS) >> BLIS_NTYPE_KER_SHIFT) + 1)

// We have to use a 32-bit type here to avoid problems with passing small enum
// constants to variadic functions. See https://github.com/flame/blis/issues/839.
typedef uint32_t kerid_t;

// Sentinel constant used to indicate the end of a variable argument function
// (See bli_cntx.c)

#define BLIS_VA_END  ((kerid_t)-1)

typedef enum ukr_e
{
	// -- Single-type kernels --

	// l1v kernels
	BLIS_ADDV_KER = BLIS_1TYPE_KER,
	BLIS_AMAXV_KER,
	BLIS_AXPBYV_KER,
	BLIS_AXPYV_KER,
	BLIS_COPYV_KER,
	BLIS_DOTV_KER,
	BLIS_DOTXV_KER,
	BLIS_INVERTV_KER,
	BLIS_INVSCALV_KER,
	BLIS_SCALV_KER,
	BLIS_SCAL2V_KER,
	BLIS_SETV_KER,
	BLIS_SUBV_KER,
	BLIS_SWAPV_KER,
	BLIS_XPBYV_KER,
	BLIS_AXPY2V_KER,
	BLIS_DOTAXPYV_KER,

	// l1f kernels
	BLIS_AXPYF_KER,
	BLIS_DOTXF_KER,
	BLIS_DOTXAXPYF_KER,

	// l3 native kernels
	BLIS_GEMMTRSM_L_UKR,
	BLIS_GEMMTRSM_U_UKR,
	BLIS_TRSM_L_UKR,
	BLIS_TRSM_U_UKR,

	// l3 1m kernels
	BLIS_GEMMTRSM1M_L_UKR,
	BLIS_GEMMTRSM1M_U_UKR,

	// gemmsup kernels
	BLIS_GEMMSUP_RRR_UKR,
	BLIS_GEMMSUP_RRC_UKR,
	BLIS_GEMMSUP_RCR_UKR,
	BLIS_GEMMSUP_RCC_UKR,
	BLIS_GEMMSUP_CRR_UKR,
	BLIS_GEMMSUP_CRC_UKR,
	BLIS_GEMMSUP_CCR_UKR,
	BLIS_GEMMSUP_CCC_UKR,
	BLIS_GEMMSUP_XXX_UKR,

	// BLIS_NUM_UKRS must after all 1-type kernels and before 2-type kernels!
	BLIS_NUM_UKRS_, BLIS_NUM_UKRS = bli_ker_idx( BLIS_NUM_UKRS_ ),

	// -- Two-type kernels --

	// pack kernels
	BLIS_PACKM_KER = BLIS_2TYPE_KER,
	BLIS_PACKM_1ER_KER,
	BLIS_PACKM_RO_KER,
	BLIS_PACKM_DIAG_KER,
	BLIS_PACKM_DIAG_1ER_KER,
	BLIS_PACKM_DIAG_RO_KER,

	// unpack kernels
	BLIS_UNPACKM_KER,

	// l3 native kernels
	BLIS_GEMM_UKR,

	// l3 1m kernels
	BLIS_GEMM1M_UKR,

	// mixed-domain kernels
	BLIS_GEMM_CCR_UKR,
	BLIS_GEMM_RCC_UKR,
	BLIS_GEMM_CRR_UKR,

	// BLIS_NUM_UKR2S must come after all kernels!
	BLIS_NUM_UKR2S_, BLIS_NUM_UKR2S = bli_ker_idx( BLIS_NUM_UKR2S_ ),

	// Force the size of ukr_t values to be as large as kerid_t
	BLIS_UKRS_END_ = BLIS_VA_END
} ukr_t;


typedef enum ukr_pref_e
{
    // l3 kernel row preferences
	BLIS_GEMM_UKR_ROW_PREF,
	BLIS_GEMMTRSM_L_UKR_ROW_PREF,
	BLIS_GEMMTRSM_U_UKR_ROW_PREF,
	BLIS_TRSM_L_UKR_ROW_PREF,
	BLIS_TRSM_U_UKR_ROW_PREF,

    // gemmsup kernel row preferences
	BLIS_GEMMSUP_RRR_UKR_ROW_PREF,
	BLIS_GEMMSUP_RRC_UKR_ROW_PREF,
	BLIS_GEMMSUP_RCR_UKR_ROW_PREF,
	BLIS_GEMMSUP_RCC_UKR_ROW_PREF,
	BLIS_GEMMSUP_CRR_UKR_ROW_PREF,
	BLIS_GEMMSUP_CRC_UKR_ROW_PREF,
	BLIS_GEMMSUP_CCR_UKR_ROW_PREF,
	BLIS_GEMMSUP_CCC_UKR_ROW_PREF,
	BLIS_GEMMSUP_XXX_UKR_ROW_PREF,

    // BLIS_NUM_UKR_PREFS must be last!
    BLIS_NUM_UKR_PREFS,

	// Force the size of ukr_pref_t values to be as large as kerid_t
	BLIS_UKR_PREFS_END_ = BLIS_VA_END
} ukr_pref_t;

typedef enum kimpl_e
{
	BLIS_REFERENCE_UKERNEL = 0,
	BLIS_VIRTUAL_UKERNEL,
	BLIS_OPTIMIZED_UKERNEL,
	BLIS_NOTAPPLIC_UKERNEL,

    // BLIS_NUM_UKR_IMPL_TYPES must be last!
	BLIS_NUM_UKR_IMPL_TYPES
} kimpl_t;


#if 0
typedef enum l3sup_e
{
	// RV = row-stored, contiguous vector-loading
	// RG = row-stored, non-contiguous gather-loading
	// CV = column-stored, contiguous vector-loading
	// CG = column-stored, non-contiguous gather-loading

	// RD = row-stored, dot-based
	// CD = col-stored, dot-based

	// RC = row-stored, column-times-column
	// CR = column-stored, row-times-row

	// GX = general-stored generic implementation

	BLIS_GEMMSUP_RV_UKR = 0,
	BLIS_GEMMSUP_RG_UKR,
	BLIS_GEMMSUP_CV_UKR,
	BLIS_GEMMSUP_CG_UKR,

	BLIS_GEMMSUP_RD_UKR,
	BLIS_GEMMSUP_CD_UKR,

	BLIS_GEMMSUP_RC_UKR,
	BLIS_GEMMSUP_CR_UKR,

	BLIS_GEMMSUP_GX_UKR,

	// BLIS_NUM_LEVEL3_SUP_UKRS must be last!
	BLIS_NUM_LEVEL3_SUP_UKRS
} l3sup_t;
#endif


typedef enum stor3_e
{
	// 3-operand storage combinations
	BLIS_RRR = 0,
	BLIS_RRC, // 1
	BLIS_RCR, // 2
	BLIS_RCC, // 3
	BLIS_CRR, // 4
	BLIS_CRC, // 5
	BLIS_CCR, // 6
	BLIS_CCC, // 7
	BLIS_XXX, // 8

#if 0
	BLIS_RRG,
	BLIS_RCG,
	BLIS_RGR,
	BLIS_RGC,
	BLIS_RGG,
	BLIS_CRG,
	BLIS_CCG,
	BLIS_CGR,
	BLIS_CGC,
	BLIS_CGG,
	BLIS_GRR,
	BLIS_GRC,
	BLIS_GRG,
	BLIS_GCR,
	BLIS_GCC,
	BLIS_GCG,
	BLIS_GGR,
	BLIS_GGC,
	BLIS_GGG,
#endif

	// BLIS_NUM_3OP_RC_COMBOS must be last!
	BLIS_NUM_3OP_RC_COMBOS
} stor3_t;


#if 0
typedef enum thridx_e
{
	BLIS_JC_IDX = 0,
	BLIS_PC_IDX,
	BLIS_IC_IDX,
	BLIS_JR_IDX,
	BLIS_IR_IDX,
	BLIS_PR_IDX
} thridx_t;
#endif

#define BLIS_NUM_LOOPS 6


// -- Operation ID type --

typedef enum opid_e
{
//
// NOTE: If/when additional type values are added to this enum,
// you must either:
// - keep the level-3 values (starting with _GEMM) beginning at
//   index 0; or
// - if the value range is moved such that it does not begin at
//   index 0, implement something like a BLIS_OPID_LEVEL3_RANGE_START
//   value that can be subtracted from the opid_t value to map it
//   to a zero-based range.
// This is needed because these level-3 opid_t values are used in
// bli_l3_ind.c to index into arrays.
//
	BLIS_GEMM = 0,
	BLIS_GEMMT,
	BLIS_HEMM,
	BLIS_HERK,
	BLIS_HER2K,
	BLIS_SYMM,
	BLIS_SYRK,
	BLIS_SYR2K,
	BLIS_TRMM3,
	BLIS_TRMM,
	BLIS_TRSM,

	// BLIS_NOID (= BLIS_NUM_LEVEL3_OPS) must be last!
	BLIS_NOID,
	BLIS_NUM_LEVEL3_OPS = BLIS_NOID,

	// Force the size of opid_t values to be as large as kerid_t
	BLIS_LEVEL3_OPS_END_ = BLIS_VA_END
} opid_t;


// -- Blocksize ID type --

typedef enum bszid_e
{
	// NOTE: the level-3 blocksizes MUST be indexed starting at zero.
	// At one point, we made this assumption in bli_cntx_set_blkszs()
	// and friends.
	BLIS_KR,
	BLIS_MR,
	BLIS_NR,
	BLIS_MC,
	BLIS_KC,
	BLIS_NC,

	// broadcast factors for packing
	BLIS_BBM,
	BLIS_BBN,

	// level-2 blocksizes
	BLIS_M2, // level-2 blocksize in m dimension
	BLIS_N2, // level-2 blocksize in n dimension

	// level-1f blocksizes
	BLIS_AF, // level-1f axpyf fusing factor
	BLIS_DF, // level-1f dotxf fusing factor
	BLIS_XF, // level-1f dotxaxpyf fusing factor

	// gemmsup thresholds
	BLIS_MT, // level-3 small/unpacked matrix threshold in m dimension
	BLIS_NT, // level-3 small/unpacked matrix threshold in n dimension
	BLIS_KT, // level-3 small/unpacked matrix threshold in k dimension

	// gemmsup block sizes
	BLIS_KR_SUP,
	BLIS_MR_SUP,
	BLIS_NR_SUP,
	BLIS_MC_SUP,
	BLIS_KC_SUP,
	BLIS_NC_SUP,

	// BLIS_NO_PART (= BLIS_NUM_BLKSZS) must be last!
	BLIS_NO_PART, // used as a placeholder when blocksizes are not applicable,
	              // such as when characterizing a packm operation.
	BLIS_NUM_BLKSZS = BLIS_NO_PART,

	// Force the size of bszid_t values to be as large as kerid_t
	BLIS_BLKSZS_END_ = BLIS_VA_END
} bszid_t;


// A convenient version of the BLIS_XX block size IDs which can be used in bitfields.
enum
{
	BLIS_THREAD_NONE = 0,
	BLIS_THREAD_KR   = 1 << BLIS_KR,
	BLIS_THREAD_MR   = 1 << BLIS_MR,
	BLIS_THREAD_NR   = 1 << BLIS_NR,
	BLIS_THREAD_MC   = 1 << BLIS_MC,
	BLIS_THREAD_KC   = 1 << BLIS_KC,
	BLIS_THREAD_NC   = 1 << BLIS_NC,
};

// -- Architecture ID type --

// NOTE: This typedef enum must be kept up-to-date with the arch_t
// string array in bli_arch.c. Whenever values are added/inserted
// OR if values are rearranged, be sure to update the string array
// in bli_arch.c.

typedef enum arch_e
{
	// NOTE: The C language standard guarantees that the first enum value
	// starts at 0.

	// Intel
	BLIS_ARCH_SKX,
	BLIS_ARCH_KNL,
	BLIS_ARCH_KNC,
	BLIS_ARCH_HASWELL,
	BLIS_ARCH_SANDYBRIDGE,
	BLIS_ARCH_PENRYN,

	// AMD
	BLIS_ARCH_ZEN3,
	BLIS_ARCH_ZEN2,
	BLIS_ARCH_ZEN,
	BLIS_ARCH_EXCAVATOR,
	BLIS_ARCH_STEAMROLLER,
	BLIS_ARCH_PILEDRIVER,
	BLIS_ARCH_BULLDOZER,

	// ARM-SVE
	BLIS_ARCH_ARMSVE,
	BLIS_ARCH_A64FX,

	// ARM-NEON (4 pipes x 128-bit vectors)
	BLIS_ARCH_ALTRAMAX,
	BLIS_ARCH_ALTRA,
	BLIS_ARCH_FIRESTORM,

	// ARM (2 pipes x 128-bit vectors)
	BLIS_ARCH_THUNDERX2,
	BLIS_ARCH_CORTEXA57,
	BLIS_ARCH_CORTEXA53,

	// ARM 32-bit (vintage)
	BLIS_ARCH_CORTEXA15,
	BLIS_ARCH_CORTEXA9,

	// IBM/Power
	BLIS_ARCH_POWER10,
	BLIS_ARCH_POWER9,
	BLIS_ARCH_POWER7,
	BLIS_ARCH_BGQ,

	// RISC-V
	BLIS_ARCH_RV32I,
	BLIS_ARCH_RV64I,
	BLIS_ARCH_RV32IV,
	BLIS_ARCH_RV64IV,

	// SiFive
	BLIS_ARCH_SIFIVE_RVV,
	BLIS_ARCH_SIFIVE_X280,

	// Generic architecture/configuration
	BLIS_ARCH_GENERIC,

	// The total number of defined architectures. This must be last in the
	// list of enums since its definition assumes that the previous enum
	// value (BLIS_ARCH_GENERIC) is given index num_archs-1.
	BLIS_NUM_ARCHS

} arch_t;


//
// -- BLIS misc. structure types -----------------------------------------------
//

// This header must be included here (or earlier) because definitions it
// provides are needed in the pool_t and related structs.
#include "bli_pthread.h"

// -- Pool block type --

typedef struct pblk_s
{
	void*     buf;
	siz_t     block_size;

} pblk_t;


// -- Pool type --

typedef struct pool_s
{
	void*     block_ptrs;
	dim_t     block_ptrs_len;

	dim_t     top_index;
	dim_t     num_blocks;

	siz_t     block_size;
	siz_t     align_size;
	siz_t     offset_size;

	malloc_ft malloc_fp;
	free_ft   free_fp;

} pool_t;


// -- Array type --

typedef struct array_s
{
	void*     buf;

	siz_t     num_elem;
	siz_t     elem_size;

} array_t;


// -- Locked pool-of-arrays-of-pools type --

typedef struct apool_s
{
	bli_pthread_mutex_t mutex;
	pool_t              pool;

	siz_t               def_array_len;

} apool_t;


// -- packing block allocator: Locked set of pools type --

typedef struct pba_s
{
	pool_t              pools[3];
	bli_pthread_mutex_t mutex;

	// These fields are used for general-purpose allocation.
	siz_t               align_size;
	malloc_ft           malloc_fp;
	free_ft             free_fp;

} pba_t;


// -- Memory object type --

typedef struct mem_s
{
	pblk_t    pblk;
	packbuf_t buf_type;
	pool_t*   pool;
	siz_t     size;
} mem_t;


// -- Control tree node type --

#define BLIS_MAX_SUB_NODES 2

struct cntl_s
{
	// Actually this is a l3_var_oft, but that type hasn't been defined yet
	void_fp var_func;
	struct
	{
		dim_t          ways;
		struct cntl_s* sub_node;
	} sub_nodes[ BLIS_MAX_SUB_NODES ];
};
typedef struct cntl_s cntl_t;


// -- Blocksize object type --

typedef struct blksz_s
{
	// Primary blocksize values.
	dim_t  v[BLIS_NUM_FP_TYPES];

	// Blocksize extensions.
	dim_t  e[BLIS_NUM_FP_TYPES];

} blksz_t;


// -- Function pointer object type --

typedef struct func_s
{
	// Kernel function address.
	void_fp ptr[BLIS_NUM_FP_TYPES];

} func_t;

typedef struct func2_s
{
	// Kernel function address.
	// A func2_t* can be cast to a func_t* in order to access
	// only the "diagonal" elements (dt,dt) (but note that to accomplish
	// this those elements are not stored in ptr[dt][dt]...see bli_func.c
	// for more details).
	void_fp ptr[BLIS_NUM_FP_TYPES][BLIS_NUM_FP_TYPES];

} func2_t;


// -- Multi-boolean object type --

typedef struct mbool_s
{
	bool v[BLIS_NUM_FP_TYPES];

} mbool_t;


// -- Auxiliary kernel info type --

// Note: This struct is used by macro-kernels to package together extra
// parameter values that may be of use to the micro-kernel without
// cluttering up the micro-kernel interface itself.

typedef struct auxinfo_s
{
	// The pack schemas of A and B.
	pack_t schema_a;
	pack_t schema_b;

	// Pointers to the micro-panels of A and B which will be used by the
	// next call to the micro-kernel.
	const void* a_next;
	const void* b_next;

	// The imaginary strides of A and B.
	inc_t is_a;
	inc_t is_b;

	// The panel strides of A and B.
	// NOTE: These are only used in situations where iteration over the
	// micropanels takes place in part within the kernel code (e.g. sup
	// millikernels).
	inc_t ps_a;
	inc_t ps_b;

	// The row and column offset of the current micro-tile in C.
	dim_t off_m;
	dim_t off_n;

	// The type to convert to on output.
	//num_t  dt_on_output;

	// (Virtual) microkernel address and additional parameters.
	void_fp ukr;
	const void* params;

} auxinfo_t;


// -- Global scalar constant data struct --

// Note: This struct is used only when statically initializing the
// global scalar constants in bli_const.c.
typedef struct constdata_s
{
	float    s;
	double   d;
	scomplex c;
	dcomplex z;
	gint_t   i;

} constdata_t;


//
// -- BLIS object type definitions ---------------------------------------------
//

typedef struct obj_s
{
	// Basic fields
	struct obj_s* root;

	dim_t         off[2];
	dim_t         dim[2];
	doff_t        diag_off;

	objbits_t     info;
	objbits_t     info2;
	siz_t         elem_size;

	void*         buffer;
	inc_t         rs;
	inc_t         cs;
	inc_t         is;

	// Bufferless scalar storage
	atom_t        scalar;

	// Pack-related fields
	dim_t         m_padded; // m dimension of matrix, including any padding
	dim_t         n_padded; // n dimension of matrix, including any padding
	inc_t         ps;       // panel stride (distance to next panel)
	inc_t         pd;       // panel dimension (the "width" of a panel:
	                        // usually MR or NR)
	dim_t         m_panel;  // m dimension of a "full" panel
	dim_t         n_panel;  // n dimension of a "full" panel

} obj_t;

// Pre-initializors. Things that must be set afterwards:
// - root object pointer
// - info bitfields: dt, target_dt, exec_dt, comp_dt
// - info2 bitfields: scalar_dt
// - elem_size
// - dims, strides
// - buffer
// - internal scalar buffer (must always set imaginary component)

#define BLIS_OBJECT_INITIALIZER \
{ \
	/* .root        = */ NULL, \
\
	/* .off         = */ { 0, 0 }, \
	/* .dim         = */ { 0, 0 }, \
	/* .diag_off    = */ 0, \
\
	/* .info        = */ 0x0 | BLIS_BITVAL_DENSE      | \
	/*                */       BLIS_BITVAL_GENERAL, \
	/* .info2       = */ 0x0, \
	/* .elem_size   = */ sizeof( float ), /* this is changed later. */ \
\
	/* .buffer      = */ NULL, \
	/* .rs          = */ 0, \
	/* .cs          = */ 0, \
	/* .is          = */ 1,  \
\
	/* .scalar      = */ { 0.0, 0.0 }, \
\
	/* .m_padded    = */ 0, \
	/* .n_padded    = */ 0, \
	/* .ps          = */ 0, \
	/* .pd          = */ 0, \
	/* .m_panel     = */ 0, \
	/* .n_panel     = */ 0, \
}

#define BLIS_OBJECT_INITIALIZER_1X1 \
{ \
	/* .root        = */ NULL, \
\
	/* .off         = */ { 0, 0 }, \
	/* .dim         = */ { 1, 1 }, \
	/* .diag_off    = */ 0, \
\
	/* .info        = */ 0x0 | BLIS_BITVAL_DENSE      | \
	/*                */       BLIS_BITVAL_GENERAL, \
	/* .info2       = */ 0x0, \
	/* .elem_size   = */ sizeof( float ), /* this is changed later. */ \
\
	/* .buffer      = */ NULL, \
	/* .rs          = */ 0, \
	/* .cs          = */ 0, \
	/* .is          = */ 1,  \
\
	/* .scalar      = */ { 0.0, 0.0 }, \
\
	/* .m_padded    = */ 0, \
	/* .n_padded    = */ 0, \
	/* .ps          = */ 0, \
	/* .pd          = */ 0, \
	/* .m_panel     = */ 0, \
	/* .n_panel     = */ 0, \
}

// Define these macros here since they must be updated if contents of
// obj_t changes.

BLIS_INLINE void bli_obj_init_full_shallow_copy_of( const obj_t* a, obj_t* b )
{
	b->root        = a->root;

	b->off[0]      = a->off[0];
	b->off[1]      = a->off[1];
	b->dim[0]      = a->dim[0];
	b->dim[1]      = a->dim[1];
	b->diag_off    = a->diag_off;

	b->info        = a->info;
	b->info2       = a->info2;
	b->elem_size   = a->elem_size;

	b->buffer      = a->buffer;
	b->rs          = a->rs;
	b->cs          = a->cs;
	b->is          = a->is;

	b->scalar      = a->scalar;

	//b->pack_mem    = a->pack_mem;
	b->m_padded    = a->m_padded;
	b->n_padded    = a->n_padded;
	b->ps          = a->ps;
	b->pd          = a->pd;
	b->m_panel     = a->m_panel;
	b->n_panel     = a->n_panel;
}

BLIS_INLINE void bli_obj_init_subpart_from( const obj_t* a, obj_t* b )
{
	b->root        = a->root;

	b->off[0]      = a->off[0];
	b->off[1]      = a->off[1];
	// Avoid copying m and n since they will be overwritten.
	//b->dim[0]      = a->dim[0];
	//b->dim[1]      = a->dim[1];
	b->diag_off    = a->diag_off;

	b->info        = a->info;
	b->info2       = a->info2;
	b->elem_size   = a->elem_size;

	b->buffer      = a->buffer;
	b->rs          = a->rs;
	b->cs          = a->cs;
	b->is          = a->is;

	b->scalar      = a->scalar;

	// Avoid copying pack_mem entry.
	// FGVZ: You should probably make sure this is right.
	//b->pack_mem    = a->pack_mem;
	b->m_padded    = a->m_padded;
	b->n_padded    = a->n_padded;
	b->ps          = a->ps;
	b->pd          = a->pd;
	b->m_panel     = a->m_panel;
	b->n_panel     = a->n_panel;
}

// Initializors for global scalar constants.
// NOTE: These must remain cpp macros since they are initializor
// expressions, not functions.

#define bli_obj_init_const( buffer0 ) \
{ \
	.root      = NULL, \
\
	.off       = { 0, 0 }, \
	.dim       = { 1, 1 }, \
	.diag_off  = 0, \
\
	.info      = 0x0 | BLIS_BITVAL_CONST_TYPE | \
	                   BLIS_BITVAL_DENSE      | \
	                   BLIS_BITVAL_GENERAL, \
	.info2     = 0x0, \
	.elem_size = sizeof( constdata_t ), \
\
	.buffer    = buffer0, \
	.rs        = 1, \
	.cs        = 1, \
	.is        = 1  \
}

#define bli_obj_init_constdata( val ) \
{ \
	.s =           ( float  )val, \
	.d =           ( double )val, \
	.c = { .real = ( float  )val, .imag = 0.0f }, \
	.z = { .real = ( double )val, .imag = 0.0 }, \
	.i =           ( gint_t )val, \
}

#define bli_obj_init_constdata_ri( valr, vali ) \
{ \
	.s =           ( float  )valr, \
	.d =           ( double )valr, \
	.c = { .real = ( float  )valr, .imag = ( float  )vali }, \
	.z = { .real = ( double )valr, .imag = ( double )vali }, \
	.i =           ( gint_t )valr, \
}


// -- Stack type --

// NB: stack_t is already taken by <signal.h>
typedef struct stck_s
{
	siz_t elem_size;
	siz_t block_len;
	siz_t max_blocks;
	siz_t size;
	siz_t capacity;

	void** blocks;

	bli_pthread_mutex_t lock;
} stck_t;


// -- Context type --

typedef struct cntx_s
{
	stck_t blkszs;
	stck_t bmults;

	stck_t ukrs;
	stck_t ukr2s;
	stck_t ukr_prefs;

	stck_t l3_sup_handlers;
} cntx_t;


// -- Runtime type --

// NOTE: The order of these fields must be kept consistent with the definition
// of the BLIS_RNTM_INITIALIZER macro in bli_rntm.h.

typedef struct rntm_s
{
	// "External" fields: these may be queried by the end-user.
	timpl_t   thread_impl;

	bool      auto_factor;

	dim_t     num_threads;
	dim_t     thrloop[ BLIS_NUM_LOOPS ];
	bool      pack_a; // enable/disable packing of left-hand matrix A.
	bool      pack_b; // enable/disable packing of right-hand matrix B.
	bool      l3_sup; // enable/disable small matrix handling in level-3 ops.
} rntm_t;


// -- Error types --

typedef enum errlev_e
{
	BLIS_NO_ERROR_CHECKING = 0,
	BLIS_FULL_ERROR_CHECKING
} errlev_t;

typedef enum err_e
{
	// Generic error codes
	BLIS_SUCCESS                               = (  -1),
	BLIS_FAILURE                               = (  -2),

	BLIS_ERROR_CODE_MIN                        = (  -9),

	// General errors
	BLIS_INVALID_ERROR_CHECKING_LEVEL          = ( -10),
	BLIS_UNDEFINED_ERROR_CODE                  = ( -11),
	BLIS_NULL_POINTER                          = ( -12),
	BLIS_NOT_YET_IMPLEMENTED                   = ( -13),
	BLIS_OUT_OF_BOUNDS                         = ( -14),
	BLIS_LOCK_FAILURE                          = ( -15),

	// Parameter-specific errors
	BLIS_INVALID_SIDE                          = ( -20),
	BLIS_INVALID_UPLO                          = ( -21),
	BLIS_INVALID_TRANS                         = ( -22),
	BLIS_INVALID_CONJ                          = ( -23),
	BLIS_INVALID_DIAG                          = ( -24),
	BLIS_INVALID_MACHVAL                       = ( -25),
	BLIS_EXPECTED_NONUNIT_DIAG                 = ( -26),

	// Datatype-specific errors
	BLIS_INVALID_DATATYPE                      = ( -30),
	BLIS_EXPECTED_FLOATING_POINT_DATATYPE      = ( -31),
	BLIS_EXPECTED_NONINTEGER_DATATYPE          = ( -32),
	BLIS_EXPECTED_NONCONSTANT_DATATYPE         = ( -33),
	BLIS_EXPECTED_REAL_DATATYPE                = ( -34),
	BLIS_EXPECTED_INTEGER_DATATYPE             = ( -35),
	BLIS_INCONSISTENT_DATATYPES                = ( -36),
	BLIS_EXPECTED_REAL_PROJ_OF                 = ( -37),
	BLIS_EXPECTED_REAL_VALUED_OBJECT           = ( -38),
	BLIS_INCONSISTENT_PRECISIONS               = ( -39),

	// Dimension-specific errors
	BLIS_NONCONFORMAL_DIMENSIONS               = ( -40),
	BLIS_EXPECTED_SCALAR_OBJECT                = ( -41),
	BLIS_EXPECTED_VECTOR_OBJECT                = ( -42),
	BLIS_UNEQUAL_VECTOR_LENGTHS                = ( -43),
	BLIS_EXPECTED_SQUARE_OBJECT                = ( -44),
	BLIS_UNEXPECTED_OBJECT_LENGTH              = ( -45),
	BLIS_UNEXPECTED_OBJECT_WIDTH               = ( -46),
	BLIS_UNEXPECTED_VECTOR_DIM                 = ( -47),
	BLIS_UNEXPECTED_DIAG_OFFSET                = ( -48),
	BLIS_NEGATIVE_DIMENSION                    = ( -49),

	// Stride-specific errors
	BLIS_INVALID_ROW_STRIDE                    = ( -50),
	BLIS_INVALID_COL_STRIDE                    = ( -51),
	BLIS_INVALID_DIM_STRIDE_COMBINATION        = ( -52),

	// Structure-specific errors
	BLIS_EXPECTED_GENERAL_OBJECT               = ( -60),
	BLIS_EXPECTED_HERMITIAN_OBJECT             = ( -61),
	BLIS_EXPECTED_SYMMETRIC_OBJECT             = ( -62),
	BLIS_EXPECTED_TRIANGULAR_OBJECT            = ( -63),

	// Storage-specific errors
	BLIS_EXPECTED_UPPER_OR_LOWER_OBJECT        = ( -70),

	// Partitioning-specific errors
	BLIS_INVALID_3x1_SUBPART                   = ( -80),
	BLIS_INVALID_1x3_SUBPART                   = ( -81),
	BLIS_INVALID_3x3_SUBPART                   = ( -82),

	// Control tree-specific errors
	BLIS_UNEXPECTED_NULL_CONTROL_TREE          = ( -90),

	// Packing-specific errors
	BLIS_PACK_SCHEMA_NOT_SUPPORTED_FOR_UNPACK  = (-100),
	BLIS_PACK_SCHEMA_NOT_SUPPORTED_FOR_PART    = (-101),

	// Buffer-specific errors
	BLIS_EXPECTED_NONNULL_OBJECT_BUFFER        = (-110),

	// Memory errors
	BLIS_MALLOC_RETURNED_NULL                  = (-120),

	// Internal memory pool errors
	BLIS_INVALID_PACKBUF                       = (-130),
	BLIS_EXHAUSTED_CONTIG_MEMORY_POOL          = (-131),
	BLIS_INSUFFICIENT_STACK_BUF_SIZE           = (-132),
	BLIS_ALIGNMENT_NOT_POWER_OF_TWO            = (-133),
	BLIS_ALIGNMENT_NOT_MULT_OF_PTR_SIZE        = (-134),

	// Object-related errors
	BLIS_EXPECTED_OBJECT_ALIAS                 = (-140),

	// Architecture-related errors
	BLIS_INVALID_ARCH_ID                       = (-150),
	BLIS_UNINITIALIZED_GKS_CNTX                = (-151),
	BLIS_INVALID_UKR_ID                        = (-152),

	// Blocksize-related errors
	BLIS_MC_DEF_NONMULTIPLE_OF_MR              = (-160),
	BLIS_MC_MAX_NONMULTIPLE_OF_MR              = (-161),
	BLIS_NC_DEF_NONMULTIPLE_OF_NR              = (-162),
	BLIS_NC_MAX_NONMULTIPLE_OF_NR              = (-163),
	BLIS_KC_DEF_NONMULTIPLE_OF_KR              = (-164),
	BLIS_KC_MAX_NONMULTIPLE_OF_KR              = (-165),
	BLIS_MR_NOT_EVEN_FOR_REAL_TYPE             = (-166),
	BLIS_PACKMR_NOT_EVEN_FOR_REAL_TYPE         = (-167),
	BLIS_NR_NOT_EVEN_FOR_REAL_TYPE             = (-168),
	BLIS_PACKNR_NOT_EVEN_FOR_REAL_TYPE         = (-169),

	BLIS_ERROR_CODE_MAX                        = (-170)
} err_t;

#endif
