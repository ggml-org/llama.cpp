/****************************************************************
Copyright 1990 - 1997 by AT&T, Lucent Technologies and Bellcore.

Permission to use, copy, modify, and distribute this software
and its documentation for any purpose and without fee is hereby
granted, provided that the above copyright notice appear in all
copies and that both that the copyright notice and this
permission notice and warranty disclaimer appear in supporting
documentation, and that the names of AT&T, Bell Laboratories,
Lucent or Bellcore or any of their entities not be used in
advertising or publicity pertaining to distribution of the
software without specific, written prior permission.

AT&T, Lucent and Bellcore disclaim all warranties with regard to
this software, including all implied warranties of
merchantability and fitness.  In no event shall AT&T, Lucent or
Bellcore be liable for any special, indirect or consequential
damages or any damages whatsoever resulting from loss of use,
data or profits, whether in an action of contract, negligence or
other tortious action, arising out of or in connection with the
use or performance of this software.
****************************************************************/

/* include/f2c_types.h.  Generated from f2c_types.h.in by configure.  */
/* include/f2c.h.  Generated from f2c.h.in by configure.  */
/* f2c.h  --  Standard Fortran to C header file */

/**  barf  [ba:rf]  2.  "He suggested using FORTRAN, and everybody barfed."

	- From The Shogakukan DICTIONARY OF NEW ENGLISH (Second edition) */

#ifndef F2C_TYPES_H
#define F2C_TYPES_H

#ifdef HAVE_BLIS_H
  #include <stdint.h>
  #define BLIS_VIA_BLASTEST
  #include "blis.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Define to the number of bits in an integer */
#define F2C_INT_BITS 32

/* Define to the number of bits in a long integer */
#define F2C_LONG_BITS 64

/* Define to the number of bits in a long long integer, if it exists */
#define F2C_LONG_LONG_BITS 64

#ifdef HAVE_BLIS_H
  #if   BLIS_BLAS_INT_TYPE_SIZE == 32
    typedef int32_t   integer;
  #elif BLIS_BLAS_INT_TYPE_SIZE == 64
    typedef int64_t   integer;
  #else
    typedef long int  integer;
  #endif
//typedef int integer;
typedef unsigned int uinteger;
#endif
#if F2C_INT_BITS == 32
# if F2C_LONG_BITS == 64
typedef long int longint;
typedef unsigned long int ulongint;
#  define INTEGER_STAR_8
# elif defined(F2C_LONG_LONG_BITS)
#  if F2C_LONG_LONG_BITS == 64
typedef long long int longint;
typedef unsigned long long int ulongint;
#  define INTEGER_STAR_8
#  endif
# endif
#endif

typedef char integer1;
typedef char *address;
typedef short int shortint;
typedef float real;
typedef double doublereal;
typedef struct { real r, i; } complex;
typedef struct { doublereal r, i; } doublecomplex;
typedef integer logical;
typedef shortint shortlogical;
typedef integer1 logical1;

#ifdef f2c_i2
/* for -i2 */
typedef short flag;
#ifndef HAVE_BLIS_H // don't re-typedef ftnlen
typedef short ftnlen;
#endif
typedef short ftnint;
#else
typedef integer flag;
#ifndef HAVE_BLIS_H // don't re-typedef ftnlen
typedef integer ftnlen;
#endif
typedef integer ftnint;
#endif

/* procedure parameter types for -A and -C++ */

#define F2C_proc_par_types 1
#ifdef __cplusplus
typedef int /* Unknown procedure type */ (*U_fp)(...);
typedef shortint (*J_fp)(...);
typedef integer (*I_fp)(...);
typedef real (*R_fp)(...);
typedef doublereal (*D_fp)(...), (*E_fp)(...);
typedef /* Complex */ void (*C_fp)(...);
typedef /* Double Complex */ void (*Z_fp)(...);
typedef logical (*L_fp)(...);
typedef shortlogical (*K_fp)(...);
typedef /* Character */ void (*H_fp)(...);
typedef /* Subroutine */ int (*S_fp)(...);
#else
typedef int /* Unknown procedure type */ (*U_fp)();
typedef shortint (*J_fp)();
typedef integer (*I_fp)();
typedef real (*R_fp)();
typedef doublereal (*D_fp)(), (*E_fp)();
typedef /* Complex */ void (*C_fp)();
typedef /* Double Complex */ void (*Z_fp)();
typedef logical (*L_fp)();
typedef shortlogical (*K_fp)();
typedef /* Character */ void (*H_fp)();
typedef /* Subroutine */ int (*S_fp)();
#endif
/* E_fp is for real functions when -R is not specified */
typedef void C_f;	/* complex function */
typedef void H_f;	/* character function */
typedef void Z_f;	/* double complex function */
typedef doublereal E_f;	/* real function with -R not specified */


#ifdef __cplusplus
}
#endif

#endif /* F2C_TYPES_H */
