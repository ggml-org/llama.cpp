/* include/f2c.h.  Generated from f2c.h.in by configure.  */
/* f2c.h  --  Standard Fortran to C header file */

/**  barf  [ba:rf]  2.  "He suggested using FORTRAN, and everybody barfed."

	- From The Shogakukan DICTIONARY OF NEW ENGLISH (Second edition) */

#ifndef F2C_TYPES_WIN_H
#define F2C_TYPES_WIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Define to the number of bits in an integer */
#define F2C_INT_BITS 32

/* Define to the number of bits in a long integer */
#define F2C_LONG_BITS 64

typedef int integer;
typedef unsigned int uinteger;
typedef __int64 longint;
typedef unsigned __int64 ulongint;
/*#define INTEGER_STAR_8*/

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
typedef short ftnlen;
typedef short ftnint;
#else
typedef integer flag;
typedef integer ftnlen;
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

