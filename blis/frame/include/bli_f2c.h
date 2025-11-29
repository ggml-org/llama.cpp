// f2c.h  --  Standard Fortran to C header file
//  barf  [ba:rf]  2.  "He suggested using FORTRAN, and everybody barfed."
//  - From The Shogakukan DICTIONARY OF NEW ENGLISH (Second edition)

#ifndef BLIS_F2C_H
#define BLIS_F2C_H

typedef f77_int bla_integer;
typedef f77_char bla_character;
//typedef char *address;
//typedef short int shortint;
typedef float bla_real;
typedef double bla_double;
typedef scomplex bla_scomplex;
typedef dcomplex bla_dcomplex;
typedef f77_int bla_logical;
//typedef short int shortlogical;
//typedef char logical1;
//typedef char integer1;
#ifdef INTEGER_STAR_8                // Adjust for integer*8.
typedef long long longint;           // system-dependent
typedef unsigned long long ulongint; // system-dependent
#define qbit_clear(a,b)	((a) & ~((ulongint)1 << (b)))
#define qbit_set(a,b)	((a) |  ((ulongint)1 << (b)))
#endif

#ifndef TRUE_
#define TRUE_ (1)
#endif

#ifndef FALSE_
#define FALSE_ (0)
#endif

// Extern is for use with -E
#ifndef Extern
#define Extern extern
#endif

// I/O stuff

#ifdef f2c_i2
// for -i2
//typedef short flag;
//typedef short ftnlen;
typedef bla_integer ftnlen;
//typedef short ftnint;
#else
//typedef long int flag;
//typedef long int ftnlen;
typedef bla_integer ftnlen;
//typedef long int ftnint;
#endif

#ifndef VOID
#define VOID void
#endif

#ifndef f2c_abs
  #define f2c_abs(x) ((x) >= 0 ? (x) : -(x))
#endif
#ifndef f2c_dabs
  #define f2c_dabs(x) (doublereal)f2c_abs(x)
#endif
#ifndef f2c_min
  #define f2c_min(a,b) ((a) <= (b) ? (a) : (b))
#endif
#ifndef f2c_max
  #define f2c_max(a,b) ((a) >= (b) ? (a) : (b))
#endif
#ifndef f2c_dmin
  #define f2c_dmin(a,b) (doublereal)f2c_min(a,b)
#endif
#ifndef f2c_dmax
  #define f2c_dmax(a,b) (doublereal)f2c_max(a,b)
#endif

#ifndef bit_test
  #define bit_test(a,b)  ((a) >> (b) & 1)
#endif

#ifndef bit_clear
  #define bit_clear(a,b) ((a) & ~((uinteger)1 << (b)))
#endif

#ifndef bit_set
  #define bit_set(a,b)   ((a) |  ((uinteger)1 << (b)))
#endif

// undef any lower-case symbols that your C compiler predefines, e.g.:

#ifndef Skip_f2c_Undefs
#undef cray
#undef gcos
#undef mc68010
#undef mc68020
#undef mips
#undef pdp11
#undef sgi
#undef sparc
#undef sun
#undef sun2
#undef sun3
#undef sun4
#undef u370
#undef u3b
#undef u3b2
#undef u3b5
#undef unix
#undef vax
#endif

#endif
