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

/* include/f2c.h.  Generated from f2c.h.in by configure.  */
/* f2c.h  --  Standard Fortran to C header file */

/**  barf  [ba:rf]  2.  "He suggested using FORTRAN, and everybody barfed."

	- From The Shogakukan DICTIONARY OF NEW ENGLISH (Second edition) */

#ifndef F2C_INCLUDE
#define F2C_INCLUDE

#include <math.h>
#include <string.h>
#ifdef _MSC_VER
# include <f2c_types_win.h>
#else
# include <f2c_types.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef INTEGER_STAR_8	/* Adjust for integer*8. */
#define qbit_clear(a,b)	((a) & ~((ulongint)1 << (b)))
#define qbit_set(a,b)	((a) |  ((ulongint)1 << (b)))
#endif

#define TRUE_ (1)
#define FALSE_ (0)

/* Extern is for use with -E */
#ifndef Extern
#define Extern extern
#endif

/* I/O stuff */

/*external read, write*/
typedef struct
{	flag cierr;
	ftnint ciunit;
	flag ciend;
	char *cifmt;
	ftnint cirec;
} cilist;

/*internal read, write*/
typedef struct
{	flag icierr;
	char *iciunit;
	flag iciend;
	char *icifmt;
	ftnint icirlen;
	ftnint icirnum;
} icilist;

/*open*/
typedef struct
{	flag oerr;
	ftnint ounit;
	char *ofnm;
	ftnlen ofnmlen;
	char *osta;
	char *oacc;
	char *ofm;
	ftnint orl;
	char *oblnk;
} olist;

/*close*/
typedef struct
{	flag cerr;
	ftnint cunit;
	char *csta;
} cllist;

/*rewind, backspace, endfile*/
typedef struct
{	flag aerr;
	ftnint aunit;
} alist;

/* inquire */
typedef struct
{	flag inerr;
	ftnint inunit;
	char *infile;
	ftnlen infilen;
	ftnint	*inex;	/*parameters in standard's order*/
	ftnint	*inopen;
	ftnint	*innum;
	ftnint	*innamed;
	char	*inname;
	ftnlen	innamlen;
	char	*inacc;
	ftnlen	inacclen;
	char	*inseq;
	ftnlen	inseqlen;
	char 	*indir;
	ftnlen	indirlen;
	char	*infmt;
	ftnlen	infmtlen;
	char	*inform;
	ftnint	informlen;
	char	*inunf;
	ftnlen	inunflen;
	ftnint	*inrecl;
	ftnint	*innrec;
	char	*inblank;
	ftnlen	inblanklen;
} inlist;

union Multitype {	/* for multiple entry points */
	integer1 g;
	shortint h;
	integer i;
	/* longint j; */
	real r;
	doublereal d;
	complex c;
	doublecomplex z;
};

typedef union Multitype Multitype;

struct Vardesc {	/* for Namelist */
	char *name;
	char *addr;
	ftnlen *dims;
	int  type;
};
typedef struct Vardesc Vardesc;

struct Namelist {
	char *name;
	Vardesc **vars;
	int nvars;
};
typedef struct Namelist Namelist;

#define abs(x) ((x) >= 0 ? (x) : -(x))
#define dabs(x) (doublereal)abs(x)
#define min(a,b) ((a) <= (b) ? (a) : (b))
#define max(a,b) ((a) >= (b) ? (a) : (b))
#define dmin(a,b) (doublereal)min(a,b)
#define dmax(a,b) (doublereal)max(a,b)
#define bit_test(a,b)	((a) >> (b) & 1)
#define bit_clear(a,b)	((a) & ~((uinteger)1 << (b)))
#define bit_set(a,b)	((a) |  ((uinteger)1 << (b)))

/* undef any lower-case symbols that your C compiler predefines, e.g.: */

#ifndef Skip_f2c_Undefs
/* #undef cray */
/* #undef gcos */
/* #undef mc68010 */
/* #undef mc68020 */
/* #undef mips */
/* #undef pdp11 */
/* #undef sgi */
/* #undef sparc */
/* #undef sun */
/* #undef sun2 */
/* #undef sun3 */
/* #undef sun4 */
/* #undef u370 */
/* #undef u3b */
/* #undef u3b2 */
/* #undef u3b5 */
/* #undef unix */
/* #undef vax */
#endif

void libf2c_init(int argc, char **argv);
void libf2c_close();

/*************************************************************
 * LIBF77
 */

/*
 * Private functions and variables in libF77
 */
extern int xargc;
extern char **xargv;
extern doublereal _0;

double f__cabs(double, double);
char *F77_aloc(integer Len, const char *whence);
void sig_die(const char*, int);
void _uninit_f2c(void *x, int type, long len);

/*
 * Public functions in libF77
 */

int abort_(void);

void c_cos(complex *r, complex *z);
void c_div(complex *c, complex *a, complex *b);
void c_exp(complex *r, complex *z);
void c_log(complex *r, complex *z);
void c_sin(complex *r, complex *z);
void c_sqrt(complex *r, complex *z);

double dtime_(float *tarray);

int ef1asc_(ftnint *a, ftnlen *la, ftnint *b, ftnlen *lb);
integer ef1cmc_(ftnint *a, ftnlen *la, ftnint *b, ftnlen *lb);

real etime_(real *tarray);

int getarg_(ftnint *n, char *s, ftnlen ls);
int getenv_(char *fname, char *value, ftnlen flen, ftnlen vlen);

shortint h_indx(char *a, char *b, ftnlen la, ftnlen lb);

integer i_indx(char *a, char *b, ftnlen la, ftnlen lb);

logical l_ge(char *a, char *b, ftnlen la, ftnlen lb);
logical l_gt(char *a, char *b, ftnlen la, ftnlen lb);
logical l_le(char *a, char *b, ftnlen la, ftnlen lb);
logical l_lt(char *a, char *b, ftnlen la, ftnlen lb);

integer lbit_bits(integer a, integer b, integer len);
integer lbit_shift(integer a, integer b);
integer lbit_cshift(integer a, integer b, integer len);

void pow_ci(complex *p, complex *a, integer *b);
double pow_dd(doublereal *ap, doublereal *bp);
double pow_di(doublereal *ap, integer *bp);
shortint pow_hh(shortint *ap, shortint *bp);
integer pow_ii(integer *ap, integer *bp);
#ifdef INTEGER_STAR_8
longint pow_qq(longint *ap, longint *bp);
#endif
double pow_ri(real *ap, integer *bp);
void pow_zi(doublecomplex*, doublecomplex*, integer*);
void pow_zz(doublecomplex *r, doublecomplex *a, doublecomplex *b);

#ifdef INTEGER_STAR_8
longint qbit_bits(longint a, integer b, integer len);
longint qbit_cshift(longint a, integer b, integer len);
longint qbit_shift(longint a, integer b);
#endif

double r_abs(real *x);
double r_acos(real *x);
double r_asin(real *x);
double r_atan(real *x);
double r_atn2(real *x, real *y);
void r_cnjg(complex *r, complex *z);
double r_cos(real *x);
double r_cosh(real *x);
double r_dim(real *a, real *b);
double r_exp(real *x);
double r_imag(complex *z);
double r_int(real *x);
double r_lg10(real *x);
double r_log(real *x);
double r_mod(real *x, real *y);
double r_nint(real *x);
double r_sign(real *a, real *b);
double r_sin(real *x);
double r_sinh(real *x);
double r_sqrt(real *x);
double r_tan(real *x);
double r_tanh(real *x);

int s_cat(char *lp, char *rpp[], ftnint rnp[], ftnint *np, ftnlen ll);
integer s_cmp(const char *a0, const char *b0, ftnlen la, ftnlen lb);
int s_paus(char *s, ftnlen n);
integer s_rnge(char *varn, ftnint offset, char *procn, ftnint line);
int s_stop(char *s, ftnlen n);

ftnint signal_(integer *sigp, void *proc);
integer system_(register char *s, ftnlen n);

void z_div(doublecomplex*, doublecomplex*, doublecomplex*);
void z_cos(doublecomplex *r, doublecomplex *z);
void z_exp(doublecomplex *r, doublecomplex *z);
void z_log(doublecomplex *r, doublecomplex *z);
void z_sin(doublecomplex *r, doublecomplex *z);
void z_sqrt(doublecomplex *r, doublecomplex *z);

/*
#ifndef F2C_NO_INLINE_H
# if defined(__GNUC__)
#  include <f2c_inline.h>
# endif
#endif
*/

#if !defined(F2C_INLINE_H)
double c_abs(const complex *z);
double d_abs(const doublereal *x);
double d_acos(const doublereal *x);
double d_asin(const doublereal *x);
double d_atan(const doublereal *x);
double d_atn2(const doublereal *x, const doublereal *y);
void d_cnjg(doublecomplex *r, const doublecomplex *z);
double d_cos(const doublereal *x);
double d_cosh(const doublereal *x);
double d_dim(const doublereal *a, const doublereal *b);
double d_exp(const doublereal *x);
double d_imag(const doublecomplex *z);
double d_int(const doublereal *x);
double d_lg10(const doublereal *x);
double d_log(const doublereal *x);
double d_mod(const doublereal *x, const doublereal *y);
double d_nint(const doublereal *x);
double d_prod(const real *x, const real *y);
double d_sign(const doublereal *a, const doublereal *b);
double d_sin(const doublereal *x);
double d_sinh(const doublereal *x);
double d_sqrt(const doublereal *x);
double d_tan(const doublereal *x);
double d_tanh(const doublereal *x);
double derf_(const doublereal *x);
double derfc_(const doublereal *x);
double erf_(const real *x);
double erfc_(const real *x);
shortint h_abs(const shortint *x);
shortint h_dim(const shortint *a, const shortint *b);
shortint h_dnnt(const doublereal *x);
shortint h_len(const char *s, ftnlen n);
shortint h_mod(const short *a, const short *b);
shortint h_nint(const real *x);
shortint h_sign(const shortint *a, const shortint *b);
shortlogical hl_ge(const char *a, const char *b, ftnlen la, ftnlen lb);
shortlogical hl_gt(const char *a, const char *b, ftnlen la, ftnlen lb);
shortlogical hl_le(const char *a, const char *b, ftnlen la, ftnlen lb);
shortlogical hl_lt(const char *a, const char *b, ftnlen la, ftnlen lb);
integer i_abs(const integer *x);
integer i_dceiling(const doublereal *x);
integer i_dim(const integer *a, const integer *b);
integer i_dnnt(const doublereal *x);
integer i_len(const char *s, ftnlen n);
integer i_len_trim(const char *s, ftnlen n);
integer i_mod(const integer *a, const integer *b);
integer i_nint(const real *x);
integer i_sign(const integer *a, const integer *b);
integer i_sceiling(const real *x);
ftnint iargc_(void);
int s_copy(char *a, const char *b, ftnlen la, ftnlen lb);
double z_abs(const doublecomplex *z);
#endif /* !F2C_INLINE_H */

/*************************************************************
 * LIBI77
 *
 * Public functions
 */

int c_dfe(cilist *a);
int c_due(cilist *a);
int c_sfe(cilist *a);
int c_sue(cilist *a);

integer e_rdfe(void);
integer e_rdue(void);
integer e_rsfe(void);
integer e_rsfi(void);
integer e_rsle(void);
integer e_rsli(void);
integer e_rsue(void);
integer e_wdfe(void);
integer e_wdue(void);
integer e_wsfi(void);
integer e_wsfe(void);
integer e_wsle(void);
integer e_wsli(void);
integer e_wsue(void);

void exit_(integer *rc);

integer f_back(alist *a);
integer f_clos(cllist *a);
integer f_end(alist *a);
void f_exit(void);
integer f_inqu(inlist *a);
integer f_open(olist *a);
integer f_rew(alist *a);
int flush_(void);

integer ftell_(integer *Unit);
int fseek_(integer *Unit, integer *offset, integer *whence);
#ifdef INTEGER_STAR_8
longint ftell64_(integer *Unit);
int fseek64_(integer *Unit, longint *offset, integer *whence);
#endif

integer s_rdfe(cilist *a);
integer s_rdue(cilist *a);
integer s_rsfi(icilist *a);
integer s_rsle(cilist *a);
integer s_rsli(icilist *a);
integer s_rsne(cilist *a);
integer s_rsni(icilist *a);
integer s_rsue(cilist *a);
integer s_wdfe(cilist *a);
integer s_wdue(cilist *a);
integer s_wsfe(cilist *a);
integer s_wsfi(icilist *a);
integer s_wsle(cilist *a);
integer s_wsli(icilist *a);
integer s_wsne(cilist *a);
integer s_wsni(icilist *a);
integer s_wsue(cilist *a);

real s_epsilon_( real* x );
double d_epsilon_( doublereal* x );

/*
 * Private functions in the F2C library
 */
extern const ftnlen f__typesize[];

#ifdef __cplusplus
}
#endif

#endif
