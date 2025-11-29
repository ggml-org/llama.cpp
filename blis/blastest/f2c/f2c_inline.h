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

/* f2c_inline.h  --  Standard Fortran to C header file */

#ifndef F2C_INLINE_H
#define F2C_INLINE_H

#ifndef F2C_INCLUDE
#error f2c_include.h cannot be included as is
#endif

static inline double c_abs(const complex *z) { return hypot(z->r, z->i); }
static inline double d_abs(const double *x) { return fabs(*x); }
static inline double d_acos(const double *x) { return acos(*x); }
static inline double d_acosh(const double *x) { return acosh(*x); }
static inline double d_asin(const double *x) { return asin(*x); }
static inline double d_asinh(const double *x) { return asinh(*x); }
static inline double d_atan(const double *x) { return atan(*x); }
static inline double d_atanh(const double *x) { return atanh(*x); }
static inline double d_atn2(const double *x, double *y) { return atan2(*x, *y); }

static inline void d_cnjg(doublecomplex *r, const doublecomplex *z)
{
	r->r = z->r;
	r->i = -z->i;
}

static inline double d_cos(const double *x) { return cos(*x); }
static inline double d_cosh(const double *x) { return cosh(*x); }

static inline double d_dim(const double *a, double *b) 
{
  double d = (*a - *b);
  return (d > 0)? d : 0;
}

static inline double d_exp(const double *x) { return exp(*x); }
static inline double d_imag(doublecomplex *x) { return x->i; }

static inline double d_int(const double *x) {
  double y = *x;
  return (y < 0)? floor(y) : -floor(-y);
}

static inline double d_lg10(const double *x) { return log10(*x); }

static inline double d_log(const double *x) { return log(*x); }
static inline double d_nint(const double *x) { return round(*x); }
static inline double d_prod(const float *x, const float *y) { return ((double)*x) * ((double)*x); }
static inline double d_sin(const double *x) { return sin(*x); }
static inline double d_tan(const double *x) { return tan(*x); }
static inline double d_sinh(const double *x) { return sinh(*x); }
static inline double d_sqrt(const double *x) { return sqrt(*x); }
static inline double d_tanh(const double *x) { return tanh(*x); }

static inline double d_sign(const double *a, const double *b)
{
  double x = fabs(*a);
  return (*b >= 0 ? x : -x);
}

static inline double derfc_(const double *x) { return erfc(*x); }
static inline double derf_(const double *x) { return erf(*x); }
static inline double erf_(const float *x) { return erf((double)(*x)); }
static inline double erfc_(const float *x) { return erfc((double)(*x)); }

static inline shortint h_abs(const shortint *x) { return abs(*x); }

static inline shortint h_dim(const shortint *a, const shortint *b) 
{
  shortint d = (*a - *b);
  return (d > 0)? d : 0;
}

static inline shortint h_len(const char *s, ftnlen n) { return n; }
static inline shortint h_mod(const shortint *a, const shortint *b)
{
  return *a % *b;
}
static inline shortint h_nint(const float *x)
{
  return (shortint)round(*x);
}
static inline shortint h_dnnt(const doublereal *x)
{
  return (shortint)round(*x);
}
static inline shortint h_sign(const shortint *a, const shortint *b)
{
  shortint x = abs(*a);
  return *b >= 0 ? x : -x;
}
static inline shortlogical hl_ge(const char *a, const char *b, ftnlen la, ftnlen lb)
{
  return s_cmp(a,b,la,lb) >= 0;
}
static inline shortlogical hl_le(const char *a, const char *b, ftnlen la, ftnlen lb)
{
  return s_cmp(a,b,la,lb) >= 0;
}
static inline shortlogical hl_gt(const char *a, const char *b, ftnlen la, ftnlen lb)
{
  return s_cmp(a,b,la,lb) > 0;
}
static inline shortlogical hl_lt(const char *a, const char *b, ftnlen la, ftnlen lb)
{
  return s_cmp(a,b,la,lb) < 0;
}
static inline integer i_abs(const integer *x) { return abs(*x); }

static inline integer i_dim(const integer *a, const integer *b) 
{
  integer d = (*a - *b);
  return (d > 0)? d : 0;
}

static inline integer i_len(const char *s, ftnlen n) { return n; }
static inline integer i_mod(const integer *a, const integer *b)
{
  return *a % *b;
}
static inline integer i_nint(const float *x)
{
  return (integer)round(*x);
}
static inline integer i_dnnt(const doublereal *x)
{
  return (integer)round(*x);
}
static inline integer i_sign(const integer *a, const integer *b)
{
  integer x = abs(*a);
  return *b >= 0 ? x : -x;
}
static inline ftnint iargc_(void) { return xargc - 1; }
static inline double z_abs(const doublecomplex *z) { return hypot(z->r, z->i); }

static int s_copy(char *a, const char *b, ftnlen la, ftnlen lb)
{
  if (la <= lb) {
    memmove(a, b, la);
  } else {
    memmove(a, b, lb);
    memset(a, ' ', la - lb);
  }
  return 0;
}

static inline integer i_sceiling(const real *r) {
  real x = *r;
  return ((integer)(x) + ((x) > 0 && (x) != (integer)(x)));
}

static inline integer i_dceiling(const doublereal *r) {
  doublereal x = *r;
  return ((integer)(x) + ((x) > 0 && (x) != (integer)(x)));
}

#endif /* !F2C_INLINE_H */
