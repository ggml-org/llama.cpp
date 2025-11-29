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

#ifndef F2C_ARITH_H
#define F2C_ARITH_H

#include <f2c_config.h>
#include <math.h>

#ifdef _MSC_VER
#define isnan _isnan
#define isinf(x) (!_finite(x))
#endif

#ifndef isnan
# define isnan(x)						 \
  (sizeof (x) == sizeof (long double) ? isnan_ld (x)		 \
   : sizeof (x) == sizeof (double) ? isnan_d (x)		 \
   : isnan_f (x))
static inline int isnan_f  (float       x) { return x != x; }
static inline int isnan_d  (double      x) { return x != x; }
static inline int isnan_ld (long double x) { return x != x; }
#endif
          
#ifndef isinf
# define isinf(x)						 \
  (sizeof (x) == sizeof (long double) ? isinf_ld (x)		 \
   : sizeof (x) == sizeof (double) ? isinf_d (x)		 \
   : isinf_f (x))
static inline int isinf_f  (float       x)
{ return !isnan (x) && isnan (x - x); }
static inline int isinf_d  (double      x)
{ return !isnan (x) && isnan (x - x); }
static inline int isinf_ld (long double x)
{ return !isnan (x) && isnan (x - x); }
#endif

#ifndef signbit
#define signbit(x) (((x) < 0)? 1 : 0)
#endif

#endif
